//! Row normalization with a centered first-order VJP, including positive subnormal mass.
#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    Backend, Dispatch, TensorMetadata, backend_extension, get_or_init_device_settings,
    autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::FloatTensor,
};
use burn::prelude::Tensor;

fn safe_mass<B: Backend>(mass: FloatTensor<B>) -> FloatTensor<B> {
    let bool_dtype = get_or_init_device_settings::<B>(&mass.device()).bool_dtype;
    let zero = B::float_equal_elem(mass.clone(), 0.into(), bool_dtype);
    B::float_mask_fill(mass, zero, 1.into())
}

#[backend_extension(
    NdArray: cfg(feature = "cpu"),
    Cube: cfg(any(feature = "vulkan", feature = "cuda")),
    Autodiff,
    Fusion: cfg(feature = "fusion")
)]
pub trait SoftNormalizeOps: Backend {
    #[fusion(meta = |x: &TensorSpec, _mass| x.clone())]
    fn soft_normalize(x: FloatTensor<Self>, mass: FloatTensor<Self>) -> FloatTensor<Self> {
        Self::float_div(x, safe_mass::<Self>(mass))
    }
}
#[cfg(feature = "cpu")]
impl SoftNormalizeOps for burn::backend::NdArray {}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
impl SoftNormalizeOps for burn_cubecl::CubeBackend {}

impl<B: Backend + SoftNormalizeOps, C: CheckpointStrategy> SoftNormalizeOps for Autodiff<B, C> {
    fn soft_normalize(x: FloatTensor<Self>, mass: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct NormalizeBackward;
        impl<B: Backend> Backward<B, 1> for NormalizeBackward {
            type State = (FloatTensor<B>, FloatTensor<B>);
            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
                let upstream = grads.consume::<B>(&ops.node);
                if let Some(parent) = &ops.parents[0] {
                    let (normalized, mass) = ops.state;
                    let mean = B::float_sum_dim(B::float_mul(upstream.clone(), normalized), 1);
                    // Center BEFORE dividing. Separate reciprocal paths overflow and yield
                    // inf-inf for the identically constant sum(normalized) loss at tiny mass.
                    let centered = B::float_sub(upstream, mean);
                    let gradient = B::float_div(centered, mass);
                    grads.register::<B>(parent.id, gradient);
                }
            }
        }
        let parents = [x.node()];
        let x = x.into_primitive();
        let mass = safe_mass::<B>(mass.into_primitive());
        match NormalizeBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep) => {
                let normalized = B::float_div(x, mass.clone());
                prep.finish((normalized.clone(), mass), normalized)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::soft_normalize(x, mass)),
        }
    }
}

/// Nonnegative rows normalize over their supplied support; all-zero rows stay zero.
/// The zero-row extension uses denominator one, matching the mask-only contract.
/// The VJP does not clamp unrepresentable derivatives or detach probabilities.
pub fn soft_normalize(x: Tensor<2>, mass: Tensor<2>) -> Tensor<2> {
    // The custom VJP already accounts for the mass's dependence on x. Reuse the
    // caller's sum without attaching a second denominator path to the graph.
    Tensor::from_dispatch(<Dispatch as SoftNormalizeOps>::soft_normalize(
        x.into_dispatch(), mass.detach().into_dispatch()))
}
