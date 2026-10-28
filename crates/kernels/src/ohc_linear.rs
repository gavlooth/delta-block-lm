//! Skinny oHC contraction with compensated input-gradient accumulation.
#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{Backend, Dispatch, TensorMetadata, backend_extension,
    autodiff::{Autodiff, checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients, ops::{Backward, Ops, OpsKind}}, tensor::FloatTensor};
use burn::prelude::Tensor;

#[backend_extension(NdArray: cfg(feature = "cpu"), Cube: cfg(any(feature = "vulkan", feature = "cuda")), Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait OhcLinearOps: Backend {
    #[fusion(meta = |x: &TensorSpec, _w, _transposed| { let [n, _] = x.shape.dims::<2>(); TensorSpec::new([n, 16].into(), x.dtype) })]
    fn ohc_linear(x: FloatTensor<Self>, w: FloatTensor<Self>, transposed: bool) -> FloatTensor<Self> {
        if transposed {
            Self::float_transpose(Self::float_matmul(Self::float_transpose(w), Self::float_transpose(x)))
        } else {
            Self::float_matmul(x, w)
        }
    }
}
#[cfg(feature = "cpu")]
impl OhcLinearOps for burn::backend::NdArray {}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
impl OhcLinearOps for burn_cubecl::CubeBackend {}

impl<B: Backend + OhcLinearOps, C: CheckpointStrategy> OhcLinearOps for Autodiff<B, C> {
    fn ohc_linear(x: FloatTensor<Self>, w: FloatTensor<Self>, transposed: bool) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct LinearBackward;
        impl<B: Backend> Backward<B, 2> for LinearBackward {
            type State = (FloatTensor<B>, FloatTensor<B>);
            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, w) = ops.state;
                if let Some(parent) = &ops.parents[0] {
                    let [n, width] = x.shape().dims::<2>();
                    let wt = B::float_transpose(w.clone());
                    let mut sum = B::float_zeros([n, width].into(), &x.device(), x.dtype().into());
                    let mut compensation = sum.clone();
                    for j in 0..16 {
                        let a = B::float_slice(dy.clone(), &[(0..n).into(), (j..j+1).into()]);
                        let b = B::float_slice(wt.clone(), &[(j..j+1).into(), (0..width).into()]);
                        // Recover product rounding as well as summation rounding:
                        // inverse gate RMS can amplify either error by 1e6.
                        // Dekker split: 4097 = 2^12 + 1 for f32's 24-bit significand.
                        let product = B::float_mul(a.clone(), b.clone());
                        let split_a = B::float_mul_scalar(a.clone(), 4097.0.into());
                        let high_a = B::float_sub(split_a.clone(), B::float_sub(split_a, a.clone()));
                        let low_a = B::float_sub(a, high_a.clone());
                        let split_b = B::float_mul_scalar(b.clone(), 4097.0.into());
                        let high_b = B::float_sub(split_b.clone(), B::float_sub(split_b, b.clone()));
                        let low_b = B::float_sub(b, high_b.clone());
                        let residual = B::float_add(
                            B::float_add(
                                B::float_add(B::float_sub(B::float_mul(high_a.clone(), high_b.clone()), product.clone()),
                                    B::float_mul(high_a, low_b.clone())),
                                B::float_mul(low_a.clone(), high_b)),
                            B::float_mul(low_a, low_b));
                        let next = B::float_add(sum.clone(), product.clone());
                        let added = B::float_sub(next.clone(), sum.clone());
                        let error = B::float_add(
                            B::float_sub(sum, B::float_sub(next.clone(), added.clone())),
                            B::float_sub(product, added));
                        compensation = B::float_add(compensation, B::float_add(error, residual));
                        sum = next;
                    }
                    grads.register::<B>(parent.id, B::float_add(sum, compensation));
                }
                if let Some(parent) = &ops.parents[1] {
                    grads.register::<B>(parent.id, B::float_matmul(B::float_transpose(x), dy));
                }
            }
        }
        let parents = [x.node(), w.node()];
        let (x, w) = (x.into_primitive(), w.into_primitive());
        let y = B::ohc_linear(x.clone(), w.clone(), transposed);
        match LinearBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((x, w), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
}

pub fn ohc_linear(x: Tensor<2>, w: Tensor<2>, transposed: bool) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as OhcLinearOps>::ohc_linear(x.into_dispatch(), w.into_dispatch(), transposed))
}
