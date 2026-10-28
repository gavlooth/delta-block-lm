//! Row normalization with a centered first-order VJP, including positive subnormal mass.
#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    AutodiffBackend, Backend, Dispatch, TensorMetadata, backend_extension, get_or_init_device_settings,
    autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::FloatTensor,
};
use burn::prelude::Tensor;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
use burn_cubecl::CubeBackend;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
use cubecl::{frontend::Tensor as View, prelude::*};

#[cfg(any(feature = "vulkan", feature = "cuda"))]
#[cube(launch)]
fn denominator_kernel(x: &View<f32>, mass: &View<f32>, denominator: &mut View<f32>) {
    let row = ABSOLUTE_POS as usize;
    if row < denominator.len() {
        let value = mass[row * mass.stride(0)];
        let bits = u32::reinterpret(value) & 0x7fffffff_u32;
        if bits < 0x00800000_u32 {
            // Arithmetic may have flushed the caller's sum. Convert significand
            // bits to normal floats BEFORE summing: this is exactly x * 2^126.
            let mut scaled = 0.0f32;
            for col in 0..x.shape(1) {
                let input = u32::reinterpret(x[row * x.stride(0) + col * x.stride(1)])
                    & 0x7fffffff_u32;
                scaled += f32::cast_from(input) * 1.1920928955078125e-7f32;
            }
            // A negative denominator records the exact power-of-two rescaling.
            denominator[row] = if scaled > 0.0f32 { -scaled } else { 1.0f32 };
        } else {
            denominator[row] = value;
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
#[cube(launch)]
fn divide_kernel(
    x: &View<f32>,
    denominator: &View<f32>,
    output: &mut View<f32>,
    #[comptime] backward: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < output.len() {
        let row = i / x.shape(1);
        let col = i % x.shape(1);
        let value = x[row * x.stride(0) + col * x.stride(1)];
        let divisor = denominator[row];
        if divisor < 0.0 {
            if backward {
                output[i] = (value / -divisor) * 8.507059173023462e37f32;
            } else {
                let bits = u32::reinterpret(value) & 0x7fffffff_u32;
                output[i] = (f32::cast_from(bits) * 1.1920928955078125e-7f32) / -divisor;
            }
        } else {
            output[i] = value / divisor;
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn launch_shape(x: &FloatTensor<CubeBackend>, n: usize) -> CubeCount {
    let groups = n.div_ceil(256);
    let (max_x, max_y, _) = x.client.properties().hardware.max_cube_count;
    let gx = groups.min(max_x.min(65535) as usize).max(1);
    assert!(groups.div_ceil(gx) <= max_y as usize);
    CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1)
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn gpu_divide(
    x: FloatTensor<CubeBackend>,
    denominator: FloatTensor<CubeBackend>,
    backward: bool,
) -> FloatTensor<CubeBackend> {
    use burn::backend::ops::FloatTensorOps;
    let shape = x.shape();
    let output = CubeBackend::float_empty(shape.clone(), &x.device, x.dtype.into());
    divide_kernel::launch(
        &x.client, launch_shape(&x, shape.num_elements()), CubeDim::new_1d(256),
        x.clone().into_tensor_arg(), denominator.into_tensor_arg(),
        output.clone().into_tensor_arg(), backward,
    );
    output
}

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
        Self::soft_normalize_forward(x, mass).0
    }
    #[fusion(meta = |x: &TensorSpec, mass: &TensorSpec| (x.clone(), mass.clone()))]
    fn soft_normalize_forward(
        x: FloatTensor<Self>, mass: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let denominator = safe_mass::<Self>(mass);
        (Self::float_div(x, denominator.clone()), denominator)
    }
    #[fusion(meta = |x: &TensorSpec, _denominator| x.clone())]
    fn soft_normalize_divide(
        x: FloatTensor<Self>, denominator: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        Self::float_div(x, denominator)
    }
}
#[cfg(feature = "cpu")]
impl SoftNormalizeOps for burn::backend::NdArray {}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
impl SoftNormalizeOps for CubeBackend {
    fn soft_normalize_forward(
        x: FloatTensor<Self>, mass: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        use burn::{backend::ops::FloatTensorOps, tensor::DType};
        assert_eq!(x.dtype, DType::F32, "soft normalization requires f32");
        let shape = x.shape();
        assert_eq!(shape.num_dims(), 2);
        assert_eq!(mass.shape().dims::<2>(), [shape.dims::<2>()[0], 1]);
        let denominator = CubeBackend::float_empty(mass.shape(), &x.device, x.dtype.into());
        denominator_kernel::launch(
            &x.client, launch_shape(&x, shape.dims::<2>()[0]), CubeDim::new_1d(256),
            x.clone().into_tensor_arg(), mass.into_tensor_arg(),
            denominator.clone().into_tensor_arg(),
        );
        (gpu_divide(x, denominator.clone(), false), denominator)
    }
    fn soft_normalize_divide(
        x: FloatTensor<Self>, denominator: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        gpu_divide(x, denominator, true)
    }
}

impl<B: Backend + SoftNormalizeOps, C: CheckpointStrategy> SoftNormalizeOps for Autodiff<B, C> {
    fn soft_normalize(x: FloatTensor<Self>, mass: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct NormalizeBackward;
        impl<B: Backend + SoftNormalizeOps> Backward<B, 1> for NormalizeBackward {
            type State = (FloatTensor<B>, FloatTensor<B>);
            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
                let upstream = grads.consume::<B>(&ops.node);
                if let Some(parent) = &ops.parents[0] {
                    let (normalized, mass) = ops.state;
                    let mean = B::float_sum_dim(B::float_mul(upstream.clone(), normalized), 1);
                    // Center BEFORE dividing. Separate reciprocal paths overflow and yield
                    // inf-inf for the identically constant sum(normalized) loss at tiny mass.
                    let centered = B::float_sub(upstream, mean);
                    let gradient = B::soft_normalize_divide(centered, mass);
                    grads.register::<B>(parent.id, gradient);
                }
            }
        }
        let parents = [x.node()];
        let x = x.into_primitive();
        let mass = mass.into_primitive();
        match NormalizeBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep) => {
                let (normalized, mass) = B::soft_normalize_forward(x, mass);
                prep.finish((normalized.clone(), mass), normalized)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::soft_normalize(x, mass)),
        }
    }
    fn soft_normalize_forward(
        x: FloatTensor<Self>, mass: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (normalized, denominator) = B::soft_normalize_forward(x.into_primitive(), mass.into_primitive());
        (Self::from_inner(normalized), Self::from_inner(denominator))
    }
    fn soft_normalize_divide(
        x: FloatTensor<Self>, denominator: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        Self::from_inner(B::soft_normalize_divide(x.into_primitive(), denominator.into_primitive()))
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
