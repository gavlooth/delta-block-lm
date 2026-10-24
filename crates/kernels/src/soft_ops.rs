//! Discrete IEEE total-order keys. Gather, not key construction, carries the VJP.
#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata,
        autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy},
        backend_extension,
        ops::IntTensorOps,
        tensor::{FloatTensor, IntTensor},
    },
    tensor::DType,
};
use burn_cubecl::CubeBackend;
use cubecl::{frontend::Tensor as View, prelude::*};

#[cube(launch)]
fn key_kernel(x: &View<f32>, keys: &mut View<i32>) {
    let i = ABSOLUTE_POS as usize;
    if i < keys.len() {
        let row = i / x.shape(1);
        let col = i % x.shape(1);
        let bits = i32::reinterpret(x[row * x.stride(0) + col * x.stride(1)]);
        keys[i] = if bits < 0 {
            bits ^ 0x7fffffff_i32
        } else {
            bits
        };
    }
}

#[backend_extension(Cube, Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait SoftOrderOps: Backend {
    #[fusion(meta = |x: &TensorSpec| TensorSpec::new(x.shape.clone(), DType::I32))]
    fn soft_order_keys(x: FloatTensor<Self>) -> IntTensor<Self>;
}
impl SoftOrderOps for CubeBackend {
    fn soft_order_keys(x: FloatTensor<Self>) -> IntTensor<Self> {
        assert_eq!(x.dtype, DType::F32, "soft selection requires f32");
        let shape = x.shape();
        assert_eq!(shape.num_dims(), 2);
        let mut span = 1usize;
        for (&dim, &stride) in x
            .meta
            .shape()
            .as_slice()
            .iter()
            .zip(x.meta.strides().iter())
        {
            assert!(dim > 0 && stride <= u32::MAX as usize);
            span = (dim - 1)
                .checked_mul(stride)
                .and_then(|n| span.checked_add(n))
                .expect("soft selection address overflow");
        }
        assert!(
            span <= u32::MAX as usize && x.handle.size() / 4 <= u32::MAX as u64,
            "soft selection exceeds 32-bit addressing"
        );
        let n = shape.num_elements();
        assert!(n <= u32::MAX as usize);
        let out = CubeBackend::int_empty(shape, &x.device, DType::I32.into());
        let groups = n.div_ceil(256);
        let (max_x, max_y, _) = x.client.properties().hardware.max_cube_count;
        let gx = groups.min(max_x.min(65535) as usize).max(1);
        assert!(groups.div_ceil(gx) <= max_y as usize);
        key_kernel::launch(
            &x.client,
            CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1),
            CubeDim::new_1d(256),
            x.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
        );
        out
    }
}
impl<B: Backend + SoftOrderOps, C: CheckpointStrategy> SoftOrderOps for Autodiff<B, C> {
    fn soft_order_keys(x: FloatTensor<Self>) -> IntTensor<Self> {
        B::soft_order_keys(x.into_primitive())
    }
}
pub fn soft_order_keys(
    x: burn::prelude::Tensor<2>,
) -> burn::prelude::Tensor<2, burn::prelude::Int> {
    burn::prelude::Tensor::from_dispatch(<Dispatch as SoftOrderOps>::soft_order_keys(
        x.into_dispatch(),
    ))
}
