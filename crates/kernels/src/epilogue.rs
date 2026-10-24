//! Pointwise SiLU only: no GEMM, normalization, or residual reordering.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::tensor::DType;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;

#[cube]
fn offset(x: &View<f32>, i: usize) -> usize {
    let c = i % x.shape(2);
    let t = i / x.shape(2) % x.shape(1);
    let b = i / (x.shape(2) * x.shape(1));
    b * x.stride(0) + t * x.stride(1) + c * x.stride(2)
}
#[cube(launch)]
fn pointwise(x: &View<f32>, dy: &View<f32>, out: &mut View<f32>, #[comptime] backward: bool) {
    let i = ABSOLUTE_POS as usize;
    if i >= out.len() {
        terminate!();
    }
    let z = x[offset(x, i)];
    let s = 1.0f32 / (1.0f32 + f32::exp(-z));
    if backward {
        out[i] = dy[offset(dy, i)] * (s + z * s * (1.0f32 - s));
    } else {
        out[i] = z * s;
    }
}
fn validate(x: &CubeTensor) {
    assert_eq!(x.dtype, DType::F32);
    let dims = x.shape().dims::<3>();
    assert!(dims.iter().all(|&n| n > 0));
    let span = dims.iter().zip(x.meta.strides().iter()).try_fold(1usize, |span, (&n, &s)| {
        (n - 1).checked_mul(s).and_then(|v| span.checked_add(v))
    }).expect("SiLU address overflow");
    assert!(span <= u32::MAX as usize);
}
pub(crate) fn launch(x: CubeTensor, dy: Option<CubeTensor>) -> CubeTensor {
    validate(&x);
    let backward = dy.is_some();
    let dy = dy.unwrap_or_else(|| x.clone());
    validate(&dy);
    assert_eq!(x.shape(), dy.shape());
    assert_eq!(x.device, dy.device);
    let out = CubeBackend::float_empty(x.shape(), &x.device, DType::F32.into());
    let groups = x.shape().num_elements().div_ceil(256);
    let (mx, my, _) = x.client.properties().hardware.max_cube_count;
    let nx = groups.min(mx.min(65535) as usize).max(1);
    assert!(groups.div_ceil(nx) <= my as usize);
    pointwise::launch(
        &x.client,
        CubeCount::Static(nx as u32, groups.div_ceil(nx) as u32, 1),
        CubeDim::new_1d(256),
        x.clone().into_tensor_arg(),
        dy.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        backward,
    );
    out
}
