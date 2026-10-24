//! exp(-exp(scale[h]) * softplus(x + bias[h,k])), including Burn's threshold 20.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::tensor::DType;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;
#[cube]
fn ix(x: &View<f32>, row: usize, h: usize, c: usize) -> usize {
    let t = row % x.shape(1);
    let b = row / x.shape(1);
    b * x.stride(0) + t * x.stride(1) + h * x.stride(2) + c * x.stride(3)
}
#[cube]
fn softplus(x: f32) -> f32 {
    if x > 20.0f32 {
        x
    } else {
        f32::log1p(f32::exp(x))
    }
}
#[cube(launch)]
fn forward_kernel(x: &View<f32>, bias: &View<f32>, scale: &View<f32>, y: &mut View<f32>) {
    let i = ABSOLUTE_POS as usize;
    let width = x.shape(3);
    let heads = x.shape(2);
    if i >= x.len() {
        terminate!();
    }
    let c = i % width;
    let h = i / width % heads;
    let row = i / (width * heads);
    let z = x[ix(x, row, h, c)] + bias[(h * width + c) * bias.stride(0)];
    y[i] = f32::exp(-softplus(z) * f32::exp(scale[h * scale.stride(0)]));
}
#[cube(launch)]
fn input_backward_kernel(
    x: &View<f32>,
    bias: &View<f32>,
    scale: &View<f32>,
    y: &View<f32>,
    dy: &View<f32>,
    dx: &mut View<f32>,
) {
    let i = ABSOLUTE_POS as usize;
    let width = x.shape(3);
    let heads = x.shape(2);
    if i >= x.len() {
        terminate!();
    }
    let c = i % width;
    let h = i / width % heads;
    let row = i / (width * heads);
    let z = x[ix(x, row, h, c)] + bias[(h * width + c) * bias.stride(0)];
    let derivative = if z > 20.0f32 {
        1.0f32
    } else {
        1.0f32 / (1.0f32 + f32::exp(-z))
    };
    dx[i] = -dy[ix(dy, row, h, c)] * y[i] * f32::exp(scale[h * scale.stride(0)]) * derivative;
}
#[cube(launch)]
fn parameter_backward_kernel(
    x: &View<f32>,
    bias: &View<f32>,
    scale: &View<f32>,
    y: &View<f32>,
    dy: &View<f32>,
    dx: &View<f32>,
    out: &mut View<f32>,
    #[comptime] is_scale: bool,
) {
    let parameter = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let width = x.shape(3);
    let heads = x.shape(2);
    let rows = x.shape(0) * x.shape(1);
    let mut sum = 0.0f32;
    let mut j = lane;
    if is_scale {
        if parameter >= heads {
            terminate!();
        }
        while j < rows * width {
            let row = j / width;
            let c = j % width;
            let z = x[ix(x, row, parameter, c)] + bias[(parameter * width + c) * bias.stride(0)];
            let i = (row * heads + parameter) * width + c;
            sum -= dy[ix(dy, row, parameter, c)]
                * y[i]
                * softplus(z)
                * f32::exp(scale[parameter * scale.stride(0)]);
            j += 64;
        }
    } else {
        if parameter >= heads * width {
            terminate!();
        }
        while j < rows {
            sum += dx[j * heads * width + parameter];
            j += 64;
        }
    }
    let mut reduction = Shared::<[f32]>::new_slice(64usize);
    reduction[lane] = sum;
    sync_cube();
    let mut offset = 32usize;
    while offset > 0 {
        if lane < offset {
            reduction[lane] += reduction[lane + offset];
        }
        sync_cube();
        offset /= 2;
    }
    if lane == 0 {
        out[parameter] = reduction[0];
    }
}
fn validate(x: &CubeTensor, bias: &CubeTensor, scale: &CubeTensor) {
    let [b, t, h, k] = x.shape().dims::<4>();
    assert!(b > 0 && t > 0 && h > 0 && k > 0);
    assert_eq!(bias.shape().dims::<1>(), [h * k]);
    assert_eq!(scale.shape().dims::<1>(), [h]);
    for a in [x, bias, scale] {
        assert_eq!(a.dtype, DType::F32);
        assert_eq!(a.device, x.device);
        let mut span = 1usize;
        for (&d, &s) in a
            .meta
            .shape()
            .as_slice()
            .iter()
            .zip(a.meta.strides().iter())
        {
            assert!(s <= u32::MAX as usize);
            span = (d - 1)
                .checked_mul(s)
                .and_then(|n| span.checked_add(n))
                .expect("decay address overflow");
        }
        assert!(span <= u32::MAX as usize && a.handle.size() / 4 <= u32::MAX as u64);
    }
}
fn count(x: &CubeTensor, groups: usize) -> CubeCount {
    let (mx, my, _) = x.client.properties().hardware.max_cube_count;
    let n = groups.min(mx.min(65535) as usize).max(1);
    assert!(groups.div_ceil(n) <= my as usize);
    CubeCount::Static(n as u32, groups.div_ceil(n) as u32, 1)
}
pub(crate) fn forward(x: CubeTensor, bias: CubeTensor, scale: CubeTensor) -> CubeTensor {
    validate(&x, &bias, &scale);
    let y = CubeBackend::float_empty(x.shape(), &x.device, DType::F32.into());
    forward_kernel::launch(
        &x.client,
        count(&x, x.shape().num_elements().div_ceil(64)),
        CubeDim::new_1d(64),
        x.clone().into_tensor_arg(),
        bias.into_tensor_arg(),
        scale.into_tensor_arg(),
        y.clone().into_tensor_arg(),
    );
    y
}
pub(crate) fn backward(
    x: CubeTensor,
    bias: CubeTensor,
    scale: CubeTensor,
    y: CubeTensor,
    dy: CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    validate(&x, &bias, &scale);
    assert_eq!(y.shape(), x.shape());
    assert_eq!(dy.shape(), x.shape());
    let dx = CubeBackend::float_empty(x.shape(), &x.device, DType::F32.into());
    let db = CubeBackend::float_empty(bias.shape(), &x.device, DType::F32.into());
    let ds = CubeBackend::float_empty(scale.shape(), &x.device, DType::F32.into());
    input_backward_kernel::launch(
        &x.client,
        count(&x, x.shape().num_elements().div_ceil(64)),
        CubeDim::new_1d(64),
        x.clone().into_tensor_arg(),
        bias.clone().into_tensor_arg(),
        scale.clone().into_tensor_arg(),
        y.clone().into_tensor_arg(),
        dy.clone().into_tensor_arg(),
        dx.clone().into_tensor_arg(),
    );
    for (output, is_scale) in [(&db, false), (&ds, true)] {
        parameter_backward_kernel::launch(
            &x.client,
            count(&x, output.shape().num_elements()),
            CubeDim::new_1d(64),
            x.clone().into_tensor_arg(),
            bias.clone().into_tensor_arg(),
            scale.clone().into_tensor_arg(),
            y.clone().into_tensor_arg(),
            dy.clone().into_tensor_arg(),
            dx.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            is_scale,
        );
    }
    (dx, db, ds)
}
