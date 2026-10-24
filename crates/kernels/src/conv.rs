//! Short depthwise causal convolution. Masks are metadata, not differentiable inputs.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::tensor::DType;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;
#[cube]
fn ix(x: &View<f32>, b: usize, t: usize, c: usize) -> usize {
    b * x.stride(0) + t * x.stride(1) + c * x.stride(2)
}
#[cube]
fn mask(valid: &View<f32>, docs: &View<f32>, b: usize, source: usize, dest: usize) -> f32 {
    let s = b * docs.stride(0) + source * docs.stride(1);
    let d = b * docs.stride(0) + dest * docs.stride(1);
    if docs[s] == docs[d] {
        valid[b * valid.stride(0) + source * valid.stride(1)]
            * valid[b * valid.stride(0) + dest * valid.stride(1)]
    } else {
        0.0f32
    }
}
#[cube(launch)]
fn forward_kernel(
    x: &View<f32>,
    w: &View<f32>,
    valid: &View<f32>,
    docs: &View<f32>,
    y: &mut View<f32>,
) {
    let i = ABSOLUTE_POS as usize;
    if i >= y.len() {
        terminate!();
    }
    let c = i % x.shape(2);
    let t = i / x.shape(2) % x.shape(1);
    let b = i / (x.shape(2) * x.shape(1));
    let mut sum = 0.0f32;
    for lag in 0..w.shape(0) {
        if lag <= t {
            sum += x[ix(x, b, t - lag, c)]
                * w[lag * w.stride(0) + c * w.stride(1)]
                * mask(valid, docs, b, t - lag, t);
        }
    }
    y[i] = sum;
}
#[cube(launch)]
fn input_backward_kernel(
    x: &View<f32>,
    w: &View<f32>,
    valid: &View<f32>,
    docs: &View<f32>,
    dy: &View<f32>,
    dx: &mut View<f32>,
) {
    let i = ABSOLUTE_POS as usize;
    if i >= dx.len() {
        terminate!();
    }
    let c = i % x.shape(2);
    let t = i / x.shape(2) % x.shape(1);
    let b = i / (x.shape(2) * x.shape(1));
    let mut sum = 0.0f32;
    for lag in 0..w.shape(0) {
        if lag < x.shape(1) - t {
            sum += dy[ix(dy, b, t + lag, c)]
                * w[lag * w.stride(0) + c * w.stride(1)]
                * mask(valid, docs, b, t, t + lag);
        }
    }
    dx[i] = sum;
}
#[cube(launch)]
fn weight_backward_kernel(
    x: &View<f32>,
    valid: &View<f32>,
    docs: &View<f32>,
    dy: &View<f32>,
    dw: &mut View<f32>,
) {
    let parameter = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let c = parameter % x.shape(2);
    let lag = parameter / x.shape(2);
    let rows = x.shape(0) * x.shape(1);
    let mut j = lane;
    let mut sum = 0.0f32;
    while j < rows {
        let b = j / x.shape(1);
        let t = j % x.shape(1);
        if t >= lag {
            sum += dy[ix(dy, b, t, c)] * x[ix(x, b, t - lag, c)] * mask(valid, docs, b, t - lag, t);
        }
        j += 64;
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
        dw[parameter] = reduction[0];
    }
}
fn validate(x: &CubeTensor, w: &CubeTensor, valid: &CubeTensor, docs: &CubeTensor) {
    let [b, t, c] = x.shape().dims::<3>();
    assert!(b > 0 && t > 0 && c > 0);
    let [width, wc] = w.shape().dims::<2>();
    assert!(width > 0);
    assert_eq!(wc, c);
    assert_eq!(valid.shape().dims::<2>(), [b, t]);
    assert_eq!(docs.shape().dims::<2>(), [b, t]);
    validate_views(x, &[x, w, valid, docs]);
}
fn validate_views(x: &CubeTensor, views: &[&CubeTensor]) {
    for a in views {
        assert_eq!(a.dtype, DType::F32);
        assert_eq!(a.device, x.device);
        assert!(a.shape().num_elements() <= u32::MAX as usize);
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
                .expect("convolution address overflow");
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
pub(crate) fn forward(
    x: CubeTensor,
    w: CubeTensor,
    valid: CubeTensor,
    docs: CubeTensor,
) -> CubeTensor {
    validate(&x, &w, &valid, &docs);
    let y = CubeBackend::float_empty(x.shape(), &x.device, DType::F32.into());
    forward_kernel::launch(
        &x.client,
        count(&x, x.shape().num_elements().div_ceil(64)),
        CubeDim::new_1d(64),
        x.clone().into_tensor_arg(),
        w.into_tensor_arg(),
        valid.into_tensor_arg(),
        docs.into_tensor_arg(),
        y.clone().into_tensor_arg(),
    );
    y
}
pub(crate) fn backward(
    x: CubeTensor,
    w: CubeTensor,
    valid: CubeTensor,
    docs: CubeTensor,
    dy: CubeTensor,
) -> (CubeTensor, CubeTensor) {
    validate(&x, &w, &valid, &docs);
    assert_eq!(dy.shape(), x.shape());
    validate_views(&x, &[&dy]);
    let dx = CubeBackend::float_empty(x.shape(), &x.device, DType::F32.into());
    let dw = CubeBackend::float_empty(w.shape(), &x.device, DType::F32.into());
    input_backward_kernel::launch(
        &x.client,
        count(&x, x.shape().num_elements().div_ceil(64)),
        CubeDim::new_1d(64),
        x.clone().into_tensor_arg(),
        w.clone().into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        docs.clone().into_tensor_arg(),
        dy.clone().into_tensor_arg(),
        dx.clone().into_tensor_arg(),
    );
    weight_backward_kernel::launch(
        &x.client,
        count(&x, w.shape().num_elements()),
        CubeDim::new_1d(64),
        x.clone().into_tensor_arg(),
        valid.into_tensor_arg(),
        docs.into_tensor_arg(),
        dy.into_tensor_arg(),
        dw.clone().into_tensor_arg(),
    );
    (dx, dw)
}
