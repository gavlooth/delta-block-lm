//! Per-token oHC write and analytic VJP. No token or document mixing.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::tensor::DType;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;

#[cube]
fn at4(x: &View<f32>, n: usize, s: usize, c: usize) -> usize {
    let t = n % x.shape(1);
    let b = n / x.shape(1);
    b * x.stride(0) + t * x.stride(1) + s * x.stride(2) + c * x.stride(3)
}
#[cube]
fn at3(x: &View<f32>, n: usize, c: usize) -> usize {
    let t = n % x.shape(1);
    let b = n / x.shape(1);
    b * x.stride(0) + t * x.stride(1) + c * x.stride(2)
}
#[cube(launch)]
fn write(x: &View<f32>, q: &View<f32>, w: &View<f32>, f: &View<f32>, y: &mut View<f32>) {
    let i = ABSOLUTE_POS as usize;
    if i >= y.len() {
        terminate!();
    }
    let d = x.shape(3);
    let c = i % d;
    let s = i / d % 4;
    let n = i / (d * 4);
    let mut value = w[at3(w, n, s)] * f[at3(f, n, c)];
    for k in 0..4 {
        value += q[at4(q, n, s, k)] * x[at4(x, n, k, c)];
    }
    y[i] = value;
}
#[cube(launch)]
fn input_vjp(q: &View<f32>, w: &View<f32>, dy: &View<f32>, dx: &mut View<f32>, df: &mut View<f32>) {
    let i = ABSOLUTE_POS as usize;
    if i >= dx.len() {
        terminate!();
    }
    let d = dx.shape(3);
    let c = i % d;
    let s = i / d % 4;
    let n = i / (d * 4);
    let mut value = 0.0f32;
    for k in 0..4 {
        value += q[at4(q, n, k, s)] * dy[at4(dy, n, k, c)];
    }
    dx[i] = value;
    if s == 0 {
        let mut value = 0.0f32;
        for k in 0..4 {
            value += w[at3(w, n, k)] * dy[at4(dy, n, k, c)];
        }
        df[n * d + c] = value;
    }
}
#[cube(launch)]
fn gate_vjp(x: &View<f32>, f: &View<f32>, dy: &View<f32>, dq: &mut View<f32>, dw: &mut View<f32>) {
    let i = ABSOLUTE_POS as usize;
    if i >= dq.len() {
        terminate!();
    }
    let k = i % 4;
    let s = i / 4 % 4;
    let n = i / 16;
    let mut value = 0.0f32;
    let mut write_value = 0.0f32;
    for c in 0..x.shape(3) {
        let upstream = dy[at4(dy, n, s, c)];
        value += upstream * x[at4(x, n, k, c)];
        if k == 0 {
            write_value += upstream * f[at3(f, n, c)];
        }
    }
    dq[i] = value;
    if k == 0 {
        dw[n * 4 + s] = write_value;
    }
}
fn grid(x: &CubeTensor, n: usize) -> CubeCount {
    let groups = n.div_ceil(256);
    let (mx, my, _) = x.client.properties().hardware.max_cube_count;
    let nx = groups.min(mx.min(65535) as usize).max(1);
    assert!(groups.div_ceil(nx) <= my as usize);
    CubeCount::Static(nx as u32, groups.div_ceil(nx) as u32, 1)
}
fn validate(x: &CubeTensor, q: &CubeTensor, w: &CubeTensor, f: &CubeTensor) {
    let [b, t, s, d] = x.shape().dims::<4>();
    assert_eq!(s, 4);
    assert!(b > 0 && t > 0 && d > 0);
    assert_eq!(q.shape().dims::<4>(), [b, t, 4, 4]);
    assert_eq!(w.shape().dims::<3>(), [b, t, 4]);
    assert_eq!(f.shape().dims::<3>(), [b, t, d]);
    for v in [x, q, w, f] {
        assert_eq!(v.dtype, DType::F32);
        assert_eq!(v.device, x.device);
        let span = v
            .meta
            .shape()
            .as_slice()
            .iter()
            .zip(v.meta.strides().iter())
            .try_fold(1usize, |acc, (&n, &stride)| {
                acc.checked_add((n - 1).checked_mul(stride)?)
            })
            .expect("oHC address overflow");
        assert!(span <= u32::MAX as usize);
    }
}
pub(crate) fn forward(x: CubeTensor, q: CubeTensor, w: CubeTensor, f: CubeTensor) -> CubeTensor {
    validate(&x, &q, &w, &f);
    let y = CubeBackend::float_empty(x.shape(), &x.device, DType::F32.into());
    write::launch(
        &x.client,
        grid(&x, y.shape().num_elements()),
        CubeDim::new_1d(256),
        x.clone().into_tensor_arg(),
        q.into_tensor_arg(),
        w.into_tensor_arg(),
        f.into_tensor_arg(),
        y.clone().into_tensor_arg(),
    );
    y
}
pub(crate) fn backward(
    x: CubeTensor,
    q: CubeTensor,
    w: CubeTensor,
    f: CubeTensor,
    dy: CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
    validate(&x, &q, &w, &f);
    assert_eq!(x.shape(), dy.shape());
    assert_eq!(x.device, dy.device);
    assert_eq!(dy.dtype, DType::F32);
    let empty = |v: &CubeTensor| CubeBackend::float_empty(v.shape(), &v.device, DType::F32.into());
    let (dx, dq, dw, df) = (empty(&x), empty(&q), empty(&w), empty(&f));
    input_vjp::launch(
        &x.client,
        grid(&x, dx.shape().num_elements()),
        CubeDim::new_1d(256),
        q.into_tensor_arg(),
        w.into_tensor_arg(),
        dy.clone().into_tensor_arg(),
        dx.clone().into_tensor_arg(),
        df.clone().into_tensor_arg(),
    );
    gate_vjp::launch(
        &x.client,
        grid(&x, dq.shape().num_elements()),
        CubeDim::new_1d(256),
        x.clone().into_tensor_arg(),
        f.into_tensor_arg(),
        dy.into_tensor_arg(),
        dq.clone().into_tensor_arg(),
        dw.clone().into_tensor_arg(),
    );
    (dx, dq, dw, df)
}
