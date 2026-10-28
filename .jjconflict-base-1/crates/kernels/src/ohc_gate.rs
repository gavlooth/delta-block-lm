//! Skinny gate projection and flattened RMS with deterministic analytic gradients.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::{prelude::Shape, tensor::DType};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{frontend::Tensor as View, prelude::*};

#[cube]
fn x_index(x: &View<f32>, n: usize, k: usize) -> usize {
    let d = x.shape(3);
    n / x.shape(1) * x.stride(0)
        + n % x.shape(1) * x.stride(1)
        + k / d * x.stride(2)
        + k % d * x.stride(3)
}
#[cube]
fn y_index(y: &View<f32>, n: usize, j: usize) -> usize {
    n / y.shape(1) * y.stride(0) + n % y.shape(1) * y.stride(1) + j * y.stride(2)
}

#[cube(launch)]
fn project(x: &View<f32>, w: &View<f32>, y: &mut View<f32>) {
    let n = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let j = lane % 16;
    let r = lane / 16;
    let mut dot = 0.0f32;
    let mut square = 0.0f32;
    let mut k = r;
    while k < 4 * x.shape(3) {
        let v = x[x_index(x, n, k)];
        dot += v * w[k * w.stride(0) + j * w.stride(1)];
        square += v * v;
        k += 16;
    }
    let mut sums = Shared::<[f32]>::new_slice(256usize);
    let mut squares = Shared::<[f32]>::new_slice(256usize);
    sums[lane] = dot;
    squares[lane] = square;
    sync_cube();
    let mut offset = 8usize;
    while offset > 0 {
        if r < offset {
            sums[lane] += sums[lane + offset * 16];
            squares[lane] += squares[lane + offset * 16];
        }
        sync_cube();
        offset /= 2;
    }
    if r == 0 {
        let inv = 1.0f32 / f32::sqrt(squares[j] / (4 * x.shape(3)) as f32 + 1e-6f32);
        y[n * 16 + j] = sums[j] * inv;
    }
}

#[cube(launch)]
fn input_vjp(
    x: &View<f32>,
    w: &View<f32>,
    y: &View<f32>,
    dy: &View<f32>,
    dx: &mut View<f32>,
    dp: &mut View<f32>,
) {
    let n = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let width = 4 * x.shape(3);
    let mut square = 0.0f32;
    let mut k = lane;
    while k < width {
        let v = x[x_index(x, n, k)];
        square += v * v;
        k += 256;
    }
    let mut sums = Shared::<[f32]>::new_slice(256usize);
    let mut dots = Shared::<[f32]>::new_slice(256usize);
    let mut adj = Shared::<[f32]>::new_slice(16usize);
    sums[lane] = square;
    let mut dot = 0.0f32;
    if lane < 16 {
        let v = dy[y_index(dy, n, lane)];
        dot = v * y[y_index(y, n, lane)];
        adj[lane] = v;
    }
    dots[lane] = dot;
    sync_cube();
    let mut offset = 128usize;
    while offset > 0 {
        if lane < offset {
            sums[lane] += sums[lane + offset];
            dots[lane] += dots[lane + offset];
        }
        sync_cube();
        offset /= 2;
    }
    let inv = 1.0f32 / f32::sqrt(sums[0] / width as f32 + 1e-6f32);
    let correction = dots[0] * inv * inv / width as f32;
    if lane < 16 {
        adj[lane] *= inv;
        dp[n * 16 + lane] = adj[lane];
    }
    sync_cube();
    k = lane;
    while k < width {
        let mut value = 0.0f32;
        for j in 0..16 {
            value += adj[j] * w[k * w.stride(0) + j * w.stride(1)];
        }
        dx[n * width + k] = value - x[x_index(x, n, k)] * correction;
        k += 256;
    }
}

// Each group accumulates a 16-by-16 weight tile over 128 tokens. Shared input
// and adjoint tiles avoid rereading each scalar for all 16 output channels.
#[cube(launch)]
fn weight_partials(x: &View<f32>, dp: &View<f32>, partial: &mut View<f32>) {
    let tile = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let width = 4 * x.shape(3);
    let tiles_k = (width + 15) / 16;
    let start_k = tile % tiles_k * 16;
    let part = tile / tiles_k;
    let k = start_k + lane / 16;
    let j = lane % 16;
    let n_total = x.shape(0) * x.shape(1);
    let mut xs = Shared::<[f32]>::new_slice(512usize);
    let mut ps = Shared::<[f32]>::new_slice(512usize);
    let mut value = 0.0f32;
    for chunk in 0..4 {
        for half in 0..2 {
            let pos = lane + half * 256;
            let n = part * 128 + chunk * 32 + pos / 16;
            let c = start_k + pos % 16;
            let mut xv = 0.0f32;
            let mut pv = 0.0f32;
            if n < n_total {
                if c < width {
                    xv = x[x_index(x, n, c)];
                }
                pv = dp[n * 16 + pos % 16];
            }
            xs[pos] = xv;
            ps[pos] = pv;
        }
        sync_cube();
        for row in 0..32 {
            value += xs[row * 16 + lane / 16] * ps[row * 16 + j];
        }
        sync_cube();
    }
    if k < width {
        partial[(part * width + k) * 16 + j] = value;
    }
}
#[cube(launch)]
fn weight_sum(partial: &View<f32>, dw: &mut View<f32>) {
    let i = ABSOLUTE_POS as usize;
    if i < dw.shape(0) * 16 {
        let mut value = 0.0f32;
        for part in 0..partial.shape(0) {
            value += partial[part * dw.shape(0) * 16 + i];
        }
        dw[i] = value;
    }
}
fn empty(x: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor {
    CubeBackend::float_empty(shape.into(), &x.device, DType::F32.into())
}
fn count(x: &CubeTensor, n: usize) -> CubeCount {
    let (mx, my, _) = x.client.properties().hardware.max_cube_count;
    let nx = n.min(mx.min(65535) as usize).max(1);
    assert!(n.div_ceil(nx) <= my as usize);
    CubeCount::Static(nx as u32, n.div_ceil(nx) as u32, 1)
}
fn validate(x: &CubeTensor, w: &CubeTensor) -> (usize, usize, usize) {
    assert_eq!(x.dtype, DType::F32);
    assert_eq!(w.dtype, DType::F32);
    assert_eq!(x.device, w.device);
    let [b, t, s, d] = x.shape().dims::<4>();
    assert_eq!(s, 4);
    assert!(b > 0 && t > 0 && d > 0 && d <= 1024);
    assert_eq!(w.shape().dims::<2>(), [4 * d, 16]);
    (b, t, d)
}
pub(crate) fn forward(x: CubeTensor, w: CubeTensor) -> CubeTensor {
    let (b, t, _) = validate(&x, &w);
    let y = empty(&x, [b, t, 16]);
    project::launch(
        &x.client,
        count(&x, b * t),
        CubeDim::new_1d(256),
        x.clone().into_tensor_arg(),
        w.into_tensor_arg(),
        y.clone().into_tensor_arg(),
    );
    y
}
pub(crate) fn backward(
    x: CubeTensor,
    w: CubeTensor,
    y: CubeTensor,
    dy: CubeTensor,
) -> (CubeTensor, CubeTensor) {
    let (b, t, d) = validate(&x, &w);
    assert_eq!(y.shape().dims::<3>(), [b, t, 16]);
    assert_eq!(dy.shape(), y.shape());
    assert_eq!(y.dtype, DType::F32);
    assert_eq!(dy.dtype, DType::F32);
    assert_eq!(y.device, x.device);
    assert_eq!(dy.device, x.device);
    let dx = empty(&x, x.shape());
    let dp = empty(&x, [b, t, 16]);
    let dw = empty(&w, [4 * d, 16]);
    let parts = (b * t).div_ceil(128);
    let partial = empty(&x, [parts, 4 * d, 16]);
    input_vjp::launch(
        &x.client,
        count(&x, b * t),
        CubeDim::new_1d(256),
        x.clone().into_tensor_arg(),
        w.into_tensor_arg(),
        y.into_tensor_arg(),
        dy.into_tensor_arg(),
        dx.clone().into_tensor_arg(),
        dp.clone().into_tensor_arg(),
    );
    weight_partials::launch(
        &x.client,
        count(&x, parts * (4 * d).div_ceil(16)),
        CubeDim::new_1d(256),
        x.clone().into_tensor_arg(),
        dp.into_tensor_arg(),
        partial.clone().into_tensor_arg(),
    );
    weight_sum::launch(
        &x.client,
        count(&x, (4 * d * 16).div_ceil(256)),
        CubeDim::new_1d(256),
        partial.into_tensor_arg(),
        dw.clone().into_tensor_arg(),
    );
    (dx, dw)
}
