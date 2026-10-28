//! Token-local gate RMS, quaternion map and normalized weighted read with analytic VJPs.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::{prelude::Shape, tensor::DType};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{frontend::Tensor as View, prelude::*};
#[cube]
fn x4(x: &View<f32>, n: usize, s: usize, c: usize) -> usize {
    let t = n % x.shape(1);
    let b = n / x.shape(1);
    b * x.stride(0) + t * x.stride(1) + s * x.stride(2) + c * x.stride(3)
}
#[cube]
fn x3(x: &View<f32>, n: usize, c: usize) -> usize {
    let t = n % x.shape(1);
    let b = n / x.shape(1);
    b * x.stride(0) + t * x.stride(1) + c * x.stride(2)
}
#[cube(launch)]
fn gate_norm(
    x: &View<f32>,
    p: &View<f32>,
    dy: &View<f32>,
    y: &mut View<f32>,
    dx: &mut View<f32>,
    #[comptime] backward: bool,
    #[comptime] lanes: usize,
) {
    let n = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let d = x.shape(3);
    let mut sq = 0.0f32;
    let mut j = lane;
    while j < 4 * d {
        let z = x[x4(x, n, j / d, j % d)];
        sq += z * z;
        j += lanes;
    }
    let mut sums = Shared::<[f32]>::new_slice(lanes);
    let mut dots = Shared::<[f32]>::new_slice(lanes);
    sums[lane] = sq;
    let mut dot = 0.0f32;
    if backward && lane < 16 {
        dot = dy[x3(dy, n, lane)] * p[x3(p, n, lane)];
    }
    dots[lane] = dot;
    sync_cube();
    let mut offset = 0usize;
    offset += lanes / 2;
    while offset > 0 {
        if lane < offset {
            sums[lane] += sums[lane + offset];
            dots[lane] += dots[lane + offset];
        }
        sync_cube();
        offset /= 2;
    }
    let inv = 1.0f32 / f32::sqrt(sums[0] / (4 * d) as f32 + 1e-6f32);
    if lane < 16 {
        if backward {
            y[n * 16 + lane] = dy[x3(dy, n, lane)] * inv;
        } else {
            y[n * 16 + lane] = p[x3(p, n, lane)] * inv;
        }
    }
    if backward {
        let correction = -dots[0] * inv * inv * inv / (4 * d) as f32;
        let mut j = lane;
        while j < 4 * d {
            dx[n * 4 * d + j] = x[x4(x, n, j / d, j % d)] * correction;
            j += lanes;
        }
    }
}
#[cube(launch)]
fn weighted_forward(
    x: &View<f32>,
    a: &View<f32>,
    gamma: &View<f32>,
    y: &mut View<f32>,
    inverse: &mut View<f32>,
    epsilon: f32,
    #[comptime] lanes: usize,
) {
    let n = CUBE_POS as usize;
    let c = UNIT_POS as usize;
    let d = x.shape(3);
    let mut sum = Shared::<[f32]>::new_slice(4 * lanes);
    for s in 0..4 {
        let mut z = 0.0f32;
        if c < d {
            z = x[x4(x, n, s, c)];
        }
        sum[s * lanes + c] = z * z;
    }
    sync_cube();
    let mut offset = 0usize;
    offset += lanes / 2;
    while offset > 0 {
        if c < offset {
            for s in 0..4 {
                sum[s * lanes + c] += sum[s * lanes + c + offset];
            }
        }
        sync_cube();
        offset /= 2;
    }
    let mut value = 0.0f32;
    for s in 0..4 {
        let inv = 1.0f32 / f32::sqrt(sum[s * lanes] / d as f32 + epsilon);
        if c == 0 {
            inverse[n * 4 + s] = inv;
        }
        if c < d {
            value += a[x3(a, n, s)] * x[x4(x, n, s, c)] * inv;
        }
    }
    if c < d {
        y[n * d + c] = value * gamma[c * gamma.stride(0)];
    }
}
#[cube(launch)]
fn weighted_input_vjp(
    x: &View<f32>,
    a: &View<f32>,
    gamma: &View<f32>,
    inv: &View<f32>,
    dy: &View<f32>,
    dx: &mut View<f32>,
    da: &mut View<f32>,
    #[comptime] lanes: usize,
) {
    let n = CUBE_POS as usize;
    let c = UNIT_POS as usize;
    let d = x.shape(3);
    let mut u = 0.0f32;
    if c < d {
        u = dy[x3(dy, n, c)] * gamma[c * gamma.stride(0)];
    }
    let mut dot = Shared::<[f32]>::new_slice(4 * lanes);
    for s in 0..4 {
        let mut v = 0.0f32;
        if c < d {
            v = u * x[x4(x, n, s, c)];
        }
        dot[s * lanes + c] = v;
    }
    sync_cube();
    let mut offset = 0usize;
    offset += lanes / 2;
    while offset > 0 {
        if c < offset {
            for s in 0..4 {
                dot[s * lanes + c] += dot[s * lanes + c + offset];
            }
        }
        sync_cube();
        offset /= 2;
    }
    for s in 0..4 {
        let inverse = inv[n * 4 + s];
        if c == 0 {
            da[n * 4 + s] = dot[s * lanes] * inverse;
        }
        if c < d {
            let z = x[x4(x, n, s, c)];
            dx[(n * 4 + s) * d + c] =
                a[x3(a, n, s)] * inverse * (u - z * dot[s * lanes] * inverse * inverse / d as f32);
        }
    }
}
#[cube(launch)]
fn weighted_gain_vjp(
    x: &View<f32>,
    a: &View<f32>,
    inv: &View<f32>,
    dy: &View<f32>,
    dg: &mut View<f32>,
) {
    let c = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let mut n = lane;
    let mut total = 0.0f32;
    while n < x.shape(0) * x.shape(1) {
        let mut value = 0.0f32;
        for s in 0..4 {
            value += a[x3(a, n, s)] * x[x4(x, n, s, c)] * inv[n * 4 + s];
        }
        total += dy[x3(dy, n, c)] * value;
        n += 64;
    }
    let mut sum = Shared::<[f32]>::new_slice(64usize);
    sum[lane] = total;
    sync_cube();
    let mut offset = 32usize;
    while offset > 0 {
        if lane < offset {
            sum[lane] += sum[lane + offset];
        }
        sync_cube();
        offset /= 2;
    }
    if lane == 0 {
        dg[c] = sum[0];
    }
}
#[cube(launch)]
fn quaternion(
    logits: &View<f32>,
    pattern: &View<f32>,
    dy: &View<f32>,
    out: &mut View<f32>,
    #[comptime] backward: bool,
) {
    let n = ABSOLUTE_POS as usize;
    if n >= logits.shape(0) * logits.shape(1) {
        terminate!();
    }
    let mut raw = Array::<f32>::new(8usize);
    let mut norm = Array::<f32>::new(2usize);
    let mut unit = Array::<f32>::new(8usize);
    let mut grad = Array::<f32>::new(8usize);
    for pair in 0..2 {
        let mut square = 0.0f32;
        for k in 0..4 {
            let j = pair * 4 + k;
            let mut v = logits[x3(logits, n, j)];
            if k == 0 {
                v += 1.0f32;
            }
            raw[j] = v;
            square += v * v;
            grad[j] = 0.0f32;
        }
        norm[pair] = 1.0f32 / f32::sqrt(square + 1e-12f32);
        for k in 0..4 {
            let j = pair * 4 + k;
            unit[j] = raw[j] * norm[pair];
        }
    }
    for i in 0..16 {
        let mut value = 0.0f32;
        for a in 0..4 {
            for b in 0..4 {
                let coefficient = pattern[(a * 4 + b) * pattern.stride(0) + i * pattern.stride(1)];
                if backward {
                    let up = dy[x3(dy, n, i)] * coefficient;
                    grad[a] += up * unit[4 + b];
                    grad[4 + b] += up * unit[a];
                } else {
                    value += coefficient * unit[a] * unit[4 + b];
                }
            }
        }
        if !backward {
            out[n * 16 + i] = value;
        }
    }
    if backward {
        for pair in 0..2 {
            let mut dot = 0.0f32;
            for k in 0..4 {
                let j = pair * 4 + k;
                dot += grad[j] * unit[j];
            }
            for k in 0..4 {
                let j = pair * 4 + k;
                out[n * 8 + j] = (grad[j] - unit[j] * dot) * norm[pair];
            }
        }
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
fn validate(tensors: &[&CubeTensor]) {
    for x in tensors {
        assert_eq!(x.dtype, DType::F32);
        assert_eq!(x.device, tensors[0].device);
        let span = x
            .meta
            .shape()
            .as_slice()
            .iter()
            .zip(x.meta.strides().iter())
            .try_fold(1usize, |acc, (&n, &s)| {
                assert!(n > 0 && s <= u32::MAX as usize);
                acc.checked_add((n - 1).checked_mul(s)?)
            })
            .expect("oHC read address overflow");
        assert!(span <= u32::MAX as usize && x.handle.size() / 4 <= u32::MAX as u64);
    }
}
pub(crate) fn gate(
    x: CubeTensor,
    p: CubeTensor,
    dy: Option<CubeTensor>,
) -> (CubeTensor, CubeTensor) {
    validate(&[&x, &p]);
    let [b, t, s, _] = x.shape().dims::<4>();
    assert_eq!(s, 4);
    assert_eq!(p.shape().dims::<3>(), [b, t, 16]);
    let backward = dy.is_some();
    let dy = dy.unwrap_or_else(|| p.clone());
    validate(&[&x, &dy]);
    assert_eq!(dy.shape(), p.shape());
    let y = empty(&p, p.shape());
    let dx = if backward {
        empty(&x, x.shape())
    } else {
        x.clone()
    };
    let lanes = 256usize;
    gate_norm::launch(
        &x.client,
        count(&x, b * t),
        CubeDim::new_1d(lanes as u32),
        x.clone().into_tensor_arg(),
        p.into_tensor_arg(),
        dy.into_tensor_arg(),
        y.clone().into_tensor_arg(),
        dx.clone().into_tensor_arg(),
        backward,
        lanes,
    );
    (y, dx)
}
pub(crate) fn weighted(
    x: CubeTensor,
    a: CubeTensor,
    g: CubeTensor,
    epsilon: f32,
) -> (CubeTensor, CubeTensor) {
    validate(&[&x, &a, &g]);
    assert!(epsilon.is_finite() && epsilon > 0.0);
    let [b, t, s, d] = x.shape().dims::<4>();
    assert_eq!(s, 4);
    assert!(d <= 1024);
    assert_eq!(a.shape().dims::<3>(), [b, t, 4]);
    assert_eq!(g.shape().dims::<1>(), [d]);
    let y = empty(&x, [b, t, d]);
    let inv = empty(&x, [b, t, 4]);
    let lanes = d.next_power_of_two();
    weighted_forward::launch(
        &x.client,
        count(&x, b * t),
        CubeDim::new_1d(lanes as u32),
        x.clone().into_tensor_arg(),
        a.into_tensor_arg(),
        g.into_tensor_arg(),
        y.clone().into_tensor_arg(),
        inv.clone().into_tensor_arg(),
        epsilon,
        lanes,
    );
    (y, inv)
}
pub(crate) fn weighted_backward(
    x: CubeTensor,
    a: CubeTensor,
    g: CubeTensor,
    inv: CubeTensor,
    dy: CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    validate(&[&x, &a, &g, &inv, &dy]);
    let [b, t, _, d] = x.shape().dims::<4>();
    assert_eq!(dy.shape().dims::<3>(), [b, t, d]);
    let (dx, da, dg) = (
        empty(&x, x.shape()),
        empty(&a, a.shape()),
        empty(&g, g.shape()),
    );
    weighted_input_vjp::launch(
        &x.client,
        count(&x, b * t),
        CubeDim::new_1d(d.next_power_of_two() as u32),
        x.clone().into_tensor_arg(),
        a.clone().into_tensor_arg(),
        g.into_tensor_arg(),
        inv.clone().into_tensor_arg(),
        dy.clone().into_tensor_arg(),
        dx.clone().into_tensor_arg(),
        da.clone().into_tensor_arg(),
        d.next_power_of_two(),
    );
    weighted_gain_vjp::launch(
        &x.client,
        count(&x, d),
        CubeDim::new_1d(64),
        x.clone().into_tensor_arg(),
        a.into_tensor_arg(),
        inv.into_tensor_arg(),
        dy.into_tensor_arg(),
        dg.clone().into_tensor_arg(),
    );
    (dx, da, dg)
}
pub(crate) fn rotation(x: CubeTensor, pattern: CubeTensor, dy: Option<CubeTensor>) -> CubeTensor {
    validate(&[&x, &pattern]);
    let [b, t, k] = x.shape().dims::<3>();
    assert_eq!(k, 8);
    assert_eq!(pattern.shape().dims::<2>(), [16, 16]);
    let backward = dy.is_some();
    let dy = dy.unwrap_or_else(|| x.clone());
    if backward {
        assert_eq!(dy.shape().dims::<3>(), [b, t, 16]);
        validate(&[&x, &dy]);
    }
    let out = empty(&x, [b, t, if backward { 8 } else { 16 }]);
    quaternion::launch(
        &x.client,
        count(&x, (b * t).div_ceil(64)),
        CubeDim::new_1d(64),
        x.clone().into_tensor_arg(),
        pattern.into_tensor_arg(),
        dy.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        backward,
    );
    out
}
