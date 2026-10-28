//! Row-local L2/RMS normalization and sigmoid-gated RMSNorm, with analytic VJPs.
//! Modes: 0 = L2, 1 = learned RMS, 2 = sigmoid gate then learned RMS.
//! Inactive weight/gate arguments alias x; they allocate nothing and have no AD edge.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::{prelude::Shape, tensor::DType};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;

#[cube]
fn ix(x: &View<f32>, row: usize, c: usize) -> usize {
    let h = row % x.shape(2);
    let t = row / x.shape(2) % x.shape(1);
    let b = row / (x.shape(2) * x.shape(1));
    b * x.stride(0) + t * x.stride(1) + h * x.stride(2) + c * x.stride(3)
}
#[cube]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + f32::exp(-x))
}

#[cube(launch)]
fn forward_kernel(
    x: &View<f32>,
    weight: &View<f32>,
    gate: &View<f32>,
    y: &mut View<f32>,
    inverse: &mut View<f32>,
    #[comptime] mode: usize,
    epsilon: f32,
    #[comptime] lanes: usize,
) {
    let row = CUBE_POS as usize;
    let c = UNIT_POS as usize;
    let width = x.shape(3);
    if row >= x.shape(0) * x.shape(1) * x.shape(2) {
        terminate!();
    }
    let mut z = 0.0f32;
    if c < width {
        z = x[ix(x, row, c)];
        if mode == 2 {
            z *= sigmoid(gate[ix(gate, row, c)]);
        }
    }
    let mut sum = Shared::<[f32]>::new_slice(lanes);
    sum[c] = z * z;
    sync_cube();
    // A mutable comptime expression is not a runtime variable in CubeCL 0.11.
    let mut offset = 0usize;
    offset += lanes / 2;
    while offset > 0 {
        if c < offset {
            sum[c] += sum[c + offset];
        }
        sync_cube();
        offset /= 2;
    }
    let mut square = sum[0];
    if mode != 0 {
        square /= width as f32;
    }
    let inv = if mode == 0 {
        1.0f32 / f32::sqrt(f32::max(square, epsilon * epsilon))
    } else {
        1.0f32 / f32::sqrt(square + epsilon)
    };
    if c == 0 {
        inverse[row] = inv;
    }
    if c < width {
        let mut value = z * inv;
        if mode != 0 {
            value *= weight[c * weight.stride(0)];
        }
        y[row * width + c] = value;
    }
}

#[cube(launch)]
fn input_backward_kernel(
    x: &View<f32>,
    weight: &View<f32>,
    gate: &View<f32>,
    inverse: &View<f32>,
    upstream: &View<f32>,
    dx: &mut View<f32>,
    dg: &mut View<f32>,
    #[comptime] mode: usize,
    #[comptime] lanes: usize,
) {
    let row = CUBE_POS as usize;
    let c = UNIT_POS as usize;
    let width = x.shape(3);
    if row >= x.shape(0) * x.shape(1) * x.shape(2) {
        terminate!();
    }
    let mut z = 0.0f32;
    let mut u = 0.0f32;
    let mut s = 1.0f32;
    if c < width {
        z = x[ix(x, row, c)];
        if mode == 2 {
            s = sigmoid(gate[ix(gate, row, c)]);
            z *= s;
        }
        u = upstream[ix(upstream, row, c)];
        if mode != 0 {
            u *= weight[c * weight.stride(0)];
        }
    }
    let mut dot = Shared::<[f32]>::new_slice(if mode == 0 { lanes * 2 } else { lanes });
    dot[c] = u * z;
    if mode == 0 {
        dot[lanes + c] = z * z;
    }
    sync_cube();
    let mut offset = 0usize;
    offset += lanes / 2;
    while offset > 0 {
        if c < offset {
            dot[c] += dot[c + offset];
            if mode == 0 {
                dot[lanes + c] += dot[lanes + c + offset];
            }
        }
        sync_cube();
        offset /= 2;
    }
    let inv = inverse[row];
    let mut correction = dot[0] * inv * inv;
    if mode != 0 {
        correction /= width as f32;
    }
    if mode == 0 {
        if dot[lanes] < 1.0e-24f32 {
            // Below the norm floor the denominator is constant, including at zero.
            correction = 0.0f32;
        }
    }
    if c < width {
        let dz = (u - z * correction) * inv;
        dx[row * width + c] = dz * s;
        if mode == 2 {
            dg[row * width + c] = dz * z * (1.0f32 - s);
        }
    }
}

#[cube(launch)]
fn weight_backward_kernel(
    x: &View<f32>,
    gate: &View<f32>,
    inverse: &View<f32>,
    upstream: &View<f32>,
    dw: &mut View<f32>,
    #[comptime] gated: bool,
) {
    let c = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let width = x.shape(3);
    if c >= width {
        terminate!();
    }
    let mut sum = 0.0f32;
    let mut row = lane;
    while row < x.shape(0) * x.shape(1) * x.shape(2) {
        let mut z = x[ix(x, row, c)];
        if gated {
            z *= sigmoid(gate[ix(gate, row, c)]);
        }
        sum += upstream[ix(upstream, row, c)] * z * inverse[row];
        row += 64;
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
        dw[c] = reduction[0];
    }
}

fn validate(tensors: &[&CubeTensor]) {
    for x in tensors {
        assert_eq!(x.dtype, DType::F32, "fused normalization requires f32");
        assert_eq!(x.device, tensors[0].device, "normalization device mismatch");
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
                .expect("normalization address overflow");
        }
        assert!(
            span <= u32::MAX as usize && x.handle.size() / 4 <= u32::MAX as u64,
            "normalization exceeds 32-bit addressing"
        );
    }
}
fn empty(x: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor {
    let shape = shape.into();
    assert!(shape.num_elements() <= u32::MAX as usize);
    CubeBackend::float_empty(shape, &x.device, DType::F32.into())
}
fn count(x: &CubeTensor, groups: usize) -> CubeCount {
    let (max_x, max_y, _) = x.client.properties().hardware.max_cube_count;
    let n = groups.min(max_x.min(65535) as usize).max(1);
    assert!(groups.div_ceil(n) <= max_y as usize);
    CubeCount::Static(n as u32, groups.div_ceil(n) as u32, 1)
}
pub(crate) fn forward(
    x: CubeTensor,
    weight: CubeTensor,
    gate: CubeTensor,
    mode: usize,
    epsilon: f32,
) -> (CubeTensor, CubeTensor) {
    validate(&[&x, &weight, &gate]);
    assert!(mode <= 2 && epsilon.is_finite() && epsilon > 0.);
    let [b, t, h, width] = x.shape().dims::<4>();
    assert!(width <= 1024, "fused normalization width must be <=1024");
    if mode != 0 {
        assert_eq!(weight.shape().dims::<1>(), [width]);
    }
    if mode == 2 {
        assert_eq!(gate.shape(), x.shape());
    }
    let lanes = width.next_power_of_two().max(32);
    let y = empty(&x, x.shape());
    let inverse = empty(&x, [b * t * h]);
    forward_kernel::launch(
        &x.client,
        count(&x, b * t * h),
        CubeDim::new_1d(lanes as u32),
        x.clone().into_tensor_arg(),
        weight.into_tensor_arg(),
        gate.into_tensor_arg(),
        y.clone().into_tensor_arg(),
        inverse.clone().into_tensor_arg(),
        mode,
        epsilon,
        lanes,
    );
    (y, inverse)
}
pub(crate) fn backward(
    x: CubeTensor,
    weight: CubeTensor,
    gate: CubeTensor,
    inverse: CubeTensor,
    upstream: CubeTensor,
    mode: usize,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    validate(&[&x, &weight, &gate, &inverse, &upstream]);
    assert_eq!(upstream.shape(), x.shape());
    let [b, t, h, width] = x.shape().dims::<4>();
    let dx = empty(&x, x.shape());
    // Unused results alias dx, with neither an allocation nor an autodiff parent.
    let dg = if mode == 2 {
        empty(&x, gate.shape())
    } else {
        dx.clone()
    };
    let dw = if mode != 0 {
        empty(&x, weight.shape())
    } else {
        dx.clone()
    };
    let lanes = width.next_power_of_two().max(32);
    input_backward_kernel::launch(
        &x.client,
        count(&x, b * t * h),
        CubeDim::new_1d(lanes as u32),
        x.clone().into_tensor_arg(),
        weight.into_tensor_arg(),
        gate.clone().into_tensor_arg(),
        inverse.clone().into_tensor_arg(),
        upstream.clone().into_tensor_arg(),
        dx.clone().into_tensor_arg(),
        dg.clone().into_tensor_arg(),
        mode,
        lanes,
    );
    if mode != 0 {
        weight_backward_kernel::launch(
            &x.client,
            count(&x, width),
            CubeDim::new_1d(64),
            x.clone().into_tensor_arg(),
            gate.into_tensor_arg(),
            inverse.into_tensor_arg(),
            upstream.into_tensor_arg(),
            dw.clone().into_tensor_arg(),
            mode == 2,
        );
    }
    (dx, dw, dg)
}
