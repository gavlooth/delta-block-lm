//! Stable row cross entropy without a vocabulary-sized log-softmax or saved probability buffer.
use crate::update::{empty, validate};
use burn::backend::TensorMetadata;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{frontend::Tensor as View, prelude::*};
#[cube(launch)]
fn forward_kernel(
    x: &View<f32>,
    target: &View<f32>,
    loss: &mut View<f32>,
    stats: &mut View<f32>,
    #[comptime] lanes: usize,
) {
    let row = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let width = x.shape(1);
    let mut maximum = 0.0f32;
    maximum -= f32::MAX;
    let mut c = lane;
    while c < width {
        maximum = f32::max(maximum, x[row * x.stride(0) + c * x.stride(1)]);
        c += lanes;
    }
    let mut shared = Shared::<[f32]>::new_slice(lanes);
    shared[lane] = maximum;
    sync_cube();
    let mut stride = 0usize;
    stride += lanes / 2;
    while stride > 0 {
        if lane < stride {
            shared[lane] = f32::max(shared[lane], shared[lane + stride]);
        }
        sync_cube();
        stride /= 2;
    }
    maximum = shared[0];
    // Every lane captures the maximum before the shared reduction buffer is reused.
    sync_cube();
    let mut sum = 0.0f32;
    c = lane;
    while c < width {
        sum += f32::exp(x[row * x.stride(0) + c * x.stride(1)] - maximum);
        c += lanes;
    }
    shared[lane] = sum;
    sync_cube();
    stride = lanes / 2;
    while stride > 0 {
        if lane < stride {
            shared[lane] += shared[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        let logsum = f32::ln(shared[0]);
        let label = target[row * target.stride(0)] as usize;
        loss[row] = (maximum - x[row * x.stride(0) + label * x.stride(1)]) + logsum;
        stats[row * 2] = maximum;
        stats[row * 2 + 1] = logsum;
    }
}
#[cube(launch)]
fn backward_kernel(
    x: &View<f32>,
    target: &View<f32>,
    stats: &View<f32>,
    upstream: &View<f32>,
    dx: &mut View<f32>,
) {
    let i = ABSOLUTE_POS as usize;
    if i < x.len() {
        let row = i / x.shape(1);
        let c = i % x.shape(1);
        let mut probability = f32::exp(
            (x[row * x.stride(0) + c * x.stride(1)] - stats[row * 2]) - stats[row * 2 + 1],
        );
        if c == target[row * target.stride(0)] as usize {
            probability -= 1.0f32;
        }
        dx[i] = probability * upstream[row * upstream.stride(0)];
    }
}
pub(crate) fn forward(x: CubeTensor, target: CubeTensor, lanes: usize) -> (CubeTensor, CubeTensor) {
    validate(&[&x, &target]);
    let [rows, width] = x.shape().dims::<2>();
    assert_eq!(target.shape().dims::<1>(), [rows]);
    assert!(width <= 16_777_216 && [64, 256].contains(&lanes));
    let loss = empty(&x, [rows]);
    let stats = empty(&x, [rows, 2]);
    assert!(rows <= x.client.properties().hardware.max_cube_count.0 as usize);
    forward_kernel::launch(
        &x.client,
        CubeCount::Static(rows as u32, 1, 1),
        CubeDim::new_1d(lanes as u32),
        x.clone().into_tensor_arg(),
        target.into_tensor_arg(),
        loss.clone().into_tensor_arg(),
        stats.clone().into_tensor_arg(),
        lanes,
    );
    (loss, stats)
}
pub(crate) fn backward(
    x: CubeTensor,
    target: CubeTensor,
    stats: CubeTensor,
    upstream: CubeTensor,
) -> CubeTensor {
    validate(&[&x, &target, &stats, &upstream]);
    let dx = empty(&x, x.shape());
    backward_kernel::launch(
        &x.client,
        CubeCount::Static(x.shape().num_elements().div_ceil(256) as u32, 1, 1),
        CubeDim::new_1d(256),
        x.clone().into_tensor_arg(),
        target.into_tensor_arg(),
        stats.into_tensor_arg(),
        upstream.into_tensor_arg(),
        dx.clone().into_tensor_arg(),
    );
    dx
}
