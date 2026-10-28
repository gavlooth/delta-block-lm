//! Differentiable chunkwise triangular solves. Only chunk boundary states recur.
//! Products are formed directly, so packed-document resets never divide by zero.
use crate::{MixerKind, ScanBackend};
use burn::prelude::*;

pub const MAX_CHUNK_SIZE: usize = 16;

#[derive(Clone, Debug)]
pub struct ScanInput {
    pub q: Tensor<4>,
    pub k: Tensor<4>,
    pub v: Tensor<4>,
    pub alpha: Tensor<4>,
    /// Scalar beta per head.
    pub erase: Tensor<4>,
    /// Scalar target per head.
    pub write: Tensor<4>,
}

#[derive(Clone, Debug)]
pub struct ScanOutput {
    pub output: Tensor<4>,
    pub final_state: Tensor<4>,
    /// Norm of the active edit AFTER decay, not norm of the total state change.
    pub scores: Tensor<4>,
}

/// Explicit dispatch; requesting fused execution never falls back to tensor operations.
pub fn scan_selected(
    backend: ScanBackend,
    kind: MixerKind,
    input: ScanInput,
    valid: Tensor<2>,
    reset: Tensor<2>,
    chunk_size: usize,
    initial: Option<Tensor<4>>,
) -> ScanOutput {
    match backend {
        ScanBackend::TensorSolver => scan(kind, input, valid, reset, chunk_size, initial),
        ScanBackend::FusedGpu | ScanBackend::FusedParallel128 | ScanBackend::FusedParallel256 => {
            assert_fused_device(&input.q.device());
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                let [batch, _, heads, keys] = input.q.dims();
                let values = input.v.dims()[3];
                let initial = initial.unwrap_or_else(|| {
                    Tensor::zeros([batch, heads, keys, values], &input.q.device())
                });
                let result = delta_kernels::kron_scan_partitioned(
                    input.q,
                    input.k,
                    input.v,
                    input.alpha,
                    input.erase,
                    input.write,
                    valid,
                    reset,
                    initial,
                    chunk_size,
                    backend.partitions(),
                );
                ScanOutput {
                    output: result.output,
                    final_state: result.final_state,
                    scores: result.scores,
                }
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            unreachable!("fused GPU scan requires the vulkan or cuda feature")
        }
    }
}

pub(crate) fn assert_fused_device(device: &Device) {
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    assert!(
        matches!(
            device.clone().inner().as_dispatch(),
            burn::backend::DispatchDevice::Cube(_)
        ),
        "fused GPU scan requires a CubeCL GPU device; no CPU fallback"
    );
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    {
        let _ = device;
        panic!("fused GPU scan requires the vulkan or cuda feature; no CPU fallback");
    }
}

/// Input/output axes are [batch,time,head,channel]; state is [batch,head,key,value].
/// Padding freezes the state. A reset on a valid token discards the previous document.
/// Supported computational chunk sizes are 1..=16, independently of context length.
pub fn scan(
    _kind: MixerKind,
    input: ScanInput,
    valid: Tensor<2>,
    reset: Tensor<2>,
    chunk_size: usize,
    initial: Option<Tensor<4>>,
) -> ScanOutput {
    assert!(
        (1..=MAX_CHUNK_SIZE).contains(&chunk_size),
        "computational chunk_size must be 1..=16"
    );
    let [batch, time, heads, keys] = input.q.dims();
    let values = input.v.dims()[3];
    let device = input.q.device();
    let mut state = initial.unwrap_or_else(|| Tensor::zeros([batch, heads, keys, values], &device));
    let mut outputs = Vec::with_capacity(time.div_ceil(chunk_size));
    let mut scores = Vec::with_capacity(time.div_ceil(chunk_size));
    for first in (0..time).step_by(chunk_size) {
        let size = chunk_size.min(time - first);
        let slice = |x: &Tensor<4>| {
            let width = x.dims()[3];
            x.clone()
                .slice([0..batch, first..first + size, 0..heads, 0..width])
                .swap_dims(1, 2)
        };
        let active = valid
            .clone()
            .slice([0..batch, first..first + size])
            .reshape([batch, 1, size, 1]);
        let resets = reset
            .clone()
            .slice([0..batch, first..first + size])
            .reshape([batch, 1, size, 1]);
        // Scale only read queries, never keys/writes or physical surprise scores.
        let q = slice(&input.q) * (active.clone() / (keys as f64).sqrt());
        let k = slice(&input.k) * active.clone();
        let v = slice(&input.v) * active.clone();
        let erase = slice(&input.erase) * active.clone();
        let write = slice(&input.write) * active.clone();
        let alpha: Tensor<4> =
            slice(&input.alpha) * (1.0f32 - resets) * active.clone() + (1.0f32 - active);
        let prefix = alpha.clone().cumprod(2);
        // decay[r,s] is the product over tokens s+1..=r, with identity on r=s.
        let mut product = Tensor::ones([batch, heads, size, keys], &device);
        let mut rows = Vec::with_capacity(size);
        for r in 0..size {
            let a = alpha.clone().slice([0..batch, 0..heads, r..r + 1, 0..keys]);
            product = (product * a).slice_assign(
                [0..batch, 0..heads, r..r + 1, 0..keys],
                Tensor::ones([batch, heads, 1, keys], &device),
            );
            rows.push(product.clone().unsqueeze_dim::<5>(2));
        }
        let decay = Tensor::cat(rows, 2);
        let factors = decay.clone() * k.clone().unsqueeze_dim::<5>(2);
        let key_overlap = |left: Tensor<4>| {
            (left.unsqueeze_dim::<5>(3) * factors.clone())
                .sum_dim(4)
                .reshape([batch, heads, size, size])
        };
        let lower = key_overlap(k.clone())
            * v.clone().matmul(v.clone().swap_dims(2, 3))
            * erase.clone();
        // A rowwise dot GEMM preserves the initial-state VJP. A multiply/sum
        // reduction loses it on GPU in the erase/solve graph, even at rank 2
        // for size=1; see docs/optimization/tensor-solver.md.
        let rhs = write
            - (k.clone() * prefix.clone())
                .matmul(state.clone())
                .unsqueeze_dim::<5>(3)
                .matmul(v.clone().unsqueeze_dim::<5>(4))
                .reshape([batch, heads, size, 1])
                * erase;
        let mut power = lower.tril(-1).neg();
        let mut inverse = Tensor::<2>::eye(size, &device).reshape([1, 1, size, size]);
        let mut covered = 1;
        while covered < size {
            inverse = inverse.clone() + power.clone().matmul(inverse);
            covered *= 2;
            if covered < size {
                power = power.clone().matmul(power);
            }
        }
        let correction = inverse.matmul(rhs);
        let edits = correction * v;
        let out = (q.clone() * prefix.clone()).matmul(state.clone())
            + key_overlap(q).tril(0).matmul(edits.clone());
        let squared_magnitude = k.clone().square().sum_dim(3) * edits.clone().square().sum_dim(3);
        // The physical norm is unchanged; its chosen subgradient at a zero edit is zero.
        let zero_edit = squared_magnitude.clone().equal_elem(0.0);
        let magnitude = squared_magnitude
            .mask_fill(zero_edit.clone(), 1.0)
            .sqrt()
            .mask_fill(zero_edit, 0.0);
        let tail = decay
            .slice([0..batch, 0..heads, size - 1..size, 0..size, 0..keys])
            .reshape([batch, heads, size, keys]);
        let final_decay = prefix
            .slice([0..batch, 0..heads, size - 1..size, 0..keys])
            .reshape([batch, heads, keys, 1]);
        state = final_decay * state + (tail * k).swap_dims(2, 3).matmul(edits);
        outputs.push(out.swap_dims(1, 2));
        scores.push(magnitude.swap_dims(1, 2));
    }
    ScanOutput {
        output: Tensor::cat(outputs, 1),
        final_state: state,
        scores: Tensor::cat(scores, 1),
    }
}

#[cfg(test)]
#[path = "chunk_tests.rs"]
mod tests;
