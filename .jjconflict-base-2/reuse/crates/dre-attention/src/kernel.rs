//! Exact bidirectional document-local attention implemented with CubeCL kernels.

use burn::prelude::*;
use dre_core::BatchLayout;

/// Computes attention for `[B,H,T,64]` tensors with the configured radius 128.
pub fn windowed_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, layout: &BatchLayout, radius: usize) -> Tensor<4> {
    let [batch, heads, tokens, width] = q.dims();
    assert_eq!(width, 64);
    assert_eq!(k.dims(), [batch, heads, tokens, width]);
    let (output, _) = super::attention_kernel::run_attention(q, k, v, layout, radius);
    output
}
