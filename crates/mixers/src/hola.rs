//! Exact original KV memory, shared by both recurrence families.
//! Reads see the previous chunk's top-capacity cache plus causal current-chunk KV.
//! Selection is discrete; gradients flow through gathered K/V and cache-read norms.
use crate::{CachePolicy, HolaBackend};
use burn::prelude::*;
use burn::tensor::activation::softmax;

/// The norms candidate changes only the two cache-read norms, not other blocks.
pub(crate) fn normalize(
    backend: HolaBackend,
    normalization: crate::NormalizationBackend,
    norm: &burn::nn::RmsNorm,
    input: Tensor<4>,
) -> Tensor<4> {
    if backend == HolaBackend::FusedGpuPairsNorms {
        crate::NormalizationBackend::FusedNorms.rms(norm, input)
    } else {
        normalization.rms(norm, input)
    }
}

pub(crate) struct CacheRead {
    pub output: Tensor<4>,
    pub final_positions: Tensor<3>,
    pub attention_pairs: usize,
}

/// Explicit opt-in dispatch. The tensor implementation below remains the
/// differentiable reference and the default until paired GPU measurements.
pub(crate) fn read_selected(
    backend: HolaBackend,
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
    sink: Tensor<1>,
) -> Tensor<4> {
    match backend {
        HolaBackend::Tensor => read(
            q, k, v, scores, valid, documents, capacity, chunk, policy, sink,
        ),
        HolaBackend::FusedGpu | HolaBackend::FusedGpuPairs | HolaBackend::FusedGpuPairsNorms => {
            assert!(
                policy != CachePolicy::None,
                "HOLA requires an active cache policy"
            );
            crate::chunk::assert_fused_device(&q.device());
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                delta_kernels::hola_read(
                    q,
                    k,
                    v,
                    scores,
                    valid,
                    documents,
                    sink,
                    capacity,
                    chunk,
                    policy == CachePolicy::Recency,
                    false,
                    backend != HolaBackend::FusedGpu,
                )
                .output
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            {
                let _ = (k, v, scores, valid, documents, capacity, chunk, sink);
                unreachable!("fused HOLA requires the vulkan or cuda feature; no CPU fallback")
            }
        }
    }
}

pub(crate) fn read_detailed_selected(
    backend: HolaBackend,
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
    sink: Tensor<1>,
) -> CacheRead {
    match backend {
        HolaBackend::Tensor => read_detailed(
            q, k, v, scores, valid, documents, capacity, chunk, policy, sink,
        ),
        HolaBackend::FusedGpu | HolaBackend::FusedGpuPairs | HolaBackend::FusedGpuPairsNorms => {
            assert!(
                policy != CachePolicy::None,
                "HOLA requires an active cache policy"
            );
            crate::chunk::assert_fused_device(&q.device());
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                let [b, t, h, _] = q.dims();
                let attention_pairs = (0..t)
                    .step_by(chunk)
                    .map(|first| {
                        let n = chunk.min(t - first);
                        b * h * n * (capacity + n + 1)
                    })
                    .sum();
                let result = delta_kernels::hola_read(
                    q,
                    k,
                    v,
                    scores,
                    valid,
                    documents,
                    sink,
                    capacity,
                    chunk,
                    policy == CachePolicy::Recency,
                    true,
                    backend != HolaBackend::FusedGpu,
                );
                CacheRead {
                    output: result.output,
                    final_positions: result.final_positions.expect("requested HOLA positions"),
                    attention_pairs,
                }
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            {
                let _ = (k, v, scores, valid, documents, capacity, chunk, sink);
                unreachable!("fused HOLA requires the vulkan or cuda feature; no CPU fallback")
            }
        }
    }
}

pub(crate) fn read(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
    sink: Tensor<1>,
) -> Tensor<4> {
    read_detailed(
        q, k, v, scores, valid, documents, capacity, chunk, policy, sink,
    )
    .output
}

pub(crate) fn read_detailed(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
    sink: Tensor<1>,
) -> CacheRead {
    read_impl::<false>(
        q, k, v, scores, valid, documents, capacity, chunk, policy, sink, None,
    )
    .0
}

/// One chunk's cache update: the slots kept (indices into cache ∪ chunk) and their positions.
pub struct CacheSelection {
    pub chunk_start: usize,
    pub kept_slots: Tensor<3, Int>,
    pub positions: Tensor<3>,
}

/// Exact production tensor selection, returning every chunk's kept slots. With `replay`,
/// those slot indices are used instead of ranking, so cache membership stays fixed while
/// the differentiable inputs change (finite-difference checks across a hard top-k).
pub(crate) fn read_with_selection(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
    sink: Tensor<1>,
    replay: Option<&[Tensor<3, Int>]>,
) -> (CacheRead, Vec<CacheSelection>) {
    read_impl::<true>(
        q, k, v, scores, valid, documents, capacity, chunk, policy, sink, replay,
    )
}

fn read_impl<const TRACE: bool>(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
    sink: Tensor<1>,
    replay: Option<&[Tensor<3, Int>]>,
) -> (CacheRead, Vec<CacheSelection>) {
    assert!(capacity > 0 && chunk > 0 && policy != CachePolicy::None);
    let [b, t, h, dk] = q.dims();
    let dv = v.dims()[3];
    let device = q.device();
    let mut ck = Tensor::zeros([b, h, capacity, dk], &device);
    let mut cv = Tensor::zeros([b, h, capacity, dv], &device);
    let mut cs = Tensor::full([b, h, capacity], -1.0e30, &device);
    let mut cp = Tensor::full([b, h, capacity], -1.0, &device);
    let mut cd = Tensor::full([b, h, capacity], -1.0, &device);
    let mut output = Vec::with_capacity(t.div_ceil(chunk));
    let mut work = 0;
    let mut selections = Vec::new();
    if let Some(replay) = replay {
        assert_eq!(
            replay.len(),
            t.div_ceil(chunk),
            "replayed cache selection has the wrong chunk count"
        );
    }
    for (chunk_index, first) in (0..t).step_by(chunk).enumerate() {
        let n = chunk.min(t - first);
        let total = capacity + n;
        let take = |x: &Tensor<4>| {
            let width = x.dims()[3];
            x.clone()
                .slice([0..b, first..first + n, 0..h, 0..width])
                .swap_dims(1, 2)
        };
        let local_valid = valid
            .clone()
            .slice([0..b, first..first + n])
            .reshape([b, 1, n])
            .repeat_dim(1, h);
        let local_doc = documents
            .clone()
            .slice([0..b, first..first + n])
            .reshape([b, 1, n])
            .repeat_dim(1, h);
        let local_pos = Tensor::<1, Int>::arange(first as i64..(first + n) as i64, &device)
            .float()
            .reshape([1, 1, n])
            .repeat_dim(0, b)
            .repeat_dim(1, h)
            .mask_fill(local_valid.clone().lower_elem(0.5), -1.0);
        let keys = Tensor::cat(vec![ck, take(&k)], 2);
        let values = Tensor::cat(vec![cv, take(&v)], 2);
        let positions = Tensor::cat(vec![cp, local_pos.clone()], 2);
        let docs = Tensor::cat(vec![cd, local_doc.clone()], 2);
        let visible = positions
            .clone()
            .unsqueeze_dim::<4>(2)
            .greater_equal_elem(0.0)
            .bool_and(
                positions
                    .clone()
                    .unsqueeze_dim::<4>(2)
                    .lower_equal(local_pos.clone().unsqueeze_dim::<4>(3)),
            )
            .bool_and(
                docs.clone()
                    .unsqueeze_dim::<4>(2)
                    .equal(local_doc.clone().unsqueeze_dim::<4>(3)),
            );
        let logits = (take(&q).matmul(keys.clone().swap_dims(2, 3)) / (dk as f64).sqrt())
            .mask_fill(visible.bool_not(), -1.0e30);
        let sink_logits = sink
            .clone()
            .reshape([1, h, 1, 1])
            .repeat_dim(0, b)
            .repeat_dim(2, n);
        let probabilities = softmax(Tensor::cat(vec![logits, sink_logits], 3), 3);
        let with_sink = Tensor::cat(
            vec![values.clone(), Tensor::zeros([b, h, 1, dv], &device)],
            2,
        );
        output.push(
            (probabilities.matmul(with_sink) * local_valid.clone().reshape([b, h, n, 1]))
                .swap_dims(1, 2),
        );
        work += b * h * n * (capacity + n + 1);
        let local_scores = if policy == CachePolicy::Recency {
            local_pos
        } else {
            take(&scores).reshape([b, h, n]).detach()
        };
        let all_scores = Tensor::cat(vec![cs, local_scores], 2);
        let last_doc = local_doc.slice([0..b, 0..h, n - 1..n]);
        let eligible = positions
            .clone()
            .greater_equal_elem(0.0)
            .bool_and(docs.clone().equal(last_doc));
        let ranked = all_scores
            .mask_fill(eligible.clone().bool_not(), -1.0e30)
            .detach();
        // Stable exact lexicographic rank: larger score first, then earlier position.
        // Capacity+chunk is small; this avoids backend-dependent top-k tie order.
        let better = ranked
            .clone()
            .unsqueeze_dim::<4>(2)
            .greater(ranked.clone().unsqueeze_dim::<4>(3));
        let equal = ranked
            .clone()
            .unsqueeze_dim::<4>(2)
            .equal(ranked.clone().unsqueeze_dim::<4>(3));
        let earlier = positions
            .clone()
            .unsqueeze_dim::<4>(2)
            .lower(positions.clone().unsqueeze_dim::<4>(3));
        let rank = better
            .bool_or(equal.bool_and(earlier))
            .int()
            .sum_dim(3)
            .reshape([b, h, total]);
        let index = match replay {
            Some(replay) => replay[chunk_index].clone(),
            None => rank.argsort(2).slice([0..b, 0..h, 0..capacity]),
        };
        let index_kept = index.clone();
        cs = ranked.gather(2, index.clone());
        cp = positions
            .gather(2, index.clone())
            .mask_fill(eligible.gather(2, index.clone()).bool_not(), -1.0);
        cd = docs.gather(2, index.clone());
        ck = keys.gather(2, index.clone().unsqueeze_dim::<4>(3).repeat_dim(3, dk));
        cv = values.gather(2, index.unsqueeze_dim::<4>(3).repeat_dim(3, dv));
        if TRACE {
            selections.push(CacheSelection {
                chunk_start: first,
                kept_slots: index_kept,
                positions: cp.clone(),
            });
        }
    }
    (
        CacheRead {
            output: Tensor::cat(output, 1),
            final_positions: cp,
            attention_pairs: work,
        },
        selections,
    )
}
