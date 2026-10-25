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

/// Discrete cache positions before the first chunk and after every chunk,
/// including a partial tail: [batch,head,ceil(time/chunk)+1,capacity].
/// This path performs selection only; it never constructs attention or Q/K/V.
pub(crate) fn schedule_selected(
    backend: HolaBackend,
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
) -> Tensor<4> {
    assert!(capacity > 0 && chunk > 0 && policy != CachePolicy::None);
    match backend {
        HolaBackend::Tensor => schedule_tensor(scores, valid, documents, capacity, chunk, policy),
        HolaBackend::FusedGpu | HolaBackend::FusedGpuPairs | HolaBackend::FusedGpuPairsNorms => {
            crate::chunk::assert_fused_device(&scores.device());
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                delta_kernels::hola_schedule(
                    scores, valid, documents, capacity, chunk, policy == CachePolicy::Recency,
                )
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            {
                let _ = (scores, valid, documents);
                unreachable!("fused HOLA requires the vulkan or cuda feature; no CPU fallback")
            }
        }
    }
}

fn schedule_tensor(
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
) -> Tensor<4> {
    let [b, t, h, width] = scores.dims();
    assert!(b > 0 && t > 0 && h > 0 && width == 1);
    assert_eq!(valid.dims(), [b, t]);
    assert_eq!(documents.dims(), [b, t]);
    let device = scores.device();
    let scores = scores.detach();
    let valid = valid.detach();
    let documents = documents.detach();
    let mut cs = Tensor::full([b, h, capacity], -1.0e30, &device);
    let mut cp = Tensor::full([b, h, capacity], -1.0, &device);
    let mut cd = Tensor::full([b, h, capacity], -1.0, &device);
    let mut snapshots = Vec::with_capacity(t.div_ceil(chunk) + 1);
    snapshots.push(cp.clone().unsqueeze_dim::<4>(2));
    for first in (0..t).step_by(chunk) {
        let n = chunk.min(t - first);
        let local_valid = valid.clone().slice([0..b, first..first + n])
            .reshape([b, 1, n]).repeat_dim(1, h);
        let local_doc = documents.clone().slice([0..b, first..first + n])
            .reshape([b, 1, n]).repeat_dim(1, h);
        let local_pos = Tensor::<1, Int>::arange(first as i64..(first + n) as i64, &device)
            .float().reshape([1, 1, n]).repeat_dim(0, b).repeat_dim(1, h)
            .mask_fill(local_valid.lower_elem(0.5), -1.0);
        let local_scores = if policy == CachePolicy::Recency {
            local_pos.clone()
        } else {
            scores.clone().slice([0..b, first..first + n, 0..h, 0..1])
                .swap_dims(1, 2).reshape([b, h, n])
        };
        let last_doc = local_doc.clone().slice([0..b, 0..h, n - 1..n]);
        (cs, cp, cd, _) = select_cache(
            Tensor::cat(vec![cs, local_scores], 2),
            Tensor::cat(vec![cp, local_pos], 2),
            Tensor::cat(vec![cd, local_doc], 2),
            last_doc,
            capacity,
            None,
        );
        snapshots.push(cp.clone().unsqueeze_dim::<4>(2));
    }
    Tensor::cat(snapshots, 2)
}

/// Shared exact tensor membership for attention reads and position-only schedules.
/// Replayed slots bypass ranking but retain the same eligibility masking.
fn select_cache(
    all_scores: Tensor<3>,
    positions: Tensor<3>,
    docs: Tensor<3>,
    last_doc: Tensor<3>,
    capacity: usize,
    replay: Option<&Tensor<3, Int>>,
) -> (Tensor<3>, Tensor<3>, Tensor<3>, Tensor<3, Int>) {
    let [b, h, total] = positions.dims();
    let eligible = positions.clone().greater_equal_elem(0.0)
        .bool_and(docs.clone().equal(last_doc));
    let ranked = all_scores.mask_fill(eligible.clone().bool_not(), -1.0e30).detach();
    let index = match replay {
        Some(index) => index.clone(),
        None => {
            // Exact lexicographic rank: larger score first, then earlier position.
            // This is the original tensor reference, independent of top-k tie order.
            let better = ranked.clone().unsqueeze_dim::<4>(2)
                .greater(ranked.clone().unsqueeze_dim::<4>(3));
            let equal = ranked.clone().unsqueeze_dim::<4>(2)
                .equal(ranked.clone().unsqueeze_dim::<4>(3));
            let earlier = positions.clone().unsqueeze_dim::<4>(2)
                .lower(positions.clone().unsqueeze_dim::<4>(3));
            let rank = better.bool_or(equal.bool_and(earlier)).int().sum_dim(3)
                .reshape([b, h, total]);
            rank.argsort(2).slice([0..b, 0..h, 0..capacity])
        }
    };
    (
        ranked.gather(2, index.clone()),
        positions.gather(2, index.clone())
            .mask_fill(eligible.gather(2, index.clone()).bool_not(), -1.0),
        docs.gather(2, index.clone()),
        index,
    )
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
        q, k, v, scores, valid, documents, capacity, chunk, policy, sink, None, None, None,
    )
    .0
}

/// One chunk's cache update: the slots kept (indices into cache ∪ chunk) and their positions.
pub struct CacheSelection {
    pub chunk_start: usize,
    pub kept_slots: Tensor<3, Int>,
    pub positions: Tensor<3>,
    pub prefix_visible: Option<Tensor<4, Bool>>,
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
        q, k, v, scores, valid, documents, capacity, chunk, policy, sink, replay, None, None,
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
    retrieval: Option<crate::prefix_retrieval::PrefixReadConfig>,
    prefix_replay: Option<&[Option<Tensor<4, Bool>>]>,
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
        let retrieval_positions = cp.clone();
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
        let mut prefix_visible = None;
        let mut logits = (take(&q).matmul(keys.clone().swap_dims(2, 3)) / (dk as f64).sqrt())
            .mask_fill(visible.bool_not(), -1.0e30);
        let mut read_values = values.clone();
        if let Some(config) = retrieval {
            let prefix = first;
            if prefix > 0 {
                let visible = match prefix_replay {
                    Some(replay) => replay[chunk_index].clone().expect("missing replayed prefix visibility"),
                    None => crate::prefix_retrieval::visibility(take(&q),&k,&valid,&documents,
                        documents.clone().slice([0..b,first..first+n]),retrieval_positions,first,config),
                };
                if TRACE { prefix_visible = Some(visible.clone()); }
                let prefix_keys = k.clone().slice([0..b,0..prefix,0..h,0..dk]).swap_dims(1,2);
                let extra = (take(&q).matmul(prefix_keys.swap_dims(2,3))/(dk as f64).sqrt())
                    .mask_fill(visible.bool_not(),-1.0e30);
                logits = Tensor::cat(vec![logits,extra],3);
                read_values = Tensor::cat(vec![read_values,v.clone().slice([0..b,0..prefix,0..h,0..dv]).swap_dims(1,2)],2);
                work += b*h*n*prefix;
            }
        }
        let sink_logits = sink
            .clone()
            .reshape([1, h, 1, 1])
            .repeat_dim(0, b)
            .repeat_dim(2, n);
        let probabilities = softmax(Tensor::cat(vec![logits, sink_logits], 3), 3);
        let with_sink = Tensor::cat(
            vec![read_values, Tensor::zeros([b, h, 1, dv], &device)],
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
        let last_doc = local_doc.slice([0..b, 0..h, n - 1..n]);
        let index;
        (cs, cp, cd, index) = select_cache(
            Tensor::cat(vec![cs, local_scores], 2),
            positions,
            docs,
            last_doc,
            capacity,
            replay.map(|slots| &slots[chunk_index]),
        );
        let index_kept = index.clone();
        ck = keys.gather(2, index.clone().unsqueeze_dim::<4>(3).repeat_dim(3, dk));
        cv = values.gather(2, index.unsqueeze_dim::<4>(3).repeat_dim(3, dv));
        if TRACE {
            selections.push(CacheSelection {
                chunk_start: first,
                kept_slots: index_kept,
                positions: cp.clone(),
                prefix_visible,
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

pub(crate) fn read_prefix_with_selection(
    q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, scores: Tensor<4>, valid: Tensor<2>, documents: Tensor<2>,
    capacity: usize, chunk: usize, policy: CachePolicy, sink: Tensor<1>,
    config: crate::prefix_retrieval::PrefixReadConfig, replay: Option<&[Tensor<3, Int>]>,
    prefix_replay: Option<&[Option<Tensor<4,Bool>>]>,
) -> (CacheRead, Vec<CacheSelection>) {
    assert!(config.backend.is_tensor(), "selection replay uses the tensor reference");
    read_impl::<true>(q,k,v,scores,valid,documents,capacity,chunk,policy,sink,replay,Some(config),prefix_replay)
}

/// Retrieval is opt-in; the ordinary HOLA reader above is unchanged.
pub(crate) fn read_prefix(
    q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, scores: Tensor<4>,
    valid: Tensor<2>, documents: Tensor<2>, capacity: usize, chunk: usize,
    policy: CachePolicy, sink: Tensor<1>, start: Option<usize>,
    config: crate::prefix_retrieval::PrefixReadConfig, in_block_reads: crate::sbd::InBlockReads,
) -> Tensor<4> {
    let [b,t,h,_] = q.dims();
    let take = |x: &Tensor<4>, a: usize, z: usize| {
        x.clone().slice([0..b,a..z,0..h,0..x.dims()[3]])
    };
    if let Some(start) = start {
        let cut = start/chunk*chunk;
        let clean = read_prefix(take(&q,0,start),take(&k,0,start),take(&v,0,start),
            take(&scores,0,start),valid.clone().slice([0..b,0..start]),
            documents.clone().slice([0..b,0..start]),capacity,chunk,policy,sink.clone(),None,config,in_block_reads);
        let positions = if cut > 0 {
            read_detailed(take(&q,0,cut),take(&k,0,cut),take(&v,0,cut),
                take(&scores,0,cut),valid.clone().slice([0..b,0..cut]),
                documents.clone().slice([0..b,0..cut]),capacity,chunk,policy,sink.clone()).final_positions
        } else { Tensor::full([b,h,capacity],-1.0,&q.device()) };
        let block = (in_block_reads == crate::sbd::InBlockReads::HolaRead)
            .then(|| (take(&k,start,t),take(&v,start,t)));
        let active = crate::prefix_retrieval::noisy_read(take(&q,start,t),k,v,valid.clone(),
            documents.clone(),documents.slice([0..b,start..t]),valid.slice([0..b,start..t]),
            positions,start,chunk,sink,config,block,None);
        return Tensor::cat(vec![clean,active],1);
    }
    if !config.backend.is_tensor() {
        crate::chunk::assert_fused_device(&q.device());
        #[cfg(any(feature="vulkan",feature="cuda"))]
        return delta_kernels::prefix_hola_read(q,k,v,scores,valid,documents,sink,
            capacity,chunk,policy == CachePolicy::Recency,config.block_size,config.blocks_per_query);
        #[cfg(not(any(feature="vulkan",feature="cuda")))]
        panic!("fused prefix retrieval requires a GPU feature; no CPU fallback");
    }
    read_impl::<false>(q,k,v,scores,valid,documents,capacity,chunk,policy,sink,None,Some(config),None).0.output
}

/// Read the exact cache at `start`: completed chunks' selected slots plus the
/// unfinished clean chunk and, when selected, temporary active-block K/V.
pub(crate) fn read_frozen(
    q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, scores: Tensor<4>,
    valid: Tensor<2>, documents: Tensor<2>, capacity: usize, chunk: usize,
    policy: CachePolicy, sink: Tensor<1>, start: usize, backend: HolaBackend,
    in_block_reads: crate::sbd::InBlockReads, noisy_backend: crate::sbd::NoisyReadBackend,
) -> Tensor<4> {
    let [b,t,h,dk] = q.dims();
    let dv = v.dims()[3];
    let n = t-start;
    let cut = start / chunk * chunk;
    let take = |x: &Tensor<4>, a: usize, z: usize| {
        let d = x.dims()[3];
        x.clone().slice([0..b,a..z,0..h,0..d])
    };
    let clean = read_selected(backend, take(&q,0,start), take(&k,0,start), take(&v,0,start),
        take(&scores,0,start), valid.clone().slice([0..b,0..start]),
        documents.clone().slice([0..b,0..start]), capacity, chunk, policy, sink.clone());
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut docs = Vec::new();
    let mut present = Vec::new();
    if cut > 0 {
        let positions = read_detailed_selected(backend, take(&q,0,cut), take(&k,0,cut), take(&v,0,cut),
            take(&scores,0,cut), valid.clone().slice([0..b,0..cut]),
            documents.clone().slice([0..b,0..cut]), capacity, chunk, policy, sink.clone()).final_positions;
        let ids = positions.clone().clamp_min(0.0).int();
        keys.push(k.clone().swap_dims(1,2).gather(2, ids.clone().unsqueeze_dim::<4>(3).expand([b,h,capacity,dk])));
        values.push(v.clone().swap_dims(1,2).gather(2, ids.clone().unsqueeze_dim::<4>(3).expand([b,h,capacity,dv])));
        docs.push(documents.clone().unsqueeze_dim::<3>(1).expand([b,h,t]).gather(2,ids));
        present.push(positions.greater_equal_elem(0.0));
    }
    if cut < start {
        keys.push(take(&k,cut,start).swap_dims(1,2));
        values.push(take(&v,cut,start).swap_dims(1,2));
        docs.push(documents.clone().slice([0..b,cut..start]).unsqueeze_dim::<3>(1).expand([b,h,start-cut]));
        present.push(valid.clone().slice([0..b,cut..start]).unsqueeze_dim::<3>(1).expand([b,h,start-cut]).greater_elem(0.5));
    }
    let query_docs = documents.slice([0..b,start..t]).reshape([b,1,n,1]);
    let visible = present.into_iter().zip(docs).map(|(present,docs)| {
        present.unsqueeze_dim::<4>(2).expand([b,h,n,docs.dims()[2]])
            .bool_and(docs.unsqueeze_dim::<4>(2).equal(query_docs.clone()))
    }).collect();
    let block = (in_block_reads == crate::sbd::InBlockReads::HolaRead)
        .then(|| (take(&k,start,t).swap_dims(1,2),take(&v,start,t).swap_dims(1,2)));
    let active = crate::sbd::noisy_hola_read(take(&q,start,t).swap_dims(1,2),keys,values,visible,block,
        valid.slice([0..b,start..t]),sink,noisy_backend);
    Tensor::cat(vec![clean,active],1)
}
