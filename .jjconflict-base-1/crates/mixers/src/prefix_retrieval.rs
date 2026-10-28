//! Completed clean-prefix blocks share HOLA's single softmax.
use burn::{prelude::*, tensor::activation::softmax};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixRetrievalBackend {
    #[default]
    Tensor,
    FusedGpu,
}
impl PrefixRetrievalBackend {
    pub fn is_tensor(&self) -> bool { *self == Self::Tensor }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixBlockRetrieval {
    pub layers: Vec<usize>,
    pub block_size: usize,
    pub blocks_per_query: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct PrefixReadConfig {
    pub block_size: usize,
    pub blocks_per_query: usize,
    pub backend: PrefixRetrievalBackend,
}

/// Own-block prefix plus selected completed earlier blocks, before the current
/// chunk. Ranking is detached and exact lexicographic (score, then block index).
/// This intentionally simple tensor reference materializes the prefix mask;
/// the GPU implementation instead writes only selected original positions.
pub(crate) fn visibility(
    query: Tensor<4>, keys: &Tensor<4>, valid: &Tensor<2>, documents: &Tensor<2>,
    query_documents: Tensor<2>, cached_positions: Tensor<3>, cut: usize,
    config: PrefixReadConfig,
) -> Tensor<4, Bool> {
    let b = query.dims()[0];
    let cuts = Tensor::<1,Int>::full([b],cut as i64,&query.device());
    visibility_batched(query,keys,valid,documents,query_documents,cached_positions,cuts,cut,config)
}
fn visibility_batched(
    query: Tensor<4>, keys: &Tensor<4>, valid: &Tensor<2>, documents: &Tensor<2>,
    query_documents: Tensor<2>, cached_positions: Tensor<3>, cuts: Tensor<1,Int>,
    max_cut: usize, config: PrefixReadConfig,
) -> Tensor<4, Bool> {
    let [b,h,n,dk] = query.dims();
    assert!(max_cut>0);
    let blocks = max_cut/config.block_size;
    let prefix = blocks*config.block_size;
    let device = query.device();
    let qdoc = query_documents.reshape([b,1,n,1]);
    let mut segments = Vec::with_capacity(2);
    if blocks>0 {
        let mut scores = Vec::with_capacity(blocks);
        let mut eligible = Vec::with_capacity(blocks);
        for j in 0..blocks {
            let a = j*config.block_size;
            let z = a+config.block_size;
            let same = documents.clone().slice([0..b,a..z]).reshape([b,1,1,config.block_size])
                .equal(qdoc.clone()).bool_and(valid.clone().slice([0..b,a..z])
                    .reshape([b,1,1,config.block_size]).greater_elem(0.5));
            let count = same.clone().float().sum_dim(3);
            let summary = (keys.clone().slice([0..b,a..z,0..h,0..dk]).swap_dims(1,2)
                .unsqueeze_dim::<5>(2)*same.float().unsqueeze_dim::<5>(4))
                .sum_dim(3).reshape([b,h,n,dk])/count.clone().clamp_min(1.0);
            scores.push((query.clone().detach()*summary.detach()).sum_dim(3)/(dk as f64).sqrt());
            eligible.push(count.greater_elem(0.0).bool_and(
                cuts.clone().greater_equal_elem(z as i64).reshape([b,1,1,1])));
        }
        let eligible = Tensor::cat(eligible,3).expand([b,h,n,blocks]);
        let score = Tensor::cat(scores,3).mask_fill(eligible.clone().bool_not(),-1.0e30).detach();
        let index = Tensor::<1,Int>::arange(0..blocks as i64,&device).reshape([1,1,1,blocks]);
        let better = score.clone().unsqueeze_dim::<5>(3).greater(score.clone().unsqueeze_dim::<5>(4));
        let equal = score.clone().unsqueeze_dim::<5>(3).equal(score.unsqueeze_dim::<5>(4));
        let earlier = index.clone().unsqueeze_dim::<5>(3).lower(index.unsqueeze_dim::<5>(4));
        let rank = better.bool_or(equal.bool_and(earlier)).int().sum_dim(4).reshape([b,h,n,blocks]);
        let selected = rank.lower_elem(config.blocks_per_query.min(blocks) as i64).bool_and(eligible);
        let token_blocks = Tensor::<1,Int>::arange(0..prefix as i64,&device)/config.block_size as i64;
        segments.push(selected.gather(3,token_blocks.reshape([1,1,1,prefix]).expand([b,h,n,prefix])));
    }
    if prefix<max_cut {
        segments.push(Tensor::<4>::zeros([b,h,n,max_cut-prefix],&device).greater_elem(0.5));
    }
    let token_positions = Tensor::<1,Int>::arange(0..max_cut as i64,&device);
    let own_start = cuts.clone()/config.block_size as i64*config.block_size as i64;
    let own = token_positions.clone().reshape([1,1,1,max_cut])
        .greater_equal(own_start.reshape([b,1,1,1]));
    let selected = Tensor::cat(segments,3).bool_or(own)
        .bool_and(token_positions.clone().reshape([1,1,1,max_cut]).lower(cuts.reshape([b,1,1,1])));
    let duplicate = cached_positions.unsqueeze_dim::<4>(3)
        .equal(token_positions.float().reshape([1,1,1,max_cut])).int().sum_dim(2)
        .reshape([b,h,1,max_cut]).greater_elem(0);
    selected.bool_and(duplicate.bool_not())
        .bool_and(documents.clone().slice([0..b,0..max_cut]).reshape([b,1,1,max_cut]).equal(qdoc))
        .bool_and(valid.clone().slice([0..b,0..max_cut]).reshape([b,1,1,max_cut]).greater_elem(0.5))
}

/// Frozen clean snapshot plus an explicitly selected active-block segment.
pub(crate) fn noisy_read(
    query: Tensor<4>, keys: Tensor<4>, values: Tensor<4>, valid: Tensor<2>,
    documents: Tensor<2>, query_documents: Tensor<2>, query_valid: Tensor<2>,
    positions: Tensor<3>, start: usize, chunk: usize, sink: Tensor<1>, config: PrefixReadConfig,
    active: Option<(Tensor<4>,Tensor<4>)>,
    summaries: Option<Tensor<4>>,
) -> Tensor<4> {
    let b = query.dims()[0];
    let starts = Tensor::full([b],start as f64,&query.device());
    let own = start/chunk*chunk-start/config.block_size*config.block_size;
    noisy_read_batched(query,keys,values,valid,documents,query_documents,query_valid,
        positions,starts,start,own,chunk,sink,config,active,summaries)
}

/// Frozen clean snapshots share one read across the noisy block batch.
pub(crate) fn noisy_read_batched(
    query: Tensor<4>, keys: Tensor<4>, values: Tensor<4>, valid: Tensor<2>,
    documents: Tensor<2>, query_documents: Tensor<2>, query_valid: Tensor<2>,
    positions: Tensor<3>, starts: Tensor<1>, max_start: usize, own: usize,
    chunk: usize, sink: Tensor<1>, config: PrefixReadConfig,
    active: Option<(Tensor<4>,Tensor<4>)>, summaries: Option<Tensor<4>>,
) -> Tensor<4> {
    let [b,n,h,dk] = query.dims();
    let t = keys.dims()[1];
    let dv = values.dims()[3];
    let capacity = positions.dims()[2];
    if config.backend==PrefixRetrievalBackend::FusedGpu {
        crate::chunk::assert_fused_device(&query.device());
        #[cfg(any(feature="vulkan",feature="cuda"))]
        return delta_kernels::prefix_read(query,keys,values,query_valid,valid,documents,
            query_documents,positions.unsqueeze_dim::<4>(2),sink,chunk,config.block_size,
            config.blocks_per_query,starts,max_start,own,true,active,summaries);
        #[cfg(not(any(feature="vulkan",feature="cuda")))]
        panic!("fused prefix retrieval requires a GPU feature; no CPU fallback");
    }
    let device = query.device();
    let ids = positions.clone().clamp_min(0.0).int();
    let mut ks = vec![keys.clone().swap_dims(1,2).gather(2,ids.clone().unsqueeze_dim::<4>(3).expand([b,h,capacity,dk]))];
    let mut vs = vec![values.clone().swap_dims(1,2).gather(2,ids.clone().unsqueeze_dim::<4>(3).expand([b,h,capacity,dv]))];
    let cache_docs = documents.clone().unsqueeze_dim::<3>(1).expand([b,h,t]).gather(2,ids);
    let mut visible = vec![positions.clone().greater_equal_elem(0.0).unsqueeze_dim::<4>(2)
        .bool_and(cache_docs.unsqueeze_dim::<4>(2).equal(query_documents.clone().reshape([b,1,n,1])))];
    if max_start>0 {
        let starts = starts.int();
        let cuts = starts.clone()/chunk as i64*chunk as i64;
        let max_cut = max_start/chunk*chunk;
        let token_positions = Tensor::<1,Int>::arange(0..max_start as i64,&device).reshape([1,1,1,max_start]);
        let local = token_positions.clone().greater_equal(cuts.clone().reshape([b,1,1,1]))
            .bool_and(token_positions.lower(starts.reshape([b,1,1,1])));
        let selected = if max_cut>0 {
            let selected = visibility_batched(query.clone().swap_dims(1,2),&keys,&valid,&documents,
                query_documents.clone(),positions,cuts,max_cut,config);
            if max_cut<max_start {
                Tensor::cat(vec![selected,Tensor::<4>::zeros([b,h,n,max_start-max_cut],&device).greater_elem(0.5)],3)
            } else {selected}
        } else {Tensor::<4>::zeros([b,h,n,max_start],&device).greater_elem(0.5)};
        visible.push(selected.bool_or(local)
            .bool_and(valid.clone().slice([0..b,0..max_start]).reshape([b,1,1,max_start]).greater_elem(0.5))
            .bool_and(documents.slice([0..b,0..max_start]).reshape([b,1,1,max_start])
                .equal(query_documents.reshape([b,1,n,1]))));
        ks.push(keys.slice([0..b,0..max_start,0..h,0..dk]).swap_dims(1,2));
        vs.push(values.slice([0..b,0..max_start,0..h,0..dv]).swap_dims(1,2));
    }
    if let Some((active_keys,active_values)) = active {
        assert_eq!(active_keys.dims(),[b,n,h,dk]);
        assert_eq!(active_values.dims(),[b,n,h,dv]);
        assert!((1..=16).contains(&n));
        ks.push(active_keys.swap_dims(1,2));
        vs.push(active_values.swap_dims(1,2));
        visible.push(query_valid.clone().reshape([b,1,1,n]).greater_elem(0.5).expand([b,h,n,n]));
    }
    let logits = (query.swap_dims(1,2).matmul(Tensor::cat(ks,2).swap_dims(2,3))/(dk as f64).sqrt())
        .mask_fill(Tensor::cat(visible,3).bool_not(),-1.0e30);
    softmax(Tensor::cat(vec![logits,sink.reshape([1,h,1,1]).expand([b,h,n,1])],3),3)
        .matmul(Tensor::cat(vec![Tensor::cat(vs,2),Tensor::zeros([b,h,1,dv],&device)],2))
        .swap_dims(1,2)*query_valid.reshape([b,n,1,1])
}

/// Full clean causal read for explicit reference and isolated cost measurements.
pub fn read(
    query: Tensor<4>, keys: Tensor<4>, values: Tensor<4>, scores: Tensor<4>,
    valid: Tensor<2>, documents: Tensor<2>, sink: Tensor<1>, capacity: usize, chunk: usize,
    config: PrefixReadConfig,
) -> Tensor<4> {
    crate::hola::read_prefix(query,keys,values,scores,valid,documents,capacity,chunk,
        crate::CachePolicy::Surprise,sink,None,config,crate::sbd::InBlockReads::None)
}
