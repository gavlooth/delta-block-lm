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
    let [b,h,n,dk] = query.dims();
    let blocks = cut / config.block_size;
    assert!(cut > 0);
    let prefix = blocks * config.block_size;
    let device = query.device();
    let qdoc = query_documents.reshape([b,1,n,1]);
    let mut segments = Vec::with_capacity(2);
    if blocks > 0 {
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
                .unsqueeze_dim::<5>(2) * same.float().unsqueeze_dim::<5>(4))
                .sum_dim(3).reshape([b,h,n,dk]) / count.clone().clamp_min(1.0);
            scores.push((query.clone().detach()*summary.detach()).sum_dim(3)/(dk as f64).sqrt());
            eligible.push(count.greater_elem(0.0));
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
        let selected = selected.gather(3,token_blocks.reshape([1,1,1,prefix]).expand([b,h,n,prefix]));
        segments.push(selected);
    }
    if prefix < cut {
        segments.push(Tensor::<4>::ones([b,h,n,cut-prefix],&device).greater_elem(0.5));
    }
    let selected = Tensor::cat(segments,3);
    let positions = Tensor::<1,Int>::arange(0..cut as i64,&device).float();
    let duplicate = cached_positions.unsqueeze_dim::<4>(3)
        .equal(positions.reshape([1,1,1,cut])).int().sum_dim(2)
        .reshape([b,h,1,cut]).greater_elem(0);
    selected.bool_and(duplicate.bool_not())
        .bool_and(documents.clone().slice([0..b,0..cut]).reshape([b,1,1,cut]).equal(qdoc))
        .bool_and(valid.clone().slice([0..b,0..cut]).reshape([b,1,1,cut]).greater_elem(0.5))
}

/// Frozen clean snapshot plus an explicitly selected active-block segment.
pub(crate) fn noisy_read(
    query: Tensor<4>, keys: Tensor<4>, values: Tensor<4>, valid: Tensor<2>,
    documents: Tensor<2>, query_documents: Tensor<2>, query_valid: Tensor<2>,
    positions: Tensor<3>, start: usize, chunk: usize, sink: Tensor<1>, config: PrefixReadConfig,
    active: Option<(Tensor<4>,Tensor<4>)>,
    summaries: Option<Tensor<4>>,
) -> Tensor<4> {
    let [b,n,h,dk] = query.dims();
    let t = keys.dims()[1];
    let dv = values.dims()[3];
    let capacity = positions.dims()[2];
    let cut = start/chunk*chunk;
    if config.backend == PrefixRetrievalBackend::FusedGpu {
        crate::chunk::assert_fused_device(&query.device());
        #[cfg(any(feature="vulkan",feature="cuda"))]
        return delta_kernels::prefix_read(query,keys,values,query_valid,valid,documents,
            query_documents,positions.unsqueeze_dim::<4>(2),sink,chunk,config.block_size,
            config.blocks_per_query,start,true,active,summaries);
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
    if cut < start {
        ks.push(keys.clone().slice([0..b,cut..start,0..h,0..dk]).swap_dims(1,2));
        vs.push(values.clone().slice([0..b,cut..start,0..h,0..dv]).swap_dims(1,2));
        visible.push(valid.clone().slice([0..b,cut..start]).reshape([b,1,1,start-cut]).greater_elem(0.5)
            .bool_and(documents.clone().slice([0..b,cut..start]).reshape([b,1,1,start-cut])
                .equal(query_documents.clone().reshape([b,1,n,1]))).expand([b,h,n,start-cut]));
    }
    let prefix = cut;
    if prefix > 0 {
        visible.push(visibility(query.clone().swap_dims(1,2),&keys,&valid,&documents,
            query_documents,positions,cut,config));
        ks.push(keys.slice([0..b,0..prefix,0..h,0..dk]).swap_dims(1,2));
        vs.push(values.slice([0..b,0..prefix,0..h,0..dv]).swap_dims(1,2));
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
