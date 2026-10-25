//! Clean-prefix snapshots for SBD recurrence, frozen HOLA reads, and wavelet history.
use burn::{prelude::*, tensor::activation::softmax};

use crate::{Mixer, MixerKind, chunk, hola};
use burn::tensor::activation::{sigmoid,silu};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InBlockReads {
    #[default]
    None,
    HolaRead,
}
impl InBlockReads {
    pub fn is_none(&self) -> bool { *self == Self::None }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoisyReadBackend {
    #[default]
    Tensor,
    FusedGpu,
}
impl NoisyReadBackend {
    pub fn is_tensor(&self) -> bool { *self == Self::Tensor }
}

/// Temporary noisy HOLA segments. Every segment participates in one softmax.
/// Inputs use [batch, head, query/key, channel]; the result uses [batch, query, head, value].
pub fn noisy_hola_read(
    query: Tensor<4>, mut keys: Vec<Tensor<4>>, mut values: Vec<Tensor<4>>,
    mut visible: Vec<Tensor<4, Bool>>, block: Option<(Tensor<4>, Tensor<4>)>,
    valid: Tensor<2>, sink: Tensor<1>, backend: NoisyReadBackend,
) -> Tensor<4> {
    let [b,h,n,dk] = query.dims();
    let device = query.device();
    if let Some((k,v)) = block {
        let mask = valid.clone().reshape([b,1,1,n]).expand([b,h,n,n]).greater_elem(0.5);
        keys.push(k);
        values.push(v);
        visible.push(mask);
    }
    assert!(!keys.is_empty(), "noisy HOLA requires a clean or active key segment");
    let dv = values[0].dims()[3];
    let keys = Tensor::cat(keys,2);
    let values = Tensor::cat(values,2);
    let visible = Tensor::cat(visible,3);
    match backend {
        NoisyReadBackend::Tensor => {
            let logits = (query.matmul(keys.swap_dims(2,3))/(dk as f64).sqrt())
                .mask_fill(visible.bool_not(),-1.0e30);
            let probabilities = softmax(Tensor::cat(vec![logits,sink.reshape([1,h,1,1]).expand([b,h,n,1])],3),3);
            probabilities.matmul(Tensor::cat(vec![values,Tensor::zeros([b,h,1,dv],&device)],2))
                .swap_dims(1,2)*valid.reshape([b,n,1,1])
        }
        NoisyReadBackend::FusedGpu => {
            crate::chunk::assert_fused_device(&device);
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            { delta_kernels::noisy_hola_read(query,keys,values,visible.float(),valid,sink).swap_dims(1,2) }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            { let _ = (keys,values,visible,valid,sink); unreachable!("fused noisy HOLA requires a GPU build; no CPU fallback") }
        }
    }
}

/// Differentiable clean tensors prepared once per layer/sequence. Snapshot work
/// scans disjoint segments with carried state, never all prefixes per block.
pub struct MixerCache {
    raw: [Tensor<3>;3],
    cache_k: Tensor<4>,
    cache_v: Tensor<4>,
    prefix_summaries: Option<Tensor<4>>,
    positions: Vec<Tensor<3>>,
    states: Vec<Tensor<4>>,
    starts: Vec<usize>,
}
fn take4(x: &Tensor<4>, a: usize, z: usize) -> Tensor<4> {
    let [b,_,h,d] = x.dims();
    x.clone().slice([0..b,a..z,0..h,0..d])
}
fn convolve(m: &Mixer, raw: Tensor<3>, weight: Tensor<2>) -> Tensor<4> {
    let [b,t,c] = raw.dims();
    let device = raw.device();
    silu(m.convolution_backend.forward(raw,weight,Tensor::ones([b,t],&device),Tensor::ones([b,t],&device)))
        .reshape([b,t,m.heads,c/m.heads])
}

/// Clean histories and recurrence snapshots with the noisy block as batch axis.
pub struct BatchedMixerCache {
    raw: [Tensor<3>;3],
    state: Tensor<4>,
    reads: Option<BatchedReadCache>,
}
struct BatchedReadCache {
    keys: Tensor<4>,
    values: Tensor<4>,
    positions: Tensor<3>,
    visible: Option<Tensor<3, Bool>>,
    summaries: Option<Tensor<4>>,
    starts: Tensor<1>,
    max_start: usize,
    own: usize,
}
fn block_rows(blocks: &[(usize,usize)], device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(burn::tensor::TensorData::new(
        blocks.iter().map(|&(row,_)| row as i64).collect::<Vec<_>>(),[blocks.len()]),device)
}
fn block_starts(blocks: &[(usize,usize)], device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(burn::tensor::TensorData::new(
        blocks.iter().map(|&(_,start)| start as i64).collect::<Vec<_>>(),[blocks.len()]),device)
}
fn history_indices(blocks: &[(usize,usize)], time: usize, length: usize, device: &Device)
    -> (Tensor<1, Int>,Tensor<3>) {
    let b = blocks.len();
    let positions = block_starts(blocks,device).reshape([b,1])
        + Tensor::<1,Int>::arange(-(length as i64)..0,device).reshape([1,length]);
    let visible = positions.clone().greater_equal_elem(0).float().reshape([b,length,1]);
    let ids = (block_rows(blocks,device).reshape([b,1])*time as i64
        + positions.clamp_min(0)).reshape([b*length]);
    (ids,visible)
}
fn gather_history(input: &Tensor<3>, ids: &Tensor<1,Int>, visible: &Tensor<3>) -> Tensor<3> {
    let [rows,time,width] = input.dims();
    let [b,length,_] = visible.dims();
    input.clone().reshape([rows*time,width]).select(0,ids.clone())
        .reshape([b,length,width])*visible.clone()
}
impl Mixer {
    pub fn prepare_sbd(&self, input: Tensor<3>, starts: &[usize]) -> MixerCache {
        let [b,t,_] = input.dims();
        let device = input.device();
        let raw = [self.q.forward(input.clone()),self.k.forward(input.clone()),self.v.forward(input.clone())];
        let rq = convolve(self,raw[0].clone(),self.q_conv.val());
        let rk = convolve(self,raw[1].clone(),self.k_conv.val());
        let rv = convolve(self,raw[2].clone(),self.v_conv.val());
        let q = self.normalization_backend.l2(rq.clone());
        let k = self.normalization_backend.l2(rk.clone());
        let v = if self.kind == MixerKind::KronLstm {self.normalization_backend.l2(rv.clone())} else {rv.clone()};
        let alpha = self.normalization_backend.decay(self.decay.forward(input.clone()).reshape([b,t,self.heads,self.key_dim]),self.decay_bias.val(),self.decay_scale.val());
        let ew = if self.kind == MixerKind::Gdn2 {self.key_dim} else {1};
        let ww = if self.kind == MixerKind::Gdn2 {self.value_dim} else {1};
        let erase = sigmoid(self.erase.forward(input.clone())).reshape([b,t,self.heads,ew]);
        let write = sigmoid(self.write.forward(input)).reshape([b,t,self.heads,ww]);
        let mut current = Tensor::zeros([b,self.heads,self.key_dim,self.value_dim],&device);
        let mut states = Vec::with_capacity(starts.len());
        let mut scores = Vec::new();
        let mut first = 0;
        for end in starts.iter().copied().chain(std::iter::once(t)) {
            if end > first {
                let result = chunk::scan_selected(self.scan_backend,self.kind,chunk::ScanInput {
                    q:take4(&q,first,end), k:take4(&k,first,end), v:take4(&v,first,end),
                    alpha:take4(&alpha,first,end),erase:take4(&erase,first,end),write:take4(&write,first,end),
                },Tensor::ones([b,end-first],&device),Tensor::zeros([b,end-first],&device),self.chunk_size,Some(current));
                current = result.final_state;
                scores.push(result.scores);
                first = end;
            }
            if states.len() < starts.len() { states.push(current.clone()); }
        }
        let scores = Tensor::cat(scores,1);
        let mut positions = Vec::new();
        let cache_k = if let Some(cache) = &self.cache {
            let cq = hola::normalize(self.hola_backend,self.normalization_backend,&cache.q_norm,rq);
            let ck = hola::normalize(self.hola_backend,self.normalization_backend,&cache.k_norm,rk);
            let (_, selections) = hola::read_with_selection(cq,ck.clone(),rv.clone(),scores,
                Tensor::ones([b,t],&device),Tensor::ones([b,t],&device),
                self.cache_capacity,self.chunk_size,self.cache_policy,cache.sink.val(),None);
            positions = selections.into_iter().map(|s| s.positions).collect();
            ck
        } else { rk };
        let prefix_summaries = self.prefix_retrieval.and_then(|config| {
            if config.backend.is_tensor() || t<config.block_size {return None;}
            #[cfg(any(feature="vulkan",feature="cuda"))]
            {Some(delta_kernels::prefix_summaries(cache_k.clone(),Tensor::ones([b,t],&device),
                Tensor::ones([b,t],&device),config.block_size))}
            #[cfg(not(any(feature="vulkan",feature="cuda")))]
            {panic!("fused prefix retrieval requires a GPU feature; no CPU fallback")}
        });
        MixerCache {raw,cache_k,cache_v:rv,prefix_summaries,positions,states,starts:starts.to_vec()}
    }
    pub fn prepare_sbd_batched(&self, input: Tensor<3>, blocks: &[(usize,usize)]) -> BatchedMixerCache {
        let [rows,time,_] = input.dims();
        assert!(!blocks.is_empty() && blocks.iter().all(|&(row,start)| row<rows && start<time));
        let device = input.device();
        let b = blocks.len();
        let mut starts = blocks.iter().map(|&(_,start)| start).collect::<Vec<_>>();
        starts.sort_unstable();
        starts.dedup();
        let clean = self.prepare_sbd(input,&starts);
        let (ids,visible) = history_indices(blocks,time,3,&device);
        let raw = clean.raw.each_ref().map(|raw| gather_history(raw,&ids,&visible));
        let state_ids = Tensor::<1,Int>::from_data(burn::tensor::TensorData::new(
            blocks.iter().map(|&(row,start)| (starts.binary_search(&start).unwrap()*rows+row) as i64)
                .collect::<Vec<_>>(),[b]),&device);
        let state = Tensor::cat(clean.states,0).select(0,state_ids);
        let reads = self.cache.as_ref().map(|_| {
            let row_ids = block_rows(blocks,&device);
            let mut snapshots = Vec::with_capacity(clean.positions.len()+1);
            snapshots.push(Tensor::full([rows,self.heads,self.cache_capacity],-1.0,&device));
            snapshots.extend(clean.positions);
            let snapshot_ids = Tensor::<1,Int>::from_data(burn::tensor::TensorData::new(
                blocks.iter().map(|&(row,start)| (start/self.chunk_size*rows+row) as i64)
                    .collect::<Vec<_>>(),[b]),&device);
            let positions = Tensor::cat(snapshots,0).select(0,snapshot_ids);
            let starts = block_starts(blocks,&device);
            let max_start = blocks.iter().map(|&(_,start)| start).max().unwrap();
            let own = self.prefix_retrieval.map_or(0,|config| blocks.iter()
                .map(|&(_,start)| start/self.chunk_size*self.chunk_size-start/config.block_size*config.block_size)
                .max().unwrap());
            if self.prefix_retrieval.is_some() {
                let keys = clean.cache_k.select(0,row_ids.clone());
                let values = clean.cache_v.select(0,row_ids.clone());
                let summaries = clean.prefix_summaries.map(|summaries| summaries.select(0,row_ids));
                BatchedReadCache {keys,values,positions,visible:None,summaries,starts:starts.float(),max_start,own}
            } else {
                let cache_ids = ((row_ids.clone().reshape([b,1,1])*time as i64
                    + positions.clone().clamp_min(0.0).int())*self.heads as i64
                    + Tensor::<1,Int>::arange(0..self.heads as i64,&device).reshape([1,self.heads,1]))
                    .reshape([b*self.heads*self.cache_capacity]);
                let cached_keys = clean.cache_k.clone().reshape([rows*time*self.heads,self.key_dim])
                    .select(0,cache_ids.clone()).reshape([b,self.heads,self.cache_capacity,self.key_dim]).swap_dims(1,2);
                let cached_values = clean.cache_v.clone().reshape([rows*time*self.heads,self.value_dim])
                    .select(0,cache_ids).reshape([b,self.heads,self.cache_capacity,self.value_dim]).swap_dims(1,2);
                let local = blocks.iter().map(|&(_,start)| start%self.chunk_size).max().unwrap();
                let mut key_segments = vec![cached_keys];
                let mut value_segments = vec![cached_values];
                let mut visible = vec![positions.clone().greater_equal_elem(0.0)];
                if local>0 {
                    let cuts = starts.clone()/self.chunk_size as i64*self.chunk_size as i64;
                    let local_positions = cuts.reshape([b,1])
                        + Tensor::<1,Int>::arange(0..local as i64,&device).reshape([1,local]);
                    let local_valid = local_positions.clone().lower(starts.clone().reshape([b,1]));
                    let local_ids = (row_ids.reshape([b,1])*time as i64
                        + local_positions.clamp_max((time-1) as i64)).reshape([b*local]);
                    key_segments.push(clean.cache_k.reshape([rows*time,self.heads,self.key_dim])
                        .select(0,local_ids.clone()).reshape([b,local,self.heads,self.key_dim]));
                    value_segments.push(clean.cache_v.reshape([rows*time,self.heads,self.value_dim])
                        .select(0,local_ids).reshape([b,local,self.heads,self.value_dim]));
                    visible.push(local_valid.unsqueeze_dim::<3>(1).expand([b,self.heads,local]));
                }
                BatchedReadCache {keys:Tensor::cat(key_segments,1),values:Tensor::cat(value_segments,1),
                    positions,visible:Some(Tensor::cat(visible,2)),summaries:None,starts:starts.float(),max_start,own}
            }
        });
        BatchedMixerCache {raw,state,reads}
    }
    pub fn forward_sbd_batched(&self, input: Tensor<3>, clean: &BatchedMixerCache, valid: Tensor<2>) -> Tensor<3> {
        self.forward_sbd_batched_profiled(input,clean,valid,&mut crate::profile::Unprofiled)
    }
    pub fn forward_sbd_batched_profiled<P: crate::profile::Sections>(
        &self, input: Tensor<3>, clean: &BatchedMixerCache, valid: Tensor<2>, profile: &mut P,
    ) -> Tensor<3> {
        let [b,n,_] = input.dims();
        let device = input.device();
        let input = input*valid.clone().unsqueeze_dim::<3>(2);
        let project = |active: Tensor<3>, index: usize, weight: Tensor<2>| {
            take4(&convolve(self,Tensor::cat(vec![clean.raw[index].clone(),active],1),weight),3,3+n)
        };
        let rq = project(self.q.forward(input.clone()),0,self.q_conv.val());
        let rk = project(self.k.forward(input.clone()),1,self.k_conv.val());
        let rv = project(self.v.forward(input.clone()),2,self.v_conv.val());
        let q = self.normalization_backend.l2(rq.clone());
        let k = self.normalization_backend.l2(rk.clone());
        let v = if self.kind==MixerKind::KronLstm {self.normalization_backend.l2(rv.clone())} else {rv.clone()};
        let ew = if self.kind==MixerKind::Gdn2 {self.key_dim} else {1};
        let ww = if self.kind==MixerKind::Gdn2 {self.value_dim} else {1};
        let initial = clean.state.clone().unsqueeze_dim::<5>(1)
            .expand([b,n,self.heads,self.key_dim,self.value_dim])
            .reshape([b*n,self.heads,self.key_dim,self.value_dim]);
        let single = |x: Tensor<4>| {let width=x.dims()[3];x.reshape([b*n,1,self.heads,width])};
        let alpha = self.normalization_backend.decay(
            self.decay.forward(input.clone()).reshape([b,n,self.heads,self.key_dim]),
            self.decay_bias.val(),self.decay_scale.val());
        let erase = sigmoid(self.erase.forward(input.clone())).reshape([b,n,self.heads,ew]);
        let write = sigmoid(self.write.forward(input.clone())).reshape([b,n,self.heads,ww]);
        let recurrent = chunk::scan_selected(self.scan_backend,self.kind,chunk::ScanInput {
            q:single(q),k:single(k),v:single(v),alpha:single(alpha),erase:single(erase),write:single(write),
        },valid.clone().reshape([b*n,1]),Tensor::zeros([b*n,1],&device),self.chunk_size,Some(initial));
        let mut mixed = recurrent.output.reshape([b,n,self.heads,self.value_dim]);
        if let (Some(cache),Some(reads)) = (&self.cache,&clean.reads) {
            mixed = profile.section("Noisy HOLA read", || {
                let query = hola::normalize(self.hola_backend,self.normalization_backend,&cache.q_norm,rq);
                let active = (self.in_block_reads==InBlockReads::HolaRead).then(|| (
                    hola::normalize(self.hola_backend,self.normalization_backend,&cache.k_norm,rk),rv));
                let exact = if let Some(config) = self.prefix_retrieval {
                    let time = reads.keys.dims()[1];
                    crate::prefix_retrieval::noisy_read_batched(query,reads.keys.clone(),reads.values.clone(),
                        Tensor::ones([b,time],&device),Tensor::ones([b,time],&device),
                        Tensor::ones([b,n],&device),valid.clone(),reads.positions.clone(),reads.starts.clone(),
                        reads.max_start,reads.own,self.chunk_size,cache.sink.val(),config,active,reads.summaries.clone())
                } else {
                    let count = reads.keys.dims()[1];
                    noisy_hola_read(query.swap_dims(1,2),vec![reads.keys.clone().swap_dims(1,2)],
                        vec![reads.values.clone().swap_dims(1,2)],
                        vec![reads.visible.as_ref().unwrap().clone().unsqueeze_dim::<4>(2).expand([b,self.heads,n,count])],
                        active.map(|(k,v)| (k.swap_dims(1,2),v.swap_dims(1,2))),valid.clone(),
                        cache.sink.val(),self.noisy_read_backend)
                };
                mixed+exact*sigmoid(cache.gate.val()).reshape([1,1,self.heads,1])
            });
        }
        let gate = self.output_gate.forward(input);
        let mixed = if self.normalization_backend==crate::NormalizationBackend::FusedNormsGates {
            self.normalization_backend.gated_rms(&self.norm,mixed,gate.reshape([b,n,self.heads,self.value_dim]))
        } else {
            self.normalization_backend.rms(&self.norm,mixed*sigmoid(gate).reshape([b,n,self.heads,self.value_dim]))
        };
        self.project_output(mixed.reshape([b,n,self.heads*self.value_dim]))*valid.unsqueeze_dim::<3>(2)
    }
    pub fn forward_sbd_cached(&self, input: Tensor<3>, clean: &MixerCache, block: usize, valid: Tensor<2>) -> Tensor<3> {
        self.forward_sbd_cached_profiled(input,clean,block,valid,&mut crate::profile::Unprofiled)
    }
    pub fn forward_sbd_cached_profiled<P: crate::profile::Sections>(&self, input: Tensor<3>, clean: &MixerCache, block: usize, valid: Tensor<2>, profile: &mut P) -> Tensor<3> {
        let [b,n,d] = input.dims();
        let start = clean.starts[block];
        let device = input.device();
        let input = input*valid.clone().unsqueeze_dim::<3>(2);
        let history = start.min(3);
        let project = |active: Tensor<3>, index: usize, weight: Tensor<2>| {
            let width = active.dims()[2];
            let combined = Tensor::cat(vec![clean.raw[index].clone().slice([0..b,start-history..start,0..width]),active],1);
            take4(&convolve(self,combined,weight),history,history+n)
        };
        let rq = project(self.q.forward(input.clone()),0,self.q_conv.val());
        let rk = project(self.k.forward(input.clone()),1,self.k_conv.val());
        let rv = project(self.v.forward(input.clone()),2,self.v_conv.val());
        let q = self.normalization_backend.l2(rq.clone());
        let k = self.normalization_backend.l2(rk.clone());
        let v = if self.kind == MixerKind::KronLstm {self.normalization_backend.l2(rv.clone())} else {rv.clone()};
        let ew = if self.kind == MixerKind::Gdn2 {self.key_dim} else {1};
        let ww = if self.kind == MixerKind::Gdn2 {self.value_dim} else {1};
        let initial = clean.states[block].clone().unsqueeze_dim::<5>(1).expand([b,n,self.heads,self.key_dim,self.value_dim])
            .reshape([b*n,self.heads,self.key_dim,self.value_dim]);
        let single = |x: Tensor<4>| {let width = x.dims()[3]; x.reshape([b*n,1,self.heads,width])};
        let alpha = self.normalization_backend.decay(self.decay.forward(input.clone()).reshape([b,n,self.heads,self.key_dim]),self.decay_bias.val(),self.decay_scale.val());
        let erase = sigmoid(self.erase.forward(input.clone())).reshape([b,n,self.heads,ew]);
        let write = sigmoid(self.write.forward(input.clone())).reshape([b,n,self.heads,ww]);
        let recurrent = chunk::scan_selected(self.scan_backend,self.kind,chunk::ScanInput {
            q:single(q),k:single(k),v:single(v),alpha:single(alpha),erase:single(erase),write:single(write),
        },valid.clone().reshape([b*n,1]),Tensor::zeros([b*n,1],&device),self.chunk_size,Some(initial));
        let mut mixed = recurrent.output.reshape([b,n,self.heads,self.value_dim]);
        if let Some(cache) = &self.cache {
            mixed = profile.section("Noisy HOLA read", || {
                let query = hola::normalize(self.hola_backend,self.normalization_backend,&cache.q_norm,rq);
                let active = (self.in_block_reads == InBlockReads::HolaRead).then(|| (
                    hola::normalize(self.hola_backend,self.normalization_backend,&cache.k_norm,rk),
                    rv,
                ));
                let cut = start/self.chunk_size*self.chunk_size;
                let exact = if let Some(config) = self.prefix_retrieval {
                    let positions = if cut > 0 { clean.positions[cut/self.chunk_size-1].clone() }
                        else { Tensor::full([b,self.heads,self.cache_capacity],-1.0,&device) };
                    let t = clean.cache_k.dims()[1];
                    crate::prefix_retrieval::noisy_read(query,clean.cache_k.clone(),clean.cache_v.clone(),
                        Tensor::ones([b,t],&device),Tensor::ones([b,t],&device),
                        Tensor::ones([b,n],&device),valid.clone(),positions,
                        start,self.chunk_size,cache.sink.val(),config,active,clean.prefix_summaries.clone())
                } else {
                    let mut keys = Vec::new();
                    let mut vals = Vec::new();
                    let mut visible = Vec::new();
                    if cut > 0 {
                        let positions = clean.positions[cut/self.chunk_size-1].clone();
                        let ids = positions.clone().clamp_min(0.0).int();
                        keys.push(clean.cache_k.clone().swap_dims(1,2).gather(2,ids.clone().unsqueeze_dim::<4>(3).expand([b,self.heads,self.cache_capacity,self.key_dim])));
                        vals.push(clean.cache_v.clone().swap_dims(1,2).gather(2,ids.unsqueeze_dim::<4>(3).expand([b,self.heads,self.cache_capacity,self.value_dim])));
                        visible.push(positions.greater_equal_elem(0.0).unsqueeze_dim::<4>(2).expand([b,self.heads,n,self.cache_capacity]));
                    }
                    if cut < start {
                        keys.push(take4(&clean.cache_k,cut,start).swap_dims(1,2));
                        vals.push(take4(&clean.cache_v,cut,start).swap_dims(1,2));
                        visible.push(Tensor::<4>::ones([b,self.heads,n,start-cut],&device).greater_elem(0.0));
                    }
                    let active = active.map(|(k,v)| (k.swap_dims(1,2),v.swap_dims(1,2)));
                    noisy_hola_read(query.swap_dims(1,2),keys,vals,visible,active,valid.clone(),cache.sink.val(),self.noisy_read_backend)
                };
                mixed + exact*sigmoid(cache.gate.val()).reshape([1,1,self.heads,1])
            });
        }
        let gate = self.output_gate.forward(input);
        let mixed = if self.normalization_backend == crate::NormalizationBackend::FusedNormsGates {
            self.normalization_backend.gated_rms(&self.norm,mixed,gate.reshape([b,n,self.heads,self.value_dim]))
        } else {
            self.normalization_backend.rms(&self.norm,mixed*sigmoid(gate).reshape([b,n,self.heads,self.value_dim]))
        };
        let _ = d;
        self.project_output(mixed.reshape([b,n,self.heads*self.value_dim]))*valid.unsqueeze_dim::<3>(2)
    }
}

pub struct WaveletCache { approximations: Vec<Tensor<3>> }
pub struct BatchedWaveletCache { histories: Vec<Tensor<3>> }
impl crate::WaveletDyn {
    pub fn prepare_sbd(&self, input: Tensor<3>) -> WaveletCache {
        let [b,t,_] = input.dims();
        let device = input.device();
        let mut approx = self.input.forward(input);
        let mut approximations = Vec::with_capacity(self.levels);
        for level in 0..self.levels {
            approximations.push(approx.clone());
            approx = crate::wavelet::smooth(approx,Tensor::ones([b,t],&device),Tensor::ones([b,t],&device),1<<level,self.backend);
        }
        WaveletCache {approximations}
    }
    pub fn prepare_sbd_batched(&self, input: Tensor<3>, blocks: &[(usize,usize)]) -> BatchedWaveletCache {
        let [rows,time,_] = input.dims();
        assert!(!blocks.is_empty() && blocks.iter().all(|&(row,start)| row<rows && start<time));
        let device = input.device();
        let clean = self.prepare_sbd(input);
        let max_history = blocks.iter().map(|&(_,start)| start).max().unwrap();
        let histories = clean.approximations.iter().enumerate().map(|(level,approx)| {
            let length = ((crate::wavelet::FILTER_TAPS-1)*(1<<level)).min(max_history).max(1);
            let (ids,visible) = history_indices(blocks,time,length,&device);
            gather_history(approx,&ids,&visible)
        }).collect();
        BatchedWaveletCache {histories}
    }
    pub fn forward_sbd_batched(&self, input: Tensor<3>, clean: &BatchedWaveletCache) -> Tensor<3> {
        let [b,n,d] = input.dims();
        let device = input.device();
        let mut approx = self.input.forward(input.clone());
        let mut details = Vec::with_capacity(self.levels);
        for level in 0..self.levels {
            let history = clean.histories[level].dims()[1];
            let combined = Tensor::cat(vec![clean.histories[level].clone(),approx.clone()],1);
            let next = crate::wavelet::smooth(combined,Tensor::ones([b,history+n],&device),
                Tensor::ones([b,history+n],&device),1<<level,self.backend)
                .slice([0..b,history..history+n,0..d]);
            details.push(approx-next.clone());
            approx = next;
        }
        let gates = sigmoid(self.gate.forward(input.clone())).reshape([b,n,self.levels+1,self.groups]);
        let bands = std::iter::once(approx).chain(details.into_iter().rev());
        let mixed = bands.enumerate().map(|(i,x)| x.reshape([b,n,self.groups,d/self.groups])
            *gates.clone().narrow(2,i,1).reshape([b,n,self.groups,1])).reduce(|a,b| a+b).unwrap().reshape([b,n,d]);
        self.output.forward(mixed*self.value.forward(input))
    }
    pub fn forward_sbd_cached(&self, input: Tensor<3>, clean: &WaveletCache, start: usize) -> Tensor<3> {
        let [b,n,d] = input.dims();
        let device = input.device();
        let mut approx = self.input.forward(input.clone());
        let mut details = Vec::with_capacity(self.levels);
        for level in 0..self.levels {
            let history = start.min((crate::wavelet::FILTER_TAPS-1)*(1<<level));
            let combined = Tensor::cat(vec![clean.approximations[level].clone().slice([0..b,start-history..start,0..d]),approx.clone()],1);
            let next = crate::wavelet::smooth(combined,Tensor::ones([b,history+n],&device),Tensor::ones([b,history+n],&device),1<<level,self.backend)
                .slice([0..b,history..history+n,0..d]);
            details.push(approx-next.clone());
            approx = next;
        }
        let gates = sigmoid(self.gate.forward(input.clone())).reshape([b,n,self.levels+1,self.groups]);
        let bands = std::iter::once(approx).chain(details.into_iter().rev());
        let mixed = bands.enumerate().map(|(i,x)| x.reshape([b,n,self.groups,d/self.groups])
            *gates.clone().narrow(2,i,1).reshape([b,n,self.groups,1])).reduce(|a,b| a+b).unwrap().reshape([b,n,d]);
        self.output.forward(mixed*self.value.forward(input))
    }
}
