//! Exact active-block attention. The caller supplies only noisy-block Q/K/V.
use burn::{prelude::*, tensor::activation::softmax};

/// Axes are [batch,time,head,channel]. Padding contributes neither keys nor queries.
pub fn attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, valid: Tensor<2>) -> Tensor<4> {
    let [b, t, h, d] = q.dims();
    let dv = v.dims()[3];
    let inactive = valid.clone().lower_elem(0.5).reshape([b,t,1,1]);
    let q = q.mask_fill(inactive.clone().expand([b,t,h,d]),0.0);
    let k = k.mask_fill(inactive.clone().expand([b,t,h,d]),0.0);
    let v = v.mask_fill(inactive.clone().expand([b,t,h,dv]),0.0);
    let visible = valid.clone().reshape([b, 1, 1, t]).greater_elem(0.5);
    let logits = q.swap_dims(1, 2).matmul(k.swap_dims(1, 2).swap_dims(2, 3)) / (d as f64).sqrt();
    let probabilities = softmax(logits.mask_fill(visible.bool_not(), -1.0e30), 3);
    probabilities.matmul(v.swap_dims(1, 2)).swap_dims(1, 2)
        .mask_fill(inactive.expand([b,t,h,dv]),0.0)
}

use crate::{Mixer, MixerKind, chunk, hola};
use burn::tensor::activation::{sigmoid,silu};

/// Differentiable clean tensors prepared once per layer/sequence. Snapshot work
/// scans disjoint segments with carried state, never all prefixes per block.
pub struct MixerCache {
    raw: [Tensor<3>;3],
    cache_k: Tensor<4>,
    cache_v: Tensor<4>,
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
impl Mixer {
    pub fn prepare_sbd(&self, input: Tensor<3>, starts: &[usize]) -> MixerCache {
        let [b,t,_] = input.dims();
        assert_eq!(b,1,"one document per training snapshot");
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
        MixerCache {raw,cache_k,cache_v:rv,positions,states,starts:starts.to_vec()}
    }
    pub fn forward_sbd_cached(&self, input: Tensor<3>, clean: &MixerCache, block: usize) -> Tensor<3> {
        let [b,n,d] = input.dims();
        let start = clean.starts[block];
        let device = input.device();
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
        let k = self.normalization_backend.l2(rk);
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
            q:single(q.clone()),k:single(k.clone()),v:single(v.clone()),alpha:single(alpha),erase:single(erase),write:single(write),
        },Tensor::ones([b*n,1],&device),Tensor::zeros([b*n,1],&device),self.chunk_size,Some(initial));
        let mut mixed = recurrent.output.reshape([b,n,self.heads,self.value_dim])
            + attention(q,k,v,Tensor::ones([b,n],&device));
        if let Some(cache) = &self.cache {
            let query = hola::normalize(self.hola_backend,self.normalization_backend,&cache.q_norm,rq).swap_dims(1,2);
            let cut = start/self.chunk_size*self.chunk_size;
            let mut keys = Vec::new();
            let mut vals = Vec::new();
            let mut visible = Vec::new();
            if cut > 0 {
                let positions = clean.positions[cut/self.chunk_size-1].clone();
                let ids = positions.clone().clamp_min(0.0).int();
                keys.push(clean.cache_k.clone().swap_dims(1,2).gather(2,ids.clone().unsqueeze_dim::<4>(3).expand([b,self.heads,self.cache_capacity,self.key_dim])));
                vals.push(clean.cache_v.clone().swap_dims(1,2).gather(2,ids.unsqueeze_dim::<4>(3).expand([b,self.heads,self.cache_capacity,self.value_dim])));
                visible.push(positions.greater_equal_elem(0.0));
            }
            if cut < start {
                keys.push(take4(&clean.cache_k,cut,start).swap_dims(1,2));
                vals.push(take4(&clean.cache_v,cut,start).swap_dims(1,2));
                visible.push(Tensor::<3>::ones([b,self.heads,start-cut],&device).greater_elem(0.0));
            }
            let keys = Tensor::cat(keys,2);
            let vals = Tensor::cat(vals,2);
            let visible = Tensor::cat(visible,2).unsqueeze_dim::<4>(2);
            let logits = (query.matmul(keys.swap_dims(2,3))/(self.key_dim as f64).sqrt()).mask_fill(visible.bool_not(),-1.0e30);
            let probabilities = softmax(Tensor::cat(vec![logits,cache.sink.val().reshape([1,self.heads,1,1]).expand([b,self.heads,n,1])],3),3);
            let exact = probabilities.matmul(Tensor::cat(vec![vals,Tensor::zeros([b,self.heads,1,self.value_dim],&device)],2)).swap_dims(1,2);
            mixed = mixed + exact*sigmoid(cache.gate.val()).reshape([1,1,self.heads,1]);
        }
        let gate = self.output_gate.forward(input);
        let mixed = if self.normalization_backend == crate::NormalizationBackend::FusedNormsGates {
            self.normalization_backend.gated_rms(&self.norm,mixed,gate.reshape([b,n,self.heads,self.value_dim]))
        } else {
            self.normalization_backend.rms(&self.norm,mixed*sigmoid(gate).reshape([b,n,self.heads,self.value_dim]))
        };
        let _ = d;
        self.project_output(mixed.reshape([b,n,self.heads*self.value_dim]))
    }
}

pub struct WaveletCache { approximations: Vec<Tensor<3>> }
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
