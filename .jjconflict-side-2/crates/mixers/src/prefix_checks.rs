//! Independent scalar f64 selection, read and analytic first-order derivatives.
use burn::prelude::*;
use serde::Serialize;
use crate::{CachePolicy,hola,prefix_retrieval::{PrefixReadConfig,PrefixRetrievalBackend}};
#[derive(Serialize)]
pub struct PrefixCheck {
    pub backend: PrefixRetrievalBackend, pub tokens: usize, pub block_size: usize,
    pub blocks_per_query: usize, pub frozen_start: Option<usize>,
    pub active_block: bool,
    pub output_max_abs: f64, pub q_gradient_max_abs: f64, pub k_gradient_max_abs: f64,
    pub v_gradient_max_abs: f64, pub sink_gradient_max_abs: f64,
    pub selection_scores_detached: bool,
}
struct Fixture {b:usize,t:usize,h:usize,q:Vec<f32>,k:Vec<f32>,v:Vec<f32>,scores:Vec<f32>,
    valid:Vec<f32>,docs:Vec<f32>,sink:Vec<f32>,up:Vec<f32>}
impl Fixture {
    fn new(t:usize) -> Self {
        let (b,h)=(2,2);
        let signal=|width,phase| (0..b*t*h*width).map(|i| (0.31*((i*13+phase) as f64*0.071).sin()) as f32).collect::<Vec<_>>();
        let mut q=signal(32,7); let k=signal(32,11);
        // Head zero makes every eligible block score tie exactly.
        for row in 0..b*t {q[row*h*32..row*h*32+32].fill(0.0);}
        let mut valid=vec![1.;b*t]; let mut docs=vec![0.;b*t];
        for batch in 0..b {for s in 0..t {
            docs[batch*t+s]=if batch==0 {0.} else if s<19 {0.} else if s<43 {1.} else {0.};
            if s==3 || s==27 || (batch==1 && (32..40).contains(&s)) || s+2>=t {valid[batch*t+s]=0.;}
        }}
        Self {b,t,h,q,k,v:signal(64,17),scores:(0..b*t*h).map(|i| ((i/h%t)%7) as f32).collect(),
            valid,docs,sink:vec![0.4,-0.7],up:signal(64,23)}
    }
    fn ix(&self,b:usize,t:usize,h:usize,c:usize,d:usize)->usize {((b*self.t+t)*self.h+h)*d+c}
    fn oracle(&self,block:usize,top:usize,start:Option<usize>,active:bool) -> Oracle {
        let mut result=Oracle {output:vec![0.;self.v.len()],dq:vec![0.;self.q.len()],dk:vec![0.;self.k.len()],
            dv:vec![0.;self.v.len()],ds:vec![0.;self.h]};
        let capacity=4; let chunk=8;
        for b in 0..self.b {for h in 0..self.h {
            let mut snapshots=vec![Vec::<usize>::new()];
            let mut cache=Vec::<usize>::new();
            for first in (0..self.t).step_by(chunk) {
                let end=(first+chunk).min(self.t);
                cache.extend(first..end);
                cache.retain(|&s|self.valid[b*self.t+s]>0.5 && self.docs[b*self.t+s]==self.docs[b*self.t+end-1]);
                cache.sort_by(|&a,&z|self.scores[(b*self.t+z)*self.h+h].total_cmp(&self.scores[(b*self.t+a)*self.h+h]).then(a.cmp(&z)));
                cache.truncate(capacity); snapshots.push(cache.clone());
            }
            for t in 0..self.t {
                if self.valid[b*self.t+t]<0.5 {continue;}
                let noisy=start.is_some_and(|s|t>=s);
                let cut=if noisy {start.unwrap()/chunk*chunk} else {t/chunk*chunk};
                let end=if noisy {start.unwrap()} else {t+1};
                let cache=&snapshots[cut/chunk];
                let doc=self.docs[b*self.t+t];
                let same=|s:usize|self.valid[b*self.t+s]>0.5 && self.docs[b*self.t+s]==doc;
                let mut sources=cache.iter().copied().filter(|&s|same(s)).collect::<Vec<_>>();
                sources.extend((cut..end).filter(|&s|same(s)));
                let mut blocks=Vec::new();
                for j in 0..cut/block {
                    let tokens=(j*block..(j+1)*block).filter(|&s|same(s)).collect::<Vec<_>>();
                    if tokens.is_empty() {continue;}
                    let mut score=0.;
                    for c in 0..32 {
                        let mean=tokens.iter().map(|&s|self.k[self.ix(b,s,h,c,32)] as f64).sum::<f64>()/tokens.len() as f64;
                        score+=self.q[self.ix(b,t,h,c,32)] as f64*mean/32f64.sqrt();
                    }
                    blocks.push((j,score,tokens));
                }
                blocks.sort_by(|a,z|z.1.total_cmp(&a.1).then(a.0.cmp(&z.0)));
                for (_,_,tokens) in blocks.into_iter().take(top) {sources.extend(tokens.into_iter().filter(|s|!cache.contains(s)));}
                let own_start=if noisy {start.unwrap()/block*block}else{t/block*block};
                for s in own_start..end {
                    if same(s) && !sources.contains(&s) {sources.push(s);}
                }
                if noisy && active {
                    sources.extend((start.unwrap()..self.t).filter(|&s|self.valid[b*self.t+s]>0.5));
                }
                let mut logits=sources.iter().map(|&s|(0..32).map(|c|self.q[self.ix(b,t,h,c,32)] as f64*self.k[self.ix(b,s,h,c,32)] as f64).sum::<f64>()/32f64.sqrt()).collect::<Vec<_>>();
                logits.push(self.sink[h] as f64);
                let maximum=logits.iter().copied().fold(f64::NEG_INFINITY,f64::max);
                let sum=logits.iter().map(|x|(x-maximum).exp()).sum::<f64>();
                let probabilities=logits.iter().map(|x|(x-maximum).exp()/sum).collect::<Vec<_>>();
                for c in 0..64 {
                    result.output[self.ix(b,t,h,c,64)]=sources.iter().enumerate().map(|(i,&s)|probabilities[i]*self.v[self.ix(b,s,h,c,64)] as f64).sum();
                }
                let delta=(0..64).map(|c|self.up[self.ix(b,t,h,c,64)] as f64*result.output[self.ix(b,t,h,c,64)]).sum::<f64>();
                result.ds[h]-=probabilities[sources.len()]*delta;
                for (i,&s) in sources.iter().enumerate() {
                    let value=(0..64).map(|c|self.up[self.ix(b,t,h,c,64)] as f64*self.v[self.ix(b,s,h,c,64)] as f64).sum::<f64>();
                    let derivative=probabilities[i]*(value-delta)/32f64.sqrt();
                    for c in 0..32 {
                        result.dq[self.ix(b,t,h,c,32)]+=derivative*self.k[self.ix(b,s,h,c,32)] as f64;
                        result.dk[self.ix(b,s,h,c,32)]+=derivative*self.q[self.ix(b,t,h,c,32)] as f64;
                    }
                    for c in 0..64 {result.dv[self.ix(b,s,h,c,64)]+=probabilities[i]*self.up[self.ix(b,t,h,c,64)] as f64;}
                }
            }
        }}
        result
    }
}
struct Oracle {output:Vec<f64>,dq:Vec<f64>,dk:Vec<f64>,dv:Vec<f64>,ds:Vec<f64>}
fn values<const D:usize>(x:Tensor<D>)->Vec<f32>{x.into_data().try_to_vec::<f32>().unwrap()}
fn compare(name:&str,actual:&[f32],reference:&[f64])->Result<f64,String>{
    let mut maximum=0f64;
    if actual.len()!=reference.len(){return Err(format!("{name}: shape mismatch"));}
    for (i,(&a,&r)) in actual.iter().zip(reference).enumerate(){
        let error=(a as f64-r).abs(); maximum=maximum.max(error);
        if !a.is_finite() || error>2e-5+3e-4*r.abs(){return Err(format!("{name}[{i}] actual={a} reference={r} error={error} tolerance={}",2e-5+3e-4*r.abs()));}
    }
    Ok(maximum)
}
pub fn reference(device:&Device)->Result<Vec<PrefixCheck>,String>{
    let mut reports=Vec::new();
    let train=device.clone().autodiff();
    #[cfg(any(feature="vulkan",feature="cuda"))]
    let backends=if matches!(device.clone().inner().as_dispatch(),burn::backend::DispatchDevice::Cube(_)) {
        vec![PrefixRetrievalBackend::Tensor,PrefixRetrievalBackend::FusedGpu]
    } else {vec![PrefixRetrievalBackend::Tensor]};
    #[cfg(not(any(feature="vulkan",feature="cuda")))]
    let backends=vec![PrefixRetrievalBackend::Tensor];
    for backend in backends {for (block,top,start,active,tokens) in [
        (16,2,None,false,81),(16,9,None,false,81),(16,2,Some(51),false,81),(16,9,Some(16),false,81),
        (32,2,Some(65),false,81),(32,2,Some(65),true,81),(16,9,Some(73),true,81),
        (64,2,None,false,161),(64,2,Some(145),true,161),(16,9,Some(73),true,89)] {
        let f=Fixture::new(tokens);
        let expected=f.oracle(block,top,start,active);
        let upload=|v:&Vec<f32>,d|Tensor::<4>::from_data(TensorData::new(v.clone(),[f.b,f.t,f.h,d]),&train).require_grad();
        let q=upload(&f.q,32);let k=upload(&f.k,32);let v=upload(&f.v,64);let scores=upload(&f.scores,1);
        let valid=Tensor::from_data(TensorData::new(f.valid.clone(),[f.b,f.t]),&train);
        let docs=Tensor::from_data(TensorData::new(f.docs.clone(),[f.b,f.t]),&train);
        let sink=Tensor::<1>::from_data(TensorData::new(f.sink.clone(),[f.h]),&train).require_grad();
        let config=PrefixReadConfig{block_size:block,blocks_per_query:top,backend};
        let out=if active {
            let s=start.unwrap();let cut=s/8*8;
            let take=|x:&Tensor<4>,a,z|x.clone().slice([0..f.b,a..z,0..f.h,0..x.dims()[3]]);
            let clean=hola::read_prefix(take(&q,0,s),take(&k,0,s),take(&v,0,s),take(&scores,0,s),
                valid.clone().slice([0..f.b,0..s]),docs.clone().slice([0..f.b,0..s]),4,8,
                CachePolicy::Surprise,sink.clone(),None,config,crate::sbd::InBlockReads::None);
            let positions=hola::read_detailed(take(&q,0,cut),take(&k,0,cut),take(&v,0,cut),take(&scores,0,cut),
                valid.clone().slice([0..f.b,0..cut]),docs.clone().slice([0..f.b,0..cut]),4,8,
                CachePolicy::Surprise,sink.clone()).final_positions;
            let noisy=crate::prefix_retrieval::noisy_read(take(&q,s,f.t),k.clone(),v.clone(),valid.clone(),docs.clone(),
                docs.slice([0..f.b,s..f.t]),valid.slice([0..f.b,s..f.t]),positions,s,8,sink.clone(),config,
                Some((take(&k,s,f.t),take(&v,s,f.t))),None);
            Tensor::cat(vec![clean,noisy],1)
        } else {
            hola::read_prefix(q.clone(),k.clone(),v.clone(),scores.clone(),valid,docs,4,8,
                CachePolicy::Surprise,sink.clone(),start,config,crate::sbd::InBlockReads::None)
        };
        let output_max_abs=compare("prefix output",&values(out.clone()),&expected.output)?;
        let up=Tensor::<4>::from_data(TensorData::new(f.up.clone(),[f.b,f.t,f.h,64]),&train);
        let grads=(out*up).sum().backward();
        let q_gradient_max_abs=compare("prefix dq",&values(q.grad(&grads).ok_or("missing dq")?),&expected.dq)?;
        let k_gradient_max_abs=compare("prefix dk",&values(k.grad(&grads).ok_or("missing dk")?),&expected.dk)?;
        let v_gradient_max_abs=compare("prefix dv",&values(v.grad(&grads).ok_or("missing dv")?),&expected.dv)?;
        let sink_gradient_max_abs=compare("prefix ds",&values(sink.grad(&grads).ok_or("missing ds")?),&expected.ds)?;
        if scores.grad(&grads).is_some(){return Err("selection scores differentiated".into());}
        reports.push(PrefixCheck{backend,tokens:f.t,block_size:block,blocks_per_query:top,frozen_start:start,active_block:active,
            output_max_abs,q_gradient_max_abs,k_gradient_max_abs,v_gradient_max_abs,sink_gradient_max_abs,selection_scores_detached:true});
    }}
    Ok(reports)
}
