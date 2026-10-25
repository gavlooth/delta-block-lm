//! Independent scalar f64 noisy-HOLA forward and full reverse-mode derivative.
use anyhow::Result;
use burn::{prelude::*,tensor::activation::sigmoid};
use delta_mixers::sbd::{NoisyReadBackend,noisy_hola_read};
use serde_json::{Value,json};

#[derive(Clone)]
struct Fixture {
    b:usize,h:usize,n:usize,dk:usize,dv:usize,
    lengths:[usize;3], q:Vec<f64>, keys:[Vec<f64>;3], values:[Vec<f64>;3],
    masks:[Vec<bool>;3], valid:Vec<bool>, sink:Vec<f64>,gate:Vec<f64>,up:Vec<f64>,
}
struct Derivative { q:Vec<f64>, keys:[Vec<f64>;3],values:[Vec<f64>;3],sink:Vec<f64>,gate:Vec<f64> }
impl Fixture {
    fn new(b:usize,h:usize,n:usize,cache:usize,unfinished:usize,dk:usize,dv:usize,seed:u64) -> Self {
        let mut rng=crate::data::Rng(seed);
        let mut random=|size| (0..size).map(|_| ((rng.next() as f64/u64::MAX as f64)*1.4-0.7) as f32 as f64).collect::<Vec<_>>();
        let lengths=[cache,unfinished,n];
        let q=random(b*h*n*dk);
        let keys=lengths.map(|m| random(b*h*m*dk));
        let values=lengths.map(|m| random(b*h*m*dv));
        let sink=random(h); let gate=random(h); let up=random(b*n*h*dv);
        let valid=(0..b*n).map(|i| i%n < n-(i/n%2+usize::from(n>3))).collect::<Vec<_>>();
        let masks=std::array::from_fn(|segment| {
            let m=lengths[segment];
            (0..b*h*m).map(|i| if segment==2 {valid[i/(h*m)*n+i%m]} else {segment==1 || i%3!=1}).collect()
        });
        Self{b,h,n,dk,dv,lengths,q,keys,values,masks,valid,sink,gate,up}
    }
    fn oracle(&self) -> (Vec<f64>,Derivative) {
        let mut y=vec![0.0;self.b*self.n*self.h*self.dv];
        let mut g=Derivative{q:vec![0.0;self.q.len()],keys:self.keys.each_ref().map(|x|vec![0.0;x.len()]),values:self.values.each_ref().map(|x|vec![0.0;x.len()]),sink:vec![0.0;self.h],gate:vec![0.0;self.h]};
        let scale=1.0/(self.dk as f64).sqrt();
        for b in 0..self.b { for h in 0..self.h { for i in 0..self.n {
            if !self.valid[b*self.n+i] {continue;}
            let qi=((b*self.h+h)*self.n+i)*self.dk;
            let yi=((b*self.n+i)*self.h+h)*self.dv;
            let gate=1.0/(1.0+(-self.gate[h]).exp());
            let mut entries=Vec::new();
            for segment in 0..3 { for j in 0..self.lengths[segment] {
                let row=(b*self.h+h)*self.lengths[segment]+j;
                if !self.masks[segment][row] {continue;}
                let dot=(0..self.dk).map(|d|self.q[qi+d]*self.keys[segment][row*self.dk+d]).sum::<f64>()*scale;
                entries.push((segment,row,dot));
            }}
            let maximum=entries.iter().map(|x|x.2).fold(self.sink[h],f64::max);
            let denominator=(self.sink[h]-maximum).exp()+entries.iter().map(|x|(x.2-maximum).exp()).sum::<f64>();
            let mut read=vec![0.0;self.dv];
            for &(segment,row,logit) in &entries {
                let p=(logit-maximum).exp()/denominator;
                for d in 0..self.dv {read[d]+=p*self.values[segment][row*self.dv+d];}
            }
            let dot=(0..self.dv).map(|d|self.up[yi+d]*read[d]).sum::<f64>();
            g.sink[h]-=(self.sink[h]-maximum).exp()/denominator*gate*dot;
            g.gate[h]+=dot*gate*(1.0-gate);
            for d in 0..self.dv {y[yi+d]=read[d]*gate;}
            for &(segment,row,logit) in &entries {
                let p=(logit-maximum).exp()/denominator;
                let adj=p*gate*((0..self.dv).map(|d|self.up[yi+d]*self.values[segment][row*self.dv+d]).sum::<f64>()-dot);
                for d in 0..self.dk {
                    g.q[qi+d]+=adj*self.keys[segment][row*self.dk+d]*scale;
                    g.keys[segment][row*self.dk+d]+=adj*self.q[qi+d]*scale;
                }
                for d in 0..self.dv {g.values[segment][row*self.dv+d]+=p*gate*self.up[yi+d];}
            }
        }}}
        (y,g)
    }
    fn objective(&self) -> f64 {self.oracle().0.iter().zip(&self.up).map(|(y,u)|y*u).sum()}
    fn field(&mut self,field:usize) -> &mut Vec<f64> {
        match field {0=>&mut self.q,1..=3=>&mut self.keys[field-1],4..=6=>&mut self.values[field-4],7=>&mut self.sink,8=>&mut self.gate,_=>unreachable!()}
    }
    fn finite_differences(&self,g:&Derivative) -> Result<Value> {
        let fields=std::iter::once(&g.q).chain(g.keys.iter()).chain(g.values.iter()).chain([&g.sink,&g.gate]);
        let mut actual=Vec::new();let mut reference=Vec::new();
        for (field,expected) in fields.enumerate() {
            for i in (0..expected.len()).step_by((expected.len()/7).max(1)) {
                let mut plus=self.clone(); let mut minus=self.clone();
                plus.field(field)[i]+=1e-5; minus.field(field)[i]-=1e-5;
                actual.push((plus.objective()-minus.objective())/2e-5);
                reference.push(expected[i]);
            }
        }
        crate::sbd::compare(&actual,&reference)
    }
    fn tensor(&self,device:&Device,backend:NoisyReadBackend) -> Result<Value> {
        let device=device.clone().autodiff();
        let t4=|x:&[f64],dims:[usize;4]| Tensor::<4>::from_data(TensorData::new(x.iter().map(|&x|x as f32).collect::<Vec<_>>(),dims),&device).require_grad();
        let q=t4(&self.q,[self.b,self.h,self.n,self.dk]);
        let keys=std::array::from_fn::<_,3,_>(|s|t4(&self.keys[s],[self.b,self.h,self.lengths[s],self.dk]));
        let values=std::array::from_fn::<_,3,_>(|s|t4(&self.values[s],[self.b,self.h,self.lengths[s],self.dv]));
        let vector=|x:&[f64]|Tensor::<1>::from_data(TensorData::new(x.iter().map(|&x|x as f32).collect::<Vec<_>>(),[x.len()]),&device).require_grad();
        let sink=vector(&self.sink);let gate=vector(&self.gate);
        let valid=Tensor::<2>::from_data(TensorData::new(self.valid.iter().map(|&x|if x{1.0f32}else{0.0}).collect::<Vec<_>>(),[self.b,self.n]),&device);
        let mut ks=Vec::new();let mut vs=Vec::new();let mut masks=Vec::new();
        for s in 0..2 {
            if self.lengths[s]==0 {continue;}
            ks.push(keys[s].clone());vs.push(values[s].clone());
            masks.push(Tensor::<3,Bool>::from_data(TensorData::new(self.masks[s].clone(),[self.b,self.h,self.lengths[s]]),&device).unsqueeze_dim::<4>(2).expand([self.b,self.h,self.n,self.lengths[s]]));
        }
        let y=noisy_hola_read(q.clone(),ks,vs,masks,Some((keys[2].clone(),values[2].clone())),valid,sink.clone(),backend)*sigmoid(gate.clone()).reshape([1,1,self.h,1]);
        let upstream=t4(&self.up,[self.b,self.n,self.h,self.dv]);
        let gradients=(y.clone()*upstream).sum().backward();
        let (expected,g)=self.oracle();
        let read=|x:Tensor<4>|x.into_data().try_to_vec::<f32>().unwrap();
        let forward=crate::sbd::compare(&read(y),&expected)?;
        let mut reports=vec![json!({"input":"query","error":crate::sbd::compare(&read(q.grad(&gradients).unwrap()),&g.q)?})];
        for s in 0..3 {
            if self.lengths[s]==0 {continue;}
            reports.push(json!({"input":format!("segment_{s}_key"),"error":crate::sbd::compare(&read(keys[s].grad(&gradients).unwrap()),&g.keys[s])?}));
            reports.push(json!({"input":format!("segment_{s}_value"),"error":crate::sbd::compare(&read(values[s].grad(&gradients).unwrap()),&g.values[s])?}));
        }
        for (name,tensor,reference) in [("sink",sink,&g.sink),("gate",gate,&g.gate)] {
            reports.push(json!({"input":name,"error":crate::sbd::compare(&tensor.grad(&gradients).unwrap().into_data().try_to_vec::<f32>().unwrap(),reference)?}));
        }
        Ok(json!({"backend":format!("{backend:?}"),"forward":forward,"full_vjp":reports}))
    }
}
pub fn run(device:&Device,gpu:bool) -> Result<Value> {
    let mut cases=Vec::new();
    for (index,(b,h,n,cache,unfinished)) in [(1,1,2,0,0),(2,2,4,4,3),(1,2,8,4,0),(2,1,16,0,7),(1,2,3,2,15)].into_iter().enumerate() {
        let fixture=Fixture::new(b,h,n,cache,unfinished,32,64,991+index as u64);
        let (_,derivative)=fixture.oracle();
        let finite=fixture.finite_differences(&derivative)?;
        let cpu=fixture.tensor(&Device::ndarray(),NoisyReadBackend::Tensor)?;
        let mut paths=vec![cpu];
        if gpu {paths.push(fixture.tensor(device,NoisyReadBackend::Tensor)?);paths.push(fixture.tensor(device,NoisyReadBackend::FusedGpu)?);}
        cases.push(json!({"batch":b,"heads":h,"queries":n,"cache_slots":cache,"unfinished_chunk":unfinished,"finite_differences":finite,"paths":paths}));
    }
    Ok(json!({"passed":true,"cases":cases,"scope":"independent scalar f64 forward and full VJP of noisy query, all three K/V segments, sink and sigmoid gate; central finite differences sampled in every nonempty input; random values, padding, empty cache, cut==start and unfinished clean chunks"}))
}
#[cfg(test)]
mod tests {
    #[test]
    fn noisy_hola_matches_independent_oracle() {super::run(&burn::prelude::Device::ndarray(),false).unwrap();}
}
