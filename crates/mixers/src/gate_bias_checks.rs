//! Whole-layer f64 oracle for arXiv:2610.26834 §2.2, Eqs. (10)–(12).
//!
//! This deliberately does not call the tensor recurrence, projection packing, or
//! existing scan oracle. Eq. (11)'s beta/c biases and the following output-gate
//! bias are nonzero. The release's L2 norm floor, per-head RMS epsilon, masking
//! and independent temporary SBD updates are included explicitly.
use crate::{CachePolicy, Mixer, MixerConfig, MixerKind, ProjectionBackend, ScanBackend};
use burn::{module::Param, nn::Linear, optim::GradientsParams, prelude::*};
use serde::Serialize;

const ATOL: f64 = 2e-5;
const RTOL: f64 = 3e-4;
const DIFFERENCE: f64 = 1e-5;
const WEIGHTS: usize = 0;
const BIASES: usize = 8;
const CONVOLUTIONS: usize = 16;
const DECAY_SCALE: usize = 19;
const DECAY_BIAS: usize = 20;
const RMS_SCALE: usize = 21;

fn values<const D: usize>(x: Tensor<D>) -> Vec<f64> {
    x.into_data().try_to_vec::<f32>().expect("f32 bias fixture").into_iter().map(f64::from).collect()
}
fn tensor<const D: usize>(x: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(x.iter().map(|&x| x as f32).collect::<Vec<_>>(), shape), device)
}
fn layers(m: &Mixer) -> [&Linear; 8] {
    [&m.q, &m.k, &m.v, &m.decay, &m.erase, &m.write, &m.output_gate, &m.output]
}
fn compare(actual: &[f64], expected: &[f64], label: &str) -> Result<f64, String> {
    if actual.len() != expected.len() { return Err(format!("{label}: element counts differ")); }
    let mut maximum = 0.0f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (a-e).abs();
        if !a.is_finite() || !e.is_finite() || error > ATOL + RTOL*e.abs() {
            return Err(format!("{label}[{i}]: {a} != f64 {e}, error {error}"));
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
fn sigmoid(x: f64) -> f64 { 1.0/(1.0+(-x).exp()) }
fn pattern(n: usize, phase: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|i| (scale*((i*17+phase) as f64*0.137).sin()) as f32 as f64).collect()
}
fn assign<const D: usize>(p: &mut Param<Tensor<D>>, data: Vec<f64>, device: &Device) {
    let shape = p.val().dims();
    *p = p.clone().map(|_| tensor(&data, shape, device).require_grad());
}
fn fixture(device: &Device, backend: ScanBackend) -> Mixer {
    let (key_dim, value_dim) = if backend.is_fused() { (32,64) } else { (3,3) };
    let mut m = MixerConfig {kind: MixerKind::KronLstm, cache_policy: CachePolicy::None,
        scan_backend: backend, d_model: 4, heads: 2, key_dim, value_dim, chunk_size: 4,
        ..Default::default()}.init(device);
    for (field, layer) in [&mut m.q,&mut m.k,&mut m.v,&mut m.decay,&mut m.erase,
        &mut m.write,&mut m.output_gate,&mut m.output].into_iter().enumerate() {
        let [a,b] = layer.weight.val().dims();
        assign(&mut layer.weight, pattern(a*b,field*11+3,0.35),device);
    }
    for (field, layer) in [&mut m.erase,&mut m.write,&mut m.output_gate].into_iter().enumerate() {
        let bias = layer.bias.as_mut().expect("KronLSTM gate bias must exist");
        let n = bias.val().dims()[0];
        let data = (0..n).map(|i| ((if (i+field)%2==0 {-0.65} else {0.45})
            + 0.03*(i%7) as f64) as f32 as f64).collect();
        assign(bias,data,device);
    }
    for (field, p) in [&mut m.q_conv,&mut m.k_conv,&mut m.v_conv].into_iter().enumerate() {
        let [lags,channels] = p.val().dims();
        let data = (0..lags*channels).map(|i| {
            let lag=i/channels;
            ([0.8,0.17,-0.09,0.06][lag]+0.007*((i+field)%5) as f64) as f32 as f64
        }).collect();
        assign(p,data,device);
    }
    assign(&mut m.decay_scale,vec![-0.7,-0.2],device);
    assign(&mut m.decay_bias,pattern(2*key_dim,19,0.4),device);
    assign(&mut m.norm.gamma,(0..value_dim).map(|i|(0.8+0.13*(i%5) as f64) as f32 as f64).collect(),device);
    m
}

#[derive(Clone)]
struct Oracle {
    d: usize, h: usize, k: usize, v: usize, epsilon: f64,
    widths: [usize; 8],
    // Input-major matrices; absent biases have empty vectors.
    parameters: Vec<Vec<f64>>,
}
impl Oracle {
    fn from_mixer(m: &Mixer) -> Self {
        let linear = layers(m);
        let mut parameters = linear.iter().map(|l| values(l.weight.val())).collect::<Vec<_>>();
        parameters.extend(linear.iter().map(|l| l.bias.as_ref().map_or_else(Vec::new,|b|values(b.val()))));
        parameters.extend([values(m.q_conv.val()),values(m.k_conv.val()),values(m.v_conv.val()),
            values(m.decay_scale.val()),values(m.decay_bias.val()),values(m.norm.gamma.val())]);
        Self {d:m.q.weight.val().dims()[0],h:m.heads,k:m.key_dim,v:m.value_dim,epsilon:m.norm.epsilon,
            widths:linear.map(|l|l.weight.val().dims()[1]),parameters}
    }
    fn linear(&self, field: usize, x: &[f64]) -> Vec<f64> {
        let width=self.widths[field];
        (0..width).map(|j| {
            let bias=self.parameters[BIASES+field].get(j).copied().unwrap_or(0.0);
            bias+x.iter().enumerate().map(|(i,&x)|x*self.parameters[WEIGHTS+field][i*width+j]).sum::<f64>()
        }).collect()
    }
    fn normalize(x: &mut [f64]) {
        let norm=x.iter().map(|x|x*x).sum::<f64>().sqrt().max(1e-12);
        for x in x { *x/=norm; }
    }
    fn output(&self, s: &Scenario) -> Vec<f64> {
        let (h,k,v,d)=(self.h,self.k,self.v,self.d);
        let mut result=Vec::new();
        for row in 0..s.batch {
            let start=s.start.unwrap_or(0);
            let time=if s.start.is_some() {start+s.noisy_time} else {s.clean_time};
            let mut input=Vec::with_capacity(time*d);
            let mut valid=Vec::with_capacity(time);
            let mut document=Vec::with_capacity(time);
            let mut reset=Vec::with_capacity(time);
            let mut current_document=0;
            for t in 0..time {
                let noisy=s.start.is_some() && t>=start;
                let position=if noisy {row*s.noisy_time+t-start} else {row*s.clean_time+t};
                let active=if noisy {s.noisy_valid[position]} else {s.clean_valid[position]};
                let resets=if noisy {false} else {s.reset[position]};
                if active && resets {current_document+=1;}
                valid.push(active);reset.push(resets);document.push(current_document);
                let source=if noisy {&s.noisy} else {&s.clean};
                input.extend(source[position*d..(position+1)*d].iter().map(|&x|if active {x} else {0.0}));
            }
            let projections: Vec<Vec<Vec<f64>>>=(0..time).map(|t|(0..7)
                .map(|field|self.linear(field,&input[t*d..(t+1)*d])).collect()).collect();
            let mut state=vec![0.0;h*k*v];
            let mut prefix=state.clone();
            for t in 0..time {
                if s.start.is_some() && t==start {prefix=state.clone();}
                let mut qkv=Vec::with_capacity(3);
                for field in 0..3 {
                    let width=self.widths[field];
                    let mut activated=vec![0.0;width];
                    for j in 0..width {
                        let mut sum=0.0;
                        for lag in 0..4.min(t+1) {
                            let earlier=t-lag;
                            if valid[t] && valid[earlier] && document[t]==document[earlier] {
                                sum+=projections[earlier][field][j]*self.parameters[CONVOLUTIONS+field][lag*width+j];
                            }
                        }
                        activated[j]=sum*sigmoid(sum);
                    }
                    let size=if field==2 {v} else {k};
                    for head in 0..h {Self::normalize(&mut activated[head*size..(head+1)*size]);}
                    qkv.push(activated);
                }
                let mut current=if s.start.is_some() && t>=start {prefix.clone()} else {state.clone()};
                let mut mixed=vec![0.0;h*v];
                if valid[t] {
                    if reset[t] {current.fill(0.0);}
                    for head in 0..h {
                        // Paper Eq. (12): componentwise retention, learned positive head power.
                        for i in 0..k {
                            let pre=projections[t][3][head*k+i]+self.parameters[DECAY_BIAS][head*k+i];
                            let retention=(-self.parameters[DECAY_SCALE][head].exp()*pre.exp().ln_1p()).exp();
                            for j in 0..v {current[(head*k+i)*v+j]*=retention;}
                        }
                        // Eq. (11) includes biases BEFORE the two separate sigmoids.
                        let beta=sigmoid(projections[t][4][head]);
                        let target=sigmoid(projections[t][5][head]);
                        let mut association=0.0;
                        for i in 0..k {for j in 0..v {
                            association+=qkv[2][head*v+j]*current[(head*k+i)*v+j]*qkv[1][head*k+i];
                        }}
                        // Eq. (10): the write target is NOT multiplied by beta.
                        let correction=target-beta*association;
                        for i in 0..k {for j in 0..v {
                            current[(head*k+i)*v+j]+=correction*qkv[2][head*v+j]*qkv[1][head*k+i];
                            mixed[head*v+j]+=current[(head*k+i)*v+j]*qkv[0][head*k+i]/(k as f64).sqrt();
                        }}
                        for j in 0..v {mixed[head*v+j]*=sigmoid(projections[t][6][head*v+j]);}
                        let rms=(mixed[head*v..(head+1)*v].iter().map(|x|x*x).sum::<f64>()/v as f64+self.epsilon).sqrt();
                        for j in 0..v {mixed[head*v+j]=mixed[head*v+j]/rms*self.parameters[RMS_SCALE][j];}
                    }
                    if s.start.is_none() || t<start {state=current;}
                }
                if s.start.is_none() || t>=start {
                    let output=self.linear(7,&mixed);
                    result.extend(output.into_iter().map(|x|if valid[t] {x} else {0.0}));
                }
            }
        }
        result
    }
    fn objective(&self, s: &Scenario) -> f64 {
        self.output(s).iter().zip(&s.upstream).map(|(x,g)|x*g).sum()
    }
}
#[derive(Clone)]
struct Scenario {
    batch: usize, clean_time: usize, noisy_time: usize, start: Option<usize>,
    clean: Vec<f64>, noisy: Vec<f64>, clean_valid: Vec<bool>, noisy_valid: Vec<bool>,
    reset: Vec<bool>, upstream: Vec<f64>,
}
impl Scenario {
    fn new(start: Option<usize>) -> Self {
        let (batch,clean_time,noisy_time,d)=(2,9,3,4);
        let clean=pattern(batch*clean_time*d,31,0.7);
        let noisy=pattern(batch*noisy_time*d,53,0.6);
        let mut clean_valid=vec![true;batch*clean_time];
        let mut reset=vec![false;batch*clean_time];
        if start.is_none() {
            clean_valid[clean_time+7..].fill(false);
            reset[4]=true;reset[clean_time+3]=true;
            reset[clean_time+8]=true; // A reset in padding must not affect state.
        }
        let mut noisy_valid=vec![true;batch*noisy_time];
        noisy_valid[batch*noisy_time-1]=false;
        let output_time=if start.is_some(){noisy_time}else{clean_time};
        Self {batch,clean_time,noisy_time,start,clean,noisy,clean_valid,noisy_valid,reset,
            upstream:pattern(batch*output_time*d,71,0.11)}
    }
    fn masks(&self, device: &Device) -> (Tensor<2>,Tensor<2>,Tensor<2>) {
        let floats=|x:&[bool]|x.iter().map(|&x|if x {1.0}else{0.0}).collect::<Vec<_>>();
        (tensor(&floats(&self.clean_valid),[self.batch,self.clean_time],device),
         tensor(&floats(&self.reset),[self.batch,self.clean_time],device),
         tensor(&floats(&self.noisy_valid),[self.batch,self.noisy_time],device))
    }
}
#[derive(Clone)]
struct Expected {
    output: Vec<f64>, parameters: Vec<Vec<f64>>, clean: Vec<f64>, noisy: Vec<f64>,
}
fn direction(n: usize, field: usize) -> Vec<f64> {pattern(n,field*23+83,0.03)}
fn derivative(values: &mut [f64], direction: &[f64], mut objective: impl FnMut(&[f64])->f64) -> f64 {
    let original=values.to_vec();
    for ((x,&old),&delta) in values.iter_mut().zip(&original).zip(direction) {*x=old+DIFFERENCE*delta;}
    let plus=objective(values);
    for ((x,&old),&delta) in values.iter_mut().zip(&original).zip(direction) {*x=old-DIFFERENCE*delta;}
    let minus=objective(values);
    values.copy_from_slice(&original);
    (plus-minus)/(2.0*DIFFERENCE)
}
fn expected(oracle: &Oracle, s: &Scenario, directional: bool) -> Expected {
    let parameters=oracle.parameters.iter().enumerate().map(|(field,data)| {
        let coordinate=matches!(field,12|13|14) || !directional;
        let mut work=data.clone();
        let directions=if coordinate {(0..data.len()).map(|i|{let mut d=vec![0.0;data.len()];d[i]=1.0;d}).collect::<Vec<_>>()}
            else if data.is_empty() {Vec::new()} else {vec![direction(data.len(),field)]};
        directions.into_iter().map(|d|derivative(&mut work,&d,|x| {
            let mut changed=oracle.clone();changed.parameters[field]=x.to_vec();changed.objective(s)
        })).collect()
    }).collect();
    let input=|noisy:bool| {
        let data=if noisy {&s.noisy}else{&s.clean};
        let directions=if directional {vec![direction(data.len(),if noisy {29}else{28})]}
            else {(0..data.len()).map(|i|{let mut d=vec![0.0;data.len()];d[i]=1.0;d}).collect()};
        let mut work=data.clone();
        directions.into_iter().map(|d|derivative(&mut work,&d,|x| {
            let mut changed=s.clone();if noisy {changed.noisy=x.to_vec();} else {changed.clean=x.to_vec();}
            oracle.objective(&changed)
        })).collect()
    };
    Expected {output:oracle.output(s),parameters,clean:input(false),noisy:if s.start.is_some(){input(true)}else{Vec::new()}}
}
fn parameter_gradients(m: &Mixer, grads: &GradientsParams) -> Result<Vec<Vec<f64>>,String> {
    fn get<const D:usize>(p:&Param<Tensor<D>>,g:&GradientsParams)->Result<Vec<f64>,String> {
        g.get::<D>(p.id).map(values).ok_or_else(||format!("missing learned parameter gradient {:?}",p.id))
    }
    let linear=layers(m);
    let mut result=linear.iter().map(|l|get(&l.weight,grads)).collect::<Result<Vec<_>,_>>()?;
    result.extend(linear.iter().map(|l|l.bias.as_ref().map_or_else(||Ok(Vec::new()),|b|get(b,grads))).collect::<Result<Vec<_>,_>>()?);
    for p in [&m.q_conv,&m.k_conv,&m.v_conv] {result.push(get(p,grads)?);}
    for p in [&m.decay_scale,&m.decay_bias,&m.norm.gamma] {result.push(get(p,grads)?);}
    Ok(result)
}
struct Capture {starts:Vec<usize>,clean:Option<crate::sbd::CleanMixer>}
impl crate::profile::Sections for Capture {
    const CAPTURE_CLEAN_MIXERS:bool=true;
    fn clean_starts(&self)->Option<&[usize]> {Some(&self.starts)}
    fn clean_mixer(&mut self,clean:crate::sbd::CleanMixer){self.clean=Some(clean);}
    fn section<T>(&mut self,_name:&'static str,run:impl FnOnce()->T)->T {run()}
}
#[derive(Serialize)]
struct Case {
    path:&'static str,projection_backend:ProjectionBackend,start:Option<usize>,
    output_max_abs:f64,gradient_max_abs:f64,bias_gradient_coordinates:usize,
    all_other_gradient_coordinates:bool,
}
fn check(m:&Mixer,s:&Scenario,e:&Expected,path:&'static str,backend:ProjectionBackend,directional:bool,device:&Device)->Result<Case,String> {
    let mut m=m.clone();m.set_projection_backend(backend);
    let clean=tensor(&s.clean,[s.batch,s.clean_time,4],device).require_grad();
    let noisy=tensor(&s.noisy,[s.batch,s.noisy_time,4],device).require_grad();
    let (valid,reset,noisy_valid)=s.masks(device);
    let output=if let Some(start)=s.start {
        match path {
            "sbd_reconstructed"=> {
                let joined=Tensor::cat(vec![clean.clone().narrow(1,0,start),noisy.clone()],1);
                let active=Tensor::cat(vec![Tensor::ones([s.batch,start],device),noisy_valid],1);
                m.forward_sbd(joined,active,Tensor::zeros([s.batch,start+s.noisy_time],device),start).narrow(1,start,s.noisy_time)
            }
            "sbd_cached"=>m.forward_sbd_cached(noisy.clone(),&m.prepare_sbd(clean.clone(),&[start]),0,noisy_valid),
            "sbd_batched"=>m.forward_sbd_batched(noisy.clone(),&m.prepare_sbd_batched(clean.clone(),&[(0,start),(1,start)]),noisy_valid),
            "sbd_captured"|"sbd_batched_captured"=> {
                let mut trace=Capture {starts:vec![start],clean:None};
                let _=m.forward_profiled(clean.clone(),valid,reset,&mut trace);
                let captured=trace.clean.expect("clean tensors captured");
                if path=="sbd_captured" {
                    m.forward_sbd_cached(noisy.clone(),&m.prepare_sbd_from_clean(&captured,&[start]),0,noisy_valid)
                } else {
                    m.forward_sbd_batched(noisy.clone(),&m.prepare_sbd_batched_from_clean(&captured,&[(0,start),(1,start)]),noisy_valid)
                }
            }
            _=>return Err(format!("unknown gate check path {path}")),
        }
    } else {m.forward_masked(clean.clone(),valid,reset)};
    let output_max_abs=compare(&values(output.clone()),&e.output,"whole-layer output")?;
    let output_time=if s.start.is_some(){s.noisy_time}else{s.clean_time};
    let grads=(output*tensor(&s.upstream,[s.batch,output_time,4],device)).sum().backward();
    let project=|data:Vec<f64>,field:usize|if directional {
        vec![data.iter().zip(direction(data.len(),field)).map(|(x,d)|x*d).sum()]
    }else{data};
    let mut gradient_max_abs=compare(&project(values(clean.grad(&grads).ok_or("missing clean input gradient")?),28),&e.clean,"clean-input gradient")?;
    if s.start.is_some() {
        gradient_max_abs=gradient_max_abs.max(compare(&project(values(noisy.grad(&grads).ok_or("missing noisy input gradient")?),29),&e.noisy,"noisy-input gradient")?);
    }
    let gradients=parameter_gradients(&m,&GradientsParams::from_grads(grads,&m))?;
    let mut bias_gradient_coordinates=0;
    for (field,(actual,expected)) in gradients.into_iter().zip(&e.parameters).enumerate() {
        let actual=if directional && !matches!(field,12|13|14) && !actual.is_empty() {
            vec![actual.iter().zip(direction(actual.len(),field)).map(|(x,d)|x*d).sum()]
        }else{actual};
        gradient_max_abs=gradient_max_abs.max(compare(&actual,expected,&format!("parameter field {field}"))?);
        if matches!(field,12|13|14) {
            bias_gradient_coordinates+=actual.len();
            if !actual.iter().any(|x|x.abs()>1e-8) {return Err(format!("field {field}: nonzero bias fixture has no learning signal"));}
        }
    }
    Ok(Case {path,projection_backend:backend,start:s.start,output_max_abs,gradient_max_abs,bias_gradient_coordinates,
        all_other_gradient_coordinates:!directional})
}

/// Ordinary and cached/noisy whole layers, not merely gate projections.
pub fn numerical_checks(device:&Device)->Result<impl Serialize + use<>,String> {
    numerical_checks_selected(device,ScanBackend::TensorSolver)
}
/// Fused scans use their supported K32/V64 shape; every bias coordinate is still checked.
/// Other fused-shape parameter/input VJPs use deterministic directional derivatives.
pub fn numerical_checks_selected(device:&Device,backend:ScanBackend)->Result<impl Serialize + use<>,String> {
    #[derive(Serialize)]
    struct Report {passed:bool,paper:&'static str,scan_backend:ScanBackend,absolute_tolerance:f64,
        relative_tolerance:f64,finite_difference:f64,nonzero_biases:bool,padding:bool,short_tail:bool,
        chunk_boundary_starts:[usize;3],cases:Vec<Case>}
    let device=device.clone().autodiff();
    let mixer=fixture(&device,backend);
    let oracle=Oracle::from_mixer(&mixer);
    let directional=backend.is_fused();
    let mut cases=Vec::new();
    for start in [None,Some(3),Some(4),Some(5)] {
        let scenario=Scenario::new(start);
        let reference=expected(&oracle,&scenario,directional);
        let paths:&[&str]=if start.is_none(){&["ordinary"]}else{
            &["sbd_reconstructed","sbd_cached","sbd_batched","sbd_captured","sbd_batched_captured"]};
        for &path in paths {
            for projection in [ProjectionBackend::Separate,ProjectionBackend::Grouped,ProjectionBackend::Merged] {
                let case=check(&mixer,&scenario,&reference,path,projection,directional,&device)
                    .map_err(|e|format!("Kron gate biases {backend:?}/{projection:?}/{path}/start={start:?}: {e}"))?;
                cases.push(case);
            }
        }
    }
    Ok(Report {passed:true,paper:"arXiv:2610.26834v1 §2.2 Eqs.10–12 and output-gate text",scan_backend:backend,
        absolute_tolerance:ATOL,relative_tolerance:RTOL,finite_difference:DIFFERENCE,nonzero_biases:true,
        padding:true,short_tail:true,chunk_boundary_starts:[3,4,5],cases})
}

#[cfg(test)]
mod tests {
    #[test]
    fn kron_gate_biases_match_paper_and_release_initialization() {
        let device = burn::prelude::Device::ndarray().autodiff();
        let mixer = crate::MixerConfig::default().init(&device);
        for (layer, width, initial) in [
            (&mixer.erase, mixer.heads, 0.0),
            (&mixer.write, mixer.heads, 0.0),
            (&mixer.output_gate, mixer.heads * mixer.value_dim, 3.0),
        ] {
            let bias = layer.bias.as_ref().expect("paper gate bias");
            assert_eq!(bias.val().dims(), [width]);
            assert!(super::values(bias.val()).iter().all(|&value| value == initial));
        }
        assert!(super::values(mixer.output_gate.weight.val()).iter().all(|&x|x==0.0));
        for log_rate in super::values(mixer.decay_scale.val()) {
            assert!(log_rate.is_finite() && (0.0..16.0).contains(&log_rate.exp()));
        }
        for bias in super::values(mixer.decay_bias.val()) {
            let dt=bias.exp().ln_1p();
            assert!((0.000999..=0.100001).contains(&dt), "initial time constant {dt}");
        }
        for layer in [&mixer.q, &mixer.k, &mixer.v, &mixer.decay, &mixer.output] {
            assert!(layer.bias.is_none());
        }
    }
    #[test]
    fn residual_projection_initialization_uses_total_model_depth() {
        let device=burn::prelude::Device::ndarray().autodiff();
        device.seed(374);
        let shallow=crate::MixerConfig {num_layers:1,..Default::default()}.init(&device);
        let first=super::values(shallow.output.weight.val());
        device.seed(374);
        let deep=crate::MixerConfig {num_layers:4,..Default::default()}.init(&device);
        let second=super::values(deep.output.weight.val());
        for (&a,&b) in first.iter().zip(&second) {
            assert!((a*0.5-b).abs()<=1e-8,"residual depth scale: {a} vs {b}");
        }
    }
    #[test]
    fn kron_and_hola_keep_the_paper_relative_readout_scale() {
        use super::*;
        let device=Device::ndarray().autodiff();
        let mut mixer=fixture(&device,ScanBackend::TensorSolver);
        let cache=MixerConfig {d_model:4,heads:2,key_dim:3,value_dim:3,
            cache_policy:CachePolicy::Surprise,..Default::default()}.init(&device);
        mixer.cache=cache.cache;
        mixer.cache_policy=CachePolicy::Surprise;
        let oracle=Oracle::from_mixer(&mixer);
        let x=vec![0.3f32 as f64,-0.4f32 as f64,0.2f32 as f64,0.7f32 as f64];
        let raw:Vec<Vec<f64>>=(0..3).map(|field| {
            oracle.linear(field,&x).iter().enumerate().map(|(i,&p)| {
                let convolved=p*oracle.parameters[CONVOLUTIONS+field][i];
                convolved*sigmoid(convolved)
            }).collect()
        }).collect();
        let cache=mixer.cache.as_ref().unwrap();
        let q_gain=values(cache.q_norm.gamma.val());
        let k_gain=values(cache.k_norm.gamma.val());
        let sink=values(cache.sink.val());
        let cache_gate=values(cache.gate.val());
        let target=oracle.linear(5,&x);
        let gate=oracle.linear(6,&x);
        let mut mixed=vec![0.0;6];
        for head in 0..2 {
            let start=head*3;
            let q=&raw[0][start..start+3];
            let k=&raw[1][start..start+3];
            let v=&raw[2][start..start+3];
            let q_l2=q.iter().map(|x|x*x).sum::<f64>().sqrt().max(1e-12);
            let k_l2=k.iter().map(|x|x*x).sum::<f64>().sqrt().max(1e-12);
            let v_l2=v.iter().map(|x|x*x).sum::<f64>().sqrt().max(1e-12);
            let overlap=q.iter().zip(k).map(|(q,k)|q*k/(q_l2*k_l2)).sum::<f64>();
            let q_rms=(q.iter().map(|x|x*x).sum::<f64>()/3.0+cache.q_norm.epsilon).sqrt();
            let k_rms=(k.iter().map(|x|x*x).sum::<f64>()/3.0+cache.k_norm.epsilon).sqrt();
            let logit=(0..3).map(|j|q[j]*q_gain[j]/q_rms*k[j]*k_gain[j]/k_rms).sum::<f64>()/3.0f64.sqrt();
            let probability=1.0/(1.0+(sink[head]-logit).exp());
            // Empty initial state: the target writes, beta cannot change it.
            for j in 0..3 {
                let recurrent=sigmoid(target[head])*overlap*v[j]/v_l2/3.0f64.sqrt();
                let exact=probability*v[j]*sigmoid(cache_gate[head]);
                mixed[start+j]=(recurrent+exact)*sigmoid(gate[start+j]);
            }
            let rms=(mixed[start..start+3].iter().map(|x|x*x).sum::<f64>()/3.0+oracle.epsilon).sqrt();
            for j in 0..3 {mixed[start+j]=mixed[start+j]/rms*oracle.parameters[RMS_SCALE][j];}
        }
        let expected=oracle.linear(7,&mixed);
        for backend in [ProjectionBackend::Separate,ProjectionBackend::Grouped,ProjectionBackend::Merged] {
            mixer.set_projection_backend(backend);
            let actual=mixer.forward_masked(tensor(&x,[1,1,4],&device),
                Tensor::ones([1,1],&device),Tensor::zeros([1,1],&device));
            compare(&values(actual),&expected,"relative Kron/HOLA scale").unwrap();
        }
    }
    #[test]
    fn kron_gate_bias_whole_layer_f64_outputs_and_gradients() {
        super::numerical_checks(&burn::prelude::Device::ndarray()).unwrap();
    }
}
