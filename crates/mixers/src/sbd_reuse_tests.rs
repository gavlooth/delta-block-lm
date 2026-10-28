use super::*;
use crate::{MixerConfig, ProjectionBackend, profile::{Sections, Unprofiled}};
use burn::{module::{ModuleVisitor, Param}, optim::GradientsParams};

struct Capture { starts: Vec<usize>, clean: Option<CleanMixer> }
impl Sections for Capture {
    const CAPTURE_CLEAN_MIXERS:bool=true;
    fn clean_starts(&self)->Option<&[usize]> {Some(&self.starts)}
    fn clean_mixer(&mut self,clean:CleanMixer) {self.clean=Some(clean);}
    fn section<T>(&mut self,_name:&'static str,run:impl FnOnce()->T)->T {run()}
}
fn values<const D:usize>(x:Tensor<D>)->Vec<f32> {x.into_data().try_to_vec::<f32>().unwrap()}
fn compare<const D:usize>(a:Tensor<D>,b:Tensor<D>) {
    let a=values(a);let b=values(b);
    assert_eq!(a.len(),b.len());
    for (a,b) in a.into_iter().zip(b) {
        assert!(a.is_finite() && b.is_finite() && (f64::from(a)-f64::from(b)).abs()<=2e-5+3e-4*f64::from(b).abs(),"{a} != {b}");
    }
}
fn capture(mixer:&Mixer,input:Tensor<3>,starts:&[usize])->CleanMixer {
    let [b,t,_]=input.dims();let device=input.device();
    let mut trace=Capture {starts:starts.to_vec(),clean:None};
    let _=mixer.forward_profiled(input,Tensor::ones([b,t],&device),Tensor::zeros([b,t],&device),&mut trace);
    trace.clean.unwrap()
}

#[test]
fn prepared_cache_retains_the_clean_projection_convolution_and_scan() {
    let device=Device::ndarray().autodiff();device.seed(8101);
    let mut mixer=MixerConfig {d_model:8,heads:2,key_dim:3,value_dim:5,chunk_size:4,..Default::default()}.init(&device);
    let input=Tensor::<3>::random([1,9,8],burn::tensor::Distribution::Uniform(-0.5,0.5),&device).require_grad();
    let clean=capture(&mixer,input.clone(),&[1,5,8]);
    let expected=mixer.prepare_sbd(input.clone(),&[1,5,8]);
    // A captured clean pass is immutable, even if the module is subsequently updated.
    // Recomputing any of its input projections, convolutions or recurrence violates this.
    mixer.q.weight=mixer.q.weight.clone().map(|x|x+0.3);
    mixer.k.weight=mixer.k.weight.clone().map(|x|x-0.2);
    mixer.v.weight=mixer.v.weight.clone().map(|x|x+0.4);
    mixer.k_conv=mixer.k_conv.clone().map(|x|x+0.1);
    mixer.v_conv=mixer.v_conv.clone().map(|x|x-0.1);
    mixer.decay.weight=mixer.decay.weight.clone().map(|x|x+0.5);
    let actual=mixer.prepare_sbd_from_clean(&clean,&[1,5,8]);
    for (a,b) in actual.raw.iter().zip(&expected.raw) {compare(a.clone(),b.clone());}
    compare(actual.cache_k.clone(),expected.cache_k.clone());
    compare(actual.cache_v.clone(),expected.cache_v.clone());
    for (a,b) in actual.states.iter().zip(&expected.states) {compare(a.clone(),b.clone());}
    let objective=|cache:MixerCache|cache.cache_k.square().mean()+cache.cache_v.square().mean()+Tensor::cat(cache.states,0).square().mean();
    let actual=objective(actual).backward();
    let expected=objective(expected).backward();
    compare(input.grad(&actual).unwrap(),input.grad(&expected).unwrap());
}

#[test]
fn merged_noisy_projections_preserve_outputs_and_all_parameter_vjps() {
    let kind = crate::MixerKind::KronLstm;
        let device=Device::ndarray().autodiff();device.seed(8317);
        let separate=MixerConfig {kind,d_model:8,heads:2,key_dim:3,value_dim:5,chunk_size:4,..Default::default()}.init(&device);
        let mut merged=separate.clone();merged.set_projection_backend(ProjectionBackend::Merged);
        let clean_input=Tensor::<3>::random([1,9,8],burn::tensor::Distribution::Uniform(-0.5,0.5),&device).require_grad();
        let noisy=Tensor::<3>::random([1,3,8],burn::tensor::Distribution::Uniform(-0.5,0.5),&device).require_grad();
        for batched in [false,true] {
            let separate_clean=capture(&separate,clean_input.clone(),&[5]);
            let merged_clean=capture(&merged,clean_input.clone(),&[5]);
            let output=|mixer:&Mixer,clean:&CleanMixer| {
                if batched {mixer.forward_sbd_batched_profiled(noisy.clone(),&mixer.prepare_sbd_batched_from_clean(clean,&[(0,5)]),Tensor::ones([1,3],&device),&mut Unprofiled)}
                else {mixer.forward_sbd_cached(noisy.clone(),&mixer.prepare_sbd_from_clean(clean,&[5]),0,Tensor::ones([1,3],&device))}
            };
            let a=output(&merged,&merged_clean);let b=output(&separate,&separate_clean);
            compare(a.clone(),b.clone());
            let a=a.square().mean().backward();let b=b.square().mean().backward();
            compare(noisy.grad(&a).unwrap(),noisy.grad(&b).unwrap());
            compare(clean_input.grad(&a).unwrap(),clean_input.grad(&b).unwrap());
            let a=GradientsParams::from_grads(a,&merged);let b=GradientsParams::from_grads(b,&separate);
            struct Compare<'a>(&'a GradientsParams,&'a GradientsParams,usize);
            impl ModuleVisitor for Compare<'_> {
                fn visit_float<const D:usize>(&mut self,p:&Param<Tensor<D>>) {
                    match (self.0.get::<D>(p.id),self.1.get::<D>(p.id)) {
                        (Some(a),Some(b))=>{compare(a,b);self.2+=1;},
                        (None,None)=>{},
                        _=>panic!("gradient presence differs"),
                    }
                }
            }
            let mut check=Compare(&a,&b,0);merged.visit(&mut check);assert!(check.2>=10);
        }
}
