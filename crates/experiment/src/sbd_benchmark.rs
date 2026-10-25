//! Paired execution-cost accounting; no evaluation, checkpoints or quality comparison.
use crate::{config::Config,data::{self,Batch,Corpus},model::{self,LanguageModel},runtime,train};
use anyhow::{Result,ensure};
use burn::{optim::ModuleOptimizer,prelude::*};
use delta_mixers::sbd::{InBlockReads,NoisyReadBackend};
use serde_json::{Value,json};
use std::time::Instant;
use crate::profile::median;

struct Runner { model:Option<LanguageModel>,optim:ModuleOptimizer,step:usize,peak:Option<u64> }
impl Runner {
    fn sample(&mut self,device:&Device) { if let Some(bytes)=runtime::sampled_reserved(device) {self.peak=Some(self.peak.unwrap_or(0).max(bytes));} }
    fn step(&mut self,c:&Config,corpus:Option<&Corpus>,device:&Device) -> Result<f64> {
        self.step_profiled(c,corpus,device,&mut delta_mixers::profile::Unprofiled)
    }
    fn step_profiled<P:delta_mixers::profile::Sections>(&mut self,c:&Config,corpus:Option<&Corpus>,device:&Device,profile:&mut P) -> Result<f64> {
        runtime::sync(device)?;
        let clock=Instant::now();
        let rows=data::examples(c,corpus,c.seed,self.step*c.batch_size,c.batch_size,"train")?;
        let batch=Batch::new(&rows,device);
        let model=self.model.take().unwrap();
        let loss=crate::sbd::training_loss_profiled(&model,c,&rows,&batch,c.seed.wrapping_add(self.step as u64),profile)?;
        self.sample(device);
        model::scalar(loss.clone())?;
        let gradients=model::checked_gradients(&model,loss)?;
        self.sample(device);
        self.model=Some(self.optim.step(c.learning_rate,model,gradients));
        runtime::sync(device)?;
        self.sample(device);
        self.step+=1;
        Ok(clock.elapsed().as_secs_f64()*1000.0)
    }
}
#[cfg(any(feature="vulkan",feature="cuda"))]
fn launches(run:impl FnOnce()->Result<()>) -> Result<Value> {
    use std::{collections::BTreeMap,sync::{Arc,Mutex}};
    #[derive(Default)] struct Count(Mutex<BTreeMap<&'static str,usize>>);
    impl cubecl::logging::LaunchObserver for Count {fn launched(&self,kernel:&'static str) {*self.0.lock().unwrap().entry(kernel).or_default()+=1;}}
    let count=Arc::new(Count::default());
    let guard=cubecl::logging::LaunchObservation::new(count.clone());
    run()?;drop(guard);
    let kernels=count.0.lock().unwrap();
    Ok(json!({"issued_per_step":kernels.values().sum::<usize>(),"by_kernel":*kernels}))
}
#[cfg(not(any(feature="vulkan",feature="cuda")))]
fn launches(run:impl FnOnce()->Result<()>) -> Result<Value> {run()?;Ok(Value::Null)}
fn paired(a:&[f64],b:&[f64]) -> Value {
    let differences=a.iter().zip(b).map(|(a,b)|a-b).collect::<Vec<_>>();
    let mut rng=data::Rng(709);
    let mut bootstrap=(0..10000).map(|_|median(&(0..differences.len()).map(|_|differences[rng.index(differences.len())]).collect::<Vec<_>>())).collect::<Vec<_>>();
    bootstrap.sort_by(f64::total_cmp);
    json!({"a_minus_b_samples_ms":differences,"a_minus_b_median_ms":median(&differences),"bootstrap_95_percent_ms":[bootstrap[249],bootstrap[9749]],"b_faster_in":differences.iter().filter(|&&x|x>0.0).count(),"pairs":differences.len()})
}
pub fn run(c:&Config,device:&Device,warmups:usize,trials:usize) -> Result<Value> {
    ensure!(warmups>=2 && trials>=11,"SBD timing requires >=2 warmups and >=11 paired trials");
    ensure!(c.sbd.is_some(),"SBD timing requires SBD enabled");
    ensure!(matches!(c.backend.as_str(),"vulkan"|"cuda") && !c.fusion,"SBD timing requires explicit non-fusion GPU execution");
    ensure!(c.arms.len()==1 && c.arms[0].cache!="none","SBD timing requires one HOLA arm");
    let target=device.clone().autodiff();
    target.seed(c.seed);
    let base=LanguageModel::new(c,&c.arms[0],&target)?;
    let corpus=train::loaded_corpus(c)?;
    let mut configs=[c.clone(),c.clone(),c.clone()];
    let modes=[(InBlockReads::None,NoisyReadBackend::Tensor),(InBlockReads::HolaRead,NoisyReadBackend::Tensor),(InBlockReads::HolaRead,NoisyReadBackend::FusedGpu)];
    let mut runners=Vec::new();
    for (i,(reads,backend)) in modes.into_iter().enumerate() {
        let s=configs[i].sbd.as_mut().unwrap();s.in_block_reads=reads;s.noisy_read_backend=backend;
        configs[i].validate()?;
        let mut model=base.clone();
        for block in &mut model.blocks {block.mixer=block.mixer.clone().with_sbd_reads(reads,backend);}
        runners.push(Runner{model:Some(model),optim:train::optimizer(&configs[i]),step:0,peak:None});
    }
    let started=Instant::now();
    let mut samples:[Vec<f64>;3]=std::array::from_fn(|_|Vec::new());
    let report_path=c.artifact_dir.join("sbd-benchmark.json");
    runtime::write_json(&report_path,&json!({"completed":false,"config":c,"warmups":warmups,"trials":trials,"finished_iterations":0,"samples_ms":samples}))?;
    for iteration in 0..warmups+trials {
        for mode in if iteration%2==0 {[0,1,2]}else{[2,1,0]} {
            ensure!(started.elapsed().as_secs_f64()<c.max_seconds,"SBD timing reached max_seconds; partial report retained");
            let ms=runners[mode].step(&configs[mode],corpus.as_ref(),&target)?;
            if iteration>=warmups {samples[mode].push(ms);}
            runtime::write_json(&report_path,&json!({"completed":false,"config":c,"warmups":warmups,"trials":trials,"finished_iterations":iteration,"last_finished_mode":mode,"last_step_ms":ms,"samples_ms":samples}))?;
        }
        runtime::write_json(&report_path,&json!({"completed":false,"config":c,"warmups":warmups,"trials":trials,"finished_iterations":iteration+1,"samples_ms":samples}))?;
    }
    let mut rows=Vec::new();
    for mode in 0..3 {
        let dispatches=launches(||{runners[mode].step(&configs[mode],corpus.as_ref(),&target)?;Ok(())})?;
        rows.push(json!({"selection":(["none_tensor","hola_read_tensor","hola_read_fused_gpu"][mode]),"median_ms":median(&samples[mode]),"samples_ms":samples[mode],"launches":dispatches,"sampled_peak_pool_reserved_bytes":runners[mode].peak}));
    }
    let layouts=paired(&samples[1],&samples[2]);
    let adopt=layouts["bootstrap_95_percent_ms"][0].as_f64().unwrap()>0.0;
    Ok(json!({"completed":true,"config":c,"device":runtime::identity(device),"precision":"float32","warmups":warmups,"trials":trials,"steps":rows,"in_block_cost_none_minus_hola":paired(&samples[0],&samples[1]),"execution_tensor_minus_fused":layouts,"fused_beyond_noise":adopt,"method":"same-session paired complete SBD steps; alternating forward/reverse mode order; identical initialization, examples and mask seeds at each step index; independent AdamW trajectories; host examples/upload, clean and detached teacher passes, masked loss, full backward, scalar readbacks, finite checks, clipping and AdamW included; no evaluation/checkpoints; cold JIT and two warmups excluded","memory_note":"pool reservation sampled after forward/backward/update; includes all three resident model/optimizer trajectories and allocator reuse, not exact live allocation or physical-device peak","launch_note":"separate warm complete step after timing; issued CubeCL kernels, not submissions or workgroups"}))
}

struct ReadTiming<'a> { device:&'a Device, milliseconds:f64, calls:usize, error:Option<String> }
impl delta_mixers::profile::Sections for ReadTiming<'_> {
    fn section<T>(&mut self,name:&'static str,run:impl FnOnce()->T) -> T {
        if name!="Noisy HOLA read" {return run();}
        if let Err(error)=runtime::sync(self.device) {self.error=Some(format!("{error:#}"));}
        let clock=Instant::now();
        let result=run();
        if let Err(error)=runtime::sync(self.device) {self.error=Some(format!("{error:#}"));}
        self.milliseconds+=clock.elapsed().as_secs_f64()*1000.0;
        self.calls+=1;
        result
    }
}

/// One warm step and one observed complete step per mode; no evaluation/checkpoint.
pub fn diagnostic(c:&Config,device:&Device) -> Result<Value> {
    ensure!(c.sbd.is_some() && c.arms.len()==1 && c.arms[0].cache!="none","SBD diagnostic requires one HOLA arm");
    ensure!(c.batch_size==1 && c.context==512,"SBD diagnostic requires B1/T512");
    let target=device.clone().autodiff();
    target.seed(c.seed);
    let base=LanguageModel::new(c,&c.arms[0],&target)?;
    let corpus=train::loaded_corpus(c)?;
    let path=c.artifact_dir.join("sbd-diagnostic.json");
    let mut rows=Vec::new();
    runtime::write_json(&path,&json!({"completed":false,"config":c,"steps":rows}))?;
    for (mode,(reads,backend)) in [(InBlockReads::None,NoisyReadBackend::Tensor),(InBlockReads::HolaRead,NoisyReadBackend::Tensor),(InBlockReads::HolaRead,NoisyReadBackend::FusedGpu)].into_iter().enumerate() {
        let mut config=c.clone();
        let s=config.sbd.as_mut().unwrap();s.in_block_reads=reads;s.noisy_read_backend=backend;
        config.validate()?;
        let mut model=base.clone();
        for block in &mut model.blocks {block.mixer=block.mixer.clone().with_sbd_reads(reads,backend);}
        let mut runner=Runner{model:Some(model),optim:train::optimizer(&config),step:0,peak:None};
        runtime::write_json(&path,&json!({"completed":false,"config":c,"next_mode":mode,"next_work":"warm complete step","steps":rows}))?;
        let warm_ms=runner.step(&config,corpus.as_ref(),&target)?;
        runtime::write_json(&path,&json!({"completed":false,"config":c,"next_mode":mode,"next_work":"observed complete step","warm_ms":warm_ms,"steps":rows}))?;
        let mut timing=ReadTiming{device:&target,milliseconds:0.0,calls:0,error:None};
        let mut milliseconds=0.0;
        let dispatches=launches(|| {milliseconds=runner.step_profiled(&config,corpus.as_ref(),&target,&mut timing)?;Ok(())})?;
        ensure!(timing.error.is_none(),"noisy-read diagnostic synchronization failed: {:?}",timing.error);
        ensure!(timing.calls%c.layers==0,"noisy HOLA call count must cover every layer");
        rows.push(json!({"selection":(["none_tensor","hola_read_tensor","hola_read_fused_gpu"][mode]),"warm_step_ms":warm_ms,"complete_step_ms":milliseconds,"noisy_block_evaluations":timing.calls/c.layers,"noisy_hola_layer_reads":timing.calls,"noisy_hola_forward_ms":timing.milliseconds,"noisy_hola_forward_percent_of_observed_step":100.0*timing.milliseconds/milliseconds,"launches":dispatches,"sampled_peak_pool_reserved_bytes":runner.peak}));
        runtime::write_json(&path,&json!({"completed":false,"config":c,"steps":rows}))?;
    }
    Ok(json!({"completed":true,"config":c,"device":runtime::identity(device),"steps":rows,"warmups_per_mode":1,"measured_steps_per_mode":1,"precision":"float32","method":"complete SBD loss/backward/checks/clipping/AdamW after one warm step per mode; synchronization only before/after each noisy HOLA forward section to expose its time; section includes query/key norms, cache gathers, segment assembly, one softmax/value read and existing sigmoid gate; clean preparation, other noisy work and backward remain outside that section","share_limitation":"forward-section time divided by the observed complete-step time, not full forward/backward HOLA cost or additive GPU utilization; inserted synchronization changes overlap, so this diagnostic is not the ordinary paired-step adoption measurement","count_method":"all layers have HOLA; noisy block evaluation count equals noisy HOLA section calls divided by layers, including detached teacher and differentiable denoising passes; all-observed and length-one blocks do not call the denoiser","memory_note":"pool reservation sampled after forward/backward/update; allocator reuse and resident model/optimizer values are included; not exact live bytes or physical-device peak"}))
}
