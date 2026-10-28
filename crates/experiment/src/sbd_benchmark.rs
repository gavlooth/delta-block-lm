//! Paired execution-cost accounting; no evaluation, checkpoints or quality comparison.
use crate::{config::Config,data::{self,Batch,Corpus},model::{self,LanguageModel},runtime,train};
use anyhow::{Result,ensure};
use burn::prelude::*;
use delta_mixers::sbd::{InBlockReads,NoisyReadBackend};
use serde_json::{Value,json};
use std::time::Instant;
use crate::profile::median;

struct Runner { model:Option<LanguageModel>,optim:train::NeuralOptimizer,step:usize,peak:Option<u64> }
impl Runner {
    fn sample(&mut self,device:&Device) { if let Some(bytes)=runtime::sampled_reserved(device) {self.peak=Some(self.peak.unwrap_or(0).max(bytes));} }
    fn step(&mut self,c:&Config,corpus:Option<&Corpus>,device:&Device) -> Result<f64> {
        self.step_profiled(c,corpus,device,&mut delta_mixers::profile::Unprofiled)
    }
    fn step_profiled<P:delta_mixers::profile::Sections>(&mut self,c:&Config,corpus:Option<&Corpus>,device:&Device,profile:&mut P) -> Result<f64> {
        let fixture=self.step;
        self.step_fixture_profiled(c,corpus,device,fixture,profile)
    }
    fn step_fixed(&mut self,c:&Config,corpus:Option<&Corpus>,device:&Device)->Result<f64>{
        self.step_fixture_profiled(c,corpus,device,0,&mut delta_mixers::profile::Unprofiled)
    }
    fn step_fixture_profiled<P:delta_mixers::profile::Sections>(&mut self,c:&Config,corpus:Option<&Corpus>,device:&Device,fixture:usize,profile:&mut P)->Result<f64>{
        runtime::sync(device)?;
        let clock=Instant::now();
        let rows=data::examples(c,corpus,c.seed,fixture*c.batch_size,c.batch_size,"train")?;
        let batch=Batch::new(&rows,device);
        let model=self.model.take().unwrap();
        let result=crate::sbd_step::backward_profiled(&model,c,&rows,&batch,c.seed.wrapping_add(fixture as u64),profile)?;
        self.sample(device);
        let gradients=result.gradients;
        let circuit_flows=result.circuit_flows;
        self.sample(device);
        self.model=Some(train::optimizer_step(c,model,&mut self.optim,gradients,circuit_flows)?);
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
        let rows_per_pass=config.sbd.as_ref().and_then(|s|s.rows_per_pass).unwrap_or(config.batch_size).min(config.batch_size);
        let backward_passes=config.batch_size.div_ceil(rows_per_pass);
        rows.push(json!({"selection":(["none_tensor","hola_read_tensor","hola_read_fused_gpu"][mode]),"warm_step_ms":warm_ms,"complete_step_ms":milliseconds,"backward_passes":backward_passes,"rows_per_pass":rows_per_pass,"noisy_block_evaluations":timing.calls/c.layers,"noisy_hola_layer_reads":timing.calls,"noisy_hola_forward_ms":timing.milliseconds,"noisy_hola_forward_percent_of_observed_step":100.0*timing.milliseconds/milliseconds,"launches":dispatches,"sampled_peak_pool_reserved_bytes":runner.peak}));
        runtime::write_json(&path,&json!({"completed":false,"config":c,"steps":rows}))?;
    }
    Ok(json!({"completed":true,"config":c,"device":runtime::identity(device),"steps":rows,"warmups_per_mode":1,"measured_steps_per_mode":1,"precision":"float32","method":"complete SBD loss/backward/checks/clipping/AdamW after one warm step per mode; synchronization only before/after each noisy HOLA forward section to expose its time; section includes query/key norms, cache gathers, segment assembly, one softmax/value read and existing sigmoid gate; clean preparation, other noisy work and backward remain outside that section","share_limitation":"forward-section time divided by the observed complete-step time, not full forward/backward HOLA cost or additive GPU utilization; inserted synchronization changes overlap, so this diagnostic is not the ordinary paired-step adoption measurement","count_method":"all layers have HOLA; noisy block evaluation count equals noisy HOLA section calls divided by layers, including detached teacher and differentiable denoising passes; all-observed and length-one blocks do not call the denoiser","memory_note":"pool reservation sampled after forward/backward/update; allocator reuse and resident model/optimizer values are included; not exact live bytes or physical-device peak"}))
}

/// Compare only block execution; model, masks and optimizer work stay identical.
pub fn batching(c:&Config,device:&Device,warmups:usize,trials:usize) -> Result<Value> {
    use crate::config::SbdBlockExecution;
    ensure!(warmups>=2 && trials>=11,"batch timing requires >=2 warmups and >=11 paired trials");
    ensure!(c.sbd.is_some() && c.arms.len()==1,"batch timing requires SBD and one arm");
    ensure!(matches!(c.backend.as_str(),"vulkan"|"cuda") && !c.fusion,"batch timing requires explicit non-fusion GPU execution");
    let target=device.clone().autodiff();
    target.seed(c.seed);
    let base=LanguageModel::new(c,&c.arms[0],&target)?;
    let corpus=train::loaded_corpus(c)?;
    let mut configs=[c.clone(),c.clone()];
    configs[0].sbd.as_mut().unwrap().block_execution=SbdBlockExecution::PerBlock;
    configs[1].sbd.as_mut().unwrap().block_execution=SbdBlockExecution::Batched;
    let mut runners=configs.iter().map(|config|Runner{model:Some(base.clone()),optim:train::optimizer(config),step:0,peak:None}).collect::<Vec<_>>();
    let mut samples:[Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
    let started=Instant::now();
    let path=c.artifact_dir.join("sbd-batching-benchmark.json");
    for iteration in 0..warmups+trials {
        for mode in if iteration%2==0 {[0,1]}else{[1,0]} {
            runtime::write_json(&path,&json!({"completed":false,"config":c,"warmups":warmups,"trials":trials,"iteration":iteration,"next_mode":mode,"samples_ms":samples}))?;
            ensure!(started.elapsed().as_secs_f64()<c.max_seconds,"batch timing reached max_seconds; partial report retained");
            let ms=runners[mode].step(&configs[mode],corpus.as_ref(),&target)?;
            if iteration>=warmups {samples[mode].push(ms);}
        }
    }
    let mut rows=Vec::new();
    for mode in 0..2 {
        let dispatches=launches(||{runners[mode].step(&configs[mode],corpus.as_ref(),&target)?;Ok(())})?;
        rows.push(json!({"selection":(["per_block","batched"][mode]),"median_ms":median(&samples[mode]),"samples_ms":samples[mode],"launches":dispatches,"sampled_peak_pool_reserved_bytes":runners[mode].peak}));
    }
    let difference=paired(&samples[0],&samples[1]);
    Ok(json!({"completed":true,"config":c,"device":runtime::identity(device),"precision":"float32","warmups":warmups,"trials":trials,"steps":rows,"per_block_minus_batched":difference,"batched_beyond_noise":difference["bootstrap_95_percent_ms"][0].as_f64().unwrap()>0.0,"method":"same-session paired complete SBD training steps, alternating execution order, identical initialization and step-specific examples/mask seeds, independent AdamW trajectories; includes uploads, both noisy passes, backward, checks, clipping and update; excludes cold JIT and warmups","memory_note":"pool reservation sampled after forward/backward/update; includes both live runners, not peak live activation bytes"}))
}

/// One batched-only diagnostic at the configured batch and context.
pub fn batching_diagnostic(c:&Config,device:&Device) -> Result<Value> {
    use crate::config::SbdBlockExecution;
    ensure!(c.sbd.is_some() && c.arms.len()==1,"batch diagnostic requires SBD and one arm");
    ensure!(c.sbd.as_ref().unwrap().block_execution==SbdBlockExecution::Batched,"batch diagnostic requires batched execution");
    let target=device.clone().autodiff();
    target.seed(c.seed);
    let base=LanguageModel::new(c,&c.arms[0],&target)?;
    let corpus=train::loaded_corpus(c)?;
    let mut runner=Runner{model:Some(base),optim:train::optimizer(c),step:0,peak:None};
    let fixed=c.sbd.as_ref().unwrap().cocommit.is_some();
    let warmups=if fixed {2}else{1};
    let mut warm=Vec::with_capacity(warmups);
    for _ in 0..warmups {
        warm.push(if fixed {runner.step_fixed(c,corpus.as_ref(),&target)?}else{runner.step(c,corpus.as_ref(),&target)?});
    }
    let mut milliseconds=0.0;
    let dispatches=launches(||{milliseconds=if fixed {runner.step_fixed(c,corpus.as_ref(),&target)?}else{runner.step(c,corpus.as_ref(),&target)?};Ok(())})?;
    let rows_per_pass=c.sbd.as_ref().and_then(|s|s.rows_per_pass).unwrap_or(c.batch_size).min(c.batch_size);
    let passes=c.batch_size.div_ceil(rows_per_pass);
    let mut report=json!({"completed":true,"config":c,"device":runtime::identity(device),"selection":"batched","warm_step_ms":warm[warmups-1],"complete_step_ms":milliseconds,"launches":dispatches,"sampled_peak_pool_reserved_bytes":runner.peak,"warmups":warmups,"measured_steps":1,"backward_passes":passes,"rows_per_pass":rows_per_pass,"precision":"float32","method":"one warm and one complete batched SBD training step, including uploads, detached teacher, second noisy pass, backward, checks, clipping and AdamW","memory_note":"pool reservation sampled after forward/backward/update, not exact peak live allocation bytes"});
    if fixed {
        report["method"]="two warmups on one fixed fixture, then one observed complete CoCommit step including both tails, queries, flows, backward, checks, clipping, AdamW and circuit update".into();
        report["warm_step_samples_ms"]=json!(warm);
        report["fixture"]=json!({"example_index":0,"mask_seed":c.seed});
    }
    Ok(report)
}

fn codpula_isolated(c:&Config,device:&Device,warmups:usize,trials:usize) -> Result<Value> {
    use crate::codpula::{Codpula,CodpulaExecution};
    let length=8;
    let chains=c.batch_size*c.context/length;
    let logits=Tensor::<3>::from_data(TensorData::new((0..chains*length*257)
        .map(|i|((i*37%101) as f32-50.0)*0.013).collect::<Vec<_>>(),[chains,length,257]),device).require_grad();
    let features=Tensor::<3>::from_data(TensorData::new((0..chains*length*c.d_model)
        .map(|i|((i*17%71) as f32-35.0)*0.011).collect::<Vec<_>>(),[chains,length,c.d_model]),device).require_grad();
    let targets=Tensor::<2,Int>::from_data(TensorData::new((0..chains*length)
        .map(|i|((i*13+7)%256) as i32).collect::<Vec<_>>(),[chains,length]),device);
    let observed=Tensor::<2,Bool>::from_data(TensorData::new((0..chains*length)
        .map(|i|i%3==0).collect::<Vec<_>>(),[chains,length]),device);
    let lengths=vec![length;chains];
    let base=Codpula::new(c.d_model,c.seed,device);
    let mut circuits=[base.clone(),base.with_execution(CodpulaExecution::FusedGpu)];
    let mut samples:[Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
    let run=|circuit:&mut Codpula|->Result<f64> {
        runtime::sync(device)?;
        let clock=Instant::now();
        let prepared=circuit.prepare(logits.clone(),features.clone(),&lengths,observed.clone(),1.0)?;
        let result=prepared.conditional_nll(targets.clone(),observed.clone(),(chains*length) as f64)?;
        model::scalar(result.loss.clone())?;
        std::hint::black_box(result.loss.backward());
        circuit.update(result.flows)?;
        runtime::sync(device)?;
        Ok(clock.elapsed().as_secs_f64()*1000.0)
    };
    for iteration in 0..warmups+trials {
        for mode in if iteration%2==0 {[0,1]}else{[1,0]} {
            let milliseconds=run(&mut circuits[mode])?;
            if iteration>=warmups {samples[mode].push(milliseconds);}
        }
    }
    let mut rows=Vec::with_capacity(2);
    for mode in 0..2 {
        let mut milliseconds=0.0;
        let dispatches=launches(||{milliseconds=run(&mut circuits[mode])?;Ok(())})?;
        rows.push(json!({"execution":(["tensor","fused_gpu"][mode]),"median_ms":median(&samples[mode]),
            "samples_ms":samples[mode],"instrumented_ms":milliseconds,"launches":dispatches}));
    }
    Ok(json!({"shape":{"chains":chains,"length":length,"states":32,"vocab":257,"d_model":c.d_model},
        "warmups":warmups,"trials":trials,"results":rows,"tensor_minus_fused":paired(&samples[0],&samples[1]),
        "scope":"resident synthetic circuit features/potentials, gate projection, local normalizers, both evidence passes, neural backward, nonnegative flows and circuit update; no backbone, token projection, uploads, neural optimizer or generation; fixed fixtures and independent identical-start circuit trajectories",
        "logical_state_bytes":(32*257+32*32+32)*4*3,
        "memory_note":"logical log rows plus two flow averages; no exact live-allocation measurement"}))
}

/// Same-model tensor/fused comparison, with the factorized cost reported separately.
pub fn codpula(c:&Config,device:&Device,warmups:usize,trials:usize) -> Result<Value> {
    use crate::{codpula::CodpulaExecution,config::{SbdCommitment,SbdJointOutput}};
    if c.sbd.as_ref().is_some_and(|s|s.joint_output==SbdJointOutput::CodpulaContextual) {
        return codpula_contextual(c,device,warmups,trials);
    }
    ensure!(warmups>=2 && trials>=11,"Codpula timing requires >=2 warmups and >=11 paired trials");
    ensure!(c.batch_size==1 && c.context==512,"Codpula paired complete-step timing requires B1/T512");
    ensure!(c.arms.len()==1 && c.sbd.as_ref().is_some_and(|s|s.joint_output==SbdJointOutput::Codpula),
        "Codpula timing requires one arm and Codpula output");
    ensure!(matches!(c.backend.as_str(),"vulkan"|"cuda") && !c.fusion,"Codpula timing requires explicit non-fusion GPU execution");
    let target=device.clone().autodiff();
    target.seed(c.seed);
    let base=LanguageModel::new(c,&c.arms[0],&target)?;
    let corpus=train::loaded_corpus(c)?;
    let isolated=codpula_isolated(c,&target,warmups,trials)?;
    runtime::write_json(&c.artifact_dir.join("codpula-isolated.json"),&isolated)?;
    let mut configs=[c.clone(),c.clone(),c.clone()];
    {
        let s=configs[0].sbd.as_mut().unwrap();
        s.joint_output=SbdJointOutput::Factorized;
        s.commitment=Some(SbdCommitment::Fixed);
        s.joint_execution=CodpulaExecution::Tensor;
    }
    configs[1].sbd.as_mut().unwrap().joint_execution=CodpulaExecution::Tensor;
    configs[2].sbd.as_mut().unwrap().joint_execution=CodpulaExecution::FusedGpu;
    let mut runners=Vec::with_capacity(3);
    for (mode,config) in configs.iter().enumerate() {
        config.validate()?;
        let mut model=base.clone();
        if mode==0 {model.codpula=None;}
        else {
            model.codpula=model.codpula.take().map(|circuit|
                circuit.with_execution(if mode==1 {CodpulaExecution::Tensor}else{CodpulaExecution::FusedGpu}));
        }
        runners.push(Runner{model:Some(model),optim:train::optimizer(config),step:0,peak:None});
    }
    let path=c.artifact_dir.join("codpula-benchmark.json");
    let started=Instant::now();
    let mut samples:[Vec<f64>;3]=std::array::from_fn(|_|Vec::new());
    for iteration in 0..warmups+trials {
        for mode in if iteration%2==0 {[0,1,2]}else{[2,1,0]} {
            runtime::write_json(&path,&json!({"completed":false,"config":c,"warmups":warmups,"trials":trials,
                "iteration":iteration,"next_mode":mode,"samples_ms":samples}))?;
            ensure!(started.elapsed().as_secs_f64()<c.max_seconds,"Codpula timing reached max_seconds; partial report retained");
            let milliseconds=runners[mode].step(&configs[mode],corpus.as_ref(),&target)?;
            if iteration>=warmups {samples[mode].push(milliseconds);}
        }
    }
    let mut rows=Vec::with_capacity(3);
    for mode in 0..3 {
        let dispatches=launches(||{runners[mode].step(&configs[mode],corpus.as_ref(),&target)?;Ok(())})?;
        rows.push(json!({"selection":(["factorized","codpula_tensor","codpula_fused_gpu"][mode]),
            "median_ms":median(&samples[mode]),"samples_ms":samples[mode],"launches":dispatches,
            "sampled_peak_pool_reserved_bytes":runners[mode].peak}));
    }
    let execution=paired(&samples[1],&samples[2]);
    let report=json!({"completed":true,"config":c,"device":runtime::identity(device),"precision":"float32",
        "warmups":warmups,"trials":trials,"isolated_circuit":isolated,"steps":rows,"execution_tensor_minus_fused":execution,
        "fused_beyond_noise":execution["bootstrap_95_percent_ms"][0].as_f64().unwrap()>0.0,
        "factorized_minus_tensor":paired(&samples[0],&samples[1]),
        "factorized_minus_fused":paired(&samples[0],&samples[2]),
        "method":"same-session interleaved complete steps, alternating mode order, common initial backbone and identical step-specific examples/mask draws; independent neural optimizer and circuit trajectories; includes uploads, clean and detached teacher passes, joint loss, neural backward/checks/clipping/AdamW and dual-flow circuit update; excludes cold compilation and warmups; factorized is cost accounting, not a quality comparison",
        "memory_note":"pool reservation sampled after forward/backward/update, including all three live runners; not exact peak live allocation"});
    runtime::write_json(&path,&report)?;
    Ok(report)
}

/// Cost accounting for the approved contextual profile, not architecture selection.
fn codpula_contextual(c:&Config,device:&Device,warmups:usize,trials:usize) -> Result<Value> {
    use crate::{codpula::{Codpula,CodpulaExecution},config::SbdJointOutput};
    ensure!(warmups==2 && trials==11,"contextual cost accounting requires two warmups and eleven pairs");
    ensure!(c.batch_size==1 && c.context==512 && c.d_model==256,"contextual paired timing requires D256/B1/T512");
    ensure!(c.arms.len()==1 && c.sbd.as_ref().is_some_and(|s|s.block_lengths==[8]),
        "contextual timing requires one arm and L8");
    ensure!(matches!(c.backend.as_str(),"vulkan"|"cuda") && !c.fusion,
        "contextual timing requires explicit non-fusion GPU execution");
    let target=device.clone().autodiff();
    target.seed(c.seed);
    let base=LanguageModel::new(c,&c.arms[0],&target)?;
    let corpus=train::loaded_corpus(c)?;
    let mut configs=[c.clone(),c.clone()];
    configs[0].sbd.as_mut().unwrap().joint_output=SbdJointOutput::Codpula;
    let mut runners=Vec::with_capacity(2);
    for (mode,config) in configs.iter_mut().enumerate() {
        config.sbd.as_mut().unwrap().joint_execution=CodpulaExecution::Tensor;
        config.validate()?;
        let mut model=base.clone();
        if mode==0 {model.codpula=Some(Codpula::new(c.d_model,c.seed,&target));}
        runners.push(Runner{model:Some(model),optim:train::optimizer(config),step:0,peak:None});
    }
    let path=c.artifact_dir.join("codpula-benchmark.json");
    let started=Instant::now();
    let mut samples:[Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
    for iteration in 0..warmups+trials {
        for mode in if iteration%2==0 {[0,1]}else{[1,0]} {
            runtime::write_json(&path,&json!({"completed":false,"config":c,"warmups":warmups,"trials":trials,
                "iteration":iteration,"next_mode":mode,"samples_ms":samples}))?;
            ensure!(started.elapsed().as_secs_f64()<c.max_seconds,
                "contextual timing reached max_seconds; partial report retained");
            let milliseconds=runners[mode].step(&configs[mode],corpus.as_ref(),&target)?;
            if iteration>=warmups {samples[mode].push(milliseconds);}
        }
    }
    let mut rows=Vec::with_capacity(2);
    for mode in 0..2 {
        let mut instrumented_ms=0.0;
        let dispatches=launches(|| {
            instrumented_ms=runners[mode].step(&configs[mode],corpus.as_ref(),&target)?;
            Ok(())
        })?;
        rows.push(json!({"selection":(["codpula","codpula_contextual"][mode]),
            "median_ms":median(&samples[mode]),"samples_ms":samples[mode],"launches":dispatches,
            "instrumented_step_ms":instrumented_ms,"sampled_peak_pool_reserved_bytes":runners[mode].peak}));
    }
    let report=json!({"completed":true,"config":c,"device":runtime::identity(device),"precision":"float32",
        "warmups":warmups,"trials":trials,"steps":rows,
        "contextual_minus_codpula":paired(&samples[1],&samples[0]),
        "method":"same-session paired full SBD optimizer steps; alternating order; common backbone, clean vocabulary head and example/mask seeds; independent neural/circuit optimizer trajectories; includes uploads, clean pass, detached teacher, student joint loss, backward, finite checks, clipping, AdamW and dual-flow update; excludes build/startup and two designated warmups; later shape-specific JIT is not separately removed; one extra instrumented step per profile reports dispatches; costs only, not a quality comparison or architecture-selection experiment",
        "memory_note":"sampled pool reservation includes both resident models/optimizers; not per-profile peak live allocations",
        "logical_extra_neural_parameters":57376,
        "logical_extra_persistent_parameter_and_adam_bytes_f32":57376*4*3,
        "logical_extra_forward_mac_per_noisy_pass":(c.batch_size*c.context/8)*c.d_model*(3*96-64)});
    runtime::write_json(&path,&report)?;
    Ok(report)
}

/// Costs only: disabled coordination and two equivalent marked-tail layouts.
pub fn cocommit(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    use crate::config::CocommitExecution;
    ensure!(warmups>=2 && trials>=11,"CoCommit timing requires >=2 warmups and >=11 interleaved pairs");
    ensure!(c.batch_size==1 && c.context==512,"CoCommit paired timing requires B1/T512");
    ensure!(c.arms.len()==1 && c.sbd.as_ref().is_some_and(|s|s.cocommit.is_some()),"CoCommit timing requires one configured arm");
    let first_tail=c.layers-c.sbd.as_ref().unwrap().cocommit.as_ref().unwrap().tail_layers;
    ensure!(c.prefix_block_retrieval.as_ref().is_none_or(|retrieval|
        retrieval.layers.iter().all(|&layer|layer<first_tail)),
        "CoCommit layout benchmark explicitly compares Doubled and requires no retrieval in the marked tail; use Separate with the complete-step diagnostic for tail retrieval");
    let target=device.clone().autodiff();
    let corpus=train::loaded_corpus(c)?;
    let mut configs=[c.clone(),c.clone(),c.clone()];
    configs[0].sbd.as_mut().unwrap().cocommit=None;
    configs[1].sbd.as_mut().unwrap().cocommit.as_mut().unwrap().execution=CocommitExecution::Separate;
    configs[2].sbd.as_mut().unwrap().cocommit.as_mut().unwrap().execution=CocommitExecution::Doubled;
    target.seed(c.seed);
    let base=LanguageModel::new(&configs[2],&c.arms[0],&target)?;
    base.materialize_parameters();
    let mut runners=Vec::with_capacity(3);
    for config in &configs {
        config.validate()?;
        let mut model=base.clone().fork(&target);
        model.cocommit_config=config.sbd.as_ref().unwrap().cocommit.clone();
        if model.cocommit_config.is_none(){model.cocommit_marker=None;}
        runners.push(Runner {model:Some(model),optim:train::optimizer(config),step:0,peak:None});
    }
    drop(base);
    let path=c.artifact_dir.join("cocommit-benchmark.json");
    let started=Instant::now();
    let mut samples:[Vec<f64>;3]=std::array::from_fn(|_|Vec::new());
    for iteration in 0..warmups+trials {
        for mode in if iteration%2==0 {[0,1,2]} else {[2,1,0]} {
            runtime::write_json(&path,&json!({"completed":false,"config":c,"warmups":warmups,"trials":trials,"iteration":iteration,"next_mode":mode,"samples_ms":samples}))?;
            ensure!(started.elapsed().as_secs_f64()<c.max_seconds,"CoCommit timing reached max_seconds; partial report retained");
            let milliseconds=runners[mode].step_fixed(&configs[mode],corpus.as_ref(),&target)?;
            if iteration>=warmups {samples[mode].push(milliseconds);}
        }
    }
    let mut steps=Vec::with_capacity(3);
    for mode in 0..3 {
        let dispatches=launches(|| {runners[mode].step_fixed(&configs[mode],corpus.as_ref(),&target)?;Ok(())})?;
        steps.push(json!({"selection":(["off","separate","doubled"][mode]),"median_ms":median(&samples[mode]),"samples_ms":samples[mode],
            "launches":dispatches,"sampled_peak_pool_reserved_bytes":runners[mode].peak}));
    }
    let layouts=paired(&samples[1],&samples[2]);
    let doubled=layouts["bootstrap_95_percent_ms"][0].as_f64().unwrap()>0.0;
    let selected=if doubled {2}else{1};
    let isolated=cocommit_tail_isolated(c,device,warmups,trials)?;
    let mut membership=delta_mixers::checks::membership_schedule_benchmark(device,warmups,trials).map_err(anyhow::Error::msg)?;
    let legacy=membership["legacy_full_read_ms"].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect::<Vec<_>>();
    let position_only=membership["position_only_ms"].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect::<Vec<_>>();
    membership["paired_saving_ms"]=paired(&legacy,&position_only);
    Ok(json!({"completed":true,"config":c,"device":runtime::identity(device),"precision":"float32","warmups":warmups,"trials":trials,
        "steps":steps,"separate_minus_doubled":layouts,"selected_execution":if doubled {"doubled"}else{"separate"},
        "off_minus_selected":paired(&samples[0],&samples[selected]),
        "isolated_tail_pair":isolated,
        "isolated_membership_schedule":membership,
        "method":"complete-step costs only; two warmups then 11+ interleaved forward/reverse mode rounds; one fixed row/mask/bundle fixture in all rounds so all timed shapes are warmed; ordinary initial parameter values shared by cloning/forking one materialized model; independent advancing AdamW trajectories; no evaluation/checkpoints/quality comparison; off has the historical objective, on both approved branches; equivalent layouts share the objective",
        "fixture":{"example_index":0,"mask_seed":c.seed,"bundle_seed":c.seed^0x434f434f4d4d4954u64},
        "adoption_rule":"select doubled only if the paired 95% saving interval excludes zero; otherwise keep shared-cache separate execution",
        "memory_note":"sampled pool reservation includes all three live runners, not exact peak live activation bytes"}))
}

fn cocommit_tail_isolated(c:&Config,device:&Device,warmups:usize,trials:usize)->Result<Value> {
    use crate::{config::{CocommitExecution,SbdFeedback},sbd_training::CleanInputs,soft::SoftState};
    let target=device.clone().autodiff();
    let mut configurations=[c.clone(),c.clone()];
    configurations[0].sbd.as_mut().unwrap().cocommit.as_mut().unwrap().execution=CocommitExecution::Separate;
    configurations[1].sbd.as_mut().unwrap().cocommit.as_mut().unwrap().execution=CocommitExecution::Doubled;
    target.seed(c.seed);
    let base=LanguageModel::new(&configurations[0],&c.arms[0],&target)?;
    base.materialize_parameters();
    let mut models=Vec::with_capacity(2);
    for config in &configurations {
        let mut model=base.clone().fork(&target);
        model.cocommit_config=config.sbd.as_ref().unwrap().cocommit.clone();
        models.push(model);
    }
    drop(base);
    let corpus=train::loaded_corpus(c)?;
    let rows=data::examples(c,corpus.as_ref(),c.seed,0,c.batch_size,"train")?;
    let blocks=crate::sbd::sample_training_blocks(c,&rows,c.seed)?;
    let bundles=crate::cocommit::sample_bundles(&blocks,c.sbd.as_ref().unwrap().cocommit.as_ref().unwrap().training_bundle_max,c.seed);
    let selected=blocks.iter().zip(&bundles).filter(|(block,_)|block.masked.len()>1).collect::<Vec<_>>();
    ensure!(!selected.is_empty(),"isolated CoCommit tail requires a multi-token block");
    let width=*c.sbd.as_ref().unwrap().block_lengths.iter().max().unwrap();
    let count=selected.len();
    let mut observed=Vec::with_capacity(count*width);
    let mut active=Vec::with_capacity(count*width);
    let mut marks=Vec::with_capacity(count*width);
    let mut starts=Vec::with_capacity(count);
    for (block,bundle) in selected {
        starts.push((block.row,block.start+1));
        for j in 0..width {
            let real=j<block.masked.len();
            active.push(real);
            observed.push(if real && !block.masked[j] {Some(rows[block.row].target[block.start+j] as usize)}else{None});
            marks.push(if real && bundle[j] {1.0f32}else{0.0});
        }
    }
    let frozen=models[0].valid();
    let mut trace=CleanInputs::default();
    let _=frozen.forward_profiled(&Batch::new(&rows,&target),&mut trace);
    let cache=frozen.training_cache_batched(&trace,&starts);
    let soft=SoftState::new(c.vocab(),c.sbd.as_ref().unwrap().candidate_width,observed,active,&target,c.soft_selection_backend)?;
    let prefix=frozen.cached_blocks_prefix_profiled(&cache,&soft,c.sbd.as_ref().unwrap().feedback==SbdFeedback::HardMask,&mut delta_mixers::profile::Unprofiled);
    let marks=Tensor::<2>::from_data(TensorData::new(marks,[count,width]),&target);
    let run=|model:&LanguageModel|->Result<(f64,f64)> {
        runtime::sync(&target)?;
        let start=Instant::now();
        let (a,b)=model.cached_blocks_pair_profiled(&cache,prefix.clone(),marks.clone(),&mut delta_mixers::profile::Unprofiled);
        runtime::sync(&target)?;
        let forward=start.elapsed().as_secs_f64()*1000.0;
        std::hint::black_box((a.logits.mean()+b.logits.mean()).backward());
        runtime::sync(&target)?;
        Ok((forward,start.elapsed().as_secs_f64()*1000.0))
    };
    let mut forward:[Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
    let mut total:[Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
    for iteration in 0..warmups+trials {
        for mode in if iteration%2==0 {[0,1]}else{[1,0]} {
            let (f,t)=run(&models[mode])?;
            if iteration>=warmups {forward[mode].push(f);total[mode].push(t);}
        }
    }
    Ok(json!({"shape":{"chains":count,"length":width,"d_model":c.d_model,"streams":if c.ohc_streams {4}else{1}},
        "warmups":warmups,"trials":trials,
        "separate":{"forward_median_ms":median(&forward[0]),"forward_backward_median_ms":median(&total[0]),"forward_samples_ms":forward[0],"forward_backward_samples_ms":total[0]},
        "doubled":{"forward_median_ms":median(&forward[1]),"forward_backward_median_ms":median(&total[1]),"forward_samples_ms":forward[1],"forward_backward_samples_ms":total[1]},
        "forward_separate_minus_doubled":paired(&forward[0],&forward[1]),"forward_backward_separate_minus_doubled":paired(&total[0],&total[1]),
        "scope":"resident clean caches and saved noisy prefix prepared using frozen model; initialized soft block inputs; both tail/readout/head branches and mean-logit VJP; no clean/shared-prefix backward, teacher, circuit, optimizer or uploads in measured section; no parameter updates"}))
}
