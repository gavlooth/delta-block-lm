//! Execution cost measurements only; no quality evaluation or training study.
use crate::{config::Config,data::{self,Batch,Corpus},model::{self,LanguageModel},profile::median,runtime,sbd,train};
use anyhow::{Result,ensure};
use burn::{optim::{ModuleOptimizer,GradientsParams},prelude::*,tensor::Distribution};
use delta_mixers::prefix_retrieval::{PrefixReadConfig,PrefixRetrievalBackend,read};
use serde_json::{Value,json};
use std::{collections::BTreeMap,sync::{Arc,Mutex},time::Instant};

fn paired(a:&[f64],b:&[f64])->Value {
    let differences=a.iter().zip(b).map(|(a,b)|a-b).collect::<Vec<_>>();
    let mut state=0x324ac8915u64;
    let mut bootstrap=Vec::with_capacity(10000);
    for _ in 0..10000 {
        let resample=(0..differences.len()).map(|_|{
            state^=state<<13;state^=state>>7;state^=state<<17;
            differences[state as usize%differences.len()]
        }).collect::<Vec<_>>();
        bootstrap.push(median(&resample));
    }
    bootstrap.sort_by(f64::total_cmp);
    let interval=[bootstrap[249],bootstrap[9749]];
    json!({"baseline_minus_candidate_samples_ms":differences,"median_saved_ms":median(&differences),
        "bootstrap_95_percent_interval_ms":interval,"bootstrap_resamples":10000,
        "candidate_faster_trials":differences.iter().filter(|&&x|x>0.).count(),"trials":differences.len(),
        "beyond_noise":interval[0]>0.})
}
#[cfg(any(feature="vulkan",feature="cuda"))]
fn launches(device:&Device,run:impl FnOnce()->Result<()>)->Result<Value>{
    #[derive(Default)] struct Count(Mutex<BTreeMap<&'static str,usize>>);
    impl cubecl::logging::LaunchObserver for Count {
        fn launched(&self,name:&'static str){*self.0.lock().unwrap().entry(name).or_default()+=1;}
    }
    runtime::sync(device)?;
    let count=Arc::new(Count::default());
    let observation=cubecl::logging::LaunchObservation::new(count.clone());
    run()?;runtime::sync(device)?;drop(observation);
    let counts=count.0.lock().unwrap();
    Ok(json!({"issued_kernels":counts.values().sum::<usize>(),"by_kernel":*counts}))
}
#[cfg(not(any(feature="vulkan",feature="cuda")))]
fn launches(_device:&Device,_run:impl FnOnce()->Result<()>)->Result<Value>{anyhow::bail!("GPU build required")}
fn isolated(c:&Config,device:&Device,warmups:usize,trials:usize)->Result<Value>{
    let train=device.clone().autodiff();train.seed(c.seed);
    let (b,t,h)=(1,2048,c.heads);
    let q=Tensor::<4>::random([b,t,h,32],Distribution::Uniform(-0.4,0.4),&train).require_grad();
    let k=Tensor::<4>::random([b,t,h,32],Distribution::Uniform(-0.4,0.4),&train).require_grad();
    let v=Tensor::<4>::random([b,t,h,64],Distribution::Uniform(-0.4,0.4),&train).require_grad();
    let scores=Tensor::<4>::random([b,t,h,1],Distribution::Uniform(0.,1.),&train);
    let valid=Tensor::ones([b,t],&train);let docs=Tensor::ones([b,t],&train);
    let sink=Tensor::zeros([h],&train).require_grad();
    let config=c.prefix_block_retrieval.as_ref().unwrap();
    let run=|mode:usize,backward:bool|->Result<()> {
        let output=read(q.clone(),k.clone(),v.clone(),scores.clone(),valid.clone(),docs.clone(),sink.clone(),
            c.cache_capacity,c.chunk_size,PrefixReadConfig {block_size:config.block_size,blocks_per_query:config.blocks_per_query,
                backend:if mode==0 {PrefixRetrievalBackend::Tensor}else{PrefixRetrievalBackend::FusedGpu}});
        if backward {let gradients=output.square().mean().backward();std::hint::black_box(q.grad(&gradients));}
        else {std::hint::black_box(output);}
        Ok(())
    };
    let mut samples:[[Vec<f64>;2];2]=core::array::from_fn(|_|core::array::from_fn(|_|vec![]));
    let mut peak=runtime::sampled_reserved(device);
    for iteration in 0..warmups+trials {for mode in if iteration%2==0 {[0,1]}else{[1,0]} {
        for backward in [false,true] {
            runtime::sync(device)?;let clock=Instant::now();run(mode,backward)?;runtime::sync(device)?;
            if iteration>=warmups {samples[mode][backward as usize].push(clock.elapsed().as_secs_f64()*1000.);}
            if let Some(bytes)=runtime::sampled_reserved(device){peak=Some(peak.unwrap_or(0).max(bytes));}
            runtime::write_json(&c.artifact_dir.join("prefix-isolated-samples.json"),
                &json!({"completed":false,"iteration":iteration,"samples":samples,"sampled_pool_reserved_peak_bytes":peak}))?;
        }
    }}
    let rows=(0..2).map(|mode|Ok(json!({"layout":(["tensor","fused_gpu"][mode]),
        "forward_samples_ms":samples[mode][0],"forward_backward_samples_ms":samples[mode][1],
        "forward_median_ms":median(&samples[mode][0]),"forward_backward_median_ms":median(&samples[mode][1]),
        "forward_launches":launches(device,||run(mode,false))?,"forward_backward_launches":launches(device,||run(mode,true))?}))).collect::<Result<Vec<_>>>()?;
    let own=(config.block_size-c.chunk_size).min((t-1)/c.chunk_size*c.chunk_size);
    let slots=c.cache_capacity+c.chunk_size+config.blocks_per_query.min((t-1)/config.block_size)*config.block_size+own;
    let prefix_pairs=(0..t).step_by(c.chunk_size).map(|first|
        b*h*c.chunk_size.min(t-first)*first).sum::<usize>();
    Ok(json!({"shape":[b,t,h,32,64],"results":rows,"paired_forward":paired(&samples[0][0],&samples[1][0]),
        "paired_forward_backward":paired(&samples[0][1],&samples[1][1]),
        "memory":{"sampled_pool_reserved_peak_bytes":peak,"full_prefix_kv_bytes":b*t*h*96*4,
            "fused_position_bytes":b*t*h*slots*4,"fused_vjp_pair_bytes":b*t*h*slots*2*4,
            "fused_statistics_bytes":b*t*h*2*4,"block_summary_bytes":b*(t/config.block_size).max(1)*h*35*4,
            "position_schedule_bytes":b*h*(t.div_ceil(c.chunk_size)+1)*c.cache_capacity*4,
            "own_block_position_bytes":b*t*h*own*4,
            "tensor_prefix_logit_elements":prefix_pairs,"tensor_prefix_logits_and_probabilities_f32_bytes":prefix_pairs*2*4,
            "structural_scope":"logical tensor-value sizes; not a live allocation sum; projection buffers may already be retained"},
        "scope":"resident clean read; synchronized forward and query/key/value/sink VJP; no projections or optimizer"}))
}
struct Runner {model:Option<LanguageModel>,optimizer:ModuleOptimizer,step:usize}
impl Runner {
    fn run(&mut self,c:&Config,corpus:Option<&Corpus>,device:&Device)->Result<f64>{
        runtime::sync(device)?;let clock=Instant::now();
        let rows=data::examples(c,corpus,c.seed,self.step*c.batch_size,c.batch_size,"train")?;
        let batch=Batch::new(&rows,device);let model=self.model.take().unwrap();
        let (objective,circuit_flows)=if c.sbd.is_some() {
            let result=sbd::training_loss(&model,c,&rows,&batch,c.seed+self.step as u64)?;
            (result.loss,result.circuit_flows)
        } else {(model.loss(model.forward(&batch),&batch),None)};
        model::scalar(objective.clone())?;
        let gradients:GradientsParams=model::checked_gradients(&model,objective)?;
        self.model=Some(train::optimizer_step(c,model,&mut self.optimizer,gradients,circuit_flows)?);
        runtime::sync(device)?;self.step+=1;
        Ok(clock.elapsed().as_secs_f64()*1000.)
    }
}
pub fn benchmark(c:&Config,device:&Device,warmups:usize,trials:usize)->Result<Value>{
    ensure!(c.backend=="vulkan" && !c.fusion && c.prefix_block_retrieval.is_some(),
        "prefix retrieval benchmark requires explicit non-fusion Vulkan retrieval");
    ensure!(warmups>=2 && trials>=11,"requires two warmups and at least eleven paired trials");
    ensure!(c.batch_size==8 && c.context==512,"complete-step cost fixture is B8/T512");
    let isolated=isolated(c,device,warmups,trials)?;
    runtime::write_json(&c.artifact_dir.join("prefix-isolated.json"),&isolated)?;
    device.memory_cleanup();
    let corpus=train::loaded_corpus(c)?;
    let train_device=device.clone().autodiff();train_device.seed(c.seed);
    let mut clean=c.clone();clean.sbd=None;
    let mut off=clean.clone();off.prefix_block_retrieval=None;off.prefix_block_retrieval_backend=PrefixRetrievalBackend::Tensor;
    let base=LanguageModel::new(&off,&c.arms[0],&train_device)?;
    let mut configs=[off,clean.clone(),clean.clone()];configs[1].prefix_block_retrieval_backend=PrefixRetrievalBackend::Tensor;
    configs[2].prefix_block_retrieval_backend=PrefixRetrievalBackend::FusedGpu;
    let retrieval=c.prefix_block_retrieval.as_ref().unwrap();
    let selected_layers=(0..c.layers).filter(|layer|retrieval.layers.contains(layer)).count();
    let mut runners=(0..3).map(|mode|{
        let mut model=base.clone();
        if mode>0 {for &layer in &retrieval.layers {model.blocks[layer].mixer.set_prefix_retrieval(PrefixReadConfig {
            block_size:retrieval.block_size,blocks_per_query:retrieval.blocks_per_query,
            backend:configs[mode].prefix_block_retrieval_backend,
        });}}
        Runner{model:Some(model),optimizer:train::optimizer(&configs[mode]),step:0}
    }).collect::<Vec<_>>();
    let mut samples:[Vec<f64>;3]=core::array::from_fn(|_|vec![]);
    let mut peak=runtime::sampled_reserved(device);
    for iteration in 0..warmups+trials {for mode in if iteration%2==0 {[0,1,2]}else{[2,1,0]} {
        let ms=runners[mode].run(&configs[mode],corpus.as_ref(),&train_device)?;
        if iteration>=warmups {samples[mode].push(ms);}
        if let Some(bytes)=runtime::sampled_reserved(device){peak=Some(peak.unwrap_or(0).max(bytes));}
        runtime::write_json(&c.artifact_dir.join("prefix-step-samples.json"),
            &json!({"completed":false,"iteration":iteration,"samples":samples,"sampled_pool_reserved_peak_bytes":peak}))?;
    }}
    let mut rows=Vec::new();
    for mode in 0..3 {
        rows.push(json!({"layout":(["off","tensor","fused_gpu"][mode]),"step_samples_ms":samples[mode],
            "step_median_ms":median(&samples[mode]),"launches":launches(device,||{
                runners[mode].run(&configs[mode],corpus.as_ref(),&train_device).map(|_|())
            })?}));
    }
    let tensor_to_fused=paired(&samples[1],&samples[2]);
    Ok(json!({"schema":1,"command":"prefix-retrieval-benchmark","completed":true,"config":clean,"source_config":c,
        "device":runtime::identity(device),"warmups":warmups,"trials":trials,"isolated":isolated,
        "complete_steps":rows,"paired_off_to_fused":paired(&samples[0],&samples[2]),
        "paired_tensor_to_fused":tensor_to_fused,"adopt_fused":tensor_to_fused["beyond_noise"],
        "memory":{"sampled_pool_reserved_peak_bytes":peak,
            "selected_layer_full_prefix_kv_bytes":selected_layers*c.batch_size*c.context*c.heads*96*4,
            "selected_layer_block_summary_bytes":selected_layers*c.batch_size*(c.context/retrieval.block_size).max(1)*c.heads*35*4,
            "sampling":"allocator pool reservations after complete samples, not a physical or per-layout live peak; K/V value footprint can share already retained training projections"},
        "protocol":"same initial model and seeded per-step fixture; complete clean causal loss, all gradients, clipping and AdamW; SBD objective omitted explicitly; alternating forward/reverse order; no heldout evaluation or quality measurements"}))
}

pub fn sbd_diagnostic(c:&Config,device:&Device)->Result<Value> {
    ensure!(c.backend=="vulkan" && c.sbd.is_some() && c.prefix_block_retrieval.is_some()
        && c.prefix_block_retrieval_backend==PrefixRetrievalBackend::FusedGpu,
        "SBD retrieval diagnostic requires explicit fused Vulkan retrieval and SBD");
    let mut config=c.clone();config.batch_size=1;config.context=512;
    let corpus=train::loaded_corpus(&config)?;
    let train_device=device.clone().autodiff();train_device.seed(config.seed);
    let model=LanguageModel::new(&config,&config.arms[0],&train_device)?;
    let mut runner=Runner{model:Some(model),optimizer:train::optimizer(&config),step:0};
    let mut milliseconds=0.;
    let dispatches=launches(device,||{
        milliseconds=runner.run(&config,corpus.as_ref(),&train_device)?;
        Ok(())
    })?;
    let counts=dispatches["by_kernel"].as_object().unwrap();
    let prefix_launches=counts.iter().filter(|(name,_)|name.contains("delta_kernels::prefix::"))
        .map(|(_,count)|count.as_u64().unwrap()).sum::<u64>();
    let prefix_select_calls=counts.iter().filter(|(name,_)|name.contains("prefix") && name.contains("select_kernel"))
        .map(|(_,count)|count.as_u64().unwrap()).sum::<u64>();
    let selected_layers=(0..config.layers).filter(|layer|config.prefix_block_retrieval.as_ref().unwrap().layers.contains(layer)).count() as u64;
    let noisy_evaluations=prefix_select_calls.saturating_sub(selected_layers)/selected_layers.max(1);
    let total=dispatches["issued_kernels"].as_u64().unwrap();
    Ok(json!({"schema":1,"command":"prefix-retrieval-sbd-diagnostic","completed":true,"config":config,
        "step_ms":milliseconds,"launches":dispatches,"selected_layer_read_kernel_launches":prefix_launches,
        "selected_layer_read_launch_share":prefix_launches as f64/total as f64,
        "prefix_select_calls":prefix_select_calls,"noisy_block_model_evaluations":noisy_evaluations,
        "block_evaluation_definition":"teacher and differentiable noisy model passes; one prefix-select call per selected layer per pass, excluding the clean forward",
        "sampled_pool_reserved_bytes":runtime::sampled_reserved(device),
        "fits_3000_seconds":milliseconds<=3_000_000.,
        "scope":"one complete B1/T512 clean plus masked-block SBD loss, all gradients, clipping and AdamW; first execution may include kernel compilation",
        "time_share":null,"time_share_note":"launch share is not time share; no synchronized per-component time attribution is inferred from asynchronous launches"}))
}

pub fn sbd_benchmark(c:&Config,device:&Device,warmups:usize,trials:usize)->Result<Value> {
    ensure!(c.backend=="vulkan" && c.sbd.is_some() && c.prefix_block_retrieval.is_some()
        && c.batch_size==1 && matches!(c.context,256|512),
        "SBD retrieval cost requires B1/T256 or B1/T512, explicit Vulkan retrieval and SBD");
    ensure!(warmups>=2 && trials>=11,"requires two warmups and at least eleven paired trials");
    let corpus=train::loaded_corpus(c)?;
    let train_device=device.clone().autodiff();train_device.seed(c.seed);
    let mut off=c.clone();off.prefix_block_retrieval=None;off.prefix_block_retrieval_backend=PrefixRetrievalBackend::Tensor;
    let base=LanguageModel::new(&off,&c.arms[0],&train_device)?;
    let mut enabled=base.clone();
    let retrieval=c.prefix_block_retrieval.as_ref().unwrap();
    for &layer in &retrieval.layers {enabled.blocks[layer].mixer.set_prefix_retrieval(PrefixReadConfig{
        block_size:retrieval.block_size,blocks_per_query:retrieval.blocks_per_query,backend:c.prefix_block_retrieval_backend,
    });}
    let configs=[off,c.clone()];
    let mut runners=[Runner{model:Some(base),optimizer:train::optimizer(c),step:0},
        Runner{model:Some(enabled),optimizer:train::optimizer(c),step:0}];
    let mut samples:[Vec<f64>;2]=core::array::from_fn(|_|vec![]);
    let mut peak=runtime::sampled_reserved(device);
    for iteration in 0..warmups+trials {for mode in if iteration%2==0 {[0,1]}else{[1,0]} {
        let ms=runners[mode].run(&configs[mode],corpus.as_ref(),&train_device)?;
        if iteration>=warmups {samples[mode].push(ms);}
        if let Some(bytes)=runtime::sampled_reserved(device){peak=Some(peak.unwrap_or(0).max(bytes));}
        runtime::write_json(&c.artifact_dir.join("prefix-sbd-step-samples.json"),
            &json!({"completed":false,"iteration":iteration,"samples":samples,"sampled_pool_reserved_peak_bytes":peak}))?;
    }}
    let mut rows=Vec::new();
    for mode in 0..2 {rows.push(json!({"mode":if mode==0 {"off"}else{"on"},
        "step_samples_ms":samples[mode],"step_median_ms":median(&samples[mode]),
        "launches":launches(device,||runners[mode].run(&configs[mode],corpus.as_ref(),&train_device).map(|_|()))?}));}
    Ok(json!({"schema":1,"command":"prefix-retrieval-sbd-benchmark","completed":true,"config":c,
        "warmups":warmups,"trials":trials,"complete_steps":rows,"paired_off_to_on":paired(&samples[0],&samples[1]),
        "sampled_pool_reserved_peak_bytes":peak,
        "protocol":"same initial model and seeded row/mask fixtures; full SBD objective, checked gradients, clipping and AdamW; alternating AB/BA; execution cost only"}))
}
