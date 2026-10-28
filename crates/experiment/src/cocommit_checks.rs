//! CoCommit numerical and execution-path checks; deliberately uses no trained comparison arms.
use crate::{
    config::{Arm, CocommitConfig, Config, SbdConfig, SbdFeedback, SbdJointOutput},
    data::{Batch, Example},
    model::{self, LanguageModel},
    sbd::{self, block_forward},
    sbd_training::CleanInputs,
    soft::SoftState,
};
use anyhow::{Context, Result, ensure};
use burn::{module::{Module,ModuleVisitor,Param},prelude::*};
use delta_mixers::{profile::Unprofiled,sbd::InBlockReads};
use crate::kron_benchmark::{parameters,parameters_with_unused,values};
use serde_json::{Value,json};
use std::{collections::BTreeMap,path::Path};
use crate::train;

fn compare(actual: &[f32], reference: &[f32]) -> Result<Value> {
    sbd::compare(actual,&reference.iter().map(|&x|f64::from(x)).collect::<Vec<_>>())
}
fn named_parameters(model:&LanguageModel)->BTreeMap<String,(Vec<usize>,Vec<f32>)> {
    struct Visitor { path:Vec<String>, values:BTreeMap<String,(Vec<usize>,Vec<f32>)> }
    impl ModuleVisitor for Visitor {
        fn enter_module(&mut self,name:&str,_:&str){self.path.push(name.to_owned());}
        fn exit_module(&mut self,_:&str,_:&str){self.path.pop();}
        fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<D>>) {
            self.values.insert(self.path.join("."),(param.val().dims().to_vec(),values(param.val())));
        }
    }
    let mut visitor=Visitor{path:Vec::new(),values:BTreeMap::new()};
    model.visit(&mut visitor);
    visitor.values
}


fn compare_parameter_gradients(
    actual: &[(Vec<usize>,Vec<f32>)],
    expected: &[(Vec<usize>,Vec<f32>)],
) -> Result<Value> {
    ensure!(actual.len()==expected.len(),"CPU/Vulkan parameter tensor count differs");
    let mut max_abs=0.0f64;
    let mut max_ratio=0.0f64;
    let mut elements=0usize;
    for (index,((actual_shape,actual),(expected_shape,expected))) in actual.iter().zip(expected).enumerate() {
        ensure!(actual_shape==expected_shape,"parameter gradient {index} shape differs");
        let report=compare(actual,expected).with_context(||format!("parameter gradient {index} shape {actual_shape:?}"))?;
        max_abs=max_abs.max(report["max_abs"].as_f64().unwrap());
        max_ratio=max_ratio.max(report["maximum_tolerance_ratio"].as_f64().unwrap());
        elements+=expected.len();
    }
    Ok(json!({"passed":true,"parameter_tensors":actual.len(),"elements":elements,"max_abs":max_abs,"maximum_tolerance_ratio":max_ratio}))
}

fn retrieval_case(device: &Device, with_moba: bool, doubled: bool) -> Result<Value> {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/lm512-step8-retrieval.json");
    let mut gpu_cfg=Config::load(&path)?;
    let sbd=gpu_cfg.sbd.as_mut().context("retrieval config lacks SBD")?;
    sbd.cocommit=Some(CocommitConfig {execution:if doubled {crate::config::CocommitExecution::Doubled}
        else {crate::config::CocommitExecution::Separate},..CocommitConfig::default()});
    sbd.joint_output=SbdJointOutput::CodpulaContextual;
    if with_moba {
        gpu_cfg.prefix_block_retrieval.as_mut().context("retrieval fixture lacks MoBA config")?.layers=vec![3];
    } else {
        gpu_cfg.prefix_block_retrieval=None;
        gpu_cfg.prefix_block_retrieval_backend=delta_mixers::prefix_retrieval::PrefixRetrievalBackend::Tensor;
    }
    ensure!(gpu_cfg.d_model==256 && gpu_cfg.heads==4 && gpu_cfg.key_dim==32 && gpu_cfg.value_dim==64
        && gpu_cfg.sbd.as_ref().unwrap().block_lengths==[8],"retrieval fixture shape differs from step-8 config");
    if with_moba {
        ensure!(gpu_cfg.prefix_block_retrieval.as_ref().is_some_and(|r|r.block_size==64 && r.blocks_per_query==2 && r.layers==[3]),
            "retrieval fixture is not the configured layer-3 MoBA path");
    }
    ensure!(gpu_cfg.backend=="vulkan","full D256 contextual CoCommit check requires the configured Vulkan backend");
    gpu_cfg.validate()?;
    let arm=gpu_cfg.arms[0].clone();
    let mut cpu_cfg=gpu_cfg.clone();
    cpu_cfg.backend="cpu".into(); cpu_cfg.fusion=false; cpu_cfg.autotune=false;
    cpu_cfg.scan_backend=delta_mixers::ScanBackend::TensorSolver;
    cpu_cfg.hola_backend=delta_mixers::HolaBackend::Tensor;
    cpu_cfg.projection_backend=delta_mixers::ProjectionBackend::Separate;
    cpu_cfg.normalization_backend=delta_mixers::NormalizationBackend::Tensor;
    cpu_cfg.convolution_backend=delta_mixers::ConvolutionBackend::Tensor;
    cpu_cfg.head_backend=crate::config::HeadBackend::Native;
    cpu_cfg.loss_backend=crate::config::LossBackend::Tensor;
    cpu_cfg.update_backend=crate::config::UpdateBackend::Tensor;
    cpu_cfg.epilogue_backend=crate::config::EpilogueBackend::Native;
    cpu_cfg.ohc_backend=delta_mixers::ohc::OhcBackend::Tensor;
    cpu_cfg.wavelet_backend=delta_mixers::WaveletBackend::Tensor;
    cpu_cfg.lngram_backend=delta_mixers::lngram::LngramBackend::Tensor;
    cpu_cfg.prefix_block_retrieval_backend=delta_mixers::prefix_retrieval::PrefixRetrievalBackend::Tensor;
    cpu_cfg.soft_selection_backend=crate::config::SoftSelectionBackend::Cpu;
    cpu_cfg.sbd.as_mut().unwrap().noisy_read_backend=delta_mixers::sbd::NoisyReadBackend::Tensor;
    cpu_cfg.validate()?;
    let cpu=Device::ndarray().autodiff();
    let gpu=device.clone().autodiff();
    cpu.seed(gpu_cfg.seed); gpu.seed(gpu_cfg.seed);
    let cpu_model=LanguageModel::new(&cpu_cfg,&arm,&cpu)?;
    let mut gpu_model=cpu_model.clone().fork(&gpu);
    gpu_model.set_scan_backend(gpu_cfg.scan_backend);
    gpu_model.set_hola_backend(gpu_cfg.hola_backend);
    gpu_model.set_projection_backend(gpu_cfg.projection_backend);
    gpu_model.set_convolution_backend(gpu_cfg.convolution_backend);
    gpu_model.set_normalization_backend(gpu_cfg.normalization_backend);
    gpu_model.set_ohc_backend(gpu_cfg.ohc_backend);
    gpu_model.set_wavelet_backend(gpu_cfg.wavelet_backend);
    gpu_model.set_lngram_backend(gpu_cfg.lngram_backend);
    gpu_model.set_noisy_read_backend(gpu_cfg.sbd.as_ref().unwrap().noisy_read_backend);
    gpu_model.head_backend=gpu_cfg.head_backend;
    gpu_model.loss_backend=gpu_cfg.loss_backend;
    gpu_model.update_backend=gpu_cfg.update_backend;
    gpu_model.epilogue_backend=gpu_cfg.epilogue_backend;
    if let Some(retrieval)=gpu_cfg.prefix_block_retrieval.as_ref() {
        for &layer in &retrieval.layers {
            gpu_model.blocks[layer].mixer.set_prefix_retrieval(delta_mixers::prefix_retrieval::PrefixReadConfig {
                block_size:retrieval.block_size,blocks_per_query:retrieval.blocks_per_query,backend:gpu_cfg.prefix_block_retrieval_backend,
            });
        }
    }
    let prefix=(0..219).map(|position|if position==0 {256} else {((position*37+11)%256) as i32}).collect::<Vec<_>>();
    let cpu_soft=sbd::state(&cpu_cfg,vec![None;8],&cpu)?;
    let gpu_soft=sbd::state(&gpu_cfg,vec![None;8],&gpu)?;
    let mut cpu_trace=CleanInputs::default(); let _=cpu_model.forward_profiled(&sbd::prefix_batch(&prefix,&cpu),&mut cpu_trace);
    let mut gpu_trace=CleanInputs::default(); let _=gpu_model.forward_profiled(&sbd::prefix_batch(&prefix,&gpu),&mut gpu_trace);
    let cpu_cache=cpu_model.training_cache(&cpu_trace,0,&[prefix.len()]);
    let gpu_cache=gpu_model.training_cache(&gpu_trace,0,&[prefix.len()]);
    let marks_cpu=Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0]],&cpu);
    let marks_gpu=Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0]],&gpu);
    let mut cpu_model=cpu_model; let mut gpu_model=gpu_model;
    for model in [&mut cpu_model,&mut gpu_model] {
        let d=model.embedding.weight.val().dims()[1];
        let marker=Tensor::<2>::from_data(TensorData::new((0..4*d).map(|i|((i*19%31) as f32-15.0)*0.0005).collect::<Vec<_>>(),[4,d]),&model.embedding.weight.val().device()).require_grad();
        model.cocommit_marker=Some(model.cocommit_marker.take().unwrap().map(|_|marker));
    }
    let cpu_prefix=cpu_model.cached_block_prefix_profiled(&cpu_cache,0,&cpu_soft,false,&mut Unprofiled);
    let gpu_prefix=gpu_model.cached_block_prefix_profiled(&gpu_cache,0,&gpu_soft,false,&mut Unprofiled);
    let cpu_rebuilt_a=crate::cocommit::reconstruct_output(&cpu_model,&prefix,&cpu_soft,false,None,&cpu);
    let gpu_rebuilt_a=crate::cocommit::reconstruct_output(&gpu_model,&prefix,&gpu_soft,false,None,&gpu);
    let (cpu_a,cpu_b)=(cpu_model.cached_block_tail_profiled(&cpu_cache,0,cpu_prefix.clone(),None,&mut Unprofiled),
        cpu_model.cached_block_tail_profiled(&cpu_cache,0,cpu_prefix,Some(marks_cpu),&mut Unprofiled));
    let (gpu_a,gpu_b)=(gpu_model.cached_block_tail_profiled(&gpu_cache,0,gpu_prefix.clone(),None,&mut Unprofiled),
        gpu_model.cached_block_tail_profiled(&gpu_cache,0,gpu_prefix,Some(marks_gpu),&mut Unprofiled));
    let cpu_rebuilt=crate::cocommit::reconstruct_output(&cpu_model,&prefix,&cpu_soft,false,Some(Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0]],&cpu)),&cpu);
    let gpu_rebuilt=crate::cocommit::reconstruct_output(&gpu_model,&prefix,&gpu_soft,false,Some(Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0]],&gpu)),&gpu);
    let reconstruction=json!({
        "CPU_A":compare(&values(cpu_a.logits.clone()),&values(cpu_rebuilt_a.logits)).context("MoBA CPU cached/reconstructed A")?,
        "CPU_B":compare(&values(cpu_b.logits.clone()),&values(cpu_rebuilt.logits)).context("MoBA CPU cached/reconstructed B")?,
        "Vulkan_A":compare(&values(gpu_a.logits.clone()),&values(gpu_rebuilt_a.logits)).context("MoBA Vulkan cached/reconstructed A")?,
        "Vulkan_B":compare(&values(gpu_b.logits.clone()),&values(gpu_rebuilt.logits)).context("MoBA Vulkan cached/reconstructed B")?
    });
    let starts=[(0,128),(0,144)];
    let cpu_batch_cache=cpu_model.training_cache_batched(&cpu_trace,&starts);
    let gpu_batch_cache=gpu_model.training_cache_batched(&gpu_trace,&starts);
    let cpu_batch_soft=SoftState::new(257,4,vec![None;16],vec![true;16],&cpu,cpu_cfg.soft_selection_backend)?;
    let gpu_batch_soft=SoftState::new(257,4,vec![None;16],vec![true;16],&gpu,gpu_cfg.soft_selection_backend)?;
    let cpu_batch_prefix=cpu_model.cached_blocks_prefix_profiled(&cpu_batch_cache,&cpu_batch_soft,false,&mut Unprofiled);
    let gpu_batch_prefix=gpu_model.cached_blocks_prefix_profiled(&gpu_batch_cache,&gpu_batch_soft,false,&mut Unprofiled);
    let cm=Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0]],&cpu);
    let gm=Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0],[0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0]],&gpu);
    let (cpu_batched_a,cpu_batched_b)=cpu_model.cached_blocks_pair_profiled(&cpu_batch_cache,cpu_batch_prefix,cm,&mut Unprofiled);
    let (gpu_batched_a,gpu_batched_b)=gpu_model.cached_blocks_pair_profiled(&gpu_batch_cache,gpu_batch_prefix,gm,&mut Unprofiled);
    let mut cpu_per_a=Vec::new(); let mut cpu_per_b=Vec::new();
    let mut gpu_per_a=Vec::new(); let mut gpu_per_b=Vec::new();
    for (index,&start) in [128usize,144].iter().enumerate() {
        let cc=cpu_model.training_cache(&cpu_trace,0,&[start]);
        let gc=gpu_model.training_cache(&gpu_trace,0,&[start]);
        let cs=sbd::state(&cpu_cfg,vec![None;8],&cpu)?;
        let gs=sbd::state(&gpu_cfg,vec![None;8],&gpu)?;
        let cp=cpu_model.cached_block_prefix_profiled(&cc,0,&cs,false,&mut Unprofiled);
        let gp=gpu_model.cached_block_prefix_profiled(&gc,0,&gs,false,&mut Unprofiled);
        let cm=Tensor::<2>::from_data(if index==0 {[[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0]]} else {[[0.0f32,1.0,0.0,1.0,0.0,1.0,0.0,1.0]]},&cpu);
        let gm=Tensor::<2>::from_data(if index==0 {[[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0]]} else {[[0.0f32,1.0,0.0,1.0,0.0,1.0,0.0,1.0]]},&gpu);
        cpu_per_a.extend(values(cpu_model.cached_block_tail_profiled(&cc,0,cp.clone(),None,&mut Unprofiled).logits));
        cpu_per_b.extend(values(cpu_model.cached_block_tail_profiled(&cc,0,cp,Some(cm),&mut Unprofiled).logits));
        gpu_per_a.extend(values(gpu_model.cached_block_tail_profiled(&gc,0,gp.clone(),None,&mut Unprofiled).logits));
        gpu_per_b.extend(values(gpu_model.cached_block_tail_profiled(&gc,0,gp,Some(gm),&mut Unprofiled).logits));
    }
    let batching=json!({
        "CPU_A":compare(&values(cpu_batched_a.logits),&cpu_per_a).context("MoBA CPU batched/per-block A")?,
        "CPU_B":compare(&values(cpu_batched_b.logits),&cpu_per_b).context("MoBA CPU batched/per-block B")?,
        "Vulkan_A":compare(&values(gpu_batched_a.logits),&gpu_per_a).context("MoBA Vulkan batched/per-block A")?,
        "Vulkan_B":compare(&values(gpu_batched_b.logits),&gpu_per_b).context("MoBA Vulkan batched/per-block B")?
    });
    let normalized=|model:&LanguageModel,soft:&SoftState,target:&Device|->Result<(f32,Vec<(Vec<usize>,Vec<f32>)>)>{
        let (cache,hidden)=crate::cocommit::prepare_reconstruction(model,&prefix,soft,false,target);
        let a=model.cached_block_tail_profiled(&cache,0,hidden.clone(),None,&mut Unprofiled);
        let b=model.cached_block_tail_profiled(&cache,0,hidden,
            Some(Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0,1.0,0.0,1.0,0.0]],target)),&mut Unprofiled);
        let clean=Tensor::<2,Int>::from_data(TensorData::new((219..227).map(|p|((p*37+11)%256) as i32).collect::<Vec<_>>(),[1,8]),target);
        let selected=Tensor::<2,Bool>::from_data([[true,false,true,false,true,false,true,false]],target);
        let (circuit_a,observed_a)=sbd::prepare_distribution(model,a,soft,1.0)?;
        let (circuit_b,observed_b)=sbd::prepare_distribution(model,b,soft,1.0)?;
        let loss=circuit_a.conditional_nll(clean.clone(),observed_a,8.0)?.loss
            +circuit_b.event_nll(clean,observed_b,selected,8.0)?.loss;
        let value=model::scalar(loss.clone())? as f32;
        let unused=model::unused_soft_parameters();
        let (checked,_) = model::checked_gradients_with_unused(model,loss,&unused,&mut Unprofiled)?;
        let gradients=parameters_with_unused(model,Some(&checked),&unused)?;
        Ok((value,gradients))
    };
    let (cpu_loss,cpu_gradient)=normalized(&cpu_model,&cpu_soft,&cpu)?;
    let (gpu_loss,gpu_gradient)=normalized(&gpu_model,&gpu_soft,&gpu)?;
    let gradients=json!({"loss":compare(&[gpu_loss],&[cpu_loss])?,
        "all_neural_and_circuit_gradients":compare_parameter_gradients(&gpu_gradient,&cpu_gradient).context("contextual CoCommit normalized event objective all gradients")?,
        "objective":"unscaled sum of contextual A conditional NLL over all eight free positions and B marked-subset event NLL, each normalized by eight valid prediction positions",
        "same_prefix_length":219,"same_active_length":8,"same_parameters":true});
    let outputs=json!({
        "A_logits":compare(&values(gpu_a.logits),&values(cpu_a.logits)).context("contextual CPU/Vulkan A")?,
        "B_logits":compare(&values(gpu_b.logits),&values(cpu_b.logits)).context("contextual CPU/Vulkan B")?
    });
    let visibility_example=Example {
        input:(0..24).map(|i|if i==0 {256} else {((i*37+11)%256) as i32}).collect(),
        target:(0..24).map(|i|((i*19+7)%256) as i32).collect(),
        weight:vec![1.0;24],groups:vec![None;24],
    };
    let visibility=json!({
        "CPU":visibility_case(&cpu_model,&cpu_cfg,&visibility_example,&cpu)?,
        "Vulkan":visibility_case(&gpu_model,&gpu_cfg,&visibility_example,&gpu)?
    });
    let cpu_single=sbd::state(&cpu_cfg,vec![None],&cpu)?;
    let gpu_single=sbd::state(&gpu_cfg,vec![None],&gpu)?;
    let cpu_singleton=block_forward(&cpu_model,&prefix,&cpu_single,false,&cpu);
    let gpu_singleton=block_forward(&gpu_model,&prefix,&gpu_single,false,&gpu);
    let cpu_causal=cpu_model.forward_output_profiled(&sbd::prefix_batch(&prefix,&cpu),&mut Unprofiled)
        .slice_positions(prefix.len()-1,1);
    let gpu_causal=gpu_model.forward_output_profiled(&sbd::prefix_batch(&prefix,&gpu),&mut Unprofiled)
        .slice_positions(prefix.len()-1,1);
    let singleton=json!({
        "CPU":compare(&values(cpu_singleton.logits),&values(cpu_causal.logits))?,
        "Vulkan":compare(&values(gpu_singleton.logits),&values(gpu_causal.logits))?
    });
    Ok(json!({"passed":true,"configuration":path,"profile":"codpula_contextual",
        "d_model":256,"heads":4,"key_dim":32,"value_dim":64,
        "active_block_length":8,"prefix_length":219,"moba_enabled":with_moba,
        "retrieval_layers":if with_moba {vec![3]} else {Vec::<usize>::new()},
        "complete_retrieval_blocks":if with_moba {3} else {0},"own_block_clean_tokens":if with_moba {Some(27)} else {None},
        "execution":if doubled {"Doubled"} else {"Separate"},
        "tail_execution":"two C-row tails share clean banks",
        "reconstruction":reconstruction,"per_block_vs_batched":batching,"CPU_vs_Vulkan_outputs":outputs,
        "contextual_visibility_O_padding_leakage":visibility,"L1_causal_bypass":singleton,
        "CPU_vs_Vulkan_all_neural_and_circuit_gradients":gradients,
        "known_negative_evidence":{"artifacts":["artifacts/cocommit/retrieval-gradient-failure.json","artifacts/cocommit/plain-sbd-retrieval-gradient-stress.json"],
            "documentation":"docs/remasking.md §15","objective":"artificial squared-loss diagnostic; not executed by this acceptance route"}}))
}
fn cocommit_checkpoint_continuation(c:&Config,device:&Device,seed:u64)->Result<Value> {
    let mut cfg=Config::default();
    cfg.task="lm".into(); cfg.backend="cpu".into();
    cfg.d_model=16; cfg.layers=4; cfg.heads=4; cfg.key_dim=32; cfg.value_dim=64;
    cfg.chunk_size=16; cfg.cache_capacity=4; cfg.context=8; cfg.batch_size=2; cfg.seed=seed;
    cfg.association_counts=vec![1];
    cfg.train_tokens=16; cfg.scan_backend=delta_mixers::ScanBackend::TensorSolver;
    cfg.hola_backend=delta_mixers::HolaBackend::Tensor;
    cfg.projection_backend=delta_mixers::ProjectionBackend::Separate;
    cfg.normalization_backend=delta_mixers::NormalizationBackend::Tensor;
    cfg.convolution_backend=delta_mixers::ConvolutionBackend::Tensor;
    cfg.head_backend=crate::config::HeadBackend::Native;
    cfg.loss_backend=crate::config::LossBackend::Tensor;
    cfg.update_backend=crate::config::UpdateBackend::Tensor;
    cfg.epilogue_backend=crate::config::EpilogueBackend::Native;
    cfg.ohc_streams=true; cfg.ohc_backend=delta_mixers::ohc::OhcBackend::Tensor;
    cfg.gradient_clip=0.05;
    cfg.learning_rate=c.learning_rate;cfg.weight_decay=c.weight_decay;
    cfg.soft_selection_backend=crate::config::SoftSelectionBackend::Cpu;
    #[cfg(any(feature="vulkan",feature="cuda"))]
    if matches!(device.clone().inner().as_dispatch(),burn::backend::DispatchDevice::Cube(_)) {
        cfg.backend=c.backend.clone();
        cfg.scan_backend=c.scan_backend;cfg.hola_backend=c.hola_backend;
        cfg.projection_backend=c.projection_backend;cfg.normalization_backend=c.normalization_backend;
        cfg.convolution_backend=c.convolution_backend;cfg.ohc_backend=c.ohc_backend;
        cfg.head_backend=c.head_backend;cfg.loss_backend=c.loss_backend;
        cfg.update_backend=c.update_backend;cfg.epilogue_backend=c.epilogue_backend;
        cfg.soft_selection_backend=c.soft_selection_backend;
    }
    let source=c.sbd.as_ref().context("checkpoint gate needs SBD config")?;
    cfg.sbd=Some(SbdConfig {block_lengths:vec![4],training_block_lengths:vec![4],feedback:source.feedback,block_execution:source.block_execution,in_block_reads:InBlockReads::HolaRead,rows_per_pass:Some(1),
        noisy_read_backend:if cfg.backend=="cpu" {delta_mixers::sbd::NoisyReadBackend::Tensor}else{source.noisy_read_backend},
        joint_output:source.joint_output,
        joint_execution:if cfg.backend=="cpu" {crate::codpula::CodpulaExecution::Tensor}else{source.joint_execution},
        cocommit:Some(CocommitConfig::default()),..SbdConfig::default()});
    let arm=Arm{name:"cocommit-checkpoint".into(),mixer:"kronlstm".into(),cache:"surprise".into()};
    cfg.arms=vec![arm.clone()]; cfg.validate()?;
    let examples=(0..2).map(|row| {
        let sequence=std::iter::once(256).chain((0..8).map(|position|((position*37+row*53+11)%256) as i32)).collect::<Vec<_>>();
        Example{input:sequence[..8].to_vec(),target:sequence[1..].to_vec(),weight:vec![1.0;8],groups:vec![None;8]}
    }).collect::<Vec<_>>();
    let train_device=device.clone().autodiff(); train_device.seed(seed);
    let step=|model:LanguageModel,optimizer:&mut train::NeuralOptimizer,rows:&[Example],step_seed:u64|->Result<(LanguageModel,f64,usize)> {
        let batch=Batch::new(rows,&train_device);
        let result=crate::sbd_step::backward(&model,&cfg,rows,&batch,step_seed)?;
        ensure!(result.passes==2 && result.rows_per_pass==1,"checkpoint fixture did not accumulate both single-row passes");
        let loss_value=result.loss;
        let clipped=parameters(&model,Some(&result.gradients))?.iter().filter(|(_,values)|
            values.iter().map(|&x|f64::from(x).powi(2)).sum::<f64>().sqrt()>f64::from(cfg.gradient_clip)).count();
        ensure!(clipped>0,"production continuation fixture did not engage clipping");
        let updated=crate::train::optimizer_step(&cfg,model,optimizer,result.gradients,result.circuit_flows)?;
        Ok((updated,loss_value,clipped))
    };
    let initial=LanguageModel::new(&cfg,&arm,&train_device)?;
    let mut uninterrupted_optimizer=crate::train::optimizer(&cfg);
    let (trained,first_loss,first_clipped)=step(initial,&mut uninterrupted_optimizer,&examples,seed+1)?;
    let circuit_time=|model:&LanguageModel|->Result<Option<f64>>{
        model.codpula.as_ref().map(|head|model::scalar(head.probability.step.val().float())).transpose()
    };
    let saved_time=circuit_time(&trained)?;
    if trained.codpula.is_some(){ensure!(saved_time==Some(1.0),"initial probability update was not recorded");}
    let stamp=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let directory=std::env::temp_dir().join(format!("cocommit-checkpoint-{}-{stamp}",std::process::id()));
    let path=directory.join("checkpoint-1");
    let state=crate::train::State {schema:1,config_sha256:cfg.training_fingerprint(cfg.resolved_tokens()?)?,
        data:json!({"fixture_sha256":crate::data::fixture_hash(&examples)}),arm:arm.name.clone(),
        seed,step:1,tokens:16,training_seconds:0.0,initial_evaluation:Value::Null};
    crate::train::checkpoint(&cfg,&path,&trained,&uninterrupted_optimizer,&state)?;
    let restored=crate::checkpoint::load_model(LanguageModel::new(&cfg,&arm,&train_device)?,path.join("model.burnpack"),&train_device)?;
    let mut restored_optimizer=crate::checkpoint::load_optimizer(&cfg,path.join("optimizer.burnpack"))?;
    let loaded_parameters=parameters(&restored,None)?;
    let trained_parameters=parameters(&trained,None)?;
    ensure_exact_parameter_bits(&loaded_parameters,&trained_parameters,"reloaded CoCommit model")?;
    let model_equal=compare_parameter_gradients(&loaded_parameters,&trained_parameters)?;
    ensure!(circuit_time(&restored)?==saved_time,"checkpoint changed circuit optimizer time");
    ensure!(values(trained.cocommit_marker.as_ref().unwrap().val().clone()).iter().any(|&x|x!=0.0),
        "diagnostic training step did not update CoCommit marker");
    let signature=crate::checkpoint::optimizer_signature(&cfg,uninterrupted_optimizer.into_bytes()?)?;
    ensure!(signature==crate::checkpoint::optimizer_signature(&cfg,restored_optimizer.into_bytes()?)?,
        "checkpoint changed AdamW state including marker moments");
    let mut uninterrupted=trained; let mut continued=restored; let mut continuation=Vec::new();
    for index in 0..1 {
        let mut next_rows=examples.clone();
        for row in &mut next_rows {
            for value in &mut row.input[1..] {*value=(*value+index as i32+1)%256;}
            for value in &mut row.target {*value=(*value+index as i32+1)%256;}
        }
        let step_seed=seed+2+index as u64;
        let (next,loss_a,clipped_a)=step(uninterrupted,&mut uninterrupted_optimizer,&next_rows,step_seed)?;
        let (loaded_next,loss_b,clipped_b)=step(continued,&mut restored_optimizer,&next_rows,step_seed)?;
        ensure!(clipped_a==clipped_b,"continued clipping decisions differ");
        ensure!(loss_a.to_bits()==loss_b.to_bits(),"CoCommit checkpoint continuation loss differs at step {}",index+2);
        let loaded_values=parameters(&loaded_next,None)?;
        let expected_values=parameters(&next,None)?;
        ensure_exact_parameter_bits(&loaded_values,&expected_values,"continued CoCommit model")?;
        let equal=compare_parameter_gradients(&loaded_values,&expected_values)?;
        ensure!(circuit_time(&loaded_next)?==circuit_time(&next)?,"continued circuit optimizer time differs");
        if next.codpula.is_some() {
            ensure!(circuit_time(&next)?==Some((index+2) as f64),"row-accumulated step did not perform exactly one circuit update");
        }
        ensure!(crate::checkpoint::optimizer_signature(&cfg,uninterrupted_optimizer.into_bytes()?)?==
            crate::checkpoint::optimizer_signature(&cfg,restored_optimizer.into_bytes()?)?,
            "CoCommit AdamW marker moments diverged at step {}",index+2);
        continuation.push(json!({"step":index+2,"loss":loss_a,"parameters":equal,"optimizer_and_marker_moments_identical":true,
            "clipped_parameter_tensors":clipped_a}));
        uninterrupted=next; continued=loaded_next;
    }
    let checkpoint_path=path.display().to_string();
    std::fs::remove_dir_all(&directory)?;
    Ok(json!({"passed":true,"checkpoint":checkpoint_path,"first_loss":first_loss,"loaded_parameters":model_equal,
        "marker_values_identical":true,"marker_optimizer_moments_identical":true,"continuation":continuation,
        "backward_path":"sbd_step::backward","batch_size":cfg.batch_size,"rows_per_pass":1,"backward_passes":2,
        "optimizer_path":"train::optimizer_step including configured clipping, AdamW and one circuit update",
        "optimizer_backend":cfg.update_backend,"feedback":source.feedback,"joint_output":source.joint_output,
        "circuit_time_identical":true,"saved_circuit_time":saved_time,"continued_circuit_time":circuit_time(&uninterrupted)?,
        "probability_rows_and_dual_momenta_included":uninterrupted.codpula.is_some(),
        "clip_threshold":cfg.gradient_clip,"first_step_clipped_tensors":first_clipped}))
}
fn model_case(c: &Config, device: &Device, ohc: bool, codpula: bool) -> Result<Value> {
    let mut cfg = c.clone();
    cfg.backend = "cpu".into();
    cfg.d_model = 16;
    cfg.layers = 4;
    cfg.heads = 4;
    cfg.key_dim = 32;
    cfg.value_dim = 64;
    cfg.chunk_size = 16;
    cfg.cache_capacity = 4;
    cfg.ohc_streams = ohc;
    cfg.ohc_backend = delta_mixers::ohc::OhcBackend::Tensor;
    cfg.wavelet_layers.clear();
    cfg.wavelet_backend=delta_mixers::WaveletBackend::Tensor;
    cfg.lngram_backend=delta_mixers::lngram::LngramBackend::Tensor;
    cfg.prefix_block_retrieval_backend=delta_mixers::prefix_retrieval::PrefixRetrievalBackend::Tensor;
    cfg.soft_selection_backend=crate::config::SoftSelectionBackend::Cpu;
    cfg.lngram_layers.clear();
    cfg.prefix_block_retrieval = None;
    cfg.scan_backend = delta_mixers::ScanBackend::TensorSolver;
    cfg.hola_backend = delta_mixers::HolaBackend::Tensor;
    cfg.projection_backend = delta_mixers::ProjectionBackend::Separate;
    cfg.normalization_backend = delta_mixers::NormalizationBackend::Tensor;
    cfg.convolution_backend = delta_mixers::ConvolutionBackend::Tensor;
    cfg.head_backend = crate::config::HeadBackend::Native;
    cfg.loss_backend = crate::config::LossBackend::Tensor;
    cfg.update_backend = crate::config::UpdateBackend::Tensor;
    cfg.epilogue_backend = crate::config::EpilogueBackend::Native;
    cfg.sbd = Some(SbdConfig {
        block_lengths: vec![1, 4],
        training_block_lengths: vec![1,4],
        in_block_reads: InBlockReads::HolaRead,
        feedback: SbdFeedback::HardMask,
        cocommit: Some(CocommitConfig {execution:crate::config::CocommitExecution::Doubled,..CocommitConfig::default()}),
        ..SbdConfig::default()
    });
    if codpula { cfg.sbd.as_mut().unwrap().joint_output=SbdJointOutput::Codpula; }
    let arm = Arm { name: "cocommit-check".into(), mixer: "kronlstm".into(), cache: "surprise".into() };
    cfg.arms=vec![arm.clone()];
    cfg.validate()?;
    let cpu = Device::ndarray().autodiff();
    cpu.seed(721);
    let reference = LanguageModel::new(&cfg, &arm, &cpu)?;
    let disabled_cfg={let mut x=cfg.clone();x.sbd.as_mut().unwrap().cocommit=None;x};
    cpu.seed(721);
    let disabled=LanguageModel::new(&disabled_cfg,&arm,&cpu)?;
    cpu.seed(721);
    let disabled_parameters=named_parameters(&disabled);
    cpu.seed(721);
    let enabled_ordinary_parameters=named_parameters(&reference).into_iter()
        .filter(|(name,_)|!name.split('.').any(|part|part=="cocommit_marker"))
        .collect::<BTreeMap<_,_>>();
    for (name,value) in &disabled_parameters {
        ensure!(enabled_ordinary_parameters.get(name)==Some(value),
            "CoCommit initialization changed ordinary parameter {name}");
    }
    let requested_device = device.clone().autodiff();
    let mut requested = reference.clone().fork(&requested_device);
    requested_scan(&mut requested,&requested_device);
    let target=&requested_device;
    let initial_marker=values(requested.cocommit_marker.as_ref().unwrap().val().clone());
    ensure!(initial_marker.iter().all(|&x|x==0.0),"CoCommit marker is not zero initialized");
    let rows=if ohc {4} else {1};
    let marker=Tensor::<2>::from_data(TensorData::new((0..rows*16).map(|i|((i*13%23) as f32-11.0)*0.003).collect::<Vec<_>>(),[rows,16]),&requested_device).require_grad();
    requested.cocommit_marker=Some(requested.cocommit_marker.take().unwrap().map(|_|marker));
    let sequence = vec![256, 3, 19, 41, 7, 83, 2, 111, 5];
    let prefix = &sequence[..5];
    let observed=vec![None;4];
    let soft = sbd::state(&cfg, observed, target)?;
    let mut legacy_model=requested.clone();
    legacy_model.cocommit_marker=None;
    let legacy = block_forward(&legacy_model, prefix, &soft, true, target);
    let no_mark = crate::cocommit::reconstruct_output(&requested, prefix, &soft, true, None, target);
    let mark_values = Tensor::<2>::from_data([[1.0f32, 0.0, 1.0, 0.0]], target);
    let marked = crate::cocommit::reconstruct_output(&requested, prefix, &soft, true, Some(mark_values), target);
    let zero_mark = crate::cocommit::reconstruct_output(&requested, prefix, &soft, true,
        Some(Tensor::<2>::zeros([1,4], target)), target);
    let legacy_logits = values(legacy.logits);
    let no_mark_logits = values(no_mark.logits);
    let marked_logits = values(marked.logits.clone());
    let zero_logits = values(zero_mark.logits);
    let baseline = compare(&no_mark_logits, &legacy_logits).context("saved-prefix reconstruction versus full cached block")?;
    let zero_equivalence = compare(&zero_logits, &no_mark_logits).context("zero-marker A/B equivalence")?;
    ensure!(marked_logits.iter().zip(&no_mark_logits).any(|(a,b)| (a-b).abs()>1e-7), "nonzero marker had no effect");
    let marker_parameter=requested.cocommit_marker.as_ref().unwrap();
    let gradients = marked.logits.sum().backward();
    let marker_gradient=values(marker_parameter.val().grad(&gradients).context("CoCommit marker gradient not found")?);
    let marker_shape = if ohc { [4,16] } else { [1,16] };
    ensure!(marker_parameter.val().dims()==marker_shape,"marker tensor has wrong full-representation shape");
    let marker_gradient_max=marker_gradient.iter().map(|x|x.abs()).fold(0.0f32,f32::max);
    ensure!(marker_gradient_max.is_finite() && marker_gradient_max > 0.0,"marked tail produced no finite marker gradient");
    let mut capture = CleanInputs::default();
    let _ = requested.forward_profiled(&crate::sbd::prefix_batch(prefix,target),&mut capture);
    let cache = requested.training_cache(&capture,0,&[prefix.len()]);
    let cached = requested.cached_block(&cache,0,&soft,true);
    let cache_repeat = compare(&values(cached.logits),&legacy_logits).context("clean-cache repeat after marked reconstruction")?;
    let shared_prefix=requested.cached_block_prefix_profiled(&cache,0,&soft,true,&mut Unprofiled);
    let shared_a=requested.cached_block_tail_profiled(&cache,0,shared_prefix.clone(),None,&mut Unprofiled);
    let shared_b=requested.cached_block_tail_profiled(&cache,0,shared_prefix.clone(),
        Some(Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0]],target)),&mut Unprofiled);
    let circuit_check=if codpula {
        let output=|source:&crate::model::DenoisingOutput| crate::model::DenoisingOutput {
            logits:source.logits.clone(),features:source.features.clone()
        };
        let (circuit_a,observed_a)=sbd::prepare_distribution(&requested,output(&shared_a),&soft,1.0)?;
        let (circuit_b,observed_b)=sbd::prepare_distribution(&requested,output(&shared_b),&soft,1.0)?;
        let clean=Tensor::<2,Int>::from_data([[3i32,19,41,7]],target);
        let targets=Tensor::<2,Bool>::from_data([[true,false,true,false]],target);
        let full_a=circuit_a.conditional_nll(clean.clone(),observed_a.clone(),1.0)?;
        let fresh_b=circuit_b.event_nll(clean.clone(),observed_b.clone(),targets.clone(),1.0)?;
        let stale_b=circuit_a.event_nll(clean.clone(),observed_a,targets,1.0)?;
        let fresh_loss=model::scalar(fresh_b.loss)?;
        let stale_loss=model::scalar(stale_b.loss)?;
        ensure!((fresh_loss-stale_loss).abs()>1e-8,"fresh marked circuit did not discriminate from stale A circuit");
        Ok::<Value,anyhow::Error>(json!({"passed":true,"fresh_marked_subset_event_loss":fresh_loss,
            "stale_circuit_subset_event_loss":stale_loss,"unmarked_full_event_loss":model::scalar(full_a.loss)?,
            "fresh_circuit_discriminates":true,"unselected_unresolved_positions_marginalized":true}))
    } else { Ok(json!({"passed":true,"not_applicable":true})) }?;
    let shared_reuse=json!({
        "unmarked":compare(&values(shared_a.logits),&no_mark_logits)?,
        "marked":compare(&values(shared_b.logits),&marked_logits)?,
        "prefix_reuse_is_immutable":compare(&values(requested.cached_block_tail_profiled(&cache,0,shared_prefix,None,&mut Unprofiled).logits),&no_mark_logits)?
    });
    let batch_capture = {
        let mut trace = CleanInputs::default();
        let _ = requested.forward_profiled(&crate::sbd::prefix_batch(&sequence[..8],target),&mut trace);
        trace
    };
    let starts = [(0,2),(0,6)];
    let batch_cache = requested.training_cache_batched(&batch_capture,&starts);
    let batch_soft = SoftState::new(cfg.vocab(),cfg.sbd.as_ref().unwrap().candidate_width,
        vec![None;8],vec![true;8],target,cfg.soft_selection_backend)?;
    let batch_prefix = requested.cached_blocks_prefix_profiled(&batch_cache,&batch_soft,true,&mut Unprofiled);
    let batched_marks = Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0],[0.0,1.0,0.0,1.0]],target);
    let separate_model={
        let mut model=requested.clone();
        model.cocommit_config.as_mut().unwrap().execution=crate::config::CocommitExecution::Separate;
        model
    };
    let (batch_a,batch_b)=requested.cached_blocks_pair_profiled(&batch_cache,batch_prefix.clone(),batched_marks.clone(),&mut Unprofiled);
    let (separate_a,separate_b)=separate_model.cached_blocks_pair_profiled(&batch_cache,batch_prefix.clone(),batched_marks.clone(),&mut Unprofiled);
    let layout_execution=json!({"doubled_rows":4,"separate_rows_per_call":2,"separate_calls_per_pair":2,
        "A":compare(&values(separate_a.logits),&values(batch_a.logits.clone()))?,
        "B":compare(&values(separate_b.logits),&values(batch_b.logits.clone()))?});
    let mut per_a=Vec::new(); let mut per_b=Vec::new();
    for (index,&start) in [2usize,6].iter().enumerate() {
        let local_cache=requested.training_cache(&batch_capture,0,&[start]);
        let local_soft=sbd::state(&cfg,vec![None;4],target)?;
        let local_prefix=requested.cached_block_prefix_profiled(&local_cache,0,&local_soft,true,&mut Unprofiled);
        let local_mark=Tensor::<2>::from_data(if index==0 { [[1.0f32,0.0,1.0,0.0]] } else { [[0.0f32,1.0,0.0,1.0]] },target);
        per_a.extend(values(requested.cached_block_tail_profiled(&local_cache,0,local_prefix.clone(),None,&mut Unprofiled).logits));
        per_b.extend(values(requested.cached_block_tail_profiled(&local_cache,0,local_prefix,Some(local_mark),&mut Unprofiled).logits));
    }
    let batch_execution = json!({
        "unmarked":compare(&values(batch_a.logits),&per_a)?,
        "marked":compare(&values(batch_b.logits),&per_b)?,
        "paired_layout_timing":layout_execution
    });
    let mut soft_feedback = sbd::state(&cfg,vec![None;4],target)?;
    let tracked_teacher = requested.cached_block(&cache,0,&soft_feedback,false);
    let frozen_model=requested.valid();
    let frozen_teacher=frozen_model.cached_block(&cache.detached(),0,&soft_feedback,false);
    ensure!(!frozen_teacher.logits.is_require_grad() && !frozen_teacher.features.is_require_grad(),
        "frozen feedback teacher unexpectedly retains gradients");
    let frozen_teacher_equivalence=json!({
        "logits":compare(&values(tracked_teacher.logits.clone()),&values(frozen_teacher.logits.clone()))?,
        "features":compare(&values(tracked_teacher.features.clone()),&values(frozen_teacher.features.clone()))?,
        "requires_grad":false
    });
    let teacher = tracked_teacher.detach();
    soft_feedback.update(teacher.logits.reshape([4,cfg.vocab()]),cfg.soft_temperature,cfg.sbd.as_ref().unwrap().candidate_width,requested.soft_confidence.as_ref().unwrap())?;
    let soft_a = crate::cocommit::reconstruct_output(&requested,prefix,&soft_feedback,false,None,target);
    let soft_b = crate::cocommit::reconstruct_output(&requested,prefix,&soft_feedback,false,Some(Tensor::<2>::zeros([1,4],target)),target);
    let soft_equivalence = compare(&values(soft_a.logits),&values(soft_b.logits)).context("soft-feedback zero marker")?;
    let cpu_soft=sbd::state(&cfg,vec![None;4],&cpu)?;
    let mut cpu_legacy=reference.clone();
    cpu_legacy.cocommit_marker=None;
    let cpu_logits=values(block_forward(&cpu_legacy,prefix,&cpu_soft,true,&cpu).logits);
    let cpu_vs_requested=compare(&legacy_logits,&cpu_logits).context("CPU versus requested-device legacy block")?;
    // A real one-position state must retain the historical causal bypass.
    let singleton = sbd::state(&cfg,vec![None],target)?;
    let singleton_full = block_forward(&requested,prefix,&singleton,true,target);
    ensure!(singleton_full.logits.dims()[1] == 1, "singleton bypass produced incorrect length");
    let disabled_state=sbd::state(&disabled_cfg,vec![None;4],&cpu)?;
    let disabled_logits=values(block_forward(&disabled,prefix,&disabled_state,true,&cpu).logits);
    let disabled_equivalence=compare(&disabled_logits,&cpu_logits).context("CoCommit-disabled model compatibility")?;
    let legacy_singleton=values(block_forward(&legacy_model,prefix,&singleton,true,target).logits);
    let causal_singleton=values(legacy_model.forward_output_profiled(&crate::sbd::prefix_batch(prefix,target),&mut Unprofiled)
        .slice_positions(prefix.len()-1,1).logits);
    let singleton_bypass=compare(&legacy_singleton,&causal_singleton).context("exact L1 causal bypass")?;
    Ok(json!({"passed":true,"ohc_streams":ohc,"device":crate::runtime::identity(target),
        "reconstruction_vs_legacy":baseline,"cpu_vs_requested_device":cpu_vs_requested,
        "zero_marker_a_b":zero_equivalence,"soft_feedback_zero_marker":soft_equivalence,
        "frozen_feedback_teacher":frozen_teacher_equivalence,
        "nonzero_marker_changes_logits":true,"marker_gradient_max_abs":marker_gradient_max,
        "shared_clean_cache_repeat":cache_repeat,"shared_prefix_reuse":shared_reuse,"per_block_vs_batched":batch_execution,
        "combined_fresh_circuit_subset_event":circuit_check,
        "disabled_model_equivalence":disabled_equivalence,"disabled_parameter_initialization_identical":true,
        "singleton_causal_bypass":singleton_bypass}))
}

const ABS_TOL: f64 = 2.0e-5;
const REL_TOL: f64 = 3.0e-4;

fn scalar_marker_reference(streams: &[f64], marker: &[f64], selected: &[bool], d_streams: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let width = marker.len();
    assert_eq!(streams.len(), selected.len() * width);
    assert_eq!(d_streams.len(), streams.len());
    let mut marked = streams.to_vec();
    let mut vjp = vec![0.0f64; width];
    for (position, &is_selected) in selected.iter().enumerate() {
        if is_selected {
            for feature in 0..width {
                let index = position * width + feature;
                marked[index] = streams[index] + marker[feature];
                vjp[feature] += d_streams[index];
            }
        }
    }
    (marked, vjp)
}

fn scalar_affine_tail_reference(prefix: &[f64], marker: &[f64], selected: &[bool], weights: &[f64]) -> Vec<f64> {
    let width = marker.len();
    assert_eq!(prefix.len(), selected.len() * width);
    assert_eq!(weights.len(), width * width);
    let mut output = vec![0.0; prefix.len()];
    for position in 0..selected.len() {
        let mut marked = vec![0.0; width];
        for feature in 0..width {
            let index = position * width + feature;
            marked[feature] = prefix[index] + if selected[position] { marker[feature] } else { 0.0 };
        }
        // Independent plain-f64 affine tail over the saved representation; not a recurrence.
        for out in 0..width {
            let mut value = 0.0;
            for input in 0..width { value += weights[out * width + input] * marked[input]; }
            output[position * width + out] = value;
        }
    }
    output
}

fn scalar_kron_temporary_reference(
    hidden:&[f64],marker:&[f64;2],selected:&[bool],saved:&[f64;4],chain:bool,
)->(Vec<f64>,Vec<f64>,Vec<f64>) {
    let mut output=Vec::new();let mut states=Vec::new();let mut scores=Vec::new();
    let mut previous=*saved;
    for (position,&marked) in selected.iter().enumerate() {
        let h=[hidden[position*2]+if marked {marker[0]}else{0.0},
            hidden[position*2+1]+if marked {marker[1]}else{0.0}];
        let q=h;let k=h.map(|x|x*0.6+0.2);let v=h.map(|x|x*0.4-0.1);
        let alpha=h.map(|x|x*0.05+0.7);
        let erase=h[0]*0.02+0.3;let write=h[1]*0.03+0.2;
        let initial=if chain {previous}else{*saved};
        let mut next=[0.0;4];let mut prediction=0.0;
        for key in 0..2 {for value in 0..2 {
            next[key*2+value]=alpha[key]*initial[key*2+value];
            prediction+=k[key]*next[key*2+value]*v[value];
        }}
        let correction=write-erase*prediction;
        for key in 0..2 {for value in 0..2 {
            next[key*2+value]+=k[key]*correction*v[value];
        }}
        for value in 0..2 {output.push((0..2).map(|key|q[key]*next[key*2+value]).sum());}
        scores.push((k.iter().map(|x|x*x).sum::<f64>()
            *v.iter().map(|x|(correction*x).powi(2)).sum::<f64>()).sqrt());
        states.extend(next);previous=next;
    }
    (output,states,scores)
}

fn kron_temporary_tail_case(device:&Device)->Result<Value> {
    let hidden=[0.25,-0.5,0.75,1.0,-1.25,0.125];
    let saved=[0.25,-0.125,0.375,0.5];
    let selected=[true,false,true];let marker=[0.125,-0.25];
    let hidden_tensor=Tensor::<2>::from_data(TensorData::new(hidden.map(|x|x as f32).to_vec(),[3,2]),device);
    let clean=Tensor::<4>::from_data(TensorData::new(saved.map(|x|x as f32).to_vec(),[1,1,2,2]),device);
    let hidden_before=values(hidden_tensor.clone());let clean_before=values(clean.clone());
    let immutable=||->Result<()> {
        ensure!(values(hidden_tensor.clone()).iter().zip(&hidden_before).all(|(a,b)|a.to_bits()==b.to_bits()),
            "temporary tail overwrote saved A/B hidden representation");
        ensure!(values(clean.clone()).iter().zip(&clean_before).all(|(a,b)|a.to_bits()==b.to_bits()),
            "temporary tail overwrote frozen clean recurrent snapshot");
        Ok(())
    };
    let run=|marker:[f64;2]| {
        let marks=Tensor::<2>::from_data([[1.0f32],[0.0],[1.0]],device);
        let marker=Tensor::<2>::from_data([marker.map(|x|x as f32)],device);
        let h=hidden_tensor.clone()+marks.matmul(marker);
        let q=h.clone().reshape([3,1,1,2]);
        let input=delta_mixers::chunk::ScanInput {
            q:q.clone(),k:q.clone()*0.6+0.2,v:q.clone()*0.4-0.1,alpha:q.clone()*0.05+0.7,
            erase:q.clone().slice([0..3,0..1,0..1,0..1])*0.02+0.3,
            write:q.slice([0..3,0..1,0..1,1..2])*0.03+0.2,
        };
        let initial=clean.clone().expand([3,1,2,2]);
        let initial_before=values(initial.clone());
        let result=delta_mixers::chunk::scan_selected(delta_mixers::ScanBackend::TensorSolver,
            delta_mixers::MixerKind::KronLstm,input,Tensor::ones([3,1],device),Tensor::zeros([3,1],device),
            1,Some(initial.clone()));
        ensure!(values(initial).iter().zip(&initial_before).all(|(a,b)|a.to_bits()==b.to_bits()),
            "temporary scan overwrote duplicated initial snapshots");
        Ok::<_,anyhow::Error>(result)
    };
    let expected_a=scalar_kron_temporary_reference(&hidden,&[0.0;2],&selected,&saved,false);
    let expected_b=scalar_kron_temporary_reference(&hidden,&marker,&selected,&saved,false);
    let a=run([0.0;2])?;immutable()?;
    let b=run(marker)?;immutable()?;
    let report=|actual:delta_mixers::chunk::ScanOutput,expected:&(Vec<f64>,Vec<f64>,Vec<f64>)|->Result<Value>{
        Ok(json!({"output":compare_f64(&values(actual.output),&expected.0)?,
            "every_temporary_state":compare_f64(&values(actual.final_state),&expected.1)?,
            "edit_scores":compare_f64(&values(actual.scores),&expected.2)?}))
    };
    let wrong=scalar_kron_temporary_reference(&hidden,&marker,&selected,&saved,true);
    let chain_gap=wrong.0.iter().zip(&expected_b.0).map(|(a,b)|(a-b).abs()).fold(0.0f64,f64::max);
    let marker_gap=expected_a.0.iter().zip(&expected_b.0).map(|(a,b)|(a-b).abs()).fold(0.0f64,f64::max);
    ensure!(chain_gap>1.0e-3,"Kron tail fixture does not distinguish incorrect sequential chaining");
    ensure!(marker_gap>1.0e-3,"Kron tail fixture does not exercise a nonzero marker effect");
    Ok(json!({"passed":true,"device":crate::runtime::identity(device),"backend":"tensor_solver",
        "reference":"independent f64 Kron decay, scalar prediction/erase/write, outer-product edit and query read",
        "shape":{"active_positions":3,"heads":1,"keys":2,"values":2,"time_per_temporary_scan":1},
        "A":report(a,&expected_a)?,"B":report(b,&expected_b)?,
        "A_B_saved_hidden_and_clean_state_bitwise_unchanged":true,"duplicated_initial_snapshots_bitwise_unchanged":true,
        "same_frozen_state_for_every_position":true,"nonzero_marker_output_gap":marker_gap,
        "incorrect_sequential_chaining_output_gap":chain_gap}))
}

fn compare_f64(actual: &[f32], reference: &[f64]) -> Result<Value> {
    ensure!(actual.len() == reference.len(), "CoCommit reference length mismatch");
    let mut max_abs = 0.0f64;
    let mut max_ratio = 0.0f64;
    for (&a, &r) in actual.iter().zip(reference) {
        let error = (f64::from(a) - r).abs();
        let tolerance = ABS_TOL + REL_TOL * r.abs();
        max_abs = max_abs.max(error);
        max_ratio = max_ratio.max(error / tolerance);
    }
    ensure!(max_ratio <= 1.0, "CoCommit comparison exceeded tolerance: max_abs={max_abs}, ratio={max_ratio}");
    Ok(json!({"passed":true,"max_abs":max_abs,"maximum_tolerance_ratio":max_ratio,"absolute_tolerance":ABS_TOL,"relative_tolerance":REL_TOL}))
}

/// Called by the ordinary experiment CLI; no second test executable or trained comparison arm.
pub fn run(c: &Config, device: &Device) -> Result<Value> {
    ensure!(matches!(c.backend.as_str(),"cpu"|"vulkan"),"CoCommit checks support CPU or Vulkan");
    let streams = [0.25, -0.5, 0.75, 1.0, -1.25, 0.125, 0.625, -0.25];
    let marker = [0.0, 0.0, 0.0, 0.0];
    let selected = [true, false];
    let cotangent = [0.3, -0.2, 0.4, 0.1, -0.7, 0.8, 0.2, -0.5];
    let (marked, marker_vjp) = scalar_marker_reference(&streams,&marker,&selected,&cotangent);
    ensure!(marked==streams,"zero marker changed the saved representation");
    ensure!(marker_vjp==cotangent[..4],"independent marker entries have incorrect VJP");
    let fixture_device = device.clone().autodiff();
    let marker_tensor = Tensor::<1>::from_data(marker.map(|x|x as f32),&fixture_device).require_grad();
    let stream_tensor = Tensor::<2>::from_data(TensorData::new(streams.map(|x|x as f32).to_vec(),[2,4]),&fixture_device);
    let selection = Tensor::<2>::from_data([[1.0f32],[0.0]],&fixture_device).expand([2,4]);
    let marked_tensor = stream_tensor + marker_tensor.clone().unsqueeze_dim::<2>(0)*selection;
    let marker_forward = compare_f64(&values(marked_tensor.clone()),&marked)?;
    let cotangent_tensor=Tensor::<2>::from_data(TensorData::new(cotangent.map(|x|x as f32).to_vec(),[2,4]),&fixture_device);
    let gradients = (marked_tensor.clone()*cotangent_tensor).sum().backward();
    let marker_backward = compare_f64(&values(marker_tensor.grad(&gradients).context("independent stream marker VJP")?),&marker_vjp)?;
    let weights = [0.1, -0.2, 0.05, 0.3, -0.15, 0.2, 0.25, -0.1, 0.4, 0.12, -0.3, 0.08, 0.2, -0.05, 0.16, 0.11];
    let affine = scalar_affine_tail_reference(&streams,&weights[..4],&selected,&weights);
    ensure!(affine.iter().all(|x|x.is_finite()),"non-finite scalar affine-tail reference");
    let tail_marker=[0.15,-0.1,0.05,0.2];
    let expected_tail=scalar_affine_tail_reference(&streams,&tail_marker,&selected,&weights);
    let matrix=Tensor::<2>::from_data(TensorData::new(weights.iter().map(|&x|x as f32).collect::<Vec<_>>(),[4,4]),&fixture_device);
    let tail_input=Tensor::<2>::from_data(TensorData::new(streams.iter().map(|&x|x as f32).collect::<Vec<_>>(),[2,4]),&fixture_device);
    let tail_marks=Tensor::<2>::from_data([[1.0f32],[0.0]],&fixture_device).expand([2,4]);
    let tail_marker_tensor=Tensor::<1>::from_data(tail_marker.map(|x|x as f32),&fixture_device);
    let tail_actual=tail_input+tail_marker_tensor.unsqueeze_dim::<2>(0)*tail_marks;
    let tail_actual=tail_actual.matmul(matrix.transpose());
    let tail_reference=compare_f64(&values(tail_actual),&expected_tail)?;
    let kron_tail=json!({"cpu":kron_temporary_tail_case(&Device::ndarray())?,
        "requested_device":kron_temporary_tail_case(device)?});
    let cpu = model_case(c,&Device::ndarray(),true,false).context("CPU oHC CoCommit checks")?;
    let ordinary = model_case(c,device,false,false).context("requested-device ordinary stream CoCommit checks")?;
    let ohc = model_case(c,device,true,false).context("requested-device oHC CoCommit checks")?;
    let circuit = model_case(c,device,true,true).context("fresh Codpula branch-circuit checks")?;
    let contextual_profiles = if c.backend=="vulkan" {
        Some(json!({"without_moba_separate":retrieval_case(device,false,false).context("D256 contextual CoCommit without MoBA, Separate")?,
            "without_moba_doubled":retrieval_case(device,false,true).context("D256 contextual CoCommit without MoBA, Doubled")?,
            "layer3_moba_separate":retrieval_case(device,true,false).context("D256 contextual CoCommit layer-3 MoBA, Separate")?}))
    } else {None};
    let implicit=serde_json::to_value(CocommitConfig::default())?;
    let mut explicit_config=CocommitConfig::default();
    explicit_config.execution=crate::config::CocommitExecution::Separate;
    let explicit=serde_json::to_value(explicit_config)?;
    ensure!(implicit==explicit && !implicit.as_object().unwrap().contains_key("execution"),
        "omitted separate execution selector changed CoCommit serialization");
    let mut fingerprint_config=Config::default();
    fingerprint_config.sbd=Some(SbdConfig{cocommit:Some(CocommitConfig::default()),..SbdConfig::default()});
    let implicit_fingerprint=fingerprint_config.pilot_fingerprint()?;
    fingerprint_config.sbd.as_mut().unwrap().cocommit.as_mut().unwrap().execution=crate::config::CocommitExecution::Separate;
    let explicit_fingerprint=fingerprint_config.pilot_fingerprint()?;
    ensure!(implicit_fingerprint==explicit_fingerprint,"omitted separate execution selector changed config fingerprint");
    let checkpoint=cocommit_checkpoint_continuation(c,device,c.seed).context("CoCommit production optimizer marker checkpoint continuation")?;
    let clipping=crate::update::clipping_checks(device)?;
    let membership=delta_mixers::checks::membership_schedule_checks(device).map_err(anyhow::Error::msg)?;
    let omitted_selector=omitted_selector_check()?;
    let training_cases=training_equivalence(c,device)?;
    let row_accumulation=row_accumulation_equivalence(device).context("full-batch versus exact row-accumulated SBD and CoCommit")?;
    Ok(json!({"passed":true,"command":"cocommit-reference","device":crate::runtime::identity(device),
        "precision":"float32","absolute_tolerance":ABS_TOL,"relative_tolerance":REL_TOL,
        "reference_fixtures":{"independent_ohc_stream_marker_forward_and_vjp":{"passed":true,"forward":marker_forward,"vjp":marker_backward,"positions":2,"streams":4,"stream_width":1,"flattened_hidden_width":4},
            "plain_f64_affine_tail":{"passed":true,"report":tail_reference,"positions":2,"width":4,"finite_values":expected_tail.len()},
            "plain_f64_kron_temporary_tail_recurrence":kron_tail},
        "cpu_ohc":cpu,"requested_device_ordinary_stream":ordinary,"requested_device_ohc":ohc,"fresh_circuit":circuit,
        "contextual_D256_CPU_Vulkan_profiles":contextual_profiles,"checkpoint_continuation":checkpoint,
        "production_clipping":clipping,"membership_schedule":membership,
        "omitted_execution_selector":{"serialization_identical":true,"pilot_fingerprint_identical":true},
        "omitted_cocommit_selector":omitted_selector,"full_training_equivalence":training_cases,
        "full_vs_row_accumulation":row_accumulation,
        "coverage":{"cached_and_reconstructed_tail":true,"zero_marker_a_b_equivalence":true,
            "nonzero_marker_effect":true,"length_one_causal_bypass":true,"soft_hard_feedback":true,
            "per_block_vs_batched":true,"contextual_D256_CPU_Vulkan_profiles":contextual_profiles.is_some(),"contextual_event_and_circuit_gradients":contextual_profiles.is_some(),
            "full_vs_row_accumulation_all_four_objectives":true,
            "checkpoint_marker_and_adamw_continuation":true,"disabled_initialization_and_identity":"disabled-model numerical equivalence exercised"}}))
}

fn requested_scan(model:&mut LanguageModel,device:&Device) {
    #[cfg(any(feature="vulkan",feature="cuda"))]
    if matches!(device.clone().inner().as_dispatch(),burn::backend::DispatchDevice::Cube(_)) {
        model.set_scan_backend(delta_mixers::ScanBackend::FusedGpu);
        model.set_hola_backend(delta_mixers::HolaBackend::FusedGpu);
        model.set_noisy_read_backend(delta_mixers::sbd::NoisyReadBackend::FusedGpu);
    }
    let _=(model,device);
}

fn omitted_selector_check()->Result<Value> {
    let mut cfg=Config::default();
    cfg.sbd=Some(SbdConfig::default());
    let encoded=serde_json::to_value(&cfg)?;
    ensure!(encoded["sbd"].get("cocommit").is_none(),"omitted CoCommit selector is serialized");
    let mut explicit=encoded.clone();
    explicit["sbd"]["cocommit"]=Value::Null;
    let explicit:Config=serde_json::from_value(explicit)?;
    ensure!(serde_json::to_value(&explicit)?==encoded,"explicit null CoCommit changed historical serialization");
    let tokens=2048;
    let historical=crate::data::hash(&serde_json::to_vec(&(&cfg,tokens,"sbd-recurrence-only-v1"))?);
    ensure!(cfg.training_fingerprint(tokens)?==historical && explicit.training_fingerprint(tokens)?==historical,
        "omitted CoCommit changed historical SBD training fingerprint");
    let historical_pilot=json!({"config":cfg,"precision":"float32","training_algorithm":cfg.scan_backend.training_algorithm(),
        "optimizer":"AdamW","sbd_model_revision":"sbd-recurrence-only-v1"});
    let pilot_hash=crate::data::hash(&serde_json::to_vec(&historical_pilot)?);
    ensure!(cfg.pilot_fingerprint()?==pilot_hash && explicit.pilot_fingerprint()?==pilot_hash,
        "omitted CoCommit changed historical SBD pilot fingerprint");
    let mut causal=cfg.clone();causal.sbd=None;
    let causal_hash=crate::data::hash(&serde_json::to_vec(&(&causal,tokens))?);
    ensure!(causal.training_fingerprint(tokens)?==causal_hash,"CoCommit changed historical causal fingerprint");
    let mut enabled=cfg.clone();enabled.sbd.as_mut().unwrap().cocommit=Some(CocommitConfig::default());
    ensure!(enabled.training_fingerprint(tokens)?!=historical && enabled.pilot_fingerprint()?!=pilot_hash,
        "enabled CoCommit has no distinct model identity");
    Ok(json!({"passed":true,"serialization_identical":true,"explicit_null_identical":true,
        "historical_sbd_training_sha256":historical,"historical_sbd_pilot_sha256":pilot_hash,
        "historical_causal_training_sha256":causal_hash,"enabled_identity_distinct":true}))
}

fn training_equivalence(c:&Config,device:&Device)->Result<Value> {
    let cpu=Device::ndarray().autodiff();
    let target=device.clone().autodiff();
    let mut reports=Vec::new();
    for joint in [false,true] { for feedback in [SbdFeedback::HardMask,SbdFeedback::SoftFeedback] {
        let mut cfg=Config::default();
        cfg.task="lm".into();cfg.backend="cpu".into();cfg.budget_file=None;
        cfg.d_model=16;cfg.layers=4;cfg.heads=4;cfg.key_dim=32;cfg.value_dim=64;
        cfg.context=20;cfg.batch_size=2;cfg.chunk_size=16;cfg.cache_capacity=4;
        cfg.association_counts=vec![1];
        cfg.ohc_streams=true;cfg.ohc_backend=delta_mixers::ohc::OhcBackend::Tensor;
        cfg.wavelet_layers=vec![0,1,2];cfg.lngram_layers=vec![0,1,2];
        cfg.lngram_routes=cfg.d_model;cfg.lngram_bits=1;cfg.lngram_memory_dim=2;
        cfg.lngram_orders=vec![2,3];
        cfg.soft_training_probability=1.0;
        cfg.sbd=Some(SbdConfig{block_lengths:vec![4],training_block_lengths:vec![4],feedback,
            block_execution:crate::config::SbdBlockExecution::Batched,in_block_reads:InBlockReads::HolaRead,
            joint_output:if joint {SbdJointOutput::Codpula}else{SbdJointOutput::Factorized},
            cocommit:Some(CocommitConfig{execution:crate::config::CocommitExecution::Doubled,..CocommitConfig::default()}),..SbdConfig::default()});
        let arm=Arm{name:"cocommit-training-gate".into(),mixer:"kronlstm".into(),cache:"surprise".into()};
        cfg.arms=vec![arm.clone()];cfg.validate()?;
        let examples=(0..2).map(|row| {
            let values=std::iter::once(256).chain((0..20).map(|j|((j*37+row*13+5)%256) as i32)).collect::<Vec<_>>();
            Example{input:values[..20].to_vec(),target:values[1..].to_vec(),weight:vec![1.0;20],groups:vec![None;20]}
        }).collect::<Vec<_>>();
        cpu.seed(1907);
        let mut reference=LanguageModel::new(&cfg,&arm,&cpu)?;
        let marker=Tensor::from_data(TensorData::new((0..64).map(|i|(i as f32-31.5)*0.0007).collect::<Vec<_>>(),[4,16]),&cpu).require_grad();
        reference.cocommit_marker=Some(reference.cocommit_marker.take().unwrap().map(|_|marker));
        let mut actual=reference.clone().fork(&target);requested_scan(&mut actual,&target);
        let collect=|model:&LanguageModel,config:&Config,device:&Device|->Result<(f32,Vec<(Vec<usize>,Vec<f32>)>)> {
            let batch=Batch::new(&examples,device);
            let result=sbd::training_loss(model,config,&examples,&batch,41)?;
            let loss=model::scalar(result.loss.clone())? as f32;
            let (gradients,_) = model::checked_gradients_with_unused(model,result.loss,&result.unused_parameters,&mut Unprofiled)?;
            Ok((loss,parameters_with_unused(model,Some(&gradients),&result.unused_parameters)?))
        };
        let (cpu_loss,cpu_gradients)=collect(&reference,&cfg,&cpu)?;
        let mut individual=cfg.clone();individual.sbd.as_mut().unwrap().block_execution=crate::config::SbdBlockExecution::PerBlock;
        let (individual_loss,individual_gradients)=collect(&reference,&individual,&cpu)?;
        let per_block=json!({"loss":compare(&[individual_loss],&[cpu_loss])?,
            "all_gradients":compare_parameter_gradients(&individual_gradients,&cpu_gradients)?});
        let (actual_loss,actual_gradients)=collect(&actual,&cfg,&target)?;
        let requested=json!({"loss":compare(&[actual_loss],&[cpu_loss])?,
            "all_gradients":compare_parameter_gradients(&actual_gradients,&cpu_gradients)?});
        let mut separate=actual.clone();separate.cocommit_config.as_mut().unwrap().execution=crate::config::CocommitExecution::Separate;
        let (separate_loss,separate_gradients)=collect(&separate,&cfg,&target)?;
        let layouts=json!({"loss":compare(&[separate_loss],&[actual_loss])?,
            "all_gradients":compare_parameter_gradients(&separate_gradients,&actual_gradients)?});
        let visibility=if !joint && feedback==SbdFeedback::HardMask {
            Some(json!({"cpu":visibility_case(&reference,&cfg,&examples[0],&cpu)?,
                "requested":visibility_case(&actual,&cfg,&examples[0],&target)?}))
        } else {None};
        reports.push(json!({"passed":true,"joint_output":joint,"feedback":feedback,"per_block_vs_batched":per_block,
            "cpu_vs_requested":requested,"doubled_vs_separate":layouts,"seed":41,"shape":{"batch":2,"context":20,"block":4,"layers":4,"width":16},
            "visibility_padding_and_communication":visibility,
            "includes_clean_loss_both_student_branches_and_all_parameter_gradients":true}));
    }}
    Ok(json!({"passed":true,"cases":reports,"device":crate::runtime::identity(device),"config_seed":c.seed}))
}

fn compare_row_backward(
    actual_model:&LanguageModel,actual:&crate::sbd_step::Backward,
    reference_model:&LanguageModel,reference:&crate::sbd_step::Backward,
)->Result<Value> {
    ensure!(actual.loss.is_finite() && reference.loss.is_finite(),"non-finite row-accumulation loss");
    let error=(actual.loss-reference.loss).abs();
    let tolerance=ABS_TOL+REL_TOL*reference.loss.abs();
    ensure!(error<=tolerance,"row-accumulation scalar loss exceeded tolerance: error={error}, tolerance={tolerance}");
    let gradients=compare_parameter_gradients(
        &parameters(actual_model,Some(&actual.gradients))?,
        &parameters(reference_model,Some(&reference.gradients))?,
    ).context("every row-accumulation parameter gradient")?;
    let flows=match (&actual.circuit_flows,&reference.circuit_flows) {
        (None,None)=>json!({"passed":true,"present":false}),
        (Some(actual),Some(reference))=>{
            ensure!(actual.active_blocks==reference.active_blocks,"row-accumulation circuit active_blocks differs");
            let mut tensors=serde_json::Map::new();
            macro_rules! tensor {
                ($field:ident)=>{
                    ensure!(actual.$field.dims()==reference.$field.dims(),"circuit flow {} shape differs",stringify!($field));
                    tensors.insert(stringify!($field).into(),
                        compare(&values(actual.$field.clone()),&values(reference.$field.clone()))
                            .with_context(||format!("row-accumulation circuit flow {}",stringify!($field)))?);
                };
            }
            tensor!(emission_positive);tensor!(emission_negative);
            tensor!(transition_positive);tensor!(transition_negative);
            tensor!(root_positive);tensor!(root_negative);
            tensor!(emission_reach);tensor!(transition_reach);tensor!(root_reach);
            json!({"passed":true,"present":true,"active_blocks":actual.active_blocks,"tensors":tensors})
        }
        _=>anyhow::bail!("row-accumulation circuit flow presence differs"),
    };
    Ok(json!({"passed":true,"loss":{"actual":actual.loss,"reference":reference.loss,"max_abs":error,
        "maximum_tolerance_ratio":error/tolerance},"all_parameter_gradients":gradients,"all_circuit_flows":flows,
        "actual_passes":actual.passes,"reference_passes":reference.passes,
        "actual_rows_per_pass":actual.rows_per_pass,"reference_rows_per_pass":reference.rows_per_pass}))
}

fn row_accumulation_equivalence(device:&Device)->Result<Value> {
    let cpu=Device::ndarray().autodiff();
    let target=device.clone().autodiff();
    let seed=41;
    let examples=(0..2).map(|row| {
        let tokens=std::iter::once(256).chain((0..20).map(|position|((position*37+row*13+5)%256) as i32)).collect::<Vec<_>>();
        Example{input:tokens[..20].to_vec(),target:tokens[1..].to_vec(),weight:vec![1.0;20],groups:vec![None;20]}
    }).collect::<Vec<_>>();
    let mut reports=Vec::new();
    for cocommit in [false,true] {for joint in [SbdJointOutput::Factorized,SbdJointOutput::Codpula] {
        let mut cfg=Config::default();
        cfg.task="lm".into();cfg.backend="cpu".into();cfg.budget_file=None;
        cfg.d_model=16;cfg.layers=4;cfg.heads=4;cfg.key_dim=32;cfg.value_dim=64;
        cfg.context=20;cfg.batch_size=2;cfg.chunk_size=16;cfg.cache_capacity=4;
        cfg.association_counts=vec![1];cfg.ohc_streams=true;
        cfg.sbd=Some(SbdConfig{block_lengths:vec![4],training_block_lengths:vec![4],feedback:SbdFeedback::SoftFeedback,
            block_execution:crate::config::SbdBlockExecution::Batched,in_block_reads:InBlockReads::HolaRead,
            joint_output:joint,cocommit:cocommit.then(CocommitConfig::default),rows_per_pass:None,
            ..SbdConfig::default()});
        let arm=Arm{name:"sbd-row-accumulation-gate".into(),mixer:"kronlstm".into(),cache:"surprise".into()};
        cfg.arms=vec![arm.clone()];cfg.validate()?;
        let mut chunked=cfg.clone();chunked.sbd.as_mut().unwrap().rows_per_pass=Some(1);chunked.validate()?;
        // The oracle draws on the complete batch, never restarts either RNG at a row boundary.
        let blocks=sbd::sample_training_blocks(&cfg,&examples,seed)?;
        ensure!(blocks==sbd::sample_training_blocks(&chunked,&examples,seed)?,"row selector changed full-batch masks");
        ensure!((0..2).all(|row|blocks.iter().any(|block|block.row==row)),"row fixture must sample active blocks in both rows");
        let maximum=cfg.sbd.as_ref().unwrap().cocommit.as_ref().map(|options|options.training_bundle_max);
        let bundles=maximum.map(|maximum|crate::cocommit::sample_bundles(&blocks,maximum,seed));
        if let Some(maximum)=maximum {
            let chunk_blocks=sbd::sample_training_blocks(&chunked,&examples,seed)?;
            ensure!(bundles.as_ref().unwrap()==&crate::cocommit::sample_bundles(&chunk_blocks,maximum,seed),
                "row selector changed full-batch CoCommit bundles");
        }
        let mut reconstructed=Vec::new();
        let mut reconstructed_bundles=Vec::new();
        let mut partition_metadata=Vec::new();
        for row_start in 0..examples.len() {
            for (index,block) in blocks.iter().enumerate().filter(|(_,block)|block.row==row_start) {
                let mut local=block.clone();local.row-=row_start;
                partition_metadata.push(json!({"original_block_index":index,"original_row":block.row,
                    "local_row":local.row,"start":local.start,"length":local.length,"masked":local.masked,
                    "bundle":bundles.as_ref().map(|bundles|&bundles[index])}));
                local.row+=row_start;reconstructed.push(local);
                if let Some(bundles)=&bundles {reconstructed_bundles.push(bundles[index].clone());}
            }
        }
        ensure!(reconstructed==blocks,"partitioning changed full sampled masks or block order");
        if let Some(bundles)=&bundles {ensure!(&reconstructed_bundles==bundles,"partitioning changed bundle-to-block alignment");}
        cpu.seed(1907);
        let mut reference=LanguageModel::new(&cfg,&arm,&cpu)?;
        if cocommit {
            let marker=Tensor::from_data(TensorData::new((0..64).map(|i|(i as f32-31.5)*0.0007).collect::<Vec<_>>(),[4,16]),&cpu).require_grad();
            reference.cocommit_marker=Some(reference.cocommit_marker.take().unwrap().map(|_|marker));
        }
        let mut actual=reference.clone().fork(&target);requested_scan(&mut actual,&target);
        let cpu_batch=Batch::new(&examples,&cpu);
        let target_batch=Batch::new(&examples,&target);
        let sampled=sbd::training_loss_sampled_profiled(&reference,&cfg,&examples,&cpu_batch,
            &blocks,bundles.as_deref(),&mut Unprofiled)?;
        let sampled_loss=model::scalar(sampled.loss.clone())?;
        let (sampled_gradients, gradient_report) = model::checked_gradients_with_unused(&reference,sampled.loss,&sampled.unused_parameters,&mut Unprofiled)?;
        let sampled=crate::sbd_step::Backward{loss:sampled_loss,
            gradients:sampled_gradients,circuit_flows:sampled.circuit_flows,
            passes:1,rows_per_pass:examples.len(),gradient_report};
        let full=crate::sbd_step::backward(&reference,&cfg,&examples,&cpu_batch,seed)?;
        let split=crate::sbd_step::backward(&reference,&chunked,&examples,&cpu_batch,seed)?;
        ensure!(full.passes==1 && full.rows_per_pass==2,"omitted row selector did not retain a full-batch pass");
        ensure!(split.passes==2 && split.rows_per_pass==1,"explicit single-row selector did not run both passes");
        ensure!(full.circuit_flows.is_some()==(joint==SbdJointOutput::Codpula),"fixture circuit flow presence differs from joint output");
        let sampled_oracle=json!({"full":compare_row_backward(&reference,&full,&reference,&sampled)?,
            "chunked":compare_row_backward(&reference,&split,&reference,&sampled)?,
            "same_full_batch_masks_and_bundles":true,"partitioned_blocks":partition_metadata});
        let cpu_report=compare_row_backward(&reference,&split,&reference,&full)?;
        let requested_full=crate::sbd_step::backward(&actual,&cfg,&examples,&target_batch,seed)?;
        let requested_split=crate::sbd_step::backward(&actual,&chunked,&examples,&target_batch,seed)?;
        ensure!(requested_full.passes==1 && requested_split.passes==2 && requested_split.rows_per_pass==1,
            "requested-device row pass metadata differs");
        let requested=compare_row_backward(&actual,&requested_split,&actual,&requested_full)?;
        let cross_device=json!({"full":compare_row_backward(&actual,&requested_full,&reference,&full)?,
            "chunked":compare_row_backward(&actual,&requested_split,&reference,&split)?});
        reports.push(json!({"passed":true,"cocommit":cocommit,"joint_output":joint,"feedback":SbdFeedback::SoftFeedback,
            "seed":seed,"model_seed":1907,"shape":{"batch":2,"context":20,"width":16,"layers":4,"block":4},
            "ohc_streams":true,"in_block_reads":InBlockReads::HolaRead,"cpu":cpu_report,
            "requested_device":requested,"cpu_vs_requested":cross_device,"direct_full_sampled_oracle":sampled_oracle}));
    }}
    Ok(json!({"passed":true,"cases":reports,"device":crate::runtime::identity(device),
        "absolute_tolerance":ABS_TOL,"relative_tolerance":REL_TOL,"loss_normalization":"full B2/T20",
        "sampled_once_before_partitioning":true,"circuit_flows":"all nine tensors and exact active_blocks; sums are unscaled"}))
}

fn visibility_case(model:&LanguageModel,c:&Config,example:&Example,device:&Device)->Result<Value> {
    let start=19;
    let make_cache=|input:Vec<i32>| {
        let mut trace=CleanInputs::default();
        let _=model.forward_profiled(&sbd::prefix_batch(&input,device),&mut trace);
        model.training_cache(&trace,0,&[start])
    };
    let cache=make_cache(example.input.clone());
    let mut changed=example.input.clone();
    for value in &mut changed[start..] {*value=(*value+97)%256;}
    let changed_cache=make_cache(changed);
    let evidence=example.target[19] as usize;
    let soft=sbd::state(c,vec![None,None,None,Some(evidence)],device)?;
    let marks=Tensor::<2>::from_data([[1.0f32,0.0,1.0,1.0]],device);
    let run=|cache:&crate::sbd_training::TrainingCache,soft:&SoftState,marks:Tensor<2>| {
        let prefix=model.cached_block_prefix_profiled(cache,0,soft,false,&mut Unprofiled);
        model.cached_block_tail_profiled(cache,0,prefix,Some(marks),&mut Unprofiled)
    };
    let original=run(&cache,&soft,marks.clone());
    let future=run(&changed_cache,&soft,marks.clone());
    let leakage=compare(&values(future.logits),&values(original.logits.clone()))?;
    let without_observed_marker=run(&cache,&soft,Tensor::<2>::from_data([[1.0f32,0.0,1.0,0.0]],device));
    let observed_marker=compare(&values(without_observed_marker.logits),&values(original.logits.clone()))?;
    let reconstructed=crate::cocommit::reconstruct_output(model,&example.input[..start],&soft,false,Some(marks),device);
    let reconstruction=compare(&values(reconstructed.logits),&values(original.logits.clone()))?;
    let padded=SoftState::new(c.vocab(),c.sbd.as_ref().unwrap().candidate_width,
        vec![None,None,None,Some(evidence),None,None,None,None],
        vec![true,true,true,true,false,false,false,false],device,c.soft_selection_backend)?;
    let padded_output=run(&cache,&padded,Tensor::<2>::ones([1,8],device));
    let corresponding=run(&cache,&soft,Tensor::<2>::ones([1,4],device));
    let padding=compare(&values(padded_output.logits.slice([0..1,0..4,0..c.vocab()])),&values(corresponding.logits))?;
    let changed_observed=sbd::state(c,vec![None,None,None,Some((evidence+1)%256)],device)?;
    let changed_output=run(&cache,&changed_observed,Tensor::<2>::zeros([1,4],device));
    let unchanged_output=run(&cache,&soft,Tensor::<2>::zeros([1,4],device));
    let first=values(unchanged_output.logits.clone().slice([0..1,0..1,0..c.vocab()]));
    let other=values(changed_output.logits.slice([0..1,0..1,0..c.vocab()]));
    ensure!(first.iter().zip(&other).any(|(a,b)|(a-b).abs()>1e-8),"right-hand observed token cannot communicate to a left query");
    let all_masked=sbd::state(c,vec![None;4],device)?;
    let no_mark=run(&cache,&all_masked,Tensor::<2>::zeros([1,4],device));
    let last_mark=run(&cache,&all_masked,Tensor::<2>::from_data([[0.0f32,0.0,0.0,1.0]],device));
    ensure!(values(no_mark.logits.slice([0..1,0..1,0..c.vocab()])).iter()
        .zip(values(last_mark.logits.slice([0..1,0..1,0..c.vocab()]))).any(|(a,b)|(a-b).abs()>1e-8),
        "right-hand marker cannot communicate to a left query");
    Ok(json!({"passed":true,"prefix":start,"computational_chunk":16,"active_length":4,"padded_length":8,
        "future_clean_input_perturbation":leakage,"observed_marker_excluded":observed_marker,
        "cached_vs_reconstructed":reconstruction,"padding":padding,
        "observed_token_right_to_left":true,"marker_right_to_left":true}))
}


fn ensure_exact_parameter_bits(actual:&[(Vec<usize>,Vec<f32>)],expected:&[(Vec<usize>,Vec<f32>)],context:&str)->Result<()> {
    ensure!(actual.len()==expected.len(),"{context}: parameter count differs");
    for (index,((shape,a),(expected_shape,b))) in actual.iter().zip(expected).enumerate(){
        ensure!(shape==expected_shape && a.len()==b.len() && a.iter().zip(b).all(|(&x,&y)|x.to_bits()==y.to_bits()),
            "{context}: parameter {index} is not bit-identical");
    }
    Ok(())
}
