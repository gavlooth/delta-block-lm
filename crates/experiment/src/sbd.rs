//! SBD recurrent adaptation: clean NTP plus masked future-block prediction.
use crate::{config::{Arm, Config, SbdFeedback}, data::{Batch, Example, Rng}, model::{self, LanguageModel}, runtime, soft::SoftState};
use anyhow::{Context, Result, ensure};
use burn::{module::Module, prelude::*, tensor::activation::log_softmax};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

fn fixture(input: Vec<i32>) -> Example {
    let n = input.len();
    Example { input, target: vec![0;n], weight: vec![1.0;n], groups: vec![None;n] }
}
fn prefix_batch(prefix: &[i32], device: &Device) -> Batch {
    Batch::new(&[fixture(prefix.to_vec())], device)
}
fn state(c: &Config, observed: Vec<Option<usize>>, device: &Device) -> Result<SoftState> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    let n = observed.len();
    SoftState::new(c.vocab(), s.candidate_width, observed, vec![true;n], device, c.soft_selection_backend)
}
/// `prefix` contains BOS and every clean token strictly before the target block.
/// Only explicit `observed` values may reveal targets inside the block.
pub fn block_forward(model: &LanguageModel, prefix: &[i32], soft: &SoftState, hard: bool, device: &Device) -> Tensor<3> {
    assert!(!prefix.is_empty());
    let n = soft.observed.len();
    let p = prefix.len();
    let d = model.embedding.weight.val().dims()[1];
    if n == 1 {
        // Exact causal limiting case: the target is predicted from the last clean input.
        return model.forward(&prefix_batch(prefix, device)).slice([0..1,p-1..p,0..model.head.weight.val().dims()[1]]);
    }
    let mut input = prefix.to_vec();
    input.resize(p+n,0);
    let mut batch = prefix_batch(&input, device);
    let valid = std::iter::repeat_n(1.0f32,p).chain(soft.active.iter().map(|&x| if x {1.0} else {0.0})).collect::<Vec<_>>();
    batch.valid = Tensor::from_data(TensorData::new(valid,[1,p+n]),device);
    let clean = model.embedding.forward(batch.input.clone().slice([0..1,0..p]));
    let noisy = soft.embeddings(model.embedding.weight.val(), model.sbd_mask.as_ref().expect("SBD mask parameter").val(), hard).reshape([1,n,d]);
    model.forward_noisy(Tensor::cat(vec![clean,noisy],1), &batch, p)
        .slice([0..1,p..p+n,0..model.head.weight.val().dims()[1]])
}
fn feedback(model: &LanguageModel, c: &Config, prefix: &[i32], soft: &mut SoftState, device: &Device) -> Result<()> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    if s.feedback == SbdFeedback::SoftFeedback && soft.observed.len() > 1 {
        // The teacher pass is detached. Selected values and the embedding table in
        // the second pass remain differentiable; no clean masked target is used.
        let first = block_forward(model, prefix, soft, false, device).detach();
        soft.update(first.reshape([soft.observed.len(),c.vocab()]), 1.0, c.soft_temperature, c.soft_steps, s.candidate_width)?;
    }
    Ok(())
}
/// Paper equation 14, divided by the number of clean targets (not by mask count).
/// Uniform sequence-level eta, independent Bernoulli masks, uniform length choice.
pub fn training_loss(model: &LanguageModel, c: &Config, rows: &[Example], batch: &Batch, seed: u64) -> Result<Tensor<1>> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    let device = batch.input.device();
    let mut trace = crate::sbd_training::CleanInputs::default();
    let clean_logits = model.forward_profiled(batch,&mut trace);
    let mut loss = model.loss(clean_logits.clone(), batch);
    // In L=1 or an all-observed draw the mask has an exactly zero derivative.
    loss = loss + model.sbd_mask.as_ref().unwrap().val().sum() * 0.0;
    let mut rng = Rng(seed);
    for (row_index,row) in rows.iter().enumerate() {
        let length = s.block_lengths[rng.index(s.block_lengths.len())];
        let eta = (rng.next() as f64 + 0.5) / (u64::MAX as f64 + 1.0);
        let starts = (0..row.target.len()).step_by(length).map(|s|s+1).collect::<Vec<_>>();
        let cache = (length > 1).then(|| model.training_cache(&trace,row_index,&starts));
        for (block_index,start) in (0..row.target.len()).step_by(length).enumerate() {
            let n = length.min(row.target.len()-start);
            let masked: Vec<bool> = (0..n).map(|_| (rng.next() as f64 / u64::MAX as f64) < eta).collect();
            if !masked.iter().any(|&x| x) { continue; }
            let observed = (0..n).map(|j| (!masked[j]).then_some(row.target[start+j] as usize)).collect();
            let mut soft = state(c, observed, &device)?;
            let logits = if n == 1 {
                clean_logits.clone().slice([row_index..row_index+1,start..start+1,0..c.vocab()])
            } else {
                let cache = cache.as_ref().unwrap();
                if s.feedback == SbdFeedback::SoftFeedback {
                    let teacher = model.cached_block(cache,block_index,&soft,false).detach();
                    soft.update(teacher.reshape([n,c.vocab()]),1.0,c.soft_temperature,c.soft_steps,s.candidate_width)?;
                }
                model.cached_block(cache,block_index,&soft,s.feedback == SbdFeedback::HardMask)
            };
            let targets = Tensor::<2,Int>::from_data(TensorData::new(row.target[start..start+n].to_vec(),[n,1]),&device);
            let weights = Tensor::<2>::from_data(TensorData::new((0..n).map(|j| if masked[j] {row.weight[start+j]} else {0.0}).collect::<Vec<_>>(),[n,1]),&device);
            let selected = log_softmax(logits.reshape([n,c.vocab()]),1).gather(1,targets);
            loss = loss - (selected*weights).sum() / (rows.len()*row.target.len()) as f64;
        }
    }
    Ok(loss)
}

pub fn evaluate(model: &LanguageModel, c: &Config, examples: &[Example], device: &Device) -> Result<Value> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    let model = model.valid();
    let mut nll = 0.0;
    let mut hits = 0;
    let mut count = 0;
    for row in examples {
      for &length in &s.block_lengths {
        for start in (0..row.target.len()).step_by(length) {
            let n = length.min(row.target.len()-start);
            let mut soft = state(c,vec![None;n],device)?;
            let prefix = &row.input[..start+1];
            feedback(&model,c,prefix,&mut soft,device)?;
            let logits = block_forward(&model,prefix,&soft,s.feedback == SbdFeedback::HardMask,device);
            let raw = log_softmax(logits,2).into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("SBD evaluation {e:?}"))?;
            for j in 0..n {
                if row.weight[start+j] == 0.0 { continue; }
                let values = &raw[j*c.vocab()..(j+1)*c.vocab()];
                let best = (0..c.vocab()).max_by(|&a,&b| values[a].total_cmp(&values[b]).then(b.cmp(&a))).unwrap();
                nll -= values[row.target[start+j] as usize] as f64;
                hits += usize::from(best == row.target[start+j] as usize);
                count += 1;
            }
        }
      }
    }
    ensure!(count > 0,"SBD evaluation needs targets");
    Ok(json!({"block_nll":nll/count as f64,"block_accuracy":hits as f64/count as f64,"predicted_tokens":count,"conditioning":"all targets masked initially; detached soft first pass when configured; every configured length evaluated on every example","block_lengths":s.block_lengths}))
}

#[derive(Default)]
struct StateTrace(Vec<Tensor<4>>);
impl delta_mixers::profile::Sections for StateTrace {
    const CAPTURE_RECURRENT_STATE: bool = true;
    fn recurrent_state(&mut self, state: &Tensor<4>) { self.0.push(state.clone()); }
    fn section<T>(&mut self, _name: &'static str, run: impl FnOnce()->T) -> T { run() }
}
fn clean_snapshot(model: &LanguageModel, prefix: &[i32], device: &Device) -> (Tensor<3>,Vec<Tensor<4>>) {
    let mut trace = StateTrace::default();
    let logits = model.forward_profiled(&prefix_batch(prefix,device),&mut trace);
    (logits,trace.0)
}

/// Mutable clean history is changed only by `replay`. Denoising uses its immutable
/// token snapshot, reconstructing tensor states without changing this history.
struct CleanPrefix {
    tokens: Vec<i32>,
    logits: Tensor<3>,
    states: Vec<Tensor<4>>,
    replayed: usize,
}
impl CleanPrefix {
    fn new(model: &LanguageModel, tokens: Vec<i32>, device: &Device) -> Self {
        let (logits,states) = clean_snapshot(model,&tokens,device);
        Self { tokens, logits, states, replayed: 0 }
    }
    fn replay(&mut self, model: &LanguageModel, finalized: &[i32], device: &Device) {
        self.tokens.extend_from_slice(finalized);
        // Reference replay reconstructs all states from the clean document; it
        // replaces, rather than appends to, the old snapshot exactly once/block.
        (self.logits,self.states) = clean_snapshot(model,&self.tokens,device);
        self.replayed += finalized.len();
    }
}

pub fn generate(model: &LanguageModel, c: &Config, prompt: &[i32], device: &Device) -> Result<Value> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    ensure!(!prompt.is_empty(),"generation requires a clean prefix");
    let model = model.valid();
    let mut clean = CleanPrefix::new(&model,prompt.to_vec(),device);
    runtime::sync(device)?;
    let clock = Instant::now();
    let mut passes = 0;
    let mut generated = Vec::new();
    let mut blocks = 0;
    while generated.len() < s.generate_tokens {
        let n = s.block_lengths[blocks%s.block_lengths.len()].min(s.generate_tokens-generated.len());
        let mut soft = state(c,vec![None;n],device)?;
        let mut final_values = vec![None;n];
        while final_values.iter().any(Option::is_none) {
            let logits = block_forward(&model,&clean.tokens,&soft,s.feedback == SbdFeedback::HardMask,device);
            passes += 1;
            let logs = log_softmax(logits.clone(),2).into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("SBD probabilities {e:?}"))?;
            let mut order: Vec<(usize,f64,usize)> = (0..n).filter(|&i| final_values[i].is_none()).map(|i| {
                let row = &logs[i*c.vocab()..(i+1)*c.vocab()];
                let best = (0..c.vocab()).max_by(|&a,&b| row[a].total_cmp(&row[b]).then(b.cmp(&a))).unwrap();
                let entropy = row.iter().map(|&x| -(x as f64).exp()*x as f64).sum();
                (i,entropy,best)
            }).collect();
            order.sort_by(|a,b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let take = if let Some(gamma) = s.entropy_threshold {
                // Eq. 8 sums the first s-1 entropies, always revealing >=1.
                let mut take = 1;
                let mut sum = 0.0;
                while take < order.len() && sum+order[take-1].1 <= gamma {
                    sum += order[take-1].1;
                    take += 1;
                }
                take
            } else { s.commit_count.min(order.len()) };
            if s.feedback == SbdFeedback::SoftFeedback {
                soft.update(logits.reshape([n,c.vocab()]).detach(),c.soft_eta,c.soft_temperature,c.soft_steps,s.candidate_width)?;
            }
            for &(i,_,id) in &order[..take] { final_values[i] = Some(id); }
            // Observed finalized positions are immutable. Keep the old distribution
            // on still-mutable positions, rather than hardening its proposals.
            let mut next = state(c,final_values.clone(),device)?;
            let mutable = Tensor::<2>::from_data(TensorData::new(final_values.iter().map(|v| if v.is_none() {1.0f32} else {0.0}).collect::<Vec<_>>(),[n,1]),device);
            let probabilities = next.candidates.dense(c.vocab()) * mutable.clone().neg().add_scalar(1.0)
                + soft.candidates.dense(c.vocab()) * mutable.clone();
            next.candidates = crate::soft::Candidates::select(probabilities,s.candidate_width,c.soft_selection_backend)?;
            next.visibility = next.visibility.clone()*mutable.clone().neg().add_scalar(1.0)+soft.visibility*mutable;
            soft = next;
        }
        let finalized: Vec<i32> = final_values.into_iter().map(|v| v.unwrap() as i32).collect();
        clean.replay(&model,&finalized,device);
        generated.extend_from_slice(&finalized);
        blocks += 1;
    }
    runtime::sync(device)?;
    let seconds = clock.elapsed().as_secs_f64();
    Ok(json!({"tokens":generated,"text":String::from_utf8_lossy(&generated.iter().map(|&x| x as u8).collect::<Vec<_>>()),"finalized_tokens":generated.len(),"finalized_tokens_per_second":generated.len() as f64/seconds,"seconds":seconds,"denoising_passes":passes,"blocks":blocks,"causally_replayed_tokens":clean.replayed,"includes_replay":true,"prefill_included":false,"replay_implementation":"full clean-prefix reconstruction once per finalized block; no incremental optimization","commit_rule":if s.entropy_threshold.is_some() {"sbd_entropy_eq8"} else {"fixed_count"}}))
}

pub fn checkpoint_report(c: &Config, arm: &Arm, seed: u64, device: &Device, checkpoint: &Path, with_generation: bool) -> Result<Value> {
    device.seed(seed);
    let model = crate::checkpoint::load_model(LanguageModel::new(c,arm,device)?,checkpoint.join("model.burnpack"),device)?;
    let corpus = crate::train::loaded_corpus(c)?;
    let examples = crate::train::evaluations(c,corpus.as_ref(),"validation")?;
    let mut report = if with_generation {
        let prompt = &examples[0].input[..examples[0].input.len().min(17)];
        generate(&model,c,prompt,device)?
    } else {json!({})};
    report["block_evaluation"] = evaluate(&model,c,&examples,device)?;
    report["clean_evaluation"] = crate::train::evaluate(&model,c,&examples,device)?;
    Ok(report)
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().expect("SBD float readback")
}
fn identical(a: &[f32], b: &[f32], label: &str) -> Result<()> {
    ensure!(a.len() == b.len() && a.iter().zip(b).all(|(a,b)| a.to_bits() == b.to_bits()), "{label} is not bit-identical");
    Ok(())
}

pub fn reference(c: &Config, device: &Device) -> Result<Value> {
    let attention = crate::sbd_reference::numerical_checks(device)?;
    let mut reports = Vec::new();
    let cpu = Device::ndarray().autodiff();
    let gpu = device.clone().autodiff();
    for mixer in ["gdn2","kronlstm"] {
        for cache in ["none","surprise"] {
            for step7 in [false,true] {
                let mut cfg = c.clone();
                cfg.sbd = Some(crate::config::SbdConfig::default());
                cfg.backend = "cpu".into();
                cfg.d_model = 16;
                cfg.layers = 1;
                cfg.heads = 4;
                cfg.key_dim = 32;
                cfg.value_dim = 64;
                cfg.chunk_size = 16;
                cfg.cache_capacity = 4;
                cfg.scan_backend = delta_mixers::ScanBackend::TensorSolver;
                cfg.hola_backend = delta_mixers::HolaBackend::Tensor;
                cfg.projection_backend = delta_mixers::ProjectionBackend::Separate;
                cfg.normalization_backend = delta_mixers::NormalizationBackend::Tensor;
                cfg.convolution_backend = delta_mixers::ConvolutionBackend::Tensor;
                cfg.head_backend = crate::config::HeadBackend::Native;
                cfg.loss_backend = crate::config::LossBackend::Tensor;
                cfg.update_backend = crate::config::UpdateBackend::Tensor;
                cfg.epilogue_backend = crate::config::EpilogueBackend::Native;
                cfg.ohc_streams = step7;
                cfg.ohc_backend = delta_mixers::ohc::OhcBackend::Tensor;
                cfg.wavelet_layers = if step7 {vec![0]} else {vec![]};
                cfg.wavelet_backend = delta_mixers::WaveletBackend::Tensor;
                cfg.lngram_layers = if step7 {vec![0]} else {vec![]};
                cfg.lngram_backend = delta_mixers::lngram::LngramBackend::Tensor;
                cfg.lngram_routes = 2;
                cfg.lngram_bits = 1;
                cfg.lngram_memory_dim = 2;
                cfg.lngram_orders = vec![2,3];
                cfg.soft_selection_backend = crate::config::SoftSelectionBackend::Cpu;
                let arm = Arm {name: format!("{mixer}-{cache}"),mixer: mixer.into(),cache: cache.into()};
                cpu.seed(72);
                let base = LanguageModel::new(&cfg,&arm,&cpu)?;
                let mut tested = base.clone().fork(&gpu);
                if c.backend != "cpu" {
                    tested.set_scan_backend(if mixer == "gdn2" {delta_mixers::ScanBackend::FusedGdn2} else {delta_mixers::ScanBackend::FusedGpu});
                    tested.set_hola_backend(delta_mixers::HolaBackend::FusedGpu);
                    tested.set_convolution_backend(c.convolution_backend);
                    tested.set_normalization_backend(c.normalization_backend);
                    tested.set_ohc_backend(c.ohc_backend);
                    tested.set_wavelet_backend(c.wavelet_backend);
                    tested.set_lngram_backend(c.lngram_backend);
                }
                let prefix: Vec<i32> = std::iter::once(256).chain((0..18).map(|x| (x*7+11)%256)).collect();
                let observed = vec![None,Some(65),None,None];
                let soft_cpu = state(&cfg,observed.clone(),&cpu)?;
                let soft_gpu = state(&cfg,observed.clone(),&gpu)?;
                let ref_logits = block_forward(&base,&prefix,&soft_cpu,false,&cpu);
                let actual_logits = block_forward(&tested,&prefix,&soft_gpu,false,&gpu);
                let logits_report = crate::sbd_reference::compare(&values(actual_logits.clone()),&values(ref_logits.clone()).iter().map(|&x| x as f64).collect::<Vec<_>>())?;
                let mut full = prefix.clone();
                full.extend([31,32,33,34,35,36]);
                let mut trace = crate::sbd_training::CleanInputs::default();
                let _ = tested.forward_profiled(&prefix_batch(&full,&gpu),&mut trace);
                let snapshot = tested.training_cache(&trace,0,&[1,9,17,19]);
                let cached = tested.cached_block(&snapshot,3,&soft_gpu,false);
                let cached_report = crate::sbd_reference::compare(&values(cached.clone()),&values(actual_logits.clone()).iter().map(|&x|x as f64).collect::<Vec<_>>())?;
                let objective = |model: &LanguageModel, logits: Tensor<3>| logits.square().mean() + model.sbd_mask.as_ref().unwrap().val().sum()*0.0;
                let ref_grads = model::checked_gradients(&base,objective(&base,ref_logits))?;
                let actual_grads = model::checked_gradients(&tested,objective(&tested,actual_logits))?;
                let cached_grads = model::checked_gradients(&tested,objective(&tested,cached))?;
                let cached_gradient_error = crate::kron_benchmark::compare_parameters("cached vs reconstructed SBD gradients",
                    &crate::kron_benchmark::parameters(&tested,Some(&cached_grads))?,
                    &crate::kron_benchmark::parameters(&tested,Some(&actual_grads))?)?;
                let reference_cached = values(tested.cached_block(&snapshot,3,&soft_gpu,false));
                for at in prefix.len()..full.len() {
                    let mut perturbed = full.clone();
                    perturbed[at] ^= 127;
                    let mut trace = crate::sbd_training::CleanInputs::default();
                    let _ = tested.forward_profiled(&prefix_batch(&perturbed,&gpu),&mut trace);
                    let snapshot = tested.training_cache(&trace,0,&[1,9,17,19]);
                    identical(&reference_cached,&values(tested.cached_block(&snapshot,3,&soft_gpu,false)),"training snapshot future locality")?;
                }
                let gradient_error = crate::kron_benchmark::compare_parameters("SBD full-model gradients",
                    &crate::kron_benchmark::parameters(&tested,Some(&actual_grads))?,
                    &crate::kron_benchmark::parameters(&base,Some(&ref_grads))?)?;
                let mut lengths = Vec::new();
                for n in [1,4,8,16] {
                    let soft = state(&cfg,vec![None;n],device)?;
                    let inference = tested.valid();
                    let before = values(block_forward(&inference,&prefix,&soft,false,device));
                    // Full sequence is intentionally not passed to block_forward.
                    // Mutate every later clean input/target independently at the caller.
                    let mut sequence = prefix.clone();
                    sequence.extend((0..n+5).map(|i| (i*13) as i32));
                    for at in prefix.len()..sequence.len() {
                        let mut perturbed = sequence.clone();
                        perturbed[at] ^= 127;
                        let after = values(block_forward(&inference,&perturbed[..prefix.len()],&soft,false,device));
                        identical(&before,&after,"future-target locality")?;
                    }
                    if n == 1 {
                        let clean = values(inference.forward(&prefix_batch(&prefix,device)).slice([0..1,prefix.len()-1..prefix.len(),0..cfg.vocab()]));
                        identical(&before,&clean,"L=1 causal limit")?;
                    }
                    lengths.push(json!({"length":n,"leakage_bit_identical":true,"boundary":prefix.len(),"computational_chunk":16}));
                }
                let inference = tested.valid();
                let mut replay = CleanPrefix::new(&inference,prefix.clone(),device);
                let frozen = values(replay.logits.clone());
                let frozen_states = replay.states.iter().cloned().map(values).collect::<Vec<_>>();
                let noisy = state(&cfg,vec![None;4],device)?;
                let _ = values(block_forward(&inference,&replay.tokens,&noisy,false,device));
                identical(&frozen,&values(replay.logits.clone()),"noisy pass changed clean snapshot")?;
                for (before,after) in frozen_states.iter().zip(&replay.states) {
                    identical(before,&values(after.clone()),"noisy pass changed recurrent state")?;
                }
                replay.replay(&inference,&[3,4,5,6],device);
                let (direct,direct_states) = clean_snapshot(&inference,&replay.tokens,device);
                identical(&values(replay.logits.clone()),&values(direct),"causal replay")?;
                for (replayed,direct) in replay.states.iter().zip(direct_states) {
                    identical(&values(replayed.clone()),&values(direct),"causal replay states")?;
                }
                let pad = SoftState::new(cfg.vocab(),4,vec![None;4],vec![true,true,false,false],device,cfg.soft_selection_backend)?;
                let padded = values(block_forward(&inference,&prefix,&pad,false,device));
                let short = state(&cfg,vec![None;2],device)?;
                let short = values(block_forward(&inference,&prefix,&short,false,device));
                // Different GEMM lengths can alter arithmetic; padding comparison uses the fixed numerical tolerance.
                let padding = crate::sbd_reference::compare(&padded[..2*cfg.vocab()],&short.iter().map(|&x| x as f64).collect::<Vec<_>>())?;
                reports.push(json!({"mixer":mixer,"cache":cache,"step7":step7,"logits":logits_report,"cached_training_logits":cached_report,"cached_training_gradient_max_abs":cached_gradient_error,"cached_training_future_bit_identical":true,"all_parameter_gradients_max_abs":gradient_error,"lengths":lengths,"padding":padding,"noisy_snapshot_unchanged":true,"replay_logits_bit_identical":true,"replay_recurrent_states_bit_identical":true,"replayed_tokens":replay.replayed}));
            }
        }
    }
    let mut soft_config = c.clone();
    soft_config.soft_selection_backend = crate::config::SoftSelectionBackend::Cpu;
    soft_config.hard_mask_only = false;
    let soft = crate::soft::episode(&soft_config,device)?;
    Ok(json!({"passed":true,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,"attention_f64":attention,"full_model_cases":reports,"soft_feedback":soft,"reset_semantics":"Each Example is one document. BOS resets every clean replay; blocks never span examples. Padding freezes recurrence and is excluded from exact attention/history.","reference_scope":"Composed existing f64 recurrent/convolution/Step7 references plus new scalar f64 active-block attention/VJP; complete noisy model CPU/GPU logits and all parameter gradients."}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn variable_evaluation_covers_every_length_on_one_example() {
        let device = Device::ndarray();
        device.seed(13);
        let mut c = Config { task:"lm".into(),d_model:16,layers:1,..Config::default() };
        c.sbd = Some(crate::config::SbdConfig {block_lengths:vec![1,4],..Default::default()});
        let arm = Arm {name:"gdn2".into(),mixer:"gdn2".into(),cache:"none".into()};
        let model = LanguageModel::new(&c,&arm,&device).unwrap();
        let example = Example {input:vec![256,65,66,67],target:vec![65,66,67,68],weight:vec![1.0;4],groups:vec![None;4]};
        let report = evaluate(&model,&c,&[example],&device).unwrap();
        assert_eq!(report["predicted_tokens"],8);
    }
}
