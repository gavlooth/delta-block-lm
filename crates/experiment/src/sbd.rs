//! SBD recurrent adaptation: clean NTP plus masked future-block prediction.
use crate::{codpula::{CircuitFlows, PreparedCircuit}, config::{Arm, Config, SbdBlockExecution, SbdCommitment, SbdFeedback, SbdJointOutput}, data::{Batch, Example, Rng}, model::{self, DenoisingOutput, LanguageModel}, runtime, soft::SoftState};
use anyhow::{Context, Result, ensure};
use burn::{module::Module, prelude::*, tensor::activation::log_softmax};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

fn fixture(input: Vec<i32>) -> Example {
    let n = input.len();
    Example { input, target: vec![0;n], weight: vec![1.0;n], groups: vec![None;n] }
}
pub(super) fn prefix_batch(prefix: &[i32], device: &Device) -> Batch {
    Batch::new(&[fixture(prefix.to_vec())], device)
}
pub(super) fn state(c: &Config, observed: Vec<Option<usize>>, device: &Device) -> Result<SoftState> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    let n = observed.len();
    SoftState::new(c.vocab(), s.candidate_width, observed, vec![true;n], device, c.soft_selection_backend)
}
/// `prefix` contains BOS and every clean token strictly before the target block.
/// Only explicit `observed` values may reveal targets inside the block.
pub fn block_forward(model: &LanguageModel, prefix: &[i32], soft: &SoftState, hard: bool, device: &Device) -> DenoisingOutput {
    assert!(!prefix.is_empty());
    let n = soft.observed.len();
    let p = prefix.len();
    let d = model.embedding.weight.val().dims()[1];
    if n == 1 {
        // Exact causal limiting case: the target is predicted from the last clean input.
        return model.forward_output_profiled(&prefix_batch(prefix, device),&mut delta_mixers::profile::Unprofiled)
            .slice_positions(p-1,1);
    }
    if model.cocommit_marker.is_some() {
        return crate::cocommit::reconstruct_output(model,prefix,soft,hard,None,device);
    }
    let mut input = prefix.to_vec();
    input.resize(p+n,0);
    let mut batch = prefix_batch(&input, device);
    let valid = std::iter::repeat_n(1.0f32,p).chain(soft.active.iter().map(|&x| if x {1.0} else {0.0})).collect::<Vec<_>>();
    batch.valid = Tensor::from_data(TensorData::new(valid,[1,p+n]),device);
    let clean = model.embedding.forward(batch.input.clone().slice([0..1,0..p]));
    let noisy = soft.embeddings(model.embedding.weight.val(), model.sbd_mask.as_ref().expect("SBD mask parameter").val(), hard).reshape([1,n,d]);
    model.forward_noisy(Tensor::cat(vec![clean,noisy],1), &batch, p)
        .slice_positions(p,n)
}
pub(super) fn prepare_distribution(model: &LanguageModel, output: DenoisingOutput, soft: &SoftState, temperature: f64) -> Result<(PreparedCircuit,Tensor<2,Bool>)> {
    let [rows,width,_] = output.logits.dims();
    ensure!(soft.active.len() == rows*width && soft.observed.len() == rows*width,"circuit feedback shape differs from output");
    let mut lengths = Vec::with_capacity(rows);
    for row in soft.active.chunks_exact(width) {
        let length = row.iter().take_while(|&&active|active).count();
        ensure!(row[length..].iter().all(|&active|!active),"circuit padding must follow all valid block positions");
        lengths.push(length);
    }
    let observed = Tensor::<2,Bool>::from_data(TensorData::new(
        soft.observed.iter().zip(&soft.active).map(|(value,&active)|active && value.is_some()).collect::<Vec<_>>(),[rows,width]),&output.logits.device());
    let circuit = model.codpula.as_ref().context("Codpula output module is required")?
        .prepare(output.logits,output.features,&lengths,observed.clone(),temperature)?;
    Ok((circuit,observed))
}
pub(super) fn prepare_circuit(model: &LanguageModel, output: DenoisingOutput, soft: &SoftState, temperature: f64) -> Result<(PreparedCircuit,Tensor<2,Int>,Tensor<2,Bool>)> {
    let [rows,width,_] = output.logits.dims();
    let device = output.logits.device();
    let (circuit,observed) = prepare_distribution(model,output,soft,temperature)?;
    let tokens = Tensor::from_data(TensorData::new(
        soft.observed.iter().map(|value|value.unwrap_or(0) as i32).collect::<Vec<_>>(),[rows,width]),&device);
    Ok((circuit,tokens,observed))
}
pub(super) fn update_feedback(model: &LanguageModel, c: &Config, output: DenoisingOutput, soft: &mut SoftState, eta: f64) -> Result<()> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    let logits = if model.codpula.is_some() && output.logits.dims()[1] > 1 {
        let (circuit,tokens,observed) = prepare_circuit(model,output.detach(),soft,c.soft_temperature)?;
        circuit.marginals(tokens,observed)?.detach().log()
    } else {
        return soft.update(output.logits.detach().reshape([soft.observed.len(),c.vocab()]),
            eta,c.soft_temperature,c.soft_steps,s.candidate_width);
    };
    soft.update(logits.reshape([soft.observed.len(),c.vocab()]),eta,1.0,c.soft_steps,s.candidate_width)
}
fn feedback(model: &LanguageModel, c: &Config, prefix: &[i32], soft: &mut SoftState, device: &Device) -> Result<()> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    if s.feedback == SbdFeedback::SoftFeedback && soft.observed.len() > 1 {
        let first = block_forward(model,prefix,soft,false,device);
        update_feedback(model,c,first,soft,1.0)?;
    }
    Ok(())
}

pub struct SbdTrainingResult {
    pub loss: Tensor<1>,
    pub circuit_flows: Option<CircuitFlows>,
    pub(crate) unused_parameters: Vec<&'static str>,
}
impl SbdTrainingResult {
    pub(super) fn add_circuit_loss(&mut self, loss: crate::codpula::CircuitLoss) {
        self.loss = self.loss.clone() + loss.loss;
        if let Some(flows) = &mut self.circuit_flows {
            flows.add(loss.flows);
        } else {
            self.circuit_flows = Some(loss.flows);
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingBlock {
    pub row: usize,
    pub start: usize,
    pub length: usize,
    pub masked: Vec<bool>,
}
/// Samples in row order, drawing every mask before skipping an all-observed block.
pub fn sample_training_blocks(c: &Config, rows: &[Example], seed: u64) -> Result<Vec<TrainingBlock>> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    let mut rng = Rng(seed);
    let mut blocks = Vec::new();
    for (row_index,row) in rows.iter().enumerate() {
        let length = s.block_lengths[rng.index(s.block_lengths.len())];
        let eta = (rng.next() as f64 + 0.5) / (u64::MAX as f64 + 1.0);
        for start in (0..row.target.len()).step_by(length) {
            let n = length.min(row.target.len()-start);
            let masked: Vec<bool> = (0..n).map(|_| (rng.next() as f64 / u64::MAX as f64) < eta).collect();
            if masked.iter().any(|&x| x) {
                blocks.push(TrainingBlock {row:row_index,start,length,masked});
            }
        }
    }
    Ok(blocks)
}
pub(super) fn masked_training_loss(logits: Tensor<3>, block: &TrainingBlock, row: &Example, row_count: usize, vocab: usize, device: &Device) -> Tensor<1> {
    let n = block.masked.len();
    let targets = Tensor::<2,Int>::from_data(TensorData::new(row.target[block.start..block.start+n].to_vec(),[n,1]),device);
    let weights = Tensor::<2>::from_data(TensorData::new((0..n)
        .map(|j| if block.masked[j] {row.weight[block.start+j]} else {0.0}).collect::<Vec<_>>(),[n,1]),device);
    let selected = log_softmax(logits.reshape([n,vocab]),1).gather(1,targets);
    -(selected*weights).sum() / (row_count*row.target.len()) as f64
}
fn training_loss_per_block<P: delta_mixers::profile::Sections>(model: &LanguageModel, c: &Config, rows: &[Example], clean_logits: Tensor<3>, trace: &crate::sbd_training::CleanInputs, blocks: &[TrainingBlock], mut result: SbdTrainingResult, profile: &mut P) -> Result<SbdTrainingResult> {
    let s = c.sbd.as_ref().unwrap();
    let device = clean_logits.device();
    let [batch_size,sequence_length,_] = clean_logits.dims();
    let denominator = (batch_size*sequence_length) as f64;
    let mut selected = blocks.iter().peekable();
    let teacher_model=(s.feedback==SbdFeedback::SoftFeedback && blocks.iter().any(|block|block.masked.len()>1)).then(||model.valid());
    for (row_index,row) in rows.iter().enumerate() {
        let Some(length) = selected.peek().filter(|block| block.row == row_index).map(|block| block.length) else {continue;};
        let starts = blocks.iter().filter(|block|block.row==row_index && block.masked.len()>1).map(|block|block.start+1).collect::<Vec<_>>();
        let cache = (length > 1).then(|| model.training_cache(trace,row_index,&starts));
        let teacher_cache=teacher_model.as_ref().and_then(|_|cache.as_ref().map(|cache|cache.detached()));
        while selected.peek().is_some_and(|block| block.row == row_index) {
            let block = selected.next().unwrap();
            let n = block.masked.len();
            if n == 1 {
                let logits = clean_logits.clone().slice([row_index..row_index+1,block.start..block.start+1,0..c.vocab()]);
                result.loss = result.loss + masked_training_loss(logits,block,row,rows.len(),c.vocab(),&device);
                continue;
            }
            let observed = (0..n).map(|j| (!block.masked[j]).then_some(row.target[block.start+j] as usize)).collect();
            let mut soft = state(c,observed,&device)?;
            let cache = cache.as_ref().unwrap();
            let block_index = starts.binary_search(&(block.start+1)).unwrap();
            if s.feedback == SbdFeedback::SoftFeedback {
                let teacher_model=teacher_model.as_ref().unwrap();
                let teacher=teacher_model.cached_block_profiled(teacher_cache.as_ref().unwrap(),block_index,&soft,false,profile);
                update_feedback(teacher_model,c,teacher,&mut soft,1.0)?;
            }
            let output = model.cached_block_profiled(cache,block_index,&soft,s.feedback == SbdFeedback::HardMask,profile);
            if model.codpula.is_some() {
                let (circuit,observed) = prepare_distribution(model,output,&soft,1.0)?;
                let clean = Tensor::<2,Int>::from_data(TensorData::new(
                    row.target[block.start..block.start+n].to_vec(),[1,n]),&device);
                result.add_circuit_loss(circuit.conditional_nll(clean,observed,denominator)?);
            } else {
                result.loss = result.loss + masked_training_loss(output.logits,block,row,rows.len(),c.vocab(),&device);
            }
        }
    }
    Ok(result)
}
fn training_loss_batched<P: delta_mixers::profile::Sections>(model: &LanguageModel, c: &Config, rows: &[Example], clean_logits: Tensor<3>, trace: &crate::sbd_training::CleanInputs, blocks: &[TrainingBlock], mut result: SbdTrainingResult, profile: &mut P) -> Result<SbdTrainingResult> {
    let s = c.sbd.as_ref().unwrap();
    let device = clean_logits.device();
    let [batch_size,sequence_length,_] = clean_logits.dims();
    let denominator = (batch_size*sequence_length) as f64;
    for block in blocks.iter().filter(|block| block.masked.len() == 1) {
        let logits = clean_logits.clone().slice([block.row..block.row+1,block.start..block.start+1,0..c.vocab()]);
        result.loss = result.loss + masked_training_loss(logits,block,&rows[block.row],rows.len(),c.vocab(),&device);
    }
    let count = blocks.iter().filter(|block| block.masked.len() > 1).count();
    if count == 0 {
        return Ok(result);
    }
    let max_length = *s.block_lengths.iter().max().unwrap();
    let mut cache_blocks = Vec::with_capacity(count);
    let mut observed = Vec::with_capacity(count*max_length);
    let mut active = Vec::with_capacity(count*max_length);
    let mut targets = Vec::with_capacity(count*max_length);
    let factorized = model.codpula.is_none();
    let mut weights = Vec::with_capacity(if factorized {count*max_length} else {0});
    let mut denominators = Vec::with_capacity(if factorized {count} else {0});
    for block in blocks.iter().filter(|block| block.masked.len() > 1) {
        cache_blocks.push((block.row,block.start+1));
        let row = &rows[block.row];
        let n = block.masked.len();
        for j in 0..max_length {
            let real = j < n;
            observed.push(if real && !block.masked[j] {Some(row.target[block.start+j] as usize)} else {None});
            active.push(real);
            targets.push(if real {row.target[block.start+j]} else {0});
            if factorized { weights.push(if real && block.masked[j] {row.weight[block.start+j]} else {0.0}); }
        }
        if factorized { denominators.push((rows.len()*row.target.len()) as f32); }
    }
    let cache = model.training_cache_batched(trace,&cache_blocks);
    let mut soft = SoftState::new(c.vocab(),s.candidate_width,observed,active,&device,c.soft_selection_backend)?;
    if s.feedback == SbdFeedback::SoftFeedback {
        let teacher_model=model.valid();
        let teacher_cache=cache.detached();
        let teacher=teacher_model.cached_blocks_profiled(&teacher_cache,&soft,false,profile);
        update_feedback(&teacher_model,c,teacher,&mut soft,1.0)?;
    }
    let output = model.cached_blocks_profiled(&cache,&soft,s.feedback == SbdFeedback::HardMask,profile);
    if model.codpula.is_some() {
        let (circuit,observed) = prepare_distribution(model,output,&soft,1.0)?;
        let clean = Tensor::<2,Int>::from_data(TensorData::new(targets,[count,max_length]),&device);
        result.add_circuit_loss(circuit.conditional_nll(clean,observed,denominator)?);
        return Ok(result);
    }
    let targets = Tensor::<2,Int>::from_data(TensorData::new(targets,[count*max_length,1]),&device);
    let weights = Tensor::<3>::from_data(TensorData::new(weights,[count,max_length,1]),&device);
    let denominators = Tensor::<1>::from_data(TensorData::new(denominators,[count]),&device);
    let selected = log_softmax(output.logits.reshape([count*max_length,c.vocab()]),1).gather(1,targets)
        .reshape([count,max_length,1]);
    result.loss = result.loss - ((selected*weights).sum_dim(1).reshape([count])/denominators).sum();
    Ok(result)
}
/// Clean loss plus noisy event NLL, divided by the number of clean targets.
/// Uniform sequence-level eta, independent Bernoulli masks, uniform length choice.
pub fn training_loss(model: &LanguageModel, c: &Config, rows: &[Example], batch: &Batch, seed: u64) -> Result<SbdTrainingResult> {
    training_loss_profiled(model,c,rows,batch,seed,&mut delta_mixers::profile::Unprofiled)
}
pub fn training_loss_profiled<P: delta_mixers::profile::Sections>(model: &LanguageModel, c: &Config, rows: &[Example], batch: &Batch, seed: u64, profile: &mut P) -> Result<SbdTrainingResult> {
    let blocks=sample_training_blocks(c,rows,seed)?;
    let bundles=c.sbd.as_ref().and_then(|s|s.cocommit.as_ref())
        .map(|options|crate::cocommit::sample_bundles(&blocks,options.training_bundle_max,seed));
    training_loss_sampled_profiled(model,c,rows,batch,&blocks,bundles.as_deref(),profile)
}
pub(crate) fn training_loss_sampled_profiled<P:delta_mixers::profile::Sections>(model:&LanguageModel,c:&Config,rows:&[Example],batch:&Batch,blocks:&[TrainingBlock],bundles:Option<&[Vec<bool>]>,profile:&mut P)->Result<SbdTrainingResult> {
    let s=c.sbd.as_ref().context("SBD config is required")?;
    if s.cocommit.is_some() {
        return crate::cocommit::training_loss_sampled_profiled(model,c,rows,batch,blocks,bundles.context("CoCommit bundles required")?,profile);
    }
    if s.joint_output.is_codpula() {
        ensure!(rows.iter().all(|row|row.weight.iter().all(|&weight|weight == 1.0)),
            "Codpula joint event loss requires unit valid target weights");
    }
    let mut trace = crate::sbd_training::CleanInputs::new(blocks.iter().filter(|block|block.masked.len()>1).map(|block|block.start+1));
    let clean_logits = model.forward_profiled(batch,&mut trace);
    let loss = model.loss(clean_logits.clone(), batch);
    let result = SbdTrainingResult {loss,circuit_flows:None,
        unused_parameters: model::unused_denoising_parameters(blocks.iter().any(|b| b.masked.len() > 1), false)};
    match s.block_execution {
        SbdBlockExecution::PerBlock => training_loss_per_block(model,c,rows,clean_logits,&trace,blocks,result,profile),
        SbdBlockExecution::Batched => training_loss_batched(model,c,rows,clean_logits,&trace,blocks,result,profile),
    }
}

fn teacher_prefix_evaluate(model: &LanguageModel, c: &Config, examples: &[Example], device: &Device) -> Result<Value> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    if s.cocommit.is_some() {
        return crate::cocommit::teacher_prefix_evaluate(model,c,examples,device);
    }
    if s.joint_output.is_codpula() {
        ensure!(examples.iter().all(|row|row.weight.iter().all(|&weight|weight == 1.0)),
            "Codpula joint event evaluation requires unit valid target weights");
    }
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
            let output = block_forward(&model,prefix,&soft,s.feedback == SbdFeedback::HardMask,device);
            let joint = model.codpula.is_some() && n > 1;
            let probabilities = if joint {
                let (circuit,tokens,observed) = prepare_circuit(&model,output,&soft,1.0)?;
                let clean = Tensor::<2,Int>::from_data(TensorData::new(row.target[start..start+n].to_vec(),[1,n]),device);
                let complete = circuit.log_partition(clean,observed.clone().bool_not())?;
                let visible = circuit.log_partition(tokens.clone(),observed.clone())?;
                nll -= model::scalar(complete-visible)?;
                circuit.marginals(tokens,observed)?.log()
            } else { log_softmax(output.logits,2) };
            let raw = probabilities.into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("SBD evaluation {e:?}"))?;
            for j in 0..n {
                if row.weight[start+j] == 0.0 { continue; }
                let values = &raw[j*c.vocab()..(j+1)*c.vocab()];
                let best = (0..c.vocab()).max_by(|&a,&b| values[a].total_cmp(&values[b]).then(b.cmp(&a))).unwrap();
                if !joint { nll -= values[row.target[start+j] as usize] as f64; }
                hits += usize::from(best == row.target[start+j] as usize);
                count += 1;
            }
        }
      }
    }
    ensure!(count > 0,"SBD evaluation needs targets");
    Ok(json!({"block_nll":nll/count as f64,"block_accuracy":hits as f64/count as f64,"predicted_tokens":count,"conditioning":"all targets masked initially; detached soft first pass when configured; every configured length evaluated on every example","block_lengths":s.block_lengths,"joint_output":s.joint_output,"nll_reduction":"sum block event NLL / valid targets"}))
}

#[derive(Default,Clone)]
pub(crate) struct DecodeScore {
    pub nll: f64,
    pub hits: usize,
    pub count: usize,
}
impl DecodeScore {
    pub(crate) fn observe(&mut self, logs: &[f32], proposals: &[usize], writes: &[usize], targets: &[i32], weights: &[f32], vocab: usize) {
        for &position in writes {
            if weights[position]==0.0 {continue;}
            self.nll-=f64::from(logs[position*vocab+targets[position] as usize]);
            self.hits+=usize::from(proposals[position]==targets[position] as usize);
            self.count+=1;
        }
    }
    pub(crate) fn report(&self) -> Value {json!({"nll":self.nll,"hits":self.hits,"count":self.count})}
}
pub fn evaluate(model: &LanguageModel, c: &Config, examples: &[Example], device: &Device) -> Result<Value> {
    let s=c.sbd.as_ref().context("SBD config is required")?;
    let mut score=DecodeScore::default();
    for &length in &s.block_lengths {
        let mut execution=c.clone();
        execution.sbd.as_mut().unwrap().block_lengths=vec![length];
        for row in examples {
            execution.sbd.as_mut().unwrap().generate_tokens=row.target.len();
            let report=generate_scored(model,&execution,&row.input[..1],Some((&row.target,&row.weight)),device)?;
            score.nll+=report["decode_score"]["nll"].as_f64().context("decode NLL is missing or nonfinite")?;
            score.hits+=report["decode_score"]["hits"].as_u64().context("decode hits missing")? as usize;
            score.count+=report["decode_score"]["count"].as_u64().context("decode token count missing")? as usize;
        }
    }
    ensure!(score.count>0,"SBD decode evaluation needs targets");
    Ok(json!({"block_nll":score.nll/score.count as f64,"block_accuracy":score.hits as f64/score.count as f64,
        "predicted_tokens":score.count,"block_lengths":s.block_lengths,"joint_output":s.joint_output,
        "conditioning":"actual decoder transitions, generated prefix and self-generated block inputs",
        "nll_reduction":"sum token marginal NLL at each final write / valid output tokens",
        "teacher_prefix_diagnostic":teacher_prefix_evaluate(model,c,examples,device)?}))
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

/// Clean token history; the next block's cache is its only required clean pass.
struct CleanPrefix {
    tokens: Vec<i32>,
    appended: usize,
}
impl CleanPrefix {
    fn new(tokens: Vec<i32>) -> Self { Self {tokens,appended:0} }
    fn append(&mut self, finalized: &[i32]) {
        self.tokens.extend_from_slice(finalized);
        self.appended+=finalized.len();
    }
}

/// Explicit full snapshots retained only by the numerical replay diagnostic.
struct ReferencePrefix {
    tokens: Vec<i32>,
    logits: Tensor<3>,
    states: Vec<Tensor<4>>,
    replayed: usize,
}
impl ReferencePrefix {
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

/// Selects only unresolved block positions. Scores are indexed by block position,
/// and non-finite scores represent unsupported candidates and are never committed.
pub(super) fn dependence_commits(unresolved: &[usize], scores: &[f64], threshold: f64, last_pass: bool) -> Vec<usize> {
    if last_pass {
        return unresolved.iter().copied().filter(|&i| scores[i].is_finite()).collect();
    }
    let selected = unresolved.iter().copied().filter(|&i| scores[i].is_finite() && scores[i] < threshold).collect::<Vec<_>>();
    if !selected.is_empty() { return selected; }
    unresolved.iter().copied().filter(|&i| scores[i].is_finite())
        .min_by(|&a,&b|scores[a].total_cmp(&scores[b]).then(a.cmp(&b))).into_iter().collect()
}

pub(super) fn advance_generation_state(
    c: &Config,
    soft: &mut SoftState,
    proposals: &[usize],
    commits: &[usize],
    feedback_logits: Tensor<2>,
    feedback_temperature: f64,
    device: &Device,
) -> Result<()> {
    let s = c.sbd.as_ref().context("SBD state transition requires SBD config")?;
    let n = soft.observed.len();
    ensure!(proposals.len() == n && commits.iter().all(|&i| i < n), "invalid SBD proposal layout");
    if s.feedback == SbdFeedback::SoftFeedback {
        soft.update(feedback_logits.detach(),c.soft_eta,feedback_temperature,c.soft_steps,s.candidate_width)?;
    }
    let mut observed = soft.observed.clone();
    for &i in commits {
        if soft.active[i] && observed[i].is_none() {
            ensure!(proposals[i] < c.vocab(), "unsupported SBD token proposal");
            observed[i] = Some(proposals[i]);
        }
    }
    let mut next = SoftState::new(c.vocab(),s.candidate_width,observed,soft.active.clone(),device,c.soft_selection_backend)?;
    let mutable = Tensor::<2>::from_data(TensorData::new(next.observed.iter().zip(&next.active)
        .map(|(value,&active)|if active && value.is_none() {1.0f32}else{0.0}).collect::<Vec<_>>(),[n,1]),device);
    let probabilities = next.candidates.dense(c.vocab()) * mutable.clone().neg().add_scalar(1.0)
        + soft.candidates.dense(c.vocab()) * mutable.clone();
    next.candidates = crate::soft::Candidates::select(probabilities,s.candidate_width,c.soft_selection_backend)?;
    next.visibility = next.visibility.clone()*mutable.clone().neg().add_scalar(1.0)+soft.visibility.clone()*mutable;
    *soft = next;
    Ok(())
}

pub fn generate(model: &LanguageModel, c: &Config, prompt: &[i32], device: &Device) -> Result<Value> {
    generate_scored(model,c,prompt,None,device)
}
pub(crate) fn generate_scored(model: &LanguageModel, c: &Config, prompt: &[i32], targets: Option<(&[i32],&[f32])>, device: &Device) -> Result<Value> {
    let s = c.sbd.as_ref().context("SBD config is required")?;
    if s.refinement.is_some() {
        return crate::remask::generate_scored(model,c,prompt,targets,device);
    }
    ensure!(!prompt.is_empty(),"generation requires a clean prefix");
    let model = model.valid();
    let mut clean = CleanPrefix::new(prompt.to_vec());
    runtime::sync(device)?;
    let clock = Instant::now();
    let mut passes = 0;
    let mut generated = Vec::new();
    let mut score=DecodeScore::default();
    let mut blocks = 0;
    let mut coordinated_passes = 0usize;
    let mut coordinated_blocks = 0usize;
    let mut proposal_rng = Rng(c.seed);
    while generated.len() < s.generate_tokens {
        let n = s.block_lengths[blocks%s.block_lengths.len()].min(s.generate_tokens-generated.len());
        let mut soft = state(c,vec![None;n],device)?;
        let mut final_values = vec![None;n];
        let mut block_passes = 0;
        let coordination_cache = if s.cocommit.is_some() && n>1 {
            coordinated_blocks+=1;
            Some(crate::cocommit::prepare_clean_cache(&model,&clean.tokens,device))
        } else {None};
        while final_values.iter().any(Option::is_none) {
            if s.cocommit.is_some() && n>1 {
                passes+=1;
                block_passes+=1;
                coordinated_passes+=1;
                let seed = if model.codpula.is_some() {proposal_rng.next()} else {0};
                let assignment = crate::cocommit::assign(&model,c,coordination_cache.as_ref().unwrap(),&soft,block_passes,seed,device)?;
                if let Some((targets,weights))=targets {
                    let start=generated.len();
                    score.observe(&assignment.host_logs,&assignment.proposals,&assignment.commits,
                        &targets[start..start+n],&weights[start..start+n],c.vocab());
                }
                advance_generation_state(c,&mut soft,&assignment.proposals,&assignment.commits,
                    assignment.feedback,assignment.feedback_temperature,device)?;
                final_values=soft.observed.clone();
                continue;
            }
            let output = if let Some(cache) = &coordination_cache {
                model.cached_block(cache,0,&soft,s.feedback == SbdFeedback::HardMask)
            } else {
                block_forward(&model,&clean.tokens,&soft,s.feedback == SbdFeedback::HardMask,device)
            };
            passes += 1;
            block_passes += 1;
            let joint = model.codpula.is_some() && n > 1;
            let (log_probabilities,proposals,scores,feedback_logits,feedback_temperature) = if joint {
                let (circuit,tokens,observed) = prepare_circuit(&model,output,&soft,c.soft_temperature)?;
                let logs = circuit.marginals(tokens.clone(),observed.clone())?.log();
                let proposals = circuit.sample(tokens.clone(),observed.clone(),proposal_rng.next())?;
                let scores = if s.commitment() == SbdCommitment::Dependence {
                    Some(circuit.dependence_scores(tokens,observed,&proposals)?.remove(0))
                } else { None };
                (logs.clone(),Some(proposals.into_iter().next().unwrap()),scores,logs,1.0)
            } else {
                (log_softmax(output.logits.clone(),2),None,None,output.logits,c.soft_temperature)
            };
            let logs = log_probabilities.into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("SBD probabilities {e:?}"))?;
            let mut order: Vec<(usize,f64,usize)> = (0..n).filter(|&i| final_values[i].is_none()).map(|i| {
                let row = &logs[i*c.vocab()..(i+1)*c.vocab()];
                let best = proposals.as_ref().map_or_else(
                    ||(0..c.vocab()).max_by(|&a,&b| row[a].total_cmp(&row[b]).then(b.cmp(&a))).unwrap(),
                    |values|values[i]);
                let entropy = row.iter().filter(|&&x|x.is_finite()).map(|&x| -(x as f64).exp()*x as f64).sum();
                (i,entropy,best)
            }).collect();
            let commits = if let Some(scores) = scores {
                let unresolved = order.iter().map(|&(i,_,_)|i).collect::<Vec<_>>();
                ensure!(unresolved.iter().all(|&i| scores[i].is_finite()),
                    "Codpula commitment requires supported finite proposal scores");
                dependence_commits(&unresolved,&scores,s.dependence_threshold(),block_passes == n)
            } else {
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
                } else { s.commit_count().min(order.len()) };
                order[..take].iter().map(|&(i,_,_)|i).collect()
            };
            let proposal_values = proposals.unwrap_or_else(|| {
                let mut values = vec![0;n];
                for &(i,_,id) in &order { values[i] = id; }
                values
            });
            if let Some((targets,weights))=targets {
                let start=generated.len();
                score.observe(&logs,&proposal_values,&commits,&targets[start..start+n],&weights[start..start+n],c.vocab());
            }
            advance_generation_state(c,&mut soft,&proposal_values,&commits,
                feedback_logits.reshape([n,c.vocab()]),feedback_temperature,device)?;
            final_values = soft.observed.clone();
        }
        drop(coordination_cache);
        let finalized: Vec<i32> = final_values.into_iter().map(|v| v.unwrap() as i32).collect();
        clean.append(&finalized);
        generated.extend_from_slice(&finalized);
        blocks += 1;
    }
    runtime::sync(device)?;
    let seconds = clock.elapsed().as_secs_f64();
    let mut report = json!({"tokens":generated,"text":String::from_utf8_lossy(&generated.iter().map(|&x| x as u8).collect::<Vec<_>>()),"finalized_tokens":generated.len(),"finalized_tokens_per_second":generated.len() as f64/seconds,"seconds":seconds,"denoising_passes":passes,"blocks":blocks,"clean_history_appended_tokens":clean.appended,"includes_clean_cache_preparation":true,"prefill_included":true,"clean_history_implementation":"finalized tokens extend clean history; the next block prepares one shared clean-prefix cache","commit_rule":if s.commitment() == SbdCommitment::Dependence {"dependence"} else if s.entropy_threshold.is_some() {"sbd_entropy_eq8"} else {"fixed_count"},"joint_output":s.joint_output,"joint_execution":s.joint_execution,"proposal_seed":c.seed,"potential_temperature":c.soft_temperature,"dependence_threshold":s.dependence_threshold(),"maximum_passes_per_block":"actual block length"});
    if targets.is_some() {report["decode_score"]=score.report();}
    if let Some(options) = &s.cocommit {
        report["cocommit"] = json!({"config":options,"shared_noisy_prefix_calls":coordinated_passes,
            "unmarked_tail_calls":coordinated_passes,"marked_tail_calls":coordinated_passes,
            "clean_prefix_cache_preparations":coordinated_blocks,
            "clean_prefix_cache_reuses":coordinated_passes.saturating_sub(coordinated_blocks),
            "circuit_preparations":if model.codpula.is_some() {2*coordinated_passes}else{0},
            "circuit_proposals":if model.codpula.is_some() {2*coordinated_passes}else{0},
            "clean_history_appends":blocks,"cost_note":"seconds include clean-cache preparation, both tails, all circuit queries, host selection and feedback"});
    }
    Ok(report)
}

fn validate_checkpoint_identity(c: &Config, arm: &Arm, seed: u64, saved: &crate::train::State) -> Result<()> {
    ensure!(c.sbd.is_some(), "SBD config is required");
    ensure!(
        saved.schema == 1
            && saved.config_sha256 == c.training_fingerprint(c.resolved_tokens()?)?
            && saved.arm == arm.name
            && saved.seed == seed,
        "SBD checkpoint mismatch: model semantics, config, arm or seed changed; dense-attention SBD checkpoints are incompatible"
    );
    Ok(())
}

pub fn checkpoint_report(c: &Config, arm: &Arm, seed: u64, device: &Device, checkpoint: &Path, with_generation: bool) -> Result<Value> {
    let saved = crate::checkpoint::load_state(checkpoint.join("state.json"))?;
    validate_checkpoint_identity(c,arm,seed,&saved)?;
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

/// Compares model readbacks with the existing fixed numerical tolerance.
pub(crate) fn compare<T: Copy + Into<f64>>(actual: &[T], reference: &[f64]) -> Result<Value> {
    const ABSOLUTE_TOLERANCE: f64 = 2e-5;
    const RELATIVE_TOLERANCE: f64 = 3e-4;
    ensure!(actual.len() == reference.len(), "SBD comparison lengths differ");
    let mut maximum_absolute_error = 0.0f64;
    let mut maximum_tolerance_ratio = 0.0f64;
    for (i, (&actual, &reference)) in actual.iter().zip(reference).enumerate() {
        let actual = actual.into();
        let error = (actual - reference).abs();
        let ratio = error / (ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * reference.abs());
        ensure!(
            actual.is_finite() && reference.is_finite() && ratio <= 1.0,
            "SBD element {i}: {actual} vs {reference}, absolute error {error}, tolerance ratio {ratio}"
        );
        maximum_absolute_error = maximum_absolute_error.max(error);
        maximum_tolerance_ratio = maximum_tolerance_ratio.max(ratio);
    }
    Ok(json!({
        "passed": true,
        "compared_elements": reference.len(),
        "max_abs": maximum_absolute_error,
        "maximum_tolerance_ratio": maximum_tolerance_ratio,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
    }))
}

pub fn reference(c: &Config, device: &Device) -> Result<Value> {
    let mut reports = Vec::new();
    let read_oracle = crate::sbd_read_reference::run(device,c.backend != "cpu")?;
    let cpu = Device::ndarray().autodiff();
    let gpu = device.clone().autodiff();
    for mixer in ["kronlstm"] {
        for cache in ["none","surprise"] {
            for step7 in [false,true] {
              for reads in [delta_mixers::sbd::InBlockReads::None,delta_mixers::sbd::InBlockReads::HolaRead] {
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
                cfg.prefix_block_retrieval_backend = delta_mixers::prefix_retrieval::PrefixRetrievalBackend::Tensor;
                cfg.prefix_block_retrieval = if cache == "surprise" {
                    c.prefix_block_retrieval.clone().map(|mut retrieval| {retrieval.layers=vec![0];retrieval})
                } else {None};
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
                cfg.arms = vec![arm.clone()];
                cfg.sbd.as_mut().unwrap().in_block_reads = reads;
                if cache == "none" && reads == delta_mixers::sbd::InBlockReads::HolaRead {
                    ensure!(cfg.validate().unwrap_err().to_string().contains("requires HOLA"),"hola_read must reject layers without HOLA");
                    continue;
                }
                cpu.seed(72);
                let base = LanguageModel::new(&cfg,&arm,&cpu)?;
                let mut tested = base.clone().fork(&gpu);
                if c.backend != "cpu" {
                    tested.set_scan_backend(delta_mixers::ScanBackend::FusedGpu);
                    tested.set_hola_backend(delta_mixers::HolaBackend::FusedGpu);
                    tested.set_noisy_read_backend(if reads.is_none() {delta_mixers::sbd::NoisyReadBackend::Tensor} else {delta_mixers::sbd::NoisyReadBackend::FusedGpu});
                    if let Some(retrieval) = &cfg.prefix_block_retrieval {
                        tested.blocks[0].mixer.set_prefix_retrieval(delta_mixers::prefix_retrieval::PrefixReadConfig {
                            block_size: retrieval.block_size, blocks_per_query: retrieval.blocks_per_query,
                            backend: c.prefix_block_retrieval_backend,
                        });
                    }
                    tested.set_convolution_backend(c.convolution_backend);
                    tested.set_normalization_backend(c.normalization_backend);
                    tested.set_ohc_backend(c.ohc_backend);
                    tested.set_wavelet_backend(c.wavelet_backend);
                    tested.set_lngram_backend(c.lngram_backend);
                }
                let prefix_length = cfg.prefix_block_retrieval.as_ref().map_or(19,|r|r.block_size+cfg.chunk_size+3);
                let prefix: Vec<i32> = std::iter::once(256).chain((0..prefix_length-1).map(|x| ((x*7+11)%256) as i32)).collect();
                let observed = vec![None,Some(65),None,None];
                let soft_cpu = state(&cfg,observed.clone(),&cpu)?;
                let soft_gpu = state(&cfg,observed.clone(),&gpu)?;
                let ref_logits = block_forward(&base,&prefix,&soft_cpu,false,&cpu).logits;
                let actual_logits = block_forward(&tested,&prefix,&soft_gpu,false,&gpu).logits;
                let logits_report = compare(&values(actual_logits.clone()),&values(ref_logits.clone()).iter().map(|&x| x as f64).collect::<Vec<_>>())?;
                let communication = if !reads.is_none() {
                    let changed = state(&cfg,vec![None,Some(97),None,None],&gpu)?;
                    let changed = values(block_forward(&tested,&prefix,&changed,false,&gpu).logits);
                    let original = values(actual_logits.clone());
                    let maximum = original.iter().zip(changed).enumerate().filter(|(i,_)| i/cfg.vocab() != 1)
                        .map(|(_,(&a,b))| (a as f64-b as f64).abs()).fold(0.0,f64::max);
                    ensure!(maximum > 0.0,"observed block token must change another query's logits through HOLA");
                    Some(maximum)
                } else {None};
                let mut full = prefix.clone();
                full.extend([31,32,33,34,35,36]);
                let mut trace = crate::sbd_training::CleanInputs::default();
                let _ = tested.forward_profiled(&prefix_batch(&full,&gpu),&mut trace);
                let snapshot = tested.training_cache(&trace,0,&[1,9,17,prefix.len()]);
                let cached = tested.cached_block(&snapshot,3,&soft_gpu,false).logits;
                let cached_report = compare(&values(cached.clone()),&values(actual_logits.clone()).iter().map(|&x|x as f64).collect::<Vec<_>>())?;
                let ref_grads = model::checked_gradients(&base,ref_logits.square().mean())?;
                let actual_grads = model::checked_gradients(&tested,actual_logits.square().mean())?;
                let cached_grads = model::checked_gradients(&tested,cached.square().mean())?;
                let cached_gradient_error = crate::kron_benchmark::compare_parameters("cached vs reconstructed SBD gradients",
                    &crate::kron_benchmark::parameters(&tested,Some(&cached_grads))?,
                    &crate::kron_benchmark::parameters(&tested,Some(&actual_grads))?)?;
                let reference_cached = values(tested.cached_block(&snapshot,3,&soft_gpu,false).logits);
                for at in prefix.len()..full.len() {
                    let mut perturbed = full.clone();
                    perturbed[at] ^= 127;
                    let mut trace = crate::sbd_training::CleanInputs::default();
                    let _ = tested.forward_profiled(&prefix_batch(&perturbed,&gpu),&mut trace);
                    let snapshot = tested.training_cache(&trace,0,&[1,9,17,prefix.len()]);
                    identical(&reference_cached,&values(tested.cached_block(&snapshot,3,&soft_gpu,false).logits),"training snapshot future locality")?;
                }
                let gradient_error = crate::kron_benchmark::compare_parameters("SBD full-model gradients",
                    &crate::kron_benchmark::parameters(&tested,Some(&actual_grads))?,
                    &crate::kron_benchmark::parameters(&base,Some(&ref_grads))?)?;
                let mut lengths = Vec::new();
                for n in [1,4,8,16] {
                    let soft = state(&cfg,vec![None;n],device)?;
                    let inference = tested.valid();
                    let before = values(block_forward(&inference,&prefix,&soft,false,device).logits);
                    // Full sequence is intentionally not passed to block_forward.
                    // Mutate every later clean input/target independently at the caller.
                    let mut sequence = prefix.clone();
                    sequence.extend((0..n+5).map(|i| (i*13) as i32));
                    for at in prefix.len()..sequence.len() {
                        let mut perturbed = sequence.clone();
                        perturbed[at] ^= 127;
                        let after = values(block_forward(&inference,&perturbed[..prefix.len()],&soft,false,device).logits);
                        identical(&before,&after,"future-target locality")?;
                    }
                    if n == 1 {
                        let clean = values(inference.forward(&prefix_batch(&prefix,device)).slice([0..1,prefix.len()-1..prefix.len(),0..cfg.vocab()]));
                        identical(&before,&clean,"L=1 causal limit")?;
                    }
                    lengths.push(json!({"length":n,"leakage_bit_identical":true,"boundary":prefix.len(),"computational_chunk":16}));
                }
                let inference = tested.valid();
                let mut replay = ReferencePrefix::new(&inference,prefix.clone(),device);
                let frozen = values(replay.logits.clone());
                let frozen_states = replay.states.iter().cloned().map(values).collect::<Vec<_>>();
                let noisy = state(&cfg,vec![None;4],device)?;
                let _ = values(block_forward(&inference,&replay.tokens,&noisy,false,device).logits);
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
                let padded = values(block_forward(&inference,&prefix,&pad,false,device).logits);
                let short = state(&cfg,vec![None;2],device)?;
                let short = values(block_forward(&inference,&prefix,&short,false,device).logits);
                // Different GEMM lengths can alter arithmetic; padding comparison uses the fixed numerical tolerance.
                let padding = compare(&padded[..2*cfg.vocab()],&short.iter().map(|&x| x as f64).collect::<Vec<_>>())?;
                let padded_cached = values(tested.cached_block(&snapshot,3,&SoftState::new(cfg.vocab(),4,vec![None;4],vec![true,true,false,false],&gpu,cfg.soft_selection_backend)?,false).logits);
                let padded_reconstructed = values(block_forward(&tested,&prefix,&SoftState::new(cfg.vocab(),4,vec![None;4],vec![true,true,false,false],&gpu,cfg.soft_selection_backend)?,false,&gpu).logits);
                let cached_padding = compare(&padded_cached,&padded_reconstructed.iter().map(|&x|x as f64).collect::<Vec<_>>())?;
                reports.push(json!({"mixer":mixer,"cache":cache,"step7":step7,"in_block_reads":reads,"observed_cross_position_max_abs":communication,"logits":logits_report,"cached_training_logits":cached_report,"cached_training_gradient_max_abs":cached_gradient_error,"cached_training_future_bit_identical":true,"all_parameter_gradients_max_abs":gradient_error,"lengths":lengths,"padding":padding,"cached_padding":cached_padding,"noisy_snapshot_unchanged":true,"replay_logits_bit_identical":true,"replay_recurrent_states_bit_identical":true,"replayed_tokens":replay.replayed}));
              }
            }
        }
    }
    let mut soft_config = c.clone();
    soft_config.soft_selection_backend = crate::config::SoftSelectionBackend::Cpu;
    soft_config.hard_mask_only = false;
    let soft = crate::soft::episode(&soft_config,device)?;
    Ok(json!({"passed":true,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,"noisy_hola_f64":read_oracle,"full_model_cases":reports,"soft_feedback":soft,"reset_semantics":"Each Example is one document. BOS resets every clean replay; blocks never span examples. Padding freezes recurrence and is excluded from HOLA keys.","reference_scope":"Independent f64 noisy-HOLA forward/full VJP and finite differences; composed existing recurrent/convolution/Step7 references; complete CPU/GPU logits and all parameter gradients, cached/reconstructed equivalence, leakage, intended observed-token communication and replay."}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single sample supports interleaving historical and current executables.
    #[test]
    #[ignore = "explicit cost run; set DELTA_COST_CONFIG to a CPU or GPU config"]
    fn warm_complete_step_and_generation_cost_sample() {
        let path = std::env::var("DELTA_COST_CONFIG").expect("DELTA_COST_CONFIG");
        let c = Config::load(Path::new(&path)).unwrap();
        let device = runtime::device(&c).unwrap();
        device.seed(c.seed);
        let target = device.clone().autodiff();
        let mut model = LanguageModel::new(&c, &c.arms[0], &target).unwrap();
        model.materialize_parameters();
        let inference = model.valid();
        let corpus = crate::train::loaded_corpus(&c).unwrap();
        let rows = crate::data::examples(&c, corpus.as_ref(), c.seed, 0, c.batch_size, "train").unwrap();
        let mut optimizer = crate::train::optimizer(&c);
        let mut step_ms = Vec::new();
        let mut loss = 0.0;
        for _ in 0..3 {
            runtime::sync(&target).unwrap();
            let clock = Instant::now();
            let batch = Batch::new(&rows, &target);
            let result = crate::sbd_step::backward(&model,&c,&rows,&batch,c.seed).unwrap();
            loss = result.loss;
            model = crate::train::optimizer_step(&c,model,&mut optimizer,result.gradients,result.circuit_flows).unwrap();
            runtime::sync(&target).unwrap();
            step_ms.push(clock.elapsed().as_secs_f64()*1000.0);
        }
        let prompt = &rows[0].input;
        let mut decode_ms = Vec::new();
        let mut output = Value::Null;
        for _ in 0..3 {
            runtime::sync(&device).unwrap();
            let clock = Instant::now();
            output = generate(&inference,&c,prompt,&device).unwrap();
            runtime::sync(&device).unwrap();
            decode_ms.push(clock.elapsed().as_secs_f64()*1000.0);
        }
        println!("COST_SAMPLE {}", json!({"config":c,"device":runtime::identity(&device),
            "seed":c.seed,"shape":{"batch":c.batch_size,"context":c.context,"model":c.d_model,"layers":c.layers},
            "training_tokens":c.tokens_per_step(),"generated_tokens":output["finalized_tokens"],
            "step_ms":step_ms,"decode_ms":decode_ms,"warmups":2,"measured_samples":1,
            "last_loss":loss,"generation":output,
            "method":"fixed fixture; complete upload/loss/backward/gradient checks/optimizer update and full generation including prompt preparation and history append; same fresh initial inference parameters; first two samples excluded"}));
    }
    #[test]
    fn plain_generation_matches_cached_training_assignments() {
        for ohc in [false,true] {
            let device=Device::ndarray();
            device.seed(913);
            let c=Config {task:"lm".into(),d_model:8,layers:2,heads:2,key_dim:2,value_dim:2,chunk_size:4,cache_capacity:2,
                ohc_streams:ohc,wavelet_layers:vec![0],lngram_layers:vec![0],lngram_routes:2,lngram_bits:1,lngram_memory_dim:2,lngram_orders:vec![2,3],
                projection_backend:delta_mixers::ProjectionBackend::Merged,
                sbd:Some(crate::config::SbdConfig {block_lengths:vec![3],generate_tokens:6,commit_count:Some(3),feedback:SbdFeedback::HardMask,..Default::default()}),
                ..Default::default()};
            let arm=Arm {name:"kronlstm-hola".into(),mixer:"kronlstm".into(),cache:"surprise".into()};
            let model=LanguageModel::new(&c,&arm,&device).unwrap();
            let mut prefix=vec![256,11,23,37,41,53,61];
            let prompt=prefix.clone();
            let mut expected=Vec::new();
            for _ in 0..2 {
                let cache=crate::cocommit::prepare_clean_cache(&model,&prefix,&device);
                let soft=state(&c,vec![None;3],&device).unwrap();
                let logits=values(model.cached_block(&cache,0,&soft,true).logits);
                let tokens=(0..3).map(|i|(0..c.vocab()).max_by(|&a,&b|
                    logits[i*c.vocab()+a].total_cmp(&logits[i*c.vocab()+b]).then(b.cmp(&a))).unwrap() as i32).collect::<Vec<_>>();
                prefix.extend_from_slice(&tokens);
                expected.extend(tokens);
            }
            let actual=generate(&model,&c,&prompt,&device).unwrap();
            assert_eq!(actual["tokens"],json!(expected));
            assert_eq!(actual["denoising_passes"],2);
        }
    }
    #[test]
    fn dependence_commitment_uses_strict_threshold_and_supported_candidates() {
        let unresolved = [3,1,2];
        // Scores are block-position indexed, not positions in the compact candidate list.
        let scores = [0.0,0.05,0.049,0.6];
        assert_eq!(dependence_commits(&unresolved,&scores,0.05,false),vec![2]);
        assert_eq!(dependence_commits(&[3,1],&scores,0.05,false),vec![1]);
        assert_eq!(dependence_commits(&[3,1],&[0.0,0.2,0.0,0.2],0.05,false),vec![1]);
        assert_eq!(dependence_commits(&unresolved,&scores,0.05,true),unresolved);
        assert_eq!(dependence_commits(&[3],&scores,0.05,false),vec![3]);
        assert_eq!(dependence_commits(&[],&scores,0.05,true),Vec::<usize>::new());
        assert_eq!(dependence_commits(&[3,1,2],&[0.0,0.2,f64::INFINITY,f64::INFINITY],0.05,true),vec![1]);
        assert_eq!(dependence_commits(&[3,1,2],&[0.0,0.2,f64::INFINITY,f64::INFINITY],0.05,false),vec![1]);
        assert_eq!(dependence_commits(&[3,2],&[0.0,0.2,0.2,0.2],0.05,false),vec![2]);
        assert_eq!(dependence_commits(&[3,2],&[0.0,0.2,f64::INFINITY,0.2],0.05,false),vec![3]);
    }
    #[test]
    fn sampled_blocks_keep_original_random_draw_order() {
        let mut c = Config {task:"lm".into(),sbd:Some(crate::config::SbdConfig {
            block_lengths:vec![1,4,8],..Default::default()
        }),..Config::default()};
        let rows = [9,5,7,4].into_iter().map(|n|fixture(vec![0;n])).collect::<Vec<_>>();
        let original = [
            (0,vec![
                (0,0,4,"1011"),(0,4,4,"1010"),(0,8,4,"1"),
                (1,0,4,"0011"),(2,1,1,"1"),(2,3,1,"1"),(2,5,1,"1"),(2,6,1,"1"),
                (3,0,4,"0011"),
            ]),
            (1,vec![
                (0,0,8,"01100110"),(0,8,8,"1"),(1,0,4,"0110"),(2,0,8,"1111111"),
            ]),
        ];
        for (seed,draws) in original {
            let expected = draws.into_iter().map(|(row,start,length,mask)| TrainingBlock {
                row,start,length,masked:mask.bytes().map(|value|value == b'1').collect(),
            }).collect::<Vec<_>>();
            for execution in [SbdBlockExecution::PerBlock,SbdBlockExecution::Batched] {
                c.sbd.as_mut().unwrap().block_execution = execution;
                assert_eq!(sample_training_blocks(&c,&rows,seed).unwrap(),expected);
            }
        }
    }
    #[test]
    fn checkpoint_rejects_old_dense_model_identity() {
        let c = Config { task: "lm".into(), sbd: Some(crate::config::SbdConfig::default()), ..Config::default() };
        let arm = &c.arms[0];
        let mut saved = crate::train::State {
            schema: 1,
            config_sha256: crate::data::hash(&serde_json::to_vec(&(&c,c.resolved_tokens().unwrap())).unwrap()),
            data: Value::Null,
            arm: arm.name.clone(),
            seed: c.seed,
            step: 0,
            tokens: 0,
            training_seconds: 0.0,
            initial_evaluation: Value::Null,
        };
        assert!(validate_checkpoint_identity(&c,arm,c.seed,&saved).is_err());
        saved.config_sha256 = c.training_fingerprint(c.resolved_tokens().unwrap()).unwrap();
        validate_checkpoint_identity(&c,arm,c.seed,&saved).unwrap();
    }
    #[test]
    fn comparison_reports_errors_and_rejects_nonfinite_or_wrong_shapes() {
        let report = compare(&[1.0001f32, 0.0], &[1.0, 0.0]).unwrap();
        assert!(report["max_abs"].as_f64().unwrap() > 0.0);
        assert_eq!(report["compared_elements"], 2);
        assert!(compare(&[1.01], &[1.0]).is_err());
        assert!(compare(&[f64::NAN], &[0.0]).is_err());
        assert!(compare(&[0.0], &[f64::INFINITY]).is_err());
        assert!(compare(&[0.0], &[]).is_err());
    }
    #[test]
    fn variable_evaluation_covers_every_length_on_one_example() {
        let device = Device::ndarray();
        device.seed(13);
        let mut c = Config { task:"lm".into(),d_model:16,layers:1,..Config::default() };
        c.sbd = Some(crate::config::SbdConfig {block_lengths:vec![1,4],..Default::default()});
        let arm = Arm {name:"kronlstm".into(),mixer:"kronlstm".into(),cache:"none".into()};
        let model = LanguageModel::new(&c,&arm,&device).unwrap();
        let example = Example {input:vec![256,65,66,67],target:vec![65,66,67,68],weight:vec![1.0;4],groups:vec![None;4]};
        let report = evaluate(&model,&c,&[example],&device).unwrap();
        assert_eq!(report["predicted_tokens"],8);
    }
    #[test]
    fn decoder_fix_validation_nll_follows_generated_prefix_float64_reference() {
        let device=Device::ndarray();
        device.seed(281);
        let c=Config {task:"lm".into(),d_model:16,layers:1,
            sbd:Some(crate::config::SbdConfig {block_lengths:vec![1],..Default::default()}),..Config::default()};
        let arm=Arm {name:"decode-score".into(),mixer:"kronlstm".into(),cache:"none".into()};
        let model=LanguageModel::new(&c,&arm,&device).unwrap();
        let mut prefix=vec![256];
        let mut targets=Vec::new();
        let mut expected_nll=0.0;
        for position in 0..3 {
            let soft=state(&c,vec![None],&device).unwrap();
            let logits=values(block_forward(&model,&prefix,&soft,false,&device).logits);
            let best=(0..c.vocab()).max_by(|&a,&b|logits[a].total_cmp(&logits[b]).then(b.cmp(&a))).unwrap();
            let gold=(best+position+1)%c.vocab();
            let maximum=logits.iter().map(|&x|f64::from(x)).fold(f64::NEG_INFINITY,f64::max);
            expected_nll+=maximum+logits.iter().map(|&x|(f64::from(x)-maximum).exp()).sum::<f64>().ln()-f64::from(logits[gold]);
            targets.push(gold as i32);
            prefix.push(best as i32);
        }
        let row=Example {input:std::iter::once(256).chain(targets[..2].iter().copied()).collect(),
            target:targets,weight:vec![1.0;3],groups:vec![None;3]};
        let report=evaluate(&model,&c,&[row],&device).unwrap();
        compare(&[report["block_nll"].as_f64().unwrap()],&[expected_nll/3.0]).unwrap();
        assert!((report["block_nll"].as_f64().unwrap()-report["teacher_prefix_diagnostic"]["block_nll"].as_f64().unwrap()).abs()>1e-6);
    }
}
