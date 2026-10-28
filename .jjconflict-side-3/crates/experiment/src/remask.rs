//! Fixed-cardinality remasking with immutable evidence and detached soft feedback.
use crate::{
    config::{Arm, Config},
    data::{self, Batch, Example, Rng},
    mask_energy::{self, Dataset, MaskEnergy, MaskPolicy, RankPair, RankState},
    model::LanguageModel,
    runtime,
    sbd_training::{BatchedTrainingCache, CleanInputs},
    soft::{Candidates, SoftState},
};
use anyhow::{Context, Result, ensure};
use burn::{module::{Module, ModuleVisitor, Param}, prelude::*, tensor::activation::log_softmax};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::{Path, PathBuf}, time::Instant};

const CALLS_PER_LEVEL: usize = 5;
const PATIENCE: usize = 2;
pub const FEATURE_VERSION: &str = "remask-2d12-v1";
const RANDOM_VERSION: &str = "splitmix64-indexed-v1-f32-24bit";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RefinementConfig {
    pub policy: MaskPolicy,
    pub energy_path: Option<PathBuf>,
    pub data_path: PathBuf,
    pub states: usize,
    pub pairs_per_state: usize,
    pub horizon: usize,
    pub fit_steps: usize,
    pub fit_batch: usize,
}
impl Default for RefinementConfig {
    fn default() -> Self {
        Self { policy: MaskPolicy::Uniform, energy_path: None,
            data_path: "artifacts/remasking/trajectories.json".into(),
            states: 512, pairs_per_state: 4, horizon: 2, fit_steps: 100, fit_batch: 512 }
    }
}
impl RefinementConfig {
    pub fn energy_path(&self) -> PathBuf {
        self.energy_path.clone().unwrap_or_else(|| "artifacts/remasking/energy.burnpack".into())
    }
}
fn settings(c: &Config) -> Result<&RefinementConfig> {
    c.sbd.as_ref().context("remasking requires SBD")?.refinement.as_ref()
        .context("remasking requires an explicit refinement selector")
}
fn host<const D: usize>(x: Tensor<D>) -> Result<Vec<f32>> {
    x.into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("remasking float readback: {e:?}"))
}
fn indexed_seed(seed: u64, stream: u64, identity: u64, call: usize) -> u64 {
    let mut random = Rng(seed ^ stream.wrapping_mul(0xd1342543de82ef95)
        ^ identity.wrapping_mul(0xa0761d6478bd642f) ^ (call as u64).wrapping_mul(0xe7037ed1a0b428db));
    random.next()
}
fn uniform(random: &mut Rng) -> f64 {
    (random.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
}
fn integer(random: &mut Rng, count: usize) -> usize {
    assert!(count > 0);
    let count = count as u64;
    let threshold = count.wrapping_neg() % count;
    loop { let value = random.next(); if value >= threshold { return (value % count) as usize; } }
}

/// A temporary embedding view, never circuit evidence or a change to retained candidates.
/// Padded values and reveal requests at permanent evidence are deliberately ignored.
pub(crate) fn provisional_input(soft: &SoftState, values: &[usize], revealed: &[bool], device: &Device) -> Result<SoftState> {
    let rows = soft.observed.len();
    ensure!(values.len() == rows && revealed.len() == rows, "provisional input shape differs from soft state");
    let k = soft.candidates.ids.dims()[1];
    let mut selected = Vec::with_capacity(rows);
    let mut ids = Vec::with_capacity(rows*k);
    let mut weights = Vec::with_capacity(rows*k);
    for i in 0..rows {
        let visible = soft.active[i] && soft.observed[i].is_none() && revealed[i];
        ensure!(!visible || values[i] <= i32::MAX as usize, "provisional token cannot fit int32");
        selected.push(visible);
        for j in 0..k {
            ids.push(if visible {values[i] as i32} else {0});
            weights.push(if visible && j == 0 {1.0f32} else {0.0});
        }
    }
    let selected = Tensor::<2,Bool>::from_data(TensorData::new(selected,[rows,1]),device);
    let mut view = soft.clone();
    view.candidates.ids = view.candidates.ids.mask_where(selected.clone().expand([rows,k]),
        Tensor::from_data(TensorData::new(ids,[rows,k]),device));
    view.candidates.weights = view.candidates.weights.mask_where(selected.clone().expand([rows,k]),
        Tensor::from_data(TensorData::new(weights,[rows,k]),device));
    view.candidates.retained_mass = view.candidates.retained_mass.mask_fill(selected.clone(),1.0);
    view.visibility = view.visibility.mask_fill(selected,1.0);
    Ok(view)
}
/// Ordinary refinement trains on detached self-generated hard inputs, never gold free tokens.
/// Uniform cardinalities cover empty, partial and fully revealed provisional inputs without
/// requiring an energy fitted to the model currently being trained.
pub(crate) fn training_provisional_input(model: &LanguageModel, c: &Config, output: crate::model::DenoisingOutput,
    soft: &mut SoftState, seeds: &[u64]) -> Result<()> {
    let [rows,width,_]=output.logits.dims();
    ensure!(seeds.len()==rows,"provisional training random identities differ");
    let proposals=if model.codpula.is_some() {
        let device=output.logits.device();
        let uniforms=seeds.iter().flat_map(|&seed| {
            let mut random=Rng(seed);
            (0..width*2).map(move |_|(random.next()>>40) as f32/16777216.0)
        }).collect::<Vec<_>>();
        let (circuit,tokens,observed)=crate::sbd::prepare_circuit(model,output.detach(),soft,c.soft_temperature)?;
        circuit.sample_with_uniforms(tokens,observed,Tensor::from_data(TensorData::new(uniforms,[rows,width,2]),&device))?
            .into_iter().flat_map(|mut row| {row.resize(width,0);row}).collect::<Vec<_>>()
    } else {crate::soft::ids_host(output.logits.reshape([rows*width,c.vocab()]).argmax(1)).into_iter().map(|x|x as usize).collect()};
    let mut revealed=vec![false;rows*width];
    for row in 0..rows {
        let mut random=Rng(seeds[row] ^ 0x70726f766973696f);
        let mut free=(row*width..(row+1)*width).filter(|&i|soft.active[i] && soft.observed[i].is_none()).collect::<Vec<_>>();
        let count=integer(&mut random,free.len()+1);
        for position in 0..count {
            let selected=position+integer(&mut random,free.len()-position);
            free.swap(position,selected);
            revealed[free[position]]=true;
        }
    }
    *soft=provisional_input(soft,&proposals,&revealed,&soft.visibility.device())?;
    Ok(())
}

#[derive(Clone)]
struct EpisodeScoring {
    targets: Vec<i32>,
    weights: Vec<f32>,
    nll: Vec<f64>,
    hits: Vec<bool>,
}
#[derive(Clone)]
struct Episode {
    soft: SoftState,
    values: Vec<usize>,
    revealed: Vec<bool>,
    lengths: Vec<usize>,
    previous_logs: Option<Tensor<2>>,
    previous_candidates: Option<Candidates>,
    previous_values: Option<Vec<usize>>,
    scoring: Option<EpisodeScoring>,
}
impl Episode {
    fn new(c: &Config, lengths: Vec<usize>, width: usize, observed: Vec<Option<usize>>, device: &Device) -> Result<Self> {
        ensure!(!lengths.is_empty() && width > 0 && width <= 16
            && lengths.iter().all(|&n| n > 0 && n <= width), "invalid remasking block lengths");
        let rows = lengths.len()*width;
        ensure!(observed.len() == rows, "remasking evidence layout differs");
        let active = lengths.iter().flat_map(|&n| (0..width).map(move |i| i<n)).collect();
        let soft = SoftState::new(c.vocab(),c.sbd.as_ref().unwrap().candidate_width,observed,active,device,c.soft_selection_backend)?;
        let values = soft.observed.iter().map(|v|v.unwrap_or(0)).collect();
        Ok(Self {soft,values,revealed:vec![false;rows],lengths,previous_logs:None,previous_candidates:None,previous_values:None,scoring:None})
    }
    fn width(&self) -> usize { self.soft.active.len()/self.lengths.len() }
    fn eligible(&self, row: usize) -> Vec<usize> {
        let width = self.width();
        (0..self.lengths[row]).filter(|&i|self.soft.observed[row*width+i].is_none()).collect()
    }
    fn validate(&self, vocab: usize) -> Result<()> {
        let width = self.width();
        ensure!(self.values.len() == self.soft.active.len() && self.revealed.len() == self.values.len(),"invalid remasking state sizes");
        for (row,&length) in self.lengths.iter().enumerate() {
            for i in 0..width {
                let flat = row*width+i;
                ensure!(self.soft.active[flat] == (i<length),"remasking padding is not a right suffix");
                if i>=length { ensure!(!self.revealed[flat],"padding cannot be provisionally revealed"); continue; }
                ensure!(self.values[flat] < vocab,"remasking token outside vocabulary");
                if let Some(value) = self.soft.observed[flat] {
                    ensure!(self.values[flat] == value && !self.revealed[flat],"permanent evidence changed");
                }
            }
        }
        Ok(())
    }
    fn apply_sets(&mut self, sets: &[Vec<usize>], cardinalities: &[usize], c: &Config, device: &Device) -> Result<()> {
        ensure!(sets.len() == self.lengths.len() && cardinalities.len() == sets.len(),"remasking set batch differs");
        let width = self.width();
        for row in 0..sets.len() {
            let eligible = self.eligible(row);
            ensure!(sets[row].len() == cardinalities[row] && sets[row].windows(2).all(|p|p[0]<p[1])
                && sets[row].iter().all(|i|eligible.binary_search(i).is_ok()),"mask set violates support or exact cardinality");
        }
        for row in 0..sets.len() {
            let eligible=self.eligible(row);
            for i in 0..width {
                self.revealed[row*width+i] = eligible.binary_search(&i).is_ok() && sets[row].binary_search(&i).is_err();
            }
        }
        // Fresh candidate feedback remains visible on masked positions; reveals use hard token inputs.
        self.soft.visibility = Tensor::from_data(TensorData::new(self.soft.observed.iter().zip(&self.soft.active)
            .map(|(value,&active)|if !active {0.0f32} else if value.is_some() {1.0} else {1.0/c.soft_steps as f32}).collect::<Vec<_>>(),[self.values.len(),1]),device);
        self.validate(c.vocab())
    }
}

fn token_uniforms(state: &Episode, keys: &[u64], seed: u64, call: usize, device: &Device) -> Tensor<3> {
    let width = state.width();
    let mut values = vec![0.0f32;state.values.len()*2];
    for (row,&key) in keys.iter().enumerate() {
        let mut random = Rng(indexed_seed(seed,1,key,call));
        for position in 0..state.lengths[row] {
            for channel in 0..2 {
                // Exactly representable FP32 uniforms cannot round up to one.
                values[(row*width+position)*2+channel] = (random.next()>>40) as f32 / 16777216.0;
            }
        }
    }
    Tensor::from_data(TensorData::new(values,[state.lengths.len(),width,2]),device)
}

fn select_branch_output(a: crate::model::DenoisingOutput, b: crate::model::DenoisingOutput,
    use_b: &[bool], device: &Device) -> crate::model::DenoisingOutput {
    let [rows,width,vocab] = a.logits.dims();
    let depth = a.features.dims()[2];
    let selected = Tensor::<2,Bool>::from_data(TensorData::new(use_b.to_vec(),[rows,1]),device)
        .expand([rows,width]);
    let logits = a.logits.mask_where(selected.clone().unsqueeze_dim::<3>(2).expand([rows,width,vocab]),b.logits);
    let features = a.features.mask_where(selected.unsqueeze_dim::<3>(2).expand([rows,width,depth]),b.features);
    crate::model::DenoisingOutput {logits,features}
}
/// A's preparation masks only O; standard decisions query temporary O+R, endpoint/correction O.
fn announcement_marks(model: &LanguageModel, c: &Config, a: crate::model::DenoisingOutput,
    state: &Episode, eligible: &[bool], carry_reveals: bool, keys: &[u64], seed: u64, call: usize,
    device: &Device, work: &mut crate::cocommit::CoordinationWork) -> Result<Vec<bool>> {
    let rows=state.lengths.len();
    let width=state.width();
    let s=c.sbd.as_ref().unwrap();
    let default_announcement=crate::config::AnnouncementConfig::default();
    let announcement=s.announcement.as_ref().unwrap_or(&default_announcement);
    let (logs,scores)=if model.codpula.is_some() {
        let (circuit,tokens,observed)=crate::sbd::prepare_circuit(model,a,&state.soft,c.soft_temperature)?;
        let (tokens,observed)=if carry_reveals {
            (Tensor::<2,Int>::from_data(TensorData::new(state.values.iter().map(|&value|value as i32).collect::<Vec<_>>(),[rows,width]),device),
                Tensor::<2,Bool>::from_data(TensorData::new(state.soft.observed.iter().zip(&state.revealed)
                    .map(|(o,&r)|o.is_some() || r).collect::<Vec<_>>(),[rows,width]),device))
        } else {(tokens,observed)};
        work.circuit_preparations+=1;
        work.message_passes+=1;
        work.evidence_queries+=1;
        work.circuit_query_dispatched_rows+=rows;
        work.circuit_query_used_rows+=rows;
        let logs=circuit.marginals(tokens.clone(),observed.clone())?.log();
        let scores=if announcement.dependence_threshold.is_some() {
            let proposals=circuit.sample_with_uniforms(tokens.clone(),observed.clone(),
                token_uniforms(state,keys,seed,call,device))?;
            work.ffbs_passes+=1;
            work.ffbs_dispatched_rows+=rows;
            work.ffbs_used_rows+=rows;
            work.message_passes+=3; // FFBS messages plus both dependence traversals.
            work.evidence_queries+=2;
            work.circuit_query_dispatched_rows+=2*rows;
            work.circuit_query_used_rows+=2*rows;
            Some(circuit.dependence_scores(tokens,observed,&proposals)?)
        } else {None};
        (host(logs)?,scores)
    } else {(host(log_softmax(a.logits/c.soft_temperature,2))?,None)};
    let mut marks=vec![false;rows*width];
    for row in 0..rows {
        let positions=(0..state.lengths[row]).filter(|&i|eligible[row*width+i]).collect::<Vec<_>>();
        let chosen=if let Some(scores)=&scores {
            ensure!(positions.iter().all(|&i|scores[row][i].is_finite()),
                "CoCommit dependence requires finite supported scores");
            crate::sbd::dependence_commits(&positions,&scores[row],announcement.dependence_threshold.unwrap(),false)
        } else {
            let mut order=positions.into_iter().map(|i| {
                let logs=&logs[(row*width+i)*c.vocab()..(row*width+i+1)*c.vocab()];
                let entropy=logs.iter().filter(|log|log.is_finite()).map(|&log|-(log as f64).exp()*log as f64).sum::<f64>();
                (i,entropy)
            }).collect::<Vec<_>>();
            order.sort_by(|a,b|a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let take=if order.is_empty() {0} else if let Some(gamma)=announcement.entropy_threshold {
                let mut take=1;
                let mut sum=0.0;
                while take<order.len() && sum+order[take-1].1<=gamma {sum+=order[take-1].1;take+=1;}
                take
            } else {announcement.count.min(order.len())};
            order[..take].iter().map(|&(i,_)|i).collect()
        };
        for i in chosen {marks[row*width+i]=true;}
    }
    Ok(marks)
}

fn coordinated_output(model: &LanguageModel, c: &Config, cache: &BatchedTrainingCache,
    input: &SoftState, state: &Episode, eligible: &[bool], carry_reveals: bool, keys: &[u64], seed: u64,
    call: usize, device: &Device) -> Result<(crate::model::DenoisingOutput,Vec<bool>,crate::cocommit::CoordinationWork)> {
    let rows=state.lengths.len();
    let width=state.width();
    let prefix=model.cached_blocks_prefix_profiled(cache,input,false,&mut delta_mixers::profile::Unprofiled);
    let a=model.cached_blocks_tail_profiled(cache,prefix.clone(),None,&mut delta_mixers::profile::Unprofiled).detach();
    let mut work=crate::cocommit::CoordinationWork {logical_blocks:rows,prefix_batches:1,prefix_rows:rows,
        a_tail_dispatches:1,a_tail_rows:rows,a_tail_used_rows:rows,head_dispatches:1,head_rows:rows,
        head_used_rows:rows,..crate::cocommit::CoordinationWork::default()};
    let marks=announcement_marks(model,c,a.clone(),state,eligible,carry_reveals,keys,seed,call,device,&mut work)?;
    let used=marks.chunks_exact(width).map(|marks|marks.iter().any(|&mark|mark)).collect::<Vec<_>>();
    let used_count=used.iter().filter(|&&used|used).count();
    work.empty_announcement_rows=rows-used_count;
    if used_count==0 {return Ok((a,marks,work));}
    let b=model.cached_blocks_tail_profiled(cache,prefix,
        Some(Tensor::from_data(TensorData::new(marks.iter().map(|&mark|if mark {1.0f32}else{0.0}).collect::<Vec<_>>(),[rows,width]),device)),
        &mut delta_mixers::profile::Unprofiled).detach();
    work.b_tail_dispatches=1;
    work.b_tail_rows=rows;
    work.b_tail_used_rows=used_count;
    work.head_dispatches+=1;
    work.head_rows+=rows;
    work.head_used_rows+=used_count;
    work.b_circuit_used_rows=if model.codpula.is_some(){used_count}else{0};
    Ok((select_branch_output(a,b,&used,device),marks,work))
}

struct Prediction {
    hidden: Tensor<3>,
    logs: Tensor<2>,
    input_visibility: Tensor<2>,
    input_revealed: Vec<bool>,
    work: crate::cocommit::CoordinationWork,
}
fn predict(model: &LanguageModel, c: &Config, cache: &BatchedTrainingCache, state: &mut Episode,
    keys: &[u64], seed: u64, call: usize, device: &Device) -> Result<Prediction> {
    let teacher_feedback=prepare_episode_feedback(model,c,cache,state)?;
    state.validate(c.vocab())?;
    ensure!(keys.len() == state.lengths.len(),"token RNG identities differ from active rows");
    let view = provisional_input(&state.soft,&state.values,&state.revealed,device)?;
    let visibility = view.visibility.clone();
    let revealed = state.revealed.clone();
    let rows = state.lengths.len();
    let width = state.width();
    let correction = c.sbd.as_ref().unwrap().training_mode == crate::tct::TrainingMode::Tct;
    let marked = model.cocommit_config.is_some();
    let (output,marks,mut work)=if marked {
        let eligible=state.soft.active.iter().zip(&state.soft.observed).zip(&revealed)
            .map(|((&active,observed),&reveal)|active && observed.is_none() && (correction || !reveal)).collect::<Vec<_>>();
        coordinated_output(model,c,cache,&view,state,&eligible,!correction,keys,seed,call,device)?
    } else {(model.cached_blocks(cache,&view,false).detach(),Vec::new(),crate::cocommit::CoordinationWork::default())};
    let used_b_count=marks.chunks_exact(width).filter(|marks|marks.iter().any(|&mark|mark)).count();
    let hidden=output.features.clone();
    if teacher_feedback {
        work.prefix_batches+=1;
        work.prefix_rows+=rows;
        work.a_tail_dispatches+=1;
        work.a_tail_rows+=rows;
        work.a_tail_used_rows+=rows;
        work.head_dispatches+=1;
        work.head_rows+=rows;
        work.head_used_rows+=rows;
    }
    let mut write_logs=None;
    let logs = if model.codpula.is_some() {
        if marked {
            work.circuit_preparations += 1;
            work.message_passes += 2;
            work.evidence_queries += 1;
            work.ffbs_passes += 1;
            work.circuit_query_dispatched_rows += rows;
            work.circuit_query_used_rows += rows;
            work.b_circuit_used_rows = used_b_count;
            work.ffbs_dispatched_rows += rows;
            work.ffbs_used_rows += rows;
        }
        let (circuit,tokens,fixed) = crate::sbd::prepare_circuit(model,output,&state.soft,c.soft_temperature)?;
        let proposal_tokens = Tensor::<2,Int>::from_data(TensorData::new(
            state.values.iter().map(|&v|v as i32).collect::<Vec<_>>(),[rows,width]),device);
        let proposal_fixed = Tensor::<2,Bool>::from_data(TensorData::new(
            state.soft.observed.iter().zip(&revealed).map(|(value,&reveal)|value.is_some() || (!correction && reveal)).collect::<Vec<_>>(),[rows,width]),device);
        if state.scoring.is_some() {
            if marked {
                work.message_passes+=1;
                work.evidence_queries+=1;
                work.circuit_query_dispatched_rows+=rows;
                work.circuit_query_used_rows+=rows;
            }
            write_logs=Some(host(circuit.marginals(proposal_tokens.clone(),proposal_fixed.clone())?.reshape([rows*width,c.vocab()]).log())?);
        }
        let proposals = circuit.sample_with_uniforms(proposal_tokens,proposal_fixed,token_uniforms(state,keys,seed,call,device))?;
        for (row,proposal) in proposals.into_iter().enumerate() {
            for (i,value) in proposal.into_iter().enumerate() { state.values[row*width+i] = value; }
        }
        // Full-support O-only marginals keep same-token previous-log changes well defined.
        // The standard token proposal above separately conditions on provisional reveals.
        circuit.marginals(tokens,fixed)?.reshape([rows*width,c.vocab()]).log()
    } else {
        let logs = log_softmax(output.logits/c.soft_temperature,2).reshape([rows*width,c.vocab()]);
        if state.scoring.is_some() {write_logs=Some(host(logs.clone())?);}
        let best = crate::soft::ids_host(logs.clone().argmax(1));
        for i in 0..state.values.len() {
            if state.soft.active[i] && state.soft.observed[i].is_none() && (correction || !revealed[i]) { state.values[i] = best[i] as usize; }
        }
        logs
    };
    if let (Some(scoring),Some(write_logs))=(&mut state.scoring,write_logs) {
        for i in 0..state.values.len() {
            if state.soft.active[i] && state.soft.observed[i].is_none() && (correction || !revealed[i]) {
                scoring.nll[i]=-f64::from(write_logs[i*c.vocab()+scoring.targets[i] as usize]);
                scoring.hits[i]=state.values[i]==scoring.targets[i] as usize;
            }
        }
    }
    let previous = state.soft.candidates.clone();
    state.soft.replace_feedback(logs.clone().detach(),1.0,c.soft_steps,c.sbd.as_ref().unwrap().candidate_width)?;
    state.previous_candidates = Some(previous);
    state.validate(c.vocab())?;
    Ok(Prediction {hidden,logs,input_visibility:visibility,input_revealed:revealed,work})
}

fn features(model: &LanguageModel, c: &Config, state: &Episode, output: &Prediction,
    cardinalities: &[usize], noise: &[f32], device: &Device) -> Result<Tensor<3>> {
    let rows = state.lengths.len();
    let width = state.width();
    let flat = rows*width;
    let depth = output.hidden.dims()[2];
    ensure!(cardinalities.len() == rows && noise.len() == rows,"feature schedule batch differs");
    let legal = Tensor::<2,Bool>::from_data(TensorData::new(state.soft.active.iter().zip(&state.soft.observed)
        .map(|(&active,value)|active && value.is_none()).collect::<Vec<_>>(),[flat,1]),device);
    let ids = Tensor::<2,Int>::from_data(TensorData::new(state.values.iter().map(|&v|v as i32).collect::<Vec<_>>(),[flat,1]),device);
    let logs = output.logs.clone();
    let probabilities = logs.clone().exp();
    let selected = logs.clone().gather(1,ids.clone()).mask_fill(legal.clone().bool_not(),0.0);
    let safe_logs = logs.clone().mask_fill(probabilities.clone().equal_elem(0.0),0.0);
    let entropy = -(probabilities.clone()*safe_logs).sum_dim(1);
    let top = Candidates::select(probabilities,2,c.soft_selection_backend)?;
    let top_logs = logs.clone().gather(1,top.ids);
    let gap = (top_logs.clone().narrow(1,0,1)-top_logs.narrow(1,1,1)).mask_fill(legal.clone().bool_not(),0.0);
    let candidate = &state.soft.candidates;
    let expectation = candidate.expected(model.embedding.weight.val().detach(),model.sbd_mask.as_ref().unwrap().val().detach(),Tensor::ones([flat,1],device),false);
    let (changed,log_change,l1) = if let (Some(previous_logs),Some(previous_values),Some(previous)) = (&state.previous_logs,&state.previous_values,&state.previous_candidates) {
        let changed = Tensor::from_data(TensorData::new(state.values.iter().zip(previous_values).map(|(a,b)|if a!=b {1.0f32}else{0.0}).collect::<Vec<_>>(),[flat,1]),device);
        let change = selected.clone()-previous_logs.clone().gather(1,ids).mask_fill(legal.clone().bool_not(),0.0);
        let k = candidate.ids.dims()[1];
        let old_k = previous.ids.dims()[1];
        let matches = candidate.ids.clone().reshape([flat,k,1]).equal(previous.ids.clone().reshape([flat,1,old_k]));
        let overlap = candidate.weights.clone().reshape([flat,k,1]).min_pair(previous.weights.clone().reshape([flat,1,old_k]))
            .mask_fill(matches.bool_not(),0.0).sum_dim(1).sum_dim(2).reshape([flat,1]);
        let l1 = (candidate.weights.clone().sum_dim(1)+previous.weights.clone().sum_dim(1)-overlap*2.0).clamp_min(0.0);
        (changed,change,l1)
    } else { (Tensor::zeros([flat,1],device),Tensor::zeros([flat,1],device),Tensor::zeros([flat,1],device)) };
    let position = Tensor::from_data(TensorData::new(state.lengths.iter().flat_map(|&n|(0..width).map(move |i|i as f32/n.saturating_sub(1).max(1) as f32)).collect::<Vec<_>>(),[flat,1]),device);
    let reveal = Tensor::from_data(TensorData::new(output.input_revealed.iter().map(|&v|if v {1.0f32}else{0.0}).collect::<Vec<_>>(),[flat,1]),device);
    let fraction = Tensor::from_data(TensorData::new((0..rows).flat_map(|row|std::iter::repeat_n(cardinalities[row] as f32/state.eligible(row).len().max(1) as f32,width)).collect::<Vec<_>>(),[flat,1]),device);
    let noise = Tensor::from_data(TensorData::new(noise.iter().flat_map(|&t|std::iter::repeat_n(t,width)).collect::<Vec<_>>(),[flat,1]),device);
    let scalars = Tensor::cat(vec![selected,entropy,gap,candidate.retained_mass.clone(),output.input_visibility.clone(),changed,log_change,l1,position,reveal,fraction,noise],1);
    Ok(Tensor::cat(vec![output.hidden.clone().reshape([flat,depth]),expectation,scalars],1)
        .mask_fill(legal.bool_not().expand([flat,2*depth+12]),0.0).reshape([rows,width,2*depth+12]).detach())
}
fn remember(state: &mut Episode, output: &Prediction) {
    state.previous_logs = Some(output.logs.clone().detach());
    state.previous_values = Some(state.values.clone());
}
fn choose_sets(state: &Episode, cardinalities: &[usize], policy: MaskPolicy, terms: Option<&(Vec<f32>,Vec<f32>)>,
    keys: &[u64], seed: u64, call: usize) -> Result<Vec<Vec<usize>>> {
    let width = state.width();
    (0..state.lengths.len()).map(|row| {
        let mut random = Rng(indexed_seed(seed,2,keys[row],call));
        let (unary,pairs) = terms.map_or((&[][..],&[][..]),|(u,p)|(&u[row*width..(row+1)*width],&p[row*width*width..(row+1)*width*width]));
        mask_energy::sample_set(unary,pairs,&state.eligible(row),cardinalities[row],policy,&mut random)
    }).collect()
}
fn energy_terms(energy: &MaskEnergy, input: Tensor<3>, state: &Episode, policy: MaskPolicy, device: &Device) -> Result<(Vec<f32>,Vec<f32>)> {
    let rows = state.lengths.len();
    let width = state.width();
    let valid = Tensor::<2,Bool>::from_data(TensorData::new(state.soft.active.iter().zip(&state.soft.observed)
        .map(|(&active,value)|active && value.is_none()).collect::<Vec<_>>(),[rows,width]),device);
    let finite = input.clone().is_finite().all().float().reshape([1]);
    let (unary,pairs) = energy.terms(input,valid,policy == MaskPolicy::Pairwise);
    // One synchronized readback contains all rows' unary and scalar pair terms.
    let mut packed = host(Tensor::cat(vec![unary.reshape([rows*width]),pairs.reshape([rows*width*width]),finite],0))?;
    let feature_finite = packed.pop().unwrap();
    ensure!(feature_finite==1.0 && packed.iter().all(|x|x.is_finite()),"nonfinite remasking features or energy terms");
    let pairs = packed.split_off(rows*width);
    Ok((packed,pairs))
}

struct EpisodeResult { state: Episode, calls: usize, row_calls: Vec<usize>, work: crate::cocommit::CoordinationWork }
fn prepare_episode_feedback(model: &LanguageModel, c: &Config, cache: &BatchedTrainingCache, state: &mut Episode) -> Result<bool> {
    if state.previous_candidates.is_none()
        && c.sbd.as_ref().unwrap().feedback == crate::config::SbdFeedback::SoftFeedback
        && state.soft.observed.iter().zip(&state.soft.active).any(|(observed,&active)|active && observed.is_none())
        && c.sbd.as_ref().unwrap().training_mode != crate::tct::TrainingMode::Tct
        && !state.revealed.iter().any(|&revealed|revealed)
        && host(state.soft.visibility.clone())?.iter().zip(&state.soft.observed).all(|(&visibility,observed)|observed.is_some() || visibility==0.0) {
        let teacher=model.cached_blocks(cache,&state.soft,false);
        crate::sbd::update_feedback(model,c,teacher,&mut state.soft,1.0)?;
        return Ok(true);
    }
    Ok(false)
}
fn apply_confidence_reveals(state: &mut Episode, before: &Episode, logs: Tensor<2>, round: usize) -> Result<()> {
    let width=state.width();
    let confidence=host(logs.exp().max_dim(1).reshape([state.values.len()]))?;
    for row in 0..state.lengths.len() {
        let range=row*width..row*width+state.lengths[row];
        let proposals=state.values[range.clone()].to_vec();
        state.values[range.clone()].copy_from_slice(&before.values[range.clone()]);
        crate::tct::confidence_reveal(round,&state.soft.observed[range.clone()],&mut state.values[range.clone()],
            &mut state.revealed[range.clone()],&proposals,&confidence[range])?;
    }
    Ok(())
}
fn episode(model: &LanguageModel, c: &Config, cache: &BatchedTrainingCache, mut state: Episode,
    keys: &[u64], seed: u64, energy: Option<&MaskEnergy>, adaptive: bool, device: &Device) -> Result<EpisodeResult> {
    let policy = settings(c)?.policy;
    let rows = state.lengths.len();
    let free = (0..rows).map(|row|state.eligible(row).len()).collect::<Vec<_>>();
    let confidence=policy==MaskPolicy::Confidence;
    let levels=if confidence && free.iter().any(|&n|n>0) {crate::tct::TRAIN_ROUNDS+1} else {free.iter().copied().max().unwrap_or(0)};
    let mut calls = 0;
    let mut work = crate::cocommit::CoordinationWork::default();
    let mut row_calls=vec![0;rows];
    for level in (0..levels).rev() {
        let mut previous: Vec<Option<Vec<usize>>> = vec![None;rows];
        let mut repeats = vec![0;rows];
        let mut done = (0..rows).map(|row|if confidence {free[row]==0} else {level>=free[row]}).collect::<Vec<_>>();
        for inner in 0..if confidence {1} else {CALLS_PER_LEVEL} {
            if done.iter().all(|&v|v) { break; }
            let before = state.clone();
            let decision_index=if confidence {levels-1-level} else {(levels-1-level)*CALLS_PER_LEVEL+inner};
            let output = predict(model,c,cache,&mut state,keys,seed,decision_index,device)?;
            work.add(output.work);
            if confidence {
                if decision_index<crate::tct::TRAIN_ROUNDS {
                    apply_confidence_reveals(&mut state,&before,output.logs.clone(),decision_index)?;
                }
            } else {
                let cardinalities = (0..rows).map(|row|level.min(free[row])).collect::<Vec<_>>();
                let noise = (0..rows).map(|row|level as f32/free[row].max(1) as f32).collect::<Vec<_>>();
                let terms = if policy == MaskPolicy::Uniform || cardinalities.iter().all(|&r|r==0) {None} else {
                    Some(energy_terms(energy.context("learned mask policy requires loaded energy")?,features(model,c,&state,&output,&cardinalities,&noise,device)?,&state,policy,device)?)
                };
                let sets = choose_sets(&state,&cardinalities,policy,terms.as_ref(),keys,seed,decision_index)?;
                state.apply_sets(&sets,&cardinalities,c,device)?;
            }
            remember(&mut state,&output);
            let width = state.width();
            for row in 0..rows {
                if done[row] {
                    // A finished lane retains its exact episode state. Its padded device work is charged below.
                    restore_row(&mut state,&before,row,device)?;
                    continue;
                }
                row_calls[row] += 1;
                let current = state.eligible(row).iter().map(|&i|state.values[row*width+i]).collect::<Vec<_>>();
                if previous[row].as_ref() == Some(&current) { repeats[row] += 1; } else { repeats[row] = 0; }
                previous[row] = Some(current);
                if !confidence && adaptive && repeats[row]>=PATIENCE { done[row] = true; }
            }
            calls += 1;
        }
    }
    Ok(EpisodeResult {state,calls,row_calls,work})
}
fn restore_row(state: &mut Episode, before: &Episode, row: usize, device: &Device) -> Result<()> {
    let width = state.width();
    let range = row*width..(row+1)*width;
    state.values[range.clone()].copy_from_slice(&before.values[range.clone()]);
    state.revealed[range.clone()].copy_from_slice(&before.revealed[range.clone()]);
    if let (Some(after),Some(old))=(&mut state.scoring,&before.scoring) {
        after.nll[range.clone()].copy_from_slice(&old.nll[range.clone()]);
        after.hits[range.clone()].copy_from_slice(&old.hits[range.clone()]);
    }
    let rows = state.values.len();
    let selected = Tensor::<2,Bool>::from_data(TensorData::new((0..rows).map(|i|range.contains(&i)).collect::<Vec<_>>(),[rows,1]),device);
    let k = state.soft.candidates.ids.dims()[1];
    state.soft.candidates.ids = state.soft.candidates.ids.clone().mask_where(selected.clone().expand([rows,k]),before.soft.candidates.ids.clone());
    state.soft.candidates.weights = state.soft.candidates.weights.clone().mask_where(selected.clone().expand([rows,k]),before.soft.candidates.weights.clone());
    state.soft.candidates.retained_mass = state.soft.candidates.retained_mass.clone().mask_where(selected.clone(),before.soft.candidates.retained_mass.clone());
    state.soft.visibility = state.soft.visibility.clone().mask_where(selected.clone(),before.soft.visibility.clone());
    state.soft.provisional = state.soft.provisional.clone().mask_where(selected.clone(),before.soft.provisional.clone());
    state.soft.retained = state.soft.retained.clone().mask_where(selected,before.soft.retained.clone());
    if let (Some(after),Some(old)) = (&mut state.previous_logs,&before.previous_logs) {
        *after = after.clone().slice_assign([range.clone(),0..after.dims()[1]],old.clone().slice([range.clone(),0..old.dims()[1]]));
    }
    if let (Some(after),Some(old)) = (&mut state.previous_values,&before.previous_values) {after[range.clone()].copy_from_slice(&old[range]);}
    Ok(())
}

fn prefix_fixture(input: Vec<i32>) -> Example {
    let n = input.len();
    Example {input,target:vec![0;n],weight:vec![1.0;n],groups:vec![None;n]}
}
fn prefix_cache(model: &LanguageModel, prefixes: &[Vec<i32>], device: &Device) -> Result<BatchedTrainingCache> {
    ensure!(!prefixes.is_empty() && !prefixes[0].is_empty() && prefixes.iter().all(|p|p.len()==prefixes[0].len()),"batched prefixes must have the same positive length");
    let rows = prefixes.iter().map(|prefix| {
        let mut input=prefix.clone();
        // Snapshot APIs require a position after the clean prefix. Its dummy token is never in the snapshot.
        input.push(0);
        prefix_fixture(input)
    }).collect::<Vec<_>>();
    let mut trace = CleanInputs::new(prefixes.iter().map(Vec::len));
    let _ = model.forward_profiled(&Batch::new(&rows,device),&mut trace);
    let blocks = (0..rows.len()).map(|row|(row,prefixes[row].len())).collect::<Vec<_>>();
    Ok(model.training_cache_batched(&trace,&blocks))
}
fn load_energy(model: &LanguageModel, c: &Config, device: &Device) -> Result<Option<MaskEnergy>> {
    let refinement = settings(c)?;
    if matches!(refinement.policy,MaskPolicy::Uniform|MaskPolicy::Confidence) { return Ok(None); }
    let path = refinement.energy_path();
    let metadata: Value = serde_json::from_slice(&std::fs::read(path.with_extension("fit.json"))?)?;
    let checkpoint_identity = metadata["checkpoint_identity"].as_str().context("energy source checkpoint identity is missing")?;
    mask_energy::validate_checkpoint(&path,2*c.d_model+12,checkpoint_identity,refinement.policy)?;
    ensure!(metadata["source_model_identity"].as_str()==Some(model_identity(model)?.as_str()),
        "mask energy was trained against another frozen model");
    Ok(Some(mask_energy::load(2*c.d_model+12,&path,device)?.valid()))
}

pub fn generate(model: &LanguageModel, c: &Config, prompt: &[i32], device: &Device) -> Result<Value> {
    generate_scored(model,c,prompt,None,device)
}
pub(crate) fn generate_scored(model: &LanguageModel, c: &Config, prompt: &[i32], targets: Option<(&[i32],&[f32])>, device: &Device) -> Result<Value> {
    let s = c.sbd.as_ref().context("remasking requires SBD")?;
    settings(c)?;
    ensure!(!prompt.is_empty() && prompt.iter().all(|&v|v>=0 && (v as usize)<c.vocab()),"generation requires a valid clean prefix");
    ensure!(s.block_lengths.iter().all(|&n|n>0 && n<=16),"exact remasking supports block lengths 1 through 16");
    let model = model.valid();
    let energy = if s.block_lengths.iter().any(|&n|n>1) && s.generate_tokens>1 {load_energy(&model,c,device)?} else {None};
    let mut prefix = prompt.to_vec();
    let mut generated = Vec::with_capacity(s.generate_tokens);
    let mut score=crate::sbd::DecodeScore::default();
    let mut blocks = 0;
    let mut calls = 0;
    let mut row_calls = 0;
    let mut coordination = crate::cocommit::CoordinationWork::default();
    let mut clean_cache=None;
    runtime::sync(device)?;
    let clock = Instant::now();
    while generated.len()<s.generate_tokens {
        let n = s.block_lengths[blocks%s.block_lengths.len()].min(s.generate_tokens-generated.len());
        let values = if n == 1 {
            // This is the exact existing causal decoder, with no mask head or current-token input.
            let soft = SoftState::new(c.vocab(),s.candidate_width,vec![None],vec![true],device,c.soft_selection_backend)?;
            let output = crate::sbd::block_forward(&model,&prefix,&soft,false,device);
            calls += 1;
            row_calls += 1;
            let logs=log_softmax(output.logits,2).reshape([1,c.vocab()]);
            let proposal=crate::soft::ids_host(logs.clone().argmax(1))[0] as usize;
            if let Some((targets,weights))=targets {
                score.observe(&host(logs)?,&[proposal],&[0],&targets[generated.len()..generated.len()+1],&weights[generated.len()..generated.len()+1],c.vocab());
            }
            vec![proposal as i32]
        } else {
            if clean_cache.is_none() {clean_cache=Some(prefix_cache(&model,&[prefix.clone()],device)?);}
            let mut state = Episode::new(c,vec![n],n,vec![None;n],device)?;
            if let Some((targets,weights))=targets {
                let range=generated.len()..generated.len()+n;
                state.scoring=Some(EpisodeScoring {targets:targets[range.clone()].to_vec(),weights:weights[range].to_vec(),
                    nll:vec![f64::NAN;n],hits:vec![false;n]});
            }
            let result = episode(&model,c,clean_cache.as_ref().unwrap(),state,&[blocks as u64],c.seed,energy.as_ref(),true,device)?;
            coordination.add(result.work);
            calls += result.calls;
            row_calls += result.row_calls[0];
            if let Some(scoring)=&result.state.scoring {
                for i in 0..n {
                    if scoring.weights[i]==0.0 {continue;}
                    ensure!(scoring.nll[i].is_finite(),"decoder output has no finite write-time score");
                    score.nll+=scoring.nll[i];
                    score.hits+=usize::from(scoring.hits[i]);
                    score.count+=1;
                }
            }
            result.state.values.into_iter().map(|v|v as i32).collect()
        };
        prefix.extend_from_slice(&values);
        generated.extend_from_slice(&values);
        blocks += 1;
        let remaining=s.generate_tokens-generated.len();
        let next_length=s.block_lengths[blocks%s.block_lengths.len()].min(remaining);
        if next_length>1 {
            coordination.final_replay_dispatches += 1;
            coordination.final_replay_rows += 1;
            clean_cache=Some(prefix_cache(&model,&[prefix.clone()],device)?);
        } else {
            clean_cache=None;
        }
    }
    runtime::sync(device)?;
    let seconds = clock.elapsed().as_secs_f64();
    let mut report=json!({"tokens":generated,"text":String::from_utf8_lossy(&generated.iter().map(|&v|v as u8).collect::<Vec<_>>()),
        "seconds":seconds,"finalized_tokens":generated.len(),"finalized_tokens_per_second":generated.len() as f64/seconds,
        "coordination_work":coordination,
        "denoising_passes":calls,"active_row_calls":row_calls,"blocks":blocks,"clean_history_appended_tokens":generated.len(),
        "includes_clean_cache_preparation":true,"prefill_included":true,"commit_rule":"delay_until_episode_end","policy":settings(c)?.policy,
        "levels":if settings(c)?.policy==MaskPolicy::Confidence {"four confidence reveals plus endpoint"}else{"K=eligible_count"},
        "calls_per_level_max":if settings(c)?.policy==MaskPolicy::Confidence {1}else{CALLS_PER_LEVEL},
        "patience":if settings(c)?.policy==MaskPolicy::Confidence {None}else{Some(PATIENCE)},
        "token_operator":if s.training_mode == crate::tct::TrainingMode::Tct {"correction_all_free"}else if model.codpula.is_some(){"indexed_seeded_ffbs_standard_carry"}else{"factorized_argmax_standard_carry"},
        "feedback":"retained_topk_masked_visibility_1_over_soft_steps","feature_definition":FEATURE_VERSION,"rng":RANDOM_VERSION,
        "proposal_evidence":if s.training_mode == crate::tct::TrainingMode::Tct {"permanent_O_only"}else{"permanent_O_plus_provisional_reveals"},"feedback_and_feature_evidence":"permanent_O_only",
        "marginal_queries_per_joint_call":if model.cocommit_config.is_some(){None}else{Some(1usize)},
        "ffbs_queries_per_joint_call":if model.cocommit_config.is_some(){None}else{Some(1usize)},
        "proposal_seed":c.seed,"potential_temperature":c.soft_temperature,"quality_evaluation":targets.is_some()});
    if targets.is_some() {report["decode_score"]=score.report();}
    Ok(report)
}

fn repeat_state(c: &Config, source: &Episode, row_indices: &[usize], device: &Device) -> Result<Episode> {
    let width = source.width();
    let flat_indices = row_indices.iter().flat_map(|&row|(row*width..(row+1)*width).map(|i|i as i32)).collect::<Vec<_>>();
    let observed = flat_indices.iter().map(|&i|source.soft.observed[i as usize]).collect();
    let mut state = Episode::new(c,row_indices.iter().map(|&row|source.lengths[row]).collect(),width,observed,device)?;
    state.values = flat_indices.iter().map(|&i|source.values[i as usize]).collect();
    state.revealed = flat_indices.iter().map(|&i|source.revealed[i as usize]).collect();
    let ids = Tensor::<1,Int>::from_data(TensorData::new(flat_indices.clone(),[flat_indices.len()]),device);
    state.soft.candidates = Candidates {
        ids: source.soft.candidates.ids.clone().select(0,ids.clone()),
        weights: source.soft.candidates.weights.clone().select(0,ids.clone()),
        retained_mass: source.soft.candidates.retained_mass.clone().select(0,ids.clone()),
    };
    state.soft.visibility = source.soft.visibility.clone().select(0,ids.clone());
    state.soft.provisional = source.soft.provisional.clone().select(0,ids.clone());
    state.soft.retained = source.soft.retained.clone().select(0,ids.clone());
    state.previous_logs = source.previous_logs.as_ref().map(|x|x.clone().select(0,ids.clone()));
    state.previous_candidates = source.previous_candidates.as_ref().map(|x|Candidates {
        ids:x.ids.clone().select(0,ids.clone()),weights:x.weights.clone().select(0,ids.clone()),
        retained_mass:x.retained_mass.clone().select(0,ids.clone()),
    });
    state.previous_values = source.previous_values.as_ref().map(|x|flat_indices.iter().map(|&i|x[i as usize]).collect());
    state.validate(c.vocab())?;
    Ok(state)
}

fn intervention_pair(eligible: &[usize], r: usize, logs: &[f32], values: &[usize], vocab: usize,
    weighted_swap: bool, random: &mut Rng) -> Result<(Vec<usize>,Vec<usize>)> {
    ensure!(r>0 && r<eligible.len(),"trajectory intervention requires nontrivial exact-cardinality support");
    let base = mask_energy::sample_set(&[],&[],eligible,r,MaskPolicy::Uniform,random)?;
    if !weighted_swap {
        loop {
            let alternative = mask_energy::sample_set(&[],&[],eligible,r,MaskPolicy::Uniform,random)?;
            if alternative!=base {return Ok((base,alternative));}
        }
    }
    let remove = integer(random,base.len());
    let allowed = eligible.iter().copied().filter(|i|base.binary_search(i).is_err()).collect::<Vec<_>>();
    let weights = allowed.iter().map(|&i| {
        let log_probability = logs[i*vocab+values[i]] as f64;
        ensure!(log_probability.is_finite() && log_probability<=0.0,"invalid detached proposal probability");
        Ok(1.0-log_probability.exp())
    }).collect::<Result<Vec<_>>>()?;
    let total = weights.iter().sum::<f64>();
    ensure!(total.is_finite(),"nonfinite proposal-swap mass");
    let insert = if total == 0.0 {integer(random,allowed.len())} else {
        let draw = uniform(random)*total;
        let mut cumulative = 0.0;
        weights.iter().position(|&weight| {cumulative+=weight;cumulative>draw})
            .unwrap_or_else(||weights.iter().rposition(|&weight|weight>0.0).unwrap())
    };
    let mut alternative = base.clone();
    alternative[remove] = allowed[insert];
    alternative.sort_unstable();
    ensure!(alternative!=base,"proposal corruption did not change its exact set");
    Ok((base,alternative))
}

fn endpoint_losses(model: &LanguageModel, c: &Config, cache: &BatchedTrainingCache, state: &Episode,
    targets: Vec<i32>, keys: &[u64], seed: u64, call: usize, device: &Device) -> Result<(Vec<f32>,crate::cocommit::CoordinationWork)> {
    let rows = state.lengths.len();
    let width = state.width();
    ensure!(targets.len() == rows*width,"endpoint target layout differs");
    for (i,value) in state.soft.observed.iter().enumerate() {
        ensure!(value.is_none_or(|value|value == targets[i] as usize),"endpoint target disagrees with permanent evidence");
    }
    // Endpoint scoring intentionally uses the retained soft state, never hard provisional inputs.
    let targets = Tensor::<2,Int>::from_data(TensorData::new(targets,[rows,width]),device);
    let denominators = Tensor::<1>::from_data(TensorData::new((0..rows).map(|row|state.eligible(row).len().max(1) as f32).collect::<Vec<_>>(),[rows]),device);
    let mut work=crate::cocommit::CoordinationWork::default();
    let loss = if model.cocommit_config.is_some() {
        let eligible = state.soft.active.iter().zip(&state.soft.observed)
            .map(|(&active,observed)| active && observed.is_none()).collect::<Vec<_>>();
        let (output,_marks,coordination)=coordinated_output(model,c,cache,&state.soft,state,&eligible,false,
            keys,seed ^ 0x454e44504f494e54,call,device)?;
        work=coordination;
        if model.codpula.is_some() {
            work.circuit_preparations+=1;
            work.evidence_queries+=2;
            work.message_passes+=2;
            work.circuit_query_dispatched_rows+=2*rows;
            work.circuit_query_used_rows+=2*rows;
            let (circuit,_,observed) = crate::sbd::prepare_circuit(model,output,&state.soft,c.soft_temperature)?;
            let full = Tensor::<2,Bool>::from_data(TensorData::new(state.soft.active.clone(),[rows,width]),device);
            (circuit.log_partition(targets.clone(),observed)?-circuit.log_partition(targets,full)?)/denominators
        } else {
            let eligible = Tensor::<2>::from_data(TensorData::new(eligible.iter().map(|&free|if free {1.0f32}else{0.0}).collect::<Vec<_>>(),[rows,width]),device);
            let logits = output.logits;
            -(log_softmax(logits/c.soft_temperature,2).reshape([rows*width,c.vocab()])
                .gather(1,targets.reshape([rows*width,1])).reshape([rows,width])*eligible).sum_dim(1).reshape([rows])/denominators
        }
    } else {
        let output = model.cached_blocks(cache,&state.soft,false).detach();
        if model.codpula.is_some() {
            let (circuit,_,observed) = crate::sbd::prepare_circuit(model,output,&state.soft,c.soft_temperature)?;
            let full = Tensor::<2,Bool>::from_data(TensorData::new(state.soft.active.clone(),[rows,width]),device);
            (circuit.log_partition(targets.clone(),observed)?-circuit.log_partition(targets,full)?)/denominators
        } else {
            let eligible = Tensor::<2>::from_data(TensorData::new(state.soft.active.iter().zip(&state.soft.observed)
                .map(|(&active,value)|if active && value.is_none(){1.0f32}else{0.0}).collect::<Vec<_>>(),[rows,width]),device);
            -(log_softmax(output.logits/c.soft_temperature,2).reshape([rows*width,c.vocab()])
                .gather(1,targets.reshape([rows*width,1])).reshape([rows,width])*eligible).sum_dim(1).reshape([rows])/denominators
        }
    };
    let values = host(loss)?;
    ensure!(values.iter().all(|value|value.is_finite()),"trajectory endpoint has nonfinite target-conditional NLL");
    Ok((values,work))
}

#[derive(Clone)]
struct SourceBlock { identity: usize, row: usize, start: usize, length: usize, r: usize }

/// Collects only training-source states; targets enter endpoint scoring and saved metadata, never inputs.
pub fn collect(model: &LanguageModel, c: &Config, rows: &[Example], checkpoint_identity: &str, device: &Device) -> Result<Value> {
    let refinement = settings(c)?;
    ensure!(refinement.states>0 && refinement.pairs_per_state>0 && refinement.horizon>0,"trajectory counts must be positive");
    ensure!(!rows.is_empty() && rows.iter().all(|row|row.input.len()==rows[0].input.len() && row.input.len()==row.target.len()),
        "trajectory source rows must have a common positive length");
    ensure!(rows.iter().all(|row|row.weight.iter().all(|&weight|weight==1.0)),"trajectory joint targets require unit-weight training rows");
    let lengths = c.sbd.as_ref().unwrap().block_lengths.iter().copied().filter(|&n|n>1).collect::<Vec<_>>();
    ensure!(!lengths.is_empty() && lengths.iter().all(|&n|n<=16),"collection requires an actual multi-token block of length at most 16");
    let mut available = Vec::new();
    for &length in &lengths {
        for (row,example) in rows.iter().enumerate() {
            for start in (0..example.target.len()).step_by(length) {
                let n = length.min(example.target.len()-start);
                if n>1 {available.push((row,start,n));}
            }
        }
    }
    ensure!(!available.is_empty(),"training source has no nontrivial remasking block");
    let mut groups = BTreeMap::<(usize,usize),Vec<SourceBlock>>::new();
    for identity in 0..refinement.states {
        let (row,start,length) = available[identity%available.len()];
        let r = 1+(identity/available.len()+identity)%(length-1);
        groups.entry((length,r)).or_default().push(SourceBlock {identity,row,start,length,r});
    }
    let model = model.valid();
    runtime::sync(device)?;
    let clock = Instant::now();
    let mut trace = CleanInputs::new(groups.values().flatten().map(|block|block.start+1));
    let _ = model.forward_profiled(&Batch::new(rows,device),&mut trace);
    let mut dataset_states = Vec::with_capacity(refinement.states);
    let mut records = Vec::with_capacity(refinement.states);
    let mut source_calls = 0usize;
    let mut branch_calls = 0usize;
    let mut endpoint_calls = 0usize;
    let mut device_dispatch_calls = 0usize;
    let mut source_work = crate::cocommit::CoordinationWork::default();
    let mut branch_work = crate::cocommit::CoordinationWork::default();
    let mut endpoint_work = crate::cocommit::CoordinationWork::default();
    let mut eligible_pairs = 0usize;
    let mut swap_pairs = 0usize;
    let mut uniform_pairs = 0usize;
    for ((length,r),blocks) in groups {
        let count = blocks.len();
        let positions = blocks.iter().map(|block|(block.row,block.start+1)).collect::<Vec<_>>();
        let cache = model.training_cache_batched(&trace,&positions);
        let keys = blocks.iter().map(|block|block.identity as u64).collect::<Vec<_>>();
        let mut state = Episode::new(c,vec![length;count],length,vec![None;count*length],device)?;
        let decision_call = (length-1-r)*CALLS_PER_LEVEL;
        let mut decision = None;
        for call in 0..=decision_call {
            let output = predict(&model,c,&cache,&mut state,&keys,c.seed,call,device)?;
            source_work.add(output.work);
            source_calls += count;
            device_dispatch_calls += 1;
            if call == decision_call { decision=Some(output); break; }
            let current_r = length-1-call/CALLS_PER_LEVEL;
            let cardinalities = vec![current_r;count];
            let sets = choose_sets(&state,&cardinalities,MaskPolicy::Uniform,None,&keys,c.seed,call)?;
            state.apply_sets(&sets,&cardinalities,c,device)?;
            remember(&mut state,&output);
        }
        let decision = decision.unwrap();
        let feature_values = host(features(&model,c,&state,&decision,&vec![r;count],&vec![r as f32/length as f32;count],device)?)?;
        ensure!(feature_values.iter().all(|value|value.is_finite()),"nonfinite trajectory feature");
        let proposal_logs = host(decision.logs.clone())?;
        let candidate_ids = crate::soft::ids_host(state.soft.candidates.ids.clone());
        let candidate_weights = host(state.soft.candidates.weights.clone())?;
        let candidate_visibility = host(state.soft.visibility.clone())?;
        let candidate_mass = host(state.soft.candidates.retained_mass.clone())?;
        let candidate_width = state.soft.candidates.ids.dims()[1];
        let feature_width = 2*c.d_model+12;
        let mut rank_states = (0..count).map(|row|RankState {length,
            features:feature_values[row*length*feature_width..(row+1)*length*feature_width].to_vec(),pairs:Vec::with_capacity(refinement.pairs_per_state)}).collect::<Vec<_>>();
        let mut pair_details = vec![Vec::<Value>::new();count];
        let mut branch_rows = Vec::with_capacity(count*refinement.pairs_per_state*2);
        let mut sets = Vec::with_capacity(branch_rows.capacity());
        let mut branch_keys = Vec::with_capacity(branch_rows.capacity());
        for row in 0..count {
            let eligible = state.eligible(row);
            for pair in 0..refinement.pairs_per_state {
                let identity = blocks[row].identity as u64*refinement.pairs_per_state as u64+pair as u64;
                let weighted = identity%2 == 0;
                let mut random = Rng(indexed_seed(c.seed,3,identity,decision_call));
                let (a,b) = intervention_pair(&eligible,r,&proposal_logs[row*length*c.vocab()..(row+1)*length*c.vocab()],
                    &state.values[row*length..(row+1)*length],c.vocab(),weighted,&mut random)?;
                if weighted {swap_pairs+=1;} else {uniform_pairs+=1;}
                pair_details[row].push(json!({"pair":pair,"sets":[a,b],"proposal":if weighted{"proposal_weighted_one_swap"}else{"independent_uniform_distinct"},"common_rng_identity":identity}));
                branch_rows.extend([row,row]);
                branch_keys.extend([identity,identity]);
                sets.extend([a,b]);
            }
        }
        let branch_positions = branch_rows.iter().map(|&row|positions[row]).collect::<Vec<_>>();
        let branch_cache = model.training_cache_batched(&trace,&branch_positions);
        let mut branches = repeat_state(c,&state,&branch_rows,device)?;
        branches.apply_sets(&sets,&vec![r;branch_rows.len()],c,device)?;
        let mut continuation = Vec::with_capacity(refinement.horizon);
        for step in 0..refinement.horizon {
            let call = decision_call+1+step;
            let output = predict(&model,c,&branch_cache,&mut branches,&branch_keys,c.seed,call,device)?;
            branch_work.add(output.work);
            branch_calls += branch_rows.len();
            device_dispatch_calls += 1;
            let continuation_r = length.saturating_sub(1+call/CALLS_PER_LEVEL);
            continuation.push(continuation_r);
            let cardinalities = vec![continuation_r;branch_rows.len()];
            let next = choose_sets(&branches,&cardinalities,MaskPolicy::Uniform,None,&branch_keys,c.seed,call)?;
            branches.apply_sets(&next,&cardinalities,c,device)?;
            remember(&mut branches,&output);
        }
        let targets = branch_rows.iter().flat_map(|&row| {
            let block = &blocks[row];
            rows[block.row].target[block.start..block.start+length].iter().copied()
        }).collect::<Vec<_>>();
        let (losses,scoring_work) = endpoint_losses(&model,c,&branch_cache,&branches,targets,&branch_keys,
            c.seed,decision_call+refinement.horizon+1,device)?;
        endpoint_work.add(scoring_work);
        endpoint_calls += branch_rows.len();
        device_dispatch_calls += 1;
        for row in 0..count {
            for pair in 0..refinement.pairs_per_state {
                let index = (row*refinement.pairs_per_state+pair)*2;
                let (a,b) = (losses[index] as f64,losses[index+1] as f64);
                let tolerance = 2e-5+3e-4*a.abs().max(b.abs());
                let eligible = if (a-b).abs()<=tolerance {0.0} else {1.0};
                eligible_pairs += usize::from(eligible>0.0);
                let (good,bad) = if a<=b {(sets[index].clone(),sets[index+1].clone())}else{(sets[index+1].clone(),sets[index].clone())};
                rank_states[row].pairs.push(RankPair {good,bad,eligible});
                pair_details[row][pair]["outcomes"] = json!([a,b]);
                pair_details[row][pair]["eligible"] = json!(eligible);
                pair_details[row][pair]["calls_per_branch"] = json!(refinement.horizon);
                pair_details[row][pair]["scoring_calls_per_branch"] = json!(1);
            }
            let block = &blocks[row];
            let targets = &rows[block.row].target[block.start..block.start+length];
            let provisional_errors = state.values[row*length..(row+1)*length].iter().zip(targets).filter(|(a,b)|**a!=**b as usize).count();
            records.push(json!({"identity":block.identity,"source_split":"train","source_fingerprint":rows[block.row].fingerprint(),
                "source_row":block.row,"block_start":block.start,"length":block.length,"r":block.r,"decision_call":decision_call,
                "clean_targets":targets,"permanent_O":state.soft.observed[row*length..(row+1)*length],
                "provisional_values":state.values[row*length..(row+1)*length],"provisional_reveals":state.revealed[row*length..(row+1)*length],
                "candidate_ids":candidate_ids[row*length*candidate_width..(row+1)*length*candidate_width],
                "candidate_weights":candidate_weights[row*length*candidate_width..(row+1)*length*candidate_width],
                "candidate_mass":candidate_mass[row*length..(row+1)*length],"visibility":candidate_visibility[row*length..(row+1)*length],
                "provisional_error_count_analysis_only":provisional_errors,"prefix_kind":"teacher_forced_training_prefix",
                "continuation_r":continuation,"pairs":pair_details[row]}));
        }
        dataset_states.extend(rank_states);
    }
    ensure!(dataset_states.len()==refinement.states,"trajectory collection did not produce every requested decision state");
    ensure!(branch_calls==refinement.states*refinement.pairs_per_state*2*refinement.horizon
        && endpoint_calls==refinement.states*refinement.pairs_per_state*2,
        "paired trajectories did not receive their identical complete horizon and scoring calls");
    let logical_batch_reads=device_dispatch_calls;
    if model.cocommit_config.is_some() {
        device_dispatch_calls=source_work.head_dispatches+branch_work.head_dispatches+endpoint_work.head_dispatches;
    }
    let metadata = json!({"feature_definition":FEATURE_VERSION,"feature_width":2*c.d_model+12,"seed":c.seed,"rng":RANDOM_VERSION,
        "temperature":c.soft_temperature,"soft_eta":c.soft_eta,"soft_steps":c.soft_steps,"candidate_width":c.sbd.as_ref().unwrap().candidate_width,
        "source_model_identity":model_identity(&model)?,"collection_mask_policy":"uniform",
        "outcome":"mean_clean_target_conditional_joint_nll_from_retained_soft_endpoint","endpoint_evidence":"permanent_O_only",
        "proposal_evidence":if c.sbd.as_ref().unwrap().training_mode == crate::tct::TrainingMode::Tct {"permanent_O_only"}else{"permanent_O_plus_provisional_reveals"},"feature_and_feedback_evidence":"permanent_O_only",
        "source_split":"train","source_calls":source_calls,"branch_calls":branch_calls,"endpoint_calls":endpoint_calls,
        "device_noisy_dispatches":device_dispatch_calls,"horizon":refinement.horizon,"pairs_per_state":refinement.pairs_per_state,
        "coordination_work":{"source":source_work,"branch":branch_work,"endpoint":endpoint_work},
        "logical_noisy_batch_reads":logical_batch_reads,"dispatch_unit":"batched tail/head API execution, not kernel launches",
        "proposal_weighted_swap_pairs":swap_pairs,"independent_uniform_pairs":uniform_pairs,"records":records});
    let dataset = Dataset {schema:1,feature_width:2*c.d_model+12,checkpoint_identity:checkpoint_identity.into(),states:dataset_states,metadata};
    if let Some(parent) = refinement.data_path.parent().filter(|path|!path.as_os_str().is_empty()) {std::fs::create_dir_all(parent)?;}
    let temporary = refinement.data_path.with_extension("json.tmp");
    std::fs::write(&temporary,serde_json::to_vec(&dataset)?)?;
    std::fs::rename(&temporary,&refinement.data_path)?;
    runtime::sync(device)?;
    Ok(json!({"dataset":refinement.data_path,"schema":1,"states":refinement.states,"pairs":refinement.states*refinement.pairs_per_state,
        "eligible_pairs":eligible_pairs,"horizon":refinement.horizon,"source_calls":source_calls,"branch_calls":branch_calls,
        "collection_mask_policy":"uniform","feature_definition":FEATURE_VERSION,
        "endpoint_calls":endpoint_calls,"total_noisy_row_calls":source_calls+branch_calls+endpoint_calls,"device_noisy_dispatches":device_dispatch_calls,
        "coordination_work":{"source":source_work,"branch":branch_work,"endpoint":endpoint_work},
        "logical_noisy_batch_reads":logical_batch_reads,"dispatch_unit":"batched tail/head API execution, not kernel launches",
        "proposal_weighted_swap_pairs":swap_pairs,"independent_uniform_pairs":uniform_pairs,"paired_calls_exactly_equal":true,
        "checkpoint_identity":checkpoint_identity,"seconds":clock.elapsed().as_secs_f64(),"quality_evaluation":false}))
}

/// Recovers the exact recorded source configuration without relaxing any training field.
pub(crate) fn recorded_source_config(path: &Path, saved: &crate::train::State) -> Result<(Config,PathBuf,usize)> {
    let metrics_path = path.parent().context("checkpoint run directory is missing")?.join("metrics.json");
    let metrics: Value = serde_json::from_slice(&std::fs::read(&metrics_path)
        .with_context(||format!("recorded source config is required to resolve checkpoint identity: {}",metrics_path.display()))?)?;
    let source: Config = serde_json::from_value(metrics.get("config").context("checkpoint metrics has no recorded source config")?.clone())?;
    let target_tokens = usize::try_from(metrics["target_tokens"].as_u64().context("checkpoint metrics has no recorded target-token budget")?)?;
    let source_fingerprint = source.training_fingerprint(target_tokens)?;
    ensure!(saved.schema==1 && source_fingerprint==saved.config_sha256
        && metrics["config_sha256"].as_str()==Some(saved.config_sha256.as_str()),
        "recorded source configuration does not reproduce the saved checkpoint training identity");
    Ok((source,metrics_path,target_tokens))
}

fn checkpoint_training_identity(c: &Config, path: &Path, saved: &crate::train::State) -> Result<(String,Option<PathBuf>)> {
    let mut requested = c.clone();
    requested.sbd.as_mut().context("remasking requires SBD")?.refinement = None;
    let fingerprint = requested.training_fingerprint(requested.resolved_tokens()?)?;
    if fingerprint==saved.config_sha256 {return Ok((fingerprint,None));}
    let (source,metrics_path,target_tokens) = recorded_source_config(path,saved)?;
    let source_fingerprint=saved.config_sha256.clone();
    // Decoder/rank-fit controls are runtime-only. Restore their exact recorded
    // value, including an explicitly trained combined profile, for comparison.
    requested.sbd.as_mut().unwrap().refinement =
        source.sbd.as_ref().context("recorded remasking source requires SBD")?.refinement.clone();
    // Output location and output-token count do not change the frozen denoiser.
    // Every model/training field must reproduce the verified source fingerprint.
    requested.artifact_dir = source.artifact_dir.clone();
    requested.sbd.as_mut().unwrap().generate_tokens = source.sbd.as_ref().unwrap().generate_tokens;
    ensure!(requested.training_fingerprint(target_tokens)?==source_fingerprint,
        "remasking checkpoint mismatch: a model or training field differs from its verified source configuration");
    Ok((source_fingerprint,Some(metrics_path)))
}

pub fn checkpoint_report(c: &Config, arm: &Arm, seed: u64, device: &Device, path: &Path, with_collection: bool) -> Result<Value> {
    settings(c)?;
    let saved: crate::train::State = serde_json::from_slice(&std::fs::read(path.join("state.json"))?)?;
    ensure!(saved.schema==1 && saved.arm==arm.name && saved.seed==seed,
        "remasking checkpoint schema, arm or seed changed");
    let (fingerprint,source_config_path) = checkpoint_training_identity(c,path,&saved)?;
    device.seed(seed);
    let model = crate::checkpoint::load_model(LanguageModel::new(c,arm,device)?,path.join("model.burnpack"),device)?;
    let checkpoint_identity = crate::checkpoint::file_sha256(&path.join("model.burnpack"))?;
    if !with_collection && !matches!(settings(c)?.policy,MaskPolicy::Uniform|MaskPolicy::Confidence) && c.sbd.as_ref().unwrap().generate_tokens>1
        && c.sbd.as_ref().unwrap().block_lengths.iter().any(|&n|n>1) {
        mask_energy::validate_checkpoint(&settings(c)?.energy_path(),2*c.d_model+12,&checkpoint_identity,settings(c)?.policy)?;
    }
    let corpus = crate::train::loaded_corpus(c)?;
    let rows = data::examples(c,corpus.as_ref(),seed,0,c.batch_size,"train")?;
    let mut execution = c.clone();
    execution.seed = seed;
    let mut report = if with_collection {
        collect(&model,&execution,&rows,&checkpoint_identity,device)?
    } else {
        let prompt = &rows[0].input[..rows[0].input.len().min(17)];
        generate(&model,&execution,prompt,device)?
    };
    report["checkpoint"] = json!(path);
    report["checkpoint_identity"] = json!(checkpoint_identity);
    report["frozen_training_fingerprint"] = json!(fingerprint);
    report["verified_source_config"] = json!(source_config_path);
    report["training_step"] = json!(saved.step);
    report["training_tokens"] = json!(saved.tokens);
    Ok(report)
}

/// Synchronized execution cost only, not a checkpoint quality comparison.
pub fn benchmark(c: &Config, arm: &Arm, seed: u64, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(trials>0 && c.batch_size>0,"remasking benchmark requires positive batch and trials");
    settings(c)?;
    device.seed(seed);
    let model = LanguageModel::new(c,arm,device)?.valid();
    // Scratch parameters measure the selected policy's execution cost, never generated-text quality.
    let energy = if matches!(settings(c)?.policy,MaskPolicy::Uniform|MaskPolicy::Confidence) {None}else{Some(MaskEnergy::new(2*c.d_model+12,device).valid())};
    let prefix_length = 512usize;
    let length = 8usize;
    let prefixes = (0..c.batch_size).map(|row| {
        let mut prefix = (0..prefix_length).map(|i|((i+row)%c.vocab()) as i32).collect::<Vec<_>>();
        prefix[0] = (c.vocab()-1) as i32;
        prefix
    }).collect::<Vec<_>>();
    runtime::sync(device)?;
    let prefill_clock = Instant::now();
    let cache = prefix_cache(&model,&prefixes,device)?;
    runtime::sync(device)?;
    let prefill_seconds = prefill_clock.elapsed().as_secs_f64();
    let keys = (0..c.batch_size).map(|row|row as u64).collect::<Vec<_>>();
    let mut seconds = Vec::with_capacity(trials);
    let mut refinement_seconds = Vec::with_capacity(trials);
    let mut history_append_seconds = Vec::with_capacity(trials);
    let mut dispatches = Vec::with_capacity(trials);
    let mut active_calls = Vec::with_capacity(trials);
    let mut coordination_samples=Vec::with_capacity(trials);
    for iteration in 0..warmups+trials {
        runtime::sync(device)?;
        let clock = Instant::now();
        let state = Episode::new(c,vec![length;c.batch_size],length,vec![None;c.batch_size*length],device)?;
        let result = episode(&model,c,&cache,state,&keys,seed,energy.as_ref(),true,device)?;
        runtime::sync(device)?;
        let refinement_elapsed = clock.elapsed().as_secs_f64();
        let completed_prefixes=prefixes.iter().enumerate().map(|(row,prefix)| {
            let mut completed=prefix.clone();
            completed.extend(result.state.values[row*length..(row+1)*length].iter().map(|&token|token as i32));
            completed
        }).collect::<Vec<_>>();
        std::hint::black_box(&completed_prefixes);
        runtime::sync(device)?;
        let elapsed = clock.elapsed().as_secs_f64();
        if iteration>=warmups {
            seconds.push(elapsed);
            refinement_seconds.push(refinement_elapsed);
            history_append_seconds.push(elapsed-refinement_elapsed);
            dispatches.push(result.calls);
            active_calls.push(result.row_calls);
            coordination_samples.push(result.work);
        }
    }
    Ok(json!({"shape":{"batch":c.batch_size,"clean_prefix":prefix_length,"episode_length":length,"depth":c.d_model},
        "seconds":seconds,"refinement_seconds":refinement_seconds,"history_append_seconds":history_append_seconds,
        "noisy_dispatches":dispatches,"active_row_calls":active_calls,"rectangular_row_calls":dispatches.iter().map(|&n|n*c.batch_size).collect::<Vec<_>>(),
        "coordination_work":coordination_samples,"noisy_dispatches_unit":"logical episode reads; use coordination_work for physical A/B execution",
        "method_version":2,"method":"resident clean-prefix cache; complete episode and token-history append; no discarded final clean forward",
        "warmups":warmups,"trials":trials,"prefill_seconds":prefill_seconds,"prefill_in_timed_episode":false,
        "clean_history_appended_tokens_per_row":length,"final_clean_forward_in_timed_episode":false,
        "prefill_processed_positions_per_row":prefix_length+1,"prefill_extra_position":"snapshot API dummy after the immutable prefix, never used as evidence",
        "state_initialization_included":true,"device":runtime::identity(device),"policy":settings(c)?.policy,
        "retained_feedback_device_bytes_estimate":c.batch_size*length*(16*c.sbd.as_ref().unwrap().candidate_width+4*c.vocab()+24),
        "memory_estimate_scope":"current/previous candidate IDs and weights, previous full log table, visibility and token buffers; excludes model, clean cache and circuit temporaries; not peak memory",
        "scratch_parameters":true,"energy_parameters":if energy.is_some(){"benchmark_only_scratch_resident"}else{"not_used"},"energy_training_outcome":false,
        "adaptive":true,"maximum_calls_per_row":length*CALLS_PER_LEVEL,"pair_energy_work":"O(batch*L^2) scalar terms, host exact subsets",
        "seed":seed,"quality_evaluation":false,"comparison_arms":false}))
}

fn log_sum(values: impl IntoIterator<Item=f64>) -> f64 {
    let values = values.into_iter().collect::<Vec<_>>();
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    if maximum == f64::NEG_INFINITY {return maximum;}
    maximum+values.iter().map(|&v|(v-maximum).exp()).sum::<f64>().ln()
}

fn reference_partition(circuit: &crate::codpula::PreparedCircuit, values: &[usize], fixed: &[bool]) -> Result<Vec<f64>> {
    let [rows,width,vocab] = circuit.logits.dims();
    let states = circuit.log_root.dims()[0];
    let root = host(circuit.log_root.clone())?;
    let transition = host(circuit.log_transition.clone())?;
    let emission = host(circuit.log_emission.clone())?;
    let logits = host(circuit.logits.clone())?;
    let mut result = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut alpha = root.iter().map(|&v|v as f64).collect::<Vec<_>>();
        for i in 0..circuit.lengths[row] {
            let mut next = vec![0.0;states];
            for state in 0..states {
                let incoming = if i==0 {alpha[state]} else {
                    log_sum((0..states).map(|previous|alpha[previous]+transition[(row*states+previous)*states+state] as f64))
                };
                let leaf = if fixed[row*width+i] {
                    let normalizer = log_sum((0..vocab).map(|token|emission[state*vocab+token] as f64+logits[(row*width+i)*vocab+token] as f64));
                    let token = values[row*width+i];
                    emission[state*vocab+token] as f64+logits[(row*width+i)*vocab+token] as f64-normalizer
                } else {0.0};
                next[state] = incoming+leaf;
            }
            alpha = next;
        }
        result.push(log_sum(alpha));
    }
    Ok(result)
}

fn reference_features(model: &LanguageModel, c: &Config, state: &Episode, output: &Prediction,
    cardinalities: &[usize], noise: &[f32]) -> Result<Vec<f64>> {
    let width = state.width();
    let depth = c.d_model;
    let hidden = host(output.hidden.clone())?;
    let logs = host(output.logs.clone())?;
    let table = host(model.embedding.weight.val())?;
    let mask = host(model.sbd_mask.as_ref().unwrap().val())?;
    let ids = crate::soft::ids_host(state.soft.candidates.ids.clone());
    let weights = host(state.soft.candidates.weights.clone())?;
    let mass = host(state.soft.candidates.retained_mass.clone())?;
    let visibility = host(output.input_visibility.clone())?;
    let previous_logs = state.previous_logs.clone().map(host).transpose()?;
    let previous_ids = state.previous_candidates.as_ref().map(|x|crate::soft::ids_host(x.ids.clone()));
    let previous_weights = state.previous_candidates.as_ref().map(|x|host(x.weights.clone())).transpose()?;
    let k = state.soft.candidates.ids.dims()[1];
    let mut result = Vec::with_capacity(state.values.len()*(2*depth+12));
    for flat in 0..state.values.len() {
        let row = flat/width;
        let i = flat%width;
        if !state.soft.active[flat] || state.soft.observed[flat].is_some() {
            result.extend(std::iter::repeat_n(0.0,2*depth+12));
            continue;
        }
        result.extend(hidden[flat*depth..(flat+1)*depth].iter().map(|&x|x as f64));
        for d in 0..depth {
            let expectation = (0..k).map(|j|weights[flat*k+j] as f64*table[ids[flat*k+j] as usize*depth+d] as f64).sum::<f64>()
                +(1.0-mass[flat] as f64)*mask[d] as f64;
            result.push(expectation);
        }
        let probabilities = &logs[flat*c.vocab()..(flat+1)*c.vocab()];
        let current_log = probabilities[state.values[flat]] as f64;
        let entropy = probabilities.iter().filter(|&&v|v.is_finite()).map(|&v|-(v as f64).exp()*v as f64).sum::<f64>();
        let mut sorted = probabilities.iter().map(|&v|v as f64).collect::<Vec<_>>();
        sorted.sort_by(|a,b|b.total_cmp(a));
        let (changed,log_change,l1) = if let (Some(old_logs),Some(old_values),Some(old_ids),Some(old_weights)) =
            (&previous_logs,&state.previous_values,&previous_ids,&previous_weights) {
            let mut union = BTreeMap::<i32,(f64,f64)>::new();
            for j in 0..k {
                union.entry(ids[flat*k+j]).or_default().0 += weights[flat*k+j] as f64;
                union.entry(old_ids[flat*k+j]).or_default().1 += old_weights[flat*k+j] as f64;
            }
            (if state.values[flat]!=old_values[flat]{1.0}else{0.0},
                current_log-old_logs[flat*c.vocab()+state.values[flat]] as f64,
                union.values().map(|(a,b)|(a-b).abs()).sum::<f64>())
        } else {(0.0,0.0,0.0)};
        result.extend([current_log,entropy,sorted[0]-sorted[1],mass[flat] as f64,visibility[flat] as f64,changed,log_change,l1,
            i as f64/state.lengths[row].saturating_sub(1).max(1) as f64,
            if output.input_revealed[flat]{1.0}else{0.0},cardinalities[row] as f64/state.eligible(row).len().max(1) as f64,noise[row] as f64]);
    }
    Ok(result)
}

/// Mechanism checks use real model/circuit calls and independent host-f64 feature and event references.
pub fn numerical_checks(c: &Config, device: &Device) -> Result<Value> {
    let arm = c.arms.first().context("remasking reference requires an arm")?;
    let mut check = c.clone();
    if check.sbd.is_none() {check.sbd=Some(Default::default());}
    check.sbd.as_mut().unwrap().refinement=Some(RefinementConfig::default());
    // Preserve the standalone proofs, then exercise composition independently below.
    check.sbd.as_mut().unwrap().cocommit=None;
    device.seed(check.seed);
    let model = LanguageModel::new(&check,arm,device)?.valid();
    let width = 4;
    let prefix = vec![(check.vocab()-1) as i32,1,2,3];
    let prefixes = vec![prefix.clone(),prefix.clone()];
    let cache = prefix_cache(&model,&prefixes,device)?;
    let mut state = Episode::new(&check,vec![3,2],width,vec![Some(1),None,None,None,None,None,None,None],device)?;
    let observed = state.soft.observed.clone();
    let prefix_before = host(model.forward_output_profiled(&Batch::new(&[prefix_fixture(prefix.clone())],device),&mut delta_mixers::profile::Unprofiled).logits)?;
    let keys = [23,29];
    let first = predict(&model,&check,&cache,&mut state,&keys,check.seed,0,device)?;
    let cardinalities = vec![1,1];
    let noise = vec![0.5,0.5];
    let first_actual = host(features(&model,&check,&state,&first,&cardinalities,&noise,device)?)?;
    let first_reference = reference_features(&model,&check,&state,&first,&cardinalities,&noise)?;
    let first_report = crate::sbd::compare(&first_actual,&first_reference)?;
    let candidate_ids = crate::soft::ids_host(state.soft.candidates.ids.clone());
    let candidate_weights = host(state.soft.candidates.weights.clone())?;
    state.apply_sets(&[vec![1],vec![0]],&cardinalities,&check,device)?;
    ensure!(candidate_ids==crate::soft::ids_host(state.soft.candidates.ids.clone())
        && candidate_weights==host(state.soft.candidates.weights.clone())?,"remasking deleted retained candidates");
    remember(&mut state,&first);
    let second = predict(&model,&check,&cache,&mut state,&keys,check.seed,1,device)?;
    let second_actual = host(features(&model,&check,&state,&second,&cardinalities,&noise,device)?)?;
    let second_reference = reference_features(&model,&check,&state,&second,&cardinalities,&noise)?;
    let second_report = crate::sbd::compare(&second_actual,&second_reference)?;
    ensure!(first_actual.iter().chain(&second_actual).all(|value|value.is_finite()),"feature reference produced a nonfinite valid feature");
    ensure!(state.soft.observed==observed && state.values[0]==1,"provisional refinement changed permanent evidence");
    ensure!(state.apply_sets(&[vec![0],vec![0]],&cardinalities,&check,device).is_err(),"immutable evidence entered remask support");
    ensure!(state.apply_sets(&[vec![1,1],vec![0]],&[2,1],&check,device).is_err(),"duplicate remask positions accepted");
    let mut padded_values = state.values.clone();
    let mut padded_reveals = state.revealed.clone();
    for i in 0..state.values.len() {
        if !state.soft.active[i] {padded_values[i]=usize::MAX;padded_reveals[i]=true;}
    }
    let view = provisional_input(&state.soft,&padded_values,&padded_reveals,device)?;
    ensure!(view.observed==observed && state.soft.observed==observed,"provisional embedding view changed evidence");
    let embedded = host(view.embeddings(model.embedding.weight.val(),model.sbd_mask.as_ref().unwrap().val(),false))?;
    let table = host(model.embedding.weight.val())?;
    for i in 0..state.values.len() {
        if state.revealed[i] && state.soft.active[i] && state.soft.observed[i].is_none() {
            let expected=&table[state.values[i]*check.d_model..(state.values[i]+1)*check.d_model];
            ensure!(embedded[i*check.d_model..(i+1)*check.d_model].iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()),
                "provisional reveal did not use its actual token embedding");
        }
    }
    for (row,&active) in state.soft.active.iter().enumerate() {
        if !active {ensure!(embedded[row*check.d_model..(row+1)*check.d_model].iter().all(|&v|v==0.0),"padding has a nonzero provisional embedding");}
    }
    let prefix_after = host(model.forward_output_profiled(&Batch::new(&[prefix_fixture(prefix.clone())],device),&mut delta_mixers::profile::Unprofiled).logits)?;
    ensure!(prefix_before.iter().zip(&prefix_after).all(|(a,b)|a.to_bits()==b.to_bits()),"noisy refinement changed the clean prefix");
    let paired = repeat_state(&check,&state,&[0,0],device)?;
    let uniforms = host(token_uniforms(&paired,&[77,77],check.seed,4,device))?;
    ensure!(uniforms[..width*2]==uniforms[width*2..],"paired token uniforms are not identical by indexed decision");
    let direct = token_uniforms(&state,&keys,check.seed,4,device);
    ensure!(host(direct.clone())?==host(token_uniforms(&state,&keys,check.seed,4,device))?,"indexed RNG is not reproducible");
    let single = Episode::new(&check,vec![3],width,observed[..width].to_vec(),device)?;
    ensure!(host(direct)?[..width*2]==host(token_uniforms(&single,&keys[..1],check.seed,4,device))?[..width*2],"padding or another row changed a token RNG stream");
    let candidate_logs = host(second.logs.clone())?;
    let eligible = state.eligible(0);
    let mut swap_rng=Rng(7);
    let mut independent_rng=Rng(9);
    let swap = intervention_pair(&eligible,1,&candidate_logs[..width*check.vocab()],&state.values[..width],check.vocab(),true,&mut swap_rng)?;
    let independent = intervention_pair(&eligible,1,&candidate_logs[..width*check.vocab()],&state.values[..width],check.vocab(),false,&mut independent_rng)?;
    ensure!(swap.0!=swap.1 && independent.0!=independent.1,"collection accepted duplicate branch sets");
    let targets = vec![1,2,3,0,4,5,0,0];
    let (actual_losses,_) = endpoint_losses(&model,&check,&cache,&state,targets.clone(),&keys,check.seed,7,device)?;
    let endpoint = model.cached_blocks(&cache,&state.soft,false).detach();
    let reference_losses = if model.codpula.is_some() {
        let (circuit,_,_) = crate::sbd::prepare_circuit(&model,endpoint,&state.soft,check.soft_temperature)?;
        let target_values = targets.iter().map(|&value|value as usize).collect::<Vec<_>>();
        let fixed = state.soft.observed.iter().map(Option::is_some).collect::<Vec<_>>();
        let denominator = reference_partition(&circuit,&target_values,&fixed)?;
        let numerator = reference_partition(&circuit,&target_values,&state.soft.active)?;
        denominator.iter().zip(numerator).enumerate().map(|(row,(a,b))|(a-b)/state.eligible(row).len() as f64).collect::<Vec<_>>()
    } else {
        let logits = host(endpoint.logits)?;
        (0..state.lengths.len()).map(|row|state.eligible(row).iter().map(|&i| {
            let flat=row*width+i;
            let logits=&logits[flat*check.vocab()..(flat+1)*check.vocab()];
            log_sum(logits.iter().map(|&v|v as f64/check.soft_temperature))-logits[targets[flat] as usize] as f64/check.soft_temperature
        }).sum::<f64>()/state.eligible(row).len() as f64).collect::<Vec<_>>()
    };
    let endpoint_report = crate::sbd::compare(&actual_losses,&reference_losses)?;
    let l1_soft = SoftState::new(check.vocab(),check.sbd.as_ref().unwrap().candidate_width,vec![None],vec![true],device,check.soft_selection_backend)?;
    let actual_l1 = host(crate::sbd::block_forward(&model,&prefix,&l1_soft,false,device).logits)?;
    let expected_l1 = host(model.forward_output_profiled(&Batch::new(&[prefix_fixture(prefix.clone())],device),&mut delta_mixers::profile::Unprofiled)
        .slice_positions(prefix.len()-1,1).logits)?;
    ensure!(actual_l1.iter().zip(expected_l1).all(|(a,b)|a.to_bits()==b.to_bits()),"L1 remasking bypass differs from causal prediction");
    let fixed_state = Episode::new(&check,vec![2],2,vec![Some(1),Some(2)],device)?;
    let fixed_cache = prefix_cache(&model,&[prefix.clone()],device)?;
    let fixed_result = episode(&model,&check,&fixed_cache,fixed_state,&[0],check.seed,None,true,device)?;
    ensure!(fixed_result.calls==0 && fixed_result.state.values==vec![1,2],"all-observed episode performed refinement");
    let mut co_check = check.clone();
    let co_options=co_check.sbd.as_mut().unwrap();
    co_options.cocommit=Some(crate::config::CocommitConfig::default());
    co_options.commitment=None;
    co_options.commit_count=None;
    co_options.entropy_threshold=None;
    co_options.dependence_threshold=None;
    co_options.announcement=Some(crate::config::AnnouncementConfig {count:1,..Default::default()});
    co_options.training_mode=crate::tct::TrainingMode::Sbd;
    co_options.in_block_reads=delta_mixers::sbd::InBlockReads::HolaRead;
    co_options.joint_output=crate::config::SbdJointOutput::CodpulaContextual;
    co_check.tct=None;
    co_check.validate()?;
    let mut co_arm=arm.clone();
    co_arm.cache="surprise".into();
    let mut co_model=LanguageModel::new(&co_check,&co_arm,device)?.valid();
    let states=crate::codpula::STATES;
    let vocab=crate::codpula::VOCAB;
    let emission=(0..states).flat_map(|state|(0..vocab)
        .map(move |token|if token%states==state {4.0f32}else{0.0})).collect::<Vec<_>>();
    let transition=(0..states).flat_map(|source|(0..states)
        .map(move |destination|if source==destination {6.0f32}else{0.0})).collect::<Vec<_>>();
    co_model.codpula.as_mut().context("contextual CoCommit fixture requires its joint head")?.probability=
        crate::codpula::CircuitState::from_logs(
            Tensor::from_data(TensorData::new(emission,[states,vocab]),device),
            Tensor::from_data(TensorData::new(transition,[states,states]),device),
            Tensor::zeros([states],device));
    // Exercise communication, not the initialization's nearly closed HOLA gate.
    // These changes condition only this numerical fixture; production parameters
    // and the independent standalone checks above remain unchanged.
    #[derive(Default)]
    struct CommunicationParameters {path:Vec<String>,gates:Vec<u64>,outputs:Vec<u64>}
    impl ModuleVisitor for CommunicationParameters {
        fn enter_module(&mut self,name:&str,_:&str) {self.path.push(name.to_owned());}
        fn exit_module(&mut self,_:&str,_:&str) {self.path.pop();}
        fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<D>>) {
            let name=self.path.join(".");
            if name=="cache.gate" || name.ends_with(".cache.gate") {self.gates.push(param.id.val());}
            if name=="output.weight" || name.ends_with(".output.weight") {self.outputs.push(param.id.val());}
        }
    }
    impl burn::module::ModuleMapper for CommunicationParameters {
        fn map_float<const D:usize>(&mut self,param:Param<Tensor<D>>) -> Param<Tensor<D>> {
            let value=param.val();
            if self.gates.contains(&param.id.val()) {
                Param::from_tensor(Tensor::full(value.dims(),4.0,&value.device()))
            } else if self.outputs.contains(&param.id.val()) {
                Param::from_tensor(value*4.0)
            } else {param}
        }
    }
    let tail_start=co_model.blocks.len()-co_check.sbd.as_ref().unwrap().cocommit.as_ref().unwrap().tail_layers;
    for block in &mut co_model.blocks[tail_start..] {
        let mut parameters=CommunicationParameters::default();
        block.mixer.visit(&mut parameters);
        ensure!(!parameters.gates.is_empty() && !parameters.outputs.is_empty(),
            "CoCommit communication fixture requires an active HOLA gate and output projection");
        block.mixer=block.mixer.clone().map(&mut parameters);
    }
    co_model.head.weight=Param::from_tensor(co_model.head.weight.val()*4.0);
    let marker_shape=co_model.cocommit_marker.as_ref().unwrap().val().dims();
    let marker_values=(0..marker_shape[0]).flat_map(|stream|(0..marker_shape[1])
        .map(move |channel|(((channel*7+stream*11)%19) as f32-9.0)*(1.0+stream as f32)*0.75)).collect::<Vec<_>>();
    co_model.cocommit_marker=Some(Param::from_tensor(Tensor::from_data(TensorData::new(marker_values,marker_shape),device)));
    let co_cache=prefix_cache(&co_model,&[prefix.clone(),prefix.clone()],device)?;
    let mut co_state=Episode::new(&co_check,vec![3,4],4,
        vec![Some(1),Some(2),Some(3),None,Some(4),None,None,None],device)?;
    let r_value=co_check.vocab()-1;
    co_state.values[5]=r_value;
    co_state.revealed[5]=true;
    let co_input_state=co_state.clone();
    let co_before=host(co_model.forward_output_profiled(&Batch::new(&[prefix_fixture(prefix.clone())],device),
        &mut delta_mixers::profile::Unprofiled).logits)?;
    let co_view=provisional_input(&co_state.soft,&co_state.values,&co_state.revealed,device)?;
    let co_prefix=co_model.cached_blocks_prefix_profiled(&co_cache,&co_view,false,&mut delta_mixers::profile::Unprofiled);
    let manual_a=co_model.cached_blocks_tail_profiled(&co_cache,co_prefix.clone(),None,&mut delta_mixers::profile::Unprofiled).detach();
    let a_decision_logs=if co_model.codpula.is_some() {
        let (circuit,_,_)=crate::sbd::prepare_circuit(&co_model,manual_a.clone(),&co_state.soft,co_check.soft_temperature)?;
        let tokens=Tensor::<2,Int>::from_data(TensorData::new(co_state.values.iter().map(|&v|v as i32).collect::<Vec<_>>(),[2,4]),device);
        let fixed=Tensor::<2,Bool>::from_data(TensorData::new(co_state.soft.observed.iter().zip(&co_state.revealed)
            .map(|(o,&r)|o.is_some() || r).collect::<Vec<_>>(),[2,4]),device);
        host(circuit.marginals(tokens,fixed)?.log())?
    } else {host(log_softmax(manual_a.logits.clone()/co_check.soft_temperature,2))?};
    let entropy=|position:usize| a_decision_logs[position*co_check.vocab()..(position+1)*co_check.vocab()]
        .iter().filter(|log|log.is_finite()).map(|&log|-(log as f64).exp()*log as f64).sum::<f64>();
    let marked_position=if entropy(6)<=entropy(7) {6}else{7};
    let outside_position=if marked_position==6 {7}else{6};
    let mut co_marks=vec![false;8];
    co_marks[marked_position]=true;
    let manual_b=co_model.cached_blocks_tail_profiled(&co_cache,co_prefix,
        Some(Tensor::from_data(TensorData::new(co_marks.iter().map(|&mark|if mark {1.0f32}else{0.0}).collect::<Vec<_>>(),[2,4]),device)),
        &mut delta_mixers::profile::Unprofiled).detach();
    let co_prediction=predict(&co_model,&co_check,&co_cache,&mut co_state,&[91,92],co_check.seed,3,device)?;
    let co_hidden=host(co_prediction.hidden.clone())?;
    let a_hidden=host(manual_a.features.clone())?;
    let b_hidden=host(manual_b.features.clone())?;
    let d=co_check.d_model;
    crate::sbd::compare(&co_hidden[..4*d],&a_hidden[..4*d].iter().map(|&value|value as f64).collect::<Vec<_>>())?;
    crate::sbd::compare(&co_hidden[4*d..8*d],&b_hidden[4*d..8*d].iter().map(|&value|value as f64).collect::<Vec<_>>())?;
    ensure!(co_state.values[5]==r_value && co_state.soft.observed==co_input_state.soft.observed,
        "standard CoCommit changed O or carried R");
    ensure!(co_prediction.work.b_tail_rows==2 && co_prediction.work.b_tail_used_rows==1
        && co_prediction.work.b_tail_dispatches==1 && co_prediction.work.prefix_batches==1,
        "mixed empty/nonempty announcement dispatch accounting differs");
    ensure!(co_state.soft.observed[4]==Some(4) && co_state.revealed[5]
        && !co_marks[4] && !co_marks[5] && co_marks.iter().filter(|&&mark|mark).count()==1,
        "CoCommit O/R/announcement sets were not separated");
    let paired_co_state=repeat_state(&co_check,&co_state,&[1,1],device)?;
    let paired_co_uniforms=host(token_uniforms(&paired_co_state,&[99,99],co_check.seed,7,device))?;
    ensure!(paired_co_uniforms[..8]==paired_co_uniforms[8..],
        "paired CoCommit keys did not share indexed proposal draws");
    let selected_b_manual=if co_model.codpula.is_some() {
        let (circuit,tokens,fixed)=crate::sbd::prepare_circuit(&co_model,manual_b,&co_state.soft,co_check.soft_temperature)?;
        let proposal_tokens=Tensor::<2,Int>::from_data(TensorData::new(co_input_state.values.iter().map(|&v|v as i32).collect::<Vec<_>>(),[2,4]),device);
        let proposal_fixed=Tensor::<2,Bool>::from_data(TensorData::new(co_input_state.soft.observed.iter().zip(&co_input_state.revealed)
            .map(|(o,&r)|o.is_some() || r).collect::<Vec<_>>(),[2,4]),device);
        let proposal=circuit.sample_with_uniforms(proposal_tokens,proposal_fixed,token_uniforms(&co_input_state,&[91,92],co_check.seed,3,device))?;
        ensure!(co_state.values[4..8]==proposal[1],"selected row did not use one coherent B joint proposal");
        circuit.marginals(tokens,fixed)?.reshape([8,co_check.vocab()]).log()
    } else {log_softmax(manual_b.logits/co_check.soft_temperature,2).reshape([8,co_check.vocab()])};
    let selected_a_manual=if co_model.codpula.is_some() {
        let (circuit,tokens,fixed)=crate::sbd::prepare_circuit(&co_model,manual_a,&co_state.soft,co_check.soft_temperature)?;
        circuit.marginals(tokens,fixed)?.reshape([8,co_check.vocab()]).log()
    } else {log_softmax(manual_a.logits/co_check.soft_temperature,2).reshape([8,co_check.vocab()])};
    let predicted_logs=host(co_prediction.logs.clone())?;
    let a_logs=host(selected_a_manual)?;
    let b_logs=host(selected_b_manual)?;
    let observable_difference=|position:usize| (position*co_check.vocab()..(position+1)*co_check.vocab()).any(|i| {
        let a=a_logs[i].exp();
        let b=b_logs[i].exp();
        (a-b).abs()>2e-5+3e-4*b.abs()
    });
    ensure!(observable_difference(marked_position),"nonzero announcement marker did not observably change B distribution");
    ensure!(observable_difference(outside_position),"fixture cannot reject per-token A/B splicing outside S");
    crate::sbd::compare(&predicted_logs[..4*co_check.vocab()].iter().map(|value|value.exp()).collect::<Vec<_>>(),
        &a_logs[..4*co_check.vocab()].iter().map(|&value|(value as f64).exp()).collect::<Vec<_>>())?;
    crate::sbd::compare(&predicted_logs[4*co_check.vocab()..].iter().map(|value|value.exp()).collect::<Vec<_>>(),
        &b_logs[4*co_check.vocab()..].iter().map(|&value|(value as f64).exp()).collect::<Vec<_>>())?;
    let mut manual_selected_logs = a_logs[..4*co_check.vocab()].to_vec();
    manual_selected_logs.extend_from_slice(&b_logs[4*co_check.vocab()..]);
    let mut manual_selected_hidden = a_hidden[..4*d].to_vec();
    manual_selected_hidden.extend_from_slice(&b_hidden[4*d..8*d]);
    let manual_prediction = Prediction {
        hidden:Tensor::from_data(TensorData::new(manual_selected_hidden,[2,4,d]),device),
        logs:Tensor::from_data(TensorData::new(manual_selected_logs,[8,co_check.vocab()]),device),
        input_visibility:co_view.visibility.clone(),input_revealed:co_state.revealed.clone(),
        work:crate::cocommit::CoordinationWork::default(),
    };
    let actual_energy_features = host(features(&co_model,&co_check,&co_state,&co_prediction,&[1,1],&[0.5,0.5],device)?)?;
    let manual_energy_features = host(features(&co_model,&co_check,&co_state,&manual_prediction,&[1,1],&[0.5,0.5],device)?)?;
    crate::sbd::compare(&actual_energy_features,&manual_energy_features.iter().map(|&value|value as f64).collect::<Vec<_>>())?;
    let candidate_ids_before_endpoint = crate::soft::ids_host(co_state.soft.candidates.ids.clone());
    let endpoint_gold=vec![1,2,3,0,4,5,6,7];
    let endpoint_gold_alternative=vec![1,2,3,0,4,8,9,10];
    let endpoint_prefix=co_model.cached_blocks_prefix_profiled(&co_cache,&co_state.soft,false,&mut delta_mixers::profile::Unprofiled);
    let endpoint_unmarked=co_model.cached_blocks_tail_profiled(&co_cache,endpoint_prefix.clone(),None,&mut delta_mixers::profile::Unprofiled).detach();
    let endpoint_selection_logs=if co_model.codpula.is_some() {
        let (circuit,tokens,observed)=crate::sbd::prepare_circuit(&co_model,endpoint_unmarked,&co_state.soft,co_check.soft_temperature)?;
        host(circuit.marginals(tokens,observed)?.log())?
    } else {host(log_softmax(endpoint_unmarked.logits/co_check.soft_temperature,2))?};
    let endpoint_entropy=|position:usize| endpoint_selection_logs[position*co_check.vocab()..(position+1)*co_check.vocab()]
        .iter().filter(|log|log.is_finite()).map(|&log|-(log as f64).exp()*log as f64).sum::<f64>();
    let endpoint_mark=(5..8).min_by(|&a,&b|endpoint_entropy(a).total_cmp(&endpoint_entropy(b)).then(a.cmp(&b))).unwrap();
    let mut endpoint_marks=vec![0.0f32;8];
    endpoint_marks[endpoint_mark]=1.0;
    let endpoint_marked=co_model.cached_blocks_tail_profiled(&co_cache,endpoint_prefix,
        Some(Tensor::from_data(TensorData::new(endpoint_marks,[2,4]),device)),&mut delta_mixers::profile::Unprofiled).detach();
    let manual_endpoint_losses=if co_model.codpula.is_some() {
        let (circuit,_,observed)=crate::sbd::prepare_circuit(&co_model,endpoint_marked,&co_state.soft,co_check.soft_temperature)?;
        let targets=Tensor::<2,Int>::from_data(TensorData::new(endpoint_gold.clone(),[2,4]),device);
        let full=Tensor::<2,Bool>::from_data(TensorData::new(co_state.soft.active.clone(),[2,4]),device);
        host((circuit.log_partition(targets.clone(),observed)?-circuit.log_partition(targets,full)?)
            /Tensor::<1>::from_data(TensorData::new(vec![1.0f32,3.0],[2]),device))?
    } else {
        let logs=host(log_softmax(endpoint_marked.logits/co_check.soft_temperature,2))?;
        vec![0.0,-(5..8).map(|i|logs[i*co_check.vocab()+endpoint_gold[i] as usize]).sum::<f32>()/3.0]
    };
    let (endpoint_a,endpoint_work_a)=endpoint_losses(&co_model,&co_check,&co_cache,&co_state,endpoint_gold,&[91,92],co_check.seed,8,device)?;
    let (endpoint_b,endpoint_work_b)=endpoint_losses(&co_model,&co_check,&co_cache,&co_state,endpoint_gold_alternative,&[91,92],co_check.seed,8,device)?;
    crate::sbd::compare(&endpoint_a,&manual_endpoint_losses.iter().map(|&loss|loss as f64).collect::<Vec<_>>())?;
    ensure!(endpoint_work_a.b_tail_used_rows==1 && endpoint_work_a.b_tail_rows==2
        && endpoint_work_a.ffbs_passes==endpoint_work_b.ffbs_passes,
        "endpoint gold values changed coordination work or batch membership");
    ensure!(endpoint_a.iter().chain(&endpoint_b).all(|value|value.is_finite())
        && crate::soft::ids_host(co_state.soft.candidates.ids.clone())==candidate_ids_before_endpoint,
        "endpoint gold scoring entered retained-soft inputs or mutated candidates");
    let mut correction_check=co_check.clone();
    correction_check.sbd.as_mut().unwrap().training_mode=crate::tct::TrainingMode::Tct;
    let mut correction_state=Episode::new(&correction_check,vec![3,4],4,
        vec![Some(1),Some(2),Some(3),None,Some(4),None,Some(5),Some(6)],device)?;
    correction_state.values[5]=r_value;
    correction_state.revealed[5]=true;
    let correction_before=correction_state.clone();
    let correction_view=provisional_input(&correction_state.soft,&correction_state.values,&correction_state.revealed,device)?;
    let correction_prefix=co_model.cached_blocks_prefix_profiled(&co_cache,&correction_view,false,&mut delta_mixers::profile::Unprofiled);
    let mut correction_marks=vec![0.0f32;8];
    correction_marks[5]=1.0; // The only free slot is R: correction must announce it.
    let correction_b=co_model.cached_blocks_tail_profiled(&co_cache,correction_prefix,
        Some(Tensor::from_data(TensorData::new(correction_marks,[2,4]),device)),&mut delta_mixers::profile::Unprofiled).detach();
    let correction_prediction=predict(&co_model,&correction_check,&co_cache,&mut correction_state,&[101,102],co_check.seed,11,device)?;
    if co_model.codpula.is_some() {
        let (circuit,tokens,observed)=crate::sbd::prepare_circuit(&co_model,correction_b,&correction_before.soft,co_check.soft_temperature)?;
        let proposal=circuit.sample_with_uniforms(tokens,observed,
            token_uniforms(&correction_before,&[101,102],co_check.seed,11,device))?;
        ensure!(correction_state.values[4..8]==proposal[1],
            "correction CoCommit used R as evidence or spliced A/B proposals");
    } else {
        let best=crate::soft::ids_host(correction_b.logits.reshape([8,co_check.vocab()]).argmax(1));
        ensure!(correction_state.values[5]==best[5] as usize,
            "factorized correction did not revise R from B");
    }
    ensure!(correction_prediction.work.b_tail_used_rows==1
        && correction_state.soft.observed==correction_before.soft.observed,
        "correction announcement omitted R or changed permanent O");
    let all_observed = Episode::new(&co_check,vec![3,3],4,
        vec![Some(1),Some(2),Some(3),None,Some(4),Some(5),Some(6),None],device)?;
    let all_observed_prediction = predict(&co_model,&co_check,&co_cache,&mut all_observed.clone(),&[93,94],co_check.seed,5,device)?;
    ensure!(all_observed_prediction.work.b_tail_dispatches==0
        && all_observed_prediction.work.paired_tail_dispatches==0,
        "all-empty CoCommit announcement performed marked B work");
    ensure!(host(co_model.forward_output_profiled(&Batch::new(&[prefix_fixture(prefix.clone())],device),
        &mut delta_mixers::profile::Unprofiled).logits)?==co_before,
        "CoCommit prediction or endpoint scoring mutated the clean prefix");
    let coordinated_report=json!({"coherent_B_vs_manual":true,"O_R_S_separated":true,
        "mixed_empty_nonempty_dispatch_rows":co_prediction.work.b_tail_rows,
        "mixed_empty_nonempty_used_rows":co_prediction.work.b_tail_used_rows,
        "empty_S_skipped_B":true,"endpoint_retained_soft_gold_only":true,"prefix_immutable":true,
        "o_only_B_energy_marginals":true,
        "nonzero_channel_stream_marker":true,"observable_B_difference_on_S_and_outside_S":true,
        "communication_fixture":{"in_block_reads":"hola_read","cache":"surprise","tail_cache_gate_logit":4.0,
            "tail_output_projection_scale":4.0,"head_projection_scale":4.0},
        "joint_fixture":{"selector":"codpula_contextual","vocab":vocab,"states":states,
            "diagonal_transition_logit":6.0,"matching_emission_logit":4.0,"other_logits":0.0,
            "normalized_strictly_positive_rows":true,"root":"uniform"},
        "proper_subset_announcement_size":1,"correction_announces_and_resamples_R":true,
        "endpoint_manual_retained_soft_NLL":true,
        "proposal_R_carried":true,"common_indexed_keys":true});
    let mut schedule_cases = Vec::new();
    for n in 1..=16 {
        let counts=(0..n).rev().map(|level|n-((n-level)*n)/n).collect::<Vec<_>>();
        ensure!(counts==(0..n).rev().collect::<Vec<_>>(),"K=n integer cardinality schedule changed");
        schedule_cases.push(json!({"eligible":n,"r":counts,"maximum_calls":n*CALLS_PER_LEVEL}));
    }
    Ok(json!({"first_call_features":first_report,"previous_call_features":second_report,"endpoint_conditional_nll":endpoint_report,
        "coordination":coordinated_report,
        "feature_definition":FEATURE_VERSION,"feature_width":2*check.d_model+12,"immutable_O":true,"immutable_prefix":true,
        "candidates_retained":true,"right_padding_ignored":true,"invalid_sets_rejected":true,"paired_indexed_uniforms":true,
        "independent_row_rng":true,"weighted_swap_and_independent_distinct":true,"l1_exact_causal_bypass":true,
        "all_observed_zero_calls":true,"schedule":schedule_cases,"proposal_evidence":if check.sbd.as_ref().unwrap().training_mode == crate::tct::TrainingMode::Tct {"permanent_O_only"}else{"permanent_O_plus_provisional_reveals"},
        "feedback_feature_endpoint_evidence":"permanent_O_only","device":runtime::identity(device),"quality_evaluation":false}))
}

struct ModelIdentity { path: Vec<String>, digest: Sha256, errors: Vec<String> }
impl ModelIdentity {
    fn data(&mut self, data: TensorData) {
        let path = self.path.join(".");
        self.digest.update((path.len() as u64).to_le_bytes());
        self.digest.update(path.as_bytes());
        let shape = format!("{:?}:{:?}",data.dtype(),data.shape());
        self.digest.update((shape.len() as u64).to_le_bytes());
        self.digest.update(shape.as_bytes());
        self.digest.update((data.as_bytes().len() as u64).to_le_bytes());
        self.digest.update(data.as_bytes());
    }
}
impl ModuleVisitor for ModelIdentity {
    fn enter_module(&mut self, name: &str, _kind: &str) {self.path.push(name.into());}
    fn exit_module(&mut self, _name: &str, _kind: &str) {self.path.pop();}
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        match param.val().try_into_data() {
            Ok(data)=>self.data(data),
            Err(error)=>self.errors.push(format!("model float identity readback: {error:?}")),
        }
    }
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D,Int>>) {
        match param.val().try_into_data() {
            Ok(data)=>self.data(data),
            Err(error)=>self.errors.push(format!("model integer identity readback: {error:?}")),
        }
    }
}
fn model_identity(model: &LanguageModel) -> Result<String> {
    let mut visitor=ModelIdentity {path:Vec::new(),digest:Sha256::new(),errors:Vec::new()};
    visitor.digest.update(b"remasking-logical-model-v1");
    model.visit(&mut visitor);
    ensure!(visitor.errors.is_empty(),"{}",visitor.errors.join("; "));
    Ok(format!("{:x}",visitor.digest.finalize()))
}

#[cfg(test)]
mod decoder_fix_tests {
    use super::*;
    fn fixture() -> (Config,crate::config::Arm,Device) {
        let device=Device::ndarray();
        device.seed(71);
        let c=Config {task:"lm".into(),d_model:16,layers:1,heads:4,key_dim:32,value_dim:64,
            sbd:Some(crate::config::SbdConfig {block_lengths:vec![8],refinement:Some(RefinementConfig::default()),
                feedback:crate::config::SbdFeedback::SoftFeedback,..Default::default()}),..Config::default()};
        let arm=crate::config::Arm {name:"provisional".into(),mixer:"kronlstm".into(),cache:"none".into()};
        (c,arm,device)
    }
    #[test]
    fn decoder_fix_first_refinement_prediction_uses_training_teacher_state() {
        let (c,arm,device)=fixture();
        let model=LanguageModel::new(&c,&arm,&device).unwrap();
        let cache=prefix_cache(&model,&[vec![256,13,19]],&device).unwrap();
        let mut episode=Episode::new(&c,vec![8],8,vec![None;8],&device).unwrap();
        let mut training=episode.soft.clone();
        let teacher=model.cached_blocks(&cache,&training,false);
        crate::sbd::update_feedback(&model,&c,teacher,&mut training,1.0).unwrap();
        let expected=model.cached_blocks(&cache,&training,false);
        let actual=predict(&model,&c,&cache,&mut episode,&[0],c.seed,0,&device).unwrap();
        assert_eq!(host(actual.input_visibility).unwrap(),host(training.visibility).unwrap());
        let expected=host(expected.features).unwrap().into_iter().map(f64::from).collect::<Vec<_>>();
        crate::sbd::compare(&host(actual.hidden).unwrap(),&expected).unwrap();
    }
    #[test]
    fn decoder_fix_provisional_training_consumes_self_generated_inputs_without_evidence() {
        let (c,arm,device)=fixture();
        let model=LanguageModel::new(&c,&arm,&device).unwrap();
        let cache=crate::cocommit::prepare_clean_cache(&model,&[256,13,19],&device);
        let initial=crate::sbd::state(&c,vec![None;8],&device).unwrap();
        let teacher=model.cached_block(&cache,0,&initial,false);
        let mut soft_only=initial.clone();
        crate::sbd::update_feedback(&model,&c,teacher.clone(),&mut soft_only,1.0).unwrap();
        let expected=host(soft_only.embeddings(model.embedding.weight.val(),model.sbd_mask.as_ref().unwrap().val(),false)).unwrap();
        let mut saw_reveal=false;
        for identity in 0..16 {
            let mut prepared=initial.clone();
            crate::sbd::prepare_training_feedback(&model,&c,teacher.clone(),&mut prepared,&[identity]).unwrap();
            assert!(prepared.observed.iter().all(Option::is_none));
            let visibility=host(prepared.visibility.clone()).unwrap();
            if visibility.iter().any(|&value|value==1.0) {
                let actual=host(prepared.embeddings(model.embedding.weight.val(),model.sbd_mask.as_ref().unwrap().val(),false)).unwrap();
                assert!(actual.iter().zip(&expected).any(|(a,b)|(a-b).abs()>1e-6));
                saw_reveal=true;
            }
        }
        assert!(saw_reveal,"ordinary refinement training never received provisional hard inputs");
    }
    #[test]
    fn decoder_fix_tct_decode_uses_four_monotone_confidence_reveals_and_endpoint() {
        let (mut c,arm,device)=fixture();
        c.sbd.as_mut().unwrap().training_mode=crate::tct::TrainingMode::Tct;
        c.sbd.as_mut().unwrap().joint_output=crate::config::SbdJointOutput::Codpula;
        c.sbd.as_mut().unwrap().refinement.as_mut().unwrap().policy=MaskPolicy::Confidence;
        let model=LanguageModel::new(&c,&arm,&device).unwrap();
        let cache=prefix_cache(&model,&[vec![256,13,19]],&device).unwrap();
        let initial=Episode::new(&c,vec![8],8,vec![None;8],&device).unwrap();
        let mut state=initial.clone();
        for round in 0..crate::tct::TRAIN_ROUNDS {
            let before=state.clone();
            let output=predict(&model,&c,&cache,&mut state,&[0],c.seed,round,&device).unwrap();
            let proposals=state.values.clone();
            let confidence=host(output.logs.clone().exp().max_dim(1).reshape([8])).unwrap();
            let mut expected_values=before.values.clone();
            let mut expected_revealed=before.revealed.clone();
            crate::tct::confidence_reveal(round,&before.soft.observed,&mut expected_values,&mut expected_revealed,&proposals,&confidence).unwrap();
            apply_confidence_reveals(&mut state,&before,output.logs,round).unwrap();
            assert_eq!(state.values,expected_values);
            assert_eq!(state.revealed,expected_revealed);
            for i in 0..8 {if before.revealed[i] {assert_eq!(state.values[i],before.values[i]);}}
        }
        assert!(state.revealed.iter().all(|&revealed|revealed));
        let result=episode(&model,&c,&cache,initial,&[0],c.seed,None,true,&device).unwrap();
        assert_eq!(result.calls,crate::tct::TRAIN_ROUNDS+1);
        assert!(result.state.revealed.iter().all(|&revealed|revealed));
    }
}
