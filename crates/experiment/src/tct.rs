//! Streaming joint trajectory correction. Provisional neural inputs are never circuit evidence.
use crate::{
    checkpoint,
    codpula::{CircuitFlows, CodpulaExecution},
    config::{Arm, Config, SbdBlockExecution, SbdFeedback, SbdJointOutput},
    data::{self, Batch, Corpus, Rng},
    model::{self, DenoisingOutput, LanguageModel},
    runtime,
    sbd::{SbdTrainingResult, TrainingBlock},
    sbd_training::CleanInputs,
    soft::{Candidates, SoftState},
    train,
};
use anyhow::{Context, Result, ensure};
use burn::{optim::{GradientsAccumulator, GradientsParams}, prelude::*, tensor::activation::log_softmax};
use crate::update::NeuralOptimizer as ModuleOptimizer;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{io::Write, path::Path, time::Instant};

pub const TRAIN_ROUNDS: usize = 4;
pub const REVEAL_THRESHOLD: f32 = 0.9;
const CHAIN_SCHEMA: u32 = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TctConfig {
    pub rows_per_pass: usize,
}
impl Default for TctConfig {
    fn default() -> Self {Self {rows_per_pass:1}}
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    #[default]
    Sbd,
    Tct,
}
impl TrainingMode {
    pub fn is_sbd(&self) -> bool { *self == Self::Sbd }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Chain {
    start: usize,
    round: usize,
    permanent: Vec<Option<usize>>,
    values: Vec<usize>,
    revealed: Vec<bool>,
    candidate_ids: Vec<i32>,
    candidate_weights: Vec<f32>,
    retained_mass: Vec<f32>,
    visibility: Vec<f32>,
    negative_entropy: Vec<f32>,
    prediction_available: Vec<bool>,
}
impl Chain {
    fn new(start: usize, length: usize, width: usize, vocab: usize) -> Self {
        Self {
            start, round: 0, permanent: vec![None; length], values: vec![0; length],
            revealed: vec![false; length],
            candidate_ids: (0..length).flat_map(|_| (0..width).map(|x| x as i32)).collect(),
            candidate_weights: vec![1.0 / width as f32; length * width],
            retained_mass: vec![width as f32 / vocab as f32; length], visibility: vec![0.0; length],
            negative_entropy:vec![0.0;length],prediction_available:vec![false;length],
        }
    }
    fn len(&self) -> usize { self.permanent.len() }
    /// Only still-masked slots are revealed. Earlier self-generated errors remain inputs.
    fn reveal(&mut self, proposal: &[usize], confidence: &[f32]) -> Result<()> {
        ensure!(self.round < TRAIN_ROUNDS && proposal.len() == self.len() && confidence.len() == self.len(), "invalid TCT reveal round or shape");
        ensure!(confidence.iter().all(|x| x.is_finite() && (0.0..=1.00001).contains(x)), "invalid TCT proposal confidence");
        let mut masked = (0..self.len()).filter(|&i| self.permanent[i].is_none() && !self.revealed[i]).collect::<Vec<_>>();
        let minimum = masked.len().div_ceil(TRAIN_ROUNDS - self.round);
        masked.sort_by(|&a, &b| confidence[b].total_cmp(&confidence[a]).then(a.cmp(&b)).then(proposal[a].cmp(&proposal[b])));
        for (rank, &i) in masked.iter().enumerate() {
            if rank < minimum || confidence[i] >= REVEAL_THRESHOLD {
                self.values[i] = proposal[i];
                self.revealed[i] = true;
            }
        }
        self.round += 1;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChainRow {
    input: Vec<i32>,
    target: Vec<i32>,
    chains: Vec<Chain>,
}

/// Host values only: no graph or parameter-stale prefix cache survives an update.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingState {
    schema: u32,
    seed: u64,
    rng: u64,
    next_example: usize,
    steps: usize,
    optimizer_start_step: usize,
    optimizer_schedule_steps: usize,
    unique_target_bytes: usize,
    scored_target_visits: usize,
    noisy_forward_calls: usize,
    noisy_block_evaluations: usize,
    completed_chains: usize,
    rows: Vec<ChainRow>,
    source_checkpoint: Option<String>,
    source_training_identity: Option<Value>,
}
impl TrainingState {
    fn batch(&self, device: &Device) -> Batch {
        let b = self.rows.len();
        let t = self.rows[0].input.len();
        let input = self.rows.iter().flat_map(|row|row.input.iter().copied()).collect::<Vec<_>>();
        let target = self.rows.iter().flat_map(|row|row.target.iter().copied()).collect::<Vec<_>>();
        let mut reset = vec![0.0; b*t];
        for row in 0..b {reset[row*t] = 1.0;}
        Batch {
            input: Tensor::from_data(TensorData::new(input,[b,t]),device),
            target: Tensor::from_data(TensorData::new(target,[b,t]),device),
            weight: Tensor::ones([b,t],device),
            valid: Tensor::ones([b,t],device),
            reset: Tensor::from_data(TensorData::new(reset,[b,t]),device),
        }
    }
    pub fn new(c: &Config, corpus: Option<&Corpus>, seed: u64) -> Result<Self> {
        validate_config(c)?;
        let mut state = Self {schema: CHAIN_SCHEMA, seed, rng: seed ^ 0x746374636861696e, next_example: 0, steps: 0,
            unique_target_bytes: 0, scored_target_visits: 0, noisy_forward_calls: 0, noisy_block_evaluations: 0,
            optimizer_start_step:0,optimizer_schedule_steps:c.resolved_tokens()?.div_ceil(c.tokens_per_step()),
            completed_chains: 0, rows: Vec::with_capacity(c.batch_size), source_checkpoint: None, source_training_identity: None};
        for _ in 0..c.batch_size {
            let row = state.load_row(c, corpus)?;
            state.rows.push(row);
        }
        Ok(state)
    }
    fn next_random(&mut self) -> u64 {
        let mut rng = Rng(self.rng);
        let result = rng.next();
        self.rng = rng.0;
        result
    }
    fn load_row(&mut self, c: &Config, corpus: Option<&Corpus>) -> Result<ChainRow> {
        let example = data::examples(c, corpus, self.seed, self.next_example, 1, "train")?.pop().context("TCT loader returned no row")?;
        ensure!(example.weight.iter().all(|&x| x == 1.0), "joint TCT requires unit target weights");
        let lengths = &c.sbd.as_ref().unwrap().block_lengths;
        let length = lengths[(self.next_random() % lengths.len() as u64) as usize];
        let width = c.sbd.as_ref().unwrap().candidate_width.min(c.vocab());
        let chains = (0..c.context).step_by(length).map(|start| Chain::new(start, length.min(c.context-start), width, c.vocab())).collect();
        self.next_example += 1;
        self.unique_target_bytes += c.context;
        Ok(ChainRow {input: example.input, target: example.target, chains})
    }
    fn refill(&mut self, c: &Config, corpus: Option<&Corpus>) -> Result<()> {
        for index in 0..self.rows.len() {
            if self.rows[index].chains.iter().all(|chain| chain.round == TRAIN_ROUNDS + 1) {
                self.completed_chains += self.rows[index].chains.len();
                self.rows[index] = self.load_row(c, corpus)?;
            }
        }
        Ok(())
    }
    fn validate(&self, c: &Config, seed: u64) -> Result<()> {
        validate_config(c)?;
        ensure!(self.schema == CHAIN_SCHEMA && self.seed == seed && self.rows.len() == c.batch_size, "TCT state schema, seed or row count mismatch");
        ensure!(self.next_example >= c.batch_size && self.unique_target_bytes == self.next_example * c.context
            && self.scored_target_visits == self.steps * c.tokens_per_step(), "TCT loader/visit counters disagree");
        let k = c.sbd.as_ref().unwrap().candidate_width.min(c.vocab());
        for row in &self.rows {
            ensure!(row.input.len() == c.context && row.target.len() == c.context && row.input[0] == 256
                && row.input[1..] == row.target[..c.context-1], "invalid saved clean TCT row");
            ensure!(row.input.iter().chain(&row.target).all(|&x| x >= 0 && (x as usize) < c.vocab()), "saved TCT token outside vocabulary");
            let mut start = 0;
            for chain in &row.chains {
                let n = chain.len();
                ensure!(n > 0 && chain.start == start && start+n <= c.context && chain.round <= TRAIN_ROUNDS, "invalid saved TCT chain layout/counter");
                ensure!(chain.values.len() == n && chain.revealed.len() == n && chain.retained_mass.len() == n && chain.visibility.len() == n
                    && chain.candidate_ids.len() == n*k && chain.candidate_weights.len() == n*k
                    && chain.negative_entropy.len()==n && chain.prediction_available.len()==n, "invalid saved TCT feedback dimensions");
                ensure!(chain.values.iter().all(|&x| x < c.vocab()) && chain.candidate_ids.iter().all(|&x| x >= 0 && (x as usize) < c.vocab()), "saved TCT candidate outside vocabulary");
                ensure!(chain.candidate_weights.iter().all(|x| x.is_finite() && *x >= 0.0) && chain.visibility.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)), "invalid saved TCT feedback values");
                ensure!(chain.negative_entropy.iter().all(|x|x.is_finite() && *x<=0.0),"invalid saved TCT negative entropy");
                for i in 0..n {
                    ensure!(chain.permanent[i].is_none_or(|x| x == row.target[start+i] as usize), "permanent TCT evidence differs from target");
                    let weight_sum = chain.candidate_weights[i*k..(i+1)*k].iter().sum::<f32>();
                    ensure!((weight_sum-1.0).abs() <= 2e-6, "saved TCT candidate weights are not normalized");
                    // Retained mass is measured before top-k normalization, not the sum above.
                    ensure!(chain.retained_mass[i].is_finite() && chain.retained_mass[i] > 0.0
                        && chain.retained_mass[i] <= 1.00001, "invalid saved TCT retained probability mass");
                }
                if chain.round == TRAIN_ROUNDS {
                    ensure!((0..n).all(|i| chain.permanent[i].is_some() || chain.revealed[i]), "TCT endpoint still has masked positions");
                }
                start += n;
            }
            ensure!(start == c.context, "saved TCT chains do not cover the row");
        }
        Ok(())
    }
}

fn validate_config(c: &Config) -> Result<()> {
    let s = c.sbd.as_ref().context("TCT requires SBD configuration")?;
    ensure!(s.training_mode == TrainingMode::Tct && !s.joint_output.is_factorized() && c.task == "lm", "TCT is the explicit joint LM training mode");
    ensure!(s.feedback == SbdFeedback::SoftFeedback && !c.hard_mask_only, "TCT requires retained soft feedback on unresolved masks");
    ensure!(!s.block_lengths.is_empty() && s.block_lengths.iter().all(|&n| n > 0) && s.candidate_width > 0, "invalid TCT lengths or feedback settings");
    Ok(())
}

fn input_state(model:&LanguageModel, c: &Config, state: &TrainingState, blocks: &[(usize, usize)], width: usize, device: &Device) -> Result<SoftState> {
    let k = c.sbd.as_ref().unwrap().candidate_width.min(c.vocab());
    let count = blocks.len()*width;
    let mut permanent = Vec::with_capacity(count);
    let mut active = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    let mut revealed = Vec::with_capacity(count);
    let mut ids = Vec::with_capacity(count*k);
    let mut weights = Vec::with_capacity(count*k);
    let mut mass = Vec::with_capacity(count);
    let mut visibility = Vec::with_capacity(count);
    for &(row, index) in blocks {
        let chain = &state.rows[row].chains[index];
        for i in 0..width {
            let real = i < chain.len();
            permanent.push(if real {chain.permanent[i]} else {None});
            active.push(real);
            values.push(if real {chain.values[i]} else {0});
            revealed.push(real && chain.revealed[i]);
            if real {
                if let Some(id) = chain.permanent[i] {
                    ids.extend((0..k).map(|j|if j == 0 {id as i32} else {0}));
                    weights.extend((0..k).map(|j|if j == 0 {1.0} else {0.0}));
                    mass.push(1.0);
                    visibility.push(1.0);
                } else {
                    ids.extend_from_slice(&chain.candidate_ids[i*k..(i+1)*k]);
                    weights.extend_from_slice(&chain.candidate_weights[i*k..(i+1)*k]);
                    mass.push(chain.retained_mass[i]);
                    visibility.push(chain.visibility[i]);
                }
            } else {
                ids.extend(std::iter::repeat_n(0, k));
                weights.extend(std::iter::repeat_n(0.0, k));
                mass.push(0.0);
                visibility.push(0.0);
            }
        }
    }
    let mut soft = SoftState::new(c.vocab(), k, permanent, active, device, c.soft_selection_backend)?;
    soft.candidates = Candidates {
        ids: Tensor::from_data(TensorData::new(ids, [count,k]), device),
        weights: Tensor::from_data(TensorData::new(weights, [count,k]), device),
        retained_mass: Tensor::from_data(TensorData::new(mass, [count,1]), device),
    };
    soft.visibility = Tensor::from_data(TensorData::new(visibility, [count,1]), device);
    let entropy=blocks.iter().flat_map(|&(r,j)|(0..width).map(move |i|state.rows[r].chains[j].negative_entropy.get(i).copied().unwrap_or(0.0))).collect::<Vec<_>>();
    let available=blocks.iter().flat_map(|&(r,j)|(0..width).map(move |i|state.rows[r].chains[j].prediction_available.get(i).copied().unwrap_or(false))).collect::<Vec<_>>();
    soft.negative_entropy=Tensor::from_data(TensorData::new(entropy,[count,1]),device);
    soft.prediction_available=Tensor::from_data(TensorData::new(available,[count,1]),device);
    soft.refresh_feedback(model.soft_confidence.as_ref().context("soft feedback head missing")?);
    crate::remask::provisional_input(&soft, &values, &revealed, device)
}

fn host<const D: usize>(x: Tensor<D>) -> Result<Vec<f32>> {
    x.into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("reading TCT values: {e:?}"))
}

struct Loss {
    loss: Tensor<1>,
    flows: Option<CircuitFlows>,
    clean: f64,
    noisy_blocks: usize,
    joint: f64,
    chains: usize,
    work: crate::cocommit::CoordinationWork,
}

/// Both execution layouts share the exact same neural inputs, O-only evidence and loss.
fn training_loss(c: &Config, model: &LanguageModel, state: &mut TrainingState, device: &Device, layout: SbdBlockExecution, bundles: Option<&[Vec<bool>]>) -> Result<Loss> {
    let chains=state.rows.iter().map(|row|row.chains.len()).sum::<usize>();
    training_loss_reduced(c,model,state,device,layout,bundles,chains as f64)
}
fn training_loss_reduced(c:&Config,model:&LanguageModel,state:&mut TrainingState,device:&Device,layout:SbdBlockExecution,bundles:Option<&[Vec<bool>]>,denominator:f64)->Result<Loss> {
    let batch=state.batch(device);
    let mut trace=CleanInputs::new(state.rows.iter().flat_map(|row|row.chains.iter().map(|chain|chain.start+1)));
    let logits=model.forward_profiled(&batch,&mut trace);
    let clean_loss=model.loss(logits,&batch);
    let clean=model::scalar(clean_loss.clone())?;
    let mut loss=clean_loss;
    let blocks=state.rows.iter().enumerate().flat_map(|(r,row)|(0..row.chains.len()).map(move |i|(r,i))).collect::<Vec<_>>();
    let count=blocks.len();
    let width=blocks.iter().map(|&(r,i)|state.rows[r].chains[i].len()).max().context("TCT requires chains")?;
    let soft=input_state(model,c,state,&blocks,width,device)?;
    let lengths=blocks.iter().map(|&(r,i)|state.rows[r].chains[i].len()).collect::<Vec<_>>();
    let mut targets=Vec::with_capacity(count*width);
    for &(r,i) in &blocks {
        let chain=&state.rows[r].chains[i];
        targets.extend((0..width).map(|j|if j<chain.len(){state.rows[r].target[chain.start+j]}else{0}));
    }
    let targets=Tensor::<2,Int>::from_data(TensorData::new(targets,[count,width]),device);
    let observed=Tensor::<2,Bool>::from_data(TensorData::new(soft.observed.iter().map(Option::is_some).collect::<Vec<_>>(),[count,width]),device);
    let locations=blocks.iter().map(|&(r,i)|(r,state.rows[r].chains[i].start+1)).collect::<Vec<_>>();
    let mut work=crate::cocommit::CoordinationWork::default();
    let mut selected_marks=vec![false;count*width];
    let (a,b)=if c.sbd.as_ref().unwrap().cocommit.is_some() {
        ensure!(layout==SbdBlockExecution::Batched,"TCT CoCommit requires batched execution");
        let bundles=bundles.context("TCT CoCommit bundle missing")?;
        ensure!(bundles.len()==count,"TCT bundle count differs");
        let padded=bundles.iter().zip(&lengths).map(|(bundle,&n)|{
            ensure!(bundle.len()==n,"TCT bundle width differs");
            Ok(bundle.iter().copied().chain(std::iter::repeat_n(false,width-n)).collect::<Vec<_>>())
        }).collect::<Result<Vec<_>>>()?;
        selected_marks=padded.iter().flatten().copied().collect();
        let cache=model.training_cache_batched(&trace,&locations);
        let (a,b,pair_work)=crate::cocommit::training_pair_profiled(model,c,&cache,&soft,&padded,&mut delta_mixers::profile::Unprofiled)?;
        work=pair_work;
        (a,Some(b))
    }else{
        let output=match layout {
            SbdBlockExecution::Batched=>model.cached_blocks(&model.training_cache_batched(&trace,&locations),&soft,false),
            SbdBlockExecution::PerBlock=>{
                let mut outputs=Vec::with_capacity(count);
                for &(r,i) in &blocks {
                    let cache=model.training_cache(&trace,r,&[state.rows[r].chains[i].start+1]);
                    outputs.push(model.cached_block(&cache,0,&input_state(model,c,state,&[(r,i)],width,device)?,false));
                }
                DenoisingOutput{logits:Tensor::cat(outputs.iter().map(|o|o.logits.clone()).collect(),0),features:Tensor::cat(outputs.into_iter().map(|o|o.features).collect(),0)}
            }
        };
        (output,None)
    };
    let (circuit,_)=crate::sbd::prepare_distribution(model,a,&soft,1.0)?;
    let weights=Tensor::from_data(TensorData::new(soft.active.iter().zip(&soft.observed).map(|(&active,o)|if active&&o.is_none(){1.0f32}else{0.0}).collect::<Vec<_>>(),[count,width]),device);
    let correction=circuit.joint_marginal_ce(targets.clone(),targets.clone(),observed.clone(),weights,denominator)?;
    let mut joint=model::scalar(correction.loss.clone())?;
    loss=loss+correction.loss;
    let mut flows=correction.flows;
    let mut probabilities=circuit.marginals(targets.clone(),observed.clone())?.detach();
    if let Some(b)=b {
        let use_b=blocks.iter().enumerate().flat_map(|(block,&(r,i))|{
            let marks=&selected_marks[block*width..(block+1)*width];
            let used=marks.iter().any(|&m|m);
            let chain=&state.rows[r].chains[i];
            (0..width).map(move |j|used && j<chain.len() && (marks[j]||chain.revealed[j]))
        }).collect::<Vec<_>>();
        let (b_circuit,_)=crate::sbd::prepare_distribution(model,b,&soft,1.0)?;
        let weights=Tensor::from_data(TensorData::new(use_b.iter().map(|&used|if used{1.0f32}else{0.0}).collect::<Vec<_>>(),[count,width]),device);
        let auxiliary=b_circuit.joint_marginal_ce(targets.clone(),targets.clone(),observed.clone(),weights,denominator)?;
        joint+=model::scalar(auxiliary.loss.clone())?;
        loss=loss+auxiliary.loss;
        flows.add(auxiliary.flows);
        let selected=Tensor::<2,Bool>::from_data(TensorData::new(use_b,[count,width]),device).unsqueeze_dim::<3>(2).expand([count,width,c.vocab()]);
        probabilities=probabilities.mask_where(selected,b_circuit.marginals(targets,observed)?.detach());
    }
    let confidence=host(probabilities.clone().max_dim(2).reshape([count*width]))?;
    let proposals=crate::soft::ids_host(probabilities.clone().reshape([count*width,c.vocab()]).argmax(1));
    let mut feedback=soft;
    feedback.update(probabilities.reshape([count*width,c.vocab()]).log(),1.0,c.sbd.as_ref().unwrap().candidate_width,
        model.soft_confidence.as_ref().context("soft feedback head missing")?)?;
    let k=c.sbd.as_ref().unwrap().candidate_width.min(c.vocab());
    let ids=feedback.candidates.ids.into_data().try_to_vec_as::<i32>().map_err(|e|anyhow::anyhow!("TCT candidate IDs: {e:?}"))?;
    let weights=host(feedback.candidates.weights)?;
    let mass=host(feedback.candidates.retained_mass)?;
    let entropy=host(feedback.negative_entropy)?;
    let visibility=host(feedback.visibility)?;
    for (block,&(r,i)) in blocks.iter().enumerate() {
        let chain=&mut state.rows[r].chains[i];
        let n=chain.len();
        if chain.round==TRAIN_ROUNDS {chain.round+=1;continue;}
        chain.reveal(&proposals[block*width..block*width+n].iter().map(|&id|id as usize).collect::<Vec<_>>(),&confidence[block*width..block*width+n])?;
        chain.candidate_ids.copy_from_slice(&ids[block*width*k..block*width*k+n*k]);
        chain.candidate_weights.copy_from_slice(&weights[block*width*k..block*width*k+n*k]);
        chain.retained_mass.copy_from_slice(&mass[block*width..block*width+n]);
        chain.negative_entropy.copy_from_slice(&entropy[block*width..block*width+n]);
        chain.visibility.copy_from_slice(&visibility[block*width..block*width+n]);
        chain.prediction_available.fill(true);
    }
    Ok(Loss{loss,flows:Some(flows),clean,joint,chains:count,noisy_blocks:count,work})
}

struct Backward {
    gradients: GradientsParams,
    flows: Option<CircuitFlows>,
    loss: f64,
    clean: f64,
    joint: f64,
    chains: usize,
    noisy_blocks: usize,
    clean_calls: usize,
    noisy_calls: usize,
    work: crate::cocommit::CoordinationWork,
    gradient_report: model::GradientReport,
}
fn sample_announcements(c:&Config,state:&mut TrainingState)->Option<Vec<Vec<Vec<bool>>>> {
    let options=c.sbd.as_ref().unwrap().cocommit.as_ref()?;
    let seed=state.next_random();
    let mut blocks=Vec::new();
    for (row,saved) in state.rows.iter().enumerate() {
        for chain in &saved.chains {
            blocks.push(TrainingBlock {row,start:chain.start,length:chain.len(),
                masked:chain.permanent.iter().zip(&chain.revealed).map(|(o,&r)|o.is_none()&&!r).collect()});
        }
    }
    let selected=crate::cocommit::sample_bundles(&blocks,options.training_bundle_max,seed);
    let mut output=state.rows.iter().map(|row|row.chains.iter().map(|_|Vec::new()).collect::<Vec<_>>()).collect::<Vec<_>>();
    for (block,bundle) in blocks.iter().zip(selected) {
        let index=state.rows[block.row].chains.iter().position(|chain|chain.start==block.start).unwrap();
        output[block.row][index]=bundle;
    }
    Some(output)
}
fn backward_passes(c:&Config,model:&LanguageModel,state:&mut TrainingState,device:&Device,layout:SbdBlockExecution)->Result<Backward> {
    let rows_per_pass=c.tct.as_ref().map_or(c.batch_size,|tct|tct.rows_per_pass).min(c.batch_size);
    ensure!(rows_per_pass>0 && state.rows.len()==c.batch_size,"invalid TCT row accumulation shape");
    let confidence_used=state.rows.iter().flat_map(|row|&row.chains).any(|chain|
        (0..chain.len()).any(|i|chain.permanent[i].is_none() && !chain.revealed[i] && chain.prediction_available[i]));
    let sampled_bundles=sample_announcements(c,state);
    let logical_chains=state.rows.iter().map(|row|row.chains.len()).sum::<usize>();
    if rows_per_pass==c.batch_size {
        let bundles=sampled_bundles.as_ref().map(|all|state.rows.iter().enumerate().flat_map(|(row,saved)|
            saved.chains.iter().enumerate().map(move |(index,_)|all[row][index].clone())).collect::<Vec<_>>());
        let result=training_loss(c,model,state,device,layout,bundles.as_deref())?;
        let loss=model::scalar(result.loss.clone())?;
        let mut unused = model::unused_denoising_parameters(result.noisy_blocks > 0, result.work.b_tail_used_rows > 0);
        if !confidence_used {unused.extend(model::unused_soft_parameters());}
        let (gradients, gradient_report)=model::checked_gradients_with_unused(model,result.loss,&unused,&mut delta_mixers::profile::Unprofiled)?;
        return Ok(Backward {gradients,flows:result.flows,loss,clean:result.clean,joint:result.joint,
            chains:result.chains,noisy_blocks:result.noisy_blocks,clean_calls:1,
            noisy_calls:match layout {SbdBlockExecution::Batched=>usize::from(result.noisy_blocks>0),SbdBlockExecution::PerBlock=>result.noisy_blocks},
            work:result.work,gradient_report});
    }
    let mut rows=std::mem::take(&mut state.rows).into_iter();
    let mut returned=Vec::with_capacity(c.batch_size);
    let mut accumulator=GradientsAccumulator::<LanguageModel>::new();
    let mut flows=None::<CircuitFlows>;
    let (mut loss,mut clean,mut joint)=(0.0,0.0,0.0);
    let (mut chains,mut noisy_blocks,mut clean_calls,mut noisy_calls)=(0,0,0,0);
    let mut work=crate::cocommit::CoordinationWork::default();
    loop {
        let selected=rows.by_ref().take(rows_per_pass).collect::<Vec<_>>();
        if selected.is_empty(){break;}
        let row_offset=returned.len();
        let mut pass_state=state.clone();
        pass_state.rows=selected;
        let mut pass_config=c.clone();
        pass_config.batch_size=pass_state.rows.len();
        let scale=pass_config.batch_size as f64/c.batch_size as f64;
        let pass_bundles=sampled_bundles.as_ref().map(|all| {
            pass_state.rows.iter().enumerate().flat_map(|(local,row)| {
                row.chains.iter().enumerate()
                    .map(move |(index,_)|all[row_offset+local][index].clone())
            }).collect::<Vec<_>>()
        });
        let result=training_loss_reduced(&pass_config,model,&mut pass_state,device,layout,
            pass_bundles.as_deref(),logical_chains as f64*scale)?;
        loss+=model::scalar(result.loss.clone())?*scale;
        clean+=result.clean*scale;
        joint+=result.joint*scale;
        chains+=result.chains;
        noisy_blocks+=result.noisy_blocks;
        clean_calls+=1;
        noisy_calls+=match layout {SbdBlockExecution::Batched=>usize::from(result.noisy_blocks>0),SbdBlockExecution::PerBlock=>result.noisy_blocks};
        work.add(result.work);
        accumulator.accumulate(model,GradientsParams::from_grads((result.loss*scale).backward(),model));
        if let Some(part)=result.flows {if let Some(total)=&mut flows {total.add(part);} else {flows=Some(part);}}
        returned.append(&mut pass_state.rows);
    }
    state.rows=returned;
    let gradients=accumulator.grads();
    let mut unused = model::unused_denoising_parameters(noisy_blocks > 0, work.b_tail_used_rows > 0);
    if !confidence_used {unused.extend(model::unused_soft_parameters());}
    let gradient_report=model::check_parameter_gradients_with_unused(model,&gradients,&unused,&mut delta_mixers::profile::Unprofiled)?;
    Ok(Backward {gradients,flows,loss,clean,joint,chains,noisy_blocks,clean_calls,noisy_calls,work,gradient_report})
}

/// One actual optimizer step, including detached pre-update chain advancement and refill.
/// The default path always uses the block-axis cache; the timing helper alone selects its oracle layout.
pub fn step(c: &Config, model: LanguageModel, optim: &mut ModuleOptimizer, state: &mut TrainingState, corpus: Option<&Corpus>, device: &Device) -> Result<(LanguageModel, Value)> {
    step_layout(c,model,optim,state,corpus,device,SbdBlockExecution::Batched)
}
fn step_layout(c: &Config, model: LanguageModel, optim: &mut ModuleOptimizer, state: &mut TrainingState, corpus: Option<&Corpus>, device: &Device, layout: SbdBlockExecution) -> Result<(LanguageModel, Value)> {
    let result = backward_passes(c,&model,state,device,layout)?;
    let model = train::optimizer_step(c,model,optim,result.gradients,result.flows)?;
    state.steps += 1;
    state.scored_target_visits += c.tokens_per_step();
    state.noisy_forward_calls += result.noisy_calls;
    state.noisy_block_evaluations += result.noisy_blocks;
    state.refill(c,corpus)?;
    Ok((model,json!({"loss":result.loss,"clean_causal_nll_per_token":result.clean,
        "gradient_counts":result.gradient_report,
        "correction_and_coordination_ce_chain_mean_token_sum":result.joint,
        "correction_objective":"A token CE over all mutable positions; optional B coordination CE on masked S and visible mutable positions",
        "chain_count":result.chains,"steps":state.steps,"unique_target_bytes_loaded":state.unique_target_bytes,
        "scored_target_visits":state.scored_target_visits,"completed_chains":state.completed_chains,"next_loader_example":state.next_example,
        "clean_forward_calls":state.steps*result.clean_calls,"row_accumulation_passes":result.clean_calls,
        "rows_per_pass":c.tct.as_ref().map_or(c.batch_size,|tct|tct.rows_per_pass).min(c.batch_size),
        "noisy_forward_calls":state.noisy_forward_calls,"noisy_block_evaluations":state.noisy_block_evaluations,
        "coordination_work":result.work,"coordination_work_units":{"dispatches":"physical neural/circuit dispatches","rows":"physical processed row units; paired tails count 2C for one doubled dispatch",
            "used_rows":"rows whose branch results are consumed by losses/feedback; query-dispatch and used rows are separately tracked",
            "logical_blocks":"logical TCT chains","prefix_batches":"batched saved-prefix cache calls","circuit_preparations":"prepared A/B circuits",
            "message_passes":"circuit message/statistics traversals","evidence_queries":"objective/feedback query operations","ffbs_passes":"joint proposal sampling calls",
            "final_replay_rows":"circuit final-replay row units"}})))
}

fn save(c: &Config, path: &Path, model: &LanguageModel, optim: &ModuleOptimizer, saved: &train::State, chains: &TrainingState) -> Result<()> {
    train::checkpoint_with_tct(c,path,model,optim,saved,Some(chains))
}

fn warm_start_identity(c: &Config, path: &Path, saved: &train::State, target_tokens: usize) -> Result<Value> {
    let mut requested = c.clone();
    requested.sbd.as_mut().context("TCT warm-start SBD settings")?.training_mode = TrainingMode::Sbd;
    requested.sbd.as_mut().unwrap().refinement = None;
    requested.tct = None;
    if requested.training_fingerprint(target_tokens)? == saved.config_sha256 {
        return Ok(json!({"source_training_sha256":saved.config_sha256,"verified_source_config":Value::Null,"allowed_changes":["training_mode","sbd.refinement","tct"]}));
    }
    let (source,source_path,source_tokens) = crate::remask::recorded_source_config(path,saved)?;
    ensure!(source.sbd.as_ref().is_some_and(|s|s.training_mode == TrainingMode::Sbd && !s.joint_output.is_factorized()),
        "TCT warm-start requires an ordinary joint SBD checkpoint");
    // A new training run has its own explicit budget and output directory. Every
    // remaining field, including data, backbone, execution and optimizer, must match.
    requested.artifact_dir = source.artifact_dir.clone();
    requested.train_tokens = source.train_tokens;
    requested.budget_file = source.budget_file.clone();
    requested.sbd.as_mut().unwrap().refinement = source.sbd.as_ref().unwrap().refinement.clone();
    requested.sbd.as_mut().unwrap().rows_per_pass = source.sbd.as_ref().unwrap().rows_per_pass;
    ensure!(requested.training_fingerprint(source_tokens)? == saved.config_sha256,
        "TCT warm-start differs from its verified source beyond training_mode, decoder refinement, row accumulation, token budget or artifact_dir");
    Ok(json!({"source_training_sha256":saved.config_sha256,"verified_source_config":source_path,
        "allowed_changes":["training_mode","sbd.refinement","sbd.rows_per_pass","tct","train_tokens","budget_file","artifact_dir"]}))
}

/// Actual joint TCT training only: no held-out evaluation or generation routine is called.
pub fn run(c: &Config, arm: &Arm, seed: u64, device: &Device, resume: Option<&Path>) -> Result<Value> {
    validate_config(c)?;
    let corpus = train::loaded_corpus(c)?;
    // Fixture selection identifies the dataset; it does not run model evaluation.
    let validation = train::evaluations(c,corpus.as_ref(),"validation")?;
    let data = train::data_identity(c,corpus.as_ref(),&validation);
    let target_tokens = c.resolved_tokens()?;
    let quantum = c.tokens_per_step();
    ensure!(target_tokens >= quantum && target_tokens % quantum == 0,"TCT token budget must be a positive multiple of batch*context");
    let mut identity_config = c.clone();
    identity_config.sbd.as_mut().unwrap().refinement = None;
    let fingerprint = identity_config.training_fingerprint(target_tokens)?;
    let run_dir = c.artifact_dir.join(format!("tct-{}-seed{seed}",arm.name));
    std::fs::create_dir_all(&run_dir)?;
    device.seed(seed);
    let target = device.clone().autodiff();
    let mut model = LanguageModel::new(c,arm,&target)?;
    let mut optim = train::optimizer(c);
    let mut chains;
    let mut previous_seconds = 0.0;
    if let Some(path) = resume {
        let saved = checkpoint::load_state(path.join("state.json"))?;
        ensure!(saved.schema == 1 && saved.data == data && saved.arm == arm.name && saved.seed == seed,"TCT resume dataset, arm or seed mismatch");
        if path.join("tct.json").exists() {
            ensure!(saved.config_sha256 == fingerprint,"TCT resume training configuration mismatch");
            chains = serde_json::from_slice::<TrainingState>(&std::fs::read(path.join("tct.json"))?)?;
            chains.validate(c,seed)?;
            ensure!(saved.step == chains.steps && saved.tokens == chains.scored_target_visits,"TCT model/chain checkpoint counters disagree");
            previous_seconds = saved.training_seconds;
        } else {
            let source_identity = warm_start_identity(c,path,&saved,target_tokens)?;
            chains = TrainingState::new(c,corpus.as_ref(),seed)?;
            chains.source_checkpoint = Some(path.to_string_lossy().into_owned());
            chains.source_training_identity = Some(source_identity);
        }
        model = checkpoint::load_model(model,path.join("model.burnpack"),&target)?;
        optim=if chains.source_checkpoint.is_some() {
            checkpoint::load_optimizer_for_continuation(c,path.join("optimizer.burnpack"))?
        }else{checkpoint::load_optimizer(c,path.join("optimizer.burnpack"))?};
        if !path.join("tct.json").exists() {
            chains.optimizer_start_step=optim.completed_steps();
            chains.optimizer_schedule_steps=optim.schedule_steps();
            if let Some(identity)=&mut chains.source_training_identity {
                identity["optimizer_start_step"]=json!(chains.optimizer_start_step);
                identity["optimizer_source_schedule_steps"]=json!(chains.optimizer_schedule_steps);
            }
        }
        ensure!(optim.completed_steps()==chains.optimizer_start_step+chains.steps
            && optim.schedule_steps()==chains.optimizer_schedule_steps,"TCT optimizer/chain schedule or counters disagree");
    } else {
        chains = TrainingState::new(c,corpus.as_ref(),seed)?;
    }
    let start_visits = chains.scored_target_visits;
    let started = Instant::now();
    let mut last_loss = Value::Null;
    let mut work_this_invocation=crate::cocommit::CoordinationWork::default();
    let mut gradient_counts = model::GradientReport::new();
    let mut log = std::fs::OpenOptions::new().create(true).write(true).append(resume.is_some()).truncate(resume.is_none()).open(run_dir.join("training.jsonl"))?;
    while chains.scored_target_visits < target_tokens && started.elapsed().as_secs_f64() < c.max_seconds {
        let result = step(c,model,&mut optim,&mut chains,corpus.as_ref(),&target)?;
        model = result.0;
        last_loss = result.1;
        model::accumulate_gradient_report(&mut gradient_counts, serde_json::from_value(last_loss["gradient_counts"].clone())?);
        if let Some(work)=last_loss.get("coordination_work") {
            work_this_invocation.add(serde_json::from_value(work.clone())?);
        }
        runtime::sync(&target)?;
        writeln!(log,"{}",json!({"step":chains.steps,"target_visits":chains.scored_target_visits,"loss":last_loss,"elapsed_seconds":started.elapsed().as_secs_f64()}))?;
        if chains.steps % c.checkpoint_every == 0 {
            let saved = train::State {schema: 1,config_sha256: fingerprint.clone(),data: data.clone(),arm: arm.name.clone(),seed,
                step: chains.steps,tokens: chains.scored_target_visits,training_seconds: previous_seconds+started.elapsed().as_secs_f64(),initial_evaluation: Value::Null};
            save(c,&run_dir.join(format!("checkpoint-{}",chains.steps)),&model,&optim,&saved,&chains)?;
        }
    }
    runtime::sync(&target)?;
    let seconds = previous_seconds+started.elapsed().as_secs_f64();
    let saved = train::State {schema: 1,config_sha256: fingerprint.clone(),data: data.clone(),arm: arm.name.clone(),seed,
        step: chains.steps,tokens: chains.scored_target_visits,training_seconds: seconds,initial_evaluation: Value::Null};
    let path = run_dir.join(format!("checkpoint-{}",chains.steps));
    save(c,&path,&model,&optim,&saved,&chains)?;
    let report = json!({"schema":1,"training_mode":"joint_tct","config":identity_config,"requested_config":c,"config_sha256":fingerprint,"arm":arm,"seed":seed,"data":data,
        "device":runtime::identity(device),"checkpoint":path,"completed":chains.scored_target_visits==target_tokens,"target_tokens":target_tokens,
        "steps":chains.steps,"target_token_visits":chains.scored_target_visits,"target_token_visits_this_invocation":chains.scored_target_visits-start_visits,
        "unique_target_bytes_loaded":chains.unique_target_bytes,"completed_chains":chains.completed_chains,"training_seconds":seconds,
        "clean_forward_calls":chains.steps,"noisy_forward_calls":chains.noisy_forward_calls,"noisy_block_evaluations":chains.noisy_block_evaluations,
        "coordination_work_this_invocation":work_this_invocation,
        "gradient_counts_this_invocation":gradient_counts,
        "coordination_work_units":{"dispatches":"physical neural/circuit dispatches","rows":"physical processed row units; paired tails count 2C for one doubled dispatch",
            "used_rows":"rows whose branch results are consumed by losses/feedback; query-dispatch and used rows are separately tracked",
            "logical_blocks":"logical TCT chains","prefix_batches":"batched saved-prefix cache calls","circuit_preparations":"prepared A/B circuits",
            "message_passes":"circuit message/statistics traversals","evidence_queries":"objective/feedback query operations","ffbs_passes":"joint proposal sampling calls",
            "final_replay_rows":"circuit final-replay row units"},
        "unique_target_bytes_note":"counts each loaded training window once, not repeated chain visits; corpus byte offsets are not deduplicated",
        "last_training_loss":last_loss,"source_sbd_checkpoint":chains.source_checkpoint,"rounds":TRAIN_ROUNDS,"input_visits_per_chain":TRAIN_ROUNDS+1,
        "source_training_identity":chains.source_training_identity,
        "threshold":REVEAL_THRESHOLD,"proposal":"greedy_conditional_marginal_argmax","evidence":"permanent_O_only",
        "inputs":"hard_provisional_reveals_and_retained_normalized_top_k_soft_feedback",
        "objective":"explicit combined design: clean causal mean plus logical-chain-mean joint Codpula NLL and token-sum correction CE; optional B joint event and coordination CE on masked S and visible mutable positions; circuit reach/count once per branch-chain",
        "optimizer_start_step":chains.optimizer_start_step,"optimizer_source_schedule_steps":chains.optimizer_schedule_steps,
        "quality_evaluation_performed":false});
    runtime::write_json(&run_dir.join("metrics.json"),&report)?;
    Ok(report)
}

fn paired(a: &[f64], b: &[f64]) -> Value {
    let differences = a.iter().zip(b).map(|(a,b)|a-b).collect::<Vec<_>>();
    let mut rng = Rng(709);
    let mut bootstrap = (0..10000).map(|_|crate::profile::median(&(0..differences.len()).map(|_| differences[rng.index(differences.len())]).collect::<Vec<_>>())).collect::<Vec<_>>();
    bootstrap.sort_by(f64::total_cmp);
    json!({"per_block_minus_batched_ms":differences,"median_difference_ms":crate::profile::median(&differences),"bootstrap_95_percent_ms":[bootstrap[249],bootstrap[9749]]})
}

/// Paired complete steps compare only cache execution, never objectives or training variants.
/// B1/T512 uses 2 warmups + 11 alternating pairs; B8/T512 is one warm + one batched diagnostic.
pub fn benchmark(c: &Config, arm: &Arm, seed: u64, device: &Device) -> Result<Value> {
    validate_config(c)?;
    ensure!(c.context == 512 && matches!(c.batch_size,1|8),"TCT cost entry requires B1/T512 or B8/T512");
    let (warmups,trials) = if c.batch_size == 1 {(2,11)} else {(1,1)};
    let corpus = train::loaded_corpus(c)?;
    let target = device.clone().autodiff();
    target.seed(seed);
    let base = LanguageModel::new(c,arm,&target)?;
    let initial = TrainingState::new(c,corpus.as_ref(),seed)?;
    if c.batch_size == 8 {
        let mut model = base;
        let mut optimizer = train::optimizer(c);
        let mut state = initial;
        let mut peak = runtime::sampled_reserved(device);
        let mut measured = 0.0;
        let mut loss = Value::Null;
        for iteration in 0..2 {
            runtime::sync(&target)?;
            let clock = Instant::now();
            let result = step(c,model,&mut optimizer,&mut state,corpus.as_ref(),&target)?;
            model = result.0;
            runtime::sync(&target)?;
            if iteration == 1 {
                measured = clock.elapsed().as_secs_f64()*1000.0;
                loss = result.1;
            }
            if let Some(value) = runtime::sampled_reserved(device) {peak = Some(peak.unwrap_or(0).max(value));}
        }
        return Ok(json!({"completed":true,"config":c,"device":runtime::identity(device),
            "precision":"float32","batch":8,"context":512,"warmups":1,"samples":1,
            "diagnostic_only":true,"batched_samples_ms":[measured],"batched_median_ms":measured,
            "last_loss":loss,"sampled_peak_reserved_bytes":peak,
            "method":"one warm and one synchronized complete batched optimizer step; clean forward, fresh prefix cache, joint plus marginal token CE/shared flows, greedy feedback, backward/checks, optimizer and chain update included",
            "memory_note":"reservation sampled after complete steps, not a physical peak"}));
    }
    let mut models = [Some(base.clone()),Some(base)];
    let mut optimizers = [train::optimizer(c),train::optimizer(c)];
    let mut states = [initial.clone(),initial];
    let mut samples: [Vec<f64>;2] = std::array::from_fn(|_|Vec::new());
    let mut losses = [Value::Null,Value::Null];
    let mut peak = runtime::sampled_reserved(device);
    let started = Instant::now();
    for iteration in 0..warmups+trials {
        for index in if iteration%2 == 0 {[0,1]} else {[1,0]} {
            ensure!(started.elapsed().as_secs_f64() < c.max_seconds,"TCT benchmark exceeded max_seconds");
            runtime::sync(&target)?;
            let clock = Instant::now();
            let layout = if index == 0 {SbdBlockExecution::PerBlock} else {SbdBlockExecution::Batched};
            let result = step_layout(c,models[index].take().unwrap(),&mut optimizers[index],&mut states[index],corpus.as_ref(),&target,layout)?;
            models[index] = Some(result.0);
            losses[index] = result.1;
            runtime::sync(&target)?;
            let milliseconds = clock.elapsed().as_secs_f64()*1000.0;
            if iteration >= warmups {samples[index].push(milliseconds);}
            if let Some(value) = runtime::sampled_reserved(device) {peak = Some(peak.unwrap_or(0).max(value));}
        }
    }
    Ok(json!({"completed":true,"config":c,"device":runtime::identity(device),"precision":"float32","batch":c.batch_size,"context":c.context,
        "warmups":warmups,"pairs":trials,"diagnostic_only":c.batch_size==8,"per_block_samples_ms":samples[0],"batched_samples_ms":samples[1],
        "per_block_median_ms":crate::profile::median(&samples[0]),"batched_median_ms":crate::profile::median(&samples[1]),
        "paired":if trials>1 {paired(&samples[0],&samples[1])} else {Value::Null},"last_losses":losses,"sampled_peak_reserved_bytes":peak,
        "method":"alternating complete optimizer steps with identical initial models/chains; clean forward, fresh prefix cache, joint plus marginal correction CE/shared flows, greedy soft feedback, backward/checks, optimizer and refill; no evaluation/checkpoint IO",
        "memory_note":"device reservation sampled after each complete step; both runners coexist; not a physical peak"}))
}

// The small oracle sums complete latent/token assignments, not tensor messages.
// Its dual-flow algebra follows the independent Codpula reference's row derivatives.
#[derive(Clone)]
struct Reference {
    fields: [Vec<f64>; 5], // emission, transition, root, neural potentials, gates
}
struct Enumerated {
    mass: f64,
    state: Vec<f64>,
    edge: Vec<f64>,
    counts: Vec<f64>,
}
fn probabilities(row: &[f64]) -> Vec<f64> {
    let maximum = row.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    let z = row.iter().map(|x|(x-maximum).exp()).sum::<f64>();
    row.iter().map(|x|(x-maximum).exp()/z).collect()
}
impl Reference {
    fn fixture() -> Self {
        Self {fields: [
            vec![0.2,-0.7,0.3, -0.5,0.8,-0.1],
            vec![0.4,-0.2, -0.3,0.5],
            vec![0.3,-0.4],
            vec![0.8,-0.1,0.3, -0.6,0.1,1.2, 0.4,-0.2,0.6],
            vec![0.2,-0.4, -0.3,0.6],
        ].map(|row|row.into_iter().map(|x|x as f32 as f64).collect())}
    }
    fn tables(&self) -> (Vec<f64>,Vec<f64>,Vec<f64>) {
        let base = self.fields[0].chunks_exact(3).flat_map(probabilities).collect::<Vec<_>>();
        let mut leaf = Vec::with_capacity(18);
        for i in 0..3 { for s in 0..2 {
            leaf.extend(probabilities(&(0..3).map(|x|base[s*3+x].ln()+self.fields[3][i*3+x]).collect::<Vec<_>>()));
        }}
        let transition = (0..2).flat_map(|s|probabilities(&(0..2).map(|t|self.fields[1][s*2+t]+self.fields[4][s*2+t]).collect::<Vec<_>>())).collect();
        (leaf,transition,probabilities(&self.fields[2]))
    }
    fn query(&self, evidence: &[Option<usize>], target: Option<&[usize]>) -> Enumerated {
        let (leaf,a,root) = self.tables();
        let mut out = Enumerated {mass:0.0,state:vec![0.0;6],edge:vec![0.0;8],counts:vec![0.0;18]};
        for path in 0usize..8 {
            let states = [path%2,(path/2)%2,(path/4)%2];
            for words in 0usize..27 {
                let tokens = [words%3,(words/3)%3,(words/9)%3];
                if (0..3).any(|i|evidence[i].is_some_and(|x|x!=tokens[i]) || target.is_some_and(|gold|gold[i]!=tokens[i])) {continue;}
                let mut p = root[states[0]];
                for i in 0..3 {
                    if i>0 {p *= a[states[i-1]*2+states[i]];}
                    p *= leaf[(i*2+states[i])*3+tokens[i]];
                }
                out.mass += p;
                for i in 0..3 {
                    out.state[i*2+states[i]] += p;
                    out.counts[(i*2+states[i])*3+tokens[i]] += p;
                    if i>0 {out.edge[(i-1)*4+states[i-1]*2+states[i]] += p;}
                }
            }
        }
        if out.mass > 0.0 {
            for values in [&mut out.state,&mut out.edge,&mut out.counts] {
                for x in values {*x /= out.mass;}
            }
        }
        out
    }
    fn nll(&self, evidence: &[Option<usize>], target: &[usize], denominator: f64) -> f64 {
        let base=self.query(evidence,None).mass.ln();
        (0..target.len()).filter(|&i|evidence[i].is_none()).map(|i|{
            let mut singleton=evidence.to_vec();singleton[i]=Some(target[i]);
            base-self.query(&singleton,None).mass.ln()
        }).sum::<f64>()/denominator
    }
    fn flows(&self,evidence:&[Option<usize>],target:&[usize])->Vec<(&'static str,Vec<f64>)> {
        let mut result=None::<Vec<(&'static str,Vec<f64>)>>;
        for i in 0..target.len() {
            if evidence[i].is_some(){continue;}
            let mut singleton=evidence.to_vec();singleton[i]=Some(target[i]);
            let part=self.event_flows(evidence,&singleton);
            if let Some(total)=&mut result {
                for (sum,(_,values)) in total.iter_mut().take(6).zip(part) {
                    for (a,b) in sum.1.iter_mut().zip(values){*a+=b;}
                }
            }else{result=Some(part);}
        }
        result.unwrap()
    }
    fn event_flows(&self, evidence: &[Option<usize>], numerator: &[Option<usize>]) -> Vec<(&'static str,Vec<f64>)> {
        let plus = self.query(numerator,None);
        let minus = self.query(evidence,None);
        let prior = self.query(&[None;3],None);
        let (leaf,a,root) = self.tables();
        let mut ep = vec![0.0;6]; let mut en = vec![0.0;6];
        let mut tp = vec![0.0;4]; let mut tn = vec![0.0;4];
        let mut er = vec![0.0;2]; let mut tr = vec![0.0;2];
        for i in 0..3 {for s in 0..2 {
            er[s] += prior.state[i*2+s];
            for x in 0..3 {
                let at = (i*2+s)*3+x;
                ep[s*3+x] += plus.counts[at]+minus.state[i*2+s]*leaf[at];
                en[s*3+x] += minus.counts[at]+plus.state[i*2+s]*leaf[at];
            }
            if i<2 {
                tr[s] += prior.state[i*2+s];
                let plus_row = plus.edge[i*4+s*2..i*4+s*2+2].iter().sum::<f64>();
                let minus_row = minus.edge[i*4+s*2..i*4+s*2+2].iter().sum::<f64>();
                for t in 0..2 {
                    tp[s*2+t] += plus.edge[i*4+s*2+t]+minus_row*a[s*2+t];
                    tn[s*2+t] += minus.edge[i*4+s*2+t]+plus_row*a[s*2+t];
                }
            }
        }}
        vec![("emission_positive",ep),("emission_negative",en),("transition_positive",tp),("transition_negative",tn),
            ("root_positive",(0..2).map(|s|plus.state[s]+root[s]).collect()),
            ("root_negative",(0..2).map(|s|minus.state[s]+root[s]).collect()),
            ("emission_reach",er),("transition_reach",tr),("root_reach",vec![1.0])]
    }
}

fn errors(actual: &[f32], reference: &[f64], name: &str) -> Result<Value> {
    ensure!(actual.len() == reference.len(),"{name}: comparison dimensions differ");
    let mut maximum = 0.0f64;
    let mut ratio = 0.0f64;
    for (i,(&a,&r)) in actual.iter().zip(reference).enumerate() {
        let error = (f64::from(a)-r).abs();
        let relative = error/(2e-5+3e-4*r.abs());
        ensure!(a.is_finite() && r.is_finite() && relative <= 1.0,"{name}[{i}]: actual={a}, reference={r}, error={error}, tolerance_ratio={relative}");
        maximum = maximum.max(error); ratio = ratio.max(relative);
    }
    Ok(json!({"passed":true,"elements":actual.len(),"max_abs":maximum,"maximum_tolerance_ratio":ratio,"absolute_tolerance":2e-5,"relative_tolerance":3e-4}))
}
fn flow_values(flows: &CircuitFlows) -> Result<Vec<(&'static str,Vec<f32>)>> {
    Ok(vec![("emission_positive",host(flows.emission_positive.clone())?),("emission_negative",host(flows.emission_negative.clone())?),
        ("transition_positive",host(flows.transition_positive.clone())?),("transition_negative",host(flows.transition_negative.clone())?),
        ("root_positive",host(flows.root_positive.clone())?),("root_negative",host(flows.root_negative.clone())?),
        ("emission_reach",host(flows.emission_reach.clone())?),("transition_reach",host(flows.transition_reach.clone())?),
        ("root_reach",host(flows.root_reach.clone())?)])
}
fn float<const D: usize>(values: &[f64], shape: [usize;D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(values.iter().map(|&x|x as f32).collect::<Vec<_>>(),shape),device)
}
fn joint_checks(device: &Device, execution: CodpulaExecution) -> Result<Value> {
    let target_device = device.clone().autodiff();
    let reference = Reference::fixture();
    let evidence = [Some(0),None,None];
    let target = [0usize,1,2];
    let observed = Tensor::<2,Bool>::from_data([[true,false,false]],&target_device);
    let pi = float(&reference.fields[0],[2,3],&target_device).require_grad();
    let a = float(&reference.fields[1],[2,2],&target_device).require_grad();
    let rho = float(&reference.fields[2],[2],&target_device).require_grad();
    let potentials = float(&reference.fields[3],[1,3,3],&target_device).require_grad();
    let gates = float(&reference.fields[4],[1,2,2],&target_device).require_grad();
    let logits = potentials.clone().mask_where(observed.clone().reshape([1,3,1]).expand([1,3,3]),potentials.clone().detach());
    let circuit = crate::codpula::PreparedCircuit::prepare_with_execution(pi.clone(),a.clone(),rho.clone(),logits,gates.clone(),vec![3],1,execution)?;
    let gold = Tensor::<2,Int>::from_data([[0,1,2]],&target_device);
    let result=circuit.marginal_ce(gold.clone(),gold.clone(),observed.clone(),Tensor::<2>::from_data([[0.0,1.0,1.0]],&target_device),1.0)?;
    let nll=errors(&host(result.loss.clone())?,&[reference.nll(&evidence,&target,1.0)],"TCT all-mutable marginal CE with chain denominator")?;
    let marked=Tensor::<2,Bool>::from_data([[false,true,false]],&target_device);
    let event=circuit.marginal_ce(gold.clone(),gold,observed.clone(),marked.float(),1.0)?;
    let marked_evidence=[Some(0),Some(1),None];
    let expected_event=reference.query(&evidence,None).mass.ln()-reference.query(&marked_evidence,None).mass.ln();
    let event_nll=errors(&host(event.loss.clone())?,&[expected_event],"TCT marked marginal token CE")?;
    ensure!(event.flows.active_blocks==1,"TCT marked event flow count differs");
    ensure!(result.flows.active_blocks == 1,"joint TCT active block count differs");
    let expected_flows = reference.flows(&evidence,&target);
    let mut flows = Vec::new();
    for ((name,actual),(expected_name,expected)) in flow_values(&result.flows)?.into_iter().zip(expected_flows) {
        ensure!(name == expected_name,"TCT flow names differ");
        flows.push(json!({"quantity":name,"errors":errors(&actual,&expected,name)?}));
    }
    let gradients = result.loss.backward();
    let actual = [
        host(pi.grad(&gradients).context("TCT emission derivative")?)?,
        host(a.grad(&gradients).context("TCT transition derivative")?)?,
        host(rho.grad(&gradients).context("TCT root derivative")?)?,
        host(potentials.grad(&gradients).context("TCT neural potential derivative")?)?,
        host(gates.grad(&gradients).context("TCT gate derivative")?)?,
    ];
    let mut derivatives = Vec::new();
    for field in 0..5 {
        let mut expected = Vec::with_capacity(reference.fields[field].len());
        for i in 0..reference.fields[field].len() {
            if field == 3 && i<3 {expected.push(0.0);continue;}
            let mut plus = reference.clone(); let mut minus = reference.clone();
            plus.fields[field][i] += 1e-5; minus.fields[field][i] -= 1e-5;
            expected.push((plus.nll(&evidence,&target,1.0)-minus.nll(&evidence,&target,1.0))/2e-5);
        }
        derivatives.push(json!({"field":field,"errors":errors(&actual[field],&expected,"TCT central-difference gradient")?}));
    }
    ensure!(actual[3][..3].iter().all(|&x|x == 0.0),"permanent evidence has token-potential gradients");
    ensure!(actual[3][3..6].iter().any(|x|x.abs()>1e-6),"wrong provisional input has no correction gradient");
    let wrong_evidence = [Some(0),Some(2),None];
    ensure!(reference.query(&wrong_evidence,None).mass > 0.0 && reference.query(&wrong_evidence,Some(&target)).mass == 0.0,
        "wrong provisional evidence must reject the gold correction event");
    Ok(json!({"passed":true,"device":runtime::identity(device),"execution":execution,"marginal_token_ce":nll,"marked_subset_token_ce":event_nll,"flows":flows,"gradients":derivatives,
        "reference":"independent exhaustive f64 sum over 8 latent paths and 27 token assignments; all-coordinate f64 central differences",
        "wrong_provisional_value":2,"gold_correction_value":1,"wrong_reveal_correction_gradient_nonzero":true,
        "permanent_token_potential_gradient_zero":true,"incorrectly_conditioned_gold_event_probability":0.0,
        "marked_target_scope":"only S is fixed to gold; remaining free positions are marginalized"}))
}

fn chain_checks() -> Result<Value> {
    let mut chain = Chain::new(0,8,4,257);
    let mut counts = Vec::new();
    let initial = serde_json::to_vec(&chain)?;
    let mut restarted: Chain = serde_json::from_slice(&initial)?;
    for round in 0..TRAIN_ROUNDS {
        let proposal = vec![round+1;8];
        let confidence = vec![0.2;8];
        chain.reveal(&proposal,&confidence)?;
        restarted.reveal(&proposal,&confidence)?;
        ensure!(chain == restarted,"TCT round replay differs");
        restarted = serde_json::from_slice(&serde_json::to_vec(&restarted)?)?;
        counts.push(chain.revealed.iter().filter(|&&x|x).count());
    }
    ensure!(counts == [2,4,6,8] && chain.values == [1,1,2,2,3,3,4,4],"scheduled reveals or stable earlier inputs differ");
    let mut threshold = Chain::new(0,4,2,257);
    threshold.reveal(&[7;4],&[0.95;4])?;
    for _ in 1..TRAIN_ROUNDS {threshold.reveal(&[9;4],&[0.1;4])?;}
    ensure!(threshold.values == [7;4] && threshold.round == TRAIN_ROUNDS,"early fully revealed input changed before endpoint");
    let mut permanent = Chain::new(0,3,2,257);
    permanent.permanent[0] = Some(8);
    permanent.values[0] = 8;
    for _ in 0..TRAIN_ROUNDS {permanent.reveal(&[9;3],&[0.99;3])?;}
    ensure!(permanent.values[0] == 8 && !permanent.revealed[0],"permanent TCT evidence changed");
    Ok(json!({"passed":true,"scheduled_reveal_counts":counts,"input_visits":TRAIN_ROUNDS+1,"ties":"ascending_position_then_token_id",
        "early_fully_revealed_input_retained_until_endpoint":true,"permanent_values_unchanged":true,"serialized_round_replay_identical":true}))
}

struct TemporaryDirectory(std::path::PathBuf);
impl Drop for TemporaryDirectory {
    fn drop(&mut self) {let _ = std::fs::remove_dir_all(&self.0);}
}
fn restart_fixture() -> (Config,Arm,Corpus) {
    use crate::config::{SbdConfig,SbdFeedback};
    let arm = Arm {name:"tct-restart".into(),mixer:"kronlstm".into(),cache:"surprise".into()};
    let c = Config {task:"lm".into(),context:32,batch_size:1,d_model:16,layers:1,heads:4,key_dim:32,value_dim:64,
        chunk_size:16,cache_capacity:4,train_tokens:32*8,checkpoint_every:2,
        sbd:Some(SbdConfig {block_lengths:vec![8],block_execution:SbdBlockExecution::Batched,joint_output:SbdJointOutput::Codpula,
            training_mode:TrainingMode::Tct,feedback:SbdFeedback::SoftFeedback,..SbdConfig::default()}),
        arms:vec![arm.clone()],..Config::default()};
    let bytes = (0..512).map(|i|((i*37+11)%256) as u8).collect::<Vec<_>>();
    let corpus = Corpus {train:vec![bytes.clone()],validation:vec![bytes.clone()],test:vec![bytes],manifest_sha256:"tct-restart-fixture-v1".into()};
    (c,arm,corpus)
}

fn restart_checks(device:&Device,execution:CodpulaExecution,cocommit:bool)->Result<Value> {
    let (mut c,arm,corpus)=restart_fixture();
    c.batch_size=2;
    c.tct=Some(TctConfig {rows_per_pass:1});
    if cocommit {
        c.sbd.as_mut().unwrap().cocommit=Some(crate::config::CocommitConfig::default());
        c.sbd.as_mut().unwrap().joint_output=SbdJointOutput::CodpulaContextual;
    }
    c.sbd.as_mut().unwrap().joint_execution=execution;
    if !execution.is_tensor(){c.backend="vulkan".into();}
    let target = device.clone().autodiff();
    target.seed(991);
    let mut model = LanguageModel::new(&c,&arm,&target)?;
    let marker_before=if cocommit {Some(host(model.cocommit_marker.as_ref().context("TCT marker missing")?.val())?)} else {None};
    let root_head_before=if cocommit {Some(host(model.codpula.as_ref().context("contextual TCT circuit missing")?.gate_head.weight.val())?)} else {None};
    let circuit_ema_before=if cocommit {Some(host(model.codpula.as_ref().unwrap().probability.emission_positive_ema.val())?)} else {None};
    let mut optim=train::optimizer(&c);
    let mut chains=TrainingState::new(&c,Some(&corpus),991)?;
    for _ in 0..2 {model=step(&c,model,&mut optim,&mut chains,Some(&corpus),&target)?.0;}
    if let (Some(before),Some(head_before),Some(ema_before))=(marker_before,root_head_before,circuit_ema_before) {
        let marker_after=host(model.cocommit_marker.as_ref().unwrap().val())?;
        let head_after=host(model.codpula.as_ref().unwrap().gate_head.weight.val())?;
        let ema_after=host(model.codpula.as_ref().unwrap().probability.emission_positive_ema.val())?;
        ensure!(before.iter().zip(&marker_after).any(|(a,b)|a.to_bits()!=b.to_bits()),
            "TCT CoCommit marker optimizer did not update");
        ensure!(head_before.chunks_exact(96).zip(head_after.chunks_exact(96))
            .any(|(a,b)|a[64..].iter().zip(&b[64..]).any(|(x,y)|x.to_bits()!=y.to_bits())),
            "TCT contextual root-head optimizer did not update");
        ensure!(ema_before.iter().zip(&ema_after).any(|(a,b)|a.to_bits()!=b.to_bits()),
            "TCT CoCommit circuit EMA did not update");
    }
    chains.validate(&c,991)?;
    ensure!(chains.rows.iter().flat_map(|r|&r.chains).all(|chain|chain.round == 2),"restart fixture did not save partial trajectories");
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let directory = TemporaryDirectory(std::env::temp_dir().join(format!("delta-tct-restart-{}-{stamp}",std::process::id())));
    std::fs::create_dir_all(&directory.0)?;
    let path = directory.0.join("checkpoint-2");
    let saved = train::State {schema:1,config_sha256:c.training_fingerprint(c.train_tokens)?,data:json!({"fixture":"tct-restart-v1"}),
        arm:arm.name.clone(),seed:991,step:chains.steps,tokens:chains.scored_target_visits,training_seconds:0.0,initial_evaluation:Value::Null};
    save(&c,&path,&model,&optim,&saved,&chains)?;
    let mut resumed = checkpoint::load_model(LanguageModel::new(&c,&arm,&target)?,path.join("model.burnpack"),&target)?;
    let mut resumed_optim = checkpoint::load_optimizer(&c,path.join("optimizer.burnpack"))?;
    let mut resumed_chains: TrainingState = serde_json::from_slice(&std::fs::read(path.join("tct.json"))?)?;
    resumed_chains.validate(&c,991)?;
    ensure!(resumed_chains == chains,"real saved chain inputs, targets, loader counters or RNG changed");
    // Unchanged model/optimizer values isolate the chain-state collision check.
    let mut changed = resumed_chains.clone();
    changed.rng = changed.rng.wrapping_add(1);
    ensure!(train::checkpoint_with_tct(&c,&path,&model,&optim,&saved,Some(&changed)).is_err(),"changed chain checkpoint collision was accepted");
    let mut losses = Vec::new();
    for _ in 0..4 {
        let uninterrupted = step(&c,model,&mut optim,&mut chains,Some(&corpus),&target)?;
        model = uninterrupted.0;
        let continuation = step(&c,resumed,&mut resumed_optim,&mut resumed_chains,Some(&corpus),&target)?;
        resumed = continuation.0;
        ensure!(chains == resumed_chains && uninterrupted.1 == continuation.1,"checkpoint restart changed chain values, RNG/refill or exact next-step losses");
        let expected = crate::kron_benchmark::parameters(&model,None)?;
        let actual = crate::kron_benchmark::parameters(&resumed,None)?;
        ensure!(actual.len() == expected.len() && actual.iter().zip(&expected).all(|((shape,a),(expected_shape,b))|
            shape == expected_shape && a.iter().zip(b).all(|(a,b)|a.to_bits() == b.to_bits())),"TCT resumed neural/circuit parameters or optimizer buffers differ");
        ensure!(checkpoint::optimizer_signature(&c,optim.into_bytes()?)? == checkpoint::optimizer_signature(&c,resumed_optim.into_bytes()?)?,
            "TCT resumed neural optimizer moments/counters differ");
        losses.push(uninterrupted.1);
    }
    ensure!(chains.next_example == 2*c.batch_size && chains.completed_chains == 4*c.batch_size && chains.rows[0].chains.iter().all(|chain|chain.round == 1),
        "restart check did not pass fully revealed endpoint and refill");
    chains.validate(&c,991)?;
    Ok(json!({"passed":true,"device":runtime::identity(device),"saved_after_input_visits":2,"continuation_steps":4,
        "checkpoint_files":["model.burnpack","optimizer.burnpack","state.json","tct.json"],"chain_rng_loader_bit_identical":true,
        "neural_and_circuit_rows_and_optimizer_buffers_bit_identical":true,"neural_optimizer_moments_bit_identical":true,
        "fully_revealed_endpoint_supervised_before_refill":true,"parameter_fresh_clean_cache_every_step":true,"losses":losses}))
}

fn layout_checks(device: &Device, execution: CodpulaExecution) -> Result<Value> {
    let (mut c,arm,corpus) = restart_fixture();
    c.sbd.as_mut().unwrap().joint_execution=execution;
    if !execution.is_tensor() {c.backend="vulkan".into();}
    // A tail block and an L1-only batch check padding and the causal limiting case.
    c.sbd.as_mut().unwrap().block_lengths = vec![7];
    let target = device.clone().autodiff();
    target.seed(173);
    let model = LanguageModel::new(&c,&arm,&target)?;
    let mut initial = TrainingState::new(&c,Some(&corpus),173)?;
    let _ = training_loss(&c,&model,&mut initial,&target,SbdBlockExecution::Batched,None)?;
    let mut wrong_revealed_inputs = 0;
    let mut soft_positions = 0;
    for row in &mut initial.rows {for chain in &mut row.chains {for i in 0..chain.len() {
        if chain.revealed[i] {
            chain.values[i] = (row.target[chain.start+i] as usize+1)%256;
            wrong_revealed_inputs += 1;
        } else {
            soft_positions += 1;
        }
    }}}
    ensure!(wrong_revealed_inputs > 0 && soft_positions > 0,"TCT fixture needs both wrong revealed inputs and unresolved soft feedback");
    let locations = initial.rows.iter().enumerate().flat_map(|(r,row)|(0..row.chains.len()).map(move |i|(r,i))).collect::<Vec<_>>();
    let input = input_state(&model,&c,&initial,&locations,7,&target)?;
    ensure!(input.observed.iter().all(Option::is_none),"provisional reveals entered permanent circuit evidence");
    let embeddings = host(model.embedding.weight.val())?;
    let mask = host(model.sbd_mask.as_ref().unwrap().val())?;
    let actual_inputs = host(input.embeddings(model.embedding.weight.val(),model.sbd_mask.as_ref().unwrap().val(),false))?;
    let mut expected_inputs = Vec::with_capacity(locations.len()*7*c.d_model);
    for &(r,index) in &locations {
        let chain = &initial.rows[r].chains[index];
        for i in 0..7 {for d in 0..c.d_model {
            let expected = if i >= chain.len() {0.0}
                else if chain.revealed[i] {f64::from(embeddings[chain.values[i]*c.d_model+d])}
                else {
                    let k = c.sbd.as_ref().unwrap().candidate_width;
                    let visibility = f64::from(chain.visibility[i]);
                    let weighted = (0..k).map(|j|f64::from(chain.candidate_weights[i*k+j])
                        *f64::from(embeddings[chain.candidate_ids[i*k+j] as usize*c.d_model+d])).sum::<f64>();
                    weighted*visibility+f64::from(mask[d])*(1.0-visibility)
                };
            expected_inputs.push(expected);
        }}
    }
    let input_composition = errors(&actual_inputs,&expected_inputs,"TCT hard-reveal and retained-soft input embeddings")?;
    let mut a = initial.clone(); let mut b = initial;
    let per_block = training_loss(&c,&model,&mut a,&target,SbdBlockExecution::PerBlock,None)?;
    let batched = training_loss(&c,&model,&mut b,&target,SbdBlockExecution::Batched,None)?;
    let loss = errors(&host(batched.loss.clone())?,&[model::scalar(per_block.loss.clone())?],"TCT cache layouts loss")?;
    let expected_gradients = model::checked_gradients(&model,per_block.loss)?;
    let actual_gradients = model::checked_gradients(&model,batched.loss)?;
    let expected = crate::kron_benchmark::parameters(&model,Some(&expected_gradients))?;
    let actual = crate::kron_benchmark::parameters(&model,Some(&actual_gradients))?;
    ensure!(actual.len() == expected.len(),"TCT layout gradient tensor count differs");
    let mut gradients = Vec::new();
    for ((shape,actual),(expected_shape,reference)) in actual.iter().zip(&expected) {
        ensure!(shape == expected_shape,"TCT layout gradient dimensions differ");
        gradients.push(errors(actual,&reference.iter().map(|&x|f64::from(x)).collect::<Vec<_>>(),"TCT cache layouts gradients")?);
    }
    let mut flows = Vec::new();
    for ((name,actual),(_,reference)) in flow_values(batched.flows.as_ref().unwrap())?.into_iter().zip(flow_values(per_block.flows.as_ref().unwrap())?) {
        flows.push(json!({"quantity":name,"errors":errors(&actual,&reference.into_iter().map(f64::from).collect::<Vec<_>>(),name)?}));
    }
    // Discrete trajectories must agree, while FP32 candidate weights use the numerical tolerance.
    for (ra,rb) in a.rows.iter().zip(&b.rows) {for (ca,cb) in ra.chains.iter().zip(&rb.chains) {
        ensure!(ca.values == cb.values && ca.revealed == cb.revealed && ca.round == cb.round && ca.candidate_ids == cb.candidate_ids,
            "TCT layout changed supplied-randomness proposals or reveals");
        errors(&cb.candidate_weights,&ca.candidate_weights.iter().map(|&x|f64::from(x)).collect::<Vec<_>>(),"TCT layout feedback")?;
    }}
    c.sbd.as_mut().unwrap().block_lengths=vec![1];
    let mut singleton_state=TrainingState::new(&c,Some(&corpus),173)?;
    let batch=singleton_state.batch(&target);
    let clean_value=model::scalar(model.loss(model.forward(&batch),&batch))?;
    let before=singleton_state.clone();
    let singleton=training_loss(&c,&model,&mut singleton_state,&target,SbdBlockExecution::Batched,None)?;
    let mut expected=0.0;
    let chain_count=before.rows.iter().map(|row|row.chains.len()).sum::<usize>();
    for row in &before.rows {for chain in &row.chains {
        let soft=crate::sbd::state(&c,vec![None],&target)?;
        let output=crate::sbd::block_forward(&model,&row.input[..chain.start+1],&soft,false,&target);
        let (circuit,tokens,observed)=crate::sbd::prepare_circuit(&model,output,&soft,1.0)?;
        let probability=host(circuit.marginals(tokens,observed)?)?;
        expected-=2.0*f64::from(probability[row.target[chain.start] as usize]).ln()/chain_count as f64;
    }}
    let singleton_objective=errors(&host(singleton.loss)?,&[clean_value+expected],"TCT singleton joint plus marginal CE")?;
    ensure!(singleton.flows.is_some() && singleton_state.rows.iter().flat_map(|row|&row.chains).all(|chain|chain.revealed[0]),
        "singleton did not consume noisy denoiser or reveal its argmax");
    Ok(json!({"passed":true,"loss":loss,"gradients":gradients,"flows":flows,"tail_block_length":4,"length_one_noisy_objective":singleton_objective,
        "input_composition":input_composition,"wrong_revealed_inputs":wrong_revealed_inputs,"unresolved_soft_positions":soft_positions,"provisional_inputs_never_evidence":true}))
}

fn warm_start_checks() -> Result<Value> {
    let (mut requested,arm,_) = restart_fixture();
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let directory = TemporaryDirectory(std::env::temp_dir().join(format!("delta-tct-source-{}-{stamp}",std::process::id())));
    std::fs::create_dir_all(&directory.0)?;
    let path = directory.0.join("checkpoint-1");
    let mut source = requested.clone();
    source.sbd.as_mut().unwrap().training_mode = TrainingMode::Sbd;
    source.artifact_dir = directory.0.join("source-output");
    source.train_tokens = 32;
    let mut saved = train::State {schema:1,config_sha256:source.training_fingerprint(32)?,data:json!({"fixture":"source-config"}),
        arm:arm.name,seed:991,step:1,tokens:32,training_seconds:0.0,initial_evaluation:Value::Null};
    let mut metrics = json!({"schema":1,"config":source,"config_sha256":saved.config_sha256,"target_tokens":32});
    runtime::write_json(&directory.0.join("metrics.json"),&metrics)?;
    requested.artifact_dir = directory.0.join("new-tct-output");
    requested.train_tokens = 32*5;
    requested.sbd.as_mut().unwrap().refinement = Some(crate::remask::RefinementConfig::default());
    let accepted = warm_start_identity(&requested,&path,&saved,requested.train_tokens)?;
    ensure!(accepted["verified_source_config"].is_string(),"changed-budget TCT warm-start did not prove source configuration");
    let mut changed_optimizer = requested.clone();
    changed_optimizer.learning_rate *= 2.0;
    ensure!(warm_start_identity(&changed_optimizer,&path,&saved,requested.train_tokens).is_err(),"changed optimizer settings accepted by warm-start");
    let mut changed_model = requested.clone();
    changed_model.d_model *= 2;
    ensure!(warm_start_identity(&changed_model,&path,&saved,requested.train_tokens).is_err(),"changed model accepted by warm-start");
    ensure!(warm_start_identity(&requested,&directory.0.join("unrecorded/checkpoint-1"),&saved,requested.train_tokens).is_err(),
        "changed-budget warm-start accepted without source proof");
    source.budget_file = Some(directory.0.join("removed-source-budget.json"));
    saved.config_sha256 = source.training_fingerprint(32)?;
    metrics["config"] = serde_json::to_value(&source)?;
    metrics["config_sha256"] = json!(saved.config_sha256);
    runtime::write_json(&directory.0.join("metrics.json"),&metrics)?;
    warm_start_identity(&requested,&path,&saved,requested.train_tokens)?;
    metrics["config"]["train_tokens"] = json!(64);
    runtime::write_json(&directory.0.join("metrics.json"),&metrics)?;
    ensure!(warm_start_identity(&requested,&path,&saved,requested.train_tokens).is_err(),"tampered source config accepted by warm-start");
    Ok(json!({"passed":true,"new_token_budget_and_output_path_accepted_with_source_proof":true,"decoder_refinement_ignored":true,
        "removed_source_budget_file_not_needed":true,"changed_optimizer_rejected":true,"changed_model_rejected":true,
        "missing_source_proof_rejected":true,"tampered_source_config_rejected":true}))
}


/// Independent joint likelihood/flows/gradient and real checkpoint continuation report.
pub fn numerical_checks(device: &Device) -> Result<Value> {
    let cpu = Device::ndarray();
    let fused=if device != &cpu {Some(json!({
        "independent_f64":joint_checks(device,CodpulaExecution::FusedGpu)?,
        "cache_layouts":layout_checks(device,CodpulaExecution::FusedGpu)?,
        "checkpoint_restart":restart_checks(device,CodpulaExecution::FusedGpu,false)?,
        "cocommit_checkpoint_restart":restart_checks(device,CodpulaExecution::FusedGpu,true)?,
        "row_accumulation":row_accumulation_checks(device,CodpulaExecution::FusedGpu,false)?,
        "cocommit_row_accumulation":row_accumulation_checks(device,CodpulaExecution::FusedGpu,true)?,
    }))}else{None};
    Ok(json!({"passed":true,"chain_schedule":chain_checks()?,"independent_f64_cpu":joint_checks(&cpu,CodpulaExecution::Tensor)?,
        "independent_f64_selected_device":joint_checks(device,CodpulaExecution::Tensor)?,"cache_layouts":layout_checks(device,CodpulaExecution::Tensor)?,
        "checkpoint_restart":restart_checks(&cpu,CodpulaExecution::Tensor,false)?,
        "cocommit_checkpoint_restart":restart_checks(&cpu,CodpulaExecution::Tensor,true)?,
        "row_accumulation":row_accumulation_checks(device,CodpulaExecution::Tensor,false)?,
        "cocommit_row_accumulation":row_accumulation_checks(device,CodpulaExecution::Tensor,true)?,
        "fused_execution":fused,"warm_start_identity":warm_start_checks()?,
        "cli_resume_smoke_approach":"With checkpoint_every=4 and target_tokens=160 unchanged, run the five-update initial trajectory to checkpoint-5; separately resume the genuine checkpoint-4 under the same fingerprint to update 5 and compare endpoint/refill state. Do not edit checkpoints or add a stop-control."}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_chain_roundtrip_uses_declared_float_precision() {
        let (c,_,corpus)=restart_fixture();
        let mut state=TrainingState::new(&c,Some(&corpus),991).unwrap();
        let chain=&mut state.rows[0].chains[0];
        chain.visibility[0]=f32::from_bits(0x1fabc123);
        chain.retained_mass[0]=f32::from_bits(0x3eaaaaab);
        // Older saves expanded f32 fields through Value/f64 before writing.
        let bytes=serde_json::to_vec(&serde_json::to_value(&state).unwrap()).unwrap();
        let restored:TrainingState=serde_json::from_slice(&bytes).unwrap();
        assert_eq!(state,restored);
        let mut changed=restored.clone();
        changed.rng=changed.rng.wrapping_add(1);
        assert_ne!(state,changed);
        let mut changed=restored;
        let value=&mut changed.rows[0].chains[0].visibility[0];
        *value=f32::from_bits(value.to_bits()+1);
        assert_ne!(state,changed);
    }
    #[test]
    fn saved_feedback_separates_normalized_weights_from_retained_mass() {
        let (c,_,corpus)=restart_fixture();
        let state=TrainingState::new(&c,Some(&corpus),991).unwrap();
        state.validate(&c,991).unwrap();
        let mut valid=state.clone();
        valid.rows[0].chains[0].retained_mass[0]=0.25;
        valid.validate(&c,991).unwrap();
        let mut unnormalized=valid.clone();
        unnormalized.rows[0].chains[0].candidate_weights[0]*=0.5;
        assert!(unnormalized.validate(&c,991).is_err());
        for invalid_mass in [0.0,-0.1,1.1,f32::NAN,f32::INFINITY] {
            let mut corrupt=valid.clone();
            corrupt.rows[0].chains[0].retained_mass[0]=invalid_mass;
            assert!(corrupt.validate(&c,991).is_err());
        }
        let mut old_weights=valid;
        let chain=&mut old_weights.rows[0].chains[0];
        for weight in &mut chain.candidate_weights {*weight*=0.25;}
        assert!(old_weights.validate(&c,991).is_err());
    }
    #[test]
    fn confidence_schedule_keeps_self_generated_inputs_and_endpoint() {
        chain_checks().unwrap();
    }
    #[test]
    fn joint_nll_flows_and_all_gradients_match_exhaustive_f64() {
        joint_checks(&Device::ndarray(),CodpulaExecution::Tensor).unwrap();
    }
    #[test]
    fn streaming_checkpoint_continues_across_endpoint_and_refill() {
        restart_checks(&Device::ndarray(),CodpulaExecution::Tensor,false).unwrap();
    }
    #[test]
    fn cocommit_restart_keeps_optimizer_and_chain_state_bit_identical() {
        restart_checks(&Device::ndarray(),CodpulaExecution::Tensor,true).unwrap();
    }
    #[test]
    fn batched_cache_matches_per_block_and_l1_is_causal() {
        layout_checks(&Device::ndarray(),CodpulaExecution::Tensor).unwrap();
    }
    #[test]
    fn warm_start_requires_full_source_identity_except_explicit_budget_and_output_path() {
        warm_start_checks().unwrap();
    }
}

fn row_accumulation_checks(device:&Device,execution:CodpulaExecution,cocommit:bool)->Result<Value> {
    let (mut c,arm,corpus)=restart_fixture();
    c.batch_size=2;
    if cocommit {
        c.sbd.as_mut().unwrap().cocommit=Some(crate::config::CocommitConfig::default());
        c.sbd.as_mut().unwrap().joint_output=SbdJointOutput::CodpulaContextual;
    }
    c.sbd.as_mut().unwrap().joint_execution=execution;
    if !execution.is_tensor(){c.backend="vulkan".into();}
    let target=device.clone().autodiff();
    target.seed(613);
    let model=LanguageModel::new(&c,&arm,&target)?;
    let mut initial=TrainingState::new(&c,Some(&corpus),613)?;
    if cocommit {
        for row in &mut initial.rows {
            let chain=&mut row.chains[0];
            chain.revealed[0]=true;
            chain.values[0]=(row.target[chain.start] as usize+1)%c.vocab();
            chain.visibility[0]=1.0;
        }
    }
    let mut bundle_state_a=initial.clone();
    let mut bundle_state_b=initial.clone();
    let bundles_a=sample_announcements(&c,&mut bundle_state_a);
    let bundles_b=sample_announcements(&c,&mut bundle_state_b);
    ensure!(bundles_a==bundles_b && bundle_state_a.rng==bundle_state_b.rng,
        "TCT row chunking changed pre-partition announcement masks or RNG");
    if let Some(bundles)=&bundles_a {
        for (row,saved) in initial.rows.iter().enumerate() {
            for (index,chain) in saved.chains.iter().enumerate().filter(|(_,chain)|chain.len()>1) {
                ensure!(bundles[row][index].iter().zip(&chain.permanent).all(|(&marked,evidence)|!marked || evidence.is_none()),
                    "CoCommit announcement escaped the all-free target set");
                ensure!(bundles[row][index].iter().any(|&marked|marked),"CoCommit omitted a nonempty announcement");
            }
        }
    }
    let mut full_state=initial.clone();
    let full=backward_passes(&c,&model,&mut full_state,&target,SbdBlockExecution::Batched)?;
    c.tct=Some(TctConfig {rows_per_pass:1});
    let mut chunked_state=initial.clone();
    let chunked=backward_passes(&c,&model,&mut chunked_state,&target,SbdBlockExecution::Batched)?;
    let loss=errors(&[chunked.loss as f32],&[full.loss],"full batch versus row-accumulated TCT loss")?;
    let expected=crate::kron_benchmark::parameters(&model,Some(&full.gradients))?;
    let actual=crate::kron_benchmark::parameters(&model,Some(&chunked.gradients))?;
    ensure!(expected.len()==actual.len(),"row accumulation gradient count differs");
    let mut gradients=Vec::with_capacity(actual.len());
    for (index,((shape,values),(reference_shape,reference))) in actual.into_iter().zip(expected).enumerate() {
        ensure!(shape==reference_shape,"row accumulation gradient shape differs");
        gradients.push(json!({"tensor":index,"errors":errors(&values,&reference.iter().map(|&x|f64::from(x)).collect::<Vec<_>>(),"row accumulation gradient")?}));
    }
    let full_flows=full.flows.context("full batch TCT flows")?;
    let chunked_flows=chunked.flows.context("row-accumulated TCT flows")?;
    ensure!(full_flows.active_blocks==chunked_flows.active_blocks,"row accumulation changed EM normalization count");
    let marker_gradient_max=if cocommit {
        let marker=model.cocommit_marker.as_ref().context("CoCommit marker missing")?;
        let gradient=host(full.gradients.get::<2>(marker.id).context("TCT CoCommit marker gradient missing")?)?;
        let maximum=gradient.iter().copied().map(f32::abs).fold(0.0,f32::max);
        ensure!(maximum.is_finite() && maximum>1e-9,"TCT CoCommit marker received no branch gradient");
        ensure!(full_flows.active_blocks==2*full.noisy_blocks,"TCT CoCommit A/B flow weighting omitted a branch");
        Some(maximum)
    } else {None};
    let mut flows=Vec::new();
    for ((name,values),(reference_name,reference)) in flow_values(&chunked_flows)?.into_iter().zip(flow_values(&full_flows)?) {
        ensure!(name==reference_name,"row accumulation flow name differs");
        flows.push(json!({"quantity":name,"errors":errors(&values,&reference.iter().map(|&x|f64::from(x)).collect::<Vec<_>>(),"row accumulation flow")?}));
    }
    ensure!(full_state.rng==chunked_state.rng,"row accumulation changed the persisted proposal stream");
    for (a,b) in full_state.rows.iter().zip(&chunked_state.rows) {
        for (a,b) in a.chains.iter().zip(&b.chains) {
            ensure!(a.values==b.values && a.revealed==b.revealed && a.permanent==b.permanent
                && a.round==b.round && a.candidate_ids==b.candidate_ids,
                "row accumulation changed token decisions, reveal choices or RNG");
            for (field,actual,reference) in [
                ("visibility",&b.visibility,&a.visibility),("candidate weights",&b.candidate_weights,&a.candidate_weights),
                ("retained mass",&b.retained_mass,&a.retained_mass),
            ] {
                let _=errors(actual,&reference.iter().map(|&x|f64::from(x)).collect::<Vec<_>>(),field)?;
            }
        }
    }
    ensure!(full.clean_calls==1 && chunked.clean_calls==2 && full.noisy_blocks==chunked.noisy_blocks,
        "row accumulation changed example count");
    if cocommit {
        ensure!(full.work.prefix_batches==1 && chunked.work.prefix_batches==2,
            "CoCommit TCT prefix batches were not accumulated by row chunk");
        ensure!(full.work.head_rows==2*full.work.logical_blocks && chunked.work.head_rows==2*chunked.work.logical_blocks,
            "CoCommit TCT head row units do not count both branches");
    }
    Ok(json!({"passed":true,"batch":2,"context":32,"rows_per_pass":1,"execution":execution,"cocommit":cocommit,"loss":loss,
        "gradients":gradients,"flows":flows,"active_blocks":full_flows.active_blocks,"marker_gradient_max":marker_gradient_max,
        "coordination_work_full_batch":full.work,"coordination_work_chunked":chunked.work,
        "same_neural_and_em_normalization":true,"same_announcement_bundles":cocommit,
        "same_chain_token_decisions_and_feedback":true,"optimizer_updates_per_full_batch":1}))
}

#[cfg(test)]
mod row_accumulation_tests {
    #[test]
    fn full_batch_and_row_chunks_keep_gradients_flows_and_rng() {
        super::row_accumulation_checks(&burn::prelude::Device::ndarray(),crate::codpula::CodpulaExecution::Tensor,false).unwrap();
    }
    #[test]
    fn cocommit_row_chunks_keep_markers_flows_and_chain_rng() {
        super::row_accumulation_checks(&burn::prelude::Device::ndarray(),crate::codpula::CodpulaExecution::Tensor,true).unwrap();
    }
}
