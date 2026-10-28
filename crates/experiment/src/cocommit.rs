//! Shared-input marked-tail training and temporary block assignment.
use crate::{
    config::{Config, SbdBlockExecution, SbdFeedback},
    data::{Batch, Example, Rng},
    model::{self, DenoisingOutput, LanguageModel},
    sbd::{self, SbdTrainingResult, TrainingBlock},
    sbd_training::{BatchedTrainingCache, CleanInputs, NoisyPrefix, TrainingCache},
    soft::SoftState,
};
use anyhow::{Context, Result, ensure};
use burn::{module::Module, prelude::*, tensor::activation::log_softmax};
use delta_mixers::profile::{Sections, Unprofiled};
use serde_json::{Value, json};

pub(super) fn sample_bundles(blocks: &[TrainingBlock], maximum: usize, seed: u64) -> Vec<Vec<bool>> {
    let mut rng = Rng(seed ^ 0x434f434f4d4d4954);
    blocks.iter().map(|block| {
        let mut marks = vec![false; block.masked.len()];
        let mut positions = block.masked.iter().enumerate().filter_map(|(i,&masked)| masked.then_some(i)).collect::<Vec<_>>();
        if positions.is_empty() { return marks; }
        let size = 1 + rng.index(positions.len().min(maximum));
        for j in 0..size {
            let other = j + rng.index(positions.len()-j);
            positions.swap(j, other);
            marks[positions[j]] = true;
        }
        marks
    }).collect()
}
fn marker_tensor(marks: &[bool], rows: usize, width: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(marks.iter().map(|&x| if x {1.0f32} else {0.0}).collect::<Vec<_>>(), [rows,width]),device)
}
fn token_loss(logits: Tensor<3>, targets: Tensor<2,Int>, weights: Vec<f32>, denominator: f64) -> Tensor<1> {
    let [rows,width,vocab] = logits.dims();
    let device = logits.device();
    let weights = Tensor::<2>::from_data(TensorData::new(weights,[rows*width,1]),&device);
    -(log_softmax(logits.reshape([rows*width,vocab]),1).gather(1,targets.reshape([rows*width,1]))*weights).sum()/denominator
}
pub(crate) fn add_branch_losses(
    model: &LanguageModel, c: &Config, a: DenoisingOutput, b: DenoisingOutput,
    soft: &SoftState, targets: Tensor<2,Int>, marks: &[bool], target_weights: &[f32],
    denominator: f64, result: &mut SbdTrainingResult,
) -> Result<Option<crate::codpula::PreparedCircuit>> {
    let [rows,width,_] = a.logits.dims();
    ensure!(marks.len()==rows*width && target_weights.len()==rows*width
        && soft.active.len()==rows*width && soft.observed.len()==rows*width,
        "CoCommit branch loss layout differs");
    let use_b = marks.chunks_exact(width).map(|row|row.iter().any(|&mark|mark)).collect::<Vec<_>>();
    let any_b=use_b.iter().any(|&used|used);
    let selected_rows = Tensor::<2,Bool>::from_data(TensorData::new(use_b,[rows,1]),&targets.device())
        .expand([rows,width]);
    let vocab = a.logits.dims()[2];
    let depth = a.features.dims()[2];
    let b = DenoisingOutput {
        logits: a.logits.clone().mask_where(selected_rows.clone().unsqueeze_dim::<3>(2).expand([rows,width,vocab]),b.logits),
        features: a.features.clone().mask_where(selected_rows.unsqueeze_dim::<3>(2).expand([rows,width,depth]),b.features),
    };
    let gold=targets.clone().into_data().try_to_vec_as::<i32>()
        .map_err(|e|anyhow::anyhow!("CoCommit targets: {e:?}"))?;
    let a_weights=soft.active.iter().zip(&soft.observed).zip(target_weights).enumerate()
        .map(|(i,((&active,visible),&weight))|if !active {0.0} else {
            weight*match visible {None=>1.0,Some(value) if *value==gold[i] as usize=>0.2,Some(_)=>0.3}
        }).collect::<Vec<_>>();
    let b_weights=a_weights.iter().zip(&soft.observed).zip(marks).enumerate()
        .map(|(i,((&weight,visible),&marked))|if !use_b_row(marks,width,i/width) || (visible.is_none() && !marked) {0.0}else{weight})
        .collect::<Vec<_>>();
    if model.codpula.is_some() {
        // Current-block visible drafts are inputs to be corrected, never immutable evidence.
        let mut mutable=soft.clone();
        mutable.observed.fill(None);
        let (a_circuit,observed)=sbd::prepare_distribution(model,a,&mutable,1.0)?;
        let weights=Tensor::from_data(TensorData::new(a_weights,[rows,width]),&targets.device());
        result.add_circuit_loss(a_circuit.joint_marginal_ce(targets.clone(),targets.clone(),observed,weights,denominator)?);
        if any_b {
            let (b_circuit,observed)=sbd::prepare_distribution(model,b,&mutable,1.0)?;
            let weights=Tensor::from_data(TensorData::new(b_weights,[rows,width]),&targets.device());
            result.add_circuit_loss(b_circuit.joint_marginal_ce(targets.clone(),targets,observed,weights,denominator)?);
            Ok(Some(b_circuit))
        } else {Ok(Some(a_circuit))}
    } else {
        result.loss=result.loss.clone()+token_loss(a.logits,targets.clone(),a_weights,denominator)
            +token_loss(b.logits,targets,b_weights,denominator);
        Ok(None)
    }
}
fn use_b_row(marks:&[bool],width:usize,row:usize)->bool {
    marks[row*width..(row+1)*width].iter().any(|&mark|mark)
}

/// Counts batched API executions, not device kernel launches or standalone-equivalent cost.
/// `*_dispatches`/`*_batches` count physical batched operations; `*_rows` count
/// padded block-row units processed (a doubled tail has one dispatch and 2C rows).
/// `*_used_rows` count results consumed, excluding discarded zero-mark B lanes.
/// Message/evidence/FFBS counts include A selection as well as selected proposal
/// and O-only feedback queries. Final clean replay is recorded separately.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct CoordinationWork {
    pub logical_blocks: usize,
    pub empty_announcement_rows: usize,
    pub prefix_batches: usize,
    pub prefix_rows: usize,
    pub a_tail_dispatches: usize,
    pub a_tail_rows: usize,
    pub b_tail_dispatches: usize,
    pub b_tail_rows: usize,
    pub paired_tail_dispatches: usize,
    pub paired_tail_rows: usize,
    pub a_tail_used_rows: usize,
    pub head_used_rows: usize,
    pub b_tail_used_rows: usize,
    pub paired_tail_used_rows: usize,
    pub circuit_query_dispatched_rows: usize,
    pub circuit_query_used_rows: usize,
    pub ffbs_dispatched_rows: usize,
    pub ffbs_used_rows: usize,
    pub b_circuit_used_rows: usize,
    pub head_dispatches: usize,
    pub head_rows: usize,
    pub circuit_preparations: usize,
    pub message_passes: usize,
    pub evidence_queries: usize,
    pub ffbs_passes: usize,
    pub final_replay_dispatches: usize,
    pub final_replay_rows: usize,
}
impl CoordinationWork {
    pub(crate) fn add(&mut self, other: Self) {
        self.logical_blocks += other.logical_blocks;
        self.empty_announcement_rows += other.empty_announcement_rows;
        self.prefix_batches += other.prefix_batches;
        self.prefix_rows += other.prefix_rows;
        self.a_tail_dispatches += other.a_tail_dispatches;
        self.a_tail_rows += other.a_tail_rows;
        self.b_tail_dispatches += other.b_tail_dispatches;
        self.b_tail_rows += other.b_tail_rows;
        self.paired_tail_dispatches += other.paired_tail_dispatches;
        self.paired_tail_rows += other.paired_tail_rows;
        self.a_tail_used_rows += other.a_tail_used_rows;
        self.head_used_rows += other.head_used_rows;
        self.b_tail_used_rows += other.b_tail_used_rows;
        self.paired_tail_used_rows += other.paired_tail_used_rows;
        self.circuit_query_dispatched_rows += other.circuit_query_dispatched_rows;
        self.circuit_query_used_rows += other.circuit_query_used_rows;
        self.ffbs_dispatched_rows += other.ffbs_dispatched_rows;
        self.ffbs_used_rows += other.ffbs_used_rows;
        self.b_circuit_used_rows += other.b_circuit_used_rows;
        self.head_dispatches += other.head_dispatches;
        self.head_rows += other.head_rows;
        self.circuit_preparations += other.circuit_preparations;
        self.message_passes += other.message_passes;
        self.evidence_queries += other.evidence_queries;
        self.ffbs_passes += other.ffbs_passes;
        self.final_replay_dispatches += other.final_replay_dispatches;
        self.final_replay_rows += other.final_replay_rows;
    }
}

pub(crate) fn training_pair_profiled<P: Sections>(
    model: &LanguageModel, c: &Config, cache: &BatchedTrainingCache, soft: &SoftState,
    bundles: &[Vec<bool>], profile: &mut P,
) -> Result<(DenoisingOutput,DenoisingOutput,CoordinationWork)> {
    let rows = bundles.len();
    ensure!(rows > 0 && soft.observed.len().is_multiple_of(rows),
        "invalid CoCommit training pair layout");
    let width = soft.observed.len()/rows;
    ensure!(bundles.iter().all(|marks| marks.len() == width), "CoCommit bundle width differs");
    let marks = bundles.iter().flat_map(|marks| marks.iter().copied()).collect::<Vec<_>>();
    let prefix = model.cached_blocks_prefix_profiled(cache,soft,
        c.sbd.as_ref().unwrap().feedback == crate::config::SbdFeedback::HardMask,profile);
    if !marks.iter().any(|&mark|mark) {
        let a = model.cached_blocks_tail_profiled(cache,prefix,None,profile);
        let work = CoordinationWork {logical_blocks:rows,empty_announcement_rows:rows,prefix_batches:1,prefix_rows:rows,
            a_tail_dispatches:1,a_tail_rows:rows,a_tail_used_rows:rows,
            head_dispatches:1,head_rows:rows,head_used_rows:rows,..CoordinationWork::default()};
        return Ok((a.clone(),a,work));
    }
    let (a,b) = model.cached_blocks_pair_profiled(cache,prefix,
        marker_tensor(&marks,rows,width,&soft.candidates.ids.device()),profile);
    let separate = model.cocommit_config.as_ref().unwrap().execution != crate::config::CocommitExecution::Doubled;
    let used_b = bundles.iter().filter(|marks|marks.iter().any(|&mark|mark)).count();
    let work = CoordinationWork {
        logical_blocks: rows, empty_announcement_rows:rows-used_b, prefix_batches: 1, prefix_rows: rows,
        a_tail_dispatches: usize::from(separate), a_tail_rows: if separate {rows} else {0},
        b_tail_dispatches: usize::from(separate), b_tail_rows: if separate {rows} else {0},
        paired_tail_dispatches: usize::from(!separate), paired_tail_rows: if separate {0} else {2*rows},
        a_tail_used_rows: if separate {rows} else {0},
        b_tail_used_rows: used_b,
        paired_tail_used_rows: if separate {0} else {rows+used_b},
        head_dispatches: if separate {2} else {1}, head_rows: rows*2, head_used_rows: rows+used_b,
        ..CoordinationWork::default()
    };
    Ok((a,b,work))
}

pub(super) fn training_loss_sampled_profiled<P:Sections>(model:&LanguageModel,c:&Config,rows:&[Example],batch:&Batch,blocks:&[TrainingBlock],bundles:&[Vec<bool>],profile:&mut P)->Result<SbdTrainingResult> {
    let s=c.sbd.as_ref().context("CoCommit requires SBD")?;
    ensure!(s.cocommit.is_some() && blocks.len()==bundles.len(),"invalid sampled CoCommit bundle layout");
    let mut trace=CleanInputs::new(blocks.iter().map(|block|block.start+1));
    let clean_logits=model.forward_profiled(batch,&mut trace);
    let denominator=(batch.input.dims()[0]*batch.input.dims()[1]) as f64;
    let device=clean_logits.device();
    let mut result=SbdTrainingResult {loss:model.loss(clean_logits,batch),circuit_flows:None,
        unused_parameters:model::unused_denoising_parameters(!blocks.is_empty(),bundles.iter().any(|b|b.iter().any(|&m|m)))};
    if s.feedback==SbdFeedback::HardMask || blocks.is_empty() {result.unused_parameters.extend(model::unused_soft_parameters());}
    if blocks.is_empty() {return Ok(result);}
    let teacher=model.valid();
    // Both execution modes consume the same detached own-draft corruption.
    let width=blocks.iter().map(|block|block.masked.len()).max().unwrap();
    let locations=blocks.iter().map(|block|(block.row,block.start+1)).collect::<Vec<_>>();
    let cache=model.training_cache_batched(&trace,&locations);
    let active=blocks.iter().flat_map(|block|(0..width).map(move |i|i<block.masked.len())).collect::<Vec<_>>();
    let mut soft=SoftState::new(c.vocab(),s.candidate_width,vec![None;blocks.len()*width],active,&device,c.soft_selection_backend)?;
    let draft=teacher.cached_blocks_profiled(&cache.detached(),&soft,true,profile);
    let draft_logs=if teacher.codpula.is_some() {
        let (circuit,tokens,observed)=sbd::prepare_circuit(&teacher,draft.detach(),&soft,1.0)?;
        circuit.marginals(tokens,observed)?.log()
    }else{log_softmax(draft.logits.detach(),2)};
    let values=crate::soft::ids_host(draft_logs.clone().reshape([blocks.len()*width,c.vocab()]).argmax(1));
    let mut targets=Vec::with_capacity(blocks.len()*width);
    let mut weights=Vec::with_capacity(blocks.len()*width);
    let mut marks=Vec::with_capacity(blocks.len()*width);
    for (r,(block,bundle)) in blocks.iter().zip(bundles).enumerate() {
        for i in 0..width {
            let real=i<block.masked.len();
            soft.observed[r*width+i]=if real && !block.masked[i] {Some(values[r*width+i] as usize)}else{None};
            targets.push(if real {rows[block.row].target[block.start+i]}else{0});
            weights.push(if real {rows[block.row].weight[block.start+i]}else{0.0});
            marks.push(real && bundle[i]);
        }
    }
    // Rebuild hard-visible draft embeddings, then optionally retain the draft's soft
    // predictions at the masked positions. No random vocabulary errors are substituted.
    soft=SoftState::new(c.vocab(),s.candidate_width,soft.observed,soft.active,&device,c.soft_selection_backend)?;
    if s.feedback==SbdFeedback::SoftFeedback {
        soft.update(draft_logs.clone().reshape([blocks.len()*width,c.vocab()]),1.0,s.candidate_width,
            model.soft_confidence.as_ref().context("soft feedback head missing")?)?;
    }
    if s.block_execution==SbdBlockExecution::PerBlock {
        for (r,block) in blocks.iter().enumerate() {
            let n=block.masked.len();
            let mut input=sbd::state(c,soft.observed[r*width..r*width+n].to_vec(),&device)?;
            if s.feedback==SbdFeedback::SoftFeedback {
                input.update(draft_logs.clone().slice([r..r+1,0..n,0..c.vocab()]).reshape([n,c.vocab()]),1.0,s.candidate_width,
                    model.soft_confidence.as_ref().context("soft feedback head missing")?)?;
            }
            let cache=model.training_cache(&trace,block.row,&[block.start+1]);
            let prefix=model.cached_block_prefix_profiled(&cache,0,&input,s.feedback==SbdFeedback::HardMask,profile);
            let a=model.cached_block_tail_profiled(&cache,0,prefix.clone(),None,profile);
            let selected=&marks[r*width..r*width+n];
            let b=if selected.iter().any(|&mark|mark) {model.cached_block_tail_profiled(&cache,0,prefix,Some(marker_tensor(selected,1,n,&device)),profile)}else{a.clone()};
            let clean=Tensor::from_data(TensorData::new(targets[r*width..r*width+n].to_vec(),[1,n]),&device);
            let _=add_branch_losses(model,c,a,b,&input,clean,selected,&weights[r*width..r*width+n],denominator,&mut result)?;
        }
        return Ok(result);
    }
    let row_bundles=marks.chunks_exact(width).map(|row|row.to_vec()).collect::<Vec<_>>();
    let (a,b,_)=training_pair_profiled(model,c,&cache,&soft,&row_bundles,profile)?;
    let targets=Tensor::from_data(TensorData::new(targets,[blocks.len(),width]),&device);
    let _=add_branch_losses(model,c,a,b,&soft,targets,&marks,&weights,denominator,&mut result)?;
    Ok(result)
}

pub(super) fn prepare_clean_cache(model: &LanguageModel, prefix: &[i32], device: &Device) -> TrainingCache {
    assert!(!prefix.is_empty());
    let mut trace = CleanInputs::new([prefix.len()]);
    let _ = model.forward_profiled(&sbd::prefix_batch(prefix,device),&mut trace);
    model.training_cache(&trace,0,&[prefix.len()])
}
pub(super) fn prepare_reconstruction(model: &LanguageModel, prefix: &[i32], soft: &SoftState, hard: bool, device: &Device) -> (TrainingCache,NoisyPrefix) {
    assert!(!prefix.is_empty() && !soft.observed.is_empty());
    let cache = prepare_clean_cache(model,prefix,device);
    let hidden = model.cached_block_prefix_profiled(&cache,0,soft,hard,&mut Unprofiled);
    (cache,hidden)
}
pub(super) fn reconstruct_output(model: &LanguageModel, prefix: &[i32], soft: &SoftState, hard: bool, marks: Option<Tensor<2>>, device: &Device) -> DenoisingOutput {
    let (cache,hidden) = prepare_reconstruction(model,prefix,soft,hard,device);
    model.cached_block_tail_profiled(&cache,0,hidden,marks,&mut Unprofiled)
}
fn checked_entropy_rows(values:&[f32],vocab:usize,active:&[bool])->Result<Vec<f64>> {
    ensure!(vocab>0 && values.len()==active.len()*vocab,"invalid CoCommit probability shape");
    values.chunks_exact(vocab).zip(active).enumerate().map(|(index,(row,&active))|{
        if !active {return Ok(0.0);}
        let mut mass=0.0f64;
        let mut entropy=0.0f64;
        for &value in row {
            ensure!(value.is_finite() || value==f32::NEG_INFINITY,
                "CoCommit probability row {index} contains NaN or positive infinity");
            if value==f32::NEG_INFINITY {continue;}
            let probability=f64::from(value).exp();
            mass+=probability;
            if probability>0.0 {entropy-=probability*f64::from(value);}
        }
        ensure!(mass.is_finite() && mass>0.0 && entropy.is_finite(),
            "CoCommit probability row {index} has zero or nonfinite support");
        Ok(entropy)
    }).collect()
}
struct Prediction {
    host_logs: Vec<f32>,
    proposals: Vec<usize>,
    feedback: Tensor<2>,
    feedback_temperature: f64,
}
fn prediction(model: &LanguageModel, c: &Config, output: DenoisingOutput, soft: &SoftState, temperature: f64, assignments: bool) -> Result<Prediction> {
    let n = soft.observed.len();
    if model.codpula.is_some() {
        let mut mutable=soft.clone();
        mutable.observed.fill(None);
        let (circuit,tokens,observed) = sbd::prepare_circuit(model,output,&mutable,temperature)?;
        let logs = circuit.marginals(tokens.clone(),observed.clone())?.log();
        let host_logs=logs.clone().into_data().try_to_vec::<f32>().map_err(|e|anyhow::anyhow!("CoCommit probabilities {e:?}"))?;
        let _=checked_entropy_rows(&host_logs,c.vocab(),&soft.active)?;
        let proposals=if assignments {greedy_tokens(&host_logs,c.vocab())}else{Vec::new()};
        Ok(Prediction { feedback: logs.reshape([n,c.vocab()]), host_logs,
            proposals, feedback_temperature:1.0 })
    } else {
        let logs = log_softmax(output.logits.clone()/temperature,2);
        let host_logs=logs.into_data().try_to_vec::<f32>().map_err(|e|anyhow::anyhow!("CoCommit probabilities {e:?}"))?;
        let _=checked_entropy_rows(&host_logs,c.vocab(),&soft.active)?;
        let proposals = if assignments {
            host_logs.chunks_exact(c.vocab()).zip(&soft.active).map(|(row,&active)|{
                if active {(0..row.len()).max_by(|&a,&b|row[a].total_cmp(&row[b]).then(b.cmp(&a))).unwrap()}else{0}
            }).collect()
        } else {Vec::new()};
        Ok(Prediction { host_logs,proposals,feedback:output.logits.reshape([n,c.vocab()]),feedback_temperature:temperature })
    }
}
pub(crate) fn confidence_bundle(logs:&[f32],vocab:usize,positions:&[usize])->Vec<usize> {
    let mut ranked=positions.iter().map(|&i|(i,logs[i*vocab..(i+1)*vocab].iter().copied().fold(f32::NEG_INFINITY,f32::max))).collect::<Vec<_>>();
    ranked.sort_by(|a,b|b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let selected=ranked.iter().filter(|(_,confidence)|f64::from(*confidence).exp()>0.7).map(|&(i,_)|i).collect::<Vec<_>>();
    if selected.is_empty() {ranked.first().map(|&(i,_)|vec![i]).unwrap_or_default()}else{selected}
}
pub(crate) fn greedy_tokens(logs:&[f32],vocab:usize)->Vec<usize> {
    logs.chunks_exact(vocab).map(|row|(0..vocab).max_by(|&a,&b|row[a].total_cmp(&row[b]).then(b.cmp(&a))).unwrap()).collect()
}
fn select(c: &Config, prediction: &Prediction, soft: &SoftState, _last_pass: bool) -> Result<Vec<usize>> {
    let unresolved=soft.observed.iter().zip(&soft.active).enumerate().filter_map(|(i,(value,&active))|(active && value.is_none()).then_some(i)).collect::<Vec<_>>();
    Ok(confidence_bundle(&prediction.host_logs,c.vocab(),&unresolved))
}
pub(super) struct Assignment {
    pub proposals: Vec<usize>,
    pub commits: Vec<usize>,
    pub feedback: Tensor<2>,
    pub feedback_temperature: f64,
    pub host_logs: Vec<f32>,
}
pub(super) fn assign(model: &LanguageModel, c: &Config, cache: &TrainingCache, soft: &SoftState, device: &Device) -> Result<Assignment> {
    let s = c.sbd.as_ref().unwrap();
    let n = soft.observed.len();
    let hidden = model.cached_block_prefix_profiled(cache,0,soft,s.feedback==SbdFeedback::HardMask,&mut Unprofiled);
    let output = model.cached_block_tail_profiled(cache,0,hidden.clone(),None,&mut Unprofiled);
    let a = prediction(model,c,output,soft,c.soft_temperature,true)?;
    let announcement = select(c,&a,soft,false)?;
    if announcement.is_empty() { return Ok(Assignment {proposals:a.proposals, commits:announcement,feedback:a.feedback,feedback_temperature:a.feedback_temperature,host_logs:a.host_logs}); }
    let mut marks = vec![false;n];
    for &i in &announcement {marks[i]=true;}
    let output = model.cached_block_tail_profiled(cache,0,hidden,Some(marker_tensor(&marks,1,n,device)),&mut Unprofiled);
    let b = prediction(model,c,output,soft,c.soft_temperature,true)?;
    let mut logs=a.host_logs;
    for i in 0..n {
        if marks[i] || soft.observed[i].is_some() {
            logs[i*c.vocab()..(i+1)*c.vocab()].copy_from_slice(&b.host_logs[i*c.vocab()..(i+1)*c.vocab()]);
        }
    }
    let proposals=greedy_tokens(&logs,c.vocab());
    let feedback=Tensor::from_data(TensorData::new(logs.clone(),[n,c.vocab()]),device);
    Ok(Assignment {proposals,commits:announcement,feedback,feedback_temperature:1.0,host_logs:logs})
}

pub(super) fn teacher_prefix_evaluate(model: &LanguageModel, c: &Config, examples: &[Example], device: &Device) -> Result<Value> {
    let s = c.sbd.as_ref().unwrap();
    let model = model.valid();
    let mut nll = 0.0;
    let mut marked_nll = 0.0;
    let mut count = 0usize;
    let mut marked_count = 0usize;
    let mut hits = 0usize;
    let mut marked_hits = 0usize;
    for row in examples {
        if model.codpula.is_some() {ensure!(row.weight.iter().all(|&w|w==1.0),"Codpula joint event evaluation requires unit valid target weights");}
        for &length in &s.block_lengths {
            for start in (0..row.target.len()).step_by(length) {
                let n = length.min(row.target.len()-start);
                let mut soft = sbd::state(c,vec![None;n],device)?;
                let prefix = &row.input[..start+1];
                let cache=prepare_clean_cache(&model,prefix,device);
                if s.feedback==SbdFeedback::SoftFeedback {
                    let teacher=model.cached_block_profiled(&cache,0,&soft,false,&mut Unprofiled);
                    sbd::update_feedback(&model,c,teacher,&mut soft)?;
                }
                let hidden=model.cached_block_prefix_profiled(&cache,0,&soft,s.feedback==SbdFeedback::HardMask,&mut Unprofiled);
                let output_a = model.cached_block_tail_profiled(&cache,0,hidden.clone(),None,&mut Unprofiled);
                let a = prediction(&model,c,output_a,&soft,1.0,true)?;
                let announcement = select(c,&a,&soft,false)?;
                let mut marks = vec![false;n];
                for &i in &announcement {marks[i]=true;}
                let output_b = model.cached_block_tail_profiled(&cache,0,hidden,Some(marker_tensor(&marks,1,n,device)),&mut Unprofiled);
                let b = prediction(&model,c,output_b,&soft,1.0,false)?;
                let av=a.host_logs;
                let bv=b.host_logs;
                for j in 0..n {
                    if row.weight[start+j]==0.0 {continue;}
                    let gold = row.target[start+j] as usize;
                    let best = |values:&[f32]| (0..c.vocab()).max_by(|&a,&b|values[a].total_cmp(&values[b]).then(b.cmp(&a))).unwrap();
                    let ar = &av[j*c.vocab()..(j+1)*c.vocab()];
                    let br = &bv[j*c.vocab()..(j+1)*c.vocab()];
                    nll-=ar[gold] as f64;
                    hits+=usize::from(best(ar)==gold);count+=1;
                    if marks[j] {
                        marked_nll-=br[gold] as f64;
                        marked_hits+=usize::from(best(br)==gold);marked_count+=1;
                    }
                }
            }
        }
    }
    ensure!(count>0,"CoCommit evaluation needs targets");
    Ok(json!({"block_nll":nll/count as f64,"block_accuracy":hits as f64/count as f64,"predicted_tokens":count,
        "marked_subset_nll":if marked_count>0 {Some(marked_nll/marked_count as f64)} else {None},
        "marked_subset_accuracy":if marked_count>0 {Some(marked_hits as f64/marked_count as f64)} else {None},"marked_predicted_tokens":marked_count,
        "block_lengths":s.block_lengths,"joint_output":s.joint_output,"cocommit":s.cocommit,
        "conditioning":"all targets initially masked; detached soft teacher; A all-target marginal CE and B announced-target marginal CE, including noisy singletons",
        "nll_reduction":"A sum marginal token CE / all valid targets; B sum announced marginal token CE / announced valid targets"}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weighted_visible_targets_are_not_hard_circuit_evidence() {
        use crate::config::{Arm,CocommitConfig,SbdConfig,SbdJointOutput};
        let device=Device::ndarray().autodiff();
        for joint in [false,true] {
            device.seed(127);
            let c=Config{task:"lm".into(),d_model:16,layers:1,
                sbd:Some(SbdConfig{cocommit:Some(CocommitConfig::default()),
                    joint_output:if joint{SbdJointOutput::Codpula}else{SbdJointOutput::Factorized},..Default::default()}),..Default::default()};
            let arm=Arm{name:"weighted-visible".into(),mixer:"kronlstm".into(),cache:"surprise".into()};
            let model=LanguageModel::new(&c,&arm,&device).unwrap();
            let soft=sbd::state(&c,vec![None,Some(9),Some(7),None],&device).unwrap();
            let a=Tensor::<3>::zeros([1,4,c.vocab()],&device).require_grad();
            let b=Tensor::<3>::zeros([1,4,c.vocab()],&device).require_grad();
            let output=|logits:Tensor<3>|DenoisingOutput{logits,features:Tensor::zeros([1,4,16],&device)};
            let clean=Tensor::<2,Int>::from_data([[3,5,7,11]],&device);
            let mut mutable=soft.clone();mutable.observed.fill(None);
            let logs=|logits:Tensor<3>| {
                if joint {
                    let (circuit,tokens,observed)=sbd::prepare_circuit(&model,output(logits),&mutable,1.0).unwrap();
                    circuit.marginals(tokens,observed).unwrap().log()
                }else{log_softmax(logits,2)}
            };
            let a_logs=logs(a.clone()).into_data().try_to_vec::<f32>().unwrap();
            let b_logs=logs(b.clone()).into_data().try_to_vec::<f32>().unwrap();
            let expected=(0..4).map(|i|{
                let target=[3usize,5,7,11][i];
                -f64::from(a_logs[i*c.vocab()+target])*[1.0,0.3,0.2,1.0][i]
                -f64::from(b_logs[i*c.vocab()+target])*[1.0,0.3,0.2,0.0][i]
            }).sum::<f64>()/4.0;
            let mut result=SbdTrainingResult{loss:Tensor::zeros([1],&device),circuit_flows:None,unused_parameters:Vec::new()};
            add_branch_losses(&model,&c,output(a.clone()),output(b.clone()),&soft,clean,&[true,false,false,false],&[1.0;4],4.0,&mut result).unwrap();
            sbd::compare(&[model::scalar(result.loss.clone()).unwrap()],&[expected]).unwrap();
            let gradients=result.loss.backward();
            let a_grad=a.grad(&gradients).unwrap().into_data().try_to_vec::<f32>().unwrap();
            let b_grad=b.grad(&gradients).unwrap().into_data().try_to_vec::<f32>().unwrap();
            for i in [1usize,2] {
                assert!(a_grad[i*c.vocab()..(i+1)*c.vocab()].iter().any(|g|g.abs()>1e-8));
                assert!(b_grad[i*c.vocab()..(i+1)*c.vocab()].iter().any(|g|g.abs()>1e-8));
            }
            assert_eq!(result.circuit_flows.is_some(),joint);
        }
    }
    #[test]
    fn confidence_selection_and_argmax_share_the_same_event() {
        let logs=[0.8f32.ln(),0.2f32.ln(),0.1f32.ln(),0.9f32.ln(),0.6f32.ln(),0.4f32.ln()];
        assert_eq!(confidence_bundle(&logs,2,&[0,1,2]),vec![1,0]);
        assert_eq!(greedy_tokens(&logs,2),vec![0,1,0]);
        assert_eq!(confidence_bundle(&logs,2,&[2]),vec![2]);
        assert!(confidence_bundle(&logs,2,&[]).is_empty());
    }
    #[test]
    fn bundles_are_masked_uniform_size_draws_with_independent_seed() {
        let blocks = vec![
            TrainingBlock { row:0,start:0,length:1,masked:vec![true] },
            TrainingBlock { row:0,start:1,length:4,masked:vec![true,false,true,true] },
            TrainingBlock { row:1,start:0,length:4,masked:vec![false,true,false,true] },
        ];
        let bundles = sample_bundles(&blocks,16,77);
        assert_eq!(bundles[0],vec![true]);
        assert_eq!(bundles,sample_bundles(&blocks,16,77));
        for (block,bundle) in blocks.iter().zip(&bundles) {
            let count=bundle.iter().filter(|&&x|x).count();
            assert!(count>0 && count<=block.masked.iter().filter(|&&x|x).count());
            assert!(bundle.iter().zip(&block.masked).all(|(&chosen,&masked)|!chosen || masked));
        }
        for bundle in sample_bundles(&blocks,1,77).iter() {
            assert_eq!(bundle.iter().filter(|&&x|x).count(),1);
        }
    }
    #[test]
    fn entropy_rejects_nonfinite_or_unsupported_active_rows() {
        let negative_infinity=f32::NEG_INFINITY;
        for row in [[f32::NAN,0.0],[f32::INFINITY,0.0],
            [negative_infinity,negative_infinity],[-1000.0,-1000.0]] {
            assert!(checked_entropy_rows(&row,2,&[true]).is_err());
        }
        assert_eq!(checked_entropy_rows(&[negative_infinity,0.0],2,&[true]).unwrap(),vec![0.0]);
        let entropy=checked_entropy_rows(&[negative_infinity,0.5f32.ln(),0.5f32.ln()],3,&[true]).unwrap();
        assert!((entropy[0]-std::f64::consts::LN_2).abs()<1e-7);
        assert_eq!(checked_entropy_rows(&[f32::NAN,f32::INFINITY],2,&[false]).unwrap(),vec![0.0]);
    }
    #[test]
    fn decoder_fix_validation_scores_marked_decoder_not_unmarked_teacher() {
        use crate::config::{Arm,CocommitConfig,SbdConfig,SbdFeedback};
        let device=Device::ndarray();
        device.seed(179);
        let c=Config {task:"lm".into(),d_model:16,layers:1,heads:4,key_dim:32,value_dim:64,
            sbd:Some(SbdConfig {block_lengths:vec![4],feedback:SbdFeedback::SoftFeedback,
                cocommit:Some(CocommitConfig::default()),..SbdConfig::default()}),..Config::default()};
        let arm=Arm {name:"validation".into(),mixer:"kronlstm".into(),cache:"surprise".into()};
        let model=LanguageModel::new(&c,&arm,&device).unwrap();
        let rows=[Example {input:vec![256,11,19,23],target:vec![11,19,23,29],weight:vec![1.0;4],groups:vec![None;4]}];
        let before=sbd::evaluate(&model,&c,&rows,&device).unwrap();
        let mut changed=model.clone();
        changed.cocommit_marker=Some(burn::module::Param::from_tensor(Tensor::from_data(
            TensorData::new((0..16).map(|i|((i*7%13) as f32-6.0)*2.0).collect::<Vec<_>>(),[1,16]),&device)));
        let after=sbd::evaluate(&changed,&c,&rows,&device).unwrap();
        assert!((before["block_nll"].as_f64().unwrap()-after["block_nll"].as_f64().unwrap()).abs()>1e-6,
            "primary validation ignores the marked branch used by generation");
        assert_eq!(before["teacher_prefix_diagnostic"]["block_nll"],after["teacher_prefix_diagnostic"]["block_nll"]);
        assert_eq!(after["predicted_tokens"],4);
    }
}
