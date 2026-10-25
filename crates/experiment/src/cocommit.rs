//! Shared-input marked-tail training and temporary block assignment.
use crate::{
    config::{Config, SbdBlockExecution, SbdCommitment, SbdFeedback},
    data::{Batch, Example, Rng},
    model::{self, DenoisingOutput, LanguageModel},
    sbd::{self, SbdTrainingResult, TrainingBlock},
    sbd_training::{CleanInputs, NoisyPrefix, TrainingCache},
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
        if block.masked.len() == 1 { return marks; }
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
fn add_branch_losses(
    model: &LanguageModel, c: &Config, a: DenoisingOutput, b: DenoisingOutput,
    soft: &SoftState, targets: Tensor<2,Int>, marks: &[bool], target_weights: &[f32],
    denominator: f64, result: &mut SbdTrainingResult,
) -> Result<()> {
    let [rows,width,_] = a.logits.dims();
    if model.codpula.is_some() {
        let (circuit,observed) = sbd::prepare_distribution(model,a,soft,1.0)?;
        result.add_circuit_loss(circuit.conditional_nll(targets.clone(),observed,denominator)?);
        let (circuit,observed) = sbd::prepare_distribution(model,b,soft,1.0)?;
        let target_mask = marker_tensor(marks,rows,width,&targets.device()).greater_elem(0.5);
        result.add_circuit_loss(circuit.event_nll(targets,observed,target_mask,denominator)?);
    } else {
        let weight = c.sbd.as_ref().unwrap().cocommit.as_ref().unwrap().visible_loss_weight as f32;
        let a_weights = soft.active.iter().zip(&soft.observed).zip(target_weights)
            .map(|((&active,observed),&w)| if !active {0.0} else if observed.is_some() {weight*w} else {w}).collect::<Vec<_>>();
        let b_weights = soft.active.iter().zip(&soft.observed).zip(marks).zip(target_weights)
            .map(|(((&active,observed),&marked),&w)| if !active {0.0} else if observed.is_some() {weight*w} else if marked {w} else {0.0}).collect::<Vec<_>>();
        result.loss = result.loss.clone() + token_loss(a.logits,targets.clone(),a_weights,denominator)
            + token_loss(b.logits,targets,b_weights,denominator);
    }
    Ok(())
}

pub(super) fn training_loss_profiled<P: Sections>(model: &LanguageModel, c: &Config, rows: &[Example], batch: &Batch, seed: u64, profile: &mut P) -> Result<SbdTrainingResult> {
    let s = c.sbd.as_ref().context("CoCommit requires SBD")?;
    let options = s.cocommit.as_ref().context("CoCommit config is required")?;
    if model.codpula.is_some() {
        ensure!(rows.iter().all(|row| row.weight.iter().all(|&w| w == 1.0)),"Codpula joint event loss requires unit valid target weights");
    }
    let mut trace = CleanInputs::default();
    let clean_logits = model.forward_profiled(batch,&mut trace);
    let [batch_size,sequence_length,_] = clean_logits.dims();
    let denominator = (batch_size*sequence_length) as f64;
    let device = clean_logits.device();
    let mut loss = model.loss(clean_logits.clone(),batch)
        + model.sbd_mask.as_ref().unwrap().val().sum()*0.0
        + model.cocommit_marker.as_ref().unwrap().val().sum()*0.0;
    if let Some(circuit) = &model.codpula {
        loss = loss + circuit.gate_head.weight.val().sum()*0.0;
        if let Some(bias) = &circuit.gate_head.bias { loss = loss + bias.val().sum()*0.0; }
    }
    let mut result = SbdTrainingResult { loss, circuit_flows: None };
    let blocks = sbd::sample_training_blocks(c,rows,seed)?;
    let bundles = sample_bundles(&blocks,options.training_bundle_max,seed);
    for block in blocks.iter().filter(|block|block.masked.len()==1) {
        let logits = clean_logits.clone().slice([block.row..block.row+1,block.start..block.start+1,0..c.vocab()]);
        result.loss = result.loss + sbd::masked_training_loss(logits,block,&rows[block.row],rows.len(),c.vocab(),&device);
    }
    if s.block_execution == SbdBlockExecution::PerBlock {
        for (block,marks) in blocks.iter().zip(&bundles).filter(|(block,_)|block.masked.len()>1) {
            let row = &rows[block.row];
            let n = block.masked.len();
            let observed = (0..n).map(|j|(!block.masked[j]).then_some(row.target[block.start+j] as usize)).collect();
            let mut soft = sbd::state(c,observed,&device)?;
            let cache = model.training_cache(&trace,block.row,&[block.start+1]);
            if s.feedback == SbdFeedback::SoftFeedback {
                let teacher = model.cached_block_profiled(&cache,0,&soft,false,profile);
                sbd::update_feedback(model,c,teacher,&mut soft,1.0)?;
            }
            let prefix = model.cached_block_prefix_profiled(&cache,0,&soft,s.feedback==SbdFeedback::HardMask,profile);
            let a = model.cached_block_tail_profiled(&cache,0,prefix.clone(),None,profile);
            let b = model.cached_block_tail_profiled(&cache,0,prefix,Some(marker_tensor(marks,1,n,&device)),profile);
            let targets = Tensor::from_data(TensorData::new(row.target[block.start..block.start+n].to_vec(),[1,n]),&device);
            add_branch_losses(model,c,a,b,&soft,targets,marks,&row.weight[block.start..block.start+n],denominator,&mut result)?;
        }
        return Ok(result);
    }
    let count = blocks.iter().filter(|block|block.masked.len()>1).count();
    if count == 0 { return Ok(result); }
    let width = *s.block_lengths.iter().max().unwrap();
    let mut cache_blocks = Vec::with_capacity(count);
    let mut observed = Vec::with_capacity(count*width);
    let mut active = Vec::with_capacity(count*width);
    let mut targets = Vec::with_capacity(count*width);
    let mut target_weights = Vec::with_capacity(count*width);
    let mut marks = Vec::with_capacity(count*width);
    for (block,bundle) in blocks.iter().zip(&bundles).filter(|(block,_)|block.masked.len()>1) {
        cache_blocks.push((block.row,block.start+1));
        let row = &rows[block.row];
        for j in 0..width {
            let real = j < block.masked.len();
            observed.push(if real && !block.masked[j] {Some(row.target[block.start+j] as usize)} else {None});
            active.push(real);
            targets.push(if real {row.target[block.start+j]} else {0});
            target_weights.push(if real {row.weight[block.start+j]} else {0.0});
            marks.push(real && bundle[j]);
        }
    }
    let cache = model.training_cache_batched(&trace,&cache_blocks);
    let mut soft = SoftState::new(c.vocab(),s.candidate_width,observed,active,&device,c.soft_selection_backend)?;
    if s.feedback == SbdFeedback::SoftFeedback {
        let teacher = model.cached_blocks_profiled(&cache,&soft,false,profile);
        sbd::update_feedback(model,c,teacher,&mut soft,1.0)?;
    }
    let prefix = model.cached_blocks_prefix_profiled(&cache,&soft,s.feedback==SbdFeedback::HardMask,profile);
    let (a,b) = model.cached_blocks_pair_profiled(&cache,prefix,marker_tensor(&marks,count,width,&device),profile);
    let targets = Tensor::from_data(TensorData::new(targets,[count,width]),&device);
    add_branch_losses(model,c,a,b,&soft,targets,&marks,&target_weights,denominator,&mut result)?;
    Ok(result)
}

pub(super) fn prepare_clean_cache(model: &LanguageModel, prefix: &[i32], device: &Device) -> TrainingCache {
    assert!(!prefix.is_empty());
    let mut trace = CleanInputs::default();
    let _ = model.forward_profiled(&sbd::prefix_batch(prefix,device),&mut trace);
    model.training_cache(&trace,0,&[prefix.len()])
}
pub(super) fn prepare_reconstruction(model: &LanguageModel, prefix: &[i32], soft: &SoftState, hard: bool, device: &Device) -> (TrainingCache,NoisyPrefix) {
    assert!(!prefix.is_empty() && soft.observed.len()>1);
    let cache = prepare_clean_cache(model,prefix,device);
    let hidden = model.cached_block_prefix_profiled(&cache,0,soft,hard,&mut Unprofiled);
    (cache,hidden)
}
pub(super) fn reconstruct_output(model: &LanguageModel, prefix: &[i32], soft: &SoftState, hard: bool, marks: Option<Tensor<2>>, device: &Device) -> DenoisingOutput {
    if soft.observed.len()==1 { return sbd::block_forward(model,prefix,soft,hard,device); }
    let (cache,hidden) = prepare_reconstruction(model,prefix,soft,hard,device);
    model.cached_block_tail_profiled(&cache,0,hidden,marks,&mut Unprofiled)
}
struct Prediction {
    logs: Tensor<3>,
    proposals: Vec<usize>,
    scores: Option<Vec<f64>>,
    feedback: Tensor<2>,
    feedback_temperature: f64,
    circuit: Option<crate::codpula::PreparedCircuit>,
}
fn prediction(model: &LanguageModel, c: &Config, output: DenoisingOutput, soft: &SoftState, seed: u64, temperature: f64, assignments: bool) -> Result<Prediction> {
    let n = soft.observed.len();
    if model.codpula.is_some() && n>1 {
        let (circuit,tokens,observed) = sbd::prepare_circuit(model,output,soft,temperature)?;
        let logs = circuit.marginals(tokens.clone(),observed.clone())?.log();
        let proposals = if assignments {circuit.sample(tokens.clone(),observed.clone(),seed)?} else {Vec::new()};
        let scores = if assignments && c.sbd.as_ref().unwrap().commitment==SbdCommitment::Dependence {
            Some(circuit.dependence_scores(tokens,observed,&proposals)?.remove(0))
        } else {None};
        Ok(Prediction { feedback: logs.clone().reshape([n,c.vocab()]), logs,
            proposals:if assignments {proposals.into_iter().next().context("missing single-block joint proposal")?} else {Vec::new()},
            scores, feedback_temperature:1.0,circuit:Some(circuit) })
    } else {
        let logs = log_softmax(output.logits.clone(),2);
        let proposals = if assignments {
            let values = logs.clone().into_data().try_to_vec::<f32>().map_err(|e|anyhow::anyhow!("CoCommit probabilities {e:?}"))?;
            values.chunks_exact(c.vocab()).map(|row| (0..row.len()).max_by(|&a,&b|row[a].total_cmp(&row[b]).then(b.cmp(&a))).unwrap()).collect()
        } else {Vec::new()};
        Ok(Prediction { logs, proposals, scores:None, feedback:output.logits.reshape([n,c.vocab()]),feedback_temperature:c.soft_temperature,circuit:None })
    }
}
fn select(c: &Config, prediction: &Prediction, soft: &SoftState, last_pass: bool) -> Result<Vec<usize>> {
    let s = c.sbd.as_ref().unwrap();
    let unresolved = soft.observed.iter().zip(&soft.active).enumerate().filter_map(|(i,(value,&active))|(active && value.is_none()).then_some(i)).collect::<Vec<_>>();
    if unresolved.is_empty() { return Ok(Vec::new()); }
    if let Some(scores) = &prediction.scores {
        ensure!(unresolved.iter().all(|&i|scores[i].is_finite()),"CoCommit dependence requires finite supported scores");
        return Ok(sbd::dependence_commits(&unresolved,scores,s.dependence_threshold,last_pass));
    }
    let logs = prediction.logs.clone().into_data().try_to_vec::<f32>().map_err(|e|anyhow::anyhow!("CoCommit entropy {e:?}"))?;
    let mut order = unresolved.into_iter().map(|i| {
        let row = &logs[i*c.vocab()..(i+1)*c.vocab()];
        let entropy = row.iter().filter(|&&x|x.is_finite()).map(|&x|-(x as f64).exp()*x as f64).sum::<f64>();
        (i,entropy)
    }).collect::<Vec<_>>();
    order.sort_by(|a,b|a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let take = if let Some(gamma) = s.entropy_threshold {
        let mut take = 1;
        let mut sum = 0.0;
        while take < order.len() && sum+order[take-1].1 <= gamma { sum+=order[take-1].1; take+=1; }
        take
    } else {s.commit_count.min(order.len())};
    Ok(order[..take].iter().map(|&(i,_)|i).collect())
}
pub(super) struct Assignment {
    pub proposals: Vec<usize>,
    pub commits: Vec<usize>,
    pub feedback: Tensor<2>,
    pub feedback_temperature: f64,
}
pub(super) fn assign(model: &LanguageModel, c: &Config, cache: &TrainingCache, soft: &SoftState, pass: usize, proposal_seed: u64, device: &Device) -> Result<Assignment> {
    let s = c.sbd.as_ref().unwrap();
    let n = soft.observed.len();
    let hidden = model.cached_block_prefix_profiled(cache,0,soft,s.feedback==SbdFeedback::HardMask,&mut Unprofiled);
    let output = model.cached_block_tail_profiled(cache,0,hidden.clone(),None,&mut Unprofiled);
    let a = prediction(model,c,output,soft,proposal_seed,c.soft_temperature,true)?;
    let announcement = select(c,&a,soft,pass==n)?;
    if announcement.is_empty() { return Ok(Assignment {proposals:a.proposals, commits:announcement,feedback:a.feedback,feedback_temperature:a.feedback_temperature}); }
    drop(a);
    let mut marks = vec![false;n];
    for &i in &announcement {marks[i]=true;}
    let output = model.cached_block_tail_profiled(cache,0,hidden,Some(marker_tensor(&marks,1,n,device)),&mut Unprofiled);
    let b = prediction(model,c,output,soft,proposal_seed,c.soft_temperature,true)?;
    let commits = if let Some(scores) = &b.scores {
        ensure!(announcement.iter().all(|&i|scores[i].is_finite()),"marked CoCommit dependence requires finite scores");
        sbd::dependence_commits(&announcement,scores,s.dependence_threshold,pass==n)
    } else {announcement};
    Ok(Assignment {proposals:b.proposals,commits,feedback:b.feedback,feedback_temperature:b.feedback_temperature})
}

pub(super) fn evaluate(model: &LanguageModel, c: &Config, examples: &[Example], device: &Device) -> Result<Value> {
    let s = c.sbd.as_ref().unwrap();
    let model = model.valid();
    let mut nll = 0.0;
    let mut marked_nll = 0.0;
    let mut count = 0usize;
    let mut marked_count = 0usize;
    let mut hits = 0usize;
    let mut marked_hits = 0usize;
    let mut rng = Rng(c.seed);
    for row in examples {
        if model.codpula.is_some() {ensure!(row.weight.iter().all(|&w|w==1.0),"Codpula joint event evaluation requires unit valid target weights");}
        for &length in &s.block_lengths {
            for start in (0..row.target.len()).step_by(length) {
                let n = length.min(row.target.len()-start);
                let mut soft = sbd::state(c,vec![None;n],device)?;
                let prefix = &row.input[..start+1];
                if n==1 {
                    let output = sbd::block_forward(&model,prefix,&soft,s.feedback==SbdFeedback::HardMask,device);
                    let logs = log_softmax(output.logits,2).into_data().try_to_vec::<f32>().map_err(|e|anyhow::anyhow!("CoCommit singleton {e:?}"))?;
                    if row.weight[start] != 0.0 {
                        nll -= logs[row.target[start] as usize] as f64;
                        let best = (0..c.vocab()).max_by(|&a,&b|logs[a].total_cmp(&logs[b]).then(b.cmp(&a))).unwrap();
                        hits += usize::from(best==row.target[start] as usize);
                        count += 1;
                    }
                    continue;
                }
                let cache=prepare_clean_cache(&model,prefix,device);
                if s.feedback==SbdFeedback::SoftFeedback {
                    let teacher=model.cached_block_profiled(&cache,0,&soft,false,&mut Unprofiled);
                    sbd::update_feedback(&model,c,teacher,&mut soft,1.0)?;
                }
                let hidden=model.cached_block_prefix_profiled(&cache,0,&soft,s.feedback==SbdFeedback::HardMask,&mut Unprofiled);
                let output_a = model.cached_block_tail_profiled(&cache,0,hidden.clone(),None,&mut Unprofiled);
                let seed = if model.codpula.is_some() {rng.next()} else {0};
                let a = prediction(&model,c,output_a,&soft,seed,1.0,true)?;
                let announcement = select(c,&a,&soft,false)?;
                let mut marks = vec![false;n];
                for &i in &announcement {marks[i]=true;}
                let output_b = model.cached_block_tail_profiled(&cache,0,hidden,Some(marker_tensor(&marks,1,n,device)),&mut Unprofiled);
                let b = prediction(&model,c,output_b,&soft,seed,1.0,false)?;
                let clean = Tensor::<2,Int>::from_data(TensorData::new(row.target[start..start+n].to_vec(),[1,n]),device);
                if let (Some(circuit_a),Some(circuit_b)) = (&a.circuit,&b.circuit) {
                    let observed = Tensor::<2,Bool>::from_data(TensorData::new(vec![false;n],[1,n]),device);
                    let all = observed.clone().bool_not();
                    nll += model::scalar(circuit_a.log_partition(clean.clone(),observed.clone())?-circuit_a.log_partition(clean.clone(),all)?)?;
                    marked_nll += model::scalar(circuit_b.log_partition(clean.clone(),observed.clone())?-circuit_b.log_partition(clean,marker_tensor(&marks,1,n,device).greater_elem(0.5))?)?;
                }
                let av = a.logs.into_data().try_to_vec::<f32>().map_err(|e|anyhow::anyhow!("CoCommit A evaluation {e:?}"))?;
                let bv = b.logs.into_data().try_to_vec::<f32>().map_err(|e|anyhow::anyhow!("CoCommit B evaluation {e:?}"))?;
                for j in 0..n {
                    if row.weight[start+j]==0.0 {continue;}
                    let gold = row.target[start+j] as usize;
                    let best = |values:&[f32]| (0..c.vocab()).max_by(|&a,&b|values[a].total_cmp(&values[b]).then(b.cmp(&a))).unwrap();
                    let ar = &av[j*c.vocab()..(j+1)*c.vocab()];
                    let br = &bv[j*c.vocab()..(j+1)*c.vocab()];
                    if model.codpula.is_none() {nll-=ar[gold] as f64;}
                    hits+=usize::from(best(ar)==gold);count+=1;
                    if marks[j] {
                        if model.codpula.is_none() {marked_nll-=br[gold] as f64;}
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
        "conditioning":"all targets initially masked; unmarked detached teacher; A full-block NLL and B announced-subset NLL reported separately; L1 causal bypass",
        "nll_reduction":"A sum full-block event NLL / all valid targets; B sum announced event NLL / announced valid targets, not full-block NLL"}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundles_are_masked_uniform_size_draws_with_independent_seed() {
        let blocks = vec![
            TrainingBlock { row:0,start:0,length:1,masked:vec![true] },
            TrainingBlock { row:0,start:1,length:4,masked:vec![true,false,true,true] },
            TrainingBlock { row:1,start:0,length:4,masked:vec![false,true,false,true] },
        ];
        let bundles = sample_bundles(&blocks,16,77);
        assert_eq!(bundles[0],vec![false]);
        assert_eq!(bundles[1..],sample_bundles(&blocks[1..],16,77));
        assert_eq!(bundles,sample_bundles(&blocks,16,77));
        for (block,bundle) in blocks.iter().zip(&bundles).skip(1) {
            let count=bundle.iter().filter(|&&x|x).count();
            assert!(count>0 && count<=block.masked.iter().filter(|&&x|x).count());
            assert!(bundle.iter().zip(&block.masked).all(|(&chosen,&masked)|!chosen || masked));
        }
        for bundle in sample_bundles(&blocks,1,77).iter().skip(1) {
            assert_eq!(bundle.iter().filter(|&&x|x).count(),1);
        }
    }
}
