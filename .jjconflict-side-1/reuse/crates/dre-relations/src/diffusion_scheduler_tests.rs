//! CPU acceptance runs actual query assembly, both source reads, graph oracle, normalization and heads.
use super::*;
use crate::denoiser::AnnotationDenoiserConfig;
use crate::pointer::{POINTER_FIELDS, PointerEndpoint, SourcePointerDocument};
use crate::training::DenoiserTrainingConfig;
use crate::training::query_chain::QueryForwardInputs;
use dre_core::{BatchLayout, DocumentLayout};
use crate::remask::REMASK_FEATURES;
use crate::remask::REFINEMENT_ROUNDS;

struct Fixture {
    model: AnnotationDenoiser,
    plan: SlotPlan,
    slots: SlotTensors,
    pairs: PairFeatures,
    cache: DocumentCache,
    context: SourcePointerContext,
    observed: Vec<[Option<PointerEndpoint>; 4]>,
    schedule: DenoiseSchedule,
}
fn fixture(device: &Device) -> Fixture {
    let layout = DocumentLayout::packed(vec![vec![2, 2], vec![1]], 4);
    let plan = SlotPlan::queries(&layout, &[vec![1, 1], vec![0]]).unwrap();
    let slots = plan.tensors(device);
    let batch = BatchLayout::new(layout, device);
    let model = AnnotationDenoiserConfig::new().with_hidden(2).with_heads(1).with_head_dim(2)
        .with_graph_hidden(4).with_labels(3).with_slot_chunk(1).init(device);
    let source = Tensor::from_data(TensorData::new((0..16).map(|index| (index as f32 * 0.37).sin()).collect::<Vec<_>>(), [2, 4, 2]), device);
    let context = SourcePointerContext::new(&plan, source.clone(), &[
        SourcePointerDocument {row: 0, document: 0, first_token: 0, text: "ab", token_offsets: &[(0, 1), (1, 2)]},
        SourcePointerDocument {row: 0, document: 1, first_token: 2, text: "éx", token_offsets: &[(0, 2), (2, 3)]},
        SourcePointerDocument {row: 1, document: 0, first_token: 0, text: "Z", token_offsets: &[(0, 1)]},
    ], model.pointer_head.none.val()).unwrap();
    let cache = model.prepare(source.clone(), source, &batch);
    let pairs = model.source_pairs(&plan);
    let mut observed = vec![[None; 4]; slots.batch * slots.slots_per_row];
    observed[0][0] = Some(PointerEndpoint {token: 0, byte_offset: 0});
    observed[0][2] = Some(PointerEndpoint {token: 1, byte_offset: 0});
    Fixture {model, plan, slots, pairs, cache, context, observed,
        schedule: DenoiseSchedule {steps: REFINEMENT_ROUNDS, eta: 0.5, temperature: 1.0, slot_chunk: 1}}
}
fn head(device:&Device)->RefinementDecider {
    let mut head=RefinementDecider::new(2,REFINEMENT_ROUNDS,device);
    head.acceptance_output.weight=burn::module::Param::from_tensor(Tensor::zeros(head.acceptance_output.weight.val().dims(),device));
    head.acceptance_output.bias=Some(burn::module::Param::from_tensor(Tensor::from_data(
        TensorData::new(vec![100.0f32;4].into_iter().chain([-100.0;4]).collect::<Vec<_>>(),[8]),device)));
    head.outcome_output.bias=Some(burn::module::Param::from_tensor(Tensor::from_data(
        TensorData::new(vec![0.0f32,100.0,0.0,100.0,100.0,0.0,0.0,100.0,100.0,0.0,0.0,100.0],[12]),device)));
    head
}
fn host<const D: usize>(tensor: &Tensor<D>) -> Vec<f32> {tensor.to_data().try_to_vec_as::<f32>().unwrap()}
fn bits<const D: usize>(tensor: &Tensor<D>) -> Vec<u32> {host(tensor).into_iter().map(f32::to_bits).collect()}
fn same_output(left: &PointerStepOutput, right: &PointerStepOutput) {
    assert_eq!(bits(&left.hidden), bits(&right.hidden));
    assert_eq!(bits(&left.annotations.relation), bits(&right.annotations.relation));
    assert_eq!(bits(&left.annotations.evidence_start), bits(&right.annotations.evidence_start));
    assert_eq!(bits(&left.annotations.evidence_end), bits(&right.annotations.evidence_end));
    for field in 0..POINTER_FIELDS {
        assert_eq!(bits(&left.pointers.token[field]), bits(&right.pointers.token[field]));
        assert_eq!(bits(&left.pointers.byte[field]), bits(&right.pointers.byte[field]));
        assert_eq!(bits(&left.pointers.byte_queries[field]), bits(&right.pointers.byte_queries[field]));
    }
}
fn same_state(left: &RefinementRound, right: &RefinementRound) {
    same_output(&left.output, &right.output);
    for (left, right) in [(&left.annotations.relation, &right.annotations.relation),
        (&left.annotations.evidence_start, &right.annotations.evidence_start),
        (&left.annotations.evidence_end, &right.annotations.evidence_end),
        (&left.annotations.visibility, &right.annotations.visibility), (&left.pointers.visibility, &right.pointers.visibility)] {
        assert_eq!(bits(left), bits(right));
    }
    assert_eq!(left.annotations.step.to_data(), right.annotations.step.to_data());
    assert_eq!(left.annotations.subject.to_data(), right.annotations.subject.to_data());
    assert_eq!(left.annotations.object.to_data(), right.annotations.object.to_data());
    assert_eq!(left.pointers.observed_offsets.to_data(), right.pointers.observed_offsets.to_data());
    assert_eq!(left.pointers.observed.to_data(), right.pointers.observed.to_data());
    for field in 0..POINTER_FIELDS {assert_eq!(bits(&left.pointers.token[field]), bits(&right.pointers.token[field]));}
    assert_eq!(bits(&left.acceptance_features),bits(&right.acceptance_features));
    assert_eq!(bits(&left.observed),bits(&right.observed));
    assert_eq!(bits(&left.acceptance_logits),bits(&right.acceptance_logits));
    assert_eq!(left.outcome_logits.as_ref().map(bits),right.outcome_logits.as_ref().map(bits));
    assert_eq!(bits(&left.remasked),bits(&right.remasked));
    assert_eq!(left.retained.filled.to_data(),right.retained.filled.to_data());
    assert_eq!(left.retained.value.to_data(),right.retained.value.to_data());
    assert_eq!(bits(&left.retained.factor_stats),bits(&right.retained.factor_stats));
    assert_eq!(left.argmaxes.to_data(), right.argmaxes.to_data());
    assert_eq!(bits(&left.conditional_relation),bits(&right.conditional_relation));
    for field in 0..4 {assert_eq!(bits(&left.conditional_tokens[field]),bits(&right.conditional_tokens[field]));}
    assert_eq!(left.supplied.values.to_data(),right.supplied.values.to_data());
    assert_eq!(left.supplied.known.clone().int().to_data(),right.supplied.known.clone().int().to_data());
}

#[test]
fn exp004_retained_refinement_eight_live_calls_share_training_inference_cpu() {
    let device=Device::ndarray();let fixture=fixture(&device);let decider=head(&device);
    let initial=PointerState::initial(&fixture.context,&fixture.observed).unwrap();
    let source_before=bits(&fixture.context.values);let supplied_before=initial.observed_offsets.to_data();
    let mut calls=Vec::new();
    let inference=fixture.model.run_pointer_with_remask_hooks(&fixture.plan,&fixture.pairs,&fixture.cache,&fixture.slots,
        &fixture.context,Some(&initial),&fixture.schedule,None,&decider,0,|_,output,_|output,|round|calls.push(round));
    let training=fixture.model.joint_query_forward(QueryForwardInputs {plan:&fixture.plan,slots:&fixture.slots,pairs:&fixture.pairs,
        cache:&fixture.cache,context:&fixture.context,observed:&fixture.observed,exploration_flips:None},
        &fixture.schedule,&DenoiserTrainingConfig::new(),&decider,0).unwrap();
    assert_eq!(calls,(1..=8).collect::<Vec<_>>());assert_eq!(inference.output.steps,8);
    assert_eq!(inference.slot_steps,vec![8,8,0,0]);assert_eq!(inference.average_steps,8.0);
    assert_eq!(training.rounds.len(),8);
    assert_eq!(inference.rounds.iter().filter(|round|round.outcome_logits.is_some()).count(),7);
    for (index,(left,right)) in inference.rounds.iter().zip(&training.rounds).enumerate() {
        same_state(left,right);
        assert_eq!(host(&left.active),vec![1.0,1.0,0.0,0.0]);
        assert_eq!(left.annotations.step.to_data().try_to_vec_as::<i32>().unwrap(),vec![(index+1) as i32,(index+1) as i32,0,0]);
        assert_eq!(left.acceptance_logits.dims(),[2,2,8]);assert_eq!(left.acceptance_features.dims(),[2,2,REMASK_FEATURES]);
        assert!(host(&left.acceptance_features).iter().all(|value|value.is_finite()));
        assert_eq!(left.pointers.observed_offsets.to_data(),supplied_before);
        assert_eq!(left.pointers.observed.to_data(),initial.observed.to_data());
        assert_eq!(left.retained.factor_stats.dims(),[2,2,72]);
        assert_eq!(left.retained.value.dims(),[2,2,11]);
        assert!(host(&left.accepted)[8..].iter().all(|&value|value==0.0));
        if index<7 {
            assert_eq!(left.outcome_logits.as_ref().unwrap().dims(),[2,2,12]);
            let actions=host(&left.remasked);
            assert_eq!(&actions[..6],&[1.0,0.0,0.0,0.0,0.0,1.0]);
            assert_eq!(&actions[6..12],&[1.0,1.0,0.0,1.0,0.0,1.0]);
            assert!(actions[12..].iter().all(|&value|value==0.0));
        } else {
            assert!(left.outcome_features.is_none());assert_eq!(host(&left.remasked),vec![0.0;24]);
        }
    }
    assert_ne!(bits(&inference.rounds[0].output.hidden),bits(&inference.rounds[7].output.hidden));
    assert_eq!(source_before,bits(&fixture.context.values));
    // Retaining does not fabricate a one-hot relation distribution.
    let decoded=inference.rounds[0].circuit.greedy_decode(&inference.rounds[0].supplied);
    let first=RemaskFrame::from_output(&inference.rounds[0].output,&initial,0,fixture.context.none,&fixture.slots.slot_valid,&decoded);
    assert_eq!(bits(&inference.retained.relation_posterior),bits(&first.factors[0]));
    assert_eq!(inference.retained.value.to_data(),training.retained.value.to_data());
    assert_eq!(host(&inference.rounds[0].acceptance_features)[96],0.0);
    assert_eq!(host(inference.rounds[0].outcome_features.as_ref().unwrap())[96],1.0);
}

#[test]
fn exp004_remasking_hides_retained_values_but_not_supplied_boundaries_cpu() {
    let device=Device::ndarray();let fixture=fixture(&device);
    let annotations=SoftState::initial(&fixture.plan,&fixture.slots,3,None,&device);
    let pointers=PointerState::initial(&fixture.context,&fixture.observed).unwrap();
    let mut retained=RetainedAnswer::empty(2,2,3,&device);
    retained.filled=Tensor::from_data(TensorData::new(vec![true;16],[2,2,4]),&device);
    retained.value=Tensor::from_data(TensorData::new(vec![
        1i32,1,0,1,1,0,0,0,1,0,1, 2,2,0,2,2,3,0,3,1,2,3,
        0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],[2,2,11]),&device);
    let hidden=fixture.model.queries_pointer_retained(&fixture.pairs,&fixture.cache,&fixture.slots,&annotations,&pointers,
        &fixture.context,&retained,&Tensor::ones([2,2,6],&device));
    let ordinary=fixture.model.queries_pointer(&fixture.pairs,&fixture.cache,&fixture.slots,&annotations,&pointers,&fixture.context);
    assert_eq!(bits(&hidden),bits(&ordinary),"a remasked retained field contributes no input");
    let visible=fixture.model.queries_pointer_retained(&fixture.pairs,&fixture.cache,&fixture.slots,&annotations,&pointers,
        &fixture.context,&retained,&Tensor::zeros([2,2,6],&device));
    assert_ne!(bits(&visible),bits(&ordinary),"retained context is actually gathered");
    let supplied_before=pointers.observed_offsets.to_data();
    let (subject,_)=fixture.model.pointer_head.fields_retained(&pointers,&fixture.context,&retained,
        &Tensor::ones([2,2,6],&device),&fixture.model.retained_markers.val());
    let (expected,_)=fixture.model.pointer_head.fields(&pointers,&fixture.context);
    assert_eq!(bits(&subject),bits(&expected));
    assert_eq!(pointers.observed_offsets.to_data(),supplied_before);
}

#[test]
fn exp004_exploration_changes_only_eligible_next_call_inputs_cpu() {
    let device=Device::ndarray();let fixture=fixture(&device);let decider=head(&device);
    let flips=Tensor::ones([7,2,2,6],&device);
    let refinement=fixture.model.joint_query_forward(QueryForwardInputs {plan:&fixture.plan,slots:&fixture.slots,pairs:&fixture.pairs,
        cache:&fixture.cache,context:&fixture.context,observed:&fixture.observed,exploration_flips:Some(&flips)},
        &fixture.schedule,&DenoiserTrainingConfig::new(),&decider,0).unwrap();
    for round in &refinement.rounds[..7] {
        let policy=host(&round.policy_remasked);let taken=host(&round.remasked);let flipped=host(&round.exploration_flipped);
        let observed=host(&round.observed);let valid=host(&round.active);
        for index in 0..policy.len() {
            if valid[index/6]>0.5&&observed[index]<0.5 {
                assert_eq!(flipped[index],1.0);assert_eq!(taken[index],1.0-policy[index]);
            } else {assert_eq!(flipped[index],0.0);assert_eq!(taken[index],0.0);}
        }
    }
    assert_eq!(host(&refinement.rounds[7].exploration_flipped),vec![0.0;24]);
}

#[test]
fn zero_mass_circuit_feedback_keeps_previous_distributions_and_finite_gradients_cpu() {
    let device=Device::ndarray().autodiff();
    for eta in [0.37,1.0] {
        let fixture=fixture(&device);
        let mut annotations=SoftState::initial(&fixture.plan,&fixture.slots,3,None,&device);
        annotations.relation=annotations.relation.require_grad();
        let mut pointers=PointerState::masked(&fixture.context);
        pointers.token=pointers.token.map(Tensor::require_grad);
        let hidden=Tensor::zeros([2,2,2],&device).require_grad();
        let relation=Tensor::from_data(TensorData::new(vec![0.2f32,-0.3,0.7].repeat(4),[2,2,3]),&device).require_grad();
        let mut logits=fixture.model.pointer_head.logits(hidden.clone(),&fixture.context,0..2);
        let mut impossible=vec![false;2*2*5];impossible[1]=true;
        logits.token[0]=logits.token[0].clone().mask_fill(
            Tensor::<3,Bool>::from_data(TensorData::new(impossible,[2,2,5]),&device),f32::NEG_INFINITY);
        let tables=fixture.model.joint_circuit.endpoint_tables(&fixture.context);
        let circuit=fixture.model.joint_circuit.call(hidden.clone(),relation.clone(),&logits.token,&tables,&fixture.slots.slot_valid);
        let mut known=vec![false;2*2*5];known[1]=true;
        let mut values=vec![0;known.len()];values[1]=1;
        let supplied=CircuitEvidence::from_host(values,known,2,2,&device);
        let (conditional_relation,conditional_tokens)=circuit.conditional_log_marginals(&supplied,true);
        assert!(host(&conditional_relation)[..3].iter().all(|&value|value==f32::NEG_INFINITY));
        let output=PointerStepOutput {annotations:StepOutput {relation:conditional_relation.clone(),
            evidence_start:Tensor::zeros([2,2,4],&device),evidence_end:Tensor::zeros([2,2,4],&device)},
            pointers:PointerLogits {token:conditional_tokens.clone(),..logits},hidden:hidden.clone()};
        let schedule=DenoiseSchedule {eta,..fixture.schedule};
        let (next,next_pointers)=advance_with_remask(&annotations,&pointers,&output,&schedule,
            &Tensor::zeros([2,2,6],&device),&fixture.slots.slot_valid);
        assert!(host(&next.relation).iter().all(|value|value.is_finite()));
        assert_eq!(bits(&next.relation.clone().slice([0..1,0..1])),bits(&annotations.relation.clone().slice([0..1,0..1])));
        let reference=if eta==1.0 {softmax(conditional_relation.clone().slice([0..1,1..2]),2)}else{
            annotations.relation.clone().slice([0..1,1..2])*(1.0-eta)
                +softmax(conditional_relation.clone().slice([0..1,1..2]),2)*eta};
        assert_eq!(bits(&next.relation.clone().slice([0..1,1..2])),bits(&reference),
            "positive-mass rows must retain the previous arithmetic exactly");
        let mut loss=next.relation.clone().slice([0..1,0..2,0..1]).sum();
        for field in 0..4 {
            assert!(host(&next_pointers.token[field]).iter().all(|value|value.is_finite()));
            if field>0 {assert_eq!(bits(&next_pointers.token[field].clone().slice([0..1,0..1])),
                bits(&pointers.token[field].clone().slice([0..1,0..1])));}
            loss=loss+next_pointers.token[field].clone().slice([0..1,0..2,0..1]).sum();
        }
        let gradients=loss.backward();
        for gradient in [host(&annotations.relation.grad(&gradients).unwrap()),host(&relation.grad(&gradients).unwrap()),
            host(&hidden.grad(&gradients).unwrap()),host(&fixture.model.joint_circuit.relation_table.val().grad(&gradients).unwrap()),
            host(&fixture.model.joint_circuit.initial_logits.val().grad(&gradients).unwrap()),
            host(&fixture.model.joint_circuit.transition_logits.val().grad(&gradients).unwrap())] {
            assert!(gradient.iter().all(|value|value.is_finite()));
        }
        for field in 0..4 {assert!(host(&pointers.token[field].grad(&gradients).unwrap()).iter().all(|value|value.is_finite()));}
    }
}

#[test]
fn exp004_normal_update_eta_endpoints_ignore_unused_nan_operands_and_gradients_cpu() {
    let device=Device::ndarray().autodiff();
    for eta in [0.0,1.0] {
        let old=Tensor::<3>::from_data(TensorData::new(
            if eta==1.0 {vec![f32::NAN;3]}else{vec![0.2f32,0.3,0.5]},[1,1,3]),&device).require_grad();
        let logits=Tensor::<3>::from_data(TensorData::new(
            if eta==0.0 {vec![f32::NAN;3]}else{vec![0.2f32,-0.3,0.7]},[1,1,3]),&device).require_grad();
        let schedule=DenoiseSchedule {steps:REFINEMENT_ROUNDS,eta,temperature:1.0,slot_chunk:1};
        let rho=Tensor::zeros([1,1,1],&device);
        let (next,_)=normal_field(&old,&logits,&rho,&schedule);
        assert!(host(&next).iter().all(|value|value.is_finite()));
        assert!((host(&next).iter().sum::<f32>()-1.0).abs()<1e-6);
        let gradients=next.slice([0..1,0..1,0..1]).sum().backward();
        let used=if eta==0.0 {
            assert!(logits.grad(&gradients).is_none());old.grad(&gradients).unwrap()
        }else{
            if let Some(gradient)=old.grad(&gradients) {assert!(host(&gradient).iter().all(|&value|value==0.0));}
            logits.grad(&gradients).unwrap()
        };
        assert!(host(&used).iter().all(|value|value.is_finite()));
        assert!(host(&used).iter().any(|value|value.abs()>1e-8));
    }
}

fn joint_transition_case(device:&Device) {
    let fixture=fixture(device);
    let mut decider=head(device);
    decider.acceptance_output.bias=Some(burn::module::Param::from_tensor(Tensor::full([8],-100.0,device)));
    let annotations=SoftState::initial(&fixture.plan,&fixture.slots,3,None,device);
    let pointers=PointerState::initial(&fixture.context,&fixture.observed).unwrap();
    let supplied=pointers.circuit_evidence(&fixture.slots.slot_valid);
    let tables=fixture.model.joint_circuit.endpoint_tables(&fixture.context);
    let mut retained=RetainedAnswer::empty(2,2,3,device);
    retained.filled=crate::plan::boolean(vec![true,true,true,false,false,false,false,false,
        true,true,true,true,true,true,true,true],[2,2,4],device);
    retained.value=crate::plan::int(vec![
        2,1,0,0,1,0,0,1,1,0,1,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0],[2,2,11],device);
    let hidden=Tensor::from_data(TensorData::new(vec![0.7f32,-0.4,0.2,0.8,
        f32::NAN,f32::NAN,f32::NAN,f32::NAN],[2,2,2]),device);
    let invalid=fixture.slots.slot_valid.clone().lower_elem(0.5).unsqueeze_dim(2);
    let clean_hidden=hidden.clone().mask_fill(invalid.clone(),0.0);
    let evidence=Tensor::zeros([2,2,4],device)
        .mask_fill(fixture.slots.outside_document(0..2),f32::NEG_INFINITY).mask_fill(invalid.clone(),0.0);
    let output=PointerStepOutput {annotations:StepOutput {
        relation:fixture.model.relation_head.forward(clean_hidden).mask_fill(invalid.clone(),0.0),
        evidence_start:evidence.clone(),evidence_end:evidence},
        pointers:fixture.model.pointer_head.logits(hidden.clone(),&fixture.context,0..2),hidden};
    let schedule=DenoiseSchedule {temperature:0.7,..fixture.schedule};
    let (round,_)=refinement_transition(&annotations,&pointers,output,&mut retained,None,&fixture.context,
        &fixture.model,&tables,&supplied,&schedule,&decider,0,&fixture.slots.slot_valid,0,None,false);
    let decoded=round.circuit.greedy_decode(&supplied);
    let decoded_host=decoded.to_data().try_to_vec_as::<i64>().unwrap();
    let proposals=round.argmaxes.to_data().try_to_vec_as::<i64>().unwrap();
    for query in 0..4 {for (field,factor) in [0,1,3,5,7].into_iter().enumerate() {
        assert_eq!(proposals[query*11+factor],decoded_host[query*5+field]);
    }}
    let dependency=round.circuit.dependency_features(&decoded,&supplied);
    assert_eq!(bits(&round.acceptance_features.clone().slice([0..2,0..2,112..118])),bits(&dependency));
    assert_eq!(bits(&round.outcome_features.as_ref().unwrap().clone().slice([0..2,0..2,112..118])),bits(&dependency));
    assert!(host(&round.acceptance_features)[2*REMASK_FEATURES..].iter().all(|&value|value==0.0));
    for field in 0..4 {
        let expected=fixture.model.pointer_head.byte_logits(round.output.pointers.byte_queries[field].clone()
            .mask_fill(invalid.clone(),0.0),&fixture.context,
            decoded.clone().slice([0..2,0..2,field+1..field+2]).reshape([2,2])).mask_fill(invalid.clone(),0.0);
        assert_eq!(bits(&round.output.pointers.byte[field]),bits(&expected));
    }
    let evidence=visible_retained_evidence(&supplied,&retained,&round.remasked,&fixture.slots.slot_valid,&fixture.context);
    let known=evidence.known.clone().int().to_data().try_to_vec_as::<i32>().unwrap();
    let values=evidence.values.to_data().try_to_vec_as::<i32>().unwrap();
    assert_eq!(&known[..5],&[0,1,1,1,1],"remasked R is hidden; supplied starts and kept retained ends remain visible");
    assert_eq!(&values[1..5],&[0,0,1,1],"supplied starts override conflicting retained starts");
    assert!(known[5..].iter().all(|&flag|flag==0),"empty and padded retained rows add no evidence");
    let (relation,tokens)=round.circuit.conditional_log_marginals(&evidence,true);
    assert_eq!(bits(&round.conditional_relation),bits(&relation));
    for field in 0..4 {assert_eq!(bits(&round.conditional_tokens[field]),bits(&tokens[field]));}
    let expected=(annotations.relation.clone()*0.5+relation.exp()*0.5)
        .mask_where(invalid.clone().expand(annotations.relation.dims()),annotations.relation.clone());
    for (actual,expected) in host(&round.annotations.relation).iter().zip(host(&expected)) {
        assert!((actual-expected).abs()<2e-7,"feedback uses circuit probabilities without applying tau a second time");
    }
    let end_probabilities=host(&tokens[1].clone().exp());
    assert!(end_probabilities[0]>0.0&&end_probabilities[1]>0.0,
        "a retained endpoint must not collapse its own feedback distribution");
    for field in 0..4 {
        let expected=(pointers.token[field].clone()*0.5+tokens[field].clone().exp()*0.5)
            .mask_where(pointers.observed.clone().slice([0..2,0..2,field..field+1]).greater_elem(0.5)
                .expand(pointers.token[field].dims()),pointers.token[field].clone())
            .mask_where(invalid.clone().expand(pointers.token[field].dims()),pointers.token[field].clone());
        for (actual,expected) in host(&round.pointers.token[field]).iter().zip(host(&expected)) {
            assert!((actual-expected).abs()<2e-7,"endpoint feedback uses the exact other-field conditional");
        }
    }
    assert_eq!(round.pointers.observed_offsets.to_data(),pointers.observed_offsets.to_data());
    let view=round.slot_view(0,0..1);
    assert_eq!(view.supplied.values.dims(),[1,1,5]);
    let (view_relation,_)=view.circuit.conditional_log_marginals(&evidence.slot_view(0,0..1),true);
    assert_eq!(bits(&view_relation),bits(&round.conditional_relation.clone().slice([0..1,0..1])));
}

#[test]
fn exp004_joint_transition_uses_circuit_proposals_feedback_bytes_and_dependency_features_cpu() {
    joint_transition_case(&Device::ndarray());
}

#[cfg(feature="cuda")]
#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn exp004_joint_transition_uses_circuit_proposals_feedback_bytes_and_dependency_features_cuda() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(),Ok("1"));
    joint_transition_case(&dre_core::gpu_device());
}

#[test]
fn a5_three_span_seven_pair_calls_use_global_rounds_and_keep_the_bank_fixed_cpu() {
    use crate::{Candidates,DocumentEntities,Entity,TokenSpan,PairEncoderConfig};
    let device=Device::ndarray();device.seed(20261023);
    let layout=DocumentLayout::one_document_per_row(&[2],2);
    let batch=BatchLayout::new(layout.clone(),&device);
    let plan=SlotPlan::queries(&layout,&[vec![2]]).unwrap();let slots=plan.tensors(&device);
    let model=AnnotationDenoiserConfig::new().with_hidden(2).with_heads(1).with_head_dim(2)
        .with_graph_hidden(4).with_labels(3).with_slot_chunk(2).init(&device).grow_a4_rounds();
    let mut decider=head(&device).grow_a4_rounds();
    decider.outcome_output.bias=Some(burn::module::Param::from_tensor(Tensor::zeros([12],&device)));
    let source=Tensor::from_data([[[0.3f32,-0.2],[0.7,0.4]]],&device);
    let documents=[SourcePointerDocument {row:0,document:0,first_token:0,text:"ab",token_offsets:&[(0,1),(1,2)]}];
    let context=SourcePointerContext::new(&plan,source.clone(),&documents,model.pointer_head.none.val()).unwrap();
    let cache=model.prepare(source.clone(),source,&batch);let pairs=model.source_pairs(&plan);
    let schedule=DenoiseSchedule {steps:10,eta:0.5,temperature:1.0,slot_chunk:2};
    let mut calls=Vec::new();
    let spans=model.run_pointer_window_with_hooks(&plan,&pairs,&cache,&slots,&context,None,&schedule,&decider,0,
        RefinementWindow {first_round:0,end_round:3,spans_only:true},None,|round|calls.push(round));
    assert_eq!(calls,vec![1,2,3]);
    assert_eq!(spans.rounds.len(),3);
    assert_eq!(spans.rounds.iter().filter(|round|round.outcome_logits.is_some()).count(),2);
    assert!(spans.rounds.last().unwrap().outcome_logits.is_none(),"no target/action crosses the query-layout change");
    for round in &spans.rounds {
        let visible=host(&round.annotations.visibility);
        assert!(visible.chunks_exact(4).all(|row|row[RELATION]==0.0&&row[EVIDENCE]==0.0));
        let retained=round.retained.filled.clone().int().to_data().try_to_vec_as::<i32>().unwrap();
        assert!(retained.chunks_exact(4).all(|row|row[0]==0&&row[3]==0));
        assert!(host(&round.remasked).chunks_exact(6).all(|row|row[0]==0.0&&row[5]==0.0));
    }
    let bank=vec![vec![DocumentEntities {entities:vec![Entity::provided(vec![TokenSpan::new(0,1)]),
        Entity::provided(vec![TokenSpan::new(1,2)])]}]];
    let pair_plan=SlotPlan::new(&layout,&bank,&Candidates::AllOrderedPairs).unwrap();
    let pair_slots=pair_plan.tensors(&device);let pair_context=context.for_plan(&pair_plan).unwrap();
    let pairs=PairEncoderConfig::new().with_hidden(2).init(&device).forward(pair_context.h_final.clone(),&pair_slots);
    let observed=pair_plan.rows()[0].iter().map(|slot| {
        let subject=pair_context.span_endpoints(0,0,slot.subject,slot.subject+1)[0];
        let object=pair_context.span_endpoints(0,0,slot.object,slot.object+1)[0];
        [Some(subject[0]),Some(subject[1]),Some(object[0]),Some(object[1])]
    }).collect::<Vec<_>>();
    let initial=PointerState::initial(&pair_context,&observed).unwrap();
    let pair=model.run_pointer_window_with_hooks(&pair_plan,&pairs,&cache,&pair_slots,&pair_context,Some(&initial),
        &schedule,&decider,0,RefinementWindow {first_round:3,end_round:10,spans_only:false},None,|round|calls.push(round));
    assert_eq!(calls,(1..=10).collect::<Vec<_>>());
    assert_eq!(pair.rounds.len(),7);
    assert_eq!(pair.rounds.iter().filter(|round|round.outcome_logits.is_some()).count(),6);
    assert_eq!(pair.rounds.last().unwrap().annotations.step.to_data().try_to_vec_as::<i32>().unwrap(),vec![10;2]);
    for round in &pair.rounds {
        assert_eq!(round.pointers.observed_offsets.to_data(),initial.observed_offsets.to_data());
        for field in 0..4 {assert_eq!(bits(&round.pointers.token[field]),bits(&initial.token[field]));}
        assert!(host(&round.acceptance_eligible).chunks_exact(4).all(|row|row[1]==0.0&&row[2]==0.0));
    }
    let training=model.joint_query_forward_window(QueryForwardInputs {plan:&pair_plan,slots:&pair_slots,pairs:&pairs,
        cache:&cache,context:&pair_context,observed:&observed,exploration_flips:None},&schedule,&DenoiserTrainingConfig::new(),
        &decider,0,RefinementWindow {first_round:3,end_round:10,spans_only:false},None).unwrap();
    for (inference,training) in pair.rounds.iter().zip(&training.rounds) {same_state(inference,training);}
    let provided=model.run_pointer_window_with_hooks(&pair_plan,&pairs,&cache,&pair_slots,&pair_context,Some(&initial),
        &schedule,&decider,0,RefinementWindow::full(10),None,|_|{});
    assert_eq!(provided.rounds.len(),10);
    assert_eq!(provided.rounds.iter().filter(|round|round.outcome_logits.is_some()).count(),9);
}

#[test]
fn a5_denoiser_new_step_rows_receive_actual_relation_gradients_cpu() {
    let device=Device::ndarray().autodiff();device.seed(20261024);
    let fixture=fixture(&device);let old_id=fixture.model.step_embedding.weight.id;
    let old=fixture.model.step_embedding.weight.val().to_data().try_to_vec_as::<f32>().unwrap();
    let model=fixture.model.grow_a4_rounds();assert_eq!(model.step_embedding.weight.id,old_id);
    let values=model.step_embedding.weight.val().to_data().try_to_vec_as::<f32>().unwrap();
    assert_eq!(&values[..16],old.as_slice());assert_eq!(&values[16..18],&old[14..16]);assert_eq!(&values[18..20],&old[14..16]);
    let mut decider=head(&device).grow_a4_rounds();
    decider.mlp.first_dropout.prob=0.0;decider.mlp.second_dropout.prob=0.0;
    let schedule=DenoiseSchedule {steps:10,..fixture.schedule};
    let output=model.run_pointer_window_with_hooks(&fixture.plan,&fixture.pairs,&fixture.cache,&fixture.slots,&fixture.context,None,
        &schedule,&decider,0,RefinementWindow::full(10),None,|_|{});
    let loss=output.rounds[8].output.annotations.relation.clone().square().sum()
        +output.rounds[9].output.annotations.relation.clone().square().sum();
    let gradients=loss.backward();
    let gradient=model.step_embedding.weight.val().grad(&gradients).unwrap().to_data().try_to_vec_as::<f32>().unwrap();
    assert!(gradient[16..18].iter().any(|value|value.abs()>1e-8));
    assert!(gradient[18..20].iter().any(|value|value.abs()>1e-8));
}

#[test]
fn a5_grown_round_tables_preserve_the_eight_call_a4_readout_cpu() {
    let device=Device::ndarray();device.seed(20261026);
    let fixture=fixture(&device);let decider=head(&device);
    let grown=fixture.model.clone().grow_a4_rounds();let grown_decider=decider.clone().grow_a4_rounds();
    let initial=PointerState::initial(&fixture.context,&fixture.observed).unwrap();
    let run=|model:&AnnotationDenoiser,head:&RefinementDecider|model.run_pointer_window_with_hooks(
        &fixture.plan,&fixture.pairs,&fixture.cache,&fixture.slots,&fixture.context,Some(&initial),
        &fixture.schedule,head,0,RefinementWindow::full(8),None,|_|{});
    let original=run(&fixture.model,&decider);let expanded=run(&grown,&grown_decider);
    assert_eq!(original.rounds.len(),8);assert_eq!(expanded.rounds.len(),8);
    for (original,expanded) in original.rounds.iter().zip(&expanded.rounds) {
        same_state(original,expanded);assert_eq!(bits(&original.retained.age),bits(&expanded.retained.age));
        assert_eq!(bits(&original.retained.relation_posterior),bits(&expanded.retained.relation_posterior));
    }
}

#[test]
fn a5_pointer_prior_changes_only_legal_source_logits_and_keeps_its_gradient_cpu() {
    let device=Device::ndarray().autodiff();device.seed(20261027);
    let fixture=fixture(&device);let model=fixture.model.grow_a4_rounds();
    let mut decider=head(&device).grow_a4_rounds();decider.mlp.first_dropout.prob=0.0;decider.mlp.second_dropout.prob=0.0;
    let schedule=DenoiseSchedule {steps:10,..fixture.schedule};
    let window=RefinementWindow {first_round:0,end_round:3,spans_only:true};
    let prior=Tensor::from_data(TensorData::new(vec![0.0f32,1000.0,0.0,1000.0,0.0,0.0,0.0,0.0,0.0,0.0],[2,5]),&device).require_grad();
    let priors=[prior.clone(),prior.clone()];
    let run=|prior|model.run_pointer_window_with_hooks(&fixture.plan,&fixture.pairs,&fixture.cache,&fixture.slots,&fixture.context,None,
        &schedule,&decider,0,window,prior,|_|{});
    let plain=run(None);let biased=run(Some(&priors));
    for field in 0..4 {
        let old=host(&plain.rounds[0].output.pointers.token[field]);
        let new=host(&biased.rounds[0].output.pointers.token[field]);
        for query in 0..2 {for token in 0..5 {
            let index=query*5+token;
            if old[index].is_finite() {
                let expected=if token==1||token==3 {1000.0}else{0.0};
                assert!((new[index]-old[index]-expected).abs()<1e-3);
            }else{assert_eq!(new[index],f32::NEG_INFINITY);}
        }}
        assert_eq!(new[4],old[4]);assert_eq!(new[9],old[9],"NONE logits have no mention prior");
    }
    let gradient=biased.rounds[0].output.pointers.token[0].clone().slice([0..1,0..1,1..2]).sum().backward();
    let values=prior.grad(&gradient).unwrap().to_data().try_to_vec_as::<f32>().unwrap();
    assert_eq!(values[1],1.0);
}
