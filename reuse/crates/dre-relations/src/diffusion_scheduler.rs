//! Shared proposal/retained-answer rollout, including global-round mentions-first windows.
use burn::prelude::*;
use burn::tensor::{DType, activation::softmax};
use crate::{AnnotationDenoiser, DenoiseOutput, DocumentCache, PairFeatures, SlotPlan, SlotTensors};
use crate::pointer::{SourcePointerContext, PointerDenoiseOutput, PointerLogits, PointerState, PointerStepOutput};
use crate::remask::{REMASK_FIELDS, RefinementDecider, RemaskFrame,
    RetainedAnswer, RetainedSnapshot, coherent_fields, observed_fields, outcome_actions};
use crate::state::{DenoiseSchedule, EVIDENCE, FIELDS, MASKED_FIELDS, RELATION, SoftState, StepOutput};
use crate::joint_circuit::{CircuitCall, CircuitEvidence, EndpointTables, CIRCUIT_FIELDS};

/// Half-open global call indices. A layout transition is terminal for its old query rows.
#[derive(Clone,Copy,Debug)]
pub struct RefinementWindow {
    pub first_round:usize,
    pub end_round:usize,
    pub spans_only:bool,
}
impl RefinementWindow {
    pub fn full(horizon:usize)->Self {Self {first_round:0,end_round:horizon,spans_only:false}}
    pub fn len(self)->usize {self.end_round-self.first_round}
    fn validate(self,horizon:usize) {
        assert!(self.first_round<self.end_round&&self.end_round<=horizon);
    }
}

fn normal_distribution(old:&Tensor<3>,logits:&Tensor<3>,schedule:&DenoiseSchedule)->Tensor<3> {
    if schedule.eta==0.0 {return old.clone().cast(DType::F32);}
    let logits=logits.clone().cast(DType::F32);
    let absent=logits.clone().max_dim(2).equal_elem(f32::NEG_INFINITY).expand(logits.dims());
    // Undefined circuit conditionals do not replace the previous soft state.
    let prediction=softmax(logits.mask_fill(absent.clone(),0.0).div_scalar(schedule.temperature),2);
    let old=old.clone().cast(DType::F32);
    let next=if schedule.eta==1.0 {prediction}else{
        old.clone().mul_scalar(1.0-schedule.eta)+prediction.mul_scalar(schedule.eta)};
    next.mask_where(absent,old)
}

/// Normal convex annotation update and visibility gain. Learned actions are applied afterwards.
pub fn normal_update(state: &SoftState, output: &StepOutput, schedule: &DenoiseSchedule) -> SoftState {
    let increment = Tensor::<1>::from_floats(MASKED_FIELDS, &state.visibility.device())
        .mul_scalar(schedule.visibility_step()).reshape([1, 1, FIELDS]);
    SoftState {relation: normal_distribution(&state.relation, &output.relation,schedule),
        evidence_start: normal_distribution(&state.evidence_start, &output.evidence_start,schedule),
        evidence_end: normal_distribution(&state.evidence_end, &output.evidence_end,schedule),
        visibility: (state.visibility.clone() + increment).clamp(0.0, 1.0),
        subject: state.subject.clone(), object: state.object.clone(), step: state.step.clone().add_scalar(1)}
}

pub fn normal_field(old: &Tensor<3>, logits: &Tensor<3>, rho: &Tensor<3>,
    schedule: &DenoiseSchedule) -> (Tensor<3>, Tensor<3>) {
    (normal_distribution(old,logits,schedule),
        (rho.clone() + schedule.visibility_step()).clamp(0.0, 1.0))
}

/// Shared training/inference controller: normal updates, then chosen masked-field resets.
/// Inactive rows are made finite before normalization and restored with boolean selection afterwards.
/// Supplied token values and raw byte offsets never change; remasking does not reset or harden Z.
pub fn advance_with_remask(annotations: &SoftState, pointers: &PointerState, output: &PointerStepOutput,
    schedule: &DenoiseSchedule, remasked: &Tensor<3>, active: &Tensor<2>) -> (SoftState, PointerState) {
    let [b, m, _] = remasked.dims();
    assert_eq!(remasked.dims(), [b, m, REMASK_FIELDS]);
    assert_eq!(active.dims(), [b, m]);
    let inactive2 = active.clone().lower_elem(0.5);
    let inactive = inactive2.clone().unsqueeze_dim(2);
    let clean = StepOutput {
        relation: output.annotations.relation.clone().mask_fill(inactive.clone(), 0.0),
        evidence_start: output.annotations.evidence_start.clone().mask_fill(inactive.clone(), 0.0),
        evidence_end: output.annotations.evidence_end.clone().mask_fill(inactive.clone(), 0.0),
    };
    let clean_pointers = PointerLogits {
        token: output.pointers.token.each_ref().map(|field| field.clone().mask_fill(inactive.clone(), 0.0)),
        byte: output.pointers.byte.clone(), byte_queries: output.pointers.byte_queries.clone(),
    };
    let clean_annotations=SoftState {
        relation:annotations.relation.clone().mask_fill(inactive.clone(),0.0),
        evidence_start:annotations.evidence_start.clone().mask_fill(inactive.clone(),0.0),
        evidence_end:annotations.evidence_end.clone().mask_fill(inactive.clone(),0.0),
        visibility:annotations.visibility.clone().mask_fill(inactive.clone(),0.0),
        subject:annotations.subject.clone(),object:annotations.object.clone(),step:annotations.step.clone(),
    };
    let clean_state=PointerState {
        token:pointers.token.each_ref().map(|field|field.clone().mask_fill(inactive.clone(),0.0)),
        visibility:pointers.visibility.clone().mask_fill(inactive.clone(),0.0),
        observed:pointers.observed.clone().mask_fill(inactive.clone(),0.0),observed_offsets:pointers.observed_offsets.clone(),
    };
    let mut next = normal_update(&clean_annotations, &clean, schedule);
    let mut pointer_next = clean_state.update(&clean_pointers, schedule);
    let relation_mask = remasked.clone().slice([0..b, 0..m, 0..1]).greater_elem(0.5);
    let evidence_mask = remasked.clone().slice([0..b, 0..m, 5..6]).greater_elem(0.5);
    let relation = next.visibility.clone().slice([0..b, 0..m, RELATION..RELATION + 1]).mask_fill(relation_mask, 0.0);
    let evidence = next.visibility.clone().slice([0..b, 0..m, EVIDENCE..EVIDENCE + 1]).mask_fill(evidence_mask, 0.0);
    next.visibility = next.visibility.slice_assign([0..b, 0..m, RELATION..RELATION + 1], relation)
        .slice_assign([0..b, 0..m, EVIDENCE..EVIDENCE + 1], evidence);
    let selected = remasked.clone().slice([0..b, 0..m, 1..5]).greater_elem(0.5)
        .bool_and(pointers.observed.clone().lower_elem(0.5));
    pointer_next.visibility = pointer_next.visibility.mask_fill(selected, 0.0);
    next.relation = next.relation.mask_where(inactive.clone().expand(annotations.relation.dims()), annotations.relation.clone());
    next.evidence_start = next.evidence_start.mask_where(inactive.clone().expand(annotations.evidence_start.dims()), annotations.evidence_start.clone());
    next.evidence_end = next.evidence_end.mask_where(inactive.clone().expand(annotations.evidence_end.dims()), annotations.evidence_end.clone());
    next.visibility = next.visibility.mask_where(inactive.clone().expand(annotations.visibility.dims()), annotations.visibility.clone());
    next.step = next.step.mask_where(inactive2, annotations.step.clone());
    pointer_next.token = std::array::from_fn(|field|
        pointer_next.token[field].clone().mask_where(inactive.clone().expand(pointers.token[field].dims()), pointers.token[field].clone()));
    pointer_next.visibility = pointer_next.visibility.mask_where(inactive.expand(pointers.visibility.dims()), pointers.visibility.clone());
    (next, pointer_next)
}

#[derive(Debug, Clone)]
pub struct RefinementRound {
    pub output: PointerStepOutput,
    /// Raw-call joint distribution, retained for the coupling objective and terminal relation readout.
    pub circuit: CircuitCall,
    /// Supplied endpoint tokens only; model-retained answers are never training evidence.
    pub supplied: CircuitEvidence,
    /// Feedback/terminal log-marginals given next-visible retained answers, excluding the field itself.
    pub conditional_relation: Tensor<3>,
    pub conditional_tokens: [Tensor<3>; 4],
    pub annotations: SoftState,
    pub pointers: PointerState,
    /// Every real query is live on every call; padding alone is inactive.
    pub active: Tensor<2>,
    pub acceptance_logits: Tensor<3>,
    pub acceptance_eligible: Tensor<3>,
    pub outcome_logits: Option<Tensor<3>>,
    pub acceptance_features: Tensor<3>,
    pub outcome_features: Option<Tensor<3>>,
    pub policy_remasked: Tensor<3>,
    pub remasked: Tensor<3>,
    pub exploration_flipped: Tensor<3>,
    pub observed: Tensor<3>,
    pub argmaxes: Tensor<3, Int>,
    pub retained: RetainedSnapshot,
    pub accepted: Tensor<3>,
    pub replaced: Tensor<3>,
}
impl RefinementRound {
    pub fn slot_view(&self,row:usize,slots:std::ops::Range<usize>)->Self {
        let view=|tensor:&Tensor<3>|tensor.clone().slice([row..row+1,slots.clone()]);
        Self {output:self.output.slot_view(row,slots.clone()),annotations:self.annotations.slot_view(row,slots.clone()),
            circuit:self.circuit.slot_view(row,slots.clone()),supplied:self.supplied.slot_view(row,slots.clone()),
            conditional_relation:view(&self.conditional_relation),
            conditional_tokens:self.conditional_tokens.each_ref().map(view),
            pointers:self.pointers.slot_view(row,slots.clone()),active:self.active.clone().slice([row..row+1,slots.clone()]),
            acceptance_logits:view(&self.acceptance_logits),acceptance_eligible:view(&self.acceptance_eligible),
            outcome_logits:self.outcome_logits.as_ref().map(view),acceptance_features:view(&self.acceptance_features),
            outcome_features:self.outcome_features.as_ref().map(view),policy_remasked:view(&self.policy_remasked),
            remasked:view(&self.remasked),exploration_flipped:view(&self.exploration_flipped),observed:view(&self.observed),
            argmaxes:self.argmaxes.clone().slice([row..row+1,slots.clone()]),retained:self.retained.slot_view(row,slots.clone()),
            accepted:view(&self.accepted),replaced:view(&self.replaced)}
    }
}

/// Detach before convex updates, rather than registering unreachable operations then detaching.
#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_query_training_state(annotations:&SoftState,pointers:&PointerState,output:&PointerStepOutput,
    schedule:&DenoiseSchedule,remasked:&Tensor<3>,valid:&Tensor<2>,detach_between_steps:bool)->(SoftState,PointerState) {
    if detach_between_steps&&annotations.relation.device().is_autodiff() {
        let output=PointerStepOutput {annotations:StepOutput {relation:output.annotations.relation.clone().inner(),
            evidence_start:output.annotations.evidence_start.clone().inner(),evidence_end:output.annotations.evidence_end.clone().inner()},
            pointers:output.pointers.without_autodiff(),hidden:output.hidden.clone().inner()};
        let (next,pointers)=advance_with_remask(&annotations.without_autodiff(),&pointers.without_autodiff(),
            &output,schedule,&remasked.clone().inner(),&valid.clone().inner());
        (next.with_autodiff(),pointers.with_autodiff())
    } else {advance_with_remask(annotations,pointers,output,schedule,remasked,valid)}
}

/// Assemble next-call evidence on device, using source-context indices uploaded once for the rollout.
/// Supplied observations win over retained answers; Bool masks combine only elementwise.
fn visible_retained_evidence(supplied:&CircuitEvidence,retained:&RetainedAnswer,
    remasked:&Tensor<3>,valid:&Tensor<2>,context:&SourcePointerContext)->CircuitEvidence {
    let [b,m,_]=supplied.values.dims();
    let visible=retained.filled.clone().select(2,context.circuit_retained_roles.clone())
        .bool_and(remasked.clone().slice([0..b,0..m,0..CIRCUIT_FIELDS]).lower_elem(0.5));
    let known=supplied.known.clone().bool_or(visible)
        .bool_and(valid.clone().greater_elem(0.5).unsqueeze_dim(2));
    let values=retained.value.clone().select(2,context.circuit_retained_factors.clone())
        .mask_where(supplied.known.clone(),supplied.values.clone()).mask_fill(known.clone().bool_not(),0);
    CircuitEvidence {values,known}
}

/// Shared hard transitions, also usable with a dense graph oracle in CPU behavior checks.
#[allow(clippy::too_many_arguments)]
pub fn refinement_transition(annotations:&SoftState,pointers:&PointerState,output:PointerStepOutput,
    retained:&mut RetainedAnswer,previous:Option<&RemaskFrame>,context:&SourcePointerContext,
    denoiser:&AnnotationDenoiser,tables:&EndpointTables,supplied:&CircuitEvidence,schedule:&DenoiseSchedule,
    decider:&RefinementDecider,no_relation:usize,valid:&Tensor<2>,round:usize,exploration_flips:Option<&Tensor<3>>,
    detach_between_steps:bool)->(RefinementRound,RemaskFrame) {
    refinement_transition_window(annotations,pointers,output,retained,previous,context,denoiser,tables,supplied,
        schedule,decider,no_relation,valid,round,exploration_flips,detach_between_steps,RefinementWindow::full(schedule.steps))
}

#[allow(clippy::too_many_arguments)]
pub fn refinement_transition_window(annotations:&SoftState,pointers:&PointerState,mut output:PointerStepOutput,
    retained:&mut RetainedAnswer,previous:Option<&RemaskFrame>,context:&SourcePointerContext,
    denoiser:&AnnotationDenoiser,tables:&EndpointTables,supplied:&CircuitEvidence,schedule:&DenoiseSchedule,
    decider:&RefinementDecider,no_relation:usize,valid:&Tensor<2>,round:usize,exploration_flips:Option<&Tensor<3>>,
    detach_between_steps:bool,window:RefinementWindow)->(RefinementRound,RemaskFrame) {
    let [b,m,_]=output.hidden.dims();let invalid=valid.clone().lower_elem(0.5).unsqueeze_dim(2);
    let observed=observed_fields(pointers).mask_fill(invalid.clone(),0.0);
    let mut normalized_output=output.clone();
    normalized_output.pointers.token=std::array::from_fn(|field|output.pointers.token[field].clone()
        .mask_fill(context.outside.clone(),f32::NEG_INFINITY));
    let circuit=denoiser.joint_circuit.call(output.hidden.clone(),output.annotations.relation.clone(),
        &normalized_output.pointers.token,tables,valid);
    let prepared=circuit.prepared_decisions(supplied);
    dre_core::profile::memory_phase("round_joint_circuit",&output.hidden.device(),||format!("{{\"round\":{},\"batch\":{b},\"slots_per_row\":{m}}}",round+1));
    let decoded=if window.spans_only {prepared.greedy_decode_mentions()}else{prepared.greedy_decode()};
    output.pointers.byte=std::array::from_fn(|field|denoiser.pointer_head.byte_logits(
        output.pointers.byte_queries[field].clone().mask_fill(invalid.clone(),0.0),context,
        decoded.clone().slice([0..b,0..m,field+1..field+2]).reshape([b,m]))
        .mask_fill(invalid.clone(),0.0));
    normalized_output.pointers.byte=output.pointers.byte.clone();
    dre_core::profile::memory_phase("round_decode_and_byte_logits",&output.hidden.device(),||format!("{{\"round\":{},\"byte_logits\":{:?}}}",
        round+1,output.pointers.byte[0].dims()));
    let dependency=prepared.dependency_features(&decoded).mask_fill(invalid.clone(),0.0);
    let frame=RemaskFrame::from_output(&normalized_output,pointers,no_relation,context.none,valid,&decoded);
    let base=frame.features(previous,&observed).mask_fill(invalid.clone(),0.0);
    let visibility=Tensor::cat(vec![annotations.visibility.clone().slice([0..b,0..m,RELATION..RELATION+1]),
        pointers.visibility.clone(),annotations.visibility.clone().slice([0..b,0..m,EVIDENCE..EVIDENCE+1])],2)
        .mask_fill(invalid.clone(),0.0);
    retained.age_one_call(schedule.steps);
    let acceptance_features=Tensor::cat(vec![retained.features(&frame,base.clone(),visibility.clone()),dependency.clone()],2)
        .mask_fill(invalid.clone(),0.0);
    let hidden=output.hidden.clone().mask_fill(invalid.clone(),0.0);
    let acceptance_logits=decider.acceptance(hidden.clone(),acceptance_features.clone(),round);
    let mut acceptance_eligible=coherent_fields(&frame,pointers,context,valid);
    if window.spans_only {
        acceptance_eligible=acceptance_eligible.slice_assign([0..b,0..m,0..1],Tensor::zeros([b,m,1],&output.hidden.device()))
            .slice_assign([0..b,0..m,3..4],Tensor::zeros([b,m,1],&output.hidden.device()));
    }
    let (accepted,replaced)=retained.update(&frame,&base,&acceptance_logits,&acceptance_eligible);
    dre_core::profile::memory_phase("round_acceptance",&output.hidden.device(),||format!("{{\"round\":{},\"acceptance_features\":{:?}}}",
        round+1,acceptance_features.dims()));
    let (outcome_features,outcome_logits,policy_remasked,remasked,exploration_flipped)=if round+1<window.end_round {
        let features=Tensor::cat(vec![retained.features(&frame,base,visibility),dependency],2).mask_fill(invalid.clone(),0.0);
        let logits=decider.outcomes(hidden,features.clone(),round);
        let outcome_observed=if window.spans_only {
            observed.clone().slice_assign([0..b,0..m,0..1],Tensor::ones([b,m,1],&output.hidden.device()))
                .slice_assign([0..b,0..m,5..6],Tensor::ones([b,m,1],&output.hidden.device()))
        }else{observed.clone()};
        let (policy,taken,flipped)=outcome_actions(&logits,&outcome_observed,valid,exploration_flips);
        (Some(features),Some(logits),policy,taken,flipped)
    } else {
        let zero=Tensor::zeros([b,m,REMASK_FIELDS],&output.hidden.device());
        (None,None,zero.clone(),zero.clone(),zero)
    };
    let evidence=visible_retained_evidence(supplied,retained,&remasked,valid,context);
    let (conditional_relation,conditional_tokens)=circuit.prepared(&evidence).conditional_log_marginals(true);
    dre_core::profile::memory_phase("round_conditional_marginals",&output.hidden.device(),||format!("{{\"round\":{}}}",round+1));
    // Circuit fields already represent the required probabilities. Only uncoupled evidence keeps tau.
    let evidence_logits=|logits:&Tensor<3>|if schedule.temperature==1.0 {logits.clone()}else{
        logits.clone().cast(DType::F32).div_scalar(schedule.temperature)};
    let feedback=PointerStepOutput {annotations:StepOutput {relation:conditional_relation.clone(),
        evidence_start:evidence_logits(&output.annotations.evidence_start),evidence_end:evidence_logits(&output.annotations.evidence_end)},
        pointers:PointerLogits {token:conditional_tokens.clone(),byte:output.pointers.byte.clone(),
            byte_queries:output.pointers.byte_queries.clone()},hidden:output.hidden.clone()};
    let feedback_schedule=DenoiseSchedule {temperature:1.0,steps:window.len(),..*schedule};
    let (mut next,next_pointers)=advance_query_training_state(annotations,pointers,&feedback,&feedback_schedule,&remasked,valid,detach_between_steps);
    if window.spans_only {
        next.relation=annotations.relation.clone();
        next.evidence_start=annotations.evidence_start.clone();
        next.evidence_end=annotations.evidence_end.clone();
        next.visibility=annotations.visibility.clone();
    }
    let argmaxes=frame.argmaxes();let history=if detach_between_steps {frame.detached()}else{frame};
    (RefinementRound {output,circuit,supplied:supplied.clone(),conditional_relation,conditional_tokens,
        annotations:next,pointers:next_pointers,active:valid.clone(),acceptance_logits,acceptance_eligible,
        outcome_logits,acceptance_features,outcome_features,policy_remasked,remasked,exploration_flipped,observed,argmaxes,
        retained:retained.snapshot(),accepted,replaced},history)
}

#[derive(Debug, Clone)]
pub struct ScheduledOutput {
    pub output: DenoiseOutput,
    pub slot_steps: Vec<usize>,
    pub average_steps: f64,
}
#[derive(Debug, Clone)]
pub struct ScheduledPointerOutput {
    pub output: PointerDenoiseOutput,
    pub rounds: Vec<RefinementRound>,
    pub retained: RetainedAnswer,
    pub slot_steps: Vec<usize>,
    pub average_steps: f64,
}

fn valid_slots(plan: &SlotPlan, slots: &SlotTensors) -> Vec<bool> {
    let mut valid = vec![false; slots.batch * slots.slots_per_row];
    debug_assert_eq!(plan.batch(),slots.batch);
    debug_assert_eq!(plan.slots_per_row(),slots.slots_per_row);
    debug_assert_eq!(plan.tokens(),slots.tokens);
    debug_assert_eq!(plan.rows().len(),slots.batch);
    for (row, entries) in plan.rows().iter().enumerate() {
        debug_assert!(entries.len()<=slots.slots_per_row);
        valid[row * slots.slots_per_row..row * slots.slots_per_row + entries.len()].fill(true);
    }
    valid
}

impl AnnotationDenoiser {
    /// Real annotation-only primitive for non-production synthetic relation experiments.
    /// No learned stop or obsolete hand-adaptive policy is applied to this path.
    pub fn run_annotation_with_hooks(&self, plan: &SlotPlan, pairs: &PairFeatures, cache: &DocumentCache,
        slots: &SlotTensors, schedule: &DenoiseSchedule, relation_prior: Option<&[f32]>,
        mut transform: impl FnMut(&SoftState, StepOutput, &[usize]) -> StepOutput,
        mut observe: impl FnMut(usize)) -> ScheduledOutput {
        self.check_schedule(schedule);
        let device = pairs.pairs.device();let (b, m, t) = (slots.batch, slots.slots_per_row, slots.tokens);
        debug_assert_eq!(plan.batch(),b);
        debug_assert_eq!(plan.slots_per_row(),m);
        debug_assert_eq!(plan.tokens(),t);
        let mut state = SoftState::initial(plan, slots, self.labels(), relation_prior, &device);
        let steps = if m == 0 {0} else {schedule.steps};
        let mut last = StepOutput {relation: Tensor::empty([b, 0, self.labels()], &device),
            evidence_start: Tensor::empty([b, 0, t], &device), evidence_end: Tensor::empty([b, 0, t], &device)};
        let indices: Vec<_> = (0..b * m).collect();
        for step in 0..steps {
            let output = self.step(pairs, cache, slots, &state, schedule.slot_chunk);
            let transformed = transform(&state, StepOutput {relation: output.relation.reshape([1, b * m, self.labels()]),
                evidence_start: output.evidence_start.reshape([1, b * m, t]), evidence_end: output.evidence_end.reshape([1, b * m, t])}, &indices);
            let output = StepOutput {relation: transformed.relation.reshape([b, m, self.labels()]),
                evidence_start: transformed.evidence_start.reshape([b, m, t]), evidence_end: transformed.evidence_end.reshape([b, m, t])};
            state = normal_update(&state, &output, schedule);last = output;observe(step + 1);
        }
        let valid = valid_slots(plan, slots);
        ScheduledOutput {output: DenoiseOutput {state, last, steps},
            slot_steps: valid.iter().map(|&real| if real {steps} else {0}).collect(),
            average_steps: if valid.iter().any(|&real| real) {steps as f64} else {0.0}}
    }
    /// Shared inference entry for a global-round interval on one fixed query layout.
    #[allow(clippy::too_many_arguments)]
    pub fn run_pointer_window_with_hooks(&self,plan:&SlotPlan,pairs:&PairFeatures,cache:&DocumentCache,
        slots:&SlotTensors,context:&SourcePointerContext,initial_pointers:Option<&PointerState>,
        schedule:&DenoiseSchedule,decider:&RefinementDecider,no_relation:usize,window:RefinementWindow,
        prior:Option<&[Tensor<2>;2]>,observe:impl FnMut(usize))->ScheduledPointerOutput {
        let initial=initial_pointers.cloned().unwrap_or_else(||PointerState::masked(context));
        self.refinement_rollout_window(plan,pairs,cache,slots,context,initial,schedule,None,decider,no_relation,
            None,false,window,prior,|_,output,_|output,observe)
    }


    pub fn run_pointer_with_remask_hooks(&self,plan:&SlotPlan,pairs:&PairFeatures,cache:&DocumentCache,
        slots:&SlotTensors,context:&SourcePointerContext,initial_pointers:Option<&PointerState>,
        schedule:&DenoiseSchedule,relation_prior:Option<&[f32]>,decider:&RefinementDecider,no_relation:usize,
        transform:impl FnMut(&SoftState,StepOutput,&[usize])->StepOutput,observe:impl FnMut(usize))->ScheduledPointerOutput {
        let initial=initial_pointers.cloned().unwrap_or_else(||PointerState::masked(context));
        self.refinement_rollout(plan,pairs,cache,slots,context,initial,schedule,relation_prior,decider,no_relation,None,false,transform,observe)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn refinement_rollout(&self,plan:&SlotPlan,pairs:&PairFeatures,cache:&DocumentCache,
        slots:&SlotTensors,context:&SourcePointerContext,initial_pointers:PointerState,schedule:&DenoiseSchedule,
        relation_prior:Option<&[f32]>,decider:&RefinementDecider,no_relation:usize,exploration_flips:Option<&Tensor<4>>,
        detach_between_steps:bool,transform:impl FnMut(&SoftState,StepOutput,&[usize])->StepOutput,
        observe:impl FnMut(usize))->ScheduledPointerOutput {
        self.refinement_rollout_window(plan,pairs,cache,slots,context,initial_pointers,schedule,relation_prior,decider,no_relation,
            exploration_flips,detach_between_steps,RefinementWindow::full(schedule.steps),None,transform,observe)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn refinement_rollout_window(&self,plan:&SlotPlan,pairs:&PairFeatures,cache:&DocumentCache,
        slots:&SlotTensors,context:&SourcePointerContext,initial_pointers:PointerState,schedule:&DenoiseSchedule,
        relation_prior:Option<&[f32]>,decider:&RefinementDecider,no_relation:usize,exploration_flips:Option<&Tensor<4>>,
        detach_between_steps:bool,window:RefinementWindow,prior:Option<&[Tensor<2>;2]>,
        mut transform:impl FnMut(&SoftState,StepOutput,&[usize])->StepOutput,mut observe:impl FnMut(usize))->ScheduledPointerOutput {
        self.check_schedule(schedule);window.validate(schedule.steps);
        let (b,m)=(slots.batch,slots.slots_per_row);let device=pairs.pairs.device();
        if let Some(flips)=exploration_flips {assert_eq!(flips.dims(),[window.len()-1,b,m,REMASK_FIELDS]);}
        let mut annotations=SoftState::initial(plan,slots,self.labels(),relation_prior,&device);
        annotations.step=annotations.step.add_scalar(window.first_round as i64);
        let mut pointers=initial_pointers;let mut retained=RetainedAnswer::empty(b,m,self.labels(),&device);
        let tables=self.joint_circuit.endpoint_tables(context);
        let supplied=pointers.circuit_evidence(&slots.slot_valid);
        let mut masked=Tensor::zeros([b,m,REMASK_FIELDS],&device);
        let mut previous=None;let mut rounds=Vec::with_capacity(window.len());let mut last=None;
        let indices:Vec<_>=(0..b*m).collect();
        if m>0 {for round in window.first_round..window.end_round {
            let mut output=self.step_pointer_retained(pairs,cache,slots,&annotations,&pointers,context,&retained,&masked,schedule.slot_chunk);
            dre_core::profile::memory_phase("round_denoiser_step",&device,||format!(
                "{{\"round\":{},\"pointer_token\":{:?},\"byte_queries\":{:?},\"hidden\":{:?}}}",round+1,
                output.pointers.token[0].dims(),output.pointers.byte_queries[0].dims(),output.hidden.dims()));
            if let Some(prior)=prior {
                for field in 0..4 {
                    let value=prior[field%2].clone().unsqueeze_dim::<3>(1).expand([b,m,slots.tokens+1]);
                    output.pointers.token[field]=(output.pointers.token[field].clone()+value)
                        .mask_fill(context.outside.clone(),f32::NEG_INFINITY);
                }
            }
            let t=slots.tokens;
            let transformed=transform(&annotations,StepOutput {relation:output.annotations.relation.reshape([1,b*m,self.labels()]),
                evidence_start:output.annotations.evidence_start.reshape([1,b*m,t]),evidence_end:output.annotations.evidence_end.reshape([1,b*m,t])},&indices);
            output.annotations=StepOutput {relation:transformed.relation.reshape([b,m,self.labels()]),
                evidence_start:transformed.evidence_start.reshape([b,m,t]),evidence_end:transformed.evidence_end.reshape([b,m,t])};
            let flips=exploration_flips.filter(|_|round+1<window.end_round)
                .map(|flips|flips.clone().slice([round-window.first_round..round-window.first_round+1]).reshape([b,m,REMASK_FIELDS]));
            let (readout,history)=refinement_transition_window(&annotations,&pointers,output,&mut retained,previous.as_ref(),context,
                self,&tables,&supplied,schedule,decider,no_relation,&slots.slot_valid,round,flips.as_ref(),detach_between_steps,window);
            annotations=readout.annotations.clone();pointers=readout.pointers.clone();masked=readout.remasked.clone();
            last=Some(readout.output.clone());previous=Some(history);rounds.push(readout);observe(round+1);
            dre_core::profile::memory_phase("refinement_round",&device,||format!(
                "{{\"round\":{},\"window\":[{},{}],\"spans_only\":{},\"batch\":{b},\"slots_per_row\":{m},\"tokens\":{}}}",
                round+1,window.first_round,window.end_round,window.spans_only,slots.tokens));
        }}
        let steps=rounds.len();let valid=valid_slots(plan,slots);
        ScheduledPointerOutput {output:PointerDenoiseOutput {annotations,pointers,last,steps},rounds,retained,
            slot_steps:valid.iter().map(|&real|if real {steps}else{0}).collect(),
            average_steps:if valid.iter().any(|&real|real) {steps as f64}else{0.0}}
    }
}

#[cfg(test)]
#[path = "diffusion_scheduler_tests.rs"]
mod tests;
