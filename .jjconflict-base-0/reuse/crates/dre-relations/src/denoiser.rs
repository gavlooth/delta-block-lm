//! Source-anchored annotation denoiser in provided-entity mode (spec §6.3–§6.5, §9).
//!
//! One refinement step for the slots of a batch:
//!
//! ```text
//! u_f  = (1 − rho_f) E_MASK,f + rho_f Σ_v Z_f,v E_f,v           masked fields (relation, evidence start/end)
//! u_f  = E_f,observed                                            observed subject / object (the entity vector)
//! q    = RMSNorm(p + Σ_f W_f u_f + E_step(s))
//! a    = S1(q; K_early, V_early)                                 SparseGlobalAttention1, per slot chunk
//! â    = GraphMixer(a; entity-sharing neighbor lists)            all slots at once
//! b    = S2(â; K_final, V_final)                                 SparseGlobalAttention2, per slot chunk
//! ℓ_rel   = W_rel RMSNorm(b) + c                                 [B, M, R]
//! ℓ_start = (W_s RMSNorm(b)) · (U_s H_final)ᵀ / √h              [B, M, T], masked to the slot's document
//! ℓ_end   = (W_e RMSNorm(b)) · (U_e H_final)ᵀ / √h
//! ```
//!
//! `S1` and `S2` ([`crate::sparse_global_attention`], spec §4.1) run at every refinement step, each with its own
//! `W_Q, W_K, W_V, W_O`; there is no SwiGLU after either. A batch with no candidate slot (`M = 0`) runs neither
//! layer and gets empty outputs.
//!
//! Evidence values are source positions, so their field embedding `E_evidence,t = U_v H_final[t]` is a projection
//! of the token's own feature, never a vocabulary entry. No output has a vocabulary axis.
//! The two banks and the three token projections are computed once per batch in [`DocumentCache`].

use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use dre_core::{BatchLayout, Precision, project};

use crate::entities::PairFeatures;
use crate::graph::{GraphMixer, GraphMixerConfig};
use crate::plan::{SlotPlan, SlotTensors};
use crate::sparse_global_attention::{
    SourceBank, SparseGlobalAttention1, SparseGlobalAttention2, SparseGlobalAttentionConfig, SparseGlobalAttentionInfo,
};
use crate::state::{DenoiseSchedule, EVIDENCE, RELATION, SoftState, StepOutput};
use crate::read::mixed_matmul;
use crate::pointer::{SourcePointerContext, PointerHead, PointerLogits, PointerState, PointerStepOutput, position_features};
use crate::remask::RetainedAnswer;
use crate::joint_circuit::{JointCircuit, JointCircuitConfig};

/// Query density is recorded in each checkpoint. Production fills its ratio from training-only counts.
#[derive(Config, Debug)]
pub struct SourceQueryConfig {
    #[config(default = 64)]
    pub minimum: usize,
    #[config(default = 10)]
    pub density_numerator: usize,
    #[config(default = 29)]
    pub density_denominator: usize,
}
impl SourceQueryConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.minimum == 0 || self.density_numerator == 0 || self.density_denominator == 0 {
            return Err("source query minimum and density ratio must be positive".into());
        }
        Ok(())
    }
    pub fn count(&self, tokens: usize) -> usize {
        assert!(self.minimum > 0 && self.density_numerator > 0 && self.density_denominator > 0);
        let numerator = tokens as u128 * self.density_numerator as u128;
        let denominator = self.density_denominator as u128;
        usize::try_from(numerator.div_ceil(denominator)).expect("source query count fits usize").max(self.minimum)
    }
}

#[derive(Config, Debug)]
pub struct AnnotationDenoiserConfig {
    #[config(default = 512)]
    pub hidden: usize,
    /// `h_global`: heads of `S1`, `S2` and the graph mixer.
    #[config(default = 8)]
    pub heads: usize,
    /// `d_global_head`: head width of `S1`, `S2` and the graph mixer.
    #[config(default = 64)]
    pub head_dim: usize,
    /// `d_graph`.
    #[config(default = 512)]
    pub graph_hidden: usize,
    /// `R`, including `NO_RELATION`.
    #[config(default = 64)]
    pub labels: usize,
    /// `K_diff`.
    #[config(default = 8)]
    pub refinement_steps: usize,
    /// Size of the step embedding table; no episode may run more steps.
    #[config(default = 8)]
    pub max_refinement_steps: usize,
    /// `η`.
    #[config(default = 0.5)]
    pub eta: f64,
    /// `τ`.
    #[config(default = 1.0)]
    pub temperature: f64,
    /// Slots per chunk for the reads and heads.
    #[config(default = 64)]
    pub slot_chunk: usize,
    #[config(default = "Precision::Fp32")]
    pub precision: Precision,
    #[config(default = "SourceQueryConfig::new()")]
    pub source_queries: SourceQueryConfig,
    #[config(default = 32)]
    pub circuit_states: usize,
    #[config(default = 8.0)]
    pub circuit_leaf_bound: f64,
}

impl AnnotationDenoiserConfig {
    /// Widths of the two sparse global-attention layers. Their BF16 query-weight gradient failed the f64 gate,
    /// so production source reads retain FP32 projections even when other denoiser projections are mixed.
    pub fn sparse_global_attention(&self) -> SparseGlobalAttentionConfig {
        SparseGlobalAttentionConfig::new()
            .with_hidden(self.hidden).with_heads(self.heads).with_head_dim(self.head_dim).with_precision(Precision::Fp32)
    }

    pub fn schedule(&self) -> DenoiseSchedule {
        DenoiseSchedule {
            steps: self.refinement_steps,
            eta: self.eta,
            temperature: self.temperature,
            slot_chunk: self.slot_chunk,
        }
    }

    pub fn init(&self, device: &Device) -> AnnotationDenoiser {
        assert!(
            (1..=self.max_refinement_steps).contains(&self.refinement_steps),
            "refinement_steps must be in 1..={}",
            self.max_refinement_steps
        );
        let h = self.hidden;
        let square = || LinearConfig::new(h, h).init(device);
        let field = || LinearConfig::new(h, h).with_bias(false).init(device);
        let embedding = Initializer::Normal { mean: 0.0, std: 0.02 };
        let sparse = self.sparse_global_attention();
        AnnotationDenoiser {
            evidence_source: square(),
            evidence_start_key: square(),
            evidence_end_key: square(),
            relation_labels: embedding.init([self.labels, h], device),
            relation_mask: embedding.init([h], device),
            evidence_start_mask: embedding.init([h], device),
            evidence_end_mask: embedding.init([h], device),
            relation_field: field(),
            subject_field: field(),
            object_field: field(),
            evidence_start_field: field(),
            evidence_end_field: field(),
            step_embedding: EmbeddingConfig::new(self.max_refinement_steps, h).init(device),
            query_norm: RmsNormConfig::new(h).init(device),
            sparse_global_attention_1: SparseGlobalAttention1::new(&sparse, device),
            graph: GraphMixerConfig::new()
                .with_hidden(h)
                .with_heads(self.heads)
                .with_head_dim(self.head_dim)
                .with_graph_hidden(self.graph_hidden)
                .with_precision(self.precision)
                .init(device),
            sparse_global_attention_2: SparseGlobalAttention2::new(&sparse, device),
            head_norm: RmsNormConfig::new(h).init(device),
            relation_head: LinearConfig::new(h, self.labels).init(device),
            evidence_start_query: square(),
            evidence_end_query: square(),
            pointer_head: PointerHead::new(h, device),
            joint_circuit: JointCircuitConfig::new(h, self.labels)
                .with_states(self.circuit_states).with_leaf_bound(self.circuit_leaf_bound).init(device),
            source_query: embedding.init([h], device),
            source_query_position: LinearConfig::new(h, h).with_bias(false).init(device),
            retained_markers: Initializer::Zeros.init([4,h], device),
            hidden: h,
            labels: self.labels,
            max_refinement_steps: self.max_refinement_steps,
            mixed: self.precision == Precision::Bf16Mixed,
        }
    }
}

#[cfg(test)]
mod precision_tests {
    use super::*;

    #[test]
    fn mixed_denoiser_keeps_sparse_global_attention_fp32() {
        let config = AnnotationDenoiserConfig::new().with_precision(Precision::Bf16Mixed);
        assert_eq!(config.sparse_global_attention().precision, Precision::Fp32);
    }
}

#[derive(Module, Debug)]
pub struct AnnotationDenoiser {
    /// `U_v`: evidence field embedding of a source position.
    pub evidence_source: Linear,
    /// `U_s`, `U_e`: token keys of the evidence boundary heads.
    pub evidence_start_key: Linear,
    pub evidence_end_key: Linear,
    /// `E_rel: [R, h]`.
    pub relation_labels: Param<Tensor<2>>,
    /// `E_MASK,f` of the masked fields.
    pub relation_mask: Param<Tensor<1>>,
    pub evidence_start_mask: Param<Tensor<1>>,
    pub evidence_end_mask: Param<Tensor<1>>,
    /// `W_f` per field.
    pub relation_field: Linear,
    pub subject_field: Linear,
    pub object_field: Linear,
    pub evidence_start_field: Linear,
    pub evidence_end_field: Linear,
    /// `E_step: [max_refinement_steps, h]`.
    pub step_embedding: Embedding,
    pub query_norm: RmsNorm,
    /// `S1`: reads the stage-5 bank after the query assembly.
    pub sparse_global_attention_1: SparseGlobalAttention1,
    pub graph: GraphMixer,
    /// `S2`: reads the stage-15 bank after the graph update.
    pub sparse_global_attention_2: SparseGlobalAttention2,
    pub head_norm: RmsNorm,
    pub relation_head: Linear,
    pub evidence_start_query: Linear,
    pub evidence_end_query: Linear,
    pub pointer_head: PointerHead,
    pub joint_circuit: JointCircuit,
    pub source_query: Param<Tensor<1>>,
    pub source_query_position: Linear,
    /// Distinguishes model-retained context from externally supplied observations.
    pub retained_markers: Param<Tensor<2>>,
    hidden: usize,
    labels: usize,
    max_refinement_steps: usize,
    mixed: bool,
}

/// Everything derived from the document once per batch and reused at every refinement step.
#[derive(Debug, Clone)]
pub struct DocumentCache {
    /// `S1`'s projection of `H_early` and `S2`'s projection of `H_final`, `[B, heads, T, head_dim]` each.
    pub early_bank: SourceBank,
    pub final_bank: SourceBank,
    /// `U_v H_final`, `[B, T, h]`.
    pub evidence_values: Tensor<3>,
    /// `U_s H_final`, `U_e H_final`, `[B, T, h]`.
    pub evidence_start_keys: Tensor<3>,
    pub evidence_end_keys: Tensor<3>,
}

impl DocumentCache {
    /// Reuse attached source projections for one physical row without recomputing the encoder or banks.
    pub fn row_view(&self,row:usize)->Self {
        let bank=|source:&SourceBank|SourceBank {
            key:source.key.clone().slice([row..row+1]),value:source.value.clone().slice([row..row+1]),
        };
        Self {early_bank:bank(&self.early_bank),final_bank:bank(&self.final_bank),
            evidence_values:self.evidence_values.clone().slice([row..row+1]),
            evidence_start_keys:self.evidence_start_keys.clone().slice([row..row+1]),
            evidence_end_keys:self.evidence_end_keys.clone().slice([row..row+1])}
    }

    pub fn without_autodiff(&self) -> Self {
        Self {
            early_bank: self.early_bank.without_autodiff(),
            final_bank: self.final_bank.without_autodiff(),
            evidence_values: self.evidence_values.clone().inner(),
            evidence_start_keys: self.evidence_start_keys.clone().inner(),
            evidence_end_keys: self.evidence_end_keys.clone().inner(),
        }
    }
}

/// Result of a full inference episode.
#[derive(Debug, Clone)]
pub struct DenoiseOutput {
    /// State after the last update; learned remasking may retain partial field visibility.
    pub state: SoftState,
    /// Logits of the last step.
    pub last: StepOutput,
    /// Number of refinement steps run (0 when the batch has no slot).
    pub steps: usize,
}

impl AnnotationDenoiser {
    /// Extend only the learned horizon, preserving the original embedding ID and rows.
    pub fn grow_a4_rounds(mut self)->Self {
        assert_eq!(self.max_refinement_steps,8,"round growth requires an A4 denoiser");
        let value=self.step_embedding.weight.val();
        let last=value.clone().slice([7..8]);
        self.step_embedding.weight=Param::initialized(self.step_embedding.weight.id,
            Tensor::cat(vec![value,last.clone(),last],0).detach()).set_require_grad(true);
        self.max_refinement_steps=10;
        self
    }

    pub fn labels(&self) -> usize {
        self.labels
    }

    pub fn max_refinement_steps(&self) -> usize {
        self.max_refinement_steps
    }

    /// Learned relation queries depend only on document-local query IDs, never on an entity shortlist.
    pub fn source_pairs(&self, plan: &SlotPlan) -> PairFeatures {
        assert!(plan.is_source_query_plan(), "source query features require a source query plan");
        let (b, m, h) = (plan.batch(), plan.slots_per_row(), self.hidden);
        let device = self.source_query.val().device();
        let mut positions = vec![0.0; b * m * h];
        let mut valid = vec![0.0; b * m];
        debug_assert_eq!(plan.rows().len(),b);
        for (row, slots) in plan.rows().iter().enumerate() {
            debug_assert!(row<b&&slots.len()<=m);
            debug_assert!(slots.iter().all(|slot|slot.document<plan.documents_in_row(row)));
            let mut previous_document = None;
            let mut position = 0;
            for (slot, entry) in slots.iter().enumerate() {
                if previous_document != Some(entry.document) { position = 0; previous_document = Some(entry.document); }
                for (channel, value) in position_features(position, h).enumerate() {
                    positions[(row * m + slot) * h + channel] = value;
                }
                valid[row * m + slot] = 1.0;
                position += 1;
            }
        }
        let positions = crate::plan::float(positions, [b, m, h], &device);
        let valid = crate::plan::float(valid, [b, m, 1], &device);
        let pairs = (self.source_query_position.forward(positions) + self.source_query.val().reshape([1, 1, h]))
            .mask_fill(valid.lower_elem(0.5), 0.0);
        PairFeatures {entities: Tensor::empty([0, h], &device), pairs}
    }

    /// Names, bank stages and widths of `S1` and `S2`, in execution order (spec §15.2).
    pub fn sparse_global_attention_layers(&self) -> [SparseGlobalAttentionInfo; 2] {
        [self.sparse_global_attention_1.info(), self.sparse_global_attention_2.info()]
    }

    /// Projects both source banks and the evidence token features once. `h_early, h_final: [B, T, h]` are the
    /// stage-5 and stage-15 readouts; padding is removed before projection and zeroed afterwards.
    pub fn prepare(&self, h_early: Tensor<3>, h_final: Tensor<3>, layout: &BatchLayout) -> DocumentCache {
        let [batch,tokens,_]=h_early.dims();
        let invalid = layout.valid.clone().lower_elem(0.5);
        let invalid_bank=invalid.clone().reshape([batch,1,tokens,1]);
        let clean_bank=|bank:SourceBank|SourceBank {
            key:bank.key.mask_fill(invalid_bank.clone(),0.0),
            value:bank.value.mask_fill(invalid_bank.clone(),0.0),
        };
        let h_early = h_early.mask_fill(invalid.clone(), 0.0);
        let h_final = h_final.mask_fill(invalid.clone(), 0.0);
        DocumentCache {
            early_bank: clean_bank(self.sparse_global_attention_1.project_bank(h_early)),
            final_bank: clean_bank(self.sparse_global_attention_2.project_bank(h_final.clone())),
            evidence_values: project(&self.evidence_source, h_final.clone(), self.mixed).mask_fill(invalid.clone(),0.0),
            evidence_start_keys: project(&self.evidence_start_key, h_final.clone(), self.mixed).mask_fill(invalid.clone(),0.0),
            evidence_end_keys: project(&self.evidence_end_key, h_final, self.mixed).mask_fill(invalid,0.0),
        }
    }

    /// `q_s` for every slot, `[B, M, h]` (§6.3).
    pub fn queries(&self, pairs: &PairFeatures, cache: &DocumentCache, slots: &SlotTensors, state: &SoftState) -> Tensor<3> {
        self.queries_impl(pairs, cache, slots, state, None, None)
    }

    /// Ordinary pointer-mode query assembly, without model-retained context.
    pub fn queries_pointer(
        &self, pairs: &PairFeatures, cache: &DocumentCache, slots: &SlotTensors,
        state: &SoftState, pointers: &PointerState, bank: &SourcePointerContext,
    ) -> Tensor<3> {
        self.queries_impl(pairs, cache, slots, state, Some((pointers, bank)), None)
    }

    pub fn queries_pointer_retained(&self,pairs:&PairFeatures,cache:&DocumentCache,slots:&SlotTensors,
        state:&SoftState,pointers:&PointerState,bank:&SourcePointerContext,retained:&RetainedAnswer,masked:&Tensor<3>)->Tensor<3> {
        self.queries_impl(pairs,cache,slots,state,Some((pointers,bank)),Some((retained,masked)))
    }

    fn queries_impl(&self, pairs: &PairFeatures, cache: &DocumentCache, slots: &SlotTensors, state: &SoftState,
        pointers: Option<(&PointerState, &SourcePointerContext)>,retained:Option<(&RetainedAnswer,&Tensor<3>)>) -> Tensor<3> {
        let (b, m, h) = (slots.batch, slots.slots_per_row, self.hidden);
        let labels = self.labels;
        debug_assert_eq!(pairs.pairs.dims(),[b,m,h]);
        debug_assert_eq!(state.relation.dims(),[b,m,labels]);
        debug_assert_eq!(state.evidence_start.dims(),[b,m,slots.tokens]);
        debug_assert_eq!(state.evidence_end.dims(),[b,m,slots.tokens]);
        debug_assert_eq!(state.visibility.dims(),[b,m,crate::state::FIELDS]);
        debug_assert_eq!(state.step.dims(),[b,m]);
        debug_assert_eq!(slots.slot_valid.dims(),[b,m]);
        debug_assert_eq!(cache.evidence_values.dims(),[b,slots.tokens,h]);
        let invalid = slots.slot_valid.clone().lower_elem(0.5).unsqueeze_dim::<3>(2);
        let visible = |field: usize| state.visibility.clone().slice([0..b, 0..m, field..field + 1])
            .mask_fill(invalid.clone(), 0.0);
        let soft = |mask: &Param<Tensor<1>>, rho: Tensor<3>, expected: Tensor<3>| {
            let hidden = mask.val().reshape([1, 1, h]).expand([b, m, h])
                .mask_fill(rho.clone().equal_elem(1.0).expand([b, m, h]), 0.0)
                * rho.clone().neg().add_scalar(1.0);
            hidden + expected.mask_fill(rho.clone().equal_elem(0.0).expand([b, m, h]), 0.0) * rho
        };

        // Remove hidden/padded values before matmul: masking the product afterwards cannot undo
        // zero-times-NaN in either the forward reduction or its shared-parameter gradient.
        let relation_rho = visible(RELATION);
        let relation_expected = mixed_matmul(
            state.relation.clone().mask_fill(relation_rho.clone().equal_elem(0.0), 0.0)
                .reshape([b * m, labels]), self.relation_labels.val(), self.mixed,
        ).reshape([b, m, h]);
        let mut relation = soft(&self.relation_mask, relation_rho, relation_expected);
        let evidence_rho = visible(EVIDENCE);
        let hidden_evidence = evidence_rho.clone().equal_elem(0.0);
        let mut start = soft(
            &self.evidence_start_mask,
            evidence_rho.clone(),
            mixed_matmul(state.evidence_start.clone().mask_fill(hidden_evidence.clone(), 0.0), cache.evidence_values.clone(), self.mixed),
        );
        let mut end = soft(&self.evidence_end_mask, evidence_rho, mixed_matmul(
            state.evidence_end.clone().mask_fill(hidden_evidence, 0.0), cache.evidence_values.clone(), self.mixed,
        ));
        if let Some((retained,masked))=retained {
            let visible=|field:usize,action:usize|retained.filled.clone().slice([0..b,0..m,field..field+1])
                .bool_and(masked.clone().slice([0..b,0..m,action..action+1]).lower_elem(0.5)).bool_and(invalid.clone().bool_not());
            let marker=|field:usize|self.retained_markers.val().slice([field..field+1]).reshape([1,1,h]);
            let shown=visible(0,0);
            let labels=retained.value.clone().slice([0..b,0..m,0..1]).mask_fill(shown.clone().bool_not(),0).reshape([b*m]);
            let value=self.relation_labels.val().select(0,labels).reshape([b,m,h])+marker(0);
            relation=relation.mask_where(shown.expand([b,m,h]),value);
            let shown=visible(3,5);
            let rows=Tensor::<1,Int>::arange(0..b as i64,&pairs.pairs.device()).reshape([b,1]).mul_scalar(slots.tokens as i64);
            let evidence=|factor:usize| {
                let token=retained.value.clone().slice([0..b,0..m,factor..factor+1])
                    .mask_fill(shown.clone().bool_not(),0).reshape([b,m])+rows.clone();
                cache.evidence_values.clone().reshape([b*slots.tokens,h]).select(0,token.reshape([b*m])).reshape([b,m,h])+marker(3)
            };
            start=start.mask_where(shown.clone().expand([b,m,h]),evidence(9));
            end=end.mask_where(shown.clone().expand([b,m,h]),evidence(10));
        }
        let (subject, object) = match pointers {
            Some((state, bank)) if slots.source_queries => match retained {
                Some((retained,masked))=>self.pointer_head.fields_retained(state,bank,retained,masked,&self.retained_markers.val()),
                None=>self.pointer_head.fields(state,bank),
            },
            _ => {
                let (subject, object) = slots.gather_pair_entities(pairs.entities.clone());
                (subject.reshape([b, m, h]), object.reshape([b, m, h]))
            }
        };
        let subject = subject.mask_fill(invalid.clone(), 0.0);
        let object = object.mask_fill(invalid.clone(), 0.0);
        let step = self.step_embedding.forward(state.step.clone().mask_fill(
            slots.slot_valid.clone().lower_elem(0.5), 0));

        self.query_norm.forward(
            pairs.pairs.clone().mask_fill(invalid.clone(), 0.0)
                + project(&self.relation_field, relation, self.mixed)
                + project(&self.subject_field, subject, self.mixed)
                + project(&self.object_field, object, self.mixed)
                + project(&self.evidence_start_field, start, self.mixed)
                + project(&self.evidence_end_field, end, self.mixed)
                + step,
        ).mask_fill(invalid, 0.0)
    }

    /// One refinement step (§6.4): `S1` per chunk, one graph update over all slots, `S2` and the heads per chunk.
    /// With no slot (`M = 0`) neither layer runs and the outputs are empty, `[B, 0, R]` and `[B, 0, T]`.
    pub fn step(
        &self,
        pairs: &PairFeatures,
        cache: &DocumentCache,
        slots: &SlotTensors,
        state: &SoftState,
        slot_chunk: usize,
    ) -> StepOutput {
        self.step_impl(pairs, cache, slots, state, slot_chunk, None, None).0
    }

    pub fn step_pointer(&self, pairs: &PairFeatures, cache: &DocumentCache, slots: &SlotTensors,
        state: &SoftState, pointers: &PointerState, bank: &SourcePointerContext, slot_chunk: usize) -> PointerStepOutput {
        let (annotations, pointers, hidden) = self.step_impl(pairs, cache, slots, state, slot_chunk, Some((pointers, bank)), None);
        PointerStepOutput { annotations, pointers: pointers.expect("pointer heads requested"), hidden: hidden.expect("pointer hidden requested") }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_pointer_retained(&self,pairs:&PairFeatures,cache:&DocumentCache,slots:&SlotTensors,state:&SoftState,
        pointers:&PointerState,bank:&SourcePointerContext,retained:&RetainedAnswer,masked:&Tensor<3>,slot_chunk:usize)->PointerStepOutput {
        let (annotations,pointers,hidden)=self.step_impl(pairs,cache,slots,state,slot_chunk,Some((pointers,bank)),Some((retained,masked)));
        PointerStepOutput {annotations,pointers:pointers.expect("pointer heads requested"),hidden:hidden.expect("pointer hidden requested")}
    }

    fn step_impl(&self, pairs: &PairFeatures, cache: &DocumentCache, slots: &SlotTensors,
        state: &SoftState, slot_chunk: usize, pointers: Option<(&PointerState, &SourcePointerContext)>,retained:Option<(&RetainedAnswer,&Tensor<3>)>)
        -> (StepOutput, Option<PointerLogits>, Option<Tensor<3>>) {
        assert!(slot_chunk > 0, "slot_chunk must be positive");
        let _profile = dre_core::profile::component("denoiser_forward", || cache.evidence_values.device());
        let (b, m, t) = (slots.batch, slots.slots_per_row, slots.tokens);
        debug_assert_eq!(slots.slot_first.dims(),[b,m]);
        debug_assert_eq!(slots.slot_last.dims(),[b,m]);
        debug_assert_eq!(slots.pair_entities.dims(),[b*m,2]);
        debug_assert_eq!(cache.early_bank.key.dims()[0],b);
        debug_assert_eq!(cache.early_bank.key.dims()[2],t);
        debug_assert_eq!(cache.final_bank.key.dims()[0],b);
        debug_assert_eq!(cache.final_bank.key.dims()[2],t);
        if m == 0 {
            let output = self.empty_output(b, t, &cache.evidence_values.device());
            let pointer = pointers.map(|(_, context)| PointerLogits::empty(b, 0, context));
            let hidden = pointers.map(|_| Tensor::empty([b, 0, self.hidden], &cache.evidence_values.device()));
            return (output, pointer, hidden);
        }
        let step_profile = dre_core::profile::exp004("denoiser_step", &pairs.pairs)
            .map(|profile| profile.dimensions(t, m, slots.graph_edges.dims()[0]));
        let chunks: Vec<std::ops::Range<usize>> =
            (0..m).step_by(slot_chunk).map(|start| start..(start + slot_chunk).min(m)).collect();
        let outside: Vec<Tensor<3, Bool>> = chunks.iter().map(|range| slots.outside_document(range.clone())).collect();

        let query_profile = dre_core::profile::component("denoiser_queries", || cache.evidence_values.device());
        let query_profile_exp004 = dre_core::profile::exp004("denoiser_query_fields", &pairs.pairs)
            .map(|profile| profile.dimensions(t, m, slots.graph_edges.dims()[0]));
        let q = self.queries_impl(pairs, cache, slots, state, pointers, retained);
        if let Some(profile) = query_profile_exp004 { profile.finish(&q); }
        drop(query_profile);
        let attention_profile = dre_core::profile::component("sparse_attention_early", || cache.evidence_values.device());
        let a = Tensor::cat(
            chunks
                .iter()
                .zip(&outside)
                .map(|(range, outside)| {
                    self.sparse_global_attention_1.forward(
                        q.clone().slice([0..b, range.clone()]),
                        &cache.early_bank,
                        outside.clone(),
                    )
                })
                .collect(),
            1,
        );
        drop(attention_profile);
        let invalid = slots.slot_valid.clone().lower_elem(0.5).unsqueeze_dim::<3>(2);
        let a = self.graph.forward(a.mask_fill(invalid.clone(),0.0), slots).mask_fill(invalid,0.0);

        let scale = 1.0 / (self.hidden as f64).sqrt();
        let mut relation = Vec::with_capacity(chunks.len());
        let mut evidence_start = Vec::with_capacity(chunks.len());
        let mut evidence_end = Vec::with_capacity(chunks.len());
        let mut pointer_chunks = Vec::new();
        let mut hidden_chunks = Vec::new();
        for (range, outside) in chunks.iter().zip(outside) {
            let attention_profile = dre_core::profile::component("sparse_attention_final", || cache.evidence_values.device());
            let b_chunk = self.sparse_global_attention_2.forward(
                a.clone().slice([0..b, range.clone()]),
                &cache.final_bank,
                outside.clone(),
            );
            drop(attention_profile);
            let _head_profile = dre_core::profile::component("pointer_relation_evidence_heads", || cache.evidence_values.device());
            let invalid = slots.slot_valid.clone().slice([0..b, range.clone()]).lower_elem(0.5).unsqueeze_dim::<3>(2);
            let normed = self.head_norm.forward(b_chunk.mask_fill(invalid.clone(), 0.0))
                .mask_fill(invalid.clone(), 0.0);
            if let Some((_, bank)) = pointers {
                pointer_chunks.push(self.pointer_head.logits(normed.clone(), bank, range.clone()));
                hidden_chunks.push(normed.clone());
            }
            relation.push(project(&self.relation_head, normed.clone(), self.mixed).mask_fill(invalid.clone(), 0.0));
            let boundary = |query: &Linear, keys: &Tensor<3>| {
                mixed_matmul(
                    project(query, normed.clone(), self.mixed), keys.clone().swap_dims(1, 2), self.mixed,
                )
                    .mul_scalar(scale)
                    .mask_fill(outside.clone(), f32::NEG_INFINITY)
                    .mask_fill(invalid.clone(), 0.0)
            };
            evidence_start.push(boundary(&self.evidence_start_query, &cache.evidence_start_keys));
            evidence_end.push(boundary(&self.evidence_end_query, &cache.evidence_end_keys));
        }
        let pointers = pointers.map(|_| PointerLogits::cat(pointer_chunks));
        let hidden = pointers.as_ref().map(|_| Tensor::cat(hidden_chunks, 1));
        let annotations = StepOutput {
            relation: Tensor::cat(relation, 1),
            evidence_start: Tensor::cat(evidence_start, 1),
            evidence_end: Tensor::cat(evidence_end, 1),
        };
        if let Some(profile) = step_profile {
            let annotation_outputs = [&annotations.relation, &annotations.evidence_start, &annotations.evidence_end];
            let pointer_outputs = pointers.as_ref().into_iter()
                .flat_map(|pointers| pointers.token.iter().chain(pointers.byte.iter()).chain(pointers.byte_queries.iter()));
            profile.finish_many(annotation_outputs.into_iter().chain(pointer_outputs));
        }
        (annotations, pointers, hidden)
    }

    /// Logits of a batch without slots: `[B, 0, R]`, `[B, 0, T]`, `[B, 0, T]`.
    fn empty_output(&self, batch: usize, tokens: usize, device: &Device) -> StepOutput {
        StepOutput {
            relation: Tensor::empty([batch, 0, self.labels], device),
            evidence_start: Tensor::empty([batch, 0, tokens], device),
            evidence_end: Tensor::empty([batch, 0, tokens], device),
        }
    }

    /// Inference episode: `schedule.steps` refinement steps from the prior state (§6.5, §9). Takes no gold input.
    /// A batch without slots (`M = 0`) runs no step and returns its empty initial state and empty logits.
    pub fn run(
        &self,
        plan: &SlotPlan,
        pairs: &PairFeatures,
        cache: &DocumentCache,
        slots: &SlotTensors,
        schedule: &DenoiseSchedule,
        relation_prior: Option<&[f32]>,
    ) -> DenoiseOutput {
        self.check_schedule(schedule);
        let device = pairs.pairs.device();
        let mut state = SoftState::initial(plan, slots, self.labels, relation_prior, &device);
        if slots.slots_per_row == 0 {
            let last = self.empty_output(slots.batch, slots.tokens, &device);
            return DenoiseOutput { state, last, steps: 0 };
        }
        let mut last = None;
        for _ in 0..schedule.steps {
            let output = self.step(pairs, cache, slots, &state, schedule.slot_chunk);
            state = state.update(&output, schedule);
            last = Some(output);
        }
        DenoiseOutput { state, last: last.expect("at least one step"), steps: schedule.steps }
    }

    /// Panics unless `1 <= steps <= max_refinement_steps`.
    pub fn check_schedule(&self, schedule: &DenoiseSchedule) {
        assert!(
            (1..=self.max_refinement_steps).contains(&schedule.steps),
            "a denoising episode runs 1..={} steps, {} requested",
            self.max_refinement_steps,
            schedule.steps
        );
        assert!(schedule.eta > 0.0 && schedule.eta <= 1.0, "eta must be in (0, 1]");
        assert!(schedule.temperature > 0.0, "temperature must be positive");
    }
}
