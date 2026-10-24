//! Relation/evidence annotation state shared by supplied pairs and source-query extraction.
//!
//! Fields per slot, in the visibility order of [`FIELDS`]:
//! - supplied subject/object entity IDs: observed constants; source-query slots leave these IDs absent;
//! - relation label: masked, `Z_rel: [B, M, R]` on the probability simplex;
//! - evidence span: masked, start and end distributions `Z_start, Z_end: [B, M, T]` over the slot's own
//!   document positions (zero elsewhere), sharing one visibility.
//!
//! Normal update before learned visibility resets (§6.5), with mixing `eta` and gain `delta = 1 / K_diff`:
//!
//! ```text
//! Ẑ_s = softmax(ℓ_s / τ),   Z_{s+1} = (1 − η) Z_s + η Ẑ_s,   rho_{s+1} = clip(rho_s + δ, 0, 1)
//! ```
//!
//! Observed fields use `η = δ = 0`. Nothing here touches source text or token ids.
//! EXP-004's four source-token boundaries and fixed observed byte offsets live in [`crate::pointer::PointerState`].
//! Its observation/progress columns are separate from this relation/evidence overlay.
//! Production joint-query refinement uses circuit conditional marginals for relation and token feedback;
//! each marginal excludes the field's own retained value. Evidence remains an independent head.

use burn::prelude::*;

use crate::plan::{SlotPlan, SlotTensors, float, int};

/// Visibility column of each field in `rho: [B, M, FIELDS]`.
pub const SUBJECT: usize = 0;
pub const RELATION: usize = 1;
pub const OBJECT: usize = 2;
pub const EVIDENCE: usize = 3;
pub const FIELDS: usize = 4;

/// `1` for fields the denoiser fills (masked in provided-entity mode), `0` for observed fields.
pub const MASKED_FIELDS: [f32; FIELDS] = [0.0, 1.0, 0.0, 1.0];

/// Refinement schedule of one denoising episode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenoiseSchedule {
    /// `K_diff`: refinement steps (production default 8; never more than the denoiser's step table).
    pub steps: usize,
    /// `η`: fixed mixing weight of a new prediction into the state.
    pub eta: f64,
    /// `τ`: softmax temperature of the field logits.
    pub temperature: f64,
    /// Slots per chunk for the two reads and the heads; the graph always sees all slots.
    pub slot_chunk: usize,
}

impl DenoiseSchedule {
    /// `δ`: visibility gained per step, so masked fields become fully visible after `steps` updates.
    pub fn visibility_step(&self) -> f64 {
        1.0 / self.steps as f64
    }

    /// `rho` of masked fields at the input of step `s`.
    pub fn visibility_at(&self, step: usize) -> f64 {
        (step as f64 * self.visibility_step()).clamp(0.0, 1.0)
    }
}

/// Field logits of one refinement step.
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// `[B, M, R]`; exactly zero at padded slots.
    pub relation: Tensor<3>,
    /// `[B, M, T]` each; `-inf` outside the slot's document, exactly zero at padded slots.
    pub evidence_start: Tensor<3>,
    pub evidence_end: Tensor<3>,
}

impl StepOutput {
    pub fn slot_view(&self, row: usize, range: std::ops::Range<usize>) -> Self {
        Self {relation: self.relation.clone().slice([row..row + 1, range.clone()]),
            evidence_start: self.evidence_start.clone().slice([row..row + 1, range.clone()]),
            evidence_end: self.evidence_end.clone().slice([row..row + 1, range])}
    }
}

/// The annotation state `(Z, rho)` of every slot at the input of a refinement step.
#[derive(Debug, Clone)]
pub struct SoftState {
    /// `Z_rel`, `[B, M, R]`.
    pub relation: Tensor<3>,
    /// `Z_start`, `Z_end`, `[B, M, T]`.
    pub evidence_start: Tensor<3>,
    pub evidence_end: Tensor<3>,
    /// `rho`, `[B, M, FIELDS]`.
    pub visibility: Tensor<3>,
    /// Supplied entity IDs, `[B,M]`; `-1` for source-query slots and padded slots.
    pub subject: Tensor<2, Int>,
    pub object: Tensor<2, Int>,
    /// Refinement step index `s` of each slot, `[B, M]`.
    pub step: Tensor<2, Int>,
}

/// Host values of a relation prior and the document-uniform evidence prior, shared by the initial state and
/// the training corruption.
pub(crate) struct HostPriors {
    pub relation: Vec<f32>,
    /// `[B·M·T]`: uniform over each slot's document positions (over the whole row for padded slots).
    pub evidence: Vec<f32>,
}

impl HostPriors {
    pub fn new(plan: &SlotPlan, labels: usize, relation_prior: Option<&[f32]>) -> Self {
        let relation = match relation_prior {
            Some(prior) => {
                assert_eq!(prior.len(), labels, "relation prior length");
                let total: f32 = prior.iter().sum();
                assert!(prior.iter().all(|&p| p >= 0.0) && total > 0.0, "relation prior must be a distribution");
                prior.iter().map(|p| p / total).collect()
            }
            None => vec![1.0 / labels as f32; labels],
        };
        let (b, m, t) = (plan.batch(), plan.slots_per_row(), plan.tokens());
        let mut evidence = vec![1.0 / t as f32; b * m * t];
        for (row, slots) in plan.rows().iter().enumerate() {
            for (k, slot) in slots.iter().enumerate() {
                let (first, last) = plan.document_range(row, slot.document);
                let weight = 1.0 / (last - first + 1) as f32;
                let base = (row * m + k) * t;
                for position in 0..t {
                    evidence[base + position] = if (first..=last).contains(&position) { weight } else { 0.0 };
                }
            }
        }
        Self { relation, evidence }
    }
}

impl SoftState {
    /// Inference start (spec §6.5): masked fields hold the prior (uniform labels unless `relation_prior` is
    /// given; uniform over the document for evidence) with `rho = 0`; observed pointers have `rho = 1`.
    /// No gold annotation enters this state.
    pub fn initial(
        plan: &SlotPlan,
        slots: &SlotTensors,
        labels: usize,
        relation_prior: Option<&[f32]>,
        device: &Device,
    ) -> Self {
        let priors = HostPriors::new(plan, labels, relation_prior);
        let (b, m, t) = (plan.batch(), plan.slots_per_row(), plan.tokens());
        let relation: Vec<f32> = priors.relation.iter().copied().cycle().take(b * m * labels).collect();
        let visibility: Vec<f32> = MASKED_FIELDS.iter().map(|&masked| 1.0 - masked).cycle().take(b * m * FIELDS).collect();
        let evidence = float(priors.evidence, [b, m, t], device);
        Self {
            relation: float(relation, [b, m, labels], device),
            evidence_start: evidence.clone(),
            evidence_end: evidence,
            visibility: float(visibility, [b, m, FIELDS], device),
            subject: slots.subject_pointer.clone(),
            object: slots.object_pointer.clone(),
            step: int(vec![0; b * m], [b, m], device),
        }
    }

    /// One §6.5 update from the step's logits.
    pub fn update(&self, output: &StepOutput, schedule: &DenoiseSchedule) -> Self {
        crate::diffusion_scheduler::normal_update(self, output, schedule)
    }

    pub fn slot_view(&self, row: usize, range: std::ops::Range<usize>) -> Self {
        Self {
            relation: self.relation.clone().slice([row..row + 1, range.clone()]),
            evidence_start: self.evidence_start.clone().slice([row..row + 1, range.clone()]),
            evidence_end: self.evidence_end.clone().slice([row..row + 1, range.clone()]),
            visibility: self.visibility.clone().slice([row..row + 1, range.clone()]),
            subject: self.subject.clone().slice([row..row + 1, range.clone()]),
            object: self.object.clone().slice([row..row + 1, range.clone()]),
            step: self.step.clone().slice([row..row + 1, range]),
        }
    }

    pub fn without_autodiff(&self) -> Self {
        Self {
            relation: self.relation.clone().inner(),
            evidence_start: self.evidence_start.clone().inner(),
            evidence_end: self.evidence_end.clone().inner(),
            visibility: self.visibility.clone().inner(),
            subject: self.subject.clone().inner(),
            object: self.object.clone().inner(),
            step: self.step.clone().inner(),
        }
    }

    /// Moves every tensor back onto the autodiff device as constants (no gradient flows into their past).
    pub fn with_autodiff(&self) -> Self {
        Self {
            relation: self.relation.clone().autodiff(),
            evidence_start: self.evidence_start.clone().autodiff(),
            evidence_end: self.evidence_end.clone().autodiff(),
            visibility: self.visibility.clone().autodiff(),
            subject: self.subject.clone().autodiff(),
            object: self.object.clone().autodiff(),
            step: self.step.clone().autodiff(),
        }
    }
}
