//! Exact five-field annotation distribution (EXP-004-A4).
//!
//! Source-dependent leaf tables are shared across calls. Leaf normalization and marginal
//! evaluation use batched matrix products; no query × position × state tensor is built.
//! Chain messages stay in log space, including likelihoods of complete gold tuples.
//! Prepare each evidence set once: decision-only preparation omits autodiff, whereas
//! conditional feedback retains it. Dependency readout evaluates only the proposed tuple.
//! Coupling selects existing call tensors, never repeating projections or leaf normalizers.
//! Undefined zero-mass conditionals remain zero-mass, not uniform. Their free-field
//! decode choice is coordinate zero, and undefined dependency features are zero.
use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{DType, IndexingUpdateOp, activation::log_softmax};
use crate::plan::{boolean, int};
use crate::pointer::SourcePointerContext;

pub const CIRCUIT_FIELDS: usize = 5;
pub const CIRCUIT_FEATURES: usize = 6;

#[derive(Config, Debug)]
pub struct JointCircuitConfig {
    pub hidden: usize,
    pub relations: usize,
    #[config(default = 32)]
    pub states: usize,
    #[config(default = 8.0)]
    pub leaf_bound: f64,
}

impl JointCircuitConfig {
    pub fn init(&self, device: &Device) -> JointCircuit {
        assert!(self.hidden > 0 && self.relations > 0 && self.states > 0);
        assert!(self.leaf_bound.is_finite() && self.leaf_bound > 0.0 && self.leaf_bound <= 8.0,
            "circuit leaf bound must be in (0, 8] for FP32 normalizers");
        let small = Initializer::Normal { mean: 0.0, std: 0.01 / (self.hidden as f64).sqrt() };
        JointCircuit {
            relation_table: Initializer::Normal { mean: 0.0, std: 0.01 / (self.relations as f64).sqrt() }
                .init([self.relations, self.states], device),
            endpoints: std::array::from_fn(|_| LinearConfig::new(self.hidden, self.states)
                .with_initializer(small.clone()).init(device)),
            initial_logits: Initializer::Zeros.init([self.states], device),
            transition_logits: Initializer::Zeros.init([4, self.states, self.states], device),
            gates: std::array::from_fn(|_| LinearConfig::new(self.hidden, self.states)
                .with_initializer(Initializer::Zeros).init(device)),
            states: self.states,
            leaf_bound: self.leaf_bound,
        }
    }
}

#[derive(Module, Debug)]
pub struct JointCircuit {
    /// Contiguous value-major relation terms, `[R,K]`; no transposed view enters elementwise kernels.
    pub relation_table: Param<Tensor<2>>,
    pub endpoints: [Linear; 4],
    pub initial_logits: Param<Tensor<1>>,
    pub transition_logits: Param<Tensor<3>>,
    pub gates: [Linear; CIRCUIT_FIELDS],
    states: usize,
    leaf_bound: f64,
}

#[derive(Debug, Clone)]
struct LeafTable {
    /// Source rows × values × K; relation tables have one shared row.
    shifted: Tensor<3>,
    exponential: Tensor<3>,
}


impl LeafTable {
    /// One `[count, K]` term per gold alternative: `row` names the source row of its query
    /// (ignored for the shared relation table), `value` its already-validated coordinate.
    fn selected_rows(&self, row: Tensor<1, Int>, value: Tensor<1, Int>, detached: bool) -> Tensor<2> {
        let [rows, positions, states] = self.shifted.dims();
        let flat = computation(&self.shifted, detached).reshape([rows * positions, states]);
        let index = if rows == 1 { value } else { row.mul_scalar(positions as i32) + value };
        flat.select(0, index)
    }

    /// Callers remove unknown, missing and out-of-axis coordinates before this selection.
    fn selected(&self, values: Tensor<3, Int>, detached: bool) -> Tensor<3> {
        let [batch, queries, _] = values.dims();
        let [rows, positions, states] = self.shifted.dims();
        let shifted = computation(&self.shifted, detached);
        if rows == 1 {
            // Relation terms are shared across the batch. Select contiguous [R,K]
            // rows directly instead of gathering through a zero-stride expanded view.
            shifted.reshape([positions, states]).select(0, values.reshape([batch * queries]))
                .reshape([batch, queries, states])
        } else {
            shifted.gather(1, values.expand([batch, queries, states]))
        }
    }
}
#[derive(Debug, Clone)]
pub struct EndpointTables {
    tables: [LeafTable; 4],
    outside: Tensor<3, Bool>,
}

/// Evidence values are integer field coordinates in order R, SS, SE, OS, OE.
/// Unknown coordinates are removed before gathers; they need not name legal values.
#[derive(Debug, Clone)]
pub struct CircuitEvidence {
    pub values: Tensor<3, Int>,
    pub known: Tensor<3, Bool>,
}

impl CircuitEvidence {
    pub fn from_host(values: Vec<i64>, known: Vec<bool>, batch: usize, queries: usize, device: &Device) -> Self {
        assert_eq!(values.len(), batch * queries * CIRCUIT_FIELDS);
        assert_eq!(known.len(), values.len());
        let values = values.into_iter().zip(&known).map(|(value, &known)| {
            if known { assert!(value >= 0); i32::try_from(value).expect("circuit evidence fits i32") } else { 0 }
        }).collect();
        Self { values: int(values, [batch, queries, CIRCUIT_FIELDS], device),
            known: boolean(known, [batch, queries, CIRCUIT_FIELDS], device) }
    }

    pub fn empty(batch: usize, queries: usize, device: &Device) -> Self {
        Self::from_host(vec![0; batch * queries * CIRCUIT_FIELDS],
            vec![false; batch * queries * CIRCUIT_FIELDS], batch, queries, device)
    }

    pub fn slot_view(&self, row: usize, range: std::ops::Range<usize>) -> Self {
        Self { values: self.values.clone().slice([row..row + 1, range.clone()]),
            known: self.known.clone().slice([row..row + 1, range]) }
    }

    pub fn cat_rows(rows: &[Self]) -> Self {
        assert!(!rows.is_empty());
        Self { values: Tensor::cat(rows.iter().map(|row| row.values.clone()).collect(), 0),
            known: Tensor::cat(rows.iter().map(|row| row.known.clone()).collect(), 0) }
    }
}

/// The caller removes unknown queries and alternatives inconsistent with supplied evidence
/// on the host. Groups are dense, nonempty, and each group belongs to exactly one flat B*M query.
#[derive(Debug, Clone)]
pub struct GoldCombinations {
    pub query: Tensor<1, Int>,
    pub values: Tensor<2, Int>,
    pub group: Tensor<1, Int>,
    pub groups: usize,
    /// Weight of each group's term in the summed loss (one by default).
    pub group_weight: Tensor<1>,
    host_query: std::sync::Arc<Vec<i64>>,
}

impl GoldCombinations {
    pub fn from_host(query: Vec<i64>, values: Vec<i64>, group: Vec<i64>, groups: usize, device: &Device) -> Self {
        Self::from_host_weighted(query, values, group, groups, vec![1.0; groups], device)
    }

    /// As [`Self::from_host`], with an explicit per-group weight in the summed loss.
    pub fn from_host_weighted(query: Vec<i64>, values: Vec<i64>, group: Vec<i64>, groups: usize,
        group_weight: Vec<f32>, device: &Device) -> Self {
        assert_eq!(group_weight.len(), groups);
        assert!(group_weight.iter().all(|weight| weight.is_finite() && *weight >= 0.0));
        let count = query.len();
        assert_eq!(values.len(), count * CIRCUIT_FIELDS);
        assert_eq!(group.len(), count);
        assert_eq!(count == 0, groups == 0);
        let mut owners = vec![None; groups];
        let mut tuples = std::collections::BTreeSet::new();
        for index in 0..count {
            assert!(query[index] >= 0 && group[index] >= 0 && (group[index] as usize) < groups);
            let owner = &mut owners[group[index] as usize];
            if let Some(owner) = owner { assert_eq!(*owner, query[index], "one query per gold group"); }
            else { *owner = Some(query[index]); }
            let tuple = &values[index * CIRCUIT_FIELDS..(index + 1) * CIRCUIT_FIELDS];
            assert!(tuple.iter().all(|&value| value >= 0));
            assert!(tuples.insert((group[index], tuple.to_vec())), "gold alternatives must be a set");
        }
        assert!(owners.iter().all(Option::is_some), "gold groups must be dense and nonempty");
        assert_eq!(owners.iter().flatten().collect::<std::collections::BTreeSet<_>>().len(), groups,
            "a known query occurs in exactly one gold group");
        let indices = |values: Vec<i64>| values.into_iter().map(|value| i32::try_from(value).expect("circuit index fits i32")).collect();
        Self { host_query: std::sync::Arc::new(query.clone()),
            query: int(indices(query), [count], device), values: int(indices(values), [count, CIRCUIT_FIELDS], device),
            group: int(indices(group), [count], device), groups,
            group_weight: crate::plan::float(group_weight, [groups], device) }
    }
}

#[derive(Debug, Clone)]
struct Leaf {
    log_probability: Tensor<3>,
    table: LeafTable,
    log_normalizer: Tensor<3>,
    coordinates: Tensor<3, Int>,
}

#[derive(Debug, Clone)]
pub struct CircuitCall {
    leaves: [Leaf; CIRCUIT_FIELDS],
    initial: Tensor<3>,
    transitions: [Tensor<4>; 4],
    missing: Tensor<3, Bool>,
}

/// Messages for one evidence set. Reuse this value for proposal and dependency
/// readouts; feedback with different retained evidence gets its own preparation.
#[derive(Debug)]
pub struct PreparedCircuit<'a> {
    call: &'a CircuitCall,
    evidence: CircuitEvidence,
    emissions: [Tensor<3>; CIRCUIT_FIELDS],
    forward: [Tensor<3>; CIRCUIT_FIELDS],
    backward: [Tensor<3>; CIRCUIT_FIELDS],
    detached: bool,
}

fn computation<const D: usize>(value: &Tensor<D>, detached: bool) -> Tensor<D> {
    if detached { value.clone().inner() } else { value.clone() }
}

fn logsumexp<const D: usize>(values: Tensor<D>, dim: usize) -> Tensor<D> {
    let maximum = values.clone().max_dim(dim);
    let absent = maximum.clone().equal_elem(f32::NEG_INFINITY);
    let maximum = maximum.mask_fill(absent.clone(), 0.0);
    let total = (values - maximum.clone()).exp().sum_dim(dim).mask_fill(absent.clone(), 1.0);
    (total.log() + maximum).mask_fill(absent, f32::NEG_INFINITY)
}

fn outside_axis<const D: usize>(indices: &Tensor<D, Int>, length: usize) -> Tensor<D, Bool> {
    indices.clone().lower_elem(0).bool_or(indices.clone().greater_equal_elem(length as i64))
}

/// A zero-mass row stays zero-mass, without evaluating -inf - -inf.
fn normalized_log_weights<const D: usize>(values: Tensor<D>, dim: usize) -> Tensor<D> {
    let dims = values.dims();
    let total = logsumexp(values.clone(), dim);
    let absent = total.clone().equal_elem(f32::NEG_INFINITY);
    let result = values.mask_fill(absent.clone().expand(dims), 0.0) - total.mask_fill(absent.clone(), 0.0);
    result.mask_fill(absent.expand(dims), f32::NEG_INFINITY)
}

fn log_softmax_with_zero_mass<const D: usize>(values: Tensor<D>, dim: usize) -> Tensor<D> {
    let absent = values.clone().max_dim(dim).equal_elem(f32::NEG_INFINITY).expand(values.dims());
    log_softmax(values.mask_fill(absent.clone(), 0.0), dim).mask_fill(absent, f32::NEG_INFINITY)
}

fn conditional_log_mass<const D: usize>(numerator: Tensor<D>, denominator: Tensor<D>) -> Tensor<D> {
    let absent = denominator.clone().equal_elem(f32::NEG_INFINITY);
    (numerator.mask_fill(absent.clone(), 0.0) - denominator.mask_fill(absent.clone(), 0.0))
        .mask_fill(absent, f32::NEG_INFINITY)
}

impl JointCircuit {
    pub fn endpoint_tables(&self, context: &SourcePointerContext) -> EndpointTables {
        let [batch, positions, _] = context.values.dims();
        // Metadata, not a device reduction: physical padding is removed before any projection.
        let missing = boolean((0..batch).flat_map(|row| (0..positions).map(move |position|
            position != context.none && context.token_offsets[row][position].is_none())).collect(),
            [batch, positions, 1], &context.values.device());
        self.make_endpoint_tables(context.values.clone(), missing, context.outside.clone())
    }

    fn make_endpoint_tables(&self, values: Tensor<3>, missing: Tensor<3, Bool>, outside: Tensor<3, Bool>) -> EndpointTables {
        let values = values.cast(DType::F32).mask_fill(missing.clone(), 0.0);
        let tables = std::array::from_fn(|field| {
            let terms = (self.endpoints[field].forward(values.clone()) / self.leaf_bound).tanh() * self.leaf_bound;
            let maximum = terms.clone().mask_fill(missing.clone(), f32::NEG_INFINITY).max_dim(1);
            let shifted = (terms - maximum).mask_fill(missing.clone(), f32::NEG_INFINITY);
            LeafTable { exponential: shifted.clone().exp(), shifted }
        });
        EndpointTables { tables, outside }
    }

    pub fn call(&self, hidden: Tensor<3>, relation_logits: Tensor<3>, token_logits: &[Tensor<3>; 4],
        tables: &EndpointTables, valid: &Tensor<2>) -> CircuitCall {
        let [batch, queries, _] = hidden.dims();
        let device = hidden.device();
        assert_eq!(valid.dims(), [batch, queries]);
        assert_eq!(relation_logits.dims()[..2], [batch, queries]);
        assert_eq!(tables.outside.dims()[..2], [batch, queries]);
        let missing = valid.clone().lower_elem(0.5).unsqueeze_dim::<3>(2);
        let hidden = hidden.cast(DType::F32).mask_fill(missing.clone(), 0.0);
        let initial = log_softmax_with_zero_mass(self.initial_logits.val().reshape([1, 1, self.states])
            + self.gates[0].forward(hidden.clone()), 2);
        let transitions = std::array::from_fn(|edge| {
            let base = self.transition_logits.val().slice([edge..edge + 1])
                .reshape([1, 1, self.states, self.states]).expand([batch, queries, self.states, self.states]);
            let gate = self.gates[edge + 1].forward(hidden.clone()).unsqueeze_dim::<4>(2)
                .expand([batch, queries, self.states, self.states]);
            log_softmax_with_zero_mass(base + gate, 3)
        });
        let relation = (self.relation_table.val() / self.leaf_bound).tanh() * self.leaf_bound;
        let relation = relation.clone() - relation.max_dim(0);
        let relation = LeafTable { shifted: relation.clone().unsqueeze_dim::<3>(0),
            exponential: relation.exp().unsqueeze_dim::<3>(0) };
        let leaves = std::array::from_fn(|field| {
            let (logits, table) = if field == 0 { (relation_logits.clone(), relation.clone()) }
                else { (token_logits[field - 1].clone(), tables.tables[field - 1].clone()) };
            let classes = logits.dims()[2];
            let logits = logits.cast(DType::F32).mask_fill(missing.clone(), 0.0);
            let logits = if field == 0 { logits } else { logits.mask_fill(tables.outside.clone(), f32::NEG_INFINITY) };
            let log_probability = log_softmax_with_zero_mass(logits, 2);
            let normalizer = log_probability.clone().exp().matmul(table.exponential.clone());
            // A leaf with no legal mass is identically zero. Its unused divisor is one,
            // selected before log so its messages cannot evaluate -inf - -inf.
            let log_normalizer = normalizer.clone().mask_fill(normalizer.equal_elem(0.0), 1.0).log();
            let coordinates = int((0..classes).map(|value| i32::try_from(value).expect("field axis fits i32")).collect(),
                [1, 1, classes], &device);
            Leaf { log_probability, table, log_normalizer, coordinates }
        });
        CircuitCall { leaves, initial, transitions, missing }
    }
}

impl CircuitCall {
    fn shape(&self) -> (usize, usize, usize) {
        let [batch, queries, states] = self.initial.dims();
        (batch, queries, states)
    }

    fn evidence(&self, evidence: &CircuitEvidence) -> CircuitEvidence {
        let (batch, queries, _) = self.shape();
        assert_eq!(evidence.values.dims(), [batch, queries, CIRCUIT_FIELDS]);
        assert_eq!(evidence.known.dims(), evidence.values.dims());
        let known = evidence.known.clone().bool_and(self.missing.clone().bool_not());
        CircuitEvidence { values: evidence.values.clone().mask_fill(known.clone().bool_not(), 0), known }
    }

    fn emissions_using(&self, evidence: &CircuitEvidence, detached: bool) -> [Tensor<3>; CIRCUIT_FIELDS] {
        let (batch, queries, _) = self.shape();
        std::array::from_fn(|field| {
            let leaf = &self.leaves[field];
            let unknown = evidence.known.clone().slice([0..batch, 0..queries, field..field + 1]).bool_not();
            let value = evidence.values.clone().slice([0..batch, 0..queries, field..field + 1]);
            let outside = outside_axis(&value, leaf.log_probability.dims()[2]);
            let unused = unknown.clone().bool_or(outside.clone());
            let value = value.mask_fill(unused.clone(), 0);
            // Unknown and out-of-range coordinates are removed BEFORE either gather.
            // Known out-of-range coordinates describe impossible evidence, not value zero.
            (computation(&leaf.log_probability, detached).gather(2, value.clone()).mask_fill(unused.clone(), 0.0)
                + leaf.table.selected(value, detached).mask_fill(unused.clone(), 0.0)
                - computation(&leaf.log_normalizer, detached).mask_fill(unused, 0.0))
                .mask_fill(outside.bool_and(unknown.bool_not()), f32::NEG_INFINITY)
        })
    }

    fn messages_using(&self, emissions: &[Tensor<3>; CIRCUIT_FIELDS], detached: bool) -> ([Tensor<3>; CIRCUIT_FIELDS], [Tensor<3>; CIRCUIT_FIELDS]) {
        let (batch, queries, states) = self.shape();
        let mut forward = Vec::with_capacity(CIRCUIT_FIELDS);
        forward.push(computation(&self.initial, detached));
        for field in 0..4 {
            forward.push(logsumexp((forward[field].clone() + emissions[field].clone()).unsqueeze_dim::<4>(3)
                .expand([batch, queries, states, states]) + computation(&self.transitions[field], detached), 2)
                .reshape([batch, queries, states]));
        }
        (forward.try_into().unwrap(), self.backward_messages_using(emissions, detached))
    }

    fn backward_messages_using(&self, emissions: &[Tensor<3>; CIRCUIT_FIELDS], detached: bool) -> [Tensor<3>; CIRCUIT_FIELDS] {
        let (batch, queries, states) = self.shape();
        let device = computation(&self.initial, detached).device();
        let mut backward = vec![Tensor::zeros([batch, queries, states], &device); CIRCUIT_FIELDS];
        for field in (0..4).rev() {
            backward[field] = logsumexp(computation(&self.transitions[field], detached)
                + (emissions[field + 1].clone() + backward[field + 1].clone()).unsqueeze_dim::<4>(2)
                    .expand([batch, queries, states, states]), 3)
                .reshape([batch, queries, states]);
        }
        backward.try_into().unwrap()
    }

    fn marginal_using(&self, field: usize, weights: Tensor<3>, detached: bool) -> Tensor<3> {
        let weights = normalized_log_weights(weights, 2);
        let leaf = &self.leaves[field];
        let probability = computation(&leaf.log_probability, detached);
        let outside = probability.clone().equal_elem(f32::NEG_INFINITY);
        let mixture = (weights - computation(&leaf.log_normalizer, detached)).exp()
            .matmul(computation(&leaf.table.exponential, detached).swap_dims(1, 2));
        let zero = mixture.clone().equal_elem(0.0).bool_or(outside);
        let log_mixture = mixture.mask_fill(zero.clone(), 1.0).log().mask_fill(zero, f32::NEG_INFINITY);
        probability + log_mixture
    }


    fn log_evidence(&self, evidence: &CircuitEvidence) -> Tensor<3> {
        let (batch, queries, states) = self.shape();
        let emissions = self.emissions_using(evidence, false);
        let mut forward = self.initial.clone();
        for field in 0..4 {
            forward = logsumexp((forward + emissions[field].clone()).unsqueeze_dim::<4>(3)
                .expand([batch, queries, states, states]) + self.transitions[field].clone(), 2)
                .reshape([batch, queries, states]);
        }
        logsumexp(forward + emissions[4].clone(), 2)
    }

    /// Prepare differentiable messages for conditional-marginal or likelihood readout.
    pub fn prepared(&self, evidence: &CircuitEvidence) -> PreparedCircuit<'_> {
        self.prepare(evidence, false)
    }

    /// Prepare decision-only messages without registering an autodiff graph.
    pub fn prepared_decisions(&self, evidence: &CircuitEvidence) -> PreparedCircuit<'_> {
        self.prepare(evidence, true)
    }

    fn prepare(&self, evidence: &CircuitEvidence, detached: bool) -> PreparedCircuit<'_> {
        let mut evidence = self.evidence(evidence);
        if detached {
            evidence.values = evidence.values.inner();
            evidence.known = evidence.known.inner();
        }
        let emissions = self.emissions_using(&evidence, detached);
        let (forward, backward) = self.messages_using(&emissions, detached);
        PreparedCircuit { call: self, evidence, emissions, forward, backward, detached }
    }

    pub fn conditional_log_marginals(&self, evidence: &CircuitEvidence, exclude_own_field: bool)
        -> (Tensor<3>, [Tensor<3>; 4]) {
        self.prepared(evidence).conditional_log_marginals(exclude_own_field)
    }

    pub fn greedy_decode(&self, supplied: &CircuitEvidence) -> Tensor<3, Int> {
        self.prepared_decisions(supplied).greedy_decode()
    }

    pub fn dependency_features(&self, decoded: &Tensor<3, Int>, supplied: &CircuitEvidence) -> Tensor<3> {
        self.prepared_decisions(supplied).dependency_features(decoded)
    }

    pub fn relation_given_spans(&self, spans: &CircuitEvidence) -> Tensor<3> {
        // The relation field is intentionally excluded even if the caller retained a label.
        let evidence = self.evidence(spans);
        let backward = self.backward_messages_using(&self.emissions_using(&evidence, false), false);
        self.marginal_using(0, self.initial.clone() + backward[0].clone(), false).mask_fill(self.missing.clone(), 0.0)
    }

    /// Reuse this call's normalized leaves and chain weights. Known queries are selected
    /// before likelihood arithmetic; finite unknown valid queries receive zero adjoints.
    /// NaN/Inf is supported in invalid padding, which call() removes before projections.
    pub fn coupling_loss_from_call(&self, gold: &GoldCombinations, supplied: &CircuitEvidence) -> (Tensor<1>, usize) {
        let (batch, queries, _) = self.shape();
        if gold.groups == 0 {
            assert_eq!(gold.query.dims()[0], 0);
            return (Tensor::zeros([1], &self.initial.device()), 0);
        }
        assert!(gold.host_query.iter().all(|&query| query >= 0 && (query as usize) < batch * queries),
            "gold combinations must address the call's actual flat query axis");
        // One batched graph for every row and document: per-document graphs multiplied kernel launches.
        self.coupling_loss_known(gold, supplied)
    }

    fn coupling_loss_known(&self, gold: &GoldCombinations, supplied: &CircuitEvidence) -> (Tensor<1>, usize) {
        let (batch, queries, states) = self.shape();
        let count = gold.query.dims()[0];
        let device = self.initial.device();
        if count == 0 { assert_eq!(gold.groups, 0); return (Tensor::zeros([1], &device), 0); }
        assert_eq!(gold.values.dims(), [count, CIRCUIT_FIELDS]);
        assert_eq!(gold.group.dims(), [count]);
        assert!(gold.groups > 0);
        let supplied = self.evidence(supplied);
        // Only host-selected known queries enter likelihood arithmetic. Select backward
        // scatters exact zeros to finite unknown rows of the shared call graph.
        let row = int(gold.host_query.iter().map(|&query| i32::try_from(query as usize / queries).expect("row fits i32")).collect(),
            [count], &device);
        let known = supplied.known.clone().reshape([batch * queries, CIRCUIT_FIELDS]).select(0, gold.query.clone());
        let supplied_values = supplied.values.clone().reshape([batch * queries, CIRCUIT_FIELDS]).select(0, gold.query.clone());
        let consistent = supplied_values.equal(gold.values.clone()).bool_or(known.clone().bool_not()).int()
            .sum_dim(1).equal_elem(CIRCUIT_FIELDS as i64);
        let missing = self.missing.clone().reshape([batch * queries, 1]).select(0, gold.query.clone());
        let mut invalid_tuple = missing.clone();
        let mut baseline = Tensor::zeros([count, 1], &device);
        let mut emissions = Vec::with_capacity(CIRCUIT_FIELDS);
        for field in 0..CIRCUIT_FIELDS {
            let leaf = &self.leaves[field];
            let classes = leaf.log_probability.dims()[2];
            let value = gold.values.clone().slice([0..count, field..field + 1]);
            let outside = outside_axis(&value, classes);
            invalid_tuple = invalid_tuple.bool_or(outside.clone());
            let unused = missing.clone().bool_or(outside);
            let value = value.mask_fill(unused.clone(), 0);
            let probability = leaf.log_probability.clone().reshape([batch * queries, classes]).select(0, gold.query.clone())
                .gather(1, value.clone()).mask_fill(unused.clone(), 0.0);
            let table = leaf.table.selected_rows(row.clone(), value.clone().reshape([count]), false)
                .mask_fill(unused.clone(), 0.0);
            let normalizer = leaf.log_normalizer.clone().reshape([batch * queries, states]).select(0, gold.query.clone()).mask_fill(unused.clone(), 0.0);
            emissions.push(probability.clone() + table - normalizer);
            baseline = baseline + probability.mask_fill(known.clone().slice([0..count, field..field + 1]).bool_or(unused), 0.0);
        }
        let mut forward = self.initial.clone().reshape([batch * queries, states]).select(0, gold.query.clone()) + emissions[0].clone();
        for edge in 0..4 {
            let transition = self.transitions[edge].clone().reshape([batch * queries, states, states]).select(0, gold.query.clone());
            forward = logsumexp(forward.unsqueeze_dim::<3>(2).expand([count, states, states]) + transition, 1)
                .reshape([count, states]) + emissions[edge + 1].clone();
        }
        let supplied_log = self.log_evidence(&supplied).reshape([batch * queries, 1]).select(0, gold.query.clone());
        let no_evidence_mass = supplied_log.clone().equal_elem(f32::NEG_INFINITY);
        let likelihood = conditional_log_mass(logsumexp(forward, 1), supplied_log);
        let absent = consistent.bool_not().bool_or(invalid_tuple).bool_or(no_evidence_mass);
        let joint = likelihood.mask_fill(absent.clone(), f32::NEG_INFINITY).reshape([count]);
        let factorized = baseline.mask_fill(absent, f32::NEG_INFINITY).reshape([count]);
        let reduce = |scores: Tensor<1>| {
            let plain = scores.clone().inner();
            let maximum = Tensor::<1>::full([gold.groups], f32::NEG_INFINITY, &plain.device())
                .select_assign(0, gold.group.clone().inner(), plain, IndexingUpdateOp::Max);
            let maximum = if device.is_autodiff() { maximum.autodiff() } else { maximum };
            let absent = maximum.clone().equal_elem(f32::NEG_INFINITY);
            let maximum = maximum.mask_fill(absent.clone(), 0.0);
            let total = Tensor::<1>::zeros([gold.groups], &device).select_assign(0, gold.group.clone(),
                (scores - maximum.clone().select(0, gold.group.clone())).exp(), IndexingUpdateOp::Add);
            // Entirely invalid groups contribute exactly zero, including their adjoints.
            (total.mask_fill(absent.clone(), 1.0).log() + maximum).mask_fill(absent, 0.0)
        };
        (((reduce(factorized) - reduce(joint)) * gold.group_weight.clone()).sum(), gold.groups)
    }

    pub fn slot_view(&self, row: usize, range: std::ops::Range<usize>) -> Self {
        Self {
            leaves: std::array::from_fn(|field| {
                let leaf = &self.leaves[field];
                Leaf { log_probability: leaf.log_probability.clone().slice([row..row + 1, range.clone()]),
                    log_normalizer: leaf.log_normalizer.clone().slice([row..row + 1, range.clone()]),
                    table: if field == 0 { leaf.table.clone() } else {
                        LeafTable { shifted: leaf.table.shifted.clone().slice([row..row + 1]),
                            exponential: leaf.table.exponential.clone().slice([row..row + 1]) }
                    }, coordinates: leaf.coordinates.clone() }
            }),
            initial: self.initial.clone().slice([row..row + 1, range.clone()]),
            transitions: std::array::from_fn(|edge| self.transitions[edge].clone().slice([row..row + 1, range.clone()])),
            missing: self.missing.clone().slice([row..row + 1, range]),
        }
    }

    /// Concatenate calls from the same circuit; their relation reweighting table is shared.
    pub fn cat_rows(rows: &[Self]) -> Self {
        assert!(!rows.is_empty());
        Self {
            leaves: std::array::from_fn(|field| Leaf {
                log_probability: Tensor::cat(rows.iter().map(|row| row.leaves[field].log_probability.clone()).collect(), 0),
                log_normalizer: Tensor::cat(rows.iter().map(|row| row.leaves[field].log_normalizer.clone()).collect(), 0),
                table: if field == 0 { rows[0].leaves[field].table.clone() } else {
                    LeafTable {
                        shifted: Tensor::cat(rows.iter().map(|row| row.leaves[field].table.shifted.clone()).collect(), 0),
                        exponential: Tensor::cat(rows.iter().map(|row| row.leaves[field].table.exponential.clone()).collect(), 0),
                    }
                }, coordinates: rows[0].leaves[field].coordinates.clone(),
            }),
            initial: Tensor::cat(rows.iter().map(|row| row.initial.clone()).collect(), 0),
            transitions: std::array::from_fn(|edge| Tensor::cat(rows.iter().map(|row| row.transitions[edge].clone()).collect(), 0)),
            missing: Tensor::cat(rows.iter().map(|row| row.missing.clone()).collect(), 0),
        }
    }
}

impl PreparedCircuit<'_> {
    fn log_evidence(&self, detached: bool) -> Tensor<3> {
        logsumexp(computation(&self.forward[4], detached) + computation(&self.emissions[4], detached), 2)
    }

    pub fn conditional_log_marginals(&self, exclude_own_field: bool) -> (Tensor<3>, [Tensor<3>; 4]) {
        let (batch, queries, _) = self.call.shape();
        let missing = if self.detached { self.call.missing.clone().inner() } else { self.call.missing.clone() };
        let logs: [Tensor<3>; CIRCUIT_FIELDS] = std::array::from_fn(|field| {
            let mut log = self.call.marginal_using(field, self.forward[field].clone() + self.backward[field].clone(), self.detached);
            if !exclude_own_field {
                // Masks are expanded to the full shape: ndarray's mask_where/mask_fill do not broadcast a narrower mask.
                let dims = log.dims();
                let known = self.evidence.known.clone().slice([0..batch, 0..queries, field..field + 1]).expand(dims);
                let value = self.evidence.values.clone().slice([0..batch, 0..queries, field..field + 1]);
                let unequal = self.call.leaves[field].coordinates.clone().expand(dims).equal(value.expand(dims)).bool_not();
                let fixed = Tensor::zeros(dims, &log.device()).mask_fill(unequal, f32::NEG_INFINITY);
                log = log.mask_where(known, fixed);
            }
            log.mask_fill(missing.clone(), 0.0)
        });
        let [relation, ss, se, os, oe] = logs;
        (relation, [ss, se, os, oe])
    }

    /// Greedy choices and their messages never participate in autodiff.
    pub fn greedy_decode(&self) -> Tensor<3, Int> {
        self.greedy_decode_ordered(false)
    }

    /// Choose endpoints while summing over the unknown relation instead of taking a relation branch.
    pub fn greedy_decode_mentions(&self)->Tensor<3,Int> {
        self.greedy_decode_ordered(true)
    }

    fn greedy_decode_ordered(&self,marginalize_relation:bool)->Tensor<3,Int> {
        let (batch, queries, states) = self.call.shape();
        let missing = self.call.missing.clone().inner();
        let mut forward = self.call.initial.clone().inner();
        let mut chosen = Vec::with_capacity(CIRCUIT_FIELDS);
        for field in 0..CIRCUIT_FIELDS {
            let known = self.evidence.known.clone().inner().slice([0..batch, 0..queries, field..field + 1]);
            let supplied = self.evidence.values.clone().inner().slice([0..batch, 0..queries, field..field + 1]);
            // Argmax and `int(...)` uploads both use the backend's default integer dtype; never force one.
            let scores = self.call.marginal_using(field, forward.clone() + self.backward[field].clone().inner(), true);
            let finite = scores.clone().is_finite();
            let absent = finite.clone().int().sum_dim(2).equal_elem(0);
            let dims = scores.dims();
            // CUDA argmax has no defined index for a row without a finite score.
            // Give it finite operands, then explicitly discard that placeholder choice.
            let index = scores.mask_fill(finite.bool_not(), f32::NEG_INFINITY)
                .mask_fill(absent.clone().expand(dims), 0.0).argmax(2).mask_fill(absent, 0);
            let index = index.clone().mask_fill(outside_axis(&index, dims[2]), 0);
            let supplied = supplied.clone().mask_fill(outside_axis(&supplied, dims[2]), 0);
            let choice = index.mask_where(known, supplied).mask_fill(missing.clone(), 0);
            if field < 4 {
                let leaf = &self.call.leaves[field];
                let emission=if field==0&&marginalize_relation {
                    self.emissions[0].clone().inner()
                }else{
                    leaf.log_probability.clone().inner().gather(2, choice.clone()).mask_fill(missing.clone(), 0.0)
                        + leaf.table.selected(choice.clone(), true).mask_fill(missing.clone(), 0.0)
                        - leaf.log_normalizer.clone().inner().mask_fill(missing.clone(), 0.0)
                };
                forward = logsumexp((forward + emission).unsqueeze_dim::<4>(3).expand([batch, queries, states, states])
                    + self.call.transitions[field].clone().inner(), 2).reshape([batch, queries, states]);
            }
            chosen.push(choice);
        }
        Tensor::cat(chosen, 2)
    }

    /// Decider inputs are constants. Evaluate only the proposed coordinates, rather
    /// than constructing ten complete relation/pointer distributions and gathering them.
    pub fn dependency_features(&self, decoded: &Tensor<3, Int>) -> Tensor<3> {
        let (batch, queries, _) = self.call.shape();
        assert_eq!(decoded.dims(), [batch, queries, CIRCUIT_FIELDS]);
        let missing = self.call.missing.clone().inner();
        let decoded = decoded.clone().inner().mask_fill(missing.clone(), 0);
        let complete = CircuitEvidence {
            values: decoded.clone(),
            known: missing.clone().bool_not().expand([batch, queries, CIRCUIT_FIELDS]),
        };
        let emissions = self.call.emissions_using(&complete, true);
        let (forward, backward) = self.call.messages_using(&emissions, true);
        let complete_log = logsumexp(forward[4].clone() + emissions[4].clone(), 2);
        let mut baseline = Tensor::zeros([batch, queries, 1], &complete_log.device());
        let mut features = Vec::with_capacity(CIRCUIT_FEATURES);
        for field in 0..CIRCUIT_FIELDS {
            let fixed = self.evidence.known.clone().inner().slice([0..batch, 0..queries, field..field + 1])
                .bool_or(missing.clone());
            let conditional = conditional_log_mass(complete_log.clone(), logsumexp(forward[field].clone() + backward[field].clone(), 2));
            let weights = self.forward[field].clone().inner() + self.backward[field].clone().inner();
            let marginal = conditional_log_mass(logsumexp(weights.clone() + emissions[field].clone(), 2), logsumexp(weights, 2));
            let undefined = conditional.clone().is_finite().bool_and(marginal.clone().is_finite()).bool_not().bool_or(fixed.clone());
            features.push(conditional.mask_fill(undefined.clone(), 0.0) - marginal.mask_fill(undefined, 0.0));
            let coordinate = decoded.clone().slice([0..batch, 0..queries, field..field + 1]);
            let unused = fixed.bool_or(outside_axis(&coordinate, self.call.leaves[field].log_probability.dims()[2]));
            let coordinate = coordinate.mask_fill(unused.clone(), 0);
            baseline = baseline + self.call.leaves[field].log_probability.clone().inner().gather(2, coordinate).mask_fill(unused, 0.0);
        }
        let supplied_log = self.log_evidence(true);
        let undefined = complete_log.clone().is_finite().bool_and(supplied_log.clone().is_finite())
            .bool_and(baseline.clone().is_finite()).bool_not().bool_or(missing);
        features.push(complete_log.mask_fill(undefined.clone(), 0.0) - supplied_log.mask_fill(undefined.clone(), 0.0)
            - baseline.mask_fill(undefined, 0.0));
        Tensor::cat(features, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{SlotPlan, float};
    use crate::pointer::SourcePointerDocument;
    use dre_core::DocumentLayout;

    const HIDDEN: [f64; 2] = [0.4, -0.6];
    const SOURCE: [[f64; 2]; 3] = [[0.2, -0.8], [0.7, 0.3], [-0.4, 0.5]];
    const LOGITS: [[f64; 3]; 5] = [
        [-0.4, 0.6, 0.1], [0.2, -0.3, 0.7], [-0.6, 0.8, 0.2],
        [0.5, 0.1, -0.5], [-0.2, 0.4, 0.9],
    ];
    const GOLD: [[usize; 5]; 2] = [[1, 0, 1, 1, 2], [2, 0, 0, 2, 1]];

    fn host<const D: usize>(value: &Tensor<D>) -> Vec<f64> {
        value.to_data().try_to_vec_as::<f32>().unwrap().into_iter().map(f64::from).collect()
    }
    fn host_int<const D: usize>(value: &Tensor<D, Int>) -> Vec<i64> {
        value.to_data().try_to_vec_as::<i64>().unwrap()
    }
    fn close(actual: f64, expected: f64, tolerance: f64) {
        if actual.is_finite() && expected.is_finite() {
            assert!((actual - expected).abs() <= tolerance * (1.0 + expected.abs()), "{actual} != {expected}");
        } else { assert_eq!(actual, expected); }
    }
    fn tensor<const D: usize>(values: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
        float(values.iter().map(|&value| value as f32).collect(), shape, device).require_grad()
    }
    fn parameters() -> Vec<f64> {
        (0..78).map(|index| (((index * 17 + 11) % 43) as f32 / 19.0 - 1.0) as f64).collect()
    }
    fn make_model(theta: &[f64], device: &Device) -> JointCircuit {
        let mut model = JointCircuitConfig::new(2, 3).with_states(2).init(device);
        let mut cursor = 0;
        let mut take = |size: usize| { let start = cursor; cursor += size; &theta[start..cursor] };
        model.relation_table = Param::from_tensor(tensor(take(6), [3, 2], device));
        for endpoint in &mut model.endpoints {
            endpoint.weight = Param::from_tensor(tensor(take(4), [2, 2], device));
            endpoint.bias = Some(Param::from_tensor(tensor(take(2), [2], device)));
        }
        model.initial_logits = Param::from_tensor(tensor(take(2), [2], device));
        model.transition_logits = Param::from_tensor(tensor(take(16), [4, 2, 2], device));
        for gate in &mut model.gates {
            gate.weight = Param::from_tensor(tensor(take(4), [2, 2], device));
            gate.bias = Some(Param::from_tensor(tensor(take(2), [2], device)));
        }
        assert_eq!(cursor, theta.len());
        model
    }
    struct Inputs {
        hidden: Tensor<3>,
        relation: Tensor<3>,
        tokens: [Tensor<3>; 4],
    }
    fn fixture_with_inputs(model: &JointCircuit, device: &Device) -> (CircuitCall, SourcePointerContext, Inputs) {
        let plan = SlotPlan::queries(&DocumentLayout::packed(vec![vec![2]], 2), &[vec![1]]).unwrap();
        let mut context = SourcePointerContext::new(&plan,
            tensor(&SOURCE[..2].concat(), [1, 2, 2], device),
            &[SourcePointerDocument { row: 0, document: 0, first_token: 0, text: "ab", token_offsets: &[(0, 1), (1, 2)] }],
            tensor(&SOURCE[2], [2], device)).unwrap();
        context.values = tensor(&SOURCE.concat(), [1, 3, 2], device);
        let inputs = Inputs {
            hidden: tensor(&HIDDEN, [1, 1, 2], device),
            relation: tensor(&LOGITS[0], [1, 1, 3], device),
            tokens: std::array::from_fn(|field| tensor(&LOGITS[field + 1], [1, 1, 3], device)),
        };
        let call = model.call(inputs.hidden.clone(), inputs.relation.clone(), &inputs.tokens,
            &model.endpoint_tables(&context), &Tensor::ones([1, 1], device));
        (call, context, inputs)
    }
    fn fixture(model: &JointCircuit, device: &Device) -> (CircuitCall, SourcePointerContext) {
        let (call, context, _) = fixture_with_inputs(model, device);
        (call, context)
    }
    fn evidence(values: [usize; 5], known: [bool; 5], device: &Device) -> CircuitEvidence {
        CircuitEvidence::from_host(values.map(|value| value as i64).to_vec(), known.to_vec(), 1, 1, device)
    }
    fn gold(device: &Device) -> GoldCombinations {
        GoldCombinations::from_host(vec![0, 0], GOLD.concat().into_iter().map(|value| value as i64).collect(), vec![0, 0], 1, device)
    }
    fn softmax(values: &[f64]) -> Vec<f64> {
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = values.iter().map(|value| (value - maximum).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.into_iter().map(|weight| weight / total).collect()
    }
    fn linear(parameters: &[f64], input: [f64; 2]) -> [f64; 2] {
        std::array::from_fn(|state| input[0] * parameters[state] + input[1] * parameters[2 + state] + parameters[4 + state])
    }
    struct Dense {
        entries: Vec<([usize; 5], f64)>,
        factorized: [Vec<f64>; 5],
    }
    impl Dense {
        // Independent FP64 enumeration over all 3^5 value tuples and 2^5 state paths.
        // It starts from raw parameters/logits, not circuit messages or normalizers.
        fn new(theta: &[f64]) -> Self { Self::with_inputs(theta, HIDDEN, SOURCE, LOGITS) }
        fn with_inputs(theta: &[f64], hidden: [f64; 2], source: [[f64; 2]; 3], logits: [[f64; 3]; 5]) -> Self {
            let factorized: [Vec<f64>; 5] = std::array::from_fn(|field| softmax(&logits[field]));
            let mut leaves = [[[0.0; 3]; 2]; 5];
            for field in 0..5 { for state in 0..2 {
                let terms: [f64; 3] = std::array::from_fn(|value| {
                    let raw = if field == 0 { theta[value * 2 + state] }
                        else { linear(&theta[6 + (field - 1) * 6..12 + (field - 1) * 6], source[value])[state] };
                    8.0 * (raw / 8.0).tanh()
                });
                let total: f64 = (0..3).map(|value| factorized[field][value] * terms[value].exp()).sum();
                for value in 0..3 { leaves[field][state][value] = factorized[field][value] * terms[value].exp() / total; }
            }}
            let gate: [[f64; 2]; 5] = std::array::from_fn(|field| linear(&theta[48 + field * 6..54 + field * 6], hidden));
            let initial = softmax(&[theta[30] + gate[0][0], theta[31] + gate[0][1]]);
            let transitions: [[[f64; 2]; 2]; 4] = std::array::from_fn(|edge| std::array::from_fn(|from| {
                softmax(&[theta[32 + edge * 4 + from * 2] + gate[edge + 1][0],
                    theta[33 + edge * 4 + from * 2] + gate[edge + 1][1]]).try_into().unwrap()
            }));
            let mut entries = Vec::with_capacity(243);
            for code in 0..243usize {
                let mut remaining = code;
                let values: [usize; 5] = std::array::from_fn(|_| { let value = remaining % 3; remaining /= 3; value });
                let mut probability = 0.0;
                for path in 0..32usize {
                    let states: [usize; 5] = std::array::from_fn(|field| (path >> field) & 1);
                    let mut weight = initial[states[0]];
                    for field in 0..5 {
                        weight *= leaves[field][states[field]][values[field]];
                        if field < 4 { weight *= transitions[field][states[field]][states[field + 1]]; }
                    }
                    probability += weight;
                }
                entries.push((values, probability));
            }
            Self { entries, factorized }
        }
        fn mass(&self, values: [usize; 5], known: [bool; 5]) -> f64 {
            self.entries.iter().filter(|(candidate, _)| (0..5).all(|field| !known[field] || candidate[field] == values[field]))
                .map(|(_, probability)| probability).sum()
        }
        fn marginal(&self, field: usize, values: [usize; 5], known: [bool; 5], exclude_own: bool) -> Vec<f64> {
            let mut known = known;
            if exclude_own { known[field] = false; }
            let denominator = self.mass(values, known);
            (0..3).map(|value| {
                if known[field] && values[field] != value { return f64::NEG_INFINITY; }
                let mut candidate = values; candidate[field] = value;
                let mut selected = known; selected[field] = true;
                (self.mass(candidate, selected) / denominator).ln()
            }).collect()
        }
        fn coupling(&self) -> f64 {
            let denominator = self.mass([0; 5], [false, true, false, false, false]);
            let joint: f64 = GOLD.iter().map(|&tuple| self.mass(tuple, [true; 5]) / denominator).sum();
            let product: f64 = GOLD.iter().map(|tuple| (0..5).filter(|&field| field != 1)
                .map(|field| self.factorized[field][tuple[field]]).product::<f64>()).sum();
            product.ln() - joint.ln()
        }
    }

    fn enumeration_case(device: &Device) {
        let theta = parameters();
        let model = make_model(&theta, device);
        let (call, _) = fixture(&model, device);
        let dense = Dense::new(&theta);
        close(dense.entries.iter().map(|(_, p)| p).sum(), 1.0, 1e-12);
        let values = [1, 0, 2, 1, 2];
        for mask in 0..32 {
            let known = std::array::from_fn(|field| mask & (1 << field) != 0);
            let supplied = evidence(values, known, device);
            let prepared = call.prepared(&supplied);
            for exclude in [false, true] {
                let (relation, endpoints) = prepared.conditional_log_marginals(exclude);
                let logs = [relation, endpoints[0].clone(), endpoints[1].clone(), endpoints[2].clone(), endpoints[3].clone()];
                for field in 0..5 {
                    let actual = host(&logs[field]);
                    let expected = dense.marginal(field, values, known, exclude);
                    for (&actual, expected) in actual.iter().zip(expected) { close(actual, expected, 3e-5); }
                    close(actual.iter().map(|value| value.exp()).sum(), 1.0, 3e-5);
                }
            }
        }
        let supplied = evidence([0; 5], [false, true, false, false, false], device);
        let prepared = call.prepared_decisions(&supplied);
        assert!(prepared.forward.iter().chain(&prepared.backward).all(|message| !message.is_tracked()));
        let decoded = host_int(&prepared.greedy_decode());
        let mut expected = [0; 5];
        let mut known = [false, true, false, false, false];
        for field in 0..5 {
            if !known[field] {
                let logs = dense.marginal(field, expected, known, false);
                expected[field] = (0..3).max_by(|&a, &b| logs[a].total_cmp(&logs[b])).unwrap();
            }
            known[field] = true;
        }
        assert_eq!(decoded, expected.map(|value| value as i64));
        let proposed = int(decoded.iter().map(|&value| value as i32).collect(), [1, 1, 5], device);
        let feature_tensor = prepared.dependency_features(&proposed);
        assert!(!feature_tensor.is_tracked(), "decision features must not backpropagate through the circuit");
        let features = host(&feature_tensor);
        for field in 0..5 {
            let expected_feature = if field == 1 { 0.0 } else {
                dense.marginal(field, expected, [true; 5], true)[expected[field]]
                    - dense.marginal(field, [0; 5], [false, true, false, false, false], false)[expected[field]]
            };
            close(features[field], expected_feature, 4e-5);
        }
        let baseline: f64 = (0..5).filter(|&field| field != 1).map(|field| dense.factorized[field][expected[field]].ln()).sum();
        close(features[5], (dense.mass(expected, [true; 5]) / dense.mass([0; 5], [false, true, false, false, false])).ln() - baseline, 4e-5);
        let spans = evidence([2, 0, 2, 1, 2], [true; 5], device);
        for (actual, expected) in host(&call.relation_given_spans(&spans)).into_iter()
            .zip(dense.marginal(0, [2, 0, 2, 1, 2], [true; 5], true)) { close(actual, expected, 3e-5); }
        let (loss, groups) = call.coupling_loss_from_call(&gold(device), &supplied);
        assert_eq!(groups, 1); close(host(&loss)[0], dense.coupling(), 4e-5);
        let joined = CircuitCall::cat_rows(&[call.clone(), call.clone()]);
        let supplied = CircuitEvidence::cat_rows(&[supplied.clone(), supplied]);
        let joined_logs = joined.conditional_log_marginals(&supplied, true).0;
        let view_logs = joined.slot_view(1, 0..1).conditional_log_marginals(&supplied.slot_view(1, 0..1), true).0;
        for (actual, expected) in host(&joined_logs).into_iter().zip(host(&view_logs).into_iter().cycle()) { close(actual, expected, 1e-6); }
        // Interleaved alternatives exercise nontrivial group reduction and source-row
        // partitioning. A supplied-inconsistent alternative is removed from BOTH masses.
        let second = [[0, 2, 1, 1, 0], [2, 1, 0, 2, 1]];
        let tuples = [second[0], GOLD[0], [0, 1, 0, 0, 0], GOLD[1], second[1]];
        let grouped = GoldCombinations::from_host(vec![1, 0, 0, 0, 1],
            tuples.concat().into_iter().map(|value| value as i64).collect(), vec![1, 0, 0, 0, 1], 2, device);
        let supplied = CircuitEvidence::cat_rows(&[
            evidence([0; 5], [false, true, false, false, false], device),
            CircuitEvidence::empty(1, 1, device),
        ]);
        let (actual, groups) = joined.coupling_loss_from_call(&grouped, &supplied);
        assert_eq!(groups, 2);
        let second_joint: f64 = second.iter().map(|&tuple| dense.mass(tuple, [true; 5])).sum();
        let second_product: f64 = second.iter().map(|tuple| (0..5).map(|field| dense.factorized[field][tuple[field]]).product::<f64>()).sum();
        close(host(&actual)[0], dense.coupling() + second_product.ln() - second_joint.ln(), 4e-5);
        // Unknown evidence coordinates are never allowed into a gather.
        let unknown = CircuitEvidence {
            values: int(vec![i32::MAX; 5], [1, 1, 5], device),
            known: boolean(vec![false; 5], [1, 1, 5], device),
        };
        let actual = host(&call.conditional_log_marginals(&unknown, true).0);
        for (actual, expected) in actual.into_iter().zip(dense.marginal(0, [0; 5], [false; 5], true)) { close(actual, expected, 3e-5); }
        let supplied = evidence(GOLD[0], [true; 5], device);
        close(host(&call.coupling_loss_from_call(&gold(device), &supplied).0)[0], 0.0, 4e-5);
    }

    fn shared_relation_case(device: &Device) {
        let theta = parameters();
        let model = make_model(&theta, device);
        assert_eq!(model.relation_table.val().dims(), [3, 2]);
        let plan = SlotPlan::queries(&DocumentLayout::packed(vec![vec![2], vec![2]], 2),
            &[vec![2], vec![2]]).unwrap();
        let context = SourcePointerContext::new(&plan, tensor(&SOURCE[..2].concat().repeat(2), [2, 2, 2], device), &[
            SourcePointerDocument { row: 0, document: 0, first_token: 0, text: "ab", token_offsets: &[(0, 1), (1, 2)] },
            SourcePointerDocument { row: 1, document: 0, first_token: 0, text: "xy", token_offsets: &[(0, 1), (1, 2)] },
        ], tensor(&SOURCE[2], [2], device)).unwrap();
        let call = model.call(tensor(&HIDDEN.repeat(4), [2, 2, 2], device),
            tensor(&LOGITS[0].repeat(4), [2, 2, 3], device),
            &std::array::from_fn(|field| tensor(&LOGITS[field + 1].repeat(4), [2, 2, 3], device)),
            &model.endpoint_tables(&context), &Tensor::ones([2, 2], device));
        assert_eq!(call.leaves[0].table.shifted.dims(), [1, 3, 2]);
        assert_eq!(call.leaves[0].table.exponential.dims(), [1, 3, 2]);
        let tuples: [[usize; 5]; 4] = [[2, 0, 1, 1, 0], [0, 1, 2, 0, 2], [1, 2, 0, 2, 1], [2, 1, 1, 0, 2]];
        let evidence = CircuitEvidence::from_host(tuples.concat().into_iter().map(|value| value as i64).collect(),
            vec![true; 20], 2, 2, device);
        let prepared = call.prepared(&evidence);
        let dense = Dense::new(&theta);
        for exclude in [false, true] {
            let (relation, endpoints) = prepared.conditional_log_marginals(exclude);
            let logs = [relation, endpoints[0].clone(), endpoints[1].clone(), endpoints[2].clone(), endpoints[3].clone()];
            for field in 0..5 {
                let actual = host(&logs[field]);
                for (query, &tuple) in tuples.iter().enumerate() {
                    for (value, expected) in dense.marginal(field, tuple, [true; 5], exclude).into_iter().enumerate() {
                        close(actual[query * 3 + value], expected, 3e-5);
                    }
                }
            }
        }
        let view = call.slot_view(1, 0..2);
        assert_eq!(view.leaves[0].table.shifted.dims(), [1, 3, 2]);
        let actual = host(&view.conditional_log_marginals(&evidence.slot_view(1, 0..2), true).0);
        for (query, tuple) in tuples[2..].iter().enumerate() {
            for (value, expected) in dense.marginal(0, *tuple, [true; 5], true).into_iter().enumerate() {
                close(actual[query * 3 + value], expected, 3e-5);
            }
        }
    }

    fn gradient_case(device: &Device) {
        let theta = parameters();
        let model = make_model(&theta, device);
        let (call, context, inputs) = fixture_with_inputs(&model, device);
        let supplied = evidence([0; 5], [false, true, false, false, false], device);
        let loss = call.coupling_loss_from_call(&gold(device), &supplied).0;
        let gradients = loss.backward();
        let mut actual = host(&model.relation_table.val().grad(&gradients).unwrap());
        for endpoint in &model.endpoints {
            actual.extend(host(&endpoint.weight.val().grad(&gradients).unwrap()));
            actual.extend(host(&endpoint.bias.as_ref().unwrap().val().grad(&gradients).unwrap()));
        }
        actual.extend(host(&model.initial_logits.val().grad(&gradients).unwrap()));
        actual.extend(host(&model.transition_logits.val().grad(&gradients).unwrap()));
        for gate in &model.gates {
            actual.extend(host(&gate.weight.val().grad(&gradients).unwrap()));
            actual.extend(host(&gate.bias.as_ref().unwrap().val().grad(&gradients).unwrap()));
        }
        assert_eq!(actual.len(), theta.len());
        for index in 0..theta.len() {
            let mut plus = theta.clone(); plus[index] += 1e-5;
            let mut minus = theta.clone(); minus[index] -= 1e-5;
            let expected = (Dense::new(&plus).coupling() - Dense::new(&minus).coupling()) / 2e-5;
            close(actual[index], expected, 8e-5);
        }
        let hidden_gradient = host(&inputs.hidden.grad(&gradients).unwrap());
        for index in 0..2 {
            let mut plus = HIDDEN; plus[index] += 1e-5;
            let mut minus = HIDDEN; minus[index] -= 1e-5;
            let expected = (Dense::with_inputs(&theta, plus, SOURCE, LOGITS).coupling()
                - Dense::with_inputs(&theta, minus, SOURCE, LOGITS).coupling()) / 2e-5;
            close(hidden_gradient[index], expected, 8e-5);
        }
        let source_gradient = host(&context.values.grad(&gradients).unwrap());
        for index in 0..6 {
            let mut plus = SOURCE; plus[index / 2][index % 2] += 1e-5;
            let mut minus = SOURCE; minus[index / 2][index % 2] -= 1e-5;
            let expected = (Dense::with_inputs(&theta, HIDDEN, plus, LOGITS).coupling()
                - Dense::with_inputs(&theta, HIDDEN, minus, LOGITS).coupling()) / 2e-5;
            close(source_gradient[index], expected, 8e-5);
        }
        for field in 0..5 {
            let actual = if field == 0 { host(&inputs.relation.grad(&gradients).unwrap()) }
                else { host(&inputs.tokens[field - 1].grad(&gradients).unwrap()) };
            for index in 0..3 {
                let mut plus = LOGITS; plus[field][index] += 1e-5;
                let mut minus = LOGITS; minus[field][index] -= 1e-5;
                let expected = (Dense::with_inputs(&theta, HIDDEN, SOURCE, plus).coupling()
                    - Dense::with_inputs(&theta, HIDDEN, SOURCE, minus).coupling()) / 2e-5;
                close(actual[index], expected, 8e-5);
            }
        }
        // Conditional-marginal gradients use a different path (matrix-product leaf readout).
        let model = make_model(&theta, device);
        let (call, _) = fixture(&model, device);
        let marginal = call.conditional_log_marginals(&supplied, true).0.slice([0..1, 0..1, 2..3]).sum();
        let gradients = marginal.backward();
        let actual = host(&model.relation_table.val().grad(&gradients).unwrap());
        for index in 0..6 {
            let mut plus = theta.clone(); plus[index] += 1e-5;
            let mut minus = theta.clone(); minus[index] -= 1e-5;
            let known = [false, true, false, false, false];
            let expected = (Dense::new(&plus).marginal(0, [0; 5], known, true)[2]
                - Dense::new(&minus).marginal(0, [0; 5], known, true)[2]) / 2e-5;
            close(actual[index], expected, 8e-5);
        }
    }

    fn identity_case(device: &Device) {
        let random = JointCircuitConfig::new(2, 3).with_states(2).init(device);
        let (call, _) = fixture(&random, device);
        let supplied = evidence([0; 5], [false, true, false, false, false], device);
        let (relation, endpoints) = call.conditional_log_marginals(&supplied, true);
        for (field, logs) in [relation, endpoints[0].clone(), endpoints[1].clone(), endpoints[2].clone(), endpoints[3].clone()].iter().enumerate() {
            for (actual, expected) in host(logs).into_iter().zip(softmax(&LOGITS[field])) { close(actual.exp(), expected, 0.015); }
        }
        close(host(&call.coupling_loss_from_call(&gold(device), &supplied).0)[0], 0.0, 0.03);
        let mut product = random;
        product.relation_table = Param::from_tensor(Tensor::zeros([3, 2], device));
        for endpoint in &mut product.endpoints {
            endpoint.weight = Param::from_tensor(Tensor::zeros([2, 2], device));
            endpoint.bias = Some(Param::from_tensor(Tensor::zeros([2], device)));
        }
        let (call, _) = fixture(&product, device);
        close(host(&call.coupling_loss_from_call(&gold(device), &supplied).0)[0], 0.0, 3e-6);
        for feature in host(&call.dependency_features(&call.greedy_decode(&supplied), &supplied)) { close(feature, 0.0, 3e-6); }
    }

    fn padding_case(device: &Device) {
        let evaluate = |dirty: bool| {
            let model = make_model(&parameters(), device);
            let layout = DocumentLayout::packed(vec![vec![2], vec![2], vec![]], 4);
            let plan = SlotPlan::queries(&layout, &[vec![2], vec![1], vec![]]).unwrap().with_slot_padding(4);
            let source_data: Vec<f64> = (0..3 * 4 * 2).map(|index| {
                let row = index / 8; let token = (index / 2) % 4;
                if dirty && (row == 2 || token >= 2) { if index % 2 == 0 { f64::NAN } else { f64::INFINITY } }
                else { index as f64 / 17.0 - 0.5 }
            }).collect();
            let source = tensor(&source_data, [3, 4, 2], device);
            let context = SourcePointerContext::new(&plan, source.clone(), &[
                SourcePointerDocument { row: 0, document: 0, first_token: 0, text: "ab", token_offsets: &[(0, 1), (1, 2)] },
                SourcePointerDocument { row: 1, document: 0, first_token: 0, text: "xy", token_offsets: &[(0, 1), (1, 2)] },
            ], tensor(&SOURCE[2], [2], device)).unwrap();
            let real = |query: usize| [0, 1, 4].contains(&query);
            let hidden = tensor(&(0..24).map(|index| if dirty && !real(index / 2) { f64::NAN } else { HIDDEN[index % 2] }).collect::<Vec<_>>(), [3, 4, 2], device);
            let relation = tensor(&(0..36).map(|index| if dirty && !real(index / 3) { f64::INFINITY } else { LOGITS[0][index % 3] }).collect::<Vec<_>>(), [3, 4, 3], device);
            let tokens: [Tensor<3>; 4] = std::array::from_fn(|field| tensor(&(0..60).map(|index| {
                let query = index / 5; let token = index % 5;
                if dirty && (!real(query) || (2..4).contains(&token)) { if token % 2 == 0 { f64::NAN } else { f64::NEG_INFINITY } }
                else { LOGITS[field + 1][if token == 4 { 2 } else { token % 2 }] }
            }).collect::<Vec<_>>(), [3, 4, 5], device));
            let tables = model.endpoint_tables(&context);
            let call = model.call(hidden.clone(), relation.clone(), &tokens, &tables, &plan.tensors(device).slot_valid);
            // Query 1 is valid but partially known, and has no coupling group.
            let mut supplied_known = vec![false; 60]; supplied_known[6] = true;
            let supplied = CircuitEvidence::from_host(vec![0; 60], supplied_known, 3, 4, device);
            let gold = GoldCombinations::from_host(vec![0, 4, 0, 4],
                vec![1, 0, 1, 1, 4, 1, 0, 1, 1, 4, 2, 0, 0, 4, 1, 2, 0, 0, 4, 1], vec![0, 1, 0, 1], 2, device);
            let loss = call.coupling_loss_from_call(&gold, &supplied).0;
            let value = host(&loss)[0];
            let gradients = loss.backward();
            let source_gradient = host(&source.grad(&gradients).unwrap());
            let hidden_gradient = host(&hidden.grad(&gradients).unwrap());
            let relation_gradient = host(&relation.grad(&gradients).unwrap());
            let token_gradients: Vec<_> = tokens.iter().flat_map(|token| host(&token.grad(&gradients).unwrap())).collect();
            let mut parameter_gradient = host(&model.relation_table.val().grad(&gradients).unwrap());
            for endpoint in &model.endpoints { parameter_gradient.extend(host(&endpoint.weight.val().grad(&gradients).unwrap())); }
            for gate in &model.gates { parameter_gradient.extend(host(&gate.weight.val().grad(&gradients).unwrap())); }
            assert!(source_gradient.iter().chain(&hidden_gradient).chain(&relation_gradient).chain(&token_gradients)
                .chain(&parameter_gradient).all(|value| value.is_finite()));
            for range in [4..8, 12..24] { assert!(source_gradient[range].iter().all(|&value| value == 0.0)); }
            for range in [2..8, 10..24] { assert!(hidden_gradient[range].iter().all(|&value| value == 0.0)); }
            for range in [3..12, 15..36] { assert!(relation_gradient[range].iter().all(|&value| value == 0.0)); }
            for field in 0..4 {
                let gradient = &token_gradients[field * 60..(field + 1) * 60];
                for index in [2, 3, 22, 23] { assert_eq!(gradient[index], 0.0); }
                for range in [5..20, 25..60] { assert!(gradient[range].iter().all(|&value| value == 0.0)); }
            }
            // Padded queries are zero even when a caller accidentally lists them as known.
            let padding = GoldCombinations::from_host(vec![3], vec![0, 0, 0, 0, 0], vec![0], 1, device);
            close(host(&call.coupling_loss_from_call(&padding, &supplied).0)[0], 0.0, 0.0);
            let empty = GoldCombinations::from_host(vec![], vec![], vec![], 0, device);
            let (empty_loss, groups) = call.coupling_loss_from_call(&empty, &supplied);
            assert_eq!(groups, 0); assert_eq!(host(&empty_loss), vec![0.0]);
            let logs = call.slot_view(0, 0..1).conditional_log_marginals(&supplied.slot_view(0, 0..1), true);
            for field in 0..4 {
                let values = host(&logs.1[field]);
                assert_eq!(values[2], f64::NEG_INFINITY); assert_eq!(values[3], f64::NEG_INFINITY);
                close(values.iter().map(|value| value.exp()).sum(), 1.0, 3e-5);
            }
            (value, parameter_gradient)
        };
        let (clean_loss, clean_gradients) = evaluate(false);
        let (dirty_loss, dirty_gradients) = evaluate(true);
        close(dirty_loss, clean_loss, 3e-5);
        for (actual, expected) in dirty_gradients.into_iter().zip(clean_gradients) { close(actual, expected, 3e-5); }
    }

    fn impossible_evidence_case(device: &Device) {
        let model = make_model(&parameters(), device);
        let (_, context, inputs) = fixture_with_inputs(&model, device);
        let tables = model.endpoint_tables(&context);
        for case in 0..4 {
            let relation = if case == 0 { tensor(&[0.0, f64::NEG_INFINITY, -1.0], [1, 1, 3], device) }
                else { inputs.relation.clone() };
            let tokens: [Tensor<3>; 4] = std::array::from_fn(|field| {
                let logits = match (case, field) {
                    (1, 0) => [0.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
                    (1, 1) => [f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY],
                    (2, 0) => [f64::NEG_INFINITY; 3],
                    _ => LOGITS[field + 1],
                };
                tensor(&logits, [1, 1, 3], device)
            });
            // No NaN input is involved: case 0 supplies a zero-probability label,
            // case 1 contradicts two forced endpoints, case 2 has an empty leaf,
            // and case 3 supplies a coordinate outside the field's value axis.
            assert!(host(&relation).iter().all(|value| !value.is_nan()));
            assert!(tokens.iter().all(|token| host(token).iter().all(|value| !value.is_nan())));
            let supplied = match case {
                0 => evidence([1, 0, 0, 0, 0], [true, false, false, false, false], device),
                1 => evidence([0, 1, 0, 0, 0], [false, true, true, false, false], device),
                2 => CircuitEvidence::empty(1, 1, device),
                _ => evidence([0, i32::MAX as usize, 0, 0, 0], [false, true, false, false, false], device),
            };
            let call = model.call(inputs.hidden.clone(), relation, &tokens, &tables, &Tensor::ones([1, 1], device));
            let prepared = call.prepared_decisions(&supplied);
            for message in prepared.emissions.iter().chain(&prepared.forward).chain(&prepared.backward) {
                assert!(host(message).iter().all(|value| !value.is_nan() && *value != f64::INFINITY));
            }
            let decoded = prepared.greedy_decode();
            let choices = host_int(&decoded);
            assert!(choices.iter().all(|&value| (0..3).contains(&value)), "{choices:?}");
            match case {
                0 => assert_eq!(choices[0], 1, "valid supplied fields remain fixed even at zero probability"),
                1 => assert_eq!(&choices[1..3], &[1, 0]),
                3 => assert_eq!(choices[1], 0),
                _ => {}
            }
            assert!(host(&prepared.dependency_features(&decoded)).iter().all(|value| value.is_finite()));
            let garbage = int(vec![i32::MAX; 5], [1, 1, 5], device);
            assert!(host(&prepared.dependency_features(&garbage)).iter().all(|value| value.is_finite()));
            for exclude in [false, true] {
                let (relation, endpoints) = prepared.conditional_log_marginals(exclude);
                for logs in std::iter::once(&relation).chain(endpoints.iter()) {
                    assert!(host(logs).iter().all(|value| !value.is_nan() && *value != f64::INFINITY));
                    assert!(host(&logs.clone().exp()).iter().all(|value| value.is_finite()));
                }
            }
            let gold = GoldCombinations::from_host(vec![0], choices, vec![0], 1, device);
            assert!(host(&call.coupling_loss_from_call(&gold, &supplied).0)[0].is_finite());
            let invalid_gold = GoldCombinations::from_host(vec![0], vec![i32::MAX as i64; 5], vec![0], 1, device);
            assert_eq!(host(&call.coupling_loss_from_call(&invalid_gold, &supplied).0), vec![0.0]);
        }
    }

    #[test]
    fn joint_circuit_dense_fp64_normalization_conditionals_greedy_features_and_groups_cpu() {
        enumeration_case(&Device::ndarray().autodiff());
    }
    #[test]
    fn joint_circuit_value_major_shared_relation_table_handles_batch_query_and_state_axes_cpu() {
        shared_relation_case(&Device::ndarray().autodiff());
    }
    #[test]
    fn joint_circuit_parameter_and_marginal_gradients_match_fp64_finite_differences_cpu() {
        gradient_case(&Device::ndarray().autodiff());
    }
    #[test]
    fn joint_circuit_small_random_import_is_near_product_and_zero_terms_are_exact_cpu() {
        identity_case(&Device::ndarray().autodiff());
    }
    #[test]
    fn joint_circuit_padding_and_unknown_rows_have_exact_zero_loss_and_gradients_cpu() {
        padding_case(&Device::ndarray().autodiff());
    }
    #[test]
    fn joint_circuit_zero_probability_conflicting_and_empty_evidence_keep_indices_valid_cpu() {
        impossible_evidence_case(&Device::ndarray().autodiff());
    }
    #[test]
    #[should_panic(expected = "gold alternatives must be a set")]
    fn joint_circuit_gold_groups_reject_duplicate_alias_tuples() {
        let device = Device::ndarray();
        GoldCombinations::from_host(vec![0, 0], vec![0; 10], vec![0, 0], 1, &device);
    }
    #[test]
    #[should_panic(expected = "one query per gold group")]
    fn joint_circuit_gold_groups_reject_mixed_query_owners() {
        let device = Device::ndarray();
        GoldCombinations::from_host(vec![0, 1], vec![0; 10], vec![0, 0], 1, &device);
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn joint_circuit_dense_fp64_conditionals_and_coupling_cuda() {
        if std::env::var("DRE_TEST_GPU").as_deref() != Ok("1") { return; }
        enumeration_case(&dre_core::gpu_device().autodiff());
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn joint_circuit_fp64_finite_difference_gradients_cuda() {
        if std::env::var("DRE_TEST_GPU").as_deref() != Ok("1") { return; }
        gradient_case(&dre_core::gpu_device().autodiff());
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn joint_circuit_dirty_padding_unknown_rows_and_identity_cuda() {
        if std::env::var("DRE_TEST_GPU").as_deref() != Ok("1") { return; }
        let device = dre_core::gpu_device().autodiff();
        padding_case(&device);
        identity_case(&device);
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn joint_circuit_value_major_shared_relation_table_handles_batch_query_and_state_axes_cuda() {
        if std::env::var("DRE_TEST_GPU").as_deref() != Ok("1") { return; }
        shared_relation_case(&dre_core::gpu_device().autodiff());
    }
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn joint_circuit_zero_probability_conflicting_and_empty_evidence_keep_indices_valid_cuda() {
        if std::env::var("DRE_TEST_GPU").as_deref() != Ok("1") { return; }
        impossible_evidence_case(&dre_core::gpu_device().autodiff());
    }
    #[test]
    fn a5_span_decode_sums_unknown_relation_before_choosing_endpoints_cpu() {
        let device=Device::ndarray().autodiff();let mut theta=vec![0.0;78];
        theta[..6].copy_from_slice(&[5.0,-5.0,-5.0,5.0,-5.0,-5.0]);
        let first_y=-6.4/2.62;let first_x=-4.0-2.2*first_y;
        let second_y=5.0/(1.1+0.5*1.3/0.6);let second_x=1.3/0.6*second_y;
        theta[6..12].copy_from_slice(&[first_x,second_x,first_y,second_y,
            2.0-0.2*first_x+0.8*first_y,-2.0-0.2*second_x+0.8*second_y]);
        theta[30]=0.51f64.ln();theta[31]=0.49f64.ln();
        for edge in 0..4 {theta[32+edge*4..36+edge*4].copy_from_slice(&[7.0,-7.0,-7.0,7.0]);}
        let model=make_model(&theta,&device);let (call,_)=fixture(&model,&device);let dense=Dense::new(&theta);
        let supplied=evidence([0;5],[false;5],&device);let prepared=call.prepared_decisions(&supplied);
        let decoded=host_int(&prepared.greedy_decode_mentions());
        let mut expected=[0;5];let mut known=[false;5];
        for field in 1..5 {
            let log=dense.marginal(field,expected,known,false);
            expected[field]=(0..3).max_by(|&left,&right|log[left].total_cmp(&log[right])).unwrap();
            known[field]=true;
        }
        assert_eq!(&decoded[1..],&expected[1..].iter().map(|&value|value as i64).collect::<Vec<_>>());
        let relation_first=host_int(&prepared.greedy_decode());
        assert_ne!(decoded[1],relation_first[1],"fixture must distinguish marginalization from a relation-first branch");
        assert!(!known[0],"the relation never becomes an observation while selecting spans");
    }
}
