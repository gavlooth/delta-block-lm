//! Exact conditional block distributions with separately updated probability rows.
use anyhow::{Result, ensure};
use burn::{
    module::{Initializer, Param, ParamId},
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation::log_softmax,
};
use serde::{Deserialize, Serialize};

pub const STATES: usize = 32;
pub const VOCAB: usize = 257;
pub const GROUP_WIDTH: usize = 4;
const MOMENTUM: f64 = 0.9;
const STEP_SIZE: f64 = 0.05;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodpulaExecution {
    #[default]
    Tensor,
    FusedGpu,
}
impl CodpulaExecution {
    pub fn is_tensor(&self) -> bool { *self == Self::Tensor }
}

/// All fields are persistent buffers, never neural optimizer parameters.
#[derive(Module, Debug)]
pub struct CircuitState {
    pub log_emission: Param<Tensor<2>>,
    pub log_transition: Param<Tensor<2>>,
    pub log_root: Param<Tensor<1>>,
    pub emission_positive_ema: Param<Tensor<2>>,
    pub emission_negative_ema: Param<Tensor<2>>,
    pub transition_positive_ema: Param<Tensor<2>>,
    pub transition_negative_ema: Param<Tensor<2>>,
    pub root_positive_ema: Param<Tensor<1>>,
    pub root_negative_ema: Param<Tensor<1>>,
    pub step: Param<Tensor<1, Int>>,
}

#[derive(Module, Debug)]
pub struct Codpula {
    pub gate_head: Linear,
    pub probability: CircuitState,
    #[module(skip)]
    execution: CodpulaExecution,
}

#[derive(Clone, Debug)]
pub struct CircuitFlows {
    pub emission_positive: Tensor<2>,
    pub emission_negative: Tensor<2>,
    pub transition_positive: Tensor<2>,
    pub transition_negative: Tensor<2>,
    pub root_positive: Tensor<1>,
    pub root_negative: Tensor<1>,
    pub emission_reach: Tensor<1>,
    pub transition_reach: Tensor<1>,
    pub root_reach: Tensor<1>,
    /// Statistics are sums until update divides by this count exactly once.
    pub active_blocks: usize,
}
impl CircuitFlows {
    pub fn add(&mut self, other: Self) {
        self.emission_positive = self.emission_positive.clone() + other.emission_positive;
        self.emission_negative = self.emission_negative.clone() + other.emission_negative;
        self.transition_positive = self.transition_positive.clone() + other.transition_positive;
        self.transition_negative = self.transition_negative.clone() + other.transition_negative;
        self.root_positive = self.root_positive.clone() + other.root_positive;
        self.root_negative = self.root_negative.clone() + other.root_negative;
        self.emission_reach = self.emission_reach.clone() + other.emission_reach;
        self.transition_reach = self.transition_reach.clone() + other.transition_reach;
        self.root_reach = self.root_reach.clone() + other.root_reach;
        self.active_blocks += other.active_blocks;
    }
}

pub struct CircuitLoss {
    pub loss: Tensor<1>,
    pub flows: CircuitFlows,
}

/// Neural outputs and local normalizers remain fixed across all evidence queries.
#[derive(Clone, Debug)]
pub struct PreparedCircuit {
    pub log_emission: Tensor<2>,
    pub log_transition: Tensor<3>,
    pub log_root: Tensor<1>,
    /// Normalized per-block roots for the contextual profile; legacy roots stay shared.
    pub contextual_log_root: Option<Tensor<2>>,
    pub logits: Tensor<3>,
    pub leaf_normalizers: Tensor<3>,
    pub lengths: Vec<usize>,
    pub valid: Tensor<2>,
    execution: CodpulaExecution,
}

#[derive(Clone, Debug)]
pub struct EvidenceQuery {
    pub log_z: Tensor<1>,
    pub state: Tensor<3>,
    /// Entry i is the edge entering position i; entry zero is always zero.
    pub edge: Tensor<4>,
    pub token: Tensor<3>,
    pub emission_counts: Tensor<4>,
}

struct Messages {
    log_z: Tensor<1>,
    incoming: Vec<Tensor<2>>,
    forward: Vec<Tensor<2>>,
    backward: Vec<Tensor<2>>,
    evidence: Vec<Tensor<2>>,
    state: Tensor<3>,
    edge: Tensor<4>,
}

struct Statistics {
    log_z: Tensor<1>,
    state: Tensor<3>,
    summed_edge: Tensor<3>,
}

fn frozen<const D: usize>(tensor: Tensor<D>) -> Param<Tensor<D>> {
    Param::from_tensor(tensor.detach()).set_require_grad(false)
}
fn finite<const D: usize>(tensor: &Tensor<D>, name: &str) -> Result<()> {
    ensure!(tensor.clone().is_finite().all().into_scalar::<bool>(), "nonfinite {name}");
    Ok(())
}
fn finite_or_negative_infinity<const D: usize>(tensor: &Tensor<D>, name: &str) -> Result<()> {
    let supported = tensor.clone().mask_fill(tensor.clone().equal_elem(f32::NEG_INFINITY), 0.0);
    finite(&supported, name)
}
fn nonnegative<const D: usize>(tensor: &Tensor<D>, name: &str) -> Result<()> {
    finite(tensor, name)?;
    ensure!(tensor.clone().greater_equal_elem(0.0).all().into_scalar::<bool>(), "negative {name}");
    Ok(())
}
fn lse<const D: usize>(value: Tensor<D>, dim: usize) -> Tensor<D> {
    let dims = value.dims();
    let maximum = value.clone().max_dim(dim).detach();
    let absent = maximum.clone().equal_elem(f32::NEG_INFINITY);
    let maximum = maximum.mask_fill(absent.clone(), 0.0);
    let total = (value.mask_fill(absent.clone().expand(dims), 0.0) - maximum.clone())
        .exp().sum_dim(dim).mask_fill(absent.clone(), 1.0);
    (total.log() + maximum).mask_fill(absent, f32::NEG_INFINITY)
}
fn tensor_values<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>> {
    tensor.into_data().try_to_vec::<f32>().map_err(|e| anyhow::anyhow!("reading circuit values: {e:?}"))
}
fn int_values<const D: usize>(tensor: Tensor<D, Int>) -> Result<Vec<i32>> {
    tensor.into_data().try_to_vec_as::<i32>().map_err(|e| anyhow::anyhow!("reading circuit tokens: {e:?}"))
}
fn bool_values<const D: usize>(tensor: Tensor<D, Bool>) -> Result<Vec<bool>> {
    tensor.into_data().try_to_vec_as::<bool>().map_err(|e| anyhow::anyhow!("reading circuit evidence: {e:?}"))
}
fn valid_tensor(lengths: &[usize], width: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(lengths.iter().flat_map(|&length| (0..width).map(move |i| if i < length { 1.0f32 } else { 0.0 })).collect::<Vec<_>>(), [lengths.len(), width]), device)
}

/// A local generator keeps circuit initialization and decoding independent of device RNG order.
struct Generator(u64);
pub(crate) fn sample_log_categorical(logs: &[f64], uniform: f64) -> Result<usize> {
    ensure!(uniform.is_finite() && (0.0..1.0).contains(&uniform), "invalid categorical uniform");
    ensure!(!logs.is_empty() && logs.iter().all(|weight| !weight.is_nan() && *weight != f64::INFINITY),
        "invalid sampling distribution");
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure!(maximum.is_finite(), "categorical distribution has zero support");
    let mut total = 0.0;
    for &weight in logs {
        if weight.is_finite() { total += (weight - maximum).exp(); }
    }
    ensure!(total.is_finite() && total > 0.0, "invalid sampling mass");
    let draw = uniform * total;
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (index, &weight) in logs.iter().enumerate() {
        if weight.is_finite() {
            let probability = (weight - maximum).exp();
            if probability > 0.0 {
                last_positive = Some(index);
                cumulative += probability;
                if draw < cumulative { return Ok(index); }
            }
        }
    }
    last_positive.ok_or_else(|| anyhow::anyhow!("invalid sampling mass"))
}

impl Generator {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut bits = self.0;
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
        bits ^= bits >> 31;
        ((bits >> 11) as f64 + 0.5) / 9007199254740992.0
    }
    fn categorical(&mut self, logs: &[f64]) -> Result<usize> {
        sample_log_categorical(logs, self.next())
    }
}

impl Codpula {
    pub fn new(d_model: usize, seed: u64, device: &Device) -> Self {
        let mut random = Generator(seed);
        let mut row = |rows: usize, columns: usize| {
            let data = (0..rows * columns).map(|_| ((random.next() - 0.5) * 0.02) as f32).collect::<Vec<_>>();
            log_softmax(Tensor::from_data(TensorData::new(data, [rows, columns]), device), 1)
        };
        let log_emission = row(STATES, VOCAB);
        let log_transition = row(STATES, STATES);
        let log_root = row(1, STATES).reshape([STATES]);
        Self {
            gate_head: LinearConfig::new(d_model, 64).with_bias(true).with_initializer(Initializer::Zeros).init(device),
            probability: CircuitState::from_logs(log_emission, log_transition, log_root),
            execution: CodpulaExecution::Tensor,
        }
    }
    /// Contextual b4 transitions and roots share one zero-initialized neural head.
    pub fn new_contextual(d_model: usize, seed: u64, device: &Device) -> Self {
        let mut random = Generator(seed);
        let mut row = |rows: usize, columns: usize, half_width: f64| {
            let data = (0..rows * columns).map(|_| ((random.next() - 0.5) * (2.0 * half_width)) as f32).collect::<Vec<_>>();
            log_softmax(Tensor::from_data(TensorData::new(data, [rows, columns]), device), 1)
        };
        let log_emission = row(STATES, VOCAB, 0.25);
        let log_transition = row(STATES, STATES, 0.1);
        let log_root = row(1, STATES, 0.01).reshape([STATES]);
        Self {
            gate_head: LinearConfig::new(3 * d_model, 64 + STATES).with_bias(true).with_initializer(Initializer::Zeros).init(device),
            probability: CircuitState::from_logs(log_emission, log_transition, log_root),
            execution: CodpulaExecution::Tensor,
        }
    }
    pub fn is_contextual(&self) -> bool { self.gate_head.weight.val().dims()[1] == 64 + STATES }
    /// Concatenate valid mean, first valid feature, and last valid feature.
    /// Lengths describe prefix-valid rows; entirely inactive rows yield all zeros.
    pub fn context_summary(&self, features: Tensor<3>, lengths: &[usize]) -> Result<Tensor<2>> {
        let [rows, width, depth] = features.dims();
        ensure!(rows > 0 && rows == lengths.len() && width > 0 && lengths.iter().all(|&n| n <= width), "invalid context pooling lengths");
        let valid = valid_tensor(lengths, width, &features.device());
        let features = features.mask_fill(valid.clone().equal_elem(0.0).reshape([rows, width, 1]).expand([rows, width, depth]), 0.0);
        finite(&features, "valid context features")?;
        let counts = valid.sum_dim(1).clamp_min(1.0);
        let mean = features.clone().sum_dim(1).reshape([rows, depth]) / counts;
        let first = features.clone().narrow(1, 0, 1).reshape([rows, depth]);
        let last_ids = Tensor::<2, Int>::from_data(TensorData::new(
            lengths.iter().map(|&length| length.saturating_sub(1) as i32).collect::<Vec<_>>(), [rows, 1]), &features.device())
            .reshape([rows, 1, 1]).expand([rows, 1, depth]);
        let last = features.gather(1, last_ids).reshape([rows, depth]);
        Ok(Tensor::cat(vec![mean, first, last], 1))
    }
    pub fn contextual_gates(&self, features: Tensor<3>, lengths: &[usize]) -> Result<(Tensor<3>, Tensor<2>)> {
        ensure!(self.is_contextual(), "contextual gates require the contextual profile");
        let rows = features.dims()[0];
        let gates = self.gate_head.forward(self.context_summary(features, lengths)?);
        Ok((gates.clone().narrow(1, 0, 64).reshape([rows, 8, 8]), gates.narrow(1, 64, STATES)))
    }
    pub fn with_execution(mut self, execution: CodpulaExecution) -> Self {
        self.execution = execution;
        self
    }
    pub fn set_execution(&mut self, execution: CodpulaExecution) { self.execution = execution; }
    /// Reapply after loading if a recorder or model mapper activates float parameters.
    pub fn freeze_state(&mut self) { self.probability.freeze_rows(); }
    pub fn gates(&self, features: Tensor<3>, lengths: &[usize]) -> Result<Tensor<3>> {
        if self.is_contextual() { return Ok(self.contextual_gates(features, lengths)?.0); }
        let [rows, width, depth] = features.dims();
        ensure!(rows == lengths.len() && width > 0 && lengths.iter().all(|&n| n <= width), "invalid gate pooling lengths");
        let valid = valid_tensor(lengths, width, &features.device());
        let features = features.mask_fill(valid.clone().equal_elem(0.0).reshape([rows, width, 1]).expand([rows, width, depth]), 0.0);
        finite(&features, "valid gate features")?;
        let counts = valid.sum_dim(1).clamp_min(1.0);
        let pooled = features.sum_dim(1).reshape([rows, depth]) / counts;
        Ok(self.gate_head.forward(pooled).reshape([rows, 8, 8]))
    }
    pub fn prepare(&self, logits: Tensor<3>, features: Tensor<3>, lengths: &[usize], observed: Tensor<2, Bool>, temperature: f64) -> Result<PreparedCircuit> {
        ensure!(temperature.is_finite() && temperature > 0.0, "invalid circuit temperature");
        let [rows, width, vocab] = logits.dims();
        ensure!(vocab == VOCAB && observed.dims() == [rows, width] && features.dims()[..2] == [rows, width], "invalid circuit neural output shape");
        // Only the token-potential path is detached. Observed features still train gates and backbone.
        let logits = logits.clone().mask_where(observed.reshape([rows, width, 1]).expand([rows, width, vocab]), logits.detach()) / temperature;
        if self.is_contextual() {
            let (gates, root_gates) = self.contextual_gates(features, lengths)?;
            PreparedCircuit::prepare_contextual_with_execution(
                self.probability.log_emission.val(), self.probability.log_transition.val(), self.probability.log_root.val(),
                logits, gates, root_gates, lengths.to_vec(), GROUP_WIDTH, self.execution,
            )
        } else {
            let gates = self.gates(features, lengths)?;
            PreparedCircuit::prepare_with_execution(
                self.probability.log_emission.val(), self.probability.log_transition.val(), self.probability.log_root.val(),
                logits, gates, lengths.to_vec(), GROUP_WIDTH, self.execution,
            )
        }
    }
    pub fn update(&mut self, flows: CircuitFlows) -> Result<()> { self.probability.update(flows) }
}

impl CircuitState {
    /// Generic state construction is also useful for independent small-model numerical checks.
    pub fn from_logs(log_emission: Tensor<2>, log_transition: Tensor<2>, log_root: Tensor<1>) -> Self {
        let [states, vocab] = log_emission.dims();
        let device = log_emission.device();
        Self {
            log_emission: frozen(log_softmax(log_emission, 1)),
            log_transition: frozen(log_softmax(log_transition, 1)),
            log_root: frozen(log_softmax(log_root, 0)),
            emission_positive_ema: frozen(Tensor::zeros([states, vocab], &device)),
            emission_negative_ema: frozen(Tensor::zeros([states, vocab], &device)),
            transition_positive_ema: frozen(Tensor::zeros([states, states], &device)),
            transition_negative_ema: frozen(Tensor::zeros([states, states], &device)),
            root_positive_ema: frozen(Tensor::zeros([states], &device)),
            root_negative_ema: frozen(Tensor::zeros([states], &device)),
            step: Param::initialized(ParamId::new(), Tensor::<1, Int>::zeros([1], &device)),
        }
    }
    pub fn freeze_rows(&mut self) {
        self.log_emission = self.log_emission.clone().set_require_grad(false);
        self.log_transition = self.log_transition.clone().set_require_grad(false);
        self.log_root = self.log_root.clone().set_require_grad(false);
        self.emission_positive_ema = self.emission_positive_ema.clone().set_require_grad(false);
        self.emission_negative_ema = self.emission_negative_ema.clone().set_require_grad(false);
        self.transition_positive_ema = self.transition_positive_ema.clone().set_require_grad(false);
        self.transition_negative_ema = self.transition_negative_ema.clone().set_require_grad(false);
        self.root_positive_ema = self.root_positive_ema.clone().set_require_grad(false);
        self.root_negative_ema = self.root_negative_ema.clone().set_require_grad(false);
    }
    pub fn update(&mut self, flows: CircuitFlows) -> Result<()> {
        if flows.active_blocks == 0 { return Ok(()); }
        let [states, vocab] = self.log_emission.val().dims();
        ensure!(flows.emission_positive.dims() == [states, vocab] && flows.emission_negative.dims() == [states, vocab]
            && flows.transition_positive.dims() == [states, states] && flows.transition_negative.dims() == [states, states]
            && flows.root_positive.dims() == [states] && flows.root_negative.dims() == [states]
            && flows.emission_reach.dims() == [states] && flows.transition_reach.dims() == [states] && flows.root_reach.dims() == [1], "invalid circuit flow shape");
        nonnegative(&flows.emission_positive, "emission positive flow")?;
        nonnegative(&flows.emission_negative, "emission negative flow")?;
        nonnegative(&flows.transition_positive, "transition positive flow")?;
        nonnegative(&flows.transition_negative, "transition negative flow")?;
        nonnegative(&flows.root_positive, "root positive flow")?;
        nonnegative(&flows.root_negative, "root negative flow")?;
        nonnegative(&flows.emission_reach, "emission reach")?;
        nonnegative(&flows.transition_reach, "transition reach")?;
        nonnegative(&flows.root_reach, "root reach")?;
        let previous_step = int_values(self.step.val())?[0];
        ensure!(previous_step >= 0 && previous_step < i32::MAX, "invalid circuit update time");
        let step = previous_step + 1;
        let divisor = flows.active_blocks as f64;
        let correction = 1.0 - MOMENTUM.powi(step);
        let ema = |old: Tensor<2>, value: Tensor<2>| old * MOMENTUM + value.detach() * ((1.0 - MOMENTUM) / divisor);
        let ep = ema(self.emission_positive_ema.val(), flows.emission_positive);
        let en = ema(self.emission_negative_ema.val(), flows.emission_negative);
        let tp = ema(self.transition_positive_ema.val(), flows.transition_positive);
        let tn = ema(self.transition_negative_ema.val(), flows.transition_negative);
        let rp = (self.root_positive_ema.val() * MOMENTUM + flows.root_positive.detach() * ((1.0 - MOMENTUM) / divisor)).reshape([1, states]);
        let rn = (self.root_negative_ema.val() * MOMENTUM + flows.root_negative.detach() * ((1.0 - MOMENTUM) / divisor)).reshape([1, states]);
        let emission = update_rows(self.log_emission.val(), ep.clone() / correction, en.clone() / correction, flows.emission_reach / divisor)?;
        let transition = update_rows(self.log_transition.val(), tp.clone() / correction, tn.clone() / correction, flows.transition_reach / divisor)?;
        let root = update_rows(self.log_root.val().reshape([1, states]), rp.clone() / correction, rn.clone() / correction, flows.root_reach / divisor)?.reshape([states]);
        // No state is written until all candidate rows pass finite validation.
        self.log_emission = self.log_emission.clone().map(|_| emission);
        self.log_transition = self.log_transition.clone().map(|_| transition);
        self.log_root = self.log_root.clone().map(|_| root);
        self.emission_positive_ema = self.emission_positive_ema.clone().map(|_| ep);
        self.emission_negative_ema = self.emission_negative_ema.clone().map(|_| en);
        self.transition_positive_ema = self.transition_positive_ema.clone().map(|_| tp);
        self.transition_negative_ema = self.transition_negative_ema.clone().map(|_| tn);
        self.root_positive_ema = self.root_positive_ema.clone().map(|_| rp.reshape([states]));
        self.root_negative_ema = self.root_negative_ema.clone().map(|_| rn.reshape([states]));
        let device = self.step.val().device();
        self.step = self.step.clone().map(|_| Tensor::from_data([step], &device));
        self.freeze_rows();
        Ok(())
    }
}

fn update_rows(log_weights: Tensor<2>, positive: Tensor<2>, negative: Tensor<2>, reach: Tensor<1>) -> Result<Tensor<2>> {
    finite(&log_weights, "base log rows")?;
    nonnegative(&positive, "positive flow average")?;
    nonnegative(&negative, "negative flow average")?;
    let [rows, columns] = log_weights.dims();
    let inactive = reach.clone().equal_elem(0.0).reshape([rows, 1]);
    let no_statistics = (positive.clone() + negative.clone()).sum_dim(1).equal_elem(0.0);
    let unchanged = inactive.clone().bool_or(no_statistics).expand([rows, columns]);
    let log_mass = log_weights.clone() + reach.mask_fill(inactive.reshape([rows]), 1.0).log().reshape([rows, 1]);
    let log_sum = |flow: Tensor<2>| {
        let other = flow.log() + STEP_SIZE.ln();
        let maximum = log_mass.clone().max_pair(other.clone());
        ((log_mass.clone() - maximum.clone()).exp() + (other - maximum.clone()).exp()).log() + maximum
    };
    let updated = log_softmax(log_weights.clone() + log_sum(positive) - log_sum(negative), 1).mask_where(unchanged, log_weights);
    finite(&updated, "updated log rows")?;
    Ok(updated.detach())
}

impl PreparedCircuit {
    pub fn from_log_potentials(log_pi: Tensor<2>, log_a: Tensor<2>, log_rho: Tensor<1>, logits: Tensor<3>, log_gates: Tensor<3>, lengths: Vec<usize>, group_size: usize) -> Result<Self> {
        Self::prepare_with_execution(log_pi, log_a, log_rho, logits, log_gates, lengths, group_size, CodpulaExecution::Tensor)
    }
    pub fn prepare_with_execution(log_pi: Tensor<2>, log_a: Tensor<2>, log_rho: Tensor<1>, logits: Tensor<3>, log_gates: Tensor<3>, lengths: Vec<usize>, group_size: usize, execution: CodpulaExecution) -> Result<Self> {
        let [rows, width, vocab] = logits.dims();
        let [states, emission_vocab] = log_pi.dims();
        ensure!(rows > 0 && width > 0 && states > 0 && vocab > 0 && emission_vocab == vocab && log_a.dims() == [states, states] && log_rho.dims() == [states], "invalid circuit dimensions");
        ensure!(lengths.len() == rows && lengths.iter().all(|&length| length <= width), "invalid circuit lengths");
        ensure!(group_size > 0 && states % group_size == 0 && log_gates.dims() == [rows, states / group_size, states / group_size], "invalid circuit gate dimensions");
        finite(&log_pi, "base emission logs")?;
        finite(&log_a, "base transition logs")?;
        finite(&log_rho, "base root logs")?;
        let device = logits.device();
        let valid = valid_tensor(&lengths, width, &device);
        let logits = logits.mask_fill(valid.clone().equal_elem(0.0).reshape([rows, width, 1]).expand([rows, width, vocab]), 0.0);
        finite_or_negative_infinity(&logits, "valid token potentials")?;
        let inactive_rows = valid.clone().sum_dim(1).equal_elem(0.0).reshape([rows, 1, 1]);
        let log_gates = log_gates.mask_fill(inactive_rows, 0.0);
        finite(&log_gates, "coupling gates")?;
        let ids = Tensor::<1, Int>::from_data(TensorData::new((0..states).map(|s| (s / group_size) as i32).collect::<Vec<_>>(), [states]), &device);
        let gates = log_gates.select(1, ids.clone()).select(2, ids);
        let log_transition = log_softmax(log_a.reshape([1, states, states]) + gates, 2);
        let log_root = log_softmax(log_rho, 0);
        let log_emission = log_softmax(log_pi, 1);
        let leaf_normalizers = if execution.is_tensor() {
            let mut normalizers = Vec::with_capacity(width);
            for i in 0..width {
                let position = logits.clone().slice([0..rows, i..i + 1, 0..vocab]).reshape([rows, 1, vocab]);
                normalizers.push(lse(position + log_emission.clone().reshape([1, states, vocab]), 2).reshape([rows, 1, states]));
            }
            Tensor::cat(normalizers, 1).mask_fill(valid.clone().equal_elem(0.0).reshape([rows, width, 1]), 0.0)
        } else {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            { delta_kernels::codpula_leaf_normalizers(logits.clone(), log_emission.clone(), valid.clone()) }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            { anyhow::bail!("fused circuit execution requires a GPU feature") }
        };
        finite(&leaf_normalizers, "leaf normalizers")?;
        Ok(Self { log_emission, log_transition, log_root, contextual_log_root: None, logits, leaf_normalizers, lengths, valid, execution })
    }
    /// Generic tensor-reference API with per-block log-root modulation.
    pub fn prepare_contextual(log_emission: Tensor<2>, log_transition: Tensor<2>, log_root: Tensor<1>, logits: Tensor<3>, gates: Tensor<3>, root_gates: Tensor<2>, lengths: Vec<usize>, group: usize) -> Result<Self> {
        Self::prepare_contextual_with_execution(log_emission, log_transition, log_root, logits, gates, root_gates, lengths, group, CodpulaExecution::Tensor)
    }
    fn prepare_contextual_with_execution(log_emission: Tensor<2>, log_transition: Tensor<2>, log_root: Tensor<1>, logits: Tensor<3>, gates: Tensor<3>, root_gates: Tensor<2>, lengths: Vec<usize>, group: usize, execution: CodpulaExecution) -> Result<Self> {
        let [rows, _, _] = logits.dims();
        let states = log_root.dims()[0];
        ensure!(root_gates.dims() == [rows, states], "invalid circuit root gate dimensions");
        let mut prepared = Self::prepare_with_execution(log_emission, log_transition, log_root, logits, gates, lengths, group, execution)?;
        let inactive = prepared.valid.clone().sum_dim(1).equal_elem(0.0).expand([rows, states]);
        let root_gates = root_gates.mask_fill(inactive, 0.0);
        finite(&root_gates, "root gates")?;
        prepared.contextual_log_root = Some(log_softmax(prepared.log_root.clone().reshape([1, states]) + root_gates, 1));
        Ok(prepared)
    }
    /// Actual roots used by every evidence query and unconditional traversal.
    pub fn root_rows(&self) -> Tensor<2> {
        self.contextual_log_root.clone().unwrap_or_else(|| {
            let (rows, _, states, _) = self.shape();
            self.log_root.clone().reshape([1, states]).expand([rows, states])
        })
    }
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    fn kernel_root(&self) -> Tensor<1> {
        match &self.contextual_log_root {
            Some(root) => root.clone().flatten(0, 1),
            None => self.log_root.clone(),
        }
    }
    fn shape(&self) -> (usize, usize, usize, usize) {
        let [rows, width, vocab] = self.logits.dims();
        (rows, width, self.log_root.dims()[0], vocab)
    }
    fn leaf(&self, position: usize) -> Tensor<3> {
        let (rows, _, states, vocab) = self.shape();
        (self.logits.clone().narrow(1, position, 1).reshape([rows, 1, vocab])
            + self.log_emission.clone().reshape([1, states, vocab])
            - self.leaf_normalizers.clone().narrow(1, position, 1).reshape([rows, states, 1])).exp()
    }
    fn evidence(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>) -> Result<(Tensor<2, Int>, Tensor<2, Bool>, Vec<Tensor<2>>)> {
        let (rows, width, states, vocab) = self.shape();
        ensure!(values.dims() == [rows, width] && fixed.dims() == [rows, width], "invalid evidence shape");
        let fixed = fixed.bool_and(self.valid.clone().greater_elem(0.0));
        let invalid_ids = values.clone().lower_elem(0).bool_or(values.clone().greater_equal_elem(vocab as i32));
        ensure!(!fixed.clone().bool_and(invalid_ids).any().into_scalar::<bool>(), "evidence token outside circuit vocabulary");
        let values = values.mask_fill(fixed.clone().bool_not(), 0);
        let mut evidence = Vec::with_capacity(width);
        for i in 0..width {
            let ids = values.clone().narrow(1, i, 1);
            let token = self.logits.clone().narrow(1, i, 1).reshape([rows, vocab]).gather(1, ids.clone());
            let base = self.log_emission.clone().reshape([1, states, vocab]).expand([rows, states, vocab]).gather(2, ids.reshape([rows, 1, 1]).expand([rows, states, 1])).reshape([rows, states]);
            let log_leaf = base + token - self.leaf_normalizers.clone().narrow(1, i, 1).reshape([rows, states]);
            evidence.push(log_leaf.mask_fill(fixed.clone().narrow(1, i, 1).bool_not().expand([rows, states]), 0.0));
        }
        Ok((values, fixed, evidence))
    }
    fn messages(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>) -> Result<Messages> {
        let (rows, width, states, _) = self.shape();
        let (_, _, evidence) = self.evidence(values, fixed)?;
        let mut forward = Vec::with_capacity(width);
        let mut incoming = Vec::with_capacity(width);
        let mut previous = self.root_rows();
        for i in 0..width {
            let before = if i == 0 { previous.clone() } else {
                lse(previous.clone().reshape([rows, states, 1]) + self.log_transition.clone(), 1).reshape([rows, states])
            };
            incoming.push(before.clone());
            let invalid = self.valid.clone().narrow(1, i, 1).equal_elem(0.0).expand([rows, states]);
            let current = (before + evidence[i].clone()).mask_where(invalid, previous);
            forward.push(current.clone());
            previous = current;
        }
        let log_z = if self.execution.is_tensor() {
            lse(previous, 1).reshape([rows])
        } else {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            { delta_kernels::codpula_log_partition(Tensor::cat(evidence.iter().map(|x| x.clone().reshape([rows, 1, states])).collect(), 1), self.log_transition.clone(), self.kernel_root(), self.valid.clone()) }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            { anyhow::bail!("fused circuit execution requires a GPU feature") }
        };
        let log_z = log_z.mask_fill(self.valid.clone().sum_dim(1).equal_elem(0.0).reshape([rows]), 0.0);
        finite(&log_z, "evidence mass")?;
        let mut backward = Vec::with_capacity(width);
        let mut next = Tensor::zeros([rows, states], &self.logits.device());
        for i in (0..width).rev() {
            backward.push(next.clone());
            if i > 0 {
                let message = lse(self.log_transition.clone() + (evidence[i].clone() + next.clone()).reshape([rows, 1, states]), 2).reshape([rows, states]);
                next = message.mask_where(self.valid.clone().narrow(1, i, 1).equal_elem(0.0).expand([rows, states]), next);
            }
        }
        backward.reverse();
        let z = log_z.clone().reshape([rows, 1]);
        let mut posterior = Vec::with_capacity(width);
        let mut edge = Vec::with_capacity(width);
        for i in 0..width {
            let invalid = self.valid.clone().narrow(1, i, 1).equal_elem(0.0);
            posterior.push((forward[i].clone() + backward[i].clone() - z.clone()).exp().mask_fill(invalid.clone().expand([rows, states]), 0.0).reshape([rows, 1, states]));
            let xi = if i == 0 { Tensor::zeros([rows, states, states], &self.logits.device()) } else {
                (forward[i - 1].clone().reshape([rows, states, 1]) + self.log_transition.clone()
                    + (evidence[i].clone() + backward[i].clone()).reshape([rows, 1, states])
                    - z.clone().reshape([rows, 1, 1])).exp().mask_fill(invalid.reshape([rows, 1, 1]), 0.0)
            };
            edge.push(xi.reshape([rows, 1, states, states]));
        }
        Ok(Messages { log_z, incoming, forward, backward, evidence, state: Tensor::cat(posterior, 1), edge: Tensor::cat(edge, 1) })
    }
    fn statistics(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>) -> Result<Statistics> {
        let (rows, _, states, _) = self.shape();
        if self.execution.is_tensor() {
            let messages = self.messages(values, fixed)?;
            return Ok(Statistics {
                log_z: messages.log_z,
                state: messages.state.detach(),
                summed_edge: messages.edge.detach().sum_dim(1).reshape([rows, states, states]),
            });
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            let (_, _, evidence) = self.evidence(values, fixed)?;
            let evidence = Tensor::cat(evidence.into_iter().map(|x| x.reshape([rows, 1, states])).collect(), 1);
            let log_z = delta_kernels::codpula_log_partition(evidence.clone(), self.log_transition.clone(), self.kernel_root(), self.valid.clone());
            let log_z = log_z.mask_fill(self.valid.clone().sum_dim(1).equal_elem(0.0).reshape([rows]), 0.0);
            finite(&log_z, "evidence mass")?;
            let (_, state, summed_edge) = delta_kernels::codpula_chain_statistics(evidence.detach(), self.log_transition.clone().detach(), self.kernel_root().detach(), self.valid.clone());
            Ok(Statistics { log_z, state, summed_edge })
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        { anyhow::bail!("fused circuit execution requires a GPU feature") }
    }
    pub fn log_partition(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>) -> Result<Tensor<1>> {
        let (rows, width, states, _) = self.shape();
        let (_, _, evidence) = self.evidence(values, fixed)?;
        let result = if self.execution.is_tensor() {
            let mut previous = self.root_rows();
            for i in 0..width {
                let incoming = if i == 0 { previous.clone() } else {
                    lse(previous.clone().reshape([rows, states, 1]) + self.log_transition.clone(), 1).reshape([rows, states])
                };
                previous = (incoming + evidence[i].clone()).mask_where(
                    self.valid.clone().narrow(1, i, 1).equal_elem(0.0).expand([rows, states]), previous,
                );
            }
            lse(previous, 1).reshape([rows])
        } else {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            { delta_kernels::codpula_log_partition(Tensor::cat(evidence.into_iter().map(|x| x.reshape([rows, 1, states])).collect(), 1), self.log_transition.clone(), self.kernel_root(), self.valid.clone()) }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            { anyhow::bail!("fused circuit execution requires a GPU feature") }
        };
        let result = result.mask_fill(self.valid.clone().sum_dim(1).equal_elem(0.0).reshape([rows]), 0.0);
        finite(&result, "evidence mass")?;
        Ok(result)
    }
    fn token_statistics(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>, state: Tensor<3>, include_counts: bool) -> Result<(Tensor<3>, Option<Tensor<4>>)> {
        let (rows, width, states, vocab) = self.shape();
        let (values, fixed, _) = self.evidence(values, fixed)?;
        let mut tokens = Vec::with_capacity(width);
        let mut counts = Vec::with_capacity(if include_counts { width } else { 0 });
        for i in 0..width {
            let gamma = state.clone().narrow(1, i, 1).reshape([rows, states, 1]);
            let ids = values.clone().narrow(1, i, 1);
            let point = Tensor::<2>::zeros([rows, vocab], &self.logits.device()).scatter(1, ids, Tensor::ones([rows, 1], &self.logits.device()), burn::tensor::IndexingUpdateOp::Add);
            let leaf = self.leaf(i);
            let marginal = (gamma.clone() * leaf.clone()).sum_dim(1).reshape([rows, vocab]);
            tokens.push(marginal.mask_where(fixed.clone().narrow(1, i, 1).expand([rows, vocab]), point.clone()).mask_fill(self.valid.clone().narrow(1, i, 1).equal_elem(0.0).expand([rows, vocab]), 0.0).reshape([rows, 1, vocab]));
            if include_counts {
                let conditional_leaf = leaf.mask_where(fixed.clone().narrow(1, i, 1).reshape([rows, 1, 1]).expand([rows, states, vocab]), point.reshape([rows, 1, vocab]).expand([rows, states, vocab]));
                counts.push((gamma * conditional_leaf).reshape([rows, 1, states, vocab]));
            }
        }
        Ok((Tensor::cat(tokens, 1), if include_counts { Some(Tensor::cat(counts, 1)) } else { None }))
    }
    /// Complete differentiable evidence statistics, including token marginals.
    pub fn query(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>) -> Result<EvidenceQuery> {
        let message = self.messages(values.clone(), fixed.clone())?;
        let (token, counts) = self.token_statistics(values, fixed, message.state.clone(), true)?;
        Ok(EvidenceQuery { log_z: message.log_z, state: message.state, edge: message.edge, token, emission_counts: counts.expect("requested emission counts") })
    }
    /// Detached teacher/inference marginals; use `query` for marginal derivatives.
    pub fn marginals(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>) -> Result<Tensor<3>> {
        let statistics = self.statistics(values.clone(), fixed.clone())?;
        if self.execution.is_tensor() {
            return Ok(self.token_statistics(values, fixed, statistics.state, false)?.0.detach());
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            let (values, fixed, _) = self.evidence(values, fixed)?;
            Ok(delta_kernels::codpula_token_marginals(
                self.logits.clone().detach(), self.log_emission.clone().detach(), self.leaf_normalizers.clone().detach(),
                statistics.state, values.float(), fixed.float(), self.valid.clone(),
            ))
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        { anyhow::bail!("fused circuit execution requires a GPU feature") }
    }
    /// Conditional block losses sum and use the caller's existing clean-target denominator.
    pub fn conditional_nll(&self, clean: Tensor<2, Int>, observed: Tensor<2, Bool>, denominator: f64) -> Result<CircuitLoss> {
        ensure!(denominator.is_finite() && denominator > 0.0, "invalid joint loss denominator");
        let (rows, width, states, vocab) = self.shape();
        ensure!(observed.dims() == [rows, width], "invalid observed mask shape");
        let plus = self.statistics(clean.clone(), self.valid.clone().greater_elem(0.0))?;
        let minus = self.statistics(clean.clone(), observed.clone())?;
        let active = observed.clone().bool_not().bool_and(self.valid.clone().greater_elem(0.0))
            .float().sum_dim(1).greater_elem(0.0).float().reshape([rows]);
        let active_blocks = active.clone().sum().into_scalar::<f32>() as usize;
        let loss = (minus.log_z.clone() - plus.log_z.clone()).mask_fill(active.clone().equal_elem(0.0), 0.0).sum() / denominator;
        let clean_ids = clean.mask_fill(self.valid.clone().equal_elem(0.0), 0);
        let observed = observed.bool_and(self.valid.clone().greater_elem(0.0));
        let gamma_plus = plus.state.detach();
        let gamma_minus = minus.state.detach();
        let a = self.log_transition.clone().detach().exp();
        let active_emission = active.clone().reshape([rows, 1, 1]);
        let (emission_positive, emission_negative) = if self.execution.is_tensor() {
            let mut positive = Tensor::zeros([states, vocab], &self.logits.device());
            let mut negative = Tensor::zeros([states, vocab], &self.logits.device());
            for i in 0..width {
                let gp = gamma_plus.clone().narrow(1, i, 1).reshape([rows, states, 1]);
                let gm = gamma_minus.clone().narrow(1, i, 1).reshape([rows, states, 1]);
                let leaf = self.leaf(i).detach();
                let point = Tensor::<2>::zeros([rows, vocab], &self.logits.device()).scatter(1, clean_ids.clone().narrow(1, i, 1), Tensor::ones([rows, 1], &self.logits.device()), burn::tensor::IndexingUpdateOp::Add).reshape([rows, 1, vocab]);
                let minus_leaf = leaf.clone().mask_where(observed.clone().narrow(1, i, 1).reshape([rows, 1, 1]).expand([rows, states, vocab]), point.clone().expand([rows, states, vocab]));
                positive = positive + ((gp.clone() * point + gm.clone() * leaf.clone()) * active_emission.clone()).sum_dim(0).reshape([states, vocab]);
                negative = negative + ((gm * minus_leaf + gp * leaf) * active_emission.clone()).sum_dim(0).reshape([states, vocab]);
            }
            (positive, negative)
        } else {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                let counts = |gamma: Tensor<3>, fixed: Tensor<2>| delta_kernels::codpula_emission_flows(
                    self.logits.clone().detach(), self.log_emission.clone().detach(), self.leaf_normalizers.clone().detach(),
                    gamma * active.clone().reshape([rows, 1, 1]), clean_ids.clone().float(), fixed, self.valid.clone(),
                );
                let unresolved = Tensor::zeros([rows, width], &self.logits.device());
                (counts(gamma_plus.clone(), self.valid.clone()) + counts(gamma_minus.clone(), unresolved.clone()),
                 counts(gamma_minus.clone(), observed.clone().float()) + counts(gamma_plus.clone(), unresolved))
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            { anyhow::bail!("fused circuit execution requires a GPU feature") }
        };
        let xi_plus = plus.summed_edge;
        let xi_minus = minus.summed_edge;
        let transition_positive = ((xi_plus.clone() + xi_minus.clone().sum_dim(2) * a.clone()) * active_emission.clone()).sum_dim(0).reshape([states, states]);
        let transition_negative = ((xi_minus + xi_plus.sum_dim(2) * a) * active_emission.clone()).sum_dim(0).reshape([states, states]);
        let rho = match &self.contextual_log_root {
            Some(root) => root.clone().detach().exp(),
            None => self.log_root.clone().detach().exp().reshape([1, states]),
        };
        let root_positive = ((gamma_plus.narrow(1, 0, 1).reshape([rows, states]) + rho.clone()) * active.clone().reshape([rows, 1])).sum_dim(0).reshape([states]);
        let root_negative = ((gamma_minus.narrow(1, 0, 1).reshape([rows, states]) + rho) * active.clone().reshape([rows, 1])).sum_dim(0).reshape([states]);
        // Traversal is unconditional under each block's current, modulated transition.
        let prior = self.statistics(Tensor::<2, Int>::zeros([rows, width], &self.logits.device()), self.valid.clone().lower_elem(0.0))?;
        let reaches = prior.state * active.clone().reshape([rows, 1, 1]);
        let emission_reach = reaches.sum_dim(0).sum_dim(1).reshape([states]);
        let transition_reach = prior.summed_edge.sum_dim(2).reshape([rows, states]) * active.reshape([rows, 1]);
        let transition_reach = transition_reach.sum_dim(0).reshape([states]);
        let flows = CircuitFlows {
            emission_positive, emission_negative, transition_positive, transition_negative, root_positive, root_negative,
            emission_reach, transition_reach,
            root_reach: Tensor::from_data([active_blocks as f32], &self.logits.device()), active_blocks,
        };
        Ok(CircuitLoss { loss, flows })
    }
    /// Forward-filter/backward-sampling draws a joint latent path before emissions.
    pub fn sample(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>, seed: u64) -> Result<Vec<Vec<usize>>> {
        let mut random = Generator(seed);
        self.sample_using(values, fixed, |weights, _, _, _| random.categorical(weights))
    }
    /// Indexed exact sampling uses [row, position, (latent-state, token)] uniforms.
    /// State draws use their actual position even though FFBS visits them backward.
    pub fn sample_with_uniforms(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>, uniforms: Tensor<3>) -> Result<Vec<Vec<usize>>> {
        let (rows, width, _, _) = self.shape();
        ensure!(uniforms.dims() == [rows, width, 2], "invalid indexed circuit uniforms");
        let uniforms = tensor_values(uniforms)?;
        self.sample_using(values, fixed, |weights, row, position, token| {
            let channel = if token { 1 } else { 0 };
            let uniform = f64::from(uniforms[(row * width + position) * 2 + channel]);
            sample_log_categorical(weights, uniform)
        })
    }
    fn sample_using<U>(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>, mut categorical: U) -> Result<Vec<Vec<usize>>>
    where
        U: FnMut(&[f64], usize, usize, bool) -> Result<usize>,
    {
        let (rows, width, states, vocab) = self.shape();
        let message = self.messages(values.clone(), fixed.clone())?;
        let values = int_values(values)?;
        let fixed = bool_values(fixed)?;
        let alpha = message.forward.into_iter().map(tensor_values).collect::<Result<Vec<_>>>()?;
        let transition = tensor_values(self.log_transition.clone())?;
        let emission = tensor_values(self.log_emission.clone())?;
        let logits = tensor_values(self.logits.clone())?;
        let normalizers = tensor_values(self.leaf_normalizers.clone())?;
        let mut result = Vec::with_capacity(rows);
        let mut weights = vec![0.0; states.max(vocab)];
        for r in 0..rows {
            let length = self.lengths[r];
            if length == 0 { result.push(Vec::new()); continue; }
            let mut path = vec![0; length];
            for s in 0..states { weights[s] = f64::from(alpha[length - 1][r * states + s]); }
            path[length - 1] = categorical(&weights[..states], r, length - 1, false)?;
            for i in (0..length - 1).rev() {
                for s in 0..states {
                    weights[s] = f64::from(alpha[i][r * states + s])
                        + f64::from(transition[(r * states + s) * states + path[i + 1]]);
                }
                path[i] = categorical(&weights[..states], r, i, false)?;
            }
            let mut tokens = Vec::with_capacity(length);
            for i in 0..length {
                if fixed[r * width + i] { tokens.push(values[r * width + i] as usize); continue; }
                for v in 0..vocab {
                    weights[v] = f64::from(emission[path[i] * vocab + v])
                        + f64::from(logits[(r * width + i) * vocab + v])
                        - f64::from(normalizers[(r * width + i) * states + path[i]]);
                }
                tokens.push(categorical(&weights[..vocab], r, i, true)?);
            }
            result.push(tokens);
        }
        Ok(result)
    }
    /// Two message traversals give every singleton and leave-one-out evidence mass.
    pub fn dependence_scores(&self, values: Tensor<2, Int>, fixed: Tensor<2, Bool>, proposals: &[Vec<usize>]) -> Result<Vec<Vec<f64>>> {
        let (rows, width, _, vocab) = self.shape();
        ensure!(proposals.len() == rows && proposals.iter().zip(&self.lengths).all(|(p, &n)| p.len() == n && p.iter().all(|&v| v < vocab)), "invalid joint proposals");
        let original_fixed = bool_values(fixed.clone())?;
        let original_values = int_values(values.clone())?;
        let mut ids = vec![0i32; rows * width];
        for r in 0..rows {
            for i in 0..self.lengths[r] {
                ids[r * width + i] = if original_fixed[r * width + i] { original_values[r * width + i] } else { proposals[r][i] as i32 };
            }
        }
        let device = self.logits.device();
        let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [rows, width]), &device);
        let base = self.messages(values.clone(), fixed.clone())?;
        let complete = self.messages(ids.clone(), self.valid.clone().greater_elem(0.0))?;
        let all_mass = tensor_values(complete.log_z)?;
        let base_mass = base.log_z.reshape([rows, 1]);
        let mut scores = self.lengths.iter().map(|&n| vec![0.0; n]).collect::<Vec<_>>();
        for i in 0..width {
            // Incoming excludes this position's emission; no division by an emission is needed.
            let cavity = lse(complete.incoming[i].clone() + complete.backward[i].clone(), 1).reshape([rows]);
            let cavity = tensor_values(cavity)?;
            let singleton = lse(base.incoming[i].clone() + complete.evidence[i].clone() + base.backward[i].clone(), 1) - base_mass.clone();
            let singleton = tensor_values(singleton.reshape([rows]))?;
            for r in 0..rows {
                if i >= self.lengths[r] || original_fixed[r * width + i] { continue; }
                let count = (0..self.lengths[r]).filter(|&j| !original_fixed[r * width + j]).count();
                if count == 1 { continue; }
                let score = ((all_mass[r] as f64 - cavity[r] as f64) - singleton[r] as f64).abs();
                ensure!(score.is_finite(), "nonfinite dependence score");
                scores[r][i] = score;
            }
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_summary_ignores_padding_and_keeps_endpoint_gradients() {
        let device = Device::ndarray().autodiff();
        let module = Codpula::new_contextual(2, 7, &device);
        let features = Tensor::<3>::from_data(TensorData::new(vec![
            2.0, 4.0, 6.0, 8.0, f32::NAN, f32::NAN,
            f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN,
            10.0, 12.0, f32::NAN, f32::NAN, f32::NAN, f32::NAN,
        ], [3, 3, 2]), &device).require_grad();
        let summary = module.context_summary(features.clone(), &[2, 0, 1]).unwrap();
        assert_eq!(tensor_values(summary.clone()).unwrap(), vec![
            4.0, 6.0, 2.0, 4.0, 6.0, 8.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 12.0, 10.0, 12.0, 10.0, 12.0,
        ]);
        let grads = summary.sum().backward();
        assert_eq!(tensor_values(features.grad(&grads).unwrap()).unwrap(), vec![
            1.5, 1.5, 1.5, 1.5, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            3.0, 3.0, 0.0, 0.0, 0.0, 0.0,
        ]);
    }

    #[test]
    fn contextual_roots_change_marginals_and_receive_only_active_gradients() {
        let device = Device::ndarray().autodiff();
        let root_gates = Tensor::<2>::from_data([[3.0f32.ln(), 0.0], [f32::NAN, f32::NAN]], &device).require_grad();
        let prepared = PreparedCircuit::prepare_contextual(
            Tensor::<2>::from_data([[0.8f32.ln(), 0.2f32.ln()], [0.2f32.ln(), 0.8f32.ln()]], &device),
            Tensor::<2>::from_data([[0.7f32.ln(), 0.3f32.ln()], [0.4f32.ln(), 0.6f32.ln()]], &device),
            Tensor::<1>::from_data([0.5f32.ln(), 0.5f32.ln()], &device),
            Tensor::zeros([2, 2, 2], &device),
            Tensor::<3>::from_data(TensorData::new(vec![0.0, 0.0, 0.0, 0.0, f32::NAN, f32::NAN, f32::NAN, f32::NAN], [2, 2, 2]), &device),
            root_gates.clone(), vec![2, 0], 1,
        ).unwrap();
        let values = Tensor::<2, Int>::from_data([[0, 0], [99, 99]], &device);
        let unknown = Tensor::<2, Bool>::from_data([[false, false], [false, false]], &device);
        let marginal = tensor_values(prepared.marginals(values.clone(), unknown).unwrap()).unwrap();
        assert!((marginal[0] - 0.65).abs() < 2e-6);
        assert!((marginal[1] - 0.35).abs() < 2e-6);
        assert_eq!(&marginal[4..], &[0.0; 4]);
        let fixed = Tensor::<2, Bool>::from_data([[true, false], [true, true]], &device);
        let log_z = prepared.log_partition(values, fixed).unwrap();
        let grads = log_z.sum().backward();
        let gradient = tensor_values(root_gates.grad(&grads).unwrap()).unwrap();
        assert!((gradient[0] - (0.6 / 0.65 - 0.75)).abs() < 2e-6);
        assert!((gradient[1] + (0.6 / 0.65 - 0.75)).abs() < 2e-6);
        assert_eq!(&gradient[2..], &[0.0, 0.0]);
    }
}
