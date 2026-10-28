//! A separate mask-set model: it scores interventions, never corrects token logits.
//! Sampling uses ordered host f64 arithmetic; fitting uses detached trajectory features
//! and differentiable FP32 selected-term sums. No denoiser parameters are registered here.
use crate::{config::Config, data::{self, Rng}, model, runtime};
use anyhow::{Context, Result, ensure};
use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    optim::GradientsParams,
    prelude::*,
    tensor::{DType, activation::softplus},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

pub const HIDDEN: usize = 32;
pub const MAX_ELIGIBLE: usize = 16;
pub const EPSILON: f64 = 0.1;
pub const TEMPERATURE: f64 = 1.0;
pub const MARGIN: f64 = 1.0;
pub const SAMPLING_IDENTITY: &str = "ascending-mask-minimum-scaled-unary-dp-unary-then-lexicographic-pairs-f64-splitmix53-mixture-v1";
pub const PRECISION_IDENTITY: &str = "features-terms-training-f32-host-ordered-f64";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskPolicy {
    #[default]
    Uniform,
    Unary,
    Pairwise,
    Confidence,
}

#[derive(Module, Debug)]
pub struct MaskEnergy {
    pub features: Linear,
    pub unary: Linear,
    pub pair: Param<Tensor<2>>,
}
impl MaskEnergy {
    /// `width` is the complete feature width (2D+12), not the backbone width D.
    pub fn new(width: usize, device: &Device) -> Self {
        assert!(width > 0);
        Self {
            features: LinearConfig::new(width, HIDDEN).init(device),
            unary: LinearConfig::new(HIDDEN, 1).init(device),
            pair: Param::from_tensor(Tensor::zeros([HIDDEN, HIDDEN], device)),
        }
    }

    /// Project each state once, then form all scalar pairs with batched matmul.
    /// Invalid features are replaced before projection, including dirty NaN padding.
    /// Pair storage is O(C L²), with no per-candidate hidden-feature expansion.
    pub fn terms(&self, features: Tensor<3>, valid: Tensor<2, Bool>, pairwise: bool)
        -> (Tensor<2>, Tensor<3>)
    {
        let [states, length, width] = features.dims();
        assert_eq!(valid.dims(), [states, length]);
        assert_eq!(self.features.weight.dims()[0], width);
        let device = features.device();
        let inactive = valid.clone().bool_not();
        let features = features.cast(DType::F32)
            .mask_fill(inactive.clone().unsqueeze_dim::<3>(2).expand([states, length, width]), 0.0);
        let z = self.features.forward(features).tanh()
            .mask_fill(inactive.clone().unsqueeze_dim::<3>(2).expand([states, length, HIDDEN]), 0.0);
        let unary = self.unary.forward(z.clone()).reshape([states, length])
            .mask_fill(inactive, 0.0);
        if !pairwise {
            return (unary, Tensor::zeros([states, length, length], &device));
        }
        let w = self.pair.val().cast(DType::F32);
        let symmetric = (w.clone() + w.transpose()) * 0.5;
        let raw = z.clone().matmul(symmetric.unsqueeze_dim(0)).matmul(z.swap_dims(1, 2));
        // Averaging both orientations also makes the returned FP32 matrix bit-symmetric.
        let pairs = (raw.clone() + raw.swap_dims(1, 2)) * 0.5;
        let positions = Tensor::<1, Int>::arange(0..length as i64, &device);
        let diagonal = positions.clone().reshape([1, length, 1])
            .equal(positions.reshape([1, 1, length]));
        let invalid = valid.clone().unsqueeze_dim(2)
            .bool_and(valid.unsqueeze_dim(1)).bool_not().bool_or(diagonal);
        (unary, pairs.mask_fill(invalid.expand([states, length, length]), 0.0))
    }
}

pub fn load(width: usize, path: &Path, device: &Device) -> Result<MaskEnergy> {
    MaskEnergy::new(width, device).try_load_file(path)
        .with_context(|| format!("loading mask energy {} with feature width {width}", path.display()))
}

/// A guided run must reject a trained head from another frozen denoiser or feature
/// definition, even when tensor dimensions happen to match.
pub fn validate_checkpoint(path: &Path, width: usize, checkpoint_identity: &str, policy: MaskPolicy) -> Result<()> {
    let metadata: Value = serde_json::from_slice(&std::fs::read(path.with_extension("fit.json"))
        .context("trained mask energy requires its fit.json identity record")?)?;
    ensure!(metadata["schema"] == 1 && metadata["command"] == "energy-fit", "unsupported mask-energy fit record");
    ensure!(metadata["checkpoint_identity"].as_str() == Some(checkpoint_identity), "mask energy source checkpoint differs");
    ensure!(metadata["feature_width"].as_u64() == Some(width as u64)
        && metadata["feature_schema"].as_str() == Some(crate::remask::FEATURE_VERSION), "mask energy feature definition differs");
    ensure!(metadata["precision"].as_str() == Some(PRECISION_IDENTITY)
        && metadata["sampling_identity"].as_str() == Some(SAMPLING_IDENTITY), "mask energy precision or sampling identity differs");
    ensure!(serde_json::from_value::<MaskPolicy>(metadata["policy"].clone())? == policy, "mask energy was fitted for a different policy");
    ensure!(metadata["optimizer_updates"].as_u64().unwrap_or(0) > 0, "mask energy has no completed fitting updates");
    ensure!(metadata["energy_sha256"].as_str() == Some(crate::checkpoint::file_sha256(path)?.as_str()), "mask energy differs from its fit record");
    Ok(())
}

/// Literal predecessor rank-loss semantics, including sanitizing inactive energies
/// before subtraction. A tie contributes neither loss nor an energy gradient.
pub fn rank_loss(good: Tensor<1>, bad: Tensor<1>, eligible: Tensor<1>, margin: f64) -> Tensor<1> {
    assert!(margin.is_finite() && margin >= 0.0);
    let denominator = eligible.clone().sum().clamp_min(1.0);
    let inactive = eligible.clone().equal_elem(0.0);
    let good = good.mask_fill(inactive.clone(), 0.0);
    let bad = bad.mask_fill(inactive.clone(), 0.0);
    (softplus(good - bad + margin, 1.0).mask_fill(inactive, 0.0) * eligible).sum() / denominator
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankPair {
    pub good: Vec<usize>,
    pub bad: Vec<usize>,
    pub eligible: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankState {
    pub length: usize,
    pub features: Vec<f32>,
    pub pairs: Vec<RankPair>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub schema: u32,
    pub feature_width: usize,
    pub checkpoint_identity: String,
    pub states: Vec<RankState>,
    #[serde(default)]
    pub metadata: Value,
}

fn check_eligible(eligible: &[usize], r: usize) -> Result<()> {
    ensure!(eligible.len() <= MAX_ELIGIBLE, "exact mask sampling supports at most {MAX_ELIGIBLE} eligible positions");
    ensure!(r <= eligible.len(), "mask cardinality {r} exceeds {} eligible positions", eligible.len());
    ensure!(eligible.windows(2).all(|w| w[0] < w[1]), "eligible positions must be unique and ascending");
    Ok(())
}
fn check_terms(unary: &[f32], pairs: &[f32], eligible: &[usize], policy: MaskPolicy) -> Result<()> {
    if policy == MaskPolicy::Uniform { return Ok(()); }
    ensure!(eligible.iter().all(|&i| i < unary.len() && unary[i].is_finite()), "invalid eligible unary term");
    if policy == MaskPolicy::Pairwise {
        ensure!(unary.len().checked_mul(unary.len()) == Some(pairs.len()), "pair terms must be a dense L by L matrix");
        for (at, &i) in eligible.iter().enumerate() {
            for &j in &eligible[at + 1..] {
                ensure!(pairs[i * unary.len() + j].is_finite(), "nonfinite pair term ({i},{j})");
            }
        }
    }
    Ok(())
}
fn uniform(rng: &mut Rng) -> f64 { (rng.next() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0) }
fn unbiased_index(rng: &mut Rng, count: usize) -> usize {
    assert!(count > 0);
    let count = count as u64;
    let reject = count.wrapping_neg() % count;
    loop {
        let value = rng.next();
        if value >= reject { return (value % count) as usize; }
    }
}
fn choose(n: usize, r: usize) -> usize {
    if r > n { return 0; }
    let r = r.min(n - r);
    (0..r).fold(1, |value, k| value * (n - k) / (k + 1))
}
fn mask_set(mask: usize, eligible: &[usize]) -> Vec<usize> {
    eligible.iter().enumerate().filter_map(|(bit, &position)|
        ((mask >> bit) & 1 == 1).then_some(position)).collect()
}
fn uniform_set_at_index(eligible: &[usize], r: usize, mut index: usize) -> Vec<usize> {
    let mut selected = Vec::with_capacity(r);
    let mut remaining = r;
    // Integer-mask order, unranked in O(n) rather than scanning all 2^n masks.
    for bit in (0..eligible.len()).rev() {
        if remaining == 0 { break; }
        let without = choose(bit, remaining);
        if index >= without {
            selected.push(eligible[bit]);
            remaining -= 1;
            index -= without;
        }
    }
    assert_eq!(remaining, 0);
    assert_eq!(index, 0);
    selected.reverse();
    selected
}
fn uniform_set(eligible: &[usize], r: usize, rng: &mut Rng) -> Vec<usize> {
    uniform_set_at_index(eligible, r, unbiased_index(rng, choose(eligible.len(), r)))
}
fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let maximum = a.max(b);
    maximum + (-(a - b).abs()).exp().ln_1p()
}
struct UnaryPartition {
    minimum_energy: Vec<f64>,
    log_residual: Vec<f64>,
    stride: usize,
}
impl UnaryPartition {
    fn inclusion(&self, unary: f64, position: usize, remaining: usize) -> f64 {
        let current = position * self.stride + remaining;
        let suffix = (position + 1) * self.stride + remaining - 1;
        let energy_difference = unary + self.minimum_energy[suffix] - self.minimum_energy[current];
        (self.log_residual[suffix] - energy_difference / TEMPERATURE - self.log_residual[current]).exp().min(1.0)
    }
}
fn unary_dp(unary: &[f64], r: usize) -> UnaryPartition {
    let stride = r + 1;
    let entries = (unary.len() + 1) * stride;
    // Keep a minimum-energy offset per entry, not just one block offset. Otherwise
    // logaddexp(-f32::MAX,-f32::MAX) loses ln(2), and equal suffix choices cease
    // to be uniform. Each entry represents log Z = log_residual - minimum/T.
    let mut minimum_energy = vec![f64::INFINITY; entries];
    let mut log_residual = vec![f64::NEG_INFINITY; entries];
    minimum_energy[unary.len() * stride] = 0.0;
    log_residual[unary.len() * stride] = 0.0;
    for j in (0..unary.len()).rev() {
        minimum_energy[j * stride] = 0.0;
        log_residual[j * stride] = 0.0;
        for k in 1..=r.min(unary.len() - j) {
            let without = (j + 1) * stride + k;
            let with = without - 1;
            let exclude_energy = minimum_energy[without];
            let include_energy = unary[j] + minimum_energy[with];
            let minimum = exclude_energy.min(include_energy);
            let exclude = if log_residual[without] == f64::NEG_INFINITY { f64::NEG_INFINITY }
                else { log_residual[without] - (exclude_energy - minimum) / TEMPERATURE };
            let include = if log_residual[with] == f64::NEG_INFINITY { f64::NEG_INFINITY }
                else { log_residual[with] - (include_energy - minimum) / TEMPERATURE };
            minimum_energy[j * stride + k] = minimum;
            log_residual[j * stride + k] = logaddexp(exclude, include);
        }
    }
    UnaryPartition { minimum_energy, log_residual, stride }
}
/// Exact suffix-DP normalizer in host f64, useful independently of sampling.
pub fn unary_log_partition(unary: &[f64], r: usize) -> Result<f64> {
    ensure!(unary.len() <= MAX_ELIGIBLE && r <= unary.len(), "unsupported unary partition dimensions");
    ensure!(unary.iter().all(|x| x.is_finite()), "nonfinite unary partition input");
    let dp = unary_dp(unary, r);
    let value = dp.log_residual[r] - dp.minimum_energy[r] / TEMPERATURE;
    ensure!(value.is_finite(), "nonfinite unary log partition");
    Ok(value)
}
fn sample_unary(unary: &[f64], eligible: &[usize], r: usize, mut draw: impl FnMut() -> f64) -> Vec<usize> {
    let dp = unary_dp(unary, r);
    let mut remaining = r;
    let mut selected = Vec::with_capacity(r);
    for j in 0..eligible.len() {
        if remaining == 0 { break; }
        if remaining == eligible.len() - j {
            selected.extend_from_slice(&eligible[j..]);
            break;
        }
        let include = dp.inclusion(unary[j], j, remaining);
        if draw() < include { selected.push(eligible[j]); remaining -= 1; }
    }
    selected
}
/// Ordered final energy: all unary terms first, then unique lexicographic pairs.
pub fn ordered_energy(unary: &[f32], pairs: &[f32], selected: &[usize], pairwise: bool) -> Result<f64> {
    ensure!(selected.windows(2).all(|w| w[0] < w[1]), "selected positions must be unique and ascending");
    check_terms(unary, pairs, selected, if pairwise { MaskPolicy::Pairwise } else { MaskPolicy::Unary })?;
    let mut energy = 0.0;
    for &i in selected { energy += f64::from(unary[i]); }
    if pairwise {
        for (at, &i) in selected.iter().enumerate() {
            for &j in &selected[at + 1..] { energy += f64::from(pairs[i * unary.len() + j]); }
        }
    }
    ensure!(energy.is_finite(), "nonfinite accumulated mask energy");
    Ok(energy)
}
fn categorical(weights: &[f64], draw: f64) -> Result<usize> {
    ensure!(draw.is_finite() && (0.0..1.0).contains(&draw), "categorical uniform must be in [0,1)");
    ensure!(!weights.is_empty() && weights.iter().all(|x| x.is_finite() && *x >= 0.0), "invalid categorical weights");
    let total: f64 = weights.iter().sum();
    ensure!(total.is_finite() && total > 0.0, "categorical total must be finite and positive");
    let threshold = draw * total;
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if weight > 0.0 {
            last_positive = Some(index);
            if cumulative > threshold { return Ok(index); }
        }
    }
    last_positive.context("no positive categorical weight")
}
fn enumerated_weights(unary: &[f32], pairs: &[f32], eligible: &[usize], r: usize) -> Result<(Vec<usize>, Vec<f64>)> {
    let mut masks = Vec::with_capacity(choose(eligible.len(), r));
    let mut scores = Vec::with_capacity(masks.capacity());
    for mask in 0usize..1usize << eligible.len() {
        if mask.count_ones() as usize != r { continue; }
        masks.push(mask);
        let mut energy = 0.0;
        for (bit, &i) in eligible.iter().enumerate() {
            if mask & (1 << bit) != 0 { energy += f64::from(unary[i]); }
        }
        for (left, &i) in eligible.iter().enumerate() {
            if mask & (1 << left) == 0 { continue; }
            for (right, &j) in eligible.iter().enumerate().skip(left + 1) {
                if mask & (1 << right) != 0 { energy += f64::from(pairs[i * unary.len() + j]); }
            }
        }
        scores.push(-energy / TEMPERATURE);
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure!(maximum.is_finite(), "invalid pairwise log weights");
    for score in &mut scores { *score = (*score - maximum).exp(); }
    Ok((masks, scores))
}
fn sample_with_branch(unary: &[f32], pairs: &[f32], eligible: &[usize], r: usize,
    policy: MaskPolicy, rng: &mut Rng) -> Result<(Vec<usize>, bool)>
{
    ensure!(policy != MaskPolicy::Confidence, "confidence reveals do not use mask energies");
    check_eligible(eligible, r)?;
    if r == 0 { return Ok((Vec::new(), false)); }
    if r == eligible.len() { return Ok((eligible.to_vec(), false)); }
    check_terms(unary, pairs, eligible, policy)?;
    let explore = policy == MaskPolicy::Uniform || uniform(rng) < EPSILON;
    if explore { return Ok((uniform_set(eligible, r, rng), true)); }
    let selected = match policy {
        MaskPolicy::Uniform | MaskPolicy::Confidence => unreachable!(),
        MaskPolicy::Unary => {
            let local = eligible.iter().map(|&i| f64::from(unary[i])).collect::<Vec<_>>();
            sample_unary(&local, eligible, r, || uniform(rng))
        }
        MaskPolicy::Pairwise => {
            let (masks, weights) = enumerated_weights(unary, pairs, eligible, r)?;
            mask_set(masks[categorical(&weights, uniform(rng))?], eligible)
        }
    };
    Ok((selected, false))
}
/// Draw the normalized epsilon-uniform/energy mixture, not a successive softmax.
/// Trivial cardinalities consume no random values and do not inspect energy terms.
pub fn sample_set(unary: &[f32], pairs: &[f32], eligible: &[usize], r: usize,
    policy: MaskPolicy, rng: &mut Rng) -> Result<Vec<usize>>
{
    Ok(sample_with_branch(unary, pairs, eligible, r, policy, rng)?.0)
}

#[derive(Clone, Debug, Serialize)]
pub struct ReferenceDistribution {
    pub sets: Vec<Vec<usize>>,
    pub energies: Vec<f64>,
    pub learned_probabilities: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub log_partition: f64,
    pub positive_learned_sets: usize,
}
/// Independent f64 oracle: recursive combinations and dense indicator contractions,
/// rather than the production integer-mask enumeration or unary recurrence.
pub fn reference_distribution(unary: &[f64], pairs: &[f64], eligible: &[usize], r: usize,
    policy: MaskPolicy) -> Result<ReferenceDistribution>
{
    ensure!(policy != MaskPolicy::Confidence, "confidence reveals do not use mask energies");
    check_eligible(eligible, r)?;
    ensure!(eligible.iter().all(|&i| i < unary.len()), "reference position outside unary array");
    ensure!(policy != MaskPolicy::Pairwise || pairs.len() == unary.len() * unary.len(), "invalid reference pair shape");
    fn combinations(n: usize, r: usize, first: usize, current: &mut Vec<usize>, output: &mut Vec<Vec<usize>>) {
        if current.len() == r { output.push(current.clone()); return; }
        let missing = r - current.len();
        for i in first..=n - missing {
            current.push(i);
            combinations(n, r, i + 1, current, output);
            current.pop();
        }
    }
    let mut local_sets = Vec::new();
    combinations(eligible.len(), r, 0, &mut Vec::with_capacity(r), &mut local_sets);
    local_sets.sort_by_key(|set| set.iter().fold(0usize, |mask, &bit| mask | (1 << bit)));
    let mut sets = Vec::with_capacity(local_sets.len());
    let mut energies = Vec::with_capacity(local_sets.len());
    for local in local_sets {
        let mut member = vec![false; eligible.len()];
        for i in local { member[i] = true; }
        let mut value = 0.0;
        let mut selected = Vec::with_capacity(r);
        for (i, &position) in eligible.iter().enumerate() {
            if member[i] {
                selected.push(position);
                if policy != MaskPolicy::Uniform { value += unary[position]; }
            }
        }
        if policy == MaskPolicy::Pairwise {
            for i in 0..member.len() {
                for j in i + 1..member.len() {
                    if member[i] && member[j] { value += pairs[eligible[i] * unary.len() + eligible[j]]; }
                }
            }
        }
        ensure!(value.is_finite(), "nonfinite reference energy");
        sets.push(selected);
        energies.push(value);
    }
    let maximum = energies.iter().map(|x| -x / TEMPERATURE).fold(f64::NEG_INFINITY, f64::max);
    let weights = energies.iter().map(|x| (-x / TEMPERATURE - maximum).exp()).collect::<Vec<_>>();
    let total: f64 = weights.iter().sum();
    ensure!(total.is_finite() && total > 0.0, "invalid reference partition");
    let learned_probabilities = weights.iter().map(|x| x / total).collect::<Vec<_>>();
    let epsilon = if policy == MaskPolicy::Uniform { 1.0 } else { EPSILON };
    let probabilities = learned_probabilities.iter().map(|x| epsilon / sets.len() as f64 + (1.0 - epsilon) * x).collect();
    let positive_learned_sets = weights.iter().filter(|&&x| x > 0.0).count();
    Ok(ReferenceDistribution { sets, energies, learned_probabilities, probabilities,
        log_partition: maximum + total.ln(), positive_learned_sets })
}

/// Differentiable selected-term sums. State indices refer to the already projected
/// unique states; only scalar unary/pair entries are gathered for each candidate.
pub fn set_energies(unary: Tensor<2>, pairs: Tensor<3>, sets: &[(usize, &[usize])], pairwise: bool) -> Result<Tensor<1>> {
    let [states, length] = unary.dims();
    ensure!(pairs.dims() == [states, length, length], "selected pair tensor shape mismatch");
    ensure!(!sets.is_empty(), "no candidate sets to score");
    let device = unary.device();
    let maximum_terms = sets.iter().map(|(_, set)| set.len() + if pairwise { set.len() * set.len().saturating_sub(1) / 2 } else { 0 }).max().unwrap().max(1);
    let sentinel = states * length + states * length * length;
    ensure!(sentinel < i32::MAX as usize, "selected term indices exceed i32");
    let mut indices = vec![sentinel as i32; sets.len() * maximum_terms];
    for (candidate, &(state, set)) in sets.iter().enumerate() {
        ensure!(state < states && set.iter().all(|&i| i < length), "candidate position outside batched terms");
        ensure!(set.windows(2).all(|w| w[0] < w[1]), "candidate positions must be unique and ascending");
        let mut at = candidate * maximum_terms;
        for &i in set { indices[at] = (state * length + i) as i32; at += 1; }
        if pairwise {
            for (left, &i) in set.iter().enumerate() {
                for &j in &set[left + 1..] {
                    indices[at] = (states * length + state * length * length + i * length + j) as i32;
                    at += 1;
                }
            }
        }
    }
    let all = Tensor::cat(vec![unary.reshape([states * length]), pairs.reshape([states * length * length]), Tensor::zeros([1], &device)], 0);
    let indices = Tensor::<1, Int>::from_data(TensorData::new(indices, [sets.len() * maximum_terms]), &device);
    Ok(all.select(0, indices).reshape([sets.len(), maximum_terms]).sum_dim(1).reshape([sets.len()]))
}

fn check_fit_gradients(energy: &MaskEnergy, gradients: &GradientsParams, pairwise: bool) -> Result<()> {
    let mut norms = vec![
        gradients.get::<2>(energy.features.weight.id).context("missing fit feature-weight gradient")?.square().sum(),
        gradients.get::<1>(energy.features.bias.as_ref().unwrap().id).context("missing fit feature-bias gradient")?.square().sum(),
        gradients.get::<2>(energy.unary.weight.id).context("missing fit unary-weight gradient")?.square().sum(),
        gradients.get::<1>(energy.unary.bias.as_ref().unwrap().id).context("missing fit unary-bias gradient")?.square().sum(),
    ];
    if pairwise {
        norms.push(gradients.get::<2>(energy.pair.id).context("missing fit pair gradient")?.square().sum());
    } else {
        ensure!(gradients.get::<2>(energy.pair.id).is_none(), "disabled pair matrix received a fit gradient");
    }
    let squared_norm = model::scalar(Tensor::cat(norms, 0).sum())?;
    ensure!(squared_norm.is_finite(), "nonfinite energy parameter gradients");
    Ok(())
}

impl Dataset {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == 1, "unsupported mask trajectory schema {}", self.schema);
        ensure!(self.feature_width > 0 && !self.checkpoint_identity.is_empty(), "trajectory feature/checkpoint identity is missing");
        ensure!(!self.states.is_empty(), "mask trajectory dataset has no states");
        for (state_index, state) in self.states.iter().enumerate() {
            ensure!((1..=MAX_ELIGIBLE).contains(&state.length), "trajectory state {state_index} has unsupported length");
            ensure!(state.features.len() == state.length * self.feature_width && state.features.iter().all(|x| x.is_finite()), "trajectory state {state_index} has invalid frozen features");
            for pair in &state.pairs {
                ensure!(pair.eligible.is_finite() && pair.eligible >= 0.0, "invalid trajectory eligibility");
                ensure!(pair.good.len() == pair.bad.len(), "trajectory candidates have different cardinalities");
                for set in [&pair.good, &pair.bad] {
                    ensure!(set.iter().all(|&i| i < state.length) && set.windows(2).all(|w| w[0] < w[1]), "invalid trajectory candidate set");
                }
                ensure!(pair.eligible == 0.0 || pair.good != pair.bad, "duplicate trajectory candidates cannot be eligible");
            }
        }
        Ok(())
    }
}

/// Fit energy parameters only, from an immutable host dataset. This command starts
/// a fresh fit; its separate fit record explicitly records that no resume is used.
pub fn fit(c: &Config, device: &Device) -> Result<Value> {
    let refinement = c.sbd.as_ref().and_then(|s| s.refinement.as_ref()).context("energy-fit requires sbd.refinement")?;
    ensure!(matches!(refinement.policy,MaskPolicy::Unary | MaskPolicy::Pairwise),
        "energy fitting requires unary or pairwise remasking");
    ensure!(refinement.fit_steps > 0 && refinement.fit_batch > 0, "energy fit counts must be positive");
    let bytes = std::fs::read(&refinement.data_path).with_context(|| format!("reading {}", refinement.data_path.display()))?;
    let dataset: Dataset = serde_json::from_slice(&bytes)?;
    dataset.validate()?;
    ensure!(dataset.feature_width == 2 * c.d_model + 12, "dataset feature width differs from configured backbone");
    ensure!(dataset.metadata["feature_definition"].as_str() == Some(crate::remask::FEATURE_VERSION), "dataset feature definition is missing or unsupported");
    ensure!(dataset.metadata["source_model_identity"].as_str().is_some_and(|identity| !identity.is_empty()), "dataset source model identity is missing");
    let pairwise = refinement.policy == MaskPolicy::Pairwise;
    let index = dataset.states.iter().enumerate().flat_map(|(state, value)| (0..value.pairs.len()).map(move |pair| (state, pair))).collect::<Vec<_>>();
    ensure!(!index.is_empty(), "mask trajectory dataset has no rank pairs");
    ensure!(dataset.states.iter().flat_map(|state| &state.pairs).any(|pair| pair.eligible > 0.0), "mask trajectory dataset has no eligible rank supervision");
    let train_device = device.clone().autodiff();
    train_device.seed(c.seed);
    let mut energy = MaskEnergy::new(dataset.feature_width, &train_device);
    energy.pair = energy.pair.map(|value| value.set_require_grad(false));
    // Keep native AdamW state/settings, with the same deterministic clipping as model training.
    let mut optimizer: burn::optim::ModuleOptimizer =
        crate::update::TensorAdamW {decay:c.weight_decay,clip:c.gradient_clip}.into();
    let parts = if pairwise { 2 } else { 1 };
    let mut losses = Vec::with_capacity(parts * refinement.fit_steps);
    let mut part_reports = Vec::with_capacity(parts);
    let mut updates = 0;
    let mut eligible_weight = 0.0f64;
    for part in 0..parts {
        let use_pairs = part == 1;
        if use_pairs {
            // Unary parameters and their Adam state continue; pair parameters
            // enter at exactly zero and have not been optimized in the first part.
            energy.pair = energy.pair.map(|_| Tensor::zeros([HIDDEN, HIDDEN], &train_device).require_grad());
        }
        let first_loss = losses.len();
        let first_update = updates;
        for step in 0..refinement.fit_steps {
            let selected = (0..refinement.fit_batch.min(index.len())).map(|offset| index[(step * refinement.fit_batch + offset) % index.len()]).collect::<Vec<_>>();
            let mut unique = BTreeMap::new();
            for &(state, _) in &selected {
                let next = unique.len();
                unique.entry(state).or_insert(next);
            }
            let states = unique.len();
            let length = unique.keys().map(|&i| dataset.states[i].length).max().unwrap();
            let mut features = vec![0.0; states * length * dataset.feature_width];
            let mut valid = vec![false; states * length];
            for (&original, &local) in &unique {
                let state = &dataset.states[original];
                let first = local * length * dataset.feature_width;
                features[first..first + state.features.len()].copy_from_slice(&state.features);
                valid[local * length..local * length + state.length].fill(true);
            }
            let mut candidates = Vec::with_capacity(2 * selected.len());
            let mut weights = Vec::with_capacity(selected.len());
            for &(state, pair) in &selected {
                let pair = &dataset.states[state].pairs[pair];
                candidates.push((unique[&state], pair.good.as_slice()));
                candidates.push((unique[&state], pair.bad.as_slice()));
                weights.push(pair.eligible);
            }
            let weight: f64 = weights.iter().map(|&x| f64::from(x)).sum();
            eligible_weight += weight;
            if weight == 0.0 { losses.push(0.0); continue; }
            let (unary, pairs) = energy.terms(
                Tensor::from_data(TensorData::new(features, [states, length, dataset.feature_width]), &train_device),
                Tensor::from_data(TensorData::new(valid, [states, length]), &train_device), use_pairs);
            let energies = set_energies(unary, pairs, &candidates, use_pairs)?.reshape([selected.len(), 2]);
            let good = energies.clone().slice([0..selected.len(), 0..1]).reshape([selected.len()]);
            let bad = energies.slice([0..selected.len(), 1..2]).reshape([selected.len()]);
            let loss = rank_loss(good, bad, Tensor::from_data(TensorData::new(weights, [selected.len()]), &train_device), MARGIN);
            let value = model::scalar(loss.clone())?;
            ensure!(value.is_finite(), "nonfinite energy rank loss at fit step {step}");
            let gradients = GradientsParams::from_grads(loss.backward(), &energy);
            check_fit_gradients(&energy, &gradients, use_pairs)?;
            energy = optimizer.step(c.learning_rate, energy, gradients);
            losses.push(value);
            updates += 1;
        }
        ensure!(updates > first_update, "energy fit part received no eligible supervision; refusing an untrained head");
        part_reports.push(json!({"part":if use_pairs {"pairwise"} else {"unary"},
            "steps":refinement.fit_steps,"optimizer_updates":updates-first_update,
            "pair_matrix_enabled":use_pairs,"losses":&losses[first_loss..]}));
    }
    let path = refinement.energy_path.clone().unwrap_or_else(|| "artifacts/remasking/energy.burnpack".into());
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) { std::fs::create_dir_all(parent)?; }
    let pending = path.with_extension("burnpack.pending");
    energy.valid().save_file(&pending)?;
    std::fs::rename(&pending, &path)?;
    let optimizer_path = path.with_extension("optimizer.burnpack");
    let optimizer_pending = optimizer_path.with_extension("burnpack.pending");
    optimizer.save(&optimizer_pending)?;
    std::fs::rename(optimizer_pending, &optimizer_path)?;
    let report = json!({
        "schema":1,"command":"energy-fit","policy":refinement.policy,"seed":c.seed,
        "dataset_path":refinement.data_path,"dataset_sha256":data::hash(&bytes),
        "checkpoint_identity":dataset.checkpoint_identity,"feature_width":dataset.feature_width,
        "feature_schema":crate::remask::FEATURE_VERSION,"precision":PRECISION_IDENTITY,
        "source_model_identity":dataset.metadata["source_model_identity"],
        "dataset_metadata":dataset.metadata,"states":dataset.states.len(),"rank_pairs":index.len(),
        "steps":parts*refinement.fit_steps,"steps_per_part":refinement.fit_steps,"parts":part_reports,
        "optimizer_updates":updates,"batch_pairs":refinement.fit_batch,
        "eligible_weight_seen":eligible_weight,"margin":MARGIN,"losses":losses,
        "optimizer":crate::train::optimizer_config(c),"learning_rate":c.learning_rate,
        "energy_path":path,"energy_sha256":crate::checkpoint::file_sha256(&path)?,
        "optimizer_path":optimizer_path,"optimizer_sha256":crate::checkpoint::file_sha256(&optimizer_path)?,
        "continuation_supported":false,"frozen_denoiser":true,"feature_gradients":false,
        "sampling_identity":SAMPLING_IDENTITY,"device":runtime::identity(device)
    });
    runtime::write_json(&path.with_extension("fit.json"), &report)?;
    Ok(report)
}

fn compare(actual: &[f32], reference: &[f64]) -> Result<Value> { crate::sbd::compare(actual, reference) }
fn host<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>> {
    tensor.into_data().try_to_vec_as::<f32>().map_err(|error| anyhow::anyhow!("mask-energy readback: {error:?}"))
}

fn sampling_checks() -> Result<Value> {
    let mut maximum_partition_error = 0.0f64;
    let mut maximum_probability_error = 0.0f64;
    let mut maximum_marginal_error = 0.0f64;
    let mut maximum_count_error = 0.0f64;
    let mut maximum_log_partition_gradient_error = 0.0f64;
    let mut cases = 0;
    for n in 0..=MAX_ELIGIBLE {
        let unary = (0..n).map(|i| ((i * 11 + n * 3) % 17) as f32 * 0.125 - 1.0).collect::<Vec<_>>();
        let mut pairs = vec![0.0f32; n * n];
        for i in 0..n { for j in i + 1..n {
            let value = ((i * 7 + j * 13) % 11) as f32 * 0.0625 - 0.3125;
            pairs[i * n + j] = value; pairs[j * n + i] = value;
        }}
        let eligible = (0..n).collect::<Vec<_>>();
        let u = unary.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        let p = pairs.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        for r in 0..=n {
            cases += 1;
            let reference = reference_distribution(&u, &p, &eligible, r, MaskPolicy::Unary)?;
            let log_z = unary_log_partition(&u, r)?;
            maximum_partition_error = maximum_partition_error.max((log_z - reference.log_partition).abs());
            let mut marginal = vec![0.0; n];
            let mut expected = vec![0.0; n];
            let dp = unary_dp(&u, r);
            for (at, ((set, &energy), &probability)) in reference.sets.iter().zip(&reference.energies).zip(&reference.learned_probabilities).enumerate() {
                ensure!(uniform_set_at_index(&eligible, r, at) == *set, "uniform integer unranking differs from enumeration");
                let production = (-energy - log_z).exp();
                let mut path_probability = 1.0;
                let mut remaining = r;
                for j in 0..n {
                    if remaining == 0 || remaining == n - j { break; }
                    let inclusion = dp.inclusion(u[j], j, remaining);
                    if set.binary_search(&j).is_ok() { path_probability *= inclusion; remaining -= 1; }
                    else { path_probability *= 1.0 - inclusion; }
                }
                maximum_probability_error = maximum_probability_error.max((production - probability).abs()).max((path_probability - probability).abs());
                for &i in set { marginal[i] += production; expected[i] += probability; }
            }
            maximum_count_error = maximum_count_error.max((marginal.iter().sum::<f64>() - r as f64).abs());
            for i in 0..n {
                maximum_marginal_error = maximum_marginal_error.max((marginal[i] - expected[i]).abs());
                let h = 1e-5;
                let mut shifted = u.clone(); shifted[i] += h;
                let plus = unary_log_partition(&shifted, r)?;
                shifted[i] -= 2.0 * h;
                let minus = unary_log_partition(&shifted, r)?;
                maximum_log_partition_gradient_error = maximum_log_partition_gradient_error.max(((plus - minus) / (2.0 * h) + expected[i]).abs());
            }
            let reference = reference_distribution(&u, &p, &eligible, r, MaskPolicy::Pairwise)?;
            let (masks, weights) = enumerated_weights(&unary, &pairs, &eligible, r)?;
            let total: f64 = weights.iter().sum();
            for (at, (&mask, &weight)) in masks.iter().zip(&weights).enumerate() {
                ensure!(mask_set(mask, &eligible) == reference.sets[at], "pairwise enumeration ordering differs");
                maximum_probability_error = maximum_probability_error.max((weight / total - reference.learned_probabilities[at]).abs());
                let ordered = ordered_energy(&unary, &pairs, &reference.sets[at], true)?;
                ensure!(ordered.to_bits() == reference.energies[at].to_bits(), "ordered pair energy differs from independent f64 contraction");
            }
        }
    }
    ensure!(maximum_partition_error < 1e-11 && maximum_probability_error < 1e-11 && maximum_marginal_error < 1e-11 && maximum_count_error < 1e-10 && maximum_log_partition_gradient_error < 1e-8, "exact sampler reference disagreement");
    let unary = vec![-0.75f32, 0.25, 1.0, -0.125];
    let mut pairs = vec![0.0f32; 16];
    pairs[1] = -1.25; pairs[4] = -1.25; pairs[11] = 0.875; pairs[14] = 0.875;
    let u = unary.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let p = pairs.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let draws = 65_536usize;
    let alpha = 0.001f64;
    let comparisons = 3 * 6 + 2;
    let limit = ((2.0 * comparisons as f64 / alpha).ln() / (2.0 * draws as f64)).sqrt();
    let mut frequencies = Vec::new();
    let mut maximum_frequency_error = 0.0f64;
    for (policy, seed) in [(MaskPolicy::Uniform, 3107), (MaskPolicy::Unary, 8861), (MaskPolicy::Pairwise, 19237)] {
        let reference = reference_distribution(&u, &p, &[0, 1, 2, 3], 2, policy)?;
        let mut rng = Rng(seed);
        let mut counts = vec![0usize; reference.sets.len()];
        let mut exploration = 0;
        for _ in 0..draws {
            let (set, branch) = sample_with_branch(&unary, &pairs, &[0, 1, 2, 3], 2, policy, &mut rng)?;
            let at = reference.sets.iter().position(|candidate| candidate == &set).context("sample outside exact support")?;
            counts[at] += 1;
            exploration += usize::from(branch);
        }
        let mut error = 0.0f64;
        for (&count, &probability) in counts.iter().zip(&reference.probabilities) {
            error = error.max((count as f64 / draws as f64 - probability).abs());
        }
        maximum_frequency_error = maximum_frequency_error.max(error);
        ensure!(error <= limit, "fixed-seed {policy:?} distribution error {error} exceeds predeclared simultaneous Hoeffding limit {limit}");
        let expected_exploration = if policy == MaskPolicy::Uniform { 1.0 } else { EPSILON };
        let exploration_error = (exploration as f64 / draws as f64 - expected_exploration).abs();
        ensure!(exploration_error <= limit, "mixture branch frequency differs");
        frequencies.push(json!({"policy":policy,"seed":seed,"draws":draws,"sets":reference.sets,"counts":counts,
            "probabilities":reference.probabilities,"max_abs_frequency_error":error,"uniform_branch_draws":exploration,"uniform_branch_error":exploration_error}));
    }
    ensure!(categorical(&[1.0, 1.0, 2.0], 0.0)? == 0 && categorical(&[1.0, 1.0, 2.0], 0.25)? == 1 && categorical(&[1.0, 1.0, 2.0], 0.5)? == 2 && categorical(&[0.0, 1.0, 0.0], 1.0 - f64::EPSILON)? == 1, "strict categorical interval mapping differs");
    for weights in [&[0.0, 0.0][..], &[f64::NAN, 1.0], &[-1.0, 2.0], &[f64::INFINITY, 1.0]] {
        ensure!(categorical(weights, 0.5).is_err(), "invalid categorical weights accepted");
    }
    ensure!(categorical(&[1.0], 1.0).is_err() && categorical(&[1.0], -0.1).is_err(), "invalid categorical uniform accepted");
    let mut rng = Rng(27);
    let before = rng.0;
    ensure!(sample_set(&[], &[], &[2, 5], 0, MaskPolicy::Pairwise, &mut rng)?.is_empty(), "r0 support differs");
    ensure!(sample_set(&[], &[], &[2, 5], 2, MaskPolicy::Pairwise, &mut rng)? == [2, 5] && rng.0 == before, "trivial selection consumes random values");
    for policy in [MaskPolicy::Uniform, MaskPolicy::Unary, MaskPolicy::Pairwise] {
        ensure!(sample_set(&[], &[], &(0..17).collect::<Vec<_>>(), 1, policy, &mut rng).is_err(), "n17 was not rejected");
        ensure!(sample_set(&unary, &pairs, &[1, 0], 1, policy, &mut rng).is_err(), "unordered eligibility accepted");
        ensure!(sample_set(&unary, &pairs, &[0, 0], 1, policy, &mut rng).is_err(), "duplicate eligibility accepted");
        ensure!(sample_set(&unary, &pairs, &[0], 2, policy, &mut rng).is_err(), "impossible cardinality accepted");
    }
    ensure!(sample_set(&[f32::NAN, 0.0], &[], &[0, 1], 1, MaskPolicy::Unary, &mut rng).is_err(), "nonfinite unary accepted");
    let extreme = reference_distribution(&[f64::from(f32::MAX), -f64::from(f32::MAX), 0.0], &[0.0; 9], &[0, 1, 2], 1, MaskPolicy::Pairwise)?;
    ensure!(extreme.positive_learned_sets == 1 && extreme.probabilities.iter().all(|x| *x >= EPSILON / 3.0), "uniform support disappeared under learned-weight underflow");
    let mut shifted_rng = Rng(9137);
    let mut zero_rng = shifted_rng.clone();
    for _ in 0..256 {
        ensure!(sample_set(&[f32::MAX; 4], &[], &[0, 1, 2, 3], 2, MaskPolicy::Unary, &mut shifted_rng)?
            == sample_set(&[0.0; 4], &[], &[0, 1, 2, 3], 2, MaskPolicy::Unary, &mut zero_rng)?,
            "huge common unary offset changed the sampled law");
    }
    let huge_suffix = unary_dp(&[0.0, f64::from(f32::MAX), f64::from(f32::MAX)], 2);
    ensure!((huge_suffix.inclusion(f64::from(f32::MAX), 1, 1) - 0.5).abs() < 1e-15,
        "huge equal suffix weights lost their multiplicity");
    let cancellation = ordered_energy(&[16_777_216.0, 1.0, -16_777_216.0], &[], &[0, 1, 2], false)?;
    ensure!(cancellation == 1.0 && cancellation > ordered_energy(&[0.5], &[], &[0], false)?, "predecessor ordered-f64 cancellation fixture failed");
    let mut draws_used = 0;
    ensure!(sample_unary(&[0.0; 3], &[0, 1, 2], 2, || { draws_used += 1; 0.0 }) == [0, 1] && draws_used == 2, "unary inclusion map differs");
    draws_used = 0;
    ensure!(sample_unary(&[0.0; 3], &[0, 1, 2], 2, || { draws_used += 1; 1.0 - f64::EPSILON }) == [1, 2] && draws_used == 1, "unary deterministic suffix differs");
    let pl_probability: f64 = 1.0 / 6.0 * 2.0 / 5.0 + 2.0 / 6.0 * 1.0 / 4.0;
    let fixed_set_probability: f64 = 2.0 / 11.0;
    ensure!((fixed_set_probability - pl_probability).abs() > 0.03, "successive-softmax counterexample disappeared");
    Ok(json!({"passed":true,"cases_all_n0_to16_and_all_r":cases,"max_log_partition_error":maximum_partition_error,
        "max_probability_error":maximum_probability_error,"max_marginal_error":maximum_marginal_error,
        "max_expected_count_error":maximum_count_error,"max_log_partition_gradient_error":maximum_log_partition_gradient_error,
        "frequencies":frequencies,"max_abs_frequency_error":maximum_frequency_error,
        "family_wise_alpha":alpha,"simultaneous_comparisons":comparisons,"simultaneous_hoeffding_limit":limit,
        "categorical_strict_interval_checks":true,"invalid_input_checks":true,"trivial_no_rng_checks":true,
        "underflow_positive_learned_sets":extreme.positive_learned_sets,"underflow_mixture_probabilities":extreme.probabilities,
        "ordered_f64_cancellation_energy":cancellation,"successive_softmax_counterexample":{"fixed_set":fixed_set_probability,"successive_softmax":pl_probability}}))
}

#[derive(Clone)]
struct OracleParameters {
    feature_weight: Vec<f64>,
    feature_bias: Vec<f64>,
    unary_weight: Vec<f64>,
    unary_bias: Vec<f64>,
    pair: Vec<f64>,
}
impl OracleParameters {
    fn zeros(width: usize) -> Self {
        Self { feature_weight: vec![0.0; width * HIDDEN], feature_bias: vec![0.0; HIDDEN],
            unary_weight: vec![0.0; HIDDEN], unary_bias: vec![0.0; 1], pair: vec![0.0; HIDDEN * HIDDEN] }
    }
    fn vectors(&self) -> [&[f64]; 5] {
        [&self.feature_weight, &self.feature_bias, &self.unary_weight, &self.unary_bias, &self.pair]
    }
    fn vector_mut(&mut self, index: usize) -> &mut Vec<f64> {
        match index {
            0 => &mut self.feature_weight, 1 => &mut self.feature_bias,
            2 => &mut self.unary_weight, 3 => &mut self.unary_bias, 4 => &mut self.pair,
            _ => unreachable!(),
        }
    }
}
struct OracleOutput {
    unary: Vec<f64>,
    pairs: Vec<f64>,
    features_gradient: Vec<f64>,
    gradient: OracleParameters,
}

/// Scalar f64 forward and full reverse derivative, with no tensor operations and
/// no calls to the production projections, selected gathers or autodiff.
fn oracle(parameters: &OracleParameters, features: &[f64], valid: &[bool],
    rows: usize, length: usize, width: usize, unary_adjoint: &[f64], pair_adjoint: &[f64]) -> OracleOutput
{
    let positions = rows * length;
    let mut z = vec![0.0; positions * HIDDEN];
    let mut unary = vec![0.0; positions];
    let mut pairs = vec![0.0; rows * length * length];
    let mut dz = vec![0.0; positions * HIDDEN];
    let mut gradient = OracleParameters::zeros(width);
    let mut features_gradient = vec![0.0; features.len()];
    let mut symmetric = vec![0.0; HIDDEN * HIDDEN];
    for a in 0..HIDDEN { for b in 0..HIDDEN {
        symmetric[a * HIDDEN + b] = (parameters.pair[a * HIDDEN + b] + parameters.pair[b * HIDDEN + a]) * 0.5;
    }}
    for i in 0..positions {
        if !valid[i] { continue; }
        for a in 0..HIDDEN {
            let mut projection = parameters.feature_bias[a];
            for k in 0..width { projection += features[i * width + k] * parameters.feature_weight[k * HIDDEN + a]; }
            z[i * HIDDEN + a] = projection.tanh();
        }
        unary[i] = parameters.unary_bias[0];
        gradient.unary_bias[0] += unary_adjoint[i];
        for a in 0..HIDDEN {
            unary[i] += z[i * HIDDEN + a] * parameters.unary_weight[a];
            gradient.unary_weight[a] += unary_adjoint[i] * z[i * HIDDEN + a];
            dz[i * HIDDEN + a] += unary_adjoint[i] * parameters.unary_weight[a];
        }
    }
    for row in 0..rows { for i in 0..length { for j in i + 1..length {
        let left = row * length + i;
        let right = row * length + j;
        if !valid[left] || !valid[right] { continue; }
        let mut value = 0.0;
        let coefficient = pair_adjoint[(row * length + i) * length + j]
            + pair_adjoint[(row * length + j) * length + i];
        for a in 0..HIDDEN { for b in 0..HIDDEN {
            value += z[left * HIDDEN + a] * symmetric[a * HIDDEN + b] * z[right * HIDDEN + b];
            dz[left * HIDDEN + a] += coefficient * symmetric[a * HIDDEN + b] * z[right * HIDDEN + b];
            dz[right * HIDDEN + b] += coefficient * symmetric[a * HIDDEN + b] * z[left * HIDDEN + a];
            let derivative = coefficient * z[left * HIDDEN + a] * z[right * HIDDEN + b] * 0.5;
            gradient.pair[a * HIDDEN + b] += derivative;
            gradient.pair[b * HIDDEN + a] += derivative;
        }}
        pairs[(row * length + i) * length + j] = value;
        pairs[(row * length + j) * length + i] = value;
    }}}
    for i in 0..positions {
        if !valid[i] { continue; }
        for a in 0..HIDDEN {
            let derivative = dz[i * HIDDEN + a] * (1.0 - z[i * HIDDEN + a].powi(2));
            gradient.feature_bias[a] += derivative;
            for k in 0..width {
                gradient.feature_weight[k * HIDDEN + a] += features[i * width + k] * derivative;
                features_gradient[i * width + k] += parameters.feature_weight[k * HIDDEN + a] * derivative;
            }
        }
    }
    OracleOutput { unary, pairs, features_gradient, gradient }
}
fn oracle_objective(output: &OracleOutput, unary_adjoint: &[f64], pair_adjoint: &[f64]) -> f64 {
    output.unary.iter().zip(unary_adjoint).map(|(x, w)| x * w).sum::<f64>()
        + output.pairs.iter().zip(pair_adjoint).map(|(x, w)| x * w).sum::<f64>()
}
fn reference_rank(good: &[f64], bad: &[f64], eligible: &[f64], margin: f64) -> (f64, Vec<f64>, Vec<f64>) {
    let denominator = eligible.iter().sum::<f64>().max(1.0);
    let mut loss = 0.0;
    let mut dg = vec![0.0; good.len()];
    let mut db = vec![0.0; bad.len()];
    for i in 0..good.len() {
        if eligible[i] == 0.0 { continue; }
        let value = good[i] - bad[i] + margin;
        loss += (value.max(0.0) + (-value.abs()).exp().ln_1p()) * eligible[i] / denominator;
        let sigmoid = if value >= 0.0 { 1.0 / (1.0 + (-value).exp()) } else { value.exp() / (1.0 + value.exp()) };
        dg[i] = sigmoid * eligible[i] / denominator;
        db[i] = -dg[i];
    }
    (loss, dg, db)
}
fn fixture_parameters(width: usize) -> OracleParameters {
    fn values(count: usize, multiplier: usize, divisor: f32) -> Vec<f64> {
        (0..count).map(|i| f64::from(((i * multiplier % 31) as f32 - 15.0) / divisor)).collect()
    }
    OracleParameters {
        feature_weight: values(width * HIDDEN, 7, 128.0),
        feature_bias: values(HIDDEN, 11, 256.0),
        unary_weight: values(HIDDEN, 13, 64.0),
        unary_bias: vec![0.125],
        // Deliberately asymmetric before symmetrization.
        pair: values(HIDDEN * HIDDEN, 17, 512.0),
    }
}
fn fixture_model(parameters: &OracleParameters, width: usize, device: &Device) -> MaskEnergy {
    fn tensor<const D: usize>(values: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
        Tensor::from_data(TensorData::new(values.iter().map(|&x| x as f32).collect::<Vec<_>>(), shape), device).require_grad()
    }
    let mut energy = MaskEnergy::new(width, device);
    energy.features.weight = energy.features.weight.map(|_| tensor(&parameters.feature_weight, [width, HIDDEN], device));
    energy.features.bias = energy.features.bias.map(|bias| bias.map(|_| tensor(&parameters.feature_bias, [HIDDEN], device)));
    energy.unary.weight = energy.unary.weight.map(|_| tensor(&parameters.unary_weight, [HIDDEN, 1], device));
    energy.unary.bias = energy.unary.bias.map(|bias| bias.map(|_| tensor(&parameters.unary_bias, [1], device)));
    energy.pair = energy.pair.map(|_| tensor(&parameters.pair, [HIDDEN, HIDDEN], device));
    energy
}
fn gradient_reports(energy: &MaskEnergy, gradients: &GradientsParams, expected: &OracleParameters) -> Result<Value> {
    Ok(json!({
        "feature_weight":compare(&host(gradients.get::<2>(energy.features.weight.id).context("missing energy feature-weight gradient")?)?, &expected.feature_weight)?,
        "feature_bias":compare(&host(gradients.get::<1>(energy.features.bias.as_ref().unwrap().id).context("missing energy feature-bias gradient")?)?, &expected.feature_bias)?,
        "unary_weight":compare(&host(gradients.get::<2>(energy.unary.weight.id).context("missing energy unary-weight gradient")?)?, &expected.unary_weight)?,
        "unary_bias":compare(&host(gradients.get::<1>(energy.unary.bias.as_ref().unwrap().id).context("missing energy unary-bias gradient")?)?, &expected.unary_bias)?,
        "symmetric_pair":compare(&host(gradients.get::<2>(energy.pair.id).context("missing energy pair gradient")?)?, &expected.pair)?
    }))
}
fn reference_set_energy(unary: &[f64], pairs: &[f64], row: usize, length: usize, set: &[usize]) -> f64 {
    let mut value = 0.0;
    for &i in set { value += unary[row * length + i]; }
    for (left, &i) in set.iter().enumerate() { for &j in &set[left + 1..] {
        value += pairs[(row * length + i) * length + j];
    }}
    value
}
fn numerical_device(device: &Device) -> Result<Value> {
    let device = device.clone().autodiff();
    let (rows, length, width) = (2, 4, 7);
    let parameters = fixture_parameters(width);
    let valid = vec![true, true, true, false, true, true, false, false];
    let features = (0..rows * length * width).map(|i| {
        if valid[i / width] { ((i * 19 % 37) as f32 - 18.0) / 32.0 } else { f32::NAN }
    }).collect::<Vec<_>>();
    let f64_features = features.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let unary_adjoint = (0..rows * length).map(|i| (i as f64 - 2.5) / 8.0).collect::<Vec<_>>();
    let pair_adjoint = (0..rows * length * length).map(|i| ((i * 7 % 17) as f64 - 8.0) / 16.0).collect::<Vec<_>>();
    let expected = oracle(&parameters, &f64_features, &valid, rows, length, width, &unary_adjoint, &pair_adjoint);
    let energy = fixture_model(&parameters, width, &device);
    let input = Tensor::<3>::from_data(TensorData::new(features.clone(), [rows, length, width]), &device).require_grad();
    let validity = Tensor::<2, Bool>::from_data(TensorData::new(valid.clone(), [rows, length]), &device);
    let (unary, pairs) = energy.terms(input.clone(), validity.clone(), true);
    let unary_values = host(unary.clone())?;
    let pair_values = host(pairs.clone())?;
    let forward_unary = compare(&unary_values, &expected.unary)?;
    let forward_pairs = compare(&pair_values, &expected.pairs)?;
    for row in 0..rows { for i in 0..length { for j in 0..length {
        let left = (row * length + i) * length + j;
        let right = (row * length + j) * length + i;
        ensure!(pair_values[left].to_bits() == pair_values[right].to_bits(), "FP32 pair terms are not exactly symmetric");
        if i == j || !valid[row * length + i] || !valid[row * length + j] {
            ensure!(pair_values[left] == 0.0, "diagonal or invalid pair is nonzero");
        }
    }}}
    let objective = (unary * Tensor::from_data(TensorData::new(unary_adjoint.iter().map(|&x| x as f32).collect::<Vec<_>>(), [rows, length]), &device)).sum()
        + (pairs * Tensor::from_data(TensorData::new(pair_adjoint.iter().map(|&x| x as f32).collect::<Vec<_>>(), [rows, length, length]), &device)).sum();
    let raw_gradients = objective.backward();
    let feature_vjp = compare(&host(input.grad(&raw_gradients).context("missing mask-energy feature VJP")?)?, &expected.features_gradient)?;
    let gradients = GradientsParams::from_grads(raw_gradients, &energy);
    let parameter_vjp = gradient_reports(&energy, &gradients, &expected.gradient)?;

    let mut central_actual = Vec::new();
    let mut central_expected = Vec::new();
    let mut shifted = parameters.clone();
    let h = 1e-5;
    for group in 0..5 {
        let count = parameters.vectors()[group].len();
        let positions = if count == 1 { vec![0] } else { vec![0, count / 2, count - 1] };
        for at in positions {
            let saved = shifted.vector_mut(group)[at];
            shifted.vector_mut(group)[at] = saved + h;
            let plus = oracle_objective(&oracle(&shifted, &f64_features, &valid, rows, length, width, &unary_adjoint, &pair_adjoint), &unary_adjoint, &pair_adjoint);
            shifted.vector_mut(group)[at] = saved - h;
            let minus = oracle_objective(&oracle(&shifted, &f64_features, &valid, rows, length, width, &unary_adjoint, &pair_adjoint), &unary_adjoint, &pair_adjoint);
            shifted.vector_mut(group)[at] = saved;
            central_actual.push((plus - minus) / (2.0 * h));
            central_expected.push(expected.gradient.vectors()[group][at]);
        }
    }
    for at in [0, width + 2, (rows * length - 3) * width + width - 1] {
        let mut input = f64_features.clone();
        input[at] += h;
        let plus = oracle_objective(&oracle(&parameters, &input, &valid, rows, length, width, &unary_adjoint, &pair_adjoint), &unary_adjoint, &pair_adjoint);
        input[at] -= 2.0 * h;
        let minus = oracle_objective(&oracle(&parameters, &input, &valid, rows, length, width, &unary_adjoint, &pair_adjoint), &unary_adjoint, &pair_adjoint);
        central_actual.push((plus - minus) / (2.0 * h));
        central_expected.push(expected.features_gradient[at]);
    }
    let central_differences = crate::sbd::compare(&central_actual, &central_expected)?;

    let rank_input = Tensor::<3>::from_data(TensorData::new(features, [rows, length, width]), &device).require_grad();
    let (unary, pairs) = energy.terms(rank_input.clone(), validity.clone(), true);
    let selected: [(usize, &[usize]); 6] = [(0, &[0, 1]), (0, &[1, 2]), (1, &[0, 1]), (1, &[0, 1]), (0, &[0, 1]), (0, &[1, 2])];
    let values = selected.iter().map(|&(row, set)| reference_set_energy(&expected.unary, &expected.pairs, row, length, set)).collect::<Vec<_>>();
    let selected_energies = set_energies(unary, pairs, &selected, true)?;
    let selected_sums = compare(&host(selected_energies.clone())?, &values)?;
    let reference_good = vec![values[0], values[2], values[4]];
    let reference_bad = vec![values[1], values[3], values[5]];
    let eligibility = [1.0f64, 0.0, 0.5];
    let (expected_loss, dg, db) = reference_rank(&reference_good, &reference_bad, &eligibility, MARGIN);
    let energies = selected_energies.reshape([3, 2]);
    let loss = rank_loss(energies.clone().slice([0..3, 0..1]).reshape([3]), energies.slice([0..3, 1..2]).reshape([3]),
        Tensor::from_data(TensorData::new(eligibility.iter().map(|&x| x as f32).collect::<Vec<_>>(), [3]), &device), MARGIN);
    let rank_forward = compare(&host(loss.clone())?, &[expected_loss])?;
    let mut rank_unary_adjoint = vec![0.0; rows * length];
    let mut rank_pair_adjoint = vec![0.0; rows * length * length];
    for (candidate, &(row, set)) in selected.iter().enumerate() {
        let coefficient = if candidate % 2 == 0 { dg[candidate / 2] } else { db[candidate / 2] };
        for &i in set { rank_unary_adjoint[row * length + i] += coefficient; }
        for (left, &i) in set.iter().enumerate() { for &j in &set[left + 1..] {
            rank_pair_adjoint[(row * length + i) * length + j] += coefficient;
        }}
    }
    let expected_rank = oracle(&parameters, &f64_features, &valid, rows, length, width, &rank_unary_adjoint, &rank_pair_adjoint);
    let raw_gradients = loss.backward();
    let rank_feature_vjp = compare(&host(rank_input.grad(&raw_gradients).context("missing rank feature VJP")?)?, &expected_rank.features_gradient)?;
    let rank_gradients = GradientsParams::from_grads(raw_gradients, &energy);
    let rank_parameter_vjp = gradient_reports(&energy, &rank_gradients, &expected_rank.gradient)?;

    let zero_pair_model = {
        let mut copy = energy.clone();
        copy.pair = copy.pair.map(|_| Tensor::zeros([HIDDEN, HIDDEN], &device).require_grad());
        copy
    };
    let clean_features = Tensor::<3>::from_data(TensorData::new(f64_features.iter().map(|&x| x as f32).collect::<Vec<_>>(), [rows, length, width]), &device);
    let (unary_only, no_pairs) = zero_pair_model.terms(clean_features.clone(), validity.clone(), false);
    let (zero_pair_unary, zero_pairs) = zero_pair_model.terms(clean_features, validity, true);
    let zero_initial_pair_unary = compare(&host(zero_pair_unary)?, &host(unary_only)?.iter().map(|&x| f64::from(x)).collect::<Vec<_>>())?;
    ensure!(host(no_pairs)?.iter().chain(host(zero_pairs)?.iter()).all(|&x| x == 0.0), "zero-initialized pair model changed unary energy");
    Ok(json!({"passed":true,"device":runtime::identity(&device),"shape":[rows,length,width],
        "forward_unary":forward_unary,"forward_pairs":forward_pairs,"feature_vjp":feature_vjp,
        "parameter_vjp":parameter_vjp,"f64_central_differences":central_differences,
        "selected_sums":selected_sums,"rank_forward":rank_forward,
        "rank_feature_vjp":rank_feature_vjp,"rank_parameter_vjp":rank_parameter_vjp,
        "zero_initial_pair_unary":zero_initial_pair_unary,"dirty_nan_padding":true,"exact_pair_symmetry":true}))
}

fn rank_checks(device: &Device) -> Result<Value> {
    let device = device.clone().autodiff();
    let fixtures = [
        (vec![0.25, f64::NAN, 100.0, -100.0, 0.0], vec![-0.5, f64::NAN, -100.0, 100.0, 0.0], vec![1.0, 0.0, 0.5, 1.0, 1.0], 1.0),
        (vec![f64::NAN, f64::INFINITY], vec![f64::NAN, f64::NEG_INFINITY], vec![0.0, 0.0], 1.0),
        (vec![0.0, 0.0], vec![0.0, 0.0], vec![0.5, 0.0], 0.0),
    ];
    let mut reports = Vec::new();
    for (good, bad, eligibility, margin) in fixtures {
        let n = good.len();
        let input = |values: &[f64]| Tensor::<1>::from_data(TensorData::new(values.iter().map(|&x| x as f32).collect::<Vec<_>>(), [n]), &device);
        let g = input(&good).require_grad();
        let b = input(&bad).require_grad();
        let (reference, dg, db) = reference_rank(&good, &bad, &eligibility, margin);
        let loss = rank_loss(g.clone(), b.clone(), input(&eligibility), margin);
        let value = compare(&host(loss.clone())?, &[reference])?;
        let gradients = loss.backward();
        let good_vjp = compare(&host(g.grad(&gradients).context("missing good rank gradient")?)?, &dg)?;
        let bad_vjp = compare(&host(b.grad(&gradients).context("missing bad rank gradient")?)?, &db)?;
        let mut numerical = Vec::new();
        let mut analytic = Vec::new();
        for i in 0..n {
            if eligibility[i] == 0.0 { continue; }
            let mut shifted = good.clone();
            let h = 1e-4;
            shifted[i] += h;
            let plus = reference_rank(&shifted, &bad, &eligibility, margin).0;
            shifted[i] -= 2.0 * h;
            let minus = reference_rank(&shifted, &bad, &eligibility, margin).0;
            numerical.push((plus - minus) / (2.0 * h));
            analytic.push(dg[i]);
        }
        reports.push(json!({"margin":margin,"eligible_weight":eligibility.iter().sum::<f64>(),
            "loss":value,"good_vjp":good_vjp,"bad_vjp":bad_vjp,
            "f64_central_differences":crate::sbd::compare(&numerical,&analytic)?}));
    }
    Ok(json!({"passed":true,"fixtures":reports,"inactive_nan_inputs":true,"all_inactive_zero_gradients":true,
        "fractional_eligibility":true,"stable_extreme_arguments":[-199,201]}))
}
fn error_maxima(value: &Value) -> (f64, f64) {
    match value {
        Value::Object(object) => {
            let mut absolute = object.get("max_abs").and_then(Value::as_f64).unwrap_or(0.0);
            let mut ratio = object.get("maximum_tolerance_ratio").and_then(Value::as_f64).unwrap_or(0.0);
            for child in object.values() {
                let (a, r) = error_maxima(child); absolute = absolute.max(a); ratio = ratio.max(r);
            }
            (absolute, ratio)
        }
        Value::Array(values) => values.iter().map(error_maxima).fold((0.0, 0.0), |(a, r), (b, s)| (a.max(b), r.max(s))),
        _ => (0.0, 0.0),
    }
}
/// Correctness only: no quality arms, parameter sweeps or calibration studies.
/// CPU and the requested accelerator each face the same independent f64 oracle.
pub fn numerical_checks(device: &Device) -> Result<Value> {
    let sampling = sampling_checks()?;
    let mut devices = vec![json!({"energy":numerical_device(device)?,"rank_loss":rank_checks(device)?})];
    #[cfg(feature = "cpu")]
    if device != &Device::ndarray() {
        devices.push(json!({"energy":numerical_device(&Device::ndarray())?,"rank_loss":rank_checks(&Device::ndarray())?}));
    }
    let device_reports = Value::Array(devices);
    let (maximum_absolute_error, maximum_tolerance_ratio) = error_maxima(&device_reports);
    Ok(json!({"passed":true,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,
        "max_abs":maximum_absolute_error,"maximum_tolerance_ratio":maximum_tolerance_ratio,
        "sampling_identity":SAMPLING_IDENTITY,"epsilon":EPSILON,"temperature":TEMPERATURE,
        "sampling":sampling,"devices":device_reports}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_set_sampling_matches_independent_enumeration() {
        sampling_checks().unwrap();
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn energy_and_rank_values_and_vjps_match_f64() {
        numerical_device(&Device::ndarray()).unwrap();
        rank_checks(&Device::ndarray()).unwrap();
    }

    #[test]
    fn dataset_rejects_duplicate_supervision_and_accepts_ties() {
        let mut data = Dataset { schema: 1, feature_width: 2, checkpoint_identity: "fixture".into(), metadata: Value::Null,
            states: vec![RankState { length: 2, features: vec![0.0; 4],
                pairs: vec![RankPair { good: vec![0], bad: vec![0], eligible: 1.0 }] }] };
        assert!(data.validate().is_err());
        data.states[0].pairs[0].eligible = 0.0;
        data.validate().unwrap();
        let restored: Dataset = serde_json::from_value(json!({"schema":1,"feature_width":2,"checkpoint_identity":"fixture","states":data.states})).unwrap();
        assert!(restored.metadata.is_null());
        restored.validate().unwrap();
    }
}
