//! CPU/Vulkan acceptance checks for legacy and contextual SBD/Codpula consumers.
//! Both profiles cover strict optimizer continuation using tiny artifacts/codpula checkpoints.
use crate::{
    checkpoint,
    codpula::{CircuitFlows, CircuitState, Codpula, CodpulaExecution, GROUP_WIDTH, PreparedCircuit, STATES, VOCAB},
    config::{Arm, Config, SbdBlockExecution, SbdCommitment, SbdConfig, SbdFeedback, SbdJointOutput},
    data::{self, Batch, Example},
    kron_benchmark::values,
    model::{self, DenoisingOutput, LanguageModel},
    runtime, sbd,
    sbd_training::CleanInputs,
    soft::SoftState,
    train,
};
use anyhow::{Context, Result, ensure};
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::{GradientsParams, ModuleOptimizer},
    prelude::*,
};
use delta_mixers::sbd::{InBlockReads, NoisyReadBackend};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, time::{SystemTime, UNIX_EPOCH}};

fn compare(actual: &[f32], reference: &[f32]) -> Result<Value> {
    sbd::compare(actual, &reference.iter().map(|&x| f64::from(x)).collect::<Vec<_>>())
}
fn identical(actual: &[f32], reference: &[f32], label: &str) -> Result<Value> {
    let report = compare(actual, reference).with_context(|| label.to_owned())?;
    ensure!(actual.iter().zip(reference).all(|(a, b)| a.to_bits() == b.to_bits()), "{label}: values are not bit-identical");
    Ok(report)
}
fn nonzero_finite(actual: &[f32], label: &str) -> Result<f64> {
    ensure!(actual.iter().all(|x| x.is_finite()), "nonfinite {label}");
    let maximum = actual.iter().map(|&x| f64::from(x).abs()).fold(0.0, f64::max);
    ensure!(maximum > 0.0, "{label} is identically zero");
    Ok(maximum)
}

#[derive(Clone)]
struct ParameterValues {
    shape: Vec<usize>,
    values: Vec<f32>,
    trainable: bool,
}
type Snapshot = BTreeMap<String, ParameterValues>;
struct Parameters<'a> {
    path: Vec<String>,
    gradients: Option<&'a GradientsParams>,
    snapshot: Snapshot,
    errors: Vec<String>,
}
impl ModuleVisitor for Parameters<'_> {
    fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_owned()); }
    fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        let name = self.path.join(".");
        let trainable = parameter.val().is_require_grad();
        let tensor = match self.gradients {
            Some(gradients) => {
                let gradient = gradients.get::<D>(parameter.id);
                if !trainable {
                    if gradient.is_some() { self.errors.push(format!("frozen parameter {name} has a neural gradient")); }
                    return;
                }
                match gradient {
                    Some(value) => value,
                    None => { self.errors.push(format!("missing neural gradient {name}")); return; }
                }
            }
            None => parameter.val(),
        };
        let shape = tensor.dims().to_vec();
        let value = ParameterValues { shape, values: values(tensor), trainable };
        if self.snapshot.insert(name.clone(), value).is_some() { self.errors.push(format!("duplicate parameter path {name}")); }
    }
}
fn snapshot(model: &impl Module, gradients: Option<&GradientsParams>) -> Result<Snapshot> {
    let mut visitor = Parameters { path: Vec::new(), gradients, snapshot: BTreeMap::new(), errors: Vec::new() };
    model.visit(&mut visitor);
    ensure!(visitor.errors.is_empty(), "{}", visitor.errors.join("; "));
    ensure!(!visitor.snapshot.is_empty(), "empty model parameter snapshot");
    Ok(visitor.snapshot)
}
fn compare_snapshots(actual: &Snapshot, reference: &Snapshot, exact: bool) -> Result<Value> {
    ensure!(actual.keys().eq(reference.keys()), "model parameter paths differ");
    let mut reports = BTreeMap::new();
    for (name, expected) in reference {
        let actual = &actual[name];
        ensure!(actual.shape == expected.shape && actual.trainable == expected.trainable, "parameter shape/trainability differs: {name}");
        let report = if exact { identical(&actual.values, &expected.values, name)? } else { compare(&actual.values, &expected.values).with_context(|| name.to_owned())? };
        reports.insert(name, report);
    }
    Ok(json!({"passed":true,"bit_identical":exact,"parameter_tensors":reports.len(),"parameters":reports}))
}
fn common(snapshot: &Snapshot) -> Snapshot {
    snapshot.iter().filter(|(name, _)| !name.split('.').any(|part| part == "codpula"))
        .map(|(name, value)| (name.clone(), value.clone())).collect()
}
fn snapshot_hash(snapshot: &Snapshot) -> Result<String> {
    let rows = snapshot.iter().map(|(name, value)| (name, &value.shape, value.trainable, value.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>())).collect::<Vec<_>>();
    Ok(data::hash(&serde_json::to_vec(&rows)?))
}
fn flow_values(flows: &CircuitFlows) -> BTreeMap<&'static str, Vec<f32>> {
    BTreeMap::from([
        ("emission_positive", values(flows.emission_positive.clone())),
        ("emission_negative", values(flows.emission_negative.clone())),
        ("transition_positive", values(flows.transition_positive.clone())),
        ("transition_negative", values(flows.transition_negative.clone())),
        ("root_positive", values(flows.root_positive.clone())),
        ("root_negative", values(flows.root_negative.clone())),
        ("emission_reach", values(flows.emission_reach.clone())),
        ("transition_reach", values(flows.transition_reach.clone())),
        ("root_reach", values(flows.root_reach.clone())),
    ])
}
fn compare_flows(actual: &CircuitFlows, reference: &CircuitFlows) -> Result<Value> {
    ensure!(actual.active_blocks == reference.active_blocks, "circuit active-block count differs");
    let expected = flow_values(reference);
    let mut reports = BTreeMap::new();
    for (name, value) in flow_values(actual) {
        ensure!(value.iter().all(|&x| x.is_finite() && x >= 0.0), "invalid nonnegative circuit statistic {name}");
        reports.insert(name, compare(&value, &expected[name]).with_context(|| format!("circuit {name}"))?);
    }
    Ok(json!({"passed":true,"active_blocks":actual.active_blocks,"statistics":reports}))
}
fn finite_flows(flows: &CircuitFlows) -> Result<Value> {
    let mut reports = BTreeMap::new();
    for (name, value) in flow_values(flows) {
        ensure!(value.iter().all(|&x|x.is_finite() && x >= 0.0),"invalid complete-step circuit statistic {name}");
        reports.insert(name,json!({"elements":value.len(),"minimum":value.iter().copied().fold(f32::INFINITY,f32::min),"maximum":value.iter().copied().fold(0.0,f32::max)}));
    }
    Ok(json!({"passed":true,"active_blocks":flows.active_blocks,"statistics":reports}))
}

fn rows(count: usize, length: usize) -> Vec<Example> {
    (0..count).map(|row| {
        let sequence = std::iter::once(256).chain((0..length).map(|position| ((position * 37 + row * 53 + 11) % 256) as i32)).collect::<Vec<_>>();
        Example { input: sequence[..length].to_vec(), target: sequence[1..].to_vec(), weight: vec![1.0; length], groups: vec![None; length] }
    }).collect()
}
fn fixture_config(c: &Config, mixer: &str, extended: bool) -> (Config, Arm) {
    let mut cfg = c.clone();
    cfg.task = "lm".into();
    cfg.backend = "cpu".into();
    cfg.seed = 72;
    cfg.fusion = false;
    cfg.autotune = false;
    cfg.d_model = 16;
    cfg.layers = 1;
    cfg.heads = 4;
    cfg.key_dim = 32;
    cfg.value_dim = 64;
    cfg.chunk_size = 16;
    cfg.cache_capacity = 4;
    cfg.context = 12;
    cfg.batch_size = 2;
    cfg.train_tokens = 24;
    cfg.budget_file = None;
    cfg.learning_rate = 0.001;
    cfg.gradient_clip = 1.0;
    cfg.scan_backend = delta_mixers::ScanBackend::TensorSolver;
    cfg.hola_backend = delta_mixers::HolaBackend::Tensor;
    cfg.projection_backend = delta_mixers::ProjectionBackend::Separate;
    cfg.normalization_backend = delta_mixers::NormalizationBackend::Tensor;
    cfg.convolution_backend = delta_mixers::ConvolutionBackend::Tensor;
    cfg.head_backend = crate::config::HeadBackend::Native;
    cfg.loss_backend = crate::config::LossBackend::Tensor;
    cfg.update_backend = crate::config::UpdateBackend::Tensor;
    cfg.epilogue_backend = crate::config::EpilogueBackend::Native;
    cfg.prefix_block_retrieval_backend = delta_mixers::prefix_retrieval::PrefixRetrievalBackend::Tensor;
    cfg.prefix_block_retrieval = extended.then_some(delta_mixers::prefix_retrieval::PrefixBlockRetrieval { layers: vec![0], block_size: 16, blocks_per_query: 1 });
    cfg.ohc_streams = extended;
    cfg.ohc_backend = delta_mixers::ohc::OhcBackend::Tensor;
    cfg.wavelet_layers = if extended { vec![0] } else { vec![] };
    cfg.wavelet_backend = delta_mixers::WaveletBackend::Tensor;
    cfg.lngram_layers = if extended { vec![0] } else { vec![] };
    cfg.lngram_backend = delta_mixers::lngram::LngramBackend::Tensor;
    cfg.lngram_routes = 2;
    cfg.lngram_bits = 1;
    cfg.lngram_memory_dim = 2;
    cfg.lngram_orders = vec![2, 3];
    cfg.soft_selection_backend = crate::config::SoftSelectionBackend::Cpu;
    cfg.soft_temperature = 0.8;
    cfg.sbd = Some(SbdConfig { block_lengths: vec![4], joint_output: SbdJointOutput::Codpula,
        in_block_reads: if extended { InBlockReads::HolaRead } else { InBlockReads::None }, ..SbdConfig::default() });
    let arm = Arm { name: format!("{mixer}-{}", if extended { "hola" } else { "none" }), mixer: mixer.into(), cache: if extended { "surprise".into() } else { "none".into() } };
    cfg.arms = vec![arm.clone()];
    (cfg, arm)
}
fn requested_backends(model: &mut LanguageModel, cfg: &mut Config, c: &Config) {
    cfg.backend = c.backend.clone();
    if c.backend != "cpu" {
        cfg.scan_backend = if cfg.arms[0].mixer == "gdn2" { delta_mixers::ScanBackend::FusedGdn2 } else { delta_mixers::ScanBackend::FusedGpu };
        cfg.hola_backend = delta_mixers::HolaBackend::FusedGpu;
        cfg.sbd.as_mut().unwrap().noisy_read_backend = if cfg.sbd.as_ref().unwrap().in_block_reads.is_none() { NoisyReadBackend::Tensor } else { NoisyReadBackend::FusedGpu };
        cfg.sbd.as_mut().unwrap().joint_execution = c.sbd.as_ref().map_or(CodpulaExecution::Tensor, |s| s.joint_execution);
        model.set_scan_backend(cfg.scan_backend);
        model.set_hola_backend(cfg.hola_backend);
        model.set_noisy_read_backend(cfg.sbd.as_ref().unwrap().noisy_read_backend);
        model.codpula.as_mut().unwrap().set_execution(cfg.sbd.as_ref().unwrap().joint_execution);
    }
}
fn soft(cfg: &Config, observed: Vec<Option<usize>>, active: Vec<bool>, device: &Device) -> Result<SoftState> {
    SoftState::new(VOCAB, cfg.sbd.as_ref().unwrap().candidate_width, observed, active, device, cfg.soft_selection_backend)
}
fn probe_gates(model: &mut LanguageModel, device: &Device) {
    let head = &mut model.codpula.as_mut().unwrap().gate_head;
    let shape = head.weight.val().dims();
    head.weight = head.weight.clone().map(|_| Tensor::from_data(TensorData::new(
        (0..shape[0] * shape[1]).map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.005).collect::<Vec<_>>(), shape), device).require_grad());
    let outputs = shape[1];
    head.bias = head.bias.take().map(|bias| bias.map(|_| Tensor::from_data(TensorData::new(
        (0..outputs).map(|i| (i as f32 - (outputs - 1) as f32 * 0.5) * 0.003).collect::<Vec<_>>(), [outputs]), device).require_grad()));
}
fn active_values(tensor: Tensor<3>, lengths: &[usize]) -> Vec<f32> {
    let [_, width, depth] = tensor.dims();
    let raw = values(tensor);
    lengths.iter().enumerate().flat_map(|(row, &length)| raw[row * width * depth..(row * width + length) * depth].iter().copied()).collect()
}

// Keep the historical draw order, f32 conversion and double row normalization explicit.
// This reference does not call either profile constructor.
fn initial_probability(seed: u64, emission_radius: f64, transition_radius: f64, device: &Device) -> CircuitState {
    let mut state = seed;
    let mut row = |rows: usize, columns: usize, radius: f64| {
        let data = (0..rows * columns).map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut bits = state;
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
            bits ^= bits >> 31;
            let uniform = ((bits >> 11) as f64 + 0.5) / 9007199254740992.0;
            ((uniform - 0.5) * (2.0 * radius)) as f32
        }).collect::<Vec<_>>();
        burn::tensor::activation::log_softmax(Tensor::from_data(TensorData::new(data, [rows, columns]), device), 1)
    };
    let emission = row(STATES, VOCAB, emission_radius);
    let transition = row(STATES, STATES, transition_radius);
    let root = row(1, STATES, 0.01).reshape([STATES]);
    CircuitState::from_logs(emission, transition, root)
}
fn initial_head(circuit: &Codpula, inputs: usize, outputs: usize) -> Result<Value> {
    ensure!(circuit.gate_head.weight.val().dims() == [inputs, outputs], "initial gate weight shape differs");
    let bias = circuit.gate_head.bias.as_ref().context("initial gate bias")?;
    ensure!(bias.val().dims() == [outputs], "initial gate bias shape differs");
    ensure!(values(circuit.gate_head.weight.val()).iter().chain(values(bias.val()).iter()).all(|&x| x == 0.0), "gate head is not initialized to zero");
    let parameters = snapshot(circuit, None)?;
    ensure!(parameters.values().filter(|value| value.trainable).count() == 2, "profile has more than one trainable linear head");
    Ok(json!({"passed":true,"weight_shape":[inputs,outputs],"bias_shape":[outputs],"head_exact_zero":true,"trainable_tensors":2}))
}

fn selector_identity(c: &Config, device: &Device) -> Result<Value> {
    let (mut cfg, arm) = fixture_config(c, "kronlstm", false);
    cfg.sbd.as_mut().unwrap().joint_output = SbdJointOutput::Factorized;
    let encoded = serde_json::to_value(&cfg)?;
    ensure!(encoded["sbd"].get("joint_output").is_none(), "factorized selector was serialized");
    let omitted: Config = serde_json::from_value(encoded.clone())?;
    let mut explicit = encoded.clone();
    explicit["sbd"]["joint_output"] = json!("factorized");
    let explicit: Config = serde_json::from_value(explicit)?;
    ensure!(serde_json::to_value(&explicit)? == encoded, "explicit factorized selector changes serialization");
    ensure!(omitted.training_fingerprint(24)? == explicit.training_fingerprint(24)? && omitted.pilot_fingerprint()? == explicit.pilot_fingerprint()?, "factorized selector changes hashes");
    device.seed(cfg.seed);
    let old_model = LanguageModel::new(&omitted, &arm, device)?;
    let baseline = snapshot(&old_model, None)?;
    device.seed(cfg.seed);
    let explicit_model = LanguageModel::new(&explicit, &arm, device)?;
    ensure!(old_model.codpula.is_none() && explicit_model.codpula.is_none(), "factorized model allocated a circuit");
    let explicit_values = snapshot(&explicit_model, None)?;
    let initial = compare_snapshots(&explicit_values, &baseline, true)?;
    cfg.sbd.as_mut().unwrap().joint_output = SbdJointOutput::Codpula;
    device.seed(cfg.seed);
    let joint = LanguageModel::new(&cfg, &arm, device)?;
    let joint_values = snapshot(&joint, None)?;
    let common_initial = compare_snapshots(&common(&joint_values), &baseline, true)?;
    let circuit = joint.codpula.as_ref().unwrap();
    let legacy_head = initial_head(circuit, cfg.d_model, 64)?;
    let legacy_probability = compare_snapshots(&snapshot(&circuit.probability,None)?, &snapshot(&initial_probability(cfg.seed,0.01,0.01,device),None)?, true)?;
    let legacy_encoded = serde_json::to_value(&cfg)?;
    ensure!(legacy_encoded["sbd"]["joint_output"] == "codpula", "legacy selector serialization changed");
    let legacy_roundtrip: Config = serde_json::from_value(legacy_encoded)?;
    ensure!(cfg.training_fingerprint(24)? == legacy_roundtrip.training_fingerprint(24)? && cfg.pilot_fingerprint()? == legacy_roundtrip.pilot_fingerprint()?, "legacy selector roundtrip changes hashes");
    ensure!(cfg.training_fingerprint(24)? != omitted.training_fingerprint(24)? && cfg.pilot_fingerprint()? != omitted.pilot_fingerprint()?, "Codpula does not change config identity");
    let legacy_training = cfg.training_fingerprint(24)?;
    let legacy_pilot = cfg.pilot_fingerprint()?;
    cfg.sbd.as_mut().unwrap().joint_output = SbdJointOutput::CodpulaContextual;
    let contextual_encoded = serde_json::to_value(&cfg)?;
    ensure!(contextual_encoded["sbd"]["joint_output"] == "codpula_contextual", "contextual selector serialization differs");
    let contextual_roundtrip: Config = serde_json::from_value(contextual_encoded)?;
    ensure!(cfg.training_fingerprint(24)? == contextual_roundtrip.training_fingerprint(24)? && cfg.pilot_fingerprint()? == contextual_roundtrip.pilot_fingerprint()?, "contextual selector roundtrip changes hashes");
    let contextual_training = cfg.training_fingerprint(24)?;
    let contextual_pilot = cfg.pilot_fingerprint()?;
    ensure!(contextual_training != legacy_training && contextual_training != omitted.training_fingerprint(24)?
        && contextual_pilot != legacy_pilot && contextual_pilot != omitted.pilot_fingerprint()?, "contextual profile does not have distinct identity");
    device.seed(cfg.seed);
    let contextual = LanguageModel::new(&cfg, &arm, device)?;
    let contextual_common = compare_snapshots(&common(&snapshot(&contextual,None)?), &baseline, true)?;
    let contextual_circuit = contextual.codpula.as_ref().context("contextual model has no circuit")?;
    let contextual_head = initial_head(contextual_circuit,3 * cfg.d_model,96)?;
    let contextual_probability = compare_snapshots(&snapshot(&contextual_circuit.probability,None)?, &snapshot(&initial_probability(cfg.seed,0.25,0.1,device),None)?, true)?;
    let shared_root = identical(&values(contextual_circuit.probability.log_root.val()),&values(circuit.probability.log_root.val()),"contextual root initialization differs from legacy")?;
    ensure!(values(contextual_circuit.probability.log_emission.val()) != values(circuit.probability.log_emission.val())
        && values(contextual_circuit.probability.log_transition.val()) != values(circuit.probability.log_transition.val()), "contextual probability initialization still uses legacy ranges");
    let full_width_legacy = initial_head(&Codpula::new(256,cfg.seed,device),256,64)?;
    let full_width_contextual = initial_head(&Codpula::new_contextual(256,cfg.seed,device),768,96)?;
    let batch = Batch::new(&rows(1, 9), device);
    let clean = identical(&values(joint.forward(&batch)), &values(old_model.forward(&batch)), "Codpula changed clean logits")?;
    let clean_loss = identical(&values(joint.loss(joint.forward(&batch), &batch)), &values(old_model.loss(old_model.forward(&batch), &batch)), "Codpula changed clean loss")?;
    let contextual_clean = identical(&values(contextual.forward(&batch)), &values(old_model.forward(&batch)), "contextual Codpula changed clean logits")?;
    let contextual_clean_loss = identical(&values(contextual.loss(contextual.forward(&batch), &batch)), &values(old_model.loss(old_model.forward(&batch), &batch)), "contextual Codpula changed clean loss")?;
    Ok(json!({"passed":true,"omitted_and_explicit_serialization_identical":true,"training_sha256":omitted.training_fingerprint(24)?,"pilot_sha256":omitted.pilot_fingerprint()?,"initial_parameter_sha256":snapshot_hash(&baseline)?,"explicit_initial_parameters":initial,"codpula_common_initial_parameters":common_initial,"clean_logits":clean,"clean_loss":clean_loss,"gate_initialization_zero":true,
        "legacy":{"training_sha256":legacy_training,"pilot_sha256":legacy_pilot,"head":legacy_head,"historical_probability_initialization":legacy_probability,"d_model_256_head":full_width_legacy},
        "contextual":{"training_sha256":cfg.training_fingerprint(24)?,"pilot_sha256":cfg.pilot_fingerprint()?,"common_initial_parameters":contextual_common,"head":contextual_head,"probability_initialization":contextual_probability,"root_matches_legacy":shared_root,"d_model_256_head":full_width_contextual,"clean_logits":contextual_clean,"clean_loss":contextual_clean_loss},
        "profile_identities_distinct":true}))
}

struct Readout {
    logits: Vec<f32>,
    features: Vec<f32>,
    marginals: Vec<f32>,
    evidence: Vec<f32>,
    loss: Tensor<1>,
    flows: CircuitFlows,
}
fn readout(model: &LanguageModel, output: DenoisingOutput, state: &SoftState, targets: Tensor<2, Int>, denominator: f64) -> Result<Readout> {
    let [count, width, _] = output.logits.dims();
    let lengths = state.active.chunks_exact(width).map(|row| row.iter().take_while(|&&x| x).count()).collect::<Vec<_>>();
    ensure!(lengths.len() == count, "readout block layout differs");
    let logits = active_values(output.logits.clone(), &lengths);
    let features = active_values(output.features.clone(), &lengths);
    let (prepared, tokens, observed) = sbd::prepare_circuit(model, output, state, 1.0)?;
    let marginals = active_values(prepared.marginals(tokens.clone(), observed.clone())?, &lengths);
    let evidence = values(prepared.log_partition(tokens, observed.clone())?);
    let result = prepared.conditional_nll(targets, observed, denominator)?;
    Ok(Readout { logits, features, marginals, evidence, loss: result.loss, flows: result.flows })
}
fn compare_readouts(actual: &Readout, reference: &Readout) -> Result<Value> {
    Ok(json!({"passed":true,"logits":compare(&actual.logits,&reference.logits)?,"final_normalized_features":compare(&actual.features,&reference.features)?,"joint_marginals":compare(&actual.marginals,&reference.marginals)?,"visible_log_mass":compare(&actual.evidence,&reference.evidence)?,"joint_nll":compare(&values(actual.loss.clone()),&values(reference.loss.clone()))?,"circuit_flows":compare_flows(&actual.flows,&reference.flows)?}))
}
fn output_paths(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    let examples = rows(2, 54);
    let batch = Batch::new(&examples, device);
    let blocks = [(0, 1), (1, 9), (0, 17), (1, 35), (0, 39)];
    let lengths = [16, 8, 4, 3, 2];
    let width = 16;
    let mut trace = CleanInputs::default();
    let _ = model.forward_profiled(&batch, &mut trace);
    let mut observations = Vec::new();
    let mut active = Vec::new();
    let mut targets = Vec::new();
    let mut reconstructed: Option<Readout> = None;
    let mut cached_reports = Vec::new();
    for (&(row, start), &length) in blocks.iter().zip(&lengths) {
        let observed = (0..length).map(|i| (i % 3 == 1).then_some(examples[row].target[start - 1 + i] as usize)).collect::<Vec<_>>();
        let state = soft(cfg, observed.clone(), vec![true; length], device)?;
        let target = examples[row].target[start - 1..start - 1 + length].to_vec();
        let target_tensor = Tensor::from_data(TensorData::new(target.clone(), [1, length]), device);
        let direct = readout(model, sbd::block_forward(model, &examples[row].input[..start], &state, false, device), &state, target_tensor.clone(), 108.0)?;
        let cache = model.training_cache(&trace, row, &[start]);
        let cached = readout(model, model.cached_block(&cache, 0, &state, false), &state, target_tensor, 108.0)?;
        cached_reports.push(compare_readouts(&cached, &direct)?);
        if let Some(total) = &mut reconstructed {
            total.logits.extend(direct.logits);
            total.features.extend(direct.features);
            total.marginals.extend(direct.marginals);
            total.evidence.extend(direct.evidence);
            total.loss = total.loss.clone() + direct.loss;
            total.flows.add(direct.flows);
        } else {
            reconstructed = Some(direct);
        }
        observations.extend(observed);
        observations.extend(std::iter::repeat_n(None, width - length));
        active.extend(std::iter::repeat_n(true, length));
        active.extend(std::iter::repeat_n(false, width - length));
        targets.extend(target);
        targets.extend(std::iter::repeat_n(0, width - length));
    }
    let reconstructed = reconstructed.context("missing reconstructed blocks")?;
    let target = Tensor::from_data(TensorData::new(targets, [blocks.len(), width]), device);
    let cache = model.training_cache_batched(&trace, &blocks);
    let state = soft(cfg, observations, active.clone(), device)?;
    let output = model.cached_blocks(&cache, &state, false);
    let batched = readout(model, output.clone(), &state, target.clone(), 108.0)?;
    let batched_report = compare_readouts(&batched, &reconstructed)?;
    let mut dirty_state = state.clone();
    let inactive = Tensor::<2, Bool>::from_data(TensorData::new(active.iter().map(|x| !x).collect::<Vec<_>>(), [active.len(), 1]), device);
    let k = dirty_state.candidates.weights.dims()[1];
    dirty_state.candidates.weights = dirty_state.candidates.weights.mask_fill(inactive.clone().expand([active.len(), k]), f32::NAN);
    dirty_state.candidates.retained_mass = dirty_state.candidates.retained_mass.mask_fill(inactive.clone(), f32::INFINITY);
    dirty_state.visibility = dirty_state.visibility.mask_fill(inactive, f32::NAN);
    for (position, &valid) in active.iter().enumerate() { if !valid { dirty_state.observed[position] = Some(256); } }
    let mut dirty_trace = CleanInputs::default();
    let _ = model.forward_profiled(&batch,&mut dirty_trace);
    let dirty_cache = model.training_cache_batched(&dirty_trace,&blocks);
    let dirty_output = model.cached_blocks(&dirty_cache, &dirty_state, false);
    let dirty = readout(model, dirty_output, &dirty_state, target.clone(), 108.0)?;
    let dirty_report = compare_readouts(&dirty, &batched)?;
    let clean_neural_gradients = model::checked_gradients(model,batched.loss.clone())?;
    let dirty_neural_gradients = model::checked_gradients(model,dirty.loss.clone())?;
    let padding_neural_gradients = compare_snapshots(&snapshot(model,Some(&dirty_neural_gradients))?,&snapshot(model,Some(&clean_neural_gradients))?,false)?;
    let clean_logits = output.logits.clone().detach().require_grad();
    let clean_features = output.features.clone().detach().require_grad();
    let clean_circuit = readout(model,DenoisingOutput { logits:clean_logits.clone(),features:clean_features.clone() },&state,target.clone(),108.0)?;
    let invalid = Tensor::<3, Bool>::from_data(TensorData::new(active.iter().map(|x| !x).collect::<Vec<_>>(), [blocks.len(), width, 1]), device);
    let logits = output.logits.detach().mask_fill(invalid.clone().expand([blocks.len(), width, VOCAB]), f32::NAN).require_grad();
    let features = output.features.detach().mask_fill(invalid.expand([blocks.len(), width, cfg.d_model]), f32::INFINITY).require_grad();
    let dirty_circuit = readout(model, DenoisingOutput { logits: logits.clone(), features: features.clone() }, &state, target, 108.0)?;
    let circuit_report = compare_readouts(&dirty_circuit, &batched)?;
    let gradients = dirty_circuit.loss.backward();
    let logit_gradient = values(logits.grad(&gradients).context("padded token-potential gradient")?);
    let feature_gradient = values(features.grad(&gradients).context("padded gate-feature gradient")?);
    let clean_gradients = clean_circuit.loss.backward();
    let logit_gradient_report = compare(&logit_gradient,&values(clean_logits.grad(&clean_gradients).context("clean token-potential gradient")?))?;
    let feature_gradient_report = compare(&feature_gradient,&values(clean_features.grad(&clean_gradients).context("clean gate-feature gradient")?))?;
    for (name, depth, gradient) in [("token", VOCAB, &logit_gradient), ("feature", cfg.d_model, &feature_gradient)] {
        ensure!(gradient.iter().all(|x| x.is_finite()), "dirty padding contaminated {name} gradients");
        for (position, &valid) in active.iter().enumerate() {
            if !valid { ensure!(gradient[position * depth..(position + 1) * depth].iter().all(|&x| x == 0.0), "padded {name} gradient is not zero"); }
        }
    }
    Ok(json!({"passed":true,"blocks":blocks,"actual_lengths":lengths,"padded_length":width,"reconstruction_vs_per_block":cached_reports,"reconstruction_vs_batched":batched_report,"dirty_padded_soft_state":dirty_report,"dirty_padded_potentials_and_gate_features":circuit_report,"dirty_padding_all_neural_gradients":padding_neural_gradients,"dirty_padding_token_potential_gradients":logit_gradient_report,"dirty_padding_gate_feature_gradients":feature_gradient_report,"padded_neural_derivatives_zero":true}))
}

fn locality_and_singletons(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    let examples = rows(1, 27);
    let prefix_length = 19;
    let mut trace = CleanInputs::default();
    let batch = Batch::new(&examples, device);
    let _ = model.forward_profiled(&batch, &mut trace);
    let cache = model.training_cache(&trace, 0, &[prefix_length]);
    let observed = vec![None, Some(examples[0].target[19] as usize), None, None];
    let state = soft(cfg, observed.clone(), vec![true; 4], device)?;
    let original = model.cached_block(&cache, 0, &state, false);
    let (prepared, tokens, fixed) = sbd::prepare_circuit(model, original.clone(), &state, 1.0)?;
    let marginals = values(prepared.marginals(tokens.clone(), fixed.clone())?);
    let evidence = values(prepared.log_partition(tokens, fixed)?);
    let mut mutations = 0;
    let input_mutations = (prefix_length..examples[0].input.len()).map(|at|(true,at));
    let target_mutations = (prefix_length-1..examples[0].target.len()).filter(|&at|at!=19).map(|at|(false,at));
    for (input,at) in input_mutations.chain(target_mutations) {
        let mut changed = examples.clone();
        if input { changed[0].input[at] ^= 127; } else { changed[0].target[at] ^= 63; }
        let mut trace = CleanInputs::default();
        let _ = model.forward_profiled(&Batch::new(&changed, device), &mut trace);
        let changed_cache = model.training_cache(&trace, 0, &[prefix_length]);
        let changed = model.cached_block(&changed_cache, 0, &state, false);
        identical(&values(changed.logits.clone()), &values(original.logits.clone()), "future clean input/target changed noisy logits")?;
        identical(&values(changed.features.clone()), &values(original.features.clone()), "future clean input/target changed final noisy features")?;
        let (prepared, tokens, fixed) = sbd::prepare_circuit(model, changed, &state, 1.0)?;
        identical(&values(prepared.marginals(tokens.clone(), fixed.clone())?), &marginals, "unrevealed target changed joint marginals")?;
        identical(&values(prepared.log_partition(tokens, fixed)?), &evidence, "unrevealed target changed evidence query")?;
        mutations += 1;
    }
    let observed_id = observed[1].unwrap();
    ensure!(marginals[VOCAB..2 * VOCAB].iter().enumerate().all(|(v, &p)| p == if v == observed_id { 1.0 } else { 0.0 }), "observed joint marginal is not its immutable point mass");
    let changed_state = soft(cfg, vec![None, Some((observed_id + 1) % VOCAB), None, None], vec![true; 4], device)?;
    let changed = model.cached_block(&cache, 0, &changed_state, false);
    let (prepared, tokens, fixed) = sbd::prepare_circuit(model, changed, &changed_state, 1.0)?;
    let changed_marginals = values(prepared.marginals(tokens, fixed)?);
    ensure!(changed_marginals != marginals, "supplied observation did not change the joint query");
    let mut singleton_reports = Vec::new();
    for observation in [None, Some(91)] {
        let state = soft(cfg, vec![observation], vec![true], device)?;
        let prefix = &examples[0].input[..prefix_length];
        let singleton = sbd::block_forward(model, prefix, &state, false, device);
        let causal = model.forward_output_profiled(&Batch::new(&[Example { input:prefix.to_vec(),target:vec![0;prefix.len()],weight:vec![1.0;prefix.len()],groups:vec![None;prefix.len()] }], device), &mut delta_mixers::profile::Unprofiled).slice_positions(prefix_length - 1, 1);
        singleton_reports.push(json!({"observed":observation,"logits":identical(&values(singleton.logits),&values(causal.logits),"Codpula L1 causal logits")?,"features":identical(&values(singleton.features),&values(causal.features),"Codpula L1 causal features")?}));
    }
    Ok(json!({"passed":true,"future_mutations":mutations,"future_features_logits_and_joint_queries_bit_identical":true,"observation_point_mass_immutable":true,"changed_explicit_observation_changes_query":true,"singletons":singleton_reports}))
}

fn teacher_and_student(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    let examples = rows(1, 23);
    let mut trace = CleanInputs::default();
    let _ = model.forward_profiled(&Batch::new(&examples,device),&mut trace);
    let clean_before = trace.0.iter().map(|(_,_,value)|values(value.clone())).collect::<Vec<_>>();
    let cache = model.training_cache(&trace,0,&[19]);
    let mut state = soft(cfg, vec![None, Some(65), None, None], vec![true; 4], device)?;
    let mut teacher = model.cached_block(&cache,0,&state,false);
    // Burn does not retain non-leaf adjoints; zero leaf probes expose the VJP without detaching the backbone.
    let teacher_potential_probe = Tensor::zeros(teacher.logits.dims(),device).require_grad();
    let teacher_feature_probe = Tensor::zeros(teacher.features.dims(),device).require_grad();
    teacher.logits = teacher.logits + teacher_potential_probe.clone();
    teacher.features = teacher.features + teacher_feature_probe.clone();
    let mut expected_state = state.clone();
    let (teacher_circuit,tokens,fixed) = sbd::prepare_circuit(model,teacher.clone().detach(),&state,cfg.soft_temperature)?;
    let expected_feedback = teacher_circuit.marginals(tokens,fixed)?.detach().log().reshape([4,VOCAB]);
    expected_state.update(expected_feedback,1.0,1.0,cfg.soft_steps,cfg.sbd.as_ref().unwrap().candidate_width)?;
    let initial_observed = values(state.candidates.dense(VOCAB)).drain(VOCAB..2 * VOCAB).collect::<Vec<_>>();
    sbd::update_feedback(model, cfg, teacher, &mut state, 1.0)?;
    identical(&values(state.candidates.dense(VOCAB))[VOCAB..2 * VOCAB], &initial_observed, "teacher changed observed feedback distribution")?;
    let temperature_report = compare(&values(state.candidates.dense(VOCAB)),&values(expected_state.candidates.dense(VOCAB)))?;
    let visibility_report = compare(&values(state.visibility.clone()),&values(expected_state.visibility.clone()))?;
    let mut student = model.cached_block(&cache,0,&state,false);
    let student_potential_probe = Tensor::zeros(student.logits.dims(),device).require_grad();
    let student_feature_probe = Tensor::zeros(student.features.dims(),device).require_grad();
    student.logits = student.logits + student_potential_probe.clone();
    student.features = student.features + student_feature_probe.clone();
    let (prepared, _, observed) = sbd::prepare_circuit(model, student, &state, 1.0)?;
    let targets = Tensor::<2, Int>::from_data([[11,65,83,121]], device);
    let loss = prepared.conditional_nll(targets, observed, 4.0)?;
    let gradients = loss.loss.backward();
    ensure!(teacher_potential_probe.grad(&gradients).is_none() && teacher_feature_probe.grad(&gradients).is_none(), "detached teacher retains a student-loss gradient");
    let student_gradient = values(student_potential_probe.grad(&gradients).context("student token-potential gradient")?);
    let maximum = nonzero_finite(&student_gradient, "student token-potential gradient")?;
    ensure!(student_gradient[VOCAB..2 * VOCAB].iter().all(|&x| x == 0.0), "observed token potentials have a gradient");
    let features_gradient = values(student_feature_probe.grad(&gradients).context("student gate-feature gradient")?);
    let observed_features_max = nonzero_finite(&features_gradient[cfg.d_model..2 * cfg.d_model], "observed student features through context gates")?;
    let table = model.embedding.weight.val().detach().require_grad();
    let mask = model.sbd_mask.as_ref().unwrap().val().detach().require_grad();
    let expected = state.embeddings(table.clone(), mask.clone(), false);
    let direction = Tensor::from_data(TensorData::new((0..4 * cfg.d_model).map(|i| ((i * 13 % 19) as f32 + 1.0) / 19.0).collect::<Vec<_>>(), [4,cfg.d_model]), device);
    let embedding_gradients = (expected * direction).sum().backward();
    ensure!(teacher_potential_probe.grad(&embedding_gradients).is_none() && teacher_feature_probe.grad(&embedding_gradients).is_none(), "selected-embedding loss differentiates through teacher");
    let table_gradient = values(table.grad(&embedding_gradients).context("selected student embedding gradient")?);
    let ids = crate::soft::ids_host(state.candidates.ids.clone());
    let k = state.candidates.ids.dims()[1];
    let mut selected_max = 0.0f64;
    for row in [0,2,3] {
        for &id in &ids[row*k..(row+1)*k] {
            selected_max = selected_max.max(table_gradient[id as usize * cfg.d_model..(id as usize + 1) * cfg.d_model].iter().map(|&x| f64::from(x).abs()).fold(0.0,f64::max));
        }
    }
    ensure!(selected_max > 0.0 && table_gradient.iter().all(|x| x.is_finite()), "selected student embeddings do not receive finite gradients");
    for ((_,kind,value),before) in trace.0.iter().zip(&clean_before) {
        identical(&values(value.clone()),before,&format!("teacher/student changed clean {kind} snapshot"))?;
    }
    Ok(json!({"passed":true,"same_clean_training_cache_for_teacher_and_student":true,"clean_snapshot_sources_bit_identical":true,"training_objective":"final student conditional NLL only; no teacher or per-round averaged loss","teacher_logits_and_features_gradients_absent":true,"teacher_potential_temperature_then_feedback_temperature_one":temperature_report,"feedback_visibility":visibility_report,"student_logit_gradient_max_abs":maximum,"observed_logit_gradient_zero":true,"observed_gate_feature_gradient_max_abs":observed_features_max,"selected_embedding_gradient_max_abs":selected_max}))
}

fn contextual_heads(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    let circuit = model.codpula.as_ref().context("contextual head circuit")?;
    let lengths = vec![4,2,0];
    let (count,width,depth) = (3,6,cfg.d_model);
    let observed_values = vec![true,false,true,true,false,false,true,false,false,false,false,false,false,false,false,false,false,false];
    let observed = Tensor::<2,Bool>::from_data(TensorData::new(observed_values.clone(),[count,width]),device);
    let targets = Tensor::<2,Int>::from_data([[11,65,83,121,0,0],[47,91,0,0,0,0],[0,0,0,0,0,0]],device);
    let logits = Tensor::from_data(TensorData::new((0..count*width*VOCAB).map(|index|
        ((index*19%97) as f32-48.0)*0.017).collect::<Vec<_>>(),[count,width,VOCAB]),device).require_grad();
    let features = Tensor::from_data(TensorData::new((0..count*width*depth).map(|index|
        ((index*23%71) as f32-35.0)*0.031).collect::<Vec<_>>(),[count,width,depth]),device).require_grad();
    let reference_logits = logits.clone().detach().require_grad();
    let reference_features = features.clone().detach().require_grad();
    // Independently slice only valid positions, rather than reusing context_summary.
    let mut summaries = Vec::new();
    for (row,&length) in lengths.iter().enumerate() {
        summaries.push(if length == 0 {
            Tensor::zeros([1,3*depth],device)
        } else {
            let valid = reference_features.clone().slice([row..row+1,0..length,0..depth]);
            let mean = valid.sum_dim(1).reshape([1,depth]) / length as f64;
            let first = reference_features.clone().slice([row..row+1,0..1,0..depth]).reshape([1,depth]);
            let last = reference_features.clone().slice([row..row+1,length-1..length,0..depth]).reshape([1,depth]);
            Tensor::cat(vec![mean,first,last],1)
        });
    }
    let projected = circuit.gate_head.forward(Tensor::cat(summaries,0));
    let gate_probe = Tensor::zeros([count,8,8],device).require_grad();
    let root_probe = Tensor::zeros([count,STATES],device).require_grad();
    let gates = projected.clone().slice([0..count,0..64]).reshape([count,8,8]) + gate_probe.clone();
    let roots = projected.slice([0..count,64..96]) + root_probe.clone();
    let temperature = 0.7;
    let detached_observed = reference_logits.clone().mask_where(
        observed.clone().reshape([count,width,1]).expand([count,width,VOCAB]),reference_logits.clone().detach()) / temperature;
    let reference = PreparedCircuit::prepare_contextual(circuit.probability.log_emission.val(),
        circuit.probability.log_transition.val(),circuit.probability.log_root.val(),detached_observed,gates,roots,lengths.clone(),GROUP_WIDTH)?;
    let actual = circuit.prepare(logits.clone(),features.clone(),&lengths,observed.clone(),temperature)?;
    let marginals = compare(&values(actual.marginals(targets.clone(),observed.clone())?),
        &values(reference.marginals(targets.clone(),observed.clone())?))?;
    let evidence = compare(&values(actual.log_partition(targets.clone(),observed.clone())?),
        &values(reference.log_partition(targets.clone(),observed.clone())?))?;
    let actual = actual.conditional_nll(targets.clone(),observed.clone(),6.0)?;
    let reference = reference.conditional_nll(targets,observed,6.0)?;
    let loss = compare(&values(actual.loss.clone()),&values(reference.loss.clone()))?;
    let flows = compare_flows(&actual.flows,&reference.flows)?;
    let actual_gradients = actual.loss.backward();
    let reference_gradients = reference.loss.backward();
    let logit_gradient = values(logits.grad(&actual_gradients).context("contextual token-potential gradient")?);
    let feature_gradient = values(features.grad(&actual_gradients).context("contextual summary-feature gradient")?);
    let logits_report = compare(&logit_gradient,&values(reference_logits.grad(&reference_gradients).context("reference contextual token gradient")?))?;
    let features_report = compare(&feature_gradient,&values(reference_features.grad(&reference_gradients).context("reference contextual feature gradient")?))?;
    let weight_gradient = values(circuit.gate_head.weight.val().grad(&actual_gradients).context("contextual head weight gradient")?);
    let bias = circuit.gate_head.bias.as_ref().context("contextual head bias")?;
    let bias_gradient = values(bias.val().grad(&actual_gradients).context("contextual head bias gradient")?);
    let weights_report = compare(&weight_gradient,&values(circuit.gate_head.weight.val().grad(&reference_gradients).context("reference contextual head weight gradient")?))?;
    let bias_report = compare(&bias_gradient,&values(bias.val().grad(&reference_gradients).context("reference contextual head bias gradient")?))?;
    let root_gradient = values(root_probe.grad(&reference_gradients).context("contextual root-loggate gradient")?);
    let transition_gradient = values(gate_probe.grad(&reference_gradients).context("contextual transition-loggate gradient")?);
    let root_max = nonzero_finite(&root_gradient[..2*STATES],"active contextual root gates")?;
    let transition_max = nonzero_finite(&transition_gradient[..2*64],"active contextual transition gates")?;
    ensure!(root_gradient[2*STATES..].iter().chain(&transition_gradient[2*64..]).all(|&x|x == 0.0),
        "inactive block trains context gates");
    let root_weight_gradient = weight_gradient.chunks_exact(96).flat_map(|row|row[64..].iter().copied()).collect::<Vec<_>>();
    let root_weights_max = nonzero_finite(&root_weight_gradient,"contextual root-head weights")?;
    let root_bias_max = nonzero_finite(&bias_gradient[64..],"contextual root-head bias")?;
    let mut observed_features = Vec::new();
    for (row,&length) in lengths.iter().enumerate() {
        for position in 0..width {
            let index = row*width+position;
            if position >= length {
                ensure!(logit_gradient[index*VOCAB..(index+1)*VOCAB].iter().chain(&feature_gradient[index*depth..(index+1)*depth]).all(|&x|x == 0.0),
                    "contextual padded token/features have a derivative");
            } else if observed_values[index] {
                ensure!(logit_gradient[index*VOCAB..(index+1)*VOCAB].iter().all(|&x|x == 0.0),"contextual observed output potential is not detached");
                observed_features.push(json!({"row":row,"position":position,"maximum":nonzero_finite(&feature_gradient[index*depth..(index+1)*depth],"observed contextual feature")?}));
            }
        }
    }
    let mut summary_gradients = Vec::new();
    for (name,index) in [("valid_mean",0),("first_valid",1),("last_valid",2)] {
        summary_gradients.push(json!({"summary":name,"weight_gradient_max_abs":nonzero_finite(&weight_gradient[index*depth*96..(index+1)*depth*96],name)?}));
    }
    Ok(json!({"passed":true,"lengths":lengths,"temperature":temperature,"independent_valid_mean_first_last_reference":true,
        "joint_marginals":marginals,"visible_log_mass":evidence,"conditional_nll":loss,"circuit_flows":flows,
        "token_potential_gradients":logits_report,"feature_gradients":features_report,"head_weight_gradients":weights_report,"head_bias_gradients":bias_report,
        "root_loggate_gradient_max_abs":root_max,"transition_loggate_gradient_max_abs":transition_max,
        "root_head_weight_gradient_max_abs":root_weights_max,"root_head_bias_gradient_max_abs":root_bias_max,
        "observed_features":observed_features,"summary_weight_gradients":summary_gradients,
        "observed_output_potentials_detached":true,"padding_and_inactive_gate_derivatives_zero":true}))
}

fn training_seed(cfg: &Config, examples: &[Example]) -> Result<u64> {
    for seed in 0..65_536 {
        let blocks = sbd::sample_training_blocks(cfg, examples, seed)?;
        if blocks.iter().any(|block| block.masked.len() > 1 && block.masked.iter().any(|x| !x)) && blocks.iter().any(|block| block.masked.iter().any(|x| *x)) { return Ok(seed); }
    }
    anyhow::bail!("no mixed observed/unresolved training fixture within 65536 seeds")
}
fn training_paths(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    let examples = rows(2, 11);
    let batch = Batch::new(&examples, device);
    let mixed_seed = training_seed(cfg, &examples)?;
    let mut reports = Vec::new();
    for feedback in [SbdFeedback::HardMask, SbdFeedback::SoftFeedback] {
        let mut per = cfg.clone();
        per.sbd.as_mut().unwrap().feedback = feedback;
        per.sbd.as_mut().unwrap().block_execution = SbdBlockExecution::PerBlock;
        let mut batched = per.clone();
        batched.sbd.as_mut().unwrap().block_execution = SbdBlockExecution::Batched;
        let reference = sbd::training_loss(model, &per, &examples, &batch, mixed_seed)?;
        let actual = sbd::training_loss(model, &batched, &examples, &batch, mixed_seed)?;
        let loss = compare(&values(actual.loss.clone()), &values(reference.loss.clone()))?;
        let flows = compare_flows(actual.circuit_flows.as_ref().context("batched circuit flows")?, reference.circuit_flows.as_ref().context("per-block circuit flows")?)?;
        let reference_gradients = model::checked_gradients(model, reference.loss)?;
        let actual_gradients = model::checked_gradients(model, actual.loss)?;
        let gradients = compare_snapshots(&snapshot(model,Some(&actual_gradients))?, &snapshot(model,Some(&reference_gradients))?, false)?;
        reports.push(json!({"feedback":feedback,"loss":loss,"all_neural_gradients":gradients,"separate_nonnegative_circuit_flows":flows}));
    }
    let mut weighted = examples.clone();
    weighted[0].weight[0] = 0.5;
    let error = sbd::training_loss(model, cfg, &weighted, &Batch::new(&weighted, device), mixed_seed).err().context("fractional joint target weight was accepted")?;
    ensure!(error.to_string().contains("unit"), "unsupported joint weights did not explain unit-weight requirement: {error}");
    let mut singleton = cfg.clone();
    singleton.sbd.as_mut().unwrap().block_lengths = vec![1];
    let result = sbd::training_loss(model, &singleton, &examples, &batch, 1)?;
    ensure!(result.circuit_flows.is_none(), "L1 bypass produces circuit updates");
    let mut factorized = singleton.clone();
    factorized.sbd.as_mut().unwrap().joint_output = SbdJointOutput::Factorized;
    let mut causal = model.clone();
    causal.codpula = None;
    let baseline = sbd::training_loss(&causal,&factorized,&examples,&batch,1)?;
    let singleton_loss = identical(&values(result.loss.clone()),&values(baseline.loss.clone()),"L1 joint/factorized loss")?;
    let singleton_gradients = snapshot(model,Some(&model::checked_gradients(model,result.loss)?))?;
    let singleton_common_gradients = compare_snapshots(&common(&singleton_gradients),&snapshot(&causal,Some(&model::checked_gradients(&causal,baseline.loss)?))?,true)?;
    for (name,gradient) in &singleton_gradients {
        if name.split('.').any(|part|part == "codpula") { ensure!(gradient.values.iter().all(|&x|x == 0.0),"L1 bypass trains context head {name}"); }
    }
    let one_row = rows(1, 4);
    let mut all_observed_seed = None;
    for seed in 0..65_536 { if sbd::sample_training_blocks(cfg,&one_row,seed)?.is_empty() { all_observed_seed = Some(seed); break; } }
    let seed = all_observed_seed.context("no all-observed training draw")?;
    let batch = Batch::new(&one_row,device);
    let result = sbd::training_loss(model,cfg,&one_row,&batch,seed)?;
    ensure!(result.circuit_flows.is_none(), "all-observed draw produces circuit updates");
    let clean = model.loss(model.forward(&batch),&batch);
    let clean_report = identical(&values(result.loss.clone()),&values(clean),"all-observed training loss is not causal loss")?;
    let _ = snapshot(model,Some(&model::checked_gradients(model,result.loss)?))?;
    Ok(json!({"passed":true,"mixed_mask_seed":mixed_seed,"sampled_blocks":sbd::sample_training_blocks(cfg,&examples,mixed_seed)?.iter().map(|block|json!({"row":block.row,"start":block.start,"length":block.length,"masked":block.masked})).collect::<Vec<_>>(),"execution_paths":reports,"fractional_weight_rejection":error.to_string(),"length_one_no_circuit_update":true,"length_one_factorized_loss":singleton_loss,"length_one_factorized_common_gradients":singleton_common_gradients,"length_one_head_gradients_exact_zero":true,"all_observed_seed":seed,"all_observed_loss":clean_report,"bypass_neural_gradients_present_and_finite":true}))
}

fn circuit_state(model: &LanguageModel) -> Result<Value> {
    let circuit = model.codpula.as_ref().context("missing circuit")?;
    let state = &circuit.probability;
    let mut normalized = BTreeMap::new();
    for (name, rows, columns, logs) in [
        ("emission", STATES, VOCAB, values(state.log_emission.val())),
        ("transition", STATES, STATES, values(state.log_transition.val())),
        ("root", 1, STATES, values(state.log_root.val())),
    ] {
        ensure!(logs.iter().all(|x| x.is_finite()), "nonfinite {name} log probabilities");
        let sums = logs.chunks_exact(columns).map(|row| row.iter().map(|&x| f64::from(x).exp()).sum::<f64>()).collect::<Vec<_>>();
        let report = sbd::compare(&sums,&vec![1.0;rows])?;
        normalized.insert(name,report);
    }
    let parameters = snapshot(model,None)?;
    for (name, parameter) in &parameters {
        ensure!(parameter.values.iter().all(|x| x.is_finite()), "nonfinite complete-step model state {name}");
        if name.split('.').any(|part| part == "probability") {
            ensure!(!parameter.trainable,"circuit state entered neural optimizer: {name}");
            if name.ends_with("ema") { ensure!(parameter.values.iter().all(|&x| x >= 0.0), "negative circuit EMA {name}"); }
        }
    }
    let step = state.step.val().into_data().try_to_vec_as::<i32>().map_err(|e|anyhow::anyhow!("circuit update step {e:?}"))?;
    Ok(json!({"passed":true,"update_step":step[0],"row_normalization":normalized,"circuit_parameters_frozen":true,"ema_state_finite_nonnegative":true}))
}
fn complete_step(model: LanguageModel, cfg: &Config, optimizer: &mut ModuleOptimizer, examples: &[Example], seed: u64, device: &Device) -> Result<(LanguageModel,Value)> {
    let before = snapshot(&model,None)?;
    let previous_circuit_step = model.codpula.as_ref().context("complete-step circuit")?.probability.step.val().into_scalar::<i32>();
    let result = sbd::training_loss(&model,cfg,examples,&Batch::new(examples,device),seed)?;
    let loss = model::scalar(result.loss.clone())?;
    let flow = result.circuit_flows.as_ref().context("complete-step fixture has no circuit statistics")?;
    let flow_report = finite_flows(flow)?;
    ensure!(flow.active_blocks > 0,"complete-step fixture has no active blocks");
    let gradients = model::checked_gradients(&model,result.loss)?;
    let gradient_values = snapshot(&model,Some(&gradients))?;
    let gradient_sha256 = snapshot_hash(&gradient_values)?;
    let root_head_gradient = if cfg.sbd.as_ref().unwrap().joint_output == SbdJointOutput::CodpulaContextual {
        let (_,weight) = gradient_values.iter().find(|(name,_)|name.ends_with("codpula.gate_head.weight")).context("complete-step contextual head gradient")?;
        ensure!(weight.shape == [3*cfg.d_model,96], "complete-step contextual gradient shape");
        let root = weight.values.chunks_exact(96).flat_map(|row|row[64..].iter().copied()).collect::<Vec<_>>();
        Some(nonzero_finite(&root,"complete-step contextual root-head gradient")?)
    } else { None };
    let clipping_norms = gradient_values.iter().map(|(name, values)| {
        let norm = values.values.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        (name.clone(),json!({"f64_norm":norm,"clipping_active":norm>f64::from(cfg.gradient_clip)}))
    }).collect::<BTreeMap<_,_>>();
    let updated = train::optimizer_step(cfg,model,optimizer,gradients,result.circuit_flows)?;
    let after = snapshot(&updated,None)?;
    let changed = |neural: bool| before.iter().filter(|(_,v)|v.trainable == neural).any(|(name,value)|value.values.iter().zip(&after[name].values).any(|(a,b)|a.to_bits()!=b.to_bits()));
    ensure!(changed(true),"complete step did not update any neural parameter");
    if root_head_gradient.is_some() {
        let (name,weight) = before.iter().find(|(name,_)|name.ends_with("codpula.gate_head.weight")).context("complete-step contextual head")?;
        ensure!(weight.values.chunks_exact(96).zip(after[name].values.chunks_exact(96))
            .any(|(old,new)|old[64..].iter().zip(&new[64..]).any(|(a,b)|a.to_bits()!=b.to_bits())),
            "complete step did not update contextual root-head weights");
    }
    ensure!(before.iter().filter(|(name,_)|name.ends_with("log_emission") || name.ends_with("log_transition") || name.ends_with("log_root")).any(|(name,value)|value.values.iter().zip(&after[name].values).any(|(a,b)|a.to_bits()!=b.to_bits())), "complete step did not update circuit probability rows");
    let circuit_report = circuit_state(&updated)?;
    ensure!(circuit_report["update_step"].as_i64() == Some(i64::from(previous_circuit_step)+1),"complete step did not advance circuit momentum time exactly once");
    let report = json!({"passed":true,"loss":loss,"mask_seed":seed,"neural_gradient_tensors":gradient_values.len(),"gradient_sha256":gradient_sha256,"gradient_norms":clipping_norms,"contextual_root_head_gradient_max_abs":root_head_gradient,"neural_parameters_changed":changed(true),"circuit_parameters_changed":true,"flows":flow_report,"state":circuit_report});
    Ok((updated,report))
}
fn checkpoint_continuation(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    let mut checkpoint_config = cfg.clone();
    checkpoint_config.context = 11;
    checkpoint_config.batch_size = 2;
    checkpoint_config.train_tokens = 66;
    let cfg = &checkpoint_config;
    let examples = rows(2,11);
    let seed = training_seed(cfg,&examples)?;
    let mut optimizer = train::optimizer(cfg);
    let (trained,first) = complete_step(model.clone(),cfg,&mut optimizer,&examples,seed,device)?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = cfg.artifact_dir.join("model-check").join(format!("{}-{stamp}",std::process::id()));
    let path = directory.join("checkpoint-1");
    let state = train::State { schema:1,config_sha256:cfg.training_fingerprint(cfg.resolved_tokens()?)?,data:json!({"fixture_sha256":data::fixture_hash(&examples)}),arm:cfg.arms[0].name.clone(),seed:cfg.seed,step:1,tokens:22,training_seconds:0.0,initial_evaluation:Value::Null };
    train::checkpoint(cfg,&path,&trained,&optimizer,&state)?;
    let restored = checkpoint::load_model(LanguageModel::new(cfg,&cfg.arms[0],device)?,path.join("model.burnpack"),device)?;
    let mut restored_optimizer = checkpoint::load_optimizer(cfg,path.join("optimizer.burnpack"))?;
    let loaded_state: train::State = serde_json::from_slice(&std::fs::read(path.join("state.json"))?)?;
    ensure!(loaded_state.schema == 1 && loaded_state.config_sha256 == cfg.training_fingerprint(cfg.resolved_tokens()?)? && loaded_state.arm == cfg.arms[0].name && loaded_state.seed == cfg.seed && loaded_state.step == 1 && loaded_state.tokens == 22 && loaded_state.data == state.data && loaded_state.training_seconds == state.training_seconds && loaded_state.initial_evaluation == state.initial_evaluation,"saved checkpoint run identity/counters changed");
    let loaded = compare_snapshots(&snapshot(&restored,None)?,&snapshot(&trained,None)?,true)?;
    ensure!(circuit_state(&restored)? == circuit_state(&trained)?,"loaded circuit update time/row state differs");
    ensure!(checkpoint::optimizer_signature(cfg,optimizer.into_bytes()?)? == checkpoint::optimizer_signature(cfg,restored_optimizer.into_bytes()?)?,"restored AdamW state differs");
    let mut steps = Vec::new();
    let mut uninterrupted = trained;
    let mut continued = restored;
    for index in 0..2 {
        let mut next_examples = examples.clone();
        for example in &mut next_examples {
            let shift = index as i32 + 1;
            for input in example.input.iter_mut().skip(1) { *input = (*input + shift) % 256; }
            for target in &mut example.target { *target = (*target + shift) % 256; }
        }
        let selected_seed = training_seed(cfg,&next_examples)?;
        let (next,actual) = complete_step(uninterrupted,cfg,&mut optimizer,&next_examples,selected_seed,device)?;
        let (loaded_next,loaded_step) = complete_step(continued,cfg,&mut restored_optimizer,&next_examples,selected_seed,device)?;
        runtime::write_json(&cfg.artifact_dir.join("continuation-diagnostic.json"),&json!({
            "continuation_step":index+2,"uninterrupted":actual,"restored":loaded_step,
            "raw_gradients_bit_identical":actual["gradient_sha256"]==loaded_step["gradient_sha256"]}))?;
        ensure!(actual["gradient_sha256"]==loaded_step["gradient_sha256"],
            "checkpoint continuation raw gradients differ before optimizer");
        let parameters = compare_snapshots(&snapshot(&loaded_next,None)?,&snapshot(&next,None)?,true)?;
        ensure!(actual["loss"] == loaded_step["loss"],"checkpoint continuation loss differs");
        ensure!(circuit_state(&loaded_next)? == circuit_state(&next)?,"checkpoint continuation circuit time differs");
        ensure!(checkpoint::optimizer_signature(cfg,optimizer.into_bytes()?)? == checkpoint::optimizer_signature(cfg,restored_optimizer.into_bytes()?)?,"checkpoint continuation AdamW moments differ");
        steps.push(json!({"step":index+2,"loss":actual["loss"],"parameters_and_circuit_ema":parameters,"optimizer_logical_state_identical":true,"circuit_update_step":circuit_state(&next)?["update_step"]}));
        uninterrupted = next;
        continued = loaded_next;
    }
    Ok(json!({"passed":true,"config":cfg,"directory":directory,"checkpoint":path,"first_complete_step":first,"loaded_float_parameters_and_circuit_ema":loaded,"loaded_optimizer_state_identical":true,"saved_run_identity_and_counters_identical":true,"continuation":steps,"payloads_retained_for_parent_cleanup":true}))
}

fn inactive_rows(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    let mut reports = Vec::new();
    for (name, active, observed) in [("all_observed",vec![true;4],vec![Some(65);4]),("inactive",vec![false;4],vec![Some(256);4])] {
        let state = soft(cfg,observed,active,device)?;
        let output = sbd::block_forward(model,&[256,31,72],&state,false,device);
        let (prepared,_,fixed) = sbd::prepare_circuit(model,output,&state,1.0)?;
        let result = prepared.conditional_nll(Tensor::<2,Int>::from_data([[65,65,65,65]],device),fixed,4.0)?;
        ensure!(model::scalar(result.loss)? == 0.0 && result.flows.active_blocks == 0,"{name} has a noisy loss or active circuit update");
        for (field,values) in flow_values(&result.flows) { ensure!(values.iter().all(|&x|x==0.0),"{name} has nonzero {field}"); }
        let mut updated = model.clone();
        updated.codpula.as_mut().unwrap().update(result.flows)?;
        let unchanged = compare_snapshots(&snapshot(&updated,None)?,&snapshot(model,None)?,true)?;
        ensure!(circuit_state(&updated)? == circuit_state(model)?,"inactive update advances circuit time");
        reports.push(json!({"case":name,"loss_and_flows_exact_zero":true,"update_identity":unchanged}));
    }
    Ok(json!({"passed":true,"cases":reports,"joint_output":cfg.sbd.as_ref().unwrap().joint_output}))
}
fn evaluation_and_generation(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    let mut cfg = cfg.clone();
    let s = cfg.sbd.as_mut().unwrap();
    s.block_lengths = vec![1,4];
    s.generate_tokens = 5;
    s.commitment = SbdCommitment::Dependence;
    let evaluation = sbd::evaluate(model,&cfg,&rows(1,5),device)?;
    ensure!(evaluation["predicted_tokens"] == 10 && evaluation["block_nll"].as_f64().is_some_and(f64::is_finite),"joint evaluation omits a configured length or has nonfinite NLL");
    let generated = sbd::generate(model,&cfg,&[256,65,66],device)?;
    let repeated = sbd::generate(model,&cfg,&[256,65,66],device)?;
    ensure!(generated["tokens"] == repeated["tokens"],"seeded joint generation is not repeatable");
    ensure!(generated["finalized_tokens"] == 5 && generated["causally_replayed_tokens"] == 5 && generated["commit_rule"] == "dependence","joint generation did not complete/replay every requested position");
    ensure!(generated["tokens"].as_array().context("generated token list")?.iter().all(|x|x.as_u64().is_some_and(|x|x<VOCAB as u64)),"joint generation leaves the full output vocabulary");
    Ok(json!({"passed":true,"evaluation":evaluation,"generation_tokens":generated["tokens"],"generation_passes":generated["denoising_passes"],"seeded_repeat_identical":true,"all_requested_tokens_finalized_and_causally_replayed":true}))
}

struct RolloutRound {
    proposals: Vec<usize>,
    commits: Vec<usize>,
    observed: Vec<Option<usize>>,
    floats: BTreeMap<&'static str,Vec<f32>>,
}
struct RolloutRun {
    rounds: Vec<RolloutRound>,
    next_random: Vec<f32>,
}
fn rollout_uniforms(round: usize, state: &SoftState, poison_unused: bool, device: &Device) -> Tensor<3> {
    let width = state.active.len();
    let draws = (0..width*2).map(|index| {
        let position = index/2;
        if poison_unused && (!state.active[position] || (index%2 == 1 && state.observed[position].is_some())) {
            f32::NAN
        } else {
            (((round*13+position*7+(index%2)*29)%89) as f32+0.5)/89.0
        }
    }).collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(draws,[1,width,2]),device)
}

// Adapted from the current predecessor scheduler's fixed-uniform, same-cache
// and absorbing-field regressions. This is a decision rollout, not its
// all-round averaged training objective.
fn rollout_run(model: &LanguageModel, cfg: &Config, device: &Device, width: usize, cached: bool) -> Result<RolloutRun> {
    const LENGTH: usize = 4;
    let examples = rows(1,23);
    let before = snapshot(model,None)?;
    let circuit_before = circuit_state(model)?;
    device.seed(8271);
    let mut trace = CleanInputs::default();
    let _ = model.forward_profiled(&Batch::new(&examples,device),&mut trace);
    let clean_before = trace.0.iter().map(|(_,_,tensor)|values(tensor.clone())).collect::<Vec<_>>();
    let cache = model.training_cache(&trace,0,&[19]);
    let active = (0..width).map(|position|position<LENGTH).collect::<Vec<_>>();
    let mut supplied = vec![None;width];
    supplied[1] = Some(65);
    let mut state = soft(cfg,supplied,active,device)?;
    let mut rounds = Vec::new();
    for round in 0..3 {
        let previous = state.observed.clone();
        let output = if cached {
            model.cached_block(&cache,0,&state,false)
        } else {
            sbd::block_forward(model,&examples[0].input[..19],&state,false,device)
        };
        let (prepared,tokens,fixed) = sbd::prepare_circuit(model,output.clone().detach(),&state,cfg.soft_temperature)?;
        let uniforms = rollout_uniforms(round,&state,false,device);
        let proposals = prepared.sample_with_uniforms(tokens.clone(),fixed.clone(),uniforms)?;
        let poisoned = prepared.sample_with_uniforms(tokens.clone(),fixed.clone(),rollout_uniforms(round,&state,true,device))?;
        ensure!(proposals == poisoned,"unused fixed/padded proposal uniforms change a rollout");
        ensure!(proposals.len() == 1 && proposals[0].len() == LENGTH,"proposal storage includes right padding");
        let proposals = proposals.into_iter().next().unwrap();
        if round == 0 {
            let inactive = DenoisingOutput {
                logits:Tensor::full([1,width,VOCAB],f32::NAN,device),
                features:Tensor::full([1,width,cfg.d_model],f32::NAN,device),
            };
            let padded_output = DenoisingOutput {
                logits:Tensor::cat(vec![inactive.logits,output.logits.clone()],0),
                features:Tensor::cat(vec![inactive.features,output.features.clone()],0),
            };
            let padded_state = soft(cfg,std::iter::repeat_n(None,width).chain(state.observed.iter().copied()).collect(),
                std::iter::repeat_n(false,width).chain(state.active.iter().copied()).collect(),device)?;
            let (padded,values,fixed) = sbd::prepare_circuit(model,padded_output.detach(),&padded_state,cfg.soft_temperature)?;
            let draws = Tensor::cat(vec![Tensor::full([1,width,2],f32::NAN,device),rollout_uniforms(round,&state,true,device)],0);
            let sampled = padded.sample_with_uniforms(values,fixed,draws)?;
            ensure!(sampled.len() == 2 && sampled[0].is_empty() && sampled[1] == proposals,
                "inserting an inactive row changes indexed proposals");
        }
        let marginals = prepared.marginals(tokens.clone(),fixed.clone())?;
        let evidence = values(prepared.log_partition(tokens.clone(),fixed.clone())?);
        let scores = prepared.dependence_scores(tokens,fixed,&[proposals.clone()])?.remove(0);
        let unresolved = (0..LENGTH).filter(|&position|state.observed[position].is_none()).collect::<Vec<_>>();
        // Existing selector tests cover competing scores/ties. Here one eligible
        // unresolved position per call guarantees a multi-call absorbing-state case.
        let eligible = if round == 2 { unresolved.clone() } else { unresolved.iter().take(1).copied().collect() };
        let commits = sbd::dependence_commits(&eligible,&scores,0.0,round == 2);
        ensure!(!commits.is_empty(),"rollout selector does not make progress");
        let feedback_logits = marginals.clone().log().reshape([width,VOCAB]);
        let mut feedback_only = state.clone();
        feedback_only.update(feedback_logits.clone().detach(),cfg.soft_eta,1.0,cfg.soft_steps,cfg.sbd.as_ref().unwrap().candidate_width)?;
        let mut stored_proposals = proposals.clone();
        stored_proposals.resize(width,0);
        sbd::advance_generation_state(cfg,&mut state,&stored_proposals,&commits,feedback_logits.clone(),1.0,device)?;
        let dense = values(state.candidates.dense(VOCAB));
        let expected_feedback = values(feedback_only.candidates.dense(VOCAB));
        let visibility = values(state.visibility.clone());
        for position in 0..width {
            if let Some(value) = previous[position] {
                ensure!(state.observed[position] == Some(value),"a later call revises supplied or committed evidence");
            } else if commits.contains(&position) {
                ensure!(state.observed[position] == Some(proposals[position]),"a selected proposal was not committed");
            } else {
                ensure!(state.observed[position].is_none(),"a deferred or padded proposal became observed");
                identical(&dense[position*VOCAB..(position+1)*VOCAB],&expected_feedback[position*VOCAB..(position+1)*VOCAB],"deferred proposal was hardened instead of retaining soft feedback")?;
            }
            if let Some(value) = state.observed[position] {
                ensure!(visibility[position] == 1.0 && dense[position*VOCAB..(position+1)*VOCAB].iter().enumerate().all(|(id,&p)|p == if id == value {1.0}else{0.0}),
                    "committed evidence is not a visible immutable point mass");
            } else if !state.active[position] {
                ensure!(visibility[position] == 0.0 && dense[position*VOCAB..(position+1)*VOCAB].iter().all(|&p|p == 0.0),"padding acquires beliefs or visibility");
            }
        }
        let mut attempted_revision = state.clone();
        let conflicting = stored_proposals.iter().map(|&value|(value+17)%VOCAB).collect::<Vec<_>>();
        let protected = (0..width).filter(|&position|state.observed[position].is_some() || !state.active[position]).collect::<Vec<_>>();
        sbd::advance_generation_state(cfg,&mut attempted_revision,&conflicting,&protected,feedback_logits,1.0,device)?;
        ensure!(attempted_revision.observed == state.observed,"repeated writes revise an absorbing observation or commit padding");
        for ((_,kind,tensor),before) in trace.0.iter().zip(&clean_before) {
            identical(&values(tensor.clone()),before,&format!("rollout changed clean {kind} snapshot"))?;
        }
        rounds.push(RolloutRound { proposals,commits,observed:state.observed[..LENGTH].to_vec(),
            floats:BTreeMap::from([
                ("final_features",active_values(output.features,&[LENGTH])),
                ("logits",active_values(output.logits,&[LENGTH])),
                ("joint_marginals",active_values(marginals,&[LENGTH])),
                ("evidence_log_mass",evidence),
                ("dependence_scores",scores.iter().map(|&value|value as f32).collect()),
                ("feedback",dense[..LENGTH*VOCAB].to_vec()),
                ("visibility",visibility[..LENGTH].to_vec()),
            ]) });
    }
    ensure!(state.observed[..LENGTH].iter().all(Option::is_some),"fixed-uniform rollout does not complete");
    let next_random = values(Tensor::<1>::random([8],burn::tensor::Distribution::Uniform(0.0,1.0),device));
    let _ = compare_snapshots(&snapshot(model,None)?,&before,true)?;
    ensure!(circuit_state(model)? == circuit_before,"decision rollout changes circuit rows, momentum or time");
    Ok(RolloutRun { rounds,next_random })
}
fn compare_rollouts(actual: &RolloutRun, expected: &RolloutRun) -> Result<Value> {
    ensure!(actual.rounds.len() == expected.rounds.len(),"rollout call counts differ");
    let mut reports = Vec::new();
    for (index,(actual,expected)) in actual.rounds.iter().zip(&expected.rounds).enumerate() {
        ensure!(actual.proposals == expected.proposals && actual.commits == expected.commits && actual.observed == expected.observed,
            "fixed-uniform rollout proposal/commit/evidence decisions differ at call {index}");
        let mut floats = BTreeMap::new();
        for (name,reference) in &expected.floats { floats.insert(name,compare(&actual.floats[name],reference)?); }
        reports.push(json!({"call":index+1,"proposals":actual.proposals,"commits":actual.commits,"observations":actual.observed,"values":floats}));
    }
    Ok(json!({"passed":true,"calls":reports,"next_device_rng":identical(&actual.next_random,&expected.next_random,"rollout changes next device RNG")?}))
}
fn scheduler_rollout(model: &LanguageModel, cfg: &Config, device: &Device) -> Result<Value> {
    device.seed(8271);
    let untouched_rng = values(Tensor::<1>::random([8],burn::tensor::Distribution::Uniform(0.0,1.0),device));
    let reconstructed = rollout_run(model,cfg,device,4,false)?;
    let cached = rollout_run(model,cfg,device,4,true)?;
    let padded = rollout_run(model,cfg,device,8,true)?;
    Ok(json!({"passed":true,"proposal_uniform_layout":"[row,position,(latent,token)], fixed separately for each call","calls":3,
        "cached_vs_reconstructed":compare_rollouts(&cached,&reconstructed)?,"padded_vs_unpadded":compare_rollouts(&padded,&cached)?,
        "rng_unchanged_without_rollout":identical(&reconstructed.next_random,&untouched_rng,"indexed rollout consumes device RNG")?,
        "same_clean_cache_all_calls":true,"clean_snapshot_and_model_state_unchanged":true,"absorbing_supplied_and_committed_values":true,
        "unused_fixed_padding_and_inactive_row_uniforms_ignored":true,"deferred_proposals_keep_soft_beliefs":true,
        "source_reused":{"diffusion_scheduler.rs":[[125,216],[265,301]],"diffusion_scheduler_tests.rs":[[35,38],[72,91],[102,182],[192,293]],"training_query_chain.rs":[[646,700]]},
        "objective_policy":"No rollout loss is added; SBD retains its detached teacher and final-student-only training objective"}))
}


fn historical_checkpoint_compatibility(device: &Device) -> Result<Value> {
    let artifacts = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
    // This Step 8 payload predates recurrent-only SBD and must remain rejected.
    let dense_directory = artifacts.join("step8/pilot-soft/kronlstm-hola-seed1");
    let dense_metrics: Value = serde_json::from_slice(&std::fs::read(dense_directory.join("metrics.json"))?)?;
    let dense: Config = serde_json::from_value(dense_metrics["config"].clone())?;
    let dense_fingerprint = dense.training_fingerprint(dense.resolved_tokens()?)?;
    ensure!(dense_metrics["config_sha256"] != dense_fingerprint,"obsolete dense-SBD revision was accepted");
    let mut explicit = dense_metrics["config"].clone();
    explicit["sbd"]["joint_output"] = json!("factorized");
    let explicit: Config = serde_json::from_value(explicit)?;
    ensure!(serde_json::to_vec(&explicit)? == serde_json::to_vec(&dense)?
        && explicit.training_fingerprint(explicit.resolved_tokens()?)? == dense_fingerprint,
        "explicit factorized changes omitted-selector serialization or identity");
    let rejection = sbd::checkpoint_report(&dense,&dense.arms[0],dense.seed,device,&dense_directory.join("checkpoint-4"),false)
        .err().context("obsolete dense-SBD checkpoint unexpectedly resumed")?.to_string();
    ensure!(rejection.contains("dense-attention"),"obsolete dense-SBD checkpoint failed for another reason: {rejection}");

    // Reuse the exact historical continuation fixture accepted by the batching turn.
    let recorded: Value = serde_json::from_slice(&std::fs::read(artifacts.join("batching/historical/checkpoint-check.json"))?)?;
    ensure!(recorded["passed"] == true,"historical continuation evidence was not accepted");
    let cfg: Config = serde_json::from_value(recorded["config"].clone())?;
    ensure!(cfg.sbd.is_none(),"pinned historical causal fixture changed");
    let source = PathBuf::from(recorded["reports"][0]["source_checkpoint"].as_str().context("historical checkpoint path")?);
    let state: train::State = serde_json::from_slice(&std::fs::read(source.join("state.json"))?)?;
    let fingerprint = cfg.training_fingerprint(cfg.resolved_tokens()?)?;
    ensure!(state.config_sha256 == fingerprint,"historical causal checkpoint identity changed");
    let arm = cfg.arms.iter().find(|arm|arm.name==state.arm).context("historical checkpoint arm")?;
    let path = source.join("model.burnpack");
    let model = checkpoint::load_model(LanguageModel::new(&cfg,arm,device)?,&path,device)?;
    ensure!(model.codpula.is_none(),"historical causal checkpoint acquired a circuit");
    let optimizer_path = source.join("optimizer.burnpack");
    let signature = checkpoint::optimizer_file_signature(&cfg,&optimizer_path)?;
    ensure!(signature["tensors"].as_object().is_some_and(|x|!x.is_empty()),"historical checkpoint contains no optimizer moments");
    Ok(json!({"passed":true,"fixture":source,"config":cfg,"checkpoint_step":state.step,
        "config_sha256":fingerprint,"model_file_sha256":checkpoint::file_sha256(&path)?,
        "optimizer_file_sha256":checkpoint::file_sha256(&optimizer_path)?,
        "omitted_explicit_factorized_serialization_identical":true,
        "obsolete_dense_sbd_rejected":rejection,
        "loaded_model_values_shapes_dtypes_and_parameter_ids_validated":true,
        "historical_payloads_unchanged":true}))
}

fn maximum_errors(value: &Value, absolute: &mut f64, ratio: &mut f64) {
    match value {
        Value::Object(fields) => {
            if let Some(x) = fields.get("max_abs").and_then(Value::as_f64) { *absolute = absolute.max(x); }
            if let Some(x) = fields.get("maximum_tolerance_ratio").and_then(Value::as_f64) { *ratio = ratio.max(x); }
            for child in fields.values() { maximum_errors(child,absolute,ratio); }
        }
        Value::Array(items) => for child in items { maximum_errors(child,absolute,ratio); },
        _ => {}
    }
}

/// Called by the ordinary experiment CLI; no second integration-test executable.
pub fn run(c: &Config, device: &Device) -> Result<Value> {
    ensure!(matches!(c.backend.as_str(),"cpu"|"vulkan"),"Codpula model checks support CPU or Vulkan");
    let clipping = crate::update::clipping_checks(device)?;
    let device = device.clone().autodiff();
    let cpu = Device::ndarray().autodiff();
    let identity = selector_identity(c,&device)?;
    let mut cases = Vec::new();
    let mut continuation = None;
    let mut contextual_continuation = None;
    // Retain the historical matrix. The contextual profile fixes KronLSTM and all
    // existing auxiliary components rather than adding another architecture sweep.
    for (mixer,extended,joint_output) in [
        ("gdn2",false,SbdJointOutput::Codpula),
        ("gdn2",true,SbdJointOutput::Codpula),
        ("kronlstm",false,SbdJointOutput::Codpula),
        ("kronlstm",true,SbdJointOutput::Codpula),
        ("kronlstm",true,SbdJointOutput::CodpulaContextual),
    ] {
            let (mut cfg,arm) = fixture_config(c,mixer,extended);
            cfg.sbd.as_mut().unwrap().joint_output = joint_output;
            cpu.seed(cfg.seed);
            let mut reference = LanguageModel::new(&cfg,&arm,&cpu)?;
            probe_gates(&mut reference,&cpu);
            let mut tested = reference.clone().fork(&device);
            requested_backends(&mut tested,&mut cfg,c);
            let examples = rows(1,23);
            let observed = vec![None,Some(65),None,None];
            let cpu_state = soft(&cfg,observed.clone(),vec![true;4],&cpu)?;
            let state = soft(&cfg,observed,vec![true;4],&device)?;
            let cpu_target = Tensor::<2,Int>::from_data([[11,65,83,121]],&cpu);
            let target = Tensor::<2,Int>::from_data([[11,65,83,121]],&device);
            let expected = readout(&reference,sbd::block_forward(&reference,&examples[0].input[..19],&cpu_state,false,&cpu),&cpu_state,cpu_target,4.0)?;
            let actual = readout(&tested,sbd::block_forward(&tested,&examples[0].input[..19],&state,false,&device),&state,target,4.0).with_context(||format!("{mixer} extended={extended}: joint readout"))?;
            let mut backend = compare_readouts(&actual,&expected)?;
            let expected_gradients = model::checked_gradients(&reference,expected.loss)?;
            let actual_gradients = model::checked_gradients(&tested,actual.loss)?;
            backend["all_neural_gradients"] = compare_snapshots(&snapshot(&tested,Some(&actual_gradients))?,&snapshot(&reference,Some(&expected_gradients))?,false)?;
            let paths = output_paths(&tested,&cfg,&device).with_context(||format!("{mixer} extended={extended}: output paths"))?;
            let locality = locality_and_singletons(&tested,&cfg,&device).with_context(||format!("{mixer} extended={extended}: locality and singletons"))?;
            let gradients = teacher_and_student(&tested,&cfg,&device).with_context(||format!("{mixer} extended={extended}: teacher/student gradients"))?;
            let training = training_paths(&tested,&cfg,&device).with_context(||format!("{mixer} extended={extended}: training paths"))?;
            let inactive = inactive_rows(&tested,&cfg,&device).with_context(||format!("{mixer} extended={extended}: inactive rows"))?;
            let inference = evaluation_and_generation(&tested,&cfg,&device).with_context(||format!("{mixer} extended={extended}: evaluation/generation"))?;
            let rollout = scheduler_rollout(&tested,&cfg,&device).with_context(||format!("{mixer} extended={extended}: fixed-uniform rollout"))?;
            let heads = if joint_output == SbdJointOutput::CodpulaContextual {
                contextual_continuation = Some(checkpoint_continuation(&tested,&cfg,&device).context("contextual checkpoint continuation")?);
                Some(json!({"cpu":contextual_heads(&reference,&cfg,&cpu)?,"requested_backend":contextual_heads(&tested,&cfg,&device)?}))
            } else {
                if mixer == "kronlstm" && !extended { continuation = Some(checkpoint_continuation(&tested,&cfg,&device).context("legacy checkpoint continuation")?); }
                None
            };
            cases.push(json!({"mixer":mixer,"joint_output":joint_output,"ohc_wavelet_lngram_and_retrieval":extended,"in_block_reads":cfg.sbd.as_ref().unwrap().in_block_reads,"joint_execution":cfg.sbd.as_ref().unwrap().joint_execution,"cpu_vs_requested_backend":backend,"output_paths":paths,"observation_and_target_locality":locality,"teacher_student_gradients":gradients,"contextual_summary_and_root_head":heads,"training_paths":training,"inactive_rows":inactive,"inference":inference,"fixed_uniform_scheduler_rollout":rollout}));
    }
    let historical = historical_checkpoint_compatibility(&device)?;
    let mut report = json!({"passed":true,"command":"codpula-model-reference","device":runtime::identity(&device),"precision":"float32","seed":72,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,"clipping":clipping,"selector_and_initialization":identity,"cases":cases,"checkpoint_continuation":continuation.context("missing legacy checkpoint continuation case")?,"contextual_checkpoint_continuation":contextual_continuation.context("missing contextual checkpoint continuation case")?,"historical_checkpoint_compatibility":historical,"old_sbd_leakage_checks":"unchanged; this module adds legacy and contextual Codpula checks"});
    let (mut absolute,mut ratio) = (0.0,0.0);
    maximum_errors(&report,&mut absolute,&mut ratio);
    report["max_abs"] = json!(absolute);
    report["maximum_tolerance_ratio"] = json!(ratio);
    Ok(report)
}

/// Strict legacy and contextual continuation regressions, without the model matrix.
pub fn continuation_reference(c: &Config, device: &Device) -> Result<Value> {
    let clipping = crate::update::clipping_checks(device)?;
    let cpu = Device::ndarray().autodiff();
    let target = device.clone().autodiff();
    let mut reports = Vec::new();
    for (joint_output,extended) in [(SbdJointOutput::Codpula,false),(SbdJointOutput::CodpulaContextual,true)] {
        let (mut cfg,arm) = fixture_config(c,"kronlstm",extended);
        cfg.sbd.as_mut().unwrap().joint_output = joint_output;
        cpu.seed(cfg.seed);
        let mut reference = LanguageModel::new(&cfg,&arm,&cpu)?;
        probe_gates(&mut reference,&cpu);
        let mut model = reference.fork(&target);
        requested_backends(&mut model,&mut cfg,c);
        reports.push(checkpoint_continuation(&model,&cfg,&target)?);
    }
    Ok(json!({"passed":true,"clipping":clipping,"continuation":reports.remove(0),"contextual_continuation":reports.remove(0)}))
}
