//! Paired execution-path benchmark. No training run, suite budget or old evidence is modified.
use crate::{
    checkpoint,
    config::{Arm, Config},
    data::{self, Batch, Example},
    model::{self, LanguageModel},
    runtime, train,
};
use anyhow::{Result, ensure};
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::{GradientsParams, ModuleOptimizer},
    prelude::*,
};
use delta_mixers::{
    HolaBackend, MixerKind, ScanBackend,
    chunk::{self, ScanInput},
};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const ATOL: f64 = 2e-5;
const RTOL: f64 = 3e-4;
pub(super) fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().expect("f32 benchmark")
}
pub(super) fn equivalent(name: &str, a: &[f32], b: &[f32]) -> Result<f64> {
    ensure!(a.len() == b.len(), "{name}: length mismatch");
    let mut maximum = 0.0f64;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let error = (f64::from(a) - f64::from(b)).abs();
        ensure!(
            a.is_finite() && b.is_finite() && error <= ATOL + RTOL * f64::from(b).abs(),
            "{name}[{i}]: {a} != {b}, error {error}"
        );
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.
    } else {
        sorted[n / 2]
    }
}
fn deadline(start: Instant, c: &Config) -> Result<()> {
    ensure!(
        start.elapsed().as_secs_f64() < c.max_seconds,
        "bounded Kron benchmark reached max_seconds={}; partial report retained",
        c.max_seconds
    );
    Ok(())
}
pub(super) struct Raw {
    kind: MixerKind,
    x: Vec<Tensor<4>>,
    valid: Tensor<2>,
    reset: Tensor<2>,
    initial: Tensor<4>,
    upstream: Tensor<4>,
}
impl Raw {
    pub(super) fn new(device: &Device, batch: usize, time: usize, seed: u64) -> Self {
        Self::new_kind(device, batch, time, seed, MixerKind::KronLstm)
    }
    fn new_kind(device: &Device, batch: usize, time: usize, seed: u64, kind: MixerKind) -> Self {
        let heads = 4;
        let widths = if kind == MixerKind::Gdn2 {
            [32, 32, 64, 32, 32, 64]
        } else {
            [32, 32, 64, 32, 1, 1]
        };
        let mut rng = data::Rng(seed);
        let x = widths
            .into_iter()
            .enumerate()
            .map(|(field, width)| {
                let mut data = (0..batch * time * heads * width)
                    .map(|_| {
                        let x = (rng.next() >> 40) as f32 / (1u32 << 24) as f32;
                        if field == 3 {
                            0.90 + 0.09 * x
                        } else if field >= 4 {
                            0.1 + 0.8 * x
                        } else {
                            x - 0.5
                        }
                    })
                    .collect::<Vec<_>>();
                if field <= 2 {
                    for row in data.chunks_exact_mut(width) {
                        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                        for x in row {
                            *x /= norm;
                        }
                    }
                }
                Tensor::from_data(TensorData::new(data, [batch, time, heads, width]), device)
            })
            .collect();
        Self {
            kind,
            x,
            valid: Tensor::ones([batch, time], device),
            reset: Tensor::zeros([batch, time], device),
            initial: Tensor::zeros([batch, heads, 32, 64], device),
            upstream: Tensor::ones([batch, time, heads, 64], device),
        }
    }
    fn run(&self, backend: ScanBackend, backward: bool) -> (chunk::ScanOutput, Vec<Tensor<4>>) {
        let x = self
            .x
            .iter()
            .map(|x| {
                if backward {
                    x.clone().detach().require_grad()
                } else {
                    x.clone().detach()
                }
            })
            .collect::<Vec<_>>();
        let initial = if backward {
            self.initial.clone().detach().require_grad()
        } else {
            self.initial.clone().detach()
        };
        let output = chunk::scan_selected(
            backend,
            self.kind,
            ScanInput {
                q: x[0].clone(),
                k: x[1].clone(),
                v: x[2].clone(),
                alpha: x[3].clone(),
                erase: x[4].clone(),
                write: x[5].clone(),
            },
            self.valid.clone(),
            self.reset.clone(),
            16,
            Some(initial.clone()),
        );
        let grads = if backward {
            let gradients = (output.output.clone() * self.upstream.clone())
                .sum()
                .backward();
            x.iter()
                .chain(std::iter::once(&initial))
                .map(|x| x.grad(&gradients).expect("raw recurrence gradient"))
                .collect()
        } else {
            Vec::new()
        };
        (output, grads)
    }
}
pub(super) fn raw_timed(
    raw: &Raw,
    backend: ScanBackend,
    backward: bool,
    device: &Device,
) -> Result<f64> {
    runtime::sync(device)?;
    let clock = Instant::now();
    let result = raw.run(backend, backward);
    runtime::sync(device)?;
    let ms = clock.elapsed().as_secs_f64() * 1000.;
    std::hint::black_box(result);
    Ok(ms)
}
struct Parameters<'a> {
    grads: Option<&'a GradientsParams>,
    tensors: Vec<(Vec<usize>, Vec<f32>)>,
    missing: usize,
}
impl ModuleVisitor for Parameters<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let tensor = if let Some(grads) = self.grads {
            match grads.get::<D>(param.id) {
                Some(g) => g,
                None => {
                    self.missing += 1;
                    return;
                }
            }
        } else {
            param.val()
        };
        self.tensors.push((tensor.dims().to_vec(), values(tensor)));
    }
}
pub(super) fn parameters(
    model: &LanguageModel,
    grads: Option<&GradientsParams>,
) -> Result<Vec<(Vec<usize>, Vec<f32>)>> {
    let mut visitor = Parameters {
        grads,
        tensors: Vec::new(),
        missing: 0,
    };
    model.visit(&mut visitor);
    ensure!(
        visitor.missing == 0,
        "{} parameters have no gradient",
        visitor.missing
    );
    Ok(visitor.tensors)
}
pub(super) fn compare_parameters(
    name: &str,
    a: &[(Vec<usize>, Vec<f32>)],
    b: &[(Vec<usize>, Vec<f32>)],
) -> Result<f64> {
    ensure!(a.len() == b.len(), "{name}: parameter count differs");
    let mut error = 0.0f64;
    for (i, ((sa, a), (sb, b))) in a.iter().zip(b).enumerate() {
        ensure!(sa == sb, "{name}: parameter {i} shape differs");
        error = error.max(equivalent(&format!("{name} parameter {i}"), a, b)?);
    }
    Ok(error)
}
type ParameterSnapshot = Vec<(Vec<usize>, Vec<f32>)>;
fn parameter_direction(parameter: usize, element: usize) -> f32 {
    let bits = (parameter as u64 + 1).wrapping_mul(2_246_822_519)
        ^ (element as u64 + 1).wrapping_mul(2_654_435_761);
    if ((bits >> 9) ^ bits) & 1 == 0 {
        0.01
    } else {
        -0.01
    }
}
struct PerturbParameters {
    scale: f64,
    index: usize,
}
impl ModuleMapper for PerturbParameters {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let current = param.val();
        let shape = current.dims();
        let direction = (0..shape.iter().product())
            .map(|i| (f64::from(parameter_direction(self.index, i)) * self.scale) as f32)
            .collect::<Vec<_>>();
        self.index += 1;
        let delta = Tensor::from_data(TensorData::new(direction, shape), &current.device());
        Param::from_tensor(current + delta)
    }
}
/// Hard HOLA top-k selection is detached and piecewise constant, so the analytic gradient
/// is the derivative with cache membership held fixed. Differencing must hold it fixed too:
/// the unperturbed forward records every block's kept slots and both perturbed forwards
/// reuse them. Models without a cache are unaffected.
#[derive(Default)]
struct FixedCacheSelection {
    layer: Option<usize>,
    kept: std::collections::BTreeMap<usize, Vec<Tensor<3, Int>>>,
    replay: bool,
}
impl delta_mixers::profile::Sections for FixedCacheSelection {
    const CAPTURE_CACHE_MEMBERSHIP: bool = true;
    fn block(&mut self, index: Option<usize>) {
        self.layer = index;
    }
    fn section<T>(&mut self, _name: &'static str, run: impl FnOnce() -> T) -> T {
        run()
    }
    fn cache_membership(
        &mut self,
        _chunk_start: usize,
        _used: bool,
        kept_slots: &Tensor<3, Int>,
        _positions: &Tensor<3>,
    ) {
        if !self.replay {
            let layer = self.layer.expect("cache selection outside a block");
            self.kept.entry(layer).or_default().push(kept_slots.clone());
        }
    }
    fn replayed_cache_selection(&self) -> Option<Vec<Tensor<3, Int>>> {
        if self.replay {
            self.layer.and_then(|layer| self.kept.get(&layer).cloned())
        } else {
            None
        }
    }
}
impl FixedCacheSelection {
    fn record(model: &LanguageModel, batch: &Batch) -> Self {
        let mut selection = Self::default();
        let _ = model.valid().forward_profiled(batch, &mut selection);
        selection.replay = true;
        selection
    }
    fn loss(&mut self, model: &LanguageModel, batch: &Batch) -> Result<f64> {
        model::scalar(model.loss(model.forward_profiled(batch, self), batch))
    }
}
fn directional_analytic(gradients: &ParameterSnapshot) -> f64 {
    gradients
        .iter()
        .enumerate()
        .flat_map(|(p, (_, values))| {
            values
                .iter()
                .enumerate()
                .map(move |(i, &g)| f64::from(g) * f64::from(parameter_direction(p, i)))
        })
        .sum::<f64>()
}
fn fixed_selection_difference(
    base: &LanguageModel,
    batch: &Batch,
    selection: &mut FixedCacheSelection,
    epsilon: f64,
) -> Result<(f64, f64)> {
    let plus = base.valid().map(&mut PerturbParameters {
        scale: epsilon,
        index: 0,
    });
    let plus_loss = selection.loss(&plus, batch)?;
    drop(plus);
    let minus = base.valid().map(&mut PerturbParameters {
        scale: -epsilon,
        index: 0,
    });
    let minus_loss = selection.loss(&minus, batch)?;
    Ok((plus_loss, minus_loss))
}
fn cpu_directional_check(
    base: &LanguageModel,
    batch: &Batch,
    gradients: &ParameterSnapshot,
) -> Result<Value> {
    let epsilon = 2e-3;
    let analytic = directional_analytic(gradients);
    let mut selection = FixedCacheSelection::record(base, batch);
    let (plus_loss, minus_loss) = fixed_selection_difference(base, batch, &mut selection, epsilon)?;
    let numerical = (plus_loss - minus_loss) / (2. * epsilon);
    let error = (analytic - numerical).abs();
    let tolerance = 3e-3 + 0.03 * analytic.abs();
    ensure!(
        analytic.is_finite() && numerical.is_finite() && error <= tolerance,
        "CPU full-parameter directional finite difference: analytic={analytic}, numerical={numerical}, absolute_error={error}, tolerance={tolerance}"
    );
    Ok(
        json!({"passed":true,"all_parameter_tensors_perturbed":gradients.len(),"epsilon":epsilon,
        "analytic":analytic,"numerical":numerical,"absolute_error":error,
        "absolute_tolerance":3e-3,"relative_tolerance":0.03,
        "cache_selection":if selection.kept.is_empty() { "no cache" } else { "kept slots replayed from the unperturbed forward" }}),
    )
}
struct CpuReference {
    base: LanguageModel,
    initial: ParameterSnapshot,
    logits: Vec<f32>,
    gradients: ParameterSnapshot,
    updated: ParameterSnapshot,
    directional_check: Value,
    loss: f64,
}
fn cpu_reference(c: &Config, arm: &Arm, rows: &[Example]) -> Result<CpuReference> {
    #[cfg(feature = "cpu")]
    {
        let device = Device::ndarray().autodiff();
        device.seed(c.seed);
        let base = LanguageModel::new(c, arm, &device)?;
        let initial = parameters(&base, None)?;
        let batch = Batch::new(rows, &device);
        let logits = base.forward(&batch);
        let objective = base.loss(logits.clone(), &batch);
        let loss = model::scalar(objective.clone())?;
        let gradients = model::checked_gradients(&base, objective)?;
        let gradient_values = parameters(&base, Some(&gradients))?;
        let directional_check = cpu_directional_check(&base, &batch, &gradient_values)?;
        let mut optimizer = train::optimizer(c);
        let updated = optimizer.step(c.learning_rate, base.clone(), gradients);
        let updated = parameters(&updated, None)?;
        Ok(CpuReference {
            base,
            initial,
            logits: values(logits),
            gradients: gradient_values,
            updated,
            directional_check,
            loss,
        })
    }
    #[cfg(not(feature = "cpu"))]
    {
        let _ = (c, arm, rows);
        anyhow::bail!("model numerical preflight requires the cpu feature")
    }
}
struct PreparedModelCase {
    config: Config,
    batch: Batch,
    runners: Vec<Runner>,
    report: Value,
}
fn prepare_model_case(
    c: &Config,
    device: &Device,
    context: usize,
    hola: bool,
    selected: Option<(ScanBackend, HolaBackend)>,
) -> Result<PreparedModelCase> {
    prepare_model_case_kind(c, device, context, hola, selected, MixerKind::KronLstm)
}
fn prepare_model_case_kind(
    c: &Config,
    device: &Device,
    context: usize,
    hola: bool,
    selected: Option<(ScanBackend, HolaBackend)>,
    kind: MixerKind,
) -> Result<PreparedModelCase> {
    let mut config = c.clone();
    config.context = context;
    config.scan_backend = ScanBackend::TensorSolver;
    config.hola_backend = HolaBackend::Tensor;
    config.projection_backend = delta_mixers::ProjectionBackend::Separate;
    config.normalization_backend = delta_mixers::NormalizationBackend::Tensor;
    config.convolution_backend = delta_mixers::ConvolutionBackend::Tensor;
    let arm = Arm {
        name: if kind == MixerKind::Gdn2 {
            if hola { "gdn-hola" } else { "gdn" }
        } else if hola {
            "kron-hola"
        } else {
            "kron"
        }
        .into(),
        mixer: if kind == MixerKind::Gdn2 {
            "gdn2"
        } else {
            "kronlstm"
        }
        .into(),
        cache: if hola { "surprise" } else { "none" }.into(),
    };
    let rows = (0..c.batch_size)
        .map(|row| Example {
            input: (0..context)
                .map(|t| ((t * 17 + row * 11 + c.seed as usize) % config.vocab()) as i32)
                .collect(),
            target: (0..context)
                .map(|t| (((t + 1) * 17 + row * 11 + c.seed as usize) % config.vocab()) as i32)
                .collect(),
            weight: vec![1.; context],
            groups: vec![None; context],
        })
        .collect::<Vec<_>>();
    let mut cpu_config = config.clone();
    cpu_config.head_backend = crate::config::HeadBackend::Native;
    cpu_config.loss_backend = crate::config::LossBackend::Tensor;
    cpu_config.update_backend = crate::config::UpdateBackend::Tensor;
    cpu_config.epilogue_backend = crate::config::EpilogueBackend::Native;
    cpu_config.ohc_backend = delta_mixers::ohc::OhcBackend::Tensor;
    cpu_config.wavelet_backend = delta_mixers::WaveletBackend::Tensor;
    cpu_config.lngram_backend = delta_mixers::lngram::LngramBackend::Tensor;
    let reference = cpu_reference(&cpu_config, &arm, &rows)?;
    let train_device = device.clone().autodiff();
    // Fork detaches the CPU graph: no model/input transfers occur in the GPU step.
    let base = reference.base.clone().fork(&train_device);
    runtime::sync(device)?;
    let batch = Batch::new(&rows, &train_device);
    let fused = if kind == MixerKind::Gdn2 {
        ScanBackend::FusedGdn2
    } else {
        ScanBackend::FusedGpu
    };
    let paths = if let Some(path) = selected {
        vec![path]
    } else if c.backend == "vulkan" {
        if hola {
            vec![
                (ScanBackend::TensorSolver, HolaBackend::Tensor),
                (fused, HolaBackend::Tensor),
                (ScanBackend::TensorSolver, HolaBackend::FusedGpu),
                (fused, HolaBackend::FusedGpu),
            ]
        } else {
            vec![
                (ScanBackend::TensorSolver, HolaBackend::Tensor),
                (fused, HolaBackend::Tensor),
            ]
        }
    } else {
        vec![(ScanBackend::TensorSolver, HolaBackend::Tensor)]
    };
    let mut runners = Vec::with_capacity(paths.len());
    let mut checks = Vec::new();
    let mut first_model = Vec::new();
    for (backend, hola_backend) in paths {
        let mut model = base.clone();
        model.set_scan_backend(backend);
        model.set_hola_backend(hola_backend);
        model.set_projection_backend(c.projection_backend);
        model.set_normalization_backend(c.normalization_backend);
        model.set_convolution_backend(c.convolution_backend);
        model.set_head_backend(c.head_backend);
        model.set_loss_backend(c.loss_backend);
        model.set_update_backend(c.update_backend);
        model.set_epilogue_backend(c.epilogue_backend);
        model.set_ohc_backend(c.ohc_backend);
        model.set_wavelet_backend(c.wavelet_backend);
        model.set_lngram_backend(c.lngram_backend);
        let initial_error = compare_parameters(
            "identical CPU initialization",
            &parameters(&model, None)?,
            &reference.initial,
        )?;
        ensure!(
            initial_error == 0.,
            "model preflight initialization must be bitwise identical"
        );
        runtime::sync(device)?;
        let clock = Instant::now();
        let logits = model.forward(&batch);
        let objective = model.loss(logits.clone(), &batch);
        let loss = model::scalar(objective.clone())?;
        let grads = model::checked_gradients(&model, objective)?;
        runtime::sync(device)?;
        first_model.push(json!({"scan_backend":backend,"hola_backend":hola_backend,
            "first_model_forward_backward_ms":clock.elapsed().as_secs_f64()*1000.}));
        let label =
            format!("actual full model T{context}/HOLA{hola}/{backend:?}/{hola_backend:?} vs CPU");
        let output_error = equivalent(
            &format!("{label} logits"),
            &values(logits),
            &reference.logits,
        )?;
        let gradient_error = compare_parameters(
            &format!("{label} all gradients"),
            &parameters(&model, Some(&grads))?,
            &reference.gradients,
        )?;
        let mut optim = train::optimizer(&config);
        let updated = optim.step(c.learning_rate, model, grads);
        runtime::sync(device)?;
        let update_error = compare_parameters(
            &format!("{label} AdamW updated parameters"),
            &parameters(&updated, None)?,
            &reference.updated,
        )?;
        checks.push(json!({"scan_backend":backend,"hola_backend":hola_backend,
            "projection_backend":c.projection_backend,"normalization_backend":c.normalization_backend,
            "convolution_backend":c.convolution_backend,
            "head_backend":c.head_backend,
            "update_backend":c.update_backend,"loss_backend":c.loss_backend,"epilogue_backend":c.epilogue_backend,
            "ohc_streams":c.ohc_streams,"ohc_backend":c.ohc_backend,
            "wavelet_layers":c.wavelet_layers,"wavelet_backend":c.wavelet_backend,
            "lngram_layers":c.lngram_layers,"lngram_backend":c.lngram_backend,
            "reference":"CPU TensorSolver/Tensor/Separate/Tensor normalization",
            "initial_parameters_max_abs":initial_error,"logits_max_abs":output_error,
            "all_parameter_gradients_max_abs":gradient_error,"AdamW_parameters_max_abs":update_error,
            "all_model_parameters_have_finite_gradients":true,"loss":loss,"cpu_loss":reference.loss}));
        runners.push(Runner {
            backend,
            hola_backend,
            model: Some(updated),
            optim,
        });
    }
    let report = json!({"context":context,"hola":hola,"batch":c.batch_size,"d_model":c.d_model,"layers":c.layers,
        "parameters":base.num_params(),"initialization":"canonical CPU parameters forked to independent runtime graph before steps",
        "initial_parameters_sha256":data::hash(&serde_json::to_vec(&reference.initial)?),
        "fixture_sha256":data::fixture_hash(&rows),"first_use":first_model,"equivalence_before_throughput":checks,
        "cpu_full_parameter_directional_check":reference.directional_check,
        "optimizer":"AdamW","learning_rate":c.learning_rate,"weight_decay":c.weight_decay,"gradient_clip":c.gradient_clip});
    Ok(PreparedModelCase {
        config,
        batch,
        runners,
        report,
    })
}
/// Actual normalized model scope only; the unresolved legacy stress diagnostic is not accepted.
pub fn model_check(c: &Config, device: &Device, configured_only: bool) -> Result<Value> {
    ensure!(
        c.backend == "cpu" || c.backend == "vulkan",
        "model preflight is local CPU/Vulkan only"
    );
    let started = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c
        .artifact_dir
        .join(format!("kron-model-check-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let report_path = directory.join("report.json");
    let mut result = json!({"schema":1,"command":"kron-model-check","scope":"actual_normalized_full_model_only",
        "completed":false,"passed":false,"global_tensor_solver_stress_accepted":false,
        "known_legacy_diagnostic":"Central Vulkan TensorSolver T7 nonzero-initial-state/zero-decay stress VJP failed separately; this command neither repairs nor accepts that path on all primitive inputs",
        "config":c,"device":runtime::identity(device),"directory":directory,
        "absolute_tolerance":ATOL,"relative_tolerance":RTOL,"cases":[]});
    result["execution_scope"] = json!(if configured_only {
        "configured"
    } else {
        "all_combinations"
    });
    result["acceptance_scope"] = json!(if configured_only {
        "Only the explicitly configured execution pair, both contexts, with and without active HOLA; legacy combinations are not certified"
    } else {
        "Every scan/HOLA execution combination; failures are retained separately, never accepted"
    });
    runtime::write_json(&report_path, &result)?;
    for context in [128, 512] {
        for hola in [false, true] {
            let paths = if configured_only {
                vec![(
                    c.scan_backend,
                    if hola {
                        c.hola_backend
                    } else {
                        HolaBackend::Tensor
                    },
                )]
            } else if c.backend == "vulkan" && hola {
                vec![
                    (ScanBackend::TensorSolver, HolaBackend::Tensor),
                    (ScanBackend::FusedGpu, HolaBackend::Tensor),
                    (ScanBackend::TensorSolver, HolaBackend::FusedGpu),
                    (ScanBackend::FusedGpu, HolaBackend::FusedGpu),
                ]
            } else if c.backend == "vulkan" {
                vec![
                    (ScanBackend::TensorSolver, HolaBackend::Tensor),
                    (ScanBackend::FusedGpu, HolaBackend::Tensor),
                ]
            } else {
                vec![(ScanBackend::TensorSolver, HolaBackend::Tensor)]
            };
            for path in paths {
                deadline(started, c)?;
                result["active_case"] = json!({"context":context,"hola":hola,"scan_backend":path.0,"hola_backend":path.1});
                runtime::write_json(&report_path, &result)?;
                let case = match prepare_model_case(c, device, context, hola, Some(path)) {
                    Ok(prepared) => {
                        let mut report = prepared.report;
                        report["passed"] = json!(true);
                        report["scan_backend"] = json!(path.0);
                        report["hola_backend"] = json!(path.1);
                        report
                    }
                    Err(error) => json!({"context":context,"hola":hola,"scan_backend":path.0,
                        "hola_backend":path.1,"passed":false,"error":format!("{error:#}")}),
                };
                result["cases"].as_array_mut().unwrap().push(case);
                runtime::write_json(&report_path, &result)?;
            }
        }
    }
    result["completed"] = json!(true);
    result["passed"] = json!(
        result["cases"]
            .as_array()
            .unwrap()
            .iter()
            .all(|case| case["passed"] == true)
    );
    result["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    result["report_path"] = json!(report_path);
    runtime::write_json(&report_path, &result)?;
    Ok(result)
}
struct Runner {
    backend: ScanBackend,
    hola_backend: HolaBackend,
    model: Option<LanguageModel>,
    optim: ModuleOptimizer,
}
impl Runner {
    fn step(&mut self, c: &Config, batch: &Batch, device: &Device) -> Result<f64> {
        runtime::sync(device)?;
        let clock = Instant::now();
        let model = self.model.take().expect("benchmark model");
        let loss = model.loss(model.forward(batch), batch);
        let _ = model::scalar(loss.clone())?;
        let grads = model::checked_gradients(&model, loss)?;
        self.model = Some(self.optim.step(c.learning_rate, model, grads));
        runtime::sync(device)?;
        Ok(clock.elapsed().as_secs_f64() * 1000.)
    }
}
fn checkpoint_cost(c: &Config, runner: &Runner, device: &Device, dir: &Path) -> Result<Value> {
    std::fs::create_dir_all(dir)?;
    runtime::sync(device)?;
    let clock = Instant::now();
    runner
        .model
        .as_ref()
        .unwrap()
        .clone()
        .save_file(dir.join("model.burnpack"))?;
    runner.optim.save(dir.join("optimizer.burnpack"))?;
    runtime::sync(device)?;
    let save = clock.elapsed().as_secs_f64() * 1000.;
    // Reuse the exact model structure/IDs, retaining the source across guarded record upload.
    let target = runner.model.as_ref().unwrap().clone();
    let clock = Instant::now();
    let restored = checkpoint::load_model(target, dir.join("model.burnpack"), device)?;
    let optimizer = checkpoint::load_optimizer(c, dir.join("optimizer.burnpack"))?;
    runtime::sync(device)?;
    let load = clock.elapsed().as_secs_f64() * 1000.;
    std::hint::black_box((restored, optimizer));
    Ok(
        json!({"save_model_and_optimizer_ms":save,"guarded_load_and_logical_validation_ms":load,"model_bytes":std::fs::metadata(dir.join("model.burnpack"))?.len(),"optimizer_bytes":std::fs::metadata(dir.join("optimizer.burnpack"))?.len(),"excluded_from_step_timing":true}),
    )
}

fn append_hola_benchmarks(
    c: &Config,
    device: &Device,
    trials: usize,
    started: Instant,
    report_path: &Path,
    result: &mut Value,
) -> Result<()> {
    result["hola_raw"] = json!([]);
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    if c.backend == "vulkan" {
        for time in [128, 512] {
            for capacity in [16, 64] {
                deadline(started, c)?;
                let report =
                    delta_mixers::checks::hola_paired_benchmark(device, 1, time, capacity, trials)
                        .map_err(anyhow::Error::msg)?;
                let summary=report.timing.iter().map(|timing| json!({
                    "hola_backend":timing.hola_backend,
                    "forward_median_ms":median(&timing.forward_ms),
                    "forward_backward_median_ms":median(&timing.forward_backward_ms),
                    "forward_backward_tokens_per_second":time as f64/(median(&timing.forward_backward_ms)/1000.)
                })).collect::<Vec<_>>();
                let mut value = serde_json::to_value(report)?;
                value["summary"] = json!(summary);
                result["hola_raw"].as_array_mut().unwrap().push(value);
                runtime::write_json(report_path, result)?;
            }
        }
    }
    Ok(())
}

/// Independent HOLA-only scope; no Kron recurrence or full-model speed claim is made.
pub fn hola_run(c: &Config, device: &Device, trials: usize) -> Result<Value> {
    ensure!(
        c.backend == "vulkan",
        "isolated paired HOLA benchmark requires local Vulkan; no CPU/fused fallback"
    );
    ensure!((3..=31).contains(&trials), "trials must be 3..=31");
    let started = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c
        .artifact_dir
        .join(format!("hola-benchmark-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let report_path = directory.join("report.json");
    let mut result = json!({
        "schema":1,"command":"hola-benchmark","scope":"isolated_hola_only","completed":false,
        "config":c,"device":runtime::identity(device),"directory":directory,"trials":trials,
        "warmups_per_mode":2,"absolute_tolerance":ATOL,"relative_tolerance":RTOL,
        "numerical_gate":"existing per-fixture f64 and same-input tensor/fused membership/output/Q/K/V/sink-gradient checks before warm throughput samples",
        "timing_method":"same resident deterministic hola_checks fixtures; synchronized paired AB/BA trials; warmups excluded; raw samples and medians",
        "first_use_scope":"first shape-specific calls before per-fixture correctness preflight; cache may already be warm; not isolated cold JIT; warm samples follow checked preflight and two warmups",
        "interpretation":"HOLA operator throughput only; neither Kron equivalence nor full-model training throughput is established",
        "hola_raw":[]
    });
    runtime::write_json(&report_path, &result)?;
    append_hola_benchmarks(
        c,
        &device.clone().autodiff(),
        trials,
        started,
        &report_path,
        &mut result,
    )?;
    result["completed"] = json!(true);
    result["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    result["report_path"] = json!(report_path);
    runtime::write_json(&report_path, &result)?;
    Ok(result)
}

fn append_model_benchmarks(
    c: &Config,
    device: &Device,
    warmups: usize,
    trials: usize,
    started: Instant,
    directory: &Path,
    report_path: &Path,
    result: &mut Value,
) -> Result<()> {
    for context in [128, 512] {
        for hola in [false, true] {
            deadline(started, c)?;
            result["active_case"] = json!({"context":context,"hola":hola});
            runtime::write_json(report_path, result)?;
            let PreparedModelCase {
                config,
                batch,
                mut runners,
                mut report,
            } = prepare_model_case(c, device, context, hola, None)?;
            report["preflight_passed"] = json!(true);
            report["measurements"] = json!([]);
            let index = result["full_model"].as_array().unwrap().len();
            result["full_model"].as_array_mut().unwrap().push(report);
            runtime::write_json(report_path, result)?;
            let mut times = vec![Vec::new(); runners.len()];
            for iteration in 0..warmups + trials {
                for offset in 0..runners.len() {
                    deadline(started, c)?;
                    let index = (iteration + offset) % runners.len();
                    let ms = runners[index].step(&config, &batch, device)?;
                    if iteration >= warmups {
                        times[index].push(ms);
                    }
                }
            }
            let mut measurements = Vec::new();
            for (i, runner) in runners.iter().enumerate() {
                deadline(started, c)?;
                let checkpoint = checkpoint_cost(
                    &config,
                    runner,
                    device,
                    &directory.join(format!(
                        "checkpoint-t{context}-hola{hola}-{:?}-{:?}",
                        runner.backend, runner.hola_backend
                    )),
                )?;
                measurements.push(json!({"scan_backend":runner.backend,"hola_backend":runner.hola_backend,
                    "step_ms":times[i],"median_step_ms":median(&times[i]),
                    "tokens_per_second":config.tokens_per_step()as f64/(median(&times[i])/1000.),"checkpoint":checkpoint}));
            }
            result["full_model"][index]["measurements"] = json!(measurements);
            runtime::write_json(report_path, result)?;
        }
    }
    Ok(())
}

/// CPU-certified actual-model scope, explicitly distinct from all-input solver certification.
pub fn model_run(
    c: &Config,
    device: &Device,
    warmups: usize,
    trials: usize,
    legacy_report: Option<&Path>,
) -> Result<Value> {
    ensure!(
        (1..=20).contains(&warmups) && (3..=31).contains(&trials),
        "warmups 1..20 and trials 3..31 required"
    );
    ensure!(
        c.backend == "cpu" || c.backend == "vulkan",
        "model benchmark is local CPU/Vulkan only"
    );
    ensure!(
        c.backend != "vulkan" || legacy_report.is_some(),
        "Vulkan model benchmark requires --legacy-stress-report with the retained TensorSolver failure evidence"
    );
    let started = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    std::fs::create_dir_all(&c.artifact_dir)?;
    let directory = c.artifact_dir.join(format!(
        "kron-model-benchmark-{}-{}-{nonce}",
        c.backend,
        std::process::id()
    ));
    std::fs::create_dir(&directory)?;
    let report_path = directory.join("report.json");
    let legacy_evidence = if let Some(path) = legacy_report {
        let bytes = std::fs::read(path)?;
        let value: Value = serde_json::from_slice(&bytes)?;
        ensure!(
            value["command"] == "kron-reference"
                && value["scan_backend"] == "tensor_solver"
                && value["config"]["backend"] == "vulkan"
                && value["passed"] == false,
            "legacy stress evidence must be an actual failed Vulkan TensorSolver kron-reference report"
        );
        let copy = directory.join("legacy-tensor-stress.json");
        std::fs::write(&copy, &bytes)?;
        json!({"certified":false,"source":path,"retained_copy":copy,"sha256":data::hash(&bytes),
            "known_issue":value["error"],"acceptance_policy":"retained failure; not repaired, suppressed or accepted by this scoped benchmark"})
    } else {
        json!({"certified":false,"scope":"CPU model only; no Vulkan legacy stress certification claimed"})
    };
    let mut result = json!({"schema":1,"command":"kron-model-benchmark","scope":"actual_normalized_full_model",
        "completed":false,"global_tensor_solver_stress_accepted":false,"legacy_gpu_stress":legacy_evidence,
        "config":c,"device":runtime::identity(device),"directory":directory,
        "warmups":warmups,"trials":trials,"absolute_tolerance":ATOL,"relative_tolerance":RTOL,
        "acceptance_contract":"independent selected-kernel f64 stress checks plus CPU full-parameter directional, logits, every parameter gradient and AdamW-update equivalence for each exact measured normalized-model case",
        "timing_method":"resident identical fixtures; synchronized paired rotating order; warmups excluded; every sample and median retained; complete AdamW steps",
        "first_use_scope":"first full-model forward/backward after independent kernel gates; persistent compilation may already be warm",
        "checkpoint_scope":"model/optimizer save and guarded logical-value-validated reload are measured separately, outside step timing",
        "interpretation":"actual CPU-certified workload comparison only; does not certify the legacy GPU TensorSolver on arbitrary recurrence inputs or claim new quality training",
        "kernel_acceptance":{},"full_model":[]});
    runtime::write_json(&report_path, &result)?;
    let scan = if c.backend == "vulkan" {
        ScanBackend::FusedGpu
    } else {
        ScanBackend::TensorSolver
    };
    let hola = if c.backend == "vulkan" {
        HolaBackend::FusedGpu
    } else {
        HolaBackend::Tensor
    };
    deadline(started, c)?;
    result["kernel_acceptance"]["kron"] = json!({
        "scan_backend":scan,
        "numerical_report":delta_mixers::checks::kron_numerical_checks_selected(device,scan,None,None).map_err(anyhow::Error::msg)?
    });
    runtime::write_json(&report_path, &result)?;
    deadline(started, c)?;
    result["kernel_acceptance"]["hola"] = json!({
        "hola_backend":hola,
        "numerical_report":delta_mixers::checks::hola_numerical_checks_selected(device,hola).map_err(anyhow::Error::msg)?
    });
    runtime::write_json(&report_path, &result)?;
    append_model_benchmarks(
        c,
        device,
        warmups,
        trials,
        started,
        &directory,
        &report_path,
        &mut result,
    )?;
    result["completed"] = json!(true);
    result["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    result["report_path"] = json!(report_path);
    runtime::write_json(&report_path, &result)?;
    Ok(result)
}

pub fn run(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(
        (1..=20).contains(&warmups) && (3..=31).contains(&trials),
        "warmups 1..20 and trials 3..31 required"
    );
    ensure!(
        c.backend == "cpu" || c.backend == "vulkan",
        "benchmark is local CPU/Vulkan only"
    );
    let started = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c.artifact_dir.join(format!(
        "kron-benchmark-{}-{}-{nonce}",
        c.backend,
        std::process::id()
    ));
    std::fs::create_dir_all(&directory)?;
    let report_path = directory.join("report.json");
    let train_device = device.clone().autodiff();
    let backends = if c.backend == "vulkan" {
        vec![ScanBackend::TensorSolver, ScanBackend::FusedGpu]
    } else {
        vec![ScanBackend::TensorSolver]
    };
    let mut result = json!({"schema":1,"command":"kron-benchmark","completed":false,"config":c,"device":runtime::identity(device),"directory":directory,"seed":c.seed,"precision":"float32","warmups":warmups,"trials":trials,"absolute_tolerance":ATOL,"relative_tolerance":RTOL,"timing_method":"device synchronization before/after each call; paired trial order alternates; medians of complete samples; resident identical fixture; no data generation, log IO or checkpoint IO in timed steps","cpu_policy":"CPU executes TensorSolver only; requesting fused GPU on CPU is an error, not a fallback","raw":[],"full_model":[]});
    runtime::write_json(&report_path, &result)?;
    // Capture the process's first call before numerical gates compile the fused specialization.
    let first_fixture = Raw::new(&train_device, 1, 128, c.seed);
    runtime::sync(device)?;
    let mut first_use = Vec::new();
    for &backend in &backends {
        deadline(started, c)?;
        first_use.push(json!({"scan_backend":backend,"forward_backward_ms":raw_timed(&first_fixture,backend,true,device)?}));
    }
    result["first_use"] = json!({"measurements":first_use,"scope":"first recurrence call in this process at B1/T128/H4/K32/V64; includes any persistent-cache miss JIT and dispatch setup; cache may already contain compiled kernels; not claimed cold compiler time"});
    runtime::write_json(&report_path, &result)?;
    let gate = Instant::now();
    result["numerical_gate"] = serde_json::to_value(
        delta_mixers::checks::kron_numerical_checks(&train_device).map_err(anyhow::Error::msg)?,
    )?;
    result["numerical_gate_seconds"] = json!(gate.elapsed().as_secs_f64());
    runtime::write_json(&report_path, &result)?;
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    if c.backend == "vulkan" {
        result["hola_numerical_gate"] = serde_json::to_value(
            delta_mixers::checks::hola_numerical_checks(&train_device)
                .map_err(anyhow::Error::msg)?,
        )?;
        runtime::write_json(&report_path, &result)?;
    }
    for (batch, time) in [(1, 128), (1, 512), (4, 512), (1, 2048)] {
        deadline(started, c)?;
        let raw = Raw::new(&train_device, batch, time, c.seed);
        runtime::sync(device)?;
        let reference = raw.run(ScanBackend::TensorSolver, true);
        let mut equivalence = Vec::new();
        for &backend in &backends {
            let actual = raw.run(backend, true);
            let mut error = equivalent(
                "raw output",
                &values(actual.0.output),
                &values(reference.0.output.clone()),
            )?;
            error = error.max(equivalent(
                "raw final state",
                &values(actual.0.final_state),
                &values(reference.0.final_state.clone()),
            )?);
            error = error.max(equivalent(
                "raw physical scores",
                &values(actual.0.scores),
                &values(reference.0.scores.clone()),
            )?);
            for (a, b) in actual.1.into_iter().zip(&reference.1) {
                error = error.max(equivalent(
                    "raw input/state gradient",
                    &values(a),
                    &values(b.clone()),
                )?);
            }
            equivalence.push(json!({"scan_backend":backend,"max_abs":error}));
        }
        let mut timings = vec![vec![Vec::new(), Vec::new()]; backends.len()];
        for iteration in 0..warmups + trials {
            for offset in 0..backends.len() {
                deadline(started, c)?;
                let index = (iteration + offset) % backends.len();
                for (mode, backward) in [false, true].into_iter().enumerate() {
                    let ms = raw_timed(&raw, backends[index], backward, device)?;
                    if iteration >= warmups {
                        timings[index][mode].push(ms);
                    }
                }
            }
        }
        let measurements=backends.iter().enumerate().map(|(i,&backend)|json!({"scan_backend":backend,"forward_ms":timings[i][0],"forward_median_ms":median(&timings[i][0]),"forward_backward_ms":timings[i][1],"forward_backward_median_ms":median(&timings[i][1]),"tokens_per_second":(batch*time)as f64/(median(&timings[i][1])/1000.)})).collect::<Vec<_>>();
        result["raw"].as_array_mut().unwrap().push(json!({"batch":batch,"context":time,"heads":4,"key_dim":32,"value_dim":64,"chunk_size":16,"equivalence_before_throughput":equivalence,"measurements":measurements}));
        runtime::write_json(&report_path, &result)?;
    }
    append_hola_benchmarks(c, &train_device, trials, started, &report_path, &mut result)?;
    append_model_benchmarks(
        c,
        device,
        warmups,
        trials,
        started,
        &directory,
        &report_path,
        &mut result,
    )?;
    result["completed"] = json!(true);
    result["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    result["report_path"] = json!(report_path);
    runtime::write_json(&report_path, &result)?;
    Ok(result)
}

/// Paired GDN-2 training paths; original tolerances, independent CPU model oracle.
pub fn gdn_run(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(
        (1..=20).contains(&warmups) && (3..=31).contains(&trials),
        "warmups 1..20 and trials 3..31 required"
    );
    ensure!(
        c.backend == "vulkan" || c.backend == "cpu",
        "GDN benchmark is local CPU/Vulkan only"
    );
    ensure!(
        c.heads == 4 && c.key_dim == 32 && c.value_dim == 64 && c.chunk_size == 16,
        "GDN benchmark fixture requires H4/K32/V64/chunk16"
    );
    let started = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c
        .artifact_dir
        .join(format!("gdn-benchmark-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let report_path = directory.join("report.json");
    let td = device.clone().autodiff();
    let backends = if c.backend == "vulkan" {
        vec![ScanBackend::TensorSolver, ScanBackend::FusedGdn2]
    } else {
        vec![ScanBackend::TensorSolver]
    };
    let mut result = json!({"schema":1,"command":"gdn-benchmark","completed":false,
        "config":c,"device":runtime::identity(device),"report_path":report_path,
        "absolute_tolerance":ATOL,"relative_tolerance":RTOL,"warmups":warmups,"trials":trials,
        "timing_method":"resident identical fixtures, synchronize before/after, alternating paired order; first-use separate; no transfers/checkpoint IO in warm timing",
        "memory_scope":"analytic f32 allocations, not measured peak GPU memory; slice/autodiff/runtime allocations excluded",
        "numerical_gate":[],"raw":[],"full_model":[]});
    runtime::write_json(&report_path, &result)?;
    for &backend in &backends {
        deadline(started, c)?;
        result["active_case"] = json!({"scope":"f64 numerical gate","scan_backend":backend});
        runtime::write_json(&report_path, &result)?;
        let gate = delta_mixers::checks::gdn_numerical_checks_selected(&td, backend, None, None)
            .map_err(anyhow::Error::msg)?;
        result["numerical_gate"]
            .as_array_mut()
            .unwrap()
            .push(json!(gate));
        runtime::write_json(&report_path, &result)?;
    }
    for time in [128usize, 512] {
        for batch_size in [1usize, 8] {
            deadline(started, c)?;
            let raw = Raw::new_kind(&td, batch_size, time, c.seed, MixerKind::Gdn2);
            let mut first_use = Vec::new();
            for &backend in &backends {
                first_use.push(json!({"scan_backend":backend,
                    "forward_backward_ms":raw_timed(&raw, backend, true, device)?}));
            }
            let mut timings = vec![vec![Vec::new(), Vec::new()]; backends.len()];
            for iteration in 0..warmups + trials {
                for offset in 0..backends.len() {
                    deadline(started, c)?;
                    let index = (iteration + offset) % backends.len();
                    for (mode, backward) in [false, true].into_iter().enumerate() {
                        let ms = raw_timed(&raw, backends[index], backward, device)?;
                        if iteration >= warmups {
                            timings[index][mode].push(ms);
                        }
                    }
                }
            }
            let measurements = backends
                .iter()
                .enumerate()
                .map(|(i, backend)| {
                    json!({
                        "scan_backend":backend,"forward_ms":timings[i][0],
                        "forward_median_ms":median(&timings[i][0]),
                        "forward_backward_ms":timings[i][1],
                        "forward_backward_median_ms":median(&timings[i][1])
                    })
                })
                .collect::<Vec<_>>();
            result["raw"].as_array_mut().unwrap().push(json!({
                "batch":batch_size,"context":time,"heads":4,"key_dim":32,"value_dim":64,"chunk_size":16,
                "first_shape_use_after_numerical_gate":first_use,"measurements":measurements,
                "fused_checkpoint_bytes":4*batch_size*4*time.div_ceil(16)*32*64,
                "fused_auxiliary_bytes":4*batch_size*time*4*66,
                "fused_backward_replay_bytes":4*batch_size*4*16*32*64,
                "fused_packed_output_bytes":4*(batch_size*time*4*65+batch_size*4*32*64),
                "fused_forward_dispatches":1,"fused_backward_dispatches":1,
                "fused_shared_bytes_per_backward_workgroup":32768,
                "fused_recurrence_work":"O(B*T*H*K*V), sequential time, no quadratic time materialization"
            }));
            runtime::write_json(&report_path, &result)?;
        }
        for hola in [false, true] {
            deadline(started, c)?;
            result["active_case"] =
                json!({"scope":"CPU full-model oracle and steps","context":time,"hola":hola});
            runtime::write_json(&report_path, &result)?;
            let mut config = c.clone();
            config.batch_size = 8;
            let PreparedModelCase {
                config,
                batch,
                mut runners,
                mut report,
            } = prepare_model_case_kind(&config, device, time, hola, None, MixerKind::Gdn2)?;
            let mut samples = vec![Vec::new(); runners.len()];
            for iteration in 0..warmups + trials {
                for offset in 0..runners.len() {
                    deadline(started, c)?;
                    let index = (iteration + offset) % runners.len();
                    let ms = runners[index].step(&config, &batch, device)?;
                    if iteration >= warmups {
                        samples[index].push(ms);
                    }
                }
            }
            report["measurements"] = json!(
                runners
                    .iter()
                    .enumerate()
                    .map(|(i, runner)| json!({
                        "scan_backend":runner.backend,"hola_backend":runner.hola_backend,
                        "step_ms":samples[i],"median_step_ms":median(&samples[i]),
                        "tokens_per_second":(8*time) as f64/(median(&samples[i])/1000.)
                    }))
                    .collect::<Vec<_>>()
            );
            result["full_model"].as_array_mut().unwrap().push(report);
            runtime::write_json(&report_path, &result)?;
        }
    }
    result["completed"] = json!(true);
    result["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    runtime::write_json(&report_path, &result)?;
    Ok(result)
}

#[derive(Default)]
struct MembershipTrace {
    layer: Option<usize>,
    entries: Vec<MembershipEntry>,
}
struct MembershipEntry {
    layer: Option<usize>,
    chunk_start: usize,
    used: bool,
    shape: [usize; 3],
    ordered: Vec<i32>,
    members: Vec<i32>,
}
impl delta_mixers::profile::Sections for MembershipTrace {
    const CAPTURE_CACHE_MEMBERSHIP: bool = true;
    fn block(&mut self, index: Option<usize>) {
        self.layer = index;
    }
    fn section<T>(&mut self, _name: &'static str, run: impl FnOnce() -> T) -> T {
        run()
    }
    fn cache_membership(
        &mut self,
        chunk_start: usize,
        used: bool,
        _kept_slots: &Tensor<3, Int>,
        positions: &Tensor<3>,
    ) {
        let shape = positions.dims();
        let ordered = values(positions.clone())
            .into_iter()
            .map(|x| x as i32)
            .collect::<Vec<_>>();
        let mut members = ordered.clone();
        // Sorting separately within each batch/head distinguishes rank reorder
        // from a change in the set that the following chunk actually reads.
        for row in members.chunks_exact_mut(shape[2]) {
            row.sort_unstable();
        }
        self.entries.push(MembershipEntry {
            layer: self.layer,
            chunk_start,
            used,
            shape,
            ordered,
            members,
        });
    }
}
impl MembershipTrace {
    fn report(&self) -> Result<Value> {
        Ok(json!(
            self.entries
                .iter()
                .map(|entry| -> Result<Value> {
                    Ok(json!({
                        "layer":entry.layer,"chunk_start":entry.chunk_start,
                        "used_by_later_chunk":entry.used,"shape":entry.shape,
                        "ordered_positions_sha256":data::hash(&serde_json::to_vec(&entry.ordered)?),
                        "membership_set_sha256":data::hash(&serde_json::to_vec(&entry.members)?)
                    }))
                })
                .collect::<Result<Vec<_>>>()?
        ))
    }
    fn changes_from(&self, base: &Self) -> Result<Value> {
        ensure!(
            self.entries.len() == base.entries.len(),
            "membership trace length changed"
        );
        let mut changes = Vec::new();
        for (entry, previous) in self.entries.iter().zip(&base.entries) {
            ensure!(
                entry.layer == previous.layer
                    && entry.chunk_start == previous.chunk_start
                    && entry.shape == previous.shape,
                "membership trace layout changed"
            );
            if entry.ordered != previous.ordered {
                changes.push(json!({
                    "layer":entry.layer,"chunk_start":entry.chunk_start,
                    "used_by_later_chunk":entry.used,
                    "set_changed":entry.members != previous.members,
                    "changed_ordered_slots":entry.ordered.iter().zip(&previous.ordered).filter(|(a,b)| a != b).count(),
                    "changed_set_slots":entry.members.iter().zip(&previous.members).filter(|(a,b)| a != b).count()
                }));
            }
        }
        Ok(json!(changes))
    }
}

/// Investigation only: never substitutes a passing epsilon for the existing gate.
/// Reports actual tensor-HOLA cache selection, not an inferred top-k reconstruction.
pub fn directional_diagnostic(c: &Config, device: &Device) -> Result<Value> {
    ensure!(c.backend == "cpu", "directional diagnostic is CPU-only");
    let started = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = c
        .artifact_dir
        .join(format!("directional-diagnostic-{nonce}.json"));
    std::fs::create_dir_all(&c.artifact_dir)?;
    let mut config = c.clone();
    config.scan_backend = ScanBackend::TensorSolver;
    config.hola_backend = HolaBackend::Tensor;
    config.projection_backend = delta_mixers::ProjectionBackend::Separate;
    config.normalization_backend = delta_mixers::NormalizationBackend::Tensor;
    config.convolution_backend = delta_mixers::ConvolutionBackend::Tensor;
    config.head_backend = crate::config::HeadBackend::Native;
    config.loss_backend = crate::config::LossBackend::Tensor;
    config.update_backend = crate::config::UpdateBackend::Tensor;
    let mut report = json!({
        "command":"gdn-directional-diagnostic","completed":false,"acceptance_changed":false,
        "original_epsilon":0.002,"absolute_tolerance":0.003,"relative_tolerance":0.03,
        "source_config":c,"cpu_oracle_config":config,"device":runtime::identity(device),
        "report_path":path,"cases":[],
        "scope":"Fixed epsilon sweep. Each sample reports the difference with hard cache decisions recomputed (free) and with the unperturbed forward's kept slots replayed (fixed, as used by the model-check gate)."
    });
    runtime::write_json(&path, &report)?;
    let td = device.clone().autodiff();
    for (kind, batch_size) in [
        (MixerKind::Gdn2, 8usize),
        (MixerKind::KronLstm, 1),
        (MixerKind::KronLstm, 8),
    ] {
        deadline(started, c)?;
        config.batch_size = batch_size;
        let arm = Arm {
            name: format!("{kind:?}-hola"),
            mixer: if kind == MixerKind::Gdn2 {
                "gdn2"
            } else {
                "kronlstm"
            }
            .into(),
            cache: "surprise".into(),
        };
        td.seed(c.seed);
        let base = LanguageModel::new(&config, &arm, &td)?;
        let rows = (0..batch_size)
            .map(|row| Example {
                input: (0..c.context)
                    .map(|t| ((t * 17 + row * 11 + c.seed as usize) % config.vocab()) as i32)
                    .collect(),
                target: (0..c.context)
                    .map(|t| (((t + 1) * 17 + row * 11 + c.seed as usize) % config.vocab()) as i32)
                    .collect(),
                weight: vec![1.; c.context],
                groups: vec![None; c.context],
            })
            .collect::<Vec<_>>();
        let batch = Batch::new(&rows, &td);
        let ordinary_loss = base.loss(base.forward(&batch), &batch);
        let ordinary_loss_value = model::scalar(ordinary_loss.clone())?;
        let grads = model::checked_gradients(&base, ordinary_loss)?;
        let snapshot = parameters(&base, Some(&grads))?;
        let analytic = directional_analytic(&snapshot);
        let mut fixed_selection = FixedCacheSelection::record(&base, &batch);
        let tolerance = 3e-3 + 0.03 * analytic.abs();
        let mut base_trace = MembershipTrace::default();
        let inference = base.valid();
        let base_logits = inference.forward_profiled(&batch, &mut base_trace);
        let traced_loss = model::scalar(inference.loss(base_logits, &batch))?;
        let index = report["cases"].as_array().unwrap().len();
        report["cases"].as_array_mut().unwrap().push(json!({
            "mixer":kind,"batch":batch_size,"context":c.context,"cache":"surprise",
            "initial_parameters_sha256":data::hash(&serde_json::to_vec(&parameters(&base, None)?)?),
            "fixture_sha256":data::fixture_hash(&rows),"all_parameter_tensors_perturbed":snapshot.len(),
            "analytic":analytic,"tolerance":tolerance,
            "ordinary_base_loss":ordinary_loss_value,"traced_base_loss":traced_loss,
            "base_memberships":base_trace.report()?,"samples":[]
        }));
        runtime::write_json(&path, &report)?;
        for epsilon in [5e-4, 1e-3, 2e-3, 4e-3] {
            deadline(started, c)?;
            let mut plus_trace = MembershipTrace::default();
            let plus = base.valid().map(&mut PerturbParameters {
                scale: epsilon,
                index: 0,
            });
            let logits = plus.forward_profiled(&batch, &mut plus_trace);
            let plus_loss = model::scalar(plus.loss(logits, &batch))?;
            drop(plus);
            let mut minus_trace = MembershipTrace::default();
            let minus = base.valid().map(&mut PerturbParameters {
                scale: -epsilon,
                index: 0,
            });
            let logits = minus.forward_profiled(&batch, &mut minus_trace);
            let minus_loss = model::scalar(minus.loss(logits, &batch))?;
            let numerical = (plus_loss - minus_loss) / (2. * epsilon);
            let error = (analytic - numerical).abs();
            let (fixed_plus, fixed_minus) =
                fixed_selection_difference(&base, &batch, &mut fixed_selection, epsilon)?;
            let fixed_numerical = (fixed_plus - fixed_minus) / (2. * epsilon);
            let fixed_error = (analytic - fixed_numerical).abs();
            report["cases"][index]["samples"].as_array_mut().unwrap().push(json!({
                "epsilon":epsilon,"per_element_direction_magnitude":0.01,
                "plus_loss":plus_loss,"minus_loss":minus_loss,
                "numerical":numerical,"absolute_error":error,
                "within_original_tolerance":analytic.is_finite() && numerical.is_finite() && error <= tolerance,
                "fixed_selection_plus_loss":fixed_plus,"fixed_selection_minus_loss":fixed_minus,
                "fixed_selection_numerical":fixed_numerical,"fixed_selection_absolute_error":fixed_error,
                "fixed_selection_within_original_tolerance":fixed_numerical.is_finite() && fixed_error <= tolerance,
                "plus_memberships":plus_trace.report()?,"minus_memberships":minus_trace.report()?,
                "plus_vs_base":plus_trace.changes_from(&base_trace)?,
                "minus_vs_base":minus_trace.changes_from(&base_trace)?,
                "plus_vs_minus":plus_trace.changes_from(&minus_trace)?
            }));
            runtime::write_json(&path, &report)?;
        }
    }
    report["completed"] = json!(true);
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    runtime::write_json(&path, &report)?;
    Ok(report)
}
