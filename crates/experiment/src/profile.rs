use crate::{
    config::{Arm, Config},
    data::{self, Batch, Corpus},
    model::{self, LanguageModel},
    runtime, train,
};
use anyhow::{Result, ensure};
use burn::{
    module::{ModuleVisitor, Param},
    optim::{GradientsParams, ModuleOptimizer},
    prelude::*,
};
use delta_mixers::profile::{Sections, Unprofiled};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Default)]
struct Timing {
    block: Option<usize>,
    totals: BTreeMap<&'static str, f64>,
    blocks: BTreeMap<(usize, &'static str), f64>,
    peak_reserved: Option<u64>,
    error: Option<String>,
}
struct Synchronized<'a> {
    device: &'a Device,
    timing: Timing,
}
impl Synchronized<'_> {
    fn sync(&mut self) {
        if let Err(error) = runtime::sync(self.device) {
            self.timing
                .error
                .get_or_insert_with(|| format!("{error:#}"));
        }
    }
}
impl Sections for Synchronized<'_> {
    fn block(&mut self, index: Option<usize>) {
        self.timing.block = index;
    }
    fn section<T>(&mut self, name: &'static str, run: impl FnOnce() -> T) -> T {
        self.sync();
        let clock = Instant::now();
        let result = run();
        self.sync();
        let ms = clock.elapsed().as_secs_f64() * 1000.;
        *self.timing.totals.entry(name).or_default() += ms;
        if let Some(block) = self.timing.block {
            *self.timing.blocks.entry((block, name)).or_default() += ms;
        }
        // Outside the section clock. Pool reservations are not live tensor bytes or a physical peak.
        if let Some(bytes) = runtime::sampled_reserved(self.device) {
            self.timing.peak_reserved = Some(self.timing.peak_reserved.unwrap_or(0).max(bytes));
        }
        result
    }
}

pub(super) struct Runner {
    model: Option<LanguageModel>,
    optim: ModuleOptimizer,
    step: usize,
}
impl Runner {
    pub(super) fn new(model: LanguageModel, c: &Config) -> Self {
        Self {
            model: Some(model),
            optim: train::optimizer(c),
            step: 0,
        }
    }
    pub(super) fn step<P: Sections>(
        &mut self,
        c: &Config,
        corpus: Option<&Corpus>,
        device: &Device,
        profile: &mut P,
    ) -> Result<f64> {
        runtime::sync(device)?;
        let clock = Instant::now();
        let rows = profile.section("Host batch preparation", || {
            data::examples(
                c,
                corpus,
                c.seed,
                self.step * c.batch_size,
                c.batch_size,
                "train",
            )
        })?;
        let batch = profile.section("Host/device: batch upload", || Batch::new(&rows, device));
        let model = self.model.take().expect("profile runner model");
        let logits = model.forward_profiled(&batch, profile);
        let objective = profile.section("Loss: log-softmax and gather", || {
            model::loss(logits, &batch)
        });
        profile.section("Host/device: loss readback", || {
            model::scalar(objective.clone())
        })?;
        let grads = model::checked_gradients_profiled(&model, objective, profile)?;
        self.model = Some(profile.section("AdamW update (including clipping)", || {
            self.optim.step(c.learning_rate, model, grads)
        }));
        runtime::sync(device)?;
        self.step += 1;
        Ok(clock.elapsed().as_secs_f64() * 1000.)
    }
}
pub(super) fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.
    } else {
        sorted[middle]
    }
}
fn ranked(samples: &[Timing], step_ms: f64, block: Option<usize>) -> Vec<Value> {
    let keys: Vec<_> = if let Some(block) = block {
        samples[0]
            .blocks
            .keys()
            .filter_map(|&(b, name)| (b == block).then_some(name))
            .collect()
    } else {
        samples[0].totals.keys().copied().collect()
    };
    let mut rows: Vec<_> = keys.into_iter().map(|name| {
        let values: Vec<_> = samples.iter().map(|sample| {
            if let Some(block) = block { sample.blocks[&(block, name)] } else { sample.totals[name] }
        }).collect();
        let ms = median(&values);
        json!({"section":name,"median_ms":ms,"percent_of_unsynchronized_step":100.*ms/step_ms,"samples_ms":values})
    }).collect();
    rows.sort_by(|a, b| {
        b["median_ms"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["median_ms"].as_f64().unwrap())
    });
    rows
}

struct Snapshot<'a> {
    grads: Option<&'a GradientsParams>,
    values: Vec<Vec<f32>>,
}
impl ModuleVisitor for Snapshot<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let tensor = match self.grads {
            Some(grads) => grads.get::<D>(param.id).expect("checked gradient"),
            None => param.val(),
        };
        self.values
            .push(tensor.into_data().try_to_vec::<f32>().expect("f32 model"));
    }
}
fn snapshot(model: &LanguageModel, grads: Option<&GradientsParams>) -> Vec<Vec<f32>> {
    let mut visitor = Snapshot {
        grads,
        values: Vec::new(),
    };
    model.visit(&mut visitor);
    visitor.values
}
fn equivalent(a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<f64> {
    ensure!(a.len() == b.len(), "profile equivalence parameter count");
    let mut maximum = 0.0f64;
    for (a, b) in a.iter().zip(b) {
        ensure!(a.len() == b.len(), "profile equivalence element count");
        for (&a, &b) in a.iter().zip(b) {
            let error = (a as f64 - b as f64).abs();
            ensure!(
                a.is_finite() && b.is_finite() && error <= 2e-5 + 3e-4 * (b as f64).abs(),
                "profile instrumentation changed numerical result: {a} vs {b}"
            );
            maximum = maximum.max(error);
        }
    }
    Ok(maximum)
}
fn instrumentation_check(
    c: &Config,
    base: &LanguageModel,
    batch: &Batch,
    device: &Device,
) -> Result<Value> {
    let plain = base.clone();
    let traced = base.clone();
    let mut profile = Synchronized {
        device,
        timing: Timing::default(),
    };
    let a = plain.forward(batch);
    let b = traced.forward_profiled(batch, &mut profile);
    let logits = equivalent(
        &[a.clone().into_data().try_to_vec::<f32>().unwrap()],
        &[b.clone().into_data().try_to_vec::<f32>().unwrap()],
    )?;
    let ga = model::checked_gradients(&plain, model::loss(a, batch))?;
    let gb = model::checked_gradients_profiled(&traced, model::loss(b, batch), &mut profile)?;
    let gradients = equivalent(&snapshot(&plain, Some(&ga)), &snapshot(&traced, Some(&gb)))?;
    let a = train::optimizer(c).step(c.learning_rate, plain, ga);
    let b = train::optimizer(c).step(c.learning_rate, traced, gb);
    let updated = equivalent(&snapshot(&a, None), &snapshot(&b, None))?;
    ensure!(
        profile.timing.error.is_none(),
        "profiling synchronization failed: {:?}",
        profile.timing.error
    );
    Ok(
        json!({"logits_max_abs":logits,"all_parameter_gradients_max_abs":gradients,"AdamW_parameters_max_abs":updated,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,"scope":"instrumentation equivalence; existing independent f64 component/model acceptance remains separate"}),
    )
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn dispatches(
    runner: &mut Runner,
    c: &Config,
    corpus: Option<&Corpus>,
    device: &Device,
) -> Result<Value> {
    use std::sync::{Arc, Mutex};
    #[derive(Default)]
    struct Count(Mutex<BTreeMap<&'static str, usize>>);
    impl cubecl::logging::LaunchObserver for Count {
        fn launched(&self, kernel: &'static str) {
            *self.0.lock().unwrap().entry(kernel).or_default() += 1;
        }
    }
    if c.backend == "cpu" {
        return Ok(json!({"available":false,"reason":"CPU backend does not issue CubeCL kernels"}));
    }
    runtime::sync(device)?;
    let counts = Arc::new(Count::default());
    let guard = cubecl::logging::LaunchObservation::new(counts.clone());
    runner.step(c, corpus, device, &mut Unprofiled)?;
    drop(guard);
    let kernels = counts.0.lock().unwrap();
    Ok(
        json!({"available":true,"issued_per_step":kernels.values().sum::<usize>(),"by_kernel":*kernels,"scope":"separate warm complete step; CubeCL launch observer counts issued kernels, not copies/submissions/workgroups; no graph replay or dry run"}),
    )
}
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
fn dispatches(
    _runner: &mut Runner,
    _c: &Config,
    _corpus: Option<&Corpus>,
    _device: &Device,
) -> Result<Value> {
    Ok(json!({"available":false,"reason":"CubeCL GPU features not enabled"}))
}

pub fn run(
    c: &Config,
    arm: &Arm,
    device: &Device,
    contexts: &[usize],
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    ensure!(
        (1..=10).contains(&warmups),
        "profile warmups must be 1..=10"
    );
    ensure!((3..=31).contains(&trials), "profile trials must be 3..=31");
    ensure!(
        !contexts.is_empty() && contexts.len() <= 8,
        "profile requires 1..=8 contexts"
    );
    let started = Instant::now();
    let deadline = c.max_seconds.min(300.);
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c
        .artifact_dir
        .join(format!("profile-{}-{nonce}", std::process::id()));
    let report_path = directory.join("report.json");
    let mut report = json!({"schema":1,"command":"profile","completed":false,"config":c,"arm":arm,"device":runtime::identity(device),"dtype":"f32","warmups_per_mode":warmups,"trials":trials,"max_seconds":deadline,"report_path":report_path,"cases":[],
        "timing":"wall-clock device.sync before and after each measured section; unsynchronized step has only outer barriers plus production scalar readbacks",
        "overlap_warning":"Synchronized section timing perturbs overlap and submission/fusion; sums of section medians may differ from the unsynchronized complete step. Percentages use the unsynchronized median and need not sum to 100%.",
        "backward_attribution":"Forward sections retain the real autodiff graph, not inference-only execution. Burn backward attribution is unavailable per section; total backward includes GradientsParams extraction. Gradient reductions/readbacks and AdamW are separate.",
        "memory_limitation":"Sampled backend pool bytes_reserved at section boundaries, not exact live/temporary allocation or physical-device peak; pool reuse, transient allocations inside sections, retained autodiff state and shared integrated-GPU memory prevent those interpretations. No exact temporary-allocation count is available.",
        "excluded":"corpus loading, initialization, instrumentation equivalence, cold compilation, warmups, checkpoint IO, evaluation, report IO; transfers within batch construction and both production scalar readbacks ARE included"});
    runtime::write_json(&report_path, &report)?;
    let outcome = (|| -> Result<()> {
        let corpus = train::loaded_corpus(c)?;
        for &context in contexts {
            let mut config = c.clone();
            config.context = context;
            config.validate()?;
            ensure!(
                started.elapsed().as_secs_f64() < deadline,
                "profile reached max_seconds={deadline}; partial report retained"
            );
            device.memory_cleanup();
            runtime::sync(device)?;
            let train_device = device.clone().autodiff();
            train_device.seed(c.seed);
            let base = LanguageModel::new(&config, arm, &train_device)?;
            let rows = data::examples(&config, corpus.as_ref(), c.seed, 0, c.batch_size, "train")?;
            let batch = Batch::new(&rows, &train_device);
            let parameters = base.num_params();
            runtime::sync(&train_device)?;
            let mut first = Runner::new(base.clone(), &config);
            let first_step_ms =
                first.step(&config, corpus.as_ref(), &train_device, &mut Unprofiled)?;
            drop(first);
            report["cases"].as_array_mut().unwrap().push(json!({"context":context,"parameters":parameters,"fixture_sha256":data::fixture_hash(&rows),"first_use":{"ordinary_step_ms":first_step_ms,"scope":"first complete ordinary training step at this context, before instrumentation/warmups; includes cache-miss JIT, fusion planning and optimizer initialization, excludes model initialization; earlier contexts may warm shared kernels; persistent cache state must be recorded separately"},"instrumentation_equivalence":null,"unsynchronized_step_samples_ms":[],"synchronized_step_samples_ms":[],"ranked_sections":[]}));
            let index = report["cases"].as_array().unwrap().len() - 1;
            runtime::write_json(&report_path, &report)?;
            let check = instrumentation_check(&config, &base, &batch, &train_device)?;
            report["cases"][index]["instrumentation_equivalence"] = check;
            let mut plain = Runner::new(base.clone(), &config);
            let mut traced = Runner::new(base, &config);
            let mut samples = Vec::new();
            let mut plain_ms = Vec::new();
            let mut traced_ms = Vec::new();
            for iteration in 0..warmups + trials {
                ensure!(
                    started.elapsed().as_secs_f64() < deadline,
                    "profile reached max_seconds={deadline}; partial report retained"
                );
                let mut profile = Synchronized {
                    device: &train_device,
                    timing: Timing::default(),
                };
                let (a, b) = if iteration % 2 == 0 {
                    let a = plain.step(&config, corpus.as_ref(), &train_device, &mut Unprofiled)?;
                    let b = traced.step(&config, corpus.as_ref(), &train_device, &mut profile)?;
                    (a, b)
                } else {
                    let b = traced.step(&config, corpus.as_ref(), &train_device, &mut profile)?;
                    let a = plain.step(&config, corpus.as_ref(), &train_device, &mut Unprofiled)?;
                    (a, b)
                };
                ensure!(
                    profile.timing.error.is_none(),
                    "profiling synchronization failed: {:?}",
                    profile.timing.error
                );
                if iteration >= warmups {
                    plain_ms.push(a);
                    traced_ms.push(b);
                    samples.push(profile.timing);
                    let step_ms = median(&plain_ms);
                    let section_rows = ranked(&samples, step_ms, None);
                    let case = &mut report["cases"][index];
                    case["unsynchronized_step_samples_ms"] = json!(plain_ms);
                    case["synchronized_step_samples_ms"] = json!(traced_ms);
                    case["unsynchronized_step_median_ms"] = json!(step_ms);
                    case["synchronized_step_median_ms"] = json!(median(&traced_ms));
                    case["sum_of_section_medians_ms"] = json!(
                        section_rows
                            .iter()
                            .map(|r| r["median_ms"].as_f64().unwrap())
                            .sum::<f64>()
                    );
                    case["ranked_sections"] = json!(section_rows);
                    case["per_block"] = json!((0..c.layers).map(|block| json!({"block":block,"ranked_sections":ranked(&samples,step_ms,Some(block))})).collect::<Vec<_>>());
                    case["sampled_pool_reserved_max_bytes"] =
                        json!(samples.iter().filter_map(|s| s.peak_reserved).max());
                    case["logical_storage"] = json!({"model_parameters_f32_bytes":parameters*4,"AdamW_two_moments_f32_bytes":parameters*8,"batch_host_uploaded_logical_bytes":c.batch_size*context*16,"scalar_readback_logical_bytes":8,"note":"logical payload only; excludes allocator padding, clipping intermediates, autodiff retention and convolution/cache temporaries; valid mask is created on device"});
                }
                report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
                runtime::write_json(&report_path, &report)?;
            }
            ensure!(
                started.elapsed().as_secs_f64() < deadline,
                "profile reached max_seconds={deadline}; partial report retained"
            );
            report["cases"][index]["dispatches"] =
                dispatches(&mut plain, &config, corpus.as_ref(), &train_device)?;
            runtime::write_json(&report_path, &report)?;
        }
        Ok(())
    })();
    report["completed"] = json!(outcome.is_ok());
    if let Err(error) = outcome {
        report["error"] = json!(format!("{error:#}"));
    }
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    runtime::write_json(&report_path, &report)?;
    Ok(report)
}

#[derive(Clone, Copy, serde::Serialize)]
#[serde(untagged)]
enum ExecutionChoice {
    Projection(delta_mixers::ProjectionBackend),
    Normalization(delta_mixers::NormalizationBackend),
    Hola(delta_mixers::HolaBackend),
}
impl ExecutionChoice {
    fn apply(self, model: &mut LanguageModel) {
        match self {
            Self::Projection(mode) => model.set_projection_backend(mode),
            Self::Normalization(mode) => model.set_normalization_backend(mode),
            Self::Hola(mode) => model.set_hola_backend(mode),
        }
    }
}

/// Paired execution-only contraction comparison on the actual guarded LM step.
pub fn projection_run(
    c: &Config,
    arm: &Arm,
    device: &Device,
    contexts: &[usize],
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    comparison_run(c, arm, device, contexts, warmups, trials, false, false)
}

pub fn normalization_run(
    c: &Config,
    arm: &Arm,
    device: &Device,
    contexts: &[usize],
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    comparison_run(c, arm, device, contexts, warmups, trials, true, false)
}

pub fn hola_run(
    c: &Config,
    arm: &Arm,
    device: &Device,
    contexts: &[usize],
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    comparison_run(c, arm, device, contexts, warmups, trials, false, true)
}

fn comparison_run(
    c: &Config,
    arm: &Arm,
    device: &Device,
    contexts: &[usize],
    warmups: usize,
    trials: usize,
    normalization: bool,
    hola: bool,
) -> Result<Value> {
    use delta_mixers::{NormalizationBackend, ProjectionBackend};
    ensure!(
        !(normalization || hola) || matches!(c.backend.as_str(), "vulkan" | "cuda"),
        "custom operator comparison requires a GPU; no CPU fallback"
    );
    ensure!(
        !hola || (!c.fusion && !c.autotune),
        "HOLA comparison requires non-fusion execution without matmul autotune"
    );
    let command = if hola {
        "hola-tune"
    } else if normalization {
        "normalization-benchmark"
    } else {
        "projection-benchmark"
    };
    let backend_key = if hola {
        "hola_backend"
    } else if normalization {
        "normalization_backend"
    } else {
        "projection_backend"
    };
    let forward_key = if hola {
        "hola_forward_samples_ms"
    } else if normalization {
        "normalization_forward_samples_ms"
    } else {
        "projection_forward_samples_ms"
    };
    let forward_median_key = if hola {
        "hola_forward_median_ms"
    } else if normalization {
        "normalization_forward_median_ms"
    } else {
        "projection_forward_median_ms"
    };
    ensure!((1..=10).contains(&warmups), "warmups must be 1..=10");
    ensure!((3..=31).contains(&trials), "trials must be 3..=31");
    ensure!(
        !contexts.is_empty() && contexts.len() <= 8,
        "requires 1..=8 contexts"
    );
    let started = Instant::now();
    let deadline = c.max_seconds.min(300.);
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c
        .artifact_dir
        .join(format!("{command}-{}-{nonce}", std::process::id()));
    let report_path = directory.join("report.json");
    let modes = if hola {
        [
            delta_mixers::HolaBackend::FusedGpu,
            delta_mixers::HolaBackend::FusedGpuPairs,
            delta_mixers::HolaBackend::FusedGpuPairsNorms,
        ]
        .map(ExecutionChoice::Hola)
    } else if normalization {
        [
            NormalizationBackend::Tensor,
            NormalizationBackend::FusedNorms,
            NormalizationBackend::FusedNormsGates,
        ]
        .map(ExecutionChoice::Normalization)
    } else {
        [
            ProjectionBackend::Separate,
            ProjectionBackend::Grouped,
            ProjectionBackend::Merged,
        ]
        .map(ExecutionChoice::Projection)
    };
    let mut report = json!({"schema":1,"command":command,"completed":false,
        "config":c,"arm":arm,"device":runtime::identity(device),"dtype":"f32",
        "warmups_per_mode":warmups,"trials":trials,"report_path":report_path,"cases":[],
        "protocol":"three independent identical-initialization trajectories; rotating forward/reverse mode order, same data and step indices; ordinary complete steps include forward, backward, checks, clipping and AdamW; separate section-barrier trajectories are not ordinary-step estimates",
        "memory_limitation":"sampled pool reserved bytes include all six scratch trajectories and allocator reuse; not physical peak or live temporary bytes"});
    runtime::write_json(&report_path, &report)?;
    let outcome = (|| -> Result<()> {
        if hola {
            report["independent_hola_reference"] = serde_json::to_value(
                delta_mixers::checks::hola_numerical_checks(device).map_err(anyhow::Error::msg)?,
            )?;
        } else if normalization {
            report["independent_normalization_reference"] = serde_json::to_value(
                delta_mixers::normalization::numerical_checks(device)
                    .map_err(anyhow::Error::msg)?,
            )?;
        } else {
            report["independent_projection_reference"] = serde_json::to_value(
                delta_mixers::projection_numerical_checks(device).map_err(anyhow::Error::msg)?,
            )?;
        }
        runtime::write_json(&report_path, &report)?;
        let corpus = train::loaded_corpus(c)?;
        for &context in contexts {
            let mut config = c.clone();
            config.context = context;
            if hola {
                config.hola_backend = delta_mixers::HolaBackend::FusedGpu;
            } else if normalization {
                config.normalization_backend = NormalizationBackend::Tensor;
            } else {
                config.projection_backend = ProjectionBackend::Separate;
            }
            config.validate()?;
            let isolated = if hola {
                Value::Null
            } else if normalization {
                serde_json::to_value(
                    delta_mixers::normalization::benchmark(
                        device,
                        context,
                        c.batch_size,
                        warmups,
                        trials,
                    )
                    .map_err(anyhow::Error::msg)?,
                )?
            } else {
                serde_json::to_value(
                    delta_mixers::projection_benchmark(
                        &arm.mixer_config(&config)?,
                        device,
                        context,
                        c.batch_size,
                        warmups,
                        trials,
                    )
                    .map_err(anyhow::Error::msg)?,
                )?
            };
            device.memory_cleanup();
            let train_device = device.clone().autodiff();
            train_device.seed(c.seed);
            let base = LanguageModel::new(&config, arm, &train_device)?;
            let rows = data::examples(&config, corpus.as_ref(), c.seed, 0, c.batch_size, "train")?;
            let batch = Batch::new(&rows, &train_device);
            let logits = base.forward(&batch);
            let gradients = model::checked_gradients(&base, model::loss(logits.clone(), &batch))?;
            let reference_logits = vec![logits.into_data().try_to_vec::<f32>().unwrap()];
            let reference_gradients = snapshot(&base, Some(&gradients));
            let updated = train::optimizer(&config).step(c.learning_rate, base.clone(), gradients);
            let reference_updated = snapshot(&updated, None);
            drop(updated);
            let mut plain = Vec::new();
            let mut traced = Vec::new();
            let mut checks = Vec::new();
            for mode in modes {
                let mut candidate = base.clone();
                mode.apply(&mut candidate);
                let logits = candidate.forward(&batch);
                let gradients =
                    model::checked_gradients(&candidate, model::loss(logits.clone(), &batch))?;
                let logits_error = equivalent(
                    &[logits.into_data().try_to_vec::<f32>().unwrap()],
                    &reference_logits,
                )?;
                let gradients_error = equivalent(
                    &snapshot(&candidate, Some(&gradients)),
                    &reference_gradients,
                )?;
                let updated =
                    train::optimizer(&config).step(c.learning_rate, candidate.clone(), gradients);
                let update_error = equivalent(&snapshot(&updated, None), &reference_updated)?;
                checks.push(json!({(backend_key):mode,"logits_max_abs":logits_error,
                    "every_parameter_gradient_max_abs":gradients_error,"AdamW_max_abs":update_error,
                    "absolute_tolerance":2e-5,"relative_tolerance":3e-4}));
                plain.push(Runner::new(candidate.clone(), &config));
                traced.push(Runner::new(candidate, &config));
            }
            let mut ordinary: [Vec<f64>; 3] = core::array::from_fn(|_| Vec::new());
            let mut sections: [Vec<Timing>; 3] = core::array::from_fn(|_| Vec::new());
            let mut synchronized: [Vec<f64>; 3] = core::array::from_fn(|_| Vec::new());
            let mut orders = Vec::new();
            for iteration in 0..warmups + trials {
                ensure!(
                    started.elapsed().as_secs_f64() < deadline,
                    "{command} reached max_seconds={deadline}"
                );
                let mut order = [iteration % 3, (iteration + 1) % 3, (iteration + 2) % 3];
                if iteration % 2 == 1 {
                    order.reverse();
                }
                for index in order {
                    let mut timing = Synchronized {
                        device: &train_device,
                        timing: Timing::default(),
                    };
                    let (ms, synchronized_ms) = if iteration % 2 == 0 {
                        let ms = plain[index].step(
                            &config,
                            corpus.as_ref(),
                            &train_device,
                            &mut Unprofiled,
                        )?;
                        let sync = traced[index].step(
                            &config,
                            corpus.as_ref(),
                            &train_device,
                            &mut timing,
                        )?;
                        (ms, sync)
                    } else {
                        let sync = traced[index].step(
                            &config,
                            corpus.as_ref(),
                            &train_device,
                            &mut timing,
                        )?;
                        let ms = plain[index].step(
                            &config,
                            corpus.as_ref(),
                            &train_device,
                            &mut Unprofiled,
                        )?;
                        (ms, sync)
                    };
                    ensure!(
                        timing.timing.error.is_none(),
                        "section synchronization failed: {:?}",
                        timing.timing.error
                    );
                    if iteration >= warmups {
                        ordinary[index].push(ms);
                        synchronized[index].push(synchronized_ms);
                        sections[index].push(timing.timing);
                    }
                }
                if iteration >= warmups {
                    orders.push(order.map(|i| modes[i]));
                }
            }
            let mut results = Vec::new();
            for index in 0..modes.len() {
                let projections: Vec<_> = sections[index]
                    .iter()
                    .map(|sample| {
                        sample
                            .totals
                            .iter()
                            .filter(|(name, _)| {
                                if hola {
                                    name.starts_with("HOLA")
                                } else if normalization {
                                    name.contains("RMSNorm")
                                        || name.starts_with("L2 ")
                                        || **name == "Residual adds"
                                } else {
                                    name.starts_with("Projection:")
                                }
                            })
                            .map(|(_, ms)| ms)
                            .sum::<f64>()
                    })
                    .collect();
                let paired_delta: Vec<_> = ordinary[0]
                    .iter()
                    .zip(&ordinary[index])
                    .map(|(old, new)| old - new)
                    .collect();
                results.push(json!({(backend_key):modes[index],"ordinary_samples_ms":ordinary[index],
                    "ordinary_median_ms":median(&ordinary[index]),"paired_savings_samples_ms":paired_delta,
                    "paired_savings_median_ms":median(&paired_delta),
                    "section_step_samples_ms":synchronized[index],(forward_key):projections,
                    (forward_median_key):median(&projections),
                    "ranked_sections":ranked(&sections[index],median(&ordinary[index]),None),
                    "sampled_pool_reserved_max_bytes":sections[index].iter().filter_map(|s| s.peak_reserved).max(),
                    "dispatches":dispatches(&mut plain[index],&config,corpus.as_ref(),&train_device)?}));
            }
            report["cases"].as_array_mut().unwrap().push(json!({"context":context,
                "fixture_sha256":data::fixture_hash(&rows),"parameters":base.num_params(),
                "equivalence":checks,"trial_orders":orders,
                (if hola { "isolated_hola" } else if normalization { "isolated_normalization" } else { "isolated_input_projections" }):isolated,
                "results":results}));
            runtime::write_json(&report_path, &report)?;
        }
        Ok(())
    })();
    report["completed"] = json!(outcome.is_ok());
    if let Err(error) = outcome {
        report["error"] = json!(format!("{error:#}"));
    }
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    runtime::write_json(&report_path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn medians_use_summed_section_time_per_trial() {
        let mut a = Timing::default();
        a.totals.insert("x", 3.);
        a.blocks.insert((0, "x"), 1.);
        let mut b = Timing::default();
        b.totals.insert("x", 7.);
        b.blocks.insert((0, "x"), 2.);
        let rows = ranked(&[a, b], 10., None);
        assert_eq!(rows[0]["median_ms"], 5.);
        assert_eq!(rows[0]["percent_of_unsynchronized_step"], 50.);
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn section_barriers_preserve_logits_gradients_and_adamw() {
        let mut c = Config::default();
        c.d_model = 8;
        c.layers = 1;
        c.context = 16;
        c.association_counts = vec![2];
        let device = Device::ndarray().autodiff();
        device.seed(719);
        for batch_size in [1, 2] {
            c.batch_size = batch_size;
            let base = LanguageModel::new(&c, &c.arms[0], &device).unwrap();
            let rows = data::examples(&c, None, c.seed, 0, c.batch_size, "train").unwrap();
            let batch = Batch::new(&rows, &device);
            instrumentation_check(&c, &base, &batch, &device).unwrap();
        }
    }
}
