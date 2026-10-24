//! Component VJP attribution alongside an unchanged, guarded full-model training step.
use crate::{
    config::{Arm, Config},
    data::{self, Batch},
    model::{self, LanguageModel},
    profile::{Runner, median},
    runtime, train,
};
use anyhow::{Result, ensure};
use burn::{
    prelude::*,
    tensor::{Distribution, activation::silu},
};
use delta_mixers::{
    attribution::{Component, Fixture},
    profile::{Sections, Unprofiled},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

struct Bound {
    start: Instant,
    seconds: f64,
}
impl Bound {
    fn check(&self) -> Result<()> {
        ensure!(
            self.start.elapsed().as_secs_f64() < self.seconds,
            "backward-profile reached max_seconds={}; partial report retained",
            self.seconds
        );
        Ok(())
    }
}

/// No section barriers in the ordinary step. The comparison brackets only backward,
/// and the already sequential safety checks/update; it does not synchronize forward sections.
struct StepTiming<'a> {
    device: &'a Device,
    totals: BTreeMap<&'static str, f64>,
    peak_reserved: Option<u64>,
    error: Option<String>,
}
impl Sections for StepTiming<'_> {
    fn section<T>(&mut self, name: &'static str, run: impl FnOnce() -> T) -> T {
        if !matches!(
            name,
            "Backward (total only)"
                | "Gradient checks: reductions"
                | "Host/device: gradient norm readback"
                | "AdamW update (including clipping)"
        ) {
            return run();
        }
        if let Err(error) = runtime::sync(self.device) {
            self.error = Some(error.to_string());
        }
        let clock = Instant::now();
        let output = run();
        if let Err(error) = runtime::sync(self.device) {
            self.error = Some(error.to_string());
        }
        *self.totals.entry(name).or_default() += clock.elapsed().as_secs_f64() * 1000.;
        // Sample outside the timer, after existing backward/update barriers only.
        if let Some(bytes) = runtime::sampled_reserved(self.device) {
            self.peak_reserved = Some(self.peak_reserved.unwrap_or(0).max(bytes));
        }
        output
    }
}

fn sample(
    device: &Device,
    instances: usize,
    backward: bool,
    run: &impl Fn(usize) -> Tensor<1>,
) -> Result<f64> {
    // Allocate harness handle storage before the clock, retain graphs/gradients through sync.
    let mut outputs = Vec::with_capacity(instances);
    let mut gradients = Vec::with_capacity(instances);
    runtime::sync(device)?;
    let clock = Instant::now();
    for index in 0..instances {
        let output = run(index);
        if backward {
            gradients.push(output.backward());
        }
        outputs.push(output);
    }
    runtime::sync(device)?;
    let ms = clock.elapsed().as_secs_f64() * 1000.;
    std::hint::black_box((outputs, gradients));
    Ok(ms)
}

fn benchmark(
    device: &Device,
    bound: &Bound,
    instances: usize,
    warmups: usize,
    trials: usize,
    run: impl Fn(usize) -> Tensor<1>,
) -> Result<Value> {
    let mut forward = Vec::with_capacity(trials);
    let mut combined = Vec::with_capacity(trials);
    for iteration in 0..warmups + trials {
        bound.check()?;
        let (f, fb) = if iteration % 2 == 0 {
            (
                sample(device, instances, false, &run)?,
                sample(device, instances, true, &run)?,
            )
        } else {
            let fb = sample(device, instances, true, &run)?;
            (sample(device, instances, false, &run)?, fb)
        };
        if iteration >= warmups {
            forward.push(f);
            combined.push(fb);
        }
    }
    let (f, fb) = (median(&forward), median(&combined));
    Ok(json!({"instances":instances,"forward_samples_ms":forward,
        "forward_backward_samples_ms":combined,"forward_median_ms":f,
        "forward_backward_median_ms":fb,"derived_backward_median_ms":fb-f}))
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn launches(device: &Device, run: impl FnOnce()) -> Result<Value> {
    use std::sync::{Arc, Mutex};
    #[derive(Default)]
    struct Count(Mutex<BTreeMap<&'static str, usize>>);
    impl cubecl::logging::LaunchObserver for Count {
        fn launched(&self, kernel: &'static str) {
            *self.0.lock().unwrap().entry(kernel).or_default() += 1;
        }
    }
    runtime::sync(device)?;
    let counts = Arc::new(Count::default());
    let guard = cubecl::logging::LaunchObservation::new(counts.clone());
    run();
    runtime::sync(device)?;
    drop(guard);
    let counts = counts.0.lock().unwrap();
    Ok(
        json!({"issued_kernels":counts.values().sum::<usize>(),"by_kernel":*counts,
        "scope":"separate warm observation; includes fill/slice/adjoint preparation, not copies or submissions; observer absent from timing trials"}),
    )
}
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
fn launches(_device: &Device, _run: impl FnOnce()) -> Result<Value> {
    anyhow::bail!("backward-profile requires a GPU build")
}

fn coverage(rows: &[Value], real_backward: f64, step: f64) -> Value {
    let f = rows
        .iter()
        .map(|r| r["forward_median_ms"].as_f64().unwrap())
        .sum::<f64>();
    let fb = rows
        .iter()
        .map(|r| r["forward_backward_median_ms"].as_f64().unwrap())
        .sum::<f64>();
    let backward = fb - f;
    json!({"sum_isolated_forward_ms":f,"sum_isolated_forward_backward_ms":fb,
        "sum_derived_backward_ms":backward,"real_full_step_backward_ms":real_backward,
        "backward_coverage_percent":100.*backward/real_backward,
        "real_minus_isolated_backward_ms":real_backward-backward,
        "sum_isolated_minus_ordinary_step_ms":fb-step,
        "warning":"not additive GPU utilization or predicted savings: independent all-ones output VJPs, multioutput sum roots, leaf-gradient accumulation/extraction, barriers and launch/overlap differences contribute to the signed residual"})
}

fn validate_shape(c: &Config) -> Result<()> {
    ensure!(
        (
            c.d_model,
            c.layers,
            c.heads,
            c.key_dim,
            c.value_dim,
            c.chunk_size,
            c.cache_capacity
        ) == (256, 4, 4, 32, 64, 16, 16)
            && c.batch_size > 0,
        "use positive batch size and the recommended D256/L4/H4/K32/V64/chunk16/cache16 model shapes"
    );
    Ok(())
}

pub fn run(
    c: &Config,
    arm: &Arm,
    device: &Device,
    contexts: &[usize],
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    use delta_mixers::{CachePolicy, HolaBackend, MixerKind, ProjectionBackend, ScanBackend};
    ensure!(
        matches!(c.backend.as_str(), "vulkan" | "cuda") && !c.fusion,
        "backward attribution requires the non-fusion GPU backend"
    );
    let mixer_config = arm.mixer_config(c)?;
    ensure!(
        mixer_config.kind == MixerKind::KronLstm
            && mixer_config.cache_policy != CachePolicy::None
            && c.scan_backend == ScanBackend::FusedGpu
            && c.hola_backend == HolaBackend::FusedGpu
            && matches!(c.projection_backend, ProjectionBackend::Merged | ProjectionBackend::MergedHeuristicOutput)
            && c.normalization_backend.is_tensor(),
        "backward attribution requires KronLSTM + HOLA, fused scan/read, merged projections and tensor normalization"
    );
    validate_shape(c)?;
    ensure!(c.task == "lm", "backward attribution requires the byte LM");
    ensure!(
        (1..=10).contains(&warmups) && (3..=31).contains(&trials),
        "warmups 1..=10 and trials 3..=31 required"
    );
    ensure!(
        !contexts.is_empty()
            && contexts.len() <= 2
            && contexts.iter().all(|t| [128, 512].contains(t)),
        "backward attribution contexts must be 128 and/or 512"
    );
    let bound = Bound {
        start: Instant::now(),
        seconds: c.max_seconds.min(300.),
    };
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = c.artifact_dir.join(format!(
        "backward-profile-{}-{nonce}/report.json",
        std::process::id()
    ));
    let mut report = json!({"schema":1,"command":"backward-profile","completed":false,
        "config":c,"arm":arm,"device":runtime::identity(device),"dtype":"f32",
        "report_path":path,"warmups_per_mode":warmups,"trials":trials,"max_seconds":bound.seconds,
        "method":"resident independent-leaf exact-shape fixtures call production operators; each layer measured, not one layer extrapolated. Both modes record autodiff. Device sync brackets each component group. Alternating forward-only/forward+backward order; fresh graph each invocation. Sum of output elements supplies all-ones VJPs; multioutput groups include their sum roots in both modes. Scan uses output only, as in the LM (selection scores are discrete; final state unused). No synthetic initial-state gradient is requested.",
        "full_step":"ordinary actual byte-LM step includes every model part, loss/backward, finite-gradient safety checks, clipping and AdamW, data preparation/upload and scalar readbacks. Separate alternating scratch trajectory times only backward/check/update boundaries. No model parts are stubbed or removed.",
        "optimizer_row":"checks+AdamW are a non-autodiff component timed after real full-model gradients; fwd=fwd+bwd=complete guarded update, backward=0; warmed moments/clipping/readback included, gradient generation excluded",
        "excluded":"initialization, corpus loading, cold JIT/first use, warmups, transfers for isolated fixtures, checkpoint/report IO. Ordinary step still includes production transfers.",
        "workgroup_count":"scan geometry is source-derived, not a hardware occupancy counter: batch*heads 64-lane workgroups in EACH forward and backward scan kernel; no partition over time/key/value. Launch observer independently counts issued kernels.",
        "cases":[]});
    runtime::write_json(&path, &report)?;
    let outcome = (|| -> Result<()> {
        let corpus = train::loaded_corpus(c)?;
        for &context in contexts {
            bound.check()?;
            let mut config = c.clone();
            config.context = context;
            config.validate()?;
            device.memory_cleanup();
            let device = device.clone().autodiff();
            device.seed(c.seed);
            let base = LanguageModel::new(&config, arm, &device)?;
            let rows = data::examples(&config, corpus.as_ref(), c.seed, 0, c.batch_size, "train")?;
            let batch = Batch::new(&rows, &device);
            let mut plain = Runner::new(base.clone(), &config);
            let mut timed = Runner::new(base.clone(), &config);
            let mut ordinary = Vec::new();
            let mut backward = Vec::new();
            let mut updates = Vec::new();
            let mut peak_reserved = None;
            for iteration in 0..warmups + trials {
                bound.check()?;
                let mut timing = StepTiming {
                    device: &device,
                    totals: BTreeMap::new(),
                    peak_reserved: None,
                    error: None,
                };
                let ms = if iteration % 2 == 0 {
                    let ms = plain.step(&config, corpus.as_ref(), &device, &mut Unprofiled)?;
                    timed.step(&config, corpus.as_ref(), &device, &mut timing)?;
                    ms
                } else {
                    timed.step(&config, corpus.as_ref(), &device, &mut timing)?;
                    plain.step(&config, corpus.as_ref(), &device, &mut Unprofiled)?
                };
                ensure!(
                    timing.error.is_none(),
                    "full step sync failed: {:?}",
                    timing.error
                );
                if iteration >= warmups {
                    if let Some(bytes) = timing.peak_reserved {
                        peak_reserved = Some(peak_reserved.unwrap_or(0).max(bytes));
                    }
                    ordinary.push(ms);
                    backward.push(timing.totals["Backward (total only)"]);
                    updates.push(
                        timing
                            .totals
                            .iter()
                            .filter(|(name, _)| **name != "Backward (total only)")
                            .map(|(_, ms)| ms)
                            .sum::<f64>(),
                    );
                }
            }
            let step_ms = median(&ordinary);
            let real_backward = median(&backward);
            let index = report["cases"].as_array().unwrap().len();
            report["cases"].as_array_mut().unwrap().push(json!({"context":context,
                "batch_size":c.batch_size,"tokens_per_step":config.tokens_per_step(),
                "full_step_tokens_per_second":1000.*config.tokens_per_step() as f64/step_ms,
                "fixture_sha256":data::fixture_hash(&rows),"parameters":base.num_params(),
                "ordinary_step_samples_ms":ordinary,"ordinary_step_median_ms":step_ms,
                "sampled_peak_pool_reserved_bytes_during_steps":peak_reserved,
                "step_memory_scope":"maximum sampled at existing backward/check/update boundaries of measured scratch steps; includes both independent model/optimizer trajectories and allocator reuse; not exact live bytes or physical-device peak",
                "real_backward_samples_ms":backward,"real_backward_median_ms":real_backward,
                "gradient_checks_AdamW_samples_ms":updates,"ranked_components":[],"scan_scaling":[]}));
            runtime::write_json(&path, &report)?;
            drop((plain, timed));
            let fixtures = base
                .blocks
                .iter()
                .map(|block| Fixture::new(&block.mixer, c.batch_size, context, &device))
                .collect::<Vec<_>>();
            let mut components = Vec::new();
            for component in Component::ALL {
                let mut result = benchmark(&device, &bound, c.layers, warmups, trials, |i| {
                    fixtures[i].forward(&base.blocks[i].mixer, component)
                })?;
                result["component"] = json!(component.name());
                components.push(result);
                report["cases"][index]["ranked_components"] = json!(components);
                runtime::write_json(&path, &report)?;
            }
            let hidden = (0..c.layers)
                .map(|_| {
                    Tensor::<3>::random(
                        [c.batch_size, context, c.d_model],
                        Distribution::Uniform(-0.5, 0.5),
                        &device,
                    )
                    .require_grad()
                })
                .collect::<Vec<_>>();
            let residual = (0..c.layers)
                .map(|_| {
                    Tensor::<3>::random(
                        [c.batch_size, context, c.d_model],
                        Distribution::Uniform(-0.5, 0.5),
                        &device,
                    )
                    .require_grad()
                })
                .collect::<Vec<_>>();
            for (name, mlp) in [
                ("MLP (up, SiLU, down)", true),
                ("Block RMSNorms and residuals", false),
            ] {
                let mut result = benchmark(&device, &bound, c.layers, warmups, trials, |i| {
                    let block = &base.blocks[i];
                    if mlp {
                        block
                            .down
                            .forward(silu(block.up.forward(hidden[i].clone())))
                            .flatten(0, 2)
                    } else {
                        base.normalization_backend
                            .rms(&block.norm, hidden[i].clone())
                            .sum()
                            + base
                                .normalization_backend
                                .rms(&block.ff_norm, hidden[i].clone())
                                .sum()
                            + (hidden[i].clone() + residual[i].clone()).sum()
                            + (hidden[i].clone() + residual[i].clone()).sum()
                    }
                })?;
                result["component"] = json!(name);
                components.push(result);
            }
            let mut final_row = benchmark(&device, &bound, 1, warmups, trials, |_| {
                model::loss(
                    base.head_forward(
                        base.normalization_backend
                            .rms(&base.norm, hidden[0].clone()),
                    ),
                    &batch,
                )
            })?;
            final_row["component"] = json!("Final norm, vocabulary head and loss");
            components.push(final_row);
            let mut embedding = benchmark(&device, &bound, 1, warmups, trials, |_| {
                base.embedding.forward(batch.input.clone()).flatten(0, 2)
            })?;
            embedding["component"] = json!("Embedding");
            components.push(embedding);
            let update = median(&updates);
            components.push(
                json!({"component":"Gradient checks and AdamW","instances":1,
                "forward_median_ms":update,"forward_backward_median_ms":update,
                "derived_backward_median_ms":0.,"non_autodiff":true,"samples_ms":updates}),
            );
            for row in &mut components {
                row["percent_of_full_step"] =
                    json!(100. * row["forward_backward_median_ms"].as_f64().unwrap() / step_ms);
            }
            components.sort_by(|a, b| {
                b["forward_backward_median_ms"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&a["forward_backward_median_ms"].as_f64().unwrap())
            });
            report["cases"][index]["coverage"] = coverage(&components, real_backward, step_ms);
            report["cases"][index]["ranked_components"] = json!(components);
            runtime::write_json(&path, &report)?;
            // Occupancy sweep changes only isolated scan batch/head counts, never the real LM.
            for batch_size in [1, 2, 4] {
                for heads in [2, 4, 8] {
                    bound.check()?;
                    let mut scan_config = arm.mixer_config(&config)?;
                    scan_config.heads = heads;
                    let mixer = scan_config.init(&device);
                    let fixture = Fixture::new(&mixer, batch_size, context, &device);
                    let mut result = benchmark(&device, &bound, 1, warmups, trials, |_| {
                        fixture.forward(&mixer, Component::Scan)
                    })?;
                    result["batch"] = json!(batch_size);
                    result["heads"] = json!(heads);
                    result["workgroups_per_scan_kernel"] = json!(batch_size * heads);
                    result["lanes_per_workgroup"] = json!(64);
                    result["tokens_heads_per_second"] = json!(
                        1000. * (batch_size * heads * context) as f64
                            / result["forward_backward_median_ms"].as_f64().unwrap()
                    );
                    result["dispatches"] = launches(&device, || {
                        let output = fixture.forward(&mixer, Component::Scan);
                        std::hint::black_box(output.backward());
                    })?;
                    report["cases"][index]["scan_scaling"]
                        .as_array_mut()
                        .unwrap()
                        .push(result);
                    runtime::write_json(&path, &report)?;
                }
            }
            report["cases"][index]["sampled_pool_reserved_bytes_after_case"] =
                json!(runtime::sampled_reserved(&device));
            report["cases"][index]["logical_scan_storage"] = json!({"state_bytes_per_layer":c.batch_size*c.heads*32*64*4,
                "checkpoint_bytes_per_layer":c.batch_size*c.heads*context.div_ceil(c.chunk_size)*32*64*4,
                "backward_replay_bytes_per_layer":c.batch_size*c.heads*c.chunk_size*32*64*4,
                "note":"logical scan storage only; pool reservation includes all scratch trajectories/fixtures and allocator reuse, not live/temporary traffic or physical-device peak"});
            runtime::write_json(&path, &report)?;
        }
        Ok(())
    })();
    report["completed"] = json!(outcome.is_ok());
    if let Err(error) = outcome {
        report["error"] = json!(format!("{error:#}"));
    }
    report["elapsed_seconds"] = json!(bound.start.elapsed().as_secs_f64());
    runtime::write_json(&path, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recommended_shapes_allow_positive_training_batches_only() {
        let mut c = Config::default();
        c.d_model = 256;
        c.layers = 4;
        c.heads = 4;
        c.key_dim = 32;
        c.value_dim = 64;
        c.chunk_size = 16;
        c.cache_capacity = 16;
        for batch in [1, 4, 8, 16, 32] {
            c.batch_size = batch;
            validate_shape(&c).unwrap();
        }
        c.batch_size = 0;
        assert!(validate_shape(&c).is_err());
        c.batch_size = 8;
        c.heads = 8;
        assert!(validate_shape(&c).is_err());
    }

    #[test]
    fn backward_coverage_uses_difference_without_clamping_or_optimizer_vjp() {
        let rows = vec![
            json!({"forward_median_ms":2.,"forward_backward_median_ms":7.}),
            json!({"forward_median_ms":3.,"forward_backward_median_ms":3.}),
        ];
        let result = coverage(&rows, 10., 12.);
        assert_eq!(result["sum_derived_backward_ms"], 5.);
        assert_eq!(result["backward_coverage_percent"], 50.);
        assert_eq!(result["real_minus_isolated_backward_ms"], 5.);
        assert_eq!(result["sum_isolated_minus_ordinary_step_ms"], -2.);
    }
}
