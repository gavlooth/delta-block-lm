//! Same-session, identical-start recurrence-only execution comparison.
use crate::{
    config::{Arm, Config},
    data::{self, Batch},
    model::{self, LanguageModel},
    profile::{self, Runner},
    runtime, train,
};
use anyhow::{Result, ensure};
use burn::module::Module;
use delta_mixers::{ScanBackend, profile::Unprofiled};
use serde_json::{Value, json};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub fn run(
    c: &Config,
    arm: &Arm,
    device: &burn::prelude::Device,
    contexts: &[usize],
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    ensure!(
        c.backend == "vulkan" && !c.fusion && !c.autotune,
        "scan comparison requires historical non-fusion/non-autotune Vulkan execution"
    );
    ensure!(
        (1..=10).contains(&warmups) && (3..=31).contains(&trials),
        "warmups 1..10, trials 3..31 required"
    );
    ensure!(
        !contexts.is_empty() && contexts.iter().all(|n| [128, 512].contains(n)),
        "contexts must be 128 or 512"
    );
    let started = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c
        .artifact_dir
        .join(format!("scan-benchmark-{}-{nonce}", std::process::id()));
    let path = directory.join("report.json");
    let modes = [
        ScanBackend::FusedGpu,
        ScanBackend::FusedParallel128,
        ScanBackend::FusedParallel256,
    ];
    let mut report = json!({"schema":1,"command":"scan-benchmark","config":c,"arm":arm,"device":runtime::identity(device),
        "completed":false,"warmups":warmups,"trials":trials,"report_path":path,"cases":[],
        "protocol":"three identical-start model/AdamW trajectories; rotating/reversed order, same real corpus batch and step indices; full guarded forward+backward+AdamW includes batch upload and scalar readbacks; separate warm launch observation outside timing",
        "absolute_tolerance":2e-5,"relative_tolerance":3e-4,
        "pool_limitation":"pool reservation after observed steps includes all three scratch trajectories and allocator reuse, not exact live or physical peak memory"});
    runtime::write_json(&path, &report)?;
    let outcome = (|| -> Result<()> {
        let corpus = train::loaded_corpus(c)?;
        for &context in contexts {
            ensure!(
                started.elapsed().as_secs_f64() < c.max_seconds,
                "scan benchmark deadline"
            );
            let mut config = c.clone();
            config.context = context;
            config.scan_backend = ScanBackend::FusedGpu;
            let td = device.clone().autodiff();
            td.seed(c.seed);
            let base = LanguageModel::new(&config, arm, &td)?;
            let rows = data::examples(&config, corpus.as_ref(), c.seed, 0, c.batch_size, "train")?;
            let batch = Batch::new(&rows, &td);
            let logits = base.forward(&batch);
            let gradients = model::checked_gradients(&base, base.loss(logits.clone(), &batch))?;
            let reference_logits = crate::kron_benchmark::values(logits);
            let reference_gradients = crate::kron_benchmark::parameters(&base, Some(&gradients))?;
            let updated = train::optimizer(&config).step(c.learning_rate, base.clone(), gradients);
            let reference_updated = crate::kron_benchmark::parameters(&updated, None)?;
            drop(updated);
            let mut runners = Vec::new();
            let mut checks = Vec::new();
            for mode in modes {
                let mut candidate = base.clone();
                candidate.set_scan_backend(mode);
                let logits = candidate.forward(&batch);
                let gradients =
                    model::checked_gradients(&candidate, candidate.loss(logits.clone(), &batch))?;
                let le = crate::kron_benchmark::equivalent(
                    "paired logits",
                    &crate::kron_benchmark::values(logits),
                    &reference_logits,
                )?;
                let ge = crate::kron_benchmark::compare_parameters(
                    "paired gradients",
                    &crate::kron_benchmark::parameters(&candidate, Some(&gradients))?,
                    &reference_gradients,
                )?;
                let updated =
                    train::optimizer(&config).step(c.learning_rate, candidate.clone(), gradients);
                let ue = crate::kron_benchmark::compare_parameters(
                    "paired AdamW",
                    &crate::kron_benchmark::parameters(&updated, None)?,
                    &reference_updated,
                )?;
                checks.push(json!({"scan_backend":mode,"logits_max_abs":le,"every_parameter_gradient_max_abs":ge,"AdamW_max_abs":ue}));
                runners.push(Runner::new(candidate, &config));
            }
            let mut times = vec![Vec::new(); 3];
            let mut orders = Vec::new();
            let mut raw = vec![vec![Vec::new(), Vec::new()]; 3];
            let fixture = crate::kron_benchmark::Raw::new(&td, c.batch_size, context, c.seed);
            for iteration in 0..warmups + trials {
                ensure!(
                    started.elapsed().as_secs_f64() < c.max_seconds,
                    "scan benchmark deadline"
                );
                let mut order = [iteration % 3, (iteration + 1) % 3, (iteration + 2) % 3];
                if iteration % 2 == 1 {
                    order.reverse();
                }
                for index in order {
                    let ms = runners[index].step(&config, corpus.as_ref(), &td, &mut Unprofiled)?;
                    for (mode, backward) in [false, true].into_iter().enumerate() {
                        let sample = crate::kron_benchmark::raw_timed(
                            &fixture,
                            modes[index],
                            backward,
                            &td,
                        )?;
                        if iteration >= warmups {
                            raw[index][mode].push(sample);
                        }
                    }
                    if iteration >= warmups {
                        times[index].push(ms);
                    }
                }
                if iteration >= warmups {
                    orders.push(order);
                }
            }
            let mut measurements = Vec::new();
            for index in 0..3 {
                let dispatches =
                    profile::dispatches(&mut runners[index], &config, corpus.as_ref(), &td)?;
                measurements.push(json!({"scan_backend":modes[index],"step_samples_ms":times[index],"step_median_ms":profile::median(&times[index]),
                    "isolated_forward_ms":raw[index][0],"isolated_forward_median_ms":profile::median(&raw[index][0]),
                    "isolated_forward_backward_ms":raw[index][1],"isolated_forward_backward_median_ms":profile::median(&raw[index][1]),"dispatches":dispatches}));
            }
            report["cases"].as_array_mut().unwrap().push(json!({"context":context,"batch":c.batch_size,"parameters":base.num_params(),
                "fixture_sha256":data::fixture_hash(&rows),"preflight":checks,"orders":orders,"measurements":measurements,
                "sampled_pool_reserved_bytes":runtime::sampled_reserved(&td)}));
            runtime::write_json(&path, &report)?;
        }
        Ok(())
    })();
    report["completed"] = json!(outcome.is_ok());
    if let Err(error) = outcome {
        report["error"] = json!(format!("{error:#}"));
    }
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    runtime::write_json(&path, &report)?;
    Ok(report)
}
