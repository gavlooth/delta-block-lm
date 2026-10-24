mod backward_profile;
mod certify;
mod checkpoint;
mod config;
mod data;
mod diagnostics;
mod gemm;
mod gemm_batched;
mod gemm_guard;
mod kron_benchmark;
mod lm_retrieval;
mod model;
mod profile;
mod runtime;
mod soft;
mod train;
use anyhow::{Context, Result, ensure};
use config::Config;
use serde_json::json;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = execute() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
fn execute() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 2,
        "usage: delta-experiment <reference|normalization-reference|kron-reference|hola-reference|kron-model-check|soft|benchmark|profile|backward-profile|gemm-micro|gemm-step|projection-benchmark|normalization-benchmark|kron-benchmark|kron-model-benchmark|hola-benchmark|smoke|train|compare|decision|checkpoint-check|certify> CONFIG.json [--arm NAME] [--seed N] [--resume CHECKPOINT_DIR] [--diagnostic-backend cpu|vulkan] [--checkpoints MANIFEST.json] [--warmups N --trials N] [--contexts 128,512] [--legacy-stress-report FILE.json] [--scan-backend tensor_solver|fused_gpu --tokens N --objective combined|output_only|final_state_only|score_only] [--hola-backend tensor|fused_gpu]"
    );
    let command = &args[0];
    let c = Config::load(Path::new(&args[1]))?;
    let mut arm_name = None;
    let mut seed = c.seed;
    let mut resume: Option<PathBuf> = None;
    let mut diagnostic_backend = None;
    let mut checkpoint_manifest: Option<PathBuf> = None;
    let mut benchmark_warmups = None;
    let mut benchmark_trials = None;
    let mut reference_scan_backend = None;
    let mut reference_hola_backend = None;
    let mut reference_tokens = None;
    let mut reference_objective = None;
    let mut legacy_stress_report: Option<PathBuf> = None;
    let mut profile_contexts: Option<Vec<usize>> = None;
    let mut model_scope: Option<&str> = None;
    let mut index = 2;
    while index < args.len() {
        let flag = &args[index];
        index += 1;
        let value = args.get(index).context("option needs a value")?;
        index += 1;
        match flag.as_str() {
            "--arm" => arm_name = Some(value.as_str()),
            "--seed" => seed = value.parse()?,
            "--resume" => resume = Some(value.into()),
            "--diagnostic-backend" => diagnostic_backend = Some(value.as_str()),
            "--checkpoints" => checkpoint_manifest = Some(value.into()),
            "--warmups" => benchmark_warmups = Some(value.parse::<usize>()?),
            "--trials" => benchmark_trials = Some(value.parse::<usize>()?),
            "--legacy-stress-report" => legacy_stress_report = Some(value.into()),
            "--model-scope" => {
                ensure!(
                    matches!(value.as_str(), "all" | "configured"),
                    "--model-scope must be all or configured"
                );
                model_scope = Some(value.as_str());
            }
            "--contexts" => {
                profile_contexts = Some(
                    value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<Vec<usize>, _>>()?,
                )
            }
            "--scan-backend" => {
                reference_scan_backend = Some(serde_json::from_value::<delta_mixers::ScanBackend>(
                    json!(value),
                )?)
            }
            "--hola-backend" => {
                reference_hola_backend = Some(serde_json::from_value::<delta_mixers::HolaBackend>(
                    json!(value),
                )?)
            }
            "--tokens" => reference_tokens = Some(value.parse::<usize>()?),
            "--objective" => {
                reference_objective = Some(match value.as_str() {
                    "combined" | "0" => 0,
                    "output_only" | "1" => 1,
                    "final_state_only" | "2" => 2,
                    "score_only" | "3" => 3,
                    _ => anyhow::bail!(
                        "objective must be combined/0, output_only/1, final_state_only/2, score_only/3"
                    ),
                })
            }
            _ => anyhow::bail!("unknown option {flag}"),
        }
    }
    ensure!(
        matches!(
            command.as_str(),
            "profile" | "backward-profile" | "projection-benchmark" | "normalization-benchmark"
        ) || profile_contexts.is_none(),
        "--contexts is only valid for profile/backward-profile/projection-benchmark/normalization-benchmark"
    );
    ensure!(
        !matches!(
            command.as_str(),
            "profile" | "backward-profile" | "projection-benchmark" | "normalization-benchmark"
        ) || (resume.is_none() && seed == c.seed),
        "profiles use configured seed and fresh scratch state, not --seed/--resume"
    );
    ensure!(
        command == "kron-model-benchmark" || legacy_stress_report.is_none(),
        "--legacy-stress-report is only valid for kron-model-benchmark"
    );
    ensure!(
        command == "kron-reference"
            || (reference_scan_backend.is_none()
                && reference_tokens.is_none()
                && reference_objective.is_none()),
        "--scan-backend, --tokens and --objective are only valid for kron-reference"
    );
    ensure!(
        command == "hola-reference" || reference_hola_backend.is_none(),
        "--hola-backend is only valid for hola-reference"
    );
    ensure!(
        matches!(
            command.as_str(),
            "kron-benchmark"
                | "kron-model-benchmark"
                | "profile"
                | "backward-profile"
                | "projection-benchmark"
                | "normalization-benchmark"
                | "gemm-micro"
                | "gemm-step"
        ) || benchmark_warmups.is_none(),
        "--warmups is only valid for Kron benchmarks/profile; hola-benchmark uses two warmups per mode"
    );
    ensure!(
        matches!(
            command.as_str(),
            "kron-benchmark"
                | "kron-model-benchmark"
                | "hola-benchmark"
                | "profile"
                | "backward-profile"
                | "projection-benchmark"
                | "normalization-benchmark"
                | "gemm-micro"
                | "gemm-step"
        ) || benchmark_trials.is_none(),
        "--trials is only valid for a benchmark command"
    );
    ensure!(
        !matches!(
            command.as_str(),
            "kron-benchmark" | "kron-model-benchmark" | "hola-benchmark"
        ) || (arm_name.is_none() && resume.is_none() && seed == c.seed),
        "benchmarks use configured/fixed paired fixtures, not --arm/--seed/--resume"
    );
    ensure!(
        !matches!(command.as_str(), "gemm-micro" | "gemm-step")
            || (arm_name.is_none() && resume.is_none() && seed == c.seed),
        "GEMM benchmarks use configured seed, first arm and fresh scratch state"
    );
    ensure!(
        model_scope.is_none() || command == "kron-model-check",
        "--model-scope is only valid for kron-model-check"
    );
    ensure!(
        diagnostic_backend.is_none() || command == "checkpoint-check",
        "--diagnostic-backend is only valid for checkpoint-check"
    );
    ensure!(
        checkpoint_manifest.is_none() || command == "certify",
        "--checkpoints is only valid for certify"
    );
    ensure!(
        !c.autotune
            || !matches!(
                command.as_str(),
                "soft"
                    | "projection-benchmark"
                    | "normalization-benchmark"
                    | "kron-benchmark"
                    | "kron-model-benchmark"
            ),
        "autotuned training certifies the configured fused LM only; this command changes GEMM execution/geometry internally"
    );
    let backend = diagnostic_backend.unwrap_or(&c.backend);
    let mut runtime_config = c.clone();
    runtime_config.backend = backend.to_owned();
    let device = runtime::device(&runtime_config)?;
    let mut gemm_preflight = Vec::new();
    if c.autotune
        && backend == "vulkan"
        && matches!(
            command.as_str(),
            "train"
                | "smoke"
                | "compare"
                | "decision"
                | "benchmark"
                | "profile"
                | "backward-profile"
                | "gemm-step"
                | "checkpoint-check"
                | "certify"
        )
    {
        let contexts = if matches!(command.as_str(), "profile" | "backward-profile") {
            profile_contexts.clone().unwrap_or_else(|| {
                if command == "backward-profile" {
                    vec![128, 512]
                } else {
                    vec![c.context]
                }
            })
        } else {
            vec![c.context]
        };
        for context in contexts {
            let mut checked = c.clone();
            checked.context = context;
            gemm_preflight.push(gemm_guard::ensure(&checked, &device)?);
        }
    }
    let mut result = match command.as_str() {
        "reference" => {
            let report = delta_mixers::numerical_checks(&device.clone().autodiff())
                .map_err(anyhow::Error::msg)?;
            json!({"schema":1,"command":"reference","device":runtime::identity(&device),"config":c,"numerical_report":report})
        }
        "normalization-reference" => {
            let report = delta_mixers::normalization::numerical_checks(&device);
            json!({"schema":1,"command":command,"passed":report.is_ok(),"config":c,
                "device":runtime::identity(&device),"numerical_report":report.as_ref().ok(),"error":report.as_ref().err()})
        }
        "kron-reference" => {
            let selected = reference_scan_backend.unwrap_or(c.scan_backend);
            let report = delta_mixers::checks::kron_numerical_checks_selected(
                &device,
                selected,
                reference_tokens,
                reference_objective,
            );
            json!({"schema":1,"command":"kron-reference","passed":report.is_ok(),"device":runtime::identity(&device),"config":c,"scan_backend":selected,"tokens":reference_tokens,"objective":reference_objective,"numerical_report":report.as_ref().ok(),"error":report.as_ref().err()})
        }
        "hola-reference" => {
            let report = if let Some(selected) = reference_hola_backend {
                delta_mixers::checks::hola_numerical_checks_selected(&device, selected)
            } else {
                delta_mixers::checks::hola_numerical_checks(&device)
            };
            let scope = match reference_hola_backend {
                Some(delta_mixers::HolaBackend::Tensor) => "tensor_float64_reference",
                Some(delta_mixers::HolaBackend::FusedGpu) => "fused_float64_reference",
                None if c.backend == "cpu" => "tensor_float64_reference",
                None => "tensor_and_fused_float64_reference",
            };
            json!({"schema":1,"command":"hola-reference","passed":report.is_ok(),"device":runtime::identity(&device),"config":c,"scope":scope,"hola_backend":reference_hola_backend,"numerical_report":report.as_ref().ok(),"error":report.as_ref().err()})
        }
        "soft" => {
            json!({"schema":1,"command":"soft","device":runtime::identity(&device),"config":c,"feedback":soft::episode(&c,&device)?})
        }
        "benchmark" => train::pilot(&c, &device)?,
        "gemm-micro" => gemm::micro(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(5),
        )?,
        "gemm-step" => gemm::step(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(1),
        )?,
        "profile" | "backward-profile" | "projection-benchmark" | "normalization-benchmark" => {
            let arm = if let Some(name) = arm_name {
                c.arms
                    .iter()
                    .find(|arm| arm.name == name)
                    .context("arm name absent from config")?
            } else {
                &c.arms[0]
            };
            let run = match command.as_str() {
                "profile" => profile::run,
                "backward-profile" => backward_profile::run,
                "projection-benchmark" => profile::projection_run,
                _ => profile::normalization_run,
            };
            let configured_context = [c.context];
            let default_contexts: &[usize] = if command == "backward-profile" {
                &[128, 512]
            } else {
                &configured_context
            };
            run(
                &c,
                arm,
                &device,
                profile_contexts.as_deref().unwrap_or(default_contexts),
                benchmark_warmups.unwrap_or(2),
                benchmark_trials.unwrap_or(5),
            )?
        }
        "kron-benchmark" => kron_benchmark::run(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(5),
        )?,
        "hola-benchmark" => kron_benchmark::hola_run(&c, &device, benchmark_trials.unwrap_or(5))?,
        "kron-model-check" => {
            kron_benchmark::model_check(&c, &device, model_scope == Some("configured"))?
        }
        "kron-model-benchmark" => kron_benchmark::model_run(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(5),
            legacy_stress_report.as_deref(),
        )?,
        "smoke" | "train" => {
            let arm = if let Some(name) = arm_name {
                c.arms
                    .iter()
                    .find(|a| a.name == name)
                    .context("arm name absent from config")?
            } else {
                &c.arms[0]
            };
            train::run(&c, arm, seed, &device, resume.as_deref())?
        }
        "compare" => {
            ensure!(
                resume.is_none(),
                "resume individual arm runs with train --arm --seed --resume"
            );
            train::compare(&c, &device, false)?
        }
        "decision" => {
            ensure!(
                resume.is_none(),
                "resume individual arm runs with train --arm --seed --resume"
            );
            train::compare(&c, &device, true)?
        }
        "checkpoint-check" => {
            diagnostics::checkpoint_check(&c, &device, backend, arm_name, seed, resume.as_deref())?
        }
        "certify" => {
            ensure!(
                arm_name.is_none() && resume.is_none() && seed == c.seed,
                "certify covers all configured seeds/arms from explicit manifest, not --arm/--seed/--resume selectors"
            );
            certify::run(
                &c,
                Path::new(&args[1]),
                &device,
                checkpoint_manifest
                    .as_deref()
                    .context("certify requires --checkpoints MANIFEST.json")?,
            )?
        }
        _ => anyhow::bail!("unknown command {command}"),
    };
    if !gemm_preflight.is_empty() {
        result["gemm_preflight"] = json!(gemm_preflight);
    }
    if !matches!(
        command.as_str(),
        "kron-benchmark"
            | "kron-model-benchmark"
            | "hola-benchmark"
            | "kron-model-check"
            | "profile"
            | "backward-profile"
            | "projection-benchmark"
            | "normalization-benchmark"
            | "gemm-micro"
    ) {
        runtime::write_json(&c.artifact_dir.join(format!("{command}.json")), &result)?;
    }
    println!("{}", serde_json::to_string(&result)?);
    ensure!(
        !matches!(
            command.as_str(),
            "profile"
                | "backward-profile"
                | "projection-benchmark"
                | "normalization-benchmark"
                | "gemm-micro"
                | "gemm-step"
        ) || result["completed"] == true,
        "profile/benchmark failed or hit its deadline; separate partial report retained"
    );
    ensure!(
        !matches!(
            command.as_str(),
            "checkpoint-check"
                | "certify"
                | "kron-reference"
                | "hola-reference"
                | "kron-model-check"
                | "normalization-reference"
                | "gemm-micro"
        ) || result["passed"] == true,
        "numerical/checkpoint diagnostic/certification failed; separate report retained"
    );
    Ok(())
}
