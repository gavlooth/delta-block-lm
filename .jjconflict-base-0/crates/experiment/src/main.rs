mod backward_profile;
mod certify;
mod checkpoint;
mod config;
mod codpula;
mod codpula_reference;
mod codpula_model_checks;
mod cocommit;
mod cocommit_checks;
mod data;
mod diagnostics;
mod epilogues;
mod gemm;
mod gemm_batched;
mod gemm_guard;
mod hola_tune;
mod io_benchmark;
mod kron_benchmark;
mod lm_retrieval;
mod lngram;
mod model;
mod mask_energy;
mod remask;
mod tct;
mod ohc;
mod profile;
mod prefix_retrieval;
mod runtime;
mod scan_benchmark;
mod sbd;
mod sbd_step;
mod sbd_training;
mod sbd_read_reference;
mod sbd_benchmark;
mod sbd_batch_reference;
mod soft;
mod soft_benchmark;
mod train;
mod update;
mod update_loss;
mod wavelet;
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
        "usage: delta-experiment <cocommit-reference|cocommit-benchmark|cocommit-diagnostic|remask-reference|remask-generate|remask-collect|energy-fit|remask-benchmark|tct-reference|tct-train|tct-benchmark|codpula-reference|codpula-model-reference|codpula-benchmark|codpula-diagnostic|prefix-retrieval-reference|prefix-retrieval-benchmark|prefix-retrieval-sbd-diagnostic|prefix-retrieval-sbd-benchmark|sbd-reference|sbd-batching-reference|sbd-batching-benchmark|sbd-batching-diagnostic|sbd-diagnostic|sbd-benchmark|sbd-train|sbd-generate|sbd-evaluate|reference|lngram-reference|lngram-benchmark|lngram-attribution|lngram-smoke|wavelet-reference|wavelet-benchmark|wavelet-smoke|ohc-reference|ohc-benchmark|ohc-learning-smoke|convolution-reference|normalization-reference|epilogue-reference|epilogue-benchmark|kron-reference|gdn-reference|gdn-directional-diagnostic|hola-reference|kron-model-check|soft|soft-reference|soft-benchmark|soft-step|update-loss-reference|update-loss-benchmark|benchmark|profile|backward-profile|gemm-micro|gemm-step|projection-benchmark|convolution-benchmark|normalization-benchmark|optimization-benchmark|scan-benchmark|hola-cost|hola-tune|kron-benchmark|gdn-benchmark|kron-model-benchmark|hola-benchmark|io-benchmark|smoke|train|compare|decision|checkpoint-check|certify> CONFIG.json [--arm NAME] [--seed N] [--resume CHECKPOINT_DIR] [--diagnostic-backend cpu|vulkan --diagnostic-output DIR] [--checkpoints MANIFEST.json] [--warmups N --trials N] [--contexts 128,512] [--legacy-stress-report FILE.json] [--scan-backend tensor_solver|fused_gpu|fused_parallel128|fused_parallel256|fused_gdn2 --tokens N --objective combined|output_only|final_state_only|score_only] [--hola-backend tensor|fused_gpu|fused_gpu_pairs|fused_gpu_pairs_norms] [--model-scope all|configured]"
    );
    let command = &args[0];
    let c = Config::load(Path::new(&args[1]))?;
    let mut arm_name = None;
    let mut seed = c.seed;
    let mut resume: Option<PathBuf> = None;
    let mut diagnostic_backend = None;
    let mut diagnostic_output: Option<PathBuf> = None;
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
            "--diagnostic-output" => diagnostic_output = Some(value.into()),
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
            "profile"
                | "backward-profile"
                | "projection-benchmark"
                | "normalization-benchmark"
                | "epilogue-benchmark"
                | "hola-tune"
                | "scan-benchmark"
                | "convolution-benchmark"
                | "optimization-benchmark"
        ) || profile_contexts.is_none(),
        "--contexts is only valid for profile/backward-profile/projection-benchmark/normalization-benchmark/hola-tune/scan-benchmark/convolution-benchmark/optimization-benchmark"
    );
    ensure!(
        !matches!(
            command.as_str(),
            "profile"
                | "backward-profile"
                | "projection-benchmark"
                | "normalization-benchmark"
                | "epilogue-benchmark"
                | "hola-tune"
                | "scan-benchmark"
                | "convolution-benchmark"
                | "optimization-benchmark"
        ) || (resume.is_none() && seed == c.seed),
        "profiles use configured seed and fresh scratch state, not --seed/--resume"
    );
    ensure!(
        command == "kron-model-benchmark" || legacy_stress_report.is_none(),
        "--legacy-stress-report is only valid for kron-model-benchmark"
    );
    ensure!(
        matches!(command.as_str(), "kron-reference" | "gdn-reference")
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
        command == "checkpoint-check" || diagnostic_output.is_none(),
        "--diagnostic-output is only valid for checkpoint-check"
    );
    ensure!(
        matches!(
            command.as_str(),
            "kron-benchmark"
                | "kron-model-benchmark"
                | "gdn-benchmark"
                | "profile"
                | "backward-profile"
                | "projection-benchmark"
                | "normalization-benchmark"
                | "epilogue-benchmark"
                | "wavelet-benchmark"
                | "hola-tune"
                | "scan-benchmark"
                | "convolution-benchmark"
                | "gemm-micro"
                | "gemm-step"
                | "update-loss-benchmark"
                | "soft-benchmark"
                | "soft-step"
                | "optimization-benchmark"
                | "io-benchmark"
                | "ohc-benchmark"
                | "lngram-benchmark"
                | "lngram-attribution"
                | "sbd-benchmark"
                | "sbd-batching-benchmark"
                | "codpula-benchmark"
                | "remask-benchmark"
                | "cocommit-benchmark"
                | "prefix-retrieval-benchmark"
                | "prefix-retrieval-sbd-benchmark"
        ) || benchmark_warmups.is_none(),
        "--warmups is only valid for Kron benchmarks/profile; hola-benchmark uses two warmups per mode"
    );
    ensure!(
        matches!(
            command.as_str(),
            "kron-benchmark"
                | "kron-model-benchmark"
                | "gdn-benchmark"
                | "hola-benchmark"
                | "profile"
                | "backward-profile"
                | "projection-benchmark"
                | "normalization-benchmark"
                | "hola-tune"
                | "epilogue-benchmark"
                | "wavelet-benchmark"
                | "scan-benchmark"
                | "convolution-benchmark"
                | "gemm-micro"
                | "gemm-step"
                | "update-loss-benchmark"
                | "soft-benchmark"
                | "soft-step"
                | "optimization-benchmark"
                | "io-benchmark"
                | "ohc-benchmark"
                | "lngram-benchmark"
                | "lngram-attribution"
                | "sbd-benchmark"
                | "sbd-batching-benchmark"
                | "remask-benchmark"
                | "codpula-benchmark"
                | "cocommit-benchmark"
                | "prefix-retrieval-benchmark"
                | "prefix-retrieval-sbd-benchmark"
        ) || benchmark_trials.is_none(),
        "--trials is only valid for a benchmark command"
    );
    ensure!(
        !matches!(
            command.as_str(),
            "kron-benchmark" | "kron-model-benchmark" | "hola-benchmark" | "gdn-benchmark"
        ) || (arm_name.is_none() && resume.is_none() && seed == c.seed),
        "benchmarks use configured/fixed paired fixtures, not --arm/--seed/--resume"
    );
    ensure!(
        !matches!(
            command.as_str(),
            "gemm-micro"
                | "gemm-step"
                | "update-loss-benchmark"
                | "update-loss-reference"
                | "gdn-directional-diagnostic"
                | "ohc-reference"
                | "ohc-benchmark"
                | "ohc-learning-smoke"
                | "ohc-attribution"
        ) || (arm_name.is_none()
            && (resume.is_none() || command == "update-loss-reference")
            && seed == c.seed),
        "benchmarks/references use configured seed and first arm; only update-loss-reference accepts an optional historical checkpoint"
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
        diagnostic_output.is_none() || command == "checkpoint-check",
        "--diagnostic-output is only valid for checkpoint-check"
    );
    ensure!(
        checkpoint_manifest.is_none() || command == "certify",
        "--checkpoints is only valid for certify"
    );
    ensure!(
        command != "io-benchmark" || (arm_name.is_none() && resume.is_none() && seed == c.seed),
        "IO benchmark uses the first configured arm, seed and fresh scratch state"
    );
    ensure!(
        !c.autotune
            || !matches!(
                command.as_str(),
                "soft"
                    | "projection-benchmark"
                    | "normalization-benchmark"
                    | "epilogue-benchmark"
                    | "optimization-benchmark"
                    | "convolution-benchmark"
                    | "hola-tune"
                    | "scan-benchmark"
                    | "update-loss-benchmark"
                    | "soft-benchmark"
                    | "soft-step"
                    | "io-benchmark"
                    | "kron-benchmark"
                    | "kron-model-benchmark"
                    | "gdn-benchmark"
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
        "remask-reference" => {
            let mut report = mask_energy::numerical_checks(&device)?;
            report["runtime"] = remask::numerical_checks(&c,&device)?;
            report
        },
        "tct-reference" => tct::numerical_checks(&device)?,
        "energy-fit" => {
            ensure!(resume.is_none() && arm_name.is_none() && seed == c.seed,
                "energy-fit starts a fresh separate fit with its configured seed");
            mask_energy::fit(&c,&device)?
        },
        "remask-generate" | "remask-collect" => {
            let arm = c.arms.iter().find(|a|arm_name.is_none_or(|name|a.name==name)).context("arm name absent from config")?;
            remask::checkpoint_report(&c,arm,seed,&device,resume.as_deref().context("remasking requires --resume CHECKPOINT")?,command=="remask-collect")?
        },
        "tct-train" => {
            let arm = c.arms.iter().find(|a|arm_name.is_none_or(|name|a.name==name)).context("arm name absent from config")?;
            tct::run(&c,arm,seed,&device,resume.as_deref())?
        },
        "tct-benchmark" => tct::benchmark(&c,&c.arms[0],seed,&device)?,
        "remask-benchmark" => remask::benchmark(&c,&c.arms[0],seed,&device,benchmark_warmups.unwrap_or(2),benchmark_trials.unwrap_or(11))?,
        "cocommit-reference" => cocommit_checks::run(&c,&device)?,
        "cocommit-benchmark" => sbd_benchmark::cocommit(&c,&device,benchmark_warmups.unwrap_or(2),benchmark_trials.unwrap_or(11))?,
        "cocommit-diagnostic" => sbd_benchmark::batching_diagnostic(&c,&device)?,
        "codpula-reference" => codpula_reference::numerical_checks(&device)?,
        "codpula-model-reference" => codpula_model_checks::run(&c,&device)?,
        "codpula-continuation-reference" => codpula_model_checks::continuation_reference(&c,&device)?,
        "codpula-benchmark" => sbd_benchmark::codpula(&c,&device,benchmark_warmups.unwrap_or(2),benchmark_trials.unwrap_or(11))?,
        "codpula-diagnostic" => sbd_benchmark::batching_diagnostic(&c,&device)?,
        "sbd-reference" => {
            let mut report=sbd::reference(&c,&device)?;
            report["block_execution"]=sbd_batch_reference::run(&c,&device)?;
            report
        },
        "sbd-batching-reference" => sbd_batch_reference::run(&c,&device)?,
        "sbd-batching-benchmark" => sbd_benchmark::batching(&c,&device,benchmark_warmups.unwrap_or(2),benchmark_trials.unwrap_or(11))?,
        "sbd-batching-diagnostic" => sbd_benchmark::batching_diagnostic(&c,&device)?,
        "sbd-benchmark" => sbd_benchmark::run(&c,&device,benchmark_warmups.unwrap_or(2),benchmark_trials.unwrap_or(11))?,
        "sbd-diagnostic" => sbd_benchmark::diagnostic(&c,&device)?,
        "sbd-generate" | "sbd-evaluate" => {
            let arm = c.arms.iter().find(|a| arm_name.is_none_or(|name| a.name == name)).context("arm name absent from config")?;
            sbd::checkpoint_report(&c,arm,seed,&device,resume.as_deref().context("sbd-generate/evaluate requires --resume CHECKPOINT")?,command == "sbd-generate")?
        }
        "lngram-reference" => lngram::reference(&c, &device)?,
        "lngram-benchmark" => lngram::benchmark(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(11),
        )?,
        "lngram-attribution" => lngram::attribution(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(11),
        )?,
        "lngram-smoke" => lngram::learning_smoke(&c, &device)?,
        "ohc-reference" => {
            let report = ohc::reference(&c, &device);
            json!({"schema":1,"command":command,"passed":report.is_ok(),"config":c,"device":runtime::identity(&device),
                "numerical_report":report.as_ref().ok(),"error":report.as_ref().err().map(|e|format!("{e:#}"))})
        }
        "ohc-benchmark" => ohc::benchmark(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(11),
        )?,
        "ohc-learning-smoke" => ohc::learning_smoke(&c, &device)?,
        "ohc-attribution" => ohc::attribution(&c, &device)?,
        "wavelet-reference" => {
            let report = wavelet::reference(&c, &device);
            json!({"schema":1,"command":command,"passed":report.is_ok(),"config":c,
                "device":runtime::identity(&device),"numerical_report":report.as_ref().ok(),
                "error":report.as_ref().err().map(|e|format!("{e:#}"))})
        }
        "wavelet-benchmark" => wavelet::benchmark(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(11),
        )?,
        "wavelet-smoke" => wavelet::smoke(&c, &device)?,
        "io-benchmark" => io_benchmark::run(
            &c,
            &c.arms[0],
            &device,
            benchmark_warmups.unwrap_or(1),
            benchmark_trials.unwrap_or(3),
        )?,
        "reference" => {
            let report = delta_mixers::numerical_checks(&device.clone().autodiff())
                .map_err(anyhow::Error::msg)?;
            json!({"schema":1,"command":"reference","device":runtime::identity(&device),"config":c,"numerical_report":report})
        }
        "convolution-reference" => {
            let report = delta_mixers::convolution::numerical_checks(&device);
            json!({"schema":1,"command":command,"passed":report.is_ok(),"config":c,
                "device":runtime::identity(&device),"numerical_report":report.as_ref().ok(),"error":report.as_ref().err()})
        }
        "normalization-reference" => {
            let report = delta_mixers::normalization::numerical_checks(&device);
            json!({"schema":1,"command":command,"passed":report.is_ok(),"config":c,
                "device":runtime::identity(&device),"numerical_report":report.as_ref().ok(),"error":report.as_ref().err()})
        }
        "epilogue-reference" => {
            let report = epilogues::numerical_checks(&device);
            json!({"schema":1,"command":command,"passed":report.is_ok(),"config":c,
                "device":runtime::identity(&device),"numerical_report":report.as_ref().ok(),
                "error":report.as_ref().err().map(|e| format!("{e:#}"))})
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
        "gdn-reference" => {
            let selected = reference_scan_backend.unwrap_or(c.scan_backend);
            let report = delta_mixers::checks::gdn_numerical_checks_selected(
                &device,
                selected,
                reference_tokens,
                reference_objective,
            );
            json!({"schema":1,"command":command,"passed":report.is_ok(),"device":runtime::identity(&device),"config":c,"scan_backend":selected,"tokens":reference_tokens,"objective":reference_objective,"numerical_report":report.as_ref().ok(),"error":report.as_ref().err()})
        }
        "gdn-benchmark" => kron_benchmark::gdn_run(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(5),
        )?,
        "gdn-directional-diagnostic" => kron_benchmark::directional_diagnostic(&c, &device)?,
        "prefix-retrieval-benchmark" => prefix_retrieval::benchmark(&c,&device,
            benchmark_warmups.unwrap_or(2),benchmark_trials.unwrap_or(11))?,
        "prefix-retrieval-sbd-diagnostic" => prefix_retrieval::sbd_diagnostic(&c,&device)?,
        "prefix-retrieval-sbd-benchmark" => prefix_retrieval::sbd_benchmark(&c,&device,
            benchmark_warmups.unwrap_or(2),benchmark_trials.unwrap_or(11))?,
        "prefix-retrieval-reference" => {
            let report = delta_mixers::prefix_checks::reference(&device);
            json!({"schema":1,"command":command,"passed":report.is_ok(),"config":c,
                "device":runtime::identity(&device),"absolute_tolerance":2e-5,"relative_tolerance":3e-4,
                "semantics":"moba_own_block_and_topk_earlier_blocks","numerical_report":report.as_ref().ok(),"error":report.as_ref().err()})
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
                Some(
                    delta_mixers::HolaBackend::FusedGpuPairs
                    | delta_mixers::HolaBackend::FusedGpuPairsNorms,
                ) => "sparse_pairs_float64_reference",
                None if c.backend == "cpu" => "tensor_float64_reference",
                None => "tensor_and_fused_float64_reference",
            };
            json!({"schema":1,"command":"hola-reference","passed":report.is_ok(),"device":runtime::identity(&device),"config":c,"scope":scope,"hola_backend":reference_hola_backend,"numerical_report":report.as_ref().ok(),"error":report.as_ref().err()})
        }
        "update-loss-reference" => {
            let report = update_loss::reference(&c, &device, resume.as_deref());
            json!({"schema":1,"command":command,"passed":report.is_ok(),"config":c,"device":runtime::identity(&device),"numerical_report":report.as_ref().ok(),"error":report.as_ref().err().map(|e|format!("{e:#}"))})
        }
        "update-loss-benchmark" => update_loss::benchmark(
            &c,
            &device,
            benchmark_warmups.unwrap_or(2),
            benchmark_trials.unwrap_or(9),
        )?,
        "soft" => {
            json!({"schema":1,"command":"soft","device":runtime::identity(&device),"config":c,"feedback":soft::episode(&c,&device)?})
        }
        "soft-reference" => {
            json!({"command":command,"passed":true,"device":runtime::identity(&device),"checks":soft_benchmark::reference(&c,&device)?})
        }
        "soft-benchmark" => soft_benchmark::run(
            &c,
            &device,
            benchmark_warmups.unwrap_or(1),
            benchmark_trials.unwrap_or(3),
        )?,
        "soft-step" => soft_benchmark::step(
            &c,
            &device,
            benchmark_warmups.unwrap_or(1),
            benchmark_trials.unwrap_or(5),
        )?,
        "benchmark" => train::pilot(&c, &device)?,
        "hola-cost" => hola_tune::cost(&c, &c.arms[0], &device, &[512], 2, 5)?,
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
        "profile"
        | "backward-profile"
        | "projection-benchmark"
        | "normalization-benchmark"
        | "epilogue-benchmark"
        | "hola-tune"
        | "scan-benchmark"
        | "convolution-benchmark"
        | "optimization-benchmark" => {
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
                "normalization-benchmark" => profile::normalization_run,
                "epilogue-benchmark" => profile::epilogue_run,
                "hola-tune" => profile::hola_run,
                "scan-benchmark" => scan_benchmark::run,
                "convolution-benchmark" => profile::convolution_run,
                "optimization-benchmark" => profile::optimization_run,
                _ => unreachable!(),
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
        "smoke" | "train" | "sbd-train" => {
            ensure!(command != "sbd-train" || c.sbd.is_some(), "sbd-train requires sbd config");
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
        "checkpoint-check" => diagnostics::checkpoint_check(
            &c,
            &device,
            backend,
            arm_name,
            seed,
            resume.as_deref(),
            diagnostic_output.as_deref(),
        )?,
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
            | "epilogue-benchmark"
            | "hola-tune"
            | "scan-benchmark"
            | "convolution-benchmark"
            | "optimization-benchmark"
            | "gemm-micro"
    ) {
        let output = diagnostic_output.as_deref().unwrap_or(&c.artifact_dir);
        runtime::write_json(&output.join(format!("{command}.json")), &result)?;
    }
    println!("{}", serde_json::to_string(&result)?);
    ensure!(
        !matches!(
            command.as_str(),
            "profile"
                | "backward-profile"
                | "projection-benchmark"
                | "normalization-benchmark"
                | "epilogue-benchmark"
                | "hola-tune"
                | "scan-benchmark"
                | "convolution-benchmark"
                | "gemm-micro"
                | "gemm-step"
                | "update-loss-benchmark"
                | "optimization-benchmark"
                | "soft-benchmark"
                | "soft-step"
                | "io-benchmark"
                | "gdn-benchmark"
                | "gdn-directional-diagnostic"
                | "ohc-benchmark"
                | "lngram-benchmark"
                | "lngram-attribution"
                | "sbd-benchmark"
                | "sbd-diagnostic"
                | "sbd-batching-benchmark"
                | "sbd-batching-diagnostic"
                | "codpula-benchmark"
                | "codpula-diagnostic"
                | "cocommit-benchmark"
                | "cocommit-diagnostic"
                | "prefix-retrieval-benchmark"
                | "prefix-retrieval-sbd-diagnostic"
                | "prefix-retrieval-sbd-benchmark"
        ) || result["completed"] == true,
        "profile/benchmark failed or hit its deadline; separate partial report retained"
    );
    ensure!(
        !matches!(
            command.as_str(),
            "checkpoint-check"
                | "certify"
                | "sbd-reference"
                | "sbd-batching-reference"
                | "codpula-reference"
                | "codpula-model-reference"
                | "cocommit-reference"
                | "kron-reference"
                | "gdn-reference"
                | "hola-reference"
                | "prefix-retrieval-reference"
                | "kron-model-check"
                | "normalization-reference"
                | "epilogue-reference"
                | "convolution-reference"
                | "wavelet-reference"
                | "gemm-micro"
                | "update-loss-reference"
                | "soft-reference"
                | "lngram-reference"
                | "lngram-smoke"
                | "ohc-reference"
                | "ohc-learning-smoke"
        ) || result["passed"] == true,
        "numerical/checkpoint diagnostic/certification failed; separate report retained"
    );
    Ok(())
}
