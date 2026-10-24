//! Actual linear GEMMs, including transposed VJPs, with first-use separated from warm execution.
use crate::{
    config::Config,
    model::LanguageModel,
    profile::{Runner, median},
    runtime, train,
};
use anyhow::{Result, ensure};
use burn::{prelude::*, tensor::Distribution};
use delta_mixers::profile::Unprofiled;
use serde_json::{Value, json};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(feature = "vulkan")]
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

fn validate(c: &Config, warmups: usize, trials: usize) -> Result<()> {
    ensure!(
        c.backend == "vulkan" && !c.fusion,
        "GEMM evaluation requires non-fusion Vulkan"
    );
    ensure!(
        c.task == "lm" && [128, 512].contains(&c.context) && c.d_model == 256 && c.layers == 4,
        "GEMM evaluation uses the actual D256/L4/context128 or512 byte-LM"
    );
    ensure!(
        (1..=5).contains(&warmups) && (1..=31).contains(&trials),
        "warmups 1..=5, trials 1..=31"
    );
    Ok(())
}

#[cfg(feature = "vulkan")]
fn capabilities() -> Value {
    let gpu = cubecl::device::WgpuDevice::default().on(cubecl::device::WgpuBackend::Vulkan);
    let client = cubecl::Device::Wgpu(gpu).client();
    json!({"cubecl_properties":format!("{:#?}", client.properties())})
}
#[cfg(not(feature = "vulkan"))]
fn capabilities() -> Value {
    Value::Null
}

#[cfg(feature = "vulkan")]
pub(super) fn dispatches(device: &Device, run: impl FnOnce()) -> Result<Value> {
    #[derive(Default)]
    struct Count(Mutex<BTreeMap<&'static str, usize>>);
    impl cubecl::logging::LaunchObserver for Count {
        fn launched(&self, name: &'static str) {
            *self.0.lock().unwrap().entry(name).or_default() += 1;
        }
    }
    runtime::sync(device)?;
    let counts = Arc::new(Count::default());
    let guard = cubecl::logging::LaunchObservation::new(counts.clone());
    run();
    runtime::sync(device)?;
    drop(guard);
    let counts = counts.0.lock().unwrap();
    Ok(json!(*counts))
}
#[cfg(not(feature = "vulkan"))]
pub(super) fn dispatches(_device: &Device, _run: impl FnOnce()) -> Result<Value> {
    Ok(Value::Null)
}

fn packed(x: Tensor<2>) -> Tensor<2> {
    let [m, n] = x.dims();
    // Flattening a true transpose requires a contiguous copy; the following reshape is metadata.
    x.reshape([m * n]).reshape([m, n])
}
fn multiply(a: Tensor<2>, b: Tensor<2>, heuristic: bool) -> Tensor<2> {
    if heuristic {
        crate::model::heuristic_matmul(a, b)
    } else {
        a.matmul(b)
    }
}
fn sample(
    device: &Device,
    a: &Tensor<2>,
    b: &Tensor<2>,
    pack: bool,
    heuristic: bool,
) -> Result<f64> {
    runtime::sync(device)?;
    let start = Instant::now();
    let (a, b) = if pack {
        (packed(a.clone()), packed(b.clone()))
    } else {
        (a.clone(), b.clone())
    };
    let out = multiply(a, b, heuristic);
    runtime::sync(device)?;
    let ms = start.elapsed().as_secs_f64() * 1000.;
    std::hint::black_box(out);
    Ok(ms)
}

pub fn micro(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    validate(c, warmups, trials)?;
    let start = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = c.artifact_dir.join(format!(
        "gemm-micro-{}-{nonce}/report.json",
        std::process::id()
    ));
    let mut report = json!({"schema":1,"command":"gemm-micro","completed":false,"config":c,
        "device":runtime::identity(device),"capabilities":capabilities(),"dtype":"f32","report_path":path,
        "warmups":warmups,"trials":trials,"cases":[],"protocol":"resident deterministic inputs; each GEMM synchronized; rotating original/pack-in-clock order; transpose packing and output allocation included; no bias, activation, weight concatenation or autodiff bookkeeping; FLOPs=2*M*N*K; independent scalar f64 probes outside timing; first use precedes warmups; partial report saved per shape"});
    runtime::write_json(&path, &report)?;
    report["legacy_solver_cases"] = json!([]);
    let outcome = (|| -> Result<()> {
        for batch in [1, 8, 16] {
            let rows = batch * c.context;
            for (name, k, n) in [
                ("mlp_up", 256, 512),
                ("mlp_down", 512, 256),
                ("merged_input", 256, 904),
                ("mixer_output", 256, 256),
                (
                    "vocabulary_head",
                    256,
                    if c.head_backend == crate::config::HeadBackend::Padded320 {
                        320
                    } else {
                        257
                    },
                ),
            ] {
                for (phase, m, n, k, at, bt) in [
                    ("forward", rows, n, k, false, false),
                    ("input_vjp", rows, k, n, false, true),
                    ("weight_vjp", k, n, rows, true, false),
                ] {
                    ensure!(
                        start.elapsed().as_secs_f64() < c.max_seconds.min(290.),
                        "gemm-micro deadline; partial report retained"
                    );
                    let heuristic = (name == "vocabulary_head"
                        && c.head_backend == crate::config::HeadBackend::Heuristic)
                        || (name == "mixer_output"
                            && c.projection_backend
                                == delta_mixers::ProjectionBackend::MergedHeuristicOutput);
                    let mut case = benchmark(
                        device, name, batch, phase, m, n, k, at, bt, heuristic, warmups, trials,
                    )?;
                    case["context"] = json!(c.context);
                    report["cases"].as_array_mut().unwrap().push(case);
                    runtime::write_json(&path, &report)?;
                }
            }
        }
        for size in [1024, 2048] {
            ensure!(
                start.elapsed().as_secs_f64() < c.max_seconds.min(290.),
                "gemm-micro deadline; partial report retained"
            );
            report["cases"].as_array_mut().unwrap().push(benchmark(
                device, "square", 0, "forward", size, size, size, false, false, false, warmups,
                trials,
            )?);
            runtime::write_json(&path, &report)?;
        }
        crate::gemm_batched::run(c, device, warmups, trials, start, |case| {
            report["legacy_solver_cases"]
                .as_array_mut()
                .unwrap()
                .push(case);
            runtime::write_json(&path, &report)
        })?;
        Ok(())
    })();
    report["completed"] = json!(outcome.is_ok());
    report["recommended_path_passed"] = json!(
        outcome.is_ok()
            && report["cases"]
                .as_array()
                .unwrap()
                .iter()
                .all(|case| case["original_passed"] == true && case["pack_passed"] == true)
    );
    report["legacy_solver_passed"] = json!(
        outcome.is_ok()
            && report["legacy_solver_cases"]
                .as_array()
                .unwrap()
                .iter()
                .all(|case| case["original_passed"] == true && case["pack_passed"] == true)
    );
    report["passed"] =
        json!(report["recommended_path_passed"] == true && report["legacy_solver_passed"] == true);
    if let Err(error) = outcome {
        report["error"] = json!(format!("{error:#}"));
    }
    report["elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    report["sampled_reserved_bytes"] = json!(runtime::sampled_reserved(device));
    runtime::write_json(&path, &report)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn benchmark(
    device: &Device,
    name: &str,
    batch: usize,
    phase: &str,
    m: usize,
    n: usize,
    k: usize,
    at: bool,
    bt: bool,
    heuristic: bool,
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    device.seed(3107);
    let a = Tensor::<2>::random(
        if at { [k, m] } else { [m, k] },
        Distribution::Uniform(-0.1, 0.1),
        device,
    );
    let b = Tensor::<2>::random(
        if bt { [n, k] } else { [k, n] },
        Distribution::Uniform(-0.1, 0.1),
        device,
    );
    let a = if at { a.swap_dims(0, 1) } else { a };
    let b = if bt { b.swap_dims(0, 1) } else { b };
    runtime::sync(device)?;
    let first_use = sample(device, &a, &b, false, heuristic)?;
    let first_pack = sample(device, &a, &b, true, heuristic)?;
    let mut original = Vec::with_capacity(trials);
    let mut pack = Vec::with_capacity(trials);
    for i in 0..warmups + trials {
        let mut times = [0.; 2];
        for mode in if i % 2 == 0 { [0, 1] } else { [1, 0] } {
            times[mode] = sample(device, &a, &b, mode == 1, heuristic)?;
        }
        if i >= warmups {
            original.push(times[0]);
            pack.push(times[1]);
        }
    }
    let kernels = dispatches(device, || {
        std::hint::black_box(multiply(a.clone(), b.clone(), heuristic));
    })?;
    let av = a.clone().into_data().try_to_vec::<f32>().unwrap();
    let bv = b.clone().into_data().try_to_vec::<f32>().unwrap();
    let cv = multiply(a.clone(), b.clone(), heuristic)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let cp = multiply(packed(a), packed(b), heuristic)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let mut error = 0.0f64;
    let mut passed = [true; 2];
    let mut probes = Vec::with_capacity(4);
    for (i, j) in [(0, 0), (m / 2, n / 2), (m - 1, n - 1), (m / 3, n / 3)] {
        let expected: f64 = (0..k)
            .map(|p| av[i * k + p] as f64 * bv[p * n + j] as f64)
            .sum();
        let actuals = [cv[i * n + j], cp[i * n + j]];
        for (mode, actual) in actuals.into_iter().enumerate() {
            let diff = (actual as f64 - expected).abs();
            passed[mode] &= actual.is_finite() && diff <= 2e-5 + 3e-4 * expected.abs();
            error = error.max(diff);
        }
        probes.push(json!({"row":i,"column":j,"expected_f64":expected,
            "original_f32":actuals[0],"packed_f32":actuals[1]}));
    }
    let flops = 2. * m as f64 * n as f64 * k as f64;
    Ok(
        json!({"name":name,"batch":batch,"phase":phase,"m":m,"n":n,"k":k,
        "lhs_transposed":at,"rhs_transposed":bt,"flops":flops,"first_use_ms":first_use,"first_pack_use_ms":first_pack,
        "forced_heuristic":heuristic,
        "original_samples_ms":original,"original_median_ms":median(&original),"original_gflops":flops/median(&original)/1e6,
        "pack_samples_ms":pack,"pack_median_ms":median(&pack),"pack_gflops":flops/median(&pack)/1e6,
        "dispatches":kernels,"scalar_probe_max_abs_error":error,"logical_operand_output_bytes":4*(m*k+k*n+m*n),
        "original_passed":passed[0],"pack_passed":passed[1],"scalar_probes":probes,
        "absolute_tolerance":2e-5,"relative_tolerance":3e-4,
        "packing_extra_bytes":if at {4*m*k} else if bt {4*k*n} else {0}}),
    )
}

/// One fresh, seed-identical trajectory per process, for cross-feature AB/BA pairing.
pub fn step(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    validate(c, warmups, trials)?;
    ensure!(
        [1, 8].contains(&c.batch_size),
        "paired acceptance batches are 1 and 8"
    );
    ensure!(c.context == 512, "paired acceptance context is512");
    let start = Instant::now();
    let corpus = train::loaded_corpus(c)?;
    let train_device = device.clone().autodiff();
    train_device.seed(c.seed);
    let base = LanguageModel::new(c, &c.arms[0], &train_device)?;
    let mut runner = Runner::new(base, c);
    let first = runner.step(c, corpus.as_ref(), &train_device, &mut Unprofiled)?;
    let mut samples = Vec::with_capacity(trials);
    for i in 0..warmups + trials {
        ensure!(
            start.elapsed().as_secs_f64() < c.max_seconds.min(290.),
            "gemm-step deadline"
        );
        let ms = runner.step(c, corpus.as_ref(), &train_device, &mut Unprofiled)?;
        if i >= warmups {
            samples.push(ms);
        }
    }
    Ok(
        json!({"schema":1,"command":"gemm-step","completed":true,"config":c,"device":runtime::identity(device),
        "first_use_step_ms":first,"samples_ms":samples,"median_ms":median(&samples),"warmups":warmups,
        "sampled_reserved_bytes":runtime::sampled_reserved(device),"elapsed_seconds":start.elapsed().as_secs_f64()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transpose_pack_preserves_logical_values() {
        let device = Device::ndarray();
        let x = Tensor::<2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device).swap_dims(0, 1);
        assert_eq!(x.clone().into_data(), packed(x).into_data());
    }
}
