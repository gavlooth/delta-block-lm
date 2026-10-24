//! Independent probes of the small batched contractions in the legacy chunked tensor solver.
use crate::{config::Config, profile::median, runtime};
use anyhow::{Result, ensure};
use burn::{prelude::*, tensor::Distribution};
use serde_json::{Value, json};
use std::time::Instant;

fn input(
    batch: usize,
    heads: usize,
    rows: usize,
    columns: usize,
    transposed: bool,
    view: bool,
    device: &Device,
) -> Tensor<4> {
    let (r, c) = if transposed {
        (columns, rows)
    } else {
        (rows, columns)
    };
    let tensor = if view && r == 16 {
        // Real chunk view from [B,T,H,D], retaining the context512 batch/time strides.
        Tensor::<4>::random(
            [batch, 512, heads, c],
            Distribution::Uniform(-0.1, 0.1),
            device,
        )
        .slice([0..batch, 0..16, 0..heads, 0..c])
        .swap_dims(1, 2)
    } else {
        Tensor::<4>::random(
            [batch, heads, r, c],
            Distribution::Uniform(-0.1, 0.1),
            device,
        )
    };
    if transposed {
        tensor.swap_dims(2, 3)
    } else {
        tensor
    }
}
fn packed(x: Tensor<4>) -> Tensor<4> {
    let shape = x.dims();
    x.reshape([shape.iter().product::<usize>()]).reshape(shape)
}
fn sample(device: &Device, a: &Tensor<4>, b: &Tensor<4>, pack: bool) -> Result<f64> {
    runtime::sync(device)?;
    let start = Instant::now();
    let (a, b) = if pack {
        (packed(a.clone()), packed(b.clone()))
    } else {
        (a.clone(), b.clone())
    };
    let output = a.matmul(b);
    runtime::sync(device)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.;
    std::hint::black_box(output);
    Ok(elapsed)
}

pub fn run(
    c: &Config,
    device: &Device,
    warmups: usize,
    trials: usize,
    started: Instant,
    mut record: impl FnMut(Value) -> Result<()>,
) -> Result<()> {
    for batch in [1, 8, 16] {
        for (name, m, k, n, at, bt, broadcast) in [
            ("solver_k_state", 16, 32, 64, false, false, false),
            ("solver_v_overlap", 16, 64, 16, false, true, false),
            ("solver_inverse_square", 16, 16, 16, false, false, false),
            ("solver_inverse_broadcast", 16, 16, 16, false, false, true),
            ("solver_correction", 16, 16, 1, false, false, false),
            ("solver_local_output", 16, 16, 64, false, false, false),
            ("solver_state_update", 32, 16, 64, true, false, false),
        ] {
            for (phase, m, n, k, at, bt, broadcast) in [
                ("forward", m, n, k, at, bt, broadcast),
                ("input_vjp", m, k, n, false, !bt, false),
                ("weight_vjp", k, n, m, !at, false, false),
            ] {
                for view in [false, true] {
                    ensure!(
                        started.elapsed().as_secs_f64() < c.max_seconds.min(290.),
                        "batched GEMM deadline; partial report retained"
                    );
                    record(benchmark(
                        device, name, batch, phase, m, n, k, at, bt, broadcast, view, warmups,
                        trials,
                    )?)?;
                }
            }
        }
    }
    Ok(())
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
    broadcast: bool,
    view: bool,
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    device.seed(3107);
    let a = input(batch, 4, m, k, at, view, device);
    let b = input(
        if broadcast { 1 } else { batch },
        if broadcast { 1 } else { 4 },
        k,
        n,
        bt,
        view,
        device,
    );
    runtime::sync(device)?;
    let first = sample(device, &a, &b, false)?;
    let first_pack = sample(device, &a, &b, true)?;
    let mut samples = [Vec::with_capacity(trials), Vec::with_capacity(trials)];
    for i in 0..warmups + trials {
        for mode in if i % 2 == 0 { [0, 1] } else { [1, 0] } {
            let ms = sample(device, &a, &b, mode == 1)?;
            if i >= warmups {
                samples[mode].push(ms);
            }
        }
    }
    let dispatches = crate::gemm::dispatches(device, || {
        std::hint::black_box(a.clone().matmul(b.clone()));
    })?;
    let av = a.clone().into_data().try_to_vec::<f32>().unwrap();
    let bv = b.clone().into_data().try_to_vec::<f32>().unwrap();
    let cv = a
        .clone()
        .matmul(b.clone())
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let cp = packed(a)
        .matmul(packed(b))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let mut passed = [true; 2];
    let mut maximum = 0.0f64;
    let mut probes = Vec::with_capacity(4);
    for (row, head, i, j) in [
        (0, 0, 0, 0),
        (batch / 2, 1, m / 2, n / 2),
        (batch - 1, 3, m - 1, n - 1),
        (batch / 3, 2, m / 3, n / 3),
    ] {
        let lane = row * 4 + head;
        let rhs = if broadcast { 0 } else { lane };
        let expected: f64 = (0..k)
            .map(|p| av[(lane * m + i) * k + p] as f64 * bv[(rhs * k + p) * n + j] as f64)
            .sum();
        let actuals = [cv[(lane * m + i) * n + j], cp[(lane * m + i) * n + j]];
        for (mode, value) in actuals.into_iter().enumerate() {
            let error = (value as f64 - expected).abs();
            passed[mode] &= value.is_finite() && error <= 2e-5 + 3e-4 * expected.abs();
            maximum = maximum.max(error);
        }
        probes.push(json!({"batch":row,"head":head,"row":i,"column":j,"expected_f64":expected,"original_f32":actuals[0],"packed_f32":actuals[1]}));
    }
    let flops = 2. * batch as f64 * 4. * m as f64 * n as f64 * k as f64;
    Ok(
        json!({"name":name,"batch":batch,"heads":4,"phase":phase,"m":m,"n":n,"k":k,
        "flops":flops,"chunk_view_parent_context":512,
        "lhs_transposed":at,"rhs_transposed":bt,"rhs_batch_broadcast":broadcast,"chunk_view":view,
        "first_use_ms":first,"first_pack_use_ms":first_pack,"original_samples_ms":samples[0],"pack_samples_ms":samples[1],
        "original_median_ms":median(&samples[0]),"pack_median_ms":median(&samples[1]),
        "original_gflops":flops/median(&samples[0])/1e6,"pack_gflops":flops/median(&samples[1])/1e6,
        "original_passed":passed[0],"pack_passed":passed[1],"scalar_probe_max_abs_error":maximum,
        "scalar_probes":probes,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,"dispatches":dispatches,
        "acceptance_scope":"legacy_tensor_solver_only_not_the_recommended_fused_scan"}),
    )
}
