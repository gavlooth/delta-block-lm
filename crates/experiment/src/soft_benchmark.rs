//! Bounded selection/feedback evaluation. This does not implement block diffusion training.
use crate::{
    config::{Config, SoftSelectionBackend as Path},
    profile::median,
    runtime,
    soft::{self, Candidates, SoftConfidence, SoftState, ids_host},
};
use anyhow::{Result, ensure};
use burn::{
    prelude::*,
    tensor::{Distribution, activation::softmax},
};
use serde_json::{Value, json};
use std::time::Instant;

fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().expect("f32 soft fixture")
}
fn close(actual: &[f32], expected: &[f64], name: &str) -> Result<f64> {
    ensure!(actual.len() == expected.len(), "{name} shape");
    let mut error = 0f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let delta = (a as f64 - e).abs();
        ensure!(
            a.is_finite() && delta <= 2e-5 + 3e-4 * e.abs(),
            "{name}[{i}]: {a} != {e}"
        );
        error = error.max(delta);
    }
    Ok(error)
}
fn state_equal(a: &SoftState, b: &SoftState) -> Result<()> {
    ensure!(
        ids_host(a.candidates.ids.clone()) == ids_host(b.candidates.ids.clone()),
        "state IDs"
    );
    ensure!(
        ids_host(a.provisional.clone()) == ids_host(b.provisional.clone()),
        "provisional IDs"
    );
    ensure!(
        ids_host(a.retained.clone()) == ids_host(b.retained.clone()),
        "retained IDs"
    );
    for (name, x, y) in [
        (
            "weights",
            a.candidates.weights.clone(),
            b.candidates.weights.clone(),
        ),
        (
            "mass",
            a.candidates.retained_mass.clone(),
            b.candidates.retained_mass.clone(),
        ),
        ("visibility", a.visibility.clone(), b.visibility.clone()),
    ] {
        close(
            &values(x),
            &values(y).into_iter().map(f64::from).collect::<Vec<_>>(),
            name,
        )?;
    }
    Ok(())
}
pub fn reference(c: &Config, device: &Device) -> Result<Value> {
    let device = device.clone().autodiff();
    let confidence = SoftConfidence::new(&device);
    let mut cases = Vec::new();
    for path in [Path::Cpu, Path::Gpu, Path::GpuTiled] {
        if c.backend == "cpu" && !path.is_cpu() {
            continue;
        }
        let mut selected_config = c.clone();
        selected_config.soft_selection_backend = path;
        let episode = soft::episode(&selected_config, &device)?;
        for (vocab, width, strided) in [
            (7, 1, false),
            (7, 4, true),
            (7, 64, false),
            (257, 16, true),
            (257, 64, false),
            (32000, 4, true),
            (32000, 64, false),
        ] {
            let rows = 4;
            let d = 3;
            let k = width.min(vocab);
            let mut pv = (0..rows * vocab)
                .map(|i| ((i * 37 + 11) % 23) as f32 / (23 * vocab) as f32)
                .collect::<Vec<_>>();
            pv[..vocab].fill(0.0);
            pv[0] = -0.0;
            pv[1] = 0.0;
            // A zero row, repeated ties, a padded row, and an ordinary tail row.
            let p = if strided {
                let mut transposed = vec![0f32; pv.len()];
                for r in 0..rows {
                    for j in 0..vocab {
                        transposed[j * rows + r] = pv[r * vocab + j];
                    }
                }
                Tensor::<2>::from_data(TensorData::new(transposed, [vocab, rows]), &device)
                    .transpose()
            } else {
                Tensor::from_data(TensorData::new(pv.clone(), [rows, vocab]), &device)
            }
            .require_grad();
            let ev = (0..vocab * d)
                .map(|i| ((i * 13) % 19) as f32 / 19.0 - 0.5)
                .collect::<Vec<_>>();
            let table = Tensor::<2>::from_data(TensorData::new(ev.clone(), [vocab, d]), &device)
                .require_grad();
            let mv = [0.2f32, -0.1, 0.3];
            let mask = Tensor::from_data(mv, &device).require_grad();
            let rho = Tensor::from_data([[0.0f32], [0.25], [1.0], [0.75]], &device).require_grad();
            let active = Tensor::from_data([[1.0f32], [1.0], [0.0], [1.0]], &device);
            let selected = Candidates::select(p.clone(), width, path)?;
            let cpu = Candidates::select(p.clone(), width, Path::Cpu)?;
            let mut oracle_ids = Vec::new();
            for row in pv.chunks_exact(vocab) {
                let mut order = (0..vocab).collect::<Vec<_>>();
                order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
                oracle_ids.extend(order[..k].iter().map(|&i| i as i32));
            }
            ensure!(
                ids_host(selected.ids.clone()) == oracle_ids && ids_host(cpu.ids) == oracle_ids,
                "stable total-order IDs {path:?} {vocab}/{width}"
            );
            let mut mass = vec![0f64; rows];
            let mut output = vec![0f64; rows * d];
            let mut dp = vec![0f64; rows * vocab];
            let mut de = vec![0f64; vocab * d];
            let mut dm = vec![0f64; d];
            let mut dr = vec![0f64; rows];
            let rv = [0.0f64, 0.25, 0.0, 0.75];
            for r in 0..rows {
                for &id in &oracle_ids[r * k..(r + 1) * k] {
                    mass[r] += pv[r * vocab + id as usize] as f64;
                }
                if r == 2 {
                    continue;
                }
                let lambda = if mass[r] > 0.0 { rv[r] } else { 0.0 };
                let mut average = vec![0.0f64; d];
                if lambda != 0.0 {
                    for &id in &oracle_ids[r*k..(r+1)*k] {
                        let id = id as usize;
                        let q = f64::from(pv[r*vocab+id])/mass[r];
                        for j in 0..d {
                            average[j] += q*f64::from(ev[id*d+j]);
                            de[id*d+j] += lambda*q;
                        }
                    }
                }
                for j in 0..d {
                    output[r*d+j] = (1.0-lambda)*f64::from(mv[j])+lambda*average[j];
                    dm[j] += 1.0-lambda;
                }
                if lambda != 0.0 {
                    for &id in &oracle_ids[r*k..(r+1)*k] {
                        let id = id as usize;
                        dp[r*vocab+id] = lambda/mass[r]
                            *(0..d).map(|j|f64::from(ev[id*d+j])-average[j]).sum::<f64>();
                    }
                    dr[r] = (0..d).map(|j|average[j]-f64::from(mv[j])).sum::<f64>();
                }
            }
            for r in 0..rows {
                for &id in &oracle_ids[r * k..(r + 1) * k] {
                    dp[r * vocab + id as usize] += 0.125;
                }
            }
            let out = selected.expected_masked(
                table.clone(),
                mask.clone(),
                rho.clone(),
                active.clone(),
                false,
            );
            let dense = soft::dense_expected_masked(
                selected.dense(vocab),
                table.clone(),
                mask.clone(),
                rho.clone(),
                active,
            );
            let mut maximum = close(&values(selected.retained_mass.clone()), &mass, "mass")?;
            maximum = maximum.max(close(&values(out.clone()), &output, "f64 expectation")?);
            close(&values(dense), &output, "dense expectation")?;
            let grads = (out.sum() + selected.retained_mass.clone().sum() * 0.125).backward();
            for (name, got, expected) in [
                (
                    "probability VJP",
                    values(p.grad(&grads).expect("p VJP")),
                    dp,
                ),
                (
                    "embedding VJP",
                    values(table.grad(&grads).expect("embedding VJP")),
                    de,
                ),
                ("mask VJP", values(mask.grad(&grads).expect("mask VJP")), dm),
                (
                    "visibility VJP",
                    values(rho.grad(&grads).expect("rho VJP")),
                    dr,
                ),
            ] {
                maximum = maximum.max(close(&got, &expected, name)?);
            }
            cases.push(json!({"path":path,"vocab":vocab,"width":width,"strided":strided,"max_abs_error":maximum}));
        }
        for invalid in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.1,
            f32::from_bits(0x80000001),
            f32::from_bits(0xffc00001),
        ] {
            ensure!(
                Candidates::select(Tensor::from_data([[0.2f32, invalid]], &device), 1, path)
                    .is_err(),
                "invalid candidate accepted"
            );
        }
        // Selection remains differentiable for accepted subnormals. In particular,
        // sum(normalized top-1) is constant, so its VJP is exactly zero, not inf-inf.
        let probability = Tensor::from_data([[-0.0f32, 0.0, f32::from_bits(1)]], &device).require_grad();
        let tiny = Candidates::select(probability.clone(), 1, path)?;
        ensure!(ids_host(tiny.ids.clone()) == vec![2], "subnormal/signed-zero ordering");
        ensure!(values(tiny.weights.clone()) == vec![1.0], "subnormal top-1 normalization");
        ensure!(values(tiny.expected(
            Tensor::ones([3,1], &device), Tensor::zeros([1], &device),
            Tensor::ones([1,1], &device), false,
        )) == vec![1.0], "subnormal sparse support was mistaken for zero mass");
        let gradients = tiny.weights.sum().backward();
        ensure!(values(probability.grad(&gradients).expect("subnormal selection VJP")) == vec![0.0;3],
            "constant normalized selection loss has a nonzero/nonfinite VJP");
        let probability = Tensor::from_data(
            [[f32::from_bits(1), f32::from_bits(3)]], &device,
        ).require_grad();
        let normalized = delta_kernels::soft_normalize(
            probability.clone(), probability.clone().sum_dim(1),
        );
        ensure!(values(normalized.clone()) == vec![0.25,0.75],
            "subnormal ratios changed during normalization");
        let gradients = normalized.sum().backward();
        ensure!(values(probability.grad(&gradients).expect("subnormal ratio VJP")) == vec![0.0;2],
            "constant subnormal ratio sum has a nonzero/nonfinite VJP");
        let probability = Tensor::from_data([[0.0f32, 0.0, f32::from_bits(1)],[0.0,0.0,0.0]], &device).require_grad();
        let output = soft::dense_expected(probability.clone(), Tensor::ones([3,1], &device),
            Tensor::zeros([1], &device), Tensor::ones([2,1], &device));
        ensure!(values(output.clone()) == vec![1.0,0.0], "subnormal/zero-row dense normalization");
        let gradients = output.sum().backward();
        ensure!(values(probability.grad(&gradients).expect("subnormal dense VJP")) == vec![0.0;6],
            "constant dense normalized loss has a nonzero/nonfinite VJP");
        let probability = Tensor::from_data([[0.0f32,0.0,0.0]], &device).require_grad();
        let zero = delta_kernels::soft_normalize(probability.clone(),probability.clone().sum_dim(1));
        ensure!(values(zero.clone()) == vec![0.0;3], "zero-row normalization extension");
        let gradients = zero.sum().backward();
        ensure!(values(probability.grad(&gradients).expect("zero-row normalization VJP")) == vec![1.0;3],
            "zero-row denominator-one derivative changed");
        let probability = Tensor::from_data([[0.0f32,0.0,f32::from_bits(1)]], &device).require_grad();
        let normalized = delta_kernels::soft_normalize(probability.clone(),probability.clone().sum_dim(1));
        let gradients = (normalized*Tensor::from_data([[0.0f32,1.0,2.0]],&device)).sum().backward();
        let derivative = values(probability.grad(&gradients).expect("unrepresentable normalization VJP"));
        ensure!(derivative == vec![f32::NEG_INFINITY,f32::NEG_INFINITY,0.0],
            "genuinely unrepresentable normalization derivatives were hidden");
        let mut a = SoftState::new(
            257,
            16,
            vec![Some(2), None, None, None],
            vec![true, true, true, false],
            &device,
            Path::Cpu,
        )?;
        let mut b = SoftState::new(257, 16, a.observed.clone(), a.active.clone(), &device, path)?;
        let mut latest_logits = None;
        for step in 0..4 {
            let mut lv = (0..4 * 257)
                .map(|i| ((i + step * 7) % 17) as f32 / 17.0 - 0.5)
                .collect::<Vec<_>>();
            lv[257..514].fill(f32::NEG_INFINITY);
            lv[514..771].fill(-0.0);
            lv[515] = 0.0;
            let left =
                Tensor::from_data(TensorData::new(lv.clone(), [4, 257]), &device).require_grad();
            let right = Tensor::from_data(TensorData::new(lv, [4, 257]), &device).require_grad();
            a.update(left.clone(), 1.0, 16, &confidence)?;
            b.update(right.clone(), 1.0, 16, &confidence)?;
            latest_logits = Some((left, right));
            a.retain(&[0, 1, 2, 3]);
            b.retain(&[0, 1, 2, 3]);
            state_equal(&a, &b)?;
            ensure!(
                values(b.visibility.clone())[0] == 1.0 && values(b.visibility.clone())[1] == 0.0
                    && values(b.visibility.clone())[2] > 0.0 && values(b.visibility.clone())[3] == 0.0,
                "learned coefficient/immutable observations"
            );
            ensure!(
                ids_host(b.provisional.clone()) == vec![-1, -1, 1, -1],
                "independent immutable/unavailable/signed-zero proposals"
            );
            ensure!(
                ids_host(b.retained.clone()) == vec![-1, -1, 1, -1],
                "independent retained proposals"
            );
        }
        let (left,right) = latest_logits.unwrap();
        let table = Tensor::ones([257,2], &device).require_grad();
        let mask = Tensor::zeros([2], &device).require_grad();
        let out = b.embeddings(table.clone(),mask.clone(),false);
        let gb = out.sum().backward();
        ensure!(left.grad(&gb).is_none() && right.grad(&gb).is_none(),
            "teacher logits received feedback gradients");
        ensure!(confidence.raw_scale.val().grad(&gb).is_some()
            && table.grad(&gb).is_some() && mask.grad(&gb).is_some(),
            "live confidence/embedding/mask gradients absent");
        for invalid in [f32::NAN, f32::INFINITY] {
            let invalid = Tensor::full([4, 257], invalid, &device);
            ensure!(
                b.update(invalid, 1.0, 16, &confidence).is_err(),
                "nonfinite logits accepted"
            );
        }
        cases.push(json!({"path":path,"episode":episode,"state_transition_rounds":4,"nonfinite_rejection":true}));
    }
    Ok(
        json!({"passed":true,"cases":cases,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,"scope":"normalized top-k selection, independent f64 outputs and VJPs, learned entropy coefficient and raw-parameter derivatives, latest-only detached prediction, immutable observations, padding, signed-zero ties, unavailable rows; no recurrent resets exist in this component"}),
    )
}
fn timed<T>(device: &Device, run: impl FnOnce() -> Result<T>) -> Result<f64> {
    runtime::sync(device)?;
    let start = Instant::now();
    let out = run()?;
    runtime::sync(device)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(out);
    Ok(ms)
}
pub fn run(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(
        c.backend == "vulkan" && !c.fusion && !c.autotune,
        "soft comparison requires heuristic non-fusion Vulkan"
    );
    ensure!(
        (1..=3).contains(&warmups) && (1..=9).contains(&trials),
        "warmups 1..3, trials 1..9"
    );
    let started = Instant::now();
    let mut records = Vec::new();
    let path = c.artifact_dir.join(format!(
        "soft-benchmark-b{}-t{}-k{}.json",
        c.batch_size, c.context, c.soft_candidate_width
    ));
    let device = device.clone().autodiff();
    let confidence = SoftConfidence::new(&device);
    for positions in [c.context] {
        for vocab in [257, 32000] {
            for width in [c.soft_candidate_width] {
                ensure!(
                    started.elapsed().as_secs_f64() < c.max_seconds.min(285.0),
                    "soft benchmark deadline; partial report saved"
                );
                let rows = c.batch_size * positions;
                device.seed(1307);
                let logits =
                    Tensor::<2>::random([rows, vocab], Distribution::Uniform(-1.0, 1.0), &device);
                let probabilities = softmax(logits.clone(), 1);
                // Initialization/JIT, fixture transfer, and resident feedback state construction are outside warm samples.
                let mut states = Vec::new();
                let mut first_state_ms = Vec::new();
                for p in [Path::Cpu, Path::Gpu, Path::GpuTiled] {
                    runtime::sync(&device)?;
                    let start = Instant::now();
                    states.push(SoftState::new(
                        vocab,
                        width,
                        vec![None; rows],
                        vec![true; rows],
                        &device,
                        p,
                    )?);
                    runtime::sync(&device)?;
                    first_state_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                let baseline = Candidates::select(probabilities.clone(), width, Path::Cpu)?;
                let baseline_ids = ids_host(baseline.ids.clone());
                for backend in [Path::Gpu, Path::GpuTiled] {
                    let candidate = Candidates::select(probabilities.clone(), width, backend)?;
                    ensure!(
                        ids_host(candidate.ids) == baseline_ids,
                        "realistic-shape candidate IDs {backend:?} {rows}/{vocab}/{width}"
                    );
                    close(
                        &values(candidate.weights),
                        &values(baseline.weights.clone())
                            .into_iter()
                            .map(f64::from)
                            .collect::<Vec<_>>(),
                        "realistic-shape weights",
                    )?;
                    close(
                        &values(candidate.retained_mass),
                        &values(baseline.retained_mass.clone())
                            .into_iter()
                            .map(f64::from)
                            .collect::<Vec<_>>(),
                        "realistic-shape retained mass",
                    )?;
                }
                let table =
                    Tensor::<2>::random([vocab, 16], Distribution::Uniform(-0.1, 0.1), &device);
                let mask = Tensor::<1>::zeros([16], &device);
                let mut select_samples: [Vec<f64>; 3] = Default::default();
                let mut feedback_samples: [Vec<f64>; 3] = Default::default();
                let mut fb_samples: [Vec<f64>; 3] = Default::default();
                for i in 0..warmups + trials {
                    for p in 0..3 {
                        let m = (p + i) % 3;
                        let backend = [Path::Cpu, Path::Gpu, Path::GpuTiled][m];
                        let ms = timed(&device, || {
                            Candidates::select(probabilities.clone(), width, backend)
                        })?;
                        let fm = timed(&device, || {
                            let s = &mut states[m];
                            s.update(logits.clone(), 1.0, width, &confidence)?;
                            s.retain(&[0]);
                            Ok(s.embeddings(table.clone(), mask.clone(), false))
                        })?;
                        let fb = timed(&device, || {
                            let p = probabilities.clone().detach().require_grad();
                            let e = table.clone().detach().require_grad();
                            let mask = mask.clone().detach().require_grad();
                            let selected = Candidates::select(p.clone(), width, backend)?;
                            let out = selected.expected(
                                e.clone(),
                                mask.clone(),
                                Tensor::full([rows, 1], 0.7, &device),
                                false,
                            );
                            let g = out.sum().backward();
                            Ok((p.grad(&g), e.grad(&g), mask.grad(&g)))
                        })?;
                        if i >= warmups {
                            select_samples[m].push(ms);
                            feedback_samples[m].push(fm);
                            fb_samples[m].push(fb);
                        }
                    }
                }
                let mut dispatches = Vec::new();
                for backend in [Path::Cpu, Path::Gpu, Path::GpuTiled] {
                    dispatches.push(crate::gemm::dispatches(&device, || {
                        std::hint::black_box(
                            Candidates::select(probabilities.clone(), width, backend).unwrap(),
                        );
                    })?);
                }
                records.push(json!({"batch":c.batch_size,"positions":positions,"rows":rows,"vocab":vocab,"width":width,"paths":["cpu","gpu","gpu_tiled"],"selection_samples_ms":select_samples,"selection_median_ms":select_samples.iter().map(|v|median(v)).collect::<Vec<_>>(),"feedback_round_samples_ms":feedback_samples,"feedback_round_median_ms":feedback_samples.iter().map(|v|median(v)).collect::<Vec<_>>(),"selection_expectation_backward_samples_ms":fb_samples,"selection_expectation_backward_median_ms":fb_samples.iter().map(|v|median(v)).collect::<Vec<_>>(),"selection_dispatches":dispatches,"probability_bytes":rows*vocab*4,"candidate_bytes":rows*width*8+rows*4,"sampled_reserved_bytes":runtime::sampled_reserved(&device)}));
                records.last_mut().unwrap()["first_state_construction_ms"] = json!(first_state_ms);
                runtime::write_json(
                    &path,
                    &json!({"completed":false,"device":runtime::identity(&device),"config":c,"records":records}),
                )?;
            }
        }
    }
    let report = json!({"completed":true,"device":runtime::identity(&device),"config":c,"warmups":warmups,"trials":trials,"records":records,"scope":"paired selection and standalone feedback, not SBD or a causal-LM speedup","elapsed_seconds":started.elapsed().as_secs_f64()});
    runtime::write_json(&path, &report)?;
    Ok(report)
}

/// Negative control: selection is inactive in the causal LM, so no model speedup is expected.
pub fn step(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(
        c.task == "lm" && c.backend == "vulkan" && !c.fusion && !c.autotune,
        "soft step control requires the recommended LM"
    );
    ensure!(
        (1..=3).contains(&warmups) && (1..=9).contains(&trials),
        "warmups 1..3, trials 1..9"
    );
    let device = device.clone().autodiff();
    let corpus = crate::train::loaded_corpus(c)?;
    device.seed(c.seed);
    let base = crate::model::LanguageModel::new(c, &c.arms[0], &device)?;
    let mut configs = [c.clone(), c.clone()];
    configs[0].soft_selection_backend = Path::Cpu;
    configs[1].soft_selection_backend = Path::Gpu;
    let mut runners = [
        crate::profile::Runner::new(base.clone(), c),
        crate::profile::Runner::new(base, c),
    ];
    let mut samples: [Vec<f64>; 2] = Default::default();
    for i in 0..warmups + trials {
        for m in if i % 2 == 0 { [0, 1] } else { [1, 0] } {
            let ms = runners[m].step(
                &configs[m],
                corpus.as_ref(),
                &device,
                &mut delta_mixers::profile::Unprofiled,
            )?;
            if i >= warmups {
                samples[m].push(ms);
            }
        }
    }
    let report = json!({"completed":true,"batch":c.batch_size,"context":c.context,"paths":["cpu","gpu"],"samples_ms":samples,"medians_ms":samples.iter().map(|v|median(v)).collect::<Vec<_>>(),"scope":"negative control: ordinary causal LM forward/backward/guarded AdamW; soft selection inactive, no SBD implementation or model speedup claimed"});
    runtime::write_json(
        &c.artifact_dir
            .join(format!("soft-step-b{}-t{}.json", c.batch_size, c.context)),
        &report,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpu_scalar_oracle_and_feedback_gradients() {
        super::reference(
            &crate::config::Config::default(),
            &burn::prelude::Device::ndarray(),
        )
        .unwrap();
    }
}
