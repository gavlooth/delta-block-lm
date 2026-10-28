use super::{
    reference::{self, Rng, Shape, Weights},
    *,
};
use burn::tensor::TensorData;
use serde::Serialize;
#[derive(Serialize)]
pub struct Report {
    pub passed: bool,
    pub cases: usize,
    pub compared_elements: usize,
    pub maximum_absolute_error: f64,
    pub maximum_tolerance_ratio: f64,
    pub causality_max_absolute_error: f64,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
}
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}
fn wide<const D: usize>(x: Tensor<D>) -> Vec<f64> {
    values(x).into_iter().map(|v| v as f64).collect()
}
fn upload<const D: usize>(v: &[f64], dims: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(
        TensorData::new(v.iter().map(|&x| x as f32).collect::<Vec<_>>(), dims),
        device,
    )
}
fn close(r: &mut Report, name: &str, a: &[f32], b: &[f64]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{name}: lengths differ"));
    }
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let e = (a as f64 - b).abs();
        let ratio = e / (2e-5 + 3e-4 * b.abs());
        if !a.is_finite() || !b.is_finite() || ratio > 1. {
            return Err(format!("{name}[{i}] {a} != {b}, tolerance ratio {ratio}"));
        }
        r.maximum_absolute_error = r.maximum_absolute_error.max(e);
        r.maximum_tolerance_ratio = r.maximum_tolerance_ratio.max(ratio);
        r.compared_elements += 1;
    }
    Ok(())
}
pub fn numerical_checks(device: &Device) -> Result<Report, String> {
    let device = device.clone().autodiff();
    device.seed(7301);
    let mut report = Report {
        passed: true,
        cases: 0,
        compared_elements: 0,
        maximum_absolute_error: 0.,
        maximum_tolerance_ratio: 0.,
        causality_max_absolute_error: 0.,
        absolute_tolerance: 2e-5,
        relative_tolerance: 3e-4,
    };
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    let gpu = matches!(
        device.clone().inner().as_dispatch(),
        burn::backend::DispatchDevice::Cube(_)
    );
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    let gpu = false;
    let backends = if gpu {
        vec![WaveletBackend::Tensor, WaveletBackend::FusedGpu]
    } else {
        vec![WaveletBackend::Tensor]
    };
    for backend in backends {
        direct_strided_smoothing(&mut report, backend, &device)?;
        // Three db10 levels reach exactly 1 + 19*(1+2+4) = 134 positions.
        let mut impulse = vec![0.0; 136];
        impulse[0] = 1.0;
        let support = undecimated_bands(upload(&impulse, [1,136,1], &device),
            Tensor::ones([1,136], &device), Tensor::ones([1,136], &device), 3, backend);
        assert_eq!(support.len(), 4);
        let coarse = values(support[0].clone());
        if coarse[133] == 0.0 || coarse[134..].iter().any(|&x| x != 0.0) {
            return Err("J3 coarse band must have exactly 134-position causal support".into());
        }
        for case in 0..4 {
            let shape = Shape {
                documents: if case == 3 {
                    vec![vec![311], vec![1, 2, 3, 19, 280]]
                } else {
                    vec![vec![1, 2, 19, 5], vec![3, 1, 7]]
                },
                tokens: if case == 3 { 317 } else { 31 },
                d: 8,
                groups: 2,
                levels: 3,
            };
            let [b, t, d] = [shape.documents.len(), shape.tokens, shape.d];
            let mut rng = Rng::new(741 + case);
            let valid = shape
                .valid()
                .iter()
                .map(|&x| if x { 1f64 } else { 0. })
                .collect::<Vec<_>>();
            let mut reset = vec![0.; shape.rows()];
            for (start, _) in shape.spans() {
                reset[start] = 1.;
            }
            let valid_t = upload(&valid, [b, t], &device);
            let reset_t = upload(&reset, [b, t], &device);
            let docs = (reset_t.clone() * valid_t.clone()).cumsum(1);
            let module = WaveletDyn::new(d, 3, 2, backend, &device);
            let weights = Weights {
                input: wide(module.input.weight.val()),
                gate: wide(module.gate.weight.val()),
                gate_bias: wide(module.gate.bias.as_ref().unwrap().val()),
                value: wide(module.value.weight.val()),
                output: wide(module.output.weight.val()),
            };
            close(&mut report, "initial gate bias probabilities",
                &weights.gate_bias.iter().map(|b| (1.0 / (1.0 + (-b).exp())) as f32).collect::<Vec<_>>(),
                &[0.88,0.88,0.12,0.12,0.12,0.12,0.12,0.12])?;
            let u = (0..shape.rows() * d)
                .map(|i| {
                    if valid[i / d] == 0. {
                        f64::NAN
                    } else if case == 1 {
                        0.
                    } else {
                        (rng.normal() * 0.4) as f32 as f64
                    }
                })
                .collect::<Vec<_>>();
            let upstream = (0..u.len())
                .map(|_| (rng.normal() * 0.2) as f32 as f64)
                .collect::<Vec<_>>();
            let oracle = reference::forward(&shape, &weights, &u);
            let expected = reference::backward(&shape, &weights, &oracle, &upstream);
            if case == 0 {
                reference::check_gradient_by_differences(
                    &shape, &weights, &u, &upstream, &expected, 3, &mut rng,
                );
            }
            let input = if case == 2 {
                let mut transposed = vec![0.; u.len()];
                for row in 0..b {
                    for time in 0..t {
                        for channel in 0..d {
                            transposed[(row * d + channel) * t + time] =
                                u[(row * t + time) * d + channel];
                        }
                    }
                }
                upload(&transposed, [b, d, t], &device).swap_dims(1, 2)
            } else {
                upload(&u, [b, t, d], &device)
            }
            .require_grad();
            let out = module.forward(input.clone(), valid_t.clone(), reset_t.clone());
            close(&mut report, "output", &values(out.clone()), &oracle.out)?;
            let grads = (out * upload(&upstream, [b, t, d], &device))
                .sum()
                .backward();
            close(
                &mut report,
                "input gradient",
                &values(input.grad(&grads).ok_or("missing input gradient")?),
                &expected.u,
            )?;
            for (name, weight, want) in [
                ("input", module.input.weight.val(), &expected.weights.input),
                ("gate", module.gate.weight.val(), &expected.weights.gate),
                ("value", module.value.weight.val(), &expected.weights.value),
                (
                    "output",
                    module.output.weight.val(),
                    &expected.weights.output,
                ),
            ] {
                close(
                    &mut report,
                    name,
                    &values(weight.grad(&grads).ok_or("missing weight gradient")?),
                    want,
                )?;
            }
            close(
                &mut report,
                "gate bias",
                &values(
                    module
                        .gate
                        .bias
                        .as_ref()
                        .unwrap()
                        .val()
                        .grad(&grads)
                        .ok_or("missing bias gradient")?,
                ),
                &expected.weights.gate_bias,
            )?;
            let clean = u
                .iter()
                .map(|&x| if x.is_nan() { 0. } else { x })
                .collect::<Vec<_>>();
            let bands = undecimated_bands(
                upload(&clean, [b, t, d], &device),
                valid_t.clone(),
                docs,
                3,
                backend,
            );
            let expected_bands = reference::bands(&shape, &clean);
            assert_eq!(bands.len(), 4, "WNet-Dyn J3 has [A3,D3,D2,D1]");
            let mut sum = vec![0f32; u.len()];
            for (band, want) in bands.into_iter().zip(expected_bands) {
                let band = values(band);
                close(&mut report, "band", &band, &want)?;
                for (s, x) in sum.iter_mut().zip(band) {
                    *s += x;
                }
            }
            close(&mut report, "telescoping reconstruction", &sum, &clean)?;
            // Alter one valid token and all of a later document independently.
            let baseline = values(module.forward(
                upload(&u, [b, t, d], &device),
                valid_t.clone(),
                reset_t.clone(),
            ));
            for (start, end) in [(4, 5), (22, 27)] {
                let mut perturbed = u.clone();
                for x in &mut perturbed[start * d..end * d] {
                    *x += 3.;
                }
                let changed = values(module.forward(
                    upload(&perturbed, [b, t, d], &device),
                    valid_t.clone(),
                    reset_t.clone(),
                ));
                for (doc_start, len) in shape.spans() {
                    for row in doc_start..doc_start + len {
                        if row < start || !(doc_start..doc_start + len).contains(&start) {
                            for c in 0..d {
                                let error =
                                    (baseline[row * d + c] - changed[row * d + c]).abs() as f64;
                                report.causality_max_absolute_error =
                                    report.causality_max_absolute_error.max(error);
                                if error != 0. {
                                    return Err(format!("causality row {row}: {error}"));
                                }
                            }
                        }
                    }
                }
            }
            report.cases += 1;
        }
    }
    Ok(report)
}
/// Exercise the kernel itself with strided signal, masks and upstream, not just strided projection input.
fn direct_strided_smoothing(
    report: &mut Report,
    backend: WaveletBackend,
    device: &Device,
) -> Result<(), String> {
    let (b, t, c) = (2, 29, 3);
    let mut signal = vec![0f64; b * t * c];
    let mut upstream = vec![0f64; b * t * c];
    let mut valid = vec![0f64; b * t];
    let mut docs = vec![0f64; b * t];
    let mut rng = Rng::new(8921);
    for row in 0..b {
        for time in 0..t {
            valid[row * t + time] = if time < t - 3 { 1. } else { 0. };
            docs[row * t + time] = if time < 7 { 0. } else { 1. };
            for channel in 0..c {
                signal[(row * t + time) * c + channel] = rng.normal() as f32 as f64;
                upstream[(row * t + time) * c + channel] = rng.normal() as f32 as f64;
            }
        }
    }
    let strided3 = |v: &[f64]| {
        let mut transposed = vec![0.; v.len()];
        for row in 0..b {
            for time in 0..t {
                for channel in 0..c {
                    transposed[(row * c + channel) * t + time] = v[(row * t + time) * c + channel];
                }
            }
        }
        upload(&transposed, [b, c, t], device).swap_dims(1, 2)
    };
    let strided2 = |v: &[f64]| {
        let transposed = (0..t)
            .flat_map(|time| (0..b).map(move |row| v[row * t + time]))
            .collect::<Vec<_>>();
        upload(&transposed, [t, b], device).swap_dims(0, 1)
    };
    for dilation in [1, 2, 8, 128] {
        let mut expected = vec![0.; signal.len()];
        let mut dx = vec![0.; signal.len()];
        for row in 0..b {
            for dest in 0..t {
                if valid[row * t + dest] == 0. {
                    continue;
                }
                for (k, weight) in smoothing_filter().into_iter().enumerate() {
                    if let Some(source) = dest.checked_sub(k * dilation) {
                        if valid[row * t + source] == 1.
                            && docs[row * t + source] == docs[row * t + dest]
                        {
                            for channel in 0..c {
                                let s = (row * t + source) * c + channel;
                                let d = (row * t + dest) * c + channel;
                                expected[d] += signal[s] * weight;
                                dx[s] += upstream[d] * weight;
                            }
                        }
                    }
                }
            }
        }
        let input = strided3(&signal).require_grad();
        let out = smooth(
            input.clone(),
            strided2(&valid),
            strided2(&docs),
            dilation,
            backend,
        );
        close(report, "strided smoothing", &values(out.clone()), &expected)?;
        let gradients = (out * strided3(&upstream)).sum().backward();
        close(
            report,
            "strided smoothing input VJP",
            &values(input.grad(&gradients).ok_or("missing smoother VJP")?),
            &dx,
        )?;
        report.cases += 1;
    }
    Ok(())
}
#[cfg(all(test, feature = "cpu"))]
mod tests {
    #[test]
    fn causal_wavelet_f64_gate() {
        super::numerical_checks(&burn::prelude::Device::ndarray()).unwrap();
    }
}
