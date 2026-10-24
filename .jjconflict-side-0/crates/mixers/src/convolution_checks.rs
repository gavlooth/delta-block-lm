//! Scalar-f64 forward and analytic VJP, including metadata and storage edge cases.
use super::ConvolutionBackend as Backend;
use burn::prelude::*;
use serde::Serialize;
#[derive(Serialize)]
pub struct Report {
    passed: bool,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    cases: Vec<Case>,
}
#[derive(Serialize)]
struct Case {
    backend: Backend,
    batch: usize,
    tokens: usize,
    channels: usize,
    width: usize,
    zero: bool,
    strided_inputs_weights_masks_adjoints: bool,
    resets_internal_tail_padding: bool,
    max_abs_error: f64,
}
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}
fn compare(label: &str, a: Vec<f32>, e: &[f64]) -> Result<f64, String> {
    if a.len() != e.len() {
        return Err(format!("{label}: length"));
    }
    let mut max = 0.0f64;
    for (i, (a, e)) in a.into_iter().zip(e).enumerate() {
        let error = (a as f64 - e).abs();
        if !a.is_finite() || error > 2e-5 + 3e-4 * e.abs() {
            return Err(format!("{label}[{i}]: {a} != {e}, error {error}"));
        }
        max = max.max(error);
    }
    Ok(max)
}
pub fn numerical_checks(device: &Device) -> Result<Report, String> {
    let device = device.clone().autodiff();
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    let gpu = matches!(
        device.clone().inner().as_dispatch(),
        burn::backend::DispatchDevice::Cube(_)
    );
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    let gpu = false;
    let modes = if gpu {
        vec![Backend::Tensor, Backend::PaddedTensor, Backend::FusedGpu]
    } else {
        vec![Backend::Tensor, Backend::PaddedTensor]
    };
    let mut cases = Vec::new();
    for (t, c, width) in [
        (1, 3, 4),
        (2, 35, 4),
        (7, 3, 4),
        (17, 128, 4),
        (19, 256, 4),
        (7, 35, 1),
    ] {
        for zero in [false, true] {
            for strided in [false, true] {
                let b = 2;
                let source: Vec<f32> = (0..b * t * c)
                    .map(|i| {
                        if zero {
                            0.
                        } else {
                            ((i * 17 + 3) as f64 * 0.013).sin() as f32
                        }
                    })
                    .collect();
                let x = if strided {
                    Tensor::<3>::from_data(TensorData::new(source, [b, c, t]), &device)
                        .swap_dims(1, 2)
                } else {
                    Tensor::<3>::from_data(TensorData::new(source, [b, t, c]), &device)
                }
                .require_grad();
                let weight: Vec<f32> = (0..width * c)
                    .map(|i| {
                        if i % 7 == 0 {
                            0.
                        } else {
                            ((i + 3) as f32 * 0.17).sin() * 0.1
                        }
                    })
                    .collect();
                let w = if strided {
                    Tensor::<2>::from_data(TensorData::new(weight, [c, width]), &device).transpose()
                } else {
                    Tensor::<2>::from_data(TensorData::new(weight, [width, c]), &device)
                }
                .require_grad();
                let metadata = |v: Vec<f32>| {
                    if strided {
                        Tensor::<2>::from_data(TensorData::new(v, [t, b]), &device).transpose()
                    } else {
                        Tensor::<2>::from_data(TensorData::new(v, [b, t]), &device)
                    }
                };
                let valid = metadata(
                    (0..b * t)
                        .map(|i| if i % 7 == 3 || i % t == t - 1 { 0. } else { 1. })
                        .collect(),
                );
                let reset = metadata(
                    (0..b * t)
                        .map(|i| {
                            if i % t == 0 || i % t == 2 || i % t == 5 {
                                1.
                            } else {
                                0.
                            }
                        })
                        .collect(),
                );
                let docs = (valid.clone() * reset).cumsum(1);
                // Force genuinely strided document IDs too, rather than merely a strided reset input.
                let docs = if strided {
                    let v = values(docs);
                    let mut storage = vec![0.; b * t];
                    for row in 0..b {
                        for time in 0..t {
                            storage[time * b + row] = v[row * t + time];
                        }
                    }
                    metadata(storage)
                } else {
                    docs
                };
                let up: Vec<f32> = (0..b * t * c)
                    .map(|i| {
                        if zero || i % 13 == 0 {
                            0.
                        } else {
                            ((i * 19 + 1) as f64 * 0.031).cos() as f32
                        }
                    })
                    .collect();
                let u = if strided {
                    Tensor::<3>::from_data(TensorData::new(up, [b, c, t]), &device).swap_dims(1, 2)
                } else {
                    Tensor::<3>::from_data(TensorData::new(up, [b, t, c]), &device)
                };
                let (xv, wv, vv, dv, uv) = (
                    values(x.clone()),
                    values(w.clone()),
                    values(valid.clone()),
                    values(docs.clone()),
                    values(u.clone()),
                );
                let mut y = vec![0.; b * t * c];
                let mut dx = y.clone();
                let mut dw = vec![0.; width * c];
                for row in 0..b {
                    for dest in 0..t {
                        for ch in 0..c {
                            for lag in 0..width.min(dest + 1) {
                                let src = dest - lag;
                                let mask = if dv[row * t + src] == dv[row * t + dest] {
                                    vv[row * t + src] as f64 * vv[row * t + dest] as f64
                                } else {
                                    0.
                                };
                                let s = (row * t + src) * c + ch;
                                let d = (row * t + dest) * c + ch;
                                let p = lag * c + ch;
                                y[d] += xv[s] as f64 * wv[p] as f64 * mask;
                                dx[s] += uv[d] as f64 * wv[p] as f64 * mask;
                                dw[p] += uv[d] as f64 * xv[s] as f64 * mask;
                            }
                        }
                    }
                }
                for &backend in &modes {
                    let out = backend.forward(x.clone(), w.clone(), valid.clone(), docs.clone());
                    let mut error = compare("conv output", values(out.clone()), &y)?;
                    let grads = (out * u.clone()).sum().backward();
                    error = error.max(compare(
                        "conv dx",
                        values(x.grad(&grads).ok_or("missing dx")?),
                        &dx,
                    )?);
                    error = error.max(compare(
                        "conv dw",
                        values(w.grad(&grads).ok_or("missing dw")?),
                        &dw,
                    )?);
                    cases.push(Case { backend, batch: b, tokens: t, channels: c, width, zero,
                        strided_inputs_weights_masks_adjoints: strided,
                        resets_internal_tail_padding: true, max_abs_error: error });
                }
            }
        }
    }
    Ok(Report { passed: true, absolute_tolerance: 2e-5, relative_tolerance: 3e-4, cases })
}
#[cfg(test)]
#[test]
fn cpu_convolution_matches_f64() {
    numerical_checks(&Device::ndarray()).unwrap();
}
#[cfg(test)]
#[test]
#[should_panic(expected = "no CPU fallback")]
fn fused_convolution_never_falls_back_to_cpu() {
    crate::MixerConfig {
        convolution_backend: Backend::FusedGpu,
        ..Default::default()
    }.init(&Device::ndarray());
}
