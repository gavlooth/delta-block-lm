//! Same-input tensor/fused HOLA checks against independent scalar references.
//! Membership-only schedules also match the old attention/selection path exactly.
//! CPU transfers happen only in these explicit checks, never in the operator.
use super::{ATOL, RTOL, compare, values};
use crate::{CachePolicy, HolaBackend, hola};
use burn::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Instant;

#[derive(Debug, Serialize)]
pub struct HolaCheck {
    pub hola_backend: HolaBackend,
    pub selection: CachePolicy,
    pub fixture: &'static str,
    pub oracle: &'static str,
    pub tokens: usize,
    pub capacity: usize,
    pub chunk: usize,
    pub output_max_abs: f64,
    pub membership_max_abs: f64,
    pub q_gradient_max_abs: f64,
    pub k_gradient_max_abs: f64,
    pub v_gradient_max_abs: f64,
    pub sink_gradient_max_abs: f64,
    pub scores_nondifferentiable: bool,
}

#[derive(Debug, Serialize)]
pub struct HolaTiming {
    pub hola_backend: HolaBackend,
    pub first_forward_ms: f64,
    pub first_forward_backward_ms: f64,
    pub forward_ms: Vec<f64>,
    pub forward_backward_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct HolaBenchmark {
    pub batch: usize,
    pub tokens: usize,
    pub heads: usize,
    pub capacity: usize,
    pub chunk: usize,
    pub warmups_per_mode: usize,
    pub paired_order: &'static str,
    pub numerical: Vec<HolaCheck>,
    pub timing: Vec<HolaTiming>,
    pub position_schedule_bytes: usize,
    pub key_lifetime_bytes: usize,
    pub softmax_statistics_bytes: usize,
    pub backward_auxiliary_bytes: usize,
    pub retained_output_bytes: usize,
    pub dense_rank_or_attention_matrix_bytes: usize,
}

struct Fixture {
    b: usize,
    t: usize,
    h: usize,
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
    name: &'static str,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    valid: Vec<f32>,
    docs: Vec<f32>,
    sink: Vec<f32>,
    upstream: Vec<f32>,
}

impl Fixture {
    fn new(
        b: usize,
        t: usize,
        capacity: usize,
        chunk: usize,
        policy: CachePolicy,
        packed: bool,
    ) -> Self {
        let h = 4;
        let signal = |width, phase| {
            (0..b * t * h * width)
                .map(|i| 0.7 * ((i * 11 + phase) as f64 * 0.09).sin() as f32)
                .collect::<Vec<_>>()
        };
        let mut q = signal(32, 3);
        let mut k = signal(32, 7);
        // Independently normalized Q and K with nonuniform learned RMS scales.
        // The full-layer fixture separately differentiates those external scales.
        for tensor in [&mut q, &mut k] {
            for row in tensor.chunks_exact_mut(32) {
                let rms = (row.iter().map(|x| x * x).sum::<f32>() / 32. + 1.0e-5).sqrt();
                for (i, x) in row.iter_mut().enumerate() {
                    *x = *x / rms * (0.8 + 0.01 * i as f32);
                }
            }
        }
        let mut valid = vec![1.; b * t];
        let mut docs = vec![0.; b * t];
        if packed {
            for row in 0..b {
                for p in 0..t {
                    let i = row * t + p;
                    if p == 8 || p == 47 {
                        valid[i] = 0.;
                    }
                    if row == 1 {
                        // Repeated document labels, internal and full-chunk padding.
                        docs[i] = if p < 37 {
                            0.
                        } else if p < 79 {
                            1.
                        } else {
                            0.
                        };
                        if (16..32).contains(&p) {
                            valid[i] = 0.;
                        }
                        if p == 31 || p + 1 == t {
                            valid[i] = 0.;
                            docs[i] = 77.;
                        }
                    }
                }
            }
        }
        let scores = (0..b * t * h)
            .map(|i| {
                let p = i / h % t;
                if valid[i / h] == 0. {
                    1000.
                } else if p == 2 {
                    8.
                } else {
                    (p % 5) as f32
                }
            })
            .collect();
        Self {
            b,
            t,
            h,
            capacity,
            chunk,
            policy,
            name: if packed {
                "ties_resets_padding_last_physical_slot"
            } else {
                "single_document_resident_cache"
            },
            q,
            k,
            v: signal(64, 13),
            scores,
            valid,
            docs,
            sink: vec![0.2, -0.7, 3., -2.],
            upstream: signal(64, 21),
        }
    }

    fn overflow() -> Self {
        let mut f = Self::new(1, 4, 1, 4, CachePolicy::Surprise, false);
        f.name = "finite_large_original_values_normalized_before_matmul";
        f.q.fill(0.);
        f.k.fill(0.);
        f.v.fill(0.);
        f.sink.fill(0.);
        f.upstream.fill(0.);
        for p in 0..4 {
            for h in 0..f.h {
                f.v[(p * f.h + h) * 64] = 1.0e38;
                // Normal f32 seed: Vulkan denormal preservation is not assumed.
                f.upstream[(p * f.h + h) * 64] = 1.0e-36;
            }
        }
        f
    }

    fn upload(&self, device: &Device) -> Inputs {
        let input = |data: &Vec<f32>, width| {
            Tensor::from_data(
                TensorData::new(data.clone(), [self.b, self.t, self.h, width]),
                device,
            )
            .require_grad()
        };
        Inputs {
            q: input(&self.q, 32),
            k: input(&self.k, 32),
            v: input(&self.v, 64),
            scores: input(&self.scores, 1),
            valid: Tensor::from_data(
                TensorData::new(self.valid.clone(), [self.b, self.t]),
                device,
            ),
            docs: Tensor::from_data(TensorData::new(self.docs.clone(), [self.b, self.t]), device),
            sink: Tensor::from_data(TensorData::new(self.sink.clone(), [self.h]), device)
                .require_grad(),
            upstream: Tensor::from_data(
                TensorData::new(self.upstream.clone(), [self.b, self.t, self.h, 64]),
                device,
            ),
        }
    }

    fn upload_strided(&self, device: &Device) -> Inputs {
        let mut x = self.upload(device);
        let strided = |data: &Vec<f32>, width| {
            let mut backing = vec![0.; self.b * self.h * self.t * width];
            for b in 0..self.b {
                for t in 0..self.t {
                    for h in 0..self.h {
                        for i in 0..width {
                            backing[((b * self.h + h) * self.t + t) * width + i] =
                                data[((b * self.t + t) * self.h + h) * width + i];
                        }
                    }
                }
            }
            Tensor::<4>::from_data(
                TensorData::new(backing, [self.b, self.h, self.t, width]),
                device,
            )
            .swap_dims(1, 2)
            .require_grad()
        };
        x.q = strided(&self.q, 32);
        x.k = strided(&self.k, 32);
        x.v = strided(&self.v, 64);
        x.scores = strided(&self.scores, 1);
        x.upstream = strided(&self.upstream, 64);
        x
    }
}

struct Inputs {
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    docs: Tensor<2>,
    sink: Tensor<1>,
    upstream: Tensor<4>,
}

impl Inputs {
    fn detailed(&self, f: &Fixture, backend: HolaBackend) -> hola::CacheRead {
        hola::read_detailed_selected(
            backend,
            self.q.clone(),
            self.k.clone(),
            self.v.clone(),
            self.scores.clone(),
            self.valid.clone(),
            self.docs.clone(),
            f.capacity,
            f.chunk,
            f.policy,
            self.sink.clone(),
        )
    }
    fn forward(&self, f: &Fixture, backend: HolaBackend) -> Tensor<4> {
        hola::read_selected(
            backend,
            self.q.clone(),
            self.k.clone(),
            self.v.clone(),
            self.scores.clone(),
            self.valid.clone(),
            self.docs.clone(),
            f.capacity,
            f.chunk,
            f.policy,
            self.sink.clone(),
        )
    }
}

struct Oracle {
    output: Vec<f64>,
    positions: Vec<f64>,
    dq: Vec<f64>,
    dk: Vec<f64>,
    dv: Vec<f64>,
    dsink: Vec<f64>,
}

fn oracle(f: &Fixture) -> Oracle {
    let mut out = Oracle {
        output: vec![0.; f.v.len()],
        positions: vec![-1.; f.b * f.h * f.capacity],
        dq: vec![0.; f.q.len()],
        dk: vec![0.; f.k.len()],
        dv: vec![0.; f.v.len()],
        dsink: vec![0.; f.h],
    };
    let scale = 1. / 32.0f64.sqrt();
    for b in 0..f.b {
        for h in 0..f.h {
            let mut cache = Vec::<usize>::new();
            for first in (0..f.t).step_by(f.chunk) {
                let end = (first + f.chunk).min(f.t);
                for t in first..end {
                    if f.valid[b * f.t + t] < 0.5 {
                        continue;
                    }
                    let visible = cache
                        .iter()
                        .copied()
                        .chain(first..=t)
                        .filter(|&s| {
                            f.valid[b * f.t + s] >= 0.5
                                && f.docs[b * f.t + s] == f.docs[b * f.t + t]
                        })
                        .collect::<Vec<_>>();
                    let qt = ((b * f.t + t) * f.h + h) * 32;
                    let ot = ((b * f.t + t) * f.h + h) * 64;
                    let logits = visible
                        .iter()
                        .map(|&s| {
                            let ks = ((b * f.t + s) * f.h + h) * 32;
                            (0..32)
                                .map(|i| f64::from(f.q[qt + i]) * f64::from(f.k[ks + i]))
                                .sum::<f64>()
                                * scale
                        })
                        .collect::<Vec<_>>();
                    let sink = f64::from(f.sink[h]);
                    let maximum = logits.iter().copied().fold(sink, f64::max);
                    let denominator = logits.iter().map(|&x| (x - maximum).exp()).sum::<f64>()
                        + (sink - maximum).exp();
                    let probabilities = logits
                        .iter()
                        .map(|&x| (x - maximum).exp() / denominator)
                        .collect::<Vec<_>>();
                    for (&s, &p) in visible.iter().zip(&probabilities) {
                        let vs = ((b * f.t + s) * f.h + h) * 64;
                        for j in 0..64 {
                            out.output[ot + j] += p * f64::from(f.v[vs + j]);
                        }
                    }
                    let delta = (0..64)
                        .map(|j| f64::from(f.upstream[ot + j]) * out.output[ot + j])
                        .sum::<f64>();
                    out.dsink[h] -= (sink - maximum).exp() / denominator * delta;
                    for (&s, &p) in visible.iter().zip(&probabilities) {
                        let ks = ((b * f.t + s) * f.h + h) * 32;
                        let vs = ((b * f.t + s) * f.h + h) * 64;
                        let dot = (0..64)
                            .map(|j| f64::from(f.upstream[ot + j]) * f64::from(f.v[vs + j]))
                            .sum::<f64>();
                        let dl = p * (dot - delta) * scale;
                        for i in 0..32 {
                            out.dq[qt + i] += dl * f64::from(f.k[ks + i]);
                            out.dk[ks + i] += dl * f64::from(f.q[qt + i]);
                        }
                        for j in 0..64 {
                            out.dv[vs + j] += p * f64::from(f.upstream[ot + j]);
                        }
                    }
                }
                cache.extend(first..end);
                let last_doc = f.docs[b * f.t + end - 1];
                cache.retain(|&s| f.valid[b * f.t + s] >= 0.5 && f.docs[b * f.t + s] == last_doc);
                cache.sort_by(|&left, &right| {
                    let score = |s| {
                        if f.policy == CachePolicy::Recency {
                            s as f32
                        } else {
                            f.scores[(b * f.t + s) * f.h + h]
                        }
                    };
                    score(right)
                        .total_cmp(&score(left))
                        .then_with(|| left.cmp(&right))
                });
                cache.truncate(f.capacity);
            }
            for (slot, &s) in cache.iter().enumerate() {
                out.positions[(b * f.h + h) * f.capacity + slot] = s as f64;
            }
        }
    }
    out
}

fn check(
    f: &Fixture,
    x: &Inputs,
    expected: &Oracle,
    backend: HolaBackend,
) -> Result<HolaCheck, String> {
    let checked: Result<HolaCheck, String> = (|| {
        let result = x.detailed(f, backend);
        let output = compare(
            "HOLA output",
            &values(result.output.clone()),
            &expected.output,
            ATOL,
            RTOL,
        )?;
        let positions = compare(
            "HOLA final membership",
            &values(result.final_positions),
            &expected.positions,
            0.,
            0.,
        )?;
        let grads = (result.output * x.upstream.clone()).sum().backward();
        let grad = |name, input: &Tensor<4>, expected: &[f64]| -> Result<f64, String> {
            compare(
                name,
                &values(
                    input
                        .grad(&grads)
                        .ok_or_else(|| format!("missing {name} gradient"))?,
                ),
                expected,
                ATOL,
                RTOL,
            )
        };
        let dq = grad("HOLA q", &x.q, &expected.dq)?;
        let dk = grad("HOLA selected original k", &x.k, &expected.dk)?;
        let dv = grad("HOLA selected original v", &x.v, &expected.dv)?;
        let dsink = compare(
            "HOLA learned sink",
            &values(x.sink.grad(&grads).ok_or("missing HOLA sink gradient")?),
            &expected.dsink,
            ATOL,
            RTOL,
        )?;
        let discrete = x.scores.grad(&grads).is_none();
        if !discrete {
            return Err("HOLA selection scores unexpectedly differentiated".into());
        }
        Ok(HolaCheck {
            hola_backend: backend,
            selection: f.policy,
            fixture: f.name,
            oracle: "independent_scalar_f64",
            tokens: f.t,
            capacity: f.capacity,
            chunk: f.chunk,
            output_max_abs: output,
            membership_max_abs: positions,
            q_gradient_max_abs: dq,
            k_gradient_max_abs: dk,
            v_gradient_max_abs: dv,
            sink_gradient_max_abs: dsink,
            scores_nondifferentiable: discrete,
        })
    })();
    checked.map_err(|error| format!(
        "HOLA backend={backend:?} policy={:?} B{}/T{}/H{}/K32/V64/capacity{}/chunk{} fixture={}: {error}",
        f.policy,f.b,f.t,f.h,f.capacity,f.chunk,f.name,
    ))
}

fn cancellation_check(device: &Device, backend: HolaBackend) -> Result<HolaCheck, String> {
    let mut f = Fixture::new(1, 1, 1, 1, CachePolicy::Surprise, false);
    f.name = "cancellation_vjp_matches_actual_forward_probability";
    f.q.fill(0.);
    f.k.fill(0.);
    f.v.fill(0.);
    f.sink.fill(0.);
    f.upstream.fill(0.);
    for h in 0..f.h {
        f.q[h * 32..h * 32 + 4].copy_from_slice(&[1.0e4, 1., 1.0e4, 1.]);
        f.k[h * 32..h * 32 + 4].copy_from_slice(&[1.0e4, 1., -1.0e4, 1.]);
        f.v[h * 64] = 1.;
        f.upstream[h * 64] = 1.;
    }
    let x = f.upload(device);
    let observed = values(x.forward(&f, backend));
    let mut expected = Oracle {
        output: observed.iter().map(|&x| f64::from(x)).collect(),
        positions: vec![0.; f.h],
        dq: vec![0.; f.q.len()],
        dk: vec![0.; f.k.len()],
        dv: vec![0.; f.v.len()],
        dsink: vec![0.; f.h],
    };
    for h in 0..f.h {
        let p = f64::from(observed[h * 64]);
        if !(0. < p && p < 1.) {
            return Err(
                "cancellation probe must have a nonsaturated finite forward probability".into(),
            );
        }
        let derivative = p * (1. - p);
        expected.dv[h * 64] = p;
        expected.dsink[h] = -derivative;
        for i in 0..32 {
            expected.dq[h * 32 + i] = derivative * f64::from(f.k[h * 32 + i]) / 32.0f64.sqrt();
            expected.dk[h * 32 + i] = derivative * f64::from(f.q[h * 32 + i]) / 32.0f64.sqrt();
        }
    }
    let mut report = check(&f, &x, &expected, backend)?;
    // This regression checks consistency with the actual f32 forward, not a
    // claimed f64 cancellation result; the regular fixtures use the f64 oracle.
    report.oracle = "analytic_vjp_of_observed_f32_forward_probability";
    Ok(report)
}

/// CPU explicitly checks the tensor reference; CubeCL checks tensor and fused
/// implementations on identical inputs. Each result names its actual backend.
pub fn hola_numerical_checks(device: &Device) -> Result<Vec<HolaCheck>, String> {
    let device = device.clone().autodiff();
    let backends = {
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            if matches!(
                device.clone().inner().as_dispatch(),
                burn::backend::DispatchDevice::Cube(_)
            ) {
                vec![
                    HolaBackend::Tensor,
                    HolaBackend::FusedGpu,
                    HolaBackend::FusedGpuPairs,
                    HolaBackend::FusedGpuPairsNorms,
                ]
            } else {
                vec![HolaBackend::Tensor]
            }
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            vec![HolaBackend::Tensor]
        }
    };
    fixture_checks(&device, &backends)
}

/// Explicit isolated backend scope; an unavailable fused scope is an error.
pub fn hola_numerical_checks_selected(
    device: &Device,
    backend: HolaBackend,
) -> Result<Vec<HolaCheck>, String> {
    let device = device.clone().autodiff();
    if backend.is_fused() {
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        if !matches!(
            device.clone().inner().as_dispatch(),
            burn::backend::DispatchDevice::Cube(_)
        ) {
            return Err("fused HOLA check requires a CubeCL GPU device; no CPU fallback".into());
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        return Err("fused HOLA check requires the vulkan or cuda feature; no CPU fallback".into());
    }
    fixture_checks(&device, &[backend])
}

fn fixture_checks(device: &Device, backends: &[HolaBackend]) -> Result<Vec<HolaCheck>, String> {
    let mut reports = Vec::new();
    for policy in [CachePolicy::Surprise, CachePolicy::Recency] {
        for (time, capacity, chunk) in [(7, 16, 16), (35, 16, 1), (83, 16, 16), (145, 64, 16)] {
            let f = Fixture::new(2, time, capacity, chunk, policy, true);
            let expected = oracle(&f);
            let x = f.upload(device);
            for &backend in backends {
                reports.push(check(&f, &x, &expected, backend)?);
            }
        }
    }
    let f = Fixture::overflow();
    let expected = oracle(&f);
    let x = f.upload(device);
    for &backend in backends {
        reports.push(check(&f, &x, &expected, backend)?);
    }
    let mut f = Fixture::new(2, 35, 16, 16, CachePolicy::Surprise, true);
    f.name = "strided_q_k_v_scores_upstream";
    let x = f.upload_strided(device);
    let expected = oracle(&f);
    for &backend in backends {
        reports.push(check(&f, &x, &expected, backend)?);
    }
    f.name = "zero_inputs_and_adjoints";
    f.q.fill(0.);
    f.k.fill(0.);
    f.v.fill(0.);
    f.upstream.fill(0.);
    f.sink.fill(0.);
    let x = f.upload_strided(device);
    let expected = oracle(&f);
    for &backend in backends {
        reports.push(check(&f, &x, &expected, backend)?);
    }
    for &backend in backends.iter().filter(|backend| backend.is_fused()) {
        reports.push(cancellation_check(device, backend)?);
    }
    Ok(reports)
}

/// Prove every position-only schedule snapshot, including the empty initial cache.
///
/// With the CPU feature, always exercise ndarray independently of the requested
/// device. A requested CubeCL device additionally exercises its tensor path and
/// every fused HOLA selector. This check performs no training or timing study.
pub fn membership_schedule_checks(device: &Device) -> Result<Value, String> {
    let mut cases = Vec::new();
    #[cfg(feature = "cpu")]
    membership_schedule_device_checks(
        &Device::ndarray().autodiff(),
        "cpu_ndarray",
        &[HolaBackend::Tensor],
        &mut cases,
    )?;
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    let configured_gpu_checked = matches!(
        device.clone().inner().as_dispatch(),
        burn::backend::DispatchDevice::Cube(_)
    );
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    let configured_gpu_checked = false;
    if configured_gpu_checked {
        membership_schedule_device_checks(
            &device.clone().autodiff(),
            "configured_gpu",
            &[
                HolaBackend::Tensor,
                HolaBackend::FusedGpu,
                HolaBackend::FusedGpuPairs,
                HolaBackend::FusedGpuPairsNorms,
            ],
            &mut cases,
        )?;
    }
    #[cfg(not(feature = "cpu"))]
    if !configured_gpu_checked {
        membership_schedule_device_checks(
            &device.clone().autodiff(),
            "configured_device",
            &[HolaBackend::Tensor],
            &mut cases,
        )?;
    }
    // CPU-only builds still use the same public device-taking interface.
    let _ = device;
    Ok(json!({
        "passed": true,
        "max_abs": 0.0,
        "precision": "float32_exact_integer_positions",
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
        "oracle": "cache_reference::evaluate.final_positions_on_repacked_truncated_prefixes",
        "oracle_qkv_width": 1,
        "legacy_comparator": "hola::read_with_selection.positions",
        "layout": ["batch", "heads", "ceil(time/chunk)+1", "capacity"],
        "initial_snapshot": "all_minus_one",
        "snapshots": "initial_and_after_every_chunk_including_partial",
        "cpu_checked": cfg!(feature = "cpu"),
        "configured_gpu_checked": configured_gpu_checked,
        "scores_nondifferentiable": true,
        "cases": cases,
    }))
}

/// Each prefix must be repacked row by row: truncating the flattened arrays
/// would silently reuse row zero's suffix as the next batch row.
fn membership_prefix_oracle(f: &Fixture, end: usize) -> Vec<f64> {
    if end == 0 {
        return vec![-1.; f.b * f.h * f.capacity];
    }
    let mut scores = Vec::with_capacity(f.b * end * f.h);
    let mut valid = Vec::with_capacity(f.b * end);
    let mut docs = Vec::with_capacity(f.b * end);
    for row in 0..f.b {
        scores.extend(
            f.scores[row * f.t * f.h..(row * f.t + end) * f.h]
                .iter()
                .map(|&x| f64::from(x)),
        );
        valid.extend(
            f.valid[row * f.t..row * f.t + end]
                .iter()
                .map(|&x| x >= 0.5),
        );
        docs.extend(
            f.docs[row * f.t..row * f.t + end]
                .iter()
                .map(|&x| x as usize),
        );
    }
    let zeros = vec![0.; f.b * end * f.h];
    crate::cache_reference::evaluate(
        &zeros,
        &zeros,
        &zeros,
        &scores,
        &valid,
        &docs,
        [f.b, end, f.h, 1, 1],
        f.capacity,
        f.chunk,
        f.policy,
        &zeros,
    )
    .final_positions
}

fn membership_schedule_device_checks(
    device: &Device,
    scope: &'static str,
    backends: &[HolaBackend],
    reports: &mut Vec<Value>,
) -> Result<(), String> {
    for policy in [CachePolicy::Surprise, CachePolicy::Recency] {
        for capacity in [1, 4, 16] {
            for chunk in [1, 16] {
                // The long case includes all packed edge cases. The short
                // case additionally leaves an underfilled, partial first chunk.
                for time in [7, 83] {
                    if time == 7 && chunk == 1 {
                        continue;
                    }
                    let mut f = Fixture::new(2, time, capacity, chunk, policy, true);
                    f.name = if time == 7 {
                        "short_partial_chunk_exact_ties"
                    } else {
                        "ties_internal_and_full_chunk_padding_resets_repeated_labels_invalid_last_doc"
                    };
                    if time == 83 {
                        // Row zero must retain eligible old entries through
                        // all-padding; row one's invalid final doc 77 must
                        // instead empty the cache at that same boundary.
                        f.valid[16..32].fill(0.);
                    }
                    for row in 0..f.b {
                        for pos in 0..f.t {
                            for head in 0..f.h {
                                f.scores[(row * f.t + pos) * f.h + head] =
                                    if f.valid[row * f.t + pos] < 0.5 {
                                        1000.
                                    } else if pos == 2 {
                                        8.
                                    } else {
                                        ((pos * (head + 1) + 2 * row) % 5) as f32
                                    };
                            }
                        }
                    }
                    let snapshots = time.div_ceil(chunk) + 1;
                    let boundaries = (0..snapshots)
                        .map(|index| (index * chunk).min(time))
                        .collect::<Vec<_>>();
                    let expected = boundaries
                        .iter()
                        .map(|&end| membership_prefix_oracle(&f, end))
                        .collect::<Vec<_>>();
                    let scores = Tensor::<4>::from_data(
                        TensorData::new(f.scores.clone(), [f.b, time, f.h, 1]),
                        device,
                    )
                    .require_grad();
                    let valid = Tensor::<2>::from_data(
                        TensorData::new(f.valid.clone(), [f.b, time]),
                        device,
                    );
                    let docs = Tensor::<2>::from_data(
                        TensorData::new(f.docs.clone(), [f.b, time]),
                        device,
                    );
                    // Keep the actual old tensor reader, not schedule_selected
                    // invoked a second time. Width one avoids unnecessary
                    // attention work while preserving its original selector.
                    let zeros = Tensor::<4>::zeros([f.b, time, f.h, 1], device);
                    let (_, selections) = hola::read_with_selection(
                        zeros.clone(),
                        zeros.clone(),
                        zeros,
                        scores.clone(),
                        valid.clone(),
                        docs.clone(),
                        capacity,
                        chunk,
                        policy,
                        Tensor::zeros([f.h], device),
                        None,
                    );
                    if selections.len() + 1 != snapshots {
                        return Err("legacy membership snapshot count mismatch".into());
                    }
                    let mut legacy = vec![vec![-1.; f.b * f.h * capacity]];
                    for (index, selection) in selections.into_iter().enumerate() {
                        if selection.chunk_start != index * chunk
                            || selection.positions.dims() != [f.b, f.h, capacity]
                        {
                            return Err("legacy membership boundary metadata mismatch".into());
                        }
                        legacy.push(
                            values(selection.positions)
                                .into_iter()
                                .map(f64::from)
                                .collect(),
                        );
                    }
                    for &backend in backends {
                        let result: Result<Value, String> = (|| {
                            let schedule = hola::schedule_selected(
                                backend,
                                scores.clone(),
                                valid.clone(),
                                docs.clone(),
                                capacity,
                                chunk,
                                policy,
                            );
                            let shape = [f.b, f.h, snapshots, capacity];
                            if schedule.dims() != shape {
                                return Err(format!(
                                    "membership schedule shape {:?} != {shape:?}",
                                    schedule.dims()
                                ));
                            }
                            if !scores.is_tracked() || schedule.is_tracked() {
                                return Err(
                                    "membership schedule must be untracked despite tracked scores".into(),
                                );
                            }
                            let observed = values(schedule.clone());
                            let mut max_abs = 0.0f64;
                            let mut legacy_max_abs = 0.0f64;
                            for index in 0..snapshots {
                                let mut boundary = Vec::with_capacity(f.b * f.h * capacity);
                                for row_head in 0..f.b * f.h {
                                    let start = (row_head * snapshots + index) * capacity;
                                    boundary.extend_from_slice(&observed[start..start + capacity]);
                                }
                                max_abs = max_abs.max(compare(
                                    &format!("membership scalar oracle boundary {index}"),
                                    &boundary,
                                    &expected[index],
                                    0.,
                                    0.,
                                )?);
                                legacy_max_abs = legacy_max_abs.max(compare(
                                    &format!("membership old reader boundary {index}"),
                                    &boundary,
                                    &legacy[index],
                                    0.,
                                    0.,
                                )?);
                            }
                            // A separate tracked leaf permits backward even
                            // though a correct membership schedule has no tape.
                            let anchor = Tensor::<1>::ones([1], device).require_grad();
                            let gradients = (schedule.sum() + anchor.clone()).backward();
                            if scores.grad(&gradients).is_some()
                                || anchor.grad(&gradients).is_none()
                            {
                                return Err(
                                    "membership scores differentiated or gradient probe did not run".into(),
                                );
                            }
                            Ok(json!({
                                "passed": true,
                                "scope": scope,
                                "hola_backend": backend,
                                "selection": policy,
                                "fixture": f.name,
                                "shape": {"batch": f.b, "tokens": time, "heads": f.h,
                                    "capacity": capacity, "chunk": chunk},
                                "score_shape": [f.b, time, f.h, 1],
                                "metadata_shape": [f.b, time],
                                "schedule_shape": shape,
                                "checked_prefix_lengths": boundaries,
                                "boundaries_checked": snapshots,
                                "max_abs": max_abs,
                                "legacy_max_abs": legacy_max_abs,
                                "scores_nondifferentiable": true,
                                "schedule_tracked": false,
                            }))
                        })();
                        reports.push(result.map_err(|error| format!(
                            "membership schedule {scope} backend={backend:?} policy={policy:?} B{}/T{time}/H{}/capacity{capacity}/chunk{chunk}: {error}",
                            f.b, f.h,
                        ))?);
                    }
                }
            }
        }
    }
    Ok(())
}

fn timed(
    f: &Fixture,
    x: &Inputs,
    device: &Device,
    backend: HolaBackend,
    backward: bool,
) -> Result<f64, String> {
    device
        .sync()
        .map_err(|e| format!("HOLA benchmark sync: {e:?}"))?;
    let clock = Instant::now();
    let result = x.forward(f, backend);
    // Keep the actual gradients alive until after synchronization. There is no
    // host read or reduction of tensor contents inside the measured interval.
    let gradients = backward.then(|| (result.clone() * x.upstream.clone()).sum().backward());
    device
        .sync()
        .map_err(|e| format!("HOLA benchmark sync: {e:?}"))?;
    let elapsed = clock.elapsed().as_secs_f64() * 1000.;
    std::hint::black_box((result, gradients));
    Ok(elapsed)
}

/// Same resident inputs, paired AB/BA order, two warmups per mode, no timed upload.
/// Independent membership/output/Q/K/V/sink checks precede all throughput samples.
pub fn hola_paired_benchmark(
    device: &Device,
    batch: usize,
    time: usize,
    capacity: usize,
    iterations: usize,
) -> Result<HolaBenchmark, String> {
    if batch == 0 || time == 0 || iterations == 0 || !(1..=64).contains(&capacity) {
        return Err("invalid HOLA benchmark dimensions/trials".into());
    }
    let device = device.clone().autodiff();
    crate::chunk::assert_fused_device(&device);
    let f = Fixture::new(batch, time, capacity, 16, CachePolicy::Surprise, false);
    let x = f.upload(&device);
    let mut timing = Vec::new();
    // Capture first shape-specific execution before correctness checks warm it.
    for backend in [HolaBackend::Tensor, HolaBackend::FusedGpu] {
        timing.push(HolaTiming {
            hola_backend: backend,
            first_forward_ms: timed(&f, &x, &device, backend, false)?,
            first_forward_backward_ms: timed(&f, &x, &device, backend, true)?,
            forward_ms: Vec::new(),
            forward_backward_ms: Vec::new(),
        });
    }
    let expected = oracle(&f);
    let mut numerical = Vec::new();
    for backend in [HolaBackend::Tensor, HolaBackend::FusedGpu] {
        numerical.push(check(&f, &x, &expected, backend)?);
    }
    for _ in 0..2 {
        for backward in [false, true] {
            for backend in [HolaBackend::Tensor, HolaBackend::FusedGpu] {
                timed(&f, &x, &device, backend, backward)?;
            }
        }
    }
    for trial in 0..iterations {
        for backward in [false, true] {
            for ordinal in 0..2 {
                let i = if trial % 2 == 0 { ordinal } else { 1 - ordinal };
                let backend = timing[i].hola_backend;
                let ms = timed(&f, &x, &device, backend, backward)?;
                if backward {
                    timing[i].forward_backward_ms.push(ms);
                } else {
                    timing[i].forward_ms.push(ms);
                }
            }
        }
    }
    let rows = batch * time * f.h;
    Ok(HolaBenchmark {
        batch,
        tokens: time,
        heads: f.h,
        capacity,
        chunk: 16,
        warmups_per_mode: 2,
        paired_order: "alternating AB/BA; identical resident Q/K/V/scores/masks/docs/sink/upstream",
        numerical,
        timing,
        position_schedule_bytes: 4 * batch * f.h * (time.div_ceil(16) + 1) * capacity,
        key_lifetime_bytes: 4 * rows,
        softmax_statistics_bytes: 8 * rows,
        backward_auxiliary_bytes: 8 * rows,
        retained_output_bytes: 4 * rows * 64,
        dense_rank_or_attention_matrix_bytes: 0,
    })
}

/// Resident forward-only membership costs; selection has no backward.
pub fn membership_schedule_benchmark(device:&Device,warmups:usize,trials:usize)->Result<serde_json::Value,String>{
    if warmups<2 || trials<11 {return Err("membership timing needs two warmups and eleven pairs".into());}
    let supported={
        #[cfg(any(feature="vulkan",feature="cuda"))]
        {matches!(device.clone().inner().as_dispatch(),burn::backend::DispatchDevice::Cube(_))}
        #[cfg(not(any(feature="vulkan",feature="cuda")))]
        {false}
    };
    if !supported{return Err("membership timing requires the configured fused GPU device".into());}
    let fixture=Fixture::new(1,512,16,16,CachePolicy::Surprise,false);
    let inputs=fixture.upload(&device.clone().autodiff());
    let q=inputs.q.detach();let k=inputs.k.detach();let v=inputs.v.detach();
    let scores=inputs.scores.detach();
    let sink=Tensor::<1>::zeros([4],device);
    let run=|fused:bool|->Result<(Tensor<4>,f64),String>{
        device.sync().map_err(|error|format!("membership pre-sync: {error:?}"))?;
        let start=Instant::now();
        let schedule=if fused {
            hola::schedule_selected(HolaBackend::FusedGpu,scores.clone(),inputs.valid.clone(),inputs.docs.clone(),16,16,CachePolicy::Surprise)
        }else{
            let (read,selections)=hola::read_with_selection(q.clone(),k.clone(),v.clone(),scores.clone(),inputs.valid.clone(),inputs.docs.clone(),16,16,CachePolicy::Surprise,sink.clone(),None);
            std::hint::black_box(read);
            let mut positions=Vec::with_capacity(selections.len()+1);
            positions.push(Tensor::<3>::full([1,4,16],-1.0,device));
            positions.extend(selections.into_iter().map(|selection|selection.positions));
            Tensor::stack(positions,2)
        };
        device.sync().map_err(|error|format!("membership post-sync: {error:?}"))?;
        Ok((schedule,start.elapsed().as_secs_f64()*1000.0))
    };
    let mut samples:[Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
    for iteration in 0..warmups+trials{
        let mut results:[Option<Tensor<4>>;2]=[None,None];
        for mode in if iteration%2==0 {[0,1]}else{[1,0]}{
            let (schedule,time)=run(mode==1)?;
            results[mode]=Some(schedule);
            if iteration>=warmups{samples[mode].push(time);}
        }
        let old=values(results[0].take().unwrap());
        let new=values(results[1].take().unwrap());
        if old!=new{return Err("timed membership schedules differ".into());}
    }
    let median=|values:&[f64]|{let mut sorted=values.to_vec();sorted.sort_by(f64::total_cmp);sorted[sorted.len()/2]};
    Ok(serde_json::json!({"shape":{"batch":1,"time":512,"heads":4,"capacity":16,"chunk":16},
        "warmups":warmups,"trials":trials,"legacy_full_read_ms":samples[0],"position_only_ms":samples[1],
        "legacy_full_read_median_ms":median(&samples[0]),"position_only_median_ms":median(&samples[1]),
        "membership_max_abs":0,"schedule_bytes":1*4*33*16*4,"backward":"not applicable: membership is discrete and detached",
        "scope":"resident no-gradient Q/K/V/scores/valid/documents; old full tensor attention read solely to obtain membership vs adopted fused selection only; excludes cache-query normalization and uploads; synchronized alternating order"}))
}
