//! Same-input tensor/fused HOLA checks against an independent scalar f64 VJP.
//! CPU transfers happen only in these explicit checks, never in the operator.
use super::{ATOL, RTOL, compare, values};
use crate::{CachePolicy, HolaBackend, hola};
use burn::prelude::*;
use serde::Serialize;
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
