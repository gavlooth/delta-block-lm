//! Numerical acceptance checks used by the experiment CLI; no training result is implied.
use crate::{
    CachePolicy, HolaBackend, MixerKind, ScanBackend,
    chunk::{self, ScanInput},
    reference::{self, Token},
};
use burn::prelude::*;
use serde::Serialize;
use std::time::Instant;
#[path = "kron_checks.rs"]
mod kron;
pub use kron::{KronCheck, kron_numerical_checks, kron_numerical_checks_selected};
#[path = "hola_checks.rs"]
mod hola;
pub use hola::{
    HolaBenchmark, HolaCheck, hola_numerical_checks, hola_numerical_checks_selected,
    hola_paired_benchmark,
};

#[derive(Debug, Serialize)]
pub struct NumericalReport {
    pub precision: &'static str,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub recurrence: Vec<RecurrenceCheck>,
    pub cache: Vec<CacheCheck>,
    pub layer: Vec<LayerCheck>,
    pub fused_kron: Vec<KronCheck>,
    pub fused_hola: Vec<HolaCheck>,
}
#[derive(Debug, Serialize)]
pub struct RecurrenceCheck {
    pub mixer: MixerKind,
    pub implementation: &'static str,
    pub tokens: usize,
    pub fixture: &'static str,
    pub final_state_probe_tokens: usize,
    pub output_max_abs: f64,
    pub state_max_abs: Option<f64>,
    pub score_max_abs: Option<f64>,
    pub gradient_max_abs: f64,
    pub continuation_max_abs: Option<f64>,
    pub first_forward_backward_ms: f64,
    pub warm_forward_backward_ms: f64,
}
#[derive(Debug, Serialize)]
pub struct CacheCheck {
    pub selection: CachePolicy,
    pub capacity: usize,
    pub output_max_abs: f64,
    pub gradient_max_abs: f64,
    pub membership_max_abs: f64,
    pub allocated_attention_pairs: usize,
}
#[derive(Debug, Serialize)]
pub struct LayerCheck {
    pub mixer: MixerKind,
    pub cache: CachePolicy,
    pub scan_backend: ScanBackend,
    pub hola_backend: HolaBackend,
    pub input_directional_gradient_error: f64,
    pub projection_gradient_error: f64,
    pub cache_parameter_gradient_errors: Vec<CacheParameterCheck>,
}
#[derive(Debug, Serialize)]
pub struct CacheParameterCheck {
    pub parameter: &'static str,
    pub absolute_error: f64,
}
const ATOL: f64 = 2.0e-5;
const RTOL: f64 = 3.0e-4;
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data()
        .try_to_vec::<f32>()
        .expect("f32 numerical fixture")
}
fn compare(
    name: &str,
    actual: &[f32],
    expected: &[f64],
    atol: f64,
    rtol: f64,
) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{name}: length {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut max = 0.0f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(a) - e).abs();
        if !a.is_finite() || !e.is_finite() || error > atol + rtol * e.abs() {
            return Err(format!(
                "{name}[{i}] {a} != {e} (absolute error {error}, tolerance {atol}+{rtol}*abs(reference))"
            ));
        }
        max = max.max(error);
    }
    Ok(max)
}
struct Fixture {
    b: usize,
    t: usize,
    h: usize,
    k: usize,
    v: usize,
    widths: [usize; 6],
    data: Vec<Vec<f32>>,
    valid: Vec<bool>,
    reset: Vec<bool>,
    upstream: Vec<f32>,
    final_adjoint: Vec<f32>,
}
impl Fixture {
    fn new(kind: MixerKind, t: usize) -> Self {
        let (b, h, k, v) = (2, 4, 32, 64);
        let widths = [
            k,
            k,
            v,
            k,
            if kind == MixerKind::Gdn2 { k } else { 1 },
            if kind == MixerKind::Gdn2 { v } else { 1 },
        ];
        let data = widths
            .iter()
            .enumerate()
            .map(|(field, &width)| {
                (0..b * t * h * width)
                    .map(|i| {
                        let signal = ((i * 13 + field * 19 + 1) as f64 * 0.037).sin() as f32;
                        if field == 3 {
                            0.91 + 0.03 * signal
                        } else if field >= 4 {
                            0.45 + 0.1 * signal
                        } else {
                            0.1 * signal
                        }
                    })
                    .collect()
            })
            .collect();
        let valid = (0..b * t)
            .map(|i| {
                let r = i / t;
                let p = i % t;
                p < t.saturating_sub(r + 1) && p != 9 && p != 11
            })
            .collect();
        let reset = (0..b * t)
            .map(|i| {
                let p = i % t;
                p == 0 || p == 5 || p == 19
            })
            .collect();
        let upstream = (0..b * t * h * v)
            .map(|i| ((i * 7 + 3) as f64 * 0.051).cos() as f32)
            .collect();
        let final_adjoint = (0..b * h * k * v)
            .map(|i| 0.03 * ((i * 3 + 1) as f64 * 0.04).sin() as f32)
            .collect();
        Self {
            b,
            t,
            h,
            k,
            v,
            widths,
            data,
            valid,
            reset,
            upstream,
            final_adjoint,
        }
    }
    fn aligned(kind: MixerKind, t: usize) -> Self {
        let mut fixture = Self::new(kind, t);
        for field in 0..6 {
            let width = fixture.widths[field];
            for (index, value) in fixture.data[field].iter_mut().enumerate() {
                *value = if field == 3 {
                    1.0
                } else if field >= 4 {
                    0.5
                } else if index % width == 0 {
                    1.0
                } else {
                    0.0
                };
            }
        }
        fixture
    }
    fn inputs(&self, device: &Device) -> Vec<Tensor<4>> {
        self.data
            .iter()
            .zip(self.widths)
            .map(|(x, width)| {
                Tensor::from_data(
                    TensorData::new(x.clone(), [self.b, self.t, self.h, width]),
                    device,
                )
                .require_grad()
            })
            .collect()
    }
    fn metadata(&self, device: &Device) -> (Tensor<2>, Tensor<2>) {
        let tensor = |x: &Vec<bool>| {
            Tensor::from_data(
                TensorData::new(
                    x.iter().map(|&x| if x { 1.0f32 } else { 0.0 }).collect(),
                    [self.b, self.t],
                ),
                device,
            )
        };
        (tensor(&self.valid), tensor(&self.reset))
    }
    fn oracle(
        &self,
        kind: MixerKind,
        include_final: bool,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let mut outputs = vec![0.; self.b * self.t * self.h * self.v];
        let mut states = vec![0.; self.b * self.h * self.k * self.v];
        let mut scores = vec![0.; self.b * self.t * self.h];
        let mut gradients = self
            .data
            .iter()
            .map(|x| vec![0.; x.len()])
            .collect::<Vec<_>>();
        for row in 0..self.b {
            for head in 0..self.h {
                let field = |f: usize, p: usize| {
                    let first = ((row * self.t + p) * self.h + head) * self.widths[f];
                    self.data[f][first..first + self.widths[f]]
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect::<Vec<_>>()
                };
                let tokens = (0..self.t)
                    .map(|p| Token {
                        q: field(0, p),
                        k: field(1, p),
                        v: field(2, p),
                        alpha: field(3, p),
                        erase: field(4, p),
                        write: field(5, p),
                    })
                    .collect::<Vec<_>>();
                let limits = row * self.t..(row + 1) * self.t;
                let result = reference::scan(
                    kind,
                    &tokens,
                    &self.valid[limits.clone()],
                    &self.reset[limits.clone()],
                    None,
                );
                let upstream = (0..self.t)
                    .map(|p| {
                        let first = ((row * self.t + p) * self.h + head) * self.v;
                        self.upstream[first..first + self.v]
                            .iter()
                            .map(|&x| f64::from(x))
                            .collect()
                    })
                    .collect::<Vec<_>>();
                let sf = (row * self.h + head) * self.k * self.v;
                let seed = self.final_adjoint[sf..sf + self.k * self.v]
                    .iter()
                    .map(|&x| if include_final { f64::from(x) } else { 0. })
                    .collect::<Vec<_>>();
                let grad = reference::backward(
                    kind,
                    &tokens,
                    &self.valid[limits.clone()],
                    &self.reset[limits],
                    &upstream,
                    &seed,
                    None,
                    &vec![0.; self.t],
                );
                states[sf..sf + self.k * self.v].copy_from_slice(&result.final_state);
                for p in 0..self.t {
                    let out = ((row * self.t + p) * self.h + head) * self.v;
                    outputs[out..out + self.v].copy_from_slice(&result.output[p]);
                    scores[(row * self.t + p) * self.h + head] = result.scores[p];
                    for (f, g) in grad.tokens[p].fields().into_iter().enumerate() {
                        let first = ((row * self.t + p) * self.h + head) * self.widths[f];
                        gradients[f][first..first + self.widths[f]].copy_from_slice(g);
                    }
                }
            }
        }
        (outputs, states, scores, gradients)
    }
}
fn scan_input(x: &[Tensor<4>]) -> ScanInput {
    ScanInput {
        q: x[0].clone(),
        k: x[1].clone(),
        v: x[2].clone(),
        alpha: x[3].clone(),
        erase: x[4].clone(),
        write: x[5].clone(),
    }
}
fn recurrence_check(
    device: &Device,
    kind: MixerKind,
    time: usize,
    aligned: bool,
) -> Result<RecurrenceCheck, String> {
    let fixture = if aligned {
        Fixture::aligned(kind, time)
    } else {
        Fixture::new(kind, time)
    };
    let expected = fixture.oracle(kind, true);
    let run = || {
        let x = fixture.inputs(device);
        let (valid, reset) = fixture.metadata(device);
        let result = chunk::scan(kind, scan_input(&x), valid, reset, 16, None);
        let out = values(result.output.clone());
        let state = values(result.final_state.clone());
        let scores = values(result.scores);
        let loss = (result.output
            * Tensor::from_data(
                TensorData::new(
                    fixture.upstream.clone(),
                    [fixture.b, time, fixture.h, fixture.v],
                ),
                device,
            ))
        .sum()
            + (result.final_state
                * Tensor::from_data(
                    TensorData::new(
                        fixture.final_adjoint.clone(),
                        [fixture.b, fixture.h, fixture.k, fixture.v],
                    ),
                    device,
                ))
            .sum();
        let gradients = loss.backward();
        let grads = x
            .iter()
            .map(|x| values(x.grad(&gradients).expect("recurrence input gradient")))
            .collect::<Vec<_>>();
        (out, state, scores, grads)
    };
    let first = Instant::now();
    let actual = run();
    let first_ms = first.elapsed().as_secs_f64() * 1000.;
    let warm = Instant::now();
    let again = run();
    let warm_ms = warm.elapsed().as_secs_f64() * 1000.;
    let output_error = compare("chunk output", &actual.0, &expected.0, ATOL, RTOL)?;
    compare("warm output", &again.0, &expected.0, ATOL, RTOL)?;
    let state_error = compare("chunk final state", &actual.1, &expected.1, ATOL, RTOL)?;
    let score_error = compare("active edit score", &actual.2, &expected.2, ATOL, RTOL)?;
    let mut gradient_error = 0.0f64;
    for f in 0..6 {
        gradient_error = gradient_error.max(compare(
            &format!("chunk gradient field {f}"),
            &actual.3[f],
            &expected.3[f],
            ATOL,
            RTOL,
        )?);
    }
    // Resume at an active-block boundary INSIDE the computational chunk (position 3).
    let split = 3.min(time - 1);
    let x = fixture.inputs(device);
    let (valid, reset) = fixture.metadata(device);
    let section = |start, end| {
        let xs = x
            .iter()
            .map(|x| {
                x.clone()
                    .slice([0..fixture.b, start..end, 0..fixture.h, 0..x.dims()[3]])
            })
            .collect::<Vec<_>>();
        (
            scan_input(&xs),
            valid.clone().slice([0..fixture.b, start..end]),
            reset.clone().slice([0..fixture.b, start..end]),
        )
    };
    let (a, av, ar) = section(0, split);
    let a = chunk::scan(kind, a, av, ar, 16, None);
    let (b, bv, br) = section(split, time);
    let b = chunk::scan(kind, b, bv, br, 16, Some(a.final_state));
    let continuation = compare(
        "inside-chunk continuation",
        &values(Tensor::cat(vec![a.output, b.output], 1)),
        &expected.0,
        ATOL,
        RTOL,
    )?;
    compare(
        "continued final state",
        &values(b.final_state),
        &expected.1,
        ATOL,
        RTOL,
    )?;
    Ok(RecurrenceCheck {
        mixer: kind,
        implementation: "burn_chunk_triangular",
        fixture: if aligned {
            "aligned_unit_keys_values"
        } else {
            "sinusoidal"
        },
        tokens: time,
        final_state_probe_tokens: 0,
        output_max_abs: output_error,
        state_max_abs: Some(state_error),
        score_max_abs: Some(score_error),
        gradient_max_abs: gradient_error,
        continuation_max_abs: Some(continuation),
        first_forward_backward_ms: first_ms,
        warm_forward_backward_ms: warm_ms,
    })
}

/// Runs CPU or GPU checks on the supplied runtime device. Device transfer/readback
/// synchronizes each timed pass; first-use and warm times are reported separately.
pub fn numerical_checks(device: &Device) -> Result<NumericalReport, String> {
    let device = device.clone().autodiff();
    let mut recurrence = Vec::new();
    for kind in [MixerKind::Gdn2, MixerKind::KronLstm] {
        for t in [7, 35] {
            recurrence.push(recurrence_check(&device, kind, t, false)?);
        }
        recurrence.push(recurrence_check(&device, kind, 35, true)?);
    }
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    if matches!(
        device.clone().inner().as_dispatch(),
        burn::backend::DispatchDevice::Cube(_)
    ) {
        for t in [7, 35] {
            for probes in [false, true] {
                recurrence.push(custom_gdn_check(&device, t, probes)?);
            }
        }
    }
    let mut cache = Vec::new();
    for policy in [CachePolicy::Surprise, CachePolicy::Recency] {
        for capacity in [3, 16] {
            cache.push(cache_check(&device, policy, capacity)?);
        }
    }
    let mut layer = Vec::new();
    for kind in [MixerKind::Gdn2, MixerKind::KronLstm] {
        for policy in [
            CachePolicy::None,
            CachePolicy::Surprise,
            CachePolicy::Recency,
        ] {
            layer.push(layer_check(
                &device,
                kind,
                policy,
                ScanBackend::TensorSolver,
                HolaBackend::Tensor,
            )?);
        }
    }
    let fused_kron = kron_numerical_checks(&device)?;
    let mut fused_hola = Vec::new();
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    if matches!(
        device.clone().inner().as_dispatch(),
        burn::backend::DispatchDevice::Cube(_)
    ) {
        for policy in [
            CachePolicy::None,
            CachePolicy::Surprise,
            CachePolicy::Recency,
        ] {
            layer.push(layer_check(
                &device,
                MixerKind::KronLstm,
                policy,
                ScanBackend::FusedGpu,
                HolaBackend::Tensor,
            )?);
        }
        fused_hola = hola_numerical_checks(&device)?;
        for kind in [MixerKind::Gdn2, MixerKind::KronLstm] {
            for policy in [CachePolicy::Surprise, CachePolicy::Recency] {
                layer.push(layer_check(
                    &device,
                    kind,
                    policy,
                    ScanBackend::TensorSolver,
                    HolaBackend::FusedGpu,
                )?);
            }
        }
        for policy in [CachePolicy::Surprise, CachePolicy::Recency] {
            layer.push(layer_check(
                &device,
                MixerKind::KronLstm,
                policy,
                ScanBackend::FusedGpu,
                HolaBackend::FusedGpu,
            )?);
        }
    }
    Ok(NumericalReport {
        precision: "f32 execution / f64 sequential analytic reference",
        absolute_tolerance: ATOL,
        relative_tolerance: RTOL,
        recurrence,
        cache,
        layer,
        fused_kron,
        fused_hola,
    })
}

fn cache_check(
    device: &Device,
    policy: CachePolicy,
    capacity: usize,
) -> Result<CacheCheck, String> {
    let (b, t, h, dk, dv) = (2, 35, 4, 32, 64);
    let data = |width, phase| {
        (0..b * t * h * width)
            .map(|i| (((i * 11 + phase) as f64 * 0.09).sin() * 0.7) as f32)
            .collect::<Vec<_>>()
    };
    let qdata = data(dk, 3);
    let kdata = data(dk, 7);
    let vdata = data(dv, 13);
    let q = Tensor::from_data(TensorData::new(qdata.clone(), [b, t, h, dk]), device).require_grad();
    let k = Tensor::from_data(TensorData::new(kdata.clone(), [b, t, h, dk]), device).require_grad();
    let v = Tensor::from_data(TensorData::new(vdata.clone(), [b, t, h, dv]), device).require_grad();
    // Deliberate exact ties, early high-surprise entries, and masked high scores.
    let scores = (0..b * t * h)
        .map(|i| {
            if (i / h) % t == 2 {
                8.0f32
            } else {
                ((i / h) % 5) as f32
            }
        })
        .collect::<Vec<_>>();
    let valid = (0..b * t)
        .map(|i| i % t < 33 && i % t != 8)
        .collect::<Vec<_>>();
    let docs = (0..b * t)
        .map(|i| if i % t < 19 { 0usize } else { 1 })
        .collect::<Vec<_>>();
    let upstream = data(dv, 21);
    let to64 = |x: &Vec<f32>| x.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let expected = crate::cache_reference::evaluate(
        &to64(&qdata),
        &to64(&kdata),
        &to64(&vdata),
        &to64(&scores),
        &valid,
        &docs,
        [b, t, h, dk, dv],
        capacity,
        16,
        policy,
        &to64(&upstream),
    );
    let vt = Tensor::from_data(
        TensorData::new(
            valid
                .iter()
                .map(|&x| if x { 1.0f32 } else { 0.0 })
                .collect(),
            [b, t],
        ),
        device,
    );
    let dt = Tensor::from_data(
        TensorData::new(docs.iter().map(|&x| x as f32).collect(), [b, t]),
        device,
    );
    let actual = crate::hola::read_detailed(
        q.clone(),
        k.clone(),
        v.clone(),
        Tensor::from_data(TensorData::new(scores, [b, t, h, 1]), device),
        vt,
        dt,
        capacity,
        16,
        policy,
        Tensor::zeros([h], device),
    );
    let error = compare(
        "HOLA exact read",
        &values(actual.output.clone()),
        &expected.output,
        ATOL,
        RTOL,
    )?;
    let membership = compare(
        "causal cache membership",
        &values(actual.final_positions),
        &expected.final_positions,
        0.,
        0.,
    )?;
    if actual.attention_pairs != expected.pairs {
        return Err("HOLA allocated attention-work accounting mismatch".into());
    }
    let gradients = (actual.output
        * Tensor::from_data(TensorData::new(upstream, [b, t, h, dv]), device))
    .sum()
    .backward();
    let mut gradient_error = 0.0f64;
    for (name, input, expected) in [
        ("cache q", q, expected.dq),
        ("cache k", k, expected.dk),
        ("cache v", v, expected.dv),
    ] {
        gradient_error = gradient_error.max(compare(
            name,
            &values(
                input
                    .grad(&gradients)
                    .ok_or_else(|| format!("missing {name} gradient"))?,
            ),
            &expected,
            ATOL,
            RTOL,
        )?);
    }
    Ok(CacheCheck {
        selection: policy,
        capacity,
        output_max_abs: error,
        gradient_max_abs: gradient_error,
        membership_max_abs: membership,
        allocated_attention_pairs: actual.attention_pairs,
    })
}

fn layer_check(
    device: &Device,
    kind: MixerKind,
    policy: CachePolicy,
    scan_backend: ScanBackend,
    hola_backend: HolaBackend,
) -> Result<LayerCheck, String> {
    use burn::module::{Module, Param};
    let config = crate::MixerConfig {
        kind,
        cache_policy: policy,
        scan_backend,
        hola_backend,
        projection_backend: crate::ProjectionBackend::Separate,
        normalization_backend: crate::NormalizationBackend::Tensor,
        convolution_backend: crate::ConvolutionBackend::Tensor,
        d_model: 8,
        heads: 1,
        key_dim: if scan_backend.is_fused() || hola_backend.is_fused() {
            32
        } else {
            4
        },
        value_dim: if scan_backend.is_fused() || hola_backend.is_fused() {
            64
        } else {
            6
        },
        chunk_size: 4,
        cache_capacity: 3,
    };
    let model = config.init(device);
    let data = (0..1 * 7 * 8)
        .map(|i| ((i * 3 + 1) as f64 * 0.19).sin() as f32)
        .collect::<Vec<_>>();
    let direction = (0..data.len())
        .map(|i| 0.02 * ((i * 5 + 2) as f64 * 0.13).cos() as f32)
        .collect::<Vec<_>>();
    let upstream = (0..data.len())
        .map(|i| 0.03 * ((i * 7 + 1) as f64 * 0.07).cos() as f32)
        .collect::<Vec<_>>();
    let x = Tensor::from_data(TensorData::new(data.clone(), [1, 7, 8]), device).require_grad();
    let valid = Tensor::from_data([[1.0f32, 1., 1., 1., 1., 1., 0.]], device);
    let reset = Tensor::from_data([[1.0f32, 0., 1., 0., 0., 0., 0.]], device);
    let weight = model.q.weight.val();
    let up = Tensor::from_data(TensorData::new(upstream.clone(), [1, 7, 8]), device);
    let gradients = (model.forward_masked(x.clone(), valid.clone(), reset.clone()) * up)
        .sum()
        .backward();
    let dx = values(x.grad(&gradients).ok_or("missing layer input gradient")?);
    let dw = values(
        weight
            .grad(&gradients)
            .ok_or("missing query projection parameter gradient")?,
    );
    if !dx.iter().chain(dw.iter()).all(|x| x.is_finite()) {
        return Err("nonfinite trainable mixer gradient".into());
    }
    let objective = |model: &crate::Mixer, input: Vec<f32>| {
        let out = values(model.forward_masked(
            Tensor::from_data(TensorData::new(input, [1, 7, 8]), device),
            valid.clone(),
            reset.clone(),
        ));
        out.iter()
            .zip(&upstream)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
    };
    let inference = model.valid();
    let epsilon = 2.0e-3f64;
    let plus = data
        .iter()
        .zip(&direction)
        .map(|(&x, &d)| x + epsilon as f32 * d)
        .collect();
    let minus = data
        .iter()
        .zip(&direction)
        .map(|(&x, &d)| x - epsilon as f32 * d)
        .collect();
    let numerical = (objective(&inference, plus) - objective(&inference, minus)) / (2. * epsilon);
    let analytic = dx
        .iter()
        .zip(&direction)
        .map(|(&x, &d)| f64::from(x) * f64::from(d))
        .sum::<f64>();
    let input_error = (numerical - analytic).abs();
    if !numerical.is_finite()
        || !analytic.is_finite()
        || !input_error.is_finite()
        || input_error > 3.0e-3 + 0.03 * analytic.abs()
    {
        return Err(format!(
            "layer input directional gradient {kind:?}/{policy:?}: {analytic} != {numerical}"
        ));
    }
    let mut plus = inference.clone();
    let mut minus = inference.clone();
    let delta = Tensor::zeros(weight.dims(), device)
        .slice_assign([0..1, 0..1], Tensor::full([1, 1], epsilon, device));
    plus.q.weight = Param::from_tensor(inference.q.weight.val() + delta.clone());
    minus.q.weight = Param::from_tensor(inference.q.weight.val() - delta);
    let numerical =
        (objective(&plus, data.clone()) - objective(&minus, data.clone())) / (2. * epsilon);
    let parameter_error = (numerical - f64::from(dw[0])).abs();
    if !numerical.is_finite()
        || !parameter_error.is_finite()
        || parameter_error > 3.0e-3 + 0.03 * f64::from(dw[0]).abs()
    {
        return Err(format!(
            "layer query weight gradient {kind:?}/{policy:?}: {} != {numerical}",
            dw[0]
        ));
    }
    let mut cache_errors = Vec::new();
    if let Some(cache) = &model.cache {
        for field in 0..4 {
            let (name, parameter) = match field {
                0 => ("cache_q_norm.gamma", cache.q_norm.gamma.val()),
                1 => ("cache_k_norm.gamma", cache.k_norm.gamma.val()),
                2 => ("cache_gate", cache.gate.val()),
                _ => ("cache_sink", cache.sink.val()),
            };
            let derivative = values(
                parameter
                    .grad(&gradients)
                    .ok_or_else(|| format!("missing {name} gradient"))?,
            )[0];
            let mut plus = inference.clone();
            let mut minus = inference.clone();
            let adjust = |model: &mut crate::Mixer, amount: f64| {
                let cache = model.cache.as_mut().unwrap();
                let parameter = match field {
                    0 => &mut cache.q_norm.gamma,
                    1 => &mut cache.k_norm.gamma,
                    2 => &mut cache.gate,
                    _ => &mut cache.sink,
                };
                let current = parameter.val();
                let delta = Tensor::zeros(current.dims(), device)
                    .slice_assign([0..1], Tensor::full([1], amount, device));
                *parameter = Param::from_tensor(current + delta);
            };
            adjust(&mut plus, epsilon);
            adjust(&mut minus, -epsilon);
            let numerical =
                (objective(&plus, data.clone()) - objective(&minus, data.clone())) / (2. * epsilon);
            let error = (numerical - f64::from(derivative)).abs();
            if !derivative.is_finite()
                || !numerical.is_finite()
                || !error.is_finite()
                || error > 3.0e-3 + 0.03 * f64::from(derivative).abs()
            {
                return Err(format!(
                    "{name} gradient {kind:?}/{policy:?}: {derivative} != {numerical}"
                ));
            }
            cache_errors.push(CacheParameterCheck {
                parameter: name,
                absolute_error: error,
            });
        }
    }
    Ok(LayerCheck {
        mixer: kind,
        cache: policy,
        scan_backend,
        hola_backend,
        input_directional_gradient_error: input_error,
        projection_gradient_error: parameter_error,
        cache_parameter_gradient_errors: cache_errors,
    })
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn custom_gdn_check(
    device: &Device,
    time: usize,
    include_state: bool,
) -> Result<RecurrenceCheck, String> {
    let fixture = Fixture::new(MixerKind::Gdn2, time);
    let expected = fixture.oracle(MixerKind::Gdn2, include_state);
    // Read every final-state row with identity queries and zero updates. These
    // 32 diagnostic tokens expose the custom operator's state without a second
    // implementation reconstructing it from the tensor solver's intermediates.
    let extended = time + if include_state { fixture.k } else { 0 };
    let mut starts = vec![1i32; fixture.b * extended];
    let mut ends = vec![0i32; fixture.b * extended];
    for row in 0..fixture.b {
        let mut start = 0;
        for p in 0..time {
            if !fixture.valid[row * time + p] {
                continue;
            }
            if fixture.reset[row * time + p] {
                start = p;
            }
            starts[row * extended + p] = start as i32;
            let next = (p + 1..time)
                .find(|&n| fixture.valid[row * time + n] && fixture.reset[row * time + n]);
            ends[row * extended + p] = match next {
                Some(next) => (p..next)
                    .rev()
                    .find(|&n| fixture.valid[row * time + n])
                    .unwrap() as i32,
                None => (extended - 1) as i32,
            };
        }
        for p in time..extended {
            starts[row * extended + p] = start as i32;
            ends[row * extended + p] = (extended - 1) as i32;
        }
    }
    let mut probes = vec![
        0.0f32;
        if include_state {
            fixture.b * fixture.k * fixture.h * fixture.k
        } else {
            0
        }
    ];
    if include_state {
        for row in 0..fixture.b {
            for p in 0..fixture.k {
                for head in 0..fixture.h {
                    probes[((row * fixture.k + p) * fixture.h + head) * fixture.k + p] = 1.;
                }
            }
        }
    }
    let run = || {
        let x = fixture.inputs(device);
        let padded = x
            .iter()
            .enumerate()
            .map(|(f, x)| {
                if !include_state {
                    return x.clone();
                }
                let probe = if f == 0 {
                    Tensor::from_data(
                        TensorData::new(
                            probes.clone(),
                            [fixture.b, fixture.k, fixture.h, fixture.k],
                        ),
                        device,
                    )
                } else if f == 3 {
                    Tensor::ones([fixture.b, fixture.k, fixture.h, fixture.widths[f]], device)
                } else {
                    Tensor::zeros([fixture.b, fixture.k, fixture.h, fixture.widths[f]], device)
                };
                Tensor::cat(vec![x.clone(), probe], 1)
            })
            .collect::<Vec<_>>();
        let all = delta_kernels::gdn2_scan(
            padded[0].clone(),
            padded[1].clone(),
            padded[2].clone(),
            padded[3].clone(),
            padded[4].clone(),
            padded[5].clone(),
            Tensor::from_data(
                TensorData::new(starts.clone(), [fixture.b, extended]),
                device,
            ),
            Tensor::from_data(TensorData::new(ends.clone(), [fixture.b, extended]), device),
        );
        let out = all
            .clone()
            .slice([0..fixture.b, 0..time, 0..fixture.h, 0..fixture.v]);
        let state = include_state.then(|| {
            all.slice([0..fixture.b, time..extended, 0..fixture.h, 0..fixture.v])
                .swap_dims(1, 2)
        });
        let actual = values(out.clone());
        let actual_state = state.as_ref().map(|state| values(state.clone()));
        let mut loss = (out
            * Tensor::from_data(
                TensorData::new(
                    fixture.upstream.clone(),
                    [fixture.b, time, fixture.h, fixture.v],
                ),
                device,
            ))
        .sum();
        if let Some(state) = state {
            loss = loss
                + (state
                    * Tensor::from_data(
                        TensorData::new(
                            fixture.final_adjoint.clone(),
                            [fixture.b, fixture.h, fixture.k, fixture.v],
                        ),
                        device,
                    ))
                .sum();
        }
        let gradients = loss.backward();
        let grads = x
            .iter()
            .map(|x| values(x.grad(&gradients).expect("custom kernel input gradient")))
            .collect::<Vec<_>>();
        (actual, actual_state, grads)
    };
    let first = Instant::now();
    let actual = run();
    let first_ms = first.elapsed().as_secs_f64() * 1000.;
    let warm = Instant::now();
    let again = run();
    let warm_ms = warm.elapsed().as_secs_f64() * 1000.;
    let output = compare("custom CubeCL output", &actual.0, &expected.0, ATOL, RTOL)?;
    let state = actual
        .1
        .as_ref()
        .map(|state| compare("custom CubeCL final state", state, &expected.1, ATOL, RTOL))
        .transpose()?;
    compare(
        "warm custom CubeCL output",
        &again.0,
        &expected.0,
        ATOL,
        RTOL,
    )?;
    let mut gradient = 0.0f64;
    for f in 0..6 {
        gradient = gradient.max(compare(
            &format!("custom CubeCL gradient field {f}"),
            &actual.2[f],
            &expected.3[f],
            ATOL,
            RTOL,
        )?);
    }
    Ok(RecurrenceCheck {
        mixer: MixerKind::Gdn2,
        implementation: "cubecl_chunk_wy",
        fixture: "sinusoidal",
        tokens: time,
        final_state_probe_tokens: if include_state { fixture.k } else { 0 },
        output_max_abs: output,
        state_max_abs: state,
        score_max_abs: None,
        gradient_max_abs: gradient,
        continuation_max_abs: None,
        first_forward_backward_ms: first_ms,
        warm_forward_backward_ms: warm_ms,
    })
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    #[test]
    fn cpu_chunk_recurrence_cache_and_layer_gradients() {
        super::numerical_checks(&burn::prelude::Device::ndarray()).unwrap();
    }
    #[test]
    #[should_panic(expected = "computational chunk_size must be 1..=16")]
    fn unsupported_computational_chunk_is_rejected_before_parameter_initialization() {
        let config = crate::MixerConfig {
            chunk_size: 17,
            ..Default::default()
        };
        config.init(&burn::prelude::Device::ndarray());
    }
    #[test]
    #[should_panic(expected = "no CPU fallback")]
    fn fused_kron_never_falls_back_to_cpu() {
        let config = crate::MixerConfig {
            kind: crate::MixerKind::KronLstm,
            scan_backend: crate::ScanBackend::FusedGpu,
            ..Default::default()
        };
        config.init(&burn::prelude::Device::ndarray());
    }
    #[test]
    #[should_panic(expected = "no CPU fallback")]
    fn fused_hola_never_falls_back_to_cpu() {
        let config = crate::MixerConfig {
            cache_policy: crate::CachePolicy::Surprise,
            hola_backend: crate::HolaBackend::FusedGpu,
            ..Default::default()
        };
        config.init(&burn::prelude::Device::ndarray());
    }
}
