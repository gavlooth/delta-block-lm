//! Plain `f64` reference of the wavelet mixer, written from `docs/references/wavelet.md`, with its analytic
//! gradient; the PLAN §8.1 comparison rule; a small deterministic random source.
//!
//! Only the db10 coefficient table is shared with the crate (it is checked separately against the db10 filter
//! conditions). Boundary handling, level geometry, gating and gradients are written independently here.

use super::smoothing_filter;

/// Batch geometry and mixer sizes of a test fixture.
#[derive(Debug, Clone)]
pub struct Shape {
    pub documents: Vec<Vec<usize>>,
    pub tokens: usize,
    pub d: usize,
    pub groups: usize,
    pub levels: usize,
}

impl Shape {
    pub fn rows(&self) -> usize {
        self.documents.len() * self.tokens
    }

    pub fn bands(&self) -> usize {
        self.levels + 1
    }

    /// `(first row, length)` of every document in the flat `[B * T]` row space.
    pub fn spans(&self) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        for (row, documents) in self.documents.iter().enumerate() {
            let mut start = row * self.tokens;
            for &length in documents {
                spans.push((start, length));
                start += length;
            }
        }
        spans
    }

    /// `true` at valid positions, `[B * T]`.
    pub fn valid(&self) -> Vec<bool> {
        let mut valid = vec![false; self.rows()];
        for (start, length) in self.spans() {
            valid[start..start + length]
                .iter_mut()
                .for_each(|v| *v = true);
        }
        valid
    }
}

/// Mixer weights, row-major `[input, output]` as in `burn::nn::Linear`.
#[derive(Debug, Clone)]
pub struct Weights {
    pub input: Vec<f64>,
    pub gate: Vec<f64>,
    pub gate_bias: Vec<f64>,
    pub value: Vec<f64>,
    pub output: Vec<f64>,
}

/// Independent causal geometry, zero extension at document start.
fn level_taps(t: usize, _length: usize, dilation: usize) -> impl Iterator<Item = (usize, f64)> {
    smoothing_filter()
        .into_iter()
        .enumerate()
        .filter_map(move |(k, w)| t.checked_sub(k * dilation).map(|source| (source, w)))
}

fn smooth(x: &[f64], length: usize, d: usize, dilation: usize) -> Vec<f64> {
    let mut out = vec![0.0; length * d];
    for t in 0..length {
        for (source, weight) in level_taps(t, length, dilation) {
            for c in 0..d {
                out[t * d + c] += weight * x[source * d + c];
            }
        }
    }
    out
}

fn smooth_transpose(grad: &[f64], length: usize, d: usize, dilation: usize) -> Vec<f64> {
    let mut out = vec![0.0; length * d];
    for t in 0..length {
        for (source, weight) in level_taps(t, length, dilation) {
            for c in 0..d {
                out[source * d + c] += weight * grad[t * d + c];
            }
        }
    }
    out
}

/// Bands `[A_J, D_J, …, D_1]` of one document `x: [length, d]`.
fn document_bands(x: &[f64], length: usize, d: usize, levels: usize) -> Vec<Vec<f64>> {
    let mut approximation = x.to_vec();
    let mut details = Vec::new();
    for level in 0..levels {
        let next = smooth(&approximation, length, d, 1 << level);
        details.push(
            approximation
                .iter()
                .zip(&next)
                .map(|(a, n)| a - n)
                .collect::<Vec<_>>(),
        );
        approximation = next;
    }
    let mut bands = vec![approximation];
    bands.extend(details.into_iter().rev());
    bands
}

/// Gradient with respect to `x` of `Σ_j <grads_j, band_j(x)>` for one document.
fn document_bands_transpose(
    grads: &[Vec<f64>],
    length: usize,
    d: usize,
    levels: usize,
) -> Vec<f64> {
    // grads[0] belongs to A_J, grads[i] to D_{J+1-i}.
    let mut approximation = grads[0].clone();
    for level in (1..=levels).rev() {
        let detail = &grads[levels + 1 - level];
        let through_smoothing: Vec<f64> = approximation
            .iter()
            .zip(detail)
            .map(|(a, g)| a - g)
            .collect();
        let previous = smooth_transpose(&through_smoothing, length, d, 1 << (level - 1));
        approximation = previous.iter().zip(detail).map(|(p, g)| p + g).collect();
    }
    approximation
}

/// Bands of a whole batch `v: [B * T * d]`; padded positions are 0.
pub fn bands(shape: &Shape, v: &[f64]) -> Vec<Vec<f64>> {
    let d = shape.d;
    let mut out = vec![vec![0.0; shape.rows() * d]; shape.bands()];
    for (start, length) in shape.spans() {
        let document = document_bands(&v[start * d..(start + length) * d], length, d, shape.levels);
        for (band, values) in out.iter_mut().zip(document) {
            band[start * d..(start + length) * d].copy_from_slice(&values);
        }
    }
    out
}

fn bands_transpose(shape: &Shape, grads: &[Vec<f64>]) -> Vec<f64> {
    let d = shape.d;
    let mut out = vec![0.0; shape.rows() * d];
    for (start, length) in shape.spans() {
        let document: Vec<Vec<f64>> = grads
            .iter()
            .map(|g| g[start * d..(start + length) * d].to_vec())
            .collect();
        out[start * d..(start + length) * d].copy_from_slice(&document_bands_transpose(
            &document,
            length,
            d,
            shape.levels,
        ));
    }
    out
}

/// `x: [rows, inner]` times `w: [inner, cols]`.
fn matmul(x: &[f64], w: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for i in 0..inner {
            let a = x[r * inner + i];
            if a != 0.0 {
                for c in 0..cols {
                    out[r * cols + c] += a * w[i * cols + c];
                }
            }
        }
    }
    out
}

/// `x^T y` for `x: [rows, a]`, `y: [rows, b]`, giving `[a, b]`.
fn matmul_tn(x: &[f64], y: &[f64], rows: usize, a: usize, b: usize) -> Vec<f64> {
    let mut out = vec![0.0; a * b];
    for r in 0..rows {
        for i in 0..a {
            let xi = x[r * a + i];
            if xi != 0.0 {
                for j in 0..b {
                    out[i * b + j] += xi * y[r * b + j];
                }
            }
        }
    }
    out
}

/// `g w^T` for `g: [rows, cols]`, `w: [inner, cols]`, giving `[rows, inner]`.
fn matmul_nt(g: &[f64], w: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * inner];
    for r in 0..rows {
        for i in 0..inner {
            out[r * inner + i] = (0..cols).map(|c| g[r * cols + c] * w[i * cols + c]).sum();
        }
    }
    out
}

/// Forward values kept for the gradient.
pub struct Forward {
    pub u: Vec<f64>,
    pub gates: Vec<f64>,
    pub value: Vec<f64>,
    pub bands: Vec<Vec<f64>>,
    pub mixed: Vec<f64>,
    pub product: Vec<f64>,
    pub out: Vec<f64>,
}

/// Mixer forward on `u: [B * T * d]` (padding rows are zeroed first, NaN included).
pub fn forward(shape: &Shape, w: &Weights, u: &[f64]) -> Forward {
    let (rows, d, groups) = (shape.rows(), shape.d, shape.groups);
    let gate_width = shape.bands() * groups;
    let group_width = d / groups;
    let valid = shape.valid();
    let u: Vec<f64> = u
        .iter()
        .enumerate()
        .map(|(i, &x)| if valid[i / d] { x } else { 0.0 })
        .collect();

    let v = matmul(&u, &w.input, rows, d, d);
    let mut gates = matmul(&u, &w.gate, rows, d, gate_width);
    for (i, g) in gates.iter_mut().enumerate() {
        *g = 1.0 / (1.0 + (-(*g + w.gate_bias[i % gate_width])).exp());
    }
    let value = matmul(&u, &w.value, rows, d, d);
    let bands = bands(shape, &v);

    let mut mixed = vec![0.0; rows * d];
    for r in 0..rows {
        for c in 0..d {
            mixed[r * d + c] = (0..shape.bands())
                .map(|j| gates[r * gate_width + j * groups + c / group_width] * bands[j][r * d + c])
                .sum();
        }
    }
    let product: Vec<f64> = mixed.iter().zip(&value).map(|(m, v)| m * v).collect();
    let out = matmul(&product, &w.output, rows, d, d);
    Forward {
        u,
        gates,
        value,
        bands,
        mixed,
        product,
        out,
    }
}

/// Gradients of `Σ out ⊙ r` with respect to the input and every weight.
pub struct Gradients {
    pub u: Vec<f64>,
    pub weights: Weights,
}

pub fn backward(shape: &Shape, w: &Weights, f: &Forward, r: &[f64]) -> Gradients {
    let (rows, d, groups) = (shape.rows(), shape.d, shape.groups);
    let gate_width = shape.bands() * groups;
    let group_width = d / groups;
    let valid = shape.valid();

    let d_output = matmul_tn(&f.product, r, rows, d, d);
    let d_product = matmul_nt(r, &w.output, rows, d, d);
    let d_mixed: Vec<f64> = d_product.iter().zip(&f.value).map(|(g, v)| g * v).collect();
    let d_value: Vec<f64> = d_product.iter().zip(&f.mixed).map(|(g, m)| g * m).collect();

    let mut d_bands = vec![vec![0.0; rows * d]; shape.bands()];
    let mut d_logits = vec![0.0; rows * gate_width];
    for r_ in 0..rows {
        for (j, d_band) in d_bands.iter_mut().enumerate() {
            for c in 0..d {
                let gate_index = r_ * gate_width + j * groups + c / group_width;
                d_band[r_ * d + c] = f.gates[gate_index] * d_mixed[r_ * d + c];
                d_logits[gate_index] += d_mixed[r_ * d + c] * f.bands[j][r_ * d + c];
            }
        }
    }
    for (g, &gate) in d_logits.iter_mut().zip(&f.gates) {
        *g *= gate * (1.0 - gate);
    }
    let d_v = bands_transpose(shape, &d_bands);

    let d_input = matmul_tn(&f.u, &d_v, rows, d, d);
    let d_gate = matmul_tn(&f.u, &d_logits, rows, d, gate_width);
    let mut d_gate_bias = vec![0.0; gate_width];
    for r_ in 0..rows {
        for i in 0..gate_width {
            d_gate_bias[i] += d_logits[r_ * gate_width + i];
        }
    }
    let d_value_weight = matmul_tn(&f.u, &d_value, rows, d, d);

    let mut d_u = matmul_nt(&d_v, &w.input, rows, d, d);
    for (acc, x) in d_u
        .iter_mut()
        .zip(matmul_nt(&d_logits, &w.gate, rows, d, gate_width))
    {
        *acc += x;
    }
    for (acc, x) in d_u
        .iter_mut()
        .zip(matmul_nt(&d_value, &w.value, rows, d, d))
    {
        *acc += x;
    }
    for (i, g) in d_u.iter_mut().enumerate() {
        if !valid[i / d] {
            *g = 0.0;
        }
    }

    Gradients {
        u: d_u,
        weights: Weights {
            input: d_input,
            gate: d_gate,
            gate_bias: d_gate_bias,
            value: d_value_weight,
            output: d_output,
        },
    }
}

fn loss(shape: &Shape, w: &Weights, u: &[f64], r: &[f64]) -> f64 {
    forward(shape, w, u)
        .out
        .iter()
        .zip(r)
        .map(|(o, r)| o * r)
        .sum()
}

fn parts(u: &[f64], w: &Weights) -> Vec<Vec<f64>> {
    vec![
        u.to_vec(),
        w.input.clone(),
        w.gate.clone(),
        w.gate_bias.clone(),
        w.value.clone(),
        w.output.clone(),
    ]
}

fn from_parts(p: &[Vec<f64>]) -> (Vec<f64>, Weights) {
    (
        p[0].clone(),
        Weights {
            input: p[1].clone(),
            gate: p[2].clone(),
            gate_bias: p[3].clone(),
            value: p[4].clone(),
            output: p[5].clone(),
        },
    )
}

/// PLAN §8.1: checks the analytic gradient by central differences along `directions` random unit directions over
/// the input and all weights. Padded input entries are excluded (NaN there is legal and has zero gradient).
pub fn check_gradient_by_differences(
    shape: &Shape,
    w: &Weights,
    u: &[f64],
    r: &[f64],
    gradients: &Gradients,
    directions: usize,
    rng: &mut Rng,
) {
    let valid = shape.valid();
    let u_clean: Vec<f64> = u
        .iter()
        .enumerate()
        .map(|(i, &x)| if valid[i / shape.d] { x } else { 0.0 })
        .collect();
    let theta = parts(&u_clean, w);
    let grad = parts(&gradients.u, &gradients.weights);
    let count: usize = theta.iter().map(Vec::len).sum();
    let rms_theta = (theta.iter().flatten().map(|x| x * x).sum::<f64>() / count as f64).sqrt();
    let h = 1e-6 * rms_theta.max(1.0);
    let grad_norm = grad.iter().flatten().map(|g| g * g).sum::<f64>().sqrt();
    let g0 = grad_norm;

    for direction in 0..directions {
        let mut v: Vec<Vec<f64>> = theta
            .iter()
            .map(|p| p.iter().map(|_| rng.normal()).collect())
            .collect();
        for (i, x) in v[0].iter_mut().enumerate() {
            if !valid[i / shape.d] {
                *x = 0.0;
            }
        }
        let norm = v.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().flatten().for_each(|x| *x /= norm);

        let shifted = |sign: f64| -> f64 {
            let p: Vec<Vec<f64>> = theta
                .iter()
                .zip(&v)
                .map(|(t, v)| t.iter().zip(v).map(|(t, v)| t + sign * h * v).collect())
                .collect();
            let (u, w) = from_parts(&p);
            loss(shape, &w, &u, r)
        };
        let difference = (shifted(1.0) - shifted(-1.0)) / (2.0 * h);
        let analytic: f64 = grad
            .iter()
            .flatten()
            .zip(v.iter().flatten())
            .map(|(g, v)| g * v)
            .sum();
        let limit = 1e-7 * grad_norm.max(g0) + 1e-6 * difference.abs();
        eprintln!(
            "oracle gradient direction {direction}: analytic {analytic:.12e} differences {difference:.12e} |error| {:.3e} limit {limit:.3e}",
            (analytic - difference).abs()
        );
        assert!(
            (analytic - difference).abs() <= limit,
            "oracle gradient disagrees with central differences"
        );
    }
}

/// SplitMix64 and Box–Muller, copied with the predecessor oracle.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn normal(&mut self) -> f64 {
        let mut uniform = || {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (a, b) = (uniform(), uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}
