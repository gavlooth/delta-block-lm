//! Scalar `f64` oracle for exact noisy-block attention and its full input VJP.
//!
//! A valid noisy query reads valid noisy keys with the same block and document IDs,
//! including keys later in its block. Optional prefix slots are snapshots supplied
//! separately for each query; only valid slots in that query's document are read.
//! The caller is responsible for taking each snapshot at the block's start, before
//! processing any noisy position. Prefix slots and noisy keys share one softmax.
//! An optional recurrent prefix readout is added after attention.
//!
//! Invalid queries and rows with no visible keys produce exactly zero, including
//! the readout contribution, and have exactly zero gradients. Masked input values
//! are not read. IDs, masks, and scale are held fixed by the VJP. Work is quadratic
//! in the noisy sequence length; this is a correctness oracle, not a training path.

use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub const ABSOLUTE_TOLERANCE: f64 = 2e-5;
pub const RELATIVE_TOLERANCE: f64 = 3e-4;

#[derive(Clone, Copy, Debug)]
pub struct SbdShape {
    pub batch: usize,
    pub heads: usize,
    pub time: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub prefix_slots: usize,
}

/// All arrays are flat, row-major slices. Prefix arrays are either all present or
/// all absent; a present readout is independent of the presence of prefix slots.
#[derive(Clone, Copy, Debug)]
pub struct SbdInput<'a> {
    pub shape: SbdShape,
    /// `[B, H, T, K]`.
    pub queries: &'a [f64],
    /// `[B, H, T, K]`.
    pub keys: &'a [f64],
    /// `[B, H, T, V]`.
    pub values: &'a [f64],
    /// `[B, T]`. IDs need not be consecutive or imply fixed-length blocks.
    pub block_ids: &'a [usize],
    /// `[B, T]`. Applies to queries and noisy keys alike.
    pub valid: &'a [bool],
    /// `[B, T]`. A block crossing documents is split by document identity.
    pub document_ids: &'a [usize],
    /// Optional `[B, H, T, P, K]` block-start prefix key snapshots.
    pub prefix_keys: Option<&'a [f64]>,
    /// Optional `[B, H, T, P, V]` block-start prefix value snapshots.
    pub prefix_values: Option<&'a [f64]>,
    /// Optional `[B, T, P]` prefix-slot validity, shared across heads.
    pub prefix_valid: Option<&'a [bool]>,
    /// Optional `[B, T, P]` prefix-slot document identity, shared across heads.
    pub prefix_document_ids: Option<&'a [usize]>,
    /// Optional `[B, H, T, V]` additive recurrent prefix readout.
    pub prefix_readout: Option<&'a [f64]>,
    pub scale: f64,
}

/// Gradients have the same layouts as the corresponding [`SbdInput`] fields.
/// Absent optional inputs have absent gradients. Repeated prefix snapshots are
/// independent inputs; callers sharing their source must sum their gradients.
#[derive(Clone, Debug)]
pub struct SbdGradients {
    pub queries: Vec<f64>,
    pub keys: Vec<f64>,
    pub values: Vec<f64>,
    pub prefix_keys: Option<Vec<f64>>,
    pub prefix_values: Option<Vec<f64>>,
    pub prefix_readout: Option<Vec<f64>>,
}

impl SbdInput<'_> {
    fn validate(&self) {
        let s = self.shape;
        let positions = s.batch * s.time;
        let rows = positions * s.heads;
        assert_eq!(self.queries.len(), rows * s.key_dim, "queries shape");
        assert_eq!(self.keys.len(), rows * s.key_dim, "keys shape");
        assert_eq!(self.values.len(), rows * s.value_dim, "values shape");
        assert_eq!(self.block_ids.len(), positions, "block_ids shape");
        assert_eq!(self.valid.len(), positions, "valid shape");
        assert_eq!(self.document_ids.len(), positions, "document_ids shape");
        match (
            self.prefix_keys,
            self.prefix_values,
            self.prefix_valid,
            self.prefix_document_ids,
        ) {
            (Some(k), Some(v), Some(valid), Some(docs)) => {
                assert_eq!(k.len(), rows * s.prefix_slots * s.key_dim, "prefix_keys shape");
                assert_eq!(v.len(), rows * s.prefix_slots * s.value_dim, "prefix_values shape");
                assert_eq!(valid.len(), positions * s.prefix_slots, "prefix_valid shape");
                assert_eq!(docs.len(), positions * s.prefix_slots, "prefix_document_ids shape");
            }
            (None, None, None, None) => {}
            _ => panic!("prefix keys, values, validity and document IDs must be supplied together"),
        }
        if let Some(readout) = self.prefix_readout {
            assert_eq!(readout.len(), rows * s.value_dim, "prefix_readout shape");
        }
    }

    /// Reuses one scratch row, first for logits and then probabilities. Masked
    /// entries become zero without touching their keys, values or metadata IDs.
    fn probabilities(&self, b: usize, h: usize, t: usize, row: &mut [f64]) -> bool {
        let s = self.shape;
        row.fill(f64::NEG_INFINITY);
        let position = b * s.time + t;
        if !self.valid[position] {
            row.fill(0.0);
            return false;
        }
        let query = ((b * s.heads + h) * s.time + t) * s.key_dim;
        let mut maximum = f64::NEG_INFINITY;
        for (j, logit) in row[..s.time].iter_mut().enumerate() {
            let key_position = b * s.time + j;
            if self.valid[key_position]
                && self.block_ids[key_position] == self.block_ids[position]
                && self.document_ids[key_position] == self.document_ids[position]
            {
                let key = ((b * s.heads + h) * s.time + j) * s.key_dim;
                let dot = (0..s.key_dim)
                    .map(|d| self.queries[query + d] * self.keys[key + d])
                    .sum::<f64>();
                *logit = self.scale * dot;
                maximum = maximum.max(*logit);
            }
        }
        if let (Some(keys), Some(valid), Some(docs)) = (
            self.prefix_keys,
            self.prefix_valid,
            self.prefix_document_ids,
        ) {
            for p in 0..s.prefix_slots {
                let meta = position * s.prefix_slots + p;
                if valid[meta] && docs[meta] == self.document_ids[position] {
                    let key = (((b * s.heads + h) * s.time + t) * s.prefix_slots + p)
                        * s.key_dim;
                    let dot = (0..s.key_dim)
                        .map(|d| self.queries[query + d] * keys[key + d])
                        .sum::<f64>();
                    row[s.time + p] = self.scale * dot;
                    maximum = maximum.max(row[s.time + p]);
                }
            }
        }
        if maximum == f64::NEG_INFINITY {
            row.fill(0.0);
            return false;
        }
        let mut denominator = 0.0;
        for probability in row.iter_mut() {
            *probability = (*probability - maximum).exp();
            denominator += *probability;
        }
        for probability in row {
            *probability /= denominator;
        }
        true
    }
}

/// Returns `[B, H, T, V]`. Softmax uses max subtraction over visible entries only.
pub fn forward(input: &SbdInput<'_>) -> Vec<f64> {
    input.validate();
    let s = input.shape;
    let mut output = vec![0.0; s.batch * s.heads * s.time * s.value_dim];
    let mut probabilities = vec![0.0; s.time + s.prefix_slots];
    for b in 0..s.batch {
        for h in 0..s.heads {
            for t in 0..s.time {
                if !input.probabilities(b, h, t, &mut probabilities) {
                    continue;
                }
                let out = ((b * s.heads + h) * s.time + t) * s.value_dim;
                for (j, &p) in probabilities[..s.time].iter().enumerate() {
                    if p == 0.0 {
                        continue;
                    }
                    let value = ((b * s.heads + h) * s.time + j) * s.value_dim;
                    for d in 0..s.value_dim {
                        output[out + d] += p * input.values[value + d];
                    }
                }
                if let Some(values) = input.prefix_values {
                    for p in 0..s.prefix_slots {
                        let probability = probabilities[s.time + p];
                        if probability == 0.0 {
                            continue;
                        }
                        let value = (((b * s.heads + h) * s.time + t) * s.prefix_slots + p)
                            * s.value_dim;
                        for d in 0..s.value_dim {
                            output[out + d] += probability * values[value + d];
                        }
                    }
                }
                if let Some(readout) = input.prefix_readout {
                    for d in 0..s.value_dim {
                        output[out + d] += readout[out + d];
                    }
                }
            }
        }
    }
    output
}

/// Computes `d(sum(forward(input) * output_gradient))` for every float input.
/// The scale, block IDs, document IDs and validity masks are constants.
pub fn vjp(input: &SbdInput<'_>, output_gradient: &[f64]) -> SbdGradients {
    input.validate();
    let s = input.shape;
    assert_eq!(output_gradient.len(), input.values.len(), "output_gradient shape");
    let mut gradient = SbdGradients {
        queries: vec![0.0; input.queries.len()],
        keys: vec![0.0; input.keys.len()],
        values: vec![0.0; input.values.len()],
        prefix_keys: input.prefix_keys.map(|x| vec![0.0; x.len()]),
        prefix_values: input.prefix_values.map(|x| vec![0.0; x.len()]),
        prefix_readout: input.prefix_readout.map(|x| vec![0.0; x.len()]),
    };
    let mut probabilities = vec![0.0; s.time + s.prefix_slots];
    let mut probability_gradient = vec![0.0; probabilities.len()];
    for b in 0..s.batch {
        for h in 0..s.heads {
            for t in 0..s.time {
                if !input.probabilities(b, h, t, &mut probabilities) {
                    continue;
                }
                let row = (b * s.heads + h) * s.time + t;
                let query = row * s.key_dim;
                let out = row * s.value_dim;
                probability_gradient.fill(0.0);
                let mut mean = 0.0;
                for (j, &p) in probabilities[..s.time].iter().enumerate() {
                    if p == 0.0 {
                        continue;
                    }
                    let value = ((b * s.heads + h) * s.time + j) * s.value_dim;
                    let dot = (0..s.value_dim)
                        .map(|d| output_gradient[out + d] * input.values[value + d])
                        .sum::<f64>();
                    probability_gradient[j] = dot;
                    mean += p * dot;
                    for d in 0..s.value_dim {
                        gradient.values[value + d] += p * output_gradient[out + d];
                    }
                }
                if let (Some(values), Some(dv)) = (input.prefix_values, gradient.prefix_values.as_mut()) {
                    for p in 0..s.prefix_slots {
                        let probability = probabilities[s.time + p];
                        if probability == 0.0 {
                            continue;
                        }
                        let value = (row * s.prefix_slots + p) * s.value_dim;
                        let dot = (0..s.value_dim)
                            .map(|d| output_gradient[out + d] * values[value + d])
                            .sum::<f64>();
                        probability_gradient[s.time + p] = dot;
                        mean += probability * dot;
                        for d in 0..s.value_dim {
                            dv[value + d] += probability * output_gradient[out + d];
                        }
                    }
                }
                for (j, &p) in probabilities[..s.time].iter().enumerate() {
                    if p == 0.0 {
                        continue;
                    }
                    let key = ((b * s.heads + h) * s.time + j) * s.key_dim;
                    let ds = input.scale * p * (probability_gradient[j] - mean);
                    for d in 0..s.key_dim {
                        gradient.queries[query + d] += ds * input.keys[key + d];
                        gradient.keys[key + d] += ds * input.queries[query + d];
                    }
                }
                if let (Some(keys), Some(dk)) = (input.prefix_keys, gradient.prefix_keys.as_mut()) {
                    for p in 0..s.prefix_slots {
                        let probability = probabilities[s.time + p];
                        if probability == 0.0 {
                            continue;
                        }
                        let key = (row * s.prefix_slots + p) * s.key_dim;
                        let ds = input.scale * probability * (probability_gradient[s.time + p] - mean);
                        for d in 0..s.key_dim {
                            gradient.queries[query + d] += ds * keys[key + d];
                            dk[key + d] += ds * input.queries[query + d];
                        }
                    }
                }
                if let Some(dr) = gradient.prefix_readout.as_mut() {
                    dr[out..out + s.value_dim]
                        .copy_from_slice(&output_gradient[out..out + s.value_dim]);
                }
            }
        }
    }
    gradient
}

/// Compares GPU `f32` readbacks (or `f64` arrays) without narrowing the reference.
/// Returns the repository's JSON max-error/tolerance report; shape mismatches,
/// nonfinite elements and any element outside `2e-5 + 3e-4 * |reference|` fail.
pub fn compare<T: Copy + Into<f64>>(actual: &[T], reference: &[f64]) -> Result<Value> {
    ensure!(actual.len() == reference.len(), "SBD comparison lengths differ");
    let mut maximum_absolute_error = 0.0f64;
    let mut maximum_tolerance_ratio = 0.0f64;
    for (i, (&actual, &reference)) in actual.iter().zip(reference).enumerate() {
        let actual = actual.into();
        let error = (actual - reference).abs();
        let ratio = error / (ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * reference.abs());
        ensure!(
            actual.is_finite() && reference.is_finite() && ratio <= 1.0,
            "SBD element {i}: {actual} vs {reference}, absolute error {error}, tolerance ratio {ratio}"
        );
        maximum_absolute_error = maximum_absolute_error.max(error);
        maximum_tolerance_ratio = maximum_tolerance_ratio.max(ratio);
    }
    Ok(json!({
        "passed": true,
        "compared_elements": reference.len(),
        "max_abs": maximum_absolute_error,
        "maximum_tolerance_ratio": maximum_tolerance_ratio,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
    }))
}

/// Runs the production tensor attention and masked recurrent-readout addition
/// against the scalar oracle, including all four production input gradients.
/// Prefix K/V are not inputs to the production active-block helper.
pub fn numerical_checks(device: &burn::prelude::Device) -> Result<Value> {
    use burn::{prelude::*, tensor::TensorData};

    fn upload<const D: usize>(x: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
        Tensor::from_data(
            TensorData::new(x.iter().map(|&v| v as f32).collect::<Vec<_>>(), shape),
            device,
        )
    }
    fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
        x.into_data().try_to_vec::<f32>().unwrap()
    }

    let device = device.clone().autodiff();
    let mut checks = Vec::new();
    for (case, time) in [5, 1, 3].into_iter().enumerate() {
        let shape = SbdShape {
            batch: 2,
            heads: 2,
            time,
            key_dim: 3,
            value_dim: 2,
            prefix_slots: 0,
        };
        let mut data = Case::random(shape, 817 + case as u64, false, true);
        data.block_ids.fill(0);
        for b in 0..shape.batch {
            for t in 0..time {
                data.document_ids[b * time + t] = b;
                data.valid[b * time + t] = match case {
                    0 => (b + t) % 3 != 1,
                    1 => b == 0,
                    _ => false,
                };
            }
        }
        // The oracle sees exactly the same representable inputs as the backend.
        for field in FloatField::ALL {
            if let Some(x) = data.field_mut(field) {
                for value in x {
                    *value = *value as f32 as f64;
                }
            }
        }
        let upstream = random_values(data.values.len(), &mut (991 + case as u64))
            .into_iter().map(|x| x as f32 as f64).collect::<Vec<_>>();
        let input = data.input();
        let expected = forward(&input);
        let expected_gradient = vjp(&input, &upstream);
        let q = upload(&data.queries, [2, 2, time, 3], &device).require_grad();
        let k = upload(&data.keys, [2, 2, time, 3], &device).require_grad();
        let v = upload(&data.values, [2, 2, time, 2], &device).require_grad();
        let readout = upload(data.prefix_readout.as_ref().unwrap(), [2, 2, time, 2], &device)
            .require_grad();
        let valid_values = data.valid.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect::<Vec<_>>();
        let valid = upload(&valid_values, [2, time], &device);
        let output = delta_mixers::sbd::attention(
            q.clone().swap_dims(1, 2),
            k.clone().swap_dims(1, 2),
            v.clone().swap_dims(1, 2),
            valid.clone(),
        ).swap_dims(1, 2) + readout.clone() * valid.reshape([2, 1, time, 1]);
        let output_check = compare(&values(output.clone()), &expected)?;
        let gradients = (output * upload(&upstream, [2, 2, time, 2], &device)).sum().backward();
        let mut gradient_checks = Vec::new();
        for (name, tensor, expected) in [
            ("queries", q, expected_gradient.queries.as_slice()),
            ("keys", k, expected_gradient.keys.as_slice()),
            ("values", v, expected_gradient.values.as_slice()),
            ("prefix_readout", readout, expected_gradient.prefix_readout.as_ref().unwrap().as_slice()),
        ] {
            let gradient = tensor.grad(&gradients)
                .ok_or_else(|| anyhow::anyhow!("SBD {name}: missing input gradient"))?;
            gradient_checks.push(json!({"input": name, "comparison": compare(&values(gradient), expected)?}));
        }
        checks.push(json!({
            "shape": [2, 2, time, 3, 2],
            "layout": "B,H,T,K/V; production helper receives B,T,H,K/V",
            "case": (["partial_padding", "singleton_and_empty", "all_masked"][case]),
            "output": output_check,
            "gradients": gradient_checks,
        }));
    }
    Ok(json!({
        "passed": true,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "cases": checks,
    }))
}

// Small deterministic inputs are shared by the tensor comparison and scalar tests.
// No random-library dependency or backend random state is needed.
fn random_values(len: usize, state: &mut u64) -> Vec<f64> {
    (0..len).map(|_| {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1.4
    }).collect()
}

#[derive(Clone)]
struct Case {
    shape: SbdShape,
    queries: Vec<f64>,
    keys: Vec<f64>,
    values: Vec<f64>,
    block_ids: Vec<usize>,
    valid: Vec<bool>,
    document_ids: Vec<usize>,
    prefix_keys: Option<Vec<f64>>,
    prefix_values: Option<Vec<f64>>,
    prefix_valid: Option<Vec<bool>>,
    prefix_document_ids: Option<Vec<usize>>,
    prefix_readout: Option<Vec<f64>>,
    scale: f64,
}

#[derive(Clone, Copy, Debug)]
enum FloatField { Queries, Keys, Values, PrefixKeys, PrefixValues, Readout }

impl FloatField {
    const ALL: [Self; 6] = [Self::Queries, Self::Keys, Self::Values, Self::PrefixKeys, Self::PrefixValues, Self::Readout];
}

impl Case {
    fn random(shape: SbdShape, mut seed: u64, prefix: bool, readout: bool) -> Self {
        let positions = shape.batch * shape.time;
        let rows = positions * shape.heads;
        let document_ids = (0..positions).map(|i| i / shape.time * 3 + (i % shape.time >= 3) as usize).collect::<Vec<_>>();
        Self {
            shape,
            queries: random_values(rows * shape.key_dim, &mut seed),
            keys: random_values(rows * shape.key_dim, &mut seed),
            values: random_values(rows * shape.value_dim, &mut seed),
            block_ids: (0..positions).map(|i| (i % shape.time) / 2 * 7 + 2).collect(),
            valid: (0..positions).map(|i| i % 5 != 1).collect(),
            document_ids: document_ids.clone(),
            prefix_keys: prefix.then(|| random_values(rows * shape.prefix_slots * shape.key_dim, &mut seed)),
            prefix_values: prefix.then(|| random_values(rows * shape.prefix_slots * shape.value_dim, &mut seed)),
            prefix_valid: prefix.then(|| (0..positions * shape.prefix_slots).map(|i| i % 3 != 1).collect()),
            prefix_document_ids: prefix.then(|| (0..positions * shape.prefix_slots).map(|i| {
                document_ids[i / shape.prefix_slots] + (i % shape.prefix_slots == 2) as usize
            }).collect()),
            prefix_readout: readout.then(|| random_values(rows * shape.value_dim, &mut seed)),
            scale: 1.0 / (shape.key_dim as f64).sqrt(),
        }
    }

    fn input(&self) -> SbdInput<'_> {
        SbdInput {
            shape: self.shape,
            queries: &self.queries,
            keys: &self.keys,
            values: &self.values,
            block_ids: &self.block_ids,
            valid: &self.valid,
            document_ids: &self.document_ids,
            prefix_keys: self.prefix_keys.as_deref(),
            prefix_values: self.prefix_values.as_deref(),
            prefix_valid: self.prefix_valid.as_deref(),
            prefix_document_ids: self.prefix_document_ids.as_deref(),
            prefix_readout: self.prefix_readout.as_deref(),
            scale: self.scale,
        }
    }

    fn field_mut(&mut self, field: FloatField) -> Option<&mut Vec<f64>> {
        match field {
            FloatField::Queries => Some(&mut self.queries),
            FloatField::Keys => Some(&mut self.keys),
            FloatField::Values => Some(&mut self.values),
            FloatField::PrefixKeys => self.prefix_keys.as_mut(),
            FloatField::PrefixValues => self.prefix_values.as_mut(),
            FloatField::Readout => self.prefix_readout.as_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: &[f64], expected: &[f64], absolute: f64, relative: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(a.is_finite() && e.is_finite() && (a - e).abs() <= absolute + relative * e.abs(),
                "element {i}: {a} vs {e}");
        }
    }

    fn shape(time: usize, prefix_slots: usize) -> SbdShape {
        SbdShape { batch: 2, heads: 2, time, key_dim: 3, value_dim: 2, prefix_slots }
    }

    fn gradient_field(g: &SbdGradients, field: FloatField) -> Option<&[f64]> {
        match field {
            FloatField::Queries => Some(&g.queries),
            FloatField::Keys => Some(&g.keys),
            FloatField::Values => Some(&g.values),
            FloatField::PrefixKeys => g.prefix_keys.as_deref(),
            FloatField::PrefixValues => g.prefix_values.as_deref(),
            FloatField::Readout => g.prefix_readout.as_deref(),
        }
    }

    #[test]
    fn central_finite_differences_full_vjp_random_shapes() {
        for (time, prefix, readout, scale) in [(5, true, true, 0.73), (4, false, false, -0.41), (1, true, false, 0.0)] {
            let mut case = Case::random(shape(time, 3), 1903 + time as u64, prefix, readout);
            case.scale = scale;
            // Force a block across a document boundary and variable block lengths.
            if time > 3 {
                case.block_ids[..time].fill(37);
                case.valid[1] = true;
            }
            let upstream = random_values(case.values.len(), &mut 8103);
            let analytic = vjp(&case.input(), &upstream);
            for field in FloatField::ALL {
                let Some(expected) = gradient_field(&analytic, field) else { continue };
                for (i, &expected) in expected.iter().enumerate() {
                    let original = case.field_mut(field).unwrap()[i];
                    let epsilon = 1e-6;
                    case.field_mut(field).unwrap()[i] = original + epsilon;
                    let plus: f64 = forward(&case.input()).iter().zip(&upstream).map(|(x, g)| x * g).sum();
                    case.field_mut(field).unwrap()[i] = original - epsilon;
                    let minus: f64 = forward(&case.input()).iter().zip(&upstream).map(|(x, g)| x * g).sum();
                    case.field_mut(field).unwrap()[i] = original;
                    let numerical = (plus - minus) / (2.0 * epsilon);
                    assert!((numerical - expected).abs() <= 3e-9 + 3e-6 * expected.abs(),
                        "{field:?}[{i}], T={time}: analytic {expected}, central difference {numerical}");
                }
            }
        }
    }

    #[test]
    fn other_blocks_documents_and_padding_are_exactly_local() {
        let mut case = Case::random(SbdShape { batch: 1, heads: 1, time: 6, key_dim: 2, value_dim: 2, prefix_slots: 3 }, 71, true, true);
        case.block_ids = vec![4, 4, 9, 9, 4, 4];
        case.document_ids = vec![1, 1, 1, 1, 2, 2];
        case.valid = vec![true, true, true, false, true, false];
        case.prefix_valid.as_mut().unwrap()[..3].copy_from_slice(&[true, false, true]);
        case.prefix_document_ids.as_mut().unwrap()[..3].copy_from_slice(&[1, 1, 2]);
        let original = forward(&case.input());
        // Other/later block, another document in the same block, and padding.
        for t in [2, 3, 4, 5] {
            for d in 0..2 {
                case.keys[t * 2 + d] = 9000.0 + d as f64;
                case.values[t * 2 + d] = -8000.0 - d as f64;
            }
            assert_eq!(&forward(&case.input())[..2], &original[..2]);
        }
        for p in [1, 2] {
            for d in 0..2 {
                case.prefix_keys.as_mut().unwrap()[p * 2 + d] = f64::NAN;
                case.prefix_values.as_mut().unwrap()[p * 2 + d] = f64::NAN;
            }
        }
        assert_eq!(&forward(&case.input())[..2], &original[..2]);
        let mut upstream = vec![0.0; case.values.len()];
        upstream[..2].copy_from_slice(&[0.3, -0.7]);
        let g = vjp(&case.input(), &upstream);
        assert!(g.keys[4..].iter().chain(&g.values[4..]).all(|&x| x == 0.0));
        assert!(g.prefix_keys.as_ref().unwrap()[2..6].iter().chain(&g.prefix_values.as_ref().unwrap()[2..6]).all(|&x| x == 0.0));
    }

    #[test]
    fn bidirectional_within_block_and_split_at_document_boundary() {
        let mut case = Case::random(SbdShape { batch: 1, heads: 1, time: 5, key_dim: 1, value_dim: 1, prefix_slots: 0 }, 3, false, false);
        case.queries.fill(0.0);
        case.keys.fill(0.0);
        case.values = vec![2.0, 6.0, 50.0, 10.0, 14.0];
        case.valid.fill(true);
        case.block_ids = vec![8, 8, 7, 8, 8];
        case.document_ids = vec![1, 1, 1, 2, 2];
        assert_eq!(forward(&case.input()), vec![4.0, 4.0, 50.0, 12.0, 12.0]);
        case.values[1] = 10.0;
        assert_eq!(forward(&case.input()), vec![6.0, 6.0, 50.0, 12.0, 12.0]);
    }

    fn permute<T: Copy>(x: &[T], s: SbdShape, heads: usize, width: usize, order: &[usize]) -> Vec<T> {
        let mut result = Vec::with_capacity(x.len());
        for b in 0..s.batch {
            for h in 0..heads {
                for &t in order {
                    let start = ((b * heads + h) * s.time + t) * width;
                    result.extend_from_slice(&x[start..start + width]);
                }
            }
        }
        result
    }

    #[test]
    fn time_permutation_preserves_outputs_and_every_gradient() {
        let mut case = Case::random(shape(5, 3), 213, true, true);
        case.valid.fill(true);
        let original = case.clone();
        let s = case.shape;
        let order = [3, 0, 4, 1, 2];
        for field in FloatField::ALL {
            let width = match field {
                FloatField::Queries | FloatField::Keys => s.key_dim,
                FloatField::Values | FloatField::Readout => s.value_dim,
                FloatField::PrefixKeys => s.prefix_slots * s.key_dim,
                FloatField::PrefixValues => s.prefix_slots * s.value_dim,
            };
            let data = case.field_mut(field).unwrap();
            *data = permute(data, s, s.heads, width, &order);
        }
        case.block_ids = permute(&case.block_ids, s, 1, 1, &order);
        case.valid = permute(&case.valid, s, 1, 1, &order);
        case.document_ids = permute(&case.document_ids, s, 1, 1, &order);
        case.prefix_valid = Some(permute(case.prefix_valid.as_ref().unwrap(), s, 1, s.prefix_slots, &order));
        case.prefix_document_ids = Some(permute(case.prefix_document_ids.as_ref().unwrap(), s, 1, s.prefix_slots, &order));
        close(&forward(&case.input()), &permute(&forward(&original.input()), s, s.heads, s.value_dim, &order), 2e-15, 2e-14);
        let upstream = random_values(case.values.len(), &mut 9091);
        let original_g = vjp(&original.input(), &upstream);
        let permuted_g = vjp(&case.input(), &permute(&upstream, s, s.heads, s.value_dim, &order));
        for field in FloatField::ALL {
            let width = match field {
                FloatField::Queries | FloatField::Keys => s.key_dim,
                FloatField::Values | FloatField::Readout => s.value_dim,
                FloatField::PrefixKeys => s.prefix_slots * s.key_dim,
                FloatField::PrefixValues => s.prefix_slots * s.value_dim,
            };
            close(gradient_field(&permuted_g, field).unwrap(),
                &permute(gradient_field(&original_g, field).unwrap(), s, s.heads, width, &order), 2e-15, 2e-14);
        }
    }

    #[test]
    fn all_masked_rows_ignore_prefix_and_readout_and_have_zero_gradients() {
        let mut case = Case::random(shape(4, 3), 1, true, true);
        case.valid.fill(false);
        case.prefix_valid.as_mut().unwrap().fill(true);
        for field in FloatField::ALL {
            case.field_mut(field).unwrap().fill(f64::NAN);
        }
        assert_eq!(forward(&case.input()), vec![0.0; case.values.len()]);
        let g = vjp(&case.input(), &vec![f64::NAN; case.values.len()]);
        for field in FloatField::ALL {
            assert!(gradient_field(&g, field).unwrap().iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn singleton_blocks_return_value_plus_readout_and_exact_vjp() {
        let mut case = Case::random(shape(4, 0), 51, false, true);
        case.valid.fill(true);
        case.block_ids = (0..case.valid.len()).collect();
        let expected = case.values.iter().zip(case.prefix_readout.as_ref().unwrap()).map(|(v, r)| v + r).collect::<Vec<_>>();
        assert_eq!(forward(&case.input()), expected);
        let upstream = random_values(case.values.len(), &mut 711);
        let g = vjp(&case.input(), &upstream);
        assert!(g.queries.iter().chain(&g.keys).all(|&x| x == 0.0));
        assert_eq!(g.values, upstream);
        assert_eq!(g.prefix_readout.as_ref().unwrap(), &upstream);
        assert!(g.prefix_keys.is_none() && g.prefix_values.is_none());
    }

    #[test]
    fn prefix_and_noisy_keys_share_the_same_softmax() {
        let mut case = Case::random(SbdShape { batch: 1, heads: 1, time: 1, key_dim: 1, value_dim: 1, prefix_slots: 3 }, 45, true, false);
        case.valid.fill(true);
        case.queries.fill(0.0);
        case.values[0] = 2.0;
        case.prefix_values = Some(vec![8.0, 200.0, 300.0]);
        case.prefix_valid = Some(vec![true, false, true]);
        case.prefix_document_ids = Some(vec![0, 0, 1]);
        assert_eq!(forward(&case.input()), vec![5.0]);
        let g = vjp(&case.input(), &[1.0]);
        assert_eq!(g.values, vec![0.5]);
        assert_eq!(g.prefix_values.unwrap(), vec![0.5, 0.0, 0.0]);
    }

    #[test]
    fn large_logits_use_stable_softmax() {
        let mut case = Case::random(SbdShape { batch: 1, heads: 1, time: 2, key_dim: 1, value_dim: 1, prefix_slots: 0 }, 5, false, false);
        case.valid.fill(true);
        case.block_ids.fill(0);
        case.queries.fill(1000.0);
        case.keys.fill(1000.0);
        case.values = vec![2.0, 6.0];
        assert_eq!(forward(&case.input()), vec![4.0, 4.0]);
        let g = vjp(&case.input(), &[0.2, -0.4]);
        assert!(g.queries.iter().chain(&g.keys).chain(&g.values).all(|x| x.is_finite()));
        case.keys[1] = -1000.0;
        assert_eq!(forward(&case.input()), vec![2.0, 2.0]);
    }

    #[test]
    fn zero_length_sequence_is_empty() {
        let case = Case::random(shape(0, 0), 5, false, false);
        assert!(forward(&case.input()).is_empty());
        let g = vjp(&case.input(), &[]);
        assert!(g.queries.is_empty() && g.keys.is_empty() && g.values.is_empty());
    }

    #[test]
    fn comparison_reports_errors_and_rejects_nonfinite_or_wrong_shapes() {
        let report = compare(&[1.0001f32, 0.0], &[1.0, 0.0]).unwrap();
        assert!(report["max_abs"].as_f64().unwrap() > 0.0);
        assert_eq!(report["compared_elements"], 2);
        assert!(compare(&[1.01], &[1.0]).is_err());
        assert!(compare(&[f64::NAN], &[0.0]).is_err());
        assert!(compare(&[0.0], &[f64::INFINITY]).is_err());
        assert!(compare(&[0.0], &[]).is_err());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn production_attention_cpu_forward_and_vjp() {
        numerical_checks(&burn::prelude::Device::ndarray()).unwrap();
    }
}
