//! Test-only tensor transcription of `reference::step`, including document resets and padding.

use burn::prelude::*;
use dre_core::BatchLayout;

pub(crate) fn scan(
    q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, alpha: Tensor<4>, erase: Tensor<4>, write: Tensor<4>,
    layout: &BatchLayout,
) -> Tensor<4> {
    let [batch, tokens, heads, keys] = q.dims();
    let values = v.dims()[3];
    let device = q.device();
    let mut output = Tensor::zeros([batch, tokens, heads, values], &device);
    for (row, documents) in layout.host.documents_per_row().iter().enumerate() {
        let mut first = 0;
        for &length in documents {
            let mut state = Tensor::<4>::zeros([1, heads, keys, values], &device);
            for token in first..first + length {
                let key = |tensor: &Tensor<4>| tensor.clone()
                    .slice([row..row + 1, token..token + 1, 0..heads, 0..keys])
                    .reshape([1, heads, keys, 1]);
                let value = |tensor: &Tensor<4>| tensor.clone()
                    .slice([row..row + 1, token..token + 1, 0..heads, 0..values])
                    .reshape([1, heads, 1, values]);
                let decayed = key(&alpha) * state;
                let read = (decayed.clone() * key(&erase) * key(&k)).sum_dim(2);
                let delta = value(&write) * value(&v) - read;
                state = decayed + key(&k) * delta;
                let result = (state.clone() * key(&q)).sum_dim(2).swap_dims(1, 2);
                output = output.slice_assign([row..row + 1, token..token + 1, 0..heads, 0..values], result);
            }
            first += length;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{self, Token, HEADS, KEY_DIM, VALUE_DIM};
    use dre_core::DocumentLayout;

    #[test]
    fn cpu_oracle_matches_f64_forward_and_backward_packed_resets_and_padding() {
        let device = Device::ndarray().autodiff();
        let (batch, tokens) = (2, 4);
        let layout = BatchLayout::new(DocumentLayout::packed(vec![vec![2, 1], vec![1, 2]], tokens), &device);
        let widths = [KEY_DIM, KEY_DIM, VALUE_DIM, KEY_DIM, KEY_DIM, VALUE_DIM];
        let data: Vec<Vec<f32>> = widths.iter().enumerate().map(|(group, &width)| {
            (0..batch * tokens * HEADS * width).map(|index| {
                let signal = ((index * 13 + group * 19 + 1) as f32 * 0.037).sin();
                if group == 3 { 0.83 + signal * 0.05 }
                else if group >= 4 { 0.45 + signal * 0.1 }
                else { signal * 0.13 }
            }).collect()
        }).collect();
        let inputs: Vec<Tensor<4>> = data.iter().zip(widths).map(|(data, width)| {
            Tensor::from_data(TensorData::new(data.clone(), [batch, tokens, HEADS, width]), &device).require_grad()
        }).collect();
        let upstream: Vec<f32> = (0..batch * tokens * HEADS * VALUE_DIM)
            .map(|index| ((index * 7 + 3) as f32 * 0.051).cos()).collect();
        let output = scan(inputs[0].clone(), inputs[1].clone(), inputs[2].clone(), inputs[3].clone(), inputs[4].clone(), inputs[5].clone(), &layout);
        let actual_output = output.clone().into_data().try_to_vec::<f32>().unwrap();
        let grads = (output * Tensor::from_data(TensorData::new(upstream.clone(), [batch, tokens, HEADS, VALUE_DIM]), &device)).sum().backward();
        let actual_gradients: Vec<Vec<f32>> = inputs.iter().map(|input| input.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap()).collect();
        let (starts, ends) = layout.host.limits_host();
        let mut expected_output = vec![0.0; actual_output.len()];
        let mut expected_gradients: Vec<Vec<f64>> = data.iter().map(|data| vec![0.0; data.len()]).collect();
        for row in 0..batch {
            for head in 0..HEADS {
                let field = |group: usize, token: usize| {
                    let first = ((row * tokens + token) * HEADS + head) * widths[group];
                    data[group][first..first + widths[group]].iter().map(|&v| f64::from(v)).collect()
                };
                let reference_tokens: Vec<_> = (0..tokens).map(|token| Token::new(
                    field(0, token), field(1, token), field(2, token), field(3, token), field(4, token), field(5, token),
                )).collect();
                let limits = row * tokens..(row + 1) * tokens;
                let reference_output = reference::scan(&reference_tokens, &starts[limits.clone()], &ends[limits.clone()]);
                let grad: Vec<Vec<f64>> = (0..tokens).map(|token| {
                    let first = ((row * tokens + token) * HEADS + head) * VALUE_DIM;
                    upstream[first..first + VALUE_DIM].iter().map(|&v| f64::from(v)).collect()
                }).collect();
                let reference_gradients = reference::backward(&reference_tokens, &starts[limits.clone()], &ends[limits], &grad);
                for token in 0..tokens {
                    let first = ((row * tokens + token) * HEADS + head) * VALUE_DIM;
                    expected_output[first..first + VALUE_DIM].copy_from_slice(&reference_output.output[token]);
                    let gradient = &reference_gradients[token];
                    for (group, values) in [&gradient.q, &gradient.k, &gradient.v, &gradient.alpha, &gradient.erase, &gradient.write].into_iter().enumerate() {
                        let first = ((row * tokens + token) * HEADS + head) * widths[group];
                        expected_gradients[group][first..first + widths[group]].copy_from_slice(values);
                    }
                }
            }
        }
        let compare = |name: &str, actual: &[f32], expected: &[f64], width: usize| {
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert!((f64::from(actual) - expected).abs() <= 1e-6 + 1e-4 * expected.abs(), "{name}[{index}]: {actual} != {expected}");
                if (index / (HEADS * width)) % tokens == 3 { assert_eq!(actual, 0.0, "{name} padding"); }
            }
        };
        compare("output", &actual_output, &expected_output, VALUE_DIM);
        for group in 0..6 { compare(&format!("gradient {group}"), &actual_gradients[group], &expected_gradients[group], widths[group]); }
    }
}
