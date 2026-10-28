use burn::prelude::*;
use burn::tensor::Int;
use dre_core::gpu_device;
use crate::spike::spike;

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn custom_op_spike() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let device = gpu_device().autodiff();
    let a = (Tensor::<2>::from_data(TensorData::new(vec![0.3f32, -0.7, 1.2, 0.5, -0.2, 0.9], [2, 3]), &device) + 0.0f32).require_grad();
    let b = (Tensor::<2>::from_data(TensorData::new(vec![0.2f32, 0.4, -0.1, 0.8, -0.3, 0.6], [2, 3]), &device) + 0.0f32).require_grad();
    let scale = Tensor::<1, Int>::from_data(TensorData::new(vec![2i32, -3], [2]), &device);
    let (out, row_sum) = spike(a.clone(), b.clone(), scale);
    let loss = (out.clone() * 1.7f32 + 0.25f32).sum() + row_sum.clone().sum() * 0.4f32;
    let gradients = loss.backward();
    device.sync().unwrap();
    let out_values = out.into_data().try_to_vec::<f32>().unwrap();
    let sums = row_sum.into_data().try_to_vec::<f32>().unwrap();
    let a_grad = a.grad(&gradients).unwrap().into_data().try_to_vec::<f32>().unwrap();
    let b_grad = b.grad(&gradients).unwrap().into_data().try_to_vec::<f32>().unwrap();
    let a_ref = [0.3f64, -0.7, 1.2, 0.5, -0.2, 0.9];
    let b_ref = [0.2f64, 0.4, -0.1, 0.8, -0.3, 0.6];
    let scales = [2.0f64, -3.0];
    let expected: Vec<f64> = (0..6).map(|i| a_ref[i] + b_ref[i] * scales[i / 3]).collect();
    for (got, reference) in out_values.iter().zip(&expected) { assert!((*got as f64 - reference).abs() < 1e-5); }
    let expected_sums = [expected[0..3].iter().sum::<f64>(), expected[3..6].iter().sum::<f64>()];
    for (got, reference) in sums.iter().zip(expected_sums) { assert!((*got as f64 - reference).abs() < 1e-5); }
    for (got, expected) in a_grad.iter().zip([2.1f32; 6]) { assert!((got - expected).abs() < 1e-6); }
    for (got, expected) in b_grad.iter().zip([4.2f32, 4.2, 4.2, -6.3, -6.3, -6.3]) { assert!((got - expected).abs() < 1e-6); }
    for direction in 0..8 {
        let da: Vec<f64> = (0..6).map(|i| ((direction * 17 + i * 13 + 3) as f64).sin()).collect();
        let db: Vec<f64> = (0..6).map(|i| ((direction * 11 + i * 19 + 5) as f64).cos()).collect();
        let norm = da.iter().chain(&db).map(|x| x * x).sum::<f64>().sqrt();
        let da: Vec<f64> = da.into_iter().map(|x| x / norm).collect();
        let db: Vec<f64> = db.into_iter().map(|x| x / norm).collect();
        let objective = |step: f64| -> f64 {
            (0..6).map(|i| {
                let z = a_ref[i] + step * da[i] + (b_ref[i] + step * db[i]) * scales[i / 3];
                1.7 * z + 0.4 * z
            }).sum()
        };
        let step = 1e-6;
        let finite_difference = (objective(step) - objective(-step)) / (2.0 * step);
        let analytical: f64 = (0..6).map(|i| 2.1 * (da[i] + db[i] * scales[i / 3])).sum();
        assert!((finite_difference - analytical).abs() < 1e-7);
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn attention_extreme_negative_single_key_is_exact() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use dre_core::{BatchLayout, DocumentLayout};

    let device = gpu_device().autodiff();
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[1], 2), &device);
    let values: Vec<f32> = (0..128).map(|i| if i < 64 { i as f32 - 32.0 } else { 17.0 }).collect();
    // All scores are finite, but the only eligible score is approximately -8e32.
    for oracle in [false, true] {
        let q = Tensor::<4>::full([1, 1, 2, 64], 1e16, &device).require_grad();
        let k = Tensor::<4>::full([1, 1, 2, 64], -1e16, &device).require_grad();
        let v = Tensor::<4>::from_data(TensorData::new(values.clone(), [1, 1, 2, 64]), &device).require_grad();
        let output = if oracle {
            crate::oracle::oracle_windowed_attention(q.clone(), k.clone(), v.clone(), &layout, 128)
        } else {
            let (output, normalization) = crate::attention_kernel::run_attention(q.clone(), k.clone(), v.clone(), &layout, 128);
            let normalization = normalization.into_data().try_to_vec::<f32>().unwrap();
            assert!(normalization[0].is_finite() && normalization[0] < -1e30);
            assert_eq!(normalization[1], 1.0);
            assert_eq!(&normalization[2..], &[0.0, 1.0]);
            output
        };
        let actual = output.clone().into_data().try_to_vec::<f32>().unwrap();
        for (i, &actual) in actual.iter().enumerate() {
            let expected = if i < 64 { values[i] } else { 0.0 };
            assert_eq!(actual, expected, "oracle={oracle}, output[{i}]");
        }
        let grads = output.sum().backward();
        for (name, gradient) in [
            ("q", q.grad(&grads).unwrap()),
            ("k", k.grad(&grads).unwrap()),
            ("v", v.grad(&grads).unwrap()),
        ] {
            for (i, actual) in gradient.into_data().try_to_vec::<f32>().unwrap().into_iter().enumerate() {
                let expected = if name == "v" && i < 64 { 1.0 } else { 0.0 };
                assert_eq!(actual, expected, "oracle={oracle}, d{name}[{i}]");
            }
        }
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn attention_extreme_negative_equal_keys_preserve_backward_normalizer() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use dre_core::{BatchLayout, DocumentLayout};

    let device = gpu_device().autodiff();
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[2], 3), &device);
    // Each valid query has two equal scores near -8e32. Adding ln(2) to an FP32
    // score rounds back to that score, so saving only LSE would double every dV.
    for oracle in [false, true] {
        let q = Tensor::<4>::full([1, 1, 3, 64], 1e16, &device).require_grad();
        let k = Tensor::<4>::full([1, 1, 3, 64], -1e16, &device).require_grad();
        let v = Tensor::<4>::zeros([1, 1, 3, 64], &device).require_grad();
        let output = if oracle {
            crate::oracle::oracle_windowed_attention(q.clone(), k.clone(), v.clone(), &layout, 128)
        } else {
            let (output, normalization) = crate::attention_kernel::run_attention(q.clone(), k.clone(), v.clone(), &layout, 128);
            let normalization = normalization.into_data().try_to_vec::<f32>().unwrap();
            for row in normalization[..4].chunks_exact(2) {
                assert!(row[0].is_finite() && row[0] < -1e30);
                assert_eq!(row[1], 2.0);
            }
            assert_eq!(&normalization[4..], &[0.0, 1.0]);
            output
        };
        assert!(output.clone().into_data().try_to_vec::<f32>().unwrap().iter().all(|&value| value == 0.0));
        let gradients = output.sum().backward();
        for (name, gradient) in [
            ("q", q.grad(&gradients).unwrap()),
            ("k", k.grad(&gradients).unwrap()),
            ("v", v.grad(&gradients).unwrap()),
        ] {
            for (index, value) in gradient.into_data().try_to_vec::<f32>().unwrap().into_iter().enumerate() {
                let expected = if name == "v" && index < 128 { 1.0 } else { 0.0 };
                assert_eq!(value, expected, "oracle={oracle}, d{name}[{index}]");
            }
        }
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn attention_backward_tiny() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use crate::kernel::windowed_attention;
    use dre_core::{BatchLayout, DocumentLayout};

    let device = gpu_device().autodiff();
    let tokens = 8usize;
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[tokens], tokens), &device);
    let make = |seed: usize| (0..tokens * 64)
        .map(|i| ((i * 31 + seed) as f32 * 0.0021).sin())
        .collect::<Vec<_>>();
    let input = |seed| Tensor::<4>::from_data(
        TensorData::new(make(seed), [1, 1, tokens, 64]), &device,
    ).require_grad();
    let q = input(1);
    let k = input(2);
    let v = input(3);
    let q_ref = input(1);
    let k_ref = input(2);
    let v_ref = input(3);
    let upstream = Tensor::<4>::from_data(
        TensorData::new(make(17), [1, tokens, 1, 64]), &device,
    ).swap_dims(1, 2);
    let actual = windowed_attention(q.clone(), k.clone(), v.clone(), &layout, 128);
    let expected = crate::oracle::oracle_windowed_attention(q_ref.clone(), k_ref.clone(), v_ref.clone(), &layout, 128);
    let actual_grads = (actual * upstream.clone()).sum().backward();
    let expected_grads = (expected * upstream).sum().backward();
    device.sync().unwrap();
    for (name, actual, expected) in [
        ("q", q.grad(&actual_grads).unwrap(), q_ref.grad(&expected_grads).unwrap()),
        ("k", k.grad(&actual_grads).unwrap(), k_ref.grad(&expected_grads).unwrap()),
        ("v", v.grad(&actual_grads).unwrap(), v_ref.grad(&expected_grads).unwrap()),
    ] {
        let actual = actual.into_data().try_to_vec::<f32>().unwrap();
        let expected = expected.into_data().try_to_vec::<f32>().unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() < 8e-4, "{name}[{index}]: {actual} != {expected}");
        }
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn attention_kernel() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use crate::kernel::windowed_attention;
    use dre_core::{BatchLayout, DocumentLayout};

    fn check(tokens: usize) {
        let device = gpu_device();
        let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[tokens], tokens), &device);
        let q: Vec<f32> = (0..tokens * 64).map(|i| ((i * 7 + 3) as f32 * 0.013).sin()).collect();
        let k: Vec<f32> = (0..tokens * 64).map(|i| ((i * 11 + 5) as f32 * 0.009).cos()).collect();
        let v: Vec<f32> = (0..tokens * 64).map(|i| ((i * 17 + 1) as f32 * 0.007).sin()).collect();
        let bf16 = std::env::var_os("DRE_BF16_GATE").is_some();
        let dtype = if bf16 { burn::tensor::DType::BF16 } else { burn::tensor::DType::F32 };
        let tq = Tensor::<4>::from_data(TensorData::new(q.clone(), [1, 1, tokens, 64]), &device);
        let tk = Tensor::<4>::from_data(TensorData::new(k.clone(), [1, 1, tokens, 64]), &device);
        let tv = Tensor::<4>::from_data(TensorData::new(v.clone(), [1, 1, tokens, 64]), &device);
        let started = std::time::Instant::now();
        let (output, normalization) = crate::attention_kernel::run_attention(tq.clone().cast(dtype), tk.clone().cast(dtype), tv.clone().cast(dtype), &layout, 128);
        device.sync().unwrap();
        println!("attention forward T={tokens}: {:?}", started.elapsed());
        let got = output.cast(burn::tensor::DType::F32).into_data().try_to_vec::<f32>().unwrap();
        let got_normalization = normalization.into_data().try_to_vec::<f32>().unwrap();
        if tokens == 100 {
            let oracle = crate::oracle::oracle_windowed_attention(tq, tk, tv, &layout, 128).into_data().try_to_vec::<f32>().unwrap();
            let _oracle_first_value = oracle[0];
        }
        let mut expected = vec![0.0f64; tokens * 64];
        for query in 0..tokens {
            let lo = query.saturating_sub(128);
            let hi = (query + 128).min(tokens - 1);
            let mut scores = Vec::with_capacity(hi - lo + 1);
            for key in lo..=hi {
                let dot: f64 = (0..64).map(|d| q[query * 64 + d] as f64 * k[key * 64 + d] as f64).sum();
                scores.push(dot / 8.0);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            if query == 0 && tokens == 100 {
                let tolerance = if bf16 { 1e-2 } else { 5e-4 };
                let got_lse = f64::from(got_normalization[0]) + f64::from(got_normalization[1]).ln();
                assert!((got_lse - (max + total.ln())).abs() < tolerance, "CubeCL LSE {got_lse} expected {}", max + total.ln());
            }
            for d in 0..64 {
                expected[query * 64 + d] = weights.iter().enumerate().map(|(i, weight)| weight * v[(lo + i) * 64 + d] as f64).sum::<f64>() / total;
            }
        }
        let scale = (expected.iter().map(|x| x*x).sum::<f64>() / expected.len() as f64).sqrt();
        let mut max_abs = 0.0f64;
        let mut max_ratio = 0.0f64;
        for (index, (actual, reference)) in got.iter().zip(expected).enumerate() {
            let error = (*actual as f64 - reference).abs();
            let tolerance = if bf16 { 1e-2 * (scale + reference.abs()) } else { 5e-4 };
            max_abs = max_abs.max(error);
            max_ratio = max_ratio.max(error / tolerance);
            assert!(error <= tolerance, "tokens={tokens}, index={index}: got {actual}, expected {reference}, error/tol={}", error/tolerance);
        }
        println!("attention f64 BF16={bf16} T={tokens}: max_abs={max_abs:.6e} max_error_ratio={max_ratio:.6e}");
    }

    let selected_lengths = std::env::var("DRE_ATTN_T").ok().map(|lengths| {
        lengths.split(',').map(|length| length.parse::<usize>().expect("DRE_ATTN_T must be comma-separated token counts")).collect::<Vec<_>>()
    });
    let defaults = [1usize, 100, 256, 257, 777, 2048];
    let forward_lengths = selected_lengths.as_deref().unwrap_or(&defaults);
    for &tokens in forward_lengths { check(tokens); }
    let tokens = selected_lengths.as_ref().and_then(|lengths| lengths.last()).copied().unwrap_or(4097);
    let device = gpu_device().autodiff();
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[tokens], tokens), &device);
    let data: Vec<f32> = (0..tokens * 64).map(|i| ((i * 3 + 2) as f32 * 0.005).sin()).collect();
    let oracle_data = data.clone();
    let q = Tensor::<4>::from_data(TensorData::new(data.clone(), [1, 1, tokens, 64]), &device).require_grad();
    let k = Tensor::<4>::from_data(TensorData::new(data.clone(), [1, 1, tokens, 64]), &device).require_grad();
    let v = Tensor::<4>::from_data(TensorData::new(data, [1, 1, tokens, 64]), &device).require_grad();
    let started = std::time::Instant::now();
    let output = windowed_attention(q.clone(), k.clone(), v.clone(), &layout, 128);
    device.sync().unwrap();
    println!("attention autodiff forward T={tokens}: {:?}", started.elapsed());
    let started = std::time::Instant::now();
    let gradients = output.clone().sum().backward();
    device.sync().unwrap();
    println!("attention backward T={tokens}: {:?}", started.elapsed());
    device.sync().unwrap();
    let got_output = output.into_data().try_to_vec::<f32>().unwrap();
    let got_dq = q.grad(&gradients).unwrap().into_data().try_to_vec::<f32>().unwrap();
    let got_dk = k.grad(&gradients).unwrap().into_data().try_to_vec::<f32>().unwrap();
    let got_dv = v.grad(&gradients).unwrap().into_data().try_to_vec::<f32>().unwrap();
    let mut ref_output = vec![0.0f64; tokens * 64];
    let mut ref_dq = vec![0.0f64; tokens * 64];
    let mut ref_dk = vec![0.0f64; tokens * 64];
    let mut ref_dv = vec![0.0f64; tokens * 64];
    for query in 0..tokens {
        let lo = query.saturating_sub(128);
        let hi = (query + 128).min(tokens - 1);
        let mut scores = Vec::with_capacity(hi - lo + 1);
        for key in lo..=hi {
            let dot: f64 = (0..64).map(|d| oracle_data[query * 64 + d] as f64 * oracle_data[key * 64 + d] as f64).sum();
            scores.push(dot / 8.0);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let total: f64 = weights.iter().sum();
        for d in 0..64 {
            ref_output[query * 64 + d] = weights.iter().enumerate().map(|(i, p)| p * oracle_data[(lo + i) * 64 + d] as f64).sum::<f64>() / total;
        }
        let correction: f64 = ref_output[query * 64..(query + 1) * 64].iter().sum();
        for (i, probability) in weights.iter().map(|p| p / total).enumerate() {
            let key = lo + i;
            let value_sum: f64 = oracle_data[key * 64..(key + 1) * 64].iter().map(|x| *x as f64).sum();
            let ds = probability * (value_sum - correction) / 8.0;
            for d in 0..64 {
                ref_dq[query * 64 + d] += ds * oracle_data[key * 64 + d] as f64;
                ref_dk[key * 64 + d] += ds * oracle_data[query * 64 + d] as f64;
                ref_dv[key * 64 + d] += probability;
            }
        }
    }
    for (got, reference) in got_output.iter().zip(ref_output) { assert!((*got as f64 - reference).abs() < 5e-4); }
    for (got, reference) in got_dq.iter().zip(ref_dq) { assert!((*got as f64 - reference).abs() < 5e-4); }
    for (got, reference) in got_dk.iter().zip(ref_dk) { assert!((*got as f64 - reference).abs() < 5e-4); }
    for (got, reference) in got_dv.iter().zip(ref_dv) { assert!((*got as f64 - reference).abs() < 5e-4); }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn attention_batch_dispatch() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use dre_core::{BatchLayout, DocumentLayout};
    let batch = 4;
    let tokens = 2048;
    let device = gpu_device().autodiff();
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[tokens; 4], tokens), &device);
    let data: Vec<f32> = (0..batch * tokens * 64).map(|i| ((i * 3 + 2) as f32 * 0.005).sin()).collect();
    let q = Tensor::<4>::from_data(TensorData::new(data.clone(), [batch, 1, tokens, 64]), &device).require_grad();
    let k = Tensor::<4>::from_data(TensorData::new(data.clone(), [batch, 1, tokens, 64]), &device).require_grad();
    let v = Tensor::<4>::from_data(TensorData::new(data.clone(), [batch, 1, tokens, 64]), &device).require_grad();
    let output = crate::kernel::windowed_attention(q.clone(), k.clone(), v.clone(), &layout, 128);
    let gradients = output.clone().sum().backward();
    device.sync().unwrap();
    let actual = output.into_data().try_to_vec::<f32>().unwrap();
    let oracle = crate::oracle::oracle_windowed_attention(q.clone(), k.clone(), v.clone(), &layout, 128)
        .into_data().try_to_vec::<f32>().unwrap();
    for tensor in [q.grad(&gradients).unwrap(), k.grad(&gradients).unwrap(), v.grad(&gradients).unwrap()] {
        let values = tensor.into_data().try_to_vec::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
    }
    for (b, query) in [(0usize, 0usize), (1, 1024), (3, 2047)] {
        for dim in 0..64 {
            let index = (b * tokens + query) * 64 + dim;
            assert!((actual[index] - oracle[index]).abs() < 5e-4);
        }
    }
    for (b, query) in [(0usize, 0usize), (1, 1024), (3, 2047)] {
        let lo = query.saturating_sub(128);
        let hi = (query + 128).min(tokens - 1);
        let mut scores = Vec::new();
        for key in lo..=hi {
            let offset_q = (b * tokens + query) * 64;
            let offset_k = (b * tokens + key) * 64;
            let dot: f64 = (0..64).map(|d| data[offset_q + d] as f64 * data[offset_k + d] as f64).sum();
            scores.push(dot / 8.0);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let total: f64 = weights.iter().sum();
        for d in 0..64 {
            let expected = weights.iter().enumerate().map(|(i, weight)| weight * data[(b * tokens + lo + i) * 64 + d] as f64).sum::<f64>() / total;
            assert!((actual[(b * tokens + query) * 64 + d] as f64 - expected).abs() < 5e-4);
        }
    }
}

#[test]
fn local_attention_packed_matches_separate_cpu() {
    local_attention_packed_matches_separate_on(Device::ndarray().autodiff());
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn local_attention_packed_matches_separate() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    local_attention_packed_matches_separate_on(gpu_device().autodiff());
}

fn local_attention_packed_matches_separate_on(device: Device) {
    use crate::LocalAttentionConfig;
    use dre_core::{BatchLayout, DocumentLayout};
    let layer = LocalAttentionConfig::new().init(&device);
    let values: Vec<f32> = (0..5 * 512).map(|i| ((i * 13 + 7) as f32 * 0.001).sin()).collect();
    let padded: Vec<f32> = values.iter().copied().chain(std::iter::repeat_n(0.0, 512)).collect();
    let packed = Tensor::<3>::from_data(TensorData::new(padded, [1, 6, 512]), &device).require_grad();
    let packed_layout = BatchLayout::new(DocumentLayout::packed(vec![vec![2, 3]], 6), &device);
    let packed_out = layer.forward(packed.clone(), &packed_layout);
    let packed_grad = packed_out.clone().sum().backward();
    assert!(packed.grad(&packed_grad).is_some());
    device.sync().unwrap();
    let packed_values = packed_out.into_data().try_to_vec::<f32>().unwrap();
    assert!(packed_values[5 * 512..].iter().all(|&value| value == 0.0));
    for (start, length) in [(0, 2), (2, 3)] {
        let data = values[start * 512..(start + length) * 512].to_vec();
        let input = Tensor::<3>::from_data(TensorData::new(data, [1, length, 512]), &device);
        let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[length], length), &device);
        let separate = layer.forward(input, &layout).into_data().try_to_vec::<f32>().unwrap();
        for token in 0..length {
            let p = &packed_values[(start + token) * 512..(start + token + 1) * 512];
            let s = &separate[token * 512..(token + 1) * 512];
            for (a, b) in p.iter().zip(s) { assert!((a - b).abs() < 2e-3); }
        }
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn m0b_attention_timing() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use burn::tensor::module::attention;
    use burn::tensor::ops::AttentionModuleOptions;
    use dre_core::{BatchLayout, DocumentLayout};
    use std::time::Instant;

    fn values(rows: usize) -> Vec<f32> {
        (0..rows * 64).map(|i| ((i * 7 + 3) as f32 * 0.013).sin()).collect()
    }
    fn kernel_run(batch: usize, tokens: usize, device: &burn::tensor::Device) -> std::time::Duration {
        let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&vec![tokens; batch], tokens), device);
        let data = values(batch * 8 * tokens);
        let q = Tensor::<4>::from_data(TensorData::new(data.clone(), [batch, 8, tokens, 64]), device).require_grad();
        let k = Tensor::<4>::from_data(TensorData::new(data.clone(), [batch, 8, tokens, 64]), device).require_grad();
        let v = Tensor::<4>::from_data(TensorData::new(data, [batch, 8, tokens, 64]), device).require_grad();
        let start = Instant::now();
        let output = crate::kernel::windowed_attention(q.clone(), k.clone(), v.clone(), &layout, 128);
        let grads = output.sum().backward();
        device.sync().unwrap();
        std::hint::black_box((q.grad(&grads), k.grad(&grads), v.grad(&grads)));
        start.elapsed()
    }
    let device = gpu_device().autodiff();
    let selected_batch = std::env::var("DRE_M0B_BATCH").ok();
    let batches = match selected_batch.as_deref() {
        Some("1") => vec![1usize],
        Some("4") => vec![4usize],
        Some(_) => panic!("DRE_M0B_BATCH must be 1 or 4"),
        None => vec![1usize, 4],
    };
    for batch in batches {
        let tokens = 2048;
        for _ in 0..3 { let _ = kernel_run(batch, tokens, &device); }
        let mut elapsed = Vec::with_capacity(5);
        for _ in 0..5 { elapsed.push(kernel_run(batch, tokens, &device).as_secs_f64() * 1000.0); }
        println!("attention kernel B={batch} T={tokens}: {elapsed:?} ms");
    }

    fn dense_run(batch: usize, tokens: usize, device: &burn::tensor::Device) -> std::time::Duration {
        let shape = [batch, 8, tokens, 64];
        let data = values(batch * 8 * tokens);
        let outside: Vec<bool> = (0..tokens).flat_map(|query| (0..tokens).map(move |key| query.abs_diff(key) > 128)).collect();
        let mask = crate::relay::materialized_attention_mask(
            Tensor::<4, Bool>::from_data(TensorData::new(outside, [1, 1, tokens, tokens]), device),
            [batch,8,tokens,tokens]);
        let q = Tensor::<4>::from_data(TensorData::new(data.clone(), shape), device).require_grad();
        let k = Tensor::<4>::from_data(TensorData::new(data.clone(), shape), device).require_grad();
        let v = Tensor::<4>::from_data(TensorData::new(data, shape), device).require_grad();
        let start = Instant::now();
        let output = attention(q.clone(), k.clone(), v.clone(), Some(mask), None, AttentionModuleOptions::default());
        let grads = output.sum().backward();
        device.sync().unwrap();
        std::hint::black_box((q.grad(&grads), k.grad(&grads), v.grad(&grads)));
        start.elapsed()
    }
    if selected_batch.is_none() {
        for _ in 0..3 { let _ = dense_run(1, 1024, &device); }
        let mut elapsed = Vec::with_capacity(5);
        for _ in 0..5 { elapsed.push(dense_run(1, 1024, &device).as_secs_f64() * 1000.0); }
        println!("Burn dense masked attention B=1 T=1024: {elapsed:?} ms");
        let _ = dense_run(1, 2048, &device);
        println!("Burn dense masked attention B=1 T=2048: {:.3} ms", dense_run(1, 2048, &device).as_secs_f64() * 1000.0);
        let _ = dense_run(4, 2048, &device);
        println!("Burn dense masked attention B=4 T=2048: {:.3} ms", dense_run(4, 2048, &device).as_secs_f64() * 1000.0);
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn local_source_attention_forced_split_matches_forward_and_backward_oracle() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use dre_core::{BatchLayout, DocumentLayout};

    let device = gpu_device().autodiff();
    let tokens = 32usize;
    let heads = 3usize;
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[tokens], tokens), &device);
    let values = |seed| (0..heads * tokens * 64)
        .map(|i| ((i * 31 + seed) as f32 * 0.0021).sin())
        .collect::<Vec<_>>();
    let tensor = |seed| Tensor::<4>::from_data(
        TensorData::new(values(seed), [1, heads, tokens, 64]),
        &device,
    ).require_grad();
    let q = tensor(1);
    let k = tensor(2);
    let v = tensor(3);
    let q_ref = tensor(1);
    let k_ref = tensor(2);
    let v_ref = tensor(3);
    let upstream = Tensor::<4>::from_data(
        TensorData::new(values(17), [1, heads, tokens, 64]),
        &device,
    );
    let expected = crate::oracle::oracle_windowed_attention(q_ref.clone(), k_ref.clone(), v_ref.clone(), &layout, 128);
    let dispatch = crate::attention_kernel::TestDispatchLimitGuard::new(2);
    let (actual, _) = crate::attention_kernel::run_attention(q.clone(), k.clone(), v.clone(), &layout, 128);
    let actual_grads = (actual.clone() * upstream.clone()).sum().backward();
    let expected_grads = (expected.clone() * upstream).sum().backward();
    device.sync().unwrap();

    let (split_dispatches, geometry) = dispatch.dispatch_stats();
    assert_eq!(split_dispatches, 2, "expected forward and shared dQ/dK-backward split launches");
    assert_eq!(geometry.logical_groups, 3);
    assert_eq!((geometry.x, geometry.y, geometry.excess_groups), (2, 2, 1));

    let actual_values = actual.into_data().try_to_vec::<f32>().unwrap();
    let expected_values = expected.into_data().try_to_vec::<f32>().unwrap();
    for (index, (actual, expected)) in actual_values.iter().zip(&expected_values).enumerate() {
        assert!((actual - expected).abs() < 8e-4, "output[{index}]: {actual} != {expected}");
    }
    for (name, actual, expected) in [
        ("q", q.grad(&actual_grads).unwrap(), q_ref.grad(&expected_grads).unwrap()),
        ("k", k.grad(&actual_grads).unwrap(), k_ref.grad(&expected_grads).unwrap()),
        ("v", v.grad(&actual_grads).unwrap(), v_ref.grad(&expected_grads).unwrap()),
    ] {
        let actual = actual.into_data().try_to_vec::<f32>().unwrap();
        let expected = expected.into_data().try_to_vec::<f32>().unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() < 1e-3, "{name}[{index}]: {actual} != {expected}");
        }
    }
}
