use burn::prelude::*;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use dre_core::{BatchLayout, DocumentLayout, gpu_device};

use super::LocalGlobalRelayAttentionConfig;

use crate::oracle::burn_source_oracle;

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn relay_attention_extreme_negative_single_effective_key_is_exact() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let device = gpu_device().autodiff();
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[1], 2), &device);
    let q = Tensor::<4>::full([1, 1, 2, 64], 1e16, &device).require_grad();
    let k = Tensor::<4>::full([1, 1, 2, 64], -1e16, &device).require_grad();
    let values: Vec<f32> = (0..128).map(|i| if i < 64 { i as f32 - 32.0 } else { 17.0 }).collect();
    let v = Tensor::<4>::from_data(TensorData::new(values.clone(), [1, 1, 2, 64]), &device).require_grad();
    // Source score -8e32, relay scores -1.6e33: all are below the old sentinel,
    // and every relay's relative weight underflows to zero, leaving exactly V_source.
    let rk = Tensor::<4>::full([1, 1, 16, 64], -2e16, &device).require_grad();
    let rv = Tensor::<4>::full([1, 1, 16, 64], 23.0, &device).require_grad();
    let (output, normalization) = crate::relay_attention_kernel::run_relay_windowed_attention(
        q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone(), &layout,
    );
    let normalization = normalization.into_data().try_to_vec::<f32>().unwrap();
    assert!(normalization[0].is_finite() && normalization[0] < -1e30);
    assert_eq!(normalization[1], 1.0);
    assert_eq!(&normalization[2..], &[0.0, 1.0]);
    for (i, actual) in output.clone().into_data().try_to_vec::<f32>().unwrap().into_iter().enumerate() {
        assert_eq!(actual, if i < 64 { values[i] } else { 0.0 }, "output[{i}]");
    }
    let grads = output.sum().backward();
    for (name, gradient) in [
        ("q", q.grad(&grads).unwrap()),
        ("k", k.grad(&grads).unwrap()),
        ("v", v.grad(&grads).unwrap()),
        ("rk", rk.grad(&grads).unwrap()),
        ("rv", rv.grad(&grads).unwrap()),
    ] {
        for (i, actual) in gradient.into_data().try_to_vec::<f32>().unwrap().into_iter().enumerate() {
            let expected = if name == "v" && i < 64 { 1.0 } else { 0.0 };
            assert_eq!(actual, expected, "d{name}[{i}]");
        }
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn relay_attention_extreme_negative_equal_keys_preserve_backward_normalizer() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let device = gpu_device().autodiff();
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[2], 2), &device);
    // Two source keys plus sixteen relay keys have identical scores near -8e32.
    // A loss on just the first query must send exactly 1/18 to every value.
    for oracle in [false, true] {
        let q = Tensor::<4>::full([1, 1, 2, 64], 1e16, &device).require_grad();
        let k = Tensor::<4>::full([1, 1, 2, 64], -1e16, &device).require_grad();
        let v = Tensor::<4>::zeros([1, 1, 2, 64], &device).require_grad();
        let rk = Tensor::<4>::full([1, 1, 16, 64], -1e16, &device).require_grad();
        let rv = Tensor::<4>::zeros([1, 1, 16, 64], &device).require_grad();
        let output = if oracle {
            burn_source_oracle(q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone(), &layout)
        } else {
            let (output, normalization) = crate::relay_attention_kernel::run_relay_windowed_attention(
                q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone(), &layout,
            );
            for row in normalization.into_data().try_to_vec::<f32>().unwrap().chunks_exact(2) {
                assert!(row[0].is_finite() && row[0] < -1e30);
                assert_eq!(row[1], 18.0);
            }
            output
        };
        assert!(output.clone().into_data().try_to_vec::<f32>().unwrap().iter().all(|&value| value == 0.0));
        let gradients = output.slice([0..1, 0..1, 0..1, 0..64]).sum().backward();
        for (name, gradient) in [
            ("q", q.grad(&gradients).unwrap()),
            ("k", k.grad(&gradients).unwrap()),
            ("v", v.grad(&gradients).unwrap()),
            ("rk", rk.grad(&gradients).unwrap()),
            ("rv", rv.grad(&gradients).unwrap()),
        ] {
            let expected = if name == "v" || name == "rv" { 1.0 / 18.0 } else { 0.0 };
            for (index, value) in gradient.into_data().try_to_vec::<f32>().unwrap().into_iter().enumerate() {
                assert!((value - expected).abs() < 1e-7, "oracle={oracle}, d{name}[{index}]: {value} != {expected}");
            }
        }
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn relay_cube_matches_burn_and_dense_masked_forward_backward() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let device = gpu_device().autodiff();
    let (tokens, documents) = (6usize, 2usize);
    let layout = BatchLayout::new(DocumentLayout::packed(vec![vec![2, 3]], tokens), &device);
    let make = |n: usize, seed: usize| -> Vec<f32> { (0..n).map(|i| ((i * 31 + seed) as f32 * 0.0021).sin()).collect() };
    let q_data = make(tokens * 64, 1);
    let k_data = make(tokens * 64, 2);
    let v_data = make(tokens * 64, 3);
    let rk_data = make(documents * 16 * 64, 4);
    let rv_data = make(documents * 16 * 64, 5);
    let tensor = |data: Vec<f32>, dims| Tensor::<4>::from_data(TensorData::new(data, dims), &device).require_grad();
    let q = tensor(q_data.clone(), [1, 1, tokens, 64]);
    let k = tensor(k_data.clone(), [1, 1, tokens, 64]);
    let v = tensor(v_data.clone(), [1, 1, tokens, 64]);
    let relay_k = tensor(rk_data.clone(), [documents, 1, 16, 64]);
    let relay_v = tensor(rv_data.clone(), [documents, 1, 16, 64]);
    let q_oracle = tensor(q_data, [1, 1, tokens, 64]);
    let k_oracle = tensor(k_data, [1, 1, tokens, 64]);
    let v_oracle = tensor(v_data, [1, 1, tokens, 64]);
    let rk_oracle = tensor(rk_data, [documents, 1, 16, 64]);
    let rv_oracle = tensor(rv_data, [documents, 1, 16, 64]);
    let q_dense = tensor(make(tokens * 64, 1), [1, 1, tokens, 64]);
    let k_dense = tensor(make(tokens * 64, 2), [1, 1, tokens, 64]);
    let v_dense = tensor(make(tokens * 64, 3), [1, 1, tokens, 64]);
    let rk_dense = tensor(make(documents * 16 * 64, 4), [documents, 1, 16, 64]);
    let rv_dense = tensor(make(documents * 16 * 64, 5), [documents, 1, 16, 64]);
    let (cube, _) = crate::relay_attention_kernel::run_relay_windowed_attention(q.clone(), k.clone(), v.clone(), relay_k.clone(), relay_v.clone(), &layout);
    let oracle = burn_source_oracle(q_oracle.clone(), k_oracle.clone(), v_oracle.clone(), rk_oracle.clone(), rv_oracle.clone(), &layout);
    let dense_k = Tensor::cat(vec![k_dense.clone(), rk_dense.clone().reshape([1, 1, documents * 16, 64])], 2);
    let dense_v = Tensor::cat(vec![v_dense.clone(), rv_dense.clone().reshape([1, 1, documents * 16, 64])], 2);
    let query_docs = [0i32, 0, 1, 1, 1, -1];
    let key_docs = [0i32, 0, 1, 1, 1, -1];
    let mask_data: Vec<bool> = (0..tokens).flat_map(|query| (0..tokens + documents * 16).map(move |key| {
        let qdoc = query_docs[query];
        if qdoc < 0 { key != 0 }
        else if key < tokens { key_docs[key] != qdoc || query.abs_diff(key) > 128 }
        else { ((key - tokens) / 16) as i32 != qdoc }
    })).collect();
    let dense_mask = crate::relay::materialized_attention_mask(
        Tensor::<4, Bool>::from_data(TensorData::new(mask_data, [1, 1, tokens, tokens + documents * 16]), &device),
        [1,1,tokens,tokens+documents*16]);
    let dense = attention(q_dense.clone(), dense_k, dense_v, Some(dense_mask), None, AttentionModuleOptions::default())
        * layout.valid.clone().reshape([1, 1, tokens, 1]);
    let cube_loss = cube.clone().sum();
    let oracle_loss = oracle.clone().sum();
    let dense_loss = dense.clone().sum();
    let cube_grads = cube_loss.backward();
    let oracle_grads = oracle_loss.backward();
    let dense_grads = dense_loss.backward();
    device.sync().unwrap();
    let cube_values = cube.into_data().try_to_vec::<f32>().unwrap();
    let oracle_values = oracle.into_data().try_to_vec::<f32>().unwrap();
    let dense_values = dense.into_data().try_to_vec::<f32>().unwrap();
    for ((actual, oracle), dense) in cube_values.iter().zip(&oracle_values).zip(dense_values) {
        assert!((actual - oracle).abs() < 8e-4);
        assert!((actual - dense).abs() < 8e-4);
    }
    for (cube_grad, (oracle_grad, dense_grad)) in [
        q.grad(&cube_grads).unwrap(), k.grad(&cube_grads).unwrap(), v.grad(&cube_grads).unwrap(),
        relay_k.grad(&cube_grads).unwrap(), relay_v.grad(&cube_grads).unwrap(),
    ].into_iter().zip([
        (q_oracle.grad(&oracle_grads).unwrap(), q_dense.grad(&dense_grads).unwrap()),
        (k_oracle.grad(&oracle_grads).unwrap(), k_dense.grad(&dense_grads).unwrap()),
        (v_oracle.grad(&oracle_grads).unwrap(), v_dense.grad(&dense_grads).unwrap()),
        (rk_oracle.grad(&oracle_grads).unwrap(), rk_dense.grad(&dense_grads).unwrap()),
        (rv_oracle.grad(&oracle_grads).unwrap(), rv_dense.grad(&dense_grads).unwrap()),
    ]) {
        let a = cube_grad.into_data().try_to_vec::<f32>().unwrap();
        let b = oracle_grad.into_data().try_to_vec::<f32>().unwrap();
        let c = dense_grad.into_data().try_to_vec::<f32>().unwrap();
        for ((actual, expected), dense) in a.iter().zip(b).zip(c) {
            assert!((actual - expected).abs() < 1e-3);
            assert!((actual - dense).abs() < 1e-3);
        }
    }

    let long = 260usize;
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[long], long), &device);
    let q = Tensor::<4>::from_data(TensorData::new(make(long * 64, 21), [1, 1, long, 64]), &device);
    let k = Tensor::<4>::from_data(TensorData::new(make(long * 64, 22), [1, 1, long, 64]), &device);
    let v = Tensor::<4>::from_data(TensorData::new(make(long * 64, 23), [1, 1, long, 64]), &device);
    let rk = Tensor::<4>::from_data(TensorData::new(make(16 * 64, 24), [1, 1, 16, 64]), &device);
    let rv = Tensor::<4>::from_data(TensorData::new(make(16 * 64, 25), [1, 1, 16, 64]), &device);
    let (cube, _) = crate::relay_attention_kernel::run_relay_windowed_attention(q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone(), &layout);
    let dense_k = Tensor::cat(vec![k, rk], 2);
    let dense_v = Tensor::cat(vec![v, rv], 2);
    let mask_data: Vec<bool> = (0..long).flat_map(|query| (0..long + 16).map(move |key| key < long && query.abs_diff(key) > 128)).collect();
    let mask = crate::relay::materialized_attention_mask(
        Tensor::<4, Bool>::from_data(TensorData::new(mask_data, [1, 1, long, long + 16]), &device),
        [1,1,long,long+16]);
    let dense = attention(q, dense_k, dense_v, Some(mask), None, AttentionModuleOptions::default());
    let actual = cube.into_data().try_to_vec::<f32>().unwrap();
    let expected = dense.into_data().try_to_vec::<f32>().unwrap();
    for (actual, expected) in actual.iter().zip(expected) { assert!((actual - expected).abs() < 8e-4); }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn relay_backward_matches_mult_head_transposed_layout_oracle() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let device = gpu_device().autodiff();
    let (tokens, heads, documents) = (6usize, 8usize, 2usize);
    let layout = BatchLayout::new(DocumentLayout::packed(vec![vec![2, 3]], tokens), &device);
    let make = |n: usize, seed: usize| -> Vec<f32> {
        (0..n).map(|i| ((i * 31 + seed) as f32 * 0.0021).sin()).collect()
    };
    let source = |seed| Tensor::<4>::from_data(
        TensorData::new(make(tokens * heads * 64, seed), [1, tokens, heads, 64]), &device,
    ).swap_dims(1, 2).require_grad();
    let relay = |seed| Tensor::<4>::from_data(
        TensorData::new(make(documents * 16 * heads * 64, seed), [documents, 16, heads, 64]), &device,
    ).swap_dims(1, 2).require_grad();
    let q = source(1);
    let k = source(2);
    let v = source(3);
    let rk = relay(4);
    let rv = relay(5);
    let q_ref = source(1);
    let k_ref = source(2);
    let v_ref = source(3);
    let rk_ref = relay(4);
    let rv_ref = relay(5);
    let upstream = Tensor::<4>::from_data(
        TensorData::new(make(tokens * heads * 64, 17), [1, tokens, heads, 64]), &device,
    ).swap_dims(1, 2);
    let (actual, _) = crate::relay_attention_kernel::run_relay_windowed_attention(
        q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone(), &layout,
    );
    let expected = burn_source_oracle(q_ref.clone(), k_ref.clone(), v_ref.clone(), rk_ref.clone(), rv_ref.clone(), &layout);
    let actual_grads = (actual * upstream.clone()).sum().backward();
    let expected_grads = (expected * upstream).sum().backward();
    device.sync().unwrap();
    for (name, actual, expected) in [
        ("q", q.grad(&actual_grads).unwrap(), q_ref.grad(&expected_grads).unwrap()),
        ("k", k.grad(&actual_grads).unwrap(), k_ref.grad(&expected_grads).unwrap()),
        ("v", v.grad(&actual_grads).unwrap(), v_ref.grad(&expected_grads).unwrap()),
        ("relay_k", rk.grad(&actual_grads).unwrap(), rk_ref.grad(&expected_grads).unwrap()),
        ("relay_v", rv.grad(&actual_grads).unwrap(), rv_ref.grad(&expected_grads).unwrap()),
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
fn relay_backward_512_documents_matches_oracle_through_last_gradient() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let device = gpu_device().autodiff();
    let (documents, tokens, heads) = (512usize, 1024usize, 8usize);
    let layout = BatchLayout::new(
        DocumentLayout::packed(vec![vec![2usize; documents]], tokens),
        &device,
    );
    let make = |n: usize, seed: usize| -> Vec<f32> {
        (0..n).map(|i| ((i * 31 + seed) as f32 * 0.0021).sin()).collect()
    };
    let source = |seed| Tensor::<4>::from_data(
        TensorData::new(make(tokens * heads * 64, seed), [1, heads, tokens, 64]),
        &device,
    ).require_grad();
    let relay = |seed| Tensor::<4>::from_data(
        TensorData::new(make(documents * heads * 16 * 64, seed), [documents, heads, 16, 64]),
        &device,
    ).require_grad();
    let q = source(1);
    let k = source(2);
    let v = source(3);
    let rk = relay(4);
    let rv = relay(5);
    let q_ref = source(1);
    let k_ref = source(2);
    let v_ref = source(3);
    let rk_ref = relay(4);
    let rv_ref = relay(5);
    let upstream = Tensor::<4>::from_data(
        TensorData::new(make(tokens * heads * 64, 17), [1, heads, tokens, 64]),
        &device,
    );
    let (actual, _) = crate::relay_attention_kernel::run_relay_windowed_attention(
        q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone(), &layout,
    );
    let expected = burn_source_oracle(
        q_ref.clone(), k_ref.clone(), v_ref.clone(), rk_ref.clone(), rv_ref.clone(), &layout,
    );
    let actual_grads = (actual * upstream.clone()).sum().backward();
    let expected_grads = (expected * upstream).sum().backward();
    device.sync().unwrap();
    for (name, actual, expected) in [
        ("relay_k", rk.grad(&actual_grads).unwrap(), rk_ref.grad(&expected_grads).unwrap()),
        ("relay_v", rv.grad(&actual_grads).unwrap(), rv_ref.grad(&expected_grads).unwrap()),
    ] {
        let actual = actual.into_data().try_to_vec::<f32>().unwrap();
        let expected = expected.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(actual.len(), documents * heads * 16 * 64);
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() < 8e-4, "{name}[{index}]: {actual} != {expected}");
        }
        let last = actual.len() - 1;
        assert!((actual[last] - expected[last]).abs() < 8e-4, "{name}[{last}]: {} != {}", actual[last], expected[last]);
    }
}

#[test]
fn relay_attention_packed_documents_match_separate_and_propagate_gradients_cpu() {
    relay_attention_packed_documents_match_separate_on(Device::ndarray().autodiff());
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn relay_attention_packed_documents_match_separate_and_propagate_gradients() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    relay_attention_packed_documents_match_separate_on(gpu_device().autodiff());
}

fn relay_attention_packed_documents_match_separate_on(device: Device) {
    let layer = LocalGlobalRelayAttentionConfig::new().init(&device);
    let mut source_values: Vec<f32> = (0..5 * 512).map(|i| ((i * 13 + 7) as f32 * 0.001).sin()).collect();
    source_values.extend(std::iter::repeat(0.0).take(512));
    let relay_values: Vec<f32> = (0..2 * 16 * 512).map(|i| ((i * 17 + 3) as f32 * 0.0007).cos()).collect();
    let source = Tensor::<3>::from_data(TensorData::new(source_values, [1, 6, 512]), &device).require_grad();
    let relays = Tensor::<3>::from_data(TensorData::new(relay_values, [2, 16, 512]), &device).require_grad();
    let packed_layout = BatchLayout::new(DocumentLayout::packed(vec![vec![2, 3]], 6), &device);
    let (packed_source, packed_relays) = layer.forward(source.clone(), relays.clone(), &packed_layout);
    let gradients = (packed_source.clone().sum() + packed_relays.clone().sum()).backward();
    assert!(source.grad(&gradients).is_some());
    assert!(relays.grad(&gradients).is_some());
    device.sync().unwrap();
    let packed_source = packed_source.into_data().try_to_vec::<f32>().unwrap();
    let packed_relays = packed_relays.into_data().try_to_vec::<f32>().unwrap();
    for (document, (start, length)) in [(0usize, 2usize), (2, 3)].into_iter().enumerate() {
        let input = Tensor::<3>::from_data(TensorData::new(source.clone().slice([0..1, start..start + length, 0..512]).into_data().try_to_vec::<f32>().unwrap(), [1, length, 512]), &device);
        let relay = relays.clone().slice([document..document + 1, 0..16, 0..512]);
        let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[length], length), &device);
        let (separate_source, separate_relays) = layer.forward(input, relay, &layout);
        let separate_source = separate_source.into_data().try_to_vec::<f32>().unwrap();
        let separate_relays = separate_relays.into_data().try_to_vec::<f32>().unwrap();
        for (actual, expected) in separate_source.iter().zip((start * 512..(start + length) * 512).map(|i| packed_source[i])) { assert!((actual - expected).abs() < 2e-4); }
        for (actual, expected) in separate_relays.iter().zip((document * 16 * 512..(document + 1) * 16 * 512).map(|i| packed_relays[i])) { assert!((actual - expected).abs() < 2e-4); }
    }
}

#[test]
fn relay_stage_pair_carries_far_token_information_across_two_hops_cpu() {
    relay_stage_pair_carries_far_token_information_on(Device::ndarray());
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn relay_stage_pair_carries_far_token_information_across_two_hops() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    relay_stage_pair_carries_far_token_information_on(gpu_device());
}

fn relay_stage_pair_carries_far_token_information_on(device: Device) {
    let tokens = 258usize;
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[tokens], tokens), &device);
    let stage_six = LocalGlobalRelayAttentionConfig::new().init(&device);
    let stage_eleven = LocalGlobalRelayAttentionConfig::new().init(&device);
    let source_a: Vec<f32> = (0..tokens * 512).map(|i| ((i * 19 + 1) as f32 * 0.0013).sin()).collect();
    let mut source_b = source_a.clone();
    for value in &mut source_b[257 * 512..258 * 512] { *value += 4.0; }
    let relay: Vec<f32> = (0..16 * 512).map(|i| ((i * 7 + 2) as f32 * 0.0009).cos()).collect();
    let run = |source: Vec<f32>| {
        let source = Tensor::<3>::from_data(TensorData::new(source, [1, tokens, 512]), &device);
        let relay = Tensor::<3>::from_data(TensorData::new(relay.clone(), [1, 16, 512]), &device);
        let (source, relay) = stage_six.forward(source, relay, &layout);
        let first = source.clone().slice([0..1, 0..1, 0..512]).into_data().try_to_vec::<f32>().unwrap();
        // The intervening source operators are identity here; the relay sidecar is passed intact.
        let (source, _) = stage_eleven.forward(source, relay, &layout);
        let second = source.slice([0..1, 0..1, 0..512]).into_data().try_to_vec::<f32>().unwrap();
        (first, second)
    };
    let (a_first, a_second) = run(source_a);
    let (b_first, b_second) = run(source_b);
    assert!(a_first.iter().zip(&b_first).all(|(a, b)| (a - b).abs() < 2e-4));
    assert!(a_second.iter().zip(&b_second).any(|(a, b)| (a - b).abs() > 1e-5));
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn hybrid_source_attention_forced_split_matches_forward_and_backward_oracle() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let device = gpu_device().autodiff();
    let (tokens, heads, documents) = (32usize, 3usize, 1usize);
    let layout = BatchLayout::new(DocumentLayout::one_document_per_row(&[tokens], tokens), &device);
    let make = |n: usize, seed: usize| -> Vec<f32> {
        (0..n).map(|i| ((i * 31 + seed) as f32 * 0.0021).sin()).collect()
    };
    let source = |seed| Tensor::<4>::from_data(
        TensorData::new(make(tokens * heads * 64, seed), [1, heads, tokens, 64]),
        &device,
    ).require_grad();
    let relay = |seed| Tensor::<4>::from_data(
        TensorData::new(make(documents * heads * 16 * 64, seed), [documents, heads, 16, 64]),
        &device,
    ).require_grad();
    let q = source(1);
    let k = source(2);
    let v = source(3);
    let rk = relay(4);
    let rv = relay(5);
    let q_ref = source(1);
    let k_ref = source(2);
    let v_ref = source(3);
    let rk_ref = relay(4);
    let rv_ref = relay(5);
    let upstream = Tensor::<4>::from_data(
        TensorData::new(make(tokens * heads * 64, 17), [1, heads, tokens, 64]),
        &device,
    );
    let expected = burn_source_oracle(
        q_ref.clone(), k_ref.clone(), v_ref.clone(), rk_ref.clone(), rv_ref.clone(), &layout,
    );
    let dispatch = crate::attention_kernel::TestDispatchLimitGuard::new(2);
    let (actual, _) = crate::relay_attention_kernel::run_relay_windowed_attention(
        q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone(), &layout,
    );
    let actual_grads = (actual.clone() * upstream.clone()).sum().backward();
    let expected_grads = (expected.clone() * upstream).sum().backward();
    device.sync().unwrap();

    let (split_dispatches, geometry) = dispatch.dispatch_stats();
    assert_eq!(split_dispatches, 2, "expected forward and shared source-backward split launches");
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
        ("relay_k", rk.grad(&actual_grads).unwrap(), rk_ref.grad(&expected_grads).unwrap()),
        ("relay_v", rv.grad(&actual_grads).unwrap(), rv_ref.grad(&expected_grads).unwrap()),
    ] {
        let actual = actual.into_data().try_to_vec::<f32>().unwrap();
        let expected = expected.into_data().try_to_vec::<f32>().unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual - expected).abs() < 1e-3, "{name}[{index}]: {actual} != {expected}");
        }
    }
}
