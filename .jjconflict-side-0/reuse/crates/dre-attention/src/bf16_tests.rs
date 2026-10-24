//! BF16 storage/FP32 accumulation compared with an independent f64 forward and analytic adjoint.
use burn::prelude::*;
use burn::tensor::DType;
use dre_core::{BatchLayout, DocumentLayout, gpu_device};

fn compare(label: &str, actual: Tensor<4>, reference: &[f64]) {
    let actual = actual.cast(DType::F32).into_data().iter::<f64>().collect::<Vec<_>>();
    let scale = (reference.iter().map(|v| v*v).sum::<f64>() / reference.len() as f64).sqrt();
    let mut max_abs = 0.0f64;
    let mut ratios = Vec::with_capacity(actual.len());
    for (&a, &r) in actual.iter().zip(reference) {
        let error = (a-r).abs();
        max_abs = max_abs.max(error);
        ratios.push(error / (1e-2 * (scale + r.abs())));
    }
    ratios.sort_by(f64::total_cmp);
    let q = |p: f64| ratios[((ratios.len()-1) as f64*p).round() as usize];
    println!("relay BF16 f64 {label}: max_abs={max_abs:.6e} ratio_p50={:.6e} ratio_p99={:.6e} ratio_max={:.6e}", q(0.5), q(0.99), ratios.last().unwrap());
    assert!(ratios.iter().all(|r| r.is_finite() && *r <= 1.0), "{label}: BF16 error exceeds 0.01*(RMS+abs(reference))");
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn bf16_relay_kernel_matches_f64_forward_backward() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    check(true);
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn bf16_windowed_kernel_matches_f64_forward_backward() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    check(false);
}

fn check(include_relays: bool) {
    let device = gpu_device().autodiff();
    assert!(device.supports_dtype(DType::BF16));
    let (heads, tokens, documents) = (2, 7, 2);
    let layout = BatchLayout::new(DocumentLayout::packed(vec![vec![2, 4]], tokens), &device);
    let make = |n: usize, salt: usize| (0..n).map(|i| (((i*37+salt*17)%101) as f64-50.0)/73.0).collect::<Vec<_>>();
    let q = make(heads*tokens*64, 1);
    let k = make(q.len(), 2);
    let v = make(q.len(), 3);
    let rk = make(documents*heads*16*64, 4);
    let rv = make(rk.len(), 5);
    let upstream = make(q.len(), 6);
    let mut output = vec![0.0; q.len()];
    let mut gradients = [vec![0.0; q.len()], vec![0.0; q.len()], vec![0.0; q.len()], vec![0.0; rk.len()], vec![0.0; rk.len()]];
    for head in 0..heads {
        for (document, start, length) in [(0, 0, 2), (1, 2, 4)] {
            for token in start..start+length {
                let qb = (head*tokens+token)*64;
                let keys = (start..start+length).map(|t| (false, (head*tokens+t)*64))
                    .chain((0..if include_relays { 16 } else { 0 }).map(|s| (true, ((document*heads+head)*16+s)*64))).collect::<Vec<_>>();
                let scores = keys.iter().map(|&(relay, base)| {
                    let key = if relay { &rk } else { &k };
                    (0..64).map(|d| q[qb+d]*key[base+d]).sum::<f64>()/8.0
                }).collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp = scores.iter().map(|s| (s-max).exp()).collect::<Vec<_>>();
                let sum = exp.iter().sum::<f64>();
                for (&(relay, base), &weight) in keys.iter().zip(&exp) {
                    let value = if relay { &rv } else { &v };
                    for d in 0..64 { output[qb+d] += weight/sum*value[base+d]; }
                }
                let correction = (0..64).map(|d| upstream[qb+d]*output[qb+d]).sum::<f64>();
                for (&(relay, base), &weight) in keys.iter().zip(&exp) {
                    let key = if relay { &rk } else { &k };
                    let value = if relay { &rv } else { &v };
                    let ds = weight/sum*((0..64).map(|d| upstream[qb+d]*value[base+d]).sum::<f64>()-correction)/8.0;
                    for d in 0..64 {
                        gradients[0][qb+d] += ds*key[base+d];
                        gradients[if relay { 3 } else { 1 }][base+d] += ds*q[qb+d];
                        gradients[if relay { 4 } else { 2 }][base+d] += weight/sum*upstream[qb+d];
                    }
                }
            }
        }
    }
    let upload = |data: &[f64], shape| Tensor::<4>::from_data(TensorData::new(data.iter().map(|x| *x as f32).collect::<Vec<_>>(), shape), &device).require_grad();
    let inputs = [upload(&q, [1, heads, tokens, 64]), upload(&k, [1, heads, tokens, 64]), upload(&v, [1, heads, tokens, 64]), upload(&rk, [documents, heads, 16, 64]), upload(&rv, [documents, heads, 16, 64])];
    let (actual, normalization) = if include_relays {
        super::relay_attention_kernel::run_relay_windowed_attention(inputs[0].clone().cast(DType::BF16), inputs[1].clone().cast(DType::BF16), inputs[2].clone().cast(DType::BF16), inputs[3].clone().cast(DType::BF16), inputs[4].clone().cast(DType::BF16), &layout)
    } else {
        super::attention_kernel::run_attention(inputs[0].clone().cast(DType::BF16), inputs[1].clone().cast(DType::BF16), inputs[2].clone().cast(DType::BF16), &layout, 128)
    };
    assert_eq!(normalization.dtype(), DType::F32);
    compare("output", actual.clone(), &output);
    let grads = (actual.cast(DType::F32)*upload(&upstream, [1, heads, tokens, 64])).sum().backward();
    for (index, (input, expected)) in inputs.into_iter().zip(&gradients).take(if include_relays { 5 } else { 3 }).enumerate() {
        compare(&format!("gradient_{index}"), input.grad(&grads).unwrap(), expected);
    }
}
