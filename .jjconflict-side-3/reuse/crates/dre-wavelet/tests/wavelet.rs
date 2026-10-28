//! PLAN §7 `wavelet` acceptance tests (spec §3.2, §11.6, §15.2) on the ndarray CPU backend, compared with the `f64`
//! reference in `reference/mod.rs` under the PLAN §8.1 rule. `wavelet_gradients_match_reference_cuda` repeats the
//! gradient oracle on the Spark GPU, including the BF16 gate.

mod reference;

use burn::prelude::*;
use burn::tensor::TensorData;
use dre_core::{BatchLayout, DocumentLayout, gpu_device};
use dre_wavelet::{
    DB10_LOW_PASS, FILTER_TAPS, SCALE_GATE_ACTIVATION, TRANSFORM, WaveletDyn, WaveletDynConfig, level_reach,
    symmetric_position, undecimated_bands,
};
use reference::{Rng, Shape, Weights, assert_close, rms};

fn batch_layout(shape: &Shape, device: &Device) -> BatchLayout {
    BatchLayout::new(DocumentLayout::packed(shape.documents.clone(), shape.tokens), device)
}

fn upload(values: &[f64], dims: [usize; 3], device: &Device) -> Tensor<3> {
    let values: Vec<f32> = values.iter().map(|&x| x as f32).collect();
    Tensor::from_data(TensorData::new(values, dims), device)
}

fn download<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_into_vec::<f32>().expect("f32 tensor data")
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&x| x as f64).collect()
}

fn weights_of(module: &WaveletDyn) -> Weights {
    let read = |tensor: Tensor<2>| widen(&download(tensor));
    Weights {
        input: read(module.input.weight.val()),
        gate: read(module.gate.weight.val()),
        gate_bias: widen(&download(module.gate.bias.as_ref().expect("gate bias").val())),
        value: read(module.value.weight.val()),
        output: read(module.output.weight.val()),
    }
}

/// Normal random input with NaN at every padded position.
fn input_with_nan_padding(shape: &Shape, rng: &mut Rng) -> Vec<f64> {
    let valid = shape.valid();
    (0..shape.rows() * shape.d).map(|i| if valid[i / shape.d] { rng.normal() } else { f64::NAN }).collect()
}

fn assert_padding_zero(name: &str, values: &[f32], shape: &Shape, width: usize) {
    let valid = shape.valid();
    let nonzero = values.iter().enumerate().filter(|&(i, &x)| !valid[i / width] && x != 0.0).count();
    assert_eq!(nonzero, 0, "{name}: {nonzero} padded entries are not exactly 0");
}

#[test]
fn wavelet_filter_and_contract_constants() {
    assert_eq!(TRANSFORM, "undecimated");
    assert_eq!(SCALE_GATE_ACTIVATION, "sigmoid");
    assert_eq!(FILTER_TAPS, 20);

    // db10 conditions: sum √2, unit energy, orthogonal to even shifts, ten vanishing moments of the high-pass.
    let h = DB10_LOW_PASS;
    assert!((h.iter().sum::<f64>() - std::f64::consts::SQRT_2).abs() < 1e-14);
    assert!((h.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-14);
    for shift in (2..FILTER_TAPS).step_by(2) {
        let product: f64 = (0..FILTER_TAPS - shift).map(|k| h[k] * h[k + shift]).sum();
        assert!(product.abs() < 1e-14, "shift {shift}: {product}");
    }
    for power in 0..10 {
        let terms: Vec<f64> =
            (0..FILTER_TAPS).map(|k| if k % 2 == 0 { 1.0 } else { -1.0 } * (k as f64).powi(power) * h[k]).collect();
        let size: f64 = terms.iter().map(|t| t.abs()).sum();
        assert!(terms.iter().sum::<f64>().abs() <= 1e-11 * size, "moment {power}");
    }

    // Reaches per dilation and the composite support of four levels (spec §3.2 estimate).
    assert_eq!([1, 2, 4, 8].map(level_reach), [(9, 10), (19, 19), (38, 38), (76, 76)]);
    let support: usize = 1 + [1, 2, 4, 8].map(|d| level_reach(d).0 + level_reach(d).1).iter().sum::<usize>();
    assert_eq!(support, 1 + (20 - 1) * (16 - 1));

    // Symmetric extension: ... x2 x1 x0 | x0 x1 x2 | x2 x1 x0 ..., and length-1 documents map everything to 0.
    let mapped: Vec<usize> = (-7..10).map(|i| symmetric_position(i, 3)).collect();
    assert_eq!(mapped, vec![0, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2]);
    assert!((-300..300).all(|i| symmetric_position(i, 1) == 0));
}

#[test]
fn wavelet_shapes_and_forward_match_reference() {
    let device = Device::ndarray();
    let precision = dre_core::Precision::Fp32;
    let config = WaveletDynConfig::new().with_precision(precision);
    assert_eq!((config.d_model, config.levels, config.groups), (512, 4, 8));
    let shape = Shape { documents: vec![vec![5, 300, 19, 400], vec![777]], tokens: 777, d: 512, groups: 8, levels: 4 };
    let layout = batch_layout(&shape, &device);
    let module = config.init(&device);
    assert_eq!(module.num_params(), 3 * 512 * 512 + 5 * 8 * 512 + 5 * 8);

    let mut rng = Rng::new(11);
    let u = input_with_nan_padding(&shape, &mut rng);
    let u_tensor = upload(&u, [2, 777, 512], &device);

    let v = module.input.forward(upload(&u.iter().map(|x| if x.is_nan() { 0.0 } else { *x }).collect::<Vec<_>>(), [2, 777, 512], &device));
    let bands = undecimated_bands(v, &layout.host, 4);
    assert_eq!(bands.len(), 5);
    for band in &bands {
        assert_eq!(band.dims(), [2, 777, 512]);
    }
    assert_eq!(module.gates(u_tensor.clone()).dims(), [2, 777, 5, 8]);

    let output = module.forward(u_tensor, &layout);
    assert_eq!(output.dims(), [2, 777, 512]);
    let output = download(output);
    assert_padding_zero("output", &output, &shape, 512);

    let reference = reference::forward(&shape, &weights_of(&module), &u);
    assert_close("forward d=512 T=777", &output, &reference.out, rms(&reference.out));
}

#[test]
fn wavelet_reconstruction_and_short_document_boundaries() {
    let device = Device::ndarray();
    // Every document is shorter than the four-level support (286), several shorter than one 20-tap filter.
    let shape = Shape {
        documents: vec![vec![1, 2, 3, 7, 19, 20, 21, 64, 300], vec![286, 285, 9]],
        tokens: 777,
        d: 16,
        groups: 8,
        levels: 4,
    };
    let host = DocumentLayout::packed(shape.documents.clone(), shape.tokens);
    let dims = [2, 777, 16];
    let valid = shape.valid();
    let mut rng = Rng::new(23);

    // Random signal: bands equal the reference, keep length T, add up to the input.
    let v: Vec<f64> = (0..shape.rows() * 16).map(|i| if valid[i / 16] { rng.normal() } else { 0.0 }).collect();
    let bands = undecimated_bands(upload(&v, dims, &device), &host, 4);
    let expected = reference::bands(&shape, &v);
    let mut sum = vec![0.0f32; v.len()];
    for (j, (band, expected)) in bands.into_iter().zip(&expected).enumerate() {
        assert_eq!(band.dims(), dims);
        let band = download(band);
        assert_close(&format!("band {j} short documents"), &band, expected, rms(&v));
        sum.iter_mut().zip(&band).for_each(|(s, b)| *s += b);
    }
    assert_close("reconstruction sum of bands = V", &sum, &v, rms(&v));

    // Constant documents stay constant in A4 and give zero details, at every document length.
    let constants: Vec<f64> = (0..shape.spans().len() * 16).map(|_| rng.normal()).collect();
    let mut constant_v = vec![0.0; shape.rows() * 16];
    for (document, (start, length)) in shape.spans().into_iter().enumerate() {
        for t in start..start + length {
            constant_v[t * 16..(t + 1) * 16].copy_from_slice(&constants[document * 16..(document + 1) * 16]);
        }
    }
    let bands = undecimated_bands(upload(&constant_v, dims, &device), &host, 4);
    for (j, band) in bands.into_iter().enumerate() {
        let expected = if j == 0 { constant_v.clone() } else { vec![0.0; constant_v.len()] };
        assert_close(&format!("band {j} of constant documents"), &download(band), &expected, rms(&constant_v));
    }

    // An impulse inside the 20-position document of row 0 stays inside that document in every band.
    let (start, length) = shape.spans()[5];
    assert_eq!(length, 20);
    let mut impulse = vec![0.0; shape.rows() * 16];
    impulse[start * 16 + 3] = 1.0;
    for (j, band) in undecimated_bands(upload(&impulse, dims, &device), &host, 4).into_iter().enumerate() {
        let band = download(band);
        let outside = band.iter().enumerate().filter(|&(i, &x)| !(start..start + length).contains(&(i / 16)) && x != 0.0).count();
        assert_eq!(outside, 0, "band {j}: impulse leaked outside its document");
    }
}

#[test]
fn wavelet_receptive_field_matches_support_estimate() {
    let device = Device::ndarray();
    let shape = Shape { documents: vec![vec![777]], tokens: 777, d: 4, groups: 1, levels: 4 };
    let host = DocumentLayout::packed(shape.documents.clone(), shape.tokens);
    let position = 388;
    let mut impulse = vec![0.0; 777 * 4];
    impulse[position * 4] = 1.0;

    let total_left: usize = [1, 2, 4, 8].map(|d| level_reach(d).0).iter().sum();
    let total_right: usize = [1, 2, 4, 8].map(|d| level_reach(d).1).iter().sum();
    let expected_widths = [286, 286, 134, 58, 20]; // A4, D4, D3, D2, D1: 1 + 19·(2^j − 1)
    let reference = reference::bands(&shape, &impulse);
    let bands = undecimated_bands(upload(&impulse, [1, 777, 4], &device), &host, 4);
    for (j, band) in bands.into_iter().enumerate() {
        let band = download(band);
        let support: Vec<usize> = (0..777).filter(|&t| band[t * 4] != 0.0).collect();
        let reference_support = (0..777).filter(|&t| reference[j][t * 4] != 0.0).count();
        eprintln!("band {j}: measured support {} positions, reference {reference_support}", support.len());
        assert_eq!(support.len(), expected_widths[j], "band {j} support");
        assert_eq!(reference_support, expected_widths[j], "band {j} reference support");
        assert!(support.iter().all(|&t| t + total_right >= position && t <= position + total_left));
        assert!((1..4).all(|c| (0..777).all(|t| band[t * 4 + c] == 0.0)), "band {j}: channels mixed");
    }
    assert_eq!(total_left + total_right + 1, 1 + (20 - 1) * (16 - 1));
}

#[test]
fn wavelet_gates_are_sigmoid_with_paper_initialisation() {
    let device = Device::ndarray();
    let module = WaveletDynConfig::new().init(&device);

    // U = 0 leaves only the bias: σ(b_0) = 0.88 for A4, σ(b_j) = 0.12 for every detail band and group.
    let gates = download(module.gates(Tensor::zeros([1, 777, 512], &device)));
    for (i, &g) in gates.iter().enumerate() {
        let expected = if (i / 8) % 5 == 0 { 0.88 } else { 0.12 };
        assert!((g as f64 - expected).abs() < 1e-6, "gate {i}: {g}");
    }
    // Independent sigmoids, not a softmax over scales: per token and group the five gates sum to 1.36, not 1.
    for token in 0..777 {
        for group in 0..8 {
            let sum: f32 = (0..5).map(|band| gates[(token * 5 + band) * 8 + group]).sum();
            assert!((sum as f64 - 1.36).abs() < 1e-5, "token {token} group {group}: gate sum {sum}");
        }
    }

    let shape = Shape { documents: vec![vec![777]], tokens: 777, d: 512, groups: 8, levels: 4 };
    let mut rng = Rng::new(5);
    let u: Vec<f64> = (0..777 * 512).map(|_| 3.0 * rng.normal()).collect();
    let gates = download(module.gates(upload(&u, [1, 777, 512], &device)));
    assert!(gates.iter().all(|&g| g > 0.0 && g < 1.0));
    let expected = reference::forward(&shape, &weights_of(&module), &u).gates;
    assert_close("gates", &gates, &expected, rms(&expected));
}

#[test]
fn wavelet_gradients_match_reference() {
    gradients_match_reference(Device::ndarray().autodiff(), dre_core::Precision::Fp32);
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn wavelet_gradients_match_reference_cuda() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let precision = if std::env::var_os("DRE_BF16_GATE").is_some() { dre_core::Precision::Bf16Mixed } else { dre_core::Precision::Fp32 };
    gradients_match_reference(gpu_device().autodiff(), precision);
}

fn gradients_match_reference(device: Device, precision: dre_core::Precision) {
    let shape = Shape { documents: vec![vec![3, 40, 9], vec![61]], tokens: 70, d: 128, groups: 8, levels: 4 };
    let layout = batch_layout(&shape, &device);
    let module = WaveletDynConfig::new().with_precision(precision).with_d_model(128).init(&device);
    let weights = weights_of(&module);

    let mut rng = Rng::new(97);
    let u = input_with_nan_padding(&shape, &mut rng);
    let r: Vec<f64> = (0..shape.rows() * 128).map(|_| rng.normal()).collect();

    let forward = reference::forward(&shape, &weights, &u);
    let expected = reference::backward(&shape, &weights, &forward, &r);
    reference::check_gradient_by_differences(&shape, &weights, &u, &r, &expected, 8, &mut rng);

    let u_tensor = upload(&u, [2, 70, 128], &device).require_grad();
    let output = module.forward(u_tensor.clone(), &layout);
    let loss = (output.clone() * upload(&r, [2, 70, 128], &device)).sum();
    let gradients = loss.backward();

    let output = download(output);
    assert_padding_zero("output", &output, &shape, 128);
    assert!(output.iter().all(|x| x.is_finite()));
    assert_close("forward d=128", &output, &forward.out, rms(&forward.out));

    let grad = |tensor: Tensor<2>| download(tensor.grad(&gradients).expect("parameter gradient"));
    let du = download(u_tensor.grad(&gradients).expect("input gradient"));
    assert!(du.iter().all(|x| x.is_finite()));
    assert_padding_zero("dL/dU", &du, &shape, 128);
    assert_close("dL/dU", &du, &expected.u, rms(&expected.u));
    assert_close("dL/dW_i", &grad(module.input.weight.val()), &expected.weights.input, rms(&expected.weights.input));
    assert_close("dL/dW_g", &grad(module.gate.weight.val()), &expected.weights.gate, rms(&expected.weights.gate));
    let bias = module.gate.bias.as_ref().expect("gate bias").val();
    let d_bias = download(bias.grad(&gradients).expect("gate bias gradient"));
    assert_close("dL/db", &d_bias, &expected.weights.gate_bias, rms(&expected.weights.gate_bias));
    assert_close("dL/dW_v", &grad(module.value.weight.val()), &expected.weights.value, rms(&expected.weights.value));
    assert_close("dL/dW_o", &grad(module.output.weight.val()), &expected.weights.output, rms(&expected.weights.output));
}

#[test]
fn wavelet_packed_rows_equal_separate_documents() {
    let device = Device::ndarray();
    let module = WaveletDynConfig::new().init(&device);
    let shape = Shape { documents: vec![vec![5, 300, 19, 400], vec![777]], tokens: 777, d: 512, groups: 8, levels: 4 };
    let mut rng = Rng::new(41);
    let u = input_with_nan_padding(&shape, &mut rng);

    let packed = download(module.forward(upload(&u, [2, 777, 512], &device), &batch_layout(&shape, &device)));
    assert_padding_zero("packed output", &packed, &shape, 512);

    for (start, length) in shape.spans() {
        let alone = Shape { documents: vec![vec![length]], tokens: length, d: 512, groups: 8, levels: 4 };
        let document = &u[start * 512..(start + length) * 512];
        let separate = download(module.forward(upload(document, [1, length, 512], &device), &batch_layout(&alone, &device)));
        let separate = widen(&separate);
        assert_close(
            &format!("packed vs separate document at row {start} length {length}"),
            &packed[start * 512..(start + length) * 512],
            &separate,
            rms(&separate),
        );
    }
}

#[test]
fn exp004_wavelet_dirty_padding_preserves_all_parameter_gradients_cpu() {
    let device=Device::ndarray().autodiff();device.seed(2741);
    let module=WaveletDynConfig::new().with_d_model(4).with_groups(2).with_levels(2).init(&device);
    for documents in [vec![vec![3,2],vec![2],vec![],vec![]],vec![vec![];4]] {
        let shape=Shape {documents,tokens:6,d:4,groups:2,levels:2};
        let layout=batch_layout(&shape,&device);let valid=shape.valid();
        let clean=(0..shape.rows()*shape.d).map(|index|if valid[index/shape.d] {
            (index%13) as f64*0.1-0.5
        } else {0.0}).collect::<Vec<_>>();
        let sentinels=[f64::NAN,f64::INFINITY,f64::NEG_INFINITY,1e30,-1e30];
        let dirty=clean.iter().enumerate().map(|(index,&value)|if valid[index/shape.d] {value}
            else {sentinels[index%sentinels.len()]}).collect::<Vec<_>>();
        let run=|values:&[f64]| {
            let input=upload(values,[4,6,4],&device).require_grad();
            let output=module.forward(input.clone(),&layout);
            let gradients=output.clone().square().sum().backward();
            let gradient=|parameter:Tensor<2>|download(parameter.grad(&gradients).expect("parameter gradient"));
            (download(output),download(input.grad(&gradients).expect("input gradient")),
                [gradient(module.input.weight.val()),gradient(module.gate.weight.val()),
                    gradient(module.value.weight.val()),gradient(module.output.weight.val())],
                download(module.gate.bias.as_ref().unwrap().val().grad(&gradients).expect("gate bias gradient")))
        };
        let expected=run(&clean);let actual=run(&dirty);
        assert_eq!(actual,expected,"wavelet must select inactive inputs before any projection");
        assert!(actual.0.iter().chain(&actual.1).chain(&actual.3).all(|value|value.is_finite()));
        assert!(actual.2.iter().flatten().all(|value|value.is_finite()));
        assert_padding_zero("CPU output",&actual.0,&shape,4);
        assert_padding_zero("CPU input gradient",&actual.1,&shape,4);
        if valid.iter().all(|&active|!active) {
            assert!(actual.2.iter().flatten().chain(&actual.3).all(|&value|value==0.0));
        } else {
            assert!(actual.1.iter().any(|&value|value!=0.0));
            assert!(actual.2.iter().flatten().any(|&value|value!=0.0));
        }
    }
}
