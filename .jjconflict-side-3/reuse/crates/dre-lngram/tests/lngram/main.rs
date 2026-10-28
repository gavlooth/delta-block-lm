//! `lngram` acceptance tests (PLAN §7, spec §3.3 / §11.5), run on the ndarray CPU backend. The two oracle comparisons
//! have `_cuda` twins that repeat them on the Spark GPU, including the BF16 gate.

mod oracle;

use std::collections::{BTreeMap, HashSet};

use burn::module::{Initializer, Param};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::Gradients;
use dre_core::{BatchLayout, DocumentLayout};
use dre_lngram::{HostRouteCodes, Lngram, LngramConfig, SurrogateGradient};
use oracle::{Batch, Evaluation, Group, Model, Surrogate};

// ---------------------------------------------------------------- helpers

/// SplitMix64 with Box–Muller normals; values are rounded to `f32` so host and device see the same numbers.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }

    fn normals(&mut self, n: usize, mean: f64, std: f64) -> Vec<f64> {
        (0..n).map(|_| (mean + std * self.normal()) as f32 as f64).collect()
    }
}

fn cpu() -> Device {
    Device::ndarray().autodiff()
}

fn tensor<const D: usize>(values: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(values.iter().map(|&v| v as f32).collect::<Vec<_>>(), shape), device)
}

fn host<const D: usize>(tensor: &Tensor<D>) -> Vec<f64> {
    tensor.to_data().iter::<f64>().collect()
}

fn host_int<const D: usize>(tensor: &Tensor<D, Int>) -> Vec<i64> {
    tensor.to_data().iter::<i64>().collect()
}

fn param_grad<const D: usize>(param: &Param<Tensor<D>>, grads: &Gradients) -> Vec<f64> {
    host(&param.grad(grads).expect("parameter gradient"))
}

fn limits(layout: &DocumentLayout) -> (Vec<i64>, Vec<i64>) {
    let (starts, ends) = layout.limits_host();
    (starts.into_iter().map(i64::from).collect(), ends.into_iter().map(i64::from).collect())
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len().max(1) as f64).sqrt()
}

/// PLAN §8.1 elementwise rule `|x − ref| ≤ atol + rtol·|ref|`, `rtol = 1e-4`, `atol = 1e-5·max(rms(ref), s₀)`
/// with `s₀` the fixture's rms of the quantity unless declared. Prints the required report.
fn compare(name: &str, actual: &[f64], reference: &[f64], declared_scale: Option<f64>) {
    assert_eq!(actual.len(), reference.len(), "{name}: length");
    let bf16 = std::env::var_os("DRE_BF16_GATE").is_some();
    let rtol = if bf16 { 1e-2 } else { 1e-4 };
    let scale = declared_scale.unwrap_or_else(|| rms(reference));
    assert!(scale > 0.0, "{name}: degenerate fixture (rms 0) without a declared scale");
    let atol = if bf16 { 1e-2 } else { 1e-5 } * rms(reference).max(scale);
    let mut max_abs = 0.0f64;
    let mut max_rel: Option<f64> = None;
    let mut ratios = Vec::with_capacity(actual.len());
    let mut failures = 0usize;
    for (&x, &r) in actual.iter().zip(reference) {
        let error = (x - r).abs();
        max_abs = max_abs.max(error);
        if r.abs() > atol {
            max_rel = Some(max_rel.unwrap_or(0.0).max(error / r.abs()));
        }
        let ratio = error / (atol + rtol * r.abs());
        if !(ratio <= 1.0) {
            failures += 1;
        }
        ratios.push(ratio);
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    let quantile = |q: f64| ratios[((ratios.len() - 1) as f64 * q).round() as usize];
    let relative = max_rel.map_or("none: all |ref| <= atol".to_string(), |v| format!("{v:.3e}"));
    println!(
        "{name}: n={} max_abs={max_abs:.3e} max_rel={relative} p50={:.3e} p99={:.3e} atol={atol:.3e}",
        actual.len(),
        quantile(0.5),
        quantile(0.99)
    );
    assert_eq!(failures, 0, "{name}: {failures} elements outside the §8.1 tolerance");
}

fn config_of(model: &Model, precision: dre_core::Precision) -> LngramConfig {
    LngramConfig::new()
        .with_precision(precision)
        .with_d_model(model.d_model)
        .with_routes(model.routes)
        .with_bits_per_route(model.bits)
        .with_memory_dim(model.memory_dim)
        .with_orders(model.orders.clone())
        .with_smoothing(model.smoothing_gamma.is_some())
        .with_smoothing_dilation(Some(model.dilation))
        .with_surrogate(match model.surrogate {
            Surrogate::OneBit => SurrogateGradient::OneBit,
            Surrogate::LocalExpectation => SurrogateGradient::LocalExpectation,
        })
        .with_surrogate_temperature(model.temperature)
        .with_surrogate_scale(model.scale)
        .with_norm_epsilon(model.epsilon)
}

/// A module holding exactly the oracle's parameters.
fn load(model: &Model, device: &Device, precision: dre_core::Precision) -> Lngram {
    let (d, kv) = (model.d_model, model.routes * model.memory_dim);
    let p1 = |v: &[f64]| Param::from_tensor(tensor(v, [v.len()], device));
    let p2 = |v: &[f64], rows: usize| Param::from_tensor(tensor(v, [rows, v.len() / rows], device));
    let mut lngram = config_of(model, precision).init(device);
    lngram.discretizer_norm.gamma = p1(&model.discretizer_gamma);
    lngram.discretizer.weight = p2(&model.discretizer, d);
    for (table, values) in lngram.tables.iter_mut().zip(&model.tables) {
        table.weight = p2(values, values.len() / model.memory_dim);
    }
    lngram.key.weight = p2(&model.key_weight, kv);
    lngram.key.bias = Some(p1(&model.key_bias));
    lngram.value.weight = p2(&model.value_weight, kv);
    lngram.value.bias = Some(p1(&model.value_bias));
    lngram.query_norm.gamma = p1(&model.query_gamma);
    lngram.key_norm.gamma = p1(&model.key_gamma);
    if let Some(smoothing) = lngram.smoothing.as_mut() {
        smoothing.norm.gamma = p1(model.smoothing_gamma.as_ref().expect("smoothing on"));
        smoothing.weight = p2(model.smoothing_weight.as_ref().expect("smoothing on"), 3);
    }
    lngram
}

fn random_model(rng: &mut Rng, d: usize, routes: usize, bits: usize, memory_dim: usize, surrogate: Surrogate) -> Model {
    let orders = vec![2, 3];
    let alphabet = 1usize << bits;
    let kv = routes * memory_dim;
    Model {
        d_model: d,
        routes,
        bits,
        memory_dim,
        discretizer_gamma: rng.normals(d, 1.0, 0.1),
        discretizer: rng.normals(d * routes * bits, 0.0, 1.0 / (d as f64).sqrt()),
        tables: orders.iter().map(|&n| rng.normals(routes * alphabet.pow(n as u32) * memory_dim, 0.0, 1.0)).collect(),
        key_weight: rng.normals(kv * d, 0.0, 1.0 / (kv as f64).sqrt()),
        key_bias: rng.normals(d, 0.0, 0.5),
        value_weight: rng.normals(kv * d, 0.0, 1.0 / (kv as f64).sqrt()),
        value_bias: rng.normals(d, 0.0, 0.5),
        query_gamma: rng.normals(d, 1.0, 0.1),
        key_gamma: rng.normals(d, 1.0, 0.1),
        smoothing_gamma: Some(rng.normals(d, 1.0, 0.1)),
        smoothing_weight: Some(rng.normals(3 * d, 0.0, 0.5)),
        dilation: 3,
        orders,
        epsilon: 1e-6,
        temperature: 1.0,
        scale: 1.0,
        surrogate,
    }
}

/// Small non-default fixture: `d=24, R=4, m=4, d_m=3`, rows `[[4,1,6],[7]]` in `T=11`. The six-token document
/// repeats `A B A B A B`, so several table rows are hit more than once. Seeds are skipped until every valid
/// discretizer logit is at least `1e-3` away from 0, so `f32` and `f64` agree on every bit.
fn small_fixture(surrogate: Surrogate) -> (Model, Batch, DocumentLayout, Evaluation) {
    let rows = vec![vec![4, 1, 6], vec![7]];
    let tokens = 11;
    let layout = DocumentLayout::packed(rows, tokens);
    let (starts, ends) = limits(&layout);
    for seed in 1.. {
        let mut rng = Rng(seed);
        let model = random_model(&mut rng, 24, 4, 4, 3, surrogate);
        let d = model.d_model;
        let mut u = rng.normals(2 * tokens * d, 0.0, 1.0);
        for (target, source) in [(7, 5), (8, 6), (9, 5), (10, 6)] {
            let copy = u[source * d..(source + 1) * d].to_vec();
            u[target * d..(target + 1) * d].copy_from_slice(&copy);
        }
        let batch = Batch {
            batch: 2,
            tokens,
            starts: starts.clone(),
            ends: ends.clone(),
            u,
            loss_weight: rng.normals(2 * tokens * d, 0.0, 1.0),
        };
        let evaluation = oracle::evaluate(&model, &batch, None);
        let width = model.routes * model.bits;
        let margin = (0..2 * tokens)
            .filter(|&p| ends[p] >= starts[p])
            .flat_map(|p| evaluation.logits[p * width..(p + 1) * width].iter().map(|z| z.abs()))
            .fold(f64::INFINITY, f64::min);
        if margin > 1e-3 {
            return (model, batch, layout, evaluation);
        }
    }
    unreachable!()
}

/// PLAN §8.1 check of the oracle's chain-rule gradient by central differences along 8 random unit directions per
/// parameter group, addresses held fixed (the discrete path is never differenced).
fn check_oracle_gradients(model: &Model, batch: &Batch, evaluation: &Evaluation) {
    let mut rng = Rng(0x5EED);
    for group in Group::all(model) {
        let gradient = group.gradient(&evaluation.chain).clone();
        let (mut probe_model, mut probe_batch) = (model.clone(), batch.clone());
        let values = group.values_mut(&mut probe_model, &mut probe_batch).clone();
        // Table directions only on the hit rows: other rows do not enter the function.
        let support: Vec<usize> = match group {
            Group::Table(o) => {
                let hit: HashSet<i64> = evaluation.addresses[o].iter().copied().filter(|&a| a >= 0).collect();
                (0..values.len()).filter(|i| hit.contains(&((i / model.memory_dim) as i64))).collect()
            }
            _ => (0..values.len()).collect(),
        };
        let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(norm > 0.0, "{group:?}: zero gradient on the fixture");
        let h = 1e-6 * rms(&values).max(1.0);
        let mut worst = 0.0f64;
        for _ in 0..8 {
            let mut direction = vec![0.0; values.len()];
            for &i in &support {
                direction[i] = rng.normal();
            }
            let length = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
            direction.iter_mut().for_each(|v| *v /= length);
            let loss_at = |step: f64| {
                let (mut m, mut b) = (model.clone(), batch.clone());
                for (x, v) in group.values_mut(&mut m, &mut b).iter_mut().zip(&direction) {
                    *x += step * v;
                }
                oracle::evaluate(&m, &b, Some(&evaluation.symbols)).loss
            };
            let difference = (loss_at(h) - loss_at(-h)) / (2.0 * h);
            let analytic: f64 = gradient.iter().zip(&direction).map(|(g, v)| g * v).sum();
            let allowed = 1e-7 * norm + 1e-6 * difference.abs();
            let error = (analytic - difference).abs();
            worst = worst.max(error / allowed);
            assert!(error <= allowed, "{group:?}: ⟨∇f,v⟩={analytic:e} vs difference {difference:e}");
        }
        println!("oracle finite differences {group:?}: worst error / allowed = {worst:.3e}");
    }
}

fn run_oracle_comparison(surrogate: Surrogate, device: Device, precision: dre_core::Precision) {
    let (model, batch, layout, evaluation) = small_fixture(surrogate);
    check_oracle_gradients(&model, &batch, &evaluation);

    let lngram = load(&model, &device, precision);
    let (b, t, d) = (batch.batch, batch.tokens, model.d_model);
    let batch_layout = BatchLayout::new(layout.clone(), &device);
    let u = tensor(&batch.u, [b, t, d], &device).require_grad();
    let output = lngram.forward(u.clone(), &batch_layout);
    let grads = (output.clone() * tensor(&batch.loss_weight, [b, t, d], &device)).sum().backward();

    // Codes and addresses: exact.
    let codes = lngram.route_codes(u.clone(), &batch_layout);
    let valid: Vec<bool> = (0..b * t).map(|p| batch.ends[p] >= batch.starts[p]).collect();
    let gpu_symbols = host_int(&codes.symbols);
    for p in (0..b * t).filter(|&p| valid[p]) {
        let range = p * model.routes..(p + 1) * model.routes;
        assert_eq!(gpu_symbols[range.clone()], evaluation.symbols[range], "symbols at position {p}");
    }
    for (o, addresses) in codes.addresses.iter().enumerate() {
        assert_eq!(host_int(addresses), evaluation.addresses[o], "addresses of order {}", model.orders[o]);
    }

    let out = host(&output);
    compare("output", &out, &evaluation.output, None);
    for p in (0..b * t).filter(|&p| !valid[p]) {
        assert!(out[p * d..(p + 1) * d].iter().all(|&v| v == 0.0), "padded output {p} is not exactly 0");
    }

    let total = &evaluation.total;
    let u_grad = host(&u.grad(&grads).expect("input gradient"));
    compare("grad u", &u_grad, &total.u, None);
    for p in (0..b * t).filter(|&p| !valid[p]) {
        assert!(u_grad[p * d..(p + 1) * d].iter().all(|&v| v == 0.0), "padded input gradient {p} is not 0");
    }
    // Discretizer: the surrogate against the independent f64 surrogate.
    compare("grad discretizer W_q (surrogate)", &param_grad(&lngram.discretizer.weight, &grads), &total.discretizer, None);
    compare("grad discretizer RMSNorm", &param_grad(&lngram.discretizer_norm.gamma, &grads), &total.discretizer_gamma, None);
    let surrogate_norm = total.discretizer.iter().map(|g| g * g).sum::<f64>().sqrt();
    assert!(surrogate_norm > 0.0, "the surrogate gives the discretizer a nonzero gradient");
    // Lookup gradients: nonzero only on indexed rows, summed over repeated indices.
    for (o, table) in lngram.tables.iter().enumerate() {
        let gradient = param_grad(&table.weight, &grads);
        let mut hits: BTreeMap<i64, usize> = BTreeMap::new();
        for &address in evaluation.addresses[o].iter().filter(|&&a| a >= 0) {
            *hits.entry(address).or_default() += 1;
        }
        let dm = model.memory_dim;
        for (row, values) in gradient.chunks(dm).enumerate() {
            if !hits.contains_key(&(row as i64)) {
                assert!(values.iter().all(|&v| v == 0.0), "order {} row {row} not indexed but has gradient", model.orders[o]);
            }
        }
        let repeated = hits.values().filter(|&&count| count > 1).count();
        assert!(repeated > 0, "fixture must repeat at least one row of order {}", model.orders[o]);
        println!("order {}: {} rows hit, {repeated} hit more than once", model.orders[o], hits.len());
        // Compare the hit rows (all other rows are exactly zero above).
        let (gpu_rows, ref_rows): (Vec<f64>, Vec<f64>) = hits
            .keys()
            .flat_map(|&row| (0..dm).map(move |c| row as usize * dm + c))
            .map(|i| (gradient[i], total.tables[o][i]))
            .unzip();
        compare(&format!("grad table order {}", model.orders[o]), &gpu_rows, &ref_rows, None);
    }
    compare("grad W_K", &param_grad(&lngram.key.weight, &grads), &total.key_weight, None);
    compare("grad b_K", &param_grad(lngram.key.bias.as_ref().unwrap(), &grads), &total.key_bias, None);
    compare("grad W_V", &param_grad(&lngram.value.weight, &grads), &total.value_weight, None);
    compare("grad b_V", &param_grad(lngram.value.bias.as_ref().unwrap(), &grads), &total.value_bias, None);
    compare("grad query RMSNorm", &param_grad(&lngram.query_norm.gamma, &grads), &total.query_gamma, None);
    compare("grad key RMSNorm", &param_grad(&lngram.key_norm.gamma, &grads), &total.key_gamma, None);
    let smoothing = lngram.smoothing.as_ref().expect("smoothing on");
    compare("grad smoothing RMSNorm", &param_grad(&smoothing.norm.gamma, &grads), &total.smoothing_gamma, None);
    compare("grad smoothing taps", &param_grad(&smoothing.weight, &grads), &total.smoothing_weight, None);
}

// ---------------------------------------------------------------- tests

/// Symbols and addresses for every combination of the declared space (spec sizes `R=64, K=16`, `n∈{2,3}`): the
/// round trip recovers the symbols, distinct suffixes get distinct rows and the rows fill `[0, R·K^n)` exactly.
/// Positions before the first full n-gram of each packed document and padding get no row.
#[test]
fn lngram_address_space_round_trip() {
    let device = Device::ndarray();
    let (routes, bits) = (64usize, 4usize);
    let alphabet = 1i64 << bits;
    let d = routes * bits;
    let config = LngramConfig::new().with_d_model(d).with_routes(routes).with_bits_per_route(bits).with_smoothing(false);
    let mut lngram = config.init(&device);
    // Identity discretizer: input channel r·m + j carries bit j of route r as ±1.
    let identity: Vec<f64> = (0..d * d).map(|i| if i / d == i % d { 1.0 } else { 0.0 }).collect();
    lngram.discretizer.weight = Param::from_tensor(tensor(&identity, [d, d], &device));
    let mut rng = Rng(7);

    for order_in_full in [2usize, 3] {
        let combinations = alphabet.pow(order_in_full as u32) as usize;
        let row0 = vec![order_in_full; combinations];
        let row1 = vec![1, 3, 2, 4];
        let tokens = combinations * order_in_full;
        let layout = DocumentLayout::packed(vec![row0, row1], tokens);
        let (starts, ends) = limits(&layout);
        // Intended symbols: in row 0, document k of route r spells combination (k + 37·r + 1) mod K^n.
        let mut symbols = vec![0i64; 2 * tokens * routes];
        for k in 0..combinations {
            for r in 0..routes {
                let combination = ((k + 37 * r + 1) % combinations) as i64;
                for i in 0..order_in_full {
                    symbols[(k * order_in_full + i) * routes + r] = (combination / alphabet.pow(i as u32)) % alphabet;
                }
            }
        }
        for p in tokens..2 * tokens {
            for r in 0..routes {
                symbols[p * routes + r] = rng.below(alphabet as usize) as i64;
            }
        }
        let u: Vec<f64> = (0..2 * tokens * d)
            .map(|i| {
                let (pr, j) = (i / bits, i % bits);
                if (symbols[pr] >> j) & 1 == 1 { 1.0 } else { -1.0 }
            })
            .collect();
        let batch_layout = BatchLayout::new(layout, &device);
        let codes = lngram.route_codes(tensor(&u, [2, tokens, d], &device), &batch_layout);
        let gpu_symbols = host_int(&codes.symbols);
        for p in (0..2 * tokens).filter(|&p| ends[p] >= starts[p]) {
            assert_eq!(gpu_symbols[p * routes..(p + 1) * routes], symbols[p * routes..(p + 1) * routes]);
        }
        for (address_tensor, &order) in codes.addresses.iter().zip(&lngram.orders) {
            let addresses = host_int(address_tensor);
            let rows = alphabet.pow(order as u32);
            for p in 0..2 * tokens {
                let t = (p % tokens) as i64;
                for r in 0..routes {
                    let address = addresses[p * routes + r];
                    if ends[p] < starts[p] || t - starts[p] < order as i64 - 1 {
                        assert_eq!(address, -1, "position {p} has no full {order}-gram");
                        continue;
                    }
                    // Round trip: route block, then base-K digits oldest first.
                    assert_eq!(address / rows, r as i64, "route of address at {p}");
                    let mut local = address % rows;
                    for i in 0..order {
                        let source = p + i + 1 - order;
                        assert_eq!(local % alphabet, symbols[source * routes + r], "digit {i} at {p}");
                        local /= alphabet;
                    }
                }
            }
            if order == order_in_full {
                let mut last: Vec<i64> = (0..combinations)
                    .flat_map(|k| {
                        let p = k * order + order - 1;
                        addresses[p * routes..(p + 1) * routes].to_vec()
                    })
                    .collect();
                last.sort_unstable();
                let expected: Vec<i64> = (0..routes as i64 * rows).collect();
                assert_eq!(last, expected, "order {order}: rows must cover [0, R·K^n) once (no aliasing)");
                println!("order {order}: {} distinct suffixes -> {} distinct rows", last.len(), expected.len());
            }
        }
    }
}

#[test]
fn lngram_matches_f64_oracle_one_bit_surrogate() {
    run_oracle_comparison(Surrogate::OneBit, cpu(), dre_core::Precision::Fp32);
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn lngram_matches_f64_oracle_one_bit_surrogate_cuda() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    run_oracle_comparison(Surrogate::OneBit, dre_core::gpu_device().autodiff(), gpu_precision());
}

#[test]
fn lngram_matches_f64_oracle_local_expectation_surrogate() {
    run_oracle_comparison(Surrogate::LocalExpectation, cpu(), dre_core::Precision::Fp32);
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn lngram_matches_f64_oracle_local_expectation_surrogate_cuda() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    run_oracle_comparison(Surrogate::LocalExpectation, dre_core::gpu_device().autodiff(), gpu_precision());
}

fn gpu_precision() -> dre_core::Precision {
    if std::env::var_os("DRE_BF16_GATE").is_some() { dre_core::Precision::Bf16Mixed } else { dre_core::Precision::Fp32 }
}

/// Spec sizes (`d=512, R=64, m=4, d_m=8`, `n∈{2,3}`, smoothing on): packed rows give the outputs and input
/// gradients of the same documents run one per row; padding content never reaches valid positions.
#[test]
fn lngram_packed_rows_equal_separate_documents() {
    let device = cpu();
    let lengths = [5usize, 1, 9, 3];
    let d = 512;
    for surrogate in [SurrogateGradient::OneBit, SurrogateGradient::LocalExpectation] {
        let mut rng = Rng(11);
        let mut lngram = LngramConfig::new()
            .with_surrogate(surrogate)
            .with_table_initializer(Initializer::Normal { mean: 0.0, std: 1.0 })
            .init(&device);
        let smoothing = lngram.smoothing.as_mut().expect("smoothing is on by default");
        smoothing.weight = Param::from_tensor(tensor(&rng.normals(3 * d, 0.0, 0.5), [3, d], &device));
        lngram.key.bias = Some(Param::from_tensor(tensor(&rng.normals(d, 0.0, 0.5), [d], &device)));

        let documents: Vec<Vec<f64>> = lengths.iter().map(|&n| rng.normals(n * d, 0.0, 1.0)).collect();
        let weights: Vec<Vec<f64>> = lengths.iter().map(|&n| rng.normals(n * d, 0.0, 1.0)).collect();

        // Packed: [[5, 1, 9], [3]] in T = 17; separate: one document per row in T = 9.
        let run = |rows: Vec<Vec<usize>>, tokens: usize, padding: f64| {
            let layout = DocumentLayout::packed(rows.clone(), tokens);
            let mut u = vec![0.0; rows.len() * tokens * d];
            let mut w = vec![0.0; rows.len() * tokens * d];
            let mut rng = Rng(99);
            for value in u.iter_mut() {
                *value = padding * rng.normal();
            }
            let mut index = 0;
            let mut placement = Vec::new();
            for (row, docs) in rows.iter().enumerate() {
                let mut start = row * tokens;
                for &n in docs {
                    u[start * d..(start + n) * d].copy_from_slice(&documents[index]);
                    w[start * d..(start + n) * d].copy_from_slice(&weights[index]);
                    placement.push(start);
                    start += n;
                    index += 1;
                }
            }
            let batch_layout = BatchLayout::new(layout.clone(), &device);
            let u_tensor = tensor(&u, [rows.len(), tokens, d], &device).require_grad();
            let output = lngram.forward(u_tensor.clone(), &batch_layout);
            let grads = (output.clone() * tensor(&w, [rows.len(), tokens, d], &device)).sum().backward();
            let out = host(&output);
            let grad = host(&u_tensor.grad(&grads).expect("input gradient"));
            let (starts, ends) = limits(&layout);
            for p in (0..rows.len() * tokens).filter(|&p| ends[p] < starts[p]) {
                assert!(out[p * d..(p + 1) * d].iter().all(|&v| v == 0.0), "padded output not 0");
                assert!(grad[p * d..(p + 1) * d].iter().all(|&v| v == 0.0), "padded input gradient not 0");
            }
            let wq = param_grad(&lngram.discretizer.weight, &grads);
            assert!(wq.iter().all(|v| v.is_finite()) && wq.iter().any(|&v| v != 0.0), "discretizer gradient");
            let gather = |values: &[f64]| -> Vec<f64> {
                placement
                    .iter()
                    .zip(&lengths)
                    .flat_map(|(&start, &n)| values[start * d..(start + n) * d].to_vec())
                    .collect()
            };
            (gather(&out), gather(&grad))
        };
        let (packed_out, packed_grad) = run(vec![vec![5, 1, 9], vec![3]], 17, 0.0);
        let (separate_out, separate_grad) = run(lengths.iter().map(|&n| vec![n]).collect(), 9, 0.0);
        let (noisy_out, noisy_grad) = run(vec![vec![5, 1, 9], vec![3]], 17, 1e3);
        compare(&format!("{surrogate:?} packed vs separate output"), &packed_out, &separate_out, None);
        compare(&format!("{surrogate:?} packed vs separate input gradient"), &packed_grad, &separate_grad, None);
        // Padding filled with values of size 1e3 instead of 0: any leak would show at that scale. Kernel choice
        // (autotune, scatter order) may differ between runs, so the check uses the §8.1 rule, not bit equality.
        compare(&format!("{surrogate:?} zero vs 1e3 padding output"), &noisy_out, &packed_out, None);
        compare(&format!("{surrogate:?} zero vs 1e3 padding input gradient"), &noisy_grad, &packed_grad, None);
    }
}

/// AdamW with weight decay 0 and fresh state: after one step only the indexed table rows change.
#[test]
fn lngram_adamw_step_changes_only_indexed_rows() {
    let (model, batch, layout, _) = small_fixture(Surrogate::OneBit);
    let device = cpu();
    let lngram = load(&model, &device, dre_core::Precision::Fp32);
    let (b, t, d) = (batch.batch, batch.tokens, model.d_model);
    let batch_layout = BatchLayout::new(layout, &device);
    let u = tensor(&batch.u, [b, t, d], &device);
    let codes = lngram.route_codes(u.clone(), &batch_layout);
    let before: Vec<Vec<f64>> = lngram.tables.iter().map(|table| host(&table.weight.val())).collect();
    let loss = (lngram.forward(u, &batch_layout) * tensor(&batch.loss_weight, [b, t, d], &device)).sum();
    let grads = GradientsParams::from_grads(loss.backward(), &lngram);
    let mut optimizer = AdamWConfig::new().with_weight_decay(0.0).init();
    let lngram = optimizer.step(1e-2, lngram, grads);
    for (o, table) in lngram.tables.iter().enumerate() {
        let after = host(&table.weight.val());
        let hit: HashSet<i64> = host_int(&codes.addresses[o]).into_iter().filter(|&a| a >= 0).collect();
        let dm = model.memory_dim;
        let mut changed = 0;
        for row in 0..after.len() / dm {
            let moved = (0..dm).any(|c| after[row * dm + c] != before[o][row * dm + c]);
            assert_eq!(moved, hit.contains(&(row as i64)), "order {} row {row}: changed={moved}", model.orders[o]);
            changed += moved as usize;
        }
        println!("order {}: {changed} rows changed, {} rows indexed", model.orders[o], hit.len());
    }
}

/// G5 toy task (reference initialization: zero tables, zero smoothing taps): regress a target that depends on the
/// bigram of token identities. With zero tables every counterfactual row difference is zero, so the discretizer
/// gradient is exactly 0 at step 0; once the tables learn, the surrogate gives it a finite nonzero gradient.
#[test]
fn lngram_toy_task_discretizer_gradient() {
    let device = cpu();
    let mut rng = Rng(2026);
    let d = 32;
    let mut lngram = LngramConfig::new().with_d_model(d).with_routes(8).with_memory_dim(4).init(&device);
    let rows = vec![vec![10, 14], vec![24], vec![7, 9, 8], vec![24]];
    let (batch, tokens, vocabulary) = (rows.len(), 24, 6);
    let layout = DocumentLayout::packed(rows, tokens);
    let (starts, ends) = limits(&layout);
    let embeddings = rng.normals(vocabulary * d, 0.0, 1.0);
    let pair_targets = rng.normals(vocabulary * vocabulary * d, 0.0, 1.0);
    let ids: Vec<usize> = (0..batch * tokens).map(|_| rng.below(vocabulary)).collect();
    let mut u = vec![0.0; batch * tokens * d];
    let mut target = vec![0.0; batch * tokens * d];
    let mut mask = vec![0.0; batch * tokens * d];
    for p in 0..batch * tokens {
        if ends[p] < starts[p] {
            continue;
        }
        u[p * d..(p + 1) * d].copy_from_slice(&embeddings[ids[p] * d..(ids[p] + 1) * d]);
        if (p % tokens) as i64 > starts[p] {
            let pair = ids[p - 1] * vocabulary + ids[p];
            target[p * d..(p + 1) * d].copy_from_slice(&pair_targets[pair * d..(pair + 1) * d]);
            mask[p * d..(p + 1) * d].iter_mut().for_each(|m| *m = 1.0);
        }
    }
    let count: f64 = mask.iter().sum();
    let batch_layout = BatchLayout::new(layout, &device);
    let u = tensor(&u, [batch, tokens, d], &device);
    let target = tensor(&target, [batch, tokens, d], &device);
    let mask = tensor(&mask, [batch, tokens, d], &device);
    let mut optimizer = AdamWConfig::new().with_weight_decay(0.0).init();
    let mut norms = Vec::new();
    let mut losses = Vec::new();
    for _ in 0..30 {
        let output = lngram.forward(u.clone(), &batch_layout);
        let loss = ((output - target.clone()).powi_scalar(2) * mask.clone()).sum().div_scalar(count);
        losses.push(host(&loss)[0]);
        let grads = loss.backward();
        let wq = param_grad(&lngram.discretizer.weight, &grads);
        norms.push(wq.iter().map(|g| g * g).sum::<f64>().sqrt());
        let grads = GradientsParams::from_grads(grads, &lngram);
        lngram = optimizer.step(1e-2, lngram, grads);
    }
    println!("toy task loss: {:?}", losses.iter().map(|l| format!("{l:.4}")).collect::<Vec<_>>());
    println!("toy task |dL/dW_q|: {:?}", norms.iter().map(|n| format!("{n:.3e}")).collect::<Vec<_>>());
    assert_eq!(norms[0], 0.0, "zero tables give no counterfactual difference at step 0");
    assert!(norms.iter().all(|n| n.is_finite()));
    assert!(norms[1..].iter().all(|&n| n > 0.0), "the discretizer gets a nonzero gradient once the tables learn");
    assert!(losses[losses.len() - 1] < losses[0], "toy loss decreases");
}

/// Route entropy, symbol and key utilization, hit diversity and code entropy: exact values on a hand-made code set,
/// then emitted per script group for codes at spec sizes.
#[test]
fn lngram_code_statistics() {
    // B=1, T=4, R=2, K=4, one order (n=2). Route 0 spells 0 1 2 3, route 1 is stuck on 1.
    let codes = HostRouteCodes {
        batch: 1,
        tokens: 4,
        routes: 2,
        alphabet: 4,
        orders: vec![2],
        valid: vec![true; 4],
        symbols: vec![0, 1, 1, 1, 2, 1, 3, 1],
        addresses: vec![vec![-1, -1, 4, 16 + 5, 9, 16 + 5, 14, 16 + 5]],
    };
    let all = codes.statistics(|_| true);
    assert_eq!(all.positions, 4);
    assert!((all.route_entropy_mean - 1.0).abs() < 1e-12 && all.route_entropy_min == 0.0);
    assert!((all.symbol_utilization - 5.0 / 8.0).abs() < 1e-12);
    assert_eq!(all.distinct_codes, 4);
    assert!((all.code_entropy - 2.0).abs() < 1e-12);
    let order = &all.orders[0];
    assert_eq!((order.lookups, order.distinct_rows, order.table_rows), (6, 4, 32));
    assert!((order.key_utilization - 4.0 / 32.0).abs() < 1e-12);
    // Hits: rows 4, 9, 14 once, row 21 three times.
    let expected_hit_entropy = -(3.0 * (1.0f64 / 6.0) * (1.0f64 / 6.0).log2() + 0.5 * 0.5f64.log2());
    assert!((order.hit_entropy - expected_hit_entropy).abs() < 1e-12);
    let groups = codes.statistics_by_group(&[0, 0, 1, 1]);
    assert_eq!(groups[&0].positions, 2);
    assert_eq!(groups[&1].orders[0].lookups, 4);

    // Emitted codes: two "scripts" with different input statistics, spec sizes.
    let device = Device::ndarray();
    let config = LngramConfig::new();
    let lngram = config.init(&device);
    let mut rng = Rng(3);
    let tokens = 64;
    let layout = DocumentLayout::packed(vec![vec![30, 34], vec![50]], tokens);
    let d = config.d_model;
    let mut u = rng.normals(2 * tokens * d, 0.0, 1.0);
    // Script 1 (second row): inputs drawn around one shared direction, so codes should concentrate.
    let shared = rng.normals(d, 0.0, 1.0);
    for p in tokens..2 * tokens {
        for c in 0..d {
            u[p * d + c] = shared[c] + 0.3 * u[p * d + c];
        }
    }
    let batch_layout = BatchLayout::new(layout.clone(), &device);
    let host_codes = lngram
        .route_codes(tensor(&u, [2, tokens, d], &device), &batch_layout)
        .to_host(&layout, config.alphabet(), &config.orders);
    let scripts: Vec<u32> = (0..2 * tokens).map(|p| (p / tokens) as u32).collect();
    for (script, stats) in host_codes.statistics_by_group(&scripts) {
        println!(
            "script {script}: positions={} route_entropy mean={:.3} min={:.3} bits, symbol_utilization={:.3}, \
             distinct_codes={} code_entropy={:.3} bits",
            stats.positions,
            stats.route_entropy_mean,
            stats.route_entropy_min,
            stats.symbol_utilization,
            stats.distinct_codes,
            stats.code_entropy
        );
        for order in &stats.orders {
            println!(
                "  order {}: lookups={} distinct_rows={} / {} key_utilization={:.3e} hit_entropy={:.3} bits",
                order.order, order.lookups, order.distinct_rows, order.table_rows, order.key_utilization, order.hit_entropy
            );
        }
        assert!(stats.route_entropy_mean <= 4.0 + 1e-12 && stats.code_entropy <= (stats.positions as f64).log2() + 1e-9);
    }
    let by_script = host_codes.statistics_by_group(&scripts);
    assert!(by_script[&1].route_entropy_mean < by_script[&0].route_entropy_mean, "concentrated script has lower entropy");
}

fn cpu_padding_lngram(device:&Device,surrogate:SurrogateGradient,orders:Vec<usize>)->Lngram {
    let mut module=LngramConfig::new().with_d_model(4).with_routes(2).with_bits_per_route(1).with_memory_dim(2)
        .with_orders(orders).with_surrogate(surrogate).with_smoothing_dilation(Some(1)).with_init_std(0.2)
        .with_table_initializer(Initializer::Normal {mean:0.0,std:0.3}).init(device);
    module.key.bias=Some(Param::from_tensor(tensor(&[0.1;4],[4],device)));
    module.value.bias=Some(Param::from_tensor(tensor(&[0.2;4],[4],device)));
    module.smoothing.as_mut().unwrap().weight=Param::from_tensor(tensor(&[0.125;12],[3,4],device));
    module
}

fn cpu_lngram_pass(module:&Lngram,layout:&BatchLayout,values:&[f64],device:&Device)
    ->(Vec<f64>,Vec<f64>,Vec<(u64,Vec<f64>)>) {
    use burn::module::{Module,ModuleVisitor};
    struct ParameterGradients<'a> {gradients:&'a Gradients,values:Vec<(u64,Vec<f64>)>}
    impl ModuleVisitor for ParameterGradients<'_> {
        fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<D>>) {
            self.values.push((param.id.val(),param_grad(param,self.gradients)));
        }
    }
    let input=tensor(values,[layout.host.batch(),layout.host.tokens(),4],device).require_grad();
    let output=module.forward(input.clone(),layout);
    let gradients=output.clone().square().sum().backward();
    let mut parameters=ParameterGradients {gradients:&gradients,values:Vec::new()};
    module.visit(&mut parameters);
    (host(&output),host(&input.grad(&gradients).expect("input gradient")),parameters.values)
}

#[test]
fn exp004_lngram_dirty_padding_preserves_all_parameter_gradients_cpu() {
    let device=Device::ndarray().autodiff();device.seed(2731);
    for surrogate in [SurrogateGradient::OneBit,SurrogateGradient::LocalExpectation] {
        let module=cpu_padding_lngram(&device,surrogate,vec![2,3]);
        for documents in [vec![vec![3,2],vec![2],vec![],vec![]],vec![vec![];4]] {
            let host_layout=DocumentLayout::packed(documents,6).with_document_padding(4);
            let valid=host_layout.document_metadata_host().source_valid;
            let layout=BatchLayout::new(host_layout,&device);
            let clean=(0..4*6*4).map(|index|if valid[index/4]==0.0 {0.0}
                else {(index%17) as f64*0.1-0.6}).collect::<Vec<_>>();
            let sentinels=[f64::NAN,f64::INFINITY,f64::NEG_INFINITY,1e30,-1e30];
            let dirty=clean.iter().enumerate().map(|(index,&value)|if valid[index/4]==0.0 {
                sentinels[index%sentinels.len()]
            } else {value}).collect::<Vec<_>>();
            let expected=cpu_lngram_pass(&module,&layout,&clean,&device);
            let actual=cpu_lngram_pass(&module,&layout,&dirty,&device);
            assert_eq!(actual,expected,"dirty inactive inputs changed the {surrogate:?} graph");
            assert!(actual.0.iter().chain(&actual.1).all(|value|value.is_finite()));
            for (_,gradient) in &actual.2 {assert!(gradient.iter().all(|value|value.is_finite()));}
            for (position,&active) in valid.iter().enumerate() {
                if active==0.0 {
                    assert!(actual.0[position*4..(position+1)*4].iter().all(|&value|value==0.0));
                    assert!(actual.1[position*4..(position+1)*4].iter().all(|&value|value==0.0));
                }
            }
            if valid.iter().all(|&active|active==0.0) {
                assert!(actual.2.iter().all(|(_,gradient)|gradient.iter().all(|&value|value==0.0)));
            } else {
                assert!(actual.2.iter().any(|(_,gradient)|gradient.iter().any(|&value|value!=0.0)));
            }
        }
    }
}

#[test]
fn exp004_lngram_incomplete_ngrams_select_poisoned_lookup_rows_before_surrogate_arithmetic_cpu() {
    let device=Device::ndarray().autodiff();device.seed(2732);
    let host_layout=DocumentLayout::packed(vec![vec![1,2],vec![2],vec![],vec![]],5).with_document_padding(4);
    let valid=host_layout.document_metadata_host().source_valid;
    let layout=BatchLayout::new(host_layout,&device);
    let clean=(0..4*5*4).map(|index|if valid[index/4]==0.0 {0.0}
        else {(index%13) as f64*0.1+0.1}).collect::<Vec<_>>();
    let dirty=clean.iter().enumerate().map(|(index,&value)|
        if valid[index/4]==0.0 {f64::NAN} else {value}).collect::<Vec<_>>();
    for surrogate in [SurrogateGradient::OneBit,SurrogateGradient::LocalExpectation] {
        let clean_module=cpu_padding_lngram(&device,surrogate,vec![3]);
        let mut dirty_module=clean_module.clone();
        for table in &mut dirty_module.tables {
            table.weight=Param::from_tensor(Tensor::<2>::full(table.weight.dims(),f32::NAN,&device));
        }
        let expected=cpu_lngram_pass(&clean_module,&layout,&clean,&device);
        let actual=cpu_lngram_pass(&dirty_module,&layout,&dirty,&device);
        assert_eq!(actual.0,expected.0,"projection biases at real short-document positions must be preserved");
        assert_eq!(actual.1,expected.1);
        assert!(actual.0.iter().any(|&value|value!=0.0));
        for ((_,actual),(_,expected)) in actual.2.iter().zip(&expected.2) {
            assert!(actual.iter().all(|value|value.is_finite()));
            assert_eq!(actual,expected,"inactive lookup values changed a real parameter gradient");
        }
        for table in &dirty_module.tables {
            let gradient=&actual.2.iter().find(|(id,_)|*id==table.weight.id.val()).unwrap().1;
            assert!(gradient.iter().all(|&value|value==0.0),"an incomplete n-gram must not train any table row");
        }
    }
}

#[test]
fn exp004_lngram_smoothing_dirty_padding_preserves_input_and_parameter_gradients_cpu() {
    let device=Device::ndarray().autodiff();device.seed(2733);
    let module=cpu_padding_lngram(&device,SurrogateGradient::OneBit,vec![2]);
    let smoothing=module.smoothing.as_ref().unwrap();
    for documents in [vec![vec![2,1],vec![1],vec![],vec![]],vec![vec![];4]] {
        let host_layout=DocumentLayout::packed(documents,6).with_document_padding(4);
        let valid=host_layout.document_metadata_host().source_valid;
        let layout=BatchLayout::new(host_layout,&device);
        let clean=(0..4*6*4).map(|index|if valid[index/4]==0.0 {0.0}
            else {(index%11) as f64*0.1-0.4}).collect::<Vec<_>>();
        let sentinels=[f64::NAN,f64::INFINITY,f64::NEG_INFINITY,1e30,-1e30];
        let dirty=clean.iter().enumerate().map(|(index,&value)|if valid[index/4]==0.0 {
            sentinels[index%sentinels.len()]
        } else {value}).collect::<Vec<_>>();
        let run=|values:&[f64]| {
            let input=tensor(values,[4,6,4],&device).require_grad();
            let output=smoothing.forward(input.clone(),&layout);
            let gradients=output.clone().square().sum().backward();
            (host(&output),host(&input.grad(&gradients).unwrap()),
                param_grad(&smoothing.norm.gamma,&gradients),param_grad(&smoothing.weight,&gradients))
        };
        let expected=run(&clean);let actual=run(&dirty);
        assert_eq!(actual,expected);
        for values in [&actual.0,&actual.1,&actual.2,&actual.3] {assert!(values.iter().all(|value|value.is_finite()));}
        for (position,&active) in valid.iter().enumerate() {
            if active==0.0 {
                assert!(actual.0[position*4..(position+1)*4].iter().all(|&value|value==0.0));
                assert!(actual.1[position*4..(position+1)*4].iter().all(|&value|value==0.0));
            }
        }
        if valid.iter().all(|&active|active==0.0) {
            assert!(actual.2.iter().chain(&actual.3).all(|&value|value==0.0));
        }
    }
}
