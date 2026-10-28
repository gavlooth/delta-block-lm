//! Independent scalar f64 output/all-gradient gate, adapted from the predecessor oracle.
use super::{Lngram, LngramBackend, LngramConfig, LngramLayout, SurrogateGradient, oracle};
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Gradients;
use oracle::{Batch, Evaluation, Group, Model, Surrogate};
use std::collections::{BTreeMap, HashSet};
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

    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }

    fn normals(&mut self, n: usize, mean: f64, std: f64) -> Vec<f64> {
        (0..n)
            .map(|_| (mean + std * self.normal()) as f32 as f64)
            .collect()
    }
}

fn tensor<const D: usize>(values: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(
        TensorData::new(values.iter().map(|&v| v as f32).collect::<Vec<_>>(), shape),
        device,
    )
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

#[derive(Clone)]
struct HostLayout {
    rows: Vec<Vec<usize>>,
    tokens: usize,
}
impl HostLayout {
    fn packed(rows: Vec<Vec<usize>>, tokens: usize) -> Self {
        Self { rows, tokens }
    }
    fn limits_host(&self) -> (Vec<i64>, Vec<i64>) {
        let mut starts = vec![1; self.rows.len() * self.tokens];
        let mut ends = vec![0; starts.len()];
        for (b, docs) in self.rows.iter().enumerate() {
            let mut start = 0;
            for &n in docs {
                for t in start..start + n {
                    starts[b * self.tokens + t] = start as i64;
                    ends[b * self.tokens + t] = (start + n - 1) as i64;
                }
                start += n;
            }
        }
        (starts, ends)
    }
    fn tensor(&self, device: &Device) -> LngramLayout {
        let (starts, ends) = self.limits_host();
        let valid = starts
            .iter()
            .zip(&ends)
            .map(|(s, e)| if e >= s { 1.0f32 } else { 0. })
            .collect();
        let reset = starts
            .iter()
            .zip(&ends)
            .enumerate()
            .map(|(p, (s, e))| {
                if e >= s && p % self.tokens == *s as usize {
                    1.0f32
                } else {
                    0.
                }
            })
            .collect();
        let shape = [self.rows.len(), self.tokens];
        LngramLayout::new(
            Tensor::from_data(TensorData::new(valid, shape), device),
            Tensor::from_data(TensorData::new(reset, shape), device),
        )
    }
}
fn limits(layout: &HostLayout) -> (Vec<i64>, Vec<i64>) {
    layout.limits_host()
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len().max(1) as f64).sqrt()
}

thread_local! { static METRICS: std::cell::RefCell<Vec<Check>> = const { std::cell::RefCell::new(Vec::new()) }; }
#[derive(Debug, serde::Serialize)]
pub struct Check {
    name: String,
    elements: usize,
    max_abs: f64,
    max_tolerance_ratio: f64,
}
fn compare(name: &str, actual: &[f64], reference: &[f64], _scale: Option<f64>) {
    assert_eq!(actual.len(), reference.len(), "{name}: length");
    let mut max_abs = 0.0f64;
    let mut ratio = 0.0f64;
    for (&x, &r) in actual.iter().zip(reference) {
        let error = (x - r).abs();
        let tolerance = 2e-5 + 3e-4 * r.abs();
        assert!(
            x.is_finite() && r.is_finite() && error <= tolerance,
            "{name}: {x} vs {r}, error {error}, tolerance {tolerance}"
        );
        max_abs = max_abs.max(error);
        ratio = ratio.max(error / tolerance);
    }
    METRICS.with(|m| {
        m.borrow_mut().push(Check {
            name: name.to_owned(),
            elements: actual.len(),
            max_abs,
            max_tolerance_ratio: ratio,
        })
    });
}

fn config_of(model: &Model, backend: LngramBackend) -> LngramConfig {
    LngramConfig::new()
        .with_backend(backend)
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
fn load(model: &Model, device: &Device, backend: LngramBackend) -> Lngram {
    let (d, kv) = (model.d_model, model.routes * model.memory_dim);
    let p1 = |v: &[f64]| Param::from_tensor(tensor(v, [v.len()], device));
    let p2 = |v: &[f64], rows: usize| Param::from_tensor(tensor(v, [rows, v.len() / rows], device));
    let mut lngram = config_of(model, backend).init(device);
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
        smoothing.weight = p2(model.smoothing_weight.as_ref().expect("smoothing on"), 4);
    }
    lngram
}

fn random_model(
    rng: &mut Rng,
    d: usize,
    routes: usize,
    bits: usize,
    memory_dim: usize,
    surrogate: Surrogate,
) -> Model {
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
        tables: orders
            .iter()
            .map(|&n| rng.normals(routes * alphabet.pow(n as u32) * memory_dim, 0.0, 1.0))
            .collect(),
        key_weight: rng.normals(kv * d, 0.0, 1.0 / (kv as f64).sqrt()),
        key_bias: rng.normals(d, 0.0, 0.5),
        value_weight: rng.normals(kv * d, 0.0, 1.0 / (kv as f64).sqrt()),
        value_bias: rng.normals(d, 0.0, 0.5),
        query_gamma: rng.normals(d, 1.0, 0.1),
        key_gamma: rng.normals(d, 1.0, 0.1),
        smoothing_gamma: Some(rng.normals(d, 1.0, 0.1)),
        smoothing_weight: Some(rng.normals(4 * d, 0.0, 0.5)),
        dilation: 3,
        orders,
        epsilon: 1e-6,
        temperature: 1.0,
        scale: 1.0,
        surrogate,
    }
}

/// Small non-default fixture: `d=24, R=6, m=4, d_m=3`, rows `[[4,1,6],[13]]` in `T=17`. The six-token document
/// repeats `A B A B A B`, so several table rows are hit more than once. The 13-token document exercises all four smoothing taps.
/// Seeds are skipped until every valid logit is at least `1e-3` from zero, so `f32` and `f64` agree on every bit.
fn small_fixture(surrogate: Surrogate) -> (Model, Batch, HostLayout, Evaluation) {
    let rows = vec![vec![4, 1, 6], vec![13]];
    let tokens = 17;
    let layout = HostLayout::packed(rows, tokens);
    let (starts, ends) = limits(&layout);
    for seed in 1.. {
        let mut rng = Rng(seed);
        let model = random_model(&mut rng, 24, 6, 4, 3, surrogate);
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
            .flat_map(|p| {
                evaluation.logits[p * width..(p + 1) * width]
                    .iter()
                    .map(|z| z.abs())
            })
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
                let hit: HashSet<i64> = evaluation.addresses[o]
                    .iter()
                    .copied()
                    .filter(|&a| a >= 0)
                    .collect();
                (0..values.len())
                    .filter(|i| hit.contains(&((i / model.memory_dim) as i64)))
                    .collect()
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
            assert!(
                error <= allowed,
                "{group:?}: ⟨∇f,v⟩={analytic:e} vs difference {difference:e}"
            );
        }
        println!("oracle finite differences {group:?}: worst error / allowed = {worst:.3e}");
    }
}

/// Replays tensor operators while retaining both normalization-output adjoints.
fn observed_zero_forward(
    layer: &Lngram,
    u: Tensor<3>,
    layout: &LngramLayout,
) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
    let [b, t, d] = u.dims();
    let padding = layout
        .valid
        .clone()
        .equal_elem(0.)
        .unsqueeze_dim::<3>(2)
        .expand([b, t, d]);
    let u = u.mask_fill(padding.clone(), 0.);
    let q_probe = Tensor::<3>::zeros([b, t, d], &u.device()).require_grad();
    let normalized_q = layer.discretizer_norm.forward(u.clone()) + q_probe.clone();
    let logits = layer.discretizer.forward(normalized_q.clone());
    let bits = layer.bits(logits.clone());
    let symbols = layer.symbols(bits.clone());
    let probabilities = burn::tensor::activation::sigmoid(
        logits
            .clone()
            .reshape([b, t, layer.routes, layer.bits_per_route])
            .mul_scalar(layer.surrogate_temperature),
    );
    let mut memories = Vec::new();
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    let cached_codes = (layer.backend == LngramBackend::FusedGpu)
        .then(|| delta_kernels::lngram_codes(logits.clone(), layer.bits_per_route));
    for (table, &order) in layer.tables.iter().zip(&layer.orders) {
        if layer.backend == LngramBackend::FusedGpu {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                memories.push(
                    delta_kernels::lngram_memory(
                        logits.clone(),
                        cached_codes.as_ref().unwrap().clone(),
                        table.weight.val(),
                        layout.valid.clone(),
                        layout.documents.clone(),
                        layer.bits_per_route,
                        order,
                        layer.surrogate == SurrogateGradient::LocalExpectation,
                        layer.surrogate_temperature,
                        layer.surrogate_scale,
                    )
                    .reshape([b, t, layer.routes, layer.memory_dim]),
                );
                continue;
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            panic!("fused Lngram reference requires GPU");
        }
        let rows = layer.alphabet().pow(order as u32);
        let local = layer.local_addresses(&symbols, order);
        let table = table.weight.val();
        let missing = super::full_ngram(layout, order)
            .bool_not()
            .reshape([b, t, 1, 1]);
        let memory = super::lookup(
            layer.backend,
            table.clone(),
            local.clone().reshape([b * t, layer.routes]),
            rows,
        )
        .reshape([b, t, layer.routes, layer.memory_dim])
        .mask_fill(
            missing
                .clone()
                .expand([b, t, layer.routes, layer.memory_dim]),
            0.,
        );
        let surrogate = match layer.surrogate {
            SurrogateGradient::OneBit => layer.one_bit_term(
                table.detach(),
                &local,
                &bits,
                &probabilities,
                memory.clone().detach(),
                order,
                &missing,
            ),
            SurrogateGradient::LocalExpectation => layer.local_expectation_term(
                table.detach(),
                &local,
                &symbols,
                &probabilities,
                order,
                &missing,
            ),
        };
        memories.push(
            memory + (surrogate.clone() - surrogate.detach()).mul_scalar(layer.surrogate_scale),
        );
    }
    let memory: Tensor<4> = Tensor::stack::<5>(memories, 2).reshape([
        b,
        t,
        layer.orders.len(),
        layer.routes * layer.memory_dim,
    ]);
    let key = layer.key_norm.forward(layer.key.forward(memory.clone()));
    let value = layer.value.forward(memory);
    let query_probe = Tensor::<3>::zeros([b, t, d], &u.device()).require_grad();
    let normalized_query = layer.query_norm.forward(u) + query_probe.clone();
    let gates = burn::tensor::activation::sigmoid(
        (normalized_query.clone().unsqueeze_dim::<4>(2) * key)
            .sum_dim(3)
            .div_scalar((d as f64).sqrt()),
    );
    let readout = (gates * value).sum_dim(2).reshape([b, t, d]);
    let output = match &layer.smoothing {
        Some(smoothing) => readout.clone() + smoothing.forward(readout, layout),
        None => readout,
    };
    (output.mask_fill(padding, 0.), q_probe, query_probe)
}

fn run_oracle_comparison(
    model: Model,
    batch: Batch,
    layout: HostLayout,
    device: Device,
    backend: LngramBackend,
    strided: bool,
) {
    let evaluation = oracle::evaluate(&model, &batch, None);

    let lngram = load(&model, &device, backend);
    let (b, t, d) = (batch.batch, batch.tokens, model.d_model);
    let batch_layout = layout.tensor(&device);
    let u = if strided {
        let mut reordered = vec![0.; batch.u.len()];
        for bi in 0..b {
            for ti in 0..t {
                for c in 0..d {
                    reordered[(bi * d + c) * t + ti] = batch.u[(bi * t + ti) * d + c];
                }
            }
        }
        tensor(&reordered, [b, d, t], &device).swap_dims(1, 2)
    } else {
        tensor(&batch.u, [b, t, d], &device)
    }
    .require_grad();
    let zero_input = batch.u.iter().all(|&v| v == 0.);
    let (output, observed) = if zero_input {
        let (output, q, query) = observed_zero_forward(&lngram, u.clone(), &batch_layout);
        compare(
            "observed zero replay output",
            &host(&output),
            &host(&lngram.forward(u.clone(), &batch_layout)),
            None,
        );
        (output, Some((q, query)))
    } else {
        (lngram.forward(u.clone(), &batch_layout), None)
    };
    let grads = (output.clone() * tensor(&batch.loss_weight, [b, t, d], &device))
        .sum()
        .backward();

    // Codes and addresses: exact.
    let codes = lngram.route_codes(u.clone(), &batch_layout);
    let valid: Vec<bool> = (0..b * t)
        .map(|p| batch.ends[p] >= batch.starts[p])
        .collect();
    let gpu_symbols = host_int(&codes.symbols);
    for p in (0..b * t).filter(|&p| valid[p]) {
        let range = p * model.routes..(p + 1) * model.routes;
        assert_eq!(
            gpu_symbols[range.clone()],
            evaluation.symbols[range],
            "symbols at position {p}"
        );
    }
    for (o, addresses) in codes.addresses.iter().enumerate() {
        assert_eq!(
            host_int(addresses),
            evaluation.addresses[o],
            "addresses of order {}",
            model.orders[o]
        );
    }

    let out = host(&output);
    compare("output", &out, &evaluation.output, None);
    for p in (0..b * t).filter(|&p| !valid[p]) {
        assert!(
            out[p * d..(p + 1) * d].iter().all(|&v| v == 0.0),
            "padded output {p} is not exactly 0"
        );
    }

    let total = &evaluation.total;
    let u_grad = host(&u.grad(&grads).expect("input gradient"));
    if let Some((q, query)) = observed {
        let q_upstream = host(&q.grad(&grads).expect("discretizer norm adjoint"));
        let query_upstream = host(&query.grad(&grads).expect("query norm adjoint"));
        let rms = (model.epsilon as f32).sqrt() as f64;
        let mut local = vec![0.; u_grad.len()];
        for i in 0..local.len() {
            local[i] = (q_upstream[i] * model.discretizer_gamma[i % d]
                + query_upstream[i] * model.query_gamma[i % d])
                / rms;
        }
        let raw = u_grad
            .iter()
            .zip(&total.u)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let observed = u_grad
            .iter()
            .zip(&local)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!(
            "zero input VJP localization: ideal-f64 max_abs={raw:e}; observed-f32 normalization-upstream f64 max_abs={observed:e}; rms={rms:e}"
        );
        compare(
            "zero input grad: f64 local VJP of observed f32 norm upstream",
            &u_grad,
            &local,
            None,
        );
    } else {
        compare("grad u", &u_grad, &total.u, None);
    }
    for p in (0..b * t).filter(|&p| !valid[p]) {
        assert!(
            u_grad[p * d..(p + 1) * d].iter().all(|&v| v == 0.0),
            "padded input gradient {p} is not 0"
        );
    }
    // Discretizer: the surrogate against the independent f64 surrogate.
    compare(
        "grad discretizer W_q (surrogate)",
        &param_grad(&lngram.discretizer.weight, &grads),
        &total.discretizer,
        None,
    );
    compare(
        "grad discretizer RMSNorm",
        &param_grad(&lngram.discretizer_norm.gamma, &grads),
        &total.discretizer_gamma,
        None,
    );
    let surrogate_norm = total.discretizer.iter().map(|g| g * g).sum::<f64>().sqrt();
    assert!(surrogate_norm.is_finite());
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
                assert!(
                    values.iter().all(|&v| v == 0.0),
                    "order {} row {row} not indexed but has gradient",
                    model.orders[o]
                );
            }
        }
        let repeated = hits.values().filter(|&&count| count > 1).count();

        println!(
            "order {}: {} rows hit, {repeated} hit more than once",
            model.orders[o],
            hits.len()
        );
        // Compare the hit rows (all other rows are exactly zero above).
        let (gpu_rows, ref_rows): (Vec<f64>, Vec<f64>) = hits
            .keys()
            .flat_map(|&row| (0..dm).map(move |c| row as usize * dm + c))
            .map(|i| (gradient[i], total.tables[o][i]))
            .unzip();
        compare(
            &format!("grad table order {}", model.orders[o]),
            &gpu_rows,
            &ref_rows,
            None,
        );
    }
    compare(
        "grad W_K",
        &param_grad(&lngram.key.weight, &grads),
        &total.key_weight,
        None,
    );
    compare(
        "grad b_K",
        &param_grad(lngram.key.bias.as_ref().unwrap(), &grads),
        &total.key_bias,
        None,
    );
    compare(
        "grad W_V",
        &param_grad(&lngram.value.weight, &grads),
        &total.value_weight,
        None,
    );
    compare(
        "grad b_V",
        &param_grad(lngram.value.bias.as_ref().unwrap(), &grads),
        &total.value_bias,
        None,
    );
    compare(
        "grad query RMSNorm",
        &param_grad(&lngram.query_norm.gamma, &grads),
        &total.query_gamma,
        None,
    );
    compare(
        "grad key RMSNorm",
        &param_grad(&lngram.key_norm.gamma, &grads),
        &total.key_gamma,
        None,
    );
    let smoothing = lngram.smoothing.as_ref().expect("smoothing on");
    compare(
        "grad smoothing RMSNorm",
        &param_grad(&smoothing.norm.gamma, &grads),
        &total.smoothing_gamma,
        None,
    );
    compare(
        "grad smoothing taps",
        &param_grad(&smoothing.weight, &grads),
        &total.smoothing_weight,
        None,
    );
}

#[derive(Debug, serde::Serialize)]
pub struct Report {
    pub passed: bool,
    pub cases: usize,
    pub checks: Vec<Check>,
    pub duplicate_index_table_gradients: Vec<LookupGradientCheck>,
}

fn causality(device: &Device, backend: LngramBackend) {
    let (model, batch, layout, _) = small_fixture(Surrogate::OneBit);
    let layer = load(&model, device, backend).valid();
    let layout = layout.tensor(device);
    let shape = [batch.batch, batch.tokens, model.d_model];
    let base = host(&layer.forward(tensor(&batch.u, shape, device), &layout));
    // Changes in the repeated document, and a complete later document, cannot affect any earlier position or other document.
    for (begin, end) in [(7, 8), (5, 11), (11, 18)] {
        let mut changed = batch.u.clone();
        for v in &mut changed[begin * model.d_model..end * model.d_model] {
            *v += 3.25;
        }
        let actual = host(&layer.forward(tensor(&changed, shape, device), &layout));
        for p in 0..batch.batch * batch.tokens {
            let same_doc =
                batch.starts[p] == batch.starts[begin] && p / batch.tokens == begin / batch.tokens;
            if p < begin || !same_doc {
                compare(
                    "causality",
                    &actual[p * model.d_model..(p + 1) * model.d_model],
                    &base[p * model.d_model..(p + 1) * model.d_model],
                    None,
                );
            }
        }
    }
}

/// Consumer-facing Tensor lookup gate: native forward, independent f64 VJP,
/// and exact raw table-gradient bits before any optimizer or clipping.
#[derive(Debug, serde::Serialize)]
pub struct LookupGradientCheck {
    layout: &'static str,
    table_shape: [usize; 3],
    index_shape: [usize; 2],
    valid_lookups: usize,
    masked_lookups: usize,
    duplicate_hits: usize,
    repeated_rows: usize,
    max_hits_per_row: usize,
    unused_rows: usize,
    backward_repeats: usize,
    native_forward_bit_identity: bool,
    repeat_raw_gradient_bit_identity: bool,
    raw_gradient_max_abs_error: f64,
    raw_gradient_max_tolerance_ratio: f64,
    raw_gradient_hash_algorithm: &'static str,
    raw_gradient_hashes: Vec<String>,
    accumulation_contract: &'static str,
}

fn raw_f32_bits<const D: usize>(value: &Tensor<D>) -> Vec<u32> {
    value.to_data().iter::<f32>().map(f32::to_bits).collect()
}

fn raw_gradient_hash(bits: &[u32]) -> String {
    // FNV-1a over explicitly little-endian f32 bits, independent of host endianness.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bits.iter().flat_map(|value| value.to_le_bytes()) {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn duplicate_index_table_gradients(device: &Device) -> Vec<LookupGradientCheck> {
    const BATCH: usize = 3;
    const TOKENS: usize = 257;
    const POSITIONS: usize = BATCH * TOKENS;
    const ROUTES: usize = 3;
    const ROWS: usize = 11;
    const MEMORY: usize = 5;
    const REPEATS: usize = 32;
    const ORDER: usize = 2;
    let device = device.clone().autodiff();
    let host_layout =
        HostLayout::packed(vec![vec![130, 73], vec![65, 64, 31], vec![]], TOKENS);
    let layout = host_layout.tensor(&device);
    let (starts, ends) = host_layout.limits_host();
    // Derive validity independently of the device's production full_ngram mask.
    let valid: Vec<bool> = (0..POSITIONS)
        .map(|p| ends[p] >= starts[p] && p % TOKENS >= starts[p] as usize + ORDER - 1)
        .collect();
    let mut values = vec![0.; ROUTES * ROWS * MEMORY];
    for (i, value) in values.iter_mut().enumerate() {
        *value = (((i * 17 + 5) % 41) as f32 * 0.03125 - 0.625) as f64;
    }
    for route in 0..ROUTES {
        // Padding reads a legal integer row containing dirty values. The existing
        // validity mask must remove these NaNs before consumer arithmetic.
        values[(route * ROWS + ROWS - 1) * MEMORY..(route + 1) * ROWS * MEMORY]
            .fill(f64::NAN);
    }
    let signed = [1.0f32, -1.0, 0.01, -0.02, 0.03, -0.01, 0.25, -0.249];
    let mut indices = vec![0i64; POSITIONS * ROUTES];
    let mut weights = vec![0.; POSITIONS * ROUTES * MEMORY];
    let mut reference_output = vec![0.; weights.len()];
    let mut reference_gradient = vec![0.; values.len()];
    let mut hits = vec![0usize; ROUTES * ROWS];
    for p in 0..POSITIONS {
        for route in 0..ROUTES {
            let row = if valid[p] {
                (p / 3 + route * 2 + usize::from(p % 7 == 0)) % 5
            } else {
                ROWS - 1
            };
            indices[p * ROUTES + route] = row as i64;
            if valid[p] {
                hits[route * ROWS + row] += 1;
            }
            for column in 0..MEMORY {
                let out = (p * ROUTES + route) * MEMORY + column;
                let table = (route * ROWS + row) * MEMORY + column;
                let weight = signed[(p * 3 + route * 5 + column) % signed.len()]
                    * (1.0 + column as f32 * 0.125 + route as f32 * 0.0625);
                weights[out] = weight as f64;
                if valid[p] {
                    reference_output[out] = values[table];
                    // Independent scalar f64 sum in ascending flattened (batch,time)
                    // order. Every contribution is the exact uploaded f32 weight.
                    reference_gradient[table] += weight as f64;
                }
            }
        }
    }
    let valid_lookups = hits.iter().sum::<usize>();
    let duplicate_hits = hits.iter().map(|&n| n.saturating_sub(1)).sum::<usize>();
    let repeated_rows = hits.iter().filter(|&&n| n > 1).count();
    let unused_rows = hits.iter().filter(|&&n| n == 0).count();
    assert!(duplicate_hits > valid_lookups / 2 && repeated_rows > 0 && unused_rows > 0);
    let padding = super::full_ngram(&layout, ORDER)
        .bool_not()
        .reshape([POSITIONS, 1, 1])
        .expand([POSITIONS, ROUTES, MEMORY]);
    let upstream = tensor(&weights, [POSITIONS, ROUTES, MEMORY], &device);
    let mut reports = Vec::new();
    for strided in [false, true] {
        let table = if strided {
            let mut reordered = vec![0.; values.len()];
            for route in 0..ROUTES {
                for row in 0..ROWS {
                    for column in 0..MEMORY {
                        reordered[(route * MEMORY + column) * ROWS + row] =
                            values[(route * ROWS + row) * MEMORY + column];
                    }
                }
            }
            tensor(&reordered, [ROUTES, MEMORY, ROWS], &device).swap_dims(1, 2)
        } else {
            tensor(&values, [ROUTES, ROWS, MEMORY], &device)
        }
        .require_grad();
        let mut route_indices = vec![0i64; indices.len()];
        for p in 0..POSITIONS {
            for route in 0..ROUTES {
                route_indices[route * POSITIONS + p] = indices[p * ROUTES + route];
            }
        }
        let index = if strided {
            Tensor::<2, Int>::from_data(
                TensorData::new(indices.clone(), [POSITIONS, ROUTES]),
                &device,
            )
            .swap_dims(0, 1)
        } else {
            Tensor::<2, Int>::from_data(
                TensorData::new(route_indices.clone(), [ROUTES, POSITIONS]),
                &device,
            )
        };
        assert_eq!(host_int(&index), route_indices, "lookup integer indices changed");
        let native = table
            .clone()
            .detach()
            .gather(
                1,
                index
                    .clone()
                    .unsqueeze_dim::<3>(2)
                    .expand([ROUTES, POSITIONS, MEMORY]),
            )
            .swap_dims(0, 1);
        let native_bits = raw_f32_bits(&native);
        let layout_name = if strided { "noncontiguous" } else { "contiguous" };
        let mut first_bits = None;
        let mut hashes = Vec::with_capacity(REPEATS);
        let mut max_abs = 0.0f64;
        let mut max_ratio = 0.0f64;
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        let gpu = matches!(device.clone().inner().as_dispatch(), burn::backend::DispatchDevice::Cube(_));
        let lookup = |table: Tensor<3>, index: Tensor<2, Int>| {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            if gpu {
                return delta_kernels::lngram_table_lookup(table, index).swap_dims(0, 1);
            }
            super::lookup(
                LngramBackend::Tensor,
                table.reshape([ROUTES * ROWS, MEMORY]),
                index.swap_dims(0, 1),
                ROWS,
            )
        };
        for repeat in 0..REPEATS {
            let output = lookup(table.clone(), index.clone());
            assert_eq!(
                raw_f32_bits(&output),
                native_bits,
                "{layout_name}: lookup forward must preserve native gather bits"
            );
            let output = output.mask_fill(padding.clone(), 0.);
            let gradients = (output.clone() * upstream.clone()).sum().backward();
            let gradient = table.grad(&gradients).expect("raw lookup table gradient");
            let bits = raw_f32_bits(&gradient);
            hashes.push(raw_gradient_hash(&bits));
            if let Some(expected) = &first_bits {
                assert_eq!(
                    &bits, expected,
                    "{layout_name}: repeat {repeat} raw table gradient changed bits"
                );
            } else {
                compare(
                    &format!("duplicate-index {layout_name} lookup output"),
                    &host(&output),
                    &reference_output,
                    None,
                );
                let actual = host(&gradient);
                compare(
                    &format!("duplicate-index {layout_name} raw table gradient"),
                    &actual,
                    &reference_gradient,
                    None,
                );
                for (&actual, &reference) in actual.iter().zip(&reference_gradient) {
                    let error = (actual - reference).abs();
                    max_abs = max_abs.max(error);
                    max_ratio = max_ratio.max(error / (2e-5 + 3e-4 * reference.abs()));
                }
                for (row, &count) in hits.iter().enumerate() {
                    if count == 0 {
                        assert!(
                            actual[row * MEMORY..(row + 1) * MEMORY]
                                .iter()
                                .all(|&value| value == 0.),
                            "{layout_name}: unused or padding-only row {row} has gradient"
                        );
                    }
                }
                first_bits = Some(bits);
            }
        }
        println!(
            "Tensor lookup {layout_name}: {duplicate_hits} duplicate hits, \
             {repeated_rows} repeated rows, {unused_rows} unused rows; \
             raw-gradient max abs {max_abs:.6e}, tolerance ratio {max_ratio:.6e}; \
             {REPEATS} backward repeats bit-identical, hashes {hashes:?}"
        );
        reports.push(LookupGradientCheck {
            layout: layout_name,
            table_shape: [ROUTES, ROWS, MEMORY],
            index_shape: [ROUTES, POSITIONS],
            valid_lookups,
            masked_lookups: POSITIONS * ROUTES - valid_lookups,
            duplicate_hits,
            repeated_rows,
            max_hits_per_row: *hits.iter().max().unwrap(),
            unused_rows,
            backward_repeats: REPEATS,
            native_forward_bit_identity: true,
            repeat_raw_gradient_bit_identity: true,
            raw_gradient_max_abs_error: max_abs,
            raw_gradient_max_tolerance_ratio: max_ratio,
            raw_gradient_hash_algorithm: "fnv1a64-little-endian-f32-bits",
            raw_gradient_hashes: hashes,
            accumulation_contract: "GPU: one owner per destination (route,row,column), \
                scalar f32 sum from +0 in ascending flattened (batch,time) position order; \
                integer indices, no float atomics. CPU: existing native gather VJP. \
                Native gather forward bits; full_ngram validity mask before arithmetic; \
                independent scalar f64 VJP tolerance 2e-5 + 3e-4*abs(reference); \
                32 raw-gradient f32-bit-identical backward repeats before clipping/optimizer.",
        });
    }
    reports
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn deterministic_table_vjp(device: &Device) {
    let device = device.clone().autodiff();
    let (model, batch, layout, _) = small_fixture(Surrogate::OneBit);
    let layer = load(&model, &device, LngramBackend::FusedGpu);
    let logits = layer
        .discretizer_logits(tensor(
            &batch.u,
            [batch.batch, batch.tokens, model.d_model],
            &device,
        ))
        .detach()
        .require_grad();
    let layout = layout.tensor(&device);
    let codes = delta_kernels::lngram_codes(logits.clone(), model.bits);
    let mut rng = Rng(0xD371);
    for (table, &order) in layer.tables.iter().zip(&layer.orders) {
        let upstream = tensor(
            &rng.normals(
                batch.batch * batch.tokens * model.routes * model.memory_dim,
                0.,
                1.,
            ),
            [batch.batch, batch.tokens, model.routes * model.memory_dim],
            &device,
        );
        let mut runs = Vec::new();
        for _ in 0..2 {
            let output = delta_kernels::lngram_memory(
                logits.clone(),
                codes.clone(),
                table.weight.val(),
                layout.valid.clone(),
                layout.documents.clone(),
                model.bits,
                order,
                false,
                model.temperature,
                model.scale,
            );
            let gradients = (output * upstream.clone()).sum().backward();
            runs.push(param_grad(&table.weight, &gradients));
        }
        assert_eq!(
            runs[0], runs[1],
            "fused table VJP must be bitwise repeatable"
        );
        compare(
            "bitwise deterministic fused table VJP",
            &runs[0],
            &runs[1],
            None,
        );
    }
}

pub fn numerical_checks(device: &Device) -> Result<Report, String> {
    METRICS.with(|m| m.borrow_mut().clear());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut cases = 0;
        let duplicate_index_table_gradients = duplicate_index_table_gradients(device);
        cases += duplicate_index_table_gradients.len();
        for mode in [Surrogate::OneBit, Surrogate::LocalExpectation] {
            let (model, batch, layout, evaluation) = small_fixture(mode);
            check_oracle_gradients(&model, &batch, &evaluation);
            let mut backends = vec![LngramBackend::Tensor, LngramBackend::FlatSelect];
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            if device.identity().is_some() {
                backends.push(LngramBackend::FusedGpu);
            }
            for backend in backends {
                run_oracle_comparison(
                    model.clone(),
                    batch.clone(),
                    layout.clone(),
                    device.clone().autodiff(),
                    backend,
                    true,
                );
                cases += 1;
                let mut zero = batch.clone();
                zero.u.fill(0.);
                run_oracle_comparison(
                    model.clone(),
                    zero,
                    layout.clone(),
                    device.clone().autodiff(),
                    backend,
                    false,
                );
                cases += 1;
                let short = HostLayout::packed(vec![vec![1], vec![], vec![2, 1]], 5);
                let (starts, ends) = short.limits_host();
                let mut rng = Rng(938);
                let edge = Batch {
                    batch: 3,
                    tokens: 5,
                    starts,
                    ends,
                    u: rng.normals(15 * model.d_model, 0., 1.),
                    loss_weight: rng.normals(15 * model.d_model, 0., 1.),
                };
                // Poison every row not actually read; masking must precede surrogate arithmetic.
                let used = oracle::evaluate(&model, &edge, None);
                let mut poisoned = model.clone();
                for (o, table) in poisoned.tables.iter_mut().enumerate() {
                    let hit: HashSet<_> = used.addresses[o]
                        .iter()
                        .copied()
                        .filter(|&a| a >= 0)
                        .collect();
                    for (row, columns) in table.chunks_mut(model.memory_dim).enumerate() {
                        if !hit.contains(&(row as i64)) {
                            columns.fill(f64::NAN);
                        }
                    }
                }
                // Alternative full-bigram rows must remain finite; poison only the all-incomplete trigram table.
                poisoned.tables[0] = model.tables[0].clone();
                let mut dirty = edge.clone();
                for p in 0..15 {
                    if dirty.ends[p] < dirty.starts[p] {
                        dirty.u[p * model.d_model..(p + 1) * model.d_model].fill(f64::NAN);
                    }
                }
                run_oracle_comparison(
                    poisoned,
                    dirty,
                    short,
                    device.clone().autodiff(),
                    backend,
                    false,
                );
                cases += 1;
            }
        }
        let mut causal_backends = vec![LngramBackend::Tensor, LngramBackend::FlatSelect];
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        if device.identity().is_some() {
            causal_backends.push(LngramBackend::FusedGpu);
            deterministic_table_vjp(device);
            cases += 1;
        }
        for backend in causal_backends {
            causality(device, backend);
            cases += 1;
        }
        (cases, duplicate_index_table_gradients)
    }));
    match result {
        Ok((cases, duplicate_index_table_gradients)) => Ok(Report {
            passed: true,
            cases,
            checks: METRICS.with(|m| m.take()),
            duplicate_index_table_gradients,
        }),
        Err(e) => Err(e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "Lngram reference assertion failed".to_string())),
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    #[test]
    fn causal_lngram_f64_outputs_and_all_gradients() {
        super::numerical_checks(&burn::prelude::Device::ndarray()).unwrap();
    }
}
