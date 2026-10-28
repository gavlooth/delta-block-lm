use crate::reference::{HEADS, KEY_DIM, Token, VALUE_DIM, backward, scan};

const GROUPS: [&str; 6] = ["q", "k", "v", "alpha", "erase", "write"];
const WIDTHS: [usize; 6] = [KEY_DIM, KEY_DIM, VALUE_DIM, KEY_DIM, KEY_DIM, VALUE_DIM];

/// SplitMix64: deterministic fixtures and directions without an external generator.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }

    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

fn draw(rng: &mut SplitMix, width: usize, center: f64, spread: f64) -> Vec<f64> {
    (0..width).map(|_| center + spread * rng.signed()).collect()
}

/// One token of the shared fixture. Keys and values are large enough that the state carries real content into
/// later tokens, so the erase and decay gradients are a material share of the gradient (checked by
/// `assert_erase_decay_nondegenerate`); `Σ erase·k²` averages about 1.1, below 2, the stability limit of the delta
/// update.
fn fixture_token(rng: &mut SplitMix) -> Token {
    let q = draw(rng, KEY_DIM, 0.0, 0.5);
    let k = draw(rng, KEY_DIM, 0.0, 0.45);
    let v = draw(rng, VALUE_DIM, 0.0, 1.0);
    let alpha = draw(rng, KEY_DIM, 0.875, 0.075);
    let erase = draw(rng, KEY_DIM, 0.5, 0.2);
    let write = draw(rng, VALUE_DIM, 0.75, 0.25);
    Token::new(q, k, v, alpha, erase, write)
}

fn group(token: &Token, index: usize) -> &[f64] {
    match index {
        0 => &token.q,
        1 => &token.k,
        2 => &token.v,
        3 => &token.alpha,
        4 => &token.erase,
        5 => &token.write,
        _ => unreachable!("GDN-2 has six input groups"),
    }
}

fn group_mut(token: &mut Token, index: usize) -> &mut [f64] {
    match index {
        0 => &mut token.q,
        1 => &mut token.k,
        2 => &mut token.v,
        3 => &mut token.alpha,
        4 => &mut token.erase,
        5 => &mut token.write,
        _ => unreachable!("GDN-2 has six input groups"),
    }
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64).sqrt()
}

/// Host `(doc_start, doc_end)` for one row built from `(is_document, length)` pieces; padding has `end < start`.
fn row_limits(pieces: &[(bool, usize)]) -> (Vec<i32>, Vec<i32>) {
    let (mut starts, mut ends) = (Vec::new(), Vec::new());
    let mut position = 0usize;
    for &(document, length) in pieces {
        for t in position..position + length {
            if document {
                starts.push(position as i32);
                ends.push((position + length - 1) as i32);
            } else {
                starts.push(t as i32);
                ends.push(t as i32 - 1);
            }
        }
        position += length;
    }
    (starts, ends)
}

/// Outcome of one PLAN §8.1 elementwise comparison, with the required error report.
struct Comparison {
    pass: bool,
    report: String,
}

/// PLAN §8.1: `|x − ref| ≤ atol + rtol·|ref|` with `rtol = 1e-4` and `atol = 1e-5·max(rms(ref), s₀)`, where the
/// declared scale `s₀` is the reference rms on this non-degenerate fixture. Elements flagged in `exact_zero`
/// (padding positions) must be exactly 0.
fn compare(name: &str, actual: &[f32], reference: &[f64], exact_zero: &[bool]) -> Comparison {
    compare_precision(name, actual, reference, exact_zero, std::env::var_os("DRE_BF16_GATE").is_some())
}

fn compare_precision(name: &str, actual: &[f32], reference: &[f64], exact_zero: &[bool], bf16: bool) -> Comparison {
    assert_eq!(actual.len(), reference.len(), "{name}: length");
    assert_eq!(exact_zero.len(), reference.len(), "{name}: mask length");
    let scale = rms(reference);
    assert!(scale > 0.0, "{name}: degenerate fixture, the reference is all zero");
    let rtol = if bf16 { 1e-2 } else { 1e-4 };
    let atol = if bf16 { 1e-2 } else { 1e-5 } * scale;
    let mut max_abs = 0.0f64;
    let mut max_rel: Option<f64> = None;
    let mut ratios = Vec::with_capacity(reference.len());
    let mut outside = 0usize;
    let mut worst: Option<(usize, f64)> = None;
    let mut nonzero_masked = 0usize;
    for (index, ((&x, &r), &zero)) in actual.iter().zip(reference).zip(exact_zero).enumerate() {
        let error = (f64::from(x) - r).abs();
        let ratio = error / (atol + rtol * r.abs());
        max_abs = max_abs.max(error);
        if r.abs() > atol {
            max_rel = Some(max_rel.unwrap_or(0.0).max(error / r.abs()));
        }
        ratios.push(ratio);
        if zero {
            assert_eq!(r, 0.0, "{name}[{index}]: the oracle is nonzero at a masked element");
            if x != 0.0 {
                nonzero_masked += 1;
            }
        } else if !(ratio <= 1.0) {
            outside += 1;
            if worst.map_or(true, |(_, largest)| ratio > largest) {
                worst = Some((index, ratio));
            }
        }
    }
    ratios.sort_by(f64::total_cmp);
    let percentile = |p: f64| ratios[((ratios.len() - 1) as f64 * p).round() as usize];
    let relative = max_rel.map_or_else(|| "none: all |ref| ≤ atol".to_string(), |value| format!("{value:.3e}"));
    let mut report = format!(
        "{name}: max abs err {max_abs:.3e}, max rel err {relative}, err/tol p50 {:.3e} p99 {:.3e} (atol {atol:.3e}, rtol {rtol:.0e})",
        percentile(0.5),
        percentile(0.99),
    );
    if let Some((index, ratio)) = worst {
        report += &format!(
            "; {outside} elements outside tolerance, worst [{index}] got {} expected {} ({ratio:.2}x tolerance)",
            actual[index], reference[index],
        );
    }
    if nonzero_masked > 0 {
        report += &format!("; {nonzero_masked} masked elements not exactly 0");
    }
    Comparison { pass: outside == 0 && nonzero_masked == 0, report }
}

fn assert_all_pass(comparisons: &[Comparison]) {
    for comparison in comparisons {
        println!("{}", comparison.report);
    }
    let failed: Vec<&str> = comparisons.iter().filter(|c| !c.pass).map(|c| c.report.as_str()).collect();
    assert!(failed.is_empty(), "PLAN §8.1 comparison failed:\n{}", failed.join("\n"));
}

fn objective(tokens: &[Token], starts: &[i32], ends: &[i32], grad: &[Vec<f64>]) -> f64 {
    let output = scan(tokens, starts, ends).output;
    output.iter().zip(grad).flat_map(|(o, g)| o.iter().zip(g)).map(|(o, g)| o * g).sum()
}

/// PLAN §8.1 differentiable-path check of the oracle's analytic gradient. For every input group, central
/// differences along 8 random unit directions over all of that group's entries must match `⟨∇f, v⟩` within
/// `1e-7·max(‖∇f‖₂, g₀) + 1e-6·|d|`, where `g₀` is the group's gradient norm on this fixture (asserted positive).
fn check_oracle_directions(tokens: &[Token], starts: &[i32], ends: &[i32], grad: &[Vec<f64>], rng: &mut SplitMix) -> Vec<Token> {
    let analytic = backward(tokens, starts, ends, grad);
    for (index, name) in GROUPS.iter().enumerate() {
        let point: Vec<f64> = tokens.iter().flat_map(|token| group(token, index).iter().copied()).collect();
        let gradient: Vec<f64> = analytic.iter().flat_map(|token| group(token, index).iter().copied()).collect();
        let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(norm > 0.0, "oracle d{name}: degenerate fixture, the analytic gradient is zero");
        let step = 1e-6 * rms(&point).max(1.0);
        for direction_index in 0..8 {
            let mut direction: Vec<f64> = (0..point.len()).map(|_| rng.normal()).collect();
            let length = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
            direction.iter_mut().for_each(|x| *x /= length);
            let shifted = |sign: f64| {
                let mut moved = tokens.to_vec();
                let mut offset = 0;
                for token in &mut moved {
                    for value in group_mut(token, index) {
                        *value += sign * step * direction[offset];
                        offset += 1;
                    }
                }
                objective(&moved, starts, ends, grad)
            };
            let numerical = (shifted(1.0) - shifted(-1.0)) / (2.0 * step);
            let projected: f64 = gradient.iter().zip(&direction).map(|(g, v)| g * v).sum();
            let error = (projected - numerical).abs();
            let tolerance = 1e-7 * norm + 1e-6 * numerical.abs();
            println!(
                "oracle d{name} direction {direction_index}: <grad, v> {projected:.9e}, central difference {numerical:.9e}, error {error:.3e} (tolerance {tolerance:.3e})"
            );
            assert!(error <= tolerance, "oracle d{name} direction {direction_index}: error {error:.3e} exceeds {tolerance:.3e}");
        }
    }
    analytic
}

/// The erase and decay gradients must be a material share of the gradient, so that a wrong or zero erase/decay
/// gradient cannot hide under the tolerance set by the larger groups.
fn assert_erase_decay_nondegenerate(label: &str, group_rms: [f64; 6]) {
    let largest = group_rms.iter().copied().fold(0.0, f64::max);
    println!("{label}: gradient rms per group {:?}", GROUPS.iter().zip(group_rms).collect::<Vec<_>>());
    for index in [3, 4] {
        assert!(
            group_rms[index] >= 1e-2 * largest,
            "{label}: d{} rms {:.3e} is below 1% of the largest group rms {largest:.3e}",
            GROUPS[index],
            group_rms[index],
        );
    }
}

/// GPU and oracle values of one input-gradient group, `[B, T, H, width]` order, with the padding mask.
struct GroupGradient {
    actual: Vec<f32>,
    expected: Vec<f64>,
    padding: Vec<bool>,
}

/// Runs the GPU scan on fixture tokens with a nonuniform upstream gradient and compares the output and all six input
/// gradients with the f64 oracle under PLAN §8.1, with padding positions exactly 0. `starts`/`ends` are row-major
/// `[B, T]`.
fn check_gpu_scan(name: &str, batch: usize, tokens: usize, starts: &[i32], ends: &[i32], interval: usize, seed: u64) -> Vec<GroupGradient> {
    check_gpu_scan_mode(name, batch, tokens, starts, ends, interval, seed, false, false)
}

fn check_gpu_scan_mode(name: &str, batch: usize, tokens: usize, starts: &[i32], ends: &[i32], interval: usize, seed: u64, chunked: bool, reverse: bool) -> Vec<GroupGradient> {
    use burn::prelude::*;
    use dre_core::gpu_device;

    assert_eq!(starts.len(), batch * tokens);
    assert_eq!(ends.len(), batch * tokens);
    let device = gpu_device().autodiff();
    let cells = batch * tokens * HEADS;
    let mut rng = SplitMix(seed);
    let mut data: Vec<Vec<f32>> = WIDTHS.iter().map(|&width| Vec::with_capacity(cells * width)).collect();
    for _ in 0..cells {
        let token = fixture_token(&mut rng);
        for (index, values) in data.iter_mut().enumerate() {
            values.extend(group(&token, index).iter().map(|&x| x as f32));
        }
    }
    let mut upstream: Vec<f32> = (0..cells * VALUE_DIM).map(|_| rng.signed() as f32).collect();
    if reverse {
        for b in 0..batch {
            for t in 0..tokens {
                let st = starts[b*tokens+t];
                let en = ends[b*tokens+t];
                if st == t as i32 && en >= st {
                    for offset in 0..(en-st+1) as usize / 2 {
                        let left = b*tokens+t+offset;
                        let right = b*tokens+en as usize-offset;
                        for (values, width) in data.iter_mut().zip(WIDTHS) {
                            for channel in 0..HEADS*width { values.swap(left*HEADS*width+channel, right*HEADS*width+channel); }
                        }
                        for channel in 0..HEADS*VALUE_DIM { upstream.swap(left*HEADS*VALUE_DIM+channel, right*HEADS*VALUE_DIM+channel); }
                    }
                }
            }
        }
    }
    let inputs: Vec<Tensor<4>> = data
        .iter()
        .zip(WIDTHS)
        .map(|(values, width)| {
            Tensor::<4>::from_data(TensorData::new(values.clone(), [batch, tokens, HEADS, width]), &device).require_grad()
        })
        .collect();
    let doc_start = Tensor::<2, Int>::from_data(TensorData::new(starts.to_vec(), [batch, tokens]), &device);
    let doc_end = Tensor::<2, Int>::from_data(TensorData::new(ends.to_vec(), [batch, tokens]), &device);
    let bf16 = !chunked && std::env::var_os("DRE_BF16_GATE").is_some();
    let dtype = if bf16 { burn::tensor::DType::BF16 } else { burn::tensor::DType::F32 };
    let output = crate::ops::gdn2_scan(
        inputs[0].clone().cast(dtype), inputs[1].clone().cast(dtype), inputs[2].clone().cast(dtype),
        inputs[3].clone(), inputs[4].clone(), inputs[5].clone(),
        doc_start, doc_end, interval, chunked,
    ).cast(burn::tensor::DType::F32);
    let actual_output = output.clone().into_data().try_to_vec::<f32>().unwrap();
    let upstream_tensor = Tensor::<4>::from_data(TensorData::new(upstream.clone(), [batch, tokens, HEADS, VALUE_DIM]), &device);
    let grads = (output * upstream_tensor).sum().backward();
    let actual: Vec<Vec<f32>> = inputs
        .iter()
        .map(|input| input.grad(&grads).expect("input gradient").into_data().try_to_vec::<f32>().unwrap())
        .collect();

    let slice = |values: &[f32], cell: usize, width: usize| -> Vec<f64> {
        values[cell * width..(cell + 1) * width].iter().map(|&x| f64::from(x)).collect()
    };
    let mut expected_output = vec![0.0f64; cells * VALUE_DIM];
    let mut expected: Vec<Vec<f64>> = WIDTHS.iter().map(|&width| vec![0.0; cells * width]).collect();
    for b in 0..batch {
        let row = b * tokens..(b + 1) * tokens;
        for h in 0..HEADS {
            let cell = |t: usize| (b * tokens + t) * HEADS + h;
            let host_tokens: Vec<Token> = (0..tokens)
                .map(|t| {
                    let c = cell(t);
                    Token::new(
                        slice(&data[0], c, KEY_DIM), slice(&data[1], c, KEY_DIM), slice(&data[2], c, VALUE_DIM),
                        slice(&data[3], c, KEY_DIM), slice(&data[4], c, KEY_DIM), slice(&data[5], c, VALUE_DIM),
                    )
                })
                .collect();
            let row_grad: Vec<Vec<f64>> = (0..tokens).map(|t| slice(&upstream, cell(t), VALUE_DIM)).collect();
            let forward = scan(&host_tokens, &starts[row.clone()], &ends[row.clone()]);
            let gradient = backward(&host_tokens, &starts[row.clone()], &ends[row.clone()], &row_grad);
            for t in 0..tokens {
                let c = cell(t);
                expected_output[c * VALUE_DIM..(c + 1) * VALUE_DIM].copy_from_slice(&forward.output[t]);
                for (index, values) in expected.iter_mut().enumerate() {
                    let width = WIDTHS[index];
                    values[c * width..(c + 1) * width].copy_from_slice(group(&gradient[t], index));
                }
            }
        }
    }

    let padding = |width: usize| -> Vec<bool> {
        (0..cells * width)
            .map(|element| {
                let position = element / (HEADS * width);
                ends[position] < starts[position]
            })
            .collect()
    };
    let mut comparisons = vec![compare_precision(&format!("{name} output"), &actual_output, &expected_output, &padding(VALUE_DIM), bf16)];
    let gradients: Vec<GroupGradient> = actual
        .into_iter()
        .zip(expected)
        .zip(WIDTHS)
        .map(|((actual, expected), width)| GroupGradient { actual, expected, padding: padding(width) })
        .collect();
    for (index, gradient) in gradients.iter().enumerate() {
        comparisons.push(compare_precision(&format!("{name} d{}", GROUPS[index]), &gradient.actual, &gradient.expected, &gradient.padding, bf16));
    }
    assert_all_pass(&comparisons);
    gradients
}

#[test]
fn gdn2_reference_reset_padding_and_gradients() {
    let mut rng = SplitMix(0x6764_6e32_0001);
    let tokens: Vec<Token> = (0..6).map(|_| fixture_token(&mut rng)).collect();
    // A four-token document, a one-token document, then one padding position.
    let (starts, ends) = row_limits(&[(true, 4), (true, 1), (false, 1)]);
    let result = scan(&tokens, &starts, &ends);
    assert!(result.output[5].iter().all(|&x| x == 0.0));
    assert!(result.states[5].iter().all(|&x| x == 0.0));
    for (range, (doc_starts, doc_ends)) in [(0..4, row_limits(&[(true, 4)])), (4..5, row_limits(&[(true, 1)]))] {
        let separate = scan(&tokens[range.clone()], &doc_starts, &doc_ends);
        for (t, output) in range.zip(&separate.output) {
            for j in 0..VALUE_DIM {
                assert!((result.output[t][j] - output[j]).abs() < 1e-12, "document reset changes output[{t}][{j}]");
            }
        }
    }

    let grad: Vec<Vec<f64>> = (0..tokens.len()).map(|_| (0..VALUE_DIM).map(|_| rng.signed()).collect()).collect();
    let analytic = check_oracle_directions(&tokens, &starts, &ends, &grad, &mut rng);
    for (index, name) in GROUPS.iter().enumerate() {
        assert!(group(&analytic[5], index).iter().all(|&g| g == 0.0), "padding d{name} is not exactly 0");
    }
    let group_rms = std::array::from_fn(|index| {
        rms(&analytic.iter().flat_map(|token| group(token, index).iter().copied()).collect::<Vec<_>>())
    });
    assert_erase_decay_nondegenerate("oracle fixture", group_rms);
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn gdn2_scan_gpu_matches_f64_and_gradients() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    // Interval 2 splits the row into three checkpoint segments; the one-token document resets the state.
    let (starts, ends) = row_limits(&[(true, 4), (true, 1), (false, 1)]);
    let gradients = check_gpu_scan("B=1 T=6", 1, 6, &starts, &ends, 2, 0x6764_6e32_0002);
    assert_erase_decay_nondegenerate("GPU fixture", std::array::from_fn(|index| rms(&gradients[index].expected)));
    // The comparison discriminates: an all-zero gradient fails it for every group, erase and decay included.
    for (index, gradient) in gradients.iter().enumerate() {
        let zeros = vec![0.0f32; gradient.expected.len()];
        assert!(
            !compare(GROUPS[index], &zeros, &gradient.expected, &gradient.padding).pass,
            "an all-zero d{} passes the comparison",
            GROUPS[index],
        );
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn gdn2_m0b_warm_forward_backward_timing() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use burn::prelude::*;
    use dre_core::{BatchLayout, DocumentLayout, gpu_device};
    use std::time::Instant;

    let device = gpu_device().autodiff();
    let selected_batch = std::env::var("DRE_M0B_BATCH").ok();
    let batches = match selected_batch.as_deref() {
        Some("1") => vec![1usize],
        Some("4") => vec![4usize],
        Some(_) => panic!("DRE_M0B_BATCH must be 1 or 4"),
        None => vec![1usize, 4],
    };
    for batch in batches {
        let tokens = 2048usize;
        let q = Tensor::<4>::zeros([batch, tokens, 4, KEY_DIM], &device).require_grad();
        let k = Tensor::<4>::zeros([batch, tokens, 4, KEY_DIM], &device).require_grad();
        let v = Tensor::<4>::ones([batch, tokens, 4, VALUE_DIM], &device).require_grad();
        let alpha = Tensor::<4>::full([batch, tokens, 4, KEY_DIM], 0.99, &device).require_grad();
        let erase = Tensor::<4>::full([batch, tokens, 4, KEY_DIM], 0.5, &device).require_grad();
        let write = Tensor::<4>::full([batch, tokens, 4, VALUE_DIM], 0.5, &device).require_grad();
        let layout = BatchLayout::new(
            DocumentLayout::packed(vec![vec![tokens]; batch], tokens),
            &device,
        );
        let measure = || {
            let output = crate::ops::gdn2_scan(
                q.clone(), k.clone(), v.clone(), alpha.clone(), erase.clone(), write.clone(),
                layout.doc_start.clone(), layout.doc_end.clone(), 32, false,
            );
            let grads = output.sum().backward();
            let _ = q.grad(&grads).unwrap();
            device.sync().unwrap();
        };
        for _ in 0..3 { measure(); }
        let mut elapsed = Vec::with_capacity(5);
        for _ in 0..5 {
            let start = Instant::now();
            measure();
            elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        println!("GDN-2 B={batch}, T={tokens}: forward+backward {elapsed:?} ms");
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn gdn2_kernel_scaling_timing() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use burn::prelude::*;
    use dre_core::{BatchLayout, DocumentLayout, gpu_device};
    use std::time::Instant;

    let device = gpu_device().autodiff();
    for tokens in [8usize, 32, 128] {
        let q = Tensor::<4>::zeros([1, tokens, 4, KEY_DIM], &device).require_grad();
        let k = Tensor::<4>::zeros([1, tokens, 4, KEY_DIM], &device).require_grad();
        let v = Tensor::<4>::ones([1, tokens, 4, VALUE_DIM], &device).require_grad();
        let alpha = Tensor::<4>::full([1, tokens, 4, KEY_DIM], 0.99, &device).require_grad();
        let erase = Tensor::<4>::full([1, tokens, 4, KEY_DIM], 0.5, &device).require_grad();
        let write = Tensor::<4>::full([1, tokens, 4, VALUE_DIM], 0.5, &device).require_grad();
        let layout = BatchLayout::new(DocumentLayout::packed(vec![vec![tokens]], tokens), &device);
        let measure = || {
            let output = crate::ops::gdn2_scan(
                q.clone(), k.clone(), v.clone(), alpha.clone(), erase.clone(), write.clone(),
                layout.doc_start.clone(), layout.doc_end.clone(), 32, false,
            );
            let grads = output.sum().backward();
            let _ = q.grad(&grads).unwrap();
            device.sync().unwrap();
        };
        measure();
        let start = Instant::now();
        measure();
        println!("GDN-2 B=1,T={tokens}: warm forward+backward {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn gdn2_scan_gpu_long_packed_rows_match_f64() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    for tokens in [33usize, 257] {
        let first = tokens / 2;
        let second = tokens - first - 1;
        let (starts, ends) = row_limits(&[(true, first), (true, second), (false, 1)]);
        check_gpu_scan(&format!("B=1 T={tokens}"), 1, tokens, &starts, &ends, 32, 0x6764_6e32_0003 + tokens as u64);
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn gdn2_scan_partial_chunk_batched_gradients_match_f64() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let (batch, tokens) = (3usize, 45usize);
    let (row_starts, row_ends) = row_limits(&[(true, 17), (true, 27), (false, 1)]);
    check_gpu_scan("B=3 T=45", batch, tokens, &row_starts.repeat(batch), &row_ends.repeat(batch), 32, 0x6764_6e32_0004);
}

/// At B=16 with the production interval 32, four heads and key width 32, the gradient reduction needs
/// 16·32·4·32 = 65,536 workgroups, one more than a WGPU dispatch dimension holds, so it is split across X and Y.
/// Rows also put padding between documents, where the reused partial-gradient buffers must be cleared.
#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn gdn2_scan_batch16_split_dispatch_matches_f64() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let (batch, tokens, interval) = (16usize, 64usize, 32usize);
    assert!(batch * interval * HEADS * KEY_DIM > u16::MAX as usize);
    let patterns: [&[(bool, usize)]; 4] = [
        &[(true, 64)],
        &[(true, 20), (true, 30), (false, 14)],
        &[(true, 10), (false, 12), (true, 42)],
        &[(true, 5), (true, 31), (false, 3), (true, 7), (false, 18)],
    ];
    let (mut starts, mut ends) = (Vec::new(), Vec::new());
    for row in 0..batch {
        let (row_starts, row_ends) = row_limits(patterns[row % patterns.len()]);
        assert_eq!(row_starts.len(), tokens);
        starts.extend(row_starts);
        ends.extend(row_ends);
    }
    check_gpu_scan("B=16 T=64", batch, tokens, &starts, &ends, interval, 0x6764_6e32_0005);
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn gdn2_chunked_packed_both_directions_f64_gate() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    for tokens in [5usize, 33, 64, 257, 2048] {
        let first = (tokens / 3).max(1);
        let (starts, ends) = row_limits(&[(true, first), (false, 1), (true, tokens-first-2), (false, 1)]);
        // Reverse each document independently, not the packed row. Mirroring the
        // seeded fixture's traversal produces independent forward/reverse inputs.
        for direction in 0..2 {
            check_gpu_scan_mode(&format!("chunked direction={direction} T={tokens}"), 1, tokens, &starts, &ends, 32,
                0x6764_6e32_1000 + tokens as u64, true, direction == 1);
        }
    }
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn gdn2_chunked_vs_serial_m0b_timing() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use burn::prelude::*;
    use dre_core::{BatchLayout, DocumentLayout, gpu_device};
    use std::time::Instant;
    let device = gpu_device().autodiff();
    let (batch, tokens) = (4usize, 2048usize);
    let mut rng = SplitMix(0x6764_6e32_2000);
    let mut data: Vec<Vec<f32>> = WIDTHS.iter().map(|&width| Vec::with_capacity(batch*tokens*HEADS*width)).collect();
    for _ in 0..batch*tokens*HEADS {
        let token = fixture_token(&mut rng);
        for (index, values) in data.iter_mut().enumerate() { values.extend(group(&token,index).iter().map(|&x| x as f32)); }
    }
    let inputs: Vec<Tensor<4>> = data.into_iter().zip(WIDTHS).map(|(values,width)|
        Tensor::<4>::from_data(TensorData::new(values,[batch,tokens,HEADS,width]),&device).require_grad()).collect();
    let layout = BatchLayout::new(DocumentLayout::packed(vec![vec![tokens];batch],tokens),&device);
    let upstream = Tensor::<4>::from_data(TensorData::new(
        (0..batch*tokens*HEADS*VALUE_DIM).map(|_| rng.signed() as f32).collect::<Vec<_>>(),
        [batch,tokens,HEADS,VALUE_DIM]),&device);
    for chunked in [false,true] {
        let measure = || {
            let output = crate::ops::gdn2_scan(inputs[0].clone(),inputs[1].clone(),inputs[2].clone(),
                inputs[3].clone(),inputs[4].clone(),inputs[5].clone(),
                layout.doc_start.clone(),layout.doc_end.clone(),32,chunked);
            let grads = (output*upstream.clone()).sum().backward();
            for input in &inputs { let _ = input.grad(&grads).unwrap(); }
            device.sync().unwrap();
        };
        for _ in 0..3 { measure(); }
        let mut times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            measure();
            times.push(start.elapsed().as_secs_f64()*1000.0);
        }
        println!("GDN-2 scan={} B=4 T=2048 one direction fwd+bwd {times:?} ms", if chunked { "chunked" } else { "serial" });
    }
}

#[test]
fn gdn2_scan_config_old_checkpoints_load() {
    use burn::config::Config;
    // Burn's Config derive supplies constructor defaults, not serde defaults.
    // This is the complete pre-switch GDN config from a checkpoint.
    let old = crate::Gdn2BidirectionalConfig::load_binary(br#"{
        "precision":"fp32","d_model":512,"heads":4,"key_dim":32,"value_dim":64,
        "conv_width":4,"checkpoint_interval":32,"norm_epsilon":0.00001
    }"#).unwrap();
    assert!(old.scan.is_none());
    let default = crate::Gdn2BidirectionalConfig::new();
    assert_eq!(default.scan.as_deref(), Some("chunked"));
    let chunked = crate::Gdn2BidirectionalConfig::load_binary(default.to_string().as_bytes()).unwrap();
    assert_eq!(chunked.scan.as_deref(), Some("chunked"));
    let serial = default.with_scan(Some("serial".to_string()));
    assert_eq!(crate::Gdn2BidirectionalConfig::load_binary(serial.to_string().as_bytes()).unwrap().scan.as_deref(), Some("serial"));
    assert_eq!(chunked.heads, old.heads);
    assert_eq!(chunked.key_dim, old.key_dim);
    assert_eq!(chunked.value_dim, old.value_dim);
}
