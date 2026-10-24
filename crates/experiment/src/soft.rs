//! Vocabulary port of reuse/crates/dre-relations/{state,diffusion_scheduler,denoiser}.rs.
//! Convex probability updates, visibility-only remasking, and weighted mask mixing are preserved.
//! This standalone episode is not SBD training and never updates recurrent state.
use crate::{config::Config, model::scalar};
use anyhow::{Result, ensure};
use burn::{prelude::*, tensor::activation::softmax};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Candidates {
    pub ids: Tensor<2, Int>,
    pub weights: Tensor<2>,
    pub retained_mass: Tensor<2>,
}
impl Candidates {
    pub fn select(probabilities: Tensor<2>, width: usize) -> Result<Self> {
        let [rows, vocab] = probabilities.dims();
        let k = width.min(vocab);
        let device = probabilities.device();
        let values = probabilities
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("candidate values {e:?}"))?;
        ensure!(
            values.iter().all(|v| v.is_finite() && *v >= 0.0),
            "candidate probabilities must be finite and nonnegative"
        );
        let mut ids = Vec::with_capacity(rows * k);
        for row in values.chunks_exact(vocab) {
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            ids.extend(order[..k].iter().map(|&i| i as i32));
        }
        let ids = Tensor::from_data(TensorData::new(ids, [rows, k]), &device);
        // Index selection is discrete; selected probabilities and gathered embeddings stay differentiable.
        let weights = probabilities.gather(1, ids.clone());
        let retained_mass = weights.clone().sum_dim(1);
        Ok(Self {
            ids,
            weights,
            retained_mass,
        })
    }
    pub fn dense(&self, vocab: usize) -> Tensor<2> {
        let [rows, _] = self.ids.dims();
        Tensor::zeros([rows, vocab], &self.weights.device()).scatter(
            1,
            self.ids.clone(),
            self.weights.clone(),
            burn::tensor::IndexingUpdateOp::Add,
        )
    }
    pub fn expected(
        &self,
        embeddings: Tensor<2>,
        mask: Tensor<1>,
        visibility: Tensor<2>,
        hard_mask: bool,
    ) -> Tensor<2> {
        let rows = self.ids.dims()[0];
        let active = Tensor::ones([rows, 1], &visibility.device());
        self.expected_masked(embeddings, mask, visibility, active, hard_mask)
    }
    pub fn expected_masked(
        &self,
        embeddings: Tensor<2>,
        mask: Tensor<1>,
        visibility: Tensor<2>,
        active: Tensor<2>,
        hard_mask: bool,
    ) -> Tensor<2> {
        let [rows, k] = self.ids.dims();
        let d = embeddings.dims()[1];
        let inactive = active.lower_elem(0.5);
        let visibility = visibility.mask_fill(inactive.clone(), 0.0);
        if hard_mask {
            let first = self.ids.clone().slice([0..rows, 0..1]).reshape([rows]);
            let visible = visibility
                .greater_equal_elem(1.0)
                .bool_and(self.retained_mass.clone().greater_elem(0.0));
            let values = embeddings
                .select(0, first)
                .mask_fill(visible.clone().bool_not().expand([rows, d]), 0.0);
            let masks = mask
                .reshape([1, d])
                .expand([rows, d])
                .mask_fill(visible.clone().bool_or(inactive).expand([rows, d]), 0.0);
            return values * visible.clone().float()
                + masks * visible.float().neg().add_scalar(1.0);
        }
        // Hide beliefs and selected values before products: zero times NaN is not zero.
        let hidden = visibility.clone().equal_elem(0.0).bool_or(inactive.clone());
        let weights = self
            .weights
            .clone()
            .mask_fill(hidden.clone().expand([rows, k]), 0.0);
        let selected = embeddings
            .select(0, self.ids.clone().reshape([rows * k]))
            .reshape([rows, k, d]);
        let unused_nonfinite = weights
            .clone()
            .equal_elem(0.0)
            .unsqueeze_dim::<3>(2)
            .expand([rows, k, d])
            .bool_and(selected.clone().is_finite().bool_not());
        let selected = selected.mask_fill(
            hidden
                .expand([rows, k])
                .unsqueeze_dim::<3>(2)
                .expand([rows, k, d])
                .bool_or(unused_nonfinite),
            0.0,
        );
        let mass = weights.clone().sum_dim(1);
        let weighted = (selected * weights.unsqueeze_dim(2))
            .sum_dim(1)
            .reshape([rows, d]);
        let mask_weight = (visibility.clone() * mass).neg().add_scalar(1.0);
        let masks = mask.reshape([1, d]).expand([rows, d]).mask_fill(
            mask_weight
                .clone()
                .equal_elem(0.0)
                .bool_or(inactive)
                .expand([rows, d]),
            0.0,
        );
        weighted * visibility + masks * mask_weight
    }
}
pub fn dense_expected(
    probabilities: Tensor<2>,
    embeddings: Tensor<2>,
    mask: Tensor<1>,
    visibility: Tensor<2>,
) -> Tensor<2> {
    let rows = probabilities.dims()[0];
    let active = Tensor::ones([rows, 1], &visibility.device());
    dense_expected_masked(probabilities, embeddings, mask, visibility, active)
}
pub fn dense_expected_masked(
    probabilities: Tensor<2>,
    embeddings: Tensor<2>,
    mask: Tensor<1>,
    visibility: Tensor<2>,
    active: Tensor<2>,
) -> Tensor<2> {
    let [rows, vocab] = probabilities.dims();
    let d = embeddings.dims()[1];
    let inactive = active.lower_elem(0.5);
    let visibility = visibility.mask_fill(inactive.clone(), 0.0);
    let hidden = visibility.clone().equal_elem(0.0).bool_or(inactive.clone());
    let probabilities = probabilities.mask_fill(hidden.expand([rows, vocab]), 0.0);
    // Keep finite zero-probability derivatives; sanitize only unused nonfinite embedding rows.
    let unused = probabilities
        .clone()
        .abs()
        .sum_dim(0)
        .equal_elem(0.0)
        .transpose()
        .expand([vocab, d]);
    let unused_nonfinite = unused.bool_and(embeddings.clone().is_finite().bool_not());
    let embeddings = embeddings.mask_fill(unused_nonfinite, 0.0);
    let mass = probabilities.clone().sum_dim(1);
    let mask_weight = (visibility.clone() * mass).neg().add_scalar(1.0);
    let masks = mask.reshape([1, d]).expand([rows, d]).mask_fill(
        mask_weight
            .clone()
            .equal_elem(0.0)
            .bool_or(inactive)
            .expand([rows, d]),
        0.0,
    );
    probabilities.matmul(embeddings) * visibility + masks * mask_weight
}
fn host<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .try_to_vec::<f32>()
        .expect("float32 feedback")
}
fn difference<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f64 {
    host(a - b).iter().fold(0.0f64, |m, v| {
        if v.is_finite() {
            m.max(v.abs() as f64)
        } else {
            f64::INFINITY
        }
    })
}

#[derive(Clone, Debug)]
pub struct SoftState {
    pub candidates: Candidates,
    pub visibility: Vec<f32>,
    pub observed: Vec<Option<usize>>,
    pub active: Vec<bool>,
    pub provisional: Vec<Option<usize>>,
    pub retained: Vec<Option<usize>>,
    vocab: usize,
}
impl SoftState {
    pub fn new(
        vocab: usize,
        width: usize,
        observed: Vec<Option<usize>>,
        active: Vec<bool>,
        device: &Device,
    ) -> Result<Self> {
        let rows = observed.len();
        ensure!(rows == active.len() && vocab > 0, "invalid feedback shape");
        let mut p = vec![0.0; rows * vocab];
        let mut visibility = vec![0.0; rows];
        for row in 0..rows {
            if active[row] {
                if let Some(id) = observed[row] {
                    ensure!(id < vocab, "observed token outside vocabulary");
                    p[row * vocab + id] = 1.0;
                    visibility[row] = 1.0;
                } else {
                    p[row * vocab..(row + 1) * vocab].fill(1.0 / vocab as f32);
                }
            }
        }
        Ok(Self {
            candidates: Candidates::select(
                Tensor::from_data(TensorData::new(p, [rows, vocab]), device),
                width,
            )?,
            visibility,
            observed,
            active,
            provisional: vec![None; rows],
            retained: vec![None; rows],
            vocab,
        })
    }
    pub fn update(
        &mut self,
        logits: Tensor<2>,
        eta: f64,
        temperature: f64,
        steps: usize,
        width: usize,
    ) -> Result<()> {
        let rows = self.visibility.len();
        ensure!(logits.dims() == [rows, self.vocab], "feedback logits shape");
        let raw = host(logits.clone());
        let old = self.candidates.dense(self.vocab);
        let mut eta_rows = vec![0.0; rows];
        for row in 0..rows {
            let values = &raw[row * self.vocab..(row + 1) * self.vocab];
            let absent = values.iter().all(|v| *v == f32::NEG_INFINITY);
            ensure!(
                values
                    .iter()
                    .all(|v| v.is_finite() || *v == f32::NEG_INFINITY),
                "nonfinite feedback logits"
            );
            if self.active[row] && self.observed[row].is_none() {
                self.visibility[row] = (self.visibility[row] + 1.0 / steps as f32).min(1.0);
                if !absent {
                    eta_rows[row] = eta as f32;
                    let best = (0..self.vocab)
                        .max_by(|&a, &b| values[a].total_cmp(&values[b]).then(b.cmp(&a)))
                        .unwrap();
                    self.provisional[row] = Some(best);
                }
            }
        }
        let device = logits.device();
        // Preserve the logits graph; only unavailable rows are sanitized, exactly as copied dense update.
        let unavailable = Tensor::<2>::from_data(
            TensorData::new(
                eta_rows
                    .iter()
                    .map(|&v| if v == 0.0 { 1.0f32 } else { 0.0 })
                    .collect::<Vec<_>>(),
                [rows, 1],
            ),
            &device,
        )
        .greater_elem(0.5);
        let prediction = softmax(
            logits.mask_fill(unavailable.expand([rows, self.vocab]), 0.0) / temperature,
            1,
        );
        let mix = Tensor::from_data(TensorData::new(eta_rows, [rows, 1]), &device);
        self.candidates = Candidates::select(
            old * mix.clone().neg().add_scalar(1.0) + prediction * mix,
            width,
        )?;
        Ok(())
    }
    pub fn remask(&mut self, positions: &[usize]) {
        for &row in positions {
            if self.active[row] && self.observed[row].is_none() {
                self.visibility[row] = 0.0;
            }
        }
    }
    pub fn retain(&mut self, positions: &[usize]) {
        for &row in positions {
            if self.active[row] && self.observed[row].is_none() {
                if let Some(value) = self.provisional[row] {
                    self.retained[row] = Some(value);
                }
            }
        }
    }
    pub fn embeddings(&self, table: Tensor<2>, mask: Tensor<1>, hard: bool) -> Tensor<2> {
        let rho = Tensor::from_data(
            TensorData::new(self.visibility.clone(), [self.visibility.len(), 1]),
            &table.device(),
        );
        let active = Tensor::<2>::from_data(
            TensorData::new(
                self.active
                    .iter()
                    .map(|&v| if v { 1.0f32 } else { 0.0 })
                    .collect::<Vec<_>>(),
                [self.active.len(), 1],
            ),
            &table.device(),
        );
        self.candidates
            .expected_masked(table, mask, rho, active, hard)
    }
}
fn endpoint_poison_checks(device: &Device) -> Result<Value> {
    for (name, rho, active, mask_values, goal) in [
        ("hidden", 0.0f32, 1.0f32, [0.2f32, 0.3], [0.2f32, 0.3]),
        (
            "padding",
            1.0f32,
            0.0f32,
            [f32::NAN, f32::INFINITY],
            [0.0f32, 0.0],
        ),
    ] {
        let table =
            Tensor::from_data([[f32::NAN, f32::INFINITY], [0.5f32, -0.25]], device).require_grad();
        let weights = Tensor::from_data([[f32::NAN, f32::INFINITY]], device).require_grad();
        let mask = Tensor::from_data(mask_values, device).require_grad();
        let visibility = Tensor::from_data([[rho]], device);
        let active = Tensor::from_data([[active]], device);
        let candidates = Candidates {
            ids: Tensor::from_data([[0i32, 1]], device),
            weights: weights.clone(),
            retained_mass: weights.clone().sum_dim(1),
        };
        let sparse = candidates.expected_masked(
            table.clone(),
            mask.clone(),
            visibility.clone(),
            active.clone(),
            false,
        );
        let dense = dense_expected_masked(
            weights.clone(),
            table.clone(),
            mask.clone(),
            visibility,
            active,
        );
        for (path, output) in [("sparse", sparse), ("dense", dense)] {
            let values = host(output.clone());
            ensure!(
                values
                    .iter()
                    .zip(goal)
                    .all(|(a, b)| a.is_finite() && (*a - b).abs() < 1e-6),
                "{name} {path} poison reached output"
            );
            let gradients = output.sum().backward();
            let weight_grad = host(weights.grad(&gradients).context("poison weight gradient")?);
            let embedding_grad = host(
                table
                    .grad(&gradients)
                    .context("poison embedding gradient")?,
            );
            ensure!(
                weight_grad
                    .iter()
                    .chain(&embedding_grad)
                    .all(|v| v.is_finite() && v.abs() < 1e-7),
                "{name} {path} poison contaminated gradients"
            );
        }
    }
    let table = Tensor::from_data([[0.2f32, 0.4], [0.6, 0.8]], device).require_grad();
    let weights = Tensor::from_data([[0.25f32, 0.75]], device).require_grad();
    let mask = Tensor::from_data([f32::NAN, f32::INFINITY], device).require_grad();
    let candidates = Candidates {
        ids: Tensor::from_data([[0i32, 1]], device),
        weights: weights.clone(),
        retained_mass: weights.clone().sum_dim(1),
    };
    let sparse = candidates.expected(
        table.clone(),
        mask.clone(),
        Tensor::ones([1, 1], device),
        false,
    );
    let dense = dense_expected(
        weights.clone(),
        table.clone(),
        mask.clone(),
        Tensor::ones([1, 1], device),
    );
    for output in [sparse, dense] {
        ensure!(
            host(output.clone()).iter().all(|v| v.is_finite()),
            "fully visible mask poison reached output"
        );
        let gradients = output.sum().backward();
        ensure!(
            host(mask.grad(&gradients).context("invisible mask gradient")?)
                .iter()
                .all(|v| v.is_finite() && v.abs() < 1e-7),
            "invisible mask contaminated gradient"
        );
        ensure!(
            host(weights.grad(&gradients).context("visible weights")?)
                .iter()
                .all(|v| v.is_finite()),
            "invisible mask contaminated weights"
        );
    }
    let poison = Tensor::from_data([[f32::NAN, 0.0f32], [0.0, 1.0]], device);
    let active_failure = dense_expected(
        Tensor::from_data([[0.5f32, 0.5]], device),
        poison,
        Tensor::zeros([2], device),
        Tensor::ones([1, 1], device),
    );
    ensure!(
        scalar(active_failure.sum()).is_err(),
        "active nonfinite embedding was silently suppressed"
    );
    Ok(
        json!({"hidden_and_padding_poison_outputs_and_gradients":"checked","fully_visible_mask_poison":"checked","active_nonfinite_failure_preserved":true}),
    )
}
pub fn episode(c: &Config, device: &Device) -> Result<Value> {
    let device = device.clone().autodiff();
    let vocab = 8;
    let d = 3;
    let table = Tensor::from_data(
        TensorData::new(
            (0..vocab * d)
                .map(|i| (i as f32 - 10.0) / 20.0)
                .collect::<Vec<_>>(),
            [vocab, d],
        ),
        &device,
    )
    .require_grad();
    let mask = Tensor::from_data([0.2f32, -0.1, 0.3], &device);
    let probabilities =
        Tensor::from_data([[0.1f32, 0.2, 0.0, 0.3, 0.1, 0.1, 0.1, 0.1]], &device).require_grad();
    let rho = Tensor::from_data([[0.7f32]], &device);
    let full = Candidates::select(probabilities.clone(), vocab)?;
    let dense = dense_expected(
        probabilities.clone(),
        table.clone(),
        mask.clone(),
        rho.clone(),
    );
    let sparse = full.expected(table.clone(), mask.clone(), rho.clone(), false);
    let expectation_error = difference(dense.clone(), sparse.clone());
    ensure!(
        expectation_error < 2e-6,
        "sparse/dense expectation mismatch"
    );
    let gd = dense.sum().backward();
    let gs = sparse.sum().backward();
    let weight_gradient_error = difference(
        probabilities
            .grad(&gd)
            .context("dense probability gradient")?,
        probabilities
            .grad(&gs)
            .context("sparse probability gradient")?,
    );
    let embedding_gradient_error = difference(
        table.grad(&gd).context("dense embedding gradient")?,
        table.grad(&gs).context("sparse embedding gradient")?,
    );
    ensure!(
        weight_gradient_error < 2e-6 && embedding_gradient_error < 2e-6,
        "feedback gradient mismatch"
    );
    let truncated_dense = Candidates::select(probabilities.clone(), 3)?;
    let truncated_sparse = Candidates::select(probabilities.clone(), 3)?;
    let dense = dense_expected(
        truncated_dense.dense(vocab),
        table.clone(),
        mask.clone(),
        rho.clone(),
    );
    let sparse = truncated_sparse.expected(table.clone(), mask.clone(), rho.clone(), false);
    let truncated_expectation_error = difference(dense.clone(), sparse.clone());
    let dense_grads = dense.sum().backward();
    let sparse_grads = sparse.sum().backward();
    let truncated_weight_gradient_error = difference(
        probabilities
            .grad(&dense_grads)
            .context("truncated dense weight gradient")?,
        probabilities
            .grad(&sparse_grads)
            .context("truncated sparse weight gradient")?,
    );
    let truncated_embedding_gradient_error = difference(
        table
            .grad(&dense_grads)
            .context("truncated dense embedding gradient")?,
        table
            .grad(&sparse_grads)
            .context("truncated sparse embedding gradient")?,
    );
    ensure!(
        truncated_expectation_error < 2e-6
            && truncated_weight_gradient_error < 2e-6
            && truncated_embedding_gradient_error < 2e-6,
        "truncated-support feedback mismatch"
    );
    let zero = Candidates::select(Tensor::zeros([1, vocab], &device), 3)?;
    ensure!(
        difference(
            zero.expected(table.clone(), mask.clone(), rho, false),
            mask.clone().reshape([1, d])
        ) < 1e-7,
        "zero mass must produce mask embedding"
    );
    let tied = Candidates::select(Tensor::ones([1, vocab], &device) / (vocab as f64), 3)?;
    ensure!(
        tied.ids
            .into_data()
            .try_to_vec_as::<i32>()
            .map_err(|e| anyhow::anyhow!("tie IDs: {e:?}"))?
            == vec![0, 1, 2],
        "ties must select ascending token ID"
    );
    let mut state = SoftState::new(
        vocab,
        c.soft_candidate_width,
        vec![Some(2), None, None, None],
        vec![true, true, true, false],
        &device,
    )?;
    let mut rounds = Vec::new();
    let mut full_state = SoftState::new(
        vocab,
        vocab,
        state.observed.clone(),
        state.active.clone(),
        &device,
    )?;
    let mut update_error = 0.0f64;
    for step in 0..c.soft_steps {
        let embedded = state.embeddings(table.clone(), mask.clone(), c.hard_mask_only);
        let logits = embedded.matmul(table.clone().transpose())
            + Tensor::from_data(
                TensorData::new(
                    (0..4 * vocab)
                        .map(|i| ((i + step * 3) % vocab) as f32 / 4.0)
                        .collect::<Vec<_>>(),
                    [4, vocab],
                ),
                &device,
            );
        let before = full_state.candidates.dense(vocab);
        let before_values = host(before.clone());
        let predicted = softmax(logits.clone() / c.soft_temperature, 1);
        let reference = before * (1.0 - c.soft_eta) + predicted * c.soft_eta;
        full_state.update(
            logits.clone(),
            c.soft_eta,
            c.soft_temperature,
            c.soft_steps,
            vocab,
        )?;
        let expected = host(reference);
        let after = host(full_state.candidates.dense(vocab));
        for row in 0..4 {
            for id in 0..vocab {
                let goal = if row == 0 || row == 3 {
                    before_values[row * vocab + id]
                } else {
                    expected[row * vocab + id]
                };
                let delta = after[row * vocab + id] - goal;
                ensure!(
                    delta.is_finite(),
                    "nonfinite dense/full-support update comparison at row {row}, token {id}"
                );
                update_error = update_error.max(delta.abs() as f64);
            }
        }
        state.update(
            logits,
            c.soft_eta,
            c.soft_temperature,
            c.soft_steps,
            c.soft_candidate_width,
        )?;
        if step == 0 || step + 1 == c.soft_steps {
            state.retain(&[1, 2]);
        }
        if step == 0 {
            let before = host(state.candidates.weights.clone());
            let retained = state.retained.clone();
            state.remask(&[0, 1, 3]);
            ensure!(
                before == host(state.candidates.weights.clone())
                    && retained == state.retained
                    && state.visibility[0] == 1.0,
                "remasking erased beliefs or changed observations"
            );
        }
        ensure!(
            state.observed[0] == Some(2)
                && state.provisional[0].is_none()
                && state.visibility[3] == 0.0,
            "observed/padding transition failure"
        );
        rounds.push(json!({"step":step,"ids":state.candidates.ids.clone().into_data().try_to_vec_as::<i32>().map_err(|e|anyhow::anyhow!("candidate IDs: {e:?}"))?,"weights":host(state.candidates.weights.clone()),"retained_mass":host(state.candidates.retained_mass.clone()),"visibility":state.visibility,"provisional":state.provisional,"retained":state.retained}));
    }
    ensure!(update_error < 2e-6, "full-support convex feedback mismatch");
    let mut unavailable = SoftState::new(vocab, vocab, vec![None], vec![true], &device)?;
    let previous = host(unavailable.candidates.weights.clone());
    unavailable.update(
        Tensor::full([1, vocab], f32::NEG_INFINITY, &device),
        c.soft_eta,
        c.soft_temperature,
        c.soft_steps,
        vocab,
    )?;
    ensure!(
        host(unavailable.candidates.weights.clone()) == previous
            && unavailable.provisional[0].is_none()
            && unavailable.visibility[0] > 0.0,
        "unavailable predictions must retain beliefs independently of visibility gain"
    );
    let final_norm = scalar(
        state
            .embeddings(table, mask, c.hard_mask_only)
            .square()
            .sum(),
    )?;
    let endpoint_checks = endpoint_poison_checks(&device)?;
    Ok(
        json!({"mechanism":"copied_dense_convex_feedback_vocabulary_port","not_sbd":true,"candidate_width":c.soft_candidate_width,"hard_mask_only":c.hard_mask_only,"expectation_max_error":expectation_error,"weight_gradient_max_error":weight_gradient_error,"embedding_gradient_max_error":embedding_gradient_error,"truncated_expectation_max_error":truncated_expectation_error,"truncated_weight_gradient_max_error":truncated_weight_gradient_error,"truncated_embedding_gradient_max_error":truncated_embedding_gradient_error,"endpoint_checks":endpoint_checks,"full_support_update_max_error":update_error,"final_embedding_squared_norm":final_norm,"rounds":rounds}),
    )
}
use anyhow::Context;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_dense_feedback_and_transitions() {
        episode(&Config::default(), &Device::ndarray()).unwrap();
    }
    #[test]
    fn agreement_oracle_rejects_nonfinite_values() {
        let device = Device::ndarray();
        assert!(
            difference(
                Tensor::<2>::from_data([[f32::NAN]], &device),
                Tensor::zeros([1, 1], &device)
            )
            .is_infinite()
        );
        assert!(
            difference(
                Tensor::<2>::from_data([[f32::INFINITY]], &device),
                Tensor::zeros([1, 1], &device)
            )
            .is_infinite()
        );
        assert_eq!(
            difference(
                Tensor::<2>::zeros([1, 1], &device),
                Tensor::zeros([1, 1], &device)
            ),
            0.0
        );
    }
}
