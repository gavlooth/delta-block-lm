//! Soft-masking: latest detached predictions, normalized top-k, and learned entropy weighting.
//! Candidate information survives remasking; completed observations remain exact inputs.
use crate::{
    config::{Config, SoftSelectionBackend},
    model::scalar,
};
use anyhow::{Result, ensure};
use burn::{
    module::{Param, ParamId},
    prelude::*,
    tensor::activation::{sigmoid, softmax, softplus},
};
use serde_json::{Value, json};

/// Paper Eq. (3), with the official constrained parameterization and initialization.
#[derive(Module, Debug)]
pub struct SoftConfidence {
    pub(crate) raw_scale: Param<Tensor<1>>,
    pub(crate) raw_centre_neg: Param<Tensor<1>>,
    pub(crate) raw_steep: Param<Tensor<1>>,
}
impl SoftConfidence {
    pub fn new(device: &Device) -> Self {
        Self::from_raw([
            (1e-6f64 / (1.0 - 1e-6)).ln(),
            0.75f64.exp_m1().ln(),
            (10.0f64 / 1.5).exp_m1().ln(),
        ], device)
    }
    pub(crate) fn from_raw(raw: [f64; 3], device: &Device) -> Self {
        let param = |value: f64| {
            let value = Tensor::full([1], value, device);
            let value = if device.is_autodiff() { value.require_grad() } else { value };
            Param::initialized(ParamId::new(), value)
        };
        Self {
            raw_scale: param(raw[0]),
            raw_centre_neg: param(raw[1]),
            raw_steep: param(raw[2]),
        }
    }
    pub fn weight(&self, negative_entropy: Tensor<2>) -> Tensor<2> {
        let scale = sigmoid(self.raw_scale.val()).reshape([1, 1]);
        let centre = -softplus(self.raw_centre_neg.val(), 1.0) - 1e-6;
        let steep = softplus(self.raw_steep.val(), 1.0) + 1e-6;
        scale * sigmoid((negative_entropy.detach() - centre.reshape([1, 1]))
            * steep.reshape([1, 1]))
    }
}

#[derive(Clone, Debug)]
pub struct Candidates {
    pub ids: Tensor<2, Int>,
    pub weights: Tensor<2>,
    pub retained_mass: Tensor<2>,
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn top_ids(keys: Tensor<2, Int>, width: usize) -> Tensor<2, Int> {
    // Burn 0.22's frontend argtopk unnecessarily asserts k < dim; the
    // backend (also used by topk_with_indices) supports full support.
    assert!(width > 0 && width <= keys.dims()[1]);
    Tensor::from_dispatch(
        <burn::backend::Dispatch as burn::backend::ops::IntTensorOps<burn::backend::Dispatch>>::int_argtopk(
            keys.into_dispatch(), 1, width,
        ),
    )
}
impl Candidates {
    pub fn select(
        probabilities: Tensor<2>,
        width: usize,
        backend: SoftSelectionBackend,
    ) -> Result<Self> {
        let [rows, vocab] = probabilities.dims();
        ensure!(
            rows > 0 && vocab > 0 && width > 0,
            "invalid candidate shape"
        );
        if backend.is_cpu() {
            return Self::select_cpu(probabilities, width);
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            let k = width.min(vocab);
            let keys = delta_kernels::soft_order_keys(probabilities.clone());
            // Integer ranges classify the original bits, including signed zero
            // and negative subnormals even on devices that flush float arithmetic.
            ensure!(
                keys.clone()
                    .greater_equal_elem(-1)
                    .bool_and(keys.clone().lower_equal_elem(0x7f7fffff_i32))
                    .all()
                    .into_scalar::<bool>(),
                "candidate probabilities must be finite and nonnegative"
            );
            let ids = if backend == SoftSelectionBackend::GpuTiled {
                const TILE: usize = 256;
                let tiles = vocab.div_ceil(TILE);
                let local_k = k.min(TILE);
                let padded = if vocab % TILE == 0 {
                    keys.reshape([rows * tiles, TILE])
                } else {
                    Tensor::<2, Int>::full([rows, tiles * TILE], i32::MIN, &probabilities.device())
                        .slice_assign([0..rows, 0..vocab], keys)
                        .reshape([rows * tiles, TILE])
                };
                let local_ids = top_ids(padded.clone(), local_k);
                let local_keys = padded
                    .gather(1, local_ids.clone())
                    .reshape([rows, tiles * local_k]);
                let offsets = Tensor::<1, Int>::arange(0..tiles as i64, &probabilities.device())
                    .reshape([1, tiles, 1])
                    * TILE as i32;
                let global_ids = (local_ids.reshape([rows, tiles, local_k]) + offsets)
                    .reshape([rows, tiles * local_k]);
                global_ids.gather(1, top_ids(local_keys, k))
            } else {
                top_ids(keys, k)
            };
            let weights = probabilities.gather(1, ids.clone());
            let retained_mass = weights.clone().sum_dim(1);
            let weights = weights / retained_mass.clone().mask_fill(retained_mass.clone().equal_elem(0.0), 1.0);
            Ok(Self {
                ids,
                weights,
                retained_mass,
            })
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        anyhow::bail!("GPU soft selection requires vulkan or cuda")
    }
    fn select_cpu(probabilities: Tensor<2>, width: usize) -> Result<Self> {
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
        let weights = weights / retained_mass.clone().mask_fill(retained_mass.clone().equal_elem(0.0), 1.0);
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
        let visibility = visibility.mask_fill(
            inactive.clone().bool_or(self.retained_mass.clone().equal_elem(0.0)), 0.0);
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
        let weighted = (selected * weights.unsqueeze_dim(2))
            .sum_dim(1)
            .reshape([rows, d]);
        let mask_weight = visibility.clone().neg().add_scalar(1.0);
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
    let mass = probabilities.clone().sum_dim(1);
    let visibility = visibility.mask_fill(mass.clone().equal_elem(0.0), 0.0);
    let probabilities = probabilities / mass.clone().mask_fill(mass.equal_elem(0.0), 1.0);
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
    let mask_weight = visibility.clone().neg().add_scalar(1.0);
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
pub(crate) fn ids_host(tensor: Tensor<2, Int>) -> Vec<i32> {
    tensor
        .into_data()
        .try_to_vec_as::<i32>()
        .expect("candidate int32 IDs")
}
fn proposals(tensor: Tensor<2, Int>) -> Vec<Option<usize>> {
    ids_host(tensor)
        .into_iter()
        .map(|id| usize::try_from(id).ok())
        .collect()
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
    /// Historical field name: now exactly learned lambda (one only for hard observations).
    pub visibility: Tensor<2>,
    /// Detached full-support statistic, not entropy of the truncated candidates.
    pub negative_entropy: Tensor<2>,
    pub prediction_available: Tensor<2, Bool>,
    pub observed: Vec<Option<usize>>,
    pub active: Vec<bool>,
    pub provisional: Tensor<2, Int>,
    pub retained: Tensor<2, Int>,
    mutable: Tensor<2, Bool>,
    active_device: Tensor<2>,
    backend: SoftSelectionBackend,
    vocab: usize,
}
impl SoftState {
    pub fn new(
        vocab: usize,
        width: usize,
        observed: Vec<Option<usize>>,
        active: Vec<bool>,
        device: &Device,
        backend: SoftSelectionBackend,
    ) -> Result<Self> {
        let rows = observed.len();
        ensure!(
            rows == active.len() && rows > 0 && vocab > 0,
            "invalid feedback shape"
        );
        let mut visibility = vec![0.0f32; rows];
        let mut observed_ids = vec![0i32; rows];
        for row in 0..rows {
            if active[row] {
                if let Some(id) = observed[row] {
                    ensure!(id < vocab, "observed token outside vocabulary");
                    observed_ids[row] = id as i32;
                    visibility[row] = 1.0;
                }
            }
        }
        let mutable = Tensor::<2>::from_data(
            TensorData::new(
                (0..rows)
                    .map(|r| {
                        if active[r] && observed[r].is_none() {
                            1.0f32
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>(),
                [rows, 1],
            ),
            device,
        )
        .greater_elem(0.5);
        let observed_values =
            Tensor::from_data(TensorData::new(visibility.clone(), [rows, 1]), device);
        let p = Tensor::full([rows, vocab], 1.0 / vocab as f32, device)
            .mask_fill(mutable.clone().bool_not().expand([rows, vocab]), 0.0)
            .scatter(
                1,
                Tensor::from_data(TensorData::new(observed_ids, [rows, 1]), device),
                observed_values,
                burn::tensor::IndexingUpdateOp::Add,
            );
        Ok(Self {
            candidates: Candidates::select(p, width, backend)?,
            visibility: Tensor::from_data(TensorData::new(visibility, [rows, 1]), device),
            negative_entropy: Tensor::zeros([rows, 1], device),
            prediction_available: Tensor::<2>::zeros([rows, 1], device).greater_elem(0.0),
            mutable,
            active_device: Tensor::from_data(
                TensorData::new(
                    active
                        .iter()
                        .map(|&v| if v { 1.0f32 } else { 0.0 })
                        .collect::<Vec<_>>(),
                    [rows, 1],
                ),
                device,
            ),
            backend,
            observed,
            active,
            provisional: Tensor::full([rows, 1], -1, device),
            retained: Tensor::full([rows, 1], -1, device),
            vocab,
        })
    }
    pub fn update(
        &mut self,
        logits: Tensor<2>,
        temperature: f64,
        width: usize,
        confidence: &SoftConfidence,
    ) -> Result<()> {
        let rows = self.observed.len();
        ensure!(logits.dims() == [rows, self.vocab], "feedback logits shape");
        ensure!(temperature.is_finite() && temperature > 0.0, "invalid feedback temperature");
        let logits = logits.detach();
        let device = logits.device();
        let (best, available) = if self.backend.is_cpu() {
            let raw = host(logits.clone());
            let mut ids = vec![-1i32; rows];
            for row in 0..rows {
                let values = &raw[row * self.vocab..(row + 1) * self.vocab];
                ensure!(
                    values
                        .iter()
                        .all(|v| v.is_finite() || *v == f32::NEG_INFINITY),
                    "nonfinite feedback logits"
                );
                if self.active[row]
                    && self.observed[row].is_none()
                    && !values.iter().all(|v| *v == f32::NEG_INFINITY)
                {
                    ids[row] = (0..self.vocab)
                        .max_by(|&a, &b| values[a].total_cmp(&values[b]).then(b.cmp(&a)))
                        .unwrap() as i32;
                }
            }
            let best = Tensor::<2, Int>::from_data(TensorData::new(ids, [rows, 1]), &device);
            (best.clone(), best.greater_equal_elem(0))
        } else {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                let keys = delta_kernels::soft_order_keys(logits.clone());
                ensure!(
                    keys.clone()
                        .greater_equal_elem(-2139095041_i32)
                        .bool_and(keys.clone().lower_equal_elem(0x7f7fffff_i32))
                        .all()
                        .into_scalar::<bool>(),
                    "nonfinite feedback logits"
                );
                let available = keys
                    .clone()
                    .max_dim(1)
                    .not_equal_elem(-2139095041_i32)
                    .bool_and(self.mutable.clone());
                let best = top_ids(keys, 1);
                (best, available)
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            anyhow::bail!("GPU soft update requires vulkan or cuda");
        };
        self.provisional = self.provisional.clone().mask_where(available.clone(), best);
        // Sanitize unavailable rows before softmax, then replace only mutable predictions.
        let prediction = softmax(
            logits.mask_fill(available.clone().bool_not().expand([rows, self.vocab]), 0.0)
                / temperature,
            1,
        );
        let entropy = (prediction.clone() * (prediction.clone() + 1e-10).log()).sum_dim(1);
        self.negative_entropy = self.negative_entropy.clone().mask_where(available.clone(), entropy).detach();
        self.prediction_available = self.prediction_available.clone().bool_or(available.clone());
        let selected = Candidates::select(prediction, width, self.backend)?;
        let [_, k] = selected.ids.dims();
        ensure!(self.candidates.ids.dims()[1] == k, "feedback candidate width changed");
        self.candidates.ids = self.candidates.ids.clone().mask_where(available.clone().expand([rows,k]), selected.ids);
        self.candidates.weights = self.candidates.weights.clone().mask_where(available.clone().expand([rows,k]), selected.weights).detach();
        self.candidates.retained_mass = self.candidates.retained_mass.clone().mask_where(available, selected.retained_mass).detach();
        self.refresh_feedback(confidence);
        Ok(())
    }
    /// Rebuild the trainable coefficient after restoring detached persistent feedback.
    pub fn refresh_feedback(&mut self, confidence: &SoftConfidence) {
        let observed = self.mutable.clone().bool_not()
            .bool_and(self.active_device.clone().greater_elem(0.5));
        self.visibility = confidence.weight(self.negative_entropy.clone())
            .mask_fill(self.prediction_available.clone().bool_not(), 0.0)
            .mask_fill(observed, 1.0)
            .mask_fill(self.active_device.clone().lower_elem(0.5), 0.0);
    }
    pub fn retain(&mut self, positions: &[usize]) {
        let mask = self
            .positions(positions)
            .bool_and(self.mutable.clone())
            .bool_and(self.provisional.clone().greater_equal_elem(0));
        self.retained = self
            .retained
            .clone()
            .mask_where(mask, self.provisional.clone());
    }
    fn positions(&self, positions: &[usize]) -> Tensor<2, Bool> {
        let rows = self.observed.len();
        let device = self.visibility.device();
        let selected = Tensor::<2, Int>::zeros([rows, 1], &device);
        if positions.is_empty() {
            return selected.greater_elem(0);
        }
        let ids = Tensor::from_data(
            TensorData::new(
                positions
                    .iter()
                    .map(|&r| {
                        assert!(r < rows);
                        r as i32
                    })
                    .collect::<Vec<_>>(),
                [positions.len(), 1],
            ),
            &device,
        );
        selected
            .scatter(
                0,
                ids,
                Tensor::ones([positions.len(), 1], &device),
                burn::tensor::IndexingUpdateOp::Add,
            )
            .greater_elem(0)
    }
    pub fn embeddings(&self, table: Tensor<2>, mask: Tensor<1>, hard: bool) -> Tensor<2> {
        self.candidates.expected_masked(
            table,
            mask,
            self.visibility.clone(),
            self.active_device.clone(),
            hard,
        )
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
    let full = Candidates::select(probabilities.clone(), vocab, c.soft_selection_backend)?;
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
    let truncated_dense = Candidates::select(probabilities.clone(), 3, c.soft_selection_backend)?;
    let truncated_sparse = Candidates::select(probabilities.clone(), 3, c.soft_selection_backend)?;
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
    let zero = Candidates::select(
        Tensor::zeros([1, vocab], &device),
        3,
        c.soft_selection_backend,
    )?;
    ensure!(
        difference(
            zero.expected(table.clone(), mask.clone(), rho, false),
            mask.clone().reshape([1, d])
        ) < 1e-7,
        "zero mass must produce mask embedding"
    );
    let tied = Candidates::select(
        Tensor::ones([1, vocab], &device) / (vocab as f64),
        3,
        c.soft_selection_backend,
    )?;
    ensure!(
        tied.ids
            .into_data()
            .try_to_vec_as::<i32>()
            .map_err(|e| anyhow::anyhow!("tie IDs: {e:?}"))?
            == vec![0, 1, 2],
        "ties must select ascending token ID"
    );
    let confidence = SoftConfidence::from_raw([0.2,-0.3,0.4], &device);
    let confidence_checks = confidence_reference(&device)?;
    let mut state = SoftState::new(vocab, c.soft_candidate_width,
        vec![Some(2), None, None, None], vec![true, true, true, false],
        &device, c.soft_selection_backend)?;
    let initial = host(state.embeddings(table.clone(), mask.clone(), false));
    ensure!(initial[d..3*d] == host(mask.clone()).repeat(2), "initial unresolved input is not binary");
    let observed_before = host(state.candidates.dense(vocab))[..vocab].to_vec();
    let mut rounds = Vec::new();
    let mut update_error = 0.0f64;
    for step in 0..4 {
        let lv = (0..4*vocab).map(|i| ((i + step*3) % vocab) as f32 / 4.0).collect::<Vec<_>>();
        let logits = Tensor::from_data(TensorData::new(lv.clone(), [4,vocab]), &device).require_grad();
        state.update(logits.clone(), c.soft_temperature, c.soft_candidate_width, &confidence)?;
        let actual_entropy = host(state.negative_entropy.clone());
        let actual_weights = host(state.candidates.weights.clone());
        let actual_mass = host(state.candidates.retained_mass.clone());
        let actual_ids = ids_host(state.candidates.ids.clone());
        let k = c.soft_candidate_width.min(vocab);
        for row in [1,2] {
            let logits = lv[row*vocab..(row+1)*vocab].iter().map(|&x| f64::from(x)/c.soft_temperature).collect::<Vec<_>>();
            let maximum = logits.iter().copied().fold(f64::NEG_INFINITY,f64::max);
            let total = logits.iter().map(|&x|(x-maximum).exp()).sum::<f64>();
            let p = logits.iter().map(|&x|(x-maximum).exp()/total).collect::<Vec<_>>();
            let entropy = p.iter().map(|&x|x*(x+1e-10).ln()).sum::<f64>();
            ensure!((f64::from(actual_entropy[row])-entropy).abs()<2e-6, "full-support entropy reference");
            let mass = actual_ids[row*k..(row+1)*k].iter().map(|&id|p[id as usize]).sum::<f64>();
            ensure!((f64::from(actual_mass[row])-mass).abs()<2e-6, "raw retained mass reference");
            for j in 0..k {
                let q = p[actual_ids[row*k+j] as usize]/mass;
                ensure!((f64::from(actual_weights[row*k+j])-q).abs()<2e-6, "normalized top-k reference");
            }
        }
        let mut fresh = SoftState::new(vocab, c.soft_candidate_width,
            state.observed.clone(), state.active.clone(), &device, c.soft_selection_backend)?;
        fresh.update(logits.clone(), c.soft_temperature, c.soft_candidate_width, &confidence)?;
        update_error = update_error.max(difference(state.candidates.weights.clone(), fresh.candidates.weights.clone()));
        ensure!(ids_host(state.candidates.ids.clone()) == ids_host(fresh.candidates.ids.clone()),
            "candidate identities depend on earlier predictions");
        ensure!(host(state.candidates.dense(vocab))[..vocab] == observed_before,
            "teacher changed permanent observations");
        ensure!(host(state.visibility.clone())[0] == 1.0 && host(state.visibility.clone())[3] == 0.0,
            "observed/padding coefficient changed");
        state.retain(&[1,2]);
        let gradients = state.embeddings(table.clone(), mask.clone(), false).sum().backward();
        ensure!(logits.grad(&gradients).is_none(), "teacher prediction received a student gradient");
        ensure!(confidence.raw_scale.val().grad(&gradients).is_some(), "confidence scale is disconnected");
        ensure!(table.grad(&gradients).is_some(), "embedding table is disconnected");
        rounds.push(json!({"step":step,"ids":ids_host(state.candidates.ids.clone()),
            "weights":host(state.candidates.weights.clone()),"retained_mass":host(state.candidates.retained_mass.clone()),
            "lambda":host(state.visibility.clone()),"provisional":proposals(state.provisional.clone()),
            "retained":proposals(state.retained.clone())}));
    }
    ensure!(update_error < 2e-6, "latest-only feedback mismatch");
    let before = host(state.candidates.weights.clone());
    let weight_before = host(state.visibility.clone());
    state.update(Tensor::full([4,vocab], f32::NEG_INFINITY, &device),
        c.soft_temperature, c.soft_candidate_width, &confidence)?;
    ensure!(host(state.candidates.weights.clone()) == before && host(state.visibility.clone()) == weight_before,
        "unavailable predictions changed retained feedback");
    let endpoint_checks = endpoint_poison_checks(&device)?;
    Ok(json!({"mechanism":"learned_entropy_latest_normalized_topk","not_sbd":true,
        "candidate_width":c.soft_candidate_width,"hard_mask_only":c.hard_mask_only,
        "expectation_max_error":expectation_error,"weight_gradient_max_error":weight_gradient_error,
        "embedding_gradient_max_error":embedding_gradient_error,"truncated_expectation_max_error":truncated_expectation_error,
        "truncated_weight_gradient_max_error":truncated_weight_gradient_error,
        "truncated_embedding_gradient_max_error":truncated_embedding_gradient_error,
        "endpoint_checks":endpoint_checks,"confidence_checks":confidence_checks,
        "latest_only_max_error":update_error,"rounds":rounds}))
}

/// Independent scalar Eq. (3) and raw-parameter derivatives, shared by CPU/GPU acceptance.
fn confidence_reference(device: &Device) -> Result<Value> {
    let initialized = SoftConfidence::new(device);
    let initial = host(initialized.weight(Tensor::from_data([[0.0f32],[-10.0]], device)));
    ensure!(initial[0] > 0.0 && initial[0] < 1.01e-6 && initial[1] < initial[0],
        "official near-zero confidence initialization changed");
    let raw = [0.2f64, -0.3, 0.4];
    let head = SoftConfidence::from_raw(raw, device);
    let entropy = [0.0f64, -0.75, -8.0];
    let sigmoid = |x:f64| 1.0/(1.0+(-x).exp());
    let scale = sigmoid(raw[0]);
    let centre = -(1.0+raw[1].exp()).ln()-1e-6;
    let steep = (1.0+raw[2].exp()).ln()+1e-6;
    let mut expected = Vec::new();
    let mut derivatives = [0.0f64;3];
    for h in entropy {
        let s = sigmoid(steep*(h-centre));
        expected.push(scale*s);
        derivatives[0] += scale*(1.0-scale)*s;
        derivatives[1] += scale*s*(1.0-s)*steep*sigmoid(raw[1]);
        derivatives[2] += scale*s*(1.0-s)*(h-centre)*sigmoid(raw[2]);
    }
    let input = Tensor::from_data([[0.0f32],[-0.75],[-8.0]], device).require_grad();
    let actual = head.weight(input.clone());
    let observed = host(actual.clone());
    for (&a,&e) in observed.iter().zip(&expected) {
        ensure!((f64::from(a)-e).abs()<2e-6, "scalar confidence mismatch");
    }
    ensure!(observed[0] > observed[1] && observed[1] > observed[2]
        && observed.iter().all(|&x| x>=0.0 && f64::from(x)<scale), "confidence constraints/entropy ordering");
    let gradients = actual.sum().backward();
    ensure!(input.grad(&gradients).is_none(), "entropy is not detached");
    for (parameter, expected) in [&head.raw_scale,&head.raw_centre_neg,&head.raw_steep].into_iter().zip(derivatives) {
        let actual = f64::from(host(parameter.val().grad(&gradients).context("confidence gradient")?)[0]);
        ensure!((actual-expected).abs()<2e-6, "scalar confidence derivative mismatch");
    }
    Ok(json!({"official_initial_lambda":initial,"entropy":entropy,"lambda":observed,
        "independent_raw_parameter_derivatives":derivatives,"teacher_entropy_detached":true}))
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
