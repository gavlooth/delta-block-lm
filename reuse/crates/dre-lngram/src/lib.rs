//! Latent n-gram lookup memory (Lngram, single table), spec §3.3.
//!
//! Hidden states are discretized into `routes` symbols of `bits_per_route` bits each. For every n-gram order the
//! suffix of route symbols ending at a position addresses one exact row of that order's table (no hashing). The
//! rows are read out through shared key/value projections with context-aware sigmoid gates, then optionally
//! smoothed by a depthwise 3-tap convolution that looks both ways inside the document.
//!
//! Equations, the counterfactual surrogate gradient of the discretizer and every adaptation of the paper are
//! documented in `docs/references/lngram.md`.

mod statistics;

use burn::module::{Initializer, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, silu};
use dre_core::{BatchLayout, DocumentLayout};

pub use statistics::{CodeStatistics, HostRouteCodes, OrderUtilization};

/// How the discretizer receives gradients through the hard addressing path (paper §2.5, App. A).
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum SurrogateGradient {
    /// One-bit counterfactual (App. A.3, `Approximate_gradient_backpropagation.py`):
    /// `∂L/∂z_j = λ τ p_j (1 − p_j) ⟨g, E_j⁽¹⁾ − E_j⁽⁰⁾⟩`, the two rows reached by forcing bit `j` to 1 or 0.
    OneBit,
    /// Local expectation over all `K` symbols of one n-gram slot (App. A.2, `Single_table_Lngram.py`):
    /// `∂L/∂z_j = λ τ Σ_c P(c|z) (β_j(c) − p_j) ⟨g, E_c⟩`.
    LocalExpectation,
}

/// Configuration of one Lngram module. Defaults are the spec §3.3 / §13 values.
#[derive(Config, Debug)]
pub struct LngramConfig {
    #[config(default = "dre_core::Precision::Fp32")]
    pub precision: dre_core::Precision,
    /// Width of the input and output features.
    #[config(default = 512)]
    pub d_model: usize,
    /// Number of discrete routes `R`.
    #[config(default = 64)]
    pub routes: usize,
    /// Bits per route `m`; each route symbol takes `K = 2^m` values.
    #[config(default = 4)]
    pub bits_per_route: usize,
    /// Width `d_m` of one table row.
    #[config(default = 8)]
    pub memory_dim: usize,
    /// N-gram orders, one table each.
    #[config(default = "vec![2, 3]")]
    pub orders: Vec<usize>,
    /// Depthwise 3-tap bidirectional smoothing of the readout (`Y = V + SiLU(conv(RMSNorm(V)))`).
    #[config(default = true)]
    pub smoothing: bool,
    /// Distance between smoothing taps; `None` uses the largest order, as the paper's dilation.
    #[config(default = "None")]
    pub smoothing_dilation: Option<usize>,
    /// Discretizer gradient rule.
    #[config(default = "SurrogateGradient::OneBit")]
    pub surrogate: SurrogateGradient,
    /// Temperature `τ` of the bit probabilities `p = σ(τ z)`.
    #[config(default = 1.0)]
    pub surrogate_temperature: f64,
    /// Global scale `λ` of the surrogate gradient.
    #[config(default = 1.0)]
    pub surrogate_scale: f64,
    /// Standard deviation of the discretizer and key/value projection weights (biases start at zero).
    #[config(default = 0.02)]
    pub init_std: f64,
    /// Initialization of the n-gram tables (the official implementation starts from zeros).
    #[config(default = "Initializer::Zeros")]
    pub table_initializer: Initializer,
    /// Epsilon of every RMSNorm in the module.
    #[config(default = 1e-6)]
    pub norm_epsilon: f64,
}

impl LngramConfig {
    /// Symbols per route, `K = 2^bits_per_route`.
    pub fn alphabet(&self) -> usize {
        1 << self.bits_per_route
    }

    /// Rows of the table for `order`: `R · K^order`.
    pub fn table_rows(&self, order: usize) -> usize {
        self.routes * self.alphabet().pow(order as u32)
    }

    pub fn init(&self, device: &Device) -> Lngram {
        assert!(!self.orders.is_empty(), "Lngram needs at least one n-gram order");
        let mut sorted = self.orders.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert!(sorted.len() == self.orders.len(), "n-gram orders must be distinct: {:?}", self.orders);
        assert!(sorted[0] >= 1, "n-gram orders start at 1: {:?}", self.orders);
        assert!((1..=16).contains(&self.bits_per_route), "bits_per_route must be in 1..=16");
        for &order in &self.orders {
            let rows = self.table_rows(order);
            assert!(rows <= i32::MAX as usize, "table for order {order} has {rows} rows, more than i32 addresses");
        }
        let weights = Initializer::Normal { mean: 0.0, std: self.init_std };
        let norm = || RmsNormConfig::new(self.d_model).with_epsilon(self.norm_epsilon).init(device);
        let projection = || {
            let mut linear = LinearConfig::new(self.routes * self.memory_dim, self.d_model)
                .with_initializer(weights.clone())
                .init(device);
            linear.bias = Some(Initializer::Zeros.init([self.d_model], device));
            linear
        };
        let smoothing = self.smoothing.then(|| Smoothing {
            norm: norm(),
            weight: Initializer::Zeros.init([3, self.d_model], device),
            dilation: self.smoothing_dilation.unwrap_or(sorted[sorted.len() - 1]),
        });
        Lngram {
            mixed: self.precision == dre_core::Precision::Bf16Mixed,
            discretizer_norm: norm(),
            discretizer: LinearConfig::new(self.d_model, self.routes * self.bits_per_route)
                .with_bias(false)
                .with_initializer(weights.clone())
                .init(device),
            tables: self
                .orders
                .iter()
                .map(|&order| {
                    EmbeddingConfig::new(self.table_rows(order), self.memory_dim)
                        .with_initializer(self.table_initializer.clone())
                        .init(device)
                })
                .collect(),
            key: projection(),
            value: projection(),
            query_norm: norm(),
            key_norm: norm(),
            smoothing,
            orders: self.orders.clone(),
            routes: self.routes,
            bits_per_route: self.bits_per_route,
            memory_dim: self.memory_dim,
            surrogate: self.surrogate,
            surrogate_temperature: self.surrogate_temperature,
            surrogate_scale: self.surrogate_scale,
        }
    }
}

/// One Lngram module: discretizer, one exact table per order, context-aware readout, optional smoothing.
#[derive(Module, Debug)]
pub struct Lngram {
    mixed: bool,
    /// RMSNorm in front of the discretizer (paper: `U = RMSNorm(H)`).
    pub discretizer_norm: RmsNorm,
    /// `W_q: [d_model, routes · bits_per_route]`, no bias.
    pub discretizer: Linear,
    /// One table per entry of `orders`, `[routes · K^n, memory_dim]`, row `g = r·K^n + Σ_i a_{t−n+1+i,r} K^i`.
    pub tables: Vec<Embedding>,
    /// `W_K, b_K: routes · memory_dim → d_model`, shared across orders.
    pub key: Linear,
    /// `W_V, b_V: routes · memory_dim → d_model`, shared across orders.
    pub value: Linear,
    /// RMSNorm of the gate query (the module input).
    pub query_norm: RmsNorm,
    /// RMSNorm of the per-order keys.
    pub key_norm: RmsNorm,
    pub smoothing: Option<Smoothing>,
    pub orders: Vec<usize>,
    pub routes: usize,
    pub bits_per_route: usize,
    pub memory_dim: usize,
    #[module(skip)]
    pub surrogate: SurrogateGradient,
    pub surrogate_temperature: f64,
    pub surrogate_scale: f64,
}

/// Depthwise 3-tap convolution over `RMSNorm(V)` with taps at `t − dilation`, `t`, `t + dilation`; taps outside
/// the position's document read zero.
#[derive(Module, Debug)]
pub struct Smoothing {
    pub norm: RmsNorm,
    /// `[3, d_model]`: rows are the taps at `t − dilation`, `t`, `t + dilation`. Starts at zero.
    pub weight: Param<Tensor<2>>,
    pub dilation: usize,
}

/// Discrete route codes of a batch, as produced by the forward pass.
#[derive(Debug, Clone)]
pub struct RouteCodes {
    /// Route symbols `a_{t,r} ∈ [0, K)`, `[B, T, R]`.
    pub symbols: Tensor<3, Int>,
    /// Per order (same order as [`Lngram::orders`]): global table row `g^{(n)}_{t,r}`, or `-1` where the position
    /// has no full n-gram inside its document (prefix positions and padding), `[B, T, R]`.
    pub addresses: Vec<Tensor<3, Int>>,
}

impl RouteCodes {
    /// Copies the codes to the host for statistics.
    pub fn to_host(&self, layout: &DocumentLayout, alphabet: usize, orders: &[usize]) -> HostRouteCodes {
        let [batch, tokens, routes] = self.symbols.dims();
        let (starts, ends) = layout.limits_host();
        let read = |tensor: &Tensor<3, Int>| tensor.to_data().iter::<i64>().collect::<Vec<_>>();
        HostRouteCodes {
            batch,
            tokens,
            routes,
            alphabet,
            orders: orders.to_vec(),
            valid: starts.iter().zip(&ends).map(|(start, end)| end >= start).collect(),
            symbols: read(&self.symbols),
            addresses: self.addresses.iter().map(read).collect(),
        }
    }
}

impl Lngram {
    /// Symbols per route, `K`.
    pub fn alphabet(&self) -> usize {
        1 << self.bits_per_route
    }

    /// `[B, T, d] -> [B, T, d]`. Lookups never cross document limits, padding never reaches valid positions and
    /// padded outputs are exactly zero.
    pub fn forward(&self, u: Tensor<3>, layout: &BatchLayout) -> Tensor<3> {
        let [batch, tokens, d_model] = u.dims();
        let padding=layout.valid.clone().equal_elem(0.0).expand([batch,tokens,d_model]);
        let u=u.mask_fill(padding.clone(),0.0);
        let logits = self.discretizer_logits(u.clone());
        let bits = self.bits(logits.clone());
        let symbols = self.symbols(bits.clone());
        // Bit probabilities only exist to carry the surrogate gradient; inference skips them.
        let probabilities = logits.is_tracked().then(|| {
            sigmoid(logits.reshape([batch, tokens, self.routes, self.bits_per_route]).mul_scalar(self.surrogate_temperature))
        });

        let mut memories = Vec::with_capacity(self.orders.len());
        for (table, &order) in self.tables.iter().zip(&self.orders) {
            let rows = self.alphabet().pow(order as u32);
            let local = self.local_addresses(&symbols, order);
            let table = table.weight.val();
            let missing=full_ngram(layout,order).bool_not().reshape([batch,tokens,1,1]);
            let memory = lookup(table.clone(), local.clone().reshape([batch * tokens, self.routes]), rows)
                .reshape([batch, tokens, self.routes, self.memory_dim])
                .mask_fill(missing.clone().expand([batch,tokens,self.routes,self.memory_dim]),0.0);
            let memory = match &probabilities {
                Some(probabilities) => {
                    let surrogate = match self.surrogate {
                        SurrogateGradient::OneBit => self.one_bit_term(
                            table.detach(),
                            &local,
                            &bits,
                            probabilities,
                            memory.clone().detach(),
                            order,
                            &missing,
                        ),
                        SurrogateGradient::LocalExpectation => {
                            self.local_expectation_term(table.detach(), &local, &symbols, probabilities, order, &missing)
                        }
                    };
                    // Forward value unchanged; the gradient of `surrogate` w.r.t. the logits is the surrogate.
                    memory + (surrogate.clone() - surrogate.detach()).mul_scalar(self.surrogate_scale)
                }
                None => memory,
            };
            memories.push(memory);
        }
        let memory: Tensor<4> = Tensor::stack::<5>(memories, 2).reshape([
            batch,
            tokens,
            self.orders.len(),
            self.routes * self.memory_dim,
        ]);

        let key = self.key_norm.forward(dre_core::project(&self.key, memory.clone(), self.mixed));
        let value = dre_core::project(&self.value, memory, self.mixed);
        let query = self.query_norm.forward(u).unsqueeze_dim::<4>(2);
        let gates = sigmoid((query * key).sum_dim(3).div_scalar((d_model as f64).sqrt()));
        let readout: Tensor<3> = (gates * value).sum_dim(2).reshape([batch, tokens, d_model]);
        let output = match &self.smoothing {
            Some(smoothing) => readout.clone() + smoothing.forward(readout, layout),
            None => readout,
        };
        output.mask_fill(padding,0.0)
    }

    /// Discretizer logits `z = RMSNorm(u) W_q`, `[B, T, routes · bits_per_route]`; bit `j` of route `r` is
    /// channel `r · bits_per_route + j`.
    pub fn discretizer_logits(&self, u: Tensor<3>) -> Tensor<3> {
        self.discretizer.forward(self.discretizer_norm.forward(u))
    }

    /// Route symbols and per-order table rows of a batch (no gradient).
    pub fn route_codes(&self, u: Tensor<3>, layout: &BatchLayout) -> RouteCodes {
        let [batch, tokens, d_model] = u.dims();
        let u=u.mask_fill(layout.valid.clone().equal_elem(0.0).expand([batch,tokens,d_model]),0.0);
        let device = u.device();
        let symbols = self.symbols(self.bits(self.discretizer_logits(u.detach())));
        let addresses = self
            .orders
            .iter()
            .map(|&order| {
                let rows = self.alphabet().pow(order as u32) as i64;
                let route_offsets = Tensor::<1, Int>::arange(0..self.routes as i64, &device)
                    .mul_scalar(rows)
                    .reshape([1, 1, self.routes]);
                let global = self.local_addresses(&symbols, order) + route_offsets;
                let missing = full_ngram(layout, order)
                    .bool_not()
                    .unsqueeze_dim::<3>(2)
                    .expand([batch, tokens, self.routes]);
                global.mask_fill(missing, -1)
            })
            .collect();
        RouteCodes { symbols, addresses }
    }

    /// Hard bits `b = 1[z > 0]`, `[B, T, R, m]`.
    fn bits(&self, logits: Tensor<3>) -> Tensor<4, Int> {
        let [batch, tokens, _] = logits.dims();
        logits.greater_elem(0.0).int().reshape([batch, tokens, self.routes, self.bits_per_route])
    }

    /// `a_{t,r} = Σ_j 2^j b_{t,r,j}`, `[B, T, R]`.
    fn symbols(&self, bits: Tensor<4, Int>) -> Tensor<3, Int> {
        let [batch, tokens, routes, width] = bits.dims();
        let weights = Tensor::<1, Int>::from_data(
            TensorData::new((0..width).map(|bit| 1i64 << bit).collect::<Vec<_>>(), [width]),
            &bits.device(),
        )
        .reshape([1, 1, 1, width]);
        (bits * weights).sum_dim(3).reshape([batch, tokens, routes])
    }

    /// Address inside the route's block, `Σ_i a_{t−n+1+i,r} K^i`, `[B, T, R]`. Positions before `n − 1` read
    /// symbol 0 for the missing slots; [`full_ngram`] masks them.
    fn local_addresses(&self, symbols: &Tensor<3, Int>, order: usize) -> Tensor<3, Int> {
        let alphabet = self.alphabet() as i64;
        (0..order)
            .map(|slot| shift_later_int(symbols.clone(), order - 1 - slot).mul_scalar(alphabet.pow(slot as u32)))
            .reduce(|sum, term| sum + term)
            .expect("order >= 1")
    }

    /// One-bit counterfactual term `X = Σ_slot Σ_j p_{slot,j} (E_j⁽¹⁾ − E_j⁽⁰⁾)` with detached rows, `[B, T, R, d_m]`.
    /// `∂⟨g, X⟩/∂z_j = τ p_j (1 − p_j) ⟨g, E_j⁽¹⁾ − E_j⁽⁰⁾⟩`, the App. A.3 surrogate before the scale `λ`.
    fn one_bit_term(
        &self,
        table: Tensor<2>,
        local: &Tensor<3, Int>,
        bits: &Tensor<4, Int>,
        probabilities: &Tensor<4>,
        current: Tensor<4>,
        order: usize,
        missing:&Tensor<4,Bool>,
    ) -> Tensor<4> {
        let [batch, tokens, routes, width] = bits.dims();
        let rows = self.alphabet().pow(order as u32);
        let alphabet = self.alphabet() as i64;
        let bit_values = Tensor::<1, Int>::from_data(
            TensorData::new((0..width).map(|bit| 1i64 << bit).collect::<Vec<_>>(), [width]),
            &bits.device(),
        )
        .reshape([1, 1, 1, width]);
        (0..order)
            .map(|slot| {
                let lag = order - 1 - slot;
                let slot_bits = shift_later_int(bits.clone(), lag);
                let slot_probabilities = shift_later_float(probabilities.clone(), lag);
                // +1 where the bit is set, −1 where it is clear.
                let sign = slot_bits.mul_scalar(2).sub_scalar(1);
                let flipped = local.clone().unsqueeze_dim::<4>(3)
                    - sign.clone() * bit_values.clone().mul_scalar(alphabet.pow(slot as u32));
                let flipped_rows = lookup(
                    table.clone(),
                    flipped.swap_dims(2, 3).reshape([batch * tokens * width, routes]),
                    rows,
                )
                .reshape([batch, tokens, width, routes, self.memory_dim])
                .swap_dims(2, 3)
                .mask_fill(missing.clone().unsqueeze_dim::<5>(4).expand([batch,tokens,routes,width,self.memory_dim]),0.0);
                // E⁽¹⁾ − E⁽⁰⁾ for every bit.
                let difference =
                    (current.clone().unsqueeze_dim::<5>(3) - flipped_rows) * sign.float().unsqueeze_dim::<5>(4);
                (slot_probabilities.unsqueeze_dim::<5>(4) * difference)
                    .sum_dim(3)
                    .reshape([batch, tokens, routes, self.memory_dim])
            })
            .reduce(|sum, term| sum + term)
            .expect("order >= 1")
    }

    /// Local expectation term `X = Σ_slot Σ_c P(c | z_slot) E_c` with detached rows, `[B, T, R, d_m]`.
    /// `∂⟨g, X⟩/∂z_j = τ Σ_c P(c|z) (β_j(c) − p_j) ⟨g, E_c⟩`, the App. A.2 surrogate before the scale `λ`.
    fn local_expectation_term(
        &self,
        table: Tensor<2>,
        local: &Tensor<3, Int>,
        symbols: &Tensor<3, Int>,
        probabilities: &Tensor<4>,
        order: usize,
        missing:&Tensor<4,Bool>,
    ) -> Tensor<4> {
        let [batch, tokens, routes] = symbols.dims();
        let device = symbols.device();
        let alphabet = self.alphabet();
        let width = self.bits_per_route;
        let rows = alphabet.pow(order as u32);
        // β_j(c) as floats, `[K, m]`.
        let code_bits = Tensor::<2>::from_data(
            TensorData::new(
                (0..alphabet).flat_map(|code| (0..width).map(move |bit| ((code >> bit) & 1) as f32)).collect(),
                [alphabet, width],
            ),
            &device,
        );
        let codes = Tensor::<1, Int>::arange(0..alphabet as i64, &device).reshape([1, 1, 1, alphabet]);
        (0..order)
            .map(|slot| {
                let lag = order - 1 - slot;
                let stride = (alphabet as i64).pow(slot as u32);
                let slot_probabilities = shift_later_float(probabilities.clone(), lag);
                let base = local.clone() - shift_later_int(symbols.clone(), lag).mul_scalar(stride);
                let candidates = base.unsqueeze_dim::<4>(3) + codes.clone().mul_scalar(stride);
                let entries = lookup(
                    table.clone(),
                    candidates.swap_dims(2, 3).reshape([batch * tokens * alphabet, routes]),
                    rows,
                )
                .reshape([batch, tokens, alphabet, routes, self.memory_dim])
                .swap_dims(2, 3)
                .mask_fill(missing.clone().unsqueeze_dim::<5>(4).expand([batch,tokens,routes,alphabet,self.memory_dim]),0.0);
                // P(c | z) = Π_j (1 − p_j) + β_j(c) (2 p_j − 1), `[B, T, R, K]`.
                let probability = (0..width)
                    .map(|bit| {
                        let p = slot_probabilities.clone().slice_dim(3, bit..bit + 1);
                        let beta = code_bits.clone().slice_dim(1, bit..bit + 1).reshape([1, 1, 1, alphabet]);
                        p.clone().neg().add_scalar(1.0) + beta * p.mul_scalar(2.0).sub_scalar(1.0)
                    })
                    .reduce(|product, factor| product * factor)
                    .expect("bits_per_route >= 1");
                (probability.unsqueeze_dim::<5>(4) * entries)
                    .sum_dim(3)
                    .reshape([batch, tokens, routes, self.memory_dim])
            })
            .reduce(|sum, term| sum + term)
            .expect("order >= 1")
    }
}

impl Smoothing {
    /// `SiLU(conv(RMSNorm(V)))`, `[B, T, d]`.
    pub fn forward(&self, readout: Tensor<3>, layout: &BatchLayout) -> Tensor<3> {
        let [batch, tokens, d_model] = readout.dims();
        let device = readout.device();
        let padding=layout.valid.clone().equal_elem(0.0).expand([batch,tokens,d_model]);
        let normed = self.norm.forward(readout.mask_fill(padding.clone(),0.0));
        let weight = self.weight.val();
        let tap = |row: usize| weight.clone().slice_dim(0, row..row + 1).reshape([1, 1, d_model]);
        let positions = positions(batch, tokens, &device);
        let outside_document = |offset: i64| {
            let source = positions.clone().add_scalar(offset);
            source
                .clone()
                .greater_equal(layout.doc_start.clone())
                .bool_and(source.lower_equal(layout.doc_end.clone()))
                .bool_not()
                .unsqueeze_dim::<3>(2).expand([batch,tokens,d_model])
        };
        let dilation = self.dilation;
        let before = shift_later_float(normed.clone(), dilation).mask_fill(outside_document(-(dilation as i64)),0.0) * tap(0);
        let after = shift_earlier_float(normed.clone(), dilation).mask_fill(outside_document(dilation as i64),0.0) * tap(2);
        silu(before + normed * tap(1) + after).mask_fill(padding,0.0)
    }
}

/// `true` where position `t` ends a full n-gram inside its document: `t − doc_start ≥ n − 1` and `t` is not
/// padding, `[B, T]`.
pub fn full_ngram(layout: &BatchLayout, order: usize) -> Tensor<2, Bool> {
    let [batch, tokens] = layout.doc_start.dims();
    let positions = positions(batch, tokens, &layout.doc_start.device());
    (positions.clone() - layout.doc_start.clone())
        .greater_equal_elem(order as i64 - 1)
        .bool_and(positions.lower_equal(layout.doc_end.clone()))
}

/// Position index of every token, `[B, T]`.
fn positions(batch: usize, tokens: usize, device: &Device) -> Tensor<2, Int> {
    Tensor::<1, Int>::arange(0..tokens as i64, device)
        .reshape([1, tokens])
        .expand([batch, tokens])
}

/// Rows `table[r · rows + local[p, r]]` for local addresses `[P, R]` into a `[R · rows, d_m]` table, `[P, R, d_m]`.
/// Gathering per route (axis 1 of `[R, rows, d_m]`) keeps the backward scatter parallel over routes and row width.
fn lookup(table: Tensor<2>, local: Tensor<2, Int>, rows: usize) -> Tensor<3> {
    let [_, memory_dim] = table.dims();
    let [count, routes] = local.dims();
    let index = local.swap_dims(0, 1).unsqueeze_dim::<3>(2).expand([routes, count, memory_dim]);
    table.reshape([routes, rows, memory_dim]).gather(1, index).swap_dims(0, 1)
}

/// `y[t] = x[t − steps]` along axis 1 for integer codes, zero for `t < steps`.
fn shift_later_int<const D: usize>(x: Tensor<D, Int>, steps: usize) -> Tensor<D, Int> {
    let mut dims = x.dims();
    let tokens = dims[1];
    if steps == 0 {
        return x;
    }
    dims[1] = steps.min(tokens);
    let zeros = Tensor::<D, Int>::zeros(dims, &x.device());
    if steps >= tokens {
        return zeros;
    }
    Tensor::cat(vec![zeros, x.slice_dim(1, 0..tokens - steps)], 1)
}

/// `y[t] = x[t − steps]` along axis 1, zero for `t < steps`.
fn shift_later_float<const D: usize>(x: Tensor<D>, steps: usize) -> Tensor<D> {
    let mut dims = x.dims();
    let tokens = dims[1];
    if steps == 0 {
        return x;
    }
    dims[1] = steps.min(tokens);
    let zeros = Tensor::<D>::zeros(dims, &x.device());
    if steps >= tokens {
        return zeros;
    }
    Tensor::cat(vec![zeros, x.slice_dim(1, 0..tokens - steps)], 1)
}

/// `y[t] = x[t + steps]` along axis 1, zero for `t + steps ≥ T`.
fn shift_earlier_float<const D: usize>(x: Tensor<D>, steps: usize) -> Tensor<D> {
    let mut dims = x.dims();
    let tokens = dims[1];
    if steps == 0 {
        return x;
    }
    dims[1] = steps.min(tokens);
    let zeros = Tensor::<D>::zeros(dims, &x.device());
    if steps >= tokens {
        return zeros;
    }
    Tensor::cat(vec![x.slice_dim(1, steps..tokens), zeros], 1)
}
