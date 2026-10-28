//! Independent `f64` reference of one Lngram module, written per position from the equations in
//! `docs/optimization/lngram.md`: a scalar reverse-mode tape gives the chain-rule gradients of the readout path,
//! and the counterfactual surrogate of the discretizer is computed directly from the App. A formulas using the
//! per-position upstream gradients `g = ∂L/∂e`.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct Var(usize);

const NO_PARENT: usize = usize::MAX;

/// Scalar reverse-mode tape. Each node has at most two parents with their local partial derivatives.
#[derive(Default)]
pub struct Tape {
    values: Vec<f64>,
    parents: Vec<[(usize, f64); 2]>,
}

impl Tape {
    fn push(&mut self, value: f64, parents: [(usize, f64); 2]) -> Var {
        self.values.push(value);
        self.parents.push(parents);
        Var(self.values.len() - 1)
    }

    pub fn leaf(&mut self, value: f64) -> Var {
        self.push(value, [(NO_PARENT, 0.0), (NO_PARENT, 0.0)])
    }

    pub fn value(&self, x: Var) -> f64 {
        self.values[x.0]
    }

    fn unary(&mut self, value: f64, a: Var, da: f64) -> Var {
        self.push(value, [(a.0, da), (NO_PARENT, 0.0)])
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let value = self.value(a) + self.value(b);
        self.push(value, [(a.0, 1.0), (b.0, 1.0)])
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let (va, vb) = (self.value(a), self.value(b));
        self.push(va * vb, [(a.0, vb), (b.0, va)])
    }

    pub fn div(&mut self, a: Var, b: Var) -> Var {
        let (va, vb) = (self.value(a), self.value(b));
        self.push(va / vb, [(a.0, 1.0 / vb), (b.0, -va / (vb * vb))])
    }

    pub fn scale(&mut self, a: Var, c: f64) -> Var {
        let value = self.value(a) * c;
        self.unary(value, a, c)
    }

    pub fn offset(&mut self, a: Var, c: f64) -> Var {
        let value = self.value(a) + c;
        self.unary(value, a, 1.0)
    }

    pub fn sqrt(&mut self, a: Var) -> Var {
        let s = self.value(a).sqrt();
        self.unary(s, a, 0.5 / s)
    }

    pub fn sigmoid(&mut self, a: Var) -> Var {
        let s = sigmoid(self.value(a));
        self.unary(s, a, s * (1.0 - s))
    }

    pub fn silu(&mut self, a: Var) -> Var {
        let x = self.value(a);
        let s = sigmoid(x);
        self.unary(x * s, a, s + x * s * (1.0 - s))
    }

    pub fn sum(&mut self, xs: &[Var]) -> Var {
        let mut total = xs[0];
        for &x in &xs[1..] {
            total = self.add(total, x);
        }
        total
    }

    pub fn dot(&mut self, a: &[Var], b: &[Var]) -> Var {
        let products: Vec<Var> = a.iter().zip(b).map(|(&x, &y)| self.mul(x, y)).collect();
        self.sum(&products)
    }

    /// `y_i = x_i / sqrt(mean(x²) + eps) · γ_i`.
    pub fn rms_norm(&mut self, x: &[Var], gamma: &[Var], eps: f64) -> Vec<Var> {
        let squares = self.dot(x, x);
        let mean = self.scale(squares, 1.0 / x.len() as f64);
        let shifted = self.offset(mean, eps);
        let rms = self.sqrt(shifted);
        x.iter()
            .zip(gamma)
            .map(|(&xi, &gi)| {
                let normed = self.div(xi, rms);
                self.mul(normed, gi)
            })
            .collect()
    }

    /// `out_o = b_o + Σ_i x_i W[i, o]` with `W` row-major `[x.len(), out]`.
    pub fn linear(&mut self, x: &[Var], weight: &[Var], bias: &[Var]) -> Vec<Var> {
        let outputs = bias.len();
        (0..outputs)
            .map(|o| {
                let column: Vec<Var> = (0..x.len()).map(|i| weight[i * outputs + o]).collect();
                let dot = self.dot(x, &column);
                self.add(dot, bias[o])
            })
            .collect()
    }

    /// Gradients of every node for the given output seeds.
    pub fn backward(&self, seeds: &[(Var, f64)]) -> Vec<f64> {
        let mut grads = vec![0.0; self.values.len()];
        for &(node, seed) in seeds {
            grads[node.0] += seed;
        }
        for node in (0..self.values.len()).rev() {
            let grad = grads[node];
            if grad == 0.0 {
                continue;
            }
            for &(parent, partial) in &self.parents[node] {
                if parent != NO_PARENT {
                    grads[parent] += grad * partial;
                }
            }
        }
        grads
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surrogate {
    OneBit,
    LocalExpectation,
}

/// Parameters, row-major as stored by Burn (`Linear` weights are `[d_input, d_output]`).
#[derive(Clone, Debug)]
pub struct Model {
    pub d_model: usize,
    pub routes: usize,
    pub bits: usize,
    pub memory_dim: usize,
    pub orders: Vec<usize>,
    pub discretizer_gamma: Vec<f64>,
    pub discretizer: Vec<f64>,
    pub tables: Vec<Vec<f64>>,
    pub key_weight: Vec<f64>,
    pub key_bias: Vec<f64>,
    pub value_weight: Vec<f64>,
    pub value_bias: Vec<f64>,
    pub query_gamma: Vec<f64>,
    pub key_gamma: Vec<f64>,
    /// `None` when smoothing is off.
    pub smoothing_gamma: Option<Vec<f64>>,
    /// `[3, d_model]`, taps at `t`, `t − dilation`, `t − 2*dilation`.
    pub smoothing_weight: Option<Vec<f64>>,
    pub dilation: usize,
    pub epsilon: f64,
    pub temperature: f64,
    pub scale: f64,
    pub surrogate: Surrogate,
}

impl Model {
    pub fn alphabet(&self) -> usize {
        1 << self.bits
    }
}

/// Inputs of one evaluation. Flat position `p = b · tokens + t`.
#[derive(Clone, Debug)]
pub struct Batch {
    pub batch: usize,
    pub tokens: usize,
    pub starts: Vec<i64>,
    pub ends: Vec<i64>,
    /// `[B, T, d]`.
    pub u: Vec<f64>,
    /// Loss `L = Σ out ⊙ loss_weight`, `[B, T, d]`.
    pub loss_weight: Vec<f64>,
}

/// Gradients with the same layout as [`Model`] / [`Batch::u`].
#[derive(Clone, Debug, Default)]
pub struct Gradients {
    pub u: Vec<f64>,
    pub discretizer_gamma: Vec<f64>,
    pub discretizer: Vec<f64>,
    pub tables: Vec<Vec<f64>>,
    pub key_weight: Vec<f64>,
    pub key_bias: Vec<f64>,
    pub value_weight: Vec<f64>,
    pub value_bias: Vec<f64>,
    pub query_gamma: Vec<f64>,
    pub key_gamma: Vec<f64>,
    pub smoothing_gamma: Vec<f64>,
    pub smoothing_weight: Vec<f64>,
}

pub struct Evaluation {
    pub loss: f64,
    /// `[B, T, d]`, exactly zero at padding.
    pub output: Vec<f64>,
    /// Discretizer logits `[B, T, R·m]`.
    pub logits: Vec<f64>,
    /// Route symbols `[B, T, R]`.
    pub symbols: Vec<i64>,
    /// Per order: global row or `-1`, `[B, T, R]`.
    pub addresses: Vec<Vec<i64>>,
    /// Chain-rule gradients with the addresses held fixed (no discretizer gradient).
    pub chain: Gradients,
    /// Chain-rule plus the counterfactual surrogate through the discretizer.
    pub total: Gradients,
}

struct Leaves {
    u: Vec<Var>,
    discretizer_gamma: Vec<Var>,
    discretizer: Vec<Var>,
    key_weight: Vec<Var>,
    key_bias: Vec<Var>,
    value_weight: Vec<Var>,
    value_bias: Vec<Var>,
    query_gamma: Vec<Var>,
    key_gamma: Vec<Var>,
    smoothing_gamma: Vec<Var>,
    smoothing_weight: Vec<Var>,
}

/// Evaluates the module. `fixed_symbols` replaces the discretized symbols (finite differences hold the
/// addresses fixed that way).
pub fn evaluate(model: &Model, batch: &Batch, fixed_symbols: Option<&[i64]>) -> Evaluation {
    let (d, routes, bits, dm) = (model.d_model, model.routes, model.bits, model.memory_dim);
    let alphabet = model.alphabet() as i64;
    let positions = batch.batch * batch.tokens;
    let width = routes * bits;
    let mut tape = Tape::default();
    let mut leaves_of = |values: &[f64]| values.iter().map(|&v| tape.leaf(v)).collect::<Vec<_>>();
    let leaves = Leaves {
        u: leaves_of(&batch.u),
        discretizer_gamma: leaves_of(&model.discretizer_gamma),
        discretizer: leaves_of(&model.discretizer),
        key_weight: leaves_of(&model.key_weight),
        key_bias: leaves_of(&model.key_bias),
        value_weight: leaves_of(&model.value_weight),
        value_bias: leaves_of(&model.value_bias),
        query_gamma: leaves_of(&model.query_gamma),
        key_gamma: leaves_of(&model.key_gamma),
        smoothing_gamma: leaves_of(model.smoothing_gamma.as_deref().unwrap_or(&[])),
        smoothing_weight: leaves_of(model.smoothing_weight.as_deref().unwrap_or(&[])),
    };
    let valid: Vec<bool> = (0..positions)
        .map(|p| batch.ends[p] >= batch.starts[p])
        .collect();
    let u_of = |p: usize| &leaves.u[p * d..(p + 1) * d];

    // Discretizer: z = RMSNorm(u) W_q, bits = 1[z > 0], a = Σ_j 2^j b_j.
    let no_bias = vec![tape.leaf(0.0); width];
    let mut logit_vars = Vec::with_capacity(positions * width);
    for p in 0..positions {
        let normed = tape.rms_norm(u_of(p), &leaves.discretizer_gamma, model.epsilon);
        let row = tape.linear(&normed, &leaves.discretizer, &no_bias);
        logit_vars.extend(row);
    }
    let logits: Vec<f64> = logit_vars.iter().map(|&z| tape.value(z)).collect();
    let symbols: Vec<i64> = match fixed_symbols {
        Some(symbols) => symbols.to_vec(),
        None => (0..positions * routes)
            .map(|pr| {
                (0..bits)
                    .map(|j| ((logits[pr * bits + j] > 0.0) as i64) << j)
                    .sum()
            })
            .collect(),
    };
    let symbol = |p: usize, r: usize| symbols[p * routes + r];

    // Exact suffix addresses; -1 without a full n-gram inside the document.
    let tokens = batch.tokens;
    let full = |p: usize, order: usize| {
        valid[p] && (p % tokens) as i64 - batch.starts[p] >= order as i64 - 1
    };
    let addresses: Vec<Vec<i64>> = model
        .orders
        .iter()
        .map(|&order| {
            (0..positions * routes)
                .map(|pr| {
                    let (p, r) = (pr / routes, pr % routes);
                    if !full(p, order) {
                        return -1;
                    }
                    let local: i64 = (0..order)
                        .map(|i| symbol(p + i + 1 - order, r) * alphabet.pow(i as u32))
                        .sum();
                    r as i64 * alphabet.pow(order as u32) + local
                })
                .collect()
        })
        .collect();

    // Lookups: one node per (order, position, route, column) so the per-position upstream gradient is visible.
    let mut table_leaves: Vec<HashMap<usize, Var>> = vec![HashMap::new(); model.orders.len()];
    let mut memory: Vec<Vec<Vec<Var>>> = vec![Vec::with_capacity(positions); model.orders.len()];
    for (o, order_addresses) in addresses.iter().enumerate() {
        for p in 0..positions {
            let mut row = Vec::with_capacity(routes * dm);
            for r in 0..routes {
                let address = order_addresses[p * routes + r];
                for c in 0..dm {
                    if address < 0 {
                        row.push(tape.leaf(0.0));
                    } else {
                        let flat = address as usize * dm + c;
                        let leaf = *table_leaves[o]
                            .entry(flat)
                            .or_insert_with(|| tape.leaf(model.tables[o][flat]));
                        row.push(tape.scale(leaf, 1.0));
                    }
                }
            }
            memory[o].push(row);
        }
    }

    // Context-aware readout: α_n = σ(RMSNorm(u)·RMSNorm(W_K e_n + b_K) / √d), V = Σ_n α_n (W_V e_n + b_V).
    let mut readout: Vec<Option<Vec<Var>>> = vec![None; positions];
    for p in 0..positions {
        if !valid[p] {
            continue;
        }
        let query = tape.rms_norm(u_of(p), &leaves.query_gamma, model.epsilon);
        let mut sum: Option<Vec<Var>> = None;
        for order_memory in &memory {
            let key = tape.linear(&order_memory[p], &leaves.key_weight, &leaves.key_bias);
            let value = tape.linear(&order_memory[p], &leaves.value_weight, &leaves.value_bias);
            let key = tape.rms_norm(&key, &leaves.key_gamma, model.epsilon);
            let score = tape.dot(&query, &key);
            let score = tape.scale(score, 1.0 / (d as f64).sqrt());
            let gate = tape.sigmoid(score);
            let gated: Vec<Var> = value.iter().map(|&v| tape.mul(gate, v)).collect();
            sum = Some(match sum {
                None => gated,
                Some(previous) => previous
                    .iter()
                    .zip(&gated)
                    .map(|(&a, &b)| tape.add(a, b))
                    .collect(),
            });
        }
        readout[p] = sum;
    }

    // Smoothing: Y = V + SiLU(Σ_k w_k ⊙ RMSNorm(V)[t + o_k]) with taps outside the document reading zero.
    let output_vars: Vec<Option<Vec<Var>>> = if model.smoothing_gamma.is_some() {
        let normed: Vec<Option<Vec<Var>>> = readout
            .iter()
            .map(|v| {
                v.as_ref()
                    .map(|v| tape.rms_norm(v, &leaves.smoothing_gamma, model.epsilon))
            })
            .collect();
        let offsets = [0, -(model.dilation as i64), -2 * model.dilation as i64];
        (0..positions)
            .map(|p| {
                let v = readout[p].as_ref()?;
                let t = (p % tokens) as i64;
                Some(
                    (0..d)
                        .map(|c| {
                            let mut terms = Vec::new();
                            for (k, &offset) in offsets.iter().enumerate() {
                                let source = t + offset;
                                if source >= batch.starts[p] && source <= batch.ends[p] {
                                    let q = (p as i64 + offset) as usize;
                                    let x = normed[q]
                                        .as_ref()
                                        .expect("in-document position is valid")[c];
                                    terms.push(tape.mul(leaves.smoothing_weight[k * d + c], x));
                                }
                            }
                            let conv = tape.sum(&terms);
                            let activated = tape.silu(conv);
                            tape.add(v[c], activated)
                        })
                        .collect(),
                )
            })
            .collect()
    } else {
        readout.clone()
    };

    let mut loss_terms = Vec::new();
    let mut output = vec![0.0; positions * d];
    for p in 0..positions {
        if let Some(y) = &output_vars[p] {
            for c in 0..d {
                output[p * d + c] = tape.value(y[c]);
                loss_terms.push(tape.scale(y[c], batch.loss_weight[p * d + c]));
            }
        }
    }
    let loss = tape.sum(&loss_terms);
    let chain_grads = tape.backward(&[(loss, 1.0)]);

    // Counterfactual surrogate for the discretizer logits (App. A), from g = ∂L/∂e at full positions.
    let mut dz = vec![0.0; positions * width];
    let tau = model.temperature;
    for (o, &order) in model.orders.iter().enumerate() {
        let table = &model.tables[o];
        let row_of = |r: usize, slots: &[i64]| -> usize {
            r * alphabet.pow(order as u32) as usize
                + slots
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| (a * alphabet.pow(i as u32)) as usize)
                    .sum::<usize>()
        };
        for p in 0..positions {
            if !full(p, order) {
                continue;
            }
            for r in 0..routes {
                let g: Vec<f64> = (0..dm)
                    .map(|c| chain_grads[memory[o][p][r * dm + c].0])
                    .collect();
                let slots: Vec<i64> = (0..order).map(|i| symbol(p + i + 1 - order, r)).collect();
                let project = |slots: &[i64]| -> f64 {
                    let row = row_of(r, slots);
                    (0..dm).map(|c| g[c] * table[row * dm + c]).sum()
                };
                for i in 0..order {
                    let s = p + i + 1 - order;
                    let probability = |j: usize| sigmoid(tau * logits[(s * routes + r) * bits + j]);
                    match model.surrogate {
                        Surrogate::OneBit => {
                            for j in 0..bits {
                                let mut one = slots.clone();
                                one[i] |= 1 << j;
                                let mut zero = slots.clone();
                                zero[i] &= !(1 << j);
                                let score = project(&one) - project(&zero);
                                let pj = probability(j);
                                dz[(s * routes + r) * bits + j] +=
                                    model.scale * tau * pj * (1.0 - pj) * score;
                            }
                        }
                        Surrogate::LocalExpectation => {
                            let p_bits: Vec<f64> = (0..bits).map(probability).collect();
                            for code in 0..alphabet {
                                let mut replaced = slots.clone();
                                replaced[i] = code;
                                let inner = project(&replaced);
                                let prob: f64 = (0..bits)
                                    .map(|j| {
                                        if (code >> j) & 1 == 1 {
                                            p_bits[j]
                                        } else {
                                            1.0 - p_bits[j]
                                        }
                                    })
                                    .product();
                                for j in 0..bits {
                                    let beta = ((code >> j) & 1) as f64;
                                    dz[(s * routes + r) * bits + j] +=
                                        model.scale * tau * prob * (beta - p_bits[j]) * inner;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let seeds: Vec<(Var, f64)> = logit_vars.iter().zip(&dz).map(|(&z, &g)| (z, g)).collect();
    let surrogate_grads = tape.backward(&seeds);

    let collect = |grads: &[f64]| {
        let read = |vars: &[Var]| vars.iter().map(|v| grads[v.0]).collect::<Vec<_>>();
        Gradients {
            u: read(&leaves.u),
            discretizer_gamma: read(&leaves.discretizer_gamma),
            discretizer: read(&leaves.discretizer),
            tables: table_leaves
                .iter()
                .zip(&model.tables)
                .map(|(map, table)| {
                    let mut gradient = vec![0.0; table.len()];
                    for (&flat, var) in map {
                        gradient[flat] = grads[var.0];
                    }
                    gradient
                })
                .collect(),
            key_weight: read(&leaves.key_weight),
            key_bias: read(&leaves.key_bias),
            value_weight: read(&leaves.value_weight),
            value_bias: read(&leaves.value_bias),
            query_gamma: read(&leaves.query_gamma),
            key_gamma: read(&leaves.key_gamma),
            smoothing_gamma: read(&leaves.smoothing_gamma),
            smoothing_weight: read(&leaves.smoothing_weight),
        }
    };
    let chain = collect(&chain_grads);
    let total_grads: Vec<f64> = chain_grads
        .iter()
        .zip(&surrogate_grads)
        .map(|(a, b)| a + b)
        .collect();
    let total = collect(&total_grads);
    Evaluation {
        loss: tape.value(loss),
        output,
        logits,
        symbols,
        addresses,
        chain,
        total,
    }
}

/// Continuous inputs that finite differences may perturb (addresses held fixed).
#[derive(Clone, Copy, Debug)]
pub enum Group {
    U,
    Table(usize),
    KeyWeight,
    KeyBias,
    ValueWeight,
    ValueBias,
    QueryGamma,
    KeyGamma,
    SmoothingGamma,
    SmoothingWeight,
}

impl Group {
    pub fn all(model: &Model) -> Vec<Group> {
        let mut groups = vec![Group::U];
        groups.extend((0..model.orders.len()).map(Group::Table));
        groups.extend([
            Group::KeyWeight,
            Group::KeyBias,
            Group::ValueWeight,
            Group::ValueBias,
            Group::QueryGamma,
            Group::KeyGamma,
        ]);
        if model.smoothing_gamma.is_some() {
            groups.extend([Group::SmoothingGamma, Group::SmoothingWeight]);
        }
        groups
    }

    pub fn values_mut<'a>(self, model: &'a mut Model, batch: &'a mut Batch) -> &'a mut Vec<f64> {
        match self {
            Group::U => &mut batch.u,
            Group::Table(o) => &mut model.tables[o],
            Group::KeyWeight => &mut model.key_weight,
            Group::KeyBias => &mut model.key_bias,
            Group::ValueWeight => &mut model.value_weight,
            Group::ValueBias => &mut model.value_bias,
            Group::QueryGamma => &mut model.query_gamma,
            Group::KeyGamma => &mut model.key_gamma,
            Group::SmoothingGamma => model.smoothing_gamma.as_mut().expect("smoothing on"),
            Group::SmoothingWeight => model.smoothing_weight.as_mut().expect("smoothing on"),
        }
    }

    pub fn gradient(self, gradients: &Gradients) -> &Vec<f64> {
        match self {
            Group::U => &gradients.u,
            Group::Table(o) => &gradients.tables[o],
            Group::KeyWeight => &gradients.key_weight,
            Group::KeyBias => &gradients.key_bias,
            Group::ValueWeight => &gradients.value_weight,
            Group::ValueBias => &gradients.value_bias,
            Group::QueryGamma => &gradients.query_gamma,
            Group::KeyGamma => &gradients.key_gamma,
            Group::SmoothingGamma => &gradients.smoothing_gamma,
            Group::SmoothingWeight => &gradients.smoothing_weight,
        }
    }
}
