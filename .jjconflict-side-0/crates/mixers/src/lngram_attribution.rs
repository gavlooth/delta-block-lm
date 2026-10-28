//! Resident independent-leaf attribution fixtures calling the production Lngram operators.
use super::{Lngram, LngramBackend, LngramLayout, full_ngram, lookup};
use burn::{
    prelude::*,
    tensor::{Distribution, activation::sigmoid},
};

#[derive(Clone, Copy, Debug)]
pub enum Part {
    Discretizer,
    CodesAddresses,
    Gather(usize),
    Counterfactual(usize),
    Readout,
    Smoothing,
    FusedMemory,
}
impl Part {
    pub fn name(self, layer: &Lngram) -> String {
        match self {
            Self::Discretizer => "Discretizer RMSNorm and GEMM".into(),
            Self::CodesAddresses => "Bit probabilities, hard codes and addresses".into(),
            Self::Gather(o) => {
                format!("Order {} gather, mask and dense table VJP", layer.orders[o])
            }
            Self::Counterfactual(o) => {
                format!("Order {} OneBit counterfactual term", layer.orders[o])
            }
            Self::Readout => "Key/value projections, query/key norms and sigmoid gates".into(),
            Self::Smoothing => "Causal smoothing norm, three taps, SiLU and residual".into(),
            Self::FusedMemory => {
                "Fused exact memory and analytic counterfactual/table VJPs, all orders".into()
            }
        }
    }
}
pub struct Fixture {
    input: Tensor<3>,
    logits: Tensor<3>,
    bits: Tensor<4, Int>,
    probabilities: Tensor<4>,
    locals: Vec<Tensor<3, Int>>,
    currents: Vec<Tensor<4>>,
    memories: Vec<Tensor<4>>,
    readout: Tensor<3>,
    layout: LngramLayout,
}
impl Fixture {
    pub fn new(layer: &Lngram, b: usize, t: usize, d: usize, device: &Device) -> Self {
        let input =
            Tensor::random([b, t, d], Distribution::Uniform(-0.5, 0.5), device).require_grad();
        let logits = layer
            .discretizer_logits(input.clone())
            .detach()
            .require_grad();
        let bits = layer.bits(logits.clone());
        let symbols = layer.symbols(bits.clone());
        let probabilities = sigmoid(
            logits
                .clone()
                .reshape([b, t, layer.routes, layer.bits_per_route])
                .mul_scalar(layer.surrogate_temperature),
        )
        .detach()
        .require_grad();
        let locals: Vec<_> = layer
            .orders
            .iter()
            .map(|&n| layer.local_addresses(&symbols, n))
            .collect();
        let memories = layer
            .orders
            .iter()
            .map(|_| {
                Tensor::random(
                    [b, t, layer.routes, layer.memory_dim],
                    Distribution::Uniform(-0.1, 0.1),
                    device,
                )
                .require_grad()
            })
            .collect();
        let readout =
            Tensor::random([b, t, d], Distribution::Uniform(-0.1, 0.1), device).require_grad();
        let layout = LngramLayout::new(Tensor::ones([b, t], device), Tensor::zeros([b, t], device));
        let currents = layer
            .tables
            .iter()
            .zip(&layer.orders)
            .enumerate()
            .map(|(o, (table, &n))| {
                lookup(
                    LngramBackend::Tensor,
                    table.weight.val().detach(),
                    locals[o].clone().reshape([b * t, layer.routes]),
                    layer.alphabet().pow(n as u32),
                )
                .reshape([b, t, layer.routes, layer.memory_dim])
                .mask_fill(
                    full_ngram(&layout, n)
                        .bool_not()
                        .reshape([b, t, 1, 1])
                        .expand([b, t, layer.routes, layer.memory_dim]),
                    0.,
                )
                .detach()
                .set_require_grad(false)
            })
            .collect();
        Self {
            input,
            logits,
            bits,
            probabilities,
            locals,
            currents,
            memories,
            readout,
            layout,
        }
    }
    pub fn parts(layer: &Lngram) -> Vec<Part> {
        let mut p = vec![Part::Discretizer, Part::CodesAddresses];
        for o in 0..layer.orders.len() {
            p.push(Part::Gather(o));
            p.push(Part::Counterfactual(o));
        }
        p.extend([Part::Readout, Part::Smoothing, Part::FusedMemory]);
        p
    }
    pub fn forward(&self, layer: &Lngram, part: Part) -> Tensor<1> {
        let [b, t, _] = self.input.dims();
        match part {
            Part::Discretizer => layer.discretizer_logits(self.input.clone()).sum(),
            Part::CodesAddresses => {
                let bits = layer.bits(self.logits.clone());
                let symbols = layer.symbols(bits);
                let probabilities =
                    sigmoid(self.logits.clone().mul_scalar(layer.surrogate_temperature));
                let addresses = layer
                    .orders
                    .iter()
                    .map(|&n| {
                        layer
                            .local_addresses(&symbols, n)
                            .mask_fill(
                                full_ngram(&self.layout, n)
                                    .bool_not()
                                    .unsqueeze_dim::<3>(2)
                                    .expand([b, t, layer.routes]),
                                0,
                            )
                            .float()
                            .sum()
                    })
                    .reduce(|a, b| a + b)
                    .unwrap();
                probabilities.sum() + addresses
            }
            Part::Gather(o) => {
                let n = layer.orders[o];
                lookup(
                    LngramBackend::Tensor,
                    layer.tables[o].weight.val(),
                    self.locals[o].clone().reshape([b * t, layer.routes]),
                    layer.alphabet().pow(n as u32),
                )
                .reshape([b, t, layer.routes, layer.memory_dim])
                .mask_fill(
                    full_ngram(&self.layout, n)
                        .bool_not()
                        .reshape([b, t, 1, 1])
                        .expand([b, t, layer.routes, layer.memory_dim]),
                    0.,
                )
                .sum()
            }
            Part::Counterfactual(o) => {
                let n = layer.orders[o];
                let missing = full_ngram(&self.layout, n).bool_not().reshape([b, t, 1, 1]);
                layer
                    .one_bit_term(
                        layer.tables[o].weight.val().detach(),
                        &self.locals[o],
                        &self.bits,
                        &self.probabilities,
                        self.currents[o].clone(),
                        n,
                        &missing,
                    )
                    .sum()
            }
            Part::Readout => {
                let mut readout = layer.clone();
                readout.smoothing = None;
                readout
                    .readout(self.input.clone(), self.memories.clone(), &self.layout)
                    .sum()
            }
            Part::Smoothing => match &layer.smoothing {
                Some(s) => {
                    (self.readout.clone() + s.forward(self.readout.clone(), &self.layout)).sum()
                }
                None => self.readout.clone().sum(),
            },
            Part::FusedMemory => {
                #[cfg(any(feature = "vulkan", feature = "cuda"))]
                {
                    let codes =
                        delta_kernels::lngram_codes(self.logits.clone(), layer.bits_per_route);
                    layer
                        .tables
                        .iter()
                        .zip(&layer.orders)
                        .map(|(table, &n)| {
                            delta_kernels::lngram_memory(
                                self.logits.clone(),
                                codes.clone(),
                                table.weight.val(),
                                self.layout.valid.clone(),
                                self.layout.documents.clone(),
                                layer.bits_per_route,
                                n,
                                false,
                                layer.surrogate_temperature,
                                layer.surrogate_scale,
                            )
                            .sum()
                        })
                        .reduce(|a, b| a + b)
                        .unwrap()
                }
                #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
                {
                    panic!("fused Lngram attribution requires GPU");
                }
            }
        }
    }
}
