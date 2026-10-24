//! Learned global-relay initialization (correction §2.2).

use burn::module::Param;
use burn::nn::Initializer;
use burn::prelude::*;

use crate::STREAMS;

#[derive(Config, Debug)]
pub struct RelayStreamInitConfig {
    #[config(default = "dre_core::Precision::Fp32")]
    pub precision: dre_core::Precision,
    #[config(default = 16)]
    pub relay_slots: usize,
    #[config(default = 512)]
    pub d_model: usize,
    #[config(default = 0.02)]
    pub perturbation_std: f64,
}

impl RelayStreamInitConfig {
    pub fn init(&self, device: &Device) -> RelayStreamInit {
        RelayStreamInit {
            mixed: self.precision == dre_core::Precision::Bf16Mixed,
            relay_vectors: Initializer::Normal { mean: 0.0, std: 0.02 }
                .init([self.relay_slots, self.d_model], device),
            scales: Initializer::Normal { mean: 1.0, std: self.perturbation_std }
                .init([STREAMS, self.d_model], device),
            offsets: Initializer::Normal { mean: 0.0, std: self.perturbation_std }
                .init([STREAMS, self.d_model], device),
        }
    }
}

/// Trainable relay vectors and trainable four-stream initialization parameters.
#[derive(Module, Debug)]
pub struct RelayStreamInit {
    mixed: bool,
    /// `[16, d]`, independent of source token or position embeddings.
    pub relay_vectors: Param<Tensor<2>>,
    /// `[4, d]` trainable stream scaling.
    pub scales: Param<Tensor<2>>,
    /// `[4, d]` trainable stream offsets.
    pub offsets: Param<Tensor<2>>,
}

impl RelayStreamInit {
    /// Broadcast learned slots to the flattened document axis: `[documents, 16, 4, d]`.
    pub fn forward(&self, documents: usize) -> Tensor<4> {
        let [slots, width] = self.relay_vectors.dims();
        let streams = self.relay_vectors.val().unsqueeze_dim::<3>(1)
            * self.scales.val().unsqueeze_dim::<3>(0)
            + self.offsets.val().unsqueeze_dim::<3>(0);
        let streams = streams.unsqueeze_dim::<4>(0).expand([documents, slots, STREAMS, width]);
        if self.mixed { streams.cast(burn::tensor::DType::BF16) } else { streams }
    }
}
