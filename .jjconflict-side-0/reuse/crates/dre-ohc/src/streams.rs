//! Four-stream initialization (spec §1.2): `X_ts = scale_s ⊙ e_t + offset_s` from the projected token
//! features `e_t ∈ R[d]`, with trainable per-stream scales and offsets that start near `1` and `0`, so the
//! four streams begin near-identical but slightly perturbed. Padded positions are exactly `0`.

use burn::module::Param;
use burn::nn::Initializer;
use burn::prelude::*;
use dre_core::BatchLayout;

use crate::STREAMS;

#[derive(Config, Debug)]
pub struct StreamInitConfig {
    #[config(default = "dre_core::Precision::Fp32")]
    pub precision: dre_core::Precision,
    #[config(default = 512)]
    pub d_model: usize,
    /// Standard deviation of the initial perturbation of scales (around 1) and offsets (around 0).
    #[config(default = 0.02)]
    pub perturbation_std: f64,
}

impl StreamInitConfig {
    pub fn init(&self, device: &Device) -> StreamInit {
        StreamInit {
            mixed: self.precision == dre_core::Precision::Bf16Mixed,
            scales: Initializer::Normal {
                mean: 1.0,
                std: self.perturbation_std,
            }
            .init([STREAMS, self.d_model], device),
            offsets: Initializer::Normal {
                mean: 0.0,
                std: self.perturbation_std,
            }
            .init([STREAMS, self.d_model], device),
        }
    }
}

/// `[B, T, d] → [B, T, 4, d]` stream initialization.
#[derive(Module, Debug)]
pub struct StreamInit {
    mixed: bool,
    /// `[4, d]`.
    pub scales: Param<Tensor<2>>,
    /// `[4, d]`.
    pub offsets: Param<Tensor<2>>,
}

impl StreamInit {
    pub fn forward(&self, features: Tensor<3>, layout: &BatchLayout) -> Tensor<4> {
        let [batch,tokens,width] = features.dims();
        let [_, d_model] = self.scales.dims();
        assert_eq!(
            width, d_model,
            "feature width {width} differs from the configured {d_model}"
        );
        let padding=layout.valid.clone().equal_elem(0.0);
        // Select before trainable multiplication: zero upstream gradients cannot cancel NaN inputs.
        let features=features.mask_fill(padding.clone().expand([batch,tokens,width]),0.0);
        let streams = features.cast(burn::tensor::DType::F32).unsqueeze_dim::<4>(2) * self.scales.val().unsqueeze::<4>()
            + self.offsets.val().unsqueeze::<4>();
        let streams=streams.mask_fill(padding.unsqueeze_dim::<4>(3).expand([batch,tokens,STREAMS,width]),0.0);
        if self.mixed { streams.cast(burn::tensor::DType::BF16) } else { streams }
    }
}
