//! Exact four-stream replication of token features, with zero padding.
use burn::prelude::*;

use super::STREAMS;

#[derive(Config, Debug)]
pub struct StreamInitConfig {
    #[config(default = 512)]
    pub d_model: usize,
}

impl StreamInitConfig {
    pub fn init(&self, _device: &Device) -> StreamInit {
        StreamInit { d_model: self.d_model }
    }
}

/// Parameter-free `[B, T, d] → [B, T, 4, d]` replication.
#[derive(Module, Debug)]
pub struct StreamInit {
    pub d_model: usize,
}

impl StreamInit {
    pub fn forward(&self, features: Tensor<3>, valid: Tensor<2>) -> Tensor<4> {
        let [batch, tokens, width] = features.dims();
        assert_eq!(width, self.d_model, "stream feature width differs from configuration");
        // Select before arithmetic: zero gradients cannot cancel invalid NaN inputs.
        let padding = valid.reshape([batch, tokens, 1]).equal_elem(0.0);
        features
            .mask_fill(padding.expand([batch, tokens, width]), 0.0)
            .cast(burn::tensor::DType::F32)
            .unsqueeze_dim::<4>(2)
            .expand([batch, tokens, STREAMS, width])
    }
}
