//! Four per-token orthogonal residual streams, migrated from dre-ohc.
//! FP32 only; stream masking uses valid tensors instead of the predecessor host layout.
mod readout;
mod reference;
pub mod rotation;
mod stage;
mod streams;
use burn::prelude::*;
pub use readout::{OhcReadout, OhcReadoutConfig};
pub use reference::numerical_checks;
use serde::{Deserialize, Serialize};
pub use stage::{OhcRead, OhcStage, OhcStageConfig};
pub use streams::{StreamInit, StreamInitConfig};
pub const STREAMS: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OhcBackend {
    #[default]
    Tensor,
    FusedGpu,
    FusedReadWrite,
    FusedGate,
    TransposedGate,
}
impl OhcBackend {
    pub fn is_tensor(&self) -> bool {
        *self == Self::Tensor
    }
    pub fn fused_read(&self) -> bool {
        matches!(
            self,
            Self::FusedReadWrite | Self::FusedGate | Self::TransposedGate
        )
    }
    /// Gate projection and flattened-stream RMS; all choices preserve the same W.
    pub fn project_gate(&self, streams: Tensor<4>, weight: Tensor<2>) -> Tensor<3> {
        let [b, t, _, d] = streams.dims();
        if *self == Self::FusedGate {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                return delta_kernels::ohc_gate_project(streams, weight);
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            {
                let _ = (streams, weight);
                panic!("fused oHC has no CPU fallback");
            }
        }
        // Normalize before the contraction so its backward sums dy * W before
        // multiplying by inverse RMS. Scaling each dy first loses cancellation
        // accuracy at near-zero inputs, where inverse RMS approaches 1e6.
        let flat = streams.clone().cast(burn::tensor::DType::F32)
            .reshape([b, t, 4 * d]);
        let flat = Self::Tensor.gate_norm(streams, flat).reshape([b * t, 4 * d]);
        let projected = delta_kernels::ohc_linear(flat, weight, *self == Self::TransposedGate);
        projected.reshape([b, t, 16])
    }
    pub fn gate_norm(&self, streams: Tensor<4>, projected: Tensor<3>) -> Tensor<3> {
        if self.fused_read() {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                return delta_kernels::ohc_gate_norm(streams, projected);
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            {
                let _ = (streams, projected);
                panic!("fused oHC has no CPU fallback");
            }
        }
        let [b, t, _, d] = streams.dims();
        let square_mean = streams.reshape([b, t, 4 * d]).square().mean_dim(2);
        // Mask before sqrt, not after an infinite sqrt derivative: this gives the
        // exact nonzero formula and a finite zero norm derivative at padded rows.
        let zero = square_mean.clone().equal_elem(0.0);
        let rms = square_mean.mask_fill(zero.clone(), 1.0).sqrt().mask_fill(zero, 0.0);
        projected * (rms + stage::GATE_NORM_EPSILON).recip()
    }
    pub fn weighted_read(
        &self,
        streams: Tensor<4>,
        weights: Tensor<3>,
    ) -> Tensor<3> {
        if self.fused_read() {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                return delta_kernels::ohc_weighted_read(streams, weights);
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            {
                let _ = (streams, weights);
                panic!("fused oHC has no CPU fallback");
            }
        }
        (streams * weights.unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .squeeze_dim::<3>(2)
    }
    pub fn write(
        &self,
        streams: Tensor<4>,
        rotation: Tensor<4>,
        gates: Tensor<3>,
        update: Tensor<3>,
    ) -> Tensor<4> {
        match self {
            Self::Tensor => {
                rotation.matmul(streams)
                    + gates.unsqueeze_dim::<4>(3) * update.unsqueeze_dim::<4>(2)
            }
            Self::FusedGpu | Self::FusedReadWrite | Self::FusedGate | Self::TransposedGate => {
                #[cfg(any(feature = "vulkan", feature = "cuda"))]
                {
                    delta_kernels::ohc_write(streams, rotation, gates, update)
                }
                #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
                {
                    let _ = (streams, rotation, gates, update);
                    panic!("fused oHC has no CPU fallback")
                }
            }
        }
    }
}
