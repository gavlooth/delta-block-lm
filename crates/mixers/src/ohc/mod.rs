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
}
impl OhcBackend {
    pub fn is_tensor(&self) -> bool {
        *self == Self::Tensor
    }
    pub fn fused_read(&self) -> bool {
        *self == Self::FusedReadWrite
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
        projected
            * (streams.reshape([b, t, 4 * d]).square().mean_dim(2) + stage::GATE_NORM_EPSILON)
                .sqrt()
                .recip()
    }
    pub fn weighted_read(
        &self,
        streams: Tensor<4>,
        weights: Tensor<3>,
        norm: &burn::nn::RmsNorm,
    ) -> Tensor<3> {
        if self.fused_read() {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                return delta_kernels::ohc_weighted_read(
                    streams,
                    weights,
                    norm.gamma.val(),
                    norm.epsilon,
                );
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            {
                let _ = (streams, weights, norm);
                panic!("fused oHC has no CPU fallback");
            }
        }
        (norm.forward(streams) * weights.unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .squeeze_dim::<3>(2)
    }
    pub fn write(
        &self,
        streams: Tensor<4>,
        rotation: Tensor<4>,
        gates: Tensor<3>,
        update: Tensor<3>,
        scale: Tensor<1>,
    ) -> Tensor<4> {
        let gates = gates * scale.unsqueeze::<3>();
        match self {
            Self::Tensor => {
                rotation.matmul(streams)
                    + gates.unsqueeze_dim::<4>(3) * update.unsqueeze_dim::<4>(2)
            }
            Self::FusedGpu | Self::FusedReadWrite => {
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
