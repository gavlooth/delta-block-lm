//! Local sliding-window attention and hybrid document-local global-relay sparse attention.

mod layer;
#[cfg(test)]
mod spike;
pub mod kernel;
mod attention_kernel;
mod oracle;
mod relay_attention_kernel;
mod relay;

pub use layer::{LocalAttention, LocalAttentionConfig};
pub use layer::ROTARY_MAX_TOKENS;
pub use relay::{LocalGlobalRelayAttention, LocalGlobalRelayAttentionConfig};

#[cfg(all(test, feature = "cuda"))]
mod bf16_tests;
#[cfg(test)]
mod relay_tests;
#[cfg(test)]
mod tests;
