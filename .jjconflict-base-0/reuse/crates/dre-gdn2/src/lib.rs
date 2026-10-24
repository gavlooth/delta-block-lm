//! Bidirectional channel-wise Gated DeltaNet-2 with a custom CubeCL scan.

mod cube;
mod chunked;
mod ops;
#[cfg(any(test, feature = "cpu-oracle"))]
mod cpu_oracle;
pub mod reference;
pub use reference::{Token as Gdn2Token, ScanResult as Gdn2ScanResult, backward as reference_backward, scan as reference_scan, step as reference_step};
mod layer;
pub use layer::{Gdn2Bidirectional, Gdn2BidirectionalConfig};

#[cfg(test)]
mod tests;
