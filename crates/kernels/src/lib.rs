//! Causal GDN-2 chunk-WY and fused recurrent KronLSTM CubeCL kernels.
//! Both use key width 32 and value width 64; batch/time/head counts are dynamic.
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod gdn2;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use ops::gdn2_scan;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod kron;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod kron_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod kron_parallel;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use kron_ops::{KronScanOutput, kron_scan, kron_scan_partitioned};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod hola;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod hola_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use hola_ops::{HolaReadOutput, hola_read};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod norm;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod norm_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use norm_ops::{l2_norm, rms_norm};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod decay;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod decay_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use decay_ops::decay;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod matmul_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use matmul_ops::heuristic_matmul;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod conv;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod conv_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use conv_ops::causal_conv;
