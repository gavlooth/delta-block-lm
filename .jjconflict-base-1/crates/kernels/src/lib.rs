//! Causal GDN-2 chunk-WY and fused recurrent KronLSTM CubeCL kernels.
//! Both use key width 32 and value width 64; batch/time/head counts are dynamic.
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod gdn2;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use ops::gdn2_scan;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod gdn_training;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod gdn_training_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use gdn_training_ops::{GdnTrainingOutput, gdn_training_scan};
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
pub use hola_ops::{HolaReadOutput, hola_read, hola_schedule};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod prefix;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod prefix_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use prefix_ops::{prefix_read, prefix_hola_read, prefix_summaries};
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
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod update;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod update_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use update_ops::{adamw_update, gradient_square_sum};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod cross_entropy;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod cross_entropy_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use cross_entropy_ops::cross_entropy_rows;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod soft_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use soft_ops::soft_order_keys;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod epilogue;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod epilogue_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use epilogue_ops::{silu_analytic, silu_vjp};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod lngram;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod lngram_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use lngram_ops::{lngram_codes, lngram_memory, lngram_table_lookup};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod wavelet;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod wavelet_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use wavelet_ops::wavelet_smooth;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod ohc;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod ohc_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use ohc_ops::ohc_write;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod ohc_read;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod ohc_read_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use ohc_read_ops::{ohc_gate_norm, ohc_quaternion, ohc_weighted_read};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod ohc_gate;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod ohc_gate_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use ohc_gate_ops::ohc_gate_project;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod noisy_hola;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod noisy_hola_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use noisy_hola_ops::noisy_hola_read;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod codpula;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
mod codpula_ops;
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use codpula::{CodpulaKernelCounters, codpula_kernel_counters};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub use codpula_ops::{
    CodpulaOps, codpula_chain_backward, codpula_chain_statistics, codpula_emission_flows,
    codpula_leaf_backward, codpula_leaf_normalizers, codpula_log_partition,
    codpula_token_marginals,
};
