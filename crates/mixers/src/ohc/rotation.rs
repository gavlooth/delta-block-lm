//! Quaternion-pair parameterization of `SO(4)` (oHC §3.6 Eq. 8, §5 Eq. 12, Appendix 8 Eq. 15).
//!
//! Eight logits `ℓ` give two unit quaternions
//! `q = (1+ℓ0, ℓ1, ℓ2, ℓ3) / sqrt(|·|² + ε)` and `r = (1+ℓ4, ℓ5, ℓ6, ℓ7) / sqrt(|·|² + ε)`,
//! and the rotation `x ↦ q x r̄` of `R⁴` has the matrix `Q = L_q R_r̄`. Every entry of `Q` is a signed product
//! `q_a r_b`, so `vec(Q)` is a fixed `{0, ±1}` matrix applied to the outer product `q rᵀ` (Appendix 8); that
//! matrix is built once on the host and applied with one matmul.

use burn::prelude::*;
use burn::tensor::TensorData;

/// `ε` of oHC Eq. (12). Small enough that `1 + ε` rounds to `1` in f32, so `ℓ = 0` gives `Q = I` exactly.
pub const QUATERNION_EPSILON: f64 = 1e-12;

/// Number of logits that define one rotation: four per quaternion.
pub const ROTATION_LOGITS: usize = 8;

/// Left multiplication `L_q` of oHC Eq. (15): entry `(i, k)` is `sign · q[index]`.
const LEFT: [[(f32, usize); 4]; 4] = [
    [(1.0, 0), (-1.0, 1), (-1.0, 2), (-1.0, 3)],
    [(1.0, 1), (1.0, 0), (-1.0, 3), (1.0, 2)],
    [(1.0, 2), (1.0, 3), (1.0, 0), (-1.0, 1)],
    [(1.0, 3), (-1.0, 2), (1.0, 1), (1.0, 0)],
];

/// Right multiplication by the conjugate, `R_r̄` of oHC Eq. (15): entry `(k, j)` is `sign · r[index]`.
const RIGHT_CONJUGATE: [[(f32, usize); 4]; 4] = [
    [(1.0, 0), (1.0, 1), (1.0, 2), (1.0, 3)],
    [(-1.0, 1), (1.0, 0), (-1.0, 3), (1.0, 2)],
    [(-1.0, 2), (1.0, 3), (1.0, 0), (-1.0, 1)],
    [(-1.0, 3), (-1.0, 2), (1.0, 1), (1.0, 0)],
];

/// `[16, 16]` row-major matrix `M` with `vec(L_q R_r̄) = vec(q rᵀ) · M`: row `a·4+b` (product `q_a r_b`),
/// column `i·4+j` (entry `Q[i][j]`). Four nonzeros per row and per column, all `±1`.
fn product_pattern() -> Vec<f32> {
    let mut pattern = vec![0.0f32; 256];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let (left_sign, a) = LEFT[i][k];
                let (right_sign, b) = RIGHT_CONJUGATE[k][j];
                pattern[(a * 4 + b) * 16 + i * 4 + j] += left_sign * right_sign;
            }
        }
    }
    pattern
}

/// Builds `Q ∈ SO(4)` per position from eight logits. Holds only constants, no trainable parameters.
#[derive(Module, Debug)]
pub struct QuaternionPair {
    /// `vec(q rᵀ) → vec(Q)`, `[16, 16]`.
    pattern: Tensor<2>,
    /// `(1, 0, 0, 0, 1, 0, 0, 0)`: the constant `1` on the two leading channels of Eq. (12), `[8]`.
    identity_offset: Tensor<1>,
}

impl QuaternionPair {
    pub fn new(device: &Device) -> Self {
        let mut offset = [0.0f32; ROTATION_LOGITS];
        offset[0] = 1.0;
        offset[4] = 1.0;
        Self {
            pattern: Tensor::from_data(TensorData::new(product_pattern(), [16, 16]), device),
            identity_offset: Tensor::from_data(
                TensorData::new(offset.to_vec(), [ROTATION_LOGITS]),
                device,
            ),
        }
    }

    /// Rotations `[B, T, 4, 4]` from logits `[B, T, 8]`; `ℓ = 0` gives the identity.
    pub fn rotation(&self, logits: Tensor<3>) -> Tensor<4> {
        let [batch, tokens, width] = logits.dims();
        assert_eq!(
            width, ROTATION_LOGITS,
            "a quaternion pair needs {ROTATION_LOGITS} logits, got {width}"
        );
        let raw = logits + self.identity_offset.clone().unsqueeze::<3>();
        let q = unit_quaternion(raw.clone().narrow(2, 0, 4));
        let r = unit_quaternion(raw.narrow(2, 4, 4));
        // products[.., a, b] = q_a r_b
        let products = q.unsqueeze_dim::<4>(3) * r.unsqueeze_dim::<4>(2);
        products
            .reshape([batch * tokens, 16])
            .matmul(self.pattern.clone())
            .reshape([batch, tokens, 4, 4])
    }
    pub fn rotation_selected(&self, logits: Tensor<3>, fused: bool) -> Tensor<4> {
        if fused {
            #[cfg(any(feature = "vulkan", feature = "cuda"))]
            {
                return delta_kernels::ohc_quaternion(logits, self.pattern.clone());
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
            {
                let _ = logits;
                panic!("fused oHC has no CPU fallback");
            }
        }
        self.rotation(logits)
    }
}

/// `v / sqrt(|v|² + ε)` along the last axis of `[B, T, 4]`.
fn unit_quaternion(raw: Tensor<3>) -> Tensor<3> {
    let norm = (raw.clone().square().sum_dim(2) + QUATERNION_EPSILON).sqrt();
    raw / norm
}
