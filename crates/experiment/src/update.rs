//! Explicit per-parameter fused AdamW. Uses Burn's exact record type and clipping semantics.
#[cfg(any(feature = "vulkan", feature = "cuda"))]
use burn::optim::AdaptiveMomentumState;
use burn::{
    optim::{AdamWState, Optimizer},
    prelude::*,
};
#[derive(Clone)]
pub struct FusedAdamW {
    pub decay: f32,
    pub clip: f32,
}
impl Optimizer for FusedAdamW {
    type State<const D: usize> = AdamWState<D>;
    fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            // Native AdamW keeps an optional AMSGrad maximum even with AMSGrad disabled.
            // It is checkpoint state, not the denominator used by this configuration.
            let (time, m, v, max_moment_2) = match state {
                Some(s) => (
                    s.momentum.time + 1,
                    Some(s.momentum.moment_1),
                    Some(s.momentum.moment_2),
                    s.momentum.max_moment_2,
                ),
                None => (1, None, None, None),
            };
            let (tensor, moment_1, moment_2) =
                delta_kernels::adamw_update(tensor, grad, m, v, time, lr, self.decay, self.clip);
            (
                tensor,
                Some(AdamWState {
                    momentum: AdaptiveMomentumState {
                        time,
                        moment_1,
                        moment_2,
                        max_moment_2,
                    },
                }),
            )
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            let _ = (lr, tensor, grad, state, self.decay, self.clip);
            panic!("fused AdamW requires a GPU feature")
        }
    }
    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

/// Lngram tables use the predecessor's 5x learning rate and zero decay.
/// Dense Adam moments are intentional: unused rows with fresh state stay bit-identical.
#[derive(Clone)]
pub struct LngramAdamW {
    pub fused: bool,
    pub clip: f32,
}
impl Optimizer for LngramAdamW {
    type State<const D: usize> = AdamWState<D>;
    fn step<const D: usize>(
        &self, lr: f64, tensor: Tensor<D>, grad: Tensor<D>, state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        if self.fused {
            FusedAdamW { decay: 0.0, clip: self.clip }.step(lr * 5.0, tensor, grad, state)
        } else {
            let grad = burn::grad_clipping::GradientClippingConfig::Norm(self.clip).init().clip_gradient(grad);
            burn::optim::AdamWConfig::new().with_weight_decay(0.0).build().step(lr * 5.0, tensor, grad, state)
        }
    }
    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}
pub fn square_sum<const D: usize>(gradient: Tensor<D>) -> Tensor<1> {
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    {
        delta_kernels::gradient_square_sum(gradient)
    }
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    {
        let _ = gradient;
        panic!("fused gradient reduction requires a GPU feature")
    }
}
