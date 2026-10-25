//! Explicit per-parameter fused AdamW. Uses Burn's exact record type and clipping semantics.
#[cfg(any(feature = "vulkan", feature = "cuda"))]
use burn::optim::AdaptiveMomentumState;
use burn::{
    optim::{AdamWState, Optimizer},
    prelude::*,
};

/// A dimension reduction owns its output; global sum uses unordered GPU atomics.
/// Keep the native AdamW state and formula while making its clipping repeatable.
pub fn clip_by_norm<const D: usize>(gradient: Tensor<D>, threshold: f32) -> Tensor<D> {
    let dtype = gradient.dtype();
    let gradient = match dtype {
        burn::tensor::DType::F16 | burn::tensor::DType::BF16 => gradient.cast(burn::tensor::DType::F32),
        _ => gradient,
    };
    let count: usize = gradient.dims().iter().product();
    let norm = gradient.clone().square().reshape([count]).sum_dim(0).sqrt();
    let minimum = gradient.dtype().finfo().unwrap_or(burn::tensor::FloatDType::F32.finfo()).min_positive;
    let coefficient = (threshold/norm.add_scalar(minimum)).clamp_max(1.0);
    (gradient*coefficient.unsqueeze()).cast(dtype)
}

#[derive(Clone)]
pub struct TensorAdamW {
    pub decay: f32,
    pub clip: f32,
}
impl Optimizer for TensorAdamW {
    type State<const D: usize> = AdamWState<D>;
    fn step<const D: usize>(&self, lr: f64, tensor: Tensor<D>, gradient: Tensor<D>, state: Option<Self::State<D>>)
        -> (Tensor<D>, Option<Self::State<D>>)
    {
        burn::optim::AdamWConfig::new().with_weight_decay(self.decay).build()
            .step(lr,tensor,clip_by_norm(gradient,self.clip),state)
    }
    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum=state.momentum.to_device(device);
        state
    }
}
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
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        if self.fused {
            FusedAdamW {
                decay: 0.0,
                clip: self.clip,
            }
            .step(lr * 5.0, tensor, grad, state)
        } else {
            let grad = clip_by_norm(grad,self.clip);
            burn::optim::AdamWConfig::new()
                .with_weight_decay(0.0)
                .build()
                .step(lr * 5.0, tensor, grad, state)
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

/// Fixed-input regression for the scalar used to rescale native optimizer gradients.
pub fn clipping_checks(device: &Device) -> anyhow::Result<serde_json::Value> {
    use anyhow::ensure;
    let input = (0..4112).map(|i| (((i*7919+17)%65521) as f32/65521.0-0.5)
        *if i%11==0 {0.3}else{0.07}).collect::<Vec<_>>();
    let gradient=Tensor::<2>::from_data(TensorData::new(input.clone(),[257,16]),device);
    let norm=input.iter().map(|&x|f64::from(x).powi(2)).sum::<f64>().sqrt();
    ensure!(norm>1.0,"clipping regression requires an active coefficient");
    let expected=input.iter().map(|&x|f64::from(x)/norm).collect::<Vec<_>>();
    let mut first=None::<Vec<f32>>;
    let mut native_norm_bits=std::collections::BTreeSet::new();
    let mut report=serde_json::Value::Null;
    for _ in 0..32 {
        native_norm_bits.insert(gradient.clone().square().sum().sqrt().into_scalar::<f32>().to_bits());
        let values=clip_by_norm(gradient.clone(),1.0).into_data().try_to_vec::<f32>()
            .map_err(|error|anyhow::anyhow!("clipping regression readback: {error:?}"))?;
        if let Some(first)=&first {
            ensure!(first.iter().zip(&values).all(|(a,b)|a.to_bits()==b.to_bits()),
                "dimension-reduced clipping is not bit-identical");
        } else {
            report=crate::sbd::compare(&values,&expected)?;
            first=Some(values);
        }
    }
    Ok(serde_json::json!({"passed":true,"shape":[257,16],"repetitions":32,
        "f64_norm":norm,"clipping_active":true,"bit_identical":true,"f64":report,
        "upstream_global_sum_distinct_norm_bits":native_norm_bits,
        "reduction":"flattened_sum_dim_without_inter_cube_atomics"}))
}

#[cfg(test)]
mod clipping_tests {
    #[test]
    fn native_clipping_matches_f64_and_repeats_exactly() {
        super::clipping_checks(&burn::prelude::Device::ndarray()).unwrap();
    }
}
