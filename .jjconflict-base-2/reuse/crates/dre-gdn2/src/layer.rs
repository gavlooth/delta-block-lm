//! GDN-2 token-mixer projections and document-aware short convolutions.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::Distribution;
use dre_core::BatchLayout;

use crate::reference::{HEADS, KEY_DIM, VALUE_DIM};

const WIDTH: usize = 512;
const DEFAULT_SCAN: &str = "chunked";

#[derive(Config, Debug)]
pub struct Gdn2BidirectionalConfig {
    #[config(default = "dre_core::Precision::Fp32")]
    pub precision: dre_core::Precision,
    #[config(default = 512)]
    pub d_model: usize,
    #[config(default = 4)]
    pub heads: usize,
    #[config(default = 32)]
    pub key_dim: usize,
    #[config(default = 64)]
    pub value_dim: usize,
    #[config(default = 4)]
    pub conv_width: usize,
    #[config(default = 32)]
    pub checkpoint_interval: usize,
    /// `serial` retains the tokenwise oracle; `chunked` enables chunkwise WY.
    /// Optional storage lets checkpoints written before this switch load without
    /// changing their parameter records. Omission uses the gated default.
    #[config(default = "Some(DEFAULT_SCAN.to_string())")]
    pub scan: Option<String>,
    #[config(default = 1e-5)]
    pub norm_epsilon: f64,
}

impl Gdn2BidirectionalConfig {
    pub fn init(&self, device: &Device) -> Gdn2Bidirectional {
        assert_eq!(self.d_model, WIDTH);
        assert_eq!(self.heads, HEADS);
        assert_eq!(self.key_dim, KEY_DIM);
        assert_eq!(self.value_dim, VALUE_DIM);
        assert!(self.conv_width > 0);
        assert!(self.checkpoint_interval > 0);
        let scan = self.scan.as_deref().unwrap_or(DEFAULT_SCAN);
        assert!(matches!(scan, "serial" | "chunked"), "gdn2.scan must be serial or chunked");
        let direction = || Direction::init(self, device);
        Gdn2Bidirectional {
            mixed: self.precision == dre_core::Precision::Bf16Mixed,
            forward: direction(),
            reverse: direction(),
            output: linear(self.d_model, self.d_model, false).init(device),
            checkpoint_interval: self.checkpoint_interval,
            chunked: scan == "chunked",
        }
    }
}

#[derive(Module, Debug)]
pub struct Gdn2Bidirectional {
    mixed: bool,
    forward: Direction,
    reverse: Direction,
    output: Linear,
    #[module(skip)]
    checkpoint_interval: usize,
    #[module(skip)]
    chunked: bool,
}

#[derive(Module, Debug)]
struct Direction {
    mixed: bool,
    q: Linear,
    k: Linear,
    v: Linear,
    decay_hidden: Linear,
    decay: Linear,
    erase: Linear,
    write: Linear,
    output_gate_hidden: Linear,
    output_gate: Linear,
    q_conv: Param<Tensor<2>>,
    k_conv: Param<Tensor<2>>,
    v_conv: Param<Tensor<2>>,
    a_log: Param<Tensor<1>>,
    dt_bias: Param<Tensor<1>>,
    output_norm: RmsNorm,
    #[module(skip)]
    conv_width: usize,
}

impl Direction {
    fn init(config: &Gdn2BidirectionalConfig, device: &Device) -> Self {
        let key_width = config.heads * config.key_dim;
        let value_width = config.heads * config.value_dim;
        let conv_init = |channels| Param::from_tensor(Tensor::random(
            [config.conv_width, channels],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ));
        let a_log_data: Vec<f32> = (0..config.heads)
            .map(|i| (1.0 + 15.0 * (i as f32 + 0.5) / config.heads as f32).ln())
            .collect();
        let dt: Vec<f32> = (0..key_width)
            .map(|i| {
                let unit = (i as f32 + 0.5) / key_width as f32;
                let dt = (0.001f32.ln() + unit * (0.1f32.ln() - 0.001f32.ln()))
                    .exp()
                    .max(1e-4);
                dt + (-(-dt).exp_m1()).ln()
            })
            .collect();
        Self {
            mixed: config.precision == dre_core::Precision::Bf16Mixed,
            q: linear(config.d_model, key_width, false).init(device),
            k: linear(config.d_model, key_width, false).init(device),
            v: linear(config.d_model, value_width, false).init(device),
            decay_hidden: linear(config.d_model, config.value_dim, false).init(device),
            decay: linear(config.value_dim, key_width, false).init(device),
            erase: linear(config.d_model, key_width, false).init(device),
            write: linear(config.d_model, value_width, false).init(device),
            output_gate_hidden: linear(config.d_model, config.value_dim, false).init(device),
            output_gate: linear(config.value_dim, value_width, true).init(device),
            q_conv: conv_init(key_width),
            k_conv: conv_init(key_width),
            v_conv: conv_init(value_width),
            a_log: Param::from_tensor(Tensor::from_data(TensorData::new(a_log_data, [config.heads]), device)),
            dt_bias: Param::from_tensor(Tensor::from_data(TensorData::new(dt, [key_width]), device)),
            output_norm: RmsNormConfig::new(config.value_dim)
                .with_epsilon(config.norm_epsilon)
                .init(device),
            conv_width: config.conv_width,
        }
    }

    fn gates(
        &self,
        input: Tensor<3>,
        layout: &BatchLayout,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>, Tensor<4>) {
        let [batch, tokens, _] = input.dims();
        let input=input.mask_fill(layout.valid.clone().lower_elem(0.5),0.0);
        let q = burn::tensor::activation::silu(causal_depthwise(
            dre_core::project(&self.q, input.clone(), self.mixed),
            self.q_conv.val(),
            layout,
            self.conv_width,
        ));
        let k = burn::tensor::activation::silu(causal_depthwise(
            dre_core::project(&self.k, input.clone(), self.mixed),
            self.k_conv.val(),
            layout,
            self.conv_width,
        ));
        let v = burn::tensor::activation::silu(causal_depthwise(
            dre_core::project(&self.v, input.clone(), self.mixed),
            self.v_conv.val(),
            layout,
            self.conv_width,
        ));
        let q = normalize_heads(q.reshape([batch, tokens, HEADS, KEY_DIM]));
        let k = normalize_heads(k.reshape([batch, tokens, HEADS, KEY_DIM]));
        let v = v.reshape([batch, tokens, HEADS, VALUE_DIM]);
        let decay = dre_core::project(
            &self.decay, dre_core::project(&self.decay_hidden, input.clone(), self.mixed), self.mixed,
        ).reshape([batch, tokens, HEADS, KEY_DIM]);
        let a_log = self.a_log.val().exp().reshape([1, 1, HEADS, 1]);
        let dt = self.dt_bias.val().reshape([1, 1, HEADS, KEY_DIM]);
        let alpha = (a_log * burn::tensor::activation::softplus(decay + dt, 1.0))
            .neg()
            .exp();
        let erase = burn::tensor::activation::sigmoid(dre_core::project(&self.erase, input.clone(), self.mixed))
            .reshape([batch, tokens, HEADS, KEY_DIM]);
        let write = burn::tensor::activation::sigmoid(dre_core::project(&self.write, input, self.mixed))
            .reshape([batch, tokens, HEADS, VALUE_DIM]);
        (q, k, v, alpha, erase, write)
    }

    fn finish(&self, recurrent: Tensor<4>, input: Tensor<3>) -> Tensor<3> {
        let [batch, tokens, _, _] = recurrent.dims();
        let values = self
            .output_norm
            .forward(recurrent.cast(burn::tensor::DType::F32))
            .reshape([batch, tokens, HEADS * VALUE_DIM]);
        let gate_values = dre_core::project(&self.output_gate_hidden, input, self.mixed);
        let gate = burn::tensor::activation::silu(dre_core::project(&self.output_gate, gate_values, self.mixed))
            .reshape([batch, tokens, HEADS * VALUE_DIM]);
        values * gate
    }

    fn forward(&self, input: Tensor<3>, layout: &BatchLayout, interval: usize, chunked: bool) -> Tensor<4> {
        let (q, k, v, alpha, erase, write) = self.gates(input, layout);
        let dtype = if self.mixed { burn::tensor::DType::BF16 } else { burn::tensor::DType::F32 };
        let (q, k, v) = (q.cast(dtype), k.cast(dtype), v.cast(dtype));
        #[cfg(any(test, feature = "cpu-oracle"))]
        if q.clone().inner().device() == Device::ndarray() {
            return crate::cpu_oracle::scan(q, k, v, alpha, erase, write, layout);
        }
        crate::ops::gdn2_scan(
            q,
            k,
            v,
            alpha,
            erase,
            write,
            layout.doc_start.clone(),
            layout.doc_end.clone(),
            interval,
            chunked,
        )
    }
}

impl Gdn2Bidirectional {
    pub fn forward(&self, input: Tensor<3>, layout: &BatchLayout) -> Tensor<3> {
        let invalid=layout.valid.clone().lower_elem(0.5);
        let input=input.mask_fill(invalid.clone(),0.0);
        let reversed_input = reverse_within_documents(input.clone(), layout);
        let forward = self.forward.forward(input.clone(), layout, self.checkpoint_interval, self.chunked);
        let reverse = self.reverse.forward(reversed_input.clone(), layout, self.checkpoint_interval, self.chunked);
        let recurrent_invalid=invalid.clone().unsqueeze_dim::<4>(3);
        let forward = self.forward.finish(forward.mask_fill(recurrent_invalid.clone(),0.0), input);
        let reverse = self.reverse.finish(reverse.mask_fill(recurrent_invalid,0.0), reversed_input);
        let reverse = reverse_within_documents(reverse, layout);
        dre_core::project(&self.output, Tensor::cat(vec![forward, reverse], 2).mask_fill(invalid.clone(),0.0), self.mixed)
            .mask_fill(invalid,0.0)
    }
}

fn reverse_within_documents(input: Tensor<3>, layout: &BatchLayout) -> Tensor<3> {
    let [batch, tokens, channels] = input.dims();
    let mut indices = Vec::with_capacity(batch * tokens);
    for documents in layout.host.documents_per_row() {
        let mut start = 0usize;
        for &length in documents {
            indices.extend((start..start + length).rev().map(|i| i as i32));
            start += length;
        }
        indices.extend((start..tokens).map(|i| i as i32));
    }
    let index: Tensor<3, Int> = Tensor::<2, Int>::from_data(
        TensorData::new(indices, [batch, tokens]),
        &input.device(),
    )
    .unsqueeze_dim::<3>(2)
    .repeat_dim(2, channels);
    input.gather(1, index)
}

fn reverse_within_documents_4d(input: Tensor<4>, layout: &BatchLayout) -> Tensor<4> {
    let [batch, tokens, heads, width] = input.dims();
    let mut indices = Vec::with_capacity(batch * tokens);
    for documents in layout.host.documents_per_row() {
        let mut start = 0usize;
        for &length in documents {
            indices.extend((start..start + length).rev().map(|i| i as i32));
            start += length;
        }
        indices.extend((start..tokens).map(|i| i as i32));
    }
    let index: Tensor<3, Int> = Tensor::<2, Int>::from_data(
        TensorData::new(indices, [batch, tokens]),
        &input.device(),
    )
    .unsqueeze_dim::<3>(2)
    .repeat_dim(2, heads);
    let index: Tensor<4, Int> = index.unsqueeze_dim::<4>(3).repeat_dim(3, width);
    input.gather(1, index)
}

fn linear(input: usize, output: usize, bias: bool) -> LinearConfig {
    LinearConfig::new(input, output)
        .with_bias(bias)
        .with_initializer(burn::nn::Initializer::XavierUniform {
            gain: 2.0f64.powf(-2.5),
        })
}

fn normalize_heads(x: Tensor<4>) -> Tensor<4> {
    let norm = (x.clone().square().sum_dim(3) + 1e-6).sqrt();
    x / norm
}
fn causal_depthwise(x: Tensor<3>, weights: Tensor<2>, layout: &BatchLayout, kernel: usize) -> Tensor<3> {
    let [batch, tokens, channels] = x.dims();
    let mut result = Tensor::zeros([batch, tokens, channels], &x.device());
    let starts = layout.doc_start.clone().unsqueeze_dim(2);
    let positions = Tensor::<1, Int>::arange(0..tokens as i64, &x.device())
        .reshape([1, tokens, 1])
        .expand([batch, tokens, 1]);
    let valid = layout.valid.clone().greater_elem(0.5);
    for lag in 0..kernel {
        let shifted = if lag == 0 {
            x.clone()
        } else if lag >= tokens {
            Tensor::zeros([batch, tokens, channels], &x.device())
        } else {
            let zeros = Tensor::zeros([batch, lag, channels], &x.device());
            let body = x.clone().slice([0..batch, 0..tokens - lag, 0..channels]);
            Tensor::cat(vec![zeros, body], 1)
        };
        let inactive = (positions.clone() - starts.clone())
            .greater_equal_scalar(lag as i64).bool_and(valid.clone()).bool_not();
        let weight = weights
            .clone()
            .slice([lag..lag + 1, 0..channels])
            .reshape([1, 1, channels]);
        result = result + shifted.mask_fill(inactive,0.0) * weight;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn2_causal_depthwise_nan_padding_preserves_values_and_actual_gradients_cpu() {
        let device=Device::ndarray().autodiff();
        let layout=BatchLayout::new(dre_core::DocumentLayout::packed(vec![vec![2,1],vec![]],5),&device);
        let weights=Param::from_tensor(Tensor::from_data(TensorData::new(
            vec![0.2f32,-0.3,0.4,0.5,-0.6,0.7],[3,2]),&device).require_grad());
        let host=|tensor:Tensor<3>|tensor.to_data().try_to_vec_as::<f32>().unwrap();
        let run=|dirty:bool| {
            let input=Tensor::<3>::from_data(TensorData::new((0..20).map(|index| {
                if index>=6 {if dirty {f32::NAN}else{0.0}}else{(index as f32-2.0)/7.0}
            }).collect::<Vec<_>>(),[2,5,2]),&device).require_grad();
            // The real production short-convolution producer, not a substituted GDN scan.
            let output=causal_depthwise(input.clone(),weights.val(),&layout,3);
            let values=host(output.clone());
            assert!(values.iter().all(|value|value.is_finite()));
            assert!(values[6..].iter().all(|&value|value==0.0));
            let grads=output.square().sum().backward();
            let input_gradient=host(input.grad(&grads).unwrap());
            let weight_gradient=weights.val().grad(&grads).unwrap().to_data().try_to_vec_as::<f32>().unwrap();
            assert!(input_gradient.iter().chain(&weight_gradient).all(|value|value.is_finite()));
            assert!(input_gradient[..6].iter().any(|value|value.abs()>1e-8));
            assert!(input_gradient[6..].iter().all(|&value|value==0.0));
            assert!(weight_gradient.iter().any(|value|value.abs()>1e-8));
            (values,input_gradient,weight_gradient)
        };
        let clean=run(false);let dirty=run(true);
        for (dirty,clean) in [(&dirty.0,&clean.0),(&dirty.1,&clean.1),(&dirty.2,&clean.2)] {
            assert_eq!(dirty.len(),clean.len());
            for (&actual,&expected) in dirty.iter().zip(clean) {
                assert!((actual-expected).abs()<1e-7+1e-6*expected.abs());
            }
        }
        let kernel=[0.2f64,-0.3,0.4,0.5,-0.6,0.7];
        for token in 0..3 {for channel in 0..2 {
            let first=if token<2 {0}else{2};
            let expected=(0..3).filter(|&lag|lag<=token-first).map(|lag| {
                f64::from(((token-lag)*2+channel) as f32-2.0)/7.0*kernel[lag*2+channel]
            }).sum::<f64>();
            assert!((f64::from(dirty.0[token*2+channel])-expected).abs()<1e-7);
        }}
    }

    #[test]
    fn gdn2_reverse_keeps_document_ranges_and_padding() {
        let device = Device::ndarray().autodiff();
        let layout = BatchLayout::new(dre_core::DocumentLayout::packed(vec![vec![2, 2]], 5), &device);
        let input = Tensor::<3>::from_data(
            TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], [1, 5, 1]),
            &device,
        );
        let reversed = reverse_within_documents(input.clone(), &layout)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(reversed, vec![1.0, 0.0, 3.0, 2.0, 4.0]);
        let restored = reverse_within_documents(
            Tensor::<3>::from_data(TensorData::new(reversed, [1, 5, 1]), &device),
            &layout,
        )
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
        assert_eq!(restored, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let input = Tensor::<4>::from_data(
            TensorData::new((0..10).map(|x| x as f32).collect(), [1, 5, 1, 2]),
            &device,
        );
        let reversed = reverse_within_documents_4d(input, &layout)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(reversed, vec![2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0, 8.0, 9.0]);
    }

    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn gdn2_bidirectional_layer_packed_matches_separate_rows() {
        assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
        let device = dre_core::gpu_device().autodiff();
        let layer = Gdn2BidirectionalConfig::new().init(&device);
        let packed_layout = BatchLayout::new(
            dre_core::DocumentLayout::packed(vec![vec![2, 1]], 4),
            &device,
        );
        let separate_layout = BatchLayout::new(
            dre_core::DocumentLayout::packed(vec![vec![2], vec![1]], 2),
            &device,
        );
        let packed_data: Vec<f32> = (0..4 * 512)
            .map(|i| if i >= 3 * 512 { 0.0 } else { ((i * 17 + 3) % 101) as f32 / 100.0 - 0.5 })
            .collect();
        let separate_data = packed_data[..2 * 512]
            .iter()
            .copied()
            .chain(packed_data[2 * 512..3 * 512].iter().copied())
            .chain(std::iter::repeat_n(0.0, 512))
            .collect();
        let packed = Tensor::<3>::from_data(TensorData::new(packed_data, [1, 4, 512]), &device);
        let separate = Tensor::<3>::from_data(TensorData::new(separate_data, [2, 2, 512]), &device);
        let packed_output = layer.forward(packed, &packed_layout).into_data().try_to_vec::<f32>().unwrap();
        let separate_output = layer.forward(separate, &separate_layout).into_data().try_to_vec::<f32>().unwrap();
        assert!(packed_output[3 * 512..].iter().all(|&x| x == 0.0));
        let pairs = (0..512).flat_map(|c| [
            (packed_output[c], separate_output[c]),
            (packed_output[512 + c], separate_output[512 + c]),
            (packed_output[2 * 512 + c], separate_output[2 * 512 + c]),
        ]).collect::<Vec<_>>();
        let s0 = (pairs.iter().map(|&(_, b)| f64::from(b).powi(2)).sum::<f64>() / pairs.len() as f64).sqrt();
        let max_abs = pairs.iter().map(|&(a, b)| (f64::from(a) - f64::from(b)).abs()).fold(0.0f64, f64::max);
        println!("packed-vs-separate layer: max_abs={max_abs:.3e}, s0={s0:.3e}");
        for (a, b) in pairs {
            assert!((f64::from(a) - f64::from(b)).abs() <= 1e-5 * s0 + 1e-4 * f64::from(b).abs());
        }
    }

    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn gdn2_layer_packed_multiple_rows_matches_separate_documents() {
        assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
        let device = dre_core::gpu_device().autodiff();
        let mut layer = Gdn2BidirectionalConfig::new().init(&device);
        fill_model(&mut layer, &device);
        layer.forward.q.weight = Param::from_tensor(Tensor::<2>::from_data(
            TensorData::new(
                (0..512 * 128).map(|i| ((i * 7 + 3) % 29) as f32 / 512.0 - 0.02).collect::<Vec<_>>(),
                [512, 128],
            ),
            &device,
        ));
        layer.forward.q_conv = Param::from_tensor(Tensor::<2>::from_data(
            TensorData::new(vec![0.025f32; 4 * 128], [4, 128]),
            &device,
        ));
        let packed_layout = BatchLayout::new(
            dre_core::DocumentLayout::packed(vec![vec![13, 9], vec![17]], 32),
            &device,
        );
        let separate_layout = BatchLayout::new(
            dre_core::DocumentLayout::packed(vec![vec![13], vec![9], vec![17]], 32),
            &device,
        );
        let mut packed_data = vec![0.0f32; 2 * 32 * 512];
        for b in 0..2 {
            let valid = if b == 0 { 22 } else { 17 };
            for t in 0..valid {
                for c in 0..512 {
                    packed_data[(b * 32 + t) * 512 + c] =
                        ((b * 7919 + t * 97 + c * 13 + 5) % 127) as f32 / 127.0 - 0.5;
                }
            }
        }
        let mut separate_data = vec![0.0f32; 3 * 32 * 512];
        for t in 0..13 {
            separate_data[t * 512..(t + 1) * 512]
                .copy_from_slice(&packed_data[t * 512..(t + 1) * 512]);
        }
        for t in 0..9 {
            separate_data[(32 + t) * 512..(33 + t) * 512]
                .copy_from_slice(&packed_data[(13 + t) * 512..(14 + t) * 512]);
        }
        for t in 0..17 {
            separate_data[(64 + t) * 512..(65 + t) * 512]
                .copy_from_slice(&packed_data[(32 + t) * 512..(33 + t) * 512]);
        }
        let packed = Tensor::<3>::from_data(TensorData::new(packed_data, [2, 32, 512]), &device);
        let separate = Tensor::<3>::from_data(TensorData::new(separate_data, [3, 32, 512]), &device);
        let raw_q_packed = layer.forward.q.forward(packed.clone()).into_data().try_to_vec::<f32>().unwrap();
        let raw_q_separate = layer.forward.q.forward(separate.clone()).into_data().try_to_vec::<f32>().unwrap();
        for channel in 0..128 {
            assert!((raw_q_packed[channel] - raw_q_separate[channel]).abs() < 2e-5,
                "raw q projection differs channel={channel}: {} vs {}", raw_q_packed[channel], raw_q_separate[channel]);
        }
        let conv_q_packed = causal_depthwise(
            layer.forward.q.forward(packed.clone()), layer.forward.q_conv.val(), &packed_layout, 4,
        ).into_data().try_to_vec::<f32>().unwrap();
        let conv_q_separate = causal_depthwise(
            layer.forward.q.forward(separate.clone()), layer.forward.q_conv.val(), &separate_layout, 4,
        ).into_data().try_to_vec::<f32>().unwrap();
        for channel in 0..128 {
            assert!((conv_q_packed[channel] - conv_q_separate[channel]).abs() < 2e-5,
                "conv q differs channel={channel}: {} vs {}", conv_q_packed[channel], conv_q_separate[channel]);
        }
        let activated_packed_tensor = burn::tensor::activation::silu(causal_depthwise(
            layer.forward.q.forward(packed.clone()), layer.forward.q_conv.val(), &packed_layout, 4,
        ).reshape([2, 32, HEADS, KEY_DIM]));
        let activated_separate_tensor = burn::tensor::activation::silu(causal_depthwise(
            layer.forward.q.forward(separate.clone()), layer.forward.q_conv.val(), &separate_layout, 4,
        ).reshape([3, 32, HEADS, KEY_DIM]));
        let activated_packed = activated_packed_tensor.clone().into_data().try_to_vec::<f32>().unwrap();
        let activated_separate = activated_separate_tensor.clone().into_data().try_to_vec::<f32>().unwrap();
        for channel in 0..128 {
            assert!((activated_packed[channel] - activated_separate[channel]).abs() < 2e-5,
                "activated q differs channel={channel}: {} vs {}", activated_packed[channel], activated_separate[channel]);
        }
        let norm_packed = (activated_packed_tensor.clone().square().sum_dim(3) + 1e-6).sqrt()
            .into_data().try_to_vec::<f32>().unwrap();
        let norm_separate = (activated_separate_tensor.clone().square().sum_dim(3) + 1e-6).sqrt()
            .into_data().try_to_vec::<f32>().unwrap();
        for head in 0..HEADS {
            assert!((norm_packed[head] - norm_separate[head]).abs() < 2e-5,
                "q norm differs head={head}: {} vs {}", norm_packed[head], norm_separate[head]);
        }
        let normalized_packed = normalize_heads(activated_packed_tensor.clone())
            .into_data().try_to_vec::<f32>().unwrap();
        let normalized_separate = normalize_heads(activated_separate_tensor.clone())
            .into_data().try_to_vec::<f32>().unwrap();
        for channel in 0..128 {
            assert!((normalized_packed[channel] - normalized_separate[channel]).abs() < 2e-5,
                "normalized q differs channel={channel}: {} vs {}", normalized_packed[channel], normalized_separate[channel]);
        }
        let (packed_q, packed_k, packed_v, packed_alpha, packed_erase, packed_write) =
            layer.forward.gates(packed.clone(), &packed_layout);
        let (separate_q, separate_k, separate_v, separate_alpha, separate_erase, separate_write) =
            layer.forward.gates(separate.clone(), &separate_layout);
        let compare = |packed: Tensor<4>, separate: Tensor<4>, name: &str| {
            let [_, _, heads, width] = packed.dims();
            let packed = packed.into_data().try_to_vec::<f32>().unwrap();
            let separate = separate.into_data().try_to_vec::<f32>().unwrap();
            for (packed_row, packed_start, separate_row, separate_start, tokens) in
                [(0usize, 0usize, 0usize, 0usize, 13usize), (0, 13, 1, 0, 9), (1, 0, 2, 0, 17)]
            {
                for t in 0..tokens {
                    for channel in 0..heads * width {
                        let a = packed[(packed_row * 32 + packed_start + t) * heads * width + channel];
                        let b = separate[(separate_row * 32 + separate_start + t) * heads * width + channel];
                        assert!((a - b).abs() < 2e-5,
                            "{name} differs doc row={packed_row} t={t} c={channel}: {a} vs {b}");
                    }
                }
            }
        };
        compare(packed_q, separate_q, "q");
        compare(packed_k, separate_k, "k");
        compare(packed_v, separate_v, "v");
        compare(packed_alpha, separate_alpha, "alpha");
        compare(packed_erase, separate_erase, "erase");
        compare(packed_write, separate_write, "write");
        let packed = layer.forward(packed, &packed_layout).into_data().try_to_vec::<f32>().unwrap();
        let separate = layer.forward(separate, &separate_layout).into_data().try_to_vec::<f32>().unwrap();
        let comparisons = [
            (0usize, 0usize, 13usize),
            (13usize, 32usize, 9usize),
            (32usize, 64usize, 17usize),
        ];
        let mut pairs = Vec::with_capacity(39 * 512);
        for (packed_start, separate_start, tokens) in comparisons {
            for t in 0..tokens {
                for c in 0..512 {
                    pairs.push((
                        packed[(packed_start + t) * 512 + c],
                        separate[(separate_start + t) * 512 + c],
                    ));
                }
            }
        }
        let s0 = (pairs.iter().map(|&(_, b)| f64::from(b).powi(2)).sum::<f64>() / pairs.len() as f64).sqrt();
        let max_abs = pairs.iter().map(|&(a, b)| (f64::from(a) - f64::from(b)).abs()).fold(0.0f64, f64::max);
        println!("multirow packed-vs-separate: max_abs={max_abs:.3e}, s0={s0:.3e}");
        for (a, b) in pairs {
            assert!((f64::from(a) - f64::from(b)).abs() <= 1e-5 * s0 + 1e-4 * f64::from(b).abs(),
                "packed layer output differs: {a} vs {b}");
        }
    }

    #[derive(Clone)]
    struct FLinear {
        input: usize,
        output: usize,
        weight: Vec<f64>,
        bias: Option<Vec<f64>>,
    }

    fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f64> {
        tensor.into_data().try_to_vec::<f32>().unwrap().into_iter().map(f64::from).collect()
    }

    impl FLinear {
        fn from_gpu(layer: &Linear) -> Self {
            let [input, output] = layer.weight.val().dims();
            Self {
                input,
                output,
                weight: values(layer.weight.val()),
                bias: layer.bias.as_ref().map(|bias| values(bias.val())),
            }
        }

        fn forward(&self, x: &[f64], rows: usize) -> Vec<f64> {
            let mut y = vec![0.0; rows * self.output];
            for row in 0..rows {
                for i in 0..self.input {
                    let value = x[row * self.input + i];
                    for j in 0..self.output {
                        y[row * self.output + j] += value * self.weight[i * self.output + j];
                    }
                }
                if let Some(bias) = &self.bias {
                    for j in 0..self.output { y[row * self.output + j] += bias[j]; }
                }
            }
            y
        }
    }

    #[derive(Clone)]
    struct FDirection {
        q: FLinear,
        k: FLinear,
        v: FLinear,
        decay_hidden: FLinear,
        decay: FLinear,
        erase: FLinear,
        write: FLinear,
        output_gate_hidden: FLinear,
        output_gate: FLinear,
        q_conv: Vec<f64>,
        k_conv: Vec<f64>,
        v_conv: Vec<f64>,
        a_log: Vec<f64>,
        dt_bias: Vec<f64>,
        norm_gamma: Vec<f64>,
        norm_epsilon: f64,
        conv_width: usize,
    }

    impl FDirection {
        fn from_gpu(layer: &super::Direction) -> Self {
            Self {
                q: FLinear::from_gpu(&layer.q),
                k: FLinear::from_gpu(&layer.k),
                v: FLinear::from_gpu(&layer.v),
                decay_hidden: FLinear::from_gpu(&layer.decay_hidden),
                decay: FLinear::from_gpu(&layer.decay),
                erase: FLinear::from_gpu(&layer.erase),
                write: FLinear::from_gpu(&layer.write),
                output_gate_hidden: FLinear::from_gpu(&layer.output_gate_hidden),
                output_gate: FLinear::from_gpu(&layer.output_gate),
                q_conv: values(layer.q_conv.val()),
                k_conv: values(layer.k_conv.val()),
                v_conv: values(layer.v_conv.val()),
                a_log: values(layer.a_log.val()),
                dt_bias: values(layer.dt_bias.val()),
                norm_gamma: values(layer.output_norm.gamma.val()),
                norm_epsilon: layer.output_norm.epsilon,
                conv_width: layer.conv_width,
            }
        }
    }

    fn fill_linear(layer: &mut Linear, device: &Device, salt: usize) {
        let [input, output] = layer.weight.val().dims();
        let weight: Vec<f32> = (0..input * output)
            .map(|i| (((i * 37 + salt * 13 + 5) % 101) as f32 - 50.0) / 101.0 * 0.04)
            .collect();
        layer.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(weight, [input, output]), device,
        ));
        if let Some(bias) = &mut layer.bias {
            let values = (0..output)
                .map(|i| (((i * 17 + salt * 7 + 3) % 53) as f32 - 26.0) / 53.0 * 0.02)
                .collect::<Vec<_>>();
            *bias = Param::from_tensor(Tensor::from_data(TensorData::new(values, [output]), device));
        }
    }

    fn fill_conv(parameter: &mut Param<Tensor<2>>, device: &Device, salt: usize) {
        let [width, channels] = parameter.val().dims();
        let values = (0..width * channels)
            .map(|i| (((i * 19 + salt * 11 + 1) % 43) as f32 - 21.0) / 43.0 * 0.1)
            .collect::<Vec<_>>();
        *parameter = Param::from_tensor(Tensor::from_data(
            TensorData::new(values, [width, channels]), device,
        ));
    }

    fn fill_direction(layer: &mut super::Direction, device: &Device, salt: usize) {
        fill_linear(&mut layer.q, device, salt + 1);
        fill_linear(&mut layer.k, device, salt + 2);
        fill_linear(&mut layer.v, device, salt + 3);
        fill_linear(&mut layer.decay_hidden, device, salt + 4);
        fill_linear(&mut layer.decay, device, salt + 5);
        fill_linear(&mut layer.erase, device, salt + 6);
        fill_linear(&mut layer.write, device, salt + 7);
        fill_linear(&mut layer.output_gate_hidden, device, salt + 8);
        fill_linear(&mut layer.output_gate, device, salt + 9);
        fill_conv(&mut layer.q_conv, device, salt);
        fill_conv(&mut layer.k_conv, device, salt + 1);
        fill_conv(&mut layer.v_conv, device, salt + 2);
        layer.a_log = Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![0.4f32, 0.7, 1.1, 1.4], [4]), device,
        ));
        let dt_width = layer.dt_bias.dims()[0];
        let dt = (0..dt_width)
            .map(|i| (((i * 13 + salt * 5) % 37) as f32 - 18.0) / 37.0 * 0.2)
            .collect::<Vec<_>>();
        layer.dt_bias = Param::from_tensor(Tensor::from_data(TensorData::new(dt, [dt_width]), device));
        let gamma_width = layer.output_norm.gamma.dims()[0];
        layer.output_norm.gamma = Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![1.0f32; gamma_width], [gamma_width]), device,
        ));
    }

    fn fill_model(layer: &mut Gdn2Bidirectional, device: &Device) {
        fill_direction(&mut layer.forward, device, 11);
        fill_direction(&mut layer.reverse, device, 47);
        fill_linear(&mut layer.output, device, 83);
    }

    #[derive(Clone)]
    struct FModel {
        forward: FDirection,
        reverse: FDirection,
        output: FLinear,
    }

    impl FModel {
        fn from_gpu(layer: &Gdn2Bidirectional) -> Self {
            Self {
                forward: FDirection::from_gpu(&layer.forward),
                reverse: FDirection::from_gpu(&layer.reverse),
                output: FLinear::from_gpu(&layer.output),
            }
        }

        fn forward(&self, input: &[f64], tokens: usize, starts: &[i32], ends: &[i32]) -> Vec<f64> {
            let reversed_input = reverse_values(input, tokens, 512, starts, ends);
            let forward_state = self.forward.scan(input, tokens, starts, ends);
            let reverse_state = self.reverse.scan(&reversed_input, tokens, starts, ends);
            let forward = self.forward.finish(&forward_state, input, tokens);
            let reverse_reversed = self.reverse.finish(&reverse_state, &reversed_input, tokens);
            let reverse = reverse_values(&reverse_reversed, tokens, 256, starts, ends);
            let mut joined = vec![0.0; tokens * 512];
            for t in 0..tokens {
                joined[t * 512..t * 512 + 256].copy_from_slice(&forward[t * 256..(t + 1) * 256]);
                joined[t * 512 + 256..(t + 1) * 512].copy_from_slice(&reverse[t * 256..(t + 1) * 256]);
            }
            let mut output = self.output.forward(&joined, tokens);
            for t in 0..tokens {
                if ends[t] < starts[t] {
                    output[t * 512..(t + 1) * 512].fill(0.0);
                }
            }
            output
        }
    }

    fn reverse_values(x: &[f64], tokens: usize, channels: usize, starts: &[i32], ends: &[i32]) -> Vec<f64> {
        let mut result = x.to_vec();
        let mut t = 0;
        while t < tokens {
            if ends[t] < starts[t] {
                t += 1;
                continue;
            }
            let start = starts[t] as usize;
            let end = ends[t] as usize;
            for offset in 0..=end - start {
                let from = start + offset;
                let to = end - offset;
                result[to * channels..(to + 1) * channels]
                    .copy_from_slice(&x[from * channels..(from + 1) * channels]);
            }
            t = end + 1;
        }
        result
    }

    fn f64_silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    fn f64_sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn f64_softplus(x: f64) -> f64 {
        if x > 30.0 { x } else { x.exp().ln_1p() }
    }

    impl FDirection {
        fn convolution(
            &self,
            x: &[f64],
            tokens: usize,
            channels: usize,
            weights: &[f64],
            starts: &[i32],
            ends: &[i32],
        ) -> Vec<f64> {
            let mut output = vec![0.0; tokens * channels];
            for t in 0..tokens {
                if ends[t] < starts[t] { continue; }
                for lag in 0..self.conv_width {
                    if t < lag { continue; }
                    let source = t - lag;
                    if starts[source] != starts[t] || ends[source] < starts[source] { continue; }
                    for channel in 0..channels {
                        output[t * channels + channel] +=
                            x[source * channels + channel] * weights[lag * channels + channel];
                    }
                }
            }
            output
        }

        fn scan(&self, input: &[f64], tokens: usize, starts: &[i32], ends: &[i32]) -> Vec<f64> {
            let q = self.convolution(
                &self.q.forward(input, tokens), tokens, HEADS * KEY_DIM, &self.q_conv, starts, ends,
            ).into_iter().map(f64_silu).collect::<Vec<_>>();
            let k = self.convolution(
                &self.k.forward(input, tokens), tokens, HEADS * KEY_DIM, &self.k_conv, starts, ends,
            ).into_iter().map(f64_silu).collect::<Vec<_>>();
            let v = self.convolution(
                &self.v.forward(input, tokens), tokens, HEADS * VALUE_DIM, &self.v_conv, starts, ends,
            ).into_iter().map(f64_silu).collect::<Vec<_>>();
            let decay_hidden = self.decay_hidden.forward(input, tokens);
            let decay = self.decay.forward(&decay_hidden, tokens);
            let erase = self.erase.forward(input, tokens).into_iter().map(f64_sigmoid).collect::<Vec<_>>();
            let write = self.write.forward(input, tokens).into_iter().map(f64_sigmoid).collect::<Vec<_>>();
            let mut output = vec![0.0; tokens * HEADS * VALUE_DIM];
            for head in 0..HEADS {
                let mut host_tokens = Vec::with_capacity(tokens);
                for t in 0..tokens {
                    let key_base = t * HEADS * KEY_DIM + head * KEY_DIM;
                    let value_base = t * HEADS * VALUE_DIM + head * VALUE_DIM;
                    let normalize = |source: &[f64]| {
                        let values = source[key_base..key_base + KEY_DIM].to_vec();
                        let norm = (values.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
                        values.into_iter().map(|x| x / norm).collect::<Vec<_>>()
                    };
                    let alpha: Vec<f64> = (0..KEY_DIM).map(|i| {
                        let index = head * KEY_DIM + i;
                        let log_decay = -self.a_log[head].exp()
                            * f64_softplus(decay[t * HEADS * KEY_DIM + index] + self.dt_bias[index]);
                        log_decay.exp()
                    }).collect();
                    host_tokens.push(crate::reference::Token::new(
                        normalize(&q),
                        normalize(&k),
                        v[value_base..value_base + VALUE_DIM].to_vec(),
                        alpha,
                        erase[key_base..key_base + KEY_DIM].to_vec(),
                        write[value_base..value_base + VALUE_DIM].to_vec(),
                    ));
                }
                let result = crate::reference::scan(&host_tokens, starts, ends);
                for t in 0..tokens {
                    output[(t * HEADS + head) * VALUE_DIM..(t * HEADS + head + 1) * VALUE_DIM]
                        .copy_from_slice(&result.output[t]);
                }
            }
            output
        }

        fn finish(&self, state: &[f64], input: &[f64], tokens: usize) -> Vec<f64> {
            let hidden = self.output_gate_hidden.forward(input, tokens);
            let gate_pre = self.output_gate.forward(&hidden, tokens);
            let mut output = vec![0.0; tokens * HEADS * VALUE_DIM];
            for t in 0..tokens {
                for head in 0..HEADS {
                    let base = (t * HEADS + head) * VALUE_DIM;
                    let mean_square = state[base..base + VALUE_DIM].iter().map(|x| x * x).sum::<f64>()
                        / VALUE_DIM as f64;
                    let inverse_rms = (mean_square + self.norm_epsilon).sqrt().recip();
                    for channel in 0..VALUE_DIM {
                        let index = base + channel;
                        output[index] = state[index] * inverse_rms * self.norm_gamma[channel]
                            * f64_silu(gate_pre[index]);
                    }
                }
            }
            output
        }
    }

    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn gdn2_layer_matches_f64() {
        assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
        check_layer_matches_f64("serial");
    }

    #[test]
    #[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
    fn gdn2_chunked_layer_matches_f64() {
        assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
        check_layer_matches_f64("chunked");
    }

    fn check_layer_matches_f64(scan: &str) {
        let device = dre_core::gpu_device().autodiff();
        let bf16 = scan == "serial" && std::env::var_os("DRE_BF16_GATE").is_some();
        let precision = if bf16 { dre_core::Precision::Bf16Mixed } else { dre_core::Precision::Fp32 };
        for tokens in [5usize, 33] {
            let first = (tokens - 1) / 2;
            let second = tokens - first - 1;
            let layout = BatchLayout::new(
                dre_core::DocumentLayout::packed(vec![vec![first, second]], tokens),
                &device,
            );
            let starts: Vec<i32> = (0..tokens)
                .map(|t| if t < first { 0 } else if t < first + second { first as i32 } else { -1 })
                .collect();
            let ends: Vec<i32> = (0..tokens)
                .map(|t| if t < first { (first - 1) as i32 } else if t < first + second { (first + second - 1) as i32 } else { -2 })
                .collect();
            let input_values: Vec<f32> = (0..tokens * 512)
                .map(|i| if ends[i / 512] < starts[i / 512] { 0.0 } else { ((i * 19 + 7) % 127) as f32 / 127.0 - 0.5 })
                .collect();
            let input_f64: Vec<f64> = input_values.iter().map(|&x| f64::from(x)).collect();
            let input = Tensor::<3>::from_data(
                TensorData::new(input_values.clone(), [1, tokens, 512]),
                &device,
            ).require_grad();
            let mut layer = Gdn2BidirectionalConfig::new().with_precision(precision).with_scan(Some(scan.to_string())).init(&device);
            fill_model(&mut layer, &device);
            let oracle = FModel::from_gpu(&layer);
            let expected = oracle.forward(&input_f64, tokens, &starts, &ends);
            let output = layer.forward(input.clone(), &layout);
            let actual = output.clone().into_data().try_to_vec::<f32>().unwrap();
            let mut squared = 0.0;
            let mut baseline_squared = 0.0;
            let valid_count = ends.iter().filter(|&&end| end >= 0).count() * 512;
            for t in 0..tokens {
                for channel in 0..512 {
                    let index = t * 512 + channel;
                    if ends[t] < starts[t] {
                        assert_eq!(actual[index], 0.0);
                    } else {
                        squared += expected[index] * expected[index];
                        baseline_squared += f64::from(actual[index]).powi(2);
                    }
                }
            }
            let rms = (squared / valid_count as f64).sqrt();
            let s0 = (baseline_squared / valid_count as f64).sqrt();
            let atol = if bf16 { 1e-2 } else { 1e-5 } * rms.max(s0);
            let mut max_abs = 0.0f64;
            let mut max_rel = 0.0f64;
            let mut passed = true;
            for (index, (&actual, &reference)) in actual.iter().zip(&expected).enumerate() {
                if ends[index / 512] < starts[index / 512] { continue; }
                let error = (f64::from(actual) - reference).abs();
                max_abs = max_abs.max(error);
                if reference.abs() > atol { max_rel = max_rel.max(error / reference.abs()); }
                passed &= error <= atol + if bf16 { 1e-2 } else { 1e-4 } * reference.abs();
            }
            println!("GDN2 layer f64 T={tokens}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, rms={rms:.3e}, s0={s0:.3e}, atol={atol:.3e}");
            assert!(passed, "T={tokens}, full-layer output differs from f64 oracle");
            let grads = output.sum().backward();
            let input_grad = input.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();

            let gradient_norm = input_grad.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            assert!(gradient_norm > 0.0, "T={tokens} input-gradient fixture must be non-degenerate");
            let h = 1e-6 * (input_f64.iter().map(|x| x * x).sum::<f64>() / input_f64.len() as f64).sqrt().max(1.0);
            for direction in 0..8 {
                let mut vector: Vec<f64> = (0..input_f64.len())
                    .map(|i| (((i * 31 + direction * 47 + 13) % 101) as f64 - 50.0) / 50.0)
                    .collect();
                let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
                for value in &mut vector { *value /= norm; }
                let mut plus = input_f64.clone();
                let mut minus = input_f64.clone();
                for i in 0..vector.len() {
                    plus[i] += h * vector[i];
                    minus[i] -= h * vector[i];
                }
                let derivative = (oracle.forward(&plus, tokens, &starts, &ends).iter().sum::<f64>()
                    - oracle.forward(&minus, tokens, &starts, &ends).iter().sum::<f64>()) / (2.0 * h);
                let gpu_directional = input_grad.iter().zip(&vector)
                    .map(|(&g, &v)| f64::from(g) * v).sum::<f64>();
                let tolerance = if bf16 { 1e-2 } else { 1e-7 } * gradient_norm
                    + if bf16 { 1e-2 } else { 1e-6 } * derivative.abs();
                println!("GDN2 directional T={tokens} BF16={bf16} direction={direction} error={:.3e} tolerance={tolerance:.3e}", (gpu_directional - derivative).abs());
                assert!((gpu_directional - derivative).abs() <= tolerance,
                    "T={tokens}, input direction {direction}: gpu={gpu_directional}, f64={derivative}, tolerance={tolerance}");
            }

            let parameter_checks = [
                (0usize, 0usize),
                (1usize, 32 * 256 + 91),
                (2usize, 512 + 19),
                (3usize, 13usize),
                (4usize, 0usize),
                (5usize, 17usize),
                (6usize, 537usize),
                (7usize, 29usize),
                (8usize, 1usize),
            ];
            for (group, index) in parameter_checks {
                let (gradient, base) = match group {
                    0 => (layer.forward.q.weight.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.forward.q.weight[index]),
                    1 => (layer.reverse.output_gate.weight.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.reverse.output_gate.weight[index]),
                    2 => (layer.output.weight.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.output.weight[index]),
                    3 => (layer.forward.q_conv.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.forward.q_conv[index]),
                    4 => (layer.forward.a_log.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.forward.a_log[index]),
                    5 => (layer.forward.dt_bias.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.forward.dt_bias[index]),
                    6 => (layer.reverse.q.weight.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.reverse.q.weight[index]),
                    7 => (layer.reverse.q_conv.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.reverse.q_conv[index]),
                    _ => (layer.reverse.a_log.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap(), oracle.reverse.a_log[index]),
                };
                let mut plus = oracle.clone();
                let mut minus = oracle.clone();
                match group {
                    0 => { plus.forward.q.weight[index] = base + h; minus.forward.q.weight[index] = base - h; }
                    1 => { plus.reverse.output_gate.weight[index] = base + h; minus.reverse.output_gate.weight[index] = base - h; }
                    2 => { plus.output.weight[index] = base + h; minus.output.weight[index] = base - h; }
                    3 => { plus.forward.q_conv[index] = base + h; minus.forward.q_conv[index] = base - h; }
                    4 => { plus.forward.a_log[index] = base + h; minus.forward.a_log[index] = base - h; }
                    5 => { plus.forward.dt_bias[index] = base + h; minus.forward.dt_bias[index] = base - h; }
                    6 => { plus.reverse.q.weight[index] = base + h; minus.reverse.q.weight[index] = base - h; }
                    7 => { plus.reverse.q_conv[index] = base + h; minus.reverse.q_conv[index] = base - h; }
                    _ => { plus.reverse.a_log[index] = base + h; minus.reverse.a_log[index] = base - h; }
                }
                let numerical = (plus.forward(&input_f64, tokens, &starts, &ends).iter().sum::<f64>()
                    - minus.forward(&input_f64, tokens, &starts, &ends).iter().sum::<f64>()) / (2.0 * h);
                let gradient_rms = (gradient.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                    / gradient.len() as f64).sqrt();
                let tolerance = if bf16 { 1e-2 } else { 1e-5 } * gradient_rms
                    + if bf16 { 1e-2 } else { 1e-4 } * numerical.abs();
                println!("GDN2 parameter T={tokens} BF16={bf16} group={group} error={:.3e} tolerance={tolerance:.3e}", (f64::from(gradient[index]) - numerical).abs());
                assert!((f64::from(gradient[index]) - numerical).abs() <= tolerance,
                    "T={tokens}, parameter group {group}[{index}] gpu={}, f64={numerical}, tolerance={tolerance}", gradient[index]);
            }
        }
    }
}
