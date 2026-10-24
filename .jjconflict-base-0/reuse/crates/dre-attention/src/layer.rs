use burn::nn::{Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig};
use burn::prelude::*;
use dre_core::BatchLayout;

/// Number of document-local positions available in attention rotary tables.
pub const ROTARY_MAX_TOKENS: usize = 4096;

/// Dimensions for the document-local attention mixer.
#[derive(Config, Debug)]
pub struct LocalAttentionConfig {
    #[config(default = "dre_core::Precision::Fp32")]
    pub precision: dre_core::Precision,
    /// Number of attention heads.
    #[config(default = 8)]
    pub heads: usize,
    /// Width of each attention head.
    #[config(default = 64)]
    pub head_dim: usize,
    /// Maximum token distance in either direction.
    #[config(default = 128)]
    pub radius: usize,
    /// Model width.
    #[config(default = 512)]
    pub d: usize,
}

impl LocalAttentionConfig {
    /// Initializes the projection and rotary parameters.
    pub fn init(&self, device: &Device) -> LocalAttention {
        assert_eq!(self.d, self.heads * self.head_dim, "model width must equal heads times head width");
        LocalAttention {
            mixed: self.precision == dre_core::Precision::Bf16Mixed,
            query: LinearConfig::new(self.d, self.d).init(device),
            key: LinearConfig::new(self.d, self.d).init(device),
            value: LinearConfig::new(self.d, self.d).init(device),
            output: LinearConfig::new(self.d, self.d).init(device),
            rotary: RotaryEncodingConfig::new(ROTARY_MAX_TOKENS, self.head_dim).init(device),
            heads: self.heads,
            head_dim: self.head_dim,
            radius: self.radius,
            d: self.d,
        }
    }
}

/// Multi-head bidirectional local attention.
#[derive(Module, Debug)]
pub struct LocalAttention {
    mixed: bool,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    rotary: RotaryEncoding,
    heads: usize,
    head_dim: usize,
    radius: usize,
    d: usize,
}

impl LocalAttention {
    /// Projects, rotates, attends within each document, then projects to model width.
    pub fn forward(&self, u: Tensor<3>, layout: &BatchLayout) -> Tensor<3> {
        let [batch, tokens, width] = u.dims();
        assert_eq!([batch, tokens, width], [layout.host.batch(), layout.host.tokens(), self.d]);
        assert_eq!(self.heads * self.head_dim, self.d);
        let shape = [batch, tokens, self.heads, self.head_dim];
        let invalid=layout.valid.clone().lower_elem(0.5);
        let u=u.mask_fill(invalid.clone(),0.0);
        let q = dre_core::project(&self.query, u.clone(), self.mixed).reshape(shape).swap_dims(1, 2);
        let k = dre_core::project(&self.key, u.clone(), self.mixed).reshape(shape).swap_dims(1, 2);
        let v = dre_core::project(&self.value, u, self.mixed).mask_fill(invalid.clone(),0.0).reshape(shape).swap_dims(1, 2);
        let (q,k) = rotate_document_sources(&self.rotary,q,k,layout);
        // BF16 local-kernel storage failed the original f64 cancellation fixture.
        // Keep just this kernel boundary FP32; the dense projections remain mixed.
        // See docs/references/precision.md for the measured gate.
        #[cfg(any(test, feature = "cpu-oracle"))]
        let attended = if q.clone().inner().device()==Device::ndarray() {
            crate::oracle::oracle_windowed_attention(q,k,v,layout,self.radius)
        } else {crate::kernel::windowed_attention(q,k,v,layout,self.radius)};
        #[cfg(not(any(test, feature = "cpu-oracle")))]
        let attended = crate::kernel::windowed_attention(q,k,v,layout,self.radius);
        let merged = attended.swap_dims(1, 2).reshape([batch, tokens, self.d]).mask_fill(invalid.clone(),0.0);
        dre_core::project(&self.output, merged, self.mixed).mask_fill(invalid,0.0)
    }

}

/// Gather the existing adjacent-pair RoPE coefficients once for both source projections.
/// Physical padding uses position zero and is removed after rotation; relay projections are not rotated.
pub(crate) fn rotate_document_sources(rotary:&RotaryEncoding,q:Tensor<4>,k:Tensor<4>,layout:&BatchLayout)->(Tensor<4>,Tensor<4>) {
    let [batch,heads,tokens,width]=q.dims();
    assert_eq!(k.dims(),q.dims());
    let coefficients=rotary.freq_complex.clone().select(0,layout.position_ids.clone().reshape([batch*tokens]))
        .reshape([batch,1,tokens,width,2]);
    let sign=Tensor::<2>::from_floats([[1.0,0.0,0.0,1.0],[0.0,-1.0,1.0,0.0]],&q.device())
        .reshape([1,1,1,2,4]);
    let invalid=layout.valid.clone().lower_elem(0.5).reshape([batch,1,tokens,1]);
    let rotate=|input:Tensor<4>| {
        let paired=input.mask_fill(invalid.clone(),0.0).reshape([batch,heads,tokens,width/2,2]).matmul(sign.clone())
            .reshape([batch,heads,tokens,width,2]);
        (paired*coefficients.clone()).sum_dim(4).reshape([batch,heads,tokens,width]).mask_fill(invalid.clone(),0.0)
    };
    (rotate(q),rotate(k))
}

#[cfg(test)]
mod padding_tests {
    use super::*;
    use burn::module::{Module,ModuleVisitor,Param};
    use dre_core::DocumentLayout;
    fn host<const D:usize>(tensor:&Tensor<D>)->Vec<f32> {tensor.to_data().try_to_vec_as::<f32>().unwrap()}
    struct Gradients<'a> {grads:&'a burn::tensor::Gradients,values:Vec<Option<Vec<f32>>>}
    impl ModuleVisitor for Gradients<'_> {
        fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<D>>) {
            self.values.push(param.val().grad(self.grads).map(|value|host(&value)));
        }
    }
    fn close(actual:&[f32],expected:&[f32]) {
        assert_eq!(actual.len(),expected.len());
        for (&a,&b) in actual.iter().zip(expected) {
            assert!(a.is_finite()&&b.is_finite()&&(a-b).abs()<2e-5+1e-4*b.abs(),"{a} != {b}");
        }
    }
    #[test]
    fn local_attention_nan_padding_outputs_and_gradients_cpu() {
        let device=Device::ndarray().autodiff();
        let model=LocalAttentionConfig::new().with_d(4).with_heads(1).with_head_dim(4).with_radius(2).init(&device);
        let layout=BatchLayout::new(DocumentLayout::packed(vec![vec![2,1],vec![]],4),&device);
        let run=|dirty:bool| {
            let input=Tensor::<3>::from_data(TensorData::new((0..32).map(|index| {
                if index>=12 {if dirty {f32::NAN}else{0.0}}else{(index as f32*0.17).sin()}
            }).collect::<Vec<_>>(),[2,4,4]),&device).require_grad();
            let output=model.forward(input.clone(),&layout);
            let values=host(&output);
            assert!(values.iter().all(|value|value.is_finite()));
            assert!(values[12..].iter().all(|&value|value==0.0));
            let weights=Tensor::from_data(TensorData::new((0..32).map(|index|1.0+(index%5) as f32).collect::<Vec<_>>(),[2,4,4]),&device);
            let grads=(output*weights).sum().backward();
            let input_gradient=host(&input.grad(&grads).unwrap());
            assert!(input_gradient[..12].iter().any(|value|value.abs()>1e-8));
            assert!(input_gradient[12..].iter().all(|&value|value==0.0));
            let mut visitor=Gradients {grads:&grads,values:Vec::new()};model.visit(&mut visitor);
            (values,input_gradient,visitor.values)
        };
        let clean=run(false);let dirty=run(true);
        close(&dirty.0,&clean.0);close(&dirty.1,&clean.1);
        assert_eq!(dirty.2.len(),clean.2.len());
        for (dirty,clean) in dirty.2.iter().zip(&clean.2) {match (dirty,clean) {
            (Some(dirty),Some(clean))=>close(dirty,clean),(None,None)=>{},_=>panic!("parameter gradient presence changed"),
        }}
    }
}
