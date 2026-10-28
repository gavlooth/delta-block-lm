//! Local source attention with document-local global relay positions.

use burn::nn::{Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig};
use burn::prelude::*;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use dre_core::BatchLayout;

#[derive(Config, Debug)]
pub struct LocalGlobalRelayAttentionConfig {
    #[config(default = "dre_core::Precision::Fp32")]
    pub precision: dre_core::Precision,
    #[config(default = 8)]
    pub heads: usize,
    #[config(default = 64)]
    pub head_dim: usize,
    #[config(default = 128)]
    pub radius: usize,
    #[config(default = 512)]
    pub d: usize,
}

impl LocalGlobalRelayAttentionConfig {
    pub fn init(&self, device: &Device) -> LocalGlobalRelayAttention {
        assert_eq!(self.d, self.heads * self.head_dim);
        LocalGlobalRelayAttention {
            mixed: self.precision == dre_core::Precision::Bf16Mixed,
            source_query: LinearConfig::new(self.d, self.d).init(device),
            source_key: LinearConfig::new(self.d, self.d).init(device),
            source_value: LinearConfig::new(self.d, self.d).init(device),
            source_output: LinearConfig::new(self.d, self.d).init(device),
            relay_query: LinearConfig::new(self.d, self.d).init(device),
            relay_key: LinearConfig::new(self.d, self.d).init(device),
            relay_value: LinearConfig::new(self.d, self.d).init(device),
            relay_output: LinearConfig::new(self.d, self.d).init(device),
            rotary: RotaryEncodingConfig::new(crate::layer::ROTARY_MAX_TOKENS, self.head_dim).init(device),
            heads: self.heads,
            head_dim: self.head_dim,
            radius: self.radius,
            d: self.d,
        }
    }
}

/// Hybrid local/global attention. Source outputs use the CubeCL online-softmax window kernel; its score storage
/// is linear in source length. Relay queries use one document-batched read over source rows and their own relays.
#[derive(Module, Debug)]
pub struct LocalGlobalRelayAttention {
    mixed: bool,
    source_query: Linear,
    source_key: Linear,
    source_value: Linear,
    source_output: Linear,
    relay_query: Linear,
    relay_key: Linear,
    relay_value: Linear,
    relay_output: Linear,
    rotary: RotaryEncoding,
    heads: usize,
    head_dim: usize,
    radius: usize,
    d: usize,
}

impl LocalGlobalRelayAttention {
    /// Inputs are source readouts `[B,T,d]` and document-indexed relay readouts `[D,16,d]`.
    /// Outputs have the same respective shapes. The host document ordering indexes relay rows.
    pub fn forward(&self, source: Tensor<3>, relays: Tensor<3>, layout: &BatchLayout) -> (Tensor<3>, Tensor<3>) {
        let [batch, tokens, width] = source.dims();
        let [documents, relay_count, relay_width] = relays.dims();
        assert_eq!([batch, tokens, width], [layout.host.batch(), layout.host.tokens(), self.d]);
        assert_eq!(relay_width, self.d);
        assert_eq!(relay_count, 16);
        assert_eq!(documents, layout.host.document_capacity());
        assert_eq!(self.heads * self.head_dim, self.d);

        let source_invalid=layout.valid.clone().lower_elem(0.5);
        let relay_invalid=layout.document_valid.clone().lower_elem(0.5);
        let source=source.mask_fill(source_invalid.clone(),0.0);
        let relays=relays.mask_fill(relay_invalid.clone(),0.0);
        let shape = [batch, tokens, self.heads, self.head_dim];
        let source_q = dre_core::project(&self.source_query, source.clone(), self.mixed).reshape(shape).swap_dims(1, 2);
        let source_k = dre_core::project(&self.source_key, source.clone(), self.mixed).reshape(shape).swap_dims(1, 2);
        let source_v = dre_core::project(&self.source_value, source, self.mixed)
            .mask_fill(source_invalid.clone(),0.0).reshape(shape).swap_dims(1, 2);
        let relay_shape = [documents, relay_count, self.heads, self.head_dim];
        let (source_q,source_k) = crate::layer::rotate_document_sources(&self.rotary,source_q,source_k,layout);
        let relay_q = dre_core::project(&self.relay_query, relays.clone(), self.mixed)
            .mask_fill(relay_invalid.clone(),0.0).reshape(relay_shape).swap_dims(1, 2);
        let relay_k = dre_core::project(&self.relay_key, relays.clone(), self.mixed)
            .mask_fill(relay_invalid.clone(),0.0).reshape(relay_shape).swap_dims(1, 2);
        let relay_v = dre_core::project(&self.relay_value, relays, self.mixed)
            .mask_fill(relay_invalid.clone(),0.0).reshape(relay_shape).swap_dims(1, 2);

        assert_eq!(self.radius, 128);
        let dtype = if self.mixed { burn::tensor::DType::BF16 } else { burn::tensor::DType::F32 };
        let source_outputs = source_windowed_attention(
            source_q.cast(dtype),source_k.clone().cast(dtype),source_v.clone().cast(dtype),
            relay_k.clone().cast(dtype),relay_v.clone().cast(dtype),layout,
        );
        let relay_outputs = relay_read(relay_q,source_k,source_v,relay_k,relay_v,layout);
        let source_merged = source_outputs.cast(burn::tensor::DType::F32).swap_dims(1, 2)
            .reshape([batch, tokens, self.d]).mask_fill(source_invalid.clone(),0.0);
        let source_output = dre_core::project(&self.source_output, source_merged, self.mixed)
            .mask_fill(source_invalid,0.0);
        let relay_merged = relay_outputs.swap_dims(1, 2).reshape([documents, relay_count, self.d])
            .mask_fill(relay_invalid.clone(),0.0);
        (source_output, dre_core::project(&self.relay_output, relay_merged, self.mixed).mask_fill(relay_invalid,0.0))
    }

}

fn source_windowed_attention(q:Tensor<4>,k:Tensor<4>,v:Tensor<4>,relay_k:Tensor<4>,relay_v:Tensor<4>,layout:&BatchLayout)->Tensor<4> {
    #[cfg(any(test, feature = "cpu-oracle"))]
    if q.clone().inner().device()==Device::ndarray() {
        return crate::oracle::burn_source_oracle(q,k,v,relay_k,relay_v,layout);
    }
    crate::relay_attention_kernel::run_relay_windowed_attention(q,k,v,relay_k,relay_v,layout).0
}

/// Burn 0.22's generic flash-attention mask reader requires physical dense mask storage.
pub(crate) fn materialized_attention_mask(mask:Tensor<4,Bool>,shape:[usize;4])->Tensor<4,Bool> {
    // Flattening repeated key vectors cannot preserve the broadcast's zero strides. Cube's
    // reshape therefore copies every logical mask element once; restoring the shape is a view.
    // An already dense mask needs only view changes. Singleton repeat is not a physical copy.
    mask.expand(shape).reshape([shape.iter().product::<usize>()]).reshape(shape)
}

fn relay_outside_mask(layout:&BatchLayout,heads:usize,relays:usize,device:&Device)->Tensor<4,Bool> {
    let documents=layout.host.document_capacity();
    let keys=layout.host.tokens()+relays;
    let relay_outside=Tensor::<3,Bool>::full([documents,1,relays],false,device);
    let shape=[documents,heads,relays,keys];
    let outside=Tensor::cat(vec![layout.document_outside.clone(),relay_outside],2)
        .reshape([documents,1,1,keys]);
    materialized_attention_mask(outside,shape)
}

/// The document axis is executable padding; only source positions from each real interval and its own 16 relays are keys.
fn relay_read(query:Tensor<4>,source_k:Tensor<4>,source_v:Tensor<4>,relay_k:Tensor<4>,relay_v:Tensor<4>,layout:&BatchLayout)->Tensor<4> {
    let [documents,heads,relays,width]=query.dims();
    let tokens=source_k.dims()[2];
    let keys=Tensor::cat(vec![source_k.select(0,layout.document_rows.clone()),relay_k],2);
    let values=Tensor::cat(vec![source_v.select(0,layout.document_rows.clone()),relay_v],2);
    let outside=relay_outside_mask(layout,heads,relays,&query.device());
    assert_eq!(keys.dims(),[documents,heads,tokens+relays,width]);
    attention(query,keys,values,Some(outside),None,AttentionModuleOptions::default())
}

#[cfg(test)]
#[path="relay_batch_tests.rs"]
mod batch_tests;

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
    fn relay_attention_nan_source_and_document_padding_outputs_and_gradients_cpu() {
        let device=Device::ndarray().autodiff();
        let model=LocalGlobalRelayAttentionConfig::new().with_d(4).with_heads(1).with_head_dim(4).init(&device);
        let layout=BatchLayout::new(DocumentLayout::packed(vec![vec![2,1],vec![]],4).with_document_padding(4),&device);
        let run=|dirty:bool| {
            let source=Tensor::<3>::from_data(TensorData::new((0..32).map(|index| {
                if index>=12 {if dirty {f32::NAN}else{0.0}}else{(index as f32*0.13).sin()}
            }).collect::<Vec<_>>(),[2,4,4]),&device).require_grad();
            let relays=Tensor::<3>::from_data(TensorData::new((0..4*16*4).map(|index| {
                if index>=2*16*4 {if dirty {f32::NAN}else{0.0}}else{(index as f32*0.07).cos()}
            }).collect::<Vec<_>>(),[4,16,4]),&device).require_grad();
            let (source_output,relay_output)=model.forward(source.clone(),relays.clone(),&layout);
            let source_values=host(&source_output);let relay_values=host(&relay_output);
            assert!(source_values.iter().chain(&relay_values).all(|value|value.is_finite()));
            assert!(source_values[12..].iter().all(|&value|value==0.0));
            assert!(relay_values[2*16*4..].iter().all(|&value|value==0.0));
            let grads=(source_output.square().sum()+relay_output.square().sum()).backward();
            let source_gradient=host(&source.grad(&grads).unwrap());let relay_gradient=host(&relays.grad(&grads).unwrap());
            assert!(source_gradient[..12].iter().any(|value|value.abs()>1e-8));
            assert!(relay_gradient[..2*16*4].iter().any(|value|value.abs()>1e-8));
            assert!(source_gradient[12..].iter().all(|&value|value==0.0));
            assert!(relay_gradient[2*16*4..].iter().all(|&value|value==0.0));
            let mut visitor=Gradients {grads:&grads,values:Vec::new()};model.visit(&mut visitor);
            (source_values,relay_values,source_gradient,relay_gradient,visitor.values)
        };
        let clean=run(false);let dirty=run(true);
        close(&dirty.0,&clean.0);close(&dirty.1,&clean.1);close(&dirty.2,&clean.2);close(&dirty.3,&clean.3);
        assert_eq!(dirty.4.len(),clean.4.len());
        for (dirty,clean) in dirty.4.iter().zip(&clean.4) {match (dirty,clean) {
            (Some(dirty),Some(clean))=>close(dirty,clean),(None,None)=>{},_=>panic!("parameter gradient presence changed"),
        }}
    }
}
