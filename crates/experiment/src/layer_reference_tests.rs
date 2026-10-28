use super::*;
use crate::{config::{Arm, Config, SbdConfig, CocommitConfig}, model::{Block, Sublayer}, sbd};
use burn::{module::{ModuleVisitor, Param}, optim::GradientsParams};
use delta_mixers::{lngram::LngramLayout, profile::Unprofiled};

// Independent residual and normalization equations, shared only by the test's input layouts.
fn reference_layer(model: &LanguageModel, block: &Block, mut x: NoisyRepresentation, valid: Tensor<2>, mut f: impl FnMut(Sublayer, Tensor<3>) -> Tensor<3>) -> NoisyRepresentation {
    let mut apply = |x: NoisyRepresentation, stage: Option<&delta_mixers::ohc::OhcStage>, kind: Option<Sublayer>| {
        let work = match &x {
            NoisyRepresentation::Hidden(x) => x.clone(),
            NoisyRepresentation::Streams(x) => stage.unwrap().read_selected(x.clone(),model.ohc_backend).work,
        };
        let input = match kind {
            Some(Sublayer::Wavelet) => {
                let norm = block.wavelet_norm.as_ref().unwrap();
                let d = work.dims()[2];
                let centered = work.clone() - work.mean_dim(2);
                let variance = centered.clone().square().mean_dim(2);
                centered / (variance + 1e-6).sqrt() * norm.gamma.val().reshape([1,1,d])
                    + norm.beta.as_ref().unwrap().val().reshape([1,1,d])
            }
            Some(Sublayer::Lngram) => work,
            Some(Sublayer::Mixer) => block.norm.forward(work),
            None => block.ff_norm.forward(work),
        };
        let update = match kind {
            Some(kind) => f(kind,input),
            None => block.down.forward(model.epilogue_backend.silu(block.up.forward(input))),
        };
        match x {
            NoisyRepresentation::Hidden(x) => NoisyRepresentation::Hidden(x+update),
            NoisyRepresentation::Streams(x) => {
                let stage=stage.unwrap();
                let read=stage.read_selected(x,model.ohc_backend);
                NoisyRepresentation::Streams(stage.write(read,update,valid.clone(),model.ohc_backend))
            }
        }
    };
    if block.wavelet.is_some() {x=apply(x,block.ohc_wavelet.as_ref(),Some(Sublayer::Wavelet));}
    if block.lngram.is_some() {x=apply(x,block.ohc_lngram.as_ref(),Some(Sublayer::Lngram));}
    x=apply(x,block.ohc_mixer.as_ref(),Some(Sublayer::Mixer));
    apply(x,block.ohc_mlp.as_ref(),None)
}

fn initial(model: &LanguageModel, hidden: Tensor<3>, valid: Tensor<2>) -> NoisyRepresentation {
    match &model.ohc_init {
        Some(init)=>NoisyRepresentation::Streams(init.forward(hidden,valid)),
        None=>NoisyRepresentation::Hidden(hidden),
    }
}
fn reference_full(model: &LanguageModel, hidden: Tensor<3>, batch: &crate::data::Batch, start: Option<usize>) -> DenoisingOutput {
    let mut x=initial(model,hidden,batch.valid.clone());
    let layout=LngramLayout::new(batch.valid.clone(),batch.reset.clone());
    for block in &model.blocks {
        x=reference_layer(model,block,x,batch.valid.clone(),|kind,input|match kind {
            Sublayer::Wavelet=>block.wavelet.as_ref().unwrap().forward_profiled(input,batch.valid.clone(),batch.reset.clone(),&mut Unprofiled),
            Sublayer::Lngram=>block.lngram.as_ref().unwrap().forward(input,&layout),
            Sublayer::Mixer=>match start {
                Some(start)=>block.mixer.forward_sbd(input,batch.valid.clone(),batch.reset.clone(),start),
                None=>block.mixer.forward_masked(input,batch.valid.clone(),batch.reset.clone()),
            }
        });
    }
    model.output_from_representation(x)
}
fn reference_cached(model: &LanguageModel, cache: &TrainingCache, soft: &SoftState, marks: bool) -> DenoisingOutput {
    let n=soft.observed.len();
    let d=model.embedding.weight.dims()[1];
    let device=model.embedding.weight.val().device();
    let valid=Tensor::ones([1,n],&device);
    let hidden=soft.embeddings(model.embedding.weight.val(),model.sbd_mask.as_ref().unwrap().val(),false).reshape([1,n,d]);
    let mut x=initial(model,hidden,valid.clone());
    let start=cache.starts[0];
    for (index,block) in model.blocks.iter().enumerate() {
        if marks && index==model.blocks.len()-1 {
            let marker=model.cocommit_marker.as_ref().unwrap().val();
            x=match x {
                NoisyRepresentation::Hidden(x)=>NoisyRepresentation::Hidden(x+marker.reshape([1,1,d])),
                NoisyRepresentation::Streams(x)=>NoisyRepresentation::Streams(x+marker.reshape([1,1,4,d])),
            };
        }
        let clean=&cache.layers[index];
        x=reference_layer(model,block,x,valid.clone(),|kind,input|match kind {
            Sublayer::Wavelet=>block.wavelet.as_ref().unwrap().forward_sbd_cached(input,clean.wavelet.as_ref().unwrap(),start),
            Sublayer::Lngram=>{
                let lngram=block.lngram.as_ref().unwrap();
                let history=start.min(lngram.orders.iter().copied().max().unwrap()-1+lngram.smoothing.as_ref().map_or(0,|s|3*s.dilation));
                let combined=Tensor::cat(vec![clean.lngram.as_ref().unwrap().clone().slice([0..1,start-history..start,0..d]),input],1);
                let layout=LngramLayout::new(Tensor::ones([1,history+n],&device),Tensor::zeros([1,history+n],&device));
                lngram.forward(combined,&layout).slice([0..1,history..history+n,0..d])
            }
            Sublayer::Mixer=>block.mixer.forward_sbd_cached(input,&clean.mixer,0,valid.clone()),
        });
    }
    model.output_from_representation(x)
}
fn assert_output_and_gradients(model: &LanguageModel, actual: DenoisingOutput, expected: DenoisingOutput) {
    for (actual,expected) in [(actual.logits.clone(),expected.logits.clone()),(actual.features.clone(),expected.features.clone())] {
        sbd::compare(&crate::kron_benchmark::values(actual),&crate::kron_benchmark::values(expected).into_iter().map(f64::from).collect::<Vec<_>>()).unwrap();
    }
    let objective=|output:DenoisingOutput|output.logits.square().mean()+output.features.square().mean();
    let actual=GradientsParams::from_grads(objective(actual).backward(),model);
    let expected=GradientsParams::from_grads(objective(expected).backward(),model);
    struct Compare<'a>(&'a GradientsParams,&'a GradientsParams,usize);
    impl ModuleVisitor for Compare<'_> {
        fn visit_float<const D:usize>(&mut self,p:&Param<Tensor<D>>) {
            match (self.0.get::<D>(p.id),self.1.get::<D>(p.id)) {
                (Some(a),Some(b))=>{
                    sbd::compare(&crate::kron_benchmark::values(a),&crate::kron_benchmark::values(b).into_iter().map(f64::from).collect::<Vec<_>>()).unwrap();
                    self.2+=1;
                }
                (None,None)=>{},
                _=>panic!("gradient presence differs for {:?}",p.id),
            }
        }
    }
    let mut compare=Compare(&actual,&expected,0);
    model.visit(&mut compare);
    assert!(compare.2>20);
}

#[test]
fn shared_layer_matches_paper_equations_and_vjps_on_every_layout() {
    for ohc in [false,true] {
        let device=Device::ndarray().autodiff();
        device.seed(481);
        let c=Config {task:"lm".into(),d_model:8,layers:2,heads:2,key_dim:2,value_dim:2,chunk_size:4,cache_capacity:2,
            wavelet_layers:vec![0,1],lngram_layers:vec![0,1],lngram_routes:8,lngram_bits:1,lngram_memory_dim:2,lngram_orders:vec![2,3],ohc_streams:ohc,
            sbd:Some(SbdConfig {cocommit:Some(CocommitConfig::default()),..Default::default()}),..Default::default()};
        let arm=Arm {name:"kronlstm-hola".into(),mixer:"kronlstm".into(),cache:"surprise".into()};
        let mut model=LanguageModel::new(&c,&arm,&device).unwrap();
        // Nontrivial affines and memory/smoothing parameters expose extra norms,
        // mean/variance mistakes and missing clean-prefix fourth-tap history.
        for block in &mut model.blocks {
            let norm=block.wavelet_norm.as_mut().unwrap();
            norm.gamma=Param::from_tensor(Tensor::from_data([0.7f32,1.2,0.8,1.1,0.9,1.3,1.4,0.6],&device));
            norm.beta=Some(Param::from_tensor(Tensor::from_data([0.2f32,-0.1,0.3,0.4,-0.2,0.1,-0.3,0.5],&device)));
            let lngram=block.lngram.as_mut().unwrap();
            for table in &mut lngram.tables {
                table.weight=Param::from_tensor(Tensor::random(table.weight.dims(),burn::tensor::Distribution::Uniform(-0.2,0.2),&device));
            }
            let smoother=lngram.smoothing.as_mut().unwrap();
            smoother.weight=Param::from_tensor(Tensor::random([4,8],burn::tensor::Distribution::Uniform(-0.2,0.2),&device));
        }
        model.materialize_parameters();
        let prefix=[256,11,23,37,41,53,61,67,71,73,79,83,89];
        let batch=sbd::prefix_batch(&prefix,&device);
        let expected=reference_full(&model,model.embedding.forward(batch.input.clone()),&batch,None);
        assert_output_and_gradients(&model,model.forward_output_profiled(&batch,&mut Unprofiled),expected);
        let soft=sbd::state(&c,vec![None;3],&device).unwrap();
        let hidden=Tensor::cat(vec![model.embedding.forward(batch.input.clone()),soft.embeddings(model.embedding.weight.val(),model.sbd_mask.as_ref().unwrap().val(),false).reshape([1,3,8])],1);
        let full=sbd::prefix_batch(&[256,11,23,37,41,53,61,67,71,73,79,83,89,0,0,0],&device);
        let expected_hidden=Tensor::cat(vec![model.embedding.forward(batch.input.clone()),soft.embeddings(model.embedding.weight.val(),model.sbd_mask.as_ref().unwrap().val(),false).reshape([1,3,8])],1);
        let expected=reference_full(&model,expected_hidden,&full,Some(prefix.len()));
        assert_output_and_gradients(&model,model.forward_noisy(hidden,&full,prefix.len()),expected);
        // Each backward consumes its own cache graph; sharing tensor values would
        // also share the clean-forward tape between the two compared objectives.
        for layout in 0..4 {
            let mut trace=CleanInputs::default();
            let _=model.forward_profiled(&full,&mut trace);
            let cache=model.training_cache(&trace,0,&[prefix.len()]);
            let batched=model.training_cache_batched(&trace,&[(0,prefix.len())]);
            let actual=match layout {
                0=>model.cached_block(&cache,0,&soft,false),
                1=>model.cached_blocks(&batched,&soft,false),
                2=>{
                    let noisy=model.cached_block_prefix_profiled(&cache,0,&soft,false,&mut Unprofiled);
                    model.cached_block_tail_profiled(&cache,0,noisy,Some(Tensor::ones([1,3],&device)),&mut Unprofiled)
                }
                _=>{
                    let noisy=model.cached_blocks_prefix_profiled(&batched,&soft,false,&mut Unprofiled);
                    model.cached_blocks_tail_profiled(&batched,noisy,Some(Tensor::ones([1,3],&device)),&mut Unprofiled)
                }
            };
            let mut reference_trace=CleanInputs::default();
            let _=model.forward_profiled(&full,&mut reference_trace);
            let reference_cache=model.training_cache(&reference_trace,0,&[prefix.len()]);
            assert_output_and_gradients(&model,actual,reference_cached(&model,&reference_cache,&soft,layout>=2));
        }
    }
}

#[test]
fn wavelet_layer_norm_matches_scalar_mean_variance_affines_and_vjp() {
    let device=Device::ndarray().autodiff();
    let raw=[2.0f32,4.0,7.0,9.0, 3.0,3.0,3.0,3.0, -4.0,1.0,2.0,8.0];
    let scale=[0.7f32,1.2,0.9,1.4];
    let bias=[0.2f32,-0.1,0.3,-0.4];
    let upstream=[0.2f32,-0.3,0.5,0.7, -0.4,0.2,0.1,-0.6, 0.1,0.9,-0.2,0.3];
    let mut norm=burn::nn::LayerNormConfig::new(4).with_epsilon(1e-6).with_bias(true).init(&device);
    norm.gamma=Param::from_tensor(Tensor::from_data(scale,&device));
    norm.beta=Some(Param::from_tensor(Tensor::from_data(bias,&device)));
    let input=Tensor::<3>::from_data(TensorData::new(raw.to_vec(),[1,3,4]),&device).require_grad();
    let output=norm.forward(input.clone());
    let gradients=(output.clone()*Tensor::from_data(TensorData::new(upstream.to_vec(),[1,3,4]),&device)).sum().backward();
    let mut expected=vec![0.0;12];
    let mut dx=vec![0.0;12];
    let mut dscale=vec![0.0;4];
    let mut dbias=vec![0.0;4];
    for row in 0..3 {
        let mean=raw[row*4..row*4+4].iter().map(|&x|f64::from(x)).sum::<f64>()/4.0;
        let centered=std::array::from_fn::<_,4,_>(|c|f64::from(raw[row*4+c])-mean);
        let variance=centered.iter().map(|x|x*x).sum::<f64>()/4.0;
        let inv=(variance+1e-6).sqrt().recip();
        let g=std::array::from_fn::<_,4,_>(|c|f64::from(upstream[row*4+c])*f64::from(scale[c]));
        let sum=g.iter().sum::<f64>();
        let dot=g.iter().zip(centered).map(|(g,x)|g*x).sum::<f64>();
        for c in 0..4 {
            let p=row*4+c;
            expected[p]=centered[c]*inv*f64::from(scale[c])+f64::from(bias[c]);
            dx[p]=inv*(g[c]-sum/4.0-centered[c]*inv*inv*dot/4.0);
            dscale[c]+=f64::from(upstream[p])*centered[c]*inv;
            dbias[c]+=f64::from(upstream[p]);
        }
    }
    sbd::compare(&crate::kron_benchmark::values(output),&expected).unwrap();
    sbd::compare(&crate::kron_benchmark::values(input.grad(&gradients).unwrap()),&dx).unwrap();
    sbd::compare(&crate::kron_benchmark::values(norm.gamma.val().grad(&gradients).unwrap()),&dscale).unwrap();
    sbd::compare(&crate::kron_benchmark::values(norm.beta.as_ref().unwrap().val().grad(&gradients).unwrap()),&dbias).unwrap();
}
