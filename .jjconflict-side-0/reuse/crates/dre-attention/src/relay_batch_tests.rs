//! CPU references retain the pre-batching document reads and source-only rotary convention.
use super::*;
use dre_core::{BatchLayout,DocumentLayout};

fn host<const D:usize>(tensor:&Tensor<D>)->Vec<f32> {tensor.to_data().try_to_vec_as::<f32>().unwrap()}
fn close(actual:&[f32],expected:&[f32],name:&str,gradient:bool) {
    assert_eq!(actual.len(),expected.len(),"{name} shape");
    let (absolute,relative)=if gradient {(1e-4,8e-4)}else{(2e-5,3e-4)};
    for (index,(&actual,&expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite()&&expected.is_finite()&&(actual-expected).abs()<absolute+relative*expected.abs(),
            "{name}[{index}]: {actual} != {expected}");
    }
}
fn rotate_reference(rotary:&RotaryEncoding,input:Tensor<4>,layout:&BatchLayout)->Tensor<4> {
    let [_,heads,_,width]=input.dims();let mut result=Tensor::zeros(input.shape(),&input.device());
    for (row,documents) in layout.host.documents_per_row().iter().enumerate() {
        let mut start=0;
        for &length in documents {
            let end=start+length;
            let piece=input.clone().slice([row..row+1,0..heads,start..end,0..width]);
            result=result.slice_assign([row..row+1,0..heads,start..end,0..width],rotary.forward(piece));
            start=end;
        }
    }
    result
}
fn legacy(model:&LocalGlobalRelayAttention,source:Tensor<3>,relays:Tensor<3>,layout:&BatchLayout)->(Tensor<3>,Tensor<3>) {
    let [batch,tokens,_]=source.dims();let real=layout.host.document_count();let capacity=layout.host.document_capacity();
    let shape=[batch,tokens,model.heads,model.head_dim];
    let source_q=rotate_reference(&model.rotary,dre_core::project(&model.source_query,source.clone(),model.mixed).reshape(shape).swap_dims(1,2),layout);
    let source_k=rotate_reference(&model.rotary,dre_core::project(&model.source_key,source.clone(),model.mixed).reshape(shape).swap_dims(1,2),layout);
    let source_v=dre_core::project(&model.source_value,source,model.mixed).reshape(shape).swap_dims(1,2);
    let relays=relays.slice([0..real]);let relay_shape=[real,16,model.heads,model.head_dim];
    // The checkpoint path rotates source Q/K only, never relay Q/K.
    let relay_q=dre_core::project(&model.relay_query,relays.clone(),model.mixed).reshape(relay_shape).swap_dims(1,2);
    let relay_k=dre_core::project(&model.relay_key,relays.clone(),model.mixed).reshape(relay_shape).swap_dims(1,2);
    let relay_v=dre_core::project(&model.relay_value,relays,model.mixed).reshape(relay_shape).swap_dims(1,2);
    let source_output=crate::oracle::burn_source_oracle(source_q,source_k.clone(),source_v.clone(),relay_k.clone(),relay_v.clone(),layout)
        .swap_dims(1,2).reshape([batch,tokens,model.d]);
    let mut reads=Vec::with_capacity(real);let mut document=0;
    for (row,lengths) in layout.host.documents_per_row().iter().enumerate() {
        let mut start=0;
        for &length in lengths {
            let end=start+length;
            let local_k=source_k.clone().slice([row..row+1,0..model.heads,start..end,0..model.head_dim]);
            let local_v=source_v.clone().slice([row..row+1,0..model.heads,start..end,0..model.head_dim]);
            let global_k=relay_k.clone().slice([document..document+1]);let global_v=relay_v.clone().slice([document..document+1]);
            let query=relay_q.clone().slice([document..document+1]);
            reads.push(attention(query,Tensor::cat(vec![local_k,global_k],2),Tensor::cat(vec![local_v,global_v],2),None,None,AttentionModuleOptions::default()));
            document+=1;start=end;
        }
    }
    let relay_output=Tensor::cat(reads,0).swap_dims(1,2).reshape([real,16,model.d]);
    let relay_output=dre_core::project(&model.relay_output,relay_output,model.mixed);
    let relay_output=if real==capacity {relay_output}else{
        let zeros=Tensor::zeros([capacity-real,16,model.d],&relay_output.device());Tensor::cat(vec![relay_output,zeros],0)
    };
    (dre_core::project(&model.source_output,source_output,model.mixed)*layout.valid.clone(),relay_output)
}
struct Snapshot {source:Vec<f32>,relays:Vec<f32>,source_gradient:Vec<f32>,relay_gradient:Vec<f32>,parameters:Vec<(&'static str,Vec<f32>)>}
fn snapshot(model:&LocalGlobalRelayAttention,padded:bool,reference:bool,device:&Device)->Snapshot {
    let rows=if padded {vec![vec![2,3],vec![1],vec![],vec![]]}else{vec![vec![2,3],vec![1]]};
    let host_layout=DocumentLayout::packed(rows,6);let host_layout=if padded {host_layout.with_document_padding(8)}else{host_layout};
    let layout=BatchLayout::new(host_layout,device);let batch=layout.host.batch();let documents=layout.host.document_capacity();
    let source=Tensor::<3>::from_data(TensorData::new((0..batch*6*64).map(|index|((index*13+7) as f32/91.0).sin()/3.0).collect::<Vec<_>>(),[batch,6,64]),device).require_grad();
    let relays=Tensor::<3>::from_data(TensorData::new((0..documents*16*64).map(|index|((index*17+3) as f32/113.0).cos()/4.0).collect::<Vec<_>>(),[documents,16,64]),device).require_grad();
    let (source_output,relay_output)=if reference {legacy(model,source.clone(),relays.clone(),&layout)}else{model.forward(source.clone(),relays.clone(),&layout)};
    let source_values=host(&source_output);let relay_values=host(&relay_output);
    let source_weight=Tensor::<3>::from_data(TensorData::new((0..batch*6*64).map(|index|((index%29) as f32-11.0)/37.0).collect::<Vec<_>>(),[batch,6,64]),device);
    let relay_weight=Tensor::<3>::from_data(TensorData::new((0..documents*16*64).map(|index|((index%23) as f32-7.0)/31.0).collect::<Vec<_>>(),[documents,16,64]),device);
    let gradients=((source_output*source_weight).sum()+(relay_output*relay_weight).sum()).backward();
    macro_rules! gradient {($name:literal,$tensor:expr)=>{($name,host(&$tensor.grad(&gradients).unwrap()))};}
    let parameters=vec![gradient!("source query",model.source_query.weight.val()),gradient!("source key",model.source_key.weight.val()),
        gradient!("source value",model.source_value.weight.val()),gradient!("source output",model.source_output.weight.val()),
        gradient!("relay query",model.relay_query.weight.val()),gradient!("relay key",model.relay_key.weight.val()),
        gradient!("relay value",model.relay_value.weight.val()),gradient!("relay output",model.relay_output.weight.val()),
        gradient!("source query bias",model.source_query.bias.as_ref().unwrap().val()),gradient!("source key bias",model.source_key.bias.as_ref().unwrap().val()),
        gradient!("source value bias",model.source_value.bias.as_ref().unwrap().val()),gradient!("source output bias",model.source_output.bias.as_ref().unwrap().val()),
        gradient!("relay query bias",model.relay_query.bias.as_ref().unwrap().val()),gradient!("relay key bias",model.relay_key.bias.as_ref().unwrap().val()),
        gradient!("relay value bias",model.relay_value.bias.as_ref().unwrap().val()),gradient!("relay output bias",model.relay_output.bias.as_ref().unwrap().val())];
    Snapshot {source:source_values,relays:relay_values,source_gradient:host(&source.grad(&gradients).unwrap()),relay_gradient:host(&relays.grad(&gradients).unwrap()),parameters}
}
fn model(device:&Device)->LocalGlobalRelayAttention {LocalGlobalRelayAttentionConfig::new().with_heads(1).with_head_dim(64).with_d(64).init(device)}
#[test]
fn exp004_batched_relay_read_matches_legacy_values_and_gradients_cpu() {
    let device=Device::ndarray().autodiff();let model=model(&device);
    for padded in [false,true] {
        let actual=snapshot(&model,padded,false,&device);let expected=snapshot(&model,padded,true,&device);
        close(&actual.source,&expected.source,"source output",false);close(&actual.relays,&expected.relays,"relay output",false);
        close(&actual.source_gradient,&expected.source_gradient,"source gradient",true);close(&actual.relay_gradient,&expected.relay_gradient,"relay gradient",true);
        for ((name,actual),(expected_name,expected)) in actual.parameters.iter().zip(&expected.parameters) {
            assert_eq!(name,expected_name);close(actual,expected,name,true);
        }
    }
}
#[test]
fn exp004_relay_batch_padding_preserves_values_leaf_and_parameter_gradients_cpu() {
    let device=Device::ndarray().autodiff();let model=model(&device);
    let exact=snapshot(&model,false,false,&device);let padded=snapshot(&model,true,false,&device);
    for (name,actual,expected,gradient) in [
        ("source output",&padded.source,&exact.source,false),("relay output",&padded.relays,&exact.relays,false),
        ("source gradient",&padded.source_gradient,&exact.source_gradient,true),("relay gradient",&padded.relay_gradient,&exact.relay_gradient,true),
    ] {
        close(&actual[..expected.len()],expected,name,gradient);
        assert!(actual[expected.len()..].iter().all(|&value|value==0.0),"inactive batch/document tail must be exactly zero: {name}");
    }
    for ((name,actual),(_,expected)) in padded.parameters.iter().zip(&exact.parameters) {close(actual,expected,name,true);}
}
#[test]
fn exp004_document_position_gather_matches_legacy_rotary_and_f64_cpu() {
    let device=Device::ndarray().autodiff();let rotary=RotaryEncodingConfig::new(4096,4).init(&device);
    let layout=BatchLayout::new(DocumentLayout::packed(vec![vec![2,3],vec![1],vec![]],6).with_document_padding(8),&device);
    let shape=[3,2,6,4];let count=3*2*6*4;
    let q_values=(0..count).map(|index|(index as f32-17.0)/73.0).collect::<Vec<_>>();
    let k_values=(0..count).map(|index|(index as f32-31.0)/91.0).collect::<Vec<_>>();
    let weights=(0..count).map(|index|((index%17) as f32-7.0)/23.0).collect::<Vec<_>>();
    let q=Tensor::<4>::from_data(TensorData::new(q_values.clone(),shape),&device).require_grad();
    let k=Tensor::<4>::from_data(TensorData::new(k_values.clone(),shape),&device).require_grad();
    let (actual_q,actual_k)=crate::layer::rotate_document_sources(&rotary,q.clone(),k.clone(),&layout);
    close(&host(&actual_q),&host(&rotate_reference(&rotary,q.clone(),&layout)),"legacy source query RoPE",false);
    close(&host(&actual_k),&host(&rotate_reference(&rotary,k.clone(),&layout)),"legacy source key RoPE",false);
    let mut expected_q=vec![0.0f32;count];let mut expected_k=expected_q.clone();let mut dq=expected_q.clone();let mut dk=expected_q.clone();
    for (row,documents) in layout.host.documents_per_row().iter().enumerate() {let mut start=0;
        for &length in documents {for token in 0..length {for head in 0..2 {for pair in 0..2 {
            let angle=token as f64*10000.0f64.powf(-((2*pair) as f64)/4.0);let (sine,cosine)=angle.sin_cos();
            let base=((row*2+head)*6+start+token)*4+pair*2;
            let rotate=|values:&[f32]|[(f64::from(values[base])*cosine-f64::from(values[base+1])*sine) as f32,
                (f64::from(values[base+1])*cosine+f64::from(values[base])*sine) as f32];
            let value=rotate(&q_values);expected_q[base]=value[0];expected_q[base+1]=value[1];
            let value=rotate(&k_values);expected_k[base]=value[0];expected_k[base+1]=value[1];
            dq[base]=2.0*q_values[base];dq[base+1]=2.0*q_values[base+1];
            dk[base]=(f64::from(weights[base])*cosine+f64::from(weights[base+1])*sine) as f32;
            dk[base+1]=(-f64::from(weights[base])*sine+f64::from(weights[base+1])*cosine) as f32;
        }}}start+=length;}
    }
    close(&host(&actual_q),&expected_q,"f64 source query RoPE",false);close(&host(&actual_k),&expected_k,"f64 source key RoPE",false);
    let weights=Tensor::<4>::from_data(TensorData::new(weights,shape),&device);
    let gradients=(actual_q.square().sum()+(actual_k*weights).sum()).backward();
    close(&host(&q.grad(&gradients).unwrap()),&dq,"f64 source query RoPE gradient",true);
    close(&host(&k.grad(&gradients).unwrap()),&dk,"f64 source key RoPE gradient",true);
}

const MASK_HEADS:usize=8;
const MASK_RELAYS:usize=16;
const MASK_WIDTH:usize=64;
const MASK_TOKENS:usize=48;
const MASK_DOCUMENTS:usize=8;

fn mask_layout(device:&Device)->BatchLayout {
    BatchLayout::new(DocumentLayout::packed(vec![vec![2,3],vec![1],vec![4],vec![]],MASK_TOKENS)
        .with_document_padding(MASK_DOCUMENTS),device)
}

/// Host document intervals, independently expanded across every head and every relay query.
fn dense_relay_mask(layout:&BatchLayout)->Vec<bool> {
    let tokens=layout.host.tokens();let keys=tokens+MASK_RELAYS;
    let mut intervals=Vec::new();
    for lengths in layout.host.documents_per_row() {
        let mut first=0;
        for &length in lengths {intervals.push(Some((first,first+length)));first+=length;}
    }
    intervals.resize(layout.host.document_capacity(),None);
    let mut values=Vec::with_capacity(intervals.len()*MASK_HEADS*MASK_RELAYS*keys);
    for interval in intervals {for _head in 0..MASK_HEADS {for _query in 0..MASK_RELAYS {for key in 0..keys {
        values.push(key<tokens&&interval.is_none_or(|(first,end)|key<first||key>=end));
    }}}}
    values
}

#[test]
fn exp004_relay_mask_materializes_broadcast_layout_and_preserves_all_logical_values_cpu() {
    use burn_std::tensor::{ReshapeAction,contiguous_strides,reshape_action};
    let device=Device::ndarray();let layout=mask_layout(&device);
    let keys=MASK_TOKENS+MASK_RELAYS;
    let shape=[MASK_DOCUMENTS,MASK_HEADS,MASK_RELAYS,keys];
    let elements=shape.iter().product::<usize>();
    // Cube expansion changes only the repeated head/query strides to zero. NdArray eagerly
    // owns its broadcast, so test the exact Cube layout analysis independently on the CPU.
    let mut broadcast=contiguous_strides(&shape);
    broadcast[0]=keys;broadcast[1]=0;broadcast[2]=0;
    assert_eq!(broadcast.to_vec(),vec![keys,0,0,1]);
    assert!(matches!(reshape_action(&Shape::new(shape),&broadcast,&Shape::new([elements])),ReshapeAction::Recompute));
    assert_eq!(MASK_DOCUMENTS*keys,512,"the original mask has only 512 physical bool elements");
    assert_eq!(elements,65536,"all heads and queries need distinct dense mask storage");
    let mask=relay_outside_mask(&layout,MASK_HEADS,MASK_RELAYS,&device);
    assert_eq!(mask.dims(),shape);
    let values=mask.to_data().try_to_vec_as::<bool>().unwrap();
    assert_eq!(values,dense_relay_mask(&layout));
    let primitive=mask.try_into_primitive::<burn::backend::NdArray>().expect("NdArray bool primitive");
    let burn::backend::ndarray::NdArrayTensor::Bool(storage)=primitive else {panic!("expected boolean storage");};
    assert_eq!(storage.view().strides(),&[
        (MASK_HEADS*MASK_RELAYS*keys) as isize,(MASK_RELAYS*keys) as isize,keys as isize,1,
    ]);
}

fn relay_read_reference(query:Tensor<4>,source_k:Tensor<4>,source_v:Tensor<4>,
    relay_k:Tensor<4>,relay_v:Tensor<4>,layout:&BatchLayout)->Tensor<4> {
    let device=query.device();
    let mut rows=Vec::new();
    for (row,lengths) in layout.host.documents_per_row().iter().enumerate() {
        rows.extend(std::iter::repeat_n(row as i32,lengths.len()));
    }
    rows.resize(layout.host.document_capacity(),0);
    let indices=Tensor::<1,Int>::from_data(TensorData::new(rows,[MASK_DOCUMENTS]),&device);
    let keys=Tensor::cat(vec![source_k.select(0,indices.clone()),relay_k],2);
    let values=Tensor::cat(vec![source_v.select(0,indices),relay_v],2);
    let outside=Tensor::<4,Bool>::from_data(TensorData::new(dense_relay_mask(layout),
        [MASK_DOCUMENTS,MASK_HEADS,MASK_RELAYS,MASK_TOKENS+MASK_RELAYS]),&device);
    // Deliberately do not use generic attention: its mask reader is the component under test.
    burn::tensor::activation::softmax((query.matmul(keys.swap_dims(2,3))/(MASK_WIDTH as f64).sqrt())
        .mask_fill(outside,f32::NEG_INFINITY),3).matmul(values)
}

struct MaskReadSnapshot {output:Vec<f32>,gradients:[Vec<f32>;5]}
fn mask_read_snapshot(layout:&BatchLayout,device:&Device,reference:bool)->MaskReadSnapshot {
    let real=layout.host.document_count();
    let inputs:[Tensor<4>;5]=std::array::from_fn(|field| {
        let source=field==1||field==2;
        let rows=if source {layout.host.batch()}else{MASK_DOCUMENTS};
        let positions=if source {MASK_TOKENS}else{MASK_RELAYS};
        let values=(0..rows*positions*MASK_HEADS*MASK_WIDTH).map(|index| {
            if !source&&index>=real*positions*MASK_HEADS*MASK_WIDTH {0.0}
            else {((index*13+field*37+7) as f32*0.0017).sin()/5.0}
        }).collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(values,[rows,positions,MASK_HEADS,MASK_WIDTH]),device).require_grad()
    });
    let view=|field:usize|inputs[field].clone().swap_dims(1,2);
    let output=if reference {relay_read_reference(view(0),view(1),view(2),view(3),view(4),layout)}
        else {relay_read(view(0),view(1),view(2),view(3),view(4),layout)};
    let values=host(&output);
    assert!(values.iter().all(|value|value.is_finite()));
    assert!(values[real*MASK_HEADS*MASK_RELAYS*MASK_WIDTH..].iter().all(|&value|value==0.0));
    let shape=[MASK_DOCUMENTS,MASK_HEADS,MASK_RELAYS,MASK_WIDTH];
    let weights=Tensor::<4>::from_data(TensorData::new((0..shape.iter().product::<usize>())
        .map(|index|((index%31) as f32-13.0)/19.0).collect::<Vec<_>>(),shape),device);
    let invalid=layout.document_valid.clone().lower_elem(0.5).reshape([MASK_DOCUMENTS,1,1,1]);
    let gradients=(output.mask_fill(invalid,0.0)*weights).sum().backward();
    let gradients=std::array::from_fn(|field| {
        let values=host(&inputs[field].grad(&gradients).expect("actual attention input gradient"));
        assert!(values.iter().all(|value|value.is_finite()));
        assert!(values.iter().any(|value|value.abs()>1e-8),"input {field} must exercise backward");
        if field==1||field==2 {
            for (row,lengths) in layout.host.documents_per_row().iter().enumerate() {
                let used=lengths.iter().sum::<usize>();
                let start=(row*MASK_TOKENS+used)*MASK_HEADS*MASK_WIDTH;
                let end=(row+1)*MASK_TOKENS*MASK_HEADS*MASK_WIDTH;
                assert!(values[start..end].iter().all(|&value|value==0.0),"unused source gradient, field {field}");
            }
        }else{
            assert!(values[real*MASK_RELAYS*MASK_HEADS*MASK_WIDTH..].iter().all(|&value|value==0.0),
                "padded document gradient, field {field}");
        }
        values
    });
    MaskReadSnapshot {output:values,gradients}
}

fn compare_mask_read(actual:&MaskReadSnapshot,expected:&MaskReadSnapshot) {
    close(&actual.output,&expected.output,"eight-head relay mask output",false);
    for field in 0..5 {close(&actual.gradients[field],&expected.gradients[field],
        &format!("eight-head relay mask input gradient {field}"),true);}
}

#[test]
fn exp004_eight_head_relay_read_matches_independent_dense_mask_values_and_gradients_cpu() {
    let device=Device::ndarray().autodiff();let layout=mask_layout(&device);
    compare_mask_read(&mask_read_snapshot(&layout,&device,false),&mask_read_snapshot(&layout,&device,true));
}

#[cfg(feature="cuda")]
#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn exp004_cuda_production_relay_dense_mask_storage_and_attention_match_independent_oracle() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    use burn::backend::{fusion::Fusion,wgpu::CubeBackend};
    let device=dre_core::gpu_device().autodiff();let layout=mask_layout(&device);
    let mask=relay_outside_mask(&layout,MASK_HEADS,MASK_RELAYS,&device);
    let fused=mask.clone().inner().try_into_primitive::<Fusion<CubeBackend>>().expect("fused Cube bool mask");
    let physical=fused.client.clone().resolve_tensor_bool::<CubeBackend>(fused);
    let shape=mask.dims();let elements=shape.iter().product::<usize>();
    assert_eq!(physical.meta.strides().to_vec(),burn_std::tensor::contiguous_strides(&shape).to_vec());
    assert!(physical.is_contiguous());
    assert!(physical.handle.size() as usize>=elements*physical.dtype.size(),
        "every logical mask element must have physical storage before flash attention");
    assert_eq!(mask.to_data().try_to_vec_as::<bool>().unwrap(),dense_relay_mask(&layout));
    compare_mask_read(&mask_read_snapshot(&layout,&device,false),&mask_read_snapshot(&layout,&device,true));
}
