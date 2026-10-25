#[cfg(feature="fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{AutodiffBackend,Backend,Dispatch,TensorMetadata,backend_extension,tensor::FloatTensor,
    autodiff::{Autodiff,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},
        grads::Gradients,ops::{Backward,Ops,OpsKind}}};
use burn::prelude::Tensor;
use burn_cubecl::CubeBackend;

#[backend_extension(Cube,Autodiff,Fusion: cfg(feature="fusion"))]
pub trait PrefixOps: Backend {
    #[fusion(meta = |scores: &TensorSpec, _valid, _documents, capacity: &usize, chunk: &usize, _recency| {
        let [b,t,h,_]=scores.shape.dims::<4>();
        TensorSpec::new([b,h,t.div_ceil(*chunk)+1,*capacity].into(),scores.dtype)
    })]
    fn prefix_schedule(scores: FloatTensor<Self>,valid: FloatTensor<Self>,documents: FloatTensor<Self>,
        capacity: usize,chunk: usize,recency: bool) -> FloatTensor<Self>;
    #[fusion(meta = |k: &TensorSpec, _valid, _documents, block: &usize| {
        let [b,t,h,_]=k.shape.dims::<4>();
        TensorSpec::new([b,(t / *block).max(1),h,35].into(),k.dtype)
    })]
    fn prefix_summaries(k: FloatTensor<Self>,valid: FloatTensor<Self>,documents: FloatTensor<Self>,block: usize) -> FloatTensor<Self>;
    #[fusion(meta = |q: &TensorSpec, _k, _v, _query_valid, _valid, _documents, _query_documents, schedule: &TensorSpec, _summaries, _sink, _ak, _av, chunk: &usize, block: &usize, top: &usize, own: &usize, _start, _frozen, active: &bool| {
        let [b,n,h,_]=q.shape.dims::<4>(); let cap=schedule.shape.dims::<4>()[3];
        (TensorSpec::new([b,n,h,64].into(),q.dtype),
         TensorSpec::new([b,n,h,cap+*chunk+*top * *block+*own+if *active {n}else{0}].into(),q.dtype),
         TensorSpec::new([b,n,h,2].into(),q.dtype))
    })]
    fn prefix_read(q: FloatTensor<Self>,k: FloatTensor<Self>,v: FloatTensor<Self>,query_valid: FloatTensor<Self>,
        valid: FloatTensor<Self>,documents: FloatTensor<Self>,query_documents: FloatTensor<Self>,
        schedule: FloatTensor<Self>,summaries: FloatTensor<Self>,sink: FloatTensor<Self>,ak: FloatTensor<Self>,av: FloatTensor<Self>,
        chunk: usize,block: usize,top: usize,own: usize,start: usize,frozen: bool,active: bool)
        -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>);
    #[fusion(meta = |q: &TensorSpec, k: &TensorSpec, v: &TensorSpec, _ids, sink: &TensorSpec, ak: &TensorSpec, av: &TensorSpec, _output, _stats, _upstream, _capacity, _chunk, _block, _top, _own, _start, _frozen, _active| {
        (q.clone(),k.clone(),v.clone(),sink.clone(),ak.clone(),av.clone())
    })]
    fn prefix_backward(q: FloatTensor<Self>,k: FloatTensor<Self>,v: FloatTensor<Self>,ids: FloatTensor<Self>,
        sink: FloatTensor<Self>,ak: FloatTensor<Self>,av: FloatTensor<Self>,output: FloatTensor<Self>,
        stats: FloatTensor<Self>,upstream: FloatTensor<Self>,capacity: usize,chunk: usize,block: usize,
        top: usize,own: usize,start: usize,frozen: bool,active: bool)
        -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>);
}
impl PrefixOps for CubeBackend {
    fn prefix_schedule(scores: FloatTensor<Self>,valid: FloatTensor<Self>,documents: FloatTensor<Self>,
        capacity: usize,chunk: usize,recency: bool) -> FloatTensor<Self> {
        crate::hola::run_schedule(scores,valid,documents,capacity,chunk,recency)
    }
    fn prefix_summaries(k: FloatTensor<Self>,valid: FloatTensor<Self>,documents: FloatTensor<Self>,block: usize) -> FloatTensor<Self> {
        crate::prefix::run_summaries(k,valid,documents,block)
    }
    fn prefix_read(q: FloatTensor<Self>,k: FloatTensor<Self>,v: FloatTensor<Self>,query_valid: FloatTensor<Self>,
        valid: FloatTensor<Self>,documents: FloatTensor<Self>,query_documents: FloatTensor<Self>,
        schedule: FloatTensor<Self>,summaries: FloatTensor<Self>,sink: FloatTensor<Self>,ak: FloatTensor<Self>,av: FloatTensor<Self>,
        chunk: usize,block: usize,top: usize,own: usize,start: usize,frozen: bool,active: bool)
        -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>) {
        crate::prefix::run_read(q,k,v,query_valid,valid,documents,query_documents,schedule,summaries,sink,ak,av,chunk,block,top,own,start,frozen,active)
    }
    fn prefix_backward(q: FloatTensor<Self>,k: FloatTensor<Self>,v: FloatTensor<Self>,ids: FloatTensor<Self>,
        sink: FloatTensor<Self>,ak: FloatTensor<Self>,av: FloatTensor<Self>,output: FloatTensor<Self>,
        stats: FloatTensor<Self>,upstream: FloatTensor<Self>,capacity: usize,chunk: usize,block: usize,
        top: usize,own: usize,start: usize,frozen: bool,active: bool)
        -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>) {
        crate::prefix::run_backward(q,k,v,ids,sink,ak,av,output,stats,upstream,capacity,chunk,block,top,own,start,frozen,active)
    }
}
impl<B: Backend+PrefixOps,C: CheckpointStrategy> PrefixOps for Autodiff<B,C> {
    fn prefix_schedule(scores: FloatTensor<Self>,valid: FloatTensor<Self>,documents: FloatTensor<Self>,
        capacity: usize,chunk: usize,recency: bool) -> FloatTensor<Self> {
        Self::from_inner(B::prefix_schedule(scores.into_primitive(),valid.into_primitive(),documents.into_primitive(),capacity,chunk,recency))
    }
    fn prefix_summaries(k: FloatTensor<Self>,valid: FloatTensor<Self>,documents: FloatTensor<Self>,block: usize) -> FloatTensor<Self> {
        Self::from_inner(B::prefix_summaries(k.into_primitive(),valid.into_primitive(),documents.into_primitive(),block))
    }
    fn prefix_read(q: FloatTensor<Self>,k: FloatTensor<Self>,v: FloatTensor<Self>,query_valid: FloatTensor<Self>,
        valid: FloatTensor<Self>,documents: FloatTensor<Self>,query_documents: FloatTensor<Self>,
        schedule: FloatTensor<Self>,summaries: FloatTensor<Self>,sink: FloatTensor<Self>,ak: FloatTensor<Self>,av: FloatTensor<Self>,
        chunk: usize,block: usize,top: usize,own: usize,start: usize,frozen: bool,active: bool)
        -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>) {
        #[derive(Debug)] struct ReadBackward;
        impl<B: Backend+PrefixOps> Backward<B,6> for ReadBackward {
            type State=(FloatTensor<B>,FloatTensor<B>,FloatTensor<B>,FloatTensor<B>,FloatTensor<B>,
                FloatTensor<B>,FloatTensor<B>,FloatTensor<B>,FloatTensor<B>,(usize,usize,usize,usize,usize,usize,bool,bool));
            fn backward(self,ops: Ops<Self::State,6>,grads: &mut Gradients,_checkpointer: &mut Checkpointer) {
                let upstream=grads.consume::<B>(&ops.node);
                let (q,k,v,ids,sink,ak,av,output,stats,(capacity,chunk,block,top,own,start,frozen,active))=ops.state;
                let (dq,dk,dv,ds,dak,dav)=B::prefix_backward(q,k,v,ids,sink,ak,av,output,stats,upstream,capacity,chunk,block,top,own,start,frozen,active);
                let [nq,nk,nv,ns,na,nb]=ops.parents;
                if let Some(node)=nq {grads.register::<B>(node.id,dq);}
                if let Some(node)=nk {grads.register::<B>(node.id,dk);}
                if let Some(node)=nv {grads.register::<B>(node.id,dv);}
                if let Some(node)=ns {grads.register::<B>(node.id,ds);}
                if let Some(node)=na {grads.register::<B>(node.id,dak);}
                if let Some(node)=nb {grads.register::<B>(node.id,dav);}
            }
        }
        let parents=[q.node(),k.node(),v.node(),sink.node(),ak.node(),av.node()];
        let (q,k,v,sink,ak,av)=(q.into_primitive(),k.into_primitive(),v.into_primitive(),sink.into_primitive(),ak.into_primitive(),av.into_primitive());
        let capacity=schedule.shape().dims::<4>()[3];
        let (output,ids,stats)=B::prefix_read(q.clone(),k.clone(),v.clone(),query_valid.into_primitive(),
            valid.into_primitive(),documents.into_primitive(),query_documents.into_primitive(),schedule.into_primitive(),summaries.into_primitive(),
            sink.clone(),ak.clone(),av.clone(),chunk,block,top,own,start,frozen,active);
        let ids_ad=Self::from_inner(ids.clone());let stats_ad=Self::from_inner(stats.clone());
        let output_ad=match ReadBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((q,k,v,ids,sink,ak,av,output.clone(),stats,(capacity,chunk,block,top,own,start,frozen,active)),output),
            OpsKind::UnTracked(prep)=>prep.finish(output),
        };
        (output_ad,ids_ad,stats_ad)
    }
    fn prefix_backward(q: FloatTensor<Self>,k: FloatTensor<Self>,v: FloatTensor<Self>,ids: FloatTensor<Self>,
        sink: FloatTensor<Self>,ak: FloatTensor<Self>,av: FloatTensor<Self>,output: FloatTensor<Self>,
        stats: FloatTensor<Self>,upstream: FloatTensor<Self>,capacity: usize,chunk: usize,block: usize,
        top: usize,own: usize,start: usize,frozen: bool,active: bool)
        -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>) {
        let (dq,dk,dv,ds,dak,dav)=B::prefix_backward(q.into_primitive(),k.into_primitive(),v.into_primitive(),ids.into_primitive(),
            sink.into_primitive(),ak.into_primitive(),av.into_primitive(),output.into_primitive(),stats.into_primitive(),
            upstream.into_primitive(),capacity,chunk,block,top,own,start,frozen,active);
        (Self::from_inner(dq),Self::from_inner(dk),Self::from_inner(dv),Self::from_inner(ds),Self::from_inner(dak),Self::from_inner(dav))
    }
}

pub fn prefix_read(q: Tensor<4>,k: Tensor<4>,v: Tensor<4>,query_valid: Tensor<2>,valid: Tensor<2>,
    documents: Tensor<2>,query_documents: Tensor<2>,schedule: Tensor<4>,sink: Tensor<1>,
    chunk: usize,block: usize,top: usize,start: usize,frozen: bool,active: Option<(Tensor<4>,Tensor<4>)>,
    summaries: Option<Tensor<4>>) -> Tensor<4> {
    assert!(block>0 && top>0 && chunk>0 && block%chunk==0);
    let time=k.dims()[1];
    assert!(time>0);
    if !frozen {assert!(start==0 && q.dims()[1]==time,"clean prefix read requires the complete sequence");}
    let candidate_limit=if frozen {start}else{time-1};
    let top=top.min(candidate_limit/block);
    let own=if frozen {start/chunk*chunk-start/block*block}
        else {(block-chunk).min((time-1)/chunk*chunk)};
    let enabled=active.is_some();let (ak,av)=active.unwrap_or_else(||(k.clone().detach(),v.clone().detach()));
    let summaries=summaries.unwrap_or_else(||if top==0 {k.clone()} else {
        prefix_summaries(k.clone(),valid.clone(),documents.clone(),block)
    });
    let (output,_,_)=<Dispatch as PrefixOps>::prefix_read(q.into_dispatch(),k.into_dispatch(),v.into_dispatch(),
        query_valid.into_dispatch(),valid.into_dispatch(),documents.into_dispatch(),query_documents.into_dispatch(),
        schedule.into_dispatch(),summaries.into_dispatch(),sink.into_dispatch(),ak.into_dispatch(),av.into_dispatch(),chunk,block,top,own,start,frozen,enabled);
    Tensor::from_dispatch(output)
}
pub fn prefix_hola_read(q: Tensor<4>,k: Tensor<4>,v: Tensor<4>,scores: Tensor<4>,valid: Tensor<2>,
    documents: Tensor<2>,sink: Tensor<1>,capacity: usize,chunk: usize,recency: bool,block: usize,top: usize) -> Tensor<4> {
    let schedule=<Dispatch as PrefixOps>::prefix_schedule(scores.into_dispatch(),valid.clone().into_dispatch(),documents.clone().into_dispatch(),capacity,chunk,recency);
    prefix_read(q,k,v,valid.clone(),valid,documents.clone(),documents,Tensor::from_dispatch(schedule),sink,chunk,block,top,0,false,None,None)
}
pub fn prefix_summaries(k: Tensor<4>,valid: Tensor<2>,documents: Tensor<2>,block: usize) -> Tensor<4> {
    Tensor::from_dispatch(<Dispatch as PrefixOps>::prefix_summaries(k.into_dispatch(),valid.into_dispatch(),documents.into_dispatch(),block))
}
