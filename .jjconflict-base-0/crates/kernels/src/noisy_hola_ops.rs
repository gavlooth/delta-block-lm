#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{AutodiffBackend,Backend,Dispatch,autodiff::{Autodiff,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,ops::{Backward,Ops,OpsKind}},backend_extension,tensor::FloatTensor};
use burn::prelude::Tensor;
use burn_cubecl::CubeBackend;

#[backend_extension(Cube,Autodiff,Fusion:cfg(feature="fusion"))]
pub trait NoisyHolaOps: Backend {
    #[fusion(meta=|q:&TensorSpec,k:&TensorSpec,_v,_visible,_valid,_sink| { let [b,h,n,_]=q.shape.dims::<4>(); let m=k.shape.dims::<4>()[2]; (TensorSpec::new([b,h,n,64].into(),q.dtype),TensorSpec::new([b,h,n,m+1].into(),q.dtype)) })]
    fn noisy_hola_read(q:FloatTensor<Self>,k:FloatTensor<Self>,v:FloatTensor<Self>,visible:FloatTensor<Self>,valid:FloatTensor<Self>,sink:FloatTensor<Self>) -> (FloatTensor<Self>,FloatTensor<Self>);
    #[fusion(meta=|q:&TensorSpec,k:&TensorSpec,v:&TensorSpec,_valid,_p,_y,_dy| { let h=q.shape.dims::<4>()[1]; (q.clone(),k.clone(),v.clone(),TensorSpec::new([h].into(),q.dtype)) })]
    fn noisy_hola_vjp(q:FloatTensor<Self>,k:FloatTensor<Self>,v:FloatTensor<Self>,valid:FloatTensor<Self>,p:FloatTensor<Self>,y:FloatTensor<Self>,dy:FloatTensor<Self>) -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>);
}
impl NoisyHolaOps for CubeBackend {
    fn noisy_hola_read(q:FloatTensor<Self>,k:FloatTensor<Self>,v:FloatTensor<Self>,visible:FloatTensor<Self>,valid:FloatTensor<Self>,sink:FloatTensor<Self>) -> (FloatTensor<Self>,FloatTensor<Self>) { crate::noisy_hola::forward(q,k,v,visible,valid,sink) }
    fn noisy_hola_vjp(q:FloatTensor<Self>,k:FloatTensor<Self>,v:FloatTensor<Self>,valid:FloatTensor<Self>,p:FloatTensor<Self>,y:FloatTensor<Self>,dy:FloatTensor<Self>) -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>) { crate::noisy_hola::backward(q,k,v,valid,p,y,dy) }
}
impl<B:Backend+NoisyHolaOps,C:CheckpointStrategy> NoisyHolaOps for Autodiff<B,C> {
    fn noisy_hola_read(q:FloatTensor<Self>,k:FloatTensor<Self>,v:FloatTensor<Self>,visible:FloatTensor<Self>,valid:FloatTensor<Self>,sink:FloatTensor<Self>) -> (FloatTensor<Self>,FloatTensor<Self>) {
        #[derive(Debug)]
        struct ReadBackward;
        impl<B:Backend+NoisyHolaOps> Backward<B,4> for ReadBackward {
            type State = (FloatTensor<B>,FloatTensor<B>,FloatTensor<B>,FloatTensor<B>,FloatTensor<B>,FloatTensor<B>);
            fn backward(self,ops:Ops<Self::State,4>,grads:&mut Gradients,_:&mut Checkpointer) {
                let dy = grads.consume::<B>(&ops.node);
                let (q,k,v,valid,p,y) = ops.state;
                let (dq,dk,dv,ds) = B::noisy_hola_vjp(q,k,v,valid,p,y,dy);
                for (node,g) in ops.parents.into_iter().zip([dq,dk,dv,ds]) { if let Some(node)=node { grads.register::<B>(node.id,g); } }
            }
        }
        let parents = [q.node(),k.node(),v.node(),sink.node()];
        let (q,k,v,visible,valid,sink) = (q.into_primitive(),k.into_primitive(),v.into_primitive(),visible.into_primitive(),valid.into_primitive(),sink.into_primitive());
        let (y,p) = B::noisy_hola_read(q.clone(),k.clone(),v.clone(),visible,valid.clone(),sink);
        let saved = Self::from_inner(p.clone());
        let y = match ReadBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((q,k,v,valid,p,y.clone()),y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        };
        (y,saved)
    }
    fn noisy_hola_vjp(q:FloatTensor<Self>,k:FloatTensor<Self>,v:FloatTensor<Self>,valid:FloatTensor<Self>,p:FloatTensor<Self>,y:FloatTensor<Self>,dy:FloatTensor<Self>) -> (FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>,FloatTensor<Self>) {
        let (dq,dk,dv,ds) = B::noisy_hola_vjp(q.into_primitive(),k.into_primitive(),v.into_primitive(),valid.into_primitive(),p.into_primitive(),y.into_primitive(),dy.into_primitive());
        (Self::from_inner(dq),Self::from_inner(dk),Self::from_inner(dv),Self::from_inner(ds))
    }
}
pub fn noisy_hola_read(q:Tensor<4>,k:Tensor<4>,v:Tensor<4>,visible:Tensor<4>,valid:Tensor<2>,sink:Tensor<1>) -> Tensor<4> {
    let (y,_) = <Dispatch as NoisyHolaOps>::noisy_hola_read(q.into_dispatch(),k.into_dispatch(),v.into_dispatch(),visible.into_dispatch(),valid.into_dispatch(),sink.into_dispatch());
    Tensor::from_dispatch(y)
}
