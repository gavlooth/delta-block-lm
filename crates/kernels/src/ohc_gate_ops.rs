use burn::backend::{
    AutodiffBackend, Backend, Dispatch,
    autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    backend_extension, tensor::FloatTensor,
};
#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::prelude::Tensor;
use burn_cubecl::CubeBackend;

#[backend_extension(Cube,Autodiff,Fusion:cfg(feature="fusion"))]
pub trait OhcGateOps: Backend {
    #[fusion(meta=|x:&TensorSpec,_w|{let[b,t,_,_]=x.shape.dims::<4>();TensorSpec::new([b,t,16].into(),x.dtype)})]
    fn ohc_gate_project(x: FloatTensor<Self>, w: FloatTensor<Self>) -> FloatTensor<Self>;
    #[fusion(meta=|x:&TensorSpec,w:&TensorSpec,_y,_dy|(x.clone(),w.clone()))]
    fn ohc_gate_project_vjp(x: FloatTensor<Self>, w: FloatTensor<Self>, y: FloatTensor<Self>, dy: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>);
}
impl OhcGateOps for CubeBackend {
    fn ohc_gate_project(x: FloatTensor<Self>, w: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::ohc_gate::forward(x, w)
    }
    fn ohc_gate_project_vjp(x: FloatTensor<Self>, w: FloatTensor<Self>, y: FloatTensor<Self>, dy: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>) {
        crate::ohc_gate::backward(x, w, y, dy)
    }
}
impl<B: Backend + OhcGateOps, C: CheckpointStrategy> OhcGateOps for Autodiff<B, C> {
    fn ohc_gate_project(x: FloatTensor<Self>, w: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct GateBackward;
        impl<B: Backend + OhcGateOps> Backward<B, 2> for GateBackward {
            type State = (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>);
            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, w, y) = ops.state;
                let (dx, dw) = B::ohc_gate_project_vjp(x, w, y, dy);
                for (node, g) in ops.parents.into_iter().zip([dx, dw]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, g);
                    }
                }
            }
        }
        let parents = [x.node(), w.node()];
        let (x, w) = (x.into_primitive(), w.into_primitive());
        let y = B::ohc_gate_project(x.clone(), w.clone());
        match GateBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((x, w, y.clone()), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
    fn ohc_gate_project_vjp(x: FloatTensor<Self>, w: FloatTensor<Self>, y: FloatTensor<Self>, dy: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (dx, dw) = B::ohc_gate_project_vjp(x.into_primitive(), w.into_primitive(), y.into_primitive(), dy.into_primitive());
        (Self::from_inner(dx), Self::from_inner(dw))
    }
}
pub fn ohc_gate_project(x: Tensor<4>, w: Tensor<2>) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as OhcGateOps>::ohc_gate_project(x.into_dispatch(), w.into_dispatch()))
}
