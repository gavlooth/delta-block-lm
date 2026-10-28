#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    AutodiffBackend, Backend, Dispatch,
    autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    backend_extension,
    tensor::FloatTensor,
};
use burn::prelude::Tensor;
use burn_cubecl::CubeBackend;
#[backend_extension(Cube, Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait ConvOps: Backend {
    #[fusion(meta = |x: &TensorSpec,_w,_valid,_docs| x.clone())]
    fn causal_conv(
        x: FloatTensor<Self>,
        w: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
    #[fusion(meta = |x: &TensorSpec,w: &TensorSpec,_valid,_docs,_dy| (x.clone(),w.clone()))]
    fn causal_conv_backward(
        x: FloatTensor<Self>,
        w: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
}
impl ConvOps for CubeBackend {
    fn causal_conv(
        x: FloatTensor<Self>,
        w: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::conv::forward(x, w, valid, docs)
    }
    fn causal_conv_backward(
        x: FloatTensor<Self>,
        w: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        crate::conv::backward(x, w, valid, docs, dy)
    }
}
impl<B: Backend + ConvOps, C: CheckpointStrategy> ConvOps for Autodiff<B, C> {
    fn causal_conv(
        x: FloatTensor<Self>,
        w: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct ConvBackward;
        impl<B: Backend + ConvOps> Backward<B, 2> for ConvBackward {
            type State = (
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
            );
            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, w, valid, docs) = ops.state;
                let (dx, dw) = B::causal_conv_backward(x, w, valid, docs, dy);
                for (node, gradient) in ops.parents.into_iter().zip([dx, dw]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, gradient);
                    }
                }
            }
        }
        let parents = [x.node(), w.node()];
        let (x, w, valid, docs) = (
            x.into_primitive(),
            w.into_primitive(),
            valid.into_primitive(),
            docs.into_primitive(),
        );
        let y = B::causal_conv(x.clone(), w.clone(), valid.clone(), docs.clone());
        match ConvBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((x, w, valid, docs), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
    fn causal_conv_backward(
        x: FloatTensor<Self>,
        w: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (dx, dw) = B::causal_conv_backward(
            x.into_primitive(),
            w.into_primitive(),
            valid.into_primitive(),
            docs.into_primitive(),
            dy.into_primitive(),
        );
        (Self::from_inner(dx), Self::from_inner(dw))
    }
}
/// f32 [batch,time,channel] convolution, weight [lag,channel]. Metadata has no VJP.
pub fn causal_conv(
    x: Tensor<3>,
    w: Tensor<2>,
    valid: Tensor<2>,
    documents: Tensor<2>,
) -> Tensor<3> {
    assert!(
        !valid.is_require_grad() && !documents.is_require_grad(),
        "convolution masks are nondifferentiable metadata"
    );
    Tensor::from_dispatch(<Dispatch as ConvOps>::causal_conv(
        x.into_dispatch(),
        w.into_dispatch(),
        valid.into_dispatch(),
        documents.into_dispatch(),
    ))
}
