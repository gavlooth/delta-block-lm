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
pub trait DecayOps: Backend {
    #[fusion(meta = |x: &TensorSpec,_bias,_scale| x.clone())]
    fn decay(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        scale: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
    #[fusion(meta = |x: &TensorSpec,bias: &TensorSpec,scale: &TensorSpec,_y,_dy| {
        (x.clone(),bias.clone(),scale.clone())
    })]
    fn decay_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        scale: FloatTensor<Self>,
        y: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}
impl DecayOps for CubeBackend {
    fn decay(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        scale: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::decay::forward(x, bias, scale)
    }
    fn decay_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        scale: FloatTensor<Self>,
        y: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        crate::decay::backward(x, bias, scale, y, dy)
    }
}
impl<B: Backend + DecayOps, C: CheckpointStrategy> DecayOps for Autodiff<B, C> {
    fn decay(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        scale: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct DecayBackward;
        impl<B: Backend + DecayOps> Backward<B, 3> for DecayBackward {
            type State = (
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
            );
            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, b, s, y) = ops.state;
                let (dx, db, ds) = B::decay_backward(x, b, s, y, dy);
                for (node, gradient) in ops.parents.into_iter().zip([dx, db, ds]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, gradient);
                    }
                }
            }
        }
        let parents = [x.node(), bias.node(), scale.node()];
        let (x, b, s) = (
            x.into_primitive(),
            bias.into_primitive(),
            scale.into_primitive(),
        );
        let y = B::decay(x.clone(), b.clone(), s.clone());
        match DecayBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((x, b, s, y.clone()), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
    fn decay_backward(
        x: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        scale: FloatTensor<Self>,
        y: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (dx, db, ds) = B::decay_backward(
            x.into_primitive(),
            bias.into_primitive(),
            scale.into_primitive(),
            y.into_primitive(),
            dy.into_primitive(),
        );
        (
            Self::from_inner(dx),
            Self::from_inner(db),
            Self::from_inner(ds),
        )
    }
}
/// Threshold-20 softplus/exp decay. x=[B,T,H,K], bias=[H*K], scale=[H].
/// Preserves all original parameter identities and supports strided f32 GPU inputs.
pub fn decay(x: Tensor<4>, bias: Tensor<1>, scale: Tensor<1>) -> Tensor<4> {
    Tensor::from_dispatch(<Dispatch as DecayOps>::decay(
        x.into_dispatch(),
        bias.into_dispatch(),
        scale.into_dispatch(),
    ))
}
