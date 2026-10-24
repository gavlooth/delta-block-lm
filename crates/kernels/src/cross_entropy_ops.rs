#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::{
    backend::{
        Backend, Dispatch,
        autodiff::{
            Autodiff,
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        backend_extension,
        tensor::FloatTensor,
    },
    prelude::Tensor,
};
use burn_cubecl::CubeBackend;
#[backend_extension(Cube, Autodiff, Fusion: cfg(feature="fusion"))]
pub trait CrossEntropyOps: Backend {
    #[fusion(meta=|x: &TensorSpec,_target,_lanes| {
        let rows = x.shape.dims::<2>()[0];
        (TensorSpec::new([rows].into(),x.dtype),TensorSpec::new([rows,2].into(),x.dtype))
    })]
    fn cross_entropy(
        x: FloatTensor<Self>,
        target: FloatTensor<Self>,
        lanes: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
    #[fusion(meta=|x: &TensorSpec,_target,_stats,_upstream| x.clone())]
    fn cross_entropy_backward(
        x: FloatTensor<Self>,
        target: FloatTensor<Self>,
        stats: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}
impl CrossEntropyOps for CubeBackend {
    fn cross_entropy(
        x: FloatTensor<Self>,
        target: FloatTensor<Self>,
        lanes: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        crate::cross_entropy::forward(x, target, lanes)
    }
    fn cross_entropy_backward(
        x: FloatTensor<Self>,
        target: FloatTensor<Self>,
        stats: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::cross_entropy::backward(x, target, stats, upstream)
    }
}
impl<B: Backend + CrossEntropyOps, C: CheckpointStrategy> CrossEntropyOps for Autodiff<B, C> {
    fn cross_entropy(
        x: FloatTensor<Self>,
        target: FloatTensor<Self>,
        lanes: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct CrossEntropyBackward;
        impl<B: Backend + CrossEntropyOps> Backward<B, 1> for CrossEntropyBackward {
            type State = (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>);
            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, target, stats) = ops.state;
                if let Some(node) = ops.parents[0].as_ref() {
                    grads.register::<B>(node.id, B::cross_entropy_backward(x, target, stats, dy));
                }
            }
        }
        let parents = [x.node()];
        let x = x.into_primitive();
        let target = target.into_primitive();
        let (loss, stats) = B::cross_entropy(x.clone(), target.clone(), lanes);
        let loss = match CrossEntropyBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((x, target, stats.clone()), loss),
            OpsKind::UnTracked(prep) => prep.finish(loss),
        };
        (
            loss,
            <Self as burn::backend::AutodiffBackend>::from_inner(stats),
        )
    }
    fn cross_entropy_backward(
        x: FloatTensor<Self>,
        target: FloatTensor<Self>,
        stats: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Self as burn::backend::AutodiffBackend>::from_inner(B::cross_entropy_backward(
            x.into_primitive(),
            target.into_primitive(),
            stats.into_primitive(),
            upstream.into_primitive(),
        ))
    }
}
/// Row negative log probability. Labels must be integral f32 in [0,vocabulary), as supplied by Batch.
/// Weighting/mean stay in tensor primitives, preserving gradients to arbitrary floating weights.
pub fn cross_entropy_rows(x: Tensor<2>, target: Tensor<1>, lanes: usize) -> Tensor<1> {
    let (loss, _) = <Dispatch as CrossEntropyOps>::cross_entropy(
        x.into_dispatch(),
        target.into_dispatch(),
        lanes,
    );
    Tensor::from_dispatch(loss)
}
