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
pub trait NormOps: Backend {
    #[fusion(meta = |x: &TensorSpec, _weight, _gate, _mode, _epsilon| {
        let [b,t,h,_] = x.shape.dims::<4>();
        (x.clone(), TensorSpec::new([b*t*h].into(),x.dtype))
    })]
    fn row_norm(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        mode: usize,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
    #[fusion(meta = |x: &TensorSpec, weight: &TensorSpec, gate: &TensorSpec, _inverse, _upstream, _mode| {
        (x.clone(), weight.clone(), gate.clone())
    })]
    fn row_norm_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        inverse: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        mode: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}
impl NormOps for CubeBackend {
    fn row_norm(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        mode: usize,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        crate::norm::forward(x, weight, gate, mode, epsilon)
    }
    fn row_norm_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        inverse: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        mode: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        crate::norm::backward(x, weight, gate, inverse, upstream, mode)
    }
}
impl<B: Backend + NormOps, C: CheckpointStrategy> NormOps for Autodiff<B, C> {
    fn row_norm(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        mode: usize,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct NormBackward;
        impl<B: Backend + NormOps> Backward<B, 3> for NormBackward {
            type State = (
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                usize,
            );
            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, w, g, inverse, mode) = ops.state;
                let (dx, dw, dg) = B::row_norm_backward(x, w, g, inverse, dy, mode);
                for (node, gradient) in ops.parents.into_iter().zip([dx, dw, dg]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, gradient);
                    }
                }
            }
        }
        let weight = if mode == 0 {
            Self::from_inner(weight.into_primitive())
        } else {
            weight
        };
        let gate = if mode != 2 {
            Self::from_inner(gate.into_primitive())
        } else {
            gate
        };
        let parents = [x.node(), weight.node(), gate.node()];
        let (x, w, g) = (
            x.into_primitive(),
            weight.into_primitive(),
            gate.into_primitive(),
        );
        let (y, inverse) = B::row_norm(x.clone(), w.clone(), g.clone(), mode, epsilon);
        let aux = <Autodiff<B, C> as AutodiffBackend>::from_inner(inverse.clone());
        let y = match NormBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((x, w, g, inverse, mode), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        };
        (y, aux)
    }
    fn row_norm_backward(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        inverse: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        mode: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (dx, dw, dg) = B::row_norm_backward(
            x.into_primitive(),
            weight.into_primitive(),
            gate.into_primitive(),
            inverse.into_primitive(),
            upstream.into_primitive(),
            mode,
        );
        (
            Self::from_inner(dx),
            Self::from_inner(dw),
            Self::from_inner(dg),
        )
    }
}
/// L2 over the last axis: x / sqrt(sum(x²) + 1e-6). Strided inputs supported.
pub fn l2_norm(x: Tensor<4>) -> Tensor<4> {
    let raw = x.into_dispatch();
    let (y, _) = <Dispatch as NormOps>::row_norm(raw.clone(), raw.clone(), raw, 0, 1e-6);
    Tensor::from_dispatch(y)
}
/// Learned RMSNorm, optionally after x * sigmoid(gate), without changing parameter storage.
/// Epsilon is the original module's epsilon (outer norms 1e-6, mixer/HOLA 1e-5).
/// First-order f32 GPU execution only; no CPU or higher-order fallback.
pub fn rms_norm<const D: usize>(
    x: Tensor<D>,
    weight: Tensor<1>,
    gate: Option<Tensor<D>>,
    epsilon: f64,
) -> Tensor<D> {
    assert!((1..=4).contains(&D));
    let dims = x.dims();
    let mode = if gate.is_some() { 2 } else { 1 };
    let x = x.unsqueeze::<4>().into_dispatch();
    let gate = gate
        .map(|g| g.unsqueeze::<4>().into_dispatch())
        .unwrap_or_else(|| x.clone());
    let (y, _) =
        <Dispatch as NormOps>::row_norm(x, weight.into_dispatch(), gate, mode, epsilon as f32);
    Tensor::<4>::from_dispatch(y).reshape(dims)
}
