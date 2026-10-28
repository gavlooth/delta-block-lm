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
#[backend_extension(Cube,Autodiff,Fusion:cfg(feature="fusion"))]
pub trait OhcReadOps: Backend {
    #[fusion(meta=|_x,p:&TensorSpec|p.clone())]
    fn ohc_gate_norm(x: FloatTensor<Self>, p: FloatTensor<Self>) -> FloatTensor<Self>;
    #[fusion(meta=|x:&TensorSpec,p:&TensorSpec,_dy|(x.clone(),p.clone()))]
    fn ohc_gate_vjp(
        x: FloatTensor<Self>,
        p: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
    #[fusion(meta=|x:&TensorSpec,a:&TensorSpec,_g,_epsilon|{let[b,t,_,d]=x.shape.dims::<4>();(TensorSpec::new([b,t,d].into(),x.dtype),a.clone())})]
    fn ohc_weighted_read(
        x: FloatTensor<Self>,
        a: FloatTensor<Self>,
        g: FloatTensor<Self>,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
    #[fusion(meta=|x:&TensorSpec,a:&TensorSpec,g:&TensorSpec,_inv,_dy|(x.clone(),a.clone(),g.clone()))]
    fn ohc_weighted_vjp(
        x: FloatTensor<Self>,
        a: FloatTensor<Self>,
        g: FloatTensor<Self>,
        inv: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
    #[fusion(meta=|x:&TensorSpec,_pattern|{let[b,t,_]=x.shape.dims::<3>();TensorSpec::new([b,t,16].into(),x.dtype)})]
    fn ohc_quaternion(x: FloatTensor<Self>, pattern: FloatTensor<Self>) -> FloatTensor<Self>;
    #[fusion(meta=|x:&TensorSpec,_pattern,_dy|x.clone())]
    fn ohc_quaternion_vjp(
        x: FloatTensor<Self>,
        pattern: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}
impl OhcReadOps for CubeBackend {
    fn ohc_gate_norm(x: FloatTensor<Self>, p: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::ohc_read::gate(x, p, None).0
    }
    fn ohc_gate_vjp(
        x: FloatTensor<Self>,
        p: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (dp, dx) = crate::ohc_read::gate(x, p, Some(dy));
        (dx, dp)
    }
    fn ohc_weighted_read(
        x: FloatTensor<Self>,
        a: FloatTensor<Self>,
        g: FloatTensor<Self>,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        crate::ohc_read::weighted(x, a, g, epsilon)
    }
    fn ohc_weighted_vjp(
        x: FloatTensor<Self>,
        a: FloatTensor<Self>,
        g: FloatTensor<Self>,
        inv: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        crate::ohc_read::weighted_backward(x, a, g, inv, dy)
    }
    fn ohc_quaternion(x: FloatTensor<Self>, pattern: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::ohc_read::rotation(x, pattern, None)
    }
    fn ohc_quaternion_vjp(
        x: FloatTensor<Self>,
        pattern: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::ohc_read::rotation(x, pattern, Some(dy))
    }
}
impl<B: Backend + OhcReadOps, C: CheckpointStrategy> OhcReadOps for Autodiff<B, C> {
    fn ohc_gate_norm(x: FloatTensor<Self>, p: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct GateBackward;
        impl<B: Backend + OhcReadOps> Backward<B, 2> for GateBackward {
            type State = (FloatTensor<B>, FloatTensor<B>);
            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, p) = ops.state;
                let (dx, dp) = B::ohc_gate_vjp(x, p, dy);
                for (node, g) in ops.parents.into_iter().zip([dx, dp]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, g);
                    }
                }
            }
        }
        let parents = [x.node(), p.node()];
        let (x, p) = (x.into_primitive(), p.into_primitive());
        let y = B::ohc_gate_norm(x.clone(), p.clone());
        match GateBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((x, p), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
    fn ohc_gate_vjp(
        x: FloatTensor<Self>,
        p: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (dx, dp) = B::ohc_gate_vjp(x.into_primitive(), p.into_primitive(), dy.into_primitive());
        (Self::from_inner(dx), Self::from_inner(dp))
    }
    fn ohc_weighted_read(
        x: FloatTensor<Self>,
        a: FloatTensor<Self>,
        g: FloatTensor<Self>,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct ReadBackward;
        impl<B: Backend + OhcReadOps> Backward<B, 3> for ReadBackward {
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
                let (x, a, g, inv) = ops.state;
                let (dx, da, dg) = B::ohc_weighted_vjp(x, a, g, inv, dy);
                for (node, g) in ops.parents.into_iter().zip([dx, da, dg]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, g);
                    }
                }
            }
        }
        let parents = [x.node(), a.node(), g.node()];
        let (x, a, g) = (x.into_primitive(), a.into_primitive(), g.into_primitive());
        let (y, inv) = B::ohc_weighted_read(x.clone(), a.clone(), g.clone(), epsilon);
        let aux = Self::from_inner(inv.clone());
        let y = match ReadBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((x, a, g, inv), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        };
        (y, aux)
    }
    fn ohc_weighted_vjp(
        x: FloatTensor<Self>,
        a: FloatTensor<Self>,
        g: FloatTensor<Self>,
        inv: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (dx, da, dg) = B::ohc_weighted_vjp(
            x.into_primitive(),
            a.into_primitive(),
            g.into_primitive(),
            inv.into_primitive(),
            dy.into_primitive(),
        );
        (
            Self::from_inner(dx),
            Self::from_inner(da),
            Self::from_inner(dg),
        )
    }
    fn ohc_quaternion(x: FloatTensor<Self>, pattern: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct RotationBackward;
        impl<B: Backend + OhcReadOps> Backward<B, 1> for RotationBackward {
            type State = (FloatTensor<B>, FloatTensor<B>);
            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, p) = ops.state;
                if let Some(node) = &ops.parents[0] {
                    grads.register::<B>(node.id, B::ohc_quaternion_vjp(x, p, dy));
                }
            }
        }
        let parents = [x.node()];
        let (x, pattern) = (x.into_primitive(), pattern.into_primitive());
        let y = B::ohc_quaternion(x.clone(), pattern.clone());
        match RotationBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((x, pattern), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
    fn ohc_quaternion_vjp(
        x: FloatTensor<Self>,
        pattern: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        Self::from_inner(B::ohc_quaternion_vjp(
            x.into_primitive(),
            pattern.into_primitive(),
            dy.into_primitive(),
        ))
    }
}
pub fn ohc_gate_norm(x: Tensor<4>, p: Tensor<3>) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as OhcReadOps>::ohc_gate_norm(
        x.into_dispatch(),
        p.into_dispatch(),
    ))
}
pub fn ohc_weighted_read(x: Tensor<4>, a: Tensor<3>, g: Tensor<1>, epsilon: f64) -> Tensor<3> {
    let (y, _) = <Dispatch as OhcReadOps>::ohc_weighted_read(
        x.into_dispatch(),
        a.into_dispatch(),
        g.into_dispatch(),
        epsilon as f32,
    );
    Tensor::from_dispatch(y)
}
pub fn ohc_quaternion(x: Tensor<3>, pattern: Tensor<2>) -> Tensor<4> {
    let [b, t, _] = x.dims();
    Tensor::<3>::from_dispatch(<Dispatch as OhcReadOps>::ohc_quaternion(
        x.into_dispatch(),
        pattern.into_dispatch(),
    ))
    .reshape([b, t, 4, 4])
}
