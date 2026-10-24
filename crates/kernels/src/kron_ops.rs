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
pub trait KronOps: Backend {
    #[fusion(meta = |q: &TensorSpec, _k, _v, _alpha, _erase, _write, _valid, _reset, initial: &TensorSpec, chunk: &usize| {
        let [batch, time, heads, _] = q.shape.dims::<4>();
        (
            TensorSpec::new([batch * time * heads * 65 + initial.shape.num_elements()].into(), q.dtype),
            TensorSpec::new([batch, heads, time.div_ceil(*chunk), 32, 64].into(), q.dtype),
            TensorSpec::new([batch, time, heads, 4].into(), q.dtype),
        )
    })]
    fn kron_scan(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        reset: FloatTensor<Self>,
        initial: FloatTensor<Self>,
        chunk: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |q: &TensorSpec, k: &TensorSpec, v: &TensorSpec, alpha: &TensorSpec, erase: &TensorSpec, _upstream, _checkpoint, _auxiliary, _valid, _reset, _chunk| {
        let [batch, _, heads, _] = q.shape.dims::<4>();
        (
            q.clone(), k.clone(), v.clone(), alpha.clone(), erase.clone(), erase.clone(),
            TensorSpec::new([batch, heads, 32, 64].into(), q.dtype),
        )
    })]
    fn kron_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        checkpoint: FloatTensor<Self>,
        auxiliary: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        reset: FloatTensor<Self>,
        chunk: usize,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );
}

impl KronOps for CubeBackend {
    fn kron_scan(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        reset: FloatTensor<Self>,
        initial: FloatTensor<Self>,
        chunk: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        crate::kron::run_scan(q, k, v, alpha, erase, write, valid, reset, initial, chunk)
    }

    fn kron_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        checkpoint: FloatTensor<Self>,
        auxiliary: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        reset: FloatTensor<Self>,
        chunk: usize,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        crate::kron::run_backward(
            q, k, v, alpha, erase, upstream, checkpoint, auxiliary, valid, reset, chunk,
        )
    }
}

impl<B: Backend + KronOps, C: CheckpointStrategy> KronOps for Autodiff<B, C> {
    fn kron_scan(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        reset: FloatTensor<Self>,
        initial: FloatTensor<Self>,
        chunk: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct ScanBackward;
        impl<B: Backend + KronOps> Backward<B, 7> for ScanBackward {
            type State = (
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                usize,
            );
            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let upstream = grads.consume::<B>(&ops.node);
                let (q, k, v, alpha, erase, checkpoint, auxiliary, valid, reset, chunk) = ops.state;
                let (dq, dk, dv, da, de, dw, ds) = B::kron_backward(
                    q, k, v, alpha, erase, upstream, checkpoint, auxiliary, valid, reset, chunk,
                );
                let [nq, nk, nv, na, ne, nw, ns] = ops.parents;
                if let Some(node) = nq {
                    grads.register::<B>(node.id, dq);
                }
                if let Some(node) = nk {
                    grads.register::<B>(node.id, dk);
                }
                if let Some(node) = nv {
                    grads.register::<B>(node.id, dv);
                }
                if let Some(node) = na {
                    grads.register::<B>(node.id, da);
                }
                if let Some(node) = ne {
                    grads.register::<B>(node.id, de);
                }
                if let Some(node) = nw {
                    grads.register::<B>(node.id, dw);
                }
                if let Some(node) = ns {
                    grads.register::<B>(node.id, ds);
                }
            }
        }
        let guards = [
            q.node(),
            k.node(),
            v.node(),
            alpha.node(),
            erase.node(),
            write.node(),
            initial.node(),
        ];
        let (q, k, v, alpha, erase, write, valid, reset, initial) = (
            q.into_primitive(),
            k.into_primitive(),
            v.into_primitive(),
            alpha.into_primitive(),
            erase.into_primitive(),
            write.into_primitive(),
            valid.into_primitive(),
            reset.into_primitive(),
            initial.into_primitive(),
        );
        let (packed, checkpoint, auxiliary) = B::kron_scan(
            q.clone(),
            k.clone(),
            v.clone(),
            alpha.clone(),
            erase.clone(),
            write,
            valid.clone(),
            reset.clone(),
            initial,
            chunk,
        );
        let checkpoint_ad = <Autodiff<B, C> as AutodiffBackend>::from_inner(checkpoint.clone());
        let auxiliary_ad = <Autodiff<B, C> as AutodiffBackend>::from_inner(auxiliary.clone());
        let packed = match ScanBackward.prepare::<C>(guards).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    q, k, v, alpha, erase, checkpoint, auxiliary, valid, reset, chunk,
                ),
                packed,
            ),
            OpsKind::UnTracked(prep) => prep.finish(packed),
        };
        (packed, checkpoint_ad, auxiliary_ad)
    }

    fn kron_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        checkpoint: FloatTensor<Self>,
        auxiliary: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        reset: FloatTensor<Self>,
        chunk: usize,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let (dq, dk, dv, da, de, dw, ds) = B::kron_backward(
            q.into_primitive(),
            k.into_primitive(),
            v.into_primitive(),
            alpha.into_primitive(),
            erase.into_primitive(),
            upstream.into_primitive(),
            checkpoint.into_primitive(),
            auxiliary.into_primitive(),
            valid.into_primitive(),
            reset.into_primitive(),
            chunk,
        );
        (
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dq),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dk),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dv),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(da),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(de),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dw),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(ds),
        )
    }
}

/// All three public outputs participate in the same seven-input analytic VJP.
#[derive(Clone, Debug)]
pub struct KronScanOutput {
    pub output: Tensor<4>,
    pub final_state: Tensor<4>,
    /// Frobenius norm of the active rank-one edit, after decay.
    pub scores: Tensor<4>,
}

/// Fused f32 Kron recurrence on a CubeCL GPU; no CPU or tensor-solver fallback.
///
/// Inputs use [batch,time,head,channel], scalar erase/write have channel width 1,
/// and initial/final state uses [batch,head,32,64]. Q/K/V normalization belongs to
/// the caller. Binary float masks are nondifferentiable: padding freezes state,
/// and a reset on a valid token discards state before applying the recurrence.
/// Chunk size 1..=16 controls checkpoint/replay memory, not recurrence semantics.
/// Scores use sign(0)=0 and zero norm subgradients at zero-norm keys/values.
/// First derivatives are supported; the custom VJP is not higher-order autodiff.
pub fn kron_scan(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    alpha: Tensor<4>,
    erase: Tensor<4>,
    write: Tensor<4>,
    valid: Tensor<2>,
    reset: Tensor<2>,
    initial: Tensor<4>,
    chunk_size: usize,
) -> KronScanOutput {
    let [batch, time, heads, keys] = q.dims();
    let values = v.dims()[3];
    let output_len = batch * time * heads * values;
    let state_len = batch * heads * keys * values;
    let score_len = batch * time * heads;
    // A single packed output node ensures every combination of output, final
    // state and score losses reaches one VJP, with Burn accumulating slice VJPs.
    let (packed, _, _) = <Dispatch as KronOps>::kron_scan(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        alpha.into_dispatch(),
        erase.into_dispatch(),
        write.into_dispatch(),
        valid.into_dispatch(),
        reset.into_dispatch(),
        initial.into_dispatch(),
        chunk_size,
    );
    let packed = Tensor::<1>::from_dispatch(packed);
    KronScanOutput {
        output: packed
            .clone()
            .slice([0..output_len])
            .reshape([batch, time, heads, values]),
        final_state: packed
            .clone()
            .slice([output_len..output_len + state_len])
            .reshape([batch, heads, keys, values]),
        scores: packed
            .slice([output_len + state_len..output_len + state_len + score_len])
            .reshape([batch, time, heads, 1]),
    }
}
