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
pub trait HolaOps: Backend {
    #[fusion(meta = |q: &TensorSpec, _k, v: &TensorSpec, _scores, _valid, _documents, _sink, capacity: &usize, chunk: &usize, _recency| {
        let [batch, time, heads, _] = q.shape.dims::<4>();
        (
            v.clone(),
            TensorSpec::new([batch, heads, time.div_ceil(*chunk) + 1, *capacity].into(), q.dtype),
            TensorSpec::new([batch, time, heads].into(), q.dtype),
            TensorSpec::new([batch, time, heads, 2].into(), q.dtype),
        )
    })]
    fn hola_read(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        scores: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        documents: FloatTensor<Self>,
        sink: FloatTensor<Self>,
        capacity: usize,
        chunk: usize,
        recency: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );

    #[fusion(meta = |q: &TensorSpec, k: &TensorSpec, v: &TensorSpec, _valid, _documents, sink: &TensorSpec, _schedule, _expiry, _output, _stats, _upstream, _capacity, _chunk| {
        (q.clone(), k.clone(), v.clone(), sink.clone())
    })]
    fn hola_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        documents: FloatTensor<Self>,
        sink: FloatTensor<Self>,
        schedule: FloatTensor<Self>,
        expiry: FloatTensor<Self>,
        output: FloatTensor<Self>,
        stats: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        capacity: usize,
        chunk: usize,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );
}

impl HolaOps for CubeBackend {
    fn hola_read(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        scores: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        documents: FloatTensor<Self>,
        sink: FloatTensor<Self>,
        capacity: usize,
        chunk: usize,
        recency: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        crate::hola::run_read(
            q, k, v, scores, valid, documents, sink, capacity, chunk, recency,
        )
    }

    fn hola_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        documents: FloatTensor<Self>,
        sink: FloatTensor<Self>,
        schedule: FloatTensor<Self>,
        expiry: FloatTensor<Self>,
        output: FloatTensor<Self>,
        stats: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        capacity: usize,
        chunk: usize,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        crate::hola::run_backward(
            q, k, v, valid, documents, sink, schedule, expiry, output, stats, upstream, capacity,
            chunk,
        )
    }
}

impl<B: Backend + HolaOps, C: CheckpointStrategy> HolaOps for Autodiff<B, C> {
    fn hola_read(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        scores: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        documents: FloatTensor<Self>,
        sink: FloatTensor<Self>,
        capacity: usize,
        chunk: usize,
        recency: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        #[derive(Debug)]
        struct ReadBackward;
        impl<B: Backend + HolaOps> Backward<B, 4> for ReadBackward {
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
                FloatTensor<B>,
                usize,
                usize,
            );
            fn backward(
                self,
                ops: Ops<Self::State, 4>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let upstream = grads.consume::<B>(&ops.node);
                let (
                    q,
                    k,
                    v,
                    valid,
                    documents,
                    sink,
                    schedule,
                    expiry,
                    output,
                    stats,
                    capacity,
                    chunk,
                ) = ops.state;
                let (dq, dk, dv, ds) = B::hola_backward(
                    q, k, v, valid, documents, sink, schedule, expiry, output, stats, upstream,
                    capacity, chunk,
                );
                let [nq, nk, nv, ns] = ops.parents;
                if let Some(node) = nq {
                    grads.register::<B>(node.id, dq);
                }
                if let Some(node) = nk {
                    grads.register::<B>(node.id, dk);
                }
                if let Some(node) = nv {
                    grads.register::<B>(node.id, dv);
                }
                if let Some(node) = ns {
                    grads.register::<B>(node.id, ds);
                }
            }
        }
        let parents = [q.node(), k.node(), v.node(), sink.node()];
        let (q, k, v, scores, valid, documents, sink) = (
            q.into_primitive(),
            k.into_primitive(),
            v.into_primitive(),
            scores.into_primitive(),
            valid.into_primitive(),
            documents.into_primitive(),
            sink.into_primitive(),
        );
        let (output, schedule, expiry, stats) = B::hola_read(
            q.clone(),
            k.clone(),
            v.clone(),
            scores,
            valid.clone(),
            documents.clone(),
            sink.clone(),
            capacity,
            chunk,
            recency,
        );
        // Metadata and selection scores intentionally create no autodiff edges.
        let schedule_ad = <Autodiff<B, C> as AutodiffBackend>::from_inner(schedule.clone());
        let expiry_ad = <Autodiff<B, C> as AutodiffBackend>::from_inner(expiry.clone());
        let stats_ad = <Autodiff<B, C> as AutodiffBackend>::from_inner(stats.clone());
        let output_ad = match ReadBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    q,
                    k,
                    v,
                    valid,
                    documents,
                    sink,
                    schedule,
                    expiry,
                    output.clone(),
                    stats,
                    capacity,
                    chunk,
                ),
                output,
            ),
            OpsKind::UnTracked(prep) => prep.finish(output),
        };
        (output_ad, schedule_ad, expiry_ad, stats_ad)
    }

    fn hola_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        documents: FloatTensor<Self>,
        sink: FloatTensor<Self>,
        schedule: FloatTensor<Self>,
        expiry: FloatTensor<Self>,
        output: FloatTensor<Self>,
        stats: FloatTensor<Self>,
        upstream: FloatTensor<Self>,
        capacity: usize,
        chunk: usize,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let (dq, dk, dv, ds) = B::hola_backward(
            q.into_primitive(),
            k.into_primitive(),
            v.into_primitive(),
            valid.into_primitive(),
            documents.into_primitive(),
            sink.into_primitive(),
            schedule.into_primitive(),
            expiry.into_primitive(),
            output.into_primitive(),
            stats.into_primitive(),
            upstream.into_primitive(),
            capacity,
            chunk,
        );
        (
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dq),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dk),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dv),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(ds),
        )
    }
}

#[derive(Clone, Debug)]
pub struct HolaReadOutput {
    pub output: Tensor<4>,
    /// Descending score, then ascending original position; empty slots are -1.
    /// Nondifferentiable, like the internal position schedule.
    pub final_positions: Option<Tensor<3>>,
}

/// Exact original-KV memory; Q/K learned RMS norms and output gate stay external.
///
/// Inputs are [batch,time,head,channel], masks/documents [batch,time], sink [head].
/// The sink is a learned logit with zero value. The prior chunk's cache and causal
/// current chunk are visible only for matching documents and valid tokens. Scores
/// select membership but are not differentiated. Padding emits zero; selection
/// uses the last physical slot's document even when that slot is padding.
/// Supports first-order Q/K/V/sink derivatives, f32 key32/value64, capacity 1..64,
/// chunk 1..16, and CubeCL GPU execution only. No higher-order VJP or CPU fallback.
pub fn hola_read(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scores: Tensor<4>,
    valid: Tensor<2>,
    documents: Tensor<2>,
    sink: Tensor<1>,
    capacity: usize,
    chunk_size: usize,
    recency: bool,
    return_final_positions: bool,
) -> HolaReadOutput {
    assert!((1..=16).contains(&chunk_size), "HOLA chunk must be 1..=16");
    let [batch, time, heads, _] = q.dims();
    let boundary = time.div_ceil(chunk_size);
    let (output, schedule, _, _) = <Dispatch as HolaOps>::hola_read(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        scores.into_dispatch(),
        valid.into_dispatch(),
        documents.into_dispatch(),
        sink.into_dispatch(),
        capacity,
        chunk_size,
        recency,
    );
    HolaReadOutput {
        output: Tensor::from_dispatch(output),
        final_positions: return_final_positions.then(|| {
            Tensor::<4>::from_dispatch(schedule)
                .slice([0..batch, 0..heads, boundary..boundary + 1, 0..capacity])
                .reshape([batch, heads, capacity])
        }),
    }
}
