# Causal GDN-2 and KronLSTM kernels

## GDN-2 chunk-WY

This crate ports the causal chunked CubeCL forward and backward operators from
`reuse/crates/dre-gdn2`. Key width 32, value width 64 and computational chunk
length 16 are preserved; batch, sequence length and head count are dynamic.
Gate accumulation and recurrent state use f32. Packed-document limits are
inclusive token indices; `doc_end < doc_start` marks padding. Valid tokens whose
`doc_start == token_index` reset before the update. Padding has identity decay
and zero update. Direct products handle reset zeros without inverse cumulative
decays.

The default build has no CubeCL runtime dependencies. `vulkan` or `cuda` enables
the selected runtime and the custom operator. Fusion and autotune are not
implicitly enabled. The optional `fusion` feature registers deferred forward
and backward operations for Burn's fusion backend, with exact output metadata
and the same CubeCL kernels. The hand-written six-input VJP remains attached to
Burn's runtime autodiff dispatch; there is no fallback to an unrelated CPU operator.
CUDA support is only a Cargo feature and is not a claim of remote execution.

`gdn2_scan(q,k,v,alpha,erase,write,doc_start,doc_end)` accepts tensor shapes
`[batch,time,head,key/value]` and returns token outputs. The recurrence is causal;
the predecessor's bidirectional encoder and serial-backend switch are deliberately
not part of this crate. Fusion metadata describes the active causal operator.

The experiment `reference` command calls `delta_mixers::numerical_checks` and,
on GPU, executes this operator's actual forward/backward against the f64 oracle.
Final states are read using 32 diagnostic identity queries with zero updates;
those tokens' cost is recorded separately in report metadata. The GDN-2 training
path uses the common differentiable chunk tensor solver, not this custom operator.
See `../mixers/README.md` for runnable bounded commands, error tolerances and the
boundary between implemented checks and observed validation.

## Fused KronLSTM

`kron_scan(q,k,v,alpha,erase,write,valid,reset,initial,chunk_size)` returns
`KronScanOutput { output, final_state, scores }`. Inputs and output use
`[batch,time,head,channel]`; key width is 32, value width is 64, and scalar
erase/write gates have channel width 1. Initial/final state uses
`[batch,head,32,64]`. Masks are binary f32 `[batch,time]` tensors, not host arrays.
All tensors must use f32 on the same CubeCL device. Strided inputs are supported
within 32-bit element addressing. Logical counts, metadata strides, reachable
spans, backing buffers and all checkpoint/replay/output allocations must fit
`u32::MAX` elements; unsupported ranges are rejected before their allocation or
launch rather than narrowed to 32 bits. Normalization of Q/K/V remains the caller's
responsibility.

For each valid token, using key-major state storage:

```text
bar[i,j] = alpha[i] * previous[i,j]
p = sum(i,j, k[i] * bar[i,j] * v[j])
c = write - erase * p
state[i,j] = bar[i,j] + c * k[i] * v[j]
output[j] = sum(i, q[i] * state[i,j])
score = abs(c) * norm(k) * norm(v)
```

A valid reset substitutes zero for `previous` before decay. Padding ignores
reset, preserves state, and returns zero output and score. Scores are physical
active-edit Frobenius norms; no surrogate cache-selection score is introduced.

Forward uses one 64-lane workgroup per batch/head, with each lane retaining one
value column of state. It writes boundary checkpoints and four scalar auxiliaries
per token (correction, bilinear prediction, key norm, value norm). Backward replays
each chunk from its checkpoint into a reusable per-head scratch buffer, then
reverses the recurrence. Replay uses the saved correction, never inverse decay,
so zero alpha and resets are valid inside or between chunks. Chunk sizes 1..=16
control checkpoint/replay storage and do not change the recurrence.

The analytic VJP includes Q/K/V, channel-wise alpha, scalar erase/write, and
initial state. Output, final state and scores are slices of one differentiable
packed node, so their upstream adjoints accumulate before the same VJP. Masks
are nondifferentiable. Score derivatives choose `sign(0)=0` and zero norm
subgradients at zero-norm keys/values. Only first derivatives are supported.
There are no CPU transfers or atomics in the scan. CPU/reference execution must
select the tensor solver explicitly; an unsupported fused device/dtype/shape is
an error, not an unreported fallback. CUDA is feature-enabled, not locally measured.

The `fusion` feature additionally registers Kron forward/backward and HOLA
selection/read/backward as opaque deferred operations. Metadata describes the
packed Kron output, boundary checkpoints, scalar auxiliaries, HOLA schedule,
expiry and softmax statistics as well as all VJP results. Burn keeps their
dependency graph on the selected GPU; execution invokes these exact kernels
and returns their handles. The custom kernels are not automatically merged
with their neighbors, and their numerical/shape/address contracts are unchanged.

### Design trade-off and measurement boundary

Three options were considered against the unchanged Burn tensor solver:

| Option | Expected benefit | Cost and limitation |
|---|---|---|
| One workgroup/head recurrence with checkpoint/replay (implemented) | One forward and one backward scan dispatch, bounded replay scratch, no Gram/triangular intermediates | Serial time dimension; low occupancy at small batch/head count; shared reductions and backward replay |
| Chunkwise Gram + triangular + boundary/output kernels | More chunk/time parallelism and potentially better throughput for larger contexts/batches | More dispatches, decay/Gram intermediates, and a substantially larger analytic backward |
| Fuse only overlaps/triangular solve, retain matmuls | Smaller initial arithmetic change | Retains chunk loop, numerous tensor launches and intermediates, and matmul dispatch overhead at tiny tiles |

The first option is selected for expected launch-overhead reduction and a small,
auditable VJP, not because a speedup has already been measured. Its checkpoint
storage is `4*batch*heads*ceil(time/chunk_size)*32*64` bytes; backward replay
scratch is `4*batch*heads*chunk_size*32*64` bytes. Per-token auxiliaries use
`16*batch*time*heads` bytes. Backward needs 24.25 KiB workgroup shared memory.
The packed result contains output, final state and scores, without a second
copy of those results.

The tensor solver remains the default until numerical checks, synchronized warm
scan forward/backward, and unchanged full-model AdamW steps establish a measured
gain. HOLA off/on and checkpoint costs must be reported separately. This design
does not imply a speedup for every batch/context or introduce a Step-7 mechanism.

## Normalization and decay

`l2_norm(x)` computes `x / sqrt(sum(x²) + 1e-6)` over the final axis.
`rms_norm(x, gamma, gate, epsilon)` computes
`z / sqrt(mean(z²) + epsilon) * gamma`, where `z=x` without a gate or
`z=x*sigmoid(gate)` with one. It uses each original `RmsNorm` module's
parameter and epsilon: outer block/final norms use 1e-6; inherited mixer and
the **separate** HOLA Q/K norms use 1e-5. No gamma is shared or reinitialized.

One workgroup per row reduces squares; the analytic input VJP reduces
`dot(dy*gamma,z)`. The parameter VJP reduces rows independently per channel,
without atomics. Forward retains one reciprocal norm per row, not normalized
copies of every input. Gated RMSNorm additionally returns the analytic sigmoid
input VJP. Ungated/L2 inactive arguments alias existing handles and have no
autodiff edges or dummy allocations.

`decay(x,bias,scale)` computes
`exp(-exp(scale[h])*softplus(x[b,t,h,k]+bias[h,k]))`, preserving beta 1 and
Burn's threshold-20 softplus identity branch. Its three analytic gradients
cover the projection, original channel bias and original head scale; bias
and scale reductions use no atomics.

All operators register through the same Cube/Autodiff/optional-Fusion
extension mechanism as Kron/HOLA. They require f32 GPU tensors and support
strided inputs with checked 32-bit addressing. Normalization widths are
1..=1024. Masks remain at their original caller boundaries, including the
input/output validity multiplications: padding is not silently moved across a
normalization or made into an extra learned-parameter contribution. Only
first derivatives are supported; CPU execution explicitly keeps tensor ops.

The two candidate scopes and bounded numerical/paired full-step commands are
documented in the mixer and experiment READMEs. `RESULTS.md` records the actual
adoption decision; kernel-count reduction alone does not establish a win.

## Source provenance

| New file | Migration input | Input SHA-256 |
|---|---|---|
| `src/gdn2.rs` | `reuse/crates/dre-gdn2/src/chunked.rs` | `4e247d18965603e0bf061381a7fc2a8f43a2f224402e60e67f5f4f39e49e575e` |
| `src/ops.rs` | `reuse/crates/dre-gdn2/src/ops.rs` | `d63d378ac8e144218fdf84ad759b4be2c367d42ddbf705d191e352638e69a154` |

The arithmetic kernels are unchanged; imports use runtime-neutral CubeCL types. The
custom dispatch wrapper removes the obsolete serial branch while preserving
the source's analytic backward implementation. Fusion uses Burn 0.22's deferred
extension registration for the active operators, rather than predecessor metadata.
Source provenance,
copy date and inherited MIT OR Apache-2.0 license remain recorded in
`reuse/PROVENANCE.json`.

## Causal Lngram memory

`lngram_codes` computes exact hard symbols once per injection.
`lngram_memory` reuses that cache for exact suffix addresses and table gather
per order. Its analytic logit VJP reads current/counterfactual table rows on
demand for OneBit or LocalExpectation; it does not retain expanded
`[B,T,R,bits_or_alphabet,memory_dim]` counterfactual tensors. One invocation per
`(route,column)` owns every table-gradient element in that route and column,
walking `(batch,time)` chronologically. Repeated hits are summed without
floating-point atomics, and a regression checks bitwise repeatability.

Inputs are FP32 strided tensors, with nondifferentiable valid/document tensors.
Incomplete suffixes and padding read no table row. The selector is explicitly
GPU-only and non-fusion; CPU keeps the tensor port. Existing tables, projections
and optimizer records are unchanged. Numerical, attribution, memory and paired
whole-step evidence are in `../../docs/optimization/lngram.md`.
