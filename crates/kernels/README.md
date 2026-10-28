# KronLSTM and supporting kernels

KronLSTM is the crate's only recurrent scan. HOLA selection/read, normalization,
convolution, projections, optimizer updates, wavelet, oHC, Lngram and joint
inference kernels remain available to the model.

The default build has no CubeCL runtime dependencies. `vulkan` or `cuda` enables
the selected runtime and custom operators. Fusion and autotune are not
implicitly enabled. The optional `fusion` feature registers supported deferred
forward and backward operations for Burn's fusion backend, with exact output
metadata and the same CubeCL kernels. Analytic VJPs remain attached to Burn's
runtime autodiff dispatch; there is no fallback to an unrelated CPU operator.
CUDA support is only a Cargo feature and is not evidence of remote execution.

See `../mixers/README.md` for runnable commands, error tolerances and the
distinction between implemented checks and observed validation.

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

## Short-chain joint inference

The explicitly selected Codpula kernels require `vulkan` or `cuda`; CPU callers
must select their tensor implementation. There is no host transfer or silent
replacement. All inputs use the same device and float dtype, either f32 or f64.
Device dtype support is checked before launch; no f64-to-f32 conversion is made.
Strided inputs and 32-bit element addresses are supported. Real model dimensions
are K=32, V=257, and 1<=L<=16; smaller K and V work for independent reference checks.

Public runtime-dispatch `Tensor<D>` functions:

| Function | Inputs | Result |
|---|---|---|
| `codpula_leaf_normalizers` | logits `[C,L,V]`, log_pi `[K,V]`, valid `[C,L]` | log local normalizers `[C,L,K]` |
| `codpula_leaf_backward` | same inputs, normalizers, upstream `[C,L,K]` | derivatives `[C,L,V]`, `[K,V]` |
| `codpula_log_partition` | log evidence `[C,L,K]`, log transition `[C,K,K]`, shared log root `[K]` or flattened contextual log roots `[C*K]`, valid `[C,L]` | logZ `[C]` |
| `codpula_chain_statistics` | same chain inputs | detached logZ, gamma `[C,L,K]`, summed_xi `[C,K,K]` |
| `codpula_chain_backward` | gamma, summed_xi, upstream `[C]` | log-evidence, log-transition, log-root derivatives |
| `codpula_token_marginals` | logits, log_pi, normalizers, gamma, tokens `[C,L]`, observed `[C,L]`, valid | detached `[C,L,V]` |
| `codpula_emission_flows` | same readout inputs | detached tied emission occurrence counts `[K,V]` |

Masks are binary floats; tokens are integral float IDs. Each chain's valid mask
must be a prefix of ones. An all-zero row is inactive: logZ, statistics and
derivatives are zero. Padding does not read padded logits/evidence.
Unsupported shapes/dtypes/devices fail; invalid evidence and non-prefix chain
masks produce nonfinite logZ, never a fabricated conditional distribution.
Callers must reject nonfinite evidence
mass before using a conditional readout. The caller also supplies normalized
log rows, constructs normalized fixed-token evidence, applies any temperature
to neural potentials, and masks observed-token-potential derivatives as required.

The analytic first-order logZ derivative is gamma for log evidence and the sum
of posterior edge occurrences for each chain's shared log transition. For a
shared `[K]` root, its derivative sums initial posterior occurrences across
chains; for flattened contextual roots `[C*K]`, autodiff retains each chain's
initial posterior separately. The direct `codpula_chain_backward` helper keeps
its original shared-root derivative contract. This shape handling uses the
same kernels, not a new execution selector or a different distribution.
Leaf-normalizer derivatives include both neural logits and shared log_pi, summed
across tied occurrences. These derivatives are with respect to the supplied
log weights; row normalization and grouped gates remain differentiable caller
operations. Direct VJP functions return detached tensors. Readout/statistics
functions deliberately do not promise gradients or higher-order derivatives.
Optional Burn fusion uses opaque deferred calls to these same kernels.

One 32-lane workgroup handles each chain's stable log-space forward/backward.
It retains at most 16x32 alpha values and a 32x32 transition matrix in shared
memory, plus two 32-element beta buffers and one scalar: 1,601 elements, 6,404
bytes for f32 or 12,808 bytes for f64. Each lane sums its own outgoing edge
occurrences. Global chain outputs retain `C + C*L*K + C*K*K` elements; backward
allocates `C*L*K + C*K*K + K` result elements with two dispatches.
An untracked log-partition call omits backward messages and posterior outputs,
allocating only its `C` logZ results; training retains the posterior statistics
needed by the analytic VJP. Posterior writes are enclosed in a positive
compile-time branch, not a compile-time early `terminate!()`: CubeCL's return
marker alone does not exclude expansion of subsequent statements. This keeps
the untracked path from writing through its unused evidence/transition aliases
and preserves prepared inputs across repeated inference queries.
Vocabulary normalizers use one 128-lane workgroup per leaf and one dispatch.
Their full VJP uses two dispatches, allocating only its two result tensors.
Autodiff skips the allocation and dispatch for each untracked parent; neural-only
training therefore does not compute a discarded shared-base derivative.
Marginals and tied emission counts each need one dispatch and only their result tensor.
No path allocates a `[C,L,K,V]` tensor or uses floating-point atomics.

`codpula_kernel_counters()` returns cumulative relaxed-atomic call counts,
submitted dispatches and allocated output bytes. Differences around synchronized
benchmark sections measure this path's submitted work, not peak allocator use,
device completion or runtime. Counters are process-global; do not mix concurrent
workloads when attributing them. All numeric and speed checks remain the
integration owner's responsibility; no measured speedup is asserted here.

The alternatives are the unchanged tensor solver, these whole-chain kernels,
and a multi-dispatch parallel message implementation. The whole-chain design
targets launch/intermediate overhead for L<=16; the tensor path remains the
comparison and default until synchronized isolated and integrated measurements
show a beyond-noise improvement. Parallel messages may help much longer chains
but add dispatches and temporary storage at these short lengths.
