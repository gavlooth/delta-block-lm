# Causal Lngram injections

## Architecture and source

The component is ported from `reuse/crates/dre-lngram/src/lib.rs`, with the
independent scalar reverse-mode oracle from its `tests/lngram/oracle.rs`.
The migration inputs are identical to the relation-extractor source snapshot.
Equations follow the predecessor's `docs/references/lngram.md` and
[Lngram §2.2–2.5 and Appendix A](https://arxiv.org/html/2605.24869v1).

The predecessor schedule is `[L,W,N,D,F,H,W,N,D,F,H,W,N,D,F]`:
`dre-encoder/src/schedule.rs` explicitly requires **three N stages**. Each N is
one module with **two n-gram orders, `[2,3]`**, not two stages. The LM maps this
to `lngram_layers: [0,1,2]` in the four-block target configuration, independently
of `lngram_orders: [2,3]`. Every selected block has a separate RMSNorm, module
and residual addition before its recurrent mixer (after wavelet when composed).
No Step 8 components are introduced.

All `lngram_*` options are absent from historical serialized configurations by
default. An empty layer list allocates no new parameters and executes the old
forward path. Enabling an injection changes serialized architecture identity
and needs a new run; existing checkpoints are not reinterpreted.

## Causal adaptations

- Runtime layout is exclusively tensors: `valid` and document IDs computed as
  `cumsum(reset * valid, time)`. No host document layout or readback is used.
- An order-n address reads the suffix ending at t, with the oldest symbol the
  least-significant base-K digit. Every suffix position must be valid and have
  the same document ID as t. Incomplete suffixes are reported as address -1.
- Current and counterfactual rows are selected to zero **before** subtraction or
  probability multiplication at incomplete suffixes. Poisoned inactive rows
  therefore do not enter surrogate arithmetic.
- Input padding is selected to zero before either input normalization; output
  padding and padded input gradients are exactly zero. Prefix positions still
  receive the original key/value projection biases: zero memory does not mean
  skipping the readout.
- The predecessor's bidirectional 3-tap smoothing is replaced with three causal
  taps at `[t, t-dilation, t-2*dilation]`, dilation `max(orders)` by default.
  Sources outside the document or at padding read zero. Tap row 0 is current,
  row 1 the first lag and row 2 the second lag. No right extension is used.
- FP32 is the active model precision; the predecessor's mixed-BF16 projection
  switch is not carried into this FP32 LM.

The discretizer uses strict `z > 0` bits (ties map to zero). Both surrogate
rules are preserved: `one_bit` (default) and `local_expectation`. Their detached
counterfactual rows add no table gradients, and inference skips all surrogate
work. Table initialization remains zero; projection weights are Normal(0,.02),
projection biases zero, RMSNorm epsilon 1e-6 and smoothing weights zero.

## Dimensions, optimizer and memory

For d_model 256, the target uses 32 routes, 4 bits and memory_dim 8. This halves
the predecessor's 64 routes in proportion to model width and keeps its alphabet
and per-route row width. The concatenated readout width is 256. Each injection
has `32 * (16^2 + 16^3) = 139,264` rows, or 1,114,112 FP32 entries:
**4,456,448 table bytes per injection; 13,369,344 bytes for all three**.
The two dense Adam moments add 26,738,688 bytes for all tables; table gradients
are dense too, although their nonzero support is only the indexed rows.

Table parameters use their own production optimizer group, learning rate x5
and weight decay zero. The x5 multiplier is this repository's choice inherited
from its predecessor, not the official implementation's default multiplier
of 3.0. Both tensor and fused AdamW use that group. With fresh state, an
unindexed row is unchanged even if its stored
value is nonzero. As with ordinary Adam, a row indexed on an earlier step can
subsequently move through momentum; this is not a sparse-moment optimizer.
`LanguageModel::lngram_table_ids` exposes the exact table parameter IDs.

At B8/T512 the one-bit detached counterfactual difference tensors retain
`B*T*R*bits*memory_dim*sum(orders)*4 = 83,886,080` bytes per injection,
251,658,240 bytes for three, excluding probabilities, projections, normalization
and smoothing intermediates. Local expectation replaces `bits` with `K=16`,
so this term is four times larger. These are logical tensor sizes, not a
measured physical memory peak. The benchmark also samples allocator pool
reservations and counts issued CubeCL launches.

## Execution designs and numerical gate

`lngram_backend: tensor` remains the default port: tables reshape to
`[routes, rows_per_route, memory_dim]` and gather within each route. Its
production `delta_kernels::lngram_table_lookup` helper preserves the native
gather forward and integer indices; GPU autodiff uses the destination-owned,
fixed-position-order table VJP described below. CPU retains the existing
native gather VJP. The real alternative `flat_select` forms global row IDs and
selects axis zero directly. They share all parameters, addressing, readout and
surrogate equations. The predecessor motivated route-gather by its more
parallel repeated-index backward scatter; the historical paired measurements
below predate this deterministic Tensor VJP correction.

The GPU-only, non-fusion `fused_gpu` path caches exact hard route symbols once
for all orders, then combines suffix addressing and current-row gather in one
kernel per order. Its analytic VJP
reads counterfactual rows on demand and reduces `g dot (E1-E0)` directly into
each logit gradient; LocalExpectation enumerates its alternative symbols
inside the kernel. No expanded counterfactual tensor is retained. Table
gradients have exclusive `(route,column)` owners, each accumulating positions
in chronological `(batch,time)` order into a zero-filled dense gradient.
There are no floating-point atomics. A direct regression checks bitwise
repeatability on repeated row hits.

`lngram-reference` runs the sequential FP64 oracle against output, input and
**every** parameter gradient under `2e-5 + 3e-4*abs(reference)`, for every
available backend and both surrogates. Cases include packed resets, short documents,
trailing padding, empty rows, zero inputs, strict-zero bits, strided inputs,
repeated row hits and poisoned inactive lookup rows. The oracle's continuous
chain-rule gradients are also independently checked by directional central
differences with hard addresses held fixed. Causality tests perturb a token,
an entire later document and a second batch row. A small three-injection LM
compares logits, all gradients and updated parameters to CPU tensor execution,
and checks the production table optimizer group's fresh-state row support.

### Tensor duplicate-index raw-gradient determinism

The retained GDN2 extended full-model Vulkan fixture exposed an exact L1
common-gradient mismatch at `blocks.0.lngram.tables.0.weight`. Loss and logits
had already passed. The failure was in the **raw gradient before any optimizer
update**, not in updated weights. This evidence is distinct from the clipping
optimizer correction in `64590d18`; that optimizer is unchanged here.

The clean-graph invariant matters: `sbd.rs::training_loss_profiled` always
executes one `model.forward_profiled` and the same `model.loss`. The contextual
arm differs through zero-valued context-head parameter terms. L1 per-block
losses slice that same `clean_logits`; there is no cached noisy forward or
circuitflows call. The factorized clone removes only Codpula. None of these
properties, the exact singleton/restart gates, or their equality requirements
is changed to accommodate the failure.

The original Tensor route lookup was
`reshape([routes,rows,memory_dim]).gather(1,index).swap_dims(0,1)`.
In Burn autodiff 0.22.1, `ops/tensor.rs:1016–1037` implements the Gather VJP
with `float_scatter` Add. CubeCL 0.22.1
`kernel/index/scatter.rs:72–120,160–220` selects floating-point atomic Add on
Vulkan and explicitly documents nondeterministic sums for duplicate indices.
The CPU implementation retains serial, owned-row accumulation. Changing the
optimizer cannot repair a raw-gradient discrepancy produced before it runs.

The narrow correction keeps native gather forward and integer index semantics.
For GPU table gradients, each destination `(route,row,column)` has one owner:
it starts at FP32 `+0` and adds matching upstream values in ascending flattened
`(batch,time)` position order. There are no floating-point atomics, parallel
reduction trees, or float-encoded indices. This is the existing Tensor Lngram
table lookup VJP, not a generic scatter replacement, a new Lngram mechanism,
an upstream vendor patch, or a backend/default-selector change.

`lngram-reference` now permanently exercises the production Tensor lookup:
the existing native gather on CPU and the exported custom-VJP helper on Vulkan.
The fixed fixture is B3/T257, R3,
11 rows per route, memory width 5, order 2, with packed resets and an empty
batch row. Only five rows per route receive valid lookups, creating many
duplicate hits; signed, cancelling FP32 upstream weights provide the table
VJP. Padding and incomplete suffixes address a legal NaN-poisoned row, removed
by the existing production `full_ngram` validity mask **before** consumer
arithmetic. Unused and padding-only table rows must have exactly zero gradient.
Both contiguous and transposed, noncontiguous tables and integer indices are
covered. An independent scalar FP64 sum uses the actual uploaded FP32 weights;
the numerical rule remains `2e-5 + 3e-4*abs(reference)`.

Each layout runs backward **32 times**, comparing all raw FP32 table-gradient
bits, before clipping or optimizer work, rather than a tolerant comparison of
updated weights. Forward bits must match native gather. The JSON
`duplicate_index_table_gradients` report records shapes, valid/masked lookup
counts, duplicate hits, repeated/unused rows, maximum hits per row, maximum
raw-gradient absolute error and tolerance ratio, repeat bit identity, every
raw-gradient hash (FNV-1a over little-endian FP32 bits), and the exact
accumulation contract. Ordinary `checks` also retain the output and gradient
errors against the independent FP64 reference.

CPU and Vulkan `lngram-reference` commands pass under
`artifacts/codpula-context/{lookup-cpu,lookup-vulkan}/`. Each layout has 1,074
valid lookups, 1,059 duplicate hits, 18 unused rows, and 32 bit-identical
raw-gradient repetitions. Vulkan maximum raw-gradient absolute error is
`1.862645149230957e-6`, tolerance ratio `0.00330107578025247`; native forward
bits match exactly. No equality requirement or numerical tolerance changed.

One work-item owns a `(route,column)` stripe: zero its table rows, then scan
positions in order, updating the uniquely owned destination cells. Work is
`O(R*rows*M + P*R*M)` with one dense table-gradient output and no expanded
`[P,rows,M]` allocation. The compatibility Tensor path can expose less GPU
parallelism; the adopted `FusedGpu` implementation is unchanged. Historical
timing tables below do not measure this correction. It is separate from
`64590d18`; no baseline rerun or fixture-backend switch was used to hide the
exposed failure.

The unchanged full Vulkan Codpula model gate also passes after this correction:
exact singleton common raw gradients, zero circuit learning on L1, and exact
saved/restored optimizer continuation are retained. Evidence:
`artifacts/codpula-context/vulkan/codpula-model-reference.json`; the failing
pre-correction log remains in `artifacts/codpula-context/commands/`.


## Commands

The exactly-zero-input cancellation probe keeps the same unnormalized random
VJP seeds as the ordinary fixture. The ordinary cases compare with the ideal
sequential FP64 equations. The zero-input probe additionally observes the two
actual FP32 normalization-output adjoints and independently evaluates their
local input VJPs in FP64; it records this oracle distinction explicitly, as in
the HOLA cancellation regression. An initial ideal-FP64 comparison failed on
one input-gradient element: FP32 0.1962738037 versus FP64 0.1964066759,
absolute error 1.32872e-4 versus tolerance 7.89220e-5. No tolerance is changed.
The predecessor and this component both use epsilon 1e-6 for all internal RMS
norms; the LM's outer norms also use 1e-6, while inherited mixer norms use the
Burn default 1e-5. Inverse RMS is approximately 1000 at zero.

From the workspace root:

```sh
../build.sh s7-lngram
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-reference artifacts/step7/lngram/cpu-reference.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-reference artifacts/step7/lngram/gpu-reference.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-benchmark artifacts/step7/lngram/b1-t128.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-benchmark artifacts/step7/lngram/b8-t512.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-smoke artifacts/step7/lngram/smoke.json
```

The benchmark alternates paired execution order with two warmups and eleven
measured trials. Isolated forward records autodiff, and F+B uses the same
square-sum objective. Complete guarded steps include real batch preparation,
upload, loss readback, every parameter gradient, finite checks, clipping and
AdamW. Cold initialization/JIT/warmups and report IO are outside warm samples.
The learning smoke is a paired 200-step tiny synthetic byte-pattern LM; its
purpose is decreasing finite loss, not heldout quality.

## Executed numerical and learning results

On Intel Arc MTL integrated graphics, Mesa 26.2.4, Vulkan, FP32, without fusion
or matmul autotuning, both backend designs and both surrogate modes passed the
13-case component gate. The maximum elementwise error/tolerance ratio was
0.06873 on CPU and GPU. The largest absolute errors (2.25083e-4 CPU and
3.76063e-4 GPU) occurred in the amplified zero-input local-VJP probe, not the
ordinary input case; all are within the unchanged elementwise rule.
The ordinary one-bit input-gradient maximum errors were 8.20144e-7 CPU and
9.46414e-7 GPU. Packing, poisoned inactive rows, exact padding, repeated table
row accumulation and causal perturbation checks all passed.

The three-injection LM gate checked all 108 parameter tensors. CPU tensor and
global-select execution matched exactly. With the target's actual fused
scan/HOLA/projection/convolution/normalization/epilogue/update selectors, GPU
versus CPU maxima were 4.76837e-7 for logits, 8.39233e-5 for all gradients and
2.94484e-6 for updated parameters; every element passed the same tolerance.
All six table parameters passed the fresh-state indexed-row test under the
production optimizer group, including the x5 learning-rate delta and zero
decay on nonzero unindexed rows.

The 200-step paired synthetic byte-pattern smoke completed with seed 1,
B1/T16/D32 and three modules (R2, bits2, memory_dim3). Mean loss over the first
and last ten steps:

| Arm | First ten | Last ten |
|---|---:|---:|
| No Lngram | 5.225845 | 0.059356 |
| Three Lngram injections | 5.227457 | 0.061236 |

Both trajectories had finite checked gradients throughout. This verifies that
the component trains without divergence on the tiny task, not that it improves
language modeling quality. Raw losses and the exact resolved configuration are
in `artifacts/step7/lngram/smoke/lngram-smoke.json`; component and CPU-model
gate reports are in the `cpu-reference/` and `gpu-reference/` directories.

## Paired cost and execution decision

Each row has eleven alternating, interleaved warm samples after two warmups.
Both workload configs are copies of `configs/lm512-optimized.json` with the
component fields and artifact locations made explicit. They use one seed,
four D256 blocks, H4/K32/V64, chunk16, capacity16, the same common initial
parameters and the same real byte-LM batches. The off arm removes only the
Lngram modules and their external norms from the common initialized model.

Median synchronized isolated cost of **one injection**, milliseconds:

| Workload | Lookup design | F (AD recording) | F+B | Observed F+B launches |
|---|---|---:|---:|---:|
| B1/T128 | Route gather (`tensor`) | 4.571 | 10.258 | 553 |
| B1/T128 | Global select (`flat_select`) | 6.405 | 50.948 | 560 |
| B8/T512 | Route gather (`tensor`) | 51.818 | 169.857 | 547 |
| B8/T512 | Global select (`flat_select`) | 54.276 | 1391.890 | 561 |

Median complete guarded optimizer step with **three injections**:

| Workload | Off | Route gather | Global select | Off / route / select launches |
|---|---:|---:|---:|---|
| B1/T128 | 70.199 ms | 103.530 ms | 224.129 ms | 1701 / 3662 / 3683 |
| B8/T512 | 854.888 ms | 1367.645 ms | 4727.498 ms | 1718 / 3661 / 3703 |

These are full steps, not isolated cost multiplied by three. Table updates,
all-parameter checks, per-parameter clipping, optimizer state and real input
preparation/upload are included. The full parameter count is 2,379,568 off and
6,221,104 on: 3,841,536 added parameters, or 15,366,144 FP32 bytes, of which
13,369,344 are the tables. End-of-benchmark allocator pool reservations were
314,889,792 bytes at B1/T128 and 1,394,923,264 at B8/T512. They include all three
trajectories and allocator reuse; they are not per-arm live memory or a
physical peak measurement.

Within-trial paired median differences and seeded 20,000-resample bootstrap
95% intervals (milliseconds):

| Workload | Three route-gather injections minus off | Route-gather saving versus global select |
|---|---:|---:|
| B1/T128 | 34.027 [30.283,37.841] | 115.248 [111.095,133.909] |
| B8/T512 | 507.525 [503.429,523.605] | 3358.087 [3346.566,3385.108] |

At B8/T512 the isolated route-gather F+B saving over global select is
1223.032 ms [1213.121,1228.674]; the forward-only saving is only
2.652 ms [1.465,4.196]. The backward scatter is the decisive difference, not
a large forward speedup. Raw samples, launch histograms and resolved configs
are in `b1-t128/lngram-benchmark.json` and
`b8-t512/lngram-benchmark.json`; `paired-summary.json` records every paired
difference and bootstrap settings.

**Initial library-only decision (superseded by the analytic comparison below):**
route gather outperforms global select. The copied port's three-injection
overhead was about 0.51 seconds per guarded step, roughly 60% over its off arm.
This is why the counterfactual path was investigated rather than accepting a
comparison with only a slower library formulation. These original numbers
remain visible; they are not the final optimized-baseline comparison.

No dependencies, backend features or crate boundaries changed. The initial
port rebuild took 5.94 seconds of Cargo build time (first integration rebuild:
10.62 seconds); the last exercised shared-cache rebuild took 6.06 seconds.
Queue wait is
separate. Workspace regression tests remain part of the parent's final merged
verification, not an additional parallel test run from this slice.

## Attribution and analytic memory design

The 169.857 ms isolated F+B number above is **one injection**; the 507.525 ms
guarded-step overhead is **three injections**. Multiplying the isolated cost
by three gives 509.571 ms, close to the observed overhead. There is not a
two-thirds unattributed gap. Independent stage medians still are not additive
full-step utilization measurements.

The first B8/T512 stage attribution used two warmups and eleven synchronized,
alternating F/FB trials, resident independent leaves, real production operators
and sum roots. Times are per injection; launch counts include the fixture roots.
Counterfactual current rows are prepared outside timing because production
reuses its actual current-row gather.

| Copied tensor-port stage | F ms | F+B ms | F / F+B launches |
|---|---:|---:|---:|
| Discretizer RMSNorm + GEMM | 5.617 | 11.947 | 9 / 36 |
| Probabilities, hard codes, suffix addresses | 2.080 | 2.164 | 94 / 108 |
| Bigram gather, mask, dense table VJP | 1.086 | 2.325 | 24 / 30 |
| Bigram OneBit counterfactual term | 8.473 | 27.582 | 55 / 86 |
| Trigram gather, mask, dense table VJP | 1.174 | 2.521 | 38 / 44 |
| Trigram OneBit counterfactual term | 13.207 | 42.571 | 89 / 136 |
| Key/value GEMMs, query/key norms, gates | 24.841 | 71.279 | 31 / 108 |
| Causal smoothing norm/taps/SiLU/residual | 4.555 | 16.690 | 67 / 133 |
| Fused memory, both orders and analytic VJPs | 1.953 | 36.647 | 7 / 19 |

Dense table AdamW for both tables of one injection was 1.584 ms, eight
launches, using real gather-gradient support and warmed moments. This includes
clipping, x5 learning rate and zero decay, but excludes gradient generation and
whole-model finite guards. The gather F+B rows already include zero-filled
dense table-gradient construction and repeated-index accumulation. The readout
and counterfactual path, not table AdamW, dominate this workload.

The tensor OneBit formulation does **not** separately expand both forced-zero
and forced-one rows. It reuses the current row `[B,T,R,memory_dim]`, expands one
bit-flipped row per bit, and forms
`sign * (current - flipped) = E1-E0`. The `[B,T,R,bits,memory_dim]` difference
is retained by the probability multiplication for every n-gram slot. Across
orders `[2,3]`, this is 83,886,080 bytes per injection and 251,658,240 for three.

Attribution also exposed a Burn 0.22 detail: `detach()` preserves a leaf's
`require_grad` setting. The predecessor's detached table therefore became a
new tracked leaf, computing orphan counterfactual-table gradients even though
those gradients could not reach the original table parameter. The tensor port
now explicitly disables tracking on frozen table/current-row values. This does
not change any trainable gradient or forward result; it removes unwanted
work. The original attribution and first fused comparison remain archived,
but the final comparison must use this corrected tensor baseline.

The analytic operator instead retains the original logits (2,097,152 bytes per
injection at B8/T512, shared across its order operators) and reads each needed
counterfactual row during backward. It retains **zero expanded counterfactual
bytes** for either surrogate. Current lookup output, projection intermediates,
tables, dense table gradients and two dense Adam moments remain real costs.
Each logit VJP owns its `(batch,position,route,bit)` output and walks only the
at-most-order suffix endings that depend on it. Each table VJP owns one
`(route,column)` and walks positions chronologically, so no floating-point
atomic or nondeterministic scatter is used. One shared code kernel precedes
one gather and two backward kernels per order, plus dense zero initialization.
Only route symbols, at most 16 bits, are stored as exactly representable FP32
integers; table addresses are still computed with integer shifts and offsets.
The shared symbol cache is 524,288 bytes per injection at B8/T512, independent
of the number of orders: 1,572,864 bytes total instead of 251,658,240 expanded
counterfactual bytes. It avoids recomputing all bits for each row column,
logit adjoint and table-gradient owner.

Raw stage report: `artifacts/step7/lngram/attribution/lngram-attribution.json`.

## Corrected baseline, refinement and final adoption

The first fused design looked substantially faster than the copied tensor
port, but that was not an adequate adoption comparison after discovering
orphan table VJPs. With frozen counterfactual rows, the tensor bigram/trigram
counterfactual F+B medians became 13.291/19.509 ms, with 71/113 launches,
instead of 27.582/42.571 ms and 86/136 launches. These are separate attribution
sessions, not a paired whole-model speedup.

Against that corrected tensor implementation, the uncached fused design's
21-pair B8/T512 guarded-step saving was only 4.956 ms with bootstrap 95%
interval [-9.877,7.614] ms. It was **not adopted** on this result. Its
33.774 ms whole-memory F+B cost was too close to the corrected library path.

Caching route symbols reduced the address recomputation in both analytic
VJPs. An initial per-order-cache candidate passed the same numerical gate and
gave a 28.979 ms paired B8 saving [22.531,31.419] over corrected tensor. The
final implementation shares one cache across all orders, eliminating the
duplicate cache allocation and code-computation launch. Rejected/earlier raw
reports remain in `clean-tensor-*` and `cached-*`; their summaries are
`clean-tensor-paired-summary.json` and `cached-paired-summary.json`.

### Final ordinary paired steps

Final shared-cache run: **21 pairs at B8/T512**, eleven at B1/T128, both after
two warmups. Config, device, dtype, masks, seed, parameters and data budgets are
unchanged. The tensor arm includes the tracking fix; no orphan
counterfactual-table gradient work is intentionally left in the baseline.

| Workload | Off | Corrected tensor | Shared-cache fused | Off / tensor / fused launches |
|---|---:|---:|---:|---|
| B8/T512 | 871.605 ms | 1280.720 ms | 1259.083 ms | 1718 / 3547 / 2881 |
| B1/T128 | 67.978 ms | 96.659 ms | 89.325 ms | 1701 / 3548 / 2861 |

Within-trial paired median fused savings over corrected tensor:

- B8/T512: **19.398 ms [13.096,27.921]**.
- B1/T128: **8.294 ms [5.224,12.066]**.

Intervals use the same seeded 20,000-resample paired-median bootstrap as above.
The intended-shape whole-step improvement is small, about 1.7% by the ratio of
ordinary medians, not the larger isolated forward gain. Three fused injections
still add **390.207 ms [378.795,395.434]** versus off. Their per-injection
isolated F+B is 129.217 ms; three such costs are about 387.652 ms, consistent
with the integrated overhead but not a substitute for measuring it.

| Workload | Corrected tensor F / F+B | Shared-cache fused F / F+B | Tensor / fused F+B launches |
|---|---:|---:|---|
| B8/T512 | 54.218 / 136.758 ms | 37.684 / 129.217 ms | 509 / 287 |
| B1/T128 | 5.289 / 10.025 ms | 2.766 / 7.889 ms | 515 / 286 |

At B8/T512 the paired isolated F+B saving is 7.890 ms [6.472,9.820], while
forward saves 16.176 ms [14.532,17.517]. Extra analytic backward computation
offsets part of the forward gain. Launch reduction alone is not the decision.

### Final per-injection attribution

| Production stage | F ms | F+B ms | F / F+B launches |
|---|---:|---:|---:|
| Discretizer RMSNorm + GEMM | 5.439 | 11.303 | 9 / 36 |
| Tensor probabilities/codes/addresses | 1.594 | 1.943 | 94 / 108 |
| Tensor bigram gather and dense VJP | 0.892 | 1.918 | 24 / 30 |
| Tensor bigram OneBit term | 8.039 | 12.290 | 55 / 71 |
| Tensor trigram gather and dense VJP | 1.324 | 2.722 | 38 / 44 |
| Tensor trigram OneBit term | 12.812 | 18.383 | 89 / 113 |
| Key/value GEMMs, norms and gates | 21.499 | 69.651 | 31 / 108 |
| Causal smoothing | 3.717 | 16.311 | 67 / 133 |
| Shared-cache fused memory, both orders | 1.438 | 23.711 | 8 / 20 |

Both-table AdamW is 1.236 ms, eight launches. The readout GEMMs/norms/gates are
now the largest remaining group; ordinary library matmuls remain in use rather
than adding an unmeasured projection implementation. These stage measurements
are independent leaf VJPs and remain non-additive.

### Final gates, memory and configuration

`optimized-reference/lngram-reference.json` returned `passed: true` for
**22 cases**: three backend designs, both surrogates, all outputs/gradients,
each backend's causality checks and bitwise deterministic table VJPs. Maximum
error/tolerance ratio remains 0.06873. The zero-input observed-intermediate
oracle distinction described above remains explicit; no tolerance changed.
The three-injection LM checked all 108 parameter tensors. Fused-versus-CPU
maxima were 4.76837e-7 for logits, 8.96454e-5 for all gradients and 2.53692e-6
for updated parameters. All six tables passed the fresh-state indexed-row,
x5 learning-rate and zero-decay checks.

The final fused 200-step smoke again passed: first-ten/last-ten losses were
5.227456/0.061236 on and 5.225845/0.059356 off, with finite checked gradients.
This is still a learning sanity check, not a quality result.

Table and optimizer storage are unchanged. The fused path retains zero
expanded counterfactual tensors and adds only the shared 0.5 MiB symbol cache
per injection alongside its existing logits. Final benchmark-end pool
reservations were 1,287,945,408 bytes at B8/T512 and 267,704,512 at B1/T128.
Both processes include tensor and fused trajectories, so these are not
per-arm live-memory or physical-peak comparisons.

**Adopt `lngram_backend: fused_gpu` for the measured explicit GPU configuration.**
It passes the numerical/CPU-model gates, has a beyond-noise intended-shape
guarded-step gain against the corrected library baseline, and removes the
large counterfactual retention. Omitted configuration still means no injections
and the tensor default. The explicit adopted config is
`artifacts/step7/lngram/optimized-b8-t512.json`. Reserve approximately 0.40
seconds of guarded-step overhead for all three injections at this workload;
remeasure the composed model before increasing context or routes. No Step 8
work or larger quality run is included.

Final raw reports are under `optimized-b8-t512/`, `optimized-b1-t128/`,
`optimized-attribution/`, `optimized-reference/` and `optimized-smoke/`.
`optimized-paired-summary.json` contains the paired differences, resampling
seed and confidence intervals.

### Exact final commands

```sh
../build.sh s7-lngram
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-reference artifacts/step7/lngram/optimized-reference.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-benchmark artifacts/step7/lngram/optimized-b8-t512.json --warmups 2 --trials 21
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-benchmark artifacts/step7/lngram/optimized-b1-t128.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-attribution artifacts/step7/lngram/optimized-attribution.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-lngram/delta-experiment lngram-smoke artifacts/step7/lngram/optimized-smoke.json
```
