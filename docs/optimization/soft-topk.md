# GPU soft top-k candidate selection

## Scope and execution identity

The selector is shared by the standalone soft episode and the active SBD
denoiser. `soft_selection_backend` explicitly selects `cpu`, native `gpu`,
or the two-stage `gpu_tiled` implementation. Fusion and matmul autotuning
remain independent. The current feedback equation follows the
[soft-masking paper, Eq. (3)](https://arxiv.org/html/2510.17206v2#S3.E3)
and [official constrained head](https://github.com/IBM/soft-masked-diffusion-language-models/blob/8c3ec65872f6943ad366ffac877a0d6d8cffe29e/coding/train/components/transparency_head.py).
The historical measurements below used the predecessor equation, not this
repair; they are not acceptance or speed evidence for the new math.

The 8.64×/24.44× figures below are isolated complete-feedback-round gains,
not full-model training or decoder speedups. Their raw paired samples are
`artifacts/soft-topk/selection-1792868611613-2908803/`; inclusion of the same
selector in a later model config does not extend that measurement's scope.

The CPU baseline downloads full probabilities, sorts each row with
`total_cmp`/ascending-ID ties, uploads IDs, and gathers selected values.
Both GPU designs replace only discrete selection and state bookkeeping.
Selected probabilities are now normalized over top-k; raw selected mass is
retained separately. No circuit vocabulary support is removed.

## Designs compared

1. **CPU baseline:** full row sort, including host transfer and ID upload.
2. **GPU native:** one stride-aware f32-to-i32 total-order-key kernel followed
   by Burn/CubeCL's native integer arg-top-k reduction. Its documented
   lower-coordinate tie rule produces ascending token IDs for equal keys.
3. **GPU tiled:** pad keys to 256-token tiles with `i32::MIN`, select up to k
   candidates per tile, then select the global k from tile candidates. Tile
   offsets recover original IDs; tile-major ordering preserves ascending-ID
   ties in the second reduction. This compares more row/tile parallelism
   against additional allocations, gathers and launches, using existing
   primitives rather than adding a second custom top-k implementation.

For f32 bits interpreted as signed i32, negative keys are XORed with
`0x7fffffff`; nonnegative keys are unchanged. Signed integer order is exactly
`f32::total_cmp`, including **+0 before -0** in descending order and positive
subnormals before either zero. No epsilon perturbation or lossy float encoding
of token IDs is used. The same keys classify valid input bit patterns, so
negative subnormals cannot become acceptable probabilities through device
float-denormal flushing. Candidate inputs allow finite nonnegative values,
including -0; logits allow finite values and -infinity but reject NaN and
+infinity, including in observed/padded rows.

Burn 0.22's public `argtopk` frontend asserts `k < dim`, although its backend
reduction (also used by `topk_with_indices`) supports `k == dim`. A small
shape-checked call to that same backend integer operation preserves full
support and vocabulary-size-one behavior without a full sort, a CPU fallback
or an unnecessary selected-value output. Width continues to clamp to vocab.

## Device-resident state and gradients

Candidate IDs, normalized weights, raw retained mass, full-distribution
negative entropy, prediction availability and proposal/retained IDs remain
on device. The historical `visibility` field is now only the learned
embedding coefficient, not an adjustable schedule. Permanent observations
have coefficient one, initial unresolved positions and padding zero.
Remasking retains soft information; only temporary reveal views become hard.

A GPU selection reads back **one boolean validity result**, not the
probability tensor. A feedback update reads one logits validity boolean and
one candidate validity boolean. Immediate `Result` rejection is preserved;
there is no silent fallback, suppressed invalid input or unchecked delayed
failure. Final report/debug snapshots intentionally read selected tensors.

Each update replaces candidates from the latest detached prediction.
The learned coefficient is scale times sigmoid of positive steepness times
negative entropy minus the negative center. Scale uses sigmoid; steepness
and negative center use softplus with the official `1e-6` offsets.
Official initialization is scale `1e-6`, center about `-0.75`, steepness
about `6.67`. Student embedding/mask values and all three confidence
parameters remain differentiable; teacher logits and probabilities do not.
The candidate distribution is normalized before embedding expectation, so
discarded top-k mass does not reduce the learned coefficient.

Normalization uses a custom first-order VJP:
`dp = (g - sum(g * normalized)) / mass`.
The centered numerator is formed before direct division. Splitting this into
reciprocal/divisor paths causes `inf - inf` for positive subnormal mass even
when the true gradient of `sum(normalized)` is exactly zero. Sparse selection
and dense references share the corrected operation; all-zero rows keep the
existing denominator-one extension. References now check subnormal outputs
and the constant-loss zero VJP, zero-row derivatives, and honest infinities
for genuinely unrepresentable derivatives. No probability floor, gradient
clamp, or selected-probability detach is introduced.

Ordinary training samples soft students with probability `0.8` and binary
students otherwise; binary draws do not run a teacher or train confidence
parameters. Inference starts binary without an added initial teacher.
Discrete remask counts/support are unchanged and are not learned by this head.
Old head-free checkpoints are incompatible under the shared model-math
revision. Current independent scalar/tensor/CPU/GPU references are in
`soft.rs` and `soft_benchmark.rs`; this repair did not execute them.

## Measurement protocol and interpretation
The rest of this document records historical predecessor-formula measurements.
Its convex-update/visibility references explain those archived artifacts only,
not controls or equations accepted by the current implementation.


Local Intel Arc MTL integrated GPU, Vulkan, Burn 0.22/CubeCL 0.11, f32,
heuristic matmuls, fusion off. Shape fixtures use seed 1307 and resident
softmax probabilities from uniform[-1,1] logits. The primary B8 rows are
128 (=8x16) and4,096 (=8x512); B1 reports16/512 rows. Vocabularies are257
and32,000; widths are4/16/64. The separate inactive causal controls cover
contexts128/512 at both batches.

Each shape compares all three implementations in one process, rotating
CPU/native/tiled order within paired trials. One warmup precedes three warm
samples; the decisive B8/512-position/k16 confirmation uses five. Outer
device synchronization brackets each sample. Selection includes
validation/readback, CPU sort/transfers where applicable, gather and retained
mass. A standalone feedback-round sample includes convex update, selection,
retain/remask and expected embeddings. A separate selection+expectation+
backward sample extracts probability, embedding and mask gradients. The
consumer fixture has embedding width **16**, not the LM's width 256; it is an
integrated existing-operation check, not a forecast of a future denoiser.

All realistic-shape IDs, selected weights and masses are compared with the
CPU path before timing. Initialization/fixture uploads, numerical checks,
first state construction/JIT, warmups, reporting and launch observation are
outside warm samples. First-state construction wall times are retained
separately; the persistent kernel cache may already be warm. Dispatch counts
are measured in a separate warm selection, never while timing.

The real causal LM does not call soft selection. `soft-step` therefore runs
paired identical-start, interleaved ordinary causal LM forward/backward,
finite-gradient checks, clipping and AdamW steps as a **negative control**;
its CPU/GPU selector variants execute identical model work. It is not a
soft-feedback training speedup or an implementation of deferred SBD. Warmup
is one paired step; five warm paired steps are reported at B1/B8 and
contexts128/512. No isolated speedup is advertised as a causal-model speedup.

Final GPU children acquire `flock /tmp/delta-block-lm-gpu.lock` and then
`flock /tmp/delta-block-lm-build.lock`, with the300s compute timeout **inside**
both locks. The sweep releases locks between configurations and records
queue wait separately from compute. Builds use the corrected
`../build.sh softtopk` helper and its private executable
`/tmp/delta-bins/softtopk/delta-experiment`; `.cache/cubecl` is a private
workspace directory. Earlier shared-cache/control measurements are labeled
provisional rather than used as final acceptance.

## Memory and work accounting

For R rows, V vocabulary and k=min(width,V): retained candidate storage is
`8*R*k + 4*R` bytes (i32 IDs, f32 weights and row mass). Native selection adds
`4*R*V` bytes of order keys, boolean classification/reduction temporaries and
the library reduction workspace. It does not materialize a pairwise VxV rank
matrix. The CPU baseline downloads `4*R*V` bytes and uploads `4*R*k` ID bytes
per selection, in addition to its host probability vector and per-row sort
scratch. GPU selection downloads only its scalar status.

For L=ceil(V/256), the measured pre-fix tiled design additionally allocates
a padded `4*R*L*256`-byte key tensor and tile-local IDs/keys/global-ID
intermediates of order `R*L*min(k,256)`. The final unadopted tiled code skips
the padded allocation/copy when V is already divisible by256.
It still performs two top-k reductions and additional packing/gathers, so
launch count alone is not a speedup. At R=4,096,V=32,000,
one probability/key tensor is **524,288,000 bytes**; k64 candidates occupy
**2,113,536 bytes**. Convex feedback still uses the historical dense update
and softmax; removing its O(RV) device tensors is not claimed here.

Reports retain sampled backend pool reservation, not an exact live or physical
GPU peak. Pool reuse, integrated shared memory and allocations inside library
reductions limit that observation. There is no hardware occupancy counter.

Native top-k is a bounded-width reduction, not a full vocabulary sort. Its
sorted-list candidate work is worst-case O(RVk), so requesting full support
k=V can become quadratic; only small full-support reference fixtures are
claimed. The tiled design reduces each tile then its retained candidates,
rather than constructing a quadratic rank matrix. The evaluated widths stay
at 4/16/64.

## Numerical evidence

`artifacts/soft-topk/soft-reference.json` passes all three paths: seven
independent scalar CPU fixtures per path plus the original soft episode and
four-round CPU/GPU transition/VJP comparison per path (24 report entries).
Outputs/masses use f64 arithmetic; analytic CPU VJPs use f64 accumulation
from the f32 fixture values.
Fixtures cover vocab7/257/32,000, clamped/full support, widths1/4/16/64,
genuine transposed strides, zero probabilities, signed zeros, positive
subnormals, repeated ties, hidden and padded rows. Outputs, retained masses
and probability/retained-mass, embedding, mask and visibility VJPs match
the independent scalar oracle at the existing 2e-5 absolute/3e-4 relative
tolerances. Maximum absolute discrepancy is **1.29745e-7**. All selected IDs
match the CPU sort exactly.

Four-round comparisons additionally check current and earlier logits'
gradients through convex updates, unavailable all-negative-infinity rows,
eta-zero proposals, retained proposals and immutable observed/padded rows.
NaN, positive/negative infinity, negative finite values, negative subnormals
and negative NaN probability patterns are rejected. Logits reject NaN and
positive infinity. Existing endpoint poison checks are retained, not weakened.
The original dense/full-support convex-update discrepancy is **0**; the
original soft/dense expectation and selected-weight VJP discrepancies are
3.72529e-9 and1.19209e-7 (embedding VJP0) on each path, within their unchanged
**2e-6** gates.

The existing CPU `reference` passes 6 recurrence,4 cache and6 full-layer
cases with the original2e-5/3e-4 tolerances. Workspace CPU tests run during
implementation passed **37 tests (27 experiment,10 mixers),0 failures**;
the new tests exercise selector identity and scalar/state-gradient oracles.
The first version of the new state-gradient test consumed an intermediate
autodiff tape before a later round; it failed explicitly and was corrected to
compare the complete four-round graphs with one backward each. The first GPU
full-support check exposed Burn's strict frontend argtopk assertion; the
checked backend call described above fixes it. Neither failure was hidden
or admitted by changing numerical tolerances.

## Causal full-step negative control

These paired ordinary forward/backward/checked-AdamW controls completed
before the global build/cache isolation fix and are **provisional control
measurements, not candidate acceptance evidence**. They use the ordinary
dev-profile experiment crate, not the optimized-host selection comparison
below. Soft selection is inactive in every row, so observed differences are
noise, not a software mechanism or an optimization regression.

| Batch | Context | CPU selector step ms | GPU selector step ms | Median paired saving ms | GPU faster /5 |
|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 1457.788 | 1464.098 | -2.729 | 1 |
| 8 | 128 | 333.643 | 341.271 | -7.738 | 1 |
| 1 | 512 | 538.149 | 538.890 | -4.464 | 2 |
| 1 | 128 | 173.755 | 172.429 | 1.257 | 3 |

Raw five paired samples and exact bounded commands:
`artifacts/soft-topk/step-1792866602867-2669256/`.
There are zero soft-selection dispatches in a causal LM step; total model
dispatches were not observed in this control. Isolated selection dispatch
counts are reported separately. Parent integration owns the final combined
model and historical-checkpoint gates, rather than repeating the same
unchanged causal work for every independent inactive component.

## Frozen-source and build isolation

Primary measured source commit:
`97b0ba065359d5463affbdb25f94c840e9cac8d3`
(`wnwytkkxzqvyqxtzopvspptuskwntotz`).
That measured private executable SHA-256 is
`d4974316921102cfc8b24266c9538de1072b28d05360831e0981cf7f855a705c`.
`artifacts/soft-topk/build-manifest.json` records hashes of all changed
source/runner files, Cargo inputs, exact build arguments and private cache
identity. The corrected helper acquired GPU then build locks, cleaned only
the three local packages, rebuilt them from this workspace and copied the
executable. Build queue wait was152s; Cargo build time was **6.79s**, within
the10s incremental budget. Registry dependencies were retained. Existing
ndarray deprecation warnings remain visible.

The isolated `soft-reference` passed from that primary binary/private cache in
**4.653s compute**, with **0.017s queue wait**. It includes the independent
visibility schedule and immutable/unavailable/signed-zero proposal assertions,
not just CPU/GPU agreement through shared bookkeeping code. Its report
retains the same1.29745e-7 maximum scalar-oracle error and unchanged tolerances.

The original dev-profile B8x16,V32k,k4 selection baseline was
1644.768ms CPU /13.572ms native /5.806ms tiled. That provisional result is
retained in `artifacts/soft-topk/soft-benchmark-b8-t16-k4.json`, but it is **not**
the final speedup denominator: extrapolating its unoptimized Rust host sort
to4096 rows would make a complete bounded sample sweep impractical.
The final comparison builds **all three paths in the same binary** with
`--config profile.dev.package.delta-experiment.opt-level=2`, alongside the
repository's already optimized dependencies. No source algorithm,
precision, workload, shader optimization flag, fusion or autotuning selector
changes between arms. This gives the GPU a conservative, optimized CPU-sort
baseline rather than claiming the cost of an unoptimized host comparator
as a GPU-kernel gain. The repository's tracked build profiles remain unchanged.

## Decisive realistic B8 comparison

Private-cache/fixed-helper source97b0ba06, R=4,096 (=8x512), k16,
one warmup/five rotating paired trials. Triples are **CPU / native GPU /
pre-fix padded tiled GPU**, synchronized milliseconds:

| Vocab | Selection | Complete existing feedback round | Selection+expectation+backward |
|---:|---:|---:|---:|
| 257 | 48.654 /3.394 /4.692 | 81.805 /9.473 /10.154 | 98.554 /50.951 /50.626 |
| 32,000 | 9536.528 /134.750 /116.269 | 13872.568 /567.555 /578.462 | 9556.361 /188.209 /171.086 |

Native is **14.34x /70.77x** faster for selection and **8.64x /24.44x**
faster for the complete existing feedback round at vocab257/32k respectively.
Every CPU/native feedback pair improves: byte-vocab paired savings span
70.773–97.742ms (median73.835ms);32k savings span
12880.183–14354.243ms (median13305.013ms). These are well beyond the observed
run-to-run drift, not a launch-count-only claim. The VJP fixture improves
1.93x /50.78x; the small-vocab embedding-gradient work remains substantial.

The padded tiled design's32k isolated selection gain does **not** establish
a complete-feedback gain: tiled minus native feedback savings range
-24.587 to67.582ms, median11.980ms in native's favor. Native has fewer
selection launches (**9 vs16**, CPU2), lower temporary storage and a simpler
execution graph. The padded tiled variant is a design comparison, not the
adopted path.

Raw paired samples/configs/launch histograms and separate lock measurements:
`artifacts/soft-topk/selection-1792868611613-2908803/`.
`decisive-summary.json` contains derived ratios and per-pair ranges.
Compute was242.406s; queue wait was158.864s, outside the compute timeout.
The32k fixture sampled3,690,008,832bytes of reserved pool after all arms and
their feedback/VJP work, not an exact native-only peak. Native first-state
construction was677.488ms at32k and322.538ms at257; these include allocation
and any cache-miss startup, not just a shader compiler timer.

For the **current D16 consumer fixture**, a practical component budget at
B8x512/k16 is≤0.7s warm feedback for32k and≤12ms for257; native's largest
observed samples are610.496ms and10.975ms. This is not a D256 denoiser,
training/convergence or future SBD latency promise. The recurrent causal
model is unchanged, and no model-level speedup is claimed.

## Existing model and checkpoint compatibility

The slice's pre-isolation `kron-model-check` passed all12 reported execution
cases at B8, including CPU logits, every parameter gradient and AdamW
parameters. Maximum logit/gradient/AdamW differences were
9.11206e-6 /2.24635e-6 /1.02613e-5 at unchanged2e-5/3e-4 tolerances.
Report:
`artifacts/soft-topk/kron-model-check-2736688-1792867196699057199/report.json`.
The existing fused Kron and tensor/fused HOLA reference commands also passed;
the known unrelated legacy arbitrary-input GPU tensor-solver defect was not
rerun or hidden.

The historical
`artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` loaded and passed
saved tensor/optimizer validation, scratch AdamW and optimizer continuation.
Its original config fingerprint remains
`c7eca1dace65f65fea8db186a6c0cc5427a320f4bf89f76a4ea0482c86dc42ca`.
The diagnostic's unchanged tolerances are1e-5 absolute/1e-4 relative.
Report:
`artifacts/soft-topk/historical/checkpoint-check.json`.
`--diagnostic-output artifacts/soft-topk/historical` relocates only reports
and scratch files; it does not modify or rehash the original config.

These model/checkpoint runs are conservatively labeled **provisional**
because they preceded the fixed-helper/private-cache acceptance protocol.
They are not substituted for the final isolated soft gate. Per parent
coordination, the shared integration runs combined model/historical gates
once; this inactive slice does not repeat every historical matrix while
the GPU queue is shared.

## Complete requested shape characterization

All24 rows below passed exact CPU-ID and selected-weight/mass preflight.
Triples are **CPU / native GPU / pre-fix padded tiled GPU**, milliseconds.
All use the same optimized-host S0 binary/hash recorded above; k16/B8/4096
has five pairs, all other rows have three. The accepted native implementation
is unchanged by the later allocation-only fix to the unadopted tiled arm.

| Batch | Rows | Vocab | k | Selection ms | Feedback round ms | Selection+expectation+backward ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 257 | 4 | 3.229 /1.470 /1.358 | 4.817 /3.763 /3.458 | 3.351 /2.407 /2.946 |
| 8 | 128 | 257 | 16 | 1.717 /1.581 /1.809 | 4.011 /3.840 /4.402 | 4.385 /4.080 /5.950 |
| 8 | 128 | 257 | 64 | 2.266 /6.660 /8.788 | 4.416 /8.362 /9.879 | 7.463 /12.696 /15.560 |
| 8 | 128 | 32,000 | 4 | 284.575 /12.533 /7.353 | 432.977 /39.832 /29.888 | 285.711 /13.054 /8.447 |
| 8 | 128 | 32,000 | 16 | 279.419 /20.378 /9.302 | 422.343 /38.775 /39.423 | 281.594 /14.396 /12.123 |
| 8 | 128 | 32,000 | 64 | 294.416 /84.009 /39.738 | 425.825 /71.457 /55.763 | 291.056 /70.032 /44.861 |
| 8 | 4096 | 257 | 4 | 48.128 /3.440 /3.611 | 72.423 /8.009 /9.908 | 61.523 /19.948 /20.848 |
| 8 | 4096 | 257 | 16 | 48.654 /3.394 /4.692 | 81.805 /9.473 /10.154 | 98.554 /50.951 /50.626 |
| 8 | 4096 | 257 | 64 | 49.188 /6.722 /16.150 | 76.988 /14.473 /23.013 | 198.574 /150.533 /163.956 |
| 8 | 4096 | 32,000 | 4 | 8845.962 /60.531 /90.132 | 13373.023 /305.102 /332.545 | 8978.509 /77.335 /104.230 |
| 8 | 4096 | 32,000 | 16 | 9536.528 /134.750 /116.269 | 13872.568 /567.555 /578.462 | 9556.361 /188.209 /171.086 |
| 8 | 4096 | 32,000 | 64 | 9000.050 /106.043 /777.422 | 13245.297 /347.623 /1021.558 | 9132.720 /258.934 /921.401 |
| 1 | 16 | 257 | 4 | 0.776 /1.321 /1.322 | 3.050 /3.725 /3.435 | 1.759 /2.206 /2.247 |
| 1 | 16 | 257 | 16 | 1.030 /1.565 /2.007 | 2.890 /4.267 /4.377 | 2.284 /2.767 /3.099 |
| 1 | 16 | 257 | 64 | 0.914 /5.826 /7.014 | 2.920 /7.678 /8.921 | 2.410 /6.629 /9.309 |
| 1 | 16 | 32,000 | 4 | 30.522 /8.107 /4.505 | 47.950 /23.024 /19.448 | 31.890 /8.745 /5.671 |
| 1 | 16 | 32,000 | 16 | 30.604 /9.136 /4.285 | 48.395 /17.981 /12.832 | 33.741 /9.842 /4.396 |
| 1 | 16 | 32,000 | 64 | 32.608 /52.042 /15.311 | 48.632 /53.527 /22.870 | 33.801 /55.303 /17.626 |
| 1 | 512 | 257 | 4 | 7.418 /1.520 /1.535 | 12.332 /4.358 /4.143 | 9.700 /4.232 /5.116 |
| 1 | 512 | 257 | 16 | 6.389 /1.967 /2.208 | 12.034 /5.054 /5.839 | 16.052 /10.104 /12.954 |
| 1 | 512 | 257 | 64 | 8.343 /4.913 /6.004 | 11.560 /7.876 /8.499 | 19.765 /17.367 /26.308 |
| 1 | 512 | 32,000 | 4 | 1120.785 /17.983 /13.757 | 1665.415 /61.525 /54.464 | 1126.847 /15.913 /15.315 |
| 1 | 512 | 32,000 | 16 | 1130.678 /23.385 /15.183 | 1662.133 /52.604 /51.361 | 1145.983 /22.161 /21.724 |
| 1 | 512 | 32,000 | 64 | 1125.322 /65.445 /111.966 | 1663.101 /88.501 /142.534 | 1162.758 /81.567 /130.271 |

Raw characterization:
`artifacts/soft-topk/selection-1792868616041-2909445/`.
Together with the decisive directory above, `evaluation.json` contains all
24 rows, source report paths, exact medians, raw feedback samples and
per-pair CPU/native savings. All measured selection launch counts are
**2 /9 /16**. Maximum sampled suite pool reservation is3,690,034,944bytes.
The11 additional commands all exited0, with411.886s aggregate compute,
1257.299s aggregate queue wait and169.739s maximum single compute hold.
Queue sums are not GPU time or a claimed wall-time total.

## Final allocation-only fix and correctness gate

The experimental tiled arm initially padded even when vocab was already
an exact256-token multiple. Final source
`2c2a7910b9e52c3c2cc65a8bd6fe7e966a0e1652`
adds only the contiguous-key reshape in that case. For4096x32,000 it avoids
a **524,288,000-byte (500MiB)** allocation and full copy. Tail vocabulary257
still pads correctly. The adopted native path, arithmetic, selectors and
historical defaults are unchanged. The original source commit was preserved,
not rewritten.

Final private binary SHA-256:
`3bfe4008bb5dbc319c2766ced4cf7e0211564b3b12825cd2cb9c4705b191682d`.
`artifacts/soft-topk/build-manifest-no-pad.json` records this source/binary
and changed-file hash, with the unchanged hashes linked to the primary
manifest. Corrected-helper build: **6.59s**, queue26s.

The final private-cache `soft-reference` again passes **all24 entries**, at
the unchanged tolerances and1.29745e-7 maximum error, in **4.234s compute /
0.015s queue wait**. Per parent direction, this fix receives correctness
validation only: **no post-fix tiled timing or speedup is claimed**, no extra
B8 timing or historical matrix is rerun, and tiled remains unadopted.

## Decision and limits

**Adopt explicit `soft_selection_backend: "gpu"` for the demonstrated B8
larger-block/large-vocabulary workload.** Keep omission/`cpu` historical;
do not change the causal recommended config. Candidate IDs, weights, mass
and feedback transitions stay device-resident, with scalar rejection checks.
The five-pair B8x512/k16 feedback gains are far beyond noise at both
vocabularies. All requested widths and B1 cases remain fully reported.

This is deliberately **not** a universal faster-on-GPU claim. B1/16-row byte
cases regress; B8/128-row byte/k64 also regresses (8.362ms native versus
4.416ms CPU feedback). B8/128-row byte/k16 is effectively near-tied.
B1/16-row32k/k64 native has a53.527ms median versus48.632ms CPU, so even
large vocab alone does not guarantee a win at tiny row counts. Keep CPU or
make a separately measured explicit choice for those small workloads.
There is no automatic fallback or shape-based semantic substitution.

**Do not adopt `gpu_tiled`.** Some small32k cases have faster isolated or
feedback samples, but the design does not win consistently across the target
regime and is much worse at4096x32k/k64 (1021.558ms pre-fix feedback versus
347.623ms native). Its final avoidable-copy fix is numerically checked but
not performance-adopted. The selector is retained explicitly for honest,
repeatable design comparison, not enabled in any recommended config.

The residual dense convex-update/softmax work and immediate scalar status
readback are real costs, not hidden. No Step7+ component, learned denoiser,
quality result, bf16/CUDA result or causal-LM speedup is included.

## Exact commands and retained inputs

From the softtopk workspace root; all performance fixtures/configs and logs
are under `artifacts/soft-topk/`. For the pre-fix tiled columns use frozen
source97b0ba06; final source2c2a7910 receives the final correctness-only gate.
The runner acquires GPU→BUILD locks, starts the300s timeout after acquiring
them and records wait/compute separately.

```sh
../build.sh softtopk --config profile.dev.package.delta-experiment.opt-level=2
BIN=/tmp/delta-bins/softtopk/delta-experiment

# The final isolated gate's bounded inner command; --locked-worker records
# queue/compute around this command while holding GPU then BUILD locks.
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock \
  timeout --signal=TERM --kill-after=5s 300s "$BIN" \
  soft-reference artifacts/soft-topk/b8-t512.json

# Primary five-pair confirmation, source97b0ba06:
bun scripts/benchmark-soft.ts "$BIN" selection 1 5 \
  artifacts/soft-topk/b8-t512-k16.json

# All other requested shapes, source97b0ba06:
bun scripts/benchmark-soft.ts "$BIN" selection 1 3 \
  artifacts/soft-topk/b8-t16-k4.json \
  artifacts/soft-topk/b8-t16-k16.json \
  artifacts/soft-topk/b8-t16-k64.json \
  artifacts/soft-topk/b8-t512-k4.json \
  artifacts/soft-topk/b8-t512-k64.json \
  artifacts/soft-topk/b1-t16-k4.json \
  artifacts/soft-topk/b1-t16-k16.json \
  artifacts/soft-topk/b1-t16-k64.json \
  artifacts/soft-topk/b1-t512-k4.json \
  artifacts/soft-topk/b1-t512-k16.json \
  artifacts/soft-topk/b1-t512-k64.json

# Executed provisional causal negative controls (ordinary dev build):
bun scripts/benchmark-soft.ts "$BIN" step 1 5 \
  artifacts/soft-topk/b8-t512.json artifacts/soft-topk/b8-t128.json \
  artifacts/soft-topk/b1-t512.json artifacts/soft-topk/b1-t128.json
```

The provisional compatibility runs used these exact binary arguments:
`reference artifacts/soft-topk/cpu.json`,
`kron-model-check artifacts/soft-topk/b8-t512.json`,
`kron-reference artifacts/soft-topk/b8-t512.json --scan-backend fused_gpu`,
`hola-reference artifacts/soft-topk/b8-t512.json`, and
`checkpoint-check artifacts/lm512-vulkan/resolved-config.json --arm
kronlstm-hola --seed 1 --resume
artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13 --diagnostic-output
artifacts/soft-topk/historical`. Their then-current external GPU lock and
300s bounds were applied; current reproductions must use GPU→BUILD order.
`cpu-tests-summary.json` retains the executed CPU test command, pass counts
and its earlier lock-policy phase. Final combined workspace/model/checkpoint
verification is owned by integration, not claimed as re-exercised here.
