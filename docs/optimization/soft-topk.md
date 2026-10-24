# GPU soft top-k candidate selection

## Scope and execution identity

This changes the existing standalone soft-feedback episode, not the causal LM
or an implementation of block diffusion/SBD. `soft_selection_backend` is an
explicit serialized selector: omitted/`cpu` preserves historical selection;
`gpu` uses the native GPU reduction; `gpu_tiled` is the alternative two-stage
GPU design. Historical configurations, parameters, checkpoints and omitted
selector fingerprints are unchanged. Fusion and matmul autotuning stay off.

The CPU baseline remains the original full probability download, per-row
`total_cmp`/ascending-ID sort, ID upload, differentiable gather and sum. Both
GPU designs replace only discrete selection and state bookkeeping. Selected
weights and retained mass still use the original gather/sum autodiff paths.

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

Candidate IDs, selected weights, retained mass, visibility, provisional IDs,
retained IDs, mutable-row masks and active-row tensors remain on the device.
Immutable observed/active host metadata is retained for episode reporting and
the explicit CPU oracle. Initial probability tables are generated on device,
not uploaded as a rows-by-vocabulary host vector. Retain/remask upload only the
requested position IDs and update device tensors; they do not download state.

A GPU selection reads back **one boolean validity result**, not the
probability tensor. A feedback update reads one logits validity boolean and
one candidate validity boolean. Immediate `Result` rejection is preserved;
there is no silent fallback, suppressed invalid input or unchecked delayed
failure. Final report/debug snapshots intentionally read selected tensors.

The convex update, temperature, visibility schedule, unavailable-row
sanitization, eta-zero proposal behavior, immutable observations,
padding, zero-mass mask expectation and visibility-only remasking retain
historical semantics. Discrete IDs have no VJP; gather/softmax/expectation
retain gradients to probabilities/logits, embeddings, mask and visibility.
No recurrent state or document reset exists in this component.

## Measurement protocol and interpretation

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

For L=ceil(V/256), the tiled alternative additionally allocates a padded
`4*R*L*256`-byte key tensor and tile-local IDs/keys/global-ID intermediates of
order `R*L*min(k,256)`. It performs two top-k reductions and additional
packing/gathers, so launch count alone is not a speedup. At R=4,096,V=32,000,
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
independent scalar-f64 fixtures per path plus the original soft episode and
four-round CPU/GPU transition/VJP comparison per path (24 report entries).
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
cases with the original2e-5/3e-4 tolerances. Final workspace CPU tests pass
**37 tests (27 experiment,10 mixers),0 failures**; the new tests exercise
selector omission/explicit identity and the f64/state-gradient oracle.
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

Final source commit:
`97b0ba065359d5463affbdb25f94c840e9cac8d3`
(`wnwytkkxzqvyqxtzopvspptuskwntotz`).
The final private executable SHA-256 is
`d4974316921102cfc8b24266c9538de1072b28d05360831e0981cf7f855a705c`.
`artifacts/soft-topk/build-manifest.json` records hashes of all changed
source/runner files, Cargo inputs, exact build arguments and private cache
identity. The corrected helper acquired GPU then build locks, cleaned only
the three local packages, rebuilt them from this workspace and copied the
executable. Build queue wait was152s; Cargo build time was **6.79s**, within
the10s incremental budget. Registry dependencies were retained. Existing
ndarray deprecation warnings remain visible.

The final isolated `soft-reference` passed from that binary/private cache in
**4.653s compute**, with **0.017s queue wait**. It includes the independent
visibility schedule and immutable/unavailable/signed-zero proposal assertions,
not just CPU/GPU agreement through shared bookkeeping code. Its report
retains the same1.29745e-7 maximum scalar-f64 error and unchanged tolerances.

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
