# Matrix-multiply efficiency

## Scope and reproducibility

This item covers MLP up/down, merged mixer inputs, dependent mixer output and
vocabulary head, each forward/input-VJP/weight-VJP, at batch1/8/16 and context512.
The model is D256/L4/H4/K32/V64, chunk16, byte vocabulary257. SiLU/residual
fusion, larger K/V, and scaling to larger contexts are separate work.

Production source: `1d038db1e3c38fafb9958d0bcf19df05394ff668`.
Diagnostic geometry correction: `7f8844f70f07205b9c51bb550ab021b0cbeac210`.
The correction does not change model execution: native linear F/dX retain
`[B,T,K] @ [1,K,N]` when T fills row tiles; only dW flattens B*T by default.
Explicit heuristic head/output paths deliberately flatten to 2D. Early 2D
F/dX microbenchmarks are flattened equivalents, **not** the actual B8/B16
native kernel geometry; their rates are provisional and must not be used as
final per-actual-shape results.

All numerical tolerances remain unchanged. Each microbenchmark samples four
entries per result against independent scalar-f64 dots, covering every listed
shape and original/packed layout. The startup guard likewise samples four
entries of each F/dX/dW on deterministic representative tensors. Neither is a
whole-output f64 proof. The configured model gate separately checks all logits,
all parameter gradients, AdamW updates, and the existing full-parameter
CPU directional check, at contexts128/512 with and without active HOLA.

## Device, peak and dispatch

Observed device: Intel Arc Graphics (Meteor Lake), Intel Core Ultra9 185H,
Mesa `26.2.4-arch1.1`, native Vulkan/SPIR-V. The physical Vulkan subgroup range is
8–32. CubeCL reports plane operations/synchronization, f32 vector loads up to4,
and48KiB shared memory; no cooperative-matrix/CMMA capability is exposed on this
backend. This does not assert that every matrix facility is absent in silicon.

A nominal upper bound is approximately4.813TFLOP/s FP32: **[INFERENCE]**
8 Xe cores ×16 vector engines/core ×8 FP32 lanes ×2 FMA operations ×2.35GHz.
The8-core/2.35GHz SKU specification and256-bit vector-engine architecture are
from [Intel's processor specification](https://www.intel.com/content/www/us/en/products/sku/236849/intel-core-ultra-9-processor-185h-24m-cache-up-to-5-10-ghz/specifications.html)
and [Intel's Xe architecture guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html).
This is a boost-clock arithmetic ceiling, not a measured sustainable peak or
hardware utilization counter. The prior whole-model MLP attribution of roughly
26GFLOP/359ms ≈72GFLOP/s is about1.5% of that bound.

A resident, synchronized large-square experiment measured81.3GFLOP/s for the
heuristic2048-square and352.3GFLOP/s for the autotuned2048-square; the best of the
1024/2048 tuned squares was380.2GFLOP/s (1024). These early shared-cache rates are
provisional measurements of this backend, not a claim about physical FP32 peak.

Resolved dependencies are Burn0.22.0, CubeCL front0.11.0 with resolved
runtime/core/WGPU/SPIR-V0.11.3, and cubek-matmul0.3.0. The historical Vulkan build
has **autotune off**. Without CMMA, Burn's default dispatch selects the
non-accelerated `SimpleUnit` family. Enabling `burn/autotune` times alternative
unit/double-buffered/GEMM candidates; it also changes other Burn autotuned ops,
not only matmul. Fusion remains off.

The runtime explicitly installs CubeCL configuration: persistent compilation
and tuning caches are enabled; environment-variable overrides do not silently
replace these application settings. Autotune config/binary mismatches fail
before device creation, preserving the meaning and serialized identity of old
configs. Tune decisions are recorded in `autotune-decisions.jsonl`.

## Why the apparent faster candidate is unsafe

The no-stage `gemm_None` candidate produced large numerical errors, not roundoff.
The recorded dispatch is
`cubek_matmul::multi_level::components::batch::gemm::matmul::base::matmul_entry::MatmulEntry`;
it also launched two `CopyKernelPack` input kernels in the head examples.
Consequently, extra user-side packing does not universally bypass internal
packing or the defect.

Precise provisional repros (f32, seeded Uniform(-0.1,0.1), scalar probes with
`abs<=2e-5 +3e-4*abs(reference)`):

| Contraction | Geometry | Observed scalar vs f64 |
|---|---|---|
| B1 head dW | `[256,512]` transposed-LHS view ×`[512,257]` | `[0,0]`0.00482840836 vs−0.05746580457 |
| B8 head dW | `[256,4096]` transposed-LHS view ×`[4096,257]` | `[0,0]`−0.05468010157 vs−0.24717845728; packing is also wrong |
| T128/B1 mixer-output F/dX | M128,N256,K256 | `gemm_None` fails independent dots; not an N257-only tail issue |
| T128/B16 mixer-output dW | M256,N256,K2048 | `gemm_None` fails independent dots |
| Legacy tensor solver | batched H4,M16,N64,K32 and related VJPs | failures include contiguous/chunk-view layouts and broadcast inverse cases |

The first head dW has LHS strides(1,256) and RHS strides(257,1).
The full reports retain all actual/expected sampled values and full dispatch
identities. These are upstream bug-note data, not a claim to have isolated the
shader's internal root cause. Original common-cache observations precede the
later workspace-cache isolation and are labeled provisional accordingly.

Neither head-only fix was sufficient: a constant-zero257→320 head and a
per-call heuristic head each passed all47 context512 GEMM cases, but both
failed the configured fused model at T128. For both, fused scan without HOLA
returned first logit0.2583878 vs CPU0.06443295, and fused scan plus fused HOLA
returned0.2474109 vs0.055922177. Both passed T512 with max logit error about5.16e-6.
The T128 ordinary mixer-output GEMM probe located the additional faulty family;
the failure is **not** confined to legacy TensorSolver/Tensor-HOLA execution.

Global candidate exclusion is not publicly exposed: Burn owns a private static
`LocalTuner`/`TunableSet`; CubeCL0.11.3 `AutotuneConfig` has no candidate filter.
No registry edit, dependency fork, vendoring patch or tolerance relaxation is used.

## Explicit guarded candidate

`configs/lm512-gemm-guarded.json` selects:

- `autotune: true` with `--features vulkan,autotune`, fusion off;
- fused scan and fused HOLA; tensor normalization/gating;
- `projection_backend: merged_heuristic_output`: inputs stay one merged GEMM,
  while dependent output F/dX/dW use the public non-autotuned Cube heuristic;
- `head_backend: heuristic`: unchanged257-column parameter, same heuristic for
  all three head contractions;
- a fresh relative cache `.cache/cubecl/gemm-guarded-private`.

MLP and merged-input GEMMs remain autotuned. Per-family pinning is a last resort,
not a whitelist of batch/context values: before training, the mandatory guard
probes every active configured geometry and refuses to start on any mismatch.
Unsupported tensor-solver/Tensor-HOLA training combinations fail fast. Native
historical head/projection selectors and all parameter names/shapes/IDs remain
unchanged by default. The alternative padded head appends constant zeros,
excludes padded logits before the loss, and creates no trained padding columns.

Guard certificates are bound to shapes/execution, actual adapter/driver,
source/dependency hashes, and the sorted persistent matmul-tune namespace digest.
Selections are warmed before reading that digest, resolving current candidate
checksums. Matching cached certificates still recheck fresh GPU values; only the
deterministic f64 expectations are reused. Startup/probe cost is outside warm
step clocks and recorded separately. The guard does not certify unknown model
families or prove every output entry correct.

## Provisional full-step experiment: reject bare autotune

The original native-head, unguarded autotune experiment used11 alternating AB/BA
pairs, seed1, context512, fresh identical trajectories. Each process reports its
first step separately, excludes two further warmups, and measures the fourth
complete guarded-training-path step. The candidate is numerically invalid;
these times are **not an accepted speedup**.

| B | Heuristic ms | Bare autotune ms | Paired saved median ms |95% paired-bootstrap interval ms |
|---:|---:|---:|---:|---:|
|1|559.97|515.39|48.93|41.98..93.36|
|8|1659.88|1177.19|412.14|193.63..519.65|

Autotune was faster in11/11 pairs at both batches, but cannot be adopted on
these results. The initial partial micro tuning command took32.5s; a subsequent
first full B8 step took12619ms versus1085ms warm. Those are partially warmed
cache observations, not a clean whole-program cold-start claim. Cargo's initial
autotune feature build took33.07s; ordinary source rebuilds were roughly2–5s.

`cargo test --workspace` on the first frozen production source passed37 tests:
27 experiment +10 mixers, no failures, plus empty kernel/doc-test suites.
This includes unchanged default optimizer/checkpoint-resume equivalence and the
new padded-head logits/VJP/parameter-shape checks. Shared integrated GPU and
historical checkpoint gates are owned by the parent integration workflow.

## Locking and artifacts

All final build/GPU transactions acquire
`/tmp/delta-block-lm-gpu.lock` **then** `/tmp/delta-block-lm-build.lock`, with
bounded compute inside the locks. Queue time is distinct from execution.
Early final-build attempts expired only in the80s queue (status75), not GPU
execution; an earlier prepare hit an outer300s queue deadline without creating
its configurations. The first successful matched build recorded190.696s queue
and11.099s execution. Registry dependencies were retained while only the three
local packages were cleaned/rebuilt.

Primary evidence roots are `artifacts/gemm-efficiency/` (provisional experiments)
and `artifacts/gemm-efficiency/final-guarded/` (isolated confirmation). Raw
stdout/stderr/process outcomes, per-shape JSON, tune decisions and preflight
certificates are retained. No artifact/data/cache directories were deleted.

## Final isolated numerical confirmation and actual-shape rates

Both feature binaries were rebuilt from the frozen source plus diagnostic fix,
under GPU→BUILD transaction locks. Final build/test transaction:247.547s queue,
6.691s execution; `cargo test --workspace` passed38 tests (28 experiment+10
mixers), including the new batch-geometry regression. Binary SHA256:

- control: `065112c4551d66344e0156e82493c23f2cf460c1c514a3ed09cc85b94bd1785b`
- candidate: `e442f4e30305023f8c01bdd5c7ecded1831c8c65b126e0f893ad501cc657c507`

`final-guarded/source-proof.json` also records Cargo/dependency/source hashes.
The isolated numerical transaction queued138.733s and executed105.968s:

- Configured full model: **4/4 cases pass**, T128/T512 × HOLAoff/on; max logits
  error5.156e-6, max all-parameter gradient error3.577e-6, max AdamW parameter
  error1.495e-5. Existing relative/absolute and directional tolerances unchanged.
- Candidate actual-rank large-shape sweep: **47/47 pass** for both original and
  packed operands; largest scalar absolute error7.908e-7.
- Control: **47/47 large +126/126 solver cases pass**.
- Candidate legacy solver:36/126 original-layout cases fail; aggregate micro
  exits1 and `passed:false`, while `recommended_path_passed:true`. This is an
  explicit diagnostic scope split, not suppression of failing tests.

The fresh private-cache solver repro confirms the no-stage problem independently
of earlier shared-cache observations: B1/H4,M16,N64,K32, contiguous forward,
`solver_k_state`, head0,row0,column0 gives0.00116826652 instead of scalar-f64
0.01825463073; packing is also wrong. Dispatch is the same GEMM `MatmulEntry`
above (plus one `CopyKernel`). Tensor-solver/Tensor-HOLA remain unsupported for
autotuned training.

GFLOP/s below are synchronized resident actual GEMMs, median of7 trials after
2 warmups; packing comparison is alternated inside each case. F/dX for native
MLP and merged inputs use B batches of M512; dW is one2D matrix with K=B*512.
The protected candidate output/head intentionally use one2D M=B*512 matrix.
All FLOP counts include B and use2*M*N*K; biases, activations, concatenation and
autodiff bookkeeping are outside these micro clocks.

| B | Linear | F heuristic / guarded | dX heuristic / guarded | dW heuristic / guarded |
|---:|---|---:|---:|---:|
|1|MLP up|43.7 /80.5|38.7 /88.8|163.4 /118.1|
|1|MLP down|38.9 /95.0|94.9 /119.4|26.8 /108.3|
|1|Merged input|86.1 /90.5|55.8 /175.0|251.8 /239.8|
|1|Mixer output|42.2 /45.5|39.2 /44.2|74.2 /112.5|
|1|Vocabulary head|40.0 /46.2|33.9 /33.1|80.1 /73.0|
|8|MLP up|69.3 /123.6|98.9 /190.0|363.1 /365.8|
|8|MLP down|66.9 /214.9|128.5 /276.4|34.1 /329.6|
|8|Merged input|98.5 /247.0|79.9 /232.9|482.4 /490.9|
|8|Mixer output|63.8 /101.2|149.4 /133.0|202.3 /191.2|
|8|Vocabulary head|98.8 /98.8|94.4 /93.2|145.4 /143.9|
|16|MLP up|71.6 /183.7|95.6 /224.1|356.6 /388.5|
|16|MLP down|66.3 /248.5|137.7 /335.6|35.2 /326.8|
|16|Merged input|113.6 /248.5|78.0 /235.7|507.1 /556.1|
|16|Mixer output|67.7 /102.7|131.8 /112.3|202.4 /210.3|
|16|Vocabulary head|105.3 /99.6|94.1 /84.4|160.6 /163.1|

Four-layer MLP GEMM-only work at B8 is25.77GFLOP:340.90→117.65ms,
75.59→219.04GFLOP/s. This does not include SiLU/residual or claim that the full
MLP component has the same latency. Merged projections benefit substantially;
head performance is intentionally conservative rather than using the faulty
fast candidate. At B1, MLP up dW itself regresses, illustrating why end-to-end
paired acceptance is required.

Packing all six MLP contractions inside their clocks is not selected:

| B | Control original / packed ms,4 layers | Guarded original / packed ms |
|---:|---:|---:|
|1|68.96 /64.63|32.36 /36.87|
|8|340.90 /326.04|117.65 /128.65|
|16|669.90 /641.55|193.65 /225.38|

The original views are better once the tuned kernels run. All individual
per-shape packing latencies/rates are retained in the JSON reports.

The final1024/2048-square control rates are81.26/83.92GFLOP/s; tuned rates are
409.94/345.87GFLOP/s. The best tested model rectangle reaches556.1GFLOP/s.
Neither is an absolute hardware ceiling. First-use latencies are separate:
1024 square531.5ms,2048 square4851.0ms, B8 merged dW1756.2ms, B16 merged
dW1561.8ms and B8 MLP-up dW1480.5ms. Across the47 cases, original first-use
latencies sum12.109s and first-packed-use latencies8.760s. These include
selection/compilation plus one execution and have a mixed warm-cache history
after the configured model gate; they are not a fabricated isolated compiler
time or pristine whole-program cold start.

## Selection and final paired confirmation

**Adopt the explicit guarded option above; reject bare autotune and global
packing. Historical defaults do not change.** Both acceptance batches improve
beyond paired noise on the private-cache, matched-source binaries:

| B/context | Control ms | Guarded ms | Speedup | Saved-time median ms |95% paired bootstrap ms | Faster pairs |
|---|---:|---:|---:|---:|---:|---:|
|1/512|560.207|510.467|1.09744×|49.011|32.903..80.052|11/11|
|8/512|1619.764|1188.295|1.36310×|430.936|390.169..442.496|11/11|

Median wall-time reductions are8.88%/26.64%; tokens/s914→1003 and2529→3447
for B1/B8 respectively. These are ordinary full guarded-training-path steps,
not sums of isolated kernels. Eleven fresh seed1 trajectories per arm are
interleaved AB/BA; first use plus two warmups are excluded, fourth step timed.
The interval is10,000 deterministic bootstrap resamples of paired saved-time
medians. No tolerance, optimizer work, historical configuration or quality
criterion was loosened.

Every candidate process passed preflight. Certificates hit in10/11 runs at
each batch, but fresh GPU values were always checked. Startup preflight median/
maximum was100.26/331.31ms at B1 and314.00/2081.48ms at B8, **outside** step
clocks. First-step medians, also separate, were control/guarded647.50/552.44ms
at B1 and1669.81/1248.38ms at B8; these are warm-persistent-cache process starts,
not claims of universal cold-start latency.

Maximum sampled allocator-reserved bytes across paired runs:

| B | Control | Guarded |
|---:|---:|---:|
|1|134,888,768|134,890,304|
|8|690,634,240|690,635,456|

These are sampled pool reservations, not physical GPU-memory measurements or
an instrumented peak. B8 transaction queue/execution26.819/137.766s; B1
131.598/55.730s. Both held GPU→BUILD locks throughout, so no sibling GPU or
build process overlapped the final timings.

### Exact bounded reproduction

Run from the repository with a private relative cache. The following payloads
match the executed commands; transaction `.process.json` files retain the
original argv and separate queue/execution times. The successful local-package
clean/rebuild and final diagnostic rebuild are likewise recorded there.
Use the shared build helper when sharing a target directory; the main workspace
used its own target. Never acquire BUILD before GPU or nest a locking helper
inside an already held transaction.

```sh
GPU=/tmp/delta-block-lm-gpu.lock
BUILD=/tmp/delta-block-lm-build.lock
ROOT=artifacts/gemm-efficiency/final-guarded
OFF=target/debug/delta-experiment-gemm-off
ON=target/debug/delta-experiment-gemm-on

flock "$GPU" flock "$BUILD" timeout 180s cargo test --workspace
flock "$GPU" flock "$BUILD" timeout 180s cargo build -p delta-experiment --features vulkan
cp target/debug/delta-experiment "$OFF"
flock "$GPU" flock "$BUILD" timeout 180s cargo build -p delta-experiment --features vulkan,autotune
cp target/debug/delta-experiment "$ON"

# Recreate the retained configs; only candidate head/projection/autotune differ.
flock "$GPU" flock "$BUILD" timeout 180s bun scripts/benchmark-gemm.ts prepare "$OFF" "$ON" configs/lm512-gemm-guarded.json "$ROOT" 8 11 heuristic merged_heuristic_output

flock "$GPU" flock "$BUILD" timeout 180s "$ON" kron-model-check "$ROOT/on-b1/config.json" --model-scope configured
flock "$GPU" flock "$BUILD" timeout 180s "$ON" gemm-micro "$ROOT/on-b8/config.json" --warmups 2 --trials 7
# Previous command intentionally exits1 for explicitly reported legacy failures;
# require recommended_path_passed=true, never pretend aggregate passed=true.
flock "$GPU" flock "$BUILD" timeout 180s "$OFF" gemm-micro "$ROOT/off-b8/config.json" --warmups 2 --trials 7
flock "$GPU" flock "$BUILD" timeout 180s bun scripts/benchmark-gemm.ts pair "$OFF" "$ON" configs/lm512-gemm-guarded.json "$ROOT" 8 11 heuristic merged_heuristic_output
flock "$GPU" flock "$BUILD" timeout 180s bun scripts/benchmark-gemm.ts pair "$OFF" "$ON" configs/lm512-gemm-guarded.json "$ROOT" 1 11 heuristic merged_heuristic_output
```

The helper `scripts/locked-gpu.ts` records queue/execution separately and writes
unique attempt artifacts. Final longer-queue transactions used explicit
1500s GPU/30s BUILD acquisition bounds and180s compute bounds, with1800s outer
orchestration allowance. The retained final reports are:

- `final-guarded/isolated-gate-summary.json` and `source-proof.json`;
- `final-guarded/paired-b1.json`, `paired-b8.json`, `final-summary.json`;
- `on-b1/kron-model-check-3088065-1792869797006935589/report.json`;
- `on-b8/gemm-micro-3093100-1792869832954886974/report.json`;
- `off-b8/gemm-micro-3098804-1792869874190115485/report.json`;
- per-pair raw stdout/stderr, including complete preflight certificates.

Paths above are relative to `artifacts/gemm-efficiency/`. Shared integrated
selector combinations and historical GPU checkpoint certification are checked
once by the parent integration workflow, not silently claimed by this isolated
matrix-efficiency result.

