# AdamW/checks/clipping and cross-entropy evaluation

## Decision

**Adopt `update_backend: "fused_gpu"` explicitly for the realistic B8/context512 pilot; keep `loss_backend: "tensor"` (or omit it).** The final isolated B8/context512 session saves31.005ms per paired complete step (1.739%;95% paired-median bootstrap interval22.088–49.058ms), while complete resident checks+clipping+AdamW falls35.122→13.685ms. Both fused loss designs pass correctness but fail the beyond-noise integrated-speed gate; the combined update+loss gain does not establish an independent loss benefit.

The measurements below are historical evidence for the former optimizer recipe, not acceptance evidence for the paper corrections described next. Tensor/moment parameter layout remains native, but optimizer recipe and global schedule-counter metadata are now required; regenerate budgets and start a fresh run. The historical component budget was **15ms for complete checks/clipping/update at B8/context512 and 2,379,568 parameters**. Its short-context anomaly does not establish a stable speedup.

## Kron paper correction (2026-10-28)

[KronLSTM Appendix A7](https://arxiv.org/html/2610.26834#A7) specifies AdamW
betas `(0.9, 0.95)`, epsilon `1e-10`, decay `0.1`, global gradient-norm clipping
at `1.0`, and a 10%-warmup cosine schedule ending at 0.1 times peak.
The selected from-scratch neural model now uses those defaults in both native
and fused updates. Existing explicit JSON learning rates and decay values remain
overrides; `learning_rate` is the peak, not a constant step rate. The configurable
fields are `optimizer_beta1`, `optimizer_beta2`, `optimizer_epsilon`,
`warmup_fraction`, and `min_learning_rate_ratio`. Setting warmup to zero and the
minimum ratio to one explicitly selects a constant schedule.

Three implementations were considered: clipping each tensor (wrong direction),
packing the whole model into a new multi-tensor optimizer (unnecessary storage
and state-layout changes), and a single module-level clipping/schedule boundary
around the existing per-tensor AdamW operations. The third keeps the existing
moment formulas and grouping while fixing every production and reference caller.
It sums squared norms over present, trainable neural gradients, derives one
coefficient, and scales every such gradient by that coefficient exactly once.
Frozen circuit probabilities do not participate. Explicit draw-specific missing
gradient allowances and per-parameter finite/nonzero/zero reporting remain in
the backward guards; NaN, infinity and squared-norm overflow still reject before
parameter writes. There is no second tensor-local clip in either update kernel.

Norm parameters, affine biases, learned Kron gate biases, `A_log`/`decay_scale`,
and `b_delta`/`decay_bias` have zero decay. The additive HOLA gate bias is also
excluded. Lngram tables retain their distinct five-times-scheduled-base-rate,
zero-decay group. The learned `soft_confidence` scalars have a separate absolute
rate: [SM Appendix B.1](https://arxiv.org/html/2510.17206v2#A2.SS1) specifies
`eta_sm=0.01`, linearly warmed over 2,500 updates and then constant;
the [official trainer](https://github.com/IBM/soft-masked-diffusion-language-models/blob/8c3ec65872f6943ad366ffac877a0d6d8cffe29e/coding/train/components/trainer.py)
puts the entire head in a zero-decay group. This direct head recipe is selected
without replacing Kron's backbone rate/schedule or importing the standalone
pretrained-backbone recipe. Overrides are `soft_confidence_learning_rate` and
`soft_confidence_warmup_steps`. The head uses the saved **global** update clock
for its schedule, including binary/no-active draws, but its Adam moments/counter
advance only when its gradient exists. Missing gradients stay absent; there is
no stale-moment aging or synthesized zero gradient. The 2,500-update warmup is
not shortened to fit a tiny smoke; a smoke does not demonstrate learned head
usefulness. Codpula's circuit-flow optimizer and its law are unchanged; the
from-scratch neural backbone is not frozen. The auxiliary mask-energy fitter
remains native and constant-rate.

Schedule updates are one-based. The warmup length is the ceiling of 10% of the
resolved update budget, capped below the whole budget so a final cosine endpoint
exists; its last warmup update reaches peak. The final budgeted update reaches
the minimum rate, and subsequent diagnostic updates stay at that floor. A
one-update budget has no warmup and uses the final floor. No token or model budget
is increased. The optimizer stores a single completed-update counter independently
of per-parameter Adam counters (which can lag for legitimately absent gradients).
It also saves the exact recipe and schedule length in `optimizer.burnpack`.
Ordinary resume rejects missing/changed recipes and schedule lengths and checks
the optimizer counter against the training step. Explicit TCT warm-start imports
the saved source horizon, recipe and global clock together with the moments;
the target budget controls only additional chain updates. It does not restart
or stretch the source cosine, which stays at its floor past the source horizon.
TCT persists its source-step offset and validates the resulting global clock
on subsequent resume. Other optimizer hyperparameter changes remain errors.
Old optimizer records cannot silently resume changed clipping/schedule semantics.

New, **not executed by the implementation worker**, acceptance cases cover the
`.8 + .8` global-clipping counterexample, unequal tensor directions, finite guards,
numeric decay exclusions (including `wavelet_norm.gamma`/`beta`, nonzero Kron
gate-bias vectors and the separate soft-confidence group), the Lngram rate, moment-level clipping checks,
native/fused updates against scalar float64, exact checkpoint continuation across
the schedule with an intermittently absent gradient, changed-target TCT horizon
import, and skip-draw→soft-draw head restart using global LR/local Adam counters.
Run the integrated
CPU test suite and `update-loss-reference` without an obsolete historical
`--resume` checkpoint. No tolerance was loosened. Historical timings and passes
below do not certify this changed recipe.


## Scope and baseline

Intel Arc MTL integrated GPU (`spirv_32902_32085`), Vulkan, f32, Burn 0.22 / CubeCL 0.11; no Burn fusion or matmul autotune. The controlled model is `configs/lm512-projections.json`: four layers, width 256, H4/K32/V64, chunk/cache 16, fused KronLSTM/HOLA, merged projections, tensor normalization, native 257-column vocabulary head, seed 1 and pinned byte-LM corpus. Batch 8 is the adoption workload; batch 1 and contexts 128/512 are reported separately. Recommended and historical config files are unchanged.

The existing batch-8/context-512 attribution is 37.967 ms for all non-autodiff checks/clipping/AdamW (2.35% of the 1,613.667 ms complete step), and 19.492 ms for final normalization/head/loss (1.21%). These historical numbers are prioritization evidence, not baselines to subtract from a different measurement session. Every new throughput session includes its own unchanged tensor-path control in the same executable.

## Compared designs and exact semantics

### Update

The historical evaluation compared ordinary Burn primitives against an explicit
per-parameter fused implementation with bounded square-sum reductions and buffer
reuse. Both originally applied tensor-local clipping. That historical clipping
rule and constants `(0.9, 0.999), epsilon=1e-5` are superseded by the correction
above; the retained numerical/timing reports below describe their original recipe.

The current fused element update receives already globally clipped gradients and
explicit beta/epsilon constants; it no longer launches its own norm reduction.
It retains
`parameter*(1-lr*weight_decay) - lr*((m/correction1)/(sqrt(v/correction2)+epsilon))`,
host f32 `1-beta.powi(time)` corrections, and optional AMSGrad maximum state
without using that maximum in ordinary AdamW's denominator. Unique contiguous
buffers can still be reused; shared/strided buffers receive independent outputs.
Clipping's active coefficient is applied once before per-parameter updates, not
folded into the learning rate or moment bias corrections.

### Loss

1. **Existing primitives:** vocabulary projection, log-softmax, target gather, weighted sum divided by weight sum.
2. **`loss_backend: "fused64"`:** 64-lane row-local stable max/sum-exp/target selection, followed by the unchanged tensor weighted mean; analytic logits VJP recomputes probabilities from saved per-row maximum/log-sum.
3. **`loss_backend: "fused256"`:** the same arithmetic with 256 lanes, trading fewer per-lane vocabulary loads for a deeper shared-memory reduction.

Both candidates eliminate the vocabulary-sized log-softmax and its generic backward graph. They retain the real logits/vocabulary matmul and weighted-mean primitives: this is not an assertion that a custom GEMM, fused final RMSNorm, or custom weight-gradient reduction was implemented. Keeping the mean in ordinary differentiable primitives preserves gradients to floating loss weights as well as logits, including fractional and zero weights. There is no padding special case or all-zero-weight fallback. Reset/validity handling in the model is untouched. Labels are exact f32 casts of the existing integer targets; supported vocabulary indices fit the exact integer range. Kernels validate f32/device/32-bit spans and support strided inputs. Omitted `loss_backend` preserves the historical implementation/fingerprint.

All production, evaluation, diagnostics, profiling and model-check callsites select loss through the model. The CPU oracle keeps tensor execution rather than certifying a GPU path against itself. The current optimizer keeps native parameter/moment tensors but requires the new recipe and global counter metadata described above; historical optimizer checkpoints must not be resumed.

## Provisional numerical and checkpoint evidence

Unchanged tolerance throughout: `2e-5 + 3e-4*abs(reference)`, with the original independent CPU directional gate's `3e-3 + .03*abs(reference)`.

- Independent scalar-f64 cross-entropy: **32 cases**, widths 1/7/35/257, short rows/tails, zero/tied and nonzero logits, fractional/zero padding weights, contiguous and genuinely transposed inputs, both lane designs. Outputs, every logits gradient and every loss-weight gradient pass. Maximum scalar-loss error 4.035e-7 and logits-gradient error 2.375e-8.
- Independent scalar-f64 AdamW plus the original Burn optimizer: **16 cases × five updates**, zero/small/clipped/large gradients, zero/nonzero decay and contiguous/transposed inputs. Every parameter and first/second moment is checked on every step. Maximum parameter error 1.306e-7. NaN, positive/negative infinity and squared-sum overflow are rejected; missing gradients are rejected.
- Config tests cover omitted historical serialization, fingerprint differences for explicit selectors, GPU-only selection and rejection of automatic-fusion combinations. Workspace CPU tests passed: **26 experiment + 10 mixer**, including original optimizer resume and partial-state rejection.
- `kron-model-check` with both candidates enabled passes all four configured cases: contexts128/512, HOLA off/on, CPU logits, every parameter gradient, AdamW update and the original full-parameter directional finite-difference gate. Largest logits/gradient/updated-parameter errors: **5.126e-6 / 2.295e-6 / 9.191e-6**. This configured-scope gate does not certify the known failing arbitrary-input legacy GPU tensor solver.
- Exact historical checkpoint `artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` loads through both optimizer implementations using the unchanged tensor/scalar/config-state validator. Three real continuation steps compare every raw gradient and updated parameter: maxima **4.285e-8 / 9.537e-7**. This is a validation/continuation check, not permission to resume a changed execution selector under an old run fingerprint.
- Benchmark state roundtrips save native-layout fused optimizer moments and load/validate them through both implementations.

Evidence root: `artifacts/update-and-loss/`. Scalar/reference and continuation report: `baseline-b1/update-loss-reference.json`. Configured model gate: `both/kron-model-check-2657068-1792866488775894497/report.json`. CPU test output is the recorded `cargo test --workspace` run; no tolerances or old failures were changed.

Review identified a valid native-checkpoint consumer case omitted by the pilot checkpoint: `max_moment_2: Some(...)` can remain in an ordinary AdamW state after disabling AMSGrad. The initial fused wrapper incorrectly asserted that it was absent, although the native loader accepts it and native ordinary AdamW preserves it. Frozen source commit `b379379d1180d9f1bc8bc038034668ad3b0a4983` removes that assertion and retains the optional tensor unchanged; the ordinary update still uses current second moment, not the stored maximum. The regression creates an AMSGrad producer checkpoint, loads it through ordinary native and fused consumers, compares three continuation updates, checks exact maximum-tensor signatures after every step, and validates another save/load roundtrip.

## Initial batch-8/context-512 experiment

This allocating-prototype session preceded the shared build-helper fix that cleans workspace-local packages before copying private binaries. Potential stale sibling rlib reuse was subsequently identified by the coordinator. Retain this session as exploratory evidence only, not final adoption evidence.

This first session used the allocating update prototype, two warmups and nine rotating/reversed paired trials of five identical-start scratch trajectories. Complete steps include data preparation/upload, actual model/loss/backward, scalar loss readback, every gradient check, per-parameter clipping, AdamW and final synchronization. Warm samples exclude initialization, compilation, checkpoint/report I/O and launch observation. The same trial index is paired across modes; trajectories have identical parameter starts and data-window sequences.

| Update / loss | Median complete step ms | Median paired saved ms | 95% paired median bootstrap interval, ms | Kernels/step |
|---|---:|---:|---:|---:|
| tensor / tensor | 1659.126 | 0 | — | 5833 |
| fused / tensor | 1729.669 | 24.637 | [-210.965,100.263] | 3922 |
| tensor / fused64 | 1868.171 | -22.793 | [-270.522,134.260] | 5815 |
| tensor / fused256 | 1833.316 | -16.685 | [-288.048,136.533] | 5815 |
| fused / fused256 | 1796.187 | -2.690 | [-253.810,130.049] | 3904 |

Positive paired differences mean faster. Differences of unpaired medians are deliberately not substituted for medians of paired differences. Baseline samples range1442.981–2306.167ms, so this session does **not** support adoption. Concurrent CPU/shared-memory activity is a possible explanation for drift, not a measured attribution. GPU workloads were serialized by the shared lock.

Resident all-parameter checks+clipping+AdamW improve **35.752→14.765ms** (medians) in this session, but the end-to-end intervals include zero. There are91 trainable tensors with HOLA: the fused path issues91 update kernels and182 square-partial kernels (one check and one clipping reduction per tensor), with the final small reductions included in the measured total. The two loss designs each replace18 total issued kernels, not the vocabulary matmul. A launch reduction or an isolated update gain is not a measured full-model speedup.

Raw report: `baseline-b8/update-loss-2600366-1792865987304971324/report.json`; reproducible initial paired summary: `initial-summary.json`.

One preliminary B1 attempt exceeded its300s **outer** deadline without creating a benchmark directory/report. Waiting for the shared GPU lock is the inferred cause; it is not a measured300s GPU step or a numerical failure. Subsequent commands put `timeout 290s` **inside** `flock`, allowing longer lock waits without unbounded GPU holds.

## Memory and work

Unchanged parameter payload **9,518,272bytes**, AdamW moments **19,036,544bytes**. The fused element update performs linear work, writes one parameter and two moment results, and does not materialize corrected moments, clipped gradients or delta arrays. The final implementation aliases unique contiguous buffers rather than necessarily allocating those three results. A square reduction uses at most64 f32 partials (256bytes) per parameter; the91-tensor check has at most23,296bytes of partials plus small scalar/cat reductions, and does not allocate a model-sized square buffer. This is a logical bound, not an exact allocator-traffic measurement.

| Batch / context | Logits and each avoided log-softmax buffer, bytes | Saved max/log-sum bytes | Row-loss bytes | Label-cast bytes |
|---|---:|---:|---:|---:|
| 1 / 128 | 131584 | 1024 | 512 | 512 |
| 1 / 512 | 526336 | 4096 | 2048 | 2048 |
| 8 / 128 | 1052672 | 8192 | 4096 | 4096 |
| 8 / 512 | 4210688 | 32768 | 16384 | 16384 |

The unavoidable head logits and logits VJP remain full vocabulary-sized tensors. Loss work is O(batch*context*vocabulary), not quadratic in context; shared reductions use256bytes or1024bytes per workgroup. The first five-trajectory session sampled825,641,280bytes of shared pool reservation, including all trajectories/retained optimizer states and allocator reuse. That is neither a candidate-specific live-byte count nor a physical GPU peak. Logical sizes are retained in `logical-memory.json`.

Those logical tensor sizes are **not net retained-memory savings**. The custom VJP retains logits where the native log-softmax VJP can retain log-probabilities instead; avoiding a4,210,688byte B8/C512 log-softmax result does not prove that much lower live or peak memory. The final reported pool reservation likewise includes all five scratch trajectories and both isolated-update states.

## Reproduction

From the `optim` workspace, build only with `../build.sh optim`. Binaries are `/tmp/delta-bins/optim/delta-experiment`, never `target/debug`. The saved `artifacts/update-and-loss/*.json` inputs copy the recommended config, disable budget output, choose a fresh artifact directory, and vary only batch/context or the explicit component selectors. All commands executing GPU work are wrapped in `flock /tmp/delta-block-lm-gpu.lock`; each actual GPU run has a deadline of at most300s.

The final build helper acquires **GPU→BUILD** locks and cleans only the three local packages before rebuilding/copying, preserving registry dependency artifacts. Manual Cargo/test commands use the same lock order, with their timeout inside both locks. This prevents stale sibling artifacts and build/measurement memory contention; never hold BUILD while waiting for GPU.

Final acceptance uses the workspace-private relative `.cache/cubecl` directory installed by the coordinator, not the former symlink into the main repository's shader cache. Earlier sessions are provisional because cross-workspace shader-cache key collisions were identified as a risk [INFERENCE], not demonstrated numerical corruption. Cold private-cache compilation is outside warm samples and remains bounded by each run deadline.

```sh
../build.sh optim
flock /tmp/delta-block-lm-gpu.lock timeout 290s /tmp/delta-bins/optim/delta-experiment update-loss-reference artifacts/update-and-loss/baseline-b1.json --resume artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13
flock /tmp/delta-block-lm-gpu.lock timeout 290s /tmp/delta-bins/optim/delta-experiment kron-model-check artifacts/update-and-loss/both.json --model-scope configured
flock /tmp/delta-block-lm-gpu.lock timeout 290s /tmp/delta-bins/optim/delta-experiment update-loss-benchmark artifacts/update-and-loss/baseline-b8.json --warmups 2 --trials 9
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 180s env CARGO_TARGET_DIR=/home/heefoo/code/delta-block-lm-ws/target-shared cargo test --workspace
bun scripts/summarize-update-loss.ts artifacts/update-and-loss/summary.json REPORT.json [...]
```

`update-loss-reference` can omit `--resume` to run just independent numerical/safety checks; supplying a historical directory explicitly adds its validated three-step continuation. The benchmark saves incremental reports before/toward its cooperative deadline and observes launches only in separate warm steps. `scripts/summarize-update-loss.ts` uses10000 deterministic bootstrap resamples (seed0x5eed) of within-trial differences; intervals are exploratory percentile intervals, not multiple-comparison-adjusted acceptance probabilities.

## Frozen source and isolated build

The production source is frozen in commit `b379379d1180d9f1bc8bc038034668ad3b0a4983` (change `mrqwrnsotxvylvyxmuuxzlrwsmxwxunz`). Remaining documentation is in evidence child `lrkouyttxtnnyuxrxtnuzvwonsqzwyyz`, so evidence snapshots do not rewrite the production commit.

- SHA256 of canonical `source_files` JSON (54 sorted inputs): `a9beab9a86581f7cc9c9f984d19b452dfe98a6eaeb9eb56b495be20cda50de5c`.
- Fixed build-helper SHA256: `987e44a9fd1711ab16924014d73b2629627e7f9e19cbe3fca4610b480aa381e3`.
- Private executable SHA256: `a54651d55a3242a43879c2a4e53398c70ca248ae1bd5d6e2164ce738f3a228a3`.
- Proof files: `artifacts/update-and-loss/source-manifest-b379379d.json` and `candidate-b379379d.json`. The manifest includes every Rust source in all three local crates, all Cargo manifests/lockfile and `.cargo/config.toml`; the source checksum was unchanged after the build.
- The fixed helper cleaned only local packages, then built default-CPU+Vulkan without fusion/autotune. Reported lock wait245s; Cargo build3.98s; outer transaction248.76s. The wait is not compilation or GPU execution time. `.cache` and `.cache/cubecl` were observed to be directories, not symlinks.

Final hardware commands use `/usr/bin/time` outside and inside the GPU lock: `outer_wall_seconds` includes queueing; `locked_execution_seconds` excludes queueing but includes process/device setup, compilation if needed, warmups, checks and report I/O. Warm complete-step samples remain independently synchronized and exclude those startup/report costs. Each locked binary invocation is limited by `timeout 290s`.

## Final isolated correctness

The frozen-source/private-binary invocation passed with `passed:true`, no error, the private relative shader cache and unchanged tolerances. Report: `artifacts/update-and-loss/isolated-reference/update-loss-reference.json`.

| Check | Cases/steps | Largest absolute difference |
|---|---:|---:|
| Scalar-f64 weighted loss | 32 | 4.035e-7 |
| Scalar-f64 row cross-entropy | 32 | 1.253e-6 |
| Every logits VJP | 32 | 2.375e-8 |
| Every floating loss-weight VJP | 32 | 9.362e-7 |
| AdamW parameter/moment oracle and native comparison | 16 × 5 updates | parameter error1.306e-7 |
| Historical checkpoint-13, every parameter gradient | 3 real updates | 4.285e-8 |
| Historical checkpoint-13, every updated parameter | 3 real updates | 9.537e-7 |
| Native AMSGrad producer→ordinary native/fused checkpoint consumers | 3 real updates + roundtrip | parameter error1.491e-8 |

The optional maximum tensor is **exactly** preserved after every consumer update and after saving/reloading the continued fused optimizer; this is a typed tensor-signature equality check, not a tolerance-based omission. Its complete native/fused checkpoint fixture is retained under `isolated-reference/optional-maximum-2903896-1792868585398371159/`. Both missing-gradient paths and NaN/±infinity/squared-overflow rejection pass. Zero/clipped gradients, transposed storage and loss padding weights remain part of the independent oracle.

Locked execution was9.25s; outer wall160.56s, hence approximately151.31s of lock queueing/process overhead outside the lock. No fresh duplicate full model/reference matrix is claimed here: the preceding passing matrices remain separately labelled provisional; the coordinator requested one final isolated correctness gate and owns the combined post-integration model/checkpoint gates.

## Decisive B8/context512 paired confirmation

Frozen source/private binary above, private shader cache, GPU/build transaction isolation, two warmups and **15 rotating/reversed paired trials** in one session. No build or another participating GPU run overlaps these samples. Complete steps retain loss and gradient readbacks, all missing/nonfinite checks, the original per-parameter clipping semantics, AdamW and actual batch uploads.

| Update / loss | Median complete step ms | Median paired saved ms | Median paired saved % | 95% paired-median bootstrap interval, ms | Kernels/step |
|---|---:|---:|---:|---:|---:|
| tensor / tensor | 1786.758 | 0 | 0 | — | 5833 |
| fused / tensor | 1760.759 | 31.005 | 1.739 | [22.088,49.058] | 3922 |
| tensor / fused64 | 1790.214 | 5.278 | 0.296 | [-14.771,31.623] | 5815 |
| tensor / fused256 | 1790.506 | -2.516 | -0.141 | [-26.652,13.285] | 5815 |
| fused / fused256 | 1756.682 | 35.893 | 1.927 | [19.568,41.362] | 3904 |

The update-only saving is positive in12/15 trials; baseline samples span1685.928–1936.835ms, so absolute throughput still drifts. The paired interval excludes zero despite that drift. Use the update-only arm for attribution: the combined arm's positive interval can be explained by the accepted update and does not rescue either loss-only interval.

| Resident actual-shape component | Tensor | Fused64 | Fused256 |
|---|---:|---:|---:|
| Final norm + native vocabulary head + loss, forward ms | 8.314 | 7.886 | 7.535 |
| Same component, forward+backward ms | 19.564 | 19.759 | 19.596 |

Resident full-parameter safety checks+clipping+AdamW: **35.122ms tensor /13.685ms fused**. This separate component result is not substituted for the measured31.005ms paired whole-step saving. Final shared pool reservation was825,509,952bytes, with the multi-trajectory/live-vs-reserved limitations described above. Both optimizer state cross-load validators pass.

Raw samples/histograms: `final-b8-c512/update-loss-2907981-1792868606011947909/report.json`; reproducible paired summary: `final-b8-summary.json`. Locked execution164.63s; outer166.81s; approximate queueing/outer process overhead2.18s. The run used `final-b8-c512.json`, `--warmups 2 --trials 15`, with the290s timeout inside the GPU lock.

## Required batch/context comparisons

All four runs use the same frozen source/private binary and two warmups. B8/context512 has15 trials; the other shapes have9. Column order below is the five independent, equal-start trajectories used within every rotating/reversed trial. These are complete forward+backward+checks+clipping+AdamW steps, not isolated kernels.

| Batch / context | Tensor update/loss ms | Fused update / tensor loss ms | Tensor update / fused64 loss ms | Tensor update / fused256 loss ms | Fused update / fused256 loss ms |
|---|---:|---:|---:|---:|---:|
| 1 / 128 | 185.563 | 170.411 | 190.088 | 192.138 | 166.696 |
| 1 / 512 | 555.236 | 551.728 | 557.646 | 572.233 | 547.707 |
| 8 / 128 | 414.309 | 357.085 | 381.264 | 524.910 | 363.451 |
| 8 / 512 | 1786.758 | 1760.759 | 1790.214 | 1790.506 | 1756.682 |

Each cell in the next table is **median paired saved ms [95% paired bootstrap interval]**, always relative to that trial's tensor/tensor step. Positive is faster; intervals crossing zero are not speedup evidence.

| Batch / context | Fused update | Fused64 loss only | Fused256 loss only | Fused update + fused256 loss |
|---|---|---|---|---|
| 1 / 128 | 16.061 [15.030,24.431] | -1.435 [-9.213,2.600] | -3.185 [-10.063,9.460] | 18.997 [15.717,23.881] |
| 1 / 512 | 7.398 [-11.921,17.424] | -17.443 [-24.575,3.100] | -11.274 [-37.023,1.449] | 16.027 [-17.245,31.542] |
| 8 / 128 | 57.224 [23.856,346.328] | -3.000 [-84.616,49.250] | 1.323 [-25.020,79.562] | 57.568 [27.280,252.725] |
| 8 / 512 | 31.005 [22.088,49.058] | 5.278 [-14.771,31.623] | -2.516 [-26.652,13.285] | 35.893 [19.568,41.362] |

Update-only paired median percentages are8.837% /1.332% /13.812% /1.739% in table order. **B1/context512 does not establish a beyond-noise integrated gain** in its9-trial session; it is not presented as a confirmed4–5% speedup from the isolated component. The adoption decision uses the explicitly requested realistic B8/context512 result.

Issued kernels/step in column order:

- B1, both contexts: **5780 /3869 /5762 /5762 /3851**.
- B8, both contexts: **5833 /3922 /5815 /5815 /3904**.

| Batch / context | Complete resident checks+clipping+AdamW tensor / fused ms | Final norm/head/loss F / FB ms, tensor | F / FB ms, fused64 | F / FB ms, fused256 | Sampled pool reservation bytes |
|---|---:|---:|---:|---:|---:|
| 1 / 128 | 35.797 /12.802 | 1.663 /2.817 | 1.306 /2.172 | 1.398 /2.243 | 228083456 |
| 1 / 512 | 38.138 /12.439 | 2.610 /4.747 | 2.394 /4.613 | 2.642 /4.994 | 252855104 |
| 8 / 128 | 205.881 /122.627 | 12.199 /28.024 | 7.764 /28.068 | 7.304 /19.779 | 340929856 |
| 8 / 512 | 35.122 /13.685 | 8.314 /19.564 | 7.886 /19.759 | 7.535 /19.596 | 825509952 |

**Short-context B8 anomaly, retained rather than hidden:** its baseline full steps range365.495–1034.618ms. Resident update samples range138.396–684.114ms for tensor and35.888–209.700ms for fused, despite the same parameter count and fixed resident gradients within that component loop. The cause is not attributed: no additional hardware experiment was run at the coordinator's request. These highly variable samples do not establish a stable13.8% short-context gain, a constant update cost at every shape, or satisfaction of the15ms B8/context512 component budget at context128. B1/context128 is separately stable enough for a positive paired interval; neither result substitutes for the decisive realistic long-context batch.

All four optimizer-state cross-load checks pass. Pool reservations include all scratch trajectories, isolated states and allocator reuse, not candidate-specific physical peaks. No loss-only interval excludes zero at any required shape; the smaller loss-only isolated F/FB values at some shapes do not justify adopting a loss selector.

Raw reports (all under `artifacts/update-and-loss/`):

- `final-b1-c128/update-loss-3004074-1792869212665013350/report.json`
- `final-b1-c512/update-loss-2968348-1792869019098050732/report.json`
- `final-b8-c128/update-loss-2978366-1792869078503342937/report.json`
- `final-b8-c512/update-loss-2907981-1792868606011947909/report.json`

`final-summary.json` contains every mode's medians, paired intervals, launch counts and component samples' medians. `shape-diagnostics.json` retains the anomalous ranges. `isolated-command-outcomes.json` binds every final command to the frozen source/binary hashes and separates queueing from locked execution:

| Command | Locked execution s | Outer wall s | Approximate queue/outer overhead s |
|---|---:|---:|---:|
| Independent reference + both checkpoint continuation regressions | 9.25 | 160.56 | 151.31 |
| B8 / 512,15 trials | 164.63 | 166.81 | 2.18 |
| B1 / 512,9 trials | 35.92 | 412.36 | 376.44 |
| B8 / 128,9 trials | 46.55 | 70.05 | 23.50 |
| B1 / 128,9 trials | 13.06 | 100.64 | 87.58 |

### Exact final bounded commands

Run from the named workspace. The two timing wrappers distinguish queueing from locked execution; each `timeout` is inside the resource lock. The four step sessions use the same already-copied binary, not a rebuild between arms.

```sh
../build.sh optim
/usr/bin/time -f 'outer_wall_seconds=%e' flock /tmp/delta-block-lm-gpu.lock /usr/bin/time -f 'locked_execution_seconds=%e' timeout 290s /tmp/delta-bins/optim/delta-experiment update-loss-reference artifacts/update-and-loss/isolated-reference.json --resume artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13
/usr/bin/time -f 'outer_wall_seconds=%e' flock /tmp/delta-block-lm-gpu.lock /usr/bin/time -f 'locked_execution_seconds=%e' timeout 290s /tmp/delta-bins/optim/delta-experiment update-loss-benchmark artifacts/update-and-loss/final-b8-c512.json --warmups 2 --trials 15
/usr/bin/time -f 'B1C512_outer_wall_seconds=%e' flock /tmp/delta-block-lm-gpu.lock /usr/bin/time -f 'B1C512_locked_execution_seconds=%e' timeout 290s /tmp/delta-bins/optim/delta-experiment update-loss-benchmark artifacts/update-and-loss/final-b1-c512.json --warmups 2 --trials 9
/usr/bin/time -f 'B8C128_outer_wall_seconds=%e' flock /tmp/delta-block-lm-gpu.lock /usr/bin/time -f 'B8C128_locked_execution_seconds=%e' timeout 290s /tmp/delta-bins/optim/delta-experiment update-loss-benchmark artifacts/update-and-loss/final-b8-c128.json --warmups 2 --trials 9
/usr/bin/time -f 'B1C128_outer_wall_seconds=%e' flock /tmp/delta-block-lm-gpu.lock /usr/bin/time -f 'B1C128_locked_execution_seconds=%e' timeout 290s /tmp/delta-bins/optim/delta-experiment update-loss-benchmark artifacts/update-and-loss/final-b1-c128.json --warmups 2 --trials 9
bun scripts/summarize-update-loss.ts artifacts/update-and-loss/final-summary.json artifacts/update-and-loss/final-b8-c512/update-loss-benchmark.json artifacts/update-and-loss/final-b1-c512/update-loss-benchmark.json artifacts/update-and-loss/final-b8-c128/update-loss-benchmark.json artifacts/update-and-loss/final-b1-c128/update-loss-benchmark.json
```

