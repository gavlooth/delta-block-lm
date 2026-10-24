# AdamW/checks/clipping and cross-entropy evaluation

## Scope and baseline

Intel Arc MTL integrated GPU (`spirv_32902_32085`), Vulkan, f32, Burn 0.22 / CubeCL 0.11; no Burn fusion or matmul autotune. The controlled model is `configs/lm512-projections.json`: four layers, width 256, H4/K32/V64, chunk/cache 16, fused KronLSTM/HOLA, merged projections, tensor normalization, native 257-column vocabulary head, seed 1 and pinned byte-LM corpus. Batch 8 is the adoption workload; batch 1 and contexts 128/512 are reported separately. Recommended and historical config files are unchanged.

The existing batch-8/context-512 attribution is 37.967 ms for all non-autodiff checks/clipping/AdamW (2.35% of the 1,613.667 ms complete step), and 19.492 ms for final normalization/head/loss (1.21%). These historical numbers are prioritization evidence, not baselines to subtract from a different measurement session. Every new throughput session includes its own unchanged tensor-path control in the same executable.

## Compared designs and exact semantics

### Update

1. **Existing Burn primitives:** all-parameter squared-norm safety check and readback, then Burn's AdamW plus configured norm clipping.
2. **Explicit `update_backend: "fused_gpu"`:** bounded square-sum kernels without a full squared-gradient tensor, the same missing-gradient check and one aggregate scalar safety readback, and one elementwise clipping/moment/bias-correction/decay/update kernel per parameter. The first prototype allocated each output/moment result; the final path reuses uniquely owned contiguous parameter/moment buffers using CubeCL output aliases. Shared or strided buffers receive independent outputs, with no copies just to make inputs contiguous.

This is **per-parameter fusion, not a claimed multi-tensor optimizer**. Burn's ModuleOptimizer exposes a per-parameter `Optimizer` extension, which permits using the exact existing `AdamWState` record and checkpoint validator. A packed multi-tensor alternative would require packing all 2,379,568 parameter/gradient/moment elements (or replacing the module state/serialization interface). The implemented primitive-reuse and per-parameter designs establish the attainable return without claiming an unimplemented dispatch architecture or changing storage. Check and clipping reductions remain separate: the optimizer cannot borrow the earlier check's per-parameter norm through Burn's extension interface. Launch counts expose that duplication rather than hiding it.

**Clipping is per parameter, not a global model norm.** Burn0.22 computes `min(threshold / (sqrt(sum(g*g)) + f32::MIN_POSITIVE), 1)` for each parameter. The all-model sum is a safety check only. Missing gradients still abort before any optimizer write; squared-sum NaN, infinity and overflow still abort via the mandatory readback. A checked loss does not replace checked gradients.

AdamW preserves Burn's f32 beta constants 0.9/0.999, factors `1-beta`, host f32 `1-beta.powi(time)` bias corrections, epsilon **1e-5**, and `parameter*(1-lr*weight_decay) - lr*((m/correction1)/(sqrt(v/correction2)+epsilon))`, without moving epsilon or folding the learning rate into bias correction. The optional AMSGrad state is not silently discarded. Parameter IDs, names/shapes, first/second moments and per-parameter time counters retain the native record layout. Omitted `update_backend` means the original implementation and serialized identity.

### Loss

1. **Existing primitives:** vocabulary projection, log-softmax, target gather, weighted sum divided by weight sum.
2. **`loss_backend: "fused64"`:** 64-lane row-local stable max/sum-exp/target selection, followed by the unchanged tensor weighted mean; analytic logits VJP recomputes probabilities from saved per-row maximum/log-sum.
3. **`loss_backend: "fused256"`:** the same arithmetic with 256 lanes, trading fewer per-lane vocabulary loads for a deeper shared-memory reduction.

Both candidates eliminate the vocabulary-sized log-softmax and its generic backward graph. They retain the real logits/vocabulary matmul and weighted-mean primitives: this is not an assertion that a custom GEMM, fused final RMSNorm, or custom weight-gradient reduction was implemented. Keeping the mean in ordinary differentiable primitives preserves gradients to floating loss weights as well as logits, including fractional and zero weights. There is no padding special case or all-zero-weight fallback. Reset/validity handling in the model is untouched. Labels are exact f32 casts of the existing integer targets; supported vocabulary indices fit the exact integer range. Kernels validate f32/device/32-bit spans and support strided inputs. Omitted `loss_backend` preserves the historical implementation/fingerprint.

All production, evaluation, diagnostics, profiling and model-check callsites select loss through the model. The CPU model oracle explicitly keeps tensor update/loss, rather than accidentally certifying a GPU candidate against itself. No new optimizer-state layout or historical config migration is needed.

## Numerical and checkpoint evidence

Unchanged tolerance throughout: `2e-5 + 3e-4*abs(reference)`, with the original independent CPU directional gate's `3e-3 + .03*abs(reference)`.

- Independent scalar-f64 cross-entropy: **32 cases**, widths 1/7/35/257, short rows/tails, zero/tied and nonzero logits, fractional/zero padding weights, contiguous and genuinely transposed inputs, both lane designs. Outputs, every logits gradient and every loss-weight gradient pass. Maximum scalar-loss error 4.035e-7 and logits-gradient error 2.375e-8.
- Independent scalar-f64 AdamW plus the original Burn optimizer: **16 cases × five updates**, zero/small/clipped/large gradients, zero/nonzero decay and contiguous/transposed inputs. Every parameter and first/second moment is checked on every step. Maximum parameter error 1.306e-7. NaN, positive/negative infinity and squared-sum overflow are rejected; missing gradients are rejected.
- Config tests cover omitted historical serialization, fingerprint differences for explicit selectors, GPU-only selection and rejection of automatic-fusion combinations. Workspace CPU tests passed: **26 experiment + 10 mixer**, including original optimizer resume and partial-state rejection.
- `kron-model-check` with both candidates enabled passes all four configured cases: contexts128/512, HOLA off/on, CPU logits, every parameter gradient, AdamW update and the original full-parameter directional finite-difference gate. Largest logits/gradient/updated-parameter errors: **5.126e-6 / 2.295e-6 / 9.191e-6**. This configured-scope gate does not certify the known failing arbitrary-input legacy GPU tensor solver.
- Exact historical checkpoint `artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` loads through both optimizer implementations using the unchanged tensor/scalar/config-state validator. Three real continuation steps compare every raw gradient and updated parameter: maxima **4.285e-8 / 9.537e-7**. This is a validation/continuation check, not permission to resume a changed execution selector under an old run fingerprint.
- Benchmark state roundtrips save native-layout fused optimizer moments and load/validate them through both implementations.

Evidence root: `artifacts/update-and-loss/`. Scalar/reference and continuation report: `baseline-b1/update-loss-reference.json`. Configured model gate: `both/kron-model-check-2657068-1792866488775894497/report.json`. CPU test output is the recorded `cargo test --workspace` run; no tolerances or old failures were changed.

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
