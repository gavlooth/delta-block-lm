# Steps 1–9: implementation and measured results

For the current optimized execution paths, measured speedups and known legacy GPU-solver limitation, see [Early GPU kernel optimization](#early-gpu-kernel-optimization). The current recommended batch-8 configuration and its combined measurements are in [Batch-8 component optimization round](#batch-8-component-optimization-round). Step 7 (oHC, wavelet mixer, three Lngram injections) is in [Step 7: oHC, wavelet and Lngram](#step-7-ohc-wavelet-and-lngram). The first sections retain the original pilot measurements.

## Scope and stopping point

Implemented and exercised the active Burn workspace, causal GDN-2 and KronLSTM, optional HOLA/recency memory, soft top-k feedback, paired associative-recall experiments, and a tiny causal language-model comparison. The Steps 1–6 sections below predate Step 7, which is reported separately at the end. Step 8 includes user-selected in-block reads through HOLA. Step 9 adds Codpula joint output and dependence-adaptive commitment; this turn's evaluation scope is numerical correctness, execution cost and a tiny smoke, not a quality study. Earlier dense-attention measurements remain historical. CoCommit and energy-guided remasking remain later work.

All runs used the local Intel Arc Graphics (MTL) through Vulkan, with float32 execution. No Spark or predecessor production process was used or modified.
Run artifacts report sampled backend-reserved memory and explicit recurrent/cache/convolution storage. Exact GPU peak memory is not available here; the integrated GPU shares system memory, so reserved-pool samples must not be presented as exact physical-device peaks.

## Implementation and numerical evidence

- `crates/kernels`: migrated CubeCL GDN-2 forward/backward operator, separately exercised against the float64 reference.
- `crates/mixers`: causal GDN-2 and KronLSTM chunk solvers, sequential float64 references, short causal convolutions, optional HOLA with surprise or matched recency selection, learned cache normalization/gate/sink.
- `crates/experiment`: full-parameter AdamW training, deterministic datasets, file configurations, soft-feedback episode, evaluation, resumable checkpoints, bounded runners and independent checkpoint certification.
- Training uses the differentiable Burn chunk tensor solver on Vulkan. These full-model timings are not timings of an optimized fused KronLSTM kernel. The GDN-2 custom-kernel check is independent of the training-path measurements.
- Initial supported computational chunks are 1–16. Larger contexts work across multiple chunks. This is not a restriction on future denoising-block sizes.

CPU and Vulkan recurrence/cache/layer checks passed. The Vulkan report compares float32 execution with float64 sequential analytic references, using absolute tolerance 2e-5 and relative tolerance 3e-4. It covers short and multi-chunk inputs, document resets, internal padding, zero-decay boundaries, final-state/continuation behavior, cache selection and gradients. Full-layer directional and parameter finite-difference checks include cache normalization, gate and sink parameters. The separate GDN-2 CubeCL operator also passed output/backward checks, including a sequence shorter than one computational chunk.

The soft-feedback episode checks dense/full-support and sparse/truncated expectation paths, selected-weight/embedding gradients, candidate mass, visibility-only remasking, retained proposals, immutable observations, padding, ties and zero-mass cases. It is the migrated diffusion foundation, not yet a trained block-denoising model.

Final `cargo test --workspace`: **16 passed, 0 failed**. The intentionally retained ndarray CPU backend emits deprecation warnings; it is the CPU numerical reference, not the local GPU path.

Numerical reports:
- `artifacts/reference-cpu/reference.json`
- `artifacts/reference-vulkan/reference.json`
- `artifacts/reference-cpu/soft.json`

## Build behavior

The first Vulkan dependency build exceeded its 120-second limit. Resuming from the compiled dependencies completed in the next bounded invocation. This cold-start result is recorded rather than hidden.

After a documentation-only edit in the mixer module, the measured Vulkan model/executable rebuild took **2.001 s wall-clock**, and the CPU test rebuild plus execution took **2.365 s**. This is a limited incremental-edit measurement, not a promise that every kernel or dependency change rebuilds that quickly. Subsequent substantive experiment changes also built within the configured deadlines.

The workspace uses clang/mold locally, no dev/test debug information, optimized external dev dependencies, unoptimized local crates, optional GPU features and a persistent CubeCL compilation cache. CPU and GPU first-use costs are recorded separately from warm throughput.

## Associative recall

Both experiments used two layers, width 128, four heads, key width 32, value width 64, computational chunk 16, cache capacity 16, batch size 1 and paired seeds 1/2/3. Each suite selected one fixed token budget from a short warm full-model pilot; all arms in that suite received the same budget.

| Arm | Context 128: mean answer accuracy | Context 512: mean answer accuracy |
|---|---:|---:|
| GDN-2 | 0.893% | 1.344% |
| GDN-2 + HOLA | 0.893% | 1.478% |
| KronLSTM | 1.488% | 1.613% |
| KronLSTM + HOLA | 1.488% | 1.747% |
| GDN-2 + recency cache | 2.381% | Not run |
| KronLSTM + recency cache | 1.786% | Not run |

Context 128 used 11,136 training tokens per run; context 512 used 11,264. Uniform guessing among the 64 possible answer values is **1.5625%**. The small differences do **not** establish useful associative recall or a mixer winner. Every seed/arm's diagnostic lower confidence bound must exceed guessing before the program emits a recall shortlist; neither suite meets that requirement. The diagnostic assumes independent Bernoulli outcomes and is not a proof when queries are correlated or evaluation data is reused.

Held-out NLL can improve simply by suppressing symbols that are never valid answers. Accordingly, NLL improvement is kept separate from demonstrated retrieval. Both HOLA candidates were retained for the independent tiny-LM comparison rather than selecting an architecture from chance-level accuracy.

## Tiny causal language model

Configuration: four layers, width 256, context 512, four heads, key width 32, value width 64, computational chunk 16, HOLA capacity 16, batch size 1, float32, AdamW, learning rate 0.001. Every arm/seed trained for **6,656 byte tokens, 13 optimizer steps**; seeds were 1/2/3. This is a learning/engineering pilot, not a useful pretrained language model or a scaling result.

Corpus: four complete Project Gutenberg books for training, Tom Sawyer for validation, Moby Dick reserved as test. Whole books are assigned before tokenization/packing. The tokenizer is UTF-8 bytes plus BOS (257 symbols). Source hashes, document hashes, split hashes and license notes are in `data/books/manifest.json`; reproducible source metadata is tracked under `configs/`. The report evaluates four deterministic, nonoverlapping stratified 512-byte validation windows. The same fixed evaluation set is used for all arms and seeds. This is a narrow validation sample from one held-out book, not a broad language benchmark. Test results are retained in artifacts but not used for the table or architecture selection.

The following values use the corrected, independently certified checkpoint evaluations. Training throughput is the pinned recorded training measurement, not a claim that certification retrained or rebenchmarked the models.

| Arm | Parameters | Mean validation NLL (nats/byte) | Seed standard deviation | Recorded training tokens/s |
|---|---:|---:|---:|---:|
| GDN-2 | 2,764,304 | 3.14977 | 0.01363 | 347.3 |
| GDN-2 + HOLA | 2,764,592 | 3.15007 | 0.01485 | 246.9 |
| KronLSTM | 2,379,280 | 3.11404 | 0.02894 | 319.2 |
| KronLSTM + HOLA | 2,379,568 | 3.11386 | 0.02834 | 228.2 |

All four arms improved on independently regenerated, repeat-checked untrained baselines in every seed. The baselines are labeled regenerated initialization, not a bitwise replay of unsaved historical initial weights.

Interpretation:
- KronLSTM is worth retaining as a challenger: fewer parameters and slightly lower loss here, but not higher measured throughput.
- HOLA has no demonstrated quality gain at this short budget and makes training slower. Its two base mixers remain available; this is not evidence to discard HOLA for a genuinely learned long-range retrieval task.
- Parameter counts differ, evaluation is narrow, and training is extremely short. These results do not prove one architecture is generally better. The program's empirical Pareto screen is descriptive, not statistical confirmation.
- The trained LMs' byte-associative-retrieval diagnostics are retained separately; next-token learning alone is not evidence that they learned the synthetic memory task.

## Checkpoint bug found and fixed

A no-training checkpoint reload initially changed one KronLSTM validation NLL from about 3.15 to 7.59; another reload returned about 3.15. Investigation reproduced fresh save/load parameter mismatches on Vulkan. Host-decoded file tensors matched all saved pre-write values, but some loaded GPU parameters retained their initializer: for example, an entire decay-bias vector became -4. CPU loading passed. Same-model repeated inference did not mutate parameters.

Verified fix: keep the constructed target model alive, synchronize its queued initialization before applying the record, synchronize after loading, and verify every parameter's logical bytes, shape, dtype and saved identity against the record. This eliminated the reproduced corruption. The exact upstream allocator/command-ordering cause remains an inference; no backend patch or tolerance increase was used.

Optimizer restoration also compares complete canonical logical state, including typed tensors/scalars and AdamW settings. Raw file hashes are not used for semantic optimizer equality because container ordering may differ. Existing checkpoint paths cannot be reused merely because config/step metadata matches: their logical model and optimizer state must also match, or the operation fails without replacing the old evidence.

The final Vulkan diagnostic passed on all four original language-model checkpoint arms, before and after an isolated scratch update. It verified exact model roundtrips, actual source optimizer restoration, and identical next-step model/optimizer results. Old failed diagnostic directories and original stdout remain available.

Pre-fix metrics were not silently edited into trusted results. The final certification batch:
- pins source configuration, checkpoint, checkpoint-state and original training-evidence hashes;
- uses the exact original metrics bytes, avoiding JavaScript/Rust floating-point reserialization differences;
- reevaluates every arm and all three seeds with guarded loading;
- repeats validation/test evaluation and regenerates untrained baselines;
- writes new, separate versioned certificates without training or changing original outcomes.

All three suites passed certification. Batch manifest and subprocess outcomes:
`artifacts/certification-batch-1792845940896-349708/command-outcomes.json`.

Final comparisons:
- `artifacts/recall128-vulkan/certified-evaluations/vulkan-349723-1792845941088426411/comparison.json`
- `artifacts/recall512-vulkan/certified-evaluations/vulkan-354437-1792845987408889792/comparison.json`
- `artifacts/lm512-vulkan/certified-evaluations/vulkan-362617-1792846070036817669/comparison.json`

Datasets, checkpoints, generated evidence, caches and local session notes are ignored by version control. This tracked file preserves the measured summary; rerun scripts preserve configuration and data provenance.

## Reproduction of the original pilot

The commands below describe the original pilot. The expanded all-path Vulkan reference now also reports the legacy GPU-solver stress failure documented below; use the explicit fused checks in the optimization section for current optimized acceptance.

See `crates/experiment/README.md` and `crates/mixers/README.md` for command details. From the workspace root:

```sh
timeout 60s cargo test --workspace
timeout 120s cargo build -p delta-experiment --features vulkan
timeout 120s target/debug/delta-experiment reference configs/reference-vulkan.json
timeout 30s target/debug/delta-experiment soft configs/reference-cpu.json
bun scripts/prepare-data.ts
# New training runs: each child command has its own deadline.
timeout 1500s bun scripts/run-experiments.ts recall128 vulkan
timeout 1500s bun scripts/run-experiments.ts recall512 vulkan
timeout 1500s bun scripts/run-experiments.ts lm512 vulkan
# Existing pre-fix runs require explicit checkpoint certification rather than cache reuse.
timeout 1000s bun scripts/certify-experiments.ts target/debug/delta-experiment
```

Do not launch an all-suite repeat merely to read results; use the certified JSON above. Conflicting old/new run identities fail deliberately. Use a distinct artifact directory and a fresh matching pilot for a changed configuration. No Step-7 component was implemented as part of this work.

## Early GPU kernel optimization

The early component-optimization rule is now in `AGENTS.md`. Every new or materially changed component must establish correctness, isolated cost and integrated cost before scaling. This includes future components when they are implemented, not permission to start Step 7.

### Implemented paths

- KronLSTM: fused f32 CubeCL scan and analytic backward, one workgroup per batch/head, chunk-boundary checkpoints and replay. Output, final state and physical edit scores share one differentiable packed result, so combined losses accumulate all adjoints correctly. All seven floating inputs, including initial state, have gradients.
- HOLA: fused stable position selection followed by per-query attention over original KV tensors. Backward visits each key's causal chunk and uninterrupted cache-resident lifetime. No floating-point atomics, copied KV cache, quadratic rank matrix or dense attention-probability matrix is required.
- Descending surprise/recency selection and earlier-position tie breaking, document matching, padding, sink gradients and external normalization/gate gradients are preserved.
- Initial optimized scan scope is f32, key width 32, value width 64, computational checkpoint chunks 1–16 and first-order derivatives. Unsupported fused execution fails explicitly, rather than silently using the tensor implementation. U32 address-span limits are checked before allocation/launch.

Historical omitted selectors still mean `tensor_solver` / `tensor`, preserving configuration and checkpoint identity. Recommended explicit configuration for the measured local workload: `configs/lm512-kron-fused.json`, with both `scan_backend` and `hola_backend` set to `fused_gpu`. Its real training pilot was exercised successfully. No Step-7 components were activated.

### Complete guarded training steps

Local Intel Arc MTL/Vulkan, f32, batch 1, four layers, width 256, key/value widths 32/64, computational chunk 16, HOLA capacity 16. Two warmups and five paired samples; table uses synchronized median complete training-step wall time. Loss reads, finite-gradient checks, backward and AdamW are included. CPU numerical comparisons and checkpoint I/O are outside these samples. These are not transfer-free GPU-compute timings.

| Context | Kron path | HOLA path | Median step, ms | Tokens/s | Speedup within context |
|---|---|---|---:|---:|---:|
| 128 | Tensor | Tensor | 547.82 | 233.7 | 1.00x |
| 128 | Fused | Tensor | 308.69 | 414.7 | 1.77x |
| 128 | Tensor | Fused | 424.74 | 301.4 | 1.29x |
| 128 | Fused | Fused | 214.06 | 598.0 | 2.56x |
| 512 | Tensor | Tensor | 2,322.06 | 220.5 | 1.00x |
| 512 | Fused | Tensor | 971.23 | 527.2 | 2.39x |
| 512 | Tensor | Fused | 1,723.34 | 297.1 | 1.35x |
| 512 | Fused | Fused | 601.02 | 851.9 | 3.86x |

Without HOLA, replacing only the scan improved the context-512 model from 1,828.03 to 598.75 ms/step (3.05x). These are pilot-shape results on the local GPU, not forecasts for CUDA/Spark, larger dimensions, mixed precision or diffusion training.

An independent pre-change five-step pilot recorded 323.9 tokens/s without HOLA and 222.0 with HOLA. It is retained at `artifacts/kron-before/budget.json`, not substituted for the paired baseline above. Device/load variation and different timing protocols matter; use the paired comparison for speedup estimates.

### Isolated HOLA cost

Same local device, four heads, key/value widths 32/64 and chunk 16. Synchronized medians after correctness checks and two warmups, five alternating paired samples:

| Context | Capacity | Tensor forward, ms | Fused forward, ms | Tensor forward+backward, ms | Fused forward+backward, ms |
|---|---:|---:|---:|---:|---:|
| 128 | 16 | 15.57 | 0.84 | 23.55 | 7.53 |
| 128 | 64 | 16.66 | 1.19 | 24.66 | 11.29 |
| 512 | 16 | 57.68 | 1.25 | 79.43 | 15.92 |
| 512 | 64 | 63.49 | 2.16 | 87.53 | 34.39 |

At context 512/capacity 16, HOLA's isolated forward gain is about 46x and forward+backward gain about 5x; neither is an equal whole-model speedup. First-call samples can include shape-specific setup/JIT and already-warm caches; they are not an isolated cold-compilation benchmark.

At batch 1/head 4/context 512/chunk 16, HOLA schedule/lifetime/softmax metadata is 33,024 bytes at capacity 16 or 58,368 at capacity 64. Backward auxiliary storage adds 16,384 bytes; retained output is separately 524,288 bytes. These are implementation buffer sizes, not measured total GPU peak memory.

### Acceptance and known legacy limitation

Fused Kron passed independent f64 output, final-state, physical-score, all-input and initial-state VJPs, including combined and separate objectives, zero decay, signed/zero correction, zero norms, padding, resets and strided/chained continuation. HOLA tensor/fused checks passed against an independent scalar oracle, including membership and Q/K/V/sink gradients at capacities 16/64. Regressions cover normalized-before-accumulation large values and consistent forward/backward arithmetic for cancellation-sensitive logits. The large-value fixture uses normal-range upstream gradients: Vulkan subnormal preservation is not assumed.

The expanded checks also exposed an **unresolved legacy GPU TensorSolver gradient defect** for a nonzero-initial-state, normalized-key/value, short-sequence stress fixture. CPU TensorSolver and fused Kron pass; GPU TensorSolver's key gradient fails the unchanged tolerance. Output-only and score-only objectives reproduce it; final-state-only passes. Rank-four overlap, direct triangular substitution and RHS reassociation experiments did not fix it and were reverted.

This failure is not hidden or accepted as a correct numerical oracle. `reference` and the strict `kron-benchmark` retain their all-path gates and report it. The workload-scoped `kron-model-check` / `kron-model-benchmark` instead:
1. retain and hash the original legacy failure evidence, explicitly setting `global_tensor_solver_stress_accepted=false`;
2. validate the fused kernels independently against f64;
3. validate each exact normalized model being timed against a genuine CPU reference, including logits, every parameter gradient, one AdamW update and a full-parameter directional finite-difference sanity check.

All measured model cases passed those scoped checks. This certifies the reported fixtures at initialization/first update, not arbitrary legacy inputs or complete training trajectories. No tolerances were loosened. The GPU tensor solver remains a historical/reference path with this documented limitation; use explicit fused paths for new supported local experiments.

Checkpoint save and guarded load/validation are measured separately. In the context-512/HOLA-on samples, save was approximately 249–259 ms and validated load approximately 326–349 ms; these costs were excluded from step speedups. A historical language-model checkpoint and optimizer continuation were also exercised successfully after adding selectors.

Final formatted workspace regression suite: 20 tests passed. GPU checks and benchmarks are explicit runtime commands, not inferred from those CPU test results.

The final host-side safety pass also added checked element counts, logical strided spans and backing-buffer limits to HOLA before allocation and launch. Seven GPU-feature host address-bound tests (Kron and HOLA) passed, and HOLA's 19 GPU numerical reports passed again after the host-only change. The first isolated GPU-feature test build exceeded 120 seconds because it compiled a separate backend feature combination; its bounded cached continuation completed. This is recorded as cold build cost, not a warm-test failure.

### Evidence and commands

- Full-model report: `artifacts/kron-kernel-benchmarks/kron-model-benchmark-vulkan-1164784-1792854492015360185/report.json`
- Isolated HOLA report: `artifacts/kron-kernel-benchmarks/hola-benchmark-970757-1792852961786446555/report.json`
- Prior full-model numerical check: `artifacts/kron-kernel-benchmarks/kron-model-check-1094285-1792853911005241139/report.json`
- Legacy failure retained for reproduction: `artifacts/kron-validation/solver-output_only/kron-reference.json`

```sh
timeout 120s cargo build -p delta-experiment --features vulkan
timeout 120s target/debug/delta-experiment kron-reference configs/reference-vulkan.json --scan-backend fused_gpu
timeout 120s target/debug/delta-experiment hola-reference configs/reference-vulkan.json
timeout 120s target/debug/delta-experiment benchmark configs/lm512-kron-fused.json
timeout 400s target/debug/delta-experiment hola-benchmark configs/kron-benchmark.json --trials 5
# Expected to exit 1 on the affected local backend; retain the real failure report.
timeout 120s target/debug/delta-experiment kron-reference configs/reference-vulkan.json --scan-backend tensor_solver --tokens 7 --objective output_only
timeout 400s target/debug/delta-experiment kron-model-benchmark configs/kron-benchmark.json --warmups 2 --trials 5 --legacy-stress-report artifacts/reference-vulkan/kron-reference.json
```

The scoped benchmark requires the retained failed report; it does not manufacture a successful legacy gate. All generated artifacts remain local/ignored. Existing old CPU and GPU configuration defaults were not silently changed, and no subsequent architecture/training-plan step was implemented.

On a fresh checkout, the explicit TensorSolver diagnostic above generates the required failure report; do not chain that expected failing command with `&&`. If all reference checks pass on another backend/version, use the strict `kron-benchmark` instead. Never manufacture legacy failure evidence to enable the scoped benchmark.

## Whole-model profile

The reusable `profile` command profiles the actual causal byte-LM training path, not isolated synthetic operators: the reference configuration, KronLSTM + HOLA with both selectors explicitly `fused_gpu`, local Intel Arc MTL/Vulkan, f32, four layers, width 256, batch 1, heads 4, K32/V64, chunk 16, HOLA capacity 16, vocabulary 257, seed 1 and the pinned training corpus. Contexts are explicitly selected without changing historical configuration defaults or old run identities. AdamW uses the configured learning rate 0.001, weight decay 0.01 and norm clipping 1.0.

Two warmups per mode precede five alternating paired trials on independent scratch training trajectories with identical data and step indices. Every measured section is bracketed by device synchronization; rows sum that section across calls/blocks *within each trial*, then report the median. The ordinary complete-step comparison has outer barriers but no added internal barriers beyond production loss/gradient scalar readbacks, and includes host batch preparation/uploads, the real autodiff forward, loss, backward, safety checks, clipping and AdamW. Corpus loading, initialization, cold first use, instrumentation preflight, checkpoint I/O, evaluation and artifact writes are excluded.

**The section barriers perturb overlap and submission/fusion.** Percentages divide by the ordinary internally-unsynchronized step median, so rows need not add to 100% and section sums are not estimates of unperturbed device utilization. Forward rows retain the training graph; backward is attributed only as one total (including gradient extraction), because Burn does not expose its time per forward section. Gradient-check reductions/readback and AdamW are separate.

The command first compares instrumented versus ordinary logits, every parameter gradient and one AdamW update using the unchanged 2e-5 absolute/3e-4 relative tolerance. This is an instrumentation check, not a replacement for the independent f64 component/full-model evidence above, and not acceptance of the known legacy tensor-solver stress defect.

Reproduction from the workspace root:

```sh
timeout 180s cargo build -p delta-experiment --features vulkan
timeout 180s target/debug/delta-experiment profile configs/lm512-kron-fused.json --contexts 128,512 --warmups 2 --trials 5
timeout 60s cargo test --workspace
```

### Measured ranked breakdown

| Context | Ordinary complete step, median ms | Section-synchronized complete step, median ms | Sum of section medians, ms | Issued kernels per complete step |
|---|---:|---:|---:|---:|
| 128 | 228.48 | 332.78 | 325.47 | 5,712 |
| 512 | 610.84 | 718.42 | 704.93 | 5,712 |

Rows below are ranked by context-512 cost; the report also ranks each context and each block independently. Percentages use each context's ordinary complete-step median. RMSNorm, projections, convolutions, activations, recurrence, HOLA and MLP values aggregate all four blocks.

| Section | Context 128 median ms | % of step | Context 512 median ms | % of step |
|---|---:|---:|---:|---:|
| Backward (total only) | 147.886 | 64.73% | 441.662 | 72.30% |
| Recurrent scan | 26.741 | 11.70% | 79.568 | 13.03% |
| AdamW update, including clipping | 31.510 | 13.79% | 28.923 | 4.73% |
| L2 normalization and gate activations | 24.659 | 10.79% | 23.083 | 3.78% |
| Short causal convolutions | 20.133 | 8.81% | 18.463 | 3.02% |
| MLP: down | 3.552 | 1.55% | 11.532 | 1.89% |
| MLP: up | 3.451 | 1.51% | 8.687 | 1.42% |
| Gradient checks: reductions | 9.436 | 4.13% | 8.060 | 1.32% |
| Projection: output gate | 3.220 | 1.41% | 6.912 | 1.13% |
| Projection: output | 3.420 | 1.50% | 6.472 | 1.06% |
| Projection: v | 3.470 | 1.52% | 6.305 | 1.03% |
| Projection: decay | 3.606 | 1.58% | 6.289 | 1.03% |
| Projection: k | 3.323 | 1.45% | 6.231 | 1.02% |
| Projection: q | 3.137 | 1.37% | 6.182 | 1.01% |
| Projection: erase | 3.065 | 1.34% | 5.689 | 0.93% |
| Projection: write | 3.110 | 1.36% | 5.466 | 0.89% |
| HOLA read | 3.489 | 1.53% | 5.135 | 0.84% |
| HOLA Q/K RMSNorms | 5.164 | 2.26% | 5.007 | 0.82% |
| Masks and document IDs | 3.897 | 1.71% | 4.356 | 0.71% |
| Residual adds | 3.607 | 1.58% | 3.596 | 0.59% |
| Mixer output RMSNorm | 2.590 | 1.13% | 2.876 | 0.47% |
| MLP: SiLU | 2.458 | 1.08% | 2.821 | 0.46% |
| Block RMSNorm: mixer | 2.627 | 1.15% | 2.765 | 0.45% |
| Block RMSNorm: MLP | 2.747 | 1.20% | 2.550 | 0.42% |
| Final norm and vocabulary head | 1.185 | 0.52% | 2.071 | 0.34% |
| Loss: log-softmax, gather and weighted reduction | 1.345 | 0.59% | 1.197 | 0.20% |
| Host/device: batch construction and upload | 0.880 | 0.39% | 1.175 | 0.19% |
| Host/device: gradient norm readback | 0.639 | 0.28% | 0.646 | 0.11% |
| Embedding | 0.476 | 0.21% | 0.640 | 0.10% |
| Host/device: loss readback | 0.578 | 0.25% | 0.509 | 0.08% |
| Host batch preparation | 0.067 | 0.03% | 0.066 | 0.01% |

The activation bucket includes convolution SiLU, decay softplus/exp, sigmoid gates, cache-gate addition/multiply and output-gate multiply. HOLA Q/K RMSNorm and mixer output RMSNorm are separate. HOLA read includes fused selection and exact read. Upload includes `Batch::new` host packing, four uploaded tensors and device-created validity mask; it is not pure DMA latency. Readback rows measure only the scalar transfers after their producing device work has completed at the preceding section barrier.

Per-block outer RMSNorm medians (zero-based block index), also present alongside every other block section in the JSON:

| Block | Context 128 mixer / MLP, ms | Context 512 mixer / MLP, ms |
|---|---:|---:|
| 0 | 0.738 / 0.709 | 0.770 / 0.658 |
| 1 | 0.640 / 0.633 | 0.713 / 0.695 |
| 2 | 0.611 / 0.622 | 0.630 / 0.554 |
| 3 | 0.561 / 0.622 | 0.644 / 0.651 |

### Optimization priorities, work and memory limits

The checklist's remaining active components are ordered using **medians of per-trial group sums**, not sums of independently computed medians:

| Remaining component group | Context 128 median ms | Context 512 median ms |
|---|---:|---:|
| All eight mixer projections | 27.55 | 49.48 |
| All RMSNorms, L2/gates and residual adds | 41.68 | 39.76 |
| AdamW + gradient reductions + gradient-norm readback | 41.58 | 39.05 |
| MLP up + SiLU + down | 9.59 | 22.92 |
| Short causal convolutions | 20.13 | 18.46 |
| Embedding + final norm/head + loss | 3.05 | 4.00 |
| Host data preparation + batch construction/upload | 0.92 | 1.49 |

Total backward dominates and remains unassigned to these forward groups. Normalization and optimizer/checks are effectively near-tied here, and the order changes at context 128. This profile does not establish that projection fusion will remove its backward cost, nor justify extrapolation to larger models. Soft top-k and GDN-2 are inactive; checkpoint I/O is excluded, so those checklist items are explicitly unranked rather than given invented zero costs.

CubeCL 0.11 exposes a launch observer: a separate warm complete step issued **5,712 kernels at each context**. Counts include library and custom kernels, not copies, workgroups or command-buffer submissions. No dry run or graph replay was used. The observer is absent from timing trials and its per-kernel histogram is retained.

Sampled maximum backend-pool **reserved** bytes were **104,464,768** at context 128 and **177,314,560** at context 512. These samples include this profiler's two scratch trajectories and are not a single-model live-memory baseline, exact temporary-allocation traffic, or a physical-device peak. Samples are taken after section boundaries and can miss inner-section transients; retained autodiff buffers, pool reuse/padding and the integrated GPU's shared memory preclude stronger claims. Exact temporary-allocation counts/peak memory remain unavailable. Logical model parameter payload is **9,518,272 bytes**; two AdamW moment tensors total **19,036,544 bytes**. Batch host uploads contain **2,048 / 8,192 logical bytes** at contexts 128/512, plus **8 bytes** of scalar readback per step; validity masks are generated on device. These payloads exclude allocator padding and intermediates.

### Evidence and regression outcome

- Completed report, raw samples, both context rankings, all per-block sections and kernel histograms: `artifacts/lm512-kron-fused/profile-1501087-1792857215248038355/report.json`.
- Both instrumentation gates passed: logits and all parameter gradients matched exactly; largest AdamW parameter difference was **7.45e-9** at both contexts.
- `timeout 60s cargo test --workspace`: **22 passed, 0 failed**, 3.22 s wall (1.87 s test build); existing ndarray deprecation/CPU-feature warnings remain visible.
- Initial incremental Vulkan build: 9.34 s (9.49 s wall); formatted rebuild after regenerable experiment-output cleanup: 3.06 s (3.12 s wall). No dependencies, backend feature defaults or numerical tolerances changed.
- The first invocation failed with filesystem `ENOSPC`; its incomplete report remains at `artifacts/lm512-kron-fused/profile-1486818-1792857084668856409/report.json` and is not mixed into the completed measurement. Only regenerable experiment build outputs under `target/` were cleaned; experiment/certification artifacts, corpus and persistent kernel cache were retained. The successful invocation completed in 21.63 s within the configuration's 90-second cooperative deadline and the enclosing 180-second hard timeout.

## Burn automatic kernel fusion

**Decision: evaluated, not adopted.** Burn 0.22 fusion can execute the complete
model with the actual custom GPU operators after fusion-aware registration.
It cuts dispatches by **54.85%**, but this local comparison shows no ordinary-step
speedup, and the unchanged independent Kron/full-model gates fail. No recommended
fusion-backend configuration was added. `configs/lm512-kron-fused.json` still
means custom fused Kron/HOLA on the **non-fusion Burn backend**.

### Backend integration and execution identity

`crates/experiment/src/runtime.rs` creates Burn's runtime-dispatched `Device`;
the existing `backend` selects ndarray, Vulkan or CUDA. Burn 0.22's `fusion`
Cargo feature changes the Cube alias from `CubeBackend` to
`Fusion<CubeBackend>` globally. Enabling the old experiment feature alone was
therefore not a safe, recorded runtime choice, and the custom extensions lacked
the implementation required by that alias.

The experiment feature now forwards through mixers to kernels. Each custom
extension uses Burn's `#[backend_extension(..., Fusion: cfg(...))]` registration
with exact shape/dtype metadata for every forward/backward result, including
Kron's packed output/checkpoints/auxiliaries and HOLA's schedule/expiry/statistics.
The generated deferred operation resolves GPU handles, calls the existing
CubeCL implementation and publishes its GPU outputs. The handwritten autodiff
VJPs remain in use. There is no CPU transfer, tensor-solver replacement or
custom-op fallback. GDN-2's compiled extension is registered too, but its GPU
fusion numerical acceptance was not measured here; it is inactive in this LM.

New config field `fusion` defaults to false and is omitted when false, preserving
historical serialized configs/fingerprints and checkpoint defaults. GPU runtime
creation rejects a config/compiled-feature mismatch **before** creating a device.
Explicit `"fusion": true` is required for a fusion binary; changing this execution
choice invalidates pilot identities and requires a fresh run/budget. The feature
remains opt-in and experimental for reproducing this negative result.

### Controlled full-step measurement

Same reference workload as the whole-model profile: Intel Arc MTL/Vulkan, f32,
B1, four layers/width256, H4/K32/V64, chunk16, HOLA capacity16, seed1, pinned
byte-LM corpus, both custom selectors `fused_gpu`, production safety checks,
clipping and AdamW. The archived resolved configs differ in fusion selection
and measurement bookkeeping only (`artifact_dir`, fresh `cache_dir`; both use
`budget_file: null` and `max_seconds: 300`). Identical context fixture hashes are
recorded in both reports. Each process uses two warmups and five trials per
mode; the fusion process ran before the non-fusion process, without concurrent
builds or other local GPU work. This is one bounded comparison, not an
interleaved cross-process confidence study.

| Context | Burn fusion | Ordinary step median, ms | Issued kernels/step | First ordinary step at this context, ms | Sampled pool reserved, bytes |
|---|---|---:|---:|---:|---:|
| 128 | Off | 217.01 | 5,712 | 2,430.42 | 104,481,024 |
| 128 | On | 245.92 | 2,579 | 7,751.93 | 110,830,464 |
| 512 | Off | 677.97 | 5,712 | 1,964.46 | 177,330,816 |
| 512 | On | 705.15 | 2,579 | 10,845.05 | 177,464,064 |

Fusion was **13.32% slower at128 and 4.01% slower at512** in this comparison.
The previous 228.48/610.84ms non-fusion report remains historical evidence;
it is not substituted for the freshly rerun control above. Ordinary raw samples:
off128 `[217.01,210.96,223.67,232.15,211.03]`, on128
`[243.16,242.76,245.92,262.85,254.17]`; off512
`[675.51,675.81,693.76,677.97,685.77]`, on512
`[742.97,694.92,697.76,705.15,706.64]` ms.

The profiler now records the first complete ordinary step before instrumentation
and warmups. Each comparison started with a separate previously absent cache
directory, without deleting `.cache/cubecl`. First-step times include cache-miss
JIT, fusion planning, optimizer initialization and execution, but exclude model
initialization; context128 may warm shared kernels for512. These are **first-use
costs, not isolated compiler times**. Whole profile process wall times were
41.14s with fusion and23.36s without. Pool samples retain the earlier caveat:
they include scratch trajectories/pool reuse and are not live temporary bytes
or physical-device peak memory. Exact allocation traffic remains unavailable.

The ordinary-step histogram contains 702 elementwise-fused launches and11
reduce-fused launches. It still contains four each of the actual Kron forward
and backward kernels, and four each of HOLA selection/read/Q-backward/KV-backward/
sink-backward: registration did not replace the custom work. Fewer launches did
not make this workload faster; fusion graph/planning and generated-kernel
efficiency are possible contributors, not established causal attribution.

### Numerical gates: retained failures, unchanged tolerances

| Check | Fusion off | Fusion on |
|---|---|---|
| Selected fused Kron independent f64 checks | Pass,12 cases | **Fail**, chained continuation score |
| Selected fused HOLA independent f64 checks | Pass,10 cases | Pass,10 cases |
| `kron-model-check`: CPU logits/every parameter gradient/AdamW, contexts128/512, HOLA off/on | Pass, all4 cases/12 execution combinations | **Fail** at context128/HOLA-on |
| Ordinary versus instrumented profile equivalence | Pass, both contexts | Pass, both contexts |

The strict fused Kron failure is B2/T7/H4/K32/V64/chunk16, combined
output/final-state/score objective, nonzero initial state, zero alpha,
signed/zero corrections, zero norms, padding and resets:
`Kron continuation score[0] 0.35687706 != 0.41130645481807415`,
absolute error0.0544293961 versus unchanged
`2e-5 + 3e-4*abs(reference)`. It occurs after the direct-scan comparisons,
during the existing sliced two-call continuation check.

The existing full-model check passes both HOLA-off execution paths at128
(CPU logits, every parameter gradient and AdamW). It then rejects the
HOLA-on TensorSolver/Tensor path at
`logits[8224]: 0.28834122 != 0.2898132`, absolute error0.0014719665.
It stops there; later fusion combinations/contexts are **not** claimed accepted.
This is a fusion-on regression in a previously passing actual-model scope,
separate from the already documented legacy arbitrary-input solver defect.
Profile instrumentation equivalence is only self-consistency and does not
override either independent failure. No fixture, tolerance, failed gate or
execution path was suppressed. A primitive-only packed-slice/reshape/uneven-time
concatenation probe passed, so a general concat defect is not established.
The precise backend cause of these failures remains unresolved; no speculative
vendor patch or synchronization workaround was introduced.

### Build/regression cost and remaining optimization overlap

The genuinely cold `vulkan,fusion` build used the previously absent
`target/fusion-cold` directory and **exceeded its300s deadline** (300.08s wall).
That lower bound is the cold-build result. A separately bounded cached
continuation completed in111.81s wall; it is not mislabeled a cold build or
summed into a fictitious uninterrupted timing. The normal non-fusion incremental
build completed in3.72s wall (Cargo3.67s). Default workspace tests passed
**24/24** in7.92s wall (test build5.41s); the matching GPU+fusion feature
workspace tests passed **31/31** in9.28s wall (test build6.72s). The latter are
CPU/host-boundary regression tests, not acceptance of the failed GPU numerical
commands. Existing ndarray-deprecation/CPU-feature warnings remain visible.

Fusion already supplies generic elementwise and reduction fusion relevant to
activation/gating/residual work, normalization, optimizer arithmetic/checks,
and parts of projection/MLP epilogues and loss expressions. In section-barrier
measurements, context512 AdamW including clipping fell30.93→15.41ms; this is
not an additive ordinary-step savings estimate, because barriers change fusion
boundaries. It does not merge independent projection matmuls, implement a
multi-tensor optimizer/combined safety reduction, replace the causal-convolution
tap loop or implement dedicated fused cross-entropy. Those checklist items are
not marked complete. Keep their accepted non-fusion ranking for now; if a later
backend fix passes the independent gates, re-profile the ordinary model before
re-ranking or implementing overlapping fusions. No Step7+ work was added.

### Evidence and exact commands

Root evidence directory: `artifacts/burn-fusion-1792857584515`; `off.json` and
`on.json` are the complete resolved comparison inputs, and `evaluation.json`
summarizes the measurements, build outcome and numerical failures. Reports:

- Fusion profile: `on/profile-1608657-1792858029919836961/report.json`.
- Non-fusion profile: `off/profile-1629422-1792858216907022031/report.json`.
- Independent component results: `on/{kron,hola}-reference.json` and
  `off/{kron,hola}-reference.json`; subprocess stdout/stderr and outcomes retained.
- Failed fusion full-model check:
  `on/kron-model-check-1622767-1792858158345523523/report.json`.
- Passed non-fusion full-model check:
  `off/kron-model-check-1660711-1792858479718010052/report.json`.
- Bounded cold build and continuation: `cold-build.json`.

Executed from the workspace root, in this order (the two fusion numerical
failures intentionally exit1 and must not be chained with `&&`):

```sh
# Empty target initially: this command reached the300s limit.
timeout 300s cargo build -p delta-experiment --features vulkan,fusion --target-dir target/fusion-cold
# Cached bounded continuation,111.81s wall:
timeout 300s cargo build -p delta-experiment --features vulkan,fusion --target-dir target/fusion-cold
timeout 300s target/fusion-cold/debug/delta-experiment profile artifacts/burn-fusion-1792857584515/on.json --contexts 128,512 --warmups 2 --trials 5
timeout 300s target/fusion-cold/debug/delta-experiment kron-reference artifacts/burn-fusion-1792857584515/on.json --scan-backend fused_gpu
timeout 300s target/fusion-cold/debug/delta-experiment hola-reference artifacts/burn-fusion-1792857584515/on.json --hola-backend fused_gpu
timeout 30s cargo fmt --all
timeout 300s target/fusion-cold/debug/delta-experiment kron-model-check artifacts/burn-fusion-1792857584515/on.json
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment profile artifacts/burn-fusion-1792857584515/off.json --contexts 128,512 --warmups 2 --trials 5
timeout 300s target/debug/delta-experiment kron-reference artifacts/burn-fusion-1792857584515/off.json --scan-backend fused_gpu
timeout 300s target/debug/delta-experiment hola-reference artifacts/burn-fusion-1792857584515/off.json --hola-backend fused_gpu
timeout 300s cargo test --workspace
timeout 300s cargo test --workspace --features vulkan,fusion --target-dir target/fusion-cold
timeout 300s target/debug/delta-experiment kron-model-check artifacts/burn-fusion-1792857584515/off.json
```

## Mixer projection contraction merging

**Decision: adopt the single merged-input contraction as an explicit execution
choice, not a new parameter layout or historical default.** The reference LM
has seven linears of the same normalized/masked input and an eighth, dependent
output linear. The latter cannot join that contraction: it consumes the
recurrent/HOLA result after gating and normalization. Two library-only designs
were compared against the unchanged separate path:

- `grouped`: Q/K/decay (384 channels), V/output-gate (512), erase/write (8):
  three common-input matrix multiplies per block.
- `merged`: concatenate the seven original weights into `[256,904]`, contract
  once and slice the seven results. The output projection stays separate.

Both retain the original eight `Linear` modules, tensor shapes/names/IDs,
initializers and AdamW slots. Concatenation is differentiable, rebuilt from
current parameter values on every forward; no stale packed-weight cache or
checkpoint migration exists. Activations, masks, convolutions, recurrence,
HOLA, normalization and optimizer arithmetic are unchanged. No custom kernel,
Burn fusion, dependency or Step-7 work was added.

`projection_backend` is omitted when `separate`, preserving old serialized
configs/fingerprints and execution. Nondefault selectors are explicit and
fingerprinted. `configs/lm512-kron-fused.json` is unchanged;
`configs/lm512-projections.json` selects `merged` in a fresh artifact namespace.
Its pilot budget must be generated anew before a quality-training comparison.

### Paired full-step and forward-section measurements

Intel Arc MTL/Vulkan, non-fusion Burn 0.22/CubeCL 0.11, f32, seed 1, pinned
byte-LM corpus, B1/D256/L4/H4/K32/V64/chunk16/cache16, both custom recurrence/HOLA
selectors `fused_gpu`, AdamW learning rate .001/weight decay .01/norm clip 1.
The reusable `projection-benchmark` runs all three variants in one process,
rotating all six order permutations. Each has independent identical-start
ordinary and section-synchronized training trajectories with the same batches
and step indices. Warmups, preflight, initialization/JIT, reporting and
checkpoint I/O are outside samples; real forward, backward, finite-gradient
checks/readbacks, clipping, AdamW and batch preparation/uploads are inside.
No other local GPU workload or build ran during the comparisons.

The expanded comparison uses **two warmups and 21 paired trials**. Forward
projection time sums all eight projection sections across four blocks within
each trial before taking the median; packing/splitting are included. These
section barriers perturb overlap and are not additive ordinary-step savings.

| Context | Execution | Complete step median, ms | Paired saving median, ms | Faster pairs / 21 | Forward projections, ms | Kernels / step | Matmul kernels / step |
|---|---|---:|---:|---:|---:|---:|---:|
| 128 | Separate | 232.734 | — | — | 27.387 | 5,712 | 123 |
| 128 | Grouped | 232.382 | 0.074 | 11 | 15.069 | 5,764 | 75 |
| 128 | Merged | 222.768 | 8.357 | 18 | 9.258 | 5,780 | 51 |
| 512 | Separate | 595.113 | — | — | 49.571 | 5,712 | 123 |
| 512 | Grouped | 583.271 | 13.476 | 15 | 30.304 | 5,764 | 75 |
| 512 | Merged | 580.235 | 20.601 | 18 | 18.070 | 5,780 | 51 |

Merged's ratios of complete-step medians improve **4.28% / 2.50%**, not the
66% / 64% suggested by forward sections. The sixth-to-sixteenth sorted paired
savings (a conservative 97.34% sign-order interval for 21 independent pairs)
are **[1.599,16.677] ms at128** and **[11.141,36.287] ms at512**. They exclude
zero; grouped's corresponding intervals include zero at both contexts.
These are small local gains, not a forecast for CUDA, larger shapes or
mixed precision. Consecutive device measurements can be correlated; the
interval is descriptive evidence under the independent-pair assumption, not
a hardware-noise guarantee.

An earlier independent-process, interleaved **11-trial** characterization is
retained, not blended into the expanded run. At128 separate/grouped/merged
medians were 194.508/195.005/192.919ms; at512
694.609/661.574/667.701ms. Merged paired median savings were 4.953/16.978ms.
The large change in baseline between runs demonstrates why old 611–678ms
profiles cannot serve as the control. The selected candidate has a consistent
positive paired estimate in both sessions; grouped is not recommended.

A **separate 21-trial confirmation with contexts reversed (512,128)** also
passed. At512 separate/grouped/merged medians were
778.354/776.231/752.816ms; merged saved a paired median25.332ms, won19/21
pairs, and had a sign-order interval[18.427,38.244]ms. At128 the medians were
274.099/264.004/258.165ms; merged saved16.094ms, won16/21, and its interval
was[2.773,31.027]ms. Both merged intervals again exclude zero; both grouped
intervals again include zero. Absolute load/clock variation remains large,
but the merged paired improvement survives separate sessions and context order.

### Isolated cost, work and memory

Same device/dtype and actual input/weight shapes, one mixer, resident
gradient-requiring input, two warmups/21 rotating paired trials. Forward
retains autodiff; forward/backward includes a sum-of-squares objective and
all seven weight/input VJPs, but excludes the dependent output projection,
recurrence, optimizer and safety checks. Packing and splitting are timed.

| Context | Execution | Forward median, ms | Forward/backward median, ms |
|---|---|---:|---:|
| 128 | Separate | 4.650 | 10.483 |
| 128 | Grouped | 3.044 | 6.919 |
| 128 | Merged | 1.977 | 5.748 |
| 512 | Separate | 7.676 | 14.283 |
| 512 | Grouped | 4.737 | 10.484 |
| 512 | Merged | 2.805 | 8.017 |

The launch observer is absent from timing trials. Separate warm complete
steps issue 123/75/51 model-wide matmul kernels. Merged removes 72 matmul
launches, but adds 56 slice, 56 slice-assign and 28 fill launches, so **total
launches increase by68**, not decrease. This is a contraction optimization,
not a dispatch-count optimization, and its measured full step earns adoption.

Per layer, transient packed weights contain **925,696 logical bytes**;
the packed output contains **462,848 / 1,851,392 bytes** at128/512. Grouped
has the same total payload split among three buffers. These are additional
packed storage and corresponding adjoints, not new model parameters or
claims of exact live peak; split results/adjoint assembly add work and
backend-dependent materialization. No quadratic token work is introduced.
The model parameter payload remains9,518,272 bytes, with19,036,544 bytes for
two AdamW moments. Maximum sampled pool reservations were220,296,512 /
285,748,224 bytes in this six-trajectory measurement. All variants share
that process pool; these values are **not** per-variant memory peaks or exact
temporary allocation counts. Autodiff retention, pool padding/reuse and
integrated-GPU shared memory prevent a physical-peak interpretation.

### Numerical and historical-checkpoint gates

- Independent scalar-f64 projection dot products and analytic VJPs passed12
  Vulkan cases: separate/grouped/merged, both GDN-2 channelwise and Kron scalar
  gate widths, B2/T7, strided inputs, tail widths and zero values/adjoints.
  Every original weight and input gradient is checked, and parameter IDs must
  remain identical. Largest absolute error was8.91e-7.
- Same-parameter full-model separate/candidate logits, every parameter
  gradient and one AdamW update pass before throughput at both contexts.
- `kron-model-check` passed **all four cases / 12 recurrence-HOLA combinations
  for each candidate**, using separate CPU projections as the independent
  normalized-model reference and all-parameter directional finite differences.
  For merged, maximum CPU logit/gradient/AdamW discrepancies were
  5.126e-6 / 2.652e-6 / 8.829e-6. Tolerances remain2e-5/3e-4, with the existing
  3e-3/0.03 directional-difference tolerance unchanged.
- Selected fused Kron passed12 existing f64 cases; GPU HOLA passed19 existing
  tensor/fused cases. The previously documented legacy arbitrary-input Vulkan
  tensor-solver defect is unchanged, not rerun or treated as accepted.
- The **historical** `artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13`
  loaded using its original resolved config, matched the original config hash
  `c7eca1dace65f65fea8db186a6c0cc5427a320f4bf89f76a4ea0482c86dc42ca`,
  and passed exact saved tensor name/shape/dtype/ID/value validation. Existing
  optimizer state was validated, and next-step model/logit continuation and
  exact canonical AdamW state before/after continuation passed. No old
  checkpoint, config, budget or fingerprint was migrated.
- CPU `reference` passed all6 recurrence,4 cache and6 full-layer cases.
  Workspace tests additionally exercise the independent CPU projection oracle
  and complete GDN-2/Kron layers with every cache policy, strided inputs,
  padding, reset-on-padding, packed resets and tail chunks, checking output,
  input gradients and every parameter gradient.
- Final `cargo test --workspace`: **26 passed, 0 failed** (19experiment,
  7mixers; test build3.33s). Final Vulkan incremental build3.85s; both remain
  within the10s model rebuild /60s test-build budgets. Formatting completed.
  Existing ndarray-deprecation/CPU-feature warnings remain visible. No build
  dependencies, feature defaults or tolerances changed.

### Evidence and exact commands

Evidence root: `artifacts/projection-evaluation`. Characterization report:
`projection-benchmark-1716094-1792858941973124613/report.json`.
Expanded full raw timing, independent projection checks, isolated samples,
kernel histograms, equivalence and fixture hashes:
`projection-benchmark-1740224-1792859145776303861/report.json`.
Reversed-context confirmation:
`projection-benchmark-1782751-1792859442864735753/report.json`.
Candidate model gates:
`grouped/kron-model-check-1758082-1792859303695851642/report.json` and
`merged/kron-model-check-1765036-1792859335337815700/report.json`.
Historical checkpoint gate:
`artifacts/lm512-vulkan/checkpoint-diagnostics/vulkan-seed1-1771773-1792859374015401969/result.json`.

The archived `comparison.json` matches the reference workload except fresh
`artifact_dir`, `budget_file:null` and `max_seconds:300`. `grouped.json` and
`merged.json` additionally select only their respective projection backend
and numerical-report namespace. Commands run from the workspace root:

```sh
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment projection-benchmark artifacts/projection-evaluation/comparison.json --contexts 128,512 --warmups 2 --trials 11
timeout 30s cargo fmt --all
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment projection-benchmark artifacts/projection-evaluation/comparison.json --contexts 128,512 --warmups 2 --trials 21
timeout 300s target/debug/delta-experiment kron-model-check artifacts/projection-evaluation/grouped.json
timeout 300s target/debug/delta-experiment kron-model-check artifacts/projection-evaluation/merged.json
timeout 300s target/debug/delta-experiment kron-reference artifacts/projection-evaluation/comparison.json --scan-backend fused_gpu
timeout 300s target/debug/delta-experiment hola-reference artifacts/projection-evaluation/comparison.json
timeout 300s target/debug/delta-experiment checkpoint-check artifacts/lm512-vulkan/resolved-config.json --arm kronlstm-hola --seed 1 --resume artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13
timeout 300s target/debug/delta-experiment projection-benchmark artifacts/projection-evaluation/comparison.json --contexts 512,128 --warmups 2 --trials 21
timeout 30s cargo fmt --all
timeout 300s cargo test --workspace
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment reference artifacts/projection-evaluation/cpu.json
```

## Normalization and gating fusion

**Decision at B1: evaluated, not adopted.** Two custom CubeCL scopes pass the
numerical gates and reduce dispatches, but neither had a repeatable
beyond-noise complete-step improvement at B1/context512. **Superseded at the
B8 training workload:** `fused_norms_gates` saves 115.26 ms per step alone
and is part of `configs/lm512-optimized.json`; see
[Batch-8 component optimization round](#batch-8-component-optimization-round).
The B1 measurements below are retained unchanged.

### Explicit scopes and preserved arithmetic

`normalization_backend` is omitted when `tensor`, preserving historical
serialized configs/fingerprints and execution selection. Nondefault choices
are explicit and fingerprinted; they require a fresh run/budget:

- `fused_norms`: one row-reduction forward and analytic input VJP for each
  L2/RMSNorm; learned RMS scales have a separate analytic row reduction.
- `fused_norms_gates`: the same norms, plus sigmoid/output multiply inside
  mixer output RMSNorm and channel-bias/head-scale/softplus/exp decay with
  analytic projection, bias and scale gradients.

L2 remains `x/sqrt(sum(x²)+1e-6)`. Outer block/final RMSNorm remains
`x/sqrt(mean(x²)+1e-6)*gamma`; the inherited mixer/HOLA norms retain their
actual **1e-5** epsilon, not a silently substituted1e-6. HOLA keeps its
separate original learned Q/K scales. Gated RMSNorm normalizes
`x*sigmoid(output_gate)`, in that order. Decay preserves beta1 and Burn's
threshold20 softplus identity branch. All original parameter names, shapes,
IDs, initializers and AdamW slots remain intact.

Validity multiplications remain at their original input/output boundaries;
recurrence/HOLA masks and document handling are unchanged. Convolution SiLU,
erase/write sigmoids, HOLA blend gate and residual additions stay on existing
primitives: this comparison tests reduction-heavy versus expanded
normalization/decay/output-gating scopes, not an assertion that every small
primitive merits a custom implementation. The new first-order f32 GPU ops
register through the same Cube/Autodiff/optional-Fusion extension mechanism
as Kron/HOLA; no CPU transfer/fallback, atomics or new backend dependency.
Normalization supports final widths1..=1024 and strided inputs within checked
32-bit addressing.

### Paired integrated measurements

Intel Arc MTL/Vulkan, non-fusion Burn0.22/CubeCL0.11, f32, seed1, pinned byte-LM
corpus, B1/D256/L4/H4/K32/V64/chunk16/cache16, merged projections and both
custom recurrence/HOLA selectors `fused_gpu`. AdamW .001, weight decay .01,
norm clip1. The reusable `normalization-benchmark` shares the projection
benchmark's six-order rotation, independent identical-start ordinary and
section-barrier training trajectories, same batches and step indices.
Ordinary samples include preparation/upload, forward, backward, every
finite-gradient check/readback, clipping and AdamW. Initialization/JIT,
preflight, reporting and checkpoint I/O are excluded. Runs were serial; no
build or other job from this evaluation overlapped timing trials.

First session, **two warmups /21 paired trials**, contexts128 then512:

| Context | Scope | Complete step median, ms | Paired saving median, ms | Faster pairs/21 | Affected forward group, ms | Kernels/step |
|---|---|---:|---:|---:|---:|---:|
| 128 | Tensor | 183.853 | — | — | 37.445 | 5,780 |
| 128 | Fused norms | 179.777 | 3.162 | 15 | 32.841 | 5,036 |
| 128 | Fused norms + gates | 179.045 | 8.875 | 17 | 29.331 | 4,836 |
| 512 | Tensor | 649.793 | — | — | 48.734 | 5,780 |
| 512 | Fused norms | 712.586 | 1.259 | 12 | 48.244 | 5,036 |
| 512 | Fused norms + gates | 647.008 | 1.956 | 13 | 40.976 | 4,836 |

The sixth-to-sixteenth sorted paired-saving intervals (nominal97.34%
sign-order coverage under independent pairs) are:
norms128 **[-2.706,13.387]**, norms+gates128 **[4.699,11.563]**;
norms512 **[-87.356,25.678]**, norms+gates512 **[-18.604,24.125]** ms.
Only the expanded scope at128 excludes zero in this session.

A separate **two-warmup/21-trial confirmation**, contexts reversed512 then128,
does not reproduce a beyond-noise gain:

| Context | Scope | Complete step median, ms | Paired saving median, ms | Faster pairs/21 | Paired interval, ms | Affected forward group, ms |
|---|---|---:|---:|---:|---|---:|
| 512 | Tensor | 632.918 | — | — | — | 51.344 |
| 512 | Fused norms | 617.000 | 8.924 | 15 | [-3.847,31.701] | 46.357 |
| 512 | Fused norms + gates | 627.531 | 18.416 | 12 | [-24.685,56.047] | 42.084 |
| 128 | Tensor | 621.937 | — | — | — | 144.180 |
| 128 | Fused norms | 549.460 | 8.766 | 11 | [-74.089,52.194] | 135.109 |
| 128 | Fused norms + gates | 515.978 | 17.923 | 15 | [-13.696,50.795] | 119.433 |

Absolute drift is substantial, especially confirmation128; medians from
different sessions are not interchangeable controls, and ratios of marginal
medians can be misleading even within a drifting session. All context512
paired intervals contain zero. Fewer launches and the positive paired point
estimates are insufficient to change the recommended path. This is an
**inconclusive integrated-speed/negative adoption result**, not evidence that
the candidates always slow training or that the much larger isolated
backward gain transfers to the model.

### Affected forward sections and isolated cost

First-session section medians below sum all four blocks within each trial.
Triples are tensor / fused norms / fused norms+gates, in ms:

| Forward section | Context128 | Context512 |
|---|---|---|
| L2 normalization and gate activations | 22.173 /20.501 /17.467 | 27.416 /28.228 /22.598 |
| HOLA Q/K RMSNorms | 4.311 /2.921 /2.858 | 6.986 /6.032 /5.490 |
| Mixer output RMSNorm | 2.500 /2.027 /2.053 | 3.855 /3.073 /3.065 |
| Block RMSNorm: mixer | 2.595 /2.033 /1.966 | 3.301 /3.048 /2.696 |
| Block RMSNorm: MLP | 2.594 /1.903 /1.897 | 3.222 /2.610 /2.661 |
| Residual adds (unchanged primitives) | 3.410 /3.380 /3.303 | 4.588 /5.036 /4.778 |

The aggregate forward column is a median of each trial's group sum, not
a sum of section medians. Final RMSNorm is also switched, but remains in
the separately reported final-norm/vocabulary-head section, outside this
group. Section barriers perturb overlap and cannot establish additive
ordinary-step savings. Total sectioned backward128 was114.569/108.874/109.548
ms and512 was461.700/501.645/470.640ms; backward is not attributed per operator.

Resident isolated representative per-block graph: three L2 norms, separate
HOLA Q/K RMSNorms, two outer RMSNorms, gated mixer RMSNorm and decay.
Forward retains autodiff; forward/backward uses a sum-of-squares objective,
all input/parameter VJPs, but no projection/scan/HOLA read/optimizer/checks:

| Context | Scope | Forward median, ms | Forward/backward median, ms |
|---|---|---:|---:|
| 128 | Tensor | 1.420 | 5.096 |
| 128 | Fused norms | 1.043 | 2.884 |
| 128 | Fused norms + gates | 0.817 | 2.645 |
| 512 | Tensor | 1.512 | 6.368 |
| 512 | Fused norms | 1.512 | 3.614 |
| 512 | Fused norms + gates | 1.138 | 3.373 |

Confirmation isolated forward/backward triples were5.240/3.553/2.774ms at128
and17.125/10.918/9.914ms at512. They corroborate an isolated backward
reduction, not a consistent full-step saving; their absolute costs also drift.

### Work, retained storage and numerical gates

- Launches fall5,780→5,036→4,836 (12.87%/16.33%). Custom norms issue33 forward,
  33 input-VJP and21 learned-scale-VJP kernels per step. The expanded scope
  additionally issues4 decay-forward,4 input-VJP and8 parameter-VJP kernels,
  while removing primitive activation/backward work. Matmuls remain51;
  Kron/HOLA custom work is unchanged. Histograms are separate warm complete
  steps; the launch observer is absent from timings.
- Norms retain one f32 inverse per row: **53,760 /215,040 logical bytes** at
  contexts128/512 for this four-layer model including final norm. Gated norms
  recompute sigmoid in VJPs rather than retain another sigmoid-sized buffer.
  They avoid the standalone gated-mixed intermediate
  (131,072 /524,288 logical bytes per block) in the expanded scope.
  Decay retains its output, with no token-sized gradient scratch. Output/input
  gradients and parameter VJPs are allocated at their real tensor sizes;
  inactive L2/ungated results alias handles, with no dummy allocations.
- Shared reductions use at most1KiB/workgroup for the measured width256 norms
  (4KiB at the supported width1024 limit) and256bytes for parameter reductions.
  Work is linear in rows/channels; there is no quadratic token intermediate.
  Model parameter payload remains9,518,272bytes, AdamW moments19,036,544bytes.
- First-session sampled shared pool reservations reached222,408,960 /
  285,775,744bytes; confirmation221,940,992 /284,934,592bytes. These include all
  six scratch trajectories, preflight and allocator reuse, not mode-specific
  live temporaries, exact allocation traffic or physical-device peaks.
- Independent scalar-f64 checks pass **116 Vulkan cases** across tensor and
  both candidates, outputs and all input/gamma/gate/bias/scale gradients.
  Widths3/32/35/64/256, zero inputs/weights/adjoints, saturated and zero gates,
  both original epsilons, softplus threshold/tails, internal/tail padding and
  genuinely strided operator inputs are covered. Largest custom absolute
  error5.80e-5 occurs among epsilon-amplified gradients; the unchanged
  `2e-5 + 3e-4*abs(reference)` criterion passes. The tensor oracle's largest
  absolute error is1.13e-4 under that same criterion.
- Before the confirmation, the fixture was tightened to pre-zero padding
  directly in strided storage, rather than materialize a masked norm input.
  The confirmation and final standalone oracle run check those genuinely
  strided L2/RMS inputs; the initial report and its timing samples are retained.
- Each candidate's `kron-model-check` passes **all4 cases/12 recurrence-HOLA
  combinations**, CPU logits, every parameter gradient and one AdamW update
  plus the original all-parameter directional finite-difference guard.
  Max logits/gradient/AdamW errors: norms4.601e-6/3.010e-6/7.984e-6;
  norms+gates4.524e-6/2.950e-6/7.313e-6. Tolerances remain2e-5/3e-4 and
  directional3e-3/.03.
- Existing selected fused Kron12 cases, GPU tensor/fused HOLA19 cases, and
  CPU reference6 recurrence/4 cache/6 full-layer cases pass. The known legacy
  arbitrary-input GPU TensorSolver stress failure is neither rerun nor hidden.
- Historical `lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` loads using its
  original resolved config and fingerprints, including original tensor
  logical values and AdamW state/continuation, under unchanged checkpoint tolerances.
- `timeout 300s cargo test --workspace`: **28 passed,0 failed** (19 experiment,
  9 mixers),4.04s wall/2.62s test build. Warm Vulkan rebuilds3.53/3.73s;
  existing ndarray/CPU-only-feature warnings remain visible. No dependency,
  profile, feature default or tolerance was changed.
- Optional Fusion-aware registration also builds successfully in the separate
  `target/fusion-cold` tree,296.99s wall within its300s bound. This was a cold
  backend-feature build, not a warm incremental measurement. It makes no new
  numerical/performance acceptance claim for the previously rejected Burn
  automatic-fusion backend.

During implementation, CubeCL rejected a mutable reduction offset initialized
from a comptime expression: generated IR attempted to store into a
`cube.index` constant. Initializing a runtime variable before adding the
comptime half-width fixed the real kernel construction issue; no compiler
patch or vendored dependency. Bounded diagnostic build/dump and return to
normal compilation took82.11/93.31s because enabling/disabling the compiler
dump rebuilt the dependency tree; normal incremental budgets above pass.
The initial shader IR is retained under `compiler-ir/`.

### Evidence and exact commands

Evidence root: `artifacts/normalization-evaluation`.
First paired report: `normalization-benchmark-1996572-1792860690933337929/report.json`;
strict-strided/reversed-context confirmation:
`normalization-benchmark-2021990-1792860876060225689/report.json`.
Both retain configs, fixture hashes, preflight, raw complete-step/section/
isolated samples, histograms and pool samples. Candidate model reports:
`fused_norms/kron-model-check-2060337-1792861092734017979/report.json` and
`fused_norms_gates/kron-model-check-2066073-1792861128945122784/report.json`.
Standalone oracle/reference reports are in that root; the historical
checkpoint command records its fresh diagnostic directory in
`artifacts/lm512-vulkan/checkpoint-check.json`.

`comparison.json` copies the current recommended workload, changing only
measurement bookkeeping (`artifact_dir`, `budget_file:null`, `max_seconds:300`).
Candidate configs add their explicit normalization selector and report namespace;
`cpu.json` explicitly restores tensor/separate CPU execution. Commands from
the workspace root (every command externally bounded; long commands backgrounded):

```sh
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment normalization-reference artifacts/normalization-evaluation/comparison.json
timeout 30s cargo fmt --all
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment normalization-benchmark artifacts/normalization-evaluation/comparison.json --contexts 128,512 --warmups 2 --trials 21
# Tightened strided fixture, then independent reversed-context confirmation:
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment normalization-benchmark artifacts/normalization-evaluation/comparison.json --contexts 512,128 --warmups 2 --trials 21
timeout 300s target/debug/delta-experiment kron-model-check artifacts/normalization-evaluation/fused_norms.json
timeout 300s target/debug/delta-experiment kron-model-check artifacts/normalization-evaluation/fused_norms_gates.json
timeout 300s target/debug/delta-experiment normalization-reference artifacts/normalization-evaluation/comparison.json
timeout 300s target/debug/delta-experiment kron-reference artifacts/normalization-evaluation/comparison.json --scan-backend fused_gpu
timeout 300s target/debug/delta-experiment hola-reference artifacts/normalization-evaluation/comparison.json
timeout 300s target/debug/delta-experiment reference artifacts/normalization-evaluation/cpu.json
timeout 300s target/debug/delta-experiment checkpoint-check artifacts/lm512-vulkan/resolved-config.json --arm kronlstm-hola --seed 1 --resume artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13
timeout 300s cargo test --workspace
timeout 300s cargo build -p delta-experiment --features vulkan,fusion --target-dir target/fusion-cold
```

Compiler-issue investigation only (not timing evidence; the pre-fix invocation
fails and is retained as such):

```sh
timeout 60s env CUBECL_DEBUG_LOG=artifacts/normalization-evaluation/compiler.log target/debug/delta-experiment normalization-reference artifacts/normalization-evaluation/comparison.json
timeout 300s env CUBECL_DEBUG_PLIRON=artifacts/normalization-evaluation/compiler-ir cargo build -p delta-experiment --features vulkan
timeout 60s env CUBECL_DEBUG_PLIRON=artifacts/normalization-evaluation/compiler-ir target/debug/delta-experiment normalization-reference artifacts/normalization-evaluation/comparison.json
```

## Backward attribution

The new bounded `backward-profile` command attributes the previously opaque
backward using **the current recommended workload**, not the historical
separate-projection baseline: `configs/lm512-projections.json`, Intel Arc MTL
(`spirv_32902_32085`), Vulkan, non-fusion Burn0.22/CubeCL0.11, f32, seed1,
B1/D256/L4/H4/K32/V64/chunk16/cache16, merged input projections, fused KronLSTM
and HOLA, tensor normalization/gating, vocabulary257 and pinned byte-LM
training windows. AdamW and its configured safety checks/clipping are unchanged.

Two warmups precede five samples per mode. Forward-only and
forward+backward order alternates; both retain real autodiff graphs and have
outer device synchronization. Fixtures use resident independent
gradient-requiring inputs and the actual layer modules/production operators.
Each of the four layers is executed in isolation and its time contributes
inside the trial before the median: **no single-layer extrapolation**.
Initialization, fixture uploads, warmups/cold first use, corpus loading,
checkpoint/report I/O and launch observation are outside warm samples.
Multioutput groups include sum-of-elements roots in **both** modes; single
outputs use Burn's all-ones output adjoint directly. Scan differentiates its
output, matching the real LM (final state unused, selection scores discrete).

Isolated inputs are seeded resident uniform[-0.5,0.5] leaves; scan Q/K/V
are normalized before detaching into independent leaves, with alpha0.98
and erase/write0.5. Validity is one, resets/document IDs zero, and initial
state is the production zero state. This is a shape-cost attribution with
synthetic adjoints, not a claim to replay the real model's activation values.

An alternating pair of independent, identical-start scratch trajectories
measures the **complete real model step**, including every model part,
loss/backward, all-parameter gradient checks/readback, clipping, AdamW and
production data transfers. One has only ordinary outer barriers and required
readbacks; the other brackets only total backward/check/update boundaries,
not forward sections. No part of the real model is stubbed or removed.
The update/check row is non-autodiff work after actual full-model gradients:
its forward and forward+backward columns both mean complete guarded update,
and its derived backward is zero.

### Ranked synchronized component costs

All times are ms. Rows are ranked by context512 **forward+backward**, not
forward cost; percentages divide by the ordinary complete-step median.
`B = FB − F` is the difference of the two warm medians, not a new backward
kernel timer. Per-layer components aggregate all four layers. Embedding,
final norm/head/loss and the complete update/check group occur once.

| Component | 128 F | 128 FB | 128 B | % step128 | 512 F | 512 FB | 512 B | % step512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fused KronLSTM scan | 18.342 | 82.043 | 63.701 | 47.73% | 66.135 | 324.605 | 258.470 | 62.68% |
| HOLA RMSNorms + fused read | 2.151 | 13.724 | 11.573 | 7.98% | 4.159 | 44.745 | 40.587 | 8.64% |
| MLP: up / SiLU / down | 3.395 | 14.258 | 10.863 | 8.29% | 15.786 | 44.727 | 28.941 | 8.64% |
| Gradient checks + AdamW (non-autodiff) | 33.660 | 33.660 | 0 | 19.58% | 32.414 | 32.414 | 0 | 6.26% |
| Merged input projections | 3.658 | 8.468 | 4.809 | 4.93% | 10.205 | 27.024 | 16.819 | 5.22% |
| Short causal convolutions: q/k/v | 7.504 | 16.179 | 8.674 | 9.41% | 7.551 | 18.146 | 10.595 | 3.50% |
| Mixer output RMSNorm + gate + output projection | 2.007 | 4.701 | 2.695 | 2.73% | 5.276 | 13.643 | 8.367 | 2.63% |
| L2 norms + gate activations | 4.104 | 10.769 | 6.664 | 6.26% | 3.919 | 10.869 | 6.950 | 2.10% |
| Block RMSNorms + residuals | 1.518 | 3.547 | 2.030 | 2.06% | 2.094 | 5.306 | 3.212 | 1.02% |
| Final norm + vocabulary head + loss | 1.223 | 2.115 | 0.892 | 1.23% | 2.098 | 4.311 | 2.213 | 0.83% |
| Mixer masks + document IDs | 1.066 | 1.323 | 0.256 | 0.77% | 1.218 | 1.648 | 0.430 | 0.32% |
| Embedding | 0.527 | 0.634 | 0.107 | 0.37% | 0.680 | 0.564 | −0.116 | 0.11% |

L2/gates includes convolution SiLU, Q/K/V L2, decay softplus/exp,
erase/write sigmoids and the HOLA blend gate. The output gate belongs to
the mixer-output group, not both groups. HOLA includes its original learned
Q/K RMSNorms, exact selection/read and sink VJP. Block RMSNorms/residuals
exclude mixer/MLP work. Final loss includes the actual weighted
log-softmax/gather/reduction, not a synthetic classifier objective.

The small embedding difference at512 is negative because independent median
noise exceeds its tiny VJP cost. It is retained, not clamped to manufacture
coverage. HOLA/MLP differ by only0.019ms at512 and should be treated as tied.

### Backward coverage and the signed isolation/overlap gap

| Measurement | Context128, ms | Context512, ms |
|---|---:|---:|
| Ordinary complete guarded step | 171.902 | 517.853 |
| Real full-step backward, including parameter-gradient extraction | 113.850 | 380.330 |
| Sum of isolated F medians (including non-autodiff update/check work) | 79.156 | 151.534 |
| Sum of isolated FB medians | 191.421 | 528.004 |
| Sum of derived isolated backward | 112.265 | 376.469 |
| Real backward minus attributed backward | **1.585** | **3.860** |
| Attributed / real backward | **98.61%** | **98.98%** |
| Isolated FB sum minus ordinary complete step | **+19.519** | **+10.151** |

Real backward is66.23%/73.44% of the ordinary step at128/512; the scan alone
accounts for55.95%/67.96% of that real backward through its isolated VJP
estimate. The isolated FB sum is111.35%/101.96% of the full step, despite
the1.58/3.86ms unassigned backward residual. This is the measurable
**launch/overlap/isolation gap**, not evidence that a model component was
removed or that the sum predicts a realizable speedup. Independent leaves,
all-ones VJPs versus real adjoints, extra multioutput roots, gradient
accumulation/extraction and changed submission/barrier overlap all contribute;
this measurement does not identify their individual shares. Ordinary data
packing/uploads and loss scalar readback are in the denominator but not
resident component fixtures. Do not clamp the gap to zero, scale component
times to force100%, or combine these absolute costs with earlier sessions.

### Scan launch geometry and occupancy scaling

Every warm isolated scan forward+backward observation issued **six kernels**:
one Kron forward, one Kron backward, three fills and one slice-assign for
adjoint preparation. The observer is absent from timing samples. Each scan
kernel launches exactly **batch×heads workgroups of64 lanes** according to
the checked production launch geometry; this is not a hardware occupancy
counter or a count of submissions/copies. Context does not increase the
workgroup count: each group loops through the entire sequence.

The sweep changes only the isolated scan batch/head counts, keeping
K32/V64/chunk16 and context fixed; the real LM remains B1/H4.
These are per-layer medians, unlike the four-layer scan row above:

| Batch | Heads | Workgroups / scan kernel | 128 F, ms | 128 FB, ms | 512 F, ms | 512 FB, ms | 512 token-heads/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 5.111 | 21.718 | 17.949 | 82.560 | 12,403 |
| 1 | 4 | 4 | 4.936 | 20.824 | 18.526 | 81.697 | 25,068 |
| 1 | 8 | 8 | 4.908 | 20.998 | 17.893 | 82.329 | 49,751 |
| 2 | 2 | 4 | 4.965 | 21.292 | 17.489 | 81.223 | 25,215 |
| 2 | 4 | 8 | 5.218 | 21.699 | 17.782 | 78.386 | 52,254 |
| 2 | 8 | 16 | 4.913 | 20.913 | 17.558 | 95.005 | 86,227 |
| 4 | 2 | 8 | 5.010 | 21.353 | 19.907 | 91.283 | 44,871 |
| 4 | 4 | 16 | 5.313 | 22.666 | 17.831 | 83.351 | 98,283 |
| 4 | 8 | 32 | 5.471 | 22.559 | 18.613 | 85.499 | 191,629 |

At512, B1/H4→B4/H8 increases independent work and workgroups8x while FB
changes81.70→85.50ms (**7.64x throughput**). At128 it changes20.82→22.56ms
(7.38x throughput). This is strong evidence that four workgroups leave
substantial parallel capacity unused, not proof of a particular occupancy
percentage or of a ready-to-use time-parallel recurrence. The prioritized
checklist now calls for a numerically verified more parallel scan/backward
design (chunk summaries/prefix or additional independent state partitions)
with unchanged resets, states, score/output VJPs and tolerances. HOLA and MLP
are the next largest differentiated groups; they are near-tied here.

Logical scan storage per layer at B1/H4:32,768bytes state,
262,144/1,048,576bytes boundary checkpoints at128/512 and524,288bytes
backward chunk replay. Sampled backend-pool reservations after each case
were52,515,456/145,303,424bytes. These include scratch trajectories, all
fixtures and allocator reuse, not physical-device peaks, exact temporary
allocation traffic or a single live production model. The sweep's larger
batch/head shapes scale those logical scan buffers, not the real model.

### Evidence, regression and exact commands

Complete report with resolved config/device, pinned fixture hashes,
raw forward/FB/full-step samples, coverage, all18 scan cases and their
launch histograms:
`artifacts/lm512-projections/backward-profile-2210483-1792862277808205190/report.json`.
The command completed in28.19s internally /28.38s process wall, within the
unchanged configuration's90s cooperative bound and180s hard bound.
No concurrent GPU work or build was used during measurement.

An earlier characterization is retained separately, not blended into these
tables: `backward-profile-2182622-1792862000232393712/report.json` in the same
artifact directory. Its complete steps were245.838/585.955ms at128/512,
backward coverage93.91%/96.71%, scan FB107.425/367.925ms across four layers,
and context512 B1/H4→B4/H8 throughput rose7.71x. Before the final run, two
fixed-size host fixture/context containers became arrays/slices rather than
heap vectors; operator arithmetic was unchanged. The absolute session
variation is much larger than that bookkeeping change, so it is not a
speedup claim. Both sessions establish scan dominance and poor occupancy.

The initial/final Vulkan builds took4.62/2.49s wall (Cargo4.57/2.45s).
Workspace tests pass **30 tests,0 failed** (20 experiment,10 mixers),3.45s
wall in final acceptance (an earlier completed run took3.37s). They include
the independent f64 existing component checks, production full-layer
gradient/update/checkpoint regressions, fresh-graph/input-VJP coverage for
every isolated fixture, and exact signed backward-coverage accounting.
An initial test compile exposed a missing const-generic result annotation
in the new test; adding `Tensor<1>` fixed it without affecting measured code.
Existing ndarray deprecation and CPU-feature warnings remain visible;
dependencies, feature defaults, historical configs/checkpoints and numerical
tolerances are unchanged.

Commands from the workspace root:

```sh
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 180s target/debug/delta-experiment backward-profile configs/lm512-projections.json --contexts 128,512 --warmups 2 --trials 5
timeout 300s cargo test --workspace
```

## Batch scaling

This follow-up varies the **real training batch**, not just the isolated
scan's head/batch launch sweep. `backward-profile` now accepts any positive
`batch_size`, while retaining D256/L4/H4/K32/V64/chunk16/cache16, contexts128/512,
KronLSTM+HOLA, fused scan/read, merged projections, tensor normalization and
the non-fusion GPU requirement. Historical configuration defaults, model
arithmetic, safety checks, checkpoint semantics and tolerances are unchanged.
The general `profile` command already supported positive configured batches.

Measurements below use context512, Intel Arc MTL/Vulkan (`spirv_32902_32085`),
Burn0.22/CubeCL0.11, f32, seed1, vocabulary257 and the same pinned byte-LM
corpus as Backward attribution. Saved inputs are copies of
`configs/lm512-projections.json` with only `batch_size`, the isolated artifact
directory and disabled budget-file output changed; B32 additionally raises
the cooperative reporting deadline from90s to300s, not model work. Two
warmups and five samples per mode, fresh autodiff graphs, alternating F/FB and
ordinary/timed scratch trajectories, all four layers, and unmodified complete guarded
training steps use the methodology above. No concurrent GPU job or build ran
during these measurements.

Full-step tokens/s is `batch_size*512*1000/median_step_ms`, not isolated scan
throughput. It includes batch preparation/upload, loss readback, backward,
finite-gradient checks, clipping and AdamW; initialization, cold first use,
warmups, corpus loading and report I/O are excluded. These are throughput
measurements, not evidence of equal optimization/convergence at different
batches or a quality comparison at equal training tokens.

### Full steps and throughput

| Batch | Tokens/step | Complete step, ms | Tokens/s | Throughput / B1 | Scan FB, ms | Scan % step |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 559.166 | 915.6 | 1.00x | 340.252 | 60.85% |
| 4 | 2,048 | 972.826 | 2105.2 | 2.30x | 337.549 | 34.70% |
| 8 | 4,096 | 1613.667 | 2538.3 | 2.77x | 339.841 | 21.06% |
| 16 | 8,192 | 3187.541 | 2570.0 | 2.81x | 600.675 | 18.84% |
| 32 | 16,384 | 6733.845 | 2433.1 | 2.66x | 1126.998 | 16.74% |

B8 captures most of this device/workload's measured throughput improvement.
B16 is only1.25% faster in tokens/s than B8 while taking1.98x step time;
B32 is4.15% slower in tokens/s than B8. These are five-sample medians from
one sequential sweep, not confidence intervals or a universal optimal batch.
Do not blend these absolute costs with earlier B1 attribution sessions.

### Ranked forward+backward components

FB includes forward and backward; all per-layer rows aggregate four layers
inside each trial before taking the median. Each batch has its own rank,
and percentages use that batch's ordinary complete-step median. The table
display order is B1 for the first table and B8 for the second. The update/
check row remains complete non-autodiff work, not an optimizer VJP.

| Component | B1 rank | B1 FB ms (% step) | B4 rank | B4 FB ms (% step) |
|---|---:|---:|---:|---:|
| Fused KronLSTM scan | 1 | 340.252 (60.85%) | 1 | 337.549 (34.70%) |
| HOLA RMSNorms and fused read | 2 | 59.434 (10.63%) | 2 | 189.034 (19.43%) |
| MLP (up, SiLU, down) | 3 | 47.167 (8.44%) | 3 | 172.640 (17.75%) |
| Gradient checks and AdamW | 4 | 33.968 (6.07%) | 7 | 37.722 (3.88%) |
| Merged input projections | 5 | 29.159 (5.21%) | 4 | 100.630 (10.34%) |
| Short causal convolutions (q/k/v) | 6 | 17.473 (3.12%) | 6 | 37.999 (3.91%) |
| Mixer output RMSNorm, gate and projection | 7 | 13.705 (2.45%) | 5 | 43.190 (4.44%) |
| L2 norms and gate activations | 8 | 11.828 (2.12%) | 8 | 28.602 (2.94%) |
| Block RMSNorms and residuals | 9 | 4.940 (0.88%) | 9 | 17.329 (1.78%) |
| Final norm, vocabulary head and loss | 10 | 4.433 (0.79%) | 10 | 10.015 (1.03%) |
| Mixer masks and document IDs | 11 | 1.735 (0.31%) | 11 | 3.338 (0.34%) |
| Embedding | 12 | 0.677 (0.12%) | 12 | 1.229 (0.13%) |

| Component | B8 rank | B8 FB ms (% step) | B16 rank | B16 FB ms (% step) | B32 rank | B32 FB ms (% step) |
|---|---:|---:|---:|---:|---:|---:|
| HOLA RMSNorms and fused read | 1 | 395.180 (24.49%) | 1 | 825.273 (25.89%) | 1 | 1511.308 (22.44%) |
| MLP (up, SiLU, down) | 2 | 358.638 (22.23%) | 2 | 706.742 (22.17%) | 2 | 1372.736 (20.39%) |
| Fused KronLSTM scan | 3 | 339.841 (21.06%) | 3 | 600.675 (18.84%) | 3 | 1126.998 (16.74%) |
| Merged input projections | 4 | 197.837 (12.26%) | 4 | 393.373 (12.34%) | 4 | 772.238 (11.47%) |
| Mixer output RMSNorm, gate and projection | 5 | 93.036 (5.77%) | 5 | 201.409 (6.32%) | 5 | 385.526 (5.73%) |
| Short causal convolutions (q/k/v) | 6 | 80.529 (4.99%) | 6 | 176.312 (5.53%) | 6 | 343.552 (5.10%) |
| L2 norms and gate activations | 7 | 56.714 (3.51%) | 7 | 145.282 (4.56%) | 7 | 298.139 (4.43%) |
| Block RMSNorms and residuals | 8 | 43.940 (2.72%) | 8 | 90.101 (2.83%) | 8 | 172.113 (2.56%) |
| Gradient checks and AdamW | 9 | 37.967 (2.35%) | 10 | 35.378 (1.11%) | 10 | 45.609 (0.68%) |
| Final norm, vocabulary head and loss | 10 | 19.492 (1.21%) | 9 | 36.054 (1.13%) | 9 | 72.225 (1.07%) |
| Mixer masks and document IDs | 11 | 6.116 (0.38%) | 11 | 15.557 (0.49%) | 11 | 20.921 (0.31%) |
| Embedding | 12 | 3.111 (0.19%) | 12 | 5.370 (0.17%) | 12 | 9.742 (0.14%) |

The original component definitions and synthetic independent-leaf VJPs are
unchanged. In particular, production model batches reset each sequence at its
start; the isolated shape-cost fixtures still use all-valid/no-reset inputs.
No tolerance was loosened to admit larger batches.

| Batch | Real backward, ms | Sum derived isolated backward, ms | Coverage | Real minus isolated backward, ms | Isolated FB sum minus step, ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 416.096 | 402.276 | 96.68% | 13.820 | 5.604 |
| 4 | 731.789 | 712.989 | 97.43% | 18.799 | 6.451 |
| 8 | 1255.497 | 1229.905 | 97.96% | 25.592 | 18.733 |
| 16 | 2512.800 | 2537.195 | 100.97% | -24.396 | 43.986 |
| 32 | 6060.365 | 4845.921 | 79.96% | 1214.444 | -602.738 |

Coverage is signed and is not normalized to100%. Isolated components are
not additive GPU utilization or guaranteed savings. B32 in particular has
**1,214.444ms unassigned backward** in the separately timed real trajectory;
its component table does not completely explain that run. It is not valid
to assign this gap to the scan or silently rescale the rows. Independent
adjoints/roots, gradient accumulation/extraction, barriers, allocator reuse
and changed overlap remain limitations of this method. The B8/B16 priority
decision is supported by97.96%/100.97% coverage, without relying on B32's
less-complete attribution.

### Largest batch and memory evidence

**B32 completed all full steps, component trials and the independent scan
sweep, with exit status0; it is the largest requested/tested batch that fits.**
No OOM boundary was sought beyond32, so this is not a claim that32 is the
hardware's maximum feasible batch. B32 took158.45s internally /158.64s
process wall, within its300s cooperative /280s hard measurement bounds.
B1/B4/B8/B16 took22.33/29.83/44.54/79.37s internally, within their90s
cooperative /180s hard bounds.

| Batch | Step-boundary peak pool, bytes | After entire case, bytes | Checkpoints/layer, MiB | Max process RSS, KiB |
|---:|---:|---:|---:|---:|
| 1 | 166,346,048 | 135,353,088 | 1 | 244,464 |
| 4 | 415,907,328 | 416,371,968 | 4 | 248,396 |
| 8 | 724,188,672 | 724,652,416 | 8 | 248,080 |
| 16 | 1,466,580,480 | 1,467,035,776 | 16 | 250,484 |
| 32 | 2,756,328,960 | 2,756,794,176 | 32 | 253,328 |

The new step memory field samples the backend's `bytes_reserved` only after
existing backward/check/update barriers in measured scratch steps, outside
the component timers. It includes both independent model/optimizer
trajectories and allocator reuse, not a single live production model, exact
temporary allocation traffic or physical-device peak. The after-case
sample additionally includes isolated fixtures and the scan sweep's reuse;
it may be above or below the step sample. At B32 this is about2.57GiB of
sampled pool reservation. Logical scan state/checkpoint/replay per layer is
`B*32KiB / B*1MiB / B*512KiB`, hence1/32/16MiB at B32.

GNU time's maximum RSS is separate host-process evidence, **not GPU peak
memory**, and must not be substituted for pool usage or added to it as an
exact physical total on the shared-memory integrated GPU. All five GNU time
reports show exit0 and0 swaps. A `/proc/meminfo` sample near B32 startup
reported81,853,516KiB total and58,329,636KiB available host memory; this is
a point sample, not a minimum or dedicated GPU budget.

### Batch-dependent optimization decision

**Do not make chunk-parallel KronLSTM the next implementation on B1
evidence alone.** The real B1→B8 scan does8x the batch work at essentially
unchanged four-layer cost340.25→339.84ms, with4→32 source-derived64-lane
workgroups per scan kernel. Its share drops60.85%→21.06%, then18.84% at
B16 and16.74% at B32. It remains material but is no longer the overriding
bottleneck: HOLA and MLP rank above it at all three realistic batches.

Use **B8/context512** as the next cost/throughput comparison baseline:
2,538tokens/s, only1.25% below B16 with roughly half its reservation; B32
does not improve throughput. `OPTIMIZATION.md` now ranks pending active
items HOLA (395.18ms,24.49%), MLP (358.64ms,22.23%), more-parallel scan
(339.84ms,21.06%), convolutions (80.53ms,4.99%), AdamW/checks
(37.97ms,2.35%), then embedding/head/loss (22.60ms,1.40% combined).
The completed projection/normalization evaluations remain completed; their
remaining costs are visible above rather than hidden or counted as free.
At B16/B32 the small head/loss versus optimizer ordering reverses, but not
the three leading pending items.

A hypothetical2x scan-only reduction saves169.92ms at B8, an estimated
1.12x complete-step speedup if all other work/overlap is unchanged,
compared with1.44x at B1. These are Amdahl-style estimates from isolated
costs, **not** measured integrated gains or promises that a new scan can
achieve2x. Retain the scan item below HOLA/MLP, with unchanged numerical
gates and intended-batch integrated acceptance. Mark the batch-size sweep
done, while leaving larger widths and contexts as separate pending work.

### Evidence, regression and exact commands

Each directory below contains the complete resolved config, fixture hash,
raw full-step/component samples, signed coverage, scan dispatch observations,
logical storage and memory fields in `report.json`:

- B1: `artifacts/batch-scaling/b1/backward-profile-2246492-1792862613800215669/`
- B4: `artifacts/batch-scaling/b4/backward-profile-2249397-1792862642666984923/`
- B8: `artifacts/batch-scaling/b8/backward-profile-2253817-1792862686076039313/`
- B16: `artifacts/batch-scaling/b16/backward-profile-2259435-1792862735929527954/`
- B32: `artifacts/batch-scaling/b32/backward-profile-2267586-1792862817757417929/`

Saved inputs are `artifacts/batch-scaling/b{1,4,8,16,32}/config.json`.
Each sibling `process-time.txt` is the corresponding GNU time report.
The Vulkan build completed in1.89s wall (Cargo1.84s).
`cargo test --workspace` **passes31 tests,0 failed** (21 experiment,
10 mixers),3.37s wall (Cargo compile2.03s). Added coverage accepts
positive B1/4/8/16/32 and rejects zero/changed recommended shapes; existing
fresh-graph/component input-VJP and instrumentation logits/all-gradient/
AdamW-equivalence regressions now exercise both B1 and B2. Existing ndarray
deprecation and CPU-feature warnings remain visible.

Exact commands from the workspace root, run sequentially:

```sh
timeout 300s cargo build -p delta-experiment --features vulkan
/usr/bin/time -v -o artifacts/batch-scaling/b1/process-time.txt timeout 180s target/debug/delta-experiment backward-profile artifacts/batch-scaling/b1/config.json --contexts 512 --warmups 2 --trials 5
/usr/bin/time -v -o artifacts/batch-scaling/b4/process-time.txt timeout 180s target/debug/delta-experiment backward-profile artifacts/batch-scaling/b4/config.json --contexts 512 --warmups 2 --trials 5
/usr/bin/time -v -o artifacts/batch-scaling/b8/process-time.txt timeout 180s target/debug/delta-experiment backward-profile artifacts/batch-scaling/b8/config.json --contexts 512 --warmups 2 --trials 5
/usr/bin/time -v -o artifacts/batch-scaling/b16/process-time.txt timeout 180s target/debug/delta-experiment backward-profile artifacts/batch-scaling/b16/config.json --contexts 512 --warmups 2 --trials 5
/usr/bin/time -v -o artifacts/batch-scaling/b32/process-time.txt timeout 280s target/debug/delta-experiment backward-profile artifacts/batch-scaling/b32/config.json --contexts 512 --warmups 2 --trials 5
timeout 300s cargo test --workspace
```

## Batch-8 component optimization round

**Decision: use `configs/lm512-optimized.json` for local Vulkan training at
B8/context512; use `configs/lm512-optimized-autotune.json` only with a
`vulkan,autotune` build.** Every selector is explicit; omitted selectors keep
their historical meaning, so historical configs, fingerprints and checkpoints
are unchanged. Intel Arc MTL/Vulkan, non-fusion Burn 0.22/CubeCL 0.11, f32,
four layers, width 256, H4/K32/V64, chunk 16, HOLA capacity 16, seed 1,
pinned byte-LM corpus. Per-component designs, rejected alternatives, memory
and work accounting are in `docs/optimization/*.md`.

### Complete-step results

All rows are interleaved paired complete guarded steps (forward, loss
readback, backward, gradient checks, clipping, AdamW) in one session, two
warmups, B8/context512. Savings are medians of within-trial differences with
20,000-resample bootstrap 95% intervals; they are not differences of
independent medians.

| Comparison | Control median, ms | Candidate median, ms | Paired saving, ms [95%] | Faster pairs |
|---|---:|---:|---|---:|
| Scan256 + HOLA pairs + fused convolution vs previous recommended path | 1603.758 | 972.667 | 629.232 [620.620, 658.318] | 11/11 |
| … plus fused normalization/gating (`fused_norms_gates`) | 1603.758 | 879.039 | 732.333 [718.443, 763.398] (45.20%) | 11/11 |
| Fused AdamW/checks on top of the above (tensor loss) | 895.674 | 871.897 | 21.081 [13.688, 39.197] | 11/11 |
| Analytic SiLU VJP on top of the above | 876.891 | 863.295 | 22.227 [9.559, 27.882] | 10/11 |
| Guarded matmul autotune + heuristic head/output on top of `lm512-optimized` | 851.795 | 505.618 | 355.617 [341.973, 364.235] | 11/11 |

Rows 3–5 are separate sessions against the optimized base, so their savings
are not summed into the first row's interval. Across sessions, the
previous recommended B8 step (≈1,604 ms, 2,554 tokens/s) falls to ≈506 ms
(≈8,100 tokens/s) with every adopted selector, about 3.2× [INFERENCE:
cross-session ratio of medians, not a paired measurement]. Kernels per step:
5,833 → 4,621 (scan/HOLA/convolution) → 3,677 (+normalization).

Rejected after measurement: padded-tensor convolution (slower at B8),
fused cross-entropy (`fused64`/`fused256`: intervals include zero at B8),
tiled GPU soft top-k as the default (isolated gain does not reach the
complete feedback round), fused-pointwise SiLU (smaller gain than the
analytic VJP alone), and unguarded matmul autotune (wrong vocabulary-head
weight VJPs, see Matrix-multiply efficiency).

### Re-profile after the round

`backward-profile` at B8/context512 on the adopted configurations, two
warmups and five trials (backward coverage 101.8% / 102.1%). Component rows
are four-layer forward+backward medians, as percentages of each ordinary
step:

| Component | `lm512-optimized`, ms | % | `lm512-optimized-autotune`, ms | % |
|---|---:|---:|---:|---:|
| Ordinary complete step | 855.0 | 100 | 512.9 | 100 |
| Merged input projections | 199.4 | 23.3 | 104.6 | 20.4 |
| Fused KronLSTM scan | 103.2 | 12.1 | 101.9 | 19.9 |
| MLP (up, SiLU, down) | 335.4 | 39.2 | 93.5 | 18.2 |
| HOLA RMSNorms and fused read | 59.9 | 7.0 | 57.9 | 11.3 |
| Mixer output RMSNorm, gate and projection | 62.8 | 7.3 | 54.4 | 10.6 |
| L2 norms and gate activations | 52.0 | 6.1 | 47.5 | 9.3 |
| Short causal convolutions | 22.2 | 2.6 | 21.0 | 4.1 |
| Final norm, vocabulary head and loss | 16.9 | 2.0 | 18.2 | 3.5 |
| Block RMSNorms and residuals | 16.4 | 1.9 | 16.6 | 3.2 |
| Gradient checks and AdamW | 15.9 | 1.9 | 12.6 | 2.5 |

Throughput: 4,791 and 7,985 tokens/s, against 2,538 tokens/s for the
previous recommended B8 profile in Batch scaling. With guarded autotune the
step is no longer dominated by one component: projections, scan and MLP are
each about a fifth, so the next round should look at the merged input
projection backward (dX/dW) and the scan before anything smaller.

### Component decisions

| Component | Adopted selector | Evidence (each measured alone) | Document |
|---|---|---|---|
| KronLSTM scan | `scan_backend: fused_parallel256` (4 key shards/value column) | B8/512 isolated F+B 86.28 → 25.79 ms; full step −17.10% [16.52, 18.17] alone | `scan.md` |
| HOLA read backward | `hola_backend: fused_gpu_pairs` (reuse query-VJP edge scalars) | Key/value VJP 60.15 → 3.98 ms per layer at B8/cap16 (pre-isolation session); full step −267.54 ms with cache RMS alone | `hola.md` |
| Short causal convolution | `convolution_backend: fused_gpu` | Four-layer F+B 77.32 → 19.16 ms | `conv-and-norms.md` |
| Normalization/gating | `normalization_backend: fused_norms_gates` | B8 full step −115.26 ms alone; previously not adopted on B1 evidence | `conv-and-norms.md` |
| AdamW, checks, clipping | `update_backend: fused_gpu` | Guarded update 35.12 → 13.69 ms; 5,833 → 3,922 kernels alone | `update-and-loss.md` |
| MLP SiLU | `epilogue_backend: analytic_backward` | Residual adds stay native; no fusion across norms or matmuls | `epilogues.md` |
| Matrix multiplies | guarded autotune, heuristic head and output projection | Rank-3 F/dX/dW preflight on every active GEMM shape before training | `gemm-efficiency.md` |
| Soft top-k selection | `soft_selection_backend: gpu` | B8×512/k16 feedback round 8.64× (V257), 24.44× (V32k); inactive in the causal LM | `soft-topk.md` |
| GDN-2 training | `scan_backend: fused_gdn2` (comparison arm) | Recurrence F+B 245.35 → 93.04 ms at B8/T512; full step with HOLA 2,609.6 → 1,710.9 ms (1.53×), see below | `gdn.md` |

### Correctness gates on the integrated tree

All with unchanged tolerances (2e-5 + 3e-4·|reference|; CPU directional
check 3e-3 + 0.03·|reference|):

- CPU workspace tests: 35 experiment + 7 kernel + 13 mixer pass (one
  ignored Vulkan regression, run separately and passing).
- Guarded-autotune recommended config: `kron-model-check` with the
  `vulkan,autotune` binary passes all four cases (max logits 4.35e-6,
  gradient 3.20e-6, AdamW 1.40e-5).
- Independent float64 references pass for scan256, HOLA pairs, convolution
  (72 cases), normalization (116), update/loss, SiLU and fused GDN-2.
- `kron-model-check --model-scope configured` on the recommended config
  passes all four context128/512 × HOLA off/on cases against the CPU oracle:
  max logits 4.35e-6, max parameter gradient 3.22e-6, max AdamW 1.58e-5.
- Historical `lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` loads and
  continues through `checkpoint-check` with its original fingerprint;
  `update-loss-reference` continues it through the fused optimizer.

### Legacy GPU tensor-solver defect: fixed

The known Vulkan `TensorSolver` key-gradient mismatch
(`artifacts/kron-validation/solver-output_only/kron-reference.json`) was the
missing initial-state erase contribution. Diagnostics showed every small
GEMM, decay and masking primitive correct in isolation; in the erase/solve
chain the VJP of the rank-4 `(X·v).sum_dim(3)` row dot product reached `X`
as exact zeros. A rank-2 flatten fixed the T7 fixture but not chunk size 1,
so the production expression is now a row-wise dot `matmul`. The new
regression `tensor_solver_initial_key_reset_padding_{cpu,vulkan}` fails
before and passes after; all three historical T7 objectives and the default
tensor-solver reference now pass. The exact CubeCL instruction at fault is
not identified; see `tensor-solver.md` for the upstream report text.

### CPU directional check now holds cache membership fixed

HOLA's surprise top-k selection is detached and piecewise constant, so the
analytic gradient is the derivative with cache membership held fixed. The
old CPU directional finite-difference gate recomputed the hard selection for
the ±ε models, and every tested ε (5e-4…4e-3) changed cache membership in
many chunks, so the difference crossed a discontinuity: GDN-2 + HOLA B8/T128
failed at ε = 2e-3 (error 3.45e-3, limit 3.32e-3) and Kron B1 at ε = 4e-3.

The gate now records the unperturbed forward's kept cache slots for every
block and chunk and replays them in both perturbed forwards
(`FixedCacheSelection` in `kron_benchmark.rs`; models without a cache are
unaffected). Epsilon 2e-3 and the tolerance 3e-3 + 0.03·|analytic| are
unchanged. The diagnostic sweep (`gdn-directional-diagnostic`,
`artifacts/integration/directional-diagnostic-fixed.stdout`) reports both:

| Case | Free selection error, ε = 5e-4 / 1e-3 / 2e-3 / 4e-3 | Fixed selection error | Limit |
|---|---|---|---:|
| GDN-2 + HOLA, B8/T128 | 1.62e-2 / 3.35e-3 / 3.45e-3 / 8.8e-4 | 1.0e-5 / 1.0e-5 / 1.09e-4 / 5.0e-5 | 3.32e-3 |
| Kron + HOLA, B1/T128 | 6.2e-4 / 6.2e-4 / 3.96e-3 / 5.57e-3 | 6.2e-4 / 9.5e-5 / 1.43e-4 / 2.4e-5 | 4.20e-3 |
| Kron + HOLA, B8/T128 | 4.38e-3 / 2.24e-3 / 4.5e-4 / 6.9e-4 | 8.9e-5 / 3.88e-4 / 3.28e-4 / 8.9e-5 | 5.03e-3 |

With membership fixed every case passes at every ε by a wide margin, which
confirms the analytic gradients; the earlier failures came from the
selection jumps.

### GDN-2 fused training scan

`bun scripts/benchmark-gdn.ts vulkan BINARY 2 5`, B8, five interleaved
trials after two warmups, every path first checked against the CPU oracle
(logits ≤ 4.72e-6, all parameter gradients ≤ 4.32e-7, AdamW ≤ 4.87e-6;
directional check passes with fixed selection). Median complete step, ms:

| Context | HOLA | Tensor solver | `fused_gdn2` | `fused_gdn2` + fused HOLA |
|---:|---|---:|---:|---:|
| 128 | off | 409.1 | 282.2 | — |
| 128 | on | 521.4 | 402.2 | 352.5 |
| 512 | off | 2,095.5 | 1,296.2 | — |
| 512 | on | 2,609.6 | 1,794.6 | 1,710.9 |

These are medians of five samples, not paired bootstrap intervals; every
trial of the fused path was faster. Isolated recurrence F+B at B8/T512:
245.35 → 93.04 ms. GDN-2 remains a comparison arm; use `fused_gdn2` with a
GPU backend when it is trained. Report:
`artifacts/gdn/benchmark-1792873067810-3535974/`.

### Open findings

- **Run-to-run GPU nondeterminism.** Two identical B1 steps produced one of
  two bitwise `head.weight` results. Recomputed steps are compared at the
  standard tolerance; checkpoint save/restore remains byte-exact.
- BF16 is not reported by this backend; f32 retained
  (`precision-and-shapes.md`). Larger K/V widths and contexts beyond 512 are
  not certified. CUDA remains unmeasured.

### Reproduction

```sh
../build.sh integration                       # vulkan
../build.sh integration --features autotune   # vulkan,autotune
flock /tmp/delta-block-lm-gpu.lock timeout 300s target/debug/delta-experiment optimization-benchmark artifacts/integration/baseline-b8.json --contexts 512 --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 300s target/debug/delta-experiment update-loss-benchmark artifacts/integration/optimized-b8.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 300s target/debug/delta-experiment epilogue-benchmark artifacts/integration/optimized-b8.json --contexts 512 --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 300s bun scripts/benchmark-gemm.ts pair HEURISTIC_BINARY GUARDED_BINARY configs/lm512-optimized.json artifacts/autotune-on-optimized-b8 8 11 heuristic merged_heuristic_output
flock /tmp/delta-block-lm-gpu.lock timeout 300s target/debug/delta-experiment kron-model-check artifacts/integration/recommended-b1.json --model-scope configured
```

Evidence: `artifacts/integration/gates/` (combined benchmark and first
gates), `artifacts/integration/batch2/` (update and SiLU over the optimized
base), `artifacts/integration/final/` (recommended-config gates, solver),
`artifacts/autotune-on-optimized-b8/paired-b8.json`,
`artifacts/kron-validation/solver-fix-verification-2/`.

## Step 7: oHC, wavelet and Lngram

**Status: implemented, causal, verified and optimized; architecture quality not
yet evaluated.** The three components were copied from the
relation-extraction repository (`dre-ohc`, `dre-wavelet`, `dre-lngram`;
byte-identical to `reuse/crates/`) and adapted to the causal byte-LM. Every
field defaults to off, so historical configs, fingerprints and checkpoints
are unchanged. The combined configuration is `configs/lm512-step7.json`
(the batch-8 optimized model plus all three components).

| Component | Configuration | Adapted for the causal LM | Adopted execution path |
|---|---|---|---|
| oHC | `ohc_streams: true` | Per-token four-stream SO(4) mixing; every sublayer (wavelet, Lngram, mixer, MLP) is one oHC stage, plus stream init and readout | `ohc_backend: fused_gate` |
| Wavelet mixer | `wavelet_layers: [0,1,2]` | Centred db10 à trous filter with symmetric extension → left-only taps `t − k·2^level` inside the document (changes the filter phase relative to WNet) | `wavelet_backend: fused_gpu` |
| Lngram | `lngram_layers: [0,1,2]`, orders `[2,3]`, 32 routes × 4 bits, memory dim 8 | Three injections as in the predecessor's three N stages; suffixes stop at the document start; three-tap smoothing → left-only | `lngram_backend: fused_gpu` |

Lngram tables: 4.46 MB per injection, 13.37 MB total, in their own optimizer
group; an AdamW step changes only indexed rows. Parameters: 2.38 M → 7.08 M
with all three components.

### Correctness

Unchanged tolerance 2e-5 + 3e-4·|reference| throughout.

- Independent float64 sequential references (outputs and all gradients,
  padding, resets, short documents, zeros) pass for each component on CPU
  and Vulkan: oHC 93 cases, wavelet 16 reports / 110,000 checks, Lngram 22
  cases with both surrogate gradients and bitwise-deterministic table VJPs.
  Causality perturbation differences are exactly zero for all three.
- `kron-model-check --model-scope configured` on `lm512-step7` passes all four
  context128/512 × HOLA cases against the CPU oracle: max logits 5.05e-6,
  parameter gradients 2.26e-6, AdamW 2.12e-5; directional check ≤ 6.88e-5.
- `lm512-optimized` model check and the historical
  `lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` check-and-continue still
  pass. CPU tests: 39 experiment + 7 kernel + 16 mixer.
- Short learning smokes (200 steps, same seed and budget) decrease loss with
  finite gradients for each component on and off. This is a stability check,
  not a quality result.
- Guarded matmul autotune refuses Step 7 configs until its preflight probes
  the new matrix multiplies.

### Optimization

Each component's adopted path against its copied tensor port, paired
complete guarded steps at B8/context512, 11 trials, measured inside the full
Step 7 configuration (the other two components enabled):

| Component | Tensor port → adopted, paired saving | Faster | Main change |
|---|---|---:|---|
| Wavelet | 755.7 ms (MAD 122.0) | 11/11 | One fixed-filter causal dilated kernel per level with analytic input VJP; launches 6,974 → 2,210 alone |
| Lngram | 67.7 ms [49.3, 74.5] | 10/11 | Shared route-symbol cache, fused suffix addressing + gather, OneBit counterfactual rows gathered on the fly (no expanded counterfactual tensors retained) |
| oHC, write then read+write fusion | 463 ms, then 408 ms (MAD 83 / 62) | 9/11, 9/11 | Fused stream write; fused per-stream RMS and weighted read |
| oHC, gate projection | 263.3 ms (13.0%, MAD 15.5) | 11/11 | Gate projection `4d → 16` fused with its RMS; deterministic tiled weight-gradient partials (no float atomics). A reshaped library GEMM saved 198.5 ms and was not selected |

The first Lngram fused design was not adopted (21-pair interval
[−9.9, 7.6] ms) after an orphaned table VJP in the tensor baseline was
corrected; the refined design above was.

### Cost of the components

| Measurement | `lm512-optimized` | `lm512-step7` |
|---|---:|---:|
| Complete step median, ms (5 trials) | 862.2 / 868.6 | 1,774.8 |
| Tokens/s | 4,751 | 2,308 |
| Sampled pool reservation | 0.56 GB | 1.53 GB |

The Step 7 run sits between two baseline runs in one session; these are
medians, not a paired interval. Incremental cost of each component with the
other two enabled (paired, before the oHC gate fusion): wavelet +507.8 ms,
Lngram +562.7 ms [551.6, 596.5], oHC +800 ms. oHC costs most in
combination because wavelet and Lngram add six more oHC stages (14 per
step). Absolute medians of the same configuration varied by up to 1.5×
between sessions on this integrated GPU, so only within-session paired
differences are used for decisions.

### Retained Step-8 visibility rules (historically exercised; see docs/sbd.md)

Step 7 is causal-only. The retained SBD paths read the clean prefix without
advancing persistent state. These component rules do not select or implement
sparse/within-block attention:

- **oHC** is per token: active-block positions run the same stages on their
  own streams. Mask and candidate embeddings enter through the same stream
  initialization as token embeddings; no stream state persists across
  positions.
- **Wavelet** taps reaching before the block read the clean prefix
  representation; taps inside the block read the active-block
  representation. With left-only taps, in-block communication is
  left-to-right; it is not a bidirectional attention path.
- **Lngram** codes come from hidden states: suffix windows reaching before
  the block use clean-prefix codes, and windows inside the block use
  active-block codes. Smoothing and suffixes never read positions after `t`.
  Table updates come only from the training loss, never from the
  coordination passes.

### Reproduction

```sh
../build.sh integration
flock /tmp/delta-block-lm-gpu.lock timeout 240s target/debug/delta-experiment ohc-reference artifacts/integration/step7/step7-b1.json
flock /tmp/delta-block-lm-gpu.lock timeout 200s target/debug/delta-experiment wavelet-reference artifacts/integration/step7/step7-b1.json
flock /tmp/delta-block-lm-gpu.lock timeout 200s target/debug/delta-experiment lngram-reference artifacts/integration/step7/step7-b1.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s target/debug/delta-experiment kron-model-check artifacts/integration/step7/step7-b1.json --model-scope configured
flock /tmp/delta-block-lm-gpu.lock timeout 600s target/debug/delta-experiment ohc-benchmark artifacts/integration/step7/paired-ohc.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 900s target/debug/delta-experiment wavelet-benchmark artifacts/integration/step7/paired-wavelet.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 900s target/debug/delta-experiment lngram-benchmark artifacts/integration/step7/paired-lngram.json --warmups 2 --trials 11
```

Details: `docs/optimization/{ohc,wavelet,lngram}.md`. Evidence:
`artifacts/integration/step7/{gates,paired,final}/`, `artifacts/step7/`.

## Step 8: recurrent block denoising through HOLA

`sbd.in_block_reads: "hola_read"` appends temporary noisy-block K/V to the frozen clean HOLA read, using one softmax and the existing sink and sigmoid gate. It adds no parameter. Every valid block query sees every valid block key, in both directions. No noisy K/V enters cache selection or the clean path; L=1 remains bit-identical clean next-token prediction. Omitted/explicit `"none"` keeps the previous serialization and fingerprints; `hola_read` changes identity and rejects any no-HOLA layer. `configs/lm512-sbd.json` selects `hola_read`.

### Numerical correctness

Vulkan-enabled workspace tests passed: 46 experiment, 7 kernel and 16 mixer tests, with one explicitly GPU-only test ignored. CPU and Intel Arc Vulkan `sbd-reference` passed 12 model cases: the eight existing mixer × HOLA × Step-7 combinations with `none`, plus `hola_read` in all four combinations having HOLA. The four no-HOLA combinations reject `hola_read`. All tests use the unchanged elementwise tolerance `2e-5 + 3e-4*abs(reference)`.

| Gate | CPU max absolute error | Vulkan max absolute error |
| --- | ---: | ---: |
| Independent f64 noisy-read forward | 3.57e-8 | 3.57e-8 |
| Independent f64 full noisy-read VJP | 5.29e-8 | 1.43e-7 |
| Oracle central finite differences | 9.42e-11 | 9.42e-11 |
| Complete-model logits vs CPU | 0 | 2.76e-7 |
| All-parameter gradients vs CPU | 0 | 1.40e-8 |
| Cached vs reconstructed logits | 1.20e-7 | 0 |
| Cached vs reconstructed gradients | 1.13e-8 | 1.40e-9 |

The maximum noisy-oracle tolerance ratio was 0.00328. Coverage includes all query/cache/clean-chunk/block K/V derivatives, sink and sigmoid gate, padding, empty cache, chunk-aligned starts and a start at position 19 inside chunk 16. Future clean-token perturbations, lengths 1/4/8/16, frozen snapshots and replay passed. Changing an explicitly observed block token changed other positions' logits. Evidence: `artifacts/inblock/{cpu,lm512}/sbd-reference.json`.

### Execution layout

`sbd.noisy_read_backend` is `"tensor"` by default or explicitly `"fused_gpu"`. The CubeCL candidate combines noisy logits, one stable softmax and the value read, with deterministic analytic query/K/V/sink VJPs; the existing tensor graph still gathers cache entries and concatenates the temporary segments. The existing sigmoid gate remains outside the kernel. It accepts f32 K32/V64, at most 16 queries, at most 127 assembled keys and computational chunk length at most 16; unsupported execution is rejected without a CPU alternative.

The fused layout is adopted in `configs/lm512-sbd.json` from the accepted B1/T512 paired evidence below. This decision concerns execution of identical math, not architecture quality. No gather/concatenation-free variant was attempted: the diagnostic times the whole noisy HOLA forward section, not the gather/concatenation share alone.

### Configured-shape timeout and single-row diagnostic

The B8/T512/L8 complete-step benchmark did not emit its first three-mode iteration report before the explicit 3100-second deadline (exit 124, 3100.12 seconds wall time). No paired timing, launch count or sampled-memory result is available at that shape. The responsible mode/component is not established by that timeout; it must not be interpreted as a measured HOLA cost. Evidence: `artifacts/inblock/lm512/sbd-benchmark-timeout.json`. The accepted adoption workload is B1/T512: its diagnostic step times below fit the required two warmups plus 11 pairs within 3000 seconds. B8/T512 was not retried.

One warm complete step and one observed complete step per mode at **B1/T512/L8**, otherwise the canonical four-layer width-256 Step-7 configuration, completed in 267.89 seconds. Each observed step performed **128 noisy block evaluations**, including detached teacher passes, and **512 layer HOLA reads**.

| B1/T512 diagnostic selection | Complete step ms | Noisy HOLA forward ms | Forward-read share | Issued kernels | Sampled pool bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none`, tensor | 43,765.69 | 4,713.70 | 10.77% | 379,689 | 1,545,177,408 |
| `hola_read`, tensor | 45,763.47 | 5,373.14 | 11.74% | 384,040 | 1,558,809,152 |
| `hola_read`, fused | 42,599.33 | 3,909.60 | 9.18% | 368,637 | 1,558,811,008 |

The forward section includes query/key norms, cache gathers, segment assembly, softmax/value read and the existing gate. Synchronization before/after each noisy HOLA read changes overlap; this is one observed step per mode, not the ordinary-step adoption comparison. The percentages exclude HOLA VJPs and are not additive GPU utilization. The high per-row/per-block evaluation and launch counts quantify work in the current SBD training path; no optimization of that loop was attempted, and the B8 timeout's cause remains unassigned. Sampled reservations include the current model/optimizer, base parameters and allocator reuse, not exact live bytes or physical-device peak. Evidence: `artifacts/inblock/b1-t512/sbd-diagnostic.json`.

### Accepted B1/T512 complete-step comparison

The same canonical model, seed 1, B1/T512/L8, two warmups and **11 same-session paired trials** completed in 1608.46 seconds. Mode order alternated forward/reverse, providing AB/BA order for each comparison. Examples, mask seeds and step indices matched across modes; each mode had its own identical-initialization AdamW trajectory.

| B1/T512 selection | Complete-step median ms | Issued kernels in separate warm step | Sampled pool bytes |
| --- | ---: | ---: | ---: |
| `none`, tensor | 44,366.60 | 217,278 | 1,729,734,592 |
| `hola_read`, tensor | 45,184.01 | 219,658 | 1,729,734,912 |
| `hola_read`, fused | 42,034.26 | 211,234 | 1,729,735,808 |

| Paired difference | Median ms | 95% paired bootstrap interval ms | Second path faster |
| --- | ---: | ---: | ---: |
| `none - hola_read(tensor)` | -601.87 | [-1,508.97, -328.46] | 2/11 |
| `hola_read(tensor) - hola_read(fused)` | +2,390.35 | [+1,667.56, +3,176.01] | 10/11 |

Thus adding the user-selected in-block segment costs a paired median 601.87 ms on the tensor path; this is cost accounting, not an architecture-selection test. The mathematically identical fused path saves a paired median 2390.35 ms beyond noise, so `sbd.noisy_read_backend: "fused_gpu"` is adopted. The omitted execution selector remains `tensor`.

Launch accounting uses a separate warm complete step at the same step index in every mode, not the median timed step. Its mask draw evaluated 35 masked blocks twice (teacher plus differentiable pass): 70 noisy block evaluations and 280 layer reads, versus the diagnostic's 128/512. Memory samples include all three resident model/optimizer trajectories and allocator reuse; their small differences are not exact temporary allocation or physical-device peak. Absolute medians are not compared across sessions. Evidence: `artifacts/inblock/b1-t512/sbd-benchmark.json`.

### Small pilot

Pilot cost accounting uses the full four-layer width-256 Step-7 model with B1/T32/L8, seed 1, two warmups and 11 same-session pairs, alternating forward/reverse order. Complete SBD steps include real examples/upload, clean loss, detached teacher pass, masked loss, full backward, scalar readbacks, finite checks, clipping and AdamW; no evaluation or checkpoint is timed.

| B1/T32 selection | Step median ms | Issued kernels in separate warm step | Sampled pool reservation bytes |
| --- | ---: | ---: | ---: |
| `none`, tensor | 727.13 | 18,311 | 415,862,976 |
| `hola_read`, tensor | 729.92 | 18,447 | 415,863,616 |
| `hola_read`, fused | 698.21 | 17,971 | 415,866,688 |

For `none - hola_read(tensor)`, the paired median was +0.84 ms, 95% paired bootstrap interval [-30.90, +26.10] ms; `hola_read` was faster in 6/11 pairs. This cost difference is not distinguishable from noise. For `tensor - fused` with `hola_read` in both paths, the paired median saving was 32.20 ms [16.49, 47.96], faster in 11/11 pairs. Absolute medians are not compared across sessions. Pool samples include all three resident model/optimizer trajectories and allocator reuse, not exact live bytes or physical-device peak. Launch counts are issued CubeCL kernels, not submissions/workgroups. Evidence: `artifacts/inblock/pilot/sbd-benchmark.json`.

Earlier training/generation numbers in [docs/sbd.md](docs/sbd.md) describe the removed separate dense-attention implementation only. No new training study, quality comparison, candidate sweep or Step-7 quality-ablation implementation was performed.

## Step 8: optional prefix block retrieval

`prefix_block_retrieval` implements MoBA coverage in selected HOLA layers:
the causal own block is always included, in addition to top-k completed
same-document earlier blocks. Original K/V share the existing softmax with
cache/chunk entries and sink; cached and current-chunk duplicates are omitted.
Selection is detached. Omission preserves historical serialization and
initialization; enabled identities include `prefix-block-moba-own-v1`.
The explicit fixture is `configs/lm512-step8-retrieval.json`; other
configurations do not enable retrieval.

Cached training and prefix reconstruction compose retrieval with
`sbd.in_block_reads: "hola_read"` through the same noisy-read helper. Its
active tuple contains HOLA-normalized noisy K and raw convolved V in
`[batch,length,head,width]` layout. All segments and the sink share one
softmax, using `prefix_block_retrieval_backend` on selected layers and
`sbd.noisy_read_backend` elsewhere. Both selectors are enabled in the
explicit retrieval config; no extra attention operator or fallback is used.

The parent CPU/GPU tensor and indexed CubeCL implementations have an
independent scalar f64 selection/read/VJP gate, including active K/V.
Semantics, memory formulas, reproduction and measurement evidence are in
[prefix-block-retrieval.md](docs/optimization/prefix-block-retrieval.md).
These measurements establish correctness and execution costs, not quality.

Parent MoBA gates passed 10 CPU and 20 Vulkan scalar-f64 cases, configured
model forward/gradient/AdamW checks, CPU/Vulkan SBD snapshots, all six
historical checkpoint arms and 68 workspace tests (one GPU-sharing test ignored).
At B8/T512 clean AdamW steps, medians were off 1863.765 ms, tensor 2234.488 ms
and fused 2015.634 ms. Fused saved paired 211.244 ms [197.175,218.854] against
tensor (11/11), but retrieval cost paired 139.102 ms versus off.
The T2048 isolated fused forward/VJP medians were 18.417/271.405 ms versus
1713.428/2901.180 ms tensor. Full-SBD B1/T512 off/on medians were
44258.335/43128.916 ms, paired saving 1869.235 ms [1129.419,2438.515] (10/11);
that includes replacing legacy frozen-read tensor assembly with fused
execution, not a quality result or a measurement of both selectors together.

### Joint-selector integration acceptance

The merged implementation supplies the active segment in both cached training
and prefix reconstruction. Retrieval-selected layers use the indexed retrieval
read; remaining layers keep the standalone noisy HOLA read. Both semantic
selectors retain their enabled identities, and omitted selectors retain
historical serialization/fingerprints. No optimization or math change was
introduced in this integration.

All ten executable acceptance commands exited 0. Workspace tests passed
**47 experiment + 7 kernel + 16 mixer tests**, with one GPU-sharing test ignored.
The GPU build completed in 8.84 seconds wall time (8.58 seconds compiling).
All new equivalence comparisons kept `2e-5 + 3e-4*abs(reference)`.

| Integrated gate | Maximum absolute error |
| --- | ---: |
| SBD without retrieval, Vulkan/CPU logits | 2.757e-7 |
| SBD without retrieval, all-parameter gradients | 1.400e-8 |
| SBD retrieval fixture, Vulkan/CPU logits, all 12 cases | 3.502e-7 |
| SBD retrieval fixture, all-parameter gradients, all cases | 2.282e-8 |
| Retrieval plus active reads, Vulkan/CPU logits, four combined cases | 2.980e-7 |
| Retrieval plus active reads, all-parameter gradients | 2.154e-8 |
| Cached/reconstructed logits, CPU, across both fixtures | 1.491e-7 |
| Cached/reconstructed gradients, CPU | 1.130e-8 |
| Cached/reconstructed logits, Vulkan | 0 |
| Cached/reconstructed gradients, Vulkan | 1.397e-9 |
| Noisy-HOLA f64 forward / full VJP | 3.566e-8 / 1.427e-7 |
| Noisy-HOLA oracle central finite differences | 9.415e-11 |
| Prefix f64 forward / full VJP, CPU | 2.417e-8 / 7.047e-7 |
| Prefix f64 forward / full VJP, Vulkan tensor/fused | 2.745e-8 / 2.097e-6 |
| Step-7 configured model logits / gradients / AdamW parameters | 5.052e-6 / 2.265e-6 / 2.119e-5 |
| Retrieval clean configured model logits / gradients / AdamW parameters | 5.737e-6 / 1.342e-6 / 6.546e-6 |
| Historical checkpoint roundtrip and optimizer-continuation logits | 0 |

The largest reported SBD elementwise tolerance ratio was **0.01365**;
the noisy-read f64 oracle ratio was **0.003271**. Prefix f64 checks passed
10 CPU and 20 Vulkan cases, including four/eight active K/V cases.
Both SBD fixtures passed 12 CPU and 12 Vulkan model cases, including
leakage, padding, clean snapshot/replay, observed-token communication and
cached/reconstructed checks. The combined Vulkan cases changed other positions'
logits by at least 0.1374 when an observed block token changed.

Both configured clean model checks passed four T128/T512, HOLA off/on cases.
Their existing directional finite-difference checks also passed unchanged:
maximum discrepancies 6.885e-5 (Step 7) and 4.130e-4 (retrieval), under their
separate pre-existing `0.003 + 0.03*abs(reference)` limit. The Step-7 AdamW
maximum above exceeds the absolute component alone but passed the unchanged
per-element relative criterion. Historical `kronlstm-hola`, seed 1,
checkpoint 13 passed its stricter `1e-5 + 1e-4*abs(reference)` comparisons;
weights and optimizer semantic state matched exactly after continuation.

Reproduction configs, exact timeout/lock-wrapped command arrays and numerical
summaries are in `artifacts/integration-step8/{commands,summary}.json`.
`sbd-{cpu,vulkan}.json` derives from `configs/lm512-sbd.json`;
`retrieval-{cpu,vulkan}.json` derives from `configs/lm512-step8-retrieval.json`.
Vulkan copies change only the artifact directory; CPU copies additionally select
the tensor CPU execution paths. `step7-clean.json` changes only the artifact
directory from `artifacts/integration/step7/step7-b1.json`;
`retrieval-clean.json` omits SBD from the retrieval config. Each runs
`kron-model-check --model-scope configured`. The historical check uses the
original resolved config and resume path, with diagnostic output isolated here.

The execution-only context-32, batch-1 retrieval smoke completed exactly
**two steps / 64 tokens**. Its generated model/optimizer payloads and the
historical diagnostic's scratch checkpoint payloads were deleted; final reports
and smoke state/metrics remain. No training study, ablation or sweep was run.
The conflict-marker scan returned no matches.

An initial test invocation omitted `CARGO_TARGET_DIR`; it was cancelled and
its newly created local target removed. All passing builds/tests used the
required shared target. Existing Burn ndarray deprecation warnings remain;
there were no acceptance failures.

## SBD block-axis execution batching

The execution-only selector `sbd.block_execution: batched` stacks selected
multi-token blocks from every row along the batch axis. Clean snapshots are
prepared for all rows together; frozen states, HOLA/retrieval starts and local
histories remain specific to each block. Soft feedback retains the detached
teacher and differentiable second pass. Padding uses the maximum configured
length, with inactive keys and zero loss weights. Singleton targets still use
clean causal logits. No parameter, model equation, training study or quality
comparison was added.

The omitted/default selector remains `per_block` and is excluded from
serialization, preserving historical fingerprints. A pinned regression checks
the pre-change length/eta/token RNG sequence, including masks for blocks that
are subsequently skipped. Both explicit execution choices use that sampler.

### Correctness

All comparisons retained `2e-5 + 3e-4*abs(reference)`.
Each backend passed 24 no-retrieval and 40 retrieval-inclusive
architecture/feedback cases: both mixers, HOLA off/on, Step-7 off/on, valid
in-block-read selections, soft feedback/hard mask, multiple rows and starts,
lengths 1/4/8/16, short/padded tails, all-observed draws and actual completed-block
retrieval selection. Loss comparisons check every parameter-gradient element,
not a sample or an aggregate direction.

| Batched versus per-block maximum absolute error | CPU | Vulkan |
| --- | ---: | ---: |
| Noisy/teacher logits | 1.63913e-7 | 4.47035e-8 |
| Total SBD loss | 1.90735e-6 | 1.90735e-6 |
| Every parameter gradient | 3.12925e-7 | 5.96047e-8 |
| Largest elementwise tolerance ratio | 0.008266 | 0.001374 |

The existing f64 noisy-HOLA and prefix-read gates, cached/reconstructed SBD
comparisons, clean configured Step-7/retrieval model gates and historical
checkpoint/optimizer continuation all passed. Workspace tests passed **49
experiment + 7 kernel + 16 mixer tests**, with one GPU-sharing test ignored.
The final GPU build took 8.47 seconds wall time; tests took 11.59 seconds.

The first B8 diagnostic failed at the existing noisy-HOLA key-VJP's
65,535-workgroup assertion, not from memory exhaustion. The existing kernel now
uses the repository's multidimensional dispatch pattern with surplus-workgroup
returns; forward and analytic VJP arithmetic are unchanged. A 67,200-workgroup
GPU fixture passed the independent scalar-f64 forward/full-VJP oracle with
maximum absolute error 3.43394e-7 and tolerance ratio 0.004088. No new read
operator or chunked alternative was needed.

### Paired complete training steps

Intel Arc integrated GPU, Vulkan, Mesa 26.2.4, f32; width 256, four layers,
H4/K32/V64, computational chunk 16, HOLA capacity 16, full Step-7 components,
L8 soft feedback, seed 1, byte vocabulary 257. Both arms use the same model,
examples, mask seeds, precision and optimizer settings. Each row below is a
separate same-process comparison with **two warmups and 11 interleaved pairs**;
order alternates, with independent AdamW trajectories from identical weights.
The measured step includes host examples/upload, clean preparation, both noisy
passes, loss/scalar readback, backward, finite checks, clipping and AdamW.
There is no evaluation or checkpoint I/O. Final-code sessions held the GPU lock
for 560.87 and 567.51 seconds respectively.

| B1/T512 workload | Per-block median | Batched median | Paired saving | 95% paired bootstrap interval | Batched faster |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOLA active-block reads, retrieval absent | 42,144.517 ms | 1,617.483 ms | 40,568.800 ms | [35,569.793, 44,211.076] ms | 11/11 |
| Same, with configured layer-3 MoBA retrieval | 44,178.291 ms | 1,819.748 ms | 42,358.543 ms | [39,231.935, 43,705.788] ms | 11/11 |

The median ratios are 26.06× and 24.28×. Paired savings are medians of
within-pair differences, not differences of marginal medians. These rows do
not measure retrieval's architecture cost against off; only execution differs
within each row. Both intervals exclude zero, so **both
`configs/lm512-sbd.json` and `configs/lm512-step8-retrieval.json` adopt explicit
`batched` execution**. Omitted configurations keep `per_block`.

| Work measurement | Per-block | Batched |
| --- | ---: | ---: |
| Issued launches, B1/T512 without retrieval | 211,269 | 24,242 |
| Issued launches, B1/T512 with retrieval | 209,934 | 24,208 |

Launch counts come from one additional complete step at the same step index
and mask seed in each arm. Masks vary with the step seed, so these counts are
not the earlier diagnostic's approximately 370k-launch draw. Sampled maximum
pool reservation was 1,931,060,160 bytes in the no-retrieval session and
2,200,545,536 bytes in the retrieval session; each includes both resident
runners and allocator reuse, not a per-arm live-activation peak.

### B8/T512 diagnostic

One batched-only no-retrieval step after one warm step completed in
**8,167.400 ms**, issuing **27,145 launches**, with **8,818,099,392 bytes**
sampled maximum pool reservation. The warm step took 9,468.817 ms.
B8 fits without block chunking. This single diagnostic establishes executable
scale, not a paired B8 speedup; sampled reservation is not exact live bytes or
physical-device peak. Retrieval-enabled B8 timing was not measured.

All small final reports, resolved inputs, lock/timeout command records and the
Bun reproduction runner are under `artifacts/batching/`. The first dispatch
failure is recorded in `b8-first-attempt.json`; pre-dispatch-fix timing is
separately retained in `before-large-grid.json` and is not the adoption
evidence above. Historical diagnostic model/optimizer payloads were removed;
numerical reports remain. No training study, ablation or sweep was run.


## Step 9: Codpula joint block output

Implemented 2026-10-25 in the `codpula` workspace. The selected model is a homogeneous
32-state HMM over actual SBD blocks of length at most 16, with full 257-symbol
emissions, shared positive log rows, and one final-feature mean-pooled 8×8 gate
grid (group width four) per block. The existing vocabulary projection supplies
token potentials. Conditional joint noisy NLL is added to the unchanged clean
loss; circuit rows use the recorded dual-flow/Anemone update, separate
bias-corrected momentum 0.9 and constant alpha 0.05. Seeded joint FFBS proposals
and dependence commitment use threshold 0.05, all unresolved positions, a
minimum-score progress fallback and at most L passes.

The full equations, decisions and current-source reuse audit are in
[`docs/codpula.md`](docs/codpula.md). Omitted/explicit factorized serialization,
fingerprints, initialization and old checkpoint behavior remain; L1 is still the
exact clean causal bypass. `configs/lm512-codpula.json` enables Codpula with
batched blocks and dependence commitment. Its circuit execution stays **tensor**.

### Numerical and continuation acceptance

The tolerance remains `2e-5 + 3e-4*abs(ref)`, unchanged.

| Gate | Max absolute error | Worst tolerance ratio |
| --- | ---: | ---: |
| CPU circuit/f64/enumeration and VJP checks | 6.385e-6 | 0.1750 |
| Vulkan circuit/f64 checks, tensor and fused | 1.2103e-5 | 0.1512 |
| Fused operations/full VJPs within that report | 8.801e-7 | 0.02940 |
| CPU complete-model matrix | 1.908e-6 | 0.09537 |
| Fused Vulkan complete-model matrix versus CPU | 2.385e-6 | 0.04744 |

Checks cover partition/evidence normalization, marginals, joint NLL and full
VJPs, separate positive/negative flows, valid Eq. (9) updates, statistical joint
samples plus exact indexed-uniform FFBS, cavity versus query scores, impossible
evidence/all-zero posterior rejection, dirty padding and noncontiguous inputs.
Model checks cover GDN-2/KronLSTM with ordinary and Step-7/retrieval features,
all SBD paths, teacher detachment, observed-feature gradients, absorbing
commitments and unchanged clean caches/device RNG. New checkpoint continuation
is exact through two additional complete neural/circuit steps, including EMA,
integer time and canonical AdamW state.

The earlier valid causal checkpoint `artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13`
still loads with saved identity
`c7eca1dace65f65fea8db186a6c0cc5427a320f4bf89f76a4ea0482c86dc42ca`.
The obsolete dense-attention Step-8 payload remains rejected, as it was before
this turn. Workspace tests pass **51 experiment + 7 kernel + 16 mixer tests**,
one GPU-sharing test explicitly ignored. Unchanged SBD and block-batching
reference matrices pass on CPU and Vulkan.

A new forward-only fused-query defect was caught before acceptance: CubeCL's
compile-time early `terminate!()` did not exclude the later posterior suffix,
which could overwrite aliased inputs. Positive compile-time inclusion of the
entire suffix fixes it; untracked repeated-query/model-inference regressions pass.

### Paired complete-step cost, not quality

Intel Arc Graphics (MTL), Mesa 26.2.4 Vulkan f32, no autotune/Burn fusion,
B1/T512/L8, D256/four layers with oHC/wavelet/Lngram/HOLA.
Two warmups and **11 interleaved samples per mode**, alternating order, in one
GPU-locked session. Initial common weights and step-specific examples/masks
match; optimizer trajectories are separate. Timings include uploads, clean and
detached teacher passes, noisy loss, backward, checks, clipping, AdamW and the
circuit update. They exclude evaluation, checkpoint I/O and cold compilation.

| Complete optimizer step | Median ms | Launches |
| --- | ---: | ---: |
| Factorized cost reference | 1522.502 | 24,242 |
| Codpula tensor, selected | 1697.014 | 27,842 |
| Codpula fused, explicit opt-in | 1631.830 | 25,578 |

Tensor Codpula adds a paired median **154.355 ms [109.492,187.927]** versus
factorized; the median-time ratio is +11.46%. Fused adds
**117.456 ms [57.658,149.567]**, median-time ratio +7.18%.

Fused versus tensor saves a paired median **60.951 ms**, but the 95% paired
bootstrap interval **[-23.545,84.668] ms includes zero**, with 8/11 faster.
**Fused is not adopted by default.** The isolated circuit is genuinely faster:
tensor/fused medians **80.350/61.654 ms**, **2716/1044 launches**, paired saving
**20.025 ms [16.577,23.311]**, 11/11 faster. That isolated gain is not sufficient
evidence for a complete-step adoption.

Sampled pool reservation is approximately **1,697,281,920 bytes** with all three
models/optimizers resident; it is not a per-path or exact live-memory peak.
The selected tensor B8/T512 path's single post-warm diagnostic is
**8000.071 ms, 30,745 launches, 8,826,550,976 bytes**. It fits without block-axis
subdivision. The earlier factorized ~8.2 s B8 figure is unpaired historical
context, not evidence that the added joint layer makes B8 faster.

### Current predecessor salvage and tiny smoke

Current predecessor inference/support/CDF/indexed-uniform, dependence
commitment and rollout/detachment tests were salvaged and adapted, rather than
treating the older `reuse/` copy as current. Relation fields, union alternatives,
all-round loss averaging, destination gates and posterior-temperature decoding
were not imported. No matching dual-flow optimizer exists there; its trainer
uses AdamW. Exact source hashes/ranges are in `reuse/PROVENANCE.json`.

Source inspection found an all-zero-row NaN in predecessor `decode.rs:56-61`:
`-infinity-(-infinity)` during posterior normalization. That decoder was not
ported; ours has an explicit zero-support sampling regression. The inventory
document in the main workspace was left untouched.

Only **96 bytes / three optimizer steps**, B1/context32/seed1, were trained.
Checkpoint CLI evaluation and generation pass: clean NLL 5.447807, joint block
NLL per target 4.463265 on one 32-token window, and eight finalized/replayed
tokens in one dependence pass. These are smoke values, not language-quality
evidence. Model/optimizer smoke and scratch payloads were deleted afterward;
small JSON reports/state/hashes remain. No study, ablation, sweep or quality
comparison was run.

Evidence: `artifacts/codpula/`, particularly `cpu-tests.txt`,
`{cpu,vulkan-fused}/codpula-model-reference.json`, `codpula-reference.json`,
`{factorized-cpu,factorized-vulkan}/sbd-reference.json`,
`b1/codpula-benchmark.json`, `b1/codpula-isolated.json`,
`codpula-diagnostic.json`, `smoke/`, `adoption.json` and
`checkpoint-cleanup.json`.
