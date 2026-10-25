# Small causal experiments (plan steps 1, 3, 5, 6 and 7)

## Measured optimized local path

Use `configs/lm512-kron-fused.json` for explicit fused KronLSTM and HOLA on the supported local Vulkan workload. Historical omitted selectors remain unchanged. The paired context-512, four-layer/width-256 guarded AdamW benchmark measured 2,322.06 ms/step for tensor/tensor versus 601.02 ms with both fused (3.86x). See the root `RESULTS.md` for scope, samples and limits; isolated operator gains are not whole-model gains.

For new runs of that supported local shape, `configs/lm512-projections.json`
additionally selects identity-preserving merged mixer input projections.
Paired, interleaved comparisons show a small repeatable full-step
gain, not the much larger forward-only reduction; see “Mixer projection
contraction merging” in `RESULTS.md`. The old config is deliberately unchanged.
Generate a fresh pilot budget for the new configuration rather than reusing
the old checkpoint/run fingerprint.

This remains the single recommended current config. The two custom
normalization/gating scopes were numerically checked but **not adopted**:
context-512 paired full-step intervals include zero in both sessions despite
fewer kernels and faster isolated backward. The config therefore deliberately
omits `normalization_backend`, retaining tensor normalization/gating. See
“Normalization and gating fusion” in `RESULTS.md` for both scopes and raw
evidence; the opt-in selectors exist for reproduction, not as a third
recommended configuration.

The expanded checks expose a legacy GPU TensorSolver key-gradient error on a short nonzero-initial-state stress case. CPU and fused Kron pass. The global `reference` / strict `kron-benchmark` intentionally still report that failure. Current optimized checks and the separately scoped model benchmark are:

```sh
timeout 120s target/debug/delta-experiment kron-reference configs/reference-vulkan.json --scan-backend fused_gpu
timeout 120s target/debug/delta-experiment hola-reference configs/reference-vulkan.json
timeout 120s target/debug/delta-experiment benchmark configs/lm512-kron-fused.json
# On the affected backend this exits 1 and creates the retained report:
timeout 120s target/debug/delta-experiment kron-reference configs/reference-vulkan.json --scan-backend tensor_solver --tokens 7 --objective output_only
timeout 400s target/debug/delta-experiment kron-model-benchmark configs/kron-benchmark.json --warmups 2 --trials 5 --legacy-stress-report artifacts/reference-vulkan/kron-reference.json
```

The scoped model benchmark requires and preserves the real failed legacy report, then checks its exact normalized workloads against CPU logits, every parameter gradient and AdamW updates, with a directional finite-difference sanity check. It never certifies arbitrary legacy-solver inputs. Checkpoints are timed separately; training-step times include production loss and finite-gradient scalar readbacks.

Do not chain the expected failing diagnostic with `&&`. If the legacy checks instead pass on your backend/version, use the strict all-path benchmark; do not construct fake failed evidence.

Run commands from the workspace root. The reference and soft-feedback commands are numerical checks, not training results. Step 7 adds opt-in causal Lngram injections; no SBD/block-denoising training is included.

## Build and local device policy

Rust 1.95+, clang and mold are required on the local x86_64 host. The root profiles disable dev/test debug information, optimize external dev dependencies, and explicitly leave workspace crates unoptimized. CPU is the default ndarray backend. Vulkan, CUDA, fusion and autotune are separate opt-ins; CPU builds do not compile custom GPU kernels. The program never connects to a remote host. CUDA support is only an opt-in build path, not an instruction to use Spark.

```sh
timeout 120s cargo build -p delta-experiment
timeout 90s target/debug/delta-experiment reference configs/reference-cpu.json
timeout 30s target/debug/delta-experiment soft configs/reference-cpu.json
# Same numerical fixture, with local Intel-adapter identity in the result:
timeout 120s cargo build -p delta-experiment --features vulkan
timeout 90s target/debug/delta-experiment reference configs/reference-vulkan.json
```

A timeout is a result; do not replace it with an unbounded build. Every GPU-enabled invocation installs a persistent CubeCL compilation cache under the configured `cache_dir` (default `.cache/cubecl`) before creating a device. Vulkan requires the actual reported adapter identity to contain Intel. Device identity and the resolved configuration are saved in JSON.

`reference` delegates to the mixer crate's float64/CPU/GPU fixture; its report identifies the algorithms actually exercised. Historical and omitted selectors remain `scan_backend: "tensor_solver"` and `hola_backend: "tensor"`. Explicit `scan_backend: "fused_gpu"` selects KronLSTM's custom scan; explicit `hola_backend: "fused_gpu"` independently selects custom HOLA. Unsupported devices/shapes fail without fallback. Selecting the fused scan for GDN-2 fails. Configuration fingerprints include nondefault selectors; omitted historical defaults serialize identically so old solver checkpoints remain readable. Training metrics record the actual arm's execution algorithm and custom-kernel use, not just its device. Do not change a selector inside an existing run directory.

Burn automatic fusion is a separate execution choice from the custom `fused_gpu`
operators. The `fusion` Cargo feature changes Burn 0.22's Cube backend globally
to `Fusion<CubeBackend>`; it is not a runtime device switch. GPU configurations
must explicitly set `"fusion": true` when using a fusion-enabled binary, and
must omit it (or set it to false) with a non-fusion binary. Mismatches fail before
device creation. CPU execution is unaffected. The false default is omitted from
serialization, preserving historical fingerprints and checkpoint loading.
Use a fresh artifact directory and regenerate any pilot budget when selecting
fusion. `delta-experiment/fusion` forwards through mixers to the kernels crate.

The custom KronLSTM, HOLA and GDN-2 extensions register deferred opaque
operations with shape/dtype metadata for every forward/backward result. Burn
resolves their GPU input handles, executes the existing CubeCL kernels and
publishes the output handles on the same device. Their handwritten VJPs remain
attached through autodiff; neither CPU transfers nor tensor-solver fallbacks
are introduced. Opaque custom kernels are scheduling boundaries, not kernels
that Burn automatically merges with adjacent operations.

The measured Burn 0.22 fusion combination is **not accepted or recommended**:
it reduced dispatches but lost ordinary-step performance and failed the
unchanged Kron continuation and full-model CPU gates. Keep using the historical
non-fusion binary with `configs/lm512-kron-fused.json`. The opt-in remains
available for reproducing the negative result; see `RESULTS.md`, “Burn automatic
kernel fusion”, for exact configs, commands and numerical failures.

Burn's `autotune` Cargo feature is independently explicit: GPU configs must
set `"autotune": true` with `--features vulkan,autotune`, and omit it (or use
false) with `--features vulkan`. A mismatch fails before device creation.
False is omitted from serialized configs, preserving historical identities.
Both compiled kernels and tuning decisions persist in the configured CubeCL
cache/environment; decisions are also logged to
`ARTIFACT_DIR/autotune-decisions.jsonl` when enabled. Tuning changes all Burn
autotuned operations, not only GEMMs. Use fresh run directories/budgets.

The accepted matrix-efficiency option is `configs/lm512-gemm-guarded.json`
with `--features vulkan,autotune`, `head_backend: "heuristic"` and
`projection_backend: "merged_heuristic_output"`, plus the mandatory startup
probe below. On the isolated D256/L4/context512 model, eleven interleaved pairs
reduce B8 step1619.76→1188.30ms and B1 560.21→510.47ms beyond paired noise.
This does **not** adopt bare autotune or change historical defaults; legacy
solver/HOLA tensor paths remain unsupported. See
`docs/optimization/gemm-efficiency.md` for every shape, source/binary proof,
startup costs and exact commands. Larger K/V/context configurations still need
their own numerical/performance acceptance.

Autotuned model execution is restricted to the local non-fusion byte-LM,
`scan_backend: "fused_gpu"`, `hola_backend: "fused_gpu"`, merged inputs and
tensor normalization/gating. Unsupported combinations fail before device creation.
`projection_backend: "merged_heuristic_output"` retains the merged input
contraction but explicitly pins the dependent output projection's forward and
both VJPs to Burn's non-autotuned Cube heuristic. `head_backend: "heuristic"`
does the same for the unchanged 257-column vocabulary parameter. The alternative
`head_backend: "padded320"` appends constant zero weight columns, slices the real
257 logits before the loss, and never adds trained parameters or optimizer state.
Omitted head selection is historical `native`; omitted projection selection is
historical `separate`. No parameter name, shape or ID changes in any mode.

Burn's private matmul tuner exposes no global candidate exclusion. Therefore,
before any autotuned training/step profiling, a fail-closed preflight probes all
five active linear families with the actual configured batch/context/width:
real rank-3 forwards plus input and parameter VJPs on deterministic representative
tensors. It checks four sampled entries of **each** F/dX/dW result against scalar
f64 dots at the existing `2e-5 + 3e-4*abs(reference)` tolerance; this is not a
whole-output f64 proof. The separate shape sweep and all-output/all-parameter
CPU model gates remain acceptance requirements. A failed sampled probe
refuses training before any real optimizer step and retains `gemm-preflight.json`.
This is not a whitelist of known batch sizes: new shapes are probed too.
Certificates under `CACHE_DIR/gemm-verdicts/` are bound to exact shapes and
execution, actual Vulkan adapter/driver information, source/dependency hashes,
and a digest of the persistent matmul tune entries. All selections are warmed
before that digest is read, resolving candidate-list checksums. Even a matching
cached certificate rechecks fresh device values; only the deterministic f64
expectations are reused. Startup/preflight cost is separate from warm step time.
Commands that internally change training contraction families reject autotune
rather than execute unprobed GEMMs.

### Matrix-multiply characterization

`gemm-micro CONFIG --warmups 2 --trials 7` is bounded by `max_seconds` (at most
290s) and an external process timeout. It measures all forward, input-VJP and
weight-VJP GEMMs for the actual D256/L4/context128 or512 byte-LM: MLP up/down,
904-channel merged inputs, mixer output, and the configured vocabulary head,
at batches1/8/16. It retains the real transposed operand views, compares
packing them **inside** each timed call, and reports synchronized latency,
`2*M*N*K` GFLOP/s, launch kernel identities, logical bytes and independent
scalar-f64 probes. First-use compile/tune costs precede warmups. Large
1024/2048 squares provide an empirical backend ceiling, not a hardware peak.
The same report includes126 small batched solver F/VJP probes: head batch4,
16-token overlaps/inverses/corrections/state updates, transposes, initial
inverse batch broadcasting and strided chunk views from a context512 parent.
These diagnose unsupported legacy execution, not the active fused scan.
`recommended_path_passed` and `legacy_solver_passed` are separate; aggregate
`passed` remains false if either scope fails, with all offending values retained.
Numerical failures retain the offending values and unchanged tolerances,
continue characterization of the remaining shapes, then exit nonzero with
`passed: false`; an invalid kernel's apparent GFLOP/s is not accepted throughput.
Partial reports survive cooperative deadline errors; the enclosing process
timeout remains the hard bound. Runtime configuration is installed explicitly:
CubeCL environment-variable overrides do not replace these application settings.
Use a fresh configured `cache_dir`, without deleting existing caches, to measure
cold tuning/compilation; replay the same directory to measure persistence.

`gemm-step` runs the unchanged complete guarded training path, reports its
first step separately, then warm samples. `scripts/benchmark-gemm.ts` pairs
separate non-autotune/autotune binaries in alternating AB/BA process order,
using identical seeds, batches, data indices and optimizer work. Its default
11 pairs each measure the fourth step of a fresh trajectory after first use
and two warmups. Raw stdout/stderr/process outcomes and paired bootstrap
intervals are retained; build and reference checks remain separate commands.
The optional final driver arguments select the candidate head
(`native|padded320|heuristic`) and projection
(`merged|merged_heuristic_output`); the control always stays historical merged
inputs/native head with autotune off. Preflight reports/costs remain outside
every ordinary-step clock.
Serialize every GPU invocation with `flock /tmp/delta-block-lm-gpu.lock`; keep
builds outside timing windows. `kron-model-check --model-scope configured`
checks only the explicitly selected scan/HOLA pair, both contexts128/512 and
with/without active HOLA, against the unchanged CPU logits, all parameter
gradients, AdamW update and full-parameter directional check. Default
`--model-scope all` retains strict acceptance of every combination and now
records each combination separately instead of stopping at its first failure.
Neither scope accepts the independently known primitive legacy solver defect.
See “Matrix-multiply efficiency” in root `RESULTS.md` for selections and commands.

Independent component checks avoid coupling HOLA or fused Kron acceptance to another execution path's regression:

```sh
timeout 90s target/debug/delta-experiment hola-reference configs/reference-cpu.json
timeout 90s target/debug/delta-experiment hola-reference configs/reference-vulkan.json
timeout 90s target/debug/delta-experiment hola-reference configs/reference-vulkan.json --hola-backend fused_gpu
timeout 90s target/debug/delta-experiment kron-reference configs/reference-vulkan.json --scan-backend fused_gpu
timeout 90s target/debug/delta-experiment kron-reference configs/reference-vulkan.json --scan-backend tensor_solver --tokens 7 --objective score_only
```

`hola-reference` reuses the existing scalar-f64 HOLA fixtures: CPU explicitly exercises Tensor, while GPU explicitly exercises Tensor and FusedGpu, with each report naming its actual implementation. Optional `--hola-backend tensor|fused_gpu` selects exactly one implementation independently of the other path's gate; requesting fused execution on CPU fails explicitly. `kron-reference` reuses the existing Kron fixtures for only the requested backend (defaulting to `scan_backend`), optionally one token length (1..=512) and one objective. Objective names are `combined`, `output_only`, `final_state_only`, `score_only` (indices 0/1/2/3 also accepted). Both commands publish JSON with `passed`, numerical results or the exact failure context, then exit nonzero on failure; they do not skip an unrelated failing fixture inside their selected scope.

## Paired KronLSTM/HOLA execution benchmark

```sh
# Build once centrally, then run bounded local commands; the runner never builds:
timeout 120s cargo build -p delta-experiment --features vulkan
bun scripts/benchmark-kron.ts vulkan target/debug/delta-experiment 2 5
# Same host-generated tensors and canonical initialized parameters on CPU/Vulkan:
bun scripts/benchmark-kron.ts both target/debug/delta-experiment 2 5
# Direct command (uses max_seconds plus this hard process deadline):
timeout 930s target/debug/delta-experiment kron-benchmark configs/kron-benchmark.json --warmups 2 --trials 5
# Independent HOLA-only operator comparison; keeps every Kron gate unchanged:
timeout 930s target/debug/delta-experiment hola-benchmark configs/kron-benchmark.json --trials 5
# Full-model numerical preflight only (no throughput), CPU truth plus directional guard:
timeout 930s target/debug/delta-experiment kron-model-check configs/kron-benchmark.json
# Actual CPU-certified model scope, preserving the separate legacy failure:
timeout 930s target/debug/delta-experiment kron-model-benchmark configs/kron-benchmark.json --warmups 2 --trials 5 --legacy-stress-report artifacts/reference-vulkan/kron-reference.json
# Same explicit scope through the bounded Bun runner:
bun scripts/benchmark-kron.ts vulkan target/debug/delta-experiment 2 5 model artifacts/reference-vulkan/kron-reference.json
```

The command writes only a unique `kron-benchmark-*` namespace, never an old suite budget, metric, checkpoint or certificate. The Bun runner records subprocess deadlines/outcomes and stops on failure. Partial reports are retained. This is scratch numerical/performance work, not a new quality-training run.

Raw recurrence fixtures cover B1/T128, B1/T512, B4/T512 and B1/T2048 at H4/K32/V64/chunk16. Full language-model AdamW steps cover contexts 128/512 and HOLA off/on, with the configured batch/width/depth (the supplied file uses B1/D256/L4). On Vulkan, HOLA-on compares all four recurrence/HOLA execution combinations. CPU executes only the tensor reference and never labels it fused. Input tensors are generated once on the host from the same seed. Model initializers are forced on CPU before one-time device transfer, with exact parameter equality and an initialization hash; no model/input transfer occurs inside timed steps.

Isolated HOLA forward/backward timings additionally cover contexts 128/512 at capacities 16/64, using two untimed warmups per mode and the requested number of paired trials. Their original-KV membership/output/read-gradient gate runs outside the timed region.

`hola-benchmark` invokes those same per-fixture gates and paired timings independently of Kron's recurrence gate. It writes a unique `hola-benchmark-*` namespace with raw samples, medians and two warmups per mode, under the configured deadline. It establishes only isolated HOLA operator throughput, not Kron equivalence or full-model training throughput. Its first shape-specific calls precede per-fixture correctness preflight; caches may already be warm, so these calls are not isolated cold-JIT measurements. Warm throughput samples follow the checked preflight and two warmups.

`kron-model-check` independently checks the actual normalized language model at contexts 128/512 with HOLA off/on, without making an all-primitive-input claim for the unresolved legacy TensorSolver stress diagnostic. It uses canonical identical parameters, a CPU full-parameter directional finite-difference guard, and CPU logits/every parameter gradient/AdamW-updated parameters as the reference for each actual execution combination. Missing or nonfinite gradients fail. Results and any failure context are retained in a unique namespace before a nonzero exit. No timing or training-quality result is produced by this command. The paired benchmark reuses this same full-model preflight before its step timing; its separate all-stress gate remains intact.

`kron-model-benchmark` independently accepts the new selected kernels against all float64 stress fixtures, then reuses the exact CPU-certified model preflight and paired full-model AdamW/checkpoint timing loop. Its declared scope is only the actual normalized model workload. On Vulkan it requires a retained failed TensorSolver `kron-reference` report, validates that provenance, copies the original bytes and records their SHA256 alongside the known issue. `global_tensor_solver_stress_accepted` remains false. The legacy arbitrary-input GPU failure is neither hidden nor accepted, and the strict `kron-benchmark` gate remains unchanged. Artifact namespaces are unique; earlier reports, suite budgets and checkpoints are not overwritten. The Bun runner defaults to strict scope; the `model` argument selects this narrower, explicitly CPU-certified comparison.

Before throughput is published, independent f64 recurrence checks cover output/final-state/physical-score values and VJPs, including initial state and single-output objectives. Each paired fixture then checks output/state/score/input-gradient equivalence; each full model checks identical initialization, logits, every parameter gradient and the first complete AdamW update. Missing/nonfinite full-model gradients fail. HOLA's isolated checks additionally certify exact cache membership and sink/read gradients.

Device barriers surround every timing. Trial order rotates across paired paths, warmups are discarded, and reports retain every sample plus medians. Process-first recurrence forward/backward is separate from warm forward and forward/backward; its wall time includes any persistent-cache-miss JIT, but is not falsely labeled isolated compiler time. First full-model forward/backward is separate from warm complete AdamW steps. Checkpoint model/optimizer save and guarded validated reload are measured separately and excluded from throughput. Resident-fixture step times exclude dataset generation, logging and checkpoint IO; they do not replace end-to-end training throughput measurements. Default selectors stay historical until explicit measured configurations are chosen.


## Whole-model section profiling

```sh
timeout 180s cargo build -p delta-experiment --features vulkan
timeout 180s target/debug/delta-experiment profile configs/lm512-kron-fused.json --contexts 128,512 --warmups 2 --trials 5
```

`profile CONFIG` uses the first configured arm (or `--arm NAME`), configured seed, and configured context unless `--contexts` supplies a comma-separated list. It creates only a unique `profile-*/report.json` under `artifact_dir`; it never trains into an existing run or changes a budget/checkpoint. Warmups are bounded to 1–10, trials to 3–31, contexts to 1–8 entries, and a cooperative deadline of `min(max_seconds, 300)` is checked between complete steps. Keep the external `timeout` for a hard process deadline. Partial reports survive a deadline/error and the command exits nonzero.

Two independent scratch model/AdamW trajectories use identical batches and step indices. Their order alternates: ordinary complete steps with only outer device barriers and production scalar readbacks, and section-synchronized complete steps with barriers before/after every measured section. Both modes discard the specified warmups and retain raw samples and medians. Corpus loading, initial model upload, first-use compilation, the numerical preflight, checkpoint I/O, evaluation and report writes are outside the warm step. Host batch preparation, batch construction/uploads, loss readback, all-parameter gradient checks/readback, backward, clipping and AdamW are inside it.

Each context also records its first complete ordinary step **before** the
instrumentation preflight and warmups, including optimizer initialization,
fusion planning and any cache-miss JIT. This is not isolated compiler time.
Earlier contexts may warm shared kernels. Use a new, recorded `cache_dir` for
an empty-cache comparison; do not delete the persistent historical cache.

Forward timings retain the actual autodiff graph; they are not inference timings. The same forward implementation is used with a no-op observer in ordinary training. Reports rank summed per-step section samples and include every block separately, including its two outer RMSNorms, HOLA Q/K RMSNorms and mixer output RMSNorm. Q/K/V/decay/erase/write/output-gate/output projections, causal convolutions, L2 normalization/gate activations, scan, HOLA read, MLP up/SiLU/down, residuals, head, loss and host transfers are separated. Backward is total-only because Burn does not expose backward time per forward section; gradient extraction is included in that total. The loss bucket includes its weighted reduction as well as log-softmax/gather. The gate bucket includes convolution SiLU, decay/gate activations and cache/output gate multiplies. The mask bucket includes causal document-ID construction and input/output masks.

**Section barriers perturb overlap, submission and fusion.** A section's percentage is its median divided by the ordinary complete-step median, not a composable fraction of device time; the sum need not equal 100%. The ordinary step is unsynchronized *internally* apart from required production readbacks; its outer barriers still ensure completed-device wall time. Per-step sums are formed before medians, not by averaging individual layer medians.

Before timing, an instrumentation preflight compares ordinary and instrumented logits, every parameter gradient and one AdamW update at the existing 2e-5 absolute/3e-4 relative tolerance. It does not replace the independent f64 component/model checks or certify the known legacy GPU tensor-solver defect. A separate warm step counts issued kernels with CubeCL's `LaunchObserver` when GPU features are enabled; counts are not workgroups, copies, command-buffer submissions or graph replays, and the observer is absent from timed trials.

Memory reporting is limited to sampled backend-pool reserved bytes after section boundaries and explicit logical parameter/moment/batch payload sizes. It is **not** exact live temporary allocation, allocation traffic or a physical-device peak: pools reuse buffers, transient inner-section peaks can be missed, autodiff retains tensors, and the local integrated GPU shares system memory.

## Backward component attribution

```sh
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 180s target/debug/delta-experiment backward-profile configs/lm512-projections.json --contexts 128,512 --warmups 2 --trials 5
```

`backward-profile CONFIG` targets the current recommended non-fusion f32
KronLSTM+HOLA model: D256/L4/H4/K32/V64/chunk16/cache16, fused scan/read,
merged projections and historical tensor normalization/gating. Any positive
configured `batch_size` is supported for both full steps and all component
fixtures; the other recommended shapes/selectors remain required rather than
silently changed. `profile` also supports configured positive batch sizes.
Contexts default to both128 and512; `--contexts` can select either/both.
Warmups1–10, trials3–31 and `min(max_seconds,300)` bound the work; keep the
external hard timeout. Every invocation saves its own
`backward-profile-*/report.json`, including a partial report on failure.
It never writes a training budget, checkpoint or historical run.

Resident independent-leaf fixtures call the production component operators
at exact model shapes. All four layers are measured, with their component
times summed inside each trial before medians, rather than multiplying one
layer's median. Both modes build fresh autodiff graphs and synchronize before
and after each component group; forward-only and forward+backward order
alternates. Multioutput groups include their sum-of-elements roots in both
modes; other operators backpropagate directly with Burn's all-ones output
adjoint. Inputs and every relevant parameter require gradients. Scan uses
only its output, matching the LM: final state is unused and HOLA selection
scores are discrete. Fixture initialization/uploads, cold first use, warmups,
corpus/checkpoint/report I/O and the launch observer are outside warm timings.

A separate alternating pair of complete real-model scratch training
trajectories supplies the ordinary full-step denominator and synchronized
total backward, gradient checks and AdamW (with clipping). No model part is
removed or replaced. The optimizer/check row is non-autodiff work after real
full-model gradients: its forward and forward+backward columns both mean the
complete guarded update, and its derived backward is zero.

The report ranks forward+backward medians and percentages of the ordinary
complete step, derives backward as the difference of the two medians, and
reports signed backward coverage and whole-step gaps. These are **not**
composable GPU utilization or predicted savings: isolation changes gradient
accumulation, sum roots, launch overhead and overlap; real backward also
extracts parameter gradients. The scan sweep independently varies batch1/2/4
and heads2/4/8 at each context, reports observed launch histograms, and gives
source-derived `batch*heads` workgroups of64 lanes in each scan kernel.
Workgroup count is not a measured hardware occupancy percentage. The report
includes full-step tokens/s (`batch_size*context*1000/ordinary_step_median_ms`)
and logical scan state/checkpoint/replay storage. Shared-pool reservations are
sampled after existing backward/check/update barriers and after the complete
case, outside timers. They include both scratch trajectories and allocator
reuse, not exact live bytes or physical-device peak. See root `RESULTS.md`,
**Backward attribution** and **Batch scaling**, for measured rankings,
coverage gaps and the batch-dependent optimization decision.

## Paired mixer projection evaluation

```sh
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment projection-benchmark configs/lm512-kron-fused.json --contexts 128,512 --warmups 2 --trials 5
```

`projection-benchmark` compares `separate`, `grouped` and `merged` contractions
within one process on independent, identical-initialization model/AdamW
trajectories with matching training batches and step indices. All six
permutations of the three execution orders are rotated. Ordinary guarded full
steps and separately section-synchronized trajectories are interleaved; the
report retains raw paired savings, complete-step samples, aggregate forward
projection sections and separate warm kernel-count observations. Section
barriers still perturb overlap and cannot establish additive step savings.
Two warmups and five trials fit the reference config's normal deadline; longer
comparisons should archive a config with a larger `max_seconds` (at most 300)
and retain the external hard timeout.

The command first checks an independent scalar-f64 projection oracle, including
every original weight/input VJP, both mixer gate widths, strided inputs, zero
values and tail widths. It then checks full-model logits, every parameter
gradient and one AdamW update against separate projections at unchanged
2e-5/3e-4 tolerances. Resident isolated forward and forward/backward measurements
include packing, splitting and input/weight gradients; the full guarded steps
include optimizer, clipping and production finite-gradient checks. The existing
`kron-model-check` additionally uses separate projections in its CPU reference,
even when the candidate config explicitly selects a merged execution path.

Selecting `"projection_backend": "grouped"` or `"merged"` changes execution but
not parameter storage or IDs. Omitted/`"separate"` preserves historical config
fingerprints, checkpoints and contraction behavior. Nondefault selectors are
included in experiment fingerprints; do not change one inside an existing
training run. The dependent output projection cannot join the common-input
contractions. These library packing candidates do not fuse activations or
change the rejected Burn-fusion recommendation.

## Paired normalization and gating evaluation

```sh
timeout 300s cargo build -p delta-experiment --features vulkan
timeout 300s target/debug/delta-experiment normalization-reference artifacts/normalization-evaluation/comparison.json
timeout 300s target/debug/delta-experiment normalization-benchmark artifacts/normalization-evaluation/comparison.json --contexts 128,512 --warmups 2 --trials 21
timeout 300s target/debug/delta-experiment kron-model-check artifacts/normalization-evaluation/fused_norms_gates.json
```

The archived comparison config uses the current merged-projection, custom
Kron/HOLA workload with `budget_file:null` and `max_seconds:300`.
`normalization-benchmark` reuses the projection comparison's paired six-order
protocol, independent identical-start ordinary/section-barrier trajectories,
same batches/step indices and complete production AdamW/safety checks.
It compares historical `tensor`, reduction-only `fused_norms`, and
`fused_norms_gates` (also sigmoid/output multiply + RMSNorm, softplus/exp decay).
Both explicit candidates preserve all parameters and the original masks and
epsilons, including separate learned HOLA norms. Omission means historical
execution and serialization; a nondefault selector changes fingerprints and
requires a fresh run/budget, not a checkpoint or optimizer migration.

Before throughput samples, the command runs an independent scalar-f64
output/every-input-and-parameter-gradient oracle and checks full-model logits,
every parameter gradient and AdamW against the tensor control at unchanged
2e-5/3e-4 tolerances. `kron-model-check` additionally compares the selected
normalization path against tensor-normalized CPU truth plus the existing
all-parameter directional guard. The isolated representative per-block graph
includes three L2 norms, HOLA Q/K RMSNorms, two outer RMSNorms, mixer gated
RMSNorm and decay; it excludes projections/scan/HOLA read/checks/AdamW.
The affected-forward group includes all RMSNorm, L2/gate and residual sections;
raw sections and dispatch histograms remain available. Section barriers and
isolated objectives are not additive full-step speedup estimates.

## Soft-feedback miniature

`soft` is a vocabulary port of the copied relation state's convex probability update, separate visibility, expected-embedding/mask mixing, and proposal/retained-value distinction. It keeps selected probability mass rather than silently renormalizing truncated top-k candidates. Discarded probability mass contributes the mask embedding. Candidate selection is discrete, but selected weights and embeddings retain gradients. Ties use ascending vocabulary ID.

The command checks sparse/full-support and truncated-support expectation and selected-weight/embedding gradients against the dense path, full-support convex updates, zero-mass behavior, deterministic ties, immutable observations, inactive padding, and visibility-only remasking that retains beliefs and answers. Hidden/padded poison values are excluded before products, and invisible mask embeddings are excluded at full visibility; endpoint poison checks cover outputs and gradients while preserving failures for active nonfinite embeddings. Each round records IDs, absolute weights, retained mass, visibility, provisional IDs and retained IDs. `soft_candidate_width`, `soft_steps`, `soft_eta`, `soft_temperature`, and `hard_mask_only` are independent configuration fields. The hard-mask-only option is only a comparison. There is no learned energy selector, circuit, commitment policy, or SBD implementation here.

## Deterministic recall and paired comparisons

```sh
# One-seed small full-model smoke for all four arms and matched recency controls:
timeout 300s bun scripts/run-experiments.ts smoke cpu
# Pilot -> one paired smoke seed -> exactly three paired decision seeds:
timeout 1500s bun scripts/run-experiments.ts recall128 vulkan
timeout 1500s bun scripts/run-experiments.ts recall512 vulkan
```

The runner does not build anything. Its optional fourth argument selects an already-built executable. It writes a resolved configuration per suite/backend and uses a separate hard deadline for each subprocess; command exit codes, elapsed time, stdout and stderr paths are recorded in `command-outcomes.json`. Any nonzero exit stops the sequence. An incomplete one-seed budget stops before the three-seed comparison. Repeating a comparison reuses completed metrics unchanged after validating the complete config/resolved-token-budget fingerprint, current corpus/evaluation identity, arm, seed, precision, optimizer, full-model training path, backend, actual device identity and measurement implementation version `guarded-validated-load-v2`. It does not reload models or re-evaluate completed runs and therefore preserves the original measured loss, training memory and timing. Older measurement versions fail with an explicit `train --resume` or evaluation-only `certify` action; they are never silently treated as corrected evaluations. Only matching incomplete runs resume `latest.json`, retaining the original fixed budget. Historical checkpoint baselines are untrusted for improvement claims until certification regenerates them; absent trusted baselines make improvement null and withhold learning eligibility. Measurement versioning does not change checkpoint configuration/data fingerprints.

The planned recall files use two layers, width 128, contexts 128/512 and the inherited four heads, key width 32, value width 64 and chunk 16. The intentionally smaller smoke file uses one layer, width 32 and context 32 only to exercise training cheaply. The context-128 file includes both matched-size recency controls; the context-512 file does not scale those controls into the entire grid. Cache capacity starts at 16 and can be changed in the file.

Each example stores distinct random keys and values, inserts distractor filler, then asks multiple queries in shuffled order. Loss is applied only to answer positions, whose answers occur later than the query input. Association count and query-to-write distance are reported separately and jointly. A hash partition of the complete input/label sequence prevents **any** example collision from crossing training/evaluation splits. Evaluation has its own fixed seed and recorded fixture hash, independent of training seeds. Training example index is a pure function of seed and step, so resume preserves data order.

`benchmark CONFIG` measures first-use forward/backward/AdamW separately, then a short warm full-model training pilot for every arm. It writes `budget.json`: slowest measured arm throughput times `comparison_seconds`, rounded down to complete batches (at least one). Both mixer families, with/without HOLA, receive exactly that fixed token budget. The budget loader requires a SHA256 identity of the entire resolved configuration plus optimizer, precision and training-path identity; changing cache capacity or optimizer settings invalidates an existing pilot as surely as changing dimensions or arms. The runner preserves an existing budget, but stale identities fail before training and require a fresh pilot/artifact directory. Hyperparameters and arms live in files, not code.

`compare CONFIG` runs one paired seed; `decision CONFIG` runs exactly three distinct paired seeds. Results include actual parameter counts, recurrent-state bytes, HOLA plus current-chunk memory, convolution-state bytes, training throughput, held-out losses, recall breakdowns and warm next-token latency. `hola_work_per_model_forward` counts allocated history, current-chunk and sink attention pairs across all layers, plus exact pairwise-rank comparisons and sort input elements; it includes masked/unfilled slots and does not pretend sort elements are sort comparisons or infer backward FLOPs. Decode timing is explicitly full-prefix recomputation, not an unimplemented incremental cache benchmark. The decision summary reports paired seed values, seed standard deviation and a quality/throughput/state-memory Pareto screen, not a single-seed winner. It withholds decision eligibility unless all budgets finish and each arm improves mean held-out NLL. A recall improvement is not a language-modeling result.

Recall decision eligibility additionally requires demonstrated retrieval rather than only improving NLL. `random_accuracy_baseline` is `1 / value_symbols` (uniform guessing among answer-value symbols, a stronger baseline than guessing across the full input vocabulary). Each arm reports each paired seed's correct count, trial count, accuracy, and two-sided 95% Wilson interval. **Every seed of every arm must have its lower bound strictly above guessing** before a recall Pareto shortlist is emitted. `heldout_learning_demonstrated` remains the separate mean held-out-NLL improvement flag; it can be true while `recall_skill_demonstrated` and `decision_eligible` are false. The Wilson calculation assumes independent, identically distributed Bernoulli outcomes, whereas queries within an example may be correlated and the fixed evaluation set is reused across seeds. Seeds are not pooled. This deliberately conservative diagnostic is not a proof of retrieval skill, statistical independence, or architectural superiority.

## Pinned document-disjoint tiny language model

The pinned corpus is six pre-1900 public-domain books obtained from Project Gutenberg, with exact source-byte SHA256 hashes in `configs/corpus.lock.json`. The original source license permits research/reuse under US copyright law; users outside the US must check local law. Source license: <https://www.gutenberg.org/policy/license.html>. Raw license-bearing originals are retained separately from extracted training text, all under ignored `data/`.

Whole books are assigned **before** extraction/tokenization: Alice, Frankenstein, Pride and Prejudice and Sherlock Holmes train; Tom Sawyer validates; Moby Dick tests. Preparation fails on changed source bytes rather than silently repinning. It records source hashes, extracted-document hashes, split hashes, the lock hash and tokenizer identity. The deterministic tokenizer is UTF-8 bytes 0–255 with BOS 256; no learned tokenizer or validation vocabulary is fitted. Each sampled window stays within one document and starts from a reset/BOS. Validation/test select one complete, nonoverlapping midpoint window per equal-width stratum across the heldout documents, rather than only book-opening title/contents pages. The recorded selection identity is `stratified-midpoint-disjoint-v1` and the actual fixture hash is retained. `eval_examples` bounds that documented evaluation subset, not a claim to full-corpus perplexity.

```sh
timeout 150s bun scripts/prepare-data.ts
timeout 180s bun scripts/run-experiments.ts lm-smoke cpu
# Planned four layers, width 256, context 512, four controlled arms:
timeout 1500s bun scripts/run-experiments.ts lm512 vulkan
```

The LM is a full trainable causal model: embedding, residual mixer blocks, RMS norms, SiLU feed-forward layers and vocabulary head. AdamW updates all model parameters; missing or nonfinite gradients are fatal before optimizer writes. Mixer shape/cache differences are reported instead of pretending equal width means equal capacity. The LM retains the same tokenizer, data order, float32 precision, optimizer, output-head architecture and budget across arms. Outputs include initial/final held-out NLL/perplexity and an explicitly synthetic byte-vocabulary associative retrieval diagnostic.

A completed short command is not proof that held-out learning or architectural gates passed. Read `completed`, `heldout_learning_demonstrated`, `recall_skill_demonstrated`, `random_accuracy_baseline`, `decision_eligible`, per-seed uncertainty diagnostics and the reference numerical report. In particular, near-chance recall accuracy must not be promoted to architectural evidence merely because NLL improved. Insufficient learning requires another explicitly chosen **file-configured** budget and fresh artifact directory; do not relabel a smoke as an architectural decision.

## Checkpoint/resume and artifact contract

Each `ARM-seedN/` directory contains `training.jsonl`, `metrics.json`, `latest.json`, and numbered checkpoint directories. A checkpoint contains model weights, AdamW moments/step counters and `state.json`; metadata is written last. Resume validates the complete resolved configuration **and token budget**, dataset identity, arm and seed, then restores optimizer and example counters. Reusing an existing numbered destination requires exact logical model values/IDs and complete canonical optimizer state, not merely matching metadata; conflicting evidence is never overwritten or published as the latest checkpoint. Comparisons reuse validated completed metrics without model loading/evaluation and genuinely resume only matching incomplete runs. This cache preserves evidence; it does not certify that a measured outcome was numerically reliable. The explicit individual `train --resume` interface is:

```sh
timeout 180s target/debug/delta-experiment train configs/lm-smoke.json \
  --arm gdn2-hola --seed 1 \
  --resume artifacts/lm-smoke/gdn2-hola-seed1/checkpoint-4
```

Use the actual checkpoint path in `latest.json`, not the example's number. Do not change configuration midway through a resumed run. Atomic JSON publication means a missing checkpoint `state.json` is not a complete checkpoint. Final evaluation and metrics are outside the training-loop soft deadline; the enclosing `timeout` remains the absolute deadline.

`sampled_peak_reserved_device_bytes` is the maximum allocator reservation sampled after forward/backward/optimizer steps. Each run requests cleanup and synchronization before constructing its model and records `reservation_baseline_after_cleanup_bytes`; the runtime may nevertheless retain reservations across runs, so the scope is explicit. This is **not** exact peak GPU allocation; `peak_gpu_memory_bytes` remains null because the runtime does not supply that measurement. The local integrated Intel GPU shares system memory. An exact external GPU-memory measurement is still needed for that plan gate, rather than inventing a value.

Unit checks are in crate source so there is no Burn-heavy forest of integration binaries. Central integration should run, once all changes land:

```sh
timeout 60s cargo test -p delta-experiment --no-run
timeout 90s cargo test -p delta-experiment
```

The test suite and experiment commands must be run by the integration owner; implementation alone does not claim measured results.

## Isolated repeat/save-load diagnostics

Local integration observed intermittent disagreement in tiny-LM evaluation after checkpoint reload. For KronLSTM seed 1 the live run's NLL was about 3.15003937, a first reload reported 7.58646894, and a later reload reported 3.15003914; smaller GDN drift was also observed. Follow-up Vulkan diagnostics proved changed parameter values on fresh unique save/load for the no-cache GDN-2 and KronLSTM arms, while the tested same-model repeats were bitwise equal and HOLA arms passed. Independent host decoding of all eight source/scratch checkpoint files matched every pre-save parameter hash exactly. One changed decay-bias tensor exactly matched its `-4` initializer. A subsequent synchronized-target A/B run on the original four Intel Vulkan LM arms passed all source/scratch model roundtrip checks and produced bitwise-equal next-AdamW-step parameters and logits. The measured fix is to keep each constructed load target alive across a device synchronization before applying record uploads, then synchronize afterward. A queued initializer/upload race is the inferred mechanism, not a fully established upstream backend defect. The original failed measurements and diagnostic directories remain evidence rather than being relabeled as reliable.

`checkpoint-check` writes only a unique `checkpoint-diagnostics/BACKEND-seedN-PID-TIMESTAMP/` directory plus its own diagnostic-command result. It never writes original run metrics, logs, optimizer state or checkpoints. The command covers all configured arms unless `--arm` is given. If a matching existing checkpoint is available, its source config/budget/data/arm/seed identity is validated before loading; otherwise a fresh fixture is explicitly labeled as such.

For each arm it records logical parameter SHA256 snapshots (shape, dtype, module path and parameter-device metadata), training-mode versus validation-mode logits, two evaluations of the same immutable model and batch, and two evaluations after a **fresh unique** save/load. Parameter values are checked before/after evaluation and serialization. A second isolated phase performs one real scratch AdamW update on the same fixed fixture to expose optimizer-produced layouts, then repeats the checks; that step is diagnostic work, not part of the original training outcome. Logit comparisons use float32 absolute tolerance `1e-5` and relative tolerance `1e-4`, reject nonfinite inputs and report exact hashes and errors. Parameter-value comparisons are bitwise and ignore parameter IDs, which may change on loading.

All active model loads now use that synchronization boundary and compare every saved logical parameter's bytes, shape, dtype and parameter ID with the loaded value; any discrepancy is fatal. No CPU staging, backend patch or relaxed numerical tolerance is used. Production optimizer loads compare the stored and restored complete typed logical state, rejecting the upstream loader's otherwise-silent omission of unreconstructable states. Canonical state includes tensor name, dtype, shape, ID and logical bytes, scalar counters, path metadata, actual AdamWConfig and learning rate. Raw container hashes are retained for forensics but are not equality criteria: tensor insertion order and file offsets can differ without changing optimizer semantics. Diagnostics load the actual source optimizer when available before scratch continuation, then compare the next AdamW step's parameters, semantic optimizer state and logits. Installed Burn 0.22.0/0.22.1 source comparisons found identical core-record loading, parameter transformation and CubeCL tensor upload code, so there was no source evidence for a patch-version-only fix.

```sh
# Six arms including recency controls; identical small fixture on each backend:
timeout 180s target/debug/delta-experiment checkpoint-check configs/checkpoint-cpu.json
timeout 180s target/debug/delta-experiment checkpoint-check configs/checkpoint-vulkan.json
# Diagnose actual Vulkan-trained weights without altering their source config/hash:
timeout 300s target/debug/delta-experiment checkpoint-check artifacts/lm512-vulkan/resolved-config.json --seed 1
timeout 300s target/debug/delta-experiment checkpoint-check artifacts/lm512-vulkan/resolved-config.json --arm kronlstm --seed 1 --diagnostic-backend cpu
```

`--diagnostic-backend` changes only the execution device and result namespace, not the original source configuration or checkpoint identity. The CPU comparison is an independent backend check, not an exact float64 language-model oracle. All complete diagnostic results are retained in their unique directories; per-arm partial results are published as the command advances, and a failed numerical comparison gives a nonzero command exit after preserving its report. Passing a small fixture does not retroactively validate the earlier discrepant LM measurements.

## Re-certifying original three-seed evaluations without training

Old evaluation caches lack the current measurement version and must not be reused as certified quality evidence. `certify CONFIG --checkpoints MANIFEST.json` requires a schema-2 manifest covering every configured arm and all three paired seeds. Each entry pins the original checkpoint/state and training-metrics file SHA256. The SHA-pinned original metrics **file bytes** are authoritative: no JavaScript-parsed/reserialized numeric snapshot is compared or allowed to override them. Rust parses that verified source and preserves a verbatim byte copy in the certificate namespace. Complete source configuration identity is checked by its recorded fingerprint, avoiding redundant cross-language floating-point Value equality. Data identity, arm, seed, budget and backend/device contract are checked. Model loading uses the validated synchronization boundary, trained validation/test evaluation is repeated twice, and source files are rehashed afterward to detect changes during certification.

Certificates and a fresh comparison are written only under a unique `certified-evaluations/BACKEND-PID-TIMESTAMP/` directory. Original training metrics, checkpoints and stdout are not overwritten. Each arm/seed also gets a fresh seeded untrained baseline under the current backend, with synchronization, finite metrics and exact repeated evaluation. It is explicitly a regenerated baseline, **not** a verified bitwise replay of historical initial weights, whose fingerprint was not recorded. Only this trusted baseline governs learning improvement; old initial/final evaluations remain historical, and missing/untrusted baselines withhold eligibility. Baseline construction/evaluation, checkpoint load and trained evaluation costs are recorded separately. New quality metrics carry `measurement_implementation_version = guarded-validated-load-v2`; original training throughput/state-memory evidence retains its pinned provenance and is not remeasured. Recall chance/uncertainty gates remain separate. A failed repeated-evaluation or immutability check withholds the shortlist and returns a nonzero exit.

```sh
# Pins original checkpoints and training evidence for all arms/seeds in these three suites,
# then runs one evaluation-only process per suite, each with a 300-second deadline:
timeout 1000s bun scripts/certify-experiments.ts target/debug/delta-experiment
# Explicit suite subset or source configuration files:
timeout 400s bun scripts/certify-experiments.ts target/debug/delta-experiment artifacts/lm512-vulkan/resolved-config.json
```

The batch script never builds or trains. It writes schema-2 manifests, command outcomes and new stdout/stderr to a unique `artifacts/certification-batch-*` namespace; every filename has a configuration-index prefix, so normalized-path collisions or repeated arguments cannot overwrite an earlier stage. Summary values and per-seed diagnostics are ordered by the explicit `paired_seeds`, independently of manifest entry order; duplicate/unknown seeds fail. For manually pinned evidence, invoke the executable directly with a 300-second limit and the manifest path. Certification creates separate versioned reports; it does not mutate old `metrics.json` files to make the ordinary comparison cache accept them.

## Causal Lngram commands

`lngram_layers` selects injection block indices; `lngram_orders` selects the
orders inside each injection. The target uses three injections at `[0,1,2]`,
each with `[2,3]` orders. Omitted layers leave historical parameters, execution
and serialized fingerprints unchanged. Enabling the component requires a new
training run. `lngram_backend` explicitly selects `tensor` (route gather,
default), `flat_select` (global row selection), or GPU-only non-fusion
`fused_gpu` (exact symbol cache and analytic memory VJPs). `lngram_surrogate`
selects `one_bit` or `local_expectation`.

`lngram-reference CONFIG` checks the independent FP64 scalar oracle, packing,
causality, both surrogate gradients, CPU-model equivalence and table optimizer
groups. `lngram-benchmark CONFIG --warmups 2 --trials 11` measures isolated F/FB
and paired complete guarded steps with the component off, tensor and fused.
`lngram-attribution CONFIG --warmups 2 --trials 11` separates the per-injection
discretizer, codes/addresses, per-order gather and OneBit terms, readout,
smoothing and table AdamW costs and launches. `lngram-smoke CONFIG` trains
paired tiny synthetic byte-pattern models for 200 steps. All reports/configs
for this slice are under `artifacts/step7/lngram/`.
See `../../docs/optimization/lngram.md` for causal adaptations, memory,
numerical-oracle scope, exact locked commands and measured results.
