# More-parallel fused KronLSTM scan

On 2026-10-28, KronLSTM's learned beta, target and output-gate biases were
restored from paper §2.2 Eq. (11) and its output-gate definition. They are
zero-initialized because the paper gives no initialization recipe. The model
revision is `kronlstm-learned-gate-biases-v1`; bias-free historical checkpoints
are rejected. All scan widths consume the same already biased, activated
beta/target gates, so their recurrence and custom VJPs need no ABI change.
The output-gate bias is applied before ordinary or fused gated RMSNorm.
The tables below remain historical evidence for the earlier parameterization,
not performance measurements of the learned-bias revision.

The baseline/paired/isolated tables below are the **pre-isolation,
provisional sessions** retained for provenance. They used the initial shared
shader cache and earlier build helper. The final isolated confirmation
section records private-cache, transactional-rebuild evidence and takes
precedence; provisional does not mean false.

## Scope and compared designs

Local Intel Arc MTL integrated GPU, Vulkan, Burn 0.22/CubeCL 0.11, f32;
D256/L4/H4/K32/V64/checkpoint chunk16/HOLA capacity16, merged projections,
tensor normalization/gating, native vocabulary head, non-fusion backend and
**autotune disabled**. Seed1; batch8 is the adoption workload, with batch1
reported separately. No later-plan component, parameter, precision, mask,
optimizer, tolerance or historical configuration was changed.

The three explicitly selectable designs are:

| `scan_backend` | Lanes/workgroup | Key entries owned by each lane | Role |
|---|---:|---:|---|
| `fused_gpu` | 64 | 32 | Retained historical fused implementation/reference |
| `fused_parallel128` | 128 | 16 | Two cooperative key shards per value column |
| `fused_parallel256` | 256 | 8 | Four cooperative key shards per value column |

This is **cooperative state partitioning, not a chunk-parallel temporal
prefix scan**. Each batch/head still has one workgroup. The token recurrence
remains sequential and exact: shards cooperatively reduce the physical
bilinear edit/correction and output, then update disjoint state entries.
Backward uses the same chunk-boundary checkpoints and bounded replay; each
lane maintains its state/adjoint shard, reduces Q/K/alpha VJPs over value
columns, and reduces the value VJP over key shards. Scalar edit/score VJPs
are shared across all shards. There are no floating atomics, reciprocal-alpha
state reconstruction, transition-summary matrices or extra global tensors.
This directly compares two plausible parallelization widths without adding
a second recurrence algorithm or relying on the known-defective legacy GPU
tensor-solver stress case as an oracle.

Omitted selectors still mean `tensor_solver`. Existing `fused_gpu` remains
the original 64-lane kernels, and historical serialization/fingerprints are
unchanged. The new paths are explicit GPU-only choices, K32/V64/chunk1..16,
with the existing f32, device, shape and u32-address checks. They share one
new kernel implementation specialized by the number of key shards. The
configuration `configs/lm512-scan.json` explicitly selects the 256-lane path;
its batch remains the historical example's1, while the measured batch8
configurations are saved under `artifacts/scan/`.

## Saved baseline before candidate throughput

`artifacts/scan/backward-profile-2592918-1792865917937561477/report.json`
contains a fresh historical B8 baseline (two warmups, five trials), before
candidate throughput measurements:

| Context | Ordinary complete step, ms | Four-layer scan F, ms | Four-layer scan F+B, ms |
|---:|---:|---:|---:|
| 512 | 1456.589 | 66.452 | 335.620 |
| 128 | 344.712 | 18.724 | 84.478 |

The context512 scan agrees closely with the earlier339.84ms attribution,
but the whole step differs from the earlier1613.67ms session. This is why
acceptance uses **same-session interleaved complete steps**, not subtraction
of old session medians.

## Paired complete training steps

Each batch is one session containing all three designs at contexts512/128,
with two warmups and11 measured trials per design/context. Three independent
identical-initialization model/AdamW trajectories receive the same pinned
byte-LM corpus windows and step indices. The six-order rotation reverses on
odd trials. The timer includes actual batch preparation/upload, all model
parts, loss and scalar readbacks, backward, finite-gradient checks, clipping
and AdamW. No section barriers are inserted. Isolated scan samples are
separate calls between complete steps. Initialization, numerical preflight,
JIT/cache misses, warmups, report I/O and checkpoint I/O are excluded.

All values below are synchronized warm median wall times in ms. Percentages
are medians of **paired per-trial reductions**, not ratios of independent
medians. Their95% intervals are percentile bootstrap intervals of the paired
median (10,000 resamples, deterministic seed1765); these describe this
session, not across-day uncertainty.

| Batch | Context | Historical64 | Parallel128 | Parallel256 | Paired reduction128 (95% interval) | Paired reduction256 (95% interval) |
|---:|---:|---:|---:|---:|---|---|
| 8 | 512 | 1466.115 | 1261.741 | 1212.279 | 13.89% [13.17,14.75] | **17.64% [15.84,18.30]** |
| 8 | 128 | 339.406 | 289.804 | 281.512 | 15.01% [13.60,16.52] | **17.28% [16.19,20.94]** |
| 1 | 512 | 559.077 | 333.680 | 294.655 | 39.37% [38.13,40.13] | **47.02% [46.41,48.56]** |
| 1 | 128 | 193.578 | 135.200 | 122.010 | 31.38% [27.73,32.65] | **33.69% [32.73,39.83]** |

Both candidates beat the historical implementation in all11 paired trials
at all four shapes. At B8/context512 the256-lane median yields approximately
3379tokens/s versus2794tokens/s for64 lanes, an approximately1.21x ratio of
median whole-step throughputs. This is not the isolated scan speedup.

Reports with every raw sample, iteration order, exact config/device, fixture
hash, preflight and launch histogram:

- B8: `artifacts/scan/scan-benchmark-2613710-1792866098757892481/report.json`
- B1: `artifacts/scan/scan-benchmark-2650481-1792866438028402952/report.json`
- Paired statistical summary: `artifacts/scan/paired-summary.json`.

A second independent B8/context512 session repeats11 trials/two warmups:
64/128/256-lane median steps are1619.167/1409.895/1373.601ms. Paired
reductions are13.96% [10.93,17.16] and15.28% [9.06,16.11], with11/11
wins for each candidate. Its absolute baseline is10.4% slower than the
first session, yet the interleaved benefit remains positive beyond noise.
Report: `artifacts/scan/scan-benchmark-2722932-1792867087374107098/report.json`;
statistics: `artifacts/scan/paired-repeat-summary.json`.

## Isolated recurrence cost

One layer, H4/K32/V64/chunk16; resident deterministic normalized Q/K/V,
all-valid/no-reset inputs, zero initial state. F+B uses an output-only
all-ones VJP and extracts all seven input/state gradients. These timing
fixtures do not replace the independent masked/reset numerical fixtures.

| Batch | Context |64 F / F+B, ms |128 F / F+B, ms |256 F / F+B, ms |
|---:|---:|---:|---:|---:|
| 8 | 512 |18.123 /83.643 |8.627 /34.987 |6.129 /25.779 |
| 8 | 128 |5.332 /22.747 |3.062 /9.713 |2.129 /7.389 |
| 1 | 512 |18.855 /85.506 |8.793 /31.764 |5.842 /20.247 |
| 1 | 128 |6.072 /25.272 |3.185 /9.507 |2.088 /6.313 |

At B8/context512 the isolated F+B ratio is3.24x; the actual full step is
only1.21x. At B1 it is4.22x versus1.90x for the complete step. Batch8
therefore remains the decision workload rather than advertising the larger
batch1 gain.

## Launches, work and memory

Full steps issue exactly the same number of kernels for all three designs:
**5833 at B8 and5780 at B1**, at both contexts. Each contains four forward
and four backward scan dispatches. The gain is inside the scan, not dispatch
elimination. Source-derived scan geometry is4 workgroups at B1 and32 at B8;
64/128/256 lanes increase the cooperative lanes, not the number of independent
batch/head workgroups. Hardware occupancy/register use was not measured.

All paths perform linear O(B*T*H*K*V) recurrence/replay/VJP work. Key
partitioning shortens each lane's serial key loop but adds shared reductions
and barriers; it introduces no quadratic work. The logical per-lane forward
state arrays contain32/16/8 floats and backward state+adjoint64/32/16 floats;
actual compiler register allocation or spills are not claimed. Shared memory
per workgroup is:

| Design | Forward bytes | Backward bytes |
|---|---:|---:|
|64 |512 |24832 |
|128 |1536 |25088 |
|256 |3072 |25600 |

Global logical storage is unchanged. Per layer, per batch item (H4):

| Buffer | Context128 bytes | Context512 bytes |
|---|---:|---:|
| Initial-state payload |32768 |32768 |
| Packed output/final-state/physical scores |165888 |565248 |
| Boundary checkpoints retained for backward |262144 |1048576 |
| Correction/bilinear/key-norm/value-norm auxiliary |8192 |32768 |
| Bounded backward replay scratch |524288 |524288 |

Multiply each row by batch size. Input retention and seven gradient tensors
are the same as the historical implementation; no new global scratch is
allocated for key partitioning. The benchmark's sampled backend pool
reservations after all trajectories are778733760/778729728bytes at B8
(context512/128) and198329664/198321792bytes at B1. Those include all three
scratch trajectories and allocator reuse across contexts, **not** a
single-model live-memory estimate, exact temporary traffic, sampled physical
GPU peak or proof of reduced memory. The fresh two-trajectory historical
profiler's724188672/252833088bytes are not a like-for-like memory comparison.

## Independent numerical acceptance

Recurrence and model absolute/relative tolerances remain **2e-5 /3e-4**.
Historical checkpoint diagnostics separately retain their **1e-5 /1e-4** gates.
Both new selectors pass all13 default independent sequential-f64 checks:
T1/7/35, combined/output-only/final-state-only/score-only VJPs, plus a genuine
transposed-physical-buffer T35 strided-input combined case. Every case checks
all outputs, final state, physical edit scores, Q/K/V/alpha/erase/write
input gradients and the initial-state gradient; T>1 additionally checks
split continuation and its gradients. Fixtures include nonzero initial
state, ignored resets on padding, internal padding, an entirely padded
chunk, document resets, individual/all-channel zero alpha, signed and
exactly zero correction, and zero key/value norms. The strided case covers
all six sequence inputs, both masks and the initial state.

The default256-lane fixtures' largest output/state/score/input/initial-state
absolute errors are6.10e-9/6.14e-9/1.25e-7/9.15e-8/1.50e-8;
continuation input/state gradients remain within9.15e-8. The128-lane
maximum input-gradient error is7.34e-8. A separate256-lane T512 check passes
all four objectives: largest output/state/score/input/initial-state errors
7.27e-9/5.86e-9/1.44e-7/7.74e-8/1.50e-8. The recurrence fixture is explicitly
B2/H4 even when its surrounding experiment config has another batch size.

The paired real-corpus model preflights also pass at both batches/contexts,
checking logits, every parameter gradient and the AdamW update against the
retained historical fused implementation from identical parameters. Across
all candidates/shapes their maximum differences are2.60e-6 logits,
2.87e-6 parameter gradients and9.64e-6 updated parameters. These GPU-pair
comparisons complement, rather than replace, the independent f64 and CPU
model gates.

The independent **CPU** `kron-model-check --model-scope configured` for256
lanes passes all four context128/512 × HOLA off/on cases. Canonical CPU
parameters are forked bitwise to Vulkan, CPU full-parameter directional
checks pass, and CPU-reference logits/every parameter gradient/AdamW updates
pass unchanged tolerances. Maximum errors across the four cases are
4.67e-6/2.45e-6/7.98e-6 respectively. Report:
`artifacts/scan/parallel256-b1/kron-model-check-2717919-1792867046099650494/report.json`.

Historical `artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13`
passes source load, repeated inference/save/reload, a scratch AdamW step and
restored optimizer continuation at the unchanged1e-5/1e-4 tolerances.
The original omitted-selector config and fingerprint are used verbatim.
To keep all new outputs inside `artifacts/scan/`, that command runs in
`artifacts/scan/historical-check`: its `data` and `.cache` paths and the
original budget file are symlinks to the existing shared inputs, while its
relative `artifacts/lm512-vulkan` output directory is local to that staging
directory. The checkpoint itself is read by absolute path and not modified.
Result: `artifacts/scan/historical-check/artifacts/lm512-vulkan/checkpoint-check.json`.

**Historical validation matrix — GDN-2 removed on 2026-10-28 by user decision.** The following counts and selector rejection cases describe the original run, not the current KronLSTM-only test inventory.

The unchanged CPU `reference` diagnostic passes. Workspace CPU tests pass:
**36 passed,0 failed** (26 experiment,10 mixer). The default test feature
set does not execute GPU kernel unit tests; the explicit GPU/f64 diagnostics
above are their numerical evidence. New selector tests cover CPU/GDN-2
rejection, serde roundtrip, distinct explicit fingerprints and unchanged
omitted-selector serialization.

## Decision and component budget

**Adopt `fused_parallel256` explicitly for this f32 K32/V64 Intel-MTL
workload.** It wins all22 intended-batch/context512 paired trials across
two independent sessions, passes the independent numerical/model/checkpoint
gates, and needs no extra global scratch or dispatches. The128-lane design
is also correct and faster than64, but256 is faster in all measured shape
medians; keep128 as a low-cost specialization for reproducible comparison,
not the recommendation. Do not change historical omitted selectors or
silently replace `fused_gpu`.

The measured intended-shape complete-step budget is approximately1.2–1.4s
at B8/context512 in these sessions, versus1.47–1.62s for64 lanes. The
isolated four-layer scan budget inferred by summing identical-layer medians
is about103–107ms F+B, down from335–352ms; it is not a new additive
whole-model profile. This scan implementation earns its maintenance cost
on actual guarded training steps, not just dispatch or batch1 evidence.

## Reproduction

From `/home/heefoo/code/delta-block-lm-ws/scan`, build only with
`../build.sh scan`. The corrected helper acquires GPU then build locks,
cleans only the three local packages, rebuilds under an inner180-second
bound and copies `/tmp/delta-bins/scan/delta-experiment`. The final protocol
uses a private relative `.cache/cubecl` directory and acquires **GPU then
build** locks for GPU execution/manual cargo commands; hard compute bounds
go inside both locks. Outer queue waits are recorded separately.
The first model-check attempt exhausted a300-second outer deadline while
waiting for the shared GPU lock, leaving an incomplete report with no case
verdicts; it is not numerical failure evidence and is excluded from acceptance.

Saved `baseline-b1.json`/`baseline-b8.json` copy
`configs/lm512-projections.json`, changing only batch, artifact namespace,
`budget_file:null` and `max_seconds:280`. Candidate configs additionally
change the explicit scan selector. CPU reference config explicitly restores
CPU/tensor/separate execution. All new artifacts are under `artifacts/scan/`.

The following are the exact **earlier provisional** commands, before the
transactional-build/private-cache protocol was introduced:

```sh
../build.sh scan
flock /tmp/delta-block-lm-gpu.lock timeout 280s /tmp/delta-bins/scan/delta-experiment backward-profile artifacts/scan/baseline-b8.json --contexts 512,128 --warmups 2 --trials 5
flock /tmp/delta-block-lm-gpu.lock timeout 280s /tmp/delta-bins/scan/delta-experiment kron-reference artifacts/scan/parallel128-b1.json
flock /tmp/delta-block-lm-gpu.lock timeout 280s /tmp/delta-bins/scan/delta-experiment kron-reference artifacts/scan/baseline-b1.json --scan-backend fused_parallel256
flock /tmp/delta-block-lm-gpu.lock timeout 280s /tmp/delta-bins/scan/delta-experiment kron-reference artifacts/scan/parallel256-b8.json --tokens 512
flock /tmp/delta-block-lm-gpu.lock timeout 280s /tmp/delta-bins/scan/delta-experiment scan-benchmark artifacts/scan/baseline-b8.json --contexts 512,128 --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 280s /tmp/delta-bins/scan/delta-experiment scan-benchmark artifacts/scan/baseline-b1.json --contexts 512,128 --warmups 2 --trials 11
timeout 280s /tmp/delta-bins/scan/delta-experiment reference artifacts/scan/cpu.json
flock /tmp/delta-block-lm-build.lock env CARGO_TARGET_DIR=/home/heefoo/code/delta-block-lm-ws/target-shared cargo test --workspace
```

The first build attempt exposed a CubeCL compile-time/runtime mutable-loop
bound type mismatch in both new kernels; it was fixed by explicitly
initializing the reduction stride as a runtime integer before multiplying
by the specialization constant. The successful incremental Vulkan build
completed in3.88s wall. The CPU-only test build was cold in the shared target
and took81.79s including dependency compilation; no dependency, backend
feature default or build profile was changed.

## Final isolated acceptance and confirmation

The fixed `../build.sh scan` transaction acquired GPU→build locks after
20s of queueing, cleaned only the three local workspace packages, compiled
all three from this workspace's paths and completed compilation in3.64s
(23.60s total wrapper wall). The final binary and source proof is retained
in `artifacts/scan/final/build-proof.json`; no source code changed after
this build.

- Binary `/tmp/delta-bins/scan/delta-experiment` SHA256:
  `d44549a2edbc921a2821f419011f0ed788f9a54b531cdebf5a4f854c6d9647ae`
- `crates/kernels/src/kron_parallel.rs` SHA256:
  `7c6ea24cd0757ac92fd1b98cfe1b968a4929500457425269a3c6550ea71a5301`
- jj change: `qzystkmyyskmtmxzozrmyuntpynrzvtu`.
- Frozen source commit: `c72c28191a52b5b7dd249ad4b5fbea9e3ca4c546`.
- Remaining validation-only evidence change: `wkwonztqtylswzxkstxvstplmkwuoxyt`.

Final commands use the workspace-private **relative** `.cache/cubecl`
directory. Config copies are retained under `artifacts/scan/final/`;
they differ from earlier configs only in their artifact namespaces.
The command prefix is exactly:

```sh
/usr/bin/time -f 'QUEUE_AND_EXEC_SECONDS=%e' \
  flock /tmp/delta-block-lm-gpu.lock \
  flock /tmp/delta-block-lm-build.lock \
  /usr/bin/time -f 'EXEC_SECONDS=%e' timeout 280s \
  /tmp/delta-bins/scan/delta-experiment
```

The compute timeout is **inside** both locks. `EXEC_SECONDS` includes process
startup, device setup, numerical gates and report I/O, but no queue wait;
the difference from `QUEUE_AND_EXEC_SECONDS` reports the approximate lock
queue separately. The retained f64 diagnostics use180s rather than the
paired benchmark's280s bound.

Final retained command arguments after the prefix above:

```text
scan-benchmark artifacts/scan/final/historical-b8.json --contexts 512,128 --warmups 2 --trials 11
scan-benchmark artifacts/scan/final/historical-b1.json --contexts 512,128 --warmups 2 --trials 11
kron-reference artifacts/scan/final/parallel128-b1.json
kron-reference artifacts/scan/final/parallel256-b1.json
kron-reference artifacts/scan/final/parallel256-t512.json --tokens 512
kron-reference artifacts/scan/final/parallel256-t128.json --tokens 128
```

On the parent's integration directive, six **not-yet-started** redundant
queued commands (individual CPU-model checks at B1/B8, historical checkpoint,
CPU tests, old-scan and HOLA reference matrices) were cancelled after a
process snapshot showed all their GPU flocks blocked and another workspace
executing. No running measurement was interrupted. The parent owns the
combined integrated CPU-model/checkpoint/workspace-test gates once; the
earlier individual passing matrices above remain provisional evidence.

This item does not certify larger K/V widths, contexts beyond512, other
devices/dtypes, higher-order derivatives or later-plan components. Those
scaling items remain pending; batch8 evidence is not width/context proof.

### Final private-cache results

Both batches completed all three selectors, both contexts, two warmups and
11 paired trials per selector/context. These are the decisive isolated
confirmation measurements, not the provisional tables above.

| Batch | Context |64 step ms |128 step ms |256 step ms | Paired reduction128 (95% interval) | Paired reduction256 (95% interval) |
|---:|---:|---:|---:|---:|---|---|
| 8 |512 |1469.864 |1253.448 |1218.261 |14.92% [13.76,15.22] |**17.10% [16.52,18.17]** |
| 8 |128 |344.218 |293.129 |282.697 |14.64% [14.09,15.84] |**17.90% [14.38,19.70]** |
| 1 |512 |542.326 |321.814 |274.617 |40.31% [39.64,41.39] |**49.48% [47.45,50.00]** |
| 1 |128 |179.683 |120.583 |107.848 |32.72% [32.05,33.79] |**41.03% [37.24,41.74]** |

Each candidate wins all11 paired trials at every shape. The intended-batch
256-lane improvement remains beyond noise, with **no numerical/model work
removed and no launch reduction**. At B8/context512 the median full-step
throughput rises approximately2787→3362tokens/s (1.21x). The128-lane
specialization is correct and useful for reproducibility but not adopted
over the consistently faster256-lane path.

| Batch | Context |64 isolated F / F+B ms |128 isolated F / F+B ms |256 isolated F / F+B ms | Kernels/step (all paths) |
|---:|---:|---:|---:|---:|---:|
|8 |512 |17.950 /86.280 |8.555 /34.840 |6.269 /25.787 |5833 |
|8 |128 |5.204 /22.621 |3.019 /9.952 |2.171 /7.698 |5833 |
|1 |512 |18.266 /83.646 |8.456 /30.372 |5.937 /19.326 |5780 |
|1 |128 |5.588 /22.519 |2.926 /8.609 |2.159 /6.336 |5780 |

Final sampled pool reservations are778733888/778717952bytes at B8 and
198329664/198321792bytes at B1 (contexts512/128). The earlier memory/work
limitations and unchanged logical buffer counts still apply.

Final independent f64 gates all pass at unchanged2e-5/3e-4 tolerances:

- `final/parallel128-b1/kron-reference.json`:13 edge/strided/continuation
  cases, maximum input-gradient error7.334e-8.
- `final/parallel256-b1/kron-reference.json`:13 edge/strided/continuation
  cases, maximum input-gradient error9.154e-8.
- `final/parallel256-t128/kron-reference.json` and
  `final/parallel256-t512/kron-reference.json`:all four objectives at each
  context, including every output, state, physical score and all seven
  input/state gradients; maximum input-gradient error7.733e-8.

All paths above are relative to `artifacts/scan/`. The final paired
real-corpus preflights also pass logits/every parameter gradient/AdamW
updates before throughput at B1/B8 and both contexts; the observed errors
are unchanged from the provisional preflights. The earlier independent
CPU-model and historical-checkpoint gates passed; the parent explicitly
owns their single combined integration rerun and the workspace tests.

Final reports:

- B8: `artifacts/scan/final/historical-b8/scan-benchmark-2821741-1792867936305942024/report.json`
- B1: `artifacts/scan/final/historical-b1/scan-benchmark-2817454-1792867902717342986/report.json`
- Statistics, source/binary proof reference and queue records:
  `artifacts/scan/final/paired-summary.json`.

Queue wait is not compute time:

| Command | Execution seconds | Approximate queue seconds | Total wrapper seconds |
|---|---:|---:|---:|
|B8 paired |80.41 |492.18 |572.59 |
|B1 paired |32.52 |458.59 |491.11 |
|128-lane default f64 |0.95 |572.60 |573.55 |
|256-lane default f64 |1.06 |491.12 |492.18 |
|256-lane T512 f64 |4.72 |565.79 |570.51 |
|256-lane T128 f64 |1.43 |570.52 |571.95 |

**Final decision: adopt the explicit `fused_parallel256` selector.** The
fixed-build/private-cache confirmation independently supports the decision
at the realistic B8 workload. Keep omitted=`tensor_solver` and explicit
`fused_gpu` historical arithmetic/fingerprints unchanged; no automatic
selector switching or fallback is introduced.

