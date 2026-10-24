# Input pipeline and checkpoint IO evaluation

## Status and scope

The implementation was verified by the integration owner, and the measurements are recorded under "Measured results" below. The integration owner exercised two revisions early on and found GPU recomputation nondeterminism, documented below; the final run passed all checks. Steps 1–6 only.

`io-benchmark` uses the configured first arm, model width/depth/selectors, optimizer, real corpus and seed. It evaluates B1/B8 × contexts 128/512. It rejects non-LM configurations, an overriding arm/seed/resume, more than three warmups, or trials outside 1–8. The default is one warmup and three measured trials, plus one separately recorded first observation. The wall-clock admission deadline is `min(max_seconds, 900)` seconds; an external timeout must bound a single blocking operation. Warmups and first observations are not counted as measured trials.

### Designs

1. **Saved baseline:** the previous `Batch::new` implementation is preserved verbatim in behavior as `io_benchmark::legacy_batch` (flat-map/collect input, target and weight; host reset vector; device ones for validity). It is diagnostic-only, not a second training selector.
2. **Exact-capacity owned packing:** `data::HostBatch` allocates the three flattened vectors with the known B×T capacity and copies each row once. `Batch::new` delegates to this packing/upload split. This removes avoidable geometric vector growth, not the necessary owned upload payload. Measured: about 0.3 ms less pack+upload per batch at B8/T512, with no step-level difference outside noise (see "Measured results"). All training/evaluation callers automatically retain the same data ordering, reset/valid masks and tensor types. No config/checkpoint fingerprint field or historical selector was added or changed.
3. **Same-step resident fixture:** prebuilt next-row tensors and identical pre-update model/optimizer state are prepared outside the full optimizer-step clock. This bounds removable input overhead without asynchronous queues. It is not advertised as a valid way to train repeatedly on stale rows.

Checkpoint comparison is the **original atomic save plus guarded, validated file restore** versus **resident same-state continuation**. The latter has no durability and is an overhead bound, not a candidate replacement for checkpointing. No validation, checksum, synchronization, optimizer-state check, config/data check, collision check, or atomic rename is removed. `train::checkpoint` and its unchanged `State` layout are exposed crate-locally so the harness exercises the real transaction instead of imitating it. `checkpoint.rs` remains unchanged.

### Attribution and correctness

The command separates:

- initial corpus file loading, manifest/content checksum and split validation;
- deterministic next-step corpus-row generation;
- exact-capacity host packing;
- upload of prepacked payloads;
- complete old/new packing plus upload;
- paired full model forward/loss/readback/backward/checked-gradients/optimizer steps for all three input designs;
- original complete atomic checkpoint save (model, optimizer, state, rename and latest pointer);
- state/config/data/counter and latest-pointer validation;
- fresh model target construction;
- guarded model load **including its mandatory logical-value/ID validation**;
- standalone logical model validation;
- optimizer load **including complete canonical moment/ID/counter/metadata/config validation**;
- the restored model/optimizer's next full step.

The separate validation time is diagnostic attribution, **not an additive part of the already validated load time**. Timing has synchronization before and after every region. Empty-queue synchronization overhead is included even in host-only timing; very small host samples must not be overinterpreted. OS page cache is uncontrolled; warm file loads are not cold-disk measurements. The first optimizer step is the first backward/update observation, not a fully cold process measurement (initial evaluation ran first). Atomic rename is the existing completion contract; no stronger fsync/power-loss guarantee is claimed.

Every uploaded batch is checked elementwise against independent row/index expectations for input, target, weight, valid and reset, including exact float bits. A CPU unit regression additionally covers distinct rows, BOS tokens, fractional/sparse weights, and row-start resets. Every full-step design starts at exactly the same model/optimizer state and uses the same next deterministic row indices. **Saving/restoring the same tensors remains byte-exact**, including model parameters and optimizer moments, shapes, dtypes, IDs, counters, metadata and config. Independently recomputed optimizer steps, including the step after restoration, use the existing `profile.rs` tolerance `abs(actual-reference) <= 2e-5 + 3e-4 * abs(reference)` for each finite f32 parameter/moment element. Tensor names/shapes/dtypes/IDs, scalar counters, metadata and non-f32 payloads remain exact even across recomputation. Reports expose maximum absolute differences, every bitwise-different tensor, and both hashes. The initial checkpoint has real nonempty optimizer moments from a genuine model step and a real initial heldout evaluation.

The whole checkpoint `state.json` must equal the expected configured run identity and counters. `latest.json` must point at the completed directory with the same step/tokens. Existing training collision and resume checks are untouched. All scratch checkpoints and reports live in a fresh `artifact_dir/io/PID-TIMESTAMP/` subtree, never in original arm run directories. Source checkpoint and continuation-reference files remain available as evidence; their aggregate disk use scales with model size and trial count.

## Integration queue (reference commands; the measured run used `artifacts/integration/io-optimized.json`)

Keep the integration owner's established feature combination; the examples below use the existing default CPU plus Vulkan features, with fusion/autotune off as in `lm512.json`. Do not run this concurrently with another GPU command.

First, in the final integration workspace, create an explicit benchmark-only config using Bun. This does not edit any historical config or require a pilot budget file:

```sh
bun -e 'const c = await Bun.file("configs/lm512.json").json(); c.budget_file = null; c.train_tokens = 8192; c.max_seconds = 900; c.artifact_dir = "artifacts/optimization-io"; c.cache_dir = ".cache/io"; await Bun.write(".cache/io-benchmark.json", JSON.stringify(c, null, 2));'
```

The width 256, four-layer model, real corpus, first GDN2 arm, precision and configured kernel paths are retained. The command's four explicit B/T overrides are written into each report. `train_tokens=8192` is a diagnostic checkpoint identity budget divisible by all four shapes; timing executes only the bounded fixture steps, not that training budget.

Queue these with the owner's single final verification batch, not mid-integration:

```sh
timeout 300s cargo test -p delta-experiment --features vulkan host_batch_preserves_row_order_sparse_weights_and_resets
timeout 300s cargo test -p delta-experiment --features vulkan checkpoint::tests
```

After the owner has built the combined `delta-experiment` binary with the same features, the bounded minimum acceptance/first-observation smoke is:

```sh
timeout 960s target/debug/delta-experiment io-benchmark .cache/io-benchmark.json --warmups 0 --trials 1
```

The measurement queue command is:

```sh
timeout 960s target/debug/delta-experiment io-benchmark .cache/io-benchmark.json --warmups 1 --trials 3
```

Use the integration owner's actual target directory if it differs. A timeout or failed exact check is evidence of an incomplete evaluation, not a performance result. The command writes its own unique result, emits JSON to stdout, and fails command completion if `completed` is false. Top-level and per-shape partial reports retain completed samples on a later deadline/failure. Failure during the initial corpus open is an ordinary command error.

## Expected evidence schema and decision

Top-level `result.json` contains `schema`, `command`, `completed`, `passed`, `error`, runtime device identity, complete original config, selected arm, warmup/trial counts, output directory, corpus load/validation milliseconds, corpus manifest SHA256 and four shape `reports`.

Each shape report contains:

- full shape-specific config, B/T, parameter count, fixture SHA256;
- first optimizer-step and first IO observations, excluded from warm summaries;
- `first_checkpoint`: path, complete state, model and optimizer file SHA256, canonical optimizer signature and initial save/load samples;
- `timings.<section>.samples_ms` and `median_ms`, including all three same-step full optimizer-step designs;
- per-iteration exact batch and checkpoint parameter/ID/optimizer-state checks, numerical recomputed-step comparisons, and cold/warmup/measured phase;
- `work`: explicit packed host payload (16×B×T bytes), resident batch logical payload (20×B×T bytes for the f32/i32 workload), retained checkpoint file bytes, four host vectors, five tensor creations, and sampled peak pool-reserved bytes.

Pool reservation is not physical/live peak memory and includes correctness fixtures. Logical payload counts do not claim allocator capacity, backend-dependent integer storage, actual PCIe traffic, GPU dispatch count, or process RSS; these are not instrumented. Source and optimizer fixture snapshots add retained memory outside the step clock. Do not sum isolated section medians into a claimed measured full-step cost.

Before scaling, the B8 samples set the IO budget: compare the paired old/new total steps and the resident overhead bound, report medians and raw spread, and decide whether row generation/upload or validated checkpoint cadence is material relative to the full step. Keep exact-capacity packing's allocation rationale distinct from its measured speed. Consider prefetch only if measurements show enough headroom to justify queue ownership, lifetime, deterministic ordering and failure-propagation complexity. Keep checkpoint validation and atomic completion regardless of their cost. The measured numbers and the resulting decisions follow.

## Measured results

Run by the integration owner on the heuristic build (binary hashes in `artifacts/integration/final2/binaries.sha256`; stdout in `artifacts/integration/final2/io-benchmark.stdout`). Config `artifacts/integration/io-optimized.json`: `lm512-projections` projections, scan `fused_parallel256`, HOLA `fused_gpu_pairs`, conv `fused_gpu`, norm `fused_norms_gates`, batch size overridden per shape, 1 warmup, 3 measured trials. All exact batch and checkpoint checks passed; recomputed steps used the tolerance described above.

Median milliseconds:

| Shape | Corpus rows | Legacy pack+upload | Exact-capacity pack+upload | Full step, legacy | Full step, exact-capacity | Full step, resident | Atomic save | Guarded model load (with validation) | Optimizer load (with validation) | Restored continuation step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1/T128 | 0.22 | 0.78 | 0.72 | 157.25 | 159.36 | 131.56 | 263.65 | 167.64 | 173.37 | 250.37 |
| B1/T512 | 0.07 | 0.98 | 0.68 | 356.93 | 323.48 | 297.50 | 278.55 | 171.13 | 184.05 | 404.75 |
| B8/T128 | 0.06 | 0.92 | 0.69 | 256.82 | 237.66 | 302.69 | 279.40 | 152.81 | 171.27 | 362.25 |
| B8/T512 | 0.11 | 1.18 | 0.90 | 959.63 | 939.59 | 1035.50 | 266.70 | 178.24 | 186.84 | 1067.70 |

The checkpoint (model, optimizer and state) is 28.61 MB. Initial corpus load and validation took 3.30 ms.

With only three trials, full-step spreads are about ±100 ms. Differences between the three input designs at the step level (including the resident fixture, which is not consistently fastest) are within that noise and must not be read as speedups. The isolated pack+upload medians are the more reliable signal.

### Decisions

- **Input pipeline:** packing plus upload is about 1 ms against a roughly 940 ms B8/T512 step, under 0.2%. No asynchronous prefetch is added; the ordering, lifetime and failure-propagation cost is not justified.
- **Exact-capacity packing:** kept. It saves about 0.28 ms per step at B8/T512 (1.18 ms to 0.90 ms), small but free of added complexity.
- **Checkpointing:** no code change. Atomic save costs about 265–280 ms at every shape. At `checkpoint_every` 4 and an optimized step of about 506 ms, a save adds roughly 13% per step on average. For long runs, raise the checkpoint interval rather than introducing an asynchronous writer; validation and atomic rename stay as they are.
- **Limits:** page cache was uncontrolled, so loads are warm-file times; the sample count is small; the step-level comparison cannot rank the input designs.


## Integration failure attribution and fixture follow-up

The integration owner exercised the first implementation with
`artifacts/integration/io-optimized.json` and retained stdout at
`artifacts/integration/batch2/io-optimized.stdout`. It failed at B1/T128 with
a `head.weight` hash difference (same parameter ID/shape/dtype). The shape
directory contained only `continuation-reference-0.burnpack`: the failure
was the **input-design next-step comparison, before atomic checkpoint save
or model/optimizer restore had executed**. The generic `validate_model`
error text called this a checkpoint-load failure, but that is not the
operation this harness was performing. This evidence does not establish
a checkpoint product defect.

Static inspection found no off-by-one step or tied embedding/head: all
designs start from the post-first-step `base`, use next-step row indices,
and the model constructs separate embedding and head parameters. However,
the original harness unnecessarily restored the optimizer from bytes for
each nominally resident comparison, without verifying that input state.
It also dropped locally generated batches before the timing wrapper's
completion barrier, unlike the training loop. The follow-up uses clones
of the actual resident optimizer and retains each timed batch through
that barrier. These remove harness confounders; they are **not a measured
proof of the cause of the observed hash difference**.

The follow-up additionally saves and self-validates an immutable resident
base record, checks model/optimizer equality before each design and
retained-state equality afterward, verifies the actual timed batch, and
self-validates the first continuation record while its source remains
alive. These source-state and serialization checks remain bit-exact and
outside timing. Cross-design recomputation reports are saved separately
as `comparison-MODE-ITERATION.json`; out-of-tolerance failure also retains
the actual candidate record. Error contexts distinguish source serialization,
pre-step state, retained-state mutation, input-design continuation,
atomic save, guarded restore, and restored continuation.

No product behavior or checkpoint safety code was changed by this
follow-up, and no product regression test was added without an established
product defect. The existing failure remains evidence, not a passing
measurement. The integration owner should rebuild and run exactly:

```sh
timeout 960s target/debug/delta-experiment io-benchmark artifacts/integration/io-optimized.json --warmups 0 --trials 1
```

### Observed GPU recomputation nondeterminism

The integration owner's rerun with source `ad4813a5` failed at
`B1 T128: iteration 1, legacy: input-design next-step mismatch before
checkpoint transaction`. The two `head.weight` hashes were exactly the
previous run's pair, now swapped:

- `a6d12a12652f11854407c4f10fbc1cdb8ec2cf539e5b4d5560db3e2f902d590b`
- `03f554c414cadcedaeb44c1cab7d5613ce5d22c454555f970992ff7b30d74fb3`

The initial-state, actual batch, retained-state and serialization checks
had passed. Thus independently recomputing the same GPU step can produce
two bitwise outcomes for `head.weight`; this is not evidence that saving
and restoring unchanged tensors loses information. A vocabulary-head
gradient reduction/matmul ordering difference is a plausible mechanism,
**not an established kernel-level root cause**.

The follow-up uses the project's existing profile numerical contract for
recomputed f32 parameter and optimizer-moment values only. The direct
checkpoint save/load validations are unchanged and byte-exact. Each
recomputation report contains `atol`, `rtol`, `max_abs_diff`,
`bitwise_equal`, and `bitwise_differing_tensors` (tensor name, both SHA256
hashes, per-tensor maximum difference and numerical pass status). Scalar
counters, IDs, shape/dtype, metadata and non-f32 payloads remain exact.
Nonfinite f32 values fail even if their bytes agree. Both input-design
comparisons and restored-next-step comparisons retain this evidence
outside measured regions.

The follow-up includes a CPU record-comparator regression covering
roundoff, out-of-tolerance values, changed IDs/counters and nonfinite
values. This tests the harness's comparison contract, not a conjectured
checkpoint product bug. The final integration run exercised it together with the full benchmark (see Resolution below).

Integration-owner comparator check (retain the established feature set):

```sh
timeout 300s cargo test -p delta-experiment --features vulkan recomputed_equivalence_tolerates_roundoff_not_identity_or_state_changes
```

### Resolution

After the harness change, the integration run completed and passed every check. Recomputed steps are compared at `2e-5 + 3e-4 * abs(reference)`; checkpoint save and restore stays byte-exact. The earlier failures are therefore explained as run-to-run GPU nondeterminism in independently recomputed steps (the swapped `head.weight` hashes above), not a checkpoint defect. The kernel-level cause of that nondeterminism was not investigated. Measured results are in "Measured results" above.

