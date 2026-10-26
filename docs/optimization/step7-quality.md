# Step 7 quality study

## Controlled comparison

The study compares five causal byte-LM arms on the KronLSTM + surprise-HOLA
backbone in `configs/lm512-optimized.json`:

| Arm | Addition to optimized backbone |
|---|---|
| off | None |
| wavelet | Blocks `[0,1,2]`, `wavelet_backend: fused_gpu` |
| Lngram | Three injections at blocks `[0,1,2]`, orders `[2,3]`, 32 routes × 4 bits, memory dimension 8, `lngram_backend: fused_gpu` |
| oHC | Four-stream residual path, `ohc_backend: fused_gate` |
| all | All three additions, exactly the component selectors in `configs/lm512-step7.json` |

All arms use batch 8, context 512, four layers, width 256, four heads, key
width 32, value width 64, computational chunk 16 and HOLA capacity 16. They
use f32 on local Intel Arc MTL/Vulkan, Burn 0.22/CubeCL 0.11 without fusion or
matmul autotune. The optimized execution selectors remain
`fused_parallel256` scan, `fused_gpu_pairs` HOLA, merged projections,
`fused_gpu` convolution, `fused_norms_gates` normalization, `fused_gpu`
updates and `analytic_backward` epilogues. Omitted head/loss selectors keep
the existing native head and tensor loss.

Every arm receives **440 optimizer steps = 1,802,240 byte tokens per seed**,
with paired seeds **1, 2 and 3** and identical deterministic training example
order within each seed. AdamW uses learning rate 0.001, weight decay 0.01,
per-parameter gradient clipping at norm 1, beta1 0.9, beta2 0.999 and epsilon
1e-5. Lngram tables retain their existing optimizer group with 5× learning
rate (0.005), zero decay and the same clipping/momentum settings; this is part
of the implemented component, not a study-specific optimizer adjustment.
There is no learning-rate schedule. Parameter counts differ by architecture;
this is equal-token/equal-step, not equal-parameter or equal-wall-time training.

The budget was chosen before observing quality. At the previously measured
1.775-second all-components step median, 440 steps take about 781 seconds,
leaving the rest of each 900-second invocation for startup, final checkpoint
and held-out evaluations. The training loop's soft limit is 820 seconds; an
incomplete invocation can resume the unchanged configuration and data order.
Checkpoint cadence is set above the run length, retaining only a final
checkpoint per run. The study rotates arm order across seeds, starting with
`all, off, wavelet, Lngram, oHC` for seed 1. Each invocation obtains the shared
GPU lock and releases it before the next command. The first invocation also
held the build lock; later invocations use only the GPU lock so CPU builds
can proceed concurrently. Training tokens/s can therefore include host
contention from concurrent compilation; use the earlier paired component
benchmarks for isolated cost comparisons.

## Corpus and evaluation

The corpus and whole-document splits are unchanged from the original
[Tiny causal language model](../../RESULTS.md#tiny-causal-language-model)
study. Training books are Alice's Adventures in Wonderland, Frankenstein,
Pride and Prejudice and The Adventures of Sherlock Holmes. Tom Sawyer is
validation; Moby Dick is test. Tokenization is UTF-8 bytes 0–255 plus BOS 256.
`data/books/manifest.json` pins the extracted document hashes and source
metadata, including validation document hash
`1dade7b8e9e86fae3dd0173c058501c07881229b824f23947641ec099482d3ef`.
Each sampled window starts from BOS, with no context before that window, and
scores all 512 next-byte targets. It is not a full-book streaming evaluation.

Two fixed validation samples are reported separately:

1. **Original sample:** four deterministic, nonoverlapping midpoint-stratified
   512-byte windows, **2,048 scored bytes**. The unchanged `certify` command
   reloads final checkpoints with the existing synchronized logical-value/ID
   checks, repeats evaluation exactly and regenerates a repeat-checked seeded
   untrained baseline. The training/certification evaluation identity remains
   unchanged.
2. **Larger sample:** 128 deterministic, nonoverlapping midpoint-stratified
   512-byte windows, **65,536 scored bytes**. The same final checkpoints are
   evaluated with a separate `evaluate` command, which validates original
   checkpoint identity and uses the same guarded loader. Trained and freshly
   regenerated untrained evaluations are each repeated exactly. This sample
   drives the interpretation, not the four-window sample.

The existing selection algorithm is `stratified-midpoint-disjoint-v1`. Tom
Sawyer has 405,626 extracted bytes, hence 792 complete 512-byte windows. For
sample size `N`, selected window index `i` is
`floor((2*i + 1) * 792 / (2*N))`; its byte offset is that index times 512.
Changing `N` changes the strata, so the 128-window sample is not an extension
that necessarily contains the original four windows. Every arm and seed uses
the same fixture for each sample size. The stored fixture hashes identify the
exact inputs and targets. This remains validation on one held-out literary
book; it is not a broader benchmark or a test-set result. Automatic training
and certification test/retrieval diagnostics are retained in artifacts but
are not used for architecture selection.

NLL is in nats per byte, lower is better. Seed spread is population standard
deviation (`sqrt(sum((x - mean)^2) / 3)`), matching the existing experiment
summary. Paired differences are `addition - off` for the same seed; negative
means lower validation NLL. Three seeds do not support a general architecture
ranking. Seed spread describes initialization/data-order variation, not an
independent-sample confidence interval over books or byte targets.

## Reproduction and artifacts

From this workspace root:

```sh
../build.sh s7q
bun scripts/step7-quality.ts study
```

The runner generates exact configurations under
`artifacts/step7-quality/configs/`, then invokes existing `train` and `certify`
commands and the new evaluation-only command. Every GPU child runs as:

```sh
flock /tmp/delta-block-lm-gpu.lock \
  bun scripts/step7-quality.ts locked LABEL COMMAND CONFIG [OPTIONS]
```

Inside the lock, the runner applies `timeout --signal=TERM --kill-after=5s
900s` to `/tmp/delta-bins/s7q/delta-experiment`. Per-command stdout, stderr,
exact arguments, exit code and resource-held elapsed time (excluding lock
wait) are retained under `artifacts/step7-quality/commands/`. Resolved source
config, binary, Cargo.lock and corpus-manifest hashes are in `study.json`.
`summary.json` records per-run metrics, seed summaries, paired differences,
checkpoint evaluation paths and total resource-held command time. Original
metrics and certificates are retained unchanged under each arm directory.

The only Rust changes add `evaluate`/`--eval-examples` routing in
`crates/experiment/src/main.rs` and a separate held-out checkpoint evaluator
in `crates/experiment/src/train.rs`. No training path, optimizer, default
configuration serialization or certification contract changes. Command
behavior is documented in `crates/experiment/README.md`; the config/command
runner is `scripts/step7-quality.ts`. No `RESULTS.md` or `PLAN.md` edits are
part of this deliverable.
