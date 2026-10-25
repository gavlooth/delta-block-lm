# SBD-style recurrent block denoising

## Objective and input realization

This is the recurrent adaptation of [Set Block Decoding, arXiv:2509.04185](https://arxiv.org/html/2509.04185v1), equations 8 and 14. It is not the paper's Transformer implementation or a reproduction of its speedups.

Step 8 adds temporary in-block reads through the existing HOLA read, selected by `sbd.in_block_reads: "hola_read"`. There is no separate block-attention operator and no new parameter. `"none"` is the omitted default and preserves the previous SBD serialization and fingerprints. `hola_read` requires HOLA in every layer; a no-HOLA arm is rejected rather than using another path.

With `sbd` absent, the architecture, config serialization, random initialization order, fingerprints, causal training and checkpoints keep their historical behavior. `configs/lm512-sbd.json` enables the Step-7 architecture with SBD. `sbd.feedback` defaults to `soft_feedback`; `hard_mask` is the comparison. Candidate width, block lengths, HOLA capacity, computational chunk length and commit count are independent.

Enabled-SBD training and pilot fingerprints retain the semantic revision introduced after dense attention removal. Selecting `hola_read` additionally changes the serialized model identity; omitted/explicit `none` keeps the previous SBD identity. Old dense-SBD checkpoints remain incompatible. Configurations without `sbd` retain their hashes.

For each training sequence, choose one entry of `sbd.block_lengths` uniformly, then partition its target positions with that length, including a short final block. A singleton entry is a fixed-length condition. The variable condition can use `[1,4,8,16]` or the paper's `[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]`. Draw one sequence-level mask probability uniformly in (0,1), and independent Bernoulli masks. Unmasked targets inside the active block are explicit observed evidence, including targets to the right of a masked position. No other future targets are visible.

The objective is clean causal mean NLL plus the sum of masked block NLLs divided by the number of clean targets. This is equation 14's sum objective with a common token normalization, not an independently mask-normalized auxiliary loss. All-observed draws contribute zero masked loss. A detached first noisy pass supplies top-k candidate IDs, original probability weights and retained mass. The second pass uses Step 3's weighted embedding plus the mask embedding, with visibility `1 / soft_steps`. Selection is discrete; the second pass's embedding values remain differentiable. The mask is a separate trainable vector. Teacher probabilities are detached deliberately, rather than training through a proposal-generation graph. Hard-mask training skips this teacher pass.

## Positions and visibility


Training retains normalized sublayer inputs from the **ordinary, unsegmented clean forward**, which also supplies the clean loss. A second O(T) state-extraction pass scans disjoint clean segments with carried recurrent state and retains the requested block-start states; it never replays each clean prefix. The configured computational chunk size stays 16, including a short segment ending inside an original chunk. HOLA selection still uses the original chunk grid, not those state-extraction segment ends. Per-block noisy work projects only active inputs plus three cached convolution taps. Wavelet preparation retains each clean approximation level once; noisy smoothing uses the required cached level history. Lngram uses only its maximum suffix plus smoothing lookback. The segmented extraction currently has a host loop per block; reducing these dependent launches is an efficiency task, not a change in visibility.
A target block starting at target index `s` sees exactly `example.input[..s+1]`: BOS followed by all clean target tokens strictly before `s`. Noisy target `s+j` occupies the same input-position identity `s+1+j` as its clean token counterpart. There is no second positional coordinate or repeated learned position embedding in this model.

Each mixer computes its clean-prefix recurrence and uses that final state as the initial state of every noisy query's **independent, temporary one-step recurrence**. Temporary states are discarded.

For `hola_read`, each head reads `[selected HOLA slots after completed chunks | clean unfinished chunk cut..start | active noisy block]` and the existing learned sink in **one softmax**, with the existing `1/sqrt(key_dim)` scale and sigmoid HOLA gate. Active keys use HOLA's key RMS normalization on the noisy convolved raw K; values use noisy convolved raw V, as clean cache values do. Every valid query sees every valid block key, including itself and keys to its right. Padded keys are excluded and padded queries return zero read output. The temporary block K/V never enters cache selection or the clean path. With `none`, the final segment is absent. Starts inside a computational chunk retain every unfinished clean key without ranking or truncation.

Wavelet taps crossing the start use clean-prefix activations and taps within the block use noisy activations. Lngram suffix codes and left-only smoothing use the same rule. Both remain left-to-right inside the block; they do not supply a bidirectional attention path. oHC stream initialization/read/write/readout is per token, including the mask/candidate inputs. No noisy stream state persists between passes. Every example is one document, with BOS/reset at its start. Training blocks cannot cross examples. Padding freezes recurrence and is excluded from HOLA reads, convolution and Step-7 history.

The length-one limit is defined explicitly: its noisy prediction is the clean next-token computation on the prefix, bit for bit. There is no current-target embedding in this limit, even for an observed singleton. This avoids changing the causal LM's next-token convention into a masked-current-token convention. Lengths greater than one use the block denoiser.

### Visibility proof sketch

The public block-forward function accepts a clean prefix and explicit soft/observed block state, not a full clean target sequence. At layer zero, these are its only token-dependent values. Inductively, clean-prefix sublayers are causal; noisy recurrence reads only that prefix's state and its own noisy projection; short convolutions, wavelet and Lngram read that prefix or the active representation; HOLA reads clean-prefix positions and, when selected, the explicitly supplied noisy block. Per-token normalization, MLP and oHC introduce no other token inputs. Thus changing a clean input/target at or after the block start cannot change predictions unless explicitly supplied as observed block evidence. Changing an observed token inside the block can change other block positions through the same HOLA softmax.

## Generation and replay

`sbd-generate` loads a checkpoint with the existing validated loader. It uses a held-out clean prompt (up to 17 input tokens), initializes an all-masked block, and repeatedly predicts against the same immutable clean token snapshot. Soft updates keep candidate distributions on uncommitted positions; finalized positions become immutable observations. Greedy assignment is deterministic. `commit_count` selects the number of lowest-entropy positions per pass. With `entropy_threshold` set, equation 8 selects the largest `s >= 1` whose first `s-1` sorted entropies sum to at most the threshold, in nats. Ties use ascending position, then ascending token ID.

Only after the entire block is finalized does the clean-prefix object append it and replace its clean logits/state reconstruction, once per block. The implementation reconstructs the clean model from the complete prefix rather than offering an optimized incremental state API. Denoising likewise reconstructs clean-prefix tensors without mutating the persistent token history. This is a correctness-first replay implementation, not a decoding acceleration result. Reported finalized tokens/s includes denoising, host entropy selection, soft updates, synchronization and replay; it excludes initial prompt prefill.

## Work and retained tensors

Clean preparation is O(T), plus one state-extraction scan segment per block per layer. Prefix-reconstruction inference is O(prefix length) per denoising pass and per completed-block replay. The active HOLA segment adds O(B H L² (K+V)) read work per pass; this is quadratic only in the active block length, not the prefix length.

`sbd.noisy_read_backend` defaults to `"tensor"`; `"fused_gpu"` explicitly selects the CubeCL noisy-read kernel with analytic query/K/V/sink derivatives. The sigmoid gate remains the existing tensor operation. Cache gathers and segment concatenation still use the tensor graph. The kernel supports f32 K32/V64, at most 16 noisy queries and 127 assembled keys, on explicit non-fusion GPU execution; unsupported shapes are rejected without a CPU alternative. This execution choice adds no model parameter.

Paired complete-step cost tables, launch counts and sampled memory are recorded in [RESULTS.md](../RESULTS.md#step-8-recurrent-block-denoising-through-hola). The complete-step benchmark is `sbd-benchmark CONFIG --warmups 2 --trials 11`. It measures the real SBD loss/backward/checks/clipping/AdamW path, not clean-only training, and performs no evaluation or checkpoint writes.

The B8/T512 attempt reached its 3100-second deadline before one three-mode iteration report; it provides no paired throughput result. The accepted B1/T512 diagnostic measured 128 noisy block evaluations and 512 layer HOLA reads per step. Its synchronized noisy-HOLA forward sections used 10.77% / 11.74% / 9.18% of observed complete-step time for `none` / tensor `hola_read` / fused `hola_read`; these are forward-only shares, not full read/VJP cost or ordinary paired-step timing. `sbd-diagnostic CONFIG` performs one warm and one observed step per mode.

The accepted B1/T512 two-warmup/11-pair session measured an in-block tensor-read cost of +601.87 ms per complete step (95% paired interval [+328.46, +1508.97]). Replacing that read by the mathematically identical fused kernel saved a paired median 2390.35 ms [+1667.56, +3176.01], faster in 10/11 pairs, so `configs/lm512-sbd.json` explicitly selects `fused_gpu`. The tensor default remains unchanged. Cache gather and segment concatenation were not replaced, and no SBD per-block loop optimization was attempted.

Historical preparation tensor counts for the earlier dense-attention implementation's canonical B8/T512/L8, four-layer H4/K32/V64 configuration were: retained block-start recurrent states, 64 MiB of f32 values; references to normalized clean sublayer inputs, 40 MiB; cached four-level wavelet approximations, 48 MiB; raw short-convolution projections, 32 MiB; HOLA K/V projections, 24 MiB; and cache-position metadata, about 256 KiB. These were structural tensor-value counts, not an incremental or peak allocator measurement; autograd intermediates, scan checkpoints, model parameters and AdamW state add memory. They have not been remeasured after removal. The historical training metrics below sampled the device pool after forward/backward/update and reported that reservation with its limitations; they do not measure the current architecture.

## Historical command examples

These workspace-root commands record the earlier implementation's entry points, not instructions for a new run. The dense-SBD checkpoints in the generation/evaluation examples are incompatible with current fingerprints and are now rejected; current checkpoint-based commands require a checkpoint produced under the revised SBD semantics.

```sh
../build.sh s8
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 900s /tmp/delta-bins/s8/delta-experiment sbd-reference configs/lm512-sbd.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 900s /tmp/delta-bins/s8/delta-experiment sbd-train configs/lm512-sbd.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 900s /tmp/delta-bins/s8/delta-experiment sbd-generate configs/lm512-sbd.json --resume artifacts/step8/lm512-soft/kronlstm-hola-seed1/checkpoint-1
flock /tmp/delta-block-lm-gpu.lock timeout 900s /tmp/delta-bins/s8/delta-experiment sbd-evaluate artifacts/step8/pilot-variable.json --resume artifacts/step8/pilot-variable/kronlstm-hola-seed1/checkpoint-4
```

The current `sbd-reference` command includes an independent scalar f64 noisy-HOLA forward/full-VJP oracle and checks both the tensor and explicit fused GPU read. GDN-2 on GPU uses `fused_gdn2` with the arm mixer set to `gdn2`. CPU uses tensor execution selectors. Autotuned matmul is not part of this variable-shape path. The historical commands above do not authorize new comparisons or sweeps.

## Verification and measurement scope

Existing sequential f64 references cover recurrence, short convolution, HOLA, wavelet, Lngram and oHC. The noisy-read oracle covers derivatives of queries, cache/clean-chunk/block keys and values, sink and sigmoid gate, with central finite differences on the oracle. Fixtures include padding, empty cache, a chunk-aligned start and an unfinished chunk. Complete-model checks cover both mixers, HOLA on/off and Step-7 on/off; `hola_read` is rejected without HOLA, and `none` still runs. A prefix of length 19 puts the block start inside chunk 16. The checks include CPU/Vulkan logits and all-parameter gradients, cached/reconstructed logits and gradients, lengths 1/4/8/16, future-token perturbations, intended observed-token communication, padding, frozen snapshots and replay. Tolerance remains `2e-5 + 3e-4*abs(reference)`.

## Verification after attention removal

The Vulkan-enabled workspace build and 44 experiment + 7 kernel + 16 mixer tests passed; one GPU-only test was ignored. The CPU `sbd-reference` executable passed the retained model checks at unchanged tolerances: `artifacts/step8/no-block-attention-cpu/sbd-reference.json`. Compatibility tests reject historical dense-attention SBD checkpoint identities while preserving omitted-SBD fingerprints. No GPU benchmark or training study was run for this removal.

## Historical results — removed dense-attention implementation

**All test counts, numerical errors, training/generation metrics and memory/timing measurements in this section describe the earlier implementation, which included dense active-block attention (with a causal special case at length one). They are retained as historical evidence only, not evidence for the current architecture after removal.** The attention-only oracle and its 11 tests mentioned below no longer exist in the current implementation.

Historical CPU verification passed: 51 experiment tests, 7 kernel tests and 16 mixer tests (one explicitly GPU-only test ignored). The then-present attention oracle contributed 11 tests, including full input VJP central finite differences. The complete CPU SBD gate passed all eight mixer × HOLA × Step-7 combinations in 4.14 seconds: active-attention f64 output max absolute error 6.26e-8, cached-vs-reconstructed noisy logits 1.08e-7 and all-parameter gradient error 6.05e-9. Evidence: `artifacts/step8/gates-cpu/sbd-reference.json`.

After correcting variable-length evaluation, the experiment suite passed 52 tests, including a regression that checks every configured length is evaluated even with one held-out example. Fixed-length metrics are unchanged. Variable block metrics from the original training `metrics.json` used L=1 only and are retained as historical evidence; the corrected report is `artifacts/step8/pilot-variable/sbd-evaluate.json`.

Intel Vulkan f32 verification also passed all eight combinations at the unchanged tolerance: scalar-f64 attention max output error 5.66e-8 and max input-gradient error 2.34e-8; complete noisy-model GPU-vs-CPU logits 1.15e-7 and every-parameter gradients 1.26e-8. Cached-training vs prefix-reconstruction logits were exactly equal on this fixture, with max gradient difference 1.16e-9. Future perturbations (including perturbations in the full clean sequence used to prepare training snapshots), the position-19 boundary inside chunk 16, length-one causal logits, frozen snapshots and replay logits/recurrent states were bit-identical in every case. Evidence: `artifacts/step8/lm512-soft/sbd-reference.json`.

After integration with Burn 0.22.1 / CubeK 0.3.1, the Vulkan-enabled workspace built, all 52 + 7 + 16 tests passed (one GPU-only test ignored), and the CPU `sbd-reference` executable passed: `artifacts/step8/integrated-cpu/sbd-reference.json`. The Vulkan reference and training smoke numbers below were measured before that dependency upgrade; they were not repeated as an upgrade benchmark.

### Historical tiny training smoke, not a quality comparison

All conditions use the full Step-7 width/depth and all three Lngram injections: 7,082,423 parameters, KronLSTM+HOLA, width 256, four layers, but **context 32, batch 1, seed 1, only 128 training byte tokens / four AdamW steps**. Each uses the same data, initial weights, optimizer, candidate width 4 and fixed commitment count 2. Initial clean validation NLL is 5.580573 in every condition. Evaluation uses one 32-token held-out window. This demonstrates end-to-end training with finite gradients, not useful language quality or an advantage for one feedback choice.

| Training condition | Clean NLL | Block NLL | Block accuracy | Finalized tokens/s | Training loop seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| L=8 soft feedback | 4.575052 | 4.275871 | 9.375% | 41.19 | 6.004 |
| L=8 hard mask | 4.565683 | 4.276025 | 9.375% | 45.14 | 2.384 |
| L=1 soft feedback | 5.033420 | 5.033420 | 9.375% | 16.74 | 0.895 |
| L=4 soft feedback | 4.675298 | 4.278239 | 9.375% | 37.42 | 7.337 |
| L=16 soft feedback | 4.595955 | 4.289566 | 9.375% | 48.46 | 1.999 |
| Variable [1,4,8,16] soft feedback | 4.679817 | 4.383110 | 9.375% | 16.56 | 6.976 |

Variable block NLL/accuracy above use the corrected evaluation-only report, covering all four configured lengths on the same held-out window (128 target predictions). Its generation episode cycles through configured lengths and finalized eight tokens in five passes; it is not the same decoding workload as a single fixed L=8 block.

Generation measures a single short episode: eight finalized tokens except L=16 (sixteen), including full-prefix reconstruction and replay, excluding prompt prefill. These are not paired throughput benchmarks; remaining first-use compilation, host work and small samples affect them. For example, the L=8 soft checkpoint generated spaces, consistent with the deliberately tiny training budget. Sampled peak device-pool reservation was 149.59–281.59 MiB across the conditions; it is not an exact allocation peak. Total measured training-loop time for all six conditions, including variable-length training, was 25.595 seconds. Every GPU command had a 900-second execution deadline and released the lock before the next command. Most observed command wall time was waiting for another GPU user, not executing these pilots.

On the L=8 soft checkpoint, entropy threshold 0.5 nats exercised equation 8 through the checkpoint-loading CLI: eight tokens, eight denoising passes, exactly eight clean replayed tokens, 22.27 finalized tokens/s. Evidence: `artifacts/step8/generation-entropy/sbd-generate.json`.

Configs are `artifacts/step8/pilot-{soft,hard,l1,l4,l16,variable}.json`. Each final-only checkpoint and original metrics file is under the matching `artifacts/step8/pilot-*/kronlstm-hola-seed1/`. The training commands were `sbd-train` with those six configs; entropy decoding used `sbd-generate artifacts/step8/generate-entropy.json --resume artifacts/step8/pilot-soft/kronlstm-hola-seed1/checkpoint-4`. No intermediate checkpoints or large dumps were retained.

These historical runs do not establish current-architecture full-context throughput or meaningful quality comparisons. Current work is restricted to numerical checks, execution smoke checks and paired timing/memory/launch measurements.
