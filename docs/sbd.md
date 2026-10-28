# SBD-style recurrent block denoising

## Objective and input realization

This is the recurrent adaptation of [Set Block Decoding, arXiv:2509.04185](https://arxiv.org/html/2509.04185v1), equations 8 and 14. It is not the paper's Transformer implementation or a reproduction of its speedups.

Step 8 adds temporary in-block reads through the existing HOLA read, selected by `sbd.in_block_reads: "hola_read"`. There is no separate block-attention operator and no new parameter. `"none"` is the omitted default and preserves the previous SBD serialization and fingerprints. `hola_read` requires HOLA in every layer; a no-HOLA arm is rejected rather than using another path.

With `sbd` absent, the architecture, config serialization, random initialization order, fingerprints, causal training and checkpoints keep their historical behavior. `configs/lm512-sbd.json` enables the Step-7 architecture with SBD. `sbd.feedback` defaults to `soft_feedback`; `hard_mask` is the comparison. Candidate width, block lengths, HOLA capacity, computational chunk length and commit count are independent.

The optional `sbd.joint_output: "codpula"` replaces the multi-token noisy output with a 32-state joint HMM over the full 257-symbol vocabulary; omitted or `"factorized"` retains the behavior described here and its historical serialization. Codpula feeds joint token marginals into sparse feedback, trains conditional joint block NLL, and can use `commitment: "dependence"` with `dependence_threshold: 0.05` and seeded joint proposals. The exact causal length-one bypass is unchanged. Decisions, inference, optimizer, numerical checks and cost measurements are specified in [Codpula](codpula.md); `configs/lm512-codpula.json` enables the new model.

The optional `sbd.cocommit` adds one learned marker over the full saved hidden representation and a shared marked-tail assignment pass; omission preserves historical model/config identity. `configs/lm512-cocommit.json` enables it with Codpula. The last complete block is the initial tail, and oHC uses independent marker entries for its four streams. Both branches share the differentiable pre-tail input and immutable clean-prefix snapshots. The marked branch is supervised on its sampled unresolved subset; factorized branches also preserve visible tokens with weight `.2`, while Codpula uses the exact marked-subset joint event and marginalizes unmarked unresolved positions. See [CoCommit](cocommit.md) for decisions, checks and measured execution cost.

The optional execution setting `sbd.rows_per_pass` processes every full-batch row in explicit groups, sums scaled gradients and unscaled circuit flows/counts, then performs one clipping/AdamW/circuit update. Full-batch mask and CoCommit-bundle sampling occurs before partitioning. Omission preserves the historical one-pass path and serialization. The CoCommit B8 profile uses two four-row groups; this is exact gradient accumulation, not fewer examples or a hidden allocation fallback.

Training reports `gradient_counts_this_invocation` per trainable parameter
tensor: optimizer steps with nonzero squared gradient norm, zero norm, or no
gradient. The last case is allowed only by an exact, draw-specific parameter
list: `sbd_mask` and the Codpula gate head when no multi-token noisy block is
scored, and `cocommit_marker` when no marked branch is used. Every other missing
gradient still stops the update. No zero-weight parameter terms are inserted
into losses, and these counts do not assert that every parameter learned.

Enabled-SBD training and pilot fingerprints retain the semantic revision introduced after dense attention removal. Selecting `hola_read` additionally changes the serialized model identity; omitted/explicit `none` keeps the previous SBD identity. Old dense-SBD checkpoints remain incompatible. Configurations without `sbd` retain their hashes.
### Configuration controls and validation

`crates/experiment/src/config.rs::Config::validate` rejects controls that the
selected execution path does not consume:

- Refinement configs omit `commit_count`, `commitment`, `dependence_threshold`
  and `entropy_threshold`. Those names describe permanent SBD commitments, not
  temporary CoCommit marks. Refinement with CoCommit can instead select
  `sbd.announcement`: `count` defaults to 2; `dependence_threshold` selects
  dependence-based announcements; `entropy_threshold` selects entropy-based
  announcements. The two thresholds are mutually exclusive. For example,
  `"announcement": {"dependence_threshold": 0.05}` marks candidates without
  claiming they are permanent observations.
- `sbd.cocommit.rounds` is removed: the implementation has one marked tail call,
  not a configurable round count.
- `sbd.cocommit.visible_loss_weight` is accepted only with factorized output;
  omission there uses 0.2. Codpula configs omit it.
- TCT configs use `tct.rows_per_pass` and omit `sbd.rows_per_pass`.
  TCT is a training mode; inference keeps the selected ordinary SBD or R²D
  uniform/learned-remasking operator rather than adopting confidence reveals.
- Codpula and contextual Codpula require `soft_temperature: 1`, matching their
  training potentials.

Outside refinement, omitted commit controls retain fixed-count decoding with
count 2 and dependence threshold 0.05. Explicit fields remain distinguishable
from omission during validation, including explicit default values.


The optional model selector `prefix_block_retrieval` extends HOLA's single
softmax with the always-visible own-block clean prefix plus top-k completed
same-document earlier blocks; `configs/lm512-step8-retrieval.json` enables it
alongside `sbd.in_block_reads: "hola_read"`.
All noisy queries use own block `floor(start/B)`, including positions across
a B boundary; cached/unfinished-chunk duplicates are excluded. The
descriptions below omit the extra retrieved/own-block clean segment. See
[prefix block retrieval](optimization/prefix-block-retrieval.md) for exact
semantics, execution selectors, independent numerical checks and costs.

For each training sequence, choose one entry of `sbd.block_lengths` uniformly, then partition its target positions with that length, including a short final block. A singleton entry is a fixed-length condition. The variable condition can use `[1,4,8,16]` or the paper's `[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]`. Draw one sequence-level mask probability uniformly in (0,1), and independent Bernoulli masks. Unmasked targets inside the active block are explicit observed evidence, including targets to the right of a masked position. No other future targets are visible.

The objective is clean causal mean NLL plus the sum of masked block NLLs divided by the number of clean targets. This is equation 14's sum objective with a common token normalization, not an independently mask-normalized auxiliary loss. All-observed draws contribute zero masked loss. A detached first noisy pass supplies top-k candidate IDs, original probability weights and retained mass. The second pass uses Step 3's weighted embedding plus the mask embedding, with visibility `1 / soft_steps`. Selection is discrete; the second pass's embedding values remain differentiable. The mask is a separate trainable vector. Teacher probabilities are detached deliberately, rather than training through a proposal-generation graph. Hard-mask training skips this teacher pass.

## Positions and visibility


Training retains normalized sublayer inputs and aliases of the clean mixer's raw projections, convolved keys/values, activated scan inputs and scores. Requested block starts are known before the clean forward: its single scan carries state across disjoint segments and retains those snapshots while also producing the clean-loss outputs. Cache preparation reuses these tensors rather than repeating projection, convolution or recurrence. The configured computational chunk size remains the maximum segment tile size; HOLA selection still uses the original chunk grid, not the snapshot segment ends. Diagnostic callers that request a new start after the clean pass may extract that missing snapshot from the captured scan inputs. Noisy projections use the configured input-projection backend, including merged execution. Wavelet preparation retains each clean approximation level once; noisy smoothing uses the required cached level history. Lngram uses only its maximum suffix plus smoothing lookback.

Cache membership preparation now reuses the existing fused position-selection schedule without computing a complete TensorHOLA attention read. The selected positions, ties, document resets, padding and chunk-boundary snapshots are unchanged; original clean K/V tensors remain the differentiable value sources. Ninety oracle/old-read comparisons have exact membership equality. Soft-feedback teachers use frozen parameters and detached cache aliases before execution, not an unused graph detached only at the output; student computation and gradients are unchanged.
A target block starting at target index `s` sees exactly `example.input[..s+1]`: BOS followed by all clean target tokens strictly before `s`. Noisy target `s+j` occupies the same input-position identity `s+1+j` as its clean token counterpart. There is no second positional coordinate or repeated learned position embedding in this model.

Each mixer computes its clean-prefix recurrence and uses that final state as the initial state of every noisy query's **independent, temporary one-step recurrence**. Temporary states are discarded.

For `hola_read`, each head reads `[selected HOLA slots after completed chunks | clean unfinished chunk cut..start | active noisy block]` and the existing learned sink in **one softmax**, with the existing `1/sqrt(key_dim)` scale and sigmoid HOLA gate. Active keys use HOLA's key RMS normalization on the noisy convolved raw K; values use noisy convolved raw V, as clean cache values do. Every valid query sees every valid block key, including itself and keys to its right. Padded keys are excluded and padded queries return zero read output. The temporary block K/V never enters cache selection or the clean path. With `none`, the final segment is absent. Starts inside a computational chunk retain every unfinished clean key without ranking or truncation.

On retrieval-selected layers the read is `[cache | clean unfinished chunk | retrieved/own-block clean keys | active noisy block]` plus sink, still with exactly one softmax. Both cached training and prefix reconstruction pass the active K/V in `[batch,length,head,width]` layout to the retrieval helper; clean candidates never include noisy K/V. `prefix_block_retrieval_backend` selects the combined read on those layers, while `sbd.noisy_read_backend` continues to select the ordinary noisy HOLA read on other layers. Unsupported execution is rejected explicitly.

Wavelet taps crossing the start use clean-prefix activations and taps within the block use noisy activations. Lngram suffix codes and left-only smoothing use the same rule. Both remain left-to-right inside the block; they do not supply a bidirectional attention path. oHC stream initialization/read/write/readout is per token, including the mask/candidate inputs. No noisy stream state persists between passes. Every example is one document, with BOS/reset at its start. Training blocks cannot cross examples. Padding freezes recurrence and is excluded from HOLA reads, convolution and Step-7 history.

The length-one limit is defined explicitly: its noisy prediction is the clean next-token computation on the prefix, bit for bit. There is no current-target embedding in this limit, even for an observed singleton. This avoids changing the causal LM's next-token convention into a masked-current-token convention. Lengths greater than one use the block denoiser.

### Visibility proof sketch

The public block-forward function accepts a clean prefix and explicit soft/observed block state, not a full clean target sequence. At layer zero, these are its only token-dependent values. Inductively, clean-prefix sublayers are causal; noisy recurrence reads only that prefix's state and its own noisy projection; short convolutions, wavelet and Lngram read that prefix or the active representation; HOLA reads clean-prefix positions and, when selected, the explicitly supplied noisy block. Per-token normalization, MLP and oHC introduce no other token inputs. Thus changing a clean input/target at or after the block start cannot change predictions unless explicitly supplied as observed block evidence. Changing an observed token inside the block can change other block positions through the same HOLA softmax.

## Generation and replay

`sbd-generate` loads a checkpoint with the existing validated loader. It uses a held-out clean prompt (up to 17 input tokens), initializes an all-masked block, and repeatedly predicts against the same immutable clean token snapshot. Soft updates keep candidate distributions on uncommitted positions; finalized positions become immutable observations. Greedy assignment is deterministic. `commit_count` selects the number of lowest-entropy positions per pass. With `entropy_threshold` set, equation 8 selects the largest `s >= 1` whose first `s-1` sorted entropies sum to at most the threshold, in nats. Ties use ascending position, then ascending token ID.

Only after the entire block is finalized does the clean-prefix object append its tokens. The next multi-token block prepares one clean-prefix cache, reused by every noisy assignment; no separate discarded clean forward runs at initialization or replay, and no clean forward runs after the final block solely to discard its output. The length-one causal special case remains unchanged. Reported finalized tokens/s includes required clean-cache preparation from the initial prompt and from extended history, denoising, host selection, soft updates and synchronization. This is an execution change, not a measured decoding speedup; paired warm full-step and decode measurements are required separately.

With CoCommit enabled, generation prepares the immutable clean layer cache once per active block and reuses it across denoising passes. Each pass still recomputes its changing noisy prefix, runs selection, then reruns only the marked tail. It writes only announced positions from the marked output and retains observation immutability. In the combined path, new token potentials and coupling gates create a new Codpula circuit; joint proposals, marginals and dependence scores are recomputed before writes. No pre-marker circuit values are used for assignment. Clean replay remains once per completed block.

## Work and retained tensors

Clean preparation is O(T): one clean forward with a state snapshot at each requested start, plus cache membership preparation. Multi-token generation performs this clean work once per block rather than per denoising pass. The active HOLA segment adds O(B H L² (K+V)) read work per pass; this is quadratic only in the active block length, not the prefix length.

`sbd.noisy_read_backend` defaults to `"tensor"`; `"fused_gpu"` explicitly selects the CubeCL noisy-read kernel on layers without retrieval, with analytic query/K/V/sink derivatives. The sigmoid gate remains the existing tensor operation. Cache gathers and segment concatenation still use the tensor graph. The kernel supports f32 K32/V64, at most 16 noisy queries and 127 assembled keys, on explicit non-fusion GPU execution; unsupported shapes are rejected without a CPU alternative. Retrieval-selected layers instead use `prefix_block_retrieval_backend` for the combined read, including active K/V. These execution choices add no model parameter.

Paired complete-step cost tables, launch counts and sampled memory are recorded in [RESULTS.md](../RESULTS.md#step-8-recurrent-block-denoising-through-hola). The complete-step benchmark is `sbd-benchmark CONFIG --warmups 2 --trials 11`. It measures the real SBD loss/backward/checks/clipping/AdamW path, not clean-only training, and performs no evaluation or checkpoint writes.

### Block-axis execution

`sbd.block_execution` selects `per_block` (the omitted default) or `batched`.
The default is excluded from serialization, so omission preserves existing
fingerprints. Both paths use the same sampler; a regression pins the original
length, sequence-mask-probability and token-mask draws, including all-observed
blocks.

The batched path prepares clean snapshots for all rows together, then stacks
selected multi-token blocks along the batch axis. Each row has its own frozen
recurrent state, convolution/wavelet/Lngram history, HOLA slots and original
prefix start. Blocks are right-padded to the maximum configured length; inactive
positions have zero loss weight and are excluded from read keys. Length-one
targets retain their clean causal logits. Soft feedback still uses a detached
teacher pass followed by the differentiable second pass, each batched.

`sbd-reference` includes the block-execution comparisons;
`sbd-batching-reference` runs only those comparisons.
`sbd-batching-benchmark CONFIG --warmups 2 --trials 11` compares complete
per-block and batched training steps with alternating execution order.
`sbd-batching-diagnostic CONFIG` measures one configured batched workload after
one warm step. Reports include issued launches and sampled pool reservation.

CPU and Vulkan acceptance each passed 24 no-retrieval and 40
retrieval-inclusive architecture/feedback cases. CPU batching maxima were
1.63913e-7 for logits, 1.90735e-6 for total loss and 3.12925e-7 for every-parameter
gradients. Vulkan maxima were 4.47035e-8, 1.90735e-6 and 5.96047e-8 respectively.
The largest elementwise tolerance ratio was 0.008266 under the unchanged
`2e-5 + 3e-4*abs(reference)` criterion. The large-grid noisy-read scalar-f64
forward/full-VJP case passed with maximum error 3.43394e-7.
Workspace tests passed 49 experiment + 7 kernel + 16 mixer tests, with one
GPU-sharing test ignored. Clean Step-7/retrieval model gates and historical
checkpoint continuation also passed. Evidence is under `artifacts/batching/`.

The batched-only B8/T512 diagnostic completed in 8167.400 ms with 27,145 launches
and 8,818,099,392 bytes sampled pool reservation. This is one observed step
after one warm step, not a paired throughput result or an exact live-memory
peak. B8 fit without chunking. Its first attempt hit the existing noisy-HOLA
key-VJP's one-dimensional 65,535-workgroup limit; extending the existing
dispatch grid fixed it without changing the read or derivative arithmetic.
The scalar-f64 regression covers 67,200 key-gradient workgroups.

Final-code B1/T512 complete-step pairs (two warmups, 11 interleaved pairs) measured:

| Workload | Per-block median | Batched median | Paired saving [95% interval] | Faster |
| --- | ---: | ---: | ---: | ---: |
| HOLA, retrieval absent | 42,144.517 ms | 1,617.483 ms | 40,568.800 [35,569.793, 44,211.076] ms | 11/11 |
| HOLA plus layer-3 MoBA | 44,178.291 ms | 1,819.748 ms | 42,358.543 [39,231.935, 43,705.788] ms | 11/11 |

Both intervals exclude zero, so `configs/lm512-sbd.json` and
`configs/lm512-step8-retrieval.json` select `batched`. An additional matched
step counted 211,269→24,242 launches without retrieval and 209,934→24,208 with
retrieval. Masks vary by step; these counts use a different draw from the
earlier 370k-launch diagnostic. Paired-session pool reservation sampled
1,931,060,160 and 2,200,545,536 bytes respectively, including both live runners.
See [RESULTS.md](../RESULTS.md#sbd-block-axis-execution-batching) for the method,
sampling limitations and exact evidence paths.


The earlier per-block B8/T512 attempt reached its 3100-second deadline before one three-mode iteration report; it provides no paired throughput result. The accepted B1/T512 diagnostic measured 128 noisy block evaluations and 512 layer HOLA reads per step. Its synchronized noisy-HOLA forward sections used 10.77% / 11.74% / 9.18% of observed complete-step time for `none` / tensor `hola_read` / fused `hola_read`; these are forward-only shares, not full read/VJP cost or ordinary paired-step timing. `sbd-diagnostic CONFIG` performs one warm and one observed step per mode.

The accepted B1/T512 two-warmup/11-pair session measured an in-block tensor-read cost of +601.87 ms per complete step (95% paired interval [+328.46, +1508.97]). Replacing that read by the mathematically identical fused kernel saved a paired median 2390.35 ms [+1667.56, +3176.01], faster in 10/11 pairs, so `configs/lm512-sbd.json` explicitly selects `fused_gpu`. The tensor default remains unchanged. Cache gather and segment concatenation were not replaced in that session, which did not attempt SBD per-block loop optimization.

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

Clean and noisy execution share one per-layer operation: optional wavelet, optional
Lngram, mixer, then MLP. The same operation handles ordinary residuals and oHC
reads/writes, including cached per-block/batched calls and the marked CoCommit
tail. CPU regression compares outputs and parameter VJPs with the pre-refactor
residual equations on both stream layouts; this is not a performance claim.

Plain SBD generation prepares the same clean-prefix cache used by training and
reuses `cached_block` for every denoising assignment. The length-one causal
special case is unchanged. The reconstructed `forward_sbd` path remains a
diagnostic comparison, not the multi-token generation implementation.

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
