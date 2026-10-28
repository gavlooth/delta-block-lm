# Architecture implementation and optimization plan

**Current backbone decision:** KronLSTM only. **GDN-2 removed on 2026-10-28 by user decision.** Earlier comparison results remain historical evidence, not supported configurations or instructions to restore the removed mixer.

Status: Steps 1–11 have separate implemented/correctness-gated profiles on CPU and local Intel Vulkan; their contextual/coordination/refinement composition was published on main as `9e67b7ac` before combined acceptance completed. The separate Step 5/6 quality gates were waived, not passed. Current Step 8 uses the explicitly selected HOLA in-block read and optional prefix retrieval; earlier removed dense-attention measurements remain historical only. The Step 7 quality ablation is cancelled, not pending. Step 12 has not started and is not authorized by this integration. Measured results and the exact partial integration command ledger are in [RESULTS.md](RESULTS.md#integrated11-publication-checkpoint).

**Scope override — explicit user instruction:** implement and optimize the given architecture. Its components are requirements, not candidates that must win architecture-selection experiments. Correctness checks and focused runtime/memory measurements may verify an implementation optimization. Do not launch quality ablations, architecture comparisons, candidate sweeps, alternative-mechanism studies or research training runs unless the user explicitly requests them. This overrides experimental comparisons, sweeps and quality-adoption gates elsewhere in this document; those passages are historical proposals, not executable tasks. Steps 9–12 are implementation/optimization work only where they implement the specified architecture, not authorization to add alternatives.

## 1. Goal and fixed decisions

Build and optimize the specified semi-autoregressive block-denoising language model. The predecessor is `diffusion-relation-extractor`; its relation-specific output heads and ontology constraints are not requirements for this LLM.

- Rust, Burn and custom CubeCL kernels. Intel Arc/Vulkan locally; DGX Spark/CUDA for larger runs after arranging GPU access. Follow `AGENTS.md` for build limits, mold, debug-free tests and experiment records.
- KronLSTM with optional HOLA is the only recurrent mixer. Preserve its shared scan, chunking, convolution and exact-memory machinery.
- Keep wavelet token mixing, orthogonal hyper-connections (oHC) and three Lngram orders in the target design. Isolate their costs and contributions rather than enabling everything in the first run.
- Use Set Block Decoding (SBD) as the initial semi-autoregressive recipe. Block length, computational chunk length and tokens committed per denoising step are separate variables.
- Preserve the predecessor's soft top-k diffusion mechanism as the block-denoising baseline: probability-weighted candidate feedback, separate visibility, and revisable proposals. Hard-mask-only feedback is an ablation, not a replacement. Add joint token prediction, coordination before writes, and revision after provisional writes on this foundation.
- Energy-guided stochastic remasking is the user's research proposal against pure random remasking. It must not be replaced by confidence ranking or a deterministic correction-value predictor.
- Preserve the specified architecture while optimizing execution time and memory. Verify numerical correctness; do not require components to earn inclusion through quality comparisons.

## 2. Sources and evidence boundaries

| Mechanism | Primary source | What the source establishes |
|---|---|---|
| GDN-2 (historical only; removed on 2026-10-28 by user decision) | [2605.22791](https://arxiv.org/abs/2605.22791) | Former channel-wise reference; no active implementation or comparison arm remains |
| KronLSTM | [2610.26834](https://arxiv.org/abs/2610.26834) | Selected associative-memory update using a scalar bilinear correction |
| HOLA | [2607.02303](https://arxiv.org/abs/2607.02303) | Selective exact KV memory and sharpened cache read, adapted to KronLSTM's committed update magnitude |
| Wavelet mixer | [2610.04720](https://arxiv.org/abs/2610.04720) | WNet token mixing; causal and SBD visibility verified on reference fixtures |
| oHC | [2609.02672](https://arxiv.org/abs/2609.02672) | Orthogonal mixing of residual streams |
| Lngram | [2605.24869](https://arxiv.org/abs/2605.24869) | N-gram conditional memory in latent space |
| SBD | [2509.04185](https://arxiv.org/abs/2509.04185) | Causal next-token prediction plus masked future-block prediction; variable training block lengths and entropy-based parallel decoding |
| Soft masking | [2510.17206](https://arxiv.org/abs/2510.17206) | Feedback using mask embeddings mixed with previous top-k predictions |
| Discrete Copula Diffusion (DCD) | [2410.01949](https://arxiv.org/abs/2410.01949) | Inference-time combination of denoiser marginals and a separate dependency model through approximate information projection |
| Codpula: Learning to Couple | [2610.25723](https://arxiv.org/abs/2610.25723) | Context-dependent token/coupling potentials, jointly trained probabilistic circuit, dual-flow update and dependence-adaptive commitment |
| CoCommit | [2607.04469v2](https://arxiv.org/html/2607.04469v2) | Mark a selected commit bundle and reuse the last few layers before writing; approximate joint-mode coordination under greedy decoding |
| Random Remask-Denoise (R²D) | [2610.30744](https://arxiv.org/abs/2610.30744) | Random remask-denoise inference; Trajectory Correction Training (TCT) exposes the denoiser to its own errors |

The downloaded discussion supplied the Codpula link. Codpula and DCD are distinct papers; the HMM joint-training mechanism is not a result of the older DCD paper. The discussion's Julia circuit was not executed and is not a verified implementation.

No cited result establishes the complete proposed combination. Exact circuit inference means exact inference inside the specified model and candidate support, not exact recovery of the true language distribution.

## 3. Recurrent backbone and memory

### Selected backbone

Use KronLSTM with the configured embedding, head dimensions, training data, precision, optimizer and token schedule. Report actual parameter counts and recurrent-state memory; no cross-mixer comparison is part of the current architecture.

Keep the existing KronLSTM configurations with no cache, HOLA surprise selection and matched-size recency selection. These isolate cache behavior without maintaining a second mixer.

HOLA retains original KV pairs, not the gated/preconditioned update factors. Its cache-read Q/K normalization is separate from recurrent-update normalization. Compare surprise selection with a matched-size recency cache. The local current-chunk attention supplied by HOLA must be counted in memory and work.

Proposed surprise scores, requiring implementation checks:
- KronLSTM: absolute scalar correction multiplied by key and value norms.

A cache may not be updated from noisy or bidirectional block representations. After completing a block, causally replay its final tokens to update recurrent states, HOLA entries and convolution histories exactly once.

### Compatible additions and alternatives

- Exact Flow Linear Attention ([2512.12602](https://arxiv.org/abs/2512.12602)): historical optional gate/update reference, not an exact solution of the selected KronLSTM + SBD model. Any extension needs a derivation and recurrent/chunked gradient checks.
- GSA-2 ([2610.02816](https://arxiv.org/abs/2610.02816)): historical alternative two-memory reference, not part of the selected architecture.
- Erase-then-Delta ([2606.26560](https://arxiv.org/abs/2606.26560)): later independent erase-address experiment; account for its extra recurrent work.
- Dynamic Linear Attention ([2606.10650](https://arxiv.org/abs/2606.10650)): later multi-state research, not exact concatenation of delta-rule states by addition.
- Hybrid recurrent backbones are not part of the KronLSTM-only architecture.

HOLA is the first selective exact-attention mechanism to test, not proof that all separate sparse/local attention is unnecessary. Keep sliding-window attention as a comparison. MSA, NSA/MoBA, CSA/HCA, BA-Att, SparseD and LoSA remain later sparse-attention references; their published kernels and settings do not establish speedups for this model. LoSA is especially relevant to repeated block denoising. Add an alternative only when a measured failure motivates it.

### Source reuse, not a fresh reimplementation

The predecessor sources needed for these components are copied under `reuse/crates/`:

| Copied source | Reuse target and necessary adaptation |
|---|---|
| `dre-core/` | Sequence layout, mixed-precision helpers and profiling; keep only dependencies needed by the new components |
| `dre-attention/` | Existing attention kernels and reference checks for exact-cache reads; any sparse/within-block attention reuse awaits explicit user selection |
| `dre-wavelet/`, `dre-ohc/`, `dre-lngram/` | Existing implementations and tests; preserve the three Lngram components and verify causal/block visibility before training |
| `dre-relations/src/state.rs` | Soft distributions and separate visibility state |
| `dre-relations/src/denoiser.rs` | Probability-weighted expected embeddings mixed with mask embeddings; adapt field embeddings to vocabulary candidates |
| `dre-relations/src/diffusion_scheduler.rs` and its tests | Convex feedback updates, visibility resets that preserve soft beliefs, and circuit-conditioned feedback |
| `dre-relations/src/joint_circuit.rs` | Existing log-space joint inference and reusable message preparation; replace the five relation/pointer fields with token positions and audit against Codpula |
| `dre-relations/src/remask.rs` | Existing proposal/retained-answer distinction; reuse state semantics without inheriting its learned policy as the new energy policy |

These are unchanged migration inputs, not a second executable workspace. The partial `dre-relations` copy is not a buildable crate. Record source provenance in `reuse/PROVENANCE.json`; port useful code into the active workspace and remove obsolete snapshots as each component is migrated. Do not import the predecessor's full trainer, datasets, run artifacts, or vendored backend tree.

The inspected feedback path uses full probability tensors and weighted matrix products; an inspected top-k call in `boundary_pair.rs` selects source boundaries. Do not mislabel that boundary selector as a reusable vocabulary top-k kernel. Preserve the existing soft mechanism while identifying which operations need a new sparse vocabulary implementation. Validate sparse top-k expectation/update results against the copied dense path before optimizing them.

## 4. Block-denoising semantics

The initial SBD reference uses a clean causal path and a noisy future-block path, with original position identities shared by corresponding tokens. The paper samples block length during training; it does not learn the block length. Its sampler's variable commitment count is also not learned block-boundary selection.

Initial block lengths: fixed 4, 8 and 16 plus a variable-length condition. These are experiment points, not a permanent maximum. Expand to larger blocks and learned/adaptive boundaries only after measuring quality and tokens/s. Use masked padding where useful; do not force the recurrent kernel's computational chunk to equal the denoising block.

Proposed recurrent adaptation:
1. Compute clean-prefix states and exact-memory contents.
2. Noisy positions read the clean state before their block. Sparse/within-block attention remains undecided pending explicit user selection. The added dense active-block path is removed; no new bidirectional attention path is currently implemented. HOLA's existing attention is unchanged.
3. Repeated denoising and coordination passes use that same clean-prefix snapshot; they do not advance persistent state.
4. Causally replay the finalized block before extending the prefix.

Verify short convolution, wavelet taps, Lngram addressing and all caches against the same visibility rules. No path may read later clean targets. Keep a test for block boundaries inside a computational chunk.

Soft top-k feedback is part of the initial denoising path. Preserve candidate IDs, weights and retained mass, rather than treating each provisional prediction as hard observed evidence. Remasking changes visibility without automatically deleting the candidate distribution or retained proposal. Candidate width is independent of block length, HOLA capacity and the number of committed positions. A dense/full-support reference and a hard-mask-only ablation isolate truncation and soft-feedback effects. Verify selected-value/embedding gradients, padding, observed-token immutability, ties and zero-mass behavior against the source semantics.

## 5. Joint prediction and commitment

Three mechanisms address related but distinct parts of the same problem. Compare them separately before composing them.

### 5.1 Codpula joint denoiser

The selected implementation is a homogeneous 32-state HMM over each actual SBD block of length at most 16. Initial, transition and emission probability rows are shared across lengths/positions, stored in log form, with full 257-symbol emissions. These are small-model choices, not the paper's 1,024-state reproduction. The decisions are recorded in [docs/codpula.md](docs/codpula.md).

The existing vocabulary head supplies token potentials from final-normalized noisy features. Original `joint_output: codpula` retains its valid-state mean and 64 biased linear log-gates exactly. Explicit `codpula_contextual` selects ordered `[valid mean, first, last]`, one zero-initialized `3D -> 96` head (64 transition plus 32 normalized-root log-gates), and moderate seeded emission/transition contrasts. Both share one b=4 transition grid across edges. Local emission, transition-row, root-row and visible-evidence normalizers are included in conditional joint NLL. Circuit rows retain the explicit nonnegative dual-flow split, summed occurrence reaches, separate bias-corrected momentum 0.9 and constant alpha 0.05; backbone/head/gate parameters retain existing AdamW.

All 257 symbols participate in circuit inference: candidate-support truncation loss is zero, while the existing width-four sparse feedback remains separate and receives detached joint marginals. The selected input-plus-coupling model, exact causal singleton bypass and old factorized default are implemented. No input-only/64-state/quality study was run in this turn.

Dependence commitment uses exact conditional joint proposals and absolute marginal-versus-conditional log-probability changes for all unresolved positions. It reuses one prepared circuit, evaluates leave-one-out probabilities with cavity messages, uses strict threshold 0.05, minimum-score progress fallback and at most L passes with final completion. Invalid evidence fails clearly.

### 5.2 CoCommit coordination

First implement factorized output + CoCommit as its own comparison:
1. Cache the hidden states before the final tail layers.
2. Run the unmarked tail to select a token bundle without writing it.
3. Add a learned marker at those positions to the saved pre-tail states.
4. Run the shared tail again, then write from the coordinated logits.

Start with one coordination round and one tail layer on the tiny model. Train the marked branch as well as the unmarked selection branch. The current authorization uses a saved-input zero-marker repeat only as a correctness equivalence gate; no trained sequential-depth comparison arm or quality study is authorized. Do not feed the first tail's output into the second and call it the paper's one-round mechanism.

CoCommit's evaluated coordination is for masked commit bundles under greedy decoding. It is not an exact joint distribution, an exact joint MAP solver, or a guarantee under independent temperature sampling. Marking already-visible tokens is a proposed extension, not an evaluated paper result.

For recurrent tails, each branch uses the same clean-prefix state and temporary active-block computation. Define marker injection across oHC streams explicitly; never write persistent state during coordination.

### 5.3 Combining the mechanisms

Compare: factorized outputs; CoCommit; Codpula with fixed-count commitment; Codpula with dependence-adaptive commitment. Only then test CoCommit + Codpula: the marked tail regenerates token potentials and coupling gates, the circuit is rebuilt for those outputs, and proposals/dependence scores are recomputed before writes. Reusing pre-marker circuit values after changing hidden states would mix two distributions.

DCD remains the older separate-model inference baseline/reference. Do not silently replace Codpula training with DCD's approximate marginal matching, or vice versa.

## 6. Energy-guided stochastic remasking

This is the user's proposed contribution: keep the exploration of random remasking, but learn its distribution from surrounding context and interactions between positions. A confident token can still be eligible for revision.

For eligible mask sets M of fixed size r, test

    P(M | x,c) = epsilon * U_r(M)
                 + (1-epsilon) * exp(-E_phi(M;x,c)/T) / Z_r(x,c)

where U_r is uniform over eligible size-r sets. Keep the remask count and eligibility identical in each comparison. Lower mask-set energy means a more useful perturbation, not merely a more probable current token assignment.

Progression:
1. Pure uniform R²D, preserving its schedule as the reference.
2. Confidence/entropy selection as diagnostic baselines, not the proposed replacement.
3. Uniform mixture with learned unary mask-set energies.
4. Add pairwise energies over local positions or a small learned dependency graph if unary guidance misses jointly wrong configurations.

For small blocks enumerate eligible size-r sets to verify normalization and sampling. A per-token softmax or sampling without replacement is not automatically the same distribution over sets. Larger blocks require a separately specified sampler whose approximation and cost are reported.

Inputs can use existing contextual states, candidate embeddings, prediction changes, and circuit conditional-probability/entropy features. Treat circuit surprisal and best-alternative log-probability gain as features or baselines, not as proof of future correction benefit. Check shortcuts where a head simply repeats its own token confidence; leave-one-out context is an optional diagnostic with its actual cost included.

Train energy rankings from paired, equal-budget refinement trajectories. If mask set A yields lower final loss than B, train E(A) < E(B); ties supply no ordering. Begin with a small frozen-denoiser trajectory set, then measure policy generalization on held-out examples. Report trajectory-generation cost separately from inference savings. The correction-value head discussed earlier is an ablation, not a substitute for stochastic mask-set energies.

Keep soft feedback as an option when a token is remasked. Pair/group remasking is important because conditioning on a collection of wrong tokens can trap single-position updates.

R²D's convergence results do not transfer automatically to state-dependent mask selection, soft feedback, or Codpula-conditioned updates. A positive uniform mixture preserves access to eligible mask sets; it does not prove the correct stationary distribution. Initially evaluate this as a refinement policy; exact target-distribution sampling needs a separate transition-kernel analysis.

## 7. Step-by-step execution checklist

Historical earlier pilot: Steps 1–7 numerical, soft-feedback, build and checkpoint-continuation checks passed; three-seed recall/tiny-LM experiments were independently reevaluated from validated checkpoints. The Step 5/6 quality gates were waived, not passed: recall was near chance on 112–248 queries per seed, and the tiny LM ran 13 updates with only 2,048 validation bytes. The KronLSTM+HOLA working baseline therefore is not a demonstrated quality winner (certified artifact paths are listed in RESULTS, Associative recall and Tiny causal language model). The Step 7 quality ablation is cancelled and must not resume. The earlier partial-Step-8/dense-attention status is superseded by the explicitly selected HOLA in-block read, optional prefix retrieval and separate Step 9–11 implementations below. Current combined acceptance is §9; Step 12 is not started or authorized.

### Step 1 — Create the small active Burn workspace

- Create crates for shared model types/references, GPU kernels, model components and the experiment executable; do not copy the predecessor application.
- Configure CPU checks without GPU features, opt-in Vulkan/CUDA runs, mold locally, debug=false for dev/test, and one integration-test binary per crate.
- Check actual Cargo profile inheritance: optimize external dependencies without accidentally optimizing every local workspace crate.
- Add file-based experiment configuration and machine-readable results from the first runnable experiment. Configurations select mixer, dimensions, cache size, seed and backend without recompilation.
- Preserve the supported kernel dimensions initially (four heads, key width 32, value width 64, computational chunk 16). Projection/model width is a separate choice.

**Deliverable:** a runnable CPU reference command and a minimal Vulkan command using the same input fixture.

**Gate:** CPU command runs, Vulkan reports the intended device, and build/rebuild times are recorded. Use a 120-second limit on the first cold build; if it expires, report that outcome rather than extending the wait silently. Warm model rebuild target is <=10 seconds; test rebuild target is <=60 seconds. Do not trigger a full-workspace test build to measure one component.

### Historical Step 2 — Former reference migration

GDN-2 removed on 2026-10-28 by user decision. The following checklist records the former migration only; its implementation, commands and copied crate have been removed.

- Copy useful reference, forward/backward and CubeCL code from `reuse/` into the active crates, keeping source provenance.
- Separate the causal recurrence needed by the LLM from the predecessor's bidirectional encoder wrapper.
- Compare chunked outputs, final states and parameter/input gradients against the float64 sequential reference.
- Cover a sequence shorter than one chunk, multiple chunks, padding, packed-document resets, and block starts inside a chunk.
- Exercise actual GPU forward and backward on a small fixture. Record first-use compilation separately from warm execution.

**Historical deliverable:** a numerically checked former reference implementation and short CPU/GPU benchmark; retained measurements are in RESULTS.

**Gate:** tolerances are specified by precision, gradient checks pass, and no document/padding information crosses its allowed boundary. Preserve finite-input failures rather than suppressing them.

### Step 3 — Migrate the existing soft top-k diffusion foundation

- Reuse the copied soft state, weighted embedding, feedback scheduler and proposal/retained-answer semantics.
- Generalize annotation values to token candidates: candidate IDs/weights, visibility, provisional values and retained values stay separate.
- Remasking must not silently delete the soft distribution. Observed prompt tokens stay fixed.
- Keep the dense probability-weighted path as the numerical reference. Port or implement sparse selection/gather/update only where needed; do not confuse source-boundary top-k with vocabulary top-k.
- Make candidate width configurable independently of block length, recurrent chunk length and HOLA capacity.

**Deliverable:** a runnable miniature token-refinement episode using the reused soft-feedback mechanism, before adding learned energy selection.

**Gate:** sparse/full-support feedback agrees with the dense reference where their support matches; embedding/weight gradients, retained mass, zero-mass behavior and visibility transitions are checked. A hard-mask-only path exists solely as a comparison.

### Step 4 — Implement KronLSTM and HOLA memory

- Implement KronLSTM's sequential float64 recurrence first, then its chunked forward/backward and GPU path.
- Implement HOLA's exact KV cache and sharpened read as shared model components where semantics allow, with mixer-specific update-magnitude scores.
- Derive KronLSTM cache-selection scores from its scalar correction and verify them against the independent reference.
- Compare surprise-based selection with a cache of the same capacity using recency selection. Count HOLA's current-chunk attention work.
- Start cache capacity at 16; measure 64 after the first pilot.

**Deliverable:** KronLSTM runnable with and without HOLA.

**Gate:** recurrence/chunk agreement and gradient checks pass; causal cache membership and padding are correct; KronLSTM executes forward/backward on the local GPU.

### Step 5 — Run the first controlled associative-recall experiment

- Start with two mixer layers, model width 128 and contexts 128/512. Keep the inherited kernel key/value widths initially.
- Generate deterministic training examples and a disjoint fixed evaluation set. Sweep the number of key/value associations to expose memory interference.
- Current configurations run KronLSTM and KronLSTM+HOLA, with the matched recency-cache option retained. Earlier four-arm results are historical: GDN-2 removed on 2026-10-28 by user decision.
- Run one seed as a smoke, then three paired seeds for an architectural decision. Use a short throughput pilot to choose and record a fixed training-token budget for the comparison.
- Report recall by association count and distance, actual parameter count, recurrent/cache memory, peak GPU memory, training throughput and warm decode latency.

**Deliverable:** one comparison report with the configurations and raw metrics.

**Historical gate:** choose a quality/cost Pareto shortlist rather than a winner from training loss or one seed. This earlier cross-mixer selection gate does not apply to the current user-selected backbone.

**Recorded outcome:** this quality gate was waived, not passed. The certified
`artifacts/recall{128,512}-vulkan/certified-evaluations/` comparisons summarized
in RESULTS remain near chance; they do not justify a quality shortlist.

### Step 6 — Check the shortlist on a tiny causal language model

- Start with four layers, model width 256 and context 512, retaining supported kernel widths. These are pilot sizes, not final architecture decisions.
- Pin a small, legally usable text corpus and tokenizer; split by document before packing or sampling. Record source/version and split hashes.
- Fix optimizer, data order, token budget, precision and output head across mixer arms. Report actual parameter/state differences.
- Measure held-out next-token loss/perplexity, throughput, memory and retrieval diagnostics for the selected KronLSTM configurations.

**Deliverable:** a resumable tiny LM checkpoint and held-out comparison.

**Gate:** the model learns on held-out data and the recall advantage survives enough to justify its cost. Resolve nonfinite states/gradients before adding mechanisms.

**Recorded outcome:** this quality/cost adoption gate was waived, not passed.
`artifacts/lm512-vulkan/certified-evaluations/vulkan-362617-1792846070036817669/comparison.json`
covers 13 optimizer steps and four 512-byte validation windows, not established
long-context learning or a surviving recall advantage.

### Step 7 — Add oHC, wavelet mixing and the three Lngram components

- Reuse the copied implementations, introducing one component at a time on the selected small backbone.
- Preserve all three Lngram components in the target design; verify the predecessor's distinction between injection count and n-gram orders before mapping configuration fields.
- Check stream mixing, table addressing and surrogate gradients, and account for residual-stream/table memory.
- Use causal masks/history in the causal model and define the future-block visibility rules needed by the next step.

**Deliverable:** the requested small backbone with attributable component measurements.

**Gate:** measure component costs and verify correctness. **DO NOT IMPLEMENT OR RUN the Step 7 quality ablation against the pre-Step-7 baseline.** Cancelled by explicit user instruction; do not resume, extend, or replace it unless the user explicitly requests it. Quality benefit remains unestablished. Keep the implemented components.

### Step 8 — Train the SBD-style block-denoising path

Implemented with the explicitly selected HOLA in-block read (`sbd.in_block_reads: "hola_read"`) and optional prefix retrieval. `docs/sbd.md`, `crates/mixers/src/sbd.rs` and `artifacts/batching/` record the selected operator and its cached/batched numerical and execution evidence. The earlier dense active-block attention and its attention-only oracle were removed; their historical measurements are not verification of the selected operator. The scope override prohibits new comparisons and sweeps without explicit user instruction.

- Add the clean causal objective plus masked future-block objective, retaining the soft top-k input/update mechanism from Step 3.
- Noisy positions read the clean state before their block without advancing persistent state. Selected HOLA in-block reads add active noisy keys/values to the existing HOLA softmax; optional prefix retrieval shares that read. There is no separate dense-attention operator.
- Verify short convolutions, wavelet taps, Lngram routing and exact-memory reads obey this visibility rule.
- Causally replay each finalized block exactly once before extending the prefix.
- Compare fixed block lengths 4/8/16 with a variable-length training condition, keeping computational chunk length independent. Include length 1 as a causal limiting-case check.
- Sweep candidate width separately. Add larger blocks and learned/adaptive boundaries only after measuring the fixed/variable references; do not impose 16 as a permanent limit.

**Deliverable:** training, soft top-k feedback, selected HOLA in-block reads and clean replay are implemented; see `docs/sbd.md` and `artifacts/batching/` for profile-specific evidence.

**Gate:** retain checks for future-target leakage and agreement between prefix replay and causal execution. The selected-operator references and block-axis checks are recorded in `docs/sbd.md`; earlier dense-attention block NLL/accuracy and tokens/s remain historical. Hard-mask-only feedback remains a historical ablation, not the default or authorization for a new run.

### Step 9 — Add Codpula's joint output and adaptive commitment

**Implementation, numerical gates and execution-cost measurement complete (2026-10-25).** Quality studies were explicitly excluded from this turn.

- 32-state homogeneous joint output, all 257 symbols, final-normalized potentials, b=4 coupling gates, required normalizers and dual-flow circuit optimizer are integrated across cached/per-block/batched/reconstruction/evaluation/generation paths.
- Independent f64/enumeration, full VJP/flow, joint-sampling, cavity, padding/observations/L1, CPU/Vulkan model and checkpoint/optimizer continuation gates pass at `2e-5 + 3e-4*abs(ref)`. Historical factorized identity is preserved; obsolete dense-SBD payloads stay rejected.
- Fused leaf/chain/marginal/flow kernels with analytic VJP pass. Keep tensor circuit execution selected: B1/T512 paired fused saving 60.951 ms has 95% interval [-23.545,84.668], despite an isolated 20.025 ms [16.577,23.311] gain. Factorized/tensor/fused complete-step medians are 1522.502/1697.014/1631.830 ms.
- Selected B8/T512 tensor diagnostic: 8000.071 ms, 30,745 launches and 8,826,550,976 bytes sampled pool reservation. Tiny context32 smoke: 96 bytes/three steps, checkpoint evaluation and eight dependence-committed/replayed tokens; disposable model/optimizer payloads removed.
- The original results above remain historical `codpula` evidence, not measurements of the explicit `codpula_contextual` revision. The revised profile changes only decisions 3, 4 and 6 for expected trainability/context capacity per cost; all other decisions and omitted-selector behavior remain unchanged. Configuration: `configs/lm512-codpula-contextual.json`; evidence namespace: `artifacts/codpula-context/`. No quality study, sweep or architecture-selection experiment is authorized.
- Explicit contextual acceptance: CPU/Vulkan tensor/fused references, ordered summaries/full head and root VJPs, normalized flows/reaches/Eq. (9), all caller layouts and exact old/contextual optimizer restarts pass unchanged tolerance. A separately committed Tensor Lngram duplicate-index gradient fix `e50cc523` retains the exact L1/restart gates and clipping `64590d18`. B1 old/contextual medians 1713.161/1691.060 ms, paired contextual-minus-old -42.608 ms [-138.390,15.668] (no resolved cost difference); full B8 diagnostic 8714.077 ms / 30,818 launches / 8,826,780,928-byte sampled pool. Only 96 bytes/three smoke steps, saved load/evaluate/generate and eight replayed tokens; no quality inference. Existing configs stay unchanged.

**Delivered:** verified joint predictor and fixed/dependence decoder support, current-predecessor salvage with provenance, exact tests and execution-cost accounting. See `docs/codpula.md`, `RESULTS.md` and `artifacts/codpula/`.

**Gate result:** valid circuit rows, conditional normalization and full-support inference pass; latency is measured. The original quality/latency study against an independently trained factorized baseline, input-only comparison and 64-state study remain unevaluated by explicit user direction, not inferred from a tiny smoke.

### Step 10 — Add CoCommit, then test the combination

**Implementation, numerical gates and execution-cost measurement complete (2026-10-25).** The authorized scope is correctness, off/on cost accounting, equivalent layouts and a tiny execution smoke, not a quality study.

- One complete final block, one round, full saved oHC streams and independent marker entries are integrated with hard/soft feedback, factorized/Codpula output, reconstruction, batching, evaluation and generation. Marked potentials/gates create a new circuit; no temporary tail writes persistent state.
- Independent f64 marker/temporary-Kron and circuit-event references, CPU/Vulkan logits/loss/every-gradient, padding/leakage, real prefix219 tail MoBA, and historical/enabled production checkpoint gates pass unchanged `2e-5 + 3e-4*abs(ref)`. Enabled restart includes clipping/AdamW/marker moments and one circuit update.
- B1/T512 medians off/separate/doubled1286.855/1408.335/1393.231ms. Selected separate overhead116.839ms[89.963,121.664]; doubled saving24.124ms[-23.326,49.555] is not established for the complete step.
- Position-only membership reuses existing fused selection and passes90 exact oracle/old-read cases; isolated paired saving98.227ms[91.325,104.213]. Frozen teachers avoid unused autodiff graphs. Explicit `sbd.rows_per_pass:4` preserves all B8/T512 work/RNG/bundles with two backward groups and one optimizer/circuit update; full/chunk all-gradient/all-flow gates pass. Selected B8 diagnostic6893.824ms,49928 launches,3,951,724,736 sampled pool bytes; original OOMs retained.
- `docs/cocommit.md` records all adopted decisions and the evidence-selected layout override. Remasking/TCT combined dispatch and provisional-evidence semantics are owned by parent integration, using the independent saved-prefix/tail/output API.

- Implement the unmarked selection tail and marked assignment tail starting from the same saved pre-tail hidden states.
- Train the marked branch; start with one tail layer and one coordination round on the tiny model.
- Check saved-h zero-marker equivalence; measure CoCommit off/on step cost and equivalent tail execution layouts only. No trained comparison arm, quality study, sweep or ablation is authorized.
- In the combined path, regenerate circuit potentials/gates and dependence scores after marked coordination. Separate-versus-combined quality comparisons require a future explicit authorization.
- Reset temporary recurrent-tail state from the same clean-prefix snapshot for each branch. No coordination pass commits persistent state.

**Deliverable:** implemented separate/combined paths, numerical and checkpoint gates, paired execution-cost evidence and a tiny execution smoke; no quality conclusion.

**Gate:** outputs, losses, full gradients, visibility and checkpoint continuation satisfy the fixed tolerances; measure complete-step cost before selecting a tail layout. Keep greedy coordination distinct from the combined stochastic joint sampler. A quality-versus-cost study is deferred, not inferred from the smoke.

### Step 11 — Implement the user's energy-guided stochastic remasking

Implemented2026-10-25; exact design, director decisions and evidence: [docs/remasking.md](docs/remasking.md).

- Optional `sbd.refinement` implements R²D with K=eligible count, M_paper4/five calls per level, patience2, current Codpula FFBS and retained soft feedback. Final commitments/prefix remain immutable; new commitments wait until the episode ends.
- Uniform, unary and pairwise are executable mask-set policies, not authorized quality-comparison arms. The learned law keeps epsilon0.1/T1 and exact fixed cardinality; unary uses stable host-f64 DP and pairwise enumerates all legal sets through L16, rejecting larger blocks. Features are contextual524->32; pairs are symmetric and include all within-block positions.
- Actual frozen-model paired trajectory collection uses512 states/four pairs/H2 by default, identical work/draws per pair, half proposal-weighted swaps and half independent sets. Clean-target conditional joint NLL determines ranking; inactive/tied pairs get zero weight. Offline fitting trains unary first, then pair terms with predecessor softplus-margin semantics. No confidence threshold or deterministic correction-value head substitutes for this mechanism.
- Explicit `sbd.training_mode:tct` implements streaming K_train4 chains, confidence0.9 plus scheduled minimum, all-free-position conditional joint NLL plus clean causal loss. Provisional inputs never become circuit evidence. Optional `tct.rows_per_pass` accumulates normalized neural gradients and unscaled circuit flows before one full-batch update; the selected B8 config uses four rows/pass, the first requested candidate that fit.
- Numerical gates: exact-law enumeration/sampling, f64 energy/rank VJPs, state/RNG/immutability/padding/L1, joint TCT flows and exact chain/optimizer restart, CPU/Vulkan parity, old identities/checkpoints. Workspace tests85 passed/one pre-existing ignored; new remasking max abs3.30547e-6, TCT including row accumulation9.53674e-7, unchanged tolerance2e-5+3e-4*abs(ref).
- Execution evidence: B1/T512 TCT per-block/batched medians17562.660/1845.395ms, 2warmups/11interleaved pairs; paired saving15711.721ms [15625.973,16349.399], so retain batched blocks. B8/T512/four-row accumulation:7267.660ms, sampled pool4399520256bytes after one warmup. B8 R²D resident-prefix episode:3342.933ms including clean replay, 38noisy batch calls; initial prefill is separate. These are execution costs, not quality evidence.
- A pre-existing GPU continuation failure was reproduced on the unmodified Codpula base and repaired separately by deterministic native AdamW clipping (64590d18); fused optimizer arithmetic/records remain unchanged, historical causal checkpoint13 still passes.

**Deliverable/gate, as explicitly decided:** implemented commands/configs, numerical and exact-law gates, tiny context32 smokes for generation/collection/fitting/guided generation/TCT, execution measurements and documentation. No quality study, ablation, sweep, calibration grid or quality/cost curves were authorized or run.

### Step 12 — Not started; outside this integration

No scaling study, adaptive boundaries, mixer alternatives, new parameterization
or sparse mechanism is authorized here. Earlier Step12 experimental proposals
are not pending execution tasks. Any such work requires a separate explicit
user request; the current deliverable composes the given architecture through
Step11 with correctness and execution-cost checks only.

## 8. Measurement and correctness requirements

- Record resolved configs, seeds, software/backend versions, actual parameter counts, data split, trained tokens, state/cache norms, gradient norms, peak memory and build/run timings.
- Separate cold build, warm rebuild, first-use GPU compilation/autotuning, and warm execution. Do not attribute all waiting to Rust compilation.
- Measure wall-clock quality frontiers and compute-matched comparisons. Count partial tail passes, causal replay, circuit queries, auxiliary models, energy scoring and policy sampling. Full-backbone NFE alone understates CoCommit and circuit costs.
- Keep generated-length/EOS rules fixed or report length differences; shorter answers can otherwise look like faster decoding.
- Verify circuit partition functions, conditional marginals, joint sampling and optimizer positivity/row sums against brute force on tiny domains. Conditional queries reuse frozen neural outputs; they are not new network evaluations with evidence edited into the input.
- Keep prompt and completed-prefix tokens immutable in the initial model. Reopening completed blocks would require rebuilding recurrent state, exact-memory entries and convolution history from an earlier snapshot; it is a separate future experiment.
- Never suppress nonfinite values or silently fall back to an independent head when the joint model fails. Preserve the failing example and report the error.
- Build and benchmark commands have explicit time limits. A timeout is a recorded outcome, not a reason to leave a long command running while withholding the available result.

## 9. Approved contextual, coordination and refinement composition

The `integrated11` composition was published on main as **`9e67b7ac`**.
Its described parents preserve contextual `a0c173c5`, remasking `b9e589b5`,
documentation-only precision diagnosis `093d0085`, and final CoCommit
`3c20ffc2`; clipping `64590d18` and deterministic Tensor Lngram `e50cc523`
remain separate ancestors. Described history was not rebased or squashed.
Publication preceded complete combined acceptance. Preserve the original
negative stress evidence and its independent primitive/cancellation diagnosis;
it is not a production fix or a replacement for the unchanged actual-training
objective gate. Exact commands and partial outcomes are in
[RESULTS: integrated11 publication checkpoint](RESULTS.md#integrated11-publication-checkpoint),
grounded in `artifacts/integrated11/commands/`.

The selected architecture and highest-return choices stand; this is composition
of the approved features, not a new mechanism or a selection experiment:

1. Keep permanent observations `O`, provisional reveals `R`, currently masked
   positions `M`, and temporary announcement marks `S` distinct. Only the
   episode-end write creates new permanent observations.
2. Reuse CoCommit's saved noisy-prefix and shared A/B tail primitives. Unmarked
   A selects a bundle; marked B restarts from the exact saved prefix, not from
   A's tail state. Position-only HOLA membership and graph-free teacher changes
   from the final CoCommit parent must be retained.
   Production remask/TCT prefix, A/B tails and circuit preparation stay batched
   across active blocks; do not call the single-block assignment helper in a row
   loop. Host bundle choices may loop over metadata from a batched readback.
   Generalize the existing output helper rather than launching one neural or
   circuit graph per block. Explicit per-block correctness oracles are not the
   production composition path.
3. Standard R²D chooses `S subset M`; B jointly samples the revisable positions
   with temporary proposal evidence `O+R` and carries R. Correction mode chooses
   `S subset A` for `A=valid minus O`, and B may revise R. Use one coherent
   B-prepared joint proposal, never splice A/B samples. Marks are neither
   evidence nor final commitments. If S is empty, use A unchanged, skip B and
   record the skipped marked pass. Feedback/energy always use the chosen
   output's O-only full marginals and final hidden states.
4. TCT retains unmarked all-free correction joint NLL given O, including wrong
   provisional R. B's additional joint event targets its announced S subset A,
   leaving other free positions marginalized. Observed-potential masking is
   only O. Detached B proposals/marginals advance the existing confidence reveal
   chain; its earlier provisional errors remain inputs as specified by TCT.
   Combine neural gradients and circuit flows/counts with CoCommit's branch
   weighting before one optimizer/circuit update.
5. Preserve exact accumulation, not a smaller batch: ordinary SBD/CoCommit uses
   explicit `sbd.rows_per_pass`, sampling full-batch masks/bundles once. TCT uses
   its separate persistent-chain `tct.rows_per_pass`; the two accumulators must
   never nest. Both retain original B*T reduction, indexed RNG and one update.
6. Record a combination-specific algorithm revision in training/pilot identity
   only when CoCommit composes with refinement or TCT. Old standalone/omitted
   identities remain unchanged. Tracked configs consist of one contextual+
   CoCommit+refinement configuration and matching TCT configuration, without
   sparse retrieval in toy defaults or a factorial grid.
7. Count logical R²D iterations separately from shared prefix, A/B tail/head
   calls, circuit preparations/queries/FFBS and final clean replay. Existing
   standalone timing tables remain historical for their own profiles and are
   not same-cost combined-model comparisons.

The publication checkpoint completed only part of final acceptance:
workspace tests, contextual/CoCommit/TCT CPU and Vulkan references, corrected
consumer fixtures, and context32 combined uniform-policy train/save/load/
evaluate/generate. The actual collection command failed before collecting;
the subsequent loader change was not rebuilt or rerun before publication.
Energy fitting/guided generation, five-update/resumed/warm-start TCT CLI,
full B8/T512 rows4 training, and historical/component/`vulkan-fused` reruns
were not executed. The complete command ledger and both original fixture
failures are in RESULTS and `artifacts/integrated11/commands/`.
These are execution proofs, not quality results. Keep negative radial-stress
evidence separate from actual normalized-objective acceptance. No quality
study, sweep, calibration grid, lower batch/context or extra training is
authorized by this status correction.
