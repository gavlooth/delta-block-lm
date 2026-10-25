# Architecture implementation and optimization plan

Status: Steps 1–7 implemented and exercised on CPU and local Intel Vulkan; Step 8 is partially implemented, with sparse/within-block attention undecided pending explicit user selection. The added dense active-block attention has been removed; HOLA's pre-existing attention is unchanged. Measured results and checkpoint-load correction are in [RESULTS.md](RESULTS.md). Prior Step-8 checks and short training smokes describe the historical dense-attention implementation, not the current architecture or a quality/speedup conclusion. Steps 9–12 are not implemented.

**Scope override — explicit user instruction:** implement and optimize the given architecture. Its components are requirements, not candidates that must win architecture-selection experiments. Correctness checks and focused runtime/memory measurements may verify an implementation optimization. Do not launch quality ablations, architecture comparisons, candidate sweeps, alternative-mechanism studies or research training runs unless the user explicitly requests them. This overrides experimental comparisons, sweeps and quality-adoption gates elsewhere in this document; those passages are historical proposals, not executable tasks. Steps 9–12 are implementation/optimization work only where they implement the specified architecture, not authorization to add alternatives.

## 1. Goal and fixed decisions

Build and optimize the specified semi-autoregressive block-denoising language model. The predecessor is `diffusion-relation-extractor`; its relation-specific output heads and ontology constraints are not requirements for this LLM.

- Rust, Burn and custom CubeCL kernels. Intel Arc/Vulkan locally; DGX Spark/CUDA for larger runs after arranging GPU access. Follow `AGENTS.md` for build limits, mold, debug-free tests and experiment records.
- Gated DeltaNet-2 (GDN-2) is the reference mixer. KronLSTM + HOLA is the principal challenger against GDN-2 + HOLA, not an assumed winner.
- Keep wavelet token mixing, orthogonal hyper-connections (oHC) and three Lngram orders in the target design. Isolate their costs and contributions rather than enabling everything in the first run.
- Use Set Block Decoding (SBD) as the initial semi-autoregressive recipe. Block length, computational chunk length and tokens committed per denoising step are separate variables.
- Preserve the predecessor's soft top-k diffusion mechanism as the block-denoising baseline: probability-weighted candidate feedback, separate visibility, and revisable proposals. Hard-mask-only feedback is an ablation, not a replacement. Add joint token prediction, coordination before writes, and revision after provisional writes on this foundation.
- Energy-guided stochastic remasking is the user's research proposal against pure random remasking. It must not be replaced by confidence ranking or a deterministic correction-value predictor.
- Preserve the specified architecture while optimizing execution time and memory. Verify numerical correctness; do not require components to earn inclusion through quality comparisons.

## 2. Sources and evidence boundaries

| Mechanism | Primary source | What the source establishes |
|---|---|---|
| GDN-2 | [2605.22791](https://arxiv.org/abs/2605.22791) | Channel-wise decay and separate channel-wise erase/write gates; chunkwise training |
| KronLSTM | [2610.26834](https://arxiv.org/abs/2610.26834) | A different associative-memory update using a scalar bilinear correction; a candidate replacement, not a drop-in extra gate |
| HOLA | [2607.02303](https://arxiv.org/abs/2607.02303) | GDN recurrent state plus a limited exact KV cache selected by committed update magnitude; sharpened cache read |
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

### Primary comparison

Use GDN-2 + HOLA and KronLSTM + HOLA with the same embedding, head dimensions where applicable, training data, precision, optimizer and token schedule. Report actual parameter counts and recurrent-state memory; equal width alone does not imply equal capacity or equal cost.

Include GDN-2 and KronLSTM without HOLA as the controls needed to isolate the cache contribution. Start on multi-query associative recall, then use a tiny causal language model before adding block denoising.

HOLA retains original KV pairs, not the gated/preconditioned update factors. Its cache-read Q/K normalization is separate from recurrent-update normalization. Compare surprise selection with a matched-size recency cache. The local current-chunk attention supplied by HOLA must be counted in memory and work.

Proposed surprise scores, requiring implementation checks:
- GDN-2: norm of the active rank-one state change after decay, not the total difference including decay.
- KronLSTM: absolute scalar correction multiplied by key and value norms.

A cache may not be updated from noisy or bidirectional block representations. After completing a block, causally replay its final tokens to update recurrent states, HOLA entries and convolution histories exactly once.

### Compatible additions and alternatives

- Preconditioned write keys: optional GDN-2 experiment after the base comparison. Preserve separate erase/read and write factors and include gradients through preconditioner statistics.
- Exact Flow Linear Attention ([2512.12602](https://arxiv.org/abs/2512.12602)): optional gate/update parameterization. Its exact solution applies to its specified continuous-time dynamics; it is not an exact solution of the full GDN-2 + channel-wise decay + SBD model. Any extension needs a derivation and recurrent/chunked gradient checks. It has nonzero arithmetic cost, even if small.
- GSA-2 ([2610.02816](https://arxiv.org/abs/2610.02816)): later alternative two-memory architecture. Do not label it a free GDN-2 add-on.
- Erase-then-Delta ([2606.26560](https://arxiv.org/abs/2606.26560)): later independent erase-address experiment; account for its extra recurrent work.
- Dynamic Linear Attention ([2606.10650](https://arxiv.org/abs/2606.10650)): later multi-state research, not exact concatenation of delta-rule states by addition.
- A layer-interleaved GDN-2/KronLSTM hybrid is conditional on evidence of complementary strengths. Do not start with a KronLSTM/GSA-2 hybrid or an all-mechanism combination.

HOLA is the first selective exact-attention mechanism to test, not proof that all separate sparse/local attention is unnecessary. Keep sliding-window attention as a comparison. MSA, NSA/MoBA, CSA/HCA, BA-Att, SparseD and LoSA remain later sparse-attention references; their published kernels and settings do not establish speedups for this model. LoSA is especially relevant to repeated block denoising. Add an alternative only when a measured failure motivates it.

### Source reuse, not a fresh reimplementation

The predecessor sources needed for these components are copied under `reuse/crates/`:

| Copied source | Reuse target and necessary adaptation |
|---|---|
| `dre-gdn2/` | Existing CubeCL chunked scan, forward/backward operators, sequential reference and tests; adapt bidirectional encoder use to causal-prefix/block semantics |
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

The existing vocabulary head supplies token potentials from final-normalized noisy features. A valid-state mean pool supplies 64 biased linear log-gates, shared across edges with group width four. Local emission, transition-row and visible-evidence normalizers are included in conditional joint NLL. Circuit rows use the explicit nonnegative dual-flow split, summed occurrence reaches, separate bias-corrected momentum 0.9 and constant alpha 0.05; backbone/head/gate parameters retain existing AdamW.

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

Steps 1–7 are implemented and exercised as small pilots. Numerical, soft-feedback, build and checkpoint-continuation checks passed; three-seed recall and tiny-LM experiments were run and independently reevaluated from validated checkpoints. Recall remains near guessing, so its historical architecture-selection gate is not met and no retrieval winner is asserted. Tiny-LM learning is demonstrated at a deliberately short budget, not at production scale. Step 7 added causal oHC, wavelet and three Lngram injections with verified gradients and optimized kernels; quality benefit remains unestablished, and the cancelled quality ablation must not be resumed. Step 8 retains clean causal plus masked-block training, soft feedback, frozen prefix reads and clean replay, but is partial: the added dense active-block attention is removed and sparse/within-block attention awaits explicit user selection. Its prior visibility checks and fixed/variable-length training smokes are historical evidence for the removed dense-attention implementation, not verification of the current architecture. Steps 9–12 remain pending. The scope override above remains in force. See [RESULTS.md](RESULTS.md) and [docs/sbd.md](docs/sbd.md).

### Step 1 — Create the small active Burn workspace

- Create crates for shared model types/references, GPU kernels, model components and the experiment executable; do not copy the predecessor application.
- Configure CPU checks without GPU features, opt-in Vulkan/CUDA runs, mold locally, debug=false for dev/test, and one integration-test binary per crate.
- Check actual Cargo profile inheritance: optimize external dependencies without accidentally optimizing every local workspace crate.
- Add file-based experiment configuration and machine-readable results from the first runnable experiment. Configurations select mixer, dimensions, cache size, seed and backend without recompilation.
- Start by preserving the copied GDN kernel dimensions (four heads, key width 32, value width 64, computational chunk 16). Projection/model width is a separate choice.

**Deliverable:** a runnable CPU reference command and a minimal Vulkan command using the same input fixture.

**Gate:** CPU command runs, Vulkan reports the intended device, and build/rebuild times are recorded. Use a 120-second limit on the first cold build; if it expires, report that outcome rather than extending the wait silently. Warm model rebuild target is <=10 seconds; test rebuild target is <=60 seconds. Do not trigger a full-workspace test build to measure one component.

### Step 2 — Migrate and verify GDN-2 before changing its recurrence

- Copy useful reference, forward/backward and CubeCL code from `reuse/` into the active crates, keeping source provenance.
- Separate the causal recurrence needed by the LLM from the predecessor's bidirectional encoder wrapper.
- Compare chunked outputs, final states and parameter/input gradients against the float64 sequential reference.
- Cover a sequence shorter than one chunk, multiple chunks, padding, packed-document resets, and block starts inside a chunk.
- Exercise actual GPU forward and backward on a small fixture. Record first-use compilation separately from warm execution.

**Deliverable:** a numerically checked GDN-2 implementation plus a short CPU/GPU benchmark.

**Gate:** tolerances are specified by precision, gradient checks pass, and no document/padding information crosses its allowed boundary. Preserve finite-input failures rather than suppressing them.

### Step 3 — Migrate the existing soft top-k diffusion foundation

- Reuse the copied soft state, weighted embedding, feedback scheduler and proposal/retained-answer semantics.
- Generalize annotation values to token candidates: candidate IDs/weights, visibility, provisional values and retained values stay separate.
- Remasking must not silently delete the soft distribution. Observed prompt tokens stay fixed.
- Keep the dense probability-weighted path as the numerical reference. Port or implement sparse selection/gather/update only where needed; do not confuse source-boundary top-k with vocabulary top-k.
- Make candidate width configurable independently of block length, recurrent chunk length and HOLA capacity.

**Deliverable:** a runnable miniature token-refinement episode using the reused soft-feedback mechanism, before adding learned energy selection.

**Gate:** sparse/full-support feedback agrees with the dense reference where their support matches; embedding/weight gradients, retained mass, zero-mass behavior and visibility transitions are checked. A hard-mask-only path exists solely as a comparison.

### Step 4 — Implement the KronLSTM challenger and HOLA memory

- Implement KronLSTM's sequential float64 recurrence first, then its chunked forward/backward and GPU path.
- Implement HOLA's exact KV cache and sharpened read as shared model components where semantics allow, with mixer-specific update-magnitude scores.
- For GDN-2, derive scores from the active edit after decay; for KronLSTM, derive them from its scalar correction. Verify both rather than assuming the same residual formula.
- Compare surprise-based selection with a cache of the same capacity using recency selection. Count HOLA's current-chunk attention work.
- Start cache capacity at 16; measure 64 after the first pilot.

**Deliverable:** GDN-2 and KronLSTM, each runnable with and without HOLA.

**Gate:** recurrence/chunk agreement and gradient checks pass; causal cache membership and padding are correct; both mixers execute forward/backward on the local GPU.

### Step 5 — Run the first controlled associative-recall experiment

- Start with two mixer layers, model width 128 and contexts 128/512. Keep the inherited kernel key/value widths initially.
- Generate deterministic training examples and a disjoint fixed evaluation set. Sweep the number of key/value associations to expose memory interference.
- Run four arms: GDN-2, GDN-2+HOLA, KronLSTM, KronLSTM+HOLA. Add the matched recency-cache comparison without scaling the entire grid.
- Run one seed as a smoke, then three paired seeds for an architectural decision. Use a short throughput pilot to choose and record a fixed training-token budget for the comparison.
- Report recall by association count and distance, actual parameter count, recurrent/cache memory, peak GPU memory, training throughput and warm decode latency.

**Deliverable:** one comparison report with the configurations and raw metrics.

**Gate:** choose a quality/cost Pareto shortlist, not a winner from training loss or a single seed. Keep both leading cache-equipped mixers if neither dominates. A synthetic recall win does not establish better language modeling.

### Step 6 — Check the shortlist on a tiny causal language model

- Start with four layers, model width 256 and context 512, retaining supported kernel widths. These are pilot sizes, not final architecture decisions.
- Pin a small, legally usable text corpus and tokenizer; split by document before packing or sampling. Record source/version and split hashes.
- Fix optimizer, data order, token budget, precision and output head across mixer arms. Report actual parameter/state differences.
- Measure held-out next-token loss/perplexity, throughput, memory and retrieval diagnostics. Select a baseline for the denoising experiments while retaining the challenger configuration.

**Deliverable:** a resumable tiny LM checkpoint and held-out comparison.

**Gate:** the model learns on held-out data and the recall advantage survives enough to justify its cost. Resolve nonfinite states/gradients before adding mechanisms.

### Step 7 — Add oHC, wavelet mixing and the three Lngram components

- Reuse the copied implementations, introducing one component at a time on the selected small backbone.
- Preserve all three Lngram components in the target design; verify the predecessor's distinction between injection count and n-gram orders before mapping configuration fields.
- Check stream mixing, table addressing and surrogate gradients, and account for residual-stream/table memory.
- Use causal masks/history in the causal model and define the future-block visibility rules needed by the next step.

**Deliverable:** the requested small backbone with attributable component measurements.

**Gate:** measure component costs and verify correctness. **DO NOT IMPLEMENT OR RUN the Step 7 quality ablation against the pre-Step-7 baseline.** Cancelled by explicit user instruction; do not resume, extend, or replace it unless the user explicitly requests it. Quality benefit remains unestablished. Keep the implemented components.

### Step 8 — Train the SBD-style block-denoising path

Partially implemented: clean/masked objectives, soft feedback, frozen prefix reads and clean replay remain. The added dense active-block attention and its attention-only oracle are removed; sparse/within-block attention is undecided pending explicit user selection. Prior reference gates and context-32 fixed/variable-length and hard-mask training smokes are historical evidence for the removed implementation, not completion or verification of the current attention architecture. See [docs/sbd.md](docs/sbd.md). The scope override prohibits new comparisons and sweeps without explicit user instruction.

- Add the clean causal objective plus masked future-block objective, retaining the soft top-k input/update mechanism from Step 3.
- Noisy positions read the clean state before their block without advancing persistent state. Sparse/within-block attention is not yet selected or implemented; do not substitute dense attention or another mechanism.
- Verify short convolutions, wavelet taps, Lngram routing and exact-memory reads obey this visibility rule.
- Causally replay each finalized block exactly once before extending the prefix.
- Compare fixed block lengths 4/8/16 with a variable-length training condition, keeping computational chunk length independent. Include length 1 as a causal limiting-case check.
- Sweep candidate width separately. Add larger blocks and learned/adaptive boundaries only after measuring the fixed/variable references; do not impose 16 as a permanent limit.

**Deliverable (partial):** training, soft top-k feedback and clean replay are implemented; the semi-autoregressive attention architecture remains incomplete pending explicit user selection.

**Gate (partial):** retain checks for future-target leakage and agreement between prefix replay and causal execution. Prior block NLL/accuracy and finalized tokens/s are historical dense-attention measurements, not current-architecture results. The removed attention oracle does not cover any future selected operator. Hard-mask-only feedback remains a historical ablation, not the default or authorization for a new run.

### Step 9 — Add Codpula's joint output and adaptive commitment

**Implementation, numerical gates and execution-cost measurement complete (2026-10-25).** Quality studies were explicitly excluded from this turn.

- 32-state homogeneous joint output, all 257 symbols, final-normalized potentials, b=4 coupling gates, required normalizers and dual-flow circuit optimizer are integrated across cached/per-block/batched/reconstruction/evaluation/generation paths.
- Independent f64/enumeration, full VJP/flow, joint-sampling, cavity, padding/observations/L1, CPU/Vulkan model and checkpoint/optimizer continuation gates pass at `2e-5 + 3e-4*abs(ref)`. Historical factorized identity is preserved; obsolete dense-SBD payloads stay rejected.
- Fused leaf/chain/marginal/flow kernels with analytic VJP pass. Keep tensor circuit execution selected: B1/T512 paired fused saving 60.951 ms has 95% interval [-23.545,84.668], despite an isolated 20.025 ms [16.577,23.311] gain. Factorized/tensor/fused complete-step medians are 1522.502/1697.014/1631.830 ms.
- Selected B8/T512 tensor diagnostic: 8000.071 ms, 30,745 launches and 8,826,550,976 bytes sampled pool reservation. Tiny context32 smoke: 96 bytes/three steps, checkpoint evaluation and eight dependence-committed/replayed tokens; disposable model/optimizer payloads removed.

**Delivered:** verified joint predictor and fixed/dependence decoder support, current-predecessor salvage with provenance, exact tests and execution-cost accounting. See `docs/codpula.md`, `RESULTS.md` and `artifacts/codpula/`.

**Gate result:** valid circuit rows, conditional normalization and full-support inference pass; latency is measured. The original quality/latency study against an independently trained factorized baseline, input-only comparison and 64-state study remain unevaluated by explicit user direction, not inferred from a tiny smoke.

### Step 10 — Add CoCommit, then test the combination

- Implement the unmarked selection tail and marked assignment tail starting from the same saved pre-tail hidden states.
- Train the marked branch; start with one tail layer and one coordination round on the tiny model.
- Check saved-h zero-marker equivalence; measure CoCommit off/on step cost and equivalent tail execution layouts only. No trained comparison arm, quality study, sweep or ablation is authorized.
- In the combined path, regenerate circuit potentials/gates and dependence scores after marked coordination. Separate-versus-combined quality comparisons require a future explicit authorization.
- Reset temporary recurrent-tail state from the same clean-prefix snapshot for each branch. No coordination pass commits persistent state.

**Deliverable:** implemented separate/combined paths, numerical and checkpoint gates, paired execution-cost evidence and a tiny execution smoke; no quality conclusion.

**Gate:** outputs, losses, full gradients, visibility and checkpoint continuation satisfy the fixed tolerances; measure complete-step cost before selecting a tail layout. Keep greedy coordination distinct from the combined stochastic joint sampler. A quality-versus-cost study is deferred, not inferred from the smoke.

### Step 11 — Test the user's energy-guided stochastic remasking

- Establish uniform R²D inside the active block and preserve the same eligible positions, remask counts and decode budget for each competing policy.
- Train unary mask-set energies from a small set of paired equal-budget refinement trajectories; retain the uniform exploration mixture.
- Compare against pure random and confidence/entropy diagnostics. Test on held-out trajectories, not the policy's training examples.
- Add pairwise mask-set energies only after the unary experiment; include cases where jointly wrong tokens cannot be fixed one at a time.
- Verify mask-set normalization/sampling by enumeration on small blocks; specify and measure any larger-block approximation.
- Keep TCT as a separate training comparison so improved denoiser training is not mistaken for a policy gain.

**Deliverable:** random versus unary-energy versus pairwise-energy quality/cost curves, with policy-data generation cost reported.

**Gate:** evidence that context-guided stochastic exploration improves on pure random at matched total work. A confidence threshold or deterministic correction-value head does not satisfy this step.

### Step 12 — Scale only supported choices and test remaining mechanisms

- Increase context, model width/depth, block length and memory/circuit capacities one axis at a time.
- Test learned/adaptive block boundaries against the measured variable-length baseline; account for selector training and inference cost.
- If the leading mixers have complementary failures, test layer-interleaved GDN-2/KronLSTM. Do not assume mixing them is better.
- Test preconditioning/exact-flow parameterizations, GSA-2, independent erase addresses or another sparse mechanism only against an identified failure/cost.
- Arrange a Spark GPU window before CUDA runs; do not interfere with predecessor production training.

**Deliverable:** a measured scaled configuration, plus rejected alternatives and reasons.

**Gate:** gains persist at realistic sequence lengths and batch sizes without exceeding memory/build limits. No full factorial sweep and no large run before the local pilot answers the cheaper questions.

## 8. Measurement and correctness requirements

- Record resolved configs, seeds, software/backend versions, actual parameter counts, data split, trained tokens, state/cache norms, gradient norms, peak memory and build/run timings.
- Separate cold build, warm rebuild, first-use GPU compilation/autotuning, and warm execution. Do not attribute all waiting to Rust compilation.
- Measure wall-clock quality frontiers and compute-matched comparisons. Count partial tail passes, causal replay, circuit queries, auxiliary models, energy scoring and policy sampling. Full-backbone NFE alone understates CoCommit and circuit costs.
- Keep generated-length/EOS rules fixed or report length differences; shorter answers can otherwise look like faster decoding.
- Verify circuit partition functions, conditional marginals, joint sampling and optimizer positivity/row sums against brute force on tiny domains. Conditional queries reuse frozen neural outputs; they are not new network evaluations with evidence edited into the input.
- Keep prompt and completed-prefix tokens immutable in the initial model. Reopening completed blocks would require rebuilding recurrent state, exact-memory entries and convolution history from an earlier snapshot; it is a separate future experiment.
- Never suppress nonfinite values or silently fall back to an independent head when the joint model fails. Preserve the failing example and report the error.
- Build and benchmark commands have explicit time limits. A timeout is a recorded outcome, not a reason to leave a long command running while withholding the available result.
