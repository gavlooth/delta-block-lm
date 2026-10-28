# Predecessor inventory: diffusion-relation-extractor → delta-block-lm

Read-only audit, 2026-10-25. Source: `/home/heefoo/Documents/code/diffusion-relation-extractor` (current working tree; Burn 0.22.1 / cubecl 0.11.1, seven vendored patches). Snapshot reference: `reuse/PROVENANCE.json` (copied 2026-10-24, 48 files hash-verified). Every hash-differing file was diffed; line numbers below are from the current predecessor tree.

Excluded by assignment: `dre-relations/src/joint_circuit.rs` and `commitment.rs` (owned by the Step-9 Codpula agent). Their dependencies are listed in §3.

## 1. Components serving remaining PLAN work

### 1.1 Step 9 — Codpula support pieces (beyond the excluded files)

| Component | Predecessor path:lines | Implements | Serves | Verification in predecessor | Adaptation for byte LM | Recommendation |
|---|---|---|---|---|---|---|
| Circuit rollout controller (current) | `dre-relations/src/diffusion_scheduler.rs` — `RefinementWindow` 11–18; `normal_update` 30–39; `refinement_transition_window` 125–134; `refinement_transition_features` 136–216; `run_pointer_window_with_hooks` 248–255; `refinement_rollout_window` 265–301 | Drives multi-call episodes: per-round proposal uniforms, detach-between-steps, supplied-evidence injection, absorb commitments between calls, no persistent-state writes; returns per-round readouts + `CommitState` | Step 9 (rollout/sampling driver), Step 10 (temporary state between branches) | `diffusion_scheduler_tests.rs`: `a5_observed_endpoints_seven_pair_round_values_decisions_rng_and_gradients_cpu` 166, `…_cuda` 170, `cached_a5_rounds_preserve_values_gradients_rng_and_commitments_cpu` 275, `shared_endpoint_tables_…` 324. Session report 2026-10-24: v6 deployed; CPU+guarded-CUDA gates passed | Replace 5 relation fields (R/SS/SE/OS/OE) with block token positions; replace span/pair windows with one block layout; keep uniform-per-round and absorb-on-commit semantics | **Port with changes** (core support piece; depends on the excluded files) |
| Denoiser step pipeline | `dre-relations/src/denoiser.rs` — `prepare_refinement` 505; `step_pointer_features` 515; `step_pointer_committed` 498; `queries_pointer_committed` 398; `DocumentCache` 231–284 (project banks once, reuse every round); `AnnotationDenoiserConfig` 68–108 | One round of S1 → graph interaction → S2 → heads on a cached, frozen layout; per-layout cached transforms (`RefinementCache`); query assembly variants (plain / supplied / committed) | Steps 8–10: same clean-prefix snapshot for every denoising/coordination pass | Scheduler integration tests exercise the real pipeline; production trained on it. Session report: per-pass caching is a first-class cost control (lazily built, discarded before optimizer) | Graph/sparse-bank reads map to NEW's own in-block attention + MoBA prefix retrieval; keep the "one prepared snapshot, many passes" pattern | **Reference only** (architecture pattern); NEW already has its own backbone, needs the pattern, not the code |
| Soft state semantics | `reuse/crates/dre-relations/src/state.rs` (unchanged since copy) — `DenoiseSchedule` 36–58; `SoftState` 79–93; impl 130–198 | Convex eta/visibility updates that preserve soft beliefs; separate visibility; observed fields immutable | Step 3 (already reused in NEW `experiment/src/soft.rs`); Step 9 conditioning | Copied; NEW `soft.rs` mirrors it | Already adapted to vocabulary candidates in NEW | **Already done** (no action) |
| Slot/layout planning | `dre-relations/src/plan.rs` — `SlotPlan` 208+; `query_axis_bucket`/`pair_axis_bucket` 196–207; `with_slot_padding` 397–413; `PAIR_SLOT_GRANULE=128` 201 | Host-built batch layout: packed rows, slot padding, bucketized execution axes to reuse compiled/autotuned kernels | Step 12 (block-length variability without kernel recompilation); Step 9 (candidate layouts) | Exercised by every training/extraction path; padding correctness covered by dirty-padding tests | Drop entity-pair slot semantics; keep bucketing/padding discipline | **Port patterns** (bucketized padded axes) |
| Multi-call joint training | `dre-relations/src/training_query_chain.rs` — `QueryForwardInputs` 19 (proposal_uniforms); `joint_query_forward` 61; `joint_query_forward_window` 69; `joint_query_forward_supplied_window` 77; `joint_query_loss` 114; `QueryScoreCache` 202; `MatchingByteTile` 175; per-round prodetach test 673 | Recurrent supervised joint-NLL across every call with exact per-field evidence; shared transitions; detached assignment; `detach_between_steps` option | Step 9 (train joint objective, dual-flow substitute), Step 11 (trajectory training base) | CPU tests `a5_*` pass; v6 production deployed on this frame (10-call /10D objective, session report 2026-10-24) | Replace relation/pointer targets with token-position targets; drop Hungarian matching unless candidate assignment is needed | **Port with changes** |
| Per-document batch loss | `dre-relations/src/training_query_batch_loss.rs` — `QueryLossDocument` 15; `PreparedBatchLoss` 58; `joint_query_batch_loss` 310 | Exact supervised losses per document; per-document evidence-sampling RNG advances explicitly handed to energy supervision | Step 11 (paired trajectory RNG discipline) | Covered by physical-batch regression tests (session report: `cpu-complete.json` 195+ cases) | Same token-position substitution | **Port concept** |
| Learned-output utilities | `dre-relations/src/learned_output.rs` — `LearnedOutputMlp` 12; `probability_entropy` 152 (exact entropy, zero-mass-safe); `connected_components` 180; `KnownBinaryTargets` 78; `zero_loss` 136 | Terminal-head MLP helpers and numerically safe feature reductions | Step 9/11 feature and terminal readouts | Entropy correction proven on CPU+CUDA (session report 2026-10-19); `probability_entropy` specifically regression-tested | Relation heads dropped; entropy/stable-reduction patterns reusable | **Reference only** |
| Final decode | `dre-relations/src/decode.rs` — `decode` 37–70; `best_span` 76–95 | Argmax label + temperature-scaled posterior from final state; ordered span decode | Step 9 write step | Unit tests (low-temperature uniform/zero-mass; ordered spans) | Reuse calibrated softmax-from-state logic for token writes | **Port with changes** (token variant). Bug: predecessor decode.rs 56–61 turns an all-zero probability row into NaN (`-inf - -inf`); its mixed-zero test misses it. Any port must reject zero-mass rows with an error and test an all-zero row. |

### 1.2 Step 10 — CoCommit coordination

No direct predecessor implementation exists (no tail-marking mechanism). Reusable patterns only:

| Pattern | Path:lines | What it offers | Recommendation |
|---|---|---|---|
| Cached pre-tail invariants reused across many passes | `denoiser.rs` `DocumentCache::prepare`/`prepare_refinement` 231–284, 505; `boundary_pair.rs` `BoundarySourceProjection`/`prepare_source_projection` 56–73, 677; `forward_with_source_projection` 717 | Compute shared projections once, rerun only the variable tail per branch — exactly the "cache hidden states before final tail layers" skeleton | **Reference only**; implement marker injection and two tail passes fresh |
| Same-snapshot multi-round control | `diffusion_scheduler.rs` `refinement_rollout_window` 265–301 | Runs every round from the same prepared state; no persistent state mutation between branches | **Reference only** |
| A3 precedent (removed) | `docs/references/refinement.md` (history) + deleted `remask.rs`: shared 2×512 trunk, round embeddings, per-field keep/remask projections, retained markers | User earlier approved an *action-outcome binary remask policy*; it was explicitly deleted in the v6 Codpula cutover and must **not** be resurrected as CoCommit | **Skip** (history only) |

### 1.3 Step 11 — energy-guided stochastic remasking + R²D/TCT

**Primary source: `dre-relations/src/energy_guidance.rs` (1,540 lines), design doc `docs/references/guidance.md`.**

Key API/lines:

| Piece | Lines | Implements |
|---|---|---|
| `EnergyGuidanceConfig`/`EnergyGuidance` | 186–210 | Energy model: local linear (context→labels) + symmetric bilinear `interaction` (compatibility, `(W+Wᵀ)/2`) + 4 learned `rule_weights` |
| `EnergyTopology` | 224–338 | Uploaded once per plan; unique entity-sharing edges; CSR `IncidentReduction` (deterministic, no atomics); inactive/padding zeroing |
| `forward_projected` / `gradient` | 745–764, 766–779 | Energy = Σ local terms + Σ edge terms; exact analytic derivative w.r.t. soft assignments |
| `correct_logits` | 781–786 | Optional approximate gradient correction `logits − λ∇E` (off by default) |
| `rerank` / `discrete_energies*` | 788–814, 842–912 | Final categorical ranking; host FP64, ordered terms, candidate-major; equal energies keep lowest index |
| `SoftConstraintFactors` | 919–1104 | Sparse polynomial rule features and CSR incident terms (type/antisymmetry/exclusion/cardinality) |
| `energy_rank_loss` | 1106–1115 | `softplus(E_gold − E_neg + margin)` with eligibility mask |
| `corrupt_plausible_graph(_per_document)` | 1117–1182 | Negative sampling: replace one eligible slot/doc with a model-plausible non-gold label; per-document RNG streams |
| Tests | unit: dirty-padding 483–537; dense-rank equivalence 565–677; CSR incident 1319–1540; integration `tests/energy_guidance.rs` (CUDA gated finite-difference) | Strong numerical coverage (f64 references, categorical=dense, FP64 final ranking) |

**User architecture decisions already made here (preserve):**
1. Energy form is **factorized**: unary MLP local term + symmetric bilinear pair term + (optional) learned rule features. Explicitly no second Transformer/whole-graph network (guidance.md "Definitions and choices").
2. Training signal is a **margin rank loss against corrupted graphs** (`energy_rank_loss`), with corruptions produced by the proposal-weighted `corrupt_plausible_graph` sampler — the exact analogue of the PLAN's "paired equal-budget refinement trajectories".
3. Final graph ranking is **host FP64 with ordered terms**, deterministic tie rule (lowest candidate index); this precision policy is fingerprinted.
4. Per-step gradient logit correction is available but **off by default** (an ablation, not the mechanism).
5. Hard ontology rules are **always on**; energy cannot disable them (19-Oct user decision). For the LLM these rules vanish, but the "hard invariants stay outside the learned energy" separation is worth keeping (observed tokens immutable).

**Transfer to Step 11:** the mask-set energy `P(M|x,c) ∝ εU + (1−ε)exp(−E_φ/T)` should reuse the *same factorized form*: unary per-position terms + pairwise position-interaction terms over eligible size-r sets; rank-training on paired trajectories can reuse `energy_rank_loss` semantics; normalization/sampling must be verified by enumeration on small blocks (PLAN §6), matching the predecessor's own exhaustive-reference testing style (categorical == dense tests).

**Warning from production:** energy feature construction hit OOM from expanding dense `[edges, classes]` tensors and padded pair axes (`energy_guidance.rs` incidents, session report 2026-10-23: A5MemoryDiagnosis; fixed via categorical gathers + CSR reductions + per-row padding to 128). Any Step-11 mask-set energy must keep feature cost linear in r and avoid materializing `O(r²)` dense terms.

**R²D / TCT:** no direct implementation in the predecessor. The closest assets are the multi-round rollout/training frames (§1.1: `diffusion_scheduler.rs`, `training_query_chain.rs`) which execute every call on the same model and expose it to its own choices — the natural carrier for TCT-style self-error exposure. The session-report note "self-conditioned states from step 0" (2026-10-13/14) refers to denoiser corruption training, not TCT.

### 1.4 Step 12 — adaptive block boundaries / scaling

| Component | Path:lines | Implements | Recommendation |
|---|---|---|---|
| Encoder growth | `dre-encoder/src/growth.rs` — `EncoderGrowth` 29; `grow_a4` 71; `ParameterRemap` 12; `NewEncoderParameter` 20 | Mid-training architecture growth preserving parameter IDs/Adam entries for retained blocks, fresh state for inserted blocks (A4→A5: 15→20 blocks) | **Port concept** if NEW wants axis-at-a-time scaling without restarting (PLAN §12) |
| Schedule/block composition | `dre-encoder/src/schedule.rs` — `spec_schedule` 74; `a5_schedule` 80; block constants 7–14 | Explicit stage composition and placement constants | **Reference only** |
| Parameter counting | `dre-encoder/src/parameters.rs` — `parameter_shapes` 46; `StageParameterCount` 84 | Exact parameter inventory for scaling reports | **Reference only** |
| Bucketed padded axes | `plan.rs` 196–207, 397–413 | Kernel-shape reuse under variable counts | **Port** (see 1.1) |
| Adaptive *budget* rule (not boundary learning) | `pointer.rs`/span-pointer tests `exp004_query_formula_and_entity_independent_plan_cpu`; EXP-004: `Q(T)=max(64,ceil(10T/29))` | Length-scaled query/cap rule, 99.994% coverage on real data | **Reference** as a baseline budget rule; no predecessor code learns boundaries |
| Bank/mention caps | `mention_head.rs` `select_top_mentions` 377; rho≈0.43 cap; 32-entity bank | Cap policies for predicted quantity | **Reference only** |

## 2. Post-copy changes in predecessor versions of copied/ported components

Snapshot (2026-10-24) vs current (2026-10-25), SHA-256 diff of all 48 reuse files. Changed files:

| Component | Change (file:line in current predecessor) | Impact on NEW |
|---|---|---|
| `dre-lngram/src/lib.rs` | Counterfactual-table fix: `table.set_require_grad(false)` replaces `detach()` (246–259) + new `counterfactual_tests` incl. `private_counterfactual_tables_preserve_values_routes_and_gradients_cpu` (548–631). Avoids orphan parameter-leaf adjoints | **NEW already carries this fix** (`mixers/src/lngram.rs` 491–557, same comment). Confirm and close. |
| `dre-wavelet/src/lib.rs` + tests | Smoothing refactored into `smoothing::Smoothing` module (19–20, 131); added ordered-band equivalence CPU/CUDA tests and registration-count tests (`tests/wavelet.rs` 343–450) | Math identical to reuse copy; NEW's `undecimated_bands` matches. Test enrichment useful; no code change needed |
| `dre-gdn2/src/layer.rs` | Added `causal_depthwise_reference` test helper + NaN-padding gradient test (292–371); `causal_depthwise` now wires `short_conv::forward` (293–295) | Test-only/production-identical; NEW's oracle convolution checks already cover the same property |
| `dre-core/src/profile.rs` | New host-only time phases: `DRE_TIME_PHASES=1`, `time_readback`/`time_phase` (57–92) | Small, useful port for cost reporting (measures host downloads without tensor work) |
| `dre-attention/Cargo.toml` 18,22; `dre-gdn2/Cargo.toml` 17–18; `dre-wavelet/Cargo.toml` 12–17 | Burn 0.22.0→0.22.1, cubecl 0.11.0→0.11.1; wavelet adds `cpu-oracle` feature | NEW already on 0.22.1/0.11.3. Carry over the seven vendored fixes conceptually: fused-Select singleton-axis lane-stride fix, scalar-fusion empty-domain fix, FP32 matmul precision, deterministic select_assign, LLVM scalar-FP32, WGPU subgroup, CUDA allocator cap (session reports 2026-10-22/25) — check each against NEW's kernels |
| `dre-relations/src/denoiser.rs`, `diffusion_scheduler.rs` | Retained-answer/remask flow replaced by commitment flow (`CommitState`, binary commitment visibility, prepared refinement cache) — Codpula evolution | Owned by the excluded files; see §3 |
| `dre-relations/src/remask.rs` | **Deleted** in predecessor (superseded by `commitment.rs`) | NEW's reuse copy remains the Step-3 source; expect no upstream maintenance |
| `dre-relations/src/joint_circuit.rs` | Changed (excluded from this audit) | — |

## 3. Dependencies of the excluded files (joint_circuit.rs, commitment.rs)

Codpula's circuit/commitment implementation relies on (do not re-port independently):
- `pointer.rs`: `PointerState::circuit_evidence` 716, `host_circuit_evidence` 702, `fields_committed` 449, `SourcePointerContext` 51–352;
- `denoiser.rs`: `step_pointer_features` 515, `prepare_refinement` 505, `queries_pointer_committed` 398;
- `diffusion_scheduler.rs`: rollout consumes `CommitState` + `CircuitCall` (125–301);
- `training_query_chain.rs`: `PreparedJointGoldEvents`/`JointGoldEvents` and joint-NLL reduction;
- `learned_output.rs` + `boundary_pair.rs`: terminal reads and candidate options consume committed labels/circuit readouts (`BoundaryCandidates`, 102–160). The source-boundary top-k selection inside `boundary_pair.rs` is a *boundary selector*, not a vocabulary top-k kernel (PLAN §3 warning reaffirmed).

## 4. Highest-value reuse summary

1. `energy_guidance.rs` → Step 11 core (energy form, rank-loss training, corruption sampler, FP64 final ranking) — port with position-mask adaptations; preserve the factorized unary+bilinear form and margin rank-loss decision.
2. `diffusion_scheduler.rs` (current) → Steps 9/10 rollout frame — port with changes as Codpula support.
3. `training_query_chain.rs` + `training_query_batch_loss.rs` → multi-call joint training + per-document RNG discipline (Step 9 training; Step 11 trajectory supervision base).
4. `denoiser.rs` prepare/step-per-pass caching → the frozen-prefix "one snapshot, many passes" pattern both CoCommit branches need.
5. Lngram counterfactual fix already present in NEW; wavelet/attention/GDN crate diffs are test/version-only.
6. `dre-core/profile.rs` time-phase util — small port for costing.
7. `growth.rs`/`parameters.rs` + bucketized padded axes → Step 12 scaling and variable-block cost controls; `Q(T)` budget rule as non-learned reference.
8. CoCommit has no predecessor implementation — reference the caching patterns and design the marker tail fresh; the deleted A3 decider stays history.
9. Preserve user decisions: factorized energy, rank-loss corruption training, FP64 ordered final ranking, hard invariants outside energy, and the exact-uniform/absorptive-commit RNG discipline.
10. Watch-outs: drop relation-specific machinery (matching, mention head, ontology, banks); re-check the seven vendored fixes against NEW's kernels on 0.11.3.

## 5. Vendored patch exposure

The predecessor carries seven local patches against published Burn/CubeCL/CubeK sources (`[patch.crates-io]` in its workspace Cargo.toml lines 43–50; each vendored crate has a `PATCHES.md`). This section audits each against NEW's actual dependency set (registry, no patches): Burn 0.22.1, cubecl 0.11.3 (incl. cubecl-wgpu, cubecl-cuda, cubecl-llvm), cubek-matmul 0.3.1 (NEW `Cargo.lock`). "Upstream" below = those published 0.22.1/0.11.3/0.3.1 sources in `~/.cargo/registry/src/`; vendor diffs were verified line-by-line. NEW runs **Vulkan (wgpu) with fusion off and autotune off by default** (opt-in, GPU-only).

| Patch (vendor file) | Exact bug and trigger | Upstream-fixed in NEW's version? | NEW exposed? | Recommended action |
|---|---|---|---|---|
| Fused-Select singleton-axis lane stride (`burn-cubecl-fusion/src/engine/codegen/kernel.rs` 559–752) | Fused codegen uses `global_stride` for vectorized index/input axes; when a fused sibling vectorizes a singleton axis (width 4/8) lanes advance by the stride and read neighbouring rows/past the tensor (no bounds checks). Trigger: fused Gather/Select/Cat along singleton axes — reproduced with 8 fused singleton Selects per update in the predecessor's `boundary_pair.rs`/`learned_output.rs` (session report 2026-10-22; 141,354/188,472 wrong values logged). Wrong values → NaN-gradient stops (6270/6327) | **No** — upstream kernel.rs still computes `stride_indices_vector = global_stride(...)` unconditionally (559–560, 588, 688–695) | **No** — fusion is compile-time optional in NEW (`#[backend_extension(... Fusion: cfg(feature="fusion"))]` in `kernels/src/*_ops.rs`) and disabled by config (`experiment/src/config.rs` 222–224, default `fusion:false` 318) | Keep fusion off, or vendor the patch + its `dre-gpu-check/tests/fused_select.rs` suite before ever enabling fusion |
| Scalar-fusion empty domain (`burn-cubecl-fusion/src/optim/elemwise/optimization.rs` 76–97, 186–196; `optim/reduce_broadcasted/optimization.rs` 162–176) | A fused kernel with an empty iteration domain but nonempty scalar output launches zero work and leaves the output buffer unwritten; nested elementwise fallbacks inside ReduceBroadcastedOptimization bypass the top-level guard. Trigger: `Zeros([0])` fused with `Zeros([1])` scalar init (session report 2026-10-25; exact diagnostic log) | **No** — upstream has neither `has_empty_reference_with_nonempty_output` nor the fallback replay | **No** — same as above (fusion off) | Same as above |
| FP32 matmul precision — silent TF32 upgrade (`cubek-matmul/src/multi_level/definition/blueprint.rs` 90–98) | Upstream auto-upgrades global f32 inputs to TF32 register/stage dtypes (`lhs_stage/rhs_stage/lhs_register/rhs_register = tf32`) whenever the client `supports_type(tf32)` — silently dropping the mantissa on intended IEEE FP32 matmuls. Vendor removes the branch: "DRE's fp32 mode is IEEE FP32, not the reduced-mantissa TF32 mode"; incompatible accelerator candidates are declined | **No** — upstream 0.3.1 still contains the TF32 branch; the vendor delta is local-only | **No on Vulkan** — cubecl-wgpu reports no tf32 support, so the branch cannot fire today. **Yes on CUDA** (cubecl-cuda maps TF32; GB10 tensor cores), where it would silently degrade every f32 GEMM, including small head-dW/solver GEMMs | Do not rely on it for the current `gemm_None` issue (see below); carry this patch when NEW moves to CUDA |
| Deterministic select_assign (`burn-cubecl/src/kernel/index/select_assign.rs`) | Upstream kernel is correct but serial per non-selected coordinate (one thread loops over all updates); predecessor's A5 profile showed `select_assign_kernel_f_f32_i_i32` = 59.7% of GPU kernel time. Vendor adds deterministic parallel narrow/wide segmented reductions, empty-no-launch, and copy-on-write handling (FP32 association order changes on narrow paths only) | **No** — upstream serial kernel unchanged (select_assign.rs 1–60) | **Partial** — NEW only uses `IndexingUpdateOp::Add` (scatter/select_assign) in dense reference paths (`experiment/src/soft.rs` 128, 343, 488), not the GPU hot path; correctness unaffected, performance cost only if heavily scattered additive updates appear | No action unless profiling shows select_assign hotspots; then port the vendor kernels |
| LLVM scalar FP32 (`cubecl-llvm/src/nvptx/codegen.rs` 39–59, 277–295) | CUDA 13.0's `sm_121a` assembler drops a live first-row result when extracting the low half of a packed FP32 register after a second vectorized reduction → wrong even-row sums (trigger: `[1,30,942]` relation softmax SUM). Vendor sets LLVM option `nvptx-no-f32x2=true`, forcing scalar FP32 registers | **No** — upstream codegen has no `NO_F32X2` option | **No** — CUDA/LLVM codegen only; NEW runs Vulkan/SPIR-V | N/A until CUDA; then vendor the option + `packed_sum_carry_uses_scalar_fp32_registers` regression |
| WGPU subgroup width (`cubecl-wgpu/src/runtime.rs`) | Adapter advertises a variable subgroup range (Intel Arc ANV: 8–32). CubeCL reports `plane_size_max = subgroup_max_size` and registers `Plane::Ops`; wgpu cannot enforce `requiredSubgroupSize`, so plane-cooperative kernels (Cubek GEMM Dot, cubecl-reduce) may assume a plane width different from the actual dispatch subgroup → wrong results/hangs. Vendor sets plane min/max = 1 and omits `Plane::Ops` for variable-width adapters (single-lane fallback; ~17% slower matmul) | **No** — upstream 0.11.3 runtime.rs still mirrors subgroup range into `plane_size_min/max` (664–669) and unconditionally inserts `Plane::Ops` (730) | **Yes, conditionally** — NEW runs the same Intel Arc/Mesa device (subgroup 8–32 documented in `docs/optimization/gemm-efficiency.md` 30–33) with registry `cubecl-wgpu` 0.11.3. NEW's own kernels use no Plane ops (grep clean), but Cubek GEMM candidates under autotune (opt-in, GPU-only; used for MLP/merged-input GEMMs in guarded configs) can enter plane-cooperative paths | Vendor the runtime patch before relying on autotuned Cubek strategies; change the cache/tune namespace when applying so stale decisions are not reused (predecessor pattern: `-dre-plane-width-1` environment namespace) |
| CUDA allocator cap (`cubecl-cuda/src/compute/storage/{gpu.rs,budget.rs}`, `context.rs`, `server.rs`, new `ptx_failure.rs`) | Feature, not a bug fix: 70 GB backing cap / 30 GB host reserve (`AllocationBudget::from_env`), process-byte accounting, allocation diagnostics, PTX-failure capture (CUBECL_PTX_FAILURE_DIR). Prevents driver OOM from killing the host during allocation | **No** — upstream 0.11.3 has a plain pool with no budget/accounting | **No** — CUDA only; NEW has no CUDA run yet | Apply when Spark/GB10 runs start (also needed for the LLVM/FP32 CUDA items above) |

**Related vendored fixes beyond the seven** (same provenance; worth knowing):

- `burn-autodiff/src/ops/tensor.rs` (3150–3153): upstream log backward computes `grad * recip(input)`, which overflows to Inf for tiny inputs even when `grad / input` is finite; vendor divides directly (comment: "reciprocal even when grad / input is finite (for example entropy)"). Upstream 0.22.1 unchanged. NEW is **not exposed** in hot paths — its losses use stable `log_softmax` (`experiment/src/model.rs` 436, `sbd.rs` 81/154), and the only `.ln()` uses are host f64 (`update_loss.rs` 87) or non-tracked kernel scalars (`wavelet/mod.rs` 162).
- `burn-cubecl/src/kernel/matmul/base.rs` (63–72) + `src/ops/module.rs` (338–344): upstream zero-K matmul returns unwritten output; vendor fills a nonempty zero-reduction output and forces `FlashUnit` for FP32 attention (F16 fragments would otherwise lower precision). Upstream 0.22.1 has neither. NEW risk low (matmul K dims are model widths; never 0), but the empty-matmul guard is cheap to adopt.

### Can the FP32-matmul patch explain NEW's `gemm_None` wrong results?

**No.** NEW's `gemm_None` failures (`docs/optimization/gemm-efficiency.md` 62–96) occur on **Vulkan**, where cubecl-wgpu does not support tf32, so the upstream TF32-upgrade branch cannot fire and the vendor patch changes nothing there. The observed errors are also not roundoff ("large numerical errors") and involve the Cubek `matmul_entry`/`Gemm` none-stage path with transposed-LHS views and `CopyKernelPack` input kernels — a distinct upstream Cubek defect on wgpu, unaffected by dtype selection. The TF32 patch only becomes relevant when NEW runs CUDA (silent mantissa loss), and the WGPU subgroup patch is the vendor fix that *can* affect autotuned Cubek GEMM candidates on NEW's current device.

### NEW-side mitigations already in place (no action)

- Autotune is off by default and restricted to explicit GPU backends (`config.rs` 227, 319, 464–465); guarded configs preflight every active geometry before training and reject mismatches (`docs/optimization/gemm-efficiency.md` 100–126).
- The `gemm_None` defect is already avoided by pinning head/merged-output contractions to the public non-autotuned heuristic (`merged_heuristic_output`/`heuristic` backends, same doc 102–110).

### 5.1 Subgroup patch test (2026-10-25)

Controlled experiment in a dedicated jj workspace (`../delta-block-lm-ws/subgroup`, rev 6426c332; test-only, no model edits): registry `cubecl-wgpu` 0.11.3 vs the identical crate with exactly the predecessor's runtime.rs subgroup hunk (plane min/max = 1 and `Plane::Ops` omitted for variable-width adapters). Both binaries built from the same commit with `--features vulkan,autotune`; each ran with a fresh autotune cache (`.cache/cubecl/subgroup-<variant>`) and `bare autotune` configs (`head_backend=native`, `projection_backend=merged`). Evidence: `artifacts/subgroup/{patched,upstream}/{gemm-micro-*,kron-model-check-*}/report.json`, `autotune-decisions.jsonl`, `artifacts/subgroup/summary.json`.

| Test | Upstream (unpatched) | Patched (subgroup hunk) |
|---|---|---|
| `gemm-micro` (47 recommended cases, 126 legacy-solver cases) | `passed=false`; recommended fails **vocabulary_head weight_vjp B8** (M256 N257 K4096, head-dW family, original+pack); **legacy solver fails** on many small batched shapes (e.g. 16×64×32, 16×32×64, 16×16×64, 32×16×64); elapsed 89.6 s | `passed=false`; recommended fails **mlp_down forward B8/B16**, **merged_input forward B8/B16**, **mlp_up input_vjp pack B16**, **square 1024** (all cases where autotune selected `gemm_None` as fastest); **legacy solver passes 126/126**; elapsed 178.1 s |
| `kron-model-check --model-scope configured` | `passed=false` — T128 both HOLA modes wrong (logits[0] 0.25838792 / 0.24741092 vs CPU 0.06443295 / 0.055922177); T512 passes | `passed=true` — all 4 cases, incl. T128 both modes |
| Selected strategies (autotune-decisions) | Big shapes → `matmul_double_unit_*`/`simple_unit_*`; small batched → `gemm_None` | (512,256,512)/(512,512,256)/(512,1024,256)/(1024³)/(2048³) → `gemm_None` fastest; (256,*,*) → `simple_unit_max` |

**Verdict:** the subgroup patch does **not** explain or fix the bare-autotune wrong results. The `gemm_None` kernel itself is numerically broken whenever it wins (the patched build still fails exactly those probes, including square 1024 and M512±256×512), so it must stay excluded/pinned regardless of plane handling. What the patch does change is *which* candidate wins autotune: it fixes the previously documented T128 model-level failure (kron-model-check) and all legacy-solver failures, at the cost of shifting `gemm_None` wins to larger shapes. Recommended action: keep the subgroup hunk (it removes a real plane-width hazard and fixes the legacy solver/T128 class) *and* keep NEW's existing heuristic pinning / guard that prevents `gemm_None` selection (docs/optimization/gemm-efficiency.md 100–126). The FP32/TF32 patch remains irrelevant here (no tf32 on Vulkan).

Decision (2026-10-25): subgroup patch not adopted. It only changes autotuned paths, which stay guarded/pinned; the default heuristic path passes every gate unpatched, and the patch removes plane operations (predecessor measured ~17% slower matmul). gemm_None remains excluded. Revisit only if autotune is extended to new shapes.
