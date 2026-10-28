# Step 10: CoCommit for recurrent block denoising

## Status and use

**Implementation decisions adopted 2026-10-25; numerical and cost evidence is recorded below after execution.** Q1–Q16 follow the highest-ROI recommendations; Q17 is a saved-input zero-marker correctness gate only. No quality study, trained comparison arm or sweep is authorized.

Source: Lin Yao, [Don't Commit Alone: Joint Token Commitment in Diffusion Language Models, arXiv:2607.04469v2](https://arxiv.org/html/2607.04469v2), especially §§3–5, Figures 1–2, Eqs. (1)–(4), Tables 1–3 and §6 Limitations. The complete HTML and [13-page PDF](https://arxiv.org/pdf/2607.04469v2) were read through the references. **This version contains no appendices.** There is no extra appendix recipe to fill its unspecified training reductions or short-mask cases.

Repository requirements: [PLAN §5.2, §5.3 and Step 10](../PLAN.md), [SBD](sbd.md), [Codpula](codpula.md), and [predecessor inventory §1.2](predecessor-inventory.md#12-step-10--cocommit-coordination). Source line references below describe the inspected working tree and will move when code changes.

The intended reader is implementing the next model comparison, not reproducing LLaDA2.1's benchmark scores. CoCommit first chooses **positions**, then predicts their **values** with an extra marked tail pass. For example, if the unmarked output chooses positions `{1,4}`, add a marker to those two saved hidden states, recompute the tail for the whole active block, and write positions 1 and 4 from the new output. Neither the first proposals nor the clean answers enter that second pass.

## 1. What the paper does

Use `D` for backbone depth, `n` for tail depth, `K` for coordination rounds, `L` for active block length, and `S` for the positions selected for commitment. The paper uses `L` for backbone depth in its `n/L` cost statement; our notation avoids confusing that depth with SBD block length.

### 1.1 Save the input to the tail, not its output

Paper §4.1 computes

\[
h=\operatorname{Prefix}(z),\qquad
\ell_A=\operatorname{Head}(\operatorname{LastN}(h)),
\]

where Prefix is the first `D-n` layers and LastN is the final `n` layers with their existing weights. Cache `h` immediately **before** LastN, before the final output normalization/head. It is not the final logits, final normalized features, attention K/V, or the output of the first tail pass. Prefix computation is shared between branches; tail weights and the output head are tied.

In the evaluated model, `D=20`, `n=3`, and `K=1` (§5.1). Table 3 compares `n=3,6,9`; deeper tails do not consistently improve quality. Our plan instead starts at `D=4,n=1,K=1`.

### 1.2 The unmarked output chooses the bundle

Paper §3.2 chooses all currently masked positions whose maximum vocabulary probability exceeds `tau_m2t=0.7`. If none qualifies, choose the most confident masked position. This creates `S` but writes nothing. Its confidence rule is not SBD's entropy rule or Codpula's dependence rule.

The marker announces position membership only. It does not contain the selected token IDs or sampled token values. All selected inputs remain masked in the paper. The evaluated bundles exclude already-visible positions; marking visible positions is an unevaluated extension (§1, §6).

### 1.3 Mark and run the same tail again

Paper Eq. (4), for a training bundle `A` or inference bundle `S`, is

\[
h^{(0)}=h,\qquad
h^{(r+1)}=\operatorname{LastN}(h^{(r)}+\mathbf1_S\odot m_r),
\qquad \ell_B=\operatorname{Head}(h^{(K)}).
\]

Each `m_r` is a learned width-sized vector, zero-initialized (§5.1). For **one** round, the assignment branch is `Head(LastN(h + marker))`, not `Head(LastN(LastN(h) + marker))`. Only when experimenting with `K>1` does Eq. (4) feed the previous marked tail's hidden output into another marked tail. Output normalization/head is applied for readout, not inserted between these hidden-state iterations.

The extra pass processes every position of the current block, not just `S`. Attention from marked positions to the rest of the block remains available. Fixed-prefix attention K/V are reused; they are not modified by the marker (§4.2).

### 1.4 Both branches are supervised

Paper §3.1 Eq. (1) partitions valid positions into masked `M`, wrong-visible `W`, and clean-visible `G` and uses ordinary clean-token cross-entropy:

\[
J_A=\lambda_m\sum_{i\in M}\operatorname{CE}(\ell_{A,i},x_i)
 +\lambda_e\sum_{i\in W}\operatorname{CE}(\ell_{A,i},x_i)
 +\lambda_c\sum_{i\in G}\operatorname{CE}(\ell_{A,i},x_i).
\]

For the marked branch, sample a size uniformly from `{s_min,...,s_max}`, then a subset `A` uniformly among masked subsets of that size. Replace only the masked sum by `sum(i in A) CE(ell_B[i],x[i])`; retain edit and clean terms on **all** `W` and `G` (§4.1). Unselected masked positions are **not** extra marked-branch targets. The clean sequence supplies labels, not second-pass inputs. There is no joint-MAP label, consistency loss, imitation of branch A's tokens, or gradient through bundle selection.

The experiment uses `(lambda_m,lambda_e,lambda_c)=(1,0.3,0.2)` and random sizes 1–16 (§5.1). It constructs wrong-visible examples using a no-gradient self-generated draft, freezes the pretrained backbone, and trains rank-16 LoRA adapters on attention projections in every layer plus the marker. These are the experimental fine-tuning recipe, not a requirement to add LoRA to our small model. The two branches share the prefix and one backward call; the paper concatenates their tails on a doubled batch. It does not state a unique branch averaging/token-normalization rule, or what to do when fewer than `s_max` masks exist.

### 1.5 Inference and limits

For each denoising step (§4.2): compute the shared prefix and unmarked tail; select `S`; if `S` is nonempty run `K` marked tail rounds; write from the resulting logits. If `S` is empty, use unmarked logits. Both mask filling (Eq. (2)) and visible-token editing (Eq. (3)) use the **marked** output when that branch ran. Visible edits still read the same pre-write block, not a block already containing this step's new fills.

The experiment uses greedy argmax, `K=1`, block length 32, 32 denoising steps and at most 16 post-steps; its default removes the visible-edit confidence threshold (§5.1). These decoder settings must not silently replace our SBD schedule. `K=0` means ordinary factorized decoding: return branch A's usual tail output, not a head applied directly to pre-tail states.

For one marked round the paper estimates an added `n/D` of a block forward, excluding additional output-head details and changed denoising counts. Its default is approximately 15% per coordinated step. Table 1's NFE/token explicitly omits this partial pass, so NFE alone is not a cost comparison. Training adds a differentiated tail/head branch, and its self-generated draft is another forward. There is no measured training-overhead percentage for our architecture.

The intended result is approximate joint-mode coordination under greedy writes, not an explicit joint distribution, exact token-sequence MAP, or joint sampling. Independent temperature draws can still split modes. One coordinated answer can be consistently wrong. There is no monotonic improvement result for `K>1` (§6).

## 2. Exact proposed mapping to our model

This section uses the recommendations in §5. Changing one of those choices requires changing its equations and tests together.

### 2.1 The tail is complete block 3

At the current four-block configuration:

| Work | Shared noisy prefix | Recomputed tail |
| --- | --- | --- |
| Token/soft embedding and oHC initialization | Once | Never repeat initialization |
| Blocks 0, 1, 2, each with wavelet, Lngram, mixer and MLP plus oHC wrappers | Once | No |
| Block 3 mixer, including short convolutions, temporary KronLSTM, HOLA and optional MoBA read, then MLP; both oHC wrappers | No | Once unmarked and once marked |
| oHC readout, final model RMSNorm, vocabulary head | No | Once per output branch |
| Codpula gate pooling/head and circuit preparation, when selected | No | Fresh for each branch output |

Thus save streams **after block 2's MLP write and before block 3's first oHC read**. Keep all four streams `[C,Lmax,4,256]`, not a readout/re-expanded approximation. Without oHC, save ordinary residual hidden states `[C,Lmax,256]` at the same block boundary. Here `C` counts active blocks, not training sequences. `n=2` would start at block 2 and must include its wavelet and Lngram operations; `n=3` would start at block 1. A tail is a suffix of complete blocks, not only mixers or only MLPs.

Keep an internal pre-tail value with either ordinary hidden states or oHC streams, plus validity/actual lengths, original starts and read-only access to the clean layer caches. Keep an internal tail runner taking that value, the same caches, and a position marker mask. Its output is the existing `DenoisingOutput { logits, features }`. Do not add a second vocabulary head or extra normalization. No output features need to be retained from branch A after its selection and loss uses finish.

### 2.2 Marker across oHC streams

Use one zero-initialized learned marker over the **complete saved hidden representation**: `m[4,256]` with independently learned stream entries when oHC is enabled, or `m[256]` without oHC. At selected active unresolved positions add:

\[
X'_{c,j,s,d}=X_{c,j,s,d}+I_{c,j}m_{s,d},
\qquad s\in\{0,1,2,3\}.
\]

`I=1` only for the training bundle or inference announcement; `I=0` on explicit observations and padding. Do not divide by four, use stream-initialization scales/offsets, add to the mask embedding, or inject after the oHC read. The next oHC read therefore sees marked streams both in its stream normalization and in its input-dependent mixing/read/write weights. This is an adaptation, not an oHC rule supplied by the CoCommit paper. Without oHC use `H'=H+I*m`.

The oHC marker's VJP is `dm[s,d]=sum(c,j) I[c,j]*dX'[c,j,s,d]`; stream gradients are not summed into one shared marker. This costs 1,024 parameters rather than 256, but allows learned stream-specific directions instead of assuming stream coordinates remain interchangeable after three blocks. Shared prefix gradients receive the sum of both branch gradients. The saved tensor stays differentiable in training; detaching it would change the objective. Adding markers must not mutate the shared tensor or clean caches in place. Padding is selected away before arithmetic so dirty padded values cannot contaminate a reduction.

Require `sbd.in_block_reads="hola_read"` for enabled CoCommit. The last-layer active HOLA K/V gives the required bidirectional path between selected positions. With `none`, this model has independent one-step noisy recurrence and left-only short convolutions: it does not reproduce the paper's within-bundle bidirectional communication. Keep that old setting valid when CoCommit is absent, rather than silently inserting attention.

### 2.3 Same clean snapshots for every branch

A frozen snapshot here means **values cannot be changed**, not that training gradients are detached. Preserve the existing differentiable clean-prefix snapshots and histories at every layer:

- clean recurrent state before the block;
- the three clean raw convolution taps;
- selected HOLA slots on the original chunk grid and all clean unfinished-chunk keys before the start;
- cached wavelet history and Lngram suffix/smoothing history;
- optional clean retrieval keys, values, summaries, document positions and block start.

Each tail call uses these same layer-specific inputs. For every noisy position independently, the existing mixer expands the clean state and performs **one temporary recurrence step**; it does not scan selected tokens sequentially and does not continue from branch A's temporary final states. Drop all temporary recurrent results after use. Block 3's marked Q/K/V and gates must be regenerated, including marked active K/V; its clean K/V and convolution taps are reused unchanged. HOLA uses one softmax over its existing clean segments, active noisy segment and sink, with the same gate and scale.

For MoBA, retain clean candidates/summaries, but recompute any query-dependent retrieval choice and attention scores from the marked queries. Do not reuse branch A's selected retrieved blocks merely because the clean candidates are unchanged. Active K/V never become clean retrieval candidates or HOLA eviction candidates.

No noisy branch may append tokens, update clean recurrence, select persistent HOLA entries, extend convolution/wavelet/Lngram history, or replace `CleanPrefix` state. Only completion of the whole block triggers the existing causal replay once. Both branches retain original input-position identities `start+j`; no round-position embedding is added.

Historical generation reconstructs clean tensors every pass. Enabled CoCommit instead prepares one immutable clean cache **per active block**, reuses it across denoising passes, and recomputes the changing noisy prefix once per pass before its two tails. Run the ordinary clean prefix with `CleanInputs`, prepare its per-layer cache at start `prefix.len()`, and discard/replace it only after completed-block causal replay. This removes repeated identical clean-prefix work without changing visibility or persistent state. `training_cache(..., starts=[prefix.len()])` and the per-block cache runner already express a prefix ending exactly at the start; the batched helper's `start < time` assertion is untouched. Reference reconstruction remains available for equivalence checks. Do not call full `block_forward` twice and call that tail reuse.

### 2.4 Training with soft feedback

Keep clean causal forward/loss and clean snapshot preparation once. Keep the original block-length/mask draws and all-observed skipping. For each actual multi-token training block:

1. In `soft_feedback`, run the existing detached **unmarked teacher** and its existing top-k update once. Hard-mask training skips it. Do not coordinate this teacher or add another feedback update between the student branches.
2. Compute the student's noisy prefix once from the resulting soft embeddings. Save the pre-tail tensor with its graph.
3. Form the random training bundle from unresolved masked positions, regardless of soft visibility. A candidate mixture is not observed evidence.
4. Run both student tails from the saved tensor and the same clean snapshots. The selected/default `execution: "separate"` makes two C-row batched calls sharing clean banks. Explicit `"doubled"` is supported only when no marked-tail mixer uses prefix retrieval: concatenate `[saved,saved+I*m]` in branch-major order, duplicate only compact tail caches and run one 2C-row tail/readout, then split A/B outputs. Never rerun the noisy prefix on 2C rows. Explicit doubled plus retrieval in the marked tail is rejected, not silently changed to separate. Both supported layouts accumulate one total loss before backward; neither is a per-block production loop.
5. Use the objective below. Update AdamW and, when enabled, the shared circuit only after both losses/statistics are collected from the same model values.

The teacher's sparse candidates retain original top-k probabilities and retained mass. Its unresolved embedding stays

\[
E_i=r_i\sum_{v\in\mathrm{topk}}p_i(v)\operatorname{embed}(v)
 +(1-r_i\sum_{v\in\mathrm{topk}}p_i(v))\operatorname{mask}.
\]

Marker addition happens much later, at the tail boundary; it neither deletes these beliefs nor turns a proposed token into evidence. Candidate IDs/probabilities are detached as now; the selected embedding table values remain differentiable. The marked and unmarked student share identical soft inputs.

For masks-only corruption, let `M` be unresolved positions and `G` the valid explicit clean observations. There are no wrong-visible tokens (`W` is empty). The proposed factorized objective is

\[
J=J_{\rm clean\ causal}
 +\frac1{BT}\sum_{b:L_b>1}\left[
 \sum_{i\in M_b}w_i\operatorname{CE}(\ell_{A,i},x_i)
 +\sum_{i\in M_b}w_i\operatorname{CE}(\ell_{B,i},x_i)
 +0.2\sum_{i\in G_b}w_i\{\operatorname{CE}(\ell_{A,i},x_i)
 +\operatorname{CE}(\ell_{B,i},x_i)\}\right].
\]

This deliberately adds visible-preservation supervision compared with current SBD, retaining the paper's cheap clean term but omitting its self-generated edit training. Apply the existing target weights; do not normalize each branch by its number of masks/markers or multiply by inverse inclusion probability. Add branch losses, do not average them. All-observed draws remain skipped, so they gain no new auxiliary term. These are explicit adaptations (Q5–Q7), not a full reproduction of the paper's editing recipe.

For `L=1`, retain the exact causal next-token bypass and its existing auxiliary masked loss, with no marker or circuit branch, including when the singleton is observed. A training batch containing no marked multi-token rows must still yield an explicit zero marker derivative for the existing all-parameter gradient checker.

### 2.5 Combined CoCommit + Codpula training

Keep the current 32-state, full-257-symbol HMM, shared token head, mean-pooled 64 coupling gates, observed-potential gradient masking, and dual-flow optimizer. The two student branches define two **different** prepared circuits `q_A` and `q_B` from their own final features and logits, with the same pre-update intrinsic probability rows. The marked branch must regenerate **both** token potentials and gate logits before preparing leaves, transitions and local normalizers.

The proposed combined objective replaces the factorized auxiliary terms above by

\[
J=J_{\rm clean\ causal}
 +\frac1{BT}\sum_b\left[-\log q_A(x_{M_b}\mid e_b)
                         -\log q_B(x_{M_b}\mid e_b,A_b)\right].
\]

Here `e_b` fixes only the already-observed clean tokens, and `A_b` is the marker input, not circuit evidence. B is supervised on every free position it can write, including unmarked positions. The two marked queries are

`minus = log Z_B(e)` and `plus = log Z_B(e union gold[M])`.

The conditional loss is `(minus-plus)/(B*T)`. The event-loss API targets all valid free positions in rows with a nonempty announcement; empty-announcement rows contribute only A. Padding and permanent observations receive no extra circuit target. This same objective is used by TCT.

The same generalized fixed masks must drive neural derivatives and circuit statistics. In probability notation, for posterior states `gamma+` and `gamma-`, normalized leaf `g`, and posterior token distribution `f+`/`f-` at a state (one-hot at a fixed token, otherwise `g`), accumulate emission positive flow `gamma+*f+ + gamma-*g` and negative flow `gamma-*f- + gamma+*g`, with the existing valid/active masks and tied-row sums. Transition/root flows retain the current two-posterior normalized-row formulas. This replaces the current numerator assumption that every leaf is fixed. Verify the difference against the full event likelihood derivative, including leaf denominators.

For shared circuit learning, sum positive/negative flows and reaches from the A and B branch contexts with `CircuitFlows::add`, and add their active-context counts. Update once using the existing context average and momentum, not once per branch and not a count of unique blocks with doubled statistics. Neural loss keeps the common `B*T` denominator; circuit statistics retain their existing separate active-context reduction. No circuit flows arise from the teacher. No new visible-token CE is applied to raw Codpula potentials: these are not calibrated token marginals, and conditioning on an observed token already makes its conditional target probability one. This difference from factorized CoCommit is explicit (Q8).

### 2.6 Factorized generation

For each current block with `L>1`:

1. Freeze the pre-write soft/observed state. Reuse this block's immutable clean cache and compute/save the changing noisy prefix once for the pass.
2. Run the unmarked tail. Choose `S` using the existing fixed `commit_count` or SBD Eq. (8) entropy policy, considering only unresolved active positions. Use existing ascending-position ties.
3. If `S` is empty, no assignment call is needed. Otherwise add its marker to the saved pre-tail tensor and rerun the whole active tail once. Do **not** change `S` using the marked confidences in this factorized path.
4. Get new token argmaxes from the marked logits, with ascending-token-ID ties. Write only `S`. The values selected before marking are discarded.
5. Pass **marked** logits to the existing soft-feedback update for deferred positions, then make `S` immutable observations with `advance_generation_state`. Its existing order updates beliefs before installing the commitments. Hard-mask mode has no belief update.
6. Continue with a new noisy prefix on the next denoising pass. Complete and replay only when all positions are observed. Since every nonempty pass commits at least one unresolved position, at most `L` passes are needed; there are no T2T edits or post-steps in this initial adaptation.

The paper's confidence selector is an optional later comparison (Q4), not a silent replacement for the SBD selector. Greedy assignment does not mean setting soft-feedback temperature to zero; retain the positive feedback temperature and deterministic argmax writes.

### 2.7 Combined generation: replace the circuit after marking

Follow PLAN §5.3 literally. Before any write:

1. From unmarked output build `q_A`; compute its evidence marginals, exact joint proposal, and dependence scores when configured. Choose announcement `S_A` using the current commitment rule. In dependence mode, score all unresolved positions as now.
2. Add the marker at `S_A` to the saved pre-tail tensor. Rerun the tail. Regenerate token potentials **and** coupling gates, then build a new `q_B` with new local emission/transition normalizers. Discard every `q_A` message, marginal, proposal and dependence score for purposes of assignment.
3. Recompute `q_B` evidence marginals, exact joint proposals and dependence scores for **all unresolved positions**. Proposals/scores use the same positive neural-potential temperature and full vocabulary as the existing Codpula path. Existing evidence is unchanged; unmarked proposals are not extra observations.
4. In dependence mode, apply the strict threshold to marked candidates only: `S_B={i in S_A : delta_B[i] < tau}`. If empty, commit the position of minimum new score within `S_A`, ties by position. On the existing last pass (`pass == L`), announce and commit every remaining unresolved position. In fixed-count/entropy mode retain the originally selected `S_A`; do not choose new unmarked positions after coordination.
5. Write the **new joint proposal** at positions `S_B`. Use `q_B` marginals, as log probabilities with `SoftState` temperature one, for deferred-position feedback. Replay once the block completes.

The new dependence scores use the full unresolved proposal set as in current Codpula, even though only announced positions may be written. Restricting *which candidates may be written* is separate from changing the score's conditioning set. Unmarked positions can affect scores and gates but cannot acquire an unannounced write in this pass. This recommended choice avoids an unlimited select/remark loop; it is not a composition proved by either paper (Q9).

Draw one proposal seed from the existing per-pass RNG, derive A and B proposals with the **same** seed (or equivalently the same indexed state/emission uniforms). Branch B still resamples its new conditional distributions; identical random numbers do not mean reused tokens. Keep branch-off RNG consumption exactly historical. No random value affects the marker itself at inference. The combined stochastic path is a Codpula extension, **not** evidence that the paper's greedy-mode result extends to sampling (Q10).

### 2.8 Metrics and evaluation

Clean validation is unchanged. The primary `sbd_validation.block_nll` now runs the actual decoder with generated clean prefixes and the same self-generated observations, feedback, announcements and reveal transitions as generation. It scores token marginal NLL from B at the positions B writes (A only for empty announcements or the causal singleton). Refinement retains each output token's last write-time score, using the actual proposal evidence; repeated refinement writes are not counted as extra output tokens. Accuracy compares finalized decoder tokens to targets. This is a decoder-conditioned token-marginal diagnostic, not a joint sequence likelihood. The former one-teacher-pass, gold-prefix A/full-block and B/subset diagnostic remains under the distinct `teacher_prefix_diagnostic` key and is not the primary decoder metric.

Record full noisy-prefix calls, unmarked-tail calls, marked-tail calls, marker positions, circuit preparations, proposals, dependence traversals, clean reconstructions and replay time separately. Tail-normalized NFE is `(prefix-block work + all tail-block work)/D`, with head/circuit time also reported; do not report `passes` alone as complete work.

## 3. Where existing paths change

| Current file:line | Required change if recommendations are adopted |
| --- | --- |
| `crates/experiment/src/config.rs:203-244,428-469,686-706` | Optional `sbd.cocommit` config, validation, defaults and model identity. Absence must preserve old serialization, fingerprints and initialization. Add a CoCommit-only semantic revision rather than changing the historical SBD revision for every config. |
| `crates/experiment/src/model.rs:58-165` | Optional zero-initialized marker parameter, created only for enabled config after ordinary parameters. Train/save it through the existing Burn module/AdamW path. |
| `crates/experiment/src/model.rs:257-328,332-427` | Share complete-block execution and final readout between clean/noisy split runners; preserve the clean path and its captured sublayer inputs. Avoid a second convention for oHC or normalization. |
| `crates/experiment/src/sbd_training.rs:15-20,65-70,90-149,169-226` | Extract noisy-prefix/pre-tail and tail operations for both per-block and batched cached execution; retain raw oHC streams; accept branch masks; share frozen caches. |
| `crates/mixers/src/sbd.rs:129-180,251-309,314-388` | Reuse snapshot preparation, independent temporary recurrence and read kernels. No recurrence mathematics change. Add branch-major concatenation of compact tail-cache tensors for the no-retrieval 2C training call; with tail retrieval, share the full banks across two C-row calls rather than replicating them. |
| `crates/experiment/src/sbd.rs:22-40,65-82` | Reconstructed-prefix split runner and unchanged unmarked teacher feedback. Reuse one prepared prefix for the two tails. |
| `crates/experiment/src/sbd.rs:107-123,124-250` | Preserve mask sampler; add independent bundle draws, two branch losses and common normalization; per-block/batched equivalence; aggregate branch circuit statistics once. |
| `crates/experiment/src/codpula.rs:65-77,228-248,558-619` | Reuse gate preparation per output; generalize the event-loss numerator and corresponding flows for marked-subset supervision; preserve existing all-unresolved call semantics. |
| `crates/experiment/src/sbd.rs:254-295,334-374,377-453` | Branch-specific evaluation, announce/assign decoding, fresh marked circuit/proposals/scores, feedback from assignment output, pass/time counts. Keep persistent replay at lines 323–329 unchanged. |
| `crates/experiment/src/soft.rs:281-447` | Existing observation and candidate update semantics stay unchanged; callers supply marked output. No marker field belongs in `SoftState`. |
| `crates/experiment/src/train.rs:318-328,449-456` | Existing single backward/optimizer call accepts the combined result; no optimizer between branches. Marker joins normal neural optimizer/checkpoint/gradient handling. |
| `crates/experiment/src/main.rs:347-359`; `sbd_batch_reference.rs:18-50`; `codpula_model_checks.rs` | Register/extend model checks using existing comparison/report conventions; keep one experiment test binary. Add independent scalar reference fixtures without copying the optimized graph. |
| `configs/lm512-sbd.json`, `configs/lm512-codpula.json`, `docs/sbd.md`, `docs/codpula.md`, `PLAN.md` | Keep historical baseline configs unchanged; create named CoCommit comparison configs only during implementation; describe enabled semantics and observed results after checks. This specification does not authorize a quality sweep. |

Predecessor reuse is limited to the cached-input/multiple-pass pattern documented in inventory §1.2. There is no existing predecessor marker implementation. Do not revive the deleted A3 remask policy, its binary decisions, or round embeddings as a substitute.

## 4. Proposed configuration and repeatable draws

The implemented profile is `sbd.cocommit` absent by default, otherwise an object with `tail_layers: 1`, `rounds: 1`, `training_bundle_max: 16`, `visible_loss_weight: 0.2` and optional `execution: "separate"` (the selected default). The explicit `"doubled"` fitting-tail alternative is supported only without prefix retrieval in the marked tail; retrieval in earlier saved-prefix layers is compatible. Tail retrieval requires explicit/default separate execution and shares banks; an explicit unsupported doubled request is rejected rather than overridden. Marker shape, masks-only corruption, branch-loss sum and write semantics remain fixed by the named CoCommit revision. Visible weight applies only to factorized output. Invalid/unknown fields, nonfinite/negative visible weight, zero bundle maximum, tail depth outside `1..=layers` and enabled CoCommit without `hola_read`/HOLA are rejected. Only one round is supported; larger K requires its own approved recipe.

Use a separate host `Rng(seed ^ 0x434F434F4D4D4954)` for bundle draws, never the mask sampler's stream. Traverse the already-sampled active training blocks in row/start order, independent of per-block/batched execution. For `m` unresolved positions, draw `s=1+rng.index(min(m,training_bundle_max))`, then use a partial Fisher–Yates shuffle of ascending unresolved indices: for `j=0..s-1`, swap with index `j+rng.index(m-j)`; select the first `s`. Singleton bypasses and skipped all-observed blocks draw nothing. Build the device indicator from those positions once. This specifies a uniform size over the feasible range and a uniform subset conditional on that size, with repeatable CPU/GPU and batching behavior. It does not change existing length/mask draws.

Checkpoint identity includes enabled semantic revision, tail depth, rounds, bundle maximum and objective weight; marker values and AdamW moments are saved. Ordinary disabled models must remain loadable exactly as before. Do not silently turn an old checkpoint into a trained CoCommit model by adding a zero marker on load; warm-start conversion, if requested later, must be explicit and initialize the new optimizer entries.

## 5. Decisions

**Each Q1–Q16 decision was decided 2026-10-25 in the user's absence from the highest-ROI recommendation; reversible.** Q17 was decided the same day by explicit direction: saved-h zero-marker equivalence only, no trained sequential-depth arm. The `RECOMMENDED` labels in Q1–Q16 below record the adopted option and its expected-return reasoning, not an open vote. Runtime layout remains subject to the explicitly authorized paired execution-cost measurement.

| Decision | Adopted behavior |
| --- | --- |
| Q1 | One complete final block as the tail. |
| Q2 | One marked coordination round. |
| Q3 | Independent `[4,d]` marker entries with oHC; `[1,d]` without it. |
| Q4 | Existing fixed-count/SBD entropy selector. |
| Q5 | Masks-only corruption and `.2` visible-preservation CE in factorized branches. |
| Q6 | Uniform feasible bundle size, uniform subset. |
| Q7 | Sum both branch losses over the unchanged clean-target denominator. |
| Q8 | Marked-subset conditional joint event loss; other masks marginalized. |
| Q9 | Fresh B scores/proposals; writes restricted to the announcement. |
| Q10 | Exact joint proposals with shared A/B random numbers. |
| Q11 | Marked-output feedback. |
| Q12 | Train existing neural parameters plus the marker. |
| Q13 | Immutable observations; no visible revision. |
| Q14 | Two shared-cache C-row tail calls by final paired measurement; optional doubled execution retained and verified, not selected. The original recommendation was reversible and subject to this measurement. |
| Q15 | Exact causal singleton bypass. |
| Q16 | Combine both branch contexts/counts and update the circuit once. |
| Q17 | Saved-h zero-marker equivalence check only; no trained sequential-depth comparison. |

Labels distinguish **paper results**, **our measurements**, **arithmetic** from the specified shapes, and **reasoned estimates** of unmeasured gains or timings. Evidence is useful, not a prerequisite for choosing a larger design: where it is absent, rank options by the most plausible mechanism-level benefit and total cost instead of withholding a recommendation. No guessed quality delta is presented as a measurement. Each recommendation below makes that expected-return judgment, including explicitly speculative ones; the exact behavior above follows these recommendations.

Shared scale for the comparisons: at B8/T512/L8 there are at most 512 active block rows and 4,096 active positions; the mask sampler gives about 455 active rows on average. One saved four-stream pre-tail tensor is **16 MiB** at the maximum shape (**arithmetic**); one extra tail/head costs an assumed **15–30% of an active noisy forward**, not of a complete training step. Our complete tensor-Codpula training diagnostic is **8.000 s**, and B1/T512 tensor median is **1.697 s** (**measured**, §6). With fixed L8, training active-row work/storage grows roughly linearly with context length; clean-prefix reconstruction and retrieval-bank costs can grow faster than active-tail costs. Larger contexts strengthen the value of shared prefix work, not a blanket preference for the smallest tail.

| ID and question | Options: expected gain versus total cost at our size | RECOMMENDED option — one-line return judgment |
| --- | --- | --- |
| **Q1 Tail depth** | **One complete block:** one bidirectional HOLA exchange and MLP, 16 MiB saved streams. **Two blocks:** another communication/transformation step plus block 2's wavelet/Lngram; probably useful for harder interactions but largely redundant on L8 (**reasoned estimate**), at more than twice the one-block tail work because of those injections. **Three blocks:** still more transformation, diminishing expected return. Paper default is 3/20; Table 3 averages are **78.72/77.50/77.65** for 3/6/9 (**paper results**, not our quality). At larger contexts with fixed L8, all active-tail alternatives scale with block rows; longer prefixes do not themselves require a deeper coordination tail. | **RECOMMENDED: one complete block** — I expect one all-to-all HOLA exchange to capture most L8 membership benefit, so a second injection-heavy block buys less quality than its greater-than-one-tail marginal compute (**estimate**, supported directionally by the paper's depth ablation). |
| **Q2 Round count** | **K1:** one marked tail. **K2 with matching training:** two marked tails, 2,048 marker parameters with four-stream markers versus 1,024 for K1 (**arithmetic**), more activations and serial inference latency; plausibly resolves a few harder ambiguities, but repeatedly transforming the same deterministic states has diminishing returns (**reasoned estimate**). **K0:** no coordination, useful as baseline. Paper evaluates K1 and gives no monotonicity guarantee for deeper iteration. Larger-context active training work scales with K; each inference round remains serial. | **RECOMMENDED: K1** — expected second-round correction is a small residual after one bidirectional exchange, unlikely to repay roughly another full marked-tail cost (**estimate**), while K0 gives up the requested coordination benefit. |
| **Q3 Marker in oHC** | **Shared broadcast vector:** 256 parameters/1 KiB, but restricts membership to the same direction in four already-transformed stream spaces. **Independent `[4,256]` entries:** 1,024 parameters/4 KiB, only **3 KiB** more weights and **12 KiB** more weights/gradients/two moments together (**arithmetic**); same activation-scale add, almost identical implementation, and can align membership with specialized stream coordinates (**reasoned gain**). **Work-vector-only marker:** 256 parameters but excludes membership from upstream oHC stream gates, reducing its useful routes (**estimate**). Paper has one marker over its ordinary hidden representation, no oHC default; parameter overhead stays constant at longer contexts. | **RECOMMENDED: independent stream entries in one full-hidden marker** — even a modest gain from adapting to four stream coordinate systems outweighs 12 KiB of extra training storage and negligible extra implementation/latency (**arithmetic cost, speculative gain**). |
| **Q4 Bundle selector** | **Existing fixed-count/entropy:** keeps a predictable byte-LM work/quality operating point and isolates coordination. **Paper confidence >.7 plus fallback:** same per-pass arithmetic and potentially better high-confidence choices; Table 2 favors confidence over matched-size entropy by **.83 GSM8K points** (**paper result**). At L8 fixed count 2 takes four passes; confidence can take one to eight (**arithmetic**); I expect a small byte model to cross .7 infrequently early in learning, making several extra prefix reconstructions more likely than major quality gains (**reasoned estimate**). **Learned selector:** could target dependence better, but needs a new learning problem and tuning/maintenance. Longer prefixes magnify added passes. | **RECOMMENDED: retain the current selector in the primary addition** — the expected risk of doubling early byte-LM passes outweighs a likely modest selector-quality gain, especially at longer prefixes (**estimate**); the paper-confidence rule remains a worthwhile separately reported comparison, not the default. |
| **Q5 Corruption and visible labels** | **Masks-only plus .2 visible CE:** adds preservation supervision to already-computed logits, O(4,096×257) loss work at maximum shape and no extra backbone call (**arithmetic**); plausibly discourages marker-induced disruption of useful visible features (**reasoned gain**). **Masked-only:** saves that small loss but leaves preservation entirely indirect. **Self-generated wrong-visible draft plus edit/clean CE:** offers error-correction robustness, but adds a full draft and new evidence/training semantics mostly useful when the decoder can edit. Paper uses the third; its **2.73-point editing ablation** is not a preservation-loss ablation. Draft/prefix cost grows with context. | **RECOMMENDED: masks-only plus .2 visible CE for factorized branches** — a weak but plausible regularization benefit is worth the small added CE work, whereas draft-error training spends a full forward on a correction capability the immutable decoder cannot use (**estimated return, arithmetic cost**). |
| **Q6 Small training bundles** | **Uniform feasible size/subset:** broad membership combinations at at most 4,096 sampled indices per B8/T512 draw (**arithmetic**). **Uniform 1–16 then clip:** similar host cost, but at m4 selects the full set with **13/16** probability instead of **1/4** (**arithmetic**), likely undertraining partial announcements (**estimate**). **Inference-selected training bundles:** reduces distribution mismatch but concentrates training on current easy decisions and adds selector/circuit synchronization; random sets should generalize better during changing model confidence (**reasoned estimate**). Paper specifies uniform 1–16 but not short-mask handling; all host work scales linearly with block rows. | **RECOMMENDED: uniform feasible size then uniform subset** — broader partial-set coverage should improve generalization more than selected-bundle matching at essentially equal or lower total cost (**reasoned estimate**), while clipping throws away diversity for no useful saving. |
| **Q7 Branch reduction** | **Sum over B×T:** preserves selection pressure while adding assignment learning. **Average branches:** same compute but reduces selector loss relative to the clean objective. **Inverse-inclusion-weight B:** can prevent small bundles receiving less total weight, but upweights individual tokens by as much as **8 at L8 / 16 at L16** (**arithmetic**), probably increasing gradient noise and emphasizing small announcements disproportionately (**estimate**). Paper leaves final reduction unspecified. Memory/latency differences are negligible at B8 and longer contexts; retuning and learning balance dominate cost. | **RECOMMENDED: sum A+B with the existing denominator** — I expect preserving a strong selector plus unamplified bundle supervision to learn more reliably than halving A or increasing B's variance, at effectively equal runtime (**reasoned estimate**). |
| **Q8 Combined marked supervision** | **Marked-subset joint event:** exactly trains the announced decision while marginalizing other masks; generalized event loss/flows cost more implementation/testing than the shortcut, but keep current two-evidence chain complexity. **Full unresolved NLL twice:** easier reuse but may teach B a generic full-block denoiser that ignores membership (**reasoned risk**). **Marked marginal CE:** targets the right positions but loses joint bundle pressure, plausibly reducing coherent assignment (**estimate**). A C512/L8 forward-backward traversal has about **7.34 million transition terms** (**arithmetic**); subset supervision does not shorten the chain, and active work scales with context. | **RECOMMENDED: marked-subset joint event NLL despite the larger implementation** — stronger membership-specific joint learning and correct marginalization should repay a reusable event-loss implementation, while the easier full-NLL route risks training away the very coordination signal (**reasoned gain, comparable runtime order**). |
| **Q9 Announcement and writes** | **Fresh B scores, write within A's set:** preserves membership conditioning, sometimes defers positions, and has fallback/last-pass completion. **Free B reselection:** may accept more tokens at the same query cost, but unannounced positions did not get marked-branch masked supervision. **Remark repeatedly:** can align membership and scores, but adds serial tail/circuit rounds and a stop rule. At L8 all recommended paths finish within eight passes (**arithmetic**); I expect many A selections to remain acceptable because marking refines rather than replaces context (**estimate**), limiting the deferral penalty. Longer prefixes increase the cost of extra denoising passes. | **RECOMMENDED: recompute scores and restrict writes to announced positions** — expected small deferral cost buys better-trained membership-matched assignments; free reselection risks errors on untrained outputs, and repeated remarking likely costs more than the residual agreement improvement (**reasoned estimate**). |
| **Q10 Combined sampling** | **Exact joint proposals/shared uniforms:** retains circuit correlations and existing decoder semantics, O(LK+LV) sampling after messages with K32/V257 (**arithmetic**). **Sequential conditional greedy:** potentially improves greedy likelihood but adds up to eight serial conditional choices at L8 and can collapse useful alternatives; it is not exact token MAP. **Independent marginal draws:** slightly simpler but directly loses sampled cross-position agreement. I expect coherent exact samples to be more valuable than the small sampler saving, while greedy's task-quality benefit depends on the application (**reasoned estimate**). Costs scale with chain batches, not prefix length per chain; CoCommit's paper evaluation is greedy-only. | **RECOMMENDED: retain exact joint proposals with shared A/B random numbers** — preserving joint coherence should earn more than the modest sampling work, whereas serial conditional greedy spends additional queries on a different decoding goal (**reasoned quality estimate, arithmetic work order**). |
| **Q11 Feedback output** | **B:** coordinated beliefs and writes with no extra model call. **A:** equally cheap, perhaps less volatile but ignores newly learned membership information. **Fresh post-write:** directly conditions on new commitments and likely improves deferred beliefs, but needs another whole denoiser/circuit call. N4096 features/logits are **4/4.016 MiB** per array (**arithmetic**); only one feedback source need remain live. I expect the next ordinary pass already to capture most post-write benefit, making an immediate extra pass redundant (**reasoned estimate**), especially when clean reconstruction grows with context. | **RECOMMENDED: marked-output feedback** — it plausibly retains most useful refinement for free, while a fresh post-write pass duplicates information the next pass soon obtains and A discards the new coordination signal (**reasoned estimate**). |
| **Q12 Trainable parameters** | **Full neural model plus marker:** learns membership-sensitive features throughout the model using the existing optimizer. **LoRA plus marker:** saves parameter-gradient/moment storage but adds adapter/checkpoint/rank maintenance and restricts updates. **Marker only:** maximally restricts learning and saves optimizer work, but is likely to teach mostly local bias rather than a new inter-position behavior (**reasoned estimate**). Roughly 7.1M parameters occupy **108 MiB** for f32 weights+gradients+two moments (**arithmetic**, not a live-memory measurement); our sampled pool reservation is **8.83 GB measured**. Full-stream marker training storage is **16 KiB** (**arithmetic**). Longer contexts mostly increase activations/history, not this storage. | **RECOMMENDED: train the full existing neural model plus marker** — expected adaptation quality outweighs limited parameter-memory savings at this size, while avoiding adapter-specific maintenance; this is deliberately larger than marker-only or low-rank training (**reasoned gain, arithmetic storage**). |
| **Q13 Visible revision** | **Immutable observations:** no error repair, but a clean coordination comparison and unchanged evidence/replay rules. **Active-block T2T:** real recovery potential; disabling editing loses **2.73 GSM8K points** in the paper, though our benefit is unknown. It needs draft-error supervision, revisable-evidence semantics and extra stopping/passes. **Mark visible tokens:** may further improve coordinated revision but adds another training distribution. I expect repair to have substantial absolute value, yet a worse return when inserted ad hoc here than when sharing Step 11's explicit error-exposure/repair training (**reasoned estimate**). Completed-block reopening is separate and becomes increasingly costly at long context. | **RECOMMENDED: keep observations immutable for Step 10 and assess repair with Step 11's training** — this is not a prediction that editing lacks benefit; concentrating repair's substantial implementation/training cost in one coherent mechanism should beat maintaining a second partial correction path (**reasoned maintenance/quality estimate**). |
| **Q14 Tail batching** | **Two C-row calls:** identical math and shared caches, with two tail launch sequences. **One 2C-row call without tail retrieval:** compact-cache concatenation adds about16MiB recurrent state at C512 plus other small caches; isolated forward/backward saves18.533ms [15.919,22.519], but full-step saving24.124ms [-23.326,49.555] includes zero (old-Codpula profile, not combined). Per-block calls would lose the established batching gain. | **SELECTED: Separate** — the isolated advantage does not establish useful complete-step savings. Explicit Doubled remains a verified no-tail-retrieval option; Doubled with marked-tail retrieval is rejected, never silently executed as Separate. |
| **Q15 Length one** | **Exact causal bypass:** maintains the established next-token computation. **Marked noisy singleton:** adds a tail/current-target embedding, changing the causal model. **Mark last prefix token:** adds a tail on clean context, not an unresolved decision. Singleton within-bundle dependence is zero (**mathematical fact**), so I expect essentially no gain from the requested coordination mechanism for either added-pass option; any generic refinement gain would need a different design. Prefix costs increase with context. | **RECOMMENDED: preserve the exact causal bypass** — the alternatives spend extra computation and compromise next-token identity for approximately zero expected within-bundle coordination gain (**reasoned consequence of a singleton decision set**). |
| **Q16 Circuit reduction** | **Average both branch contexts/update once:** preserves established optimizer scale at at most **1,024 contexts** for C512 (**arithmetic**). **Count unique blocks with summed flows:** effectively increases circuit pressure without strengthening neural loss correspondingly. **Update after each branch:** adds a probability/momentum update and makes B see different rows. I expect consistent shared-model statistics and unchanged step scale to reduce optimizer imbalance/retuning more than stronger or alternating updates improve learning (**reasoned estimate**). Context growth changes counts, not desired scaling. | **RECOMMENDED: combine contexts/counts and update once** — stable scale and a common pre-update objective should improve learning reliability at equal or lower cost; doubling pressure or serial updates likely buys tuning work rather than better coordination (**reasoned estimate**). |
| **Q17 Correctness check** | **Saved-h zero-marker repeat:** proves split reuse without training another model. A trained sequential-depth arm could test attribution, but the user has not authorized comparison arms or quality studies. | **DECIDED: saved-h zero-marker equivalence only** — exact/near-exact equality is a correctness gate; no sequential-depth training or quality comparison is implemented or run. |

Zero marker initialization itself is specified by the paper; no random marker initialization is proposed. The separate bundle RNG, strict padding exclusion, unchanged clean replay and cache reuse are implementation requirements, not additional model choices.

## 6. Cost at B8/T512/L8

### 6.1 Known baseline versus assumptions

Measured current tensor-Codpula training baseline: **8,000.071 ms** for one post-warm B8/T512 diagnostic, **30,745 launches**, **8,826,550,976 bytes sampled pool reservation** (PLAN Step 9). B1/T512 complete-step tensor median is **1,697.014 ms**, factorized median **1,522.502 ms**. These are optimizer steps, **not generation forwards**, and pool reservation is not live peak memory. No CoCommit overhead has been measured. The estimates below are planning scenarios, not timing promises or upper limits.

`N=B*T=4096`, maximum `C=N/L=512` independent active chains. With the existing uniform mask probability, the expected skipped all-observed fraction is `1/(L+1)=1/9`, giving about **455 active chains / 3,641 processed positions**, not 512 chains on every draw. Expect 2,048 genuinely masked targets. All training/forward counts include observed positions inside active chains.

### 6.2 Additional work and memory

For the proposed factorized training path the additional neural work is **one differentiated block-3 tail plus readout/head on all active positions**. The student prefix is not recomputed; its backward receives the accumulated A/B gradient. The detached teacher, clean scan, state extraction and optimizer remain once. Ignoring final heads and layer differences, this is one extra differentiated layer compared with four for the original student, not 25% of the **whole** step. Blocks 0–2 contain injections that block 3 lacks.

At the maximum N4096 size, the extra MLP alone has `N*(256*512+512*256)` = **1.074 billion multiply-accumulates**; the extra shared vocabulary projection is `N*256*257` = **269.5 million multiply-accumulates**. Mixer/gate projections, normalization, recurrence, oHC and attention add work. These are forward counts; backward adds its own projection and state-gradient work. The current ordinary HOLA segment has at most `16 cache + 15 unfinished + 8 active = 39` keys per query; its QK/read contribution is up to `N*4*39*(32+64)` = **61.34 million multiply-accumulates**, excluding softmax and retrieval. MoBA changes this key count and must be costed separately.

Combined training additionally prepares a marked circuit and computes its two event queries, neural VJP and positive/negative flows/reaches. It adds roughly another **student circuit's** work, not another teacher/marginal-feedback pass and not generation dependence scoring. The marked event can have fewer target positions but messages still traverse its full actual chain. The added clean-visible factorized CE is small compared with the new tail; no self-generated draft cost is assumed.

| Retained/temporary value at the maximum shape, f32 | Tensor-value size |
| --- | ---: |
| Saved pre-tail oHC streams `[512,8,4,256]` | **16 MiB** (about 14.22 MiB at expected active count) |
| Same without oHC | **4 MiB** |
| One output feature array | **4 MiB** |
| One `[512,8,257]` logit/marginal array | **4.016 MiB** |
| Existing one-layer recurrent snapshot `[512,4,32,64]` | **16 MiB**, reused, not another persistent model state |
| Marker parameters | **4 KiB** with oHC, **1 KiB** without; gradient and two AdamW moments bring total training storage to **16/4 KiB** |
| One fully materialized marked `[C,L,32,257]` leaf tensor, if chosen | **128.5 MiB**, avoid materializing merely to duplicate evidence queries |

Autograd tail activations, gradient accumulation, circuit statistics and backend temporaries add memory beyond this table. Do not add16MiB to sampled reservation and call it a measured peak. Explicit no-tail-retrieval Doubled duplicates compact tail caches, about16MiB recurrent state plus convolution/clean-read segments at this shape; both branches need backward activations either way. Selected Separate shares the original banks, including with tail retrieval. Unsupported Doubled plus tail retrieval is rejected; these are tensor-size estimates, not measured allocator peaks.

### 6.3 Training-step planning range

A transparent model is

`T_new = T_baseline + T_marked_tail_forward_and_backward + T_marked_head/loss + T_marked_circuit + added_dispatches`.

For scheduling only, assume the extra differentiated tail/head costs **5–20% of the complete baseline**, and a marked circuit/extra reductions and dispatches add **1–10%** in the combined case. Those fractions are assumptions: the equal-depth estimate, injection placement and the existing factorized/tensor difference motivate measuring them, but do not establish them. This gives the following deliberately broad **scenario**:

| Workload | Assumed added time | Assumed resulting step |
| --- | ---: | ---: |
| B8/T512 tensor Codpula + CoCommit | **0.48–2.40 s** | **8.48–10.40 s**, from the observed 8.00 s |
| B1/T512 tensor Codpula + CoCommit | **0.10–0.51 s** | **1.80–2.21 s**, from the observed 1.697 s |
| B1/T512 factorized + CoCommit | **0.08–0.30 s**, assuming only the 5–20% neural term | **1.60–1.83 s**, from 1.523 s |

Launch/memory effects can put actual values outside these ranges. In particular B8 runtime cannot be inferred by multiplying the B1 circuit difference by eight. No B8 factorized paired baseline is inferred from the Codpula diagnostic.

### 6.4 Per-generation-pass estimate

For one block-row batch define `P` as clean-cache preparation cost, `F` as the active unmarked noisy forward including its ordinary readout/head, `Q` as one Codpula preparation/marginal/proposal/dependence evaluation, and `R` as its number of denoising passes. Enabled generation pays `P` once per block rather than once per pass. The arithmetic per-pass accounting is

- factorized old pass: `P+F`; enabled amortized pass: `P/R+F+tail+head+selection`;
- combined old pass: `P+F+Q_A`; enabled amortized pass: `P/R+F+Q_A+tail+head+Q_B`.

For `n=1,D=4`, use **0.15–0.30 F** as a planning scenario for `tail+head`: block 3 omits injections, but readout/head is repeated and small launches may dominate. The combined path also roughly doubles the circuit-side generation queries; that is not a second backbone call. Scores use the existing two-traversal cavity optimization, not `2m+2` full neural evaluations. For C512/L8 the second circuit's local emission work is `N*32*257` = **33.69 million leaf terms**, marginal readout another 33.69 million, and two forward/backward traversals about **14.68 million transition terms**, plus sampling, local/gate normalization and gathers. Teacher/student training-flow work is absent in generation.

For a numerical sensitivity example, **if** a B8/T512/L8 block-row evaluation measured `F=100 ms` and `Q_A=10 ms`, added active work would be **15–30 ms** factorized or **25–40 ms** combined, giving amortized totals `P/R+115–130 ms` or `P/R+135–150 ms`. Reusing the clean cache additionally removes `(1-1/R)*P` per pass compared with reconstruction; this may outweigh the added tail at long prefixes. If F instead measured 1 s, scale the neural addition to 150–300 ms. These input times are hypothetical, not new measurements. Training timings provide no defensible absolute generation latency; measure `P,F,Q,R` before reporting one.

B8/T512/L8 above means **512 independent L8 block-row evaluations**, as in batched training/forward cost accounting. Actual generation currently handles one active block per sequence, so eight live sequences have eight L8 rows, not 512; T512 describes the prefix length, not 512 simultaneously generated positions per sequence. At B1 there is one L8 active block. Report both workloads distinctly. Full block generation uses between one and eight such denoising passes at L8, plus replay; CoCommit can change that pass count, so per-pass overhead does not predict tokens/s.

## 7. Correctness plan

The implementation checks below use the unchanged elementwise criterion for f32 tensor/GPU comparisons; executed results are recorded in §9:

`abs(actual - ref) <= 2e-5 + 3e-4 * abs(ref)`.

Use independent plain sequential f64 references for every changed calculation, not a second invocation of the same Burn graph. Existing scalar recurrence, convolution, HOLA, oHC, wavelet/Lngram and HMM references supply components; compose them with an independently written split/marker/loss calculation. Check tiny reference-versus-reference results more strictly before applying the f32 criterion.

1. **Split equivalence:** ordinary unmarked output versus prefix+tail at each supported cut; ordinary hidden and oHC versions; final normalized features and all logits. Zero-marker B equals A under deterministic execution. Cache saved streams before any tail normalization. Test a fixture that deliberately feeds A's output into B and establishes that it is a different function, not the reference.
2. **Marker and branch gradients:** scalar additive forward/VJP with independent `[4,256]` marker entries, including a fixture distinguishing it from shared-stream gradient summation; finite differences of a nonzero marker and shared prefix/tail/head parameters, zero-initialized marker receiving learning signal, branch-A+branch-B gradient sum, no detach at the shared student prefix. Check marked-only mask loss, visible `.2` terms, common `B*T` reduction, no inverse bundle-size weighting, and exact zero contribution from unselected masked CE.
3. **All call paths:** per-block versus batched, cached versus reconstructed, selected Separate versus explicit fitting no-tail-retrieval Doubled, and shared-bank Separate with real tail retrieval. Check clear validation failure for explicit Doubled plus tail retrieval; do not substitute another layout. Cover CPU/Vulkan and retained supported architecture/read/feedback paths, comparing every parameter gradient, not only logits/loss. No backend is silently chosen for an unsupported shape.
4. **Leakage and immutability:** perturb clean targets at/after the block start while keeping supplied noisy inputs/evidence fixed. Pre-tail states and both outputs must stay unchanged; labels affect only loss. Change clean training suffixes used to prepare caches and check the same. Supply a right-hand observed token and verify intended communication; change a right-hand marker with nonzero marker weights and verify a left query can respond through HOLA. Snapshot recurrence, selected slots, histories and token prefix before/after each branch and compare exactly. Call B before A and after A with identical inputs and verify equality.
5. **Boundary cases:** L1 exact causal identity and no coordination; L2/4/8/16, short final blocks, start at BOS and start inside chunk 16 (prefix length 19), no unfinished chunk versus unfinished chunk, empty HOLA slots, dirty padded values, all-observed/skipped blocks, one unresolved token, marker at first/last valid token, zero marker and zero hidden values, ties and noncontiguous inputs where supported. Padding contributes no marker gradient, no key or pooling term, and no circuit transition.
6. **Soft feedback:** same teacher and candidate IDs/weights/mass for A and B inputs; no feedback update between tails; teacher proposal probabilities detached but selected student embedding values differentiable; deferred positions update once from B, observed positions remain immutable, no gold/proposal IDs become marked evidence.
7. **Bundle selection and repeatability:** independent mask/bundle RNG regression; exact size/subset mapping with supplied draws, feasible-size cases, all-observed/singleton draw omission, row order unaffected by batching, original mask draws unchanged. Greedy token/position ties remain ordered. K0 returns A, not `Head(h)`.
8. **Combined event reference:** enumerate V2/3, latent states2/3, L2–4 for every observation/target subset, including unselected masks between selected positions. Compare `Z(e)`, `Z(e+gold[A])`, event NLL, conditional normalization, all neural VJPs, positive/negative flows/reaches and one dual-flow update. Signed flow difference must match independent f64 finite differences of the event objective with local normalizers. All-target event matches old conditional NLL; empty target has zero loss/flows. Do not condition the denominator on gold at unmarked positions.
9. **Fresh-circuit test:** choose a nonzero marker that changes both logits and pooled features. Record A/B potentials, gates and normalizers. Compare B messages/proposals/scores with enumeration under B, and include a fixture where deliberately reusing A's circuit or A's proposals changes the result. Test new-score threshold equality (strict `<`), fallback within announcement, full unresolved conditioning set, last-pass completion and absence of writes outside the announcement. Invalid evidence fails clearly.
10. **RNG and joint proposals:** same-seed A/B calculations use new conditional rows and preserve explicit evidence. With zero marker, common random numbers reproduce proposals/scores. Compare exact sampler conditional rows to enumeration; any frequency test uses a stated statistical interval, not the floating-point tolerance. Generation off retains old RNG consumption.
11. **Persistent replay and checkpoint:** no writes in either temporary tail; replay exactly once per completed block; replayed logits/states equal ordinary clean forward of finalized tokens. Save/load marker and moments, continue one deterministic step, compare parameters/loss/circuit state. Omitted CoCommit preserves historical config hashes, initialization, outputs and checkpoint loading; enabled incompatible math is rejected.

Extend existing reference commands/reports and the crate's existing test binary. The later implementation must pass these gates before any large run, and must retain failure evidence rather than loosening tolerances.

## 8. Optimization and comparison plan

1. **Reuse before new kernels.** Keep merged projections, fused short convolution, the selected recurrence kernel, fused normalization/gates, oHC `fused_gate`, analytic SiLU backward and noisy-HOLA fused read. The tail is ordinary active-block computation with a different input. Reuse the existing prefix retrieval implementation. No new attention operator, special marker predictor or persistent state is needed.
2. **One shared noisy prefix.** Avoid recomputing embeddings, oHC initialization or blocks 0–2 for assignment. Keep clean history once and regenerate only active tail Q/K/V. Avoid copying full clean banks or every layer's recurrent state for a single-layer tail. A small per-stream marker add and its reduction are initially tensor operations; fuse into the first oHC read only if measured dispatch/traffic justifies it, while still exposing the same marked streams to every oHC calculation.
3. **Batch blocks and compatible branches.** Retain selected Separate C-row calls with shared caches. Explicit no-tail-retrieval Doubled uses branch-major2C rows and compact-cache concatenation; its full-step evidence does not establish a saving. Tail retrieval requires Separate, rejecting an explicit Doubled request. Inference depends on A's selection and remains serial A then B; never fall back to the old per-block production loop.
4. **Reuse circuit kernels, not old circuit values.** Tensor circuit remains selected because its current integrated fused comparison did not establish a saving. The marked event uses the existing leaf/message/flow kernels with generalized evidence masks; there is no need for a new circuit architecture. Compare the existing fused path only with the new event VJP/reference gates passing. Within one B circuit, reuse leaf normalizers and messages across proposals/dependence queries; never reuse them across A/B changed neural outputs.
5. **Measure complete steps and generation.** At B1/T512 and B8/T512/L8 record identical parameters, masks/bundles/seeds, actual chain counts, f32 dtype, device/backend selectors and warm compilation status. Obtain synchronized isolated tail forward and forward/backward timings plus full loss/backward/gradient checks/clipping/AdamW/circuit update timings. Use two warmups and 11 interleaved pairs, alternating order as in current benchmarks; report paired differences and 95% intervals. Record launch counts and sampled reserved memory with their limitations. Measure actual single/eight-active-block decoding separately from the C512 training-shaped forward batch, and include host selection, circuit queries, feedback and replay.
6. **Scope of measurements.** Run saved-h zero-marker equivalence as correctness, CoCommit off/on complete-step timing as cost accounting, and equivalent tail-layout paired timing. Run only the authorized few-step context32 smoke and checkpoint/evaluation/generation checks. Do not run trained comparison arms, sequential-depth models, quality studies, sweeps or ablations. Include all partial tails, queries, feedback and replay in reported cost.

The numerical gates and measured component budget below determine implementation readiness, not quality improvement. The planning scenario remains an estimate until measurements replace it. No quality conclusion follows from a correctness gate, a cost comparison or the tiny execution smoke.

## 9. Implementation and numerical results

The enabled selector is `sbd.cocommit`; omission keeps historical serialization, fingerprints, initialization and checkpoint compatibility. `configs/lm512-cocommit.json` adds it to the existing Codpula profile. Factorized and Codpula outputs share the same saved-prefix/tail APIs. `cocommit.rs` handles both branch losses, bundle selection and marked reassignment; `sbd_training.rs` exposes per-block and batched prefix, tail and paired-tail calls. These APIs do not write tokens or persistent recurrent/cache/history state.

The marker is a trainable zero-initialized parameter appended after ordinary model initialization. Its gradient is enabled explicitly. Both student branches retain gradients through the same saved prefix. Teacher feedback is detached by definition; its execution uses frozen model parameters and detached cache aliases rather than building an unused graph before detaching its outputs. The teacher still uses the same selected membership, embeddings, gates, circuit and soft-top-k update.

### 9.1 Numerical and restart gates

All comparisons retain `2e-5 + 3e-4*abs(reference)`; no tolerance was enlarged. The reference command covers independent f64 additive marker/VJP and saved-input tail calculations, zero-marker A/B identity, ordinary and oHC representations, cached/reconstructed/per-block/batched layouts, hard/soft feedback, factorized/Codpula complete training losses and every parameter gradient, leakage and immutable clean snapshots, singleton causal identity, and a fresh marked circuit with subset-event loss. Generalized event enumeration and circuit VJPs/flows are also exercised by `codpula-reference`.

The independent f64 temporary Kron-tail oracle includes decay, prediction/erase/write, outer-product state edit and query read. Each of three active positions starts from the same duplicated clean state. CPU/Vulkan maximum output/state difference is `2.3545074445685543e-8`; saved hidden/state/snapshots remain bitwise unchanged. A deliberately chained reference differs by `.12951689897`, so the test distinguishes the required independent temporary recurrence from an incorrect recurrent active-block scan. Separate full-model fused gates cover the configured K32/V64 execution.

Real prefix retrieval uses the Step-8 retrieval configuration's D256/H4/K32/V64 tail with a 219-token prefix and L8 active block, not an empty or synthetic no-retrieval substitute. The normalized training/event objective passed CPU/Vulkan logits, loss and all-gradient checks: maximum absolute difference `1.5735626220703125e-5`, maximum tolerance ratio `0.53899813473`, loss difference `9.5367431640625e-7`.

An artificial unnormalized squared-logit/feature stress objective had failed at ratio `11.6143281239` with CoCommit. The required same-shape/same-weights plain SBD+MoBA discriminator also failed, at ratio `21.0718978301`: this is case (a), a pre-existing ill-conditioned-objective result, not a CoCommit-specific pass. Both failure reports are retained. The approved normalized objective above is the acceptance gate; no retrieval path or gradient group was excluded.

Enabled checkpoint continuation uses the actual `train::optimizer_step`, including soft feedback, Codpula flow update, active clipping at threshold `.05`, configured AdamW, and marker moments. Saved/loaded/continued parameters and losses are bitwise equal, marker moment signatures match, and circuit update time advances identically from 1 to 2. The historical causal `checkpoint-check` also passed with bitwise-equal next-step parameters, zero logit error and exact optimizer moments before/after continuation. These are production-step restart gates, not direct AdamW-only tests.

The final enabled continuation also uses two one-row backward groups at B2 before that single production optimizer step. Full-versus-grouped checks cover ordinary SBD and CoCommit, factorized and Codpula, soft feedback/oHC/in-block reads, CPU and Vulkan, loss, every parameter gradient, all nine flow tensors and exact context counts. The final accumulation-only maximum tolerance ratio is `.02246959246`; CPU/Vulkan across the enlarged fixtures reaches maximum absolute gradient difference `2.193450927734375e-5` and maximum ratio `.25555908324` (different elements). The real-retrieval fixture remains the largest accepted tolerance ratio (about `.539`); absolute error alone is not the criterion.

Probability-row selection now rejects NaN, positive infinity, and unsupported/all-zero rows before proposals or entropy. Legitimate `-inf` zero mass contributes zero to entropy; no uniform replacement is used. Focused regression cases cover these failures and supported point-mass/mixed-zero rows.

### 9.2 Exact execution changes and memory failures

The original unchanged B8/T512/L8 diagnostic failed requesting `3,489,660,928` bytes while a complete TensorHOLA read was computed only to obtain cache membership. Position-only preparation now invokes the existing fused selection schedule directly and shares its ranking, stable ties, reset/document/padding and cache-membership semantics. It does not compute/discard attention values or change model math. Ninety fitting-shape cases compare CPU, Tensor and fused selectors against both an independent scalar oracle and the old full-read path at every cache boundary; maximum membership difference is exactly zero.

After that change, B8 progressed to teacher circuit marginals but again requested `3,489,660,928` bytes while retaining an unused teacher autodiff graph. Both failures are preserved (`b8-oom.json`, `b8-feedback-oom.json`); neither supplies a completed step timing. The no-gradient teacher execution described above removes that unused graph without changing feedback values or student gradients.

Frozen teacher logits/features match tracked-then-detached execution bitwise and have no gradient tracking. This removed unused graphs but did not by itself make repeated full B8 steps fit: a section-synchronized first step returned with sampled pool reservation reaching `5,424,969,344` bytes, then the next ordinary warm step still failed the same `3,489,660,928`-byte request. `b8-memory-trace.json` retains that diagnosis. Explicit row accumulation below addresses the remaining complete-step memory pressure without a hidden retry.

The explicit execution selector `sbd.rows_per_pass` divides full-batch examples into row groups before backward. Masks and CoCommit bundles are sampled once in original full-batch row order; each group receives its corresponding immutable choices. Local losses are multiplied by `group_rows/full_batch_rows`, parameter gradients are summed, and circuit flows/counts are summed without a per-group update. Finite-gradient checks, clipping, AdamW and circuit row updates occur once after all groups. This preserves total examples, tokens, denominators and RNG consumption; it is not a smaller workload or an allocation-error fallback. The B8 profile explicitly selects four rows per pass (two groups); B1 naturally uses one group.

With that explicit two-group execution and the evidence-selected separate tails, the unchanged B8/T512/L8 workload completed: **6,893.824 ms**, **49,928 launches**, **3,951,724,736 bytes** sampled pool reservation. Two warmups were `14,261.864` and `6,266.990 ms`; the first includes first-use compilation. This is one post-warmup diagnostic, not a paired speedup against the historical 8.00 s result. All 4,096 clean targets and the full sampled noisy work are included, with one final optimizer/circuit update. Pool samples are not physical-device peak memory. The earlier doubled diagnostic (`6,243.454 ms`, `48,903` launches, `5,888,443,520` sampled bytes) is retained separately; these single diagnostics do not establish a B8 layout speedup.

### 9.3 Composition hooks

The cached prefix/tail/pair output API is independent of absorbing writes. R²D and TCT integration must call the coordinated A/B denoiser when this selector is enabled rather than silently using unmarked `cached_blocks`. Their temporary announcements/provisional reveals must remain distinct from immutable final observations; episode completion owns final writes. Step-11 semantics are not changed here, and the parent integrates that combined path.

### 9.4 Paired execution cost and selection

B1/T512/L8 uses the same materialized initial ordinary parameters, example index 0 and mask seed 1, independent optimizer histories, two warmups and 11 alternating interleaved trials. Full steps include host examples/uploads, the clean pass, frozen teacher/feedback, student prefix, both tails/heads, circuit/event/flow work, scalar reads, complete backward, finite-gradient checks, clipping and AdamW. No training quality comparison is made.

| Full step | Median ms | Issued launches | Sampled pool reservation, bytes |
| --- | ---: | ---: | ---: |
| CoCommit off | 1,286.855 | 23,032 | 1,946,849,280 |
| On, separate tails — selected | 1,408.335 | 26,022 | 1,946,848,192 |
| On, doubled tail batch | 1,393.231 | 25,506 | 1,946,850,624 |

The paired selected-on overhead is **116.839 ms**, bootstrap 95% interval **[89.963,121.664] ms** (about 9.1% of the off median). Separate-minus-doubled paired saving is **24.124 ms [−23.326,49.555]**; the complete-step interval includes zero, so **separate is the selected default**, following the recorded measurement rule rather than the earlier speculative 2C recommendation. Both paths remain numerically verified. Retrieval always shares banks rather than duplicating full clean KV.

Supported combinations are enforced in configuration validation and at the
paired-output boundary: Separate supports tail retrieval; Doubled supports
no-tail-retrieval tails only. Earlier saved-prefix retrieval does not preclude
Doubled. The real layer3-MoBA acceptance fixture selects Separate explicitly.

Resident isolated tails do show a doubled saving: separate/doubled forward medians **33.772/25.977 ms**, forward+backward **101.624/83.565 ms**; paired gains **7.547 [7.346,8.115] ms** and **18.533 [15.919,22.519] ms**. This does not establish a complete-step saving.

Position-only schedule preparation is adopted independently: old membership-through-full-read median **99.272 ms**, position-only **1.038 ms**, paired saving **98.227 [91.325,104.213] ms**, exact positions on every pair. This resident B1/H4/T512/C16/chunk16 measurement excludes uploads/query normalization; schedule storage is **8,448 bytes**, and membership has no backward. Whole-model gains cannot be obtained by multiplying this isolated difference by the layer count.

Pool reservation samples include all three resident benchmark models and allocator reuse, not independent per-model live memory or physical-device peaks. Reports: `artifacts/cocommit/paired/cocommit-benchmark.json`, `cocommit-reference.json`, `cocommit-diagnostic.json`, `historical/checkpoint-check.json`, `adoption.json`, and `workspace-tests.json`.

### 9.5 Authorized execution smoke

The context32/B1 run trained exactly **96 target bytes in three steps**, then loaded its final checkpoint for block evaluation and generation. Generation finalized and causally replayed **eight tokens**, one block and one denoising pass. These outputs exercise enabled inference, not model quality. Inference-only construction leaves the marker untracked; training construction enables its gradient. The smoke's unused recall-count validation setting was set to `[2]` so the context32 config is valid; B8/T512 measurements are unchanged.

Disposable smoke model/optimizer payloads, historical diagnostic copies and an earlier failed temporary fixture's payloads were removed; configurations, state, metrics and numerical reports remain. `checkpoint-cleanup.json` records seven removed payload files. Original historical checkpoints and main's working files remain untouched. No training study, trained comparison arm, sweep or sequential-depth comparison was run.

