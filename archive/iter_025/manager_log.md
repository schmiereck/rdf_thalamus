# Research Manager Log - Iteration 025

## Iteration 025 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints — Iteration 025

1. **Treat iter_025 as a localization probe, not a constructive attempt.** The journal already commits to ID-contrastive on the shared CNN; the Planner must structure the iteration so that the *informational outcome is the same in both directions*. Specifically: in addition to (or as a parallel arm to) end-to-end ID-contrastive shaping z_dyn, include a **supervised linear probe directly on z_dyn** (and ideally a frozen-encoder probe baseline) so that the result decomposes cleanly into "objective insufficient" vs. "architecture ceiling." Without that decomposition we will spend iter_026 swapping objectives again.

2. **Pre-declare the falsification criterion and a fresh seed batch before any code is written.** Carry the 0.10 delta_R2_color threshold forward unchanged for direct comparability with iter_021–024, and explicitly state in advance: (a) what counts as "ID-contrastive succeeded" (threshold cleared, collapse ≤ 1/5, on a *new* seed set disjoint from iter_021–024), (b) what counts as "architecture ceiling reached" (linear probe also ≤ 0.10 under direct supervision), and (c) what the agreed next move is for each of the four quadrants. The seed-reuse limitation called out in iter_024 §6 must be closed this iteration, not deferred again.

3. **Resist scope creep toward "separate encoder for z_dyn" inside iter_025.** That is the obvious next architectural move if the ceiling probe fails, but bundling it into the same iteration would re-couple objective and architecture and re-create exactly the disambiguation problem this iteration is meant to solve. Also: language in the report must stay restrained — "consistent with an architecture-level bottleneck", not "shows that the architecture cannot encode identity" — and any positive ID-contrastive result must be qualified as supervised (slot IDs are privileged information), not as evidence that the decoder-free self-supervised problem is solved.

---

## Iteration 025 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
The failure of identity encoding in z_dyn (delta_R2_color < 0.10 across iter_021-024)
is attributable to the objective being insufficiently discriminative, NOT to the
architecture being incapable. Specifically:

H1 (Architecture Capacity): The shared-CNN dual-stream NonParametricJEPASpatial
encoder CAN encode object identity in z_dyn when provided with a direct supervised
color regression loss that backpropagates through the encoder. Under this condition,
delta_R2_color ≥ 0.10 (mean over non-collapsed seeds) with collapse rate ≤ 1/5.

H2 (ID-Contrastive Viability): A color-similarity-based contrastive objective
(using privileged environment slot IDs to define positive/negative pairs) is
sufficient as a self-supervised proxy for identity encoding, achieving delta_R2_color
≥ 0.10 with collapse rate ≤ 1/5.

The full hypothesis (H1 AND H2) is falsified if H1 fails (architecture ceiling
reached). Partial falsification (H1 holds, H2 fails) means the architecture CAN
encode but the contrastive formulation is insufficient — a less severe outcome.

**Proposed Falsification Criterion:**
PRIMARY FALSIFICATION (H1 — Architecture Ceiling):
Arm B (Supervised Color Probe + VICReg, d_max=8) fails to achieve delta_R2_color ≥ 0.10
(mean over non-collapsed seeds) OR collapse rate > 3/5, on the fresh seed set
[7, 17, 31, 53, 71] disjoint from iter_021-024.

If H1 is falsified: the architecture cannot route identity information to z_dyn
under ANY objective through the shared CNN + soft-argmax centroid head. The next
iteration must modify the architecture (separate z_dyn encoder or relax decoder-free).
Language: "consistent with an architecture-level bottleneck on identity encoding."

SECONDARY FALSIFICATION (H2 — ID-Contrastive):
Arm C (ID-Contrastive + VICReg) fails to achieve delta_R2_color ≥ 0.10 with
collapse rate ≤ 1/5, while Arm B succeeds.
If H2 is falsified but H1 holds: the architecture CAN encode, but the contrastive
formulation is insufficient. Try direct supervised as training objective or stronger
contrastive variants. Language: "contrastive formulation insufficient; architecture
not the bottleneck."

FOUR OUTCOME QUADRANTS (pre-declared next moves):
B succeeds, C succeeds → H1+H2 confirmed; continue developing ID-contrastive
B succeeds, C fails → H1 confirmed, H2 refuted; try direct supervised objective
B fails, C succeeds → Check for implementation bugs (supervised ≥ contrastive)
B fails, C fails → Architecture ceiling; next iteration: separate z_dyn encoder

Any positive ID-contrastive result is qualified as supervised (slot IDs are
privileged information), NOT as evidence that the decoder-free self-supervised
problem is solved.

**Proposed Method:**
EXPERIMENT DESIGN: 4 arms × 5 fresh seeds × 5000 steps = 20 runs.

Fresh seeds: [7, 17, 31, 53, 71] — disjoint from [42, 123, 456, 789, 999].

ARM CONFIGURATIONS:

Arm A: JEPA+VICReg Control (5 seeds)
  - primary_objective="jepa", var_weight=25, cov_weight=25, sim_weight=25
  - d_max=8, d_t=3, dyn_readout="centroid_gated", pos_encoding="none"
  - CCR covariance mode (ccr_smooth_weight=10, ccr_spatial_weight=10)
  - gdasr_log_only=True
  - Provides baseline incidental identity encoding; directly comparable to
    iter_022-024 control arms but on fresh seeds.

Arm B: Supervised Color Probe + VICReg (5 seeds) [CRITICAL DIAGNOSTIC]
  - primary_objective="jepa" (KEEPS JEPA as readout, preserving the prediction
    pathway for centroid tracking and surprise readout)
  - ADDITIONAL supervised_color_loss on z_dyn with supervised_weight=25.0
  - Color probe head: per-channel linear mapping z_dyn[:, d] → 3D RGB of the
    matched object (nn.Parameter weight (d_max, 3) + bias (d_max, 3))
  - Channel-to-object matching: sort z_coord[:, :d_t] and info["positions"][:, :N]
    by position → monotonic assignment (reliable per tracking quality metrics)
  - supervised_loss = MSE(color_pred, colors_target) averaged over d_t channels
    and 3 color channels
  - VICReg on z_dyn (var_weight=25, cov_weight=25) — prevents collapse
  - CCR covariance mode, same as control
  - d_max=8, d_t=3, dyn_readout="centroid_gated"
  - Gradient flow: supervised_loss → color_probe_head → z_dyn → encoder
  - This is the architecture ceiling probe: if z_dyn cannot encode identity
    even under this strong signal, the architecture is the bottleneck.

Arm C: ID-Contrastive (Color-Similarity Metric Learning) + VICReg (5 seeds)
  - primary_objective="jepa" (KEEPS JEPA as readout)
  - ADDITIONAL id_contrastive_loss on z_dyn with contrastive_weight=25.0
  - Implementation: 
    1. For each sample b in batch, match channels to objects via sorted positions
    2. Collect all (z_dyn[b,d], color[matched_obj[b,d]]) pairs across the batch
       → n = B * d_t = 96 pairs
    3. Compute target color similarity matrix: c_sim[i,j] = cosine_sim(color_i, color_j)
    4. Compute z_dyn distance matrix: z_dist[i,j] = |z_dyn[i] - z_dyn[j]|
    5. Loss: MSE(z_dist_normalized, (1 - c_sim) * z_scale) where z_scale is
       the current std of z_dyn distances (adaptive scaling to avoid collapse)
    6. Alternative (simpler): SupCon with discretized color class labels.
       Discretize each object's color into one of 8 bins based on which RGB
       quadrant it falls in (R>G>B, R>B>G, G>R>B, etc.). Apply SupCon loss
       with these labels. This gives a clear positive/negative structure.
  - VICReg on z_dyn (var_weight=25, cov_weight=25)
  - CCR covariance mode, same as control
  - d_max=8, d_t=3, dyn_readout="centroid_gated"
  - Uses privileged information (environment colors) to define identity pairs.

Arm D: Supervised Color Probe + VICReg, d_max=16 (5 seeds)
  - Same as Arm B but with d_max=16
  - Tests whether increased latent channel capacity improves supervised encoding
  - d_max=16, d_t=3 (frozen at 3 active channels, same as all other arms)

FILES TO CREATE/MODIFY:

1. src/models_dual_stream.py:
   - Add color_probe parameters to NonParametricJEPASpatial.__init__()
     (weight: Parameter(d_max, 3), bias: Parameter(d_max, 3))
   - Add compute_supervised_color_loss() method that:
     (a) receives z_coord, z_dyn, and ground-truth positions/colors tensors
     (b) sorts z_coord channels and positions by value for monotonic matching
     (c) gathers z_dyn and colors in sorted order
     (d) computes color_pred = z_dyn_sorted * weight + bias (per-channel linear)
     (e) returns MSE(color_pred, colors_sorted[:,:d_t,:])
   - Add compute_id_contrastive_loss() method that:
     (a) receives z_dyn, positions, colors tensors
     (b) matches channels to objects via sorted positions
     (c) discretizes object colors into 8 bins (RGB quadrant)
     (d) applies SupCon loss with these discrete labels
     (e) returns the contrastive loss
   - These are computed EXTERNALLY in the training loop (not inside forward()),
     similar to how multi-step SFA was handled in iter_024.

2. src/run_phase0_id_probe.py (NEW):
   - Main experiment runner, based on run_phase0_sfa_multistep.py structure
   - Extended ReplayBuffer: stores (x_hist, x_target, positions, colors, radii)
   - 4 arms × 5 seeds × 5000 steps
   - For Arms B, D: compute supervised_color_loss after model forward pass;
     add to total loss before backward()
   - For Arm C: compute id_contrastive_loss after model forward pass;
     add to total loss before backward()
   - Same evaluation suite as iter_024: semantic probes, collapse checks,
     centroid MSE, tracking quality, normalized temporal variance,
     within/between trajectory variance, shuffled-frame control
   - Results saved to archive/iter_025/results/

3. src/pre_registration.md: Updated with this plan.

TRAINING PROTOCOL:
- 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000
- d_t=3 frozen, gdasr_log_only=True (M3 preserved)
- VICReg: var_weight=25, cov_weight=25 (batch-level, M1 preserved)
- All arms: centroid_gated dyn_readout, CCR covariance mode
- supervised_weight=25.0 (matching var_weight scale; if collapse occurs,
  ramp 0.1→25.0 over 500 steps as fallback)
- contrastive_weight=25.0 for Arm C
- Color probe head initialized with small random weights (std=0.01)

EVALUATION PROTOCOL:
- Same as iter_024: semantic probes, collapse check (per_dim_std < 0.5),
  centroid MSE, tracking quality, normalized temporal variance,
  within/between trajectory variance, shuffled-frame control
- PRIMARY METRIC: delta_R2_color (frozen-encoder linear probe, same computation
  as iter_021-024 for direct comparability)
- ADDITIONAL: report the training-time color_probe_loss convergence for Arms B/D
  and contrastive_loss convergence for Arm C — these show whether the objective
  is being optimized successfully even if the downstream metric doesn't improve
- Checkpoint evaluation at step 2000 (monitoring only; ALL 5000 steps must complete)

CONSTRAINTS (per Manager pre-planning hints):
- Fresh seed set [7, 17, 31, 53, 71], disjoint from iter_021-024
- No separate encoder for z_dyn in this iteration (resist scope creep)
- Language: "consistent with an architecture-level bottleneck" not "architecture
  cannot encode identity"; positive ID-contrastive results qualified as supervised
- The informational outcome must be the same in both directions (both success
  and failure of ID-contrastive are informative when paired with the supervised arm)

---

## Iteration 025 -> Planner [Strategic Guidance]

Manager's Note — Iteration 025 (Critique of Proposed Plan)

The plan is well-aligned with the localization-probe framing from pre-planning, and the four-quadrant decision tree is exactly the disambiguation discipline this iteration needs. Three concerns require revision before execution:

1. **The "channel-to-object matching via sorted positions" step is a silent confound that can fake or mask the ceiling result.** Arms B, C, D all match `z_dyn[:, d]` to object `d` by sorting `z_coord[:, :d_t]` against `info["positions"]`. Whenever sort orders disagree (collisions, near-equal positions, swaps across timesteps), the supervised target attached to a given z_dyn channel flips between objects — the loss then teaches z_dyn to encode *position rank*, not identity, and a positive Arm B result is compatible with "architecture can encode position rank" rather than "architecture can encode identity." Conversely, persistent mismatch will depress Arm B and produce a false ceiling result. **Required before execution:** (a) report, on the existing tracking-quality metric, the empirical mismatch rate for the sorted-position assignment over the eval set, and (b) add a Hungarian / minimum-cost assignment as an alternative match (or an oracle "true ID" assignment derived from the physics engine slot IDs the journal already commits to using) and run Arm B under both. If the two assignments disagree by more than a few percent on `delta_R2_color`, the ceiling claim is not interpretable.

2. **Falsification thresholds are inherited but not validated against the new probe; pre-register a noise floor before the run.** The 0.10 `delta_R2_color` threshold was chosen against self-supervised objectives. Under a *supervised* color probe that backprops through the encoder (Arm B), the natural ceiling is likely well above 0.10, so passing 0.10 is weak evidence of "architecture sufficient" — it could be barely-above-trivial. Pre-register, in `src/pre_registration.md`: (i) the expected null/floor (frozen random encoder, fresh seeds) and (ii) a *separation* criterion, e.g. Arm B must clear the floor by ≥ X and Arm C must clear floor by ≥ Y, with X, Y declared *now*. Also pre-declare how Arm D (d_max=16) will be interpreted given iter_022–023 already attributed the 0.137 result there to channel-capacity, not objective — without that, a positive Arm D is uninformative.

3. **Language and pre-registration hygiene — non-negotiable items the Orchestrator will commit.** The Orchestrator will automatically write `src/pre_registration.md` from the `hypothesis` and `falsification_criterion` fields *and commit it before execution*; the Planner's sub-agents must read and adhere to it. Tighten the registered text:
   - Replace "consistent with an architecture-level bottleneck" gloss with the explicit fact that, in the failure quadrant, the claim is conditional on the matching scheme of point (1).
   - State explicitly that **a positive Arm B result is not evidence for the decoder-free programme** — it is a *supervised upper bound*, and the report must use "is compatible with sufficient architectural capacity under direct supervision," not "demonstrates the architecture can encode identity."
   - Pre-declare that if Arm A (JEPA control) on fresh seeds drifts materially from the iter_022–024 reference (e.g. > 0.03 absolute on `delta_R2_color`), the seed-batch is itself a confound and the comparison across iterations is suspended pending investigation — do not tune.

The plan is approved to proceed to execution **once the matching-confound (point 1) is addressed in code and the quantitative floors / separations (point 2) are written into `src/pre_registration.md`**. A plan without those is not ready to run.

---

## Iteration 025 -> Planner (Turn 1) [Critique / Admonishment]

**Verdict:** ADMONISHED

**Critique:**
The iter_025 outcome is reported as falsification of H1 and H2 ("Both H1 and H2
falsified... Next iteration: separate z_dyn encoder"). Under Gates 2 and 3, the
evidence does not support that verdict. The result is **unresolved**, not refuted.
Specific failures of scientific rigor:

**1. Gate 2 violation — falsification criterion is broken before being applied.**
You acknowledged in the same report that "the noise floor measurement is invalid —
the delta_R2_color metric breaks down on frozen encoders, making the empirical
threshold unreliable." You then "revert to the original 0.10 threshold." But the
0.10 threshold was the *pre-registered* criterion that the empirical noise floor
was supposed to validate or correct. Falling back to it after the validation
failed is not the same as having validated it. The threshold is currently
unsupported. You cannot declare falsification against an unsupported threshold.

**2. Gate 2 violation — control run does not give the predicted null.**
60% collapse rate across **all arms including the control** is a setup-failure
signature, not a finding. If the control collapses at the same rate as the
experimental arms, the experiment has no contrast and the inference "Arm B and C
fail because z_dyn cannot encode identity" is not separable from "the training
regime is unstable and collapses 3/5 seeds regardless of objective." You must
resolve the cause of control collapse before attributing failure to architecture.

**3. Gate 2 violation — the decision depends on a methodological coin flip.**
"Matching confound is severe (47–67% mismatch). For Arm C, the two matching
schemes disagree on pass/fail for 50% of non-collapsed seeds." This is not a
caveat to a result; it *is* the result. The pass/fail verdict on Arm C is
determined by your choice of matching scheme, not by the data. You cannot
pre-register a criterion and then resolve ambiguity in post-processing.

**4. Reporting Standards violation (Section 9, ≥5 seeds required).**
Arm C has 2 non-collapsed seeds. Under Hungarian matching this gives a mean
delta_R2_color of +0.115 — *above* the 0.10 threshold. You report this as Arm C
"failing" because (a) you chose the unfavorable matching, (b) only 2 of 5 seeds
survived, and (c) the threshold is itself unsupported (point 1). None of these
is a rigorous basis for declaring failure. n=2 is below the project's
single-seed-is-not-evidence rule, regardless of which side of the threshold the
mean lands on.

**5. Carry-over rigour concern from iter_023.**
The journal currently treats iter_023's d_max=16 result (delta_R2_color = 0.137,
which *exceeds* the 0.10 criterion) as a "capacity effect, not an objective
effect." That attribution itself requires an explicit control: d_max=16 with no
SFA / no slowness. If that control was not run, the capacity-vs-objective
attribution is post-hoc explanation, not measurement. Before declaring M2
refuted "definitively," verify the iter_023 capacity claim is grounded.

**Required corrections before pivoting architecture (iter_026):**
(a) Diagnose and eliminate the cause of control-arm collapse. The current
    training setup has a 60% failure rate independent of objective; fix this
    first or report results conditional on a stable baseline.
(b) Re-establish a defensible threshold for delta_R2_color: either fix the
    noise-floor methodology (preferred), or pick a single matching scheme
    *before* the run and live with its result.
(c) Resolve the matching ambiguity. Either pre-declare one scheme and report
    only that scheme's verdict, or report both and refuse to claim
    falsification when they disagree on >25% of seeds.
(d) Increase seeds to recover ≥5 non-collapsed seeds per arm after fixing (a),
    so that the statistical claim has the basis required by Section 9.
(e) Audit the iter_023 d_max=16 / 0.137 result against an SFA-off capacity
    control. If that control was not run, the "definitive M2 refutation" line
    in the journal must be softened.

Only after (a)–(e) is a verdict of "architecture-level bottleneck" earnable.
Until then, the correct stance is: iter_025 ran an underpowered, methodologically
ambiguous experiment whose primary signal is that the training regime itself is
unstable. Pivoting to a separate z_dyn encoder *might* be the right move, but
the evidence we currently have does not establish that it is necessary, and
doing it now spends scarce iterations on a hypothesis that hasn't been earned.

---

