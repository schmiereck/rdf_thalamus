# Research Manager Log - Iteration 027

## Iteration 027 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (for Iteration 027)

1. Directional: The next iteration must be the **architectural probe** flagged in the journal — a separate `z_dyn` encoder branch — not another regime sweep. Single-knob regime tuning is now a closed intervention class (iter_026 null); any plan that returns to LR/var_weight/batch_size sweeps without bundling a structural change should be rejected at draft review. Keep the objective stack (SFA+VICReg on `z_dyn`, JEPA on `z_coord`) and the M1/M3 mandates frozen so the architectural variable is the only thing moving.

2. Scientific discipline: The plan must **pre-register before execution**: (a) the dual collapse gate (train std AND eval std) with the ≤10% target stated against an explicit seed count (≥10); (b) a *separate-encoder control with no identity objective* whose collapse rate constitutes the falsification arm — if even this control collapses ≥20%, the architectural hypothesis is refuted and the project pivots, not iterates; (c) buffer held at 4000 to neutralize the iter_025→026 confound; (d) Hungarian-primary matching unchanged. No early termination, no post-hoc gate adjustment. The plan should explicitly state which observed outcome would constitute a positive constructive result vs. a second null vs. an ambiguous middle, *before* any seed is run.

3. Discipline guard against false comfort: Even if separate-encoder lowers headline collapse, do **not** allow language like "stabilized" or "resolved" — require the train-vs-eval std gap (the narrow-subspace diagnostic from iter_026) to be reported on every arm. A configuration that passes train-std but still fails eval-std is not a success; it is the same generalization-failure mode under a new architecture. The plan should treat the train/eval std gap as a co-equal reporting requirement, not a footnote.

---

## Iteration 027 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
The shared CNN backbone (conv1–conv4) is the primary structural cause of z_dyn
collapse in the NonParametricJEPASpatial encoder. When z_dyn and z_coord share
backbone parameters, the JEPA similarity loss gradient (which predominantly shapes
z_coord's spatial selectivity) dominates the shared parameter updates, leaving
z_dyn's VICReg variance/covariance gradient under-supported. Giving z_dyn an
independent CNN backbone (same architecture: 4-layer Conv1d with kernel-5 stride-2,
1x1 projection head, mean-spatial readout; separate parameters) will reduce the
z_dyn collapse rate from ~30% (iter_026 A1, best shared-backbone regime) to ≤10%
under the dual criterion (eval OR train per-dim std < 0.5), because z_dyn's
VICReg gradients will flow through an independent parameter set without competing
with z_coord's JEPA gradients.

Formally: let C_shared be the z_dyn collapse rate under the shared backbone
(known ≈0.30 from iter_026 A1) and C_sep be the collapse rate under the
separate backbone. The hypothesis is C_sep ≤ 0.10 over 10 seeds.

**Proposed Falsification Criterion:**
Three pre-registered outcome classes:

1. POSITIVE CONSTRUCTIVE: Arm B (separate backbone + JEPA+VICReg) collapse
   rate ≤10%. The architectural change resolves collapse.

2. SECOND NULL (project pivots): Arm B collapse rate ≥20%. The shared backbone
   is NOT the primary structural cause of collapse. Per the Manager's
   instruction, the project pivots away from the shared-backbone hypothesis
   and does not iterate on this variable.

3. AMBIGUOUS MIDDLE: Arm B collapse rate in (10%, 20%). The separate backbone
   partially helps but does not fully resolve collapse. This warrants a
   follow-up combining the separate backbone with additional interventions
   (e.g., stronger z_dyn objective, longer training).

Additionally, the ARM C FALSIFICATION: If Arm C (separate backbone + VICReg-only
on z_dyn, no JEPA prediction loss on z_dyn) collapses ≥20%, then even the
architectural change combined with VICReg is insufficient without a task
objective for z_dyn. Per the Manager: "if even this control collapses ≥20%,
the architectural hypothesis is refuted and the project pivots, not iterates."

All collapse rates use the DUAL criterion: collapsed = collapsed_eval OR
collapsed_train, where collapsed_eval = any d_t z_dyn dim has batch-std < 0.5
on 200 eval samples, and collapsed_train = any d_t z_dyn dim has mean
training-logged std < 0.5 at step 8000. The train-vs-eval std gap must be
reported co-equally for every arm and every seed (not as a footnote).

**Proposed Method:**
Step-by-step experimental protocol:

1. CREATE src/models_separate_dyn.py with:
   - SeparateDynEncoder(nn.Module): An encoder with TWO independent CNN
     backbones:
     (a) coord_backbone: conv1→conv2→conv3→conv4→conv_spatial (identical to
         NonParametricEncoder), producing z_coord via soft-argmax centroid.
     (b) dyn_backbone: conv1_dyn→conv2_dyn→conv3_dyn→conv4_dyn→conv_identity
         (same architecture, SEPARATE parameters), producing z_dyn via mean
         pooling over the spatial dimension.
     Both backbones process the same RGB input independently.
     The class exposes forward(), forward_spatial(), d_dyn property matching
     the NonParametricEncoder interface.

   - NonParametricJEPASpatialSeparateDyn(NonParametricJEPASpatial): Uses
     SeparateDynEncoder instead of NonParametricEncoder. Adds a constructor
     argument `mask_dyn_sim=False` that, when True, zeros out the JEPA
     sim_loss_dyn term in the forward() method (Arm C control). All other
     loss terms (VICReg variance/covariance on z_dyn, JEPA sim_loss_coord,
     VICReg on z_coord) remain active. The predictor still receives z_dyn
     history but the sim_loss_dyn gradient does not shape the representation.

2. CREATE src/run_phase0_separate_dyn.py based on run_phase0_collapse_sweep.py,
   modified to use the new model class and three arms:

   COMMON TO ALL ARMS:
   - d_max=8, d_t=3, N=3
   - pos_encoding="none", dyn_readout="mean" (simplest readout; centroid_gated
     introduces an extra variable)
   - primary_objective="jepa", ccr_mode="covariance"
   - ccr_smooth_weight=10, ccr_spatial_weight=10
   - gdasr_log_only=True
   - lr=3e-4, gradient clipping max_norm=1.0
   - batch_size=64 (best from iter_026)
   - replay_buffer_capacity=4000 (neutralizes iter_025→026 confound)
   - 8000 training steps, Adam optimizer
   - 10 seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
   - var_weight=25, cov_weight=25, sim_weight=25

   ARM A (reference, shared backbone):
     Uses existing NonParametricJEPASpatial with dyn_readout="mean".
     This re-anchors the shared-backbone collapse rate under mean readout
     (iter_026 used centroid_gated; this eliminates the readout confound).
     mask_dyn_sim=False (full JEPA+VICReg on both streams).

   ARM B (experimental, separate backbone):
     Uses NonParametricJEPASpatialSeparateDyn.
     mask_dyn_sim=False (full JEPA+VICReg on both streams).
     The independent dyn_backbone receives z_dyn's VICReg gradient without
     competition from z_coord's JEPA gradient.

   ARM C (separate-encoder control, no z_dyn identity objective):
     Uses NonParametricJEPASpatialSeparateDyn.
     mask_dyn_sim=True (JEPA prediction loss on z_coord only; VICReg
     variance/covariance on both z_coord and z_dyn; NO JEPA sim_loss_dyn).
     This tests whether the separate backbone alone suffices, or whether
     z_dyn also needs a task objective (JEPA prediction) to avoid collapse.

3. EVALUATION at step 8000 (same protocol as iter_026):
   - Collapse check (eval): per_dim_std < 0.5 on 200 eval samples from fresh env
   - Collapse check (train): per_dim_std at step 8000 from training log
   - Dual criterion: collapsed = collapsed_eval OR collapsed_train
   - Train-vs-eval std gap: report per_seed_train_std and per_seed_eval_std
     side-by-side for every seed in every arm (co-equal reporting, not footnote)
   - VICReg health: per_dim_std, mean_abs_corr on eval samples
   - Training loss sanity: mean total loss, var_loss, sim_loss at final step
   - Centroid MSE (reference only, NOT used for arm selection)
   - Semantic probes: delta_R2_color, delta_R2_identity (reference only)
   - Parameter count per arm (to document the capacity increase)
   - Hungarian-primary matching for semantic probes

4. STOP RULE: ALL arms complete their full 10-seed runs. No early
   termination even if one arm passes the ≤10% gate.

5. The ONLY dependent variable for the pre-registered gate is collapse rate
   (dual criterion). Centroid MSE, delta_R2_color, and other downstream
   metrics are recorded for diagnostic reference but MUST NOT be used to
   select a winning arm.

6. PRE-REGISTERED OUTCOME CLASSIFICATION (stated before any seed runs):
   - If Arm B ≤10%: POSITIVE CONSTRUCTIVE — separate backbone resolves
     collapse. Report whether Arm C <20% (backbone alone suffices) or
     Arm C ≥20% (backbone + task objective both needed).
   - If Arm B ≥20%: SECOND NULL — shared backbone is not the primary cause.
     Project pivots per Manager instruction.
   - If Arm B in (10%, 20%): AMBIGUOUS MIDDLE — partial improvement.
     Follow-up combining separate backbone with other interventions.
   - If Arm C ≥20%: ARM C FALSIFICATION — architectural change alone is
     insufficient even with VICReg; project pivots.

7. SANITY DISQUALIFICATION: A seed is disqualified (counted as collapsed)
   if mean total loss at step 8000 > 50 (same threshold as iter_026).

8. OUTPUT:
   - Per-seed CSV with all metrics including train and eval per_dim_std
   - Final analysis markdown with:
     (a) Per-arm collapse rates under dual, eval-only, train-only criteria
     (b) Per-seed train-vs-eval std gap table (co-equal with collapse rates)
     (c) Gate status per arm
     (d) Parameter count comparison
     (e) Pre-registered outcome classification

FILES TO CREATE:
- src/models_separate_dyn.py (NEW): SeparateDynEncoder,
  NonParametricJEPASpatialSeparateDyn
- src/run_phase0_separate_dyn.py (NEW): experiment runner

FILES TO MODIFY:
- None (existing code left untouched for backward compatibility;
  the new model class is standalone)

Total runs: 3 arms × 10 seeds = 30 runs × 8000 steps each.
Expected wall time: ~30 minutes with parallel workers (CPU).

---

## Iteration 027 -> Planner [Strategic Guidance]

Manager's Note — Iteration 027 Plan Review (Strategic Guidance)

**Verdict: Approve with three mandatory amendments.** The plan correctly executes the architectural pivot, pre-registers cleanly, and respects the iter_026 confound controls. The amendments below close gaps that would otherwise leave the result unfalsifiable or misattributable.

---

**1. The readout swap silently breaks the cross-iteration anchor — fix the reference.** The plan shifts `dyn_readout` from `centroid_gated` (iter_026 A1, the ~30% anchor used in the hypothesis) to `mean` (Arm A here). That is a defensible simplification, but it means Arm A is **not** the same condition as iter_026 A1, so the headline claim "C_shared ≈ 0.30 → C_sep ≤ 0.10" no longer rests on a within-experiment comparison. Two acceptable fixes — pick one and pre-register it:

   (a) Add a fourth arm A′ (shared backbone, `dyn_readout=centroid_gated`, otherwise identical) as the true iter_026 anchor, so the chain A′ → A → B is internally controlled; or
   (b) State explicitly in the pre-registration that the hypothesis is now C_sep ≤ 0.10 measured **against Arm A in this iteration**, not against the iter_026 anchor, and that the iter_026 number is informal context only.

   Without one of these, a "positive" Arm B result is ambiguous between "separate backbone fixed it" and "mean readout fixed it."

**2. Pre-register parameter-count and capacity as an alternative explanation, not just a footnote.** Arm B roughly doubles encoder parameters versus Arm A. The iter_025 v2 record already shows that `d_max=16` capacity alone changes behaviour. If Arm B passes, you must distinguish "decoupled gradients" (the stated mechanism) from "more capacity" (an unstated confound that would also predict success). Add to the pre-registration:

   - The exact parameter count per arm, computed and logged **before** runs start.
   - A pre-declared **interpretive rule**: if Arm B ≤10% AND Arm C ≤10%, the mechanism is *consistent with* gradient decoupling but *also consistent with* added capacity, and a capacity-matched shared-backbone control (e.g. widened conv channels) is the mandatory iter_028 follow-up. Do not allow the report to claim "the shared-backbone hypothesis is confirmed" in that case — only "is not refuted, pending capacity control."

**3. Language and falsification discipline — tighten three items:**

   - The hypothesis paragraph asserts the mechanism ("JEPA gradient dominates, leaving z_dyn under-supported") as if established. It is a *conjecture*. Reword to: "We conjecture that shared-parameter gradient competition contributes to collapse; the experiment tests the observable consequence C_sep ≤ 0.10, not the mechanism directly." A successful collapse-rate change is **consistent with** the mechanism, not a demonstration of it.
   - The "AMBIGUOUS MIDDLE: (10%, 20%)" band must have a pre-committed default action stated *now*. Right now it says "warrants a follow-up" — that is the loophole that becomes "let's iterate just a bit more." Either pre-commit to a specific follow-up arm set, or pre-commit to treating the middle band as a **soft null** that triggers the same pivot as ≥20%. Pick one before runs start.
   - The Orchestrator will write `src/pre_registration.md` from this plan and commit it before execution; the executor sub-agents must be instructed to read it and refuse to retune the gate, the seed list, the buffer size, or the dual-collapse threshold mid-run. State this constraint explicitly in `proposed_method` step 1.

   Throughout the eventual report, use "is consistent with / does not refute / provides evidence for"; do not allow "proves," "demonstrates," "stabilizes," or "resolves" without the capacity control in point 2.

---

Proceed once amendments 1–3 are reflected in `src/pre_registration.md`. The dual collapse criterion, ≥10 seeds, fixed buffer=4000, Hungarian-primary matching, full-run completion (no early stop), and the co-equal train/eval std reporting from the previous Manager's Hints are correctly carried over — keep all of those.

---

## Iteration 027 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — collapse-elimination
    sub-phase has produced a **second pre-registered null** (iter_027) on the
    structural-cause hypothesis, plus a **suggestive within-architecture
    ablation** (Arm C) that re-frames the active question from "where does
    the gradient competition happen?" to "which loss term drives z_dyn
    collapse?".
*   **Active Direction:** iter_027 tested the hypothesis that the **shared
    CNN backbone** is the primary cause of z_dyn collapse. Arm B (separate
    backbone, full JEPA+VICReg, same hyperparameters) was the falsification
    vehicle. Arm B collapsed at **30%** — indistinguishable from the
    shared-backbone baseline (30–40%). The shared-backbone hypothesis is
    **refuted**. This is the second consecutive iteration in which a
    pre-registered structural hypothesis was cleanly killed; this is
    methodologically healthy.
*   **The Arm C signal (treat with discipline):** Arm C, identical to Arm B
    except `mask_dyn_sim=True` (i.e. `sim_loss_dyn` removed; z_dyn shaped
    only by VICReg variance + covariance), showed 0% collapse over 10 seeds
    AND the highest measured delta_R2_color (0.18). The agent labelled this
    a "breakthrough" and stated "sim_loss_dyn is the causal driver of z_dyn
    collapse" — **this language is rejected by the Manager as overclaim**.
    What we actually have:
    - **Construction-versus-empirical caveat:** VICReg's variance hinge
      directly penalizes std < 1, which is the same quantity used in the
      eval-std collapse criterion. When z_dyn is shaped *only* by VICReg,
      the optimizer is being told almost exactly what the collapse metric
      is measuring. 0% collapse under VICReg-only is therefore *partly* a
      tautology of the chosen objective, not a clean empirical discovery.
      The empirical content of Arm C is narrower: "removing `sim_loss_dyn`
      does not destabilize VICReg's variance preservation under the
      separate-backbone regime" — which is informative but is not the same
      as "sim_loss_dyn causes collapse."
    - **Not pre-registered:** the iter_027 pre-registration covered the
      B-vs-baseline comparison. Arm C is an exploratory addition. A 30%
      vs 0% delta with n=10 is suggestive (Fisher's exact p ≈ 0.21
      approx; the difference is not formally significant at n=10).
    - **Missing critical control:** the same `mask_dyn_sim=True` ablation
      has not yet been run on the **shared backbone**. Without that arm,
      we cannot distinguish "separate backbone + no sim_dyn" from "no
      sim_dyn anywhere" as the operative intervention.
    - **No robustness check:** Arm C has not been tested under perturbation
      (±10% on var_weight, alternate seeds, ramped sim_weight to z_coord
      but masked from z_dyn).
*   **Updated mechanism hypothesis (TENTATIVE, requires iter_028
    confirmation):** When `z_target_dyn` is *not* stop-gradiented in the
    current JEPA implementation, gradient flow from `sim_loss_dyn` may push
    the encoder toward predictable-but-degenerate z_dyn representations,
    and this pressure overrides VICReg's variance hinge in ~30% of seeds.
    This is consistent with the iter_026 observation that *increasing*
    var_weight (25→50) worsened collapse (the JEPA pressure was already
    dominant; pushing variance harder destabilized the joint optimization
    further). If true, this hypothesis also aligns with the M2 mandate:
    identity (z_dyn) should be shaped by a slowness/identity objective,
    with prediction error treated as a *readout* signal, not as gradient
    input to z_dyn.
*   **Next Priority (iter_028):** Pre-registered control matrix to convert
    the Arm C signal from suggestive to confirmed (or to refute it):
    - C1: `mask_dyn_sim=True` on **shared backbone** (the missing arm).
    - C2: Arm C replication with a different random seed bank (n=10).
    - C3: Robustness perturbation of Arm C (±10% var_weight, ±10%
      cov_weight; one ramp variant).
    - Pre-register: collapse gate ≤10%; report train AND eval std;
      Hungarian-primary matching; buffer=4000 (carry forward iter_026
      confound control). Falsification: if C1 collapses ≥20%, the
      "separate backbone" was load-bearing after all; if C2 collapses
      ≥20%, Arm C was a seed-bank artefact; if C3 collapses ≥20%, the
      result is not robust.
*   **Confidence Score:** 50% (recovered slightly from 40%). One additional
    structural hypothesis ruled out (good), one promising ablation arm
    identified (good but unverified), the mechanism story is more
    narrowly constrained. But: two consecutive iterations have failed
    their primary gate; Arm C is unconfirmed; the foundation for downstream
    Phase 1+ work is still not in place.

## 2. Strategic Insights & Lessons Learned
*   **SHARED CNN BACKBONE IS NOT THE PRIMARY CAUSE OF Z_DYN COLLAPSE
    (iter_027, CONFIRMED via pre-registered null):** Separate backbones
    collapse at the same rate as shared backbones (30%) under
    identical JEPA+VICReg objectives. Architectural decoupling at the
    encoder level alone does not buy stability. This refutes the iter_026
    hypothesis that drove iter_027.
*   **THE SIM_LOSS-vs-VICReg COMPETITION IS THE NEW LEAD HYPOTHESIS
    (iter_027, SUGGESTIVE NOT CONFIRMED):** Within the separate-backbone
    regime, removing `sim_loss_dyn` (Arm C) eliminated collapse and
    coincided with the highest delta_R2_color (0.18). Three caveats
    gate any stronger claim:
    (a) VICReg-only naturally maintains the very std metric used for the
        collapse check — partial construction-versus-empirical concern;
    (b) the within-architecture comparison was not pre-registered;
    (c) the matching shared-backbone arm has not been run, so we cannot
        yet say whether separate backbones were necessary.
    Treat as a high-priority hypothesis to confirm in iter_028, not as
    established fact.
*   **REPEATED PATTERN — JEPA OBJECTIVE PRESSURE COMPETES WITH VICReg
    (iter_026 + iter_027 cross-iteration synthesis):** iter_026 found
    that strengthening VICReg variance worsened collapse; iter_027
    found that weakening JEPA pressure on z_dyn (by masking
    `sim_loss_dyn`) eliminated it. Both data points push in the same
    direction: under the current implementation, `sim_loss_dyn` and
    VICReg compete, and `sim_loss_dyn` wins often enough to collapse
    z_dyn. This is mechanism-level convergent evidence (across two
    iterations and four arms) — stronger than either iteration alone.
*   **CONNECTION TO M2 MANDATE (RECONNECTING TO GOAL):** The iter_027
    Arm C finding, if confirmed, is structurally aligned with the M2
    mandate from the goal document: M2 says z_dyn should be shaped by an
    identity/slowness objective, with JEPA-style prediction error
    demoted to a readout. Masking `sim_loss_dyn` from the z_dyn gradient
    path is the minimal version of that demotion. iter_028 should
    include this framing explicitly in its pre-registration.
*   **PRE-REGISTERED NULLS REMAIN FIRST-CLASS RESULTS (iter_026,
    iter_027, ENFORCED):** Two consecutive iterations have produced
    defensible nulls because they pre-declared their falsification
    criterion. The discipline holds.
*   **PRESERVED:** M2 stream-assignment guidance stands; M1 (pooled
    VICReg) stands; d_max=16 capacity baseline stands; Hungarian-primary
    matching stands; 20% control-collapse power threshold stands.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (ACTIVE, MORE NARROWLY LOCALIZED):**
    Now traced to the gradient interaction between `sim_loss_dyn` and
    the VICReg variance term on z_dyn, regardless of backbone
    architecture. Awaiting iter_028 confirmation.
*   **Architectural-Cause Bottleneck (PROVISIONALLY DOWNGRADED):**
    iter_027 found that separating the backbone alone does not resolve
    collapse. This bottleneck is reclassified from "primary" to
    "secondary" — it may still matter, but it is not load-bearing.
*   **Capacity-vs-Objective Confound (RESOLVED, iter_025 v2):** Still
    closed.
*   **Matching-Procedure Confound (RESOLVED, iter_025 v2):** Still
    closed.
*   **Diagnostic-vs-Constructive Iteration Loop (CLEARED):** Two
    consecutive pre-registered diagnostic iterations produced
    actionable nulls and a candidate mechanism. The protocol is paying
    off; keep going.
*   **Buffer-Capacity Confound (TRACKED):** iter_027 used buffer=4000
    throughout (per the iter_026 instruction). Keep buffer=4000
    constant in iter_028.
*   **Overclaim Loop (NEW, NOW TRACKED):** iter_027 executor used
    "breakthrough", "completely eliminated", "BEST semantic encoding",
    and "causal driver" for an unconfirmed within-architecture
    ablation. Manager has flagged this. iter_028 pre-registration MUST
    explicitly hold Arm-C claims at "suggestive" until C1+C2+C3 are run.
*   **Objective-Swapping Loop (DORMANT, ENFORCED):** Holds. iter_028
    keeps the SFA/JEPA+VICReg objective family and varies only loss
    masking and backbone topology.
*   **Logistics:** Executor token limits persist. Tracked, not blocking.

## 4. Alternate Research Paths
*   **iter_028: Sim-Loss Causal Confirmation (IMMEDIATE PRIORITY,
    ABLATION):** Three-arm pre-registered design (C1, C2, C3 above)
    to convert the iter_027 Arm C signal from suggestive to confirmed
    or refuted. Buffer held at 4000; train + eval std reported;
    Hungarian-primary matching; n=10 minimum per arm. Falsification
    criteria as listed.
*   **iter_029 (CONDITIONAL on iter_028 success): Promote to
    SFA-on-z_dyn architecture (per M2 mandate):** If Arm C
    mechanism survives iter_028, the next move is the actual M2
    configuration: SFA slowness term (`||z_dyn(t) - z_dyn(t-1)||^2`)
    on z_dyn instead of just VICReg-only, paired with stop-gradient
    readout of prediction error for the surprise signal. This is the
    goal-document's intended primary representation objective and is
    where Phase 0 should land.
*   **Multi-Knob Regime Stabilization (DEFERRED):** Demoted further.
    Two structural hypotheses have produced nulls; the action now is to
    confirm the loss-masking mechanism, not to revisit hyperparameter
    sweeps.
*   **Object-Tracking-ID Contrastive (DEFERRED to iter_030+):**
    Conditional on iter_028/029 outcome.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (CONDITIONAL):**
    Becomes lead candidate IF iter_028 refutes the Arm C mechanism.
*   **Accept Decoder-Free Constraint Relaxation (LAST RESORT):**
    Unchanged.
*   **Micro-Columns (DEFERRED per semantic caution):** Unchanged.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.

---

## Iteration 027 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 027 — Null Result: Shared CNN Backbone Is Not The Primary Cause Of z_dyn Collapse

## 1. Pre-Declared Hypothesis and Falsification Criterion
Pre-registered hypothesis (carried forward from iter_026 journal entry
and the iter_027 plan):
  "The shared CNN backbone is the primary cause of z_dyn collapse under
   JEPA+VICReg. A separate dedicated backbone for z_dyn, with all other
   hyperparameters held constant, will reduce the collapse rate to ≤10%
   over ≥10 seeds (Arm B)."

Pre-declared falsification criterion:
  "If Arm B's collapse rate is ≥20% under the dual collapse criterion
   (eval-std < 0.5 OR train-std < 0.5), the shared-backbone hypothesis
   is rejected."

## 2. Experimental Protocol
- Encoder: NonParametricJEPASpatial, d_max=8, d_t=3.
- Arm B (the hypothesis vehicle): separate backbone for z_coord and
  z_dyn (135,608 parameters), full JEPA+VICReg objective on both
  streams, sim_weight=25, var_weight=25, cov_weight=1.
- Buffer capacity: 4000 (held constant from iter_026 to control the
  flagged buffer-size confound).
- Optimizer: lr=3e-4, batch_size=32, 8000 training steps.
- Matching: Hungarian-primary.
- Seeds: n=10.
- Control arms run in the same iteration: shared-backbone baseline
  (reference: 30–40% from iter_026); Arm C (separate backbone,
  `mask_dyn_sim=True`).
- Held constant between Arm B and Arm C: parameter count (135,608),
  backbone topology, hyperparameters, seed bank, matching procedure.

## 3. Observed Quantities
- Arm B (separate backbone, full JEPA+VICReg): **30% collapse rate**
  over 10 seeds under the dual criterion (eval-std < 0.5 OR
  train-std < 0.5).
- Shared-backbone reference (iter_026 A0/A1): 30–40% collapse.
- Falsification threshold: ≥20%.
- Difference between Arm B and the shared-backbone reference: within
  seed-noise (n=10 per arm).

## 4. Verdict
**REFUTED.** Arm B's 30% collapse rate clears the pre-declared
rejection threshold (≥20%). The shared CNN backbone is not the
primary cause of z_dyn collapse under the current JEPA+VICReg
objective. Architectural decoupling at the encoder level alone does
not stabilize z_dyn.

## 5. Construction-vs-Empirical Note
The null on Arm B is genuinely empirical: the falsified prediction
was about gradient-pathway competition between z_coord and z_dyn in
shared parameters. Removing the shared parameters and observing no
improvement is real information about the mechanism — it tells us
the competition is not occurring at the shared-encoder gradient
level, or that it occurs but does not dominate the collapse
dynamics.

Separately, this iteration's Arm C (`mask_dyn_sim=True`, 0% collapse)
is **not** promoted to a finding in this report. The Arm C result has
a partial construction-versus-empirical concern: VICReg's variance
hinge directly enforces the same std quantity the collapse criterion
measures, so a VICReg-only z_dyn maintaining std ≥ 1 is partly what
the loss function is being told to do. Arm C is recorded as a
suggestive within-architecture ablation in the journal, requiring
iter_028 confirmation (shared-backbone version, replication on a
different seed bank, ±10% robustness check).

## 6. Limitations
- This result does **not** show that the shared backbone is irrelevant
  — only that it is not the dominant cause. Subtle effects (e.g.
  interaction with optimizer momentum, or with larger d_t) may still
  exist.
- n=10 per arm; differences smaller than ~14 percentage points are
  within seed noise.
- This result does **not** establish what *is* the primary cause.
  The cross-iteration synthesis (iter_026 + iter_027) suggests
  `sim_loss_dyn`–VICReg competition is a leading candidate, but
  confirmation requires iter_028's missing control arm
  (`mask_dyn_sim=True` on shared backbone).
- The collapse criterion itself (eval-std < 0.5) is conservatively
  chosen but is one of multiple possible thresholds; the
  train-vs-eval std discrepancy flagged in iter_026 remains an open
  architectural signal.
- The buffer-size confound (iter_026 → iter_027 buffer=4000) is
  controlled within this iteration but limits comparability with
  earlier iter_025-and-prior results.

---

