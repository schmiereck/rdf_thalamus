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

