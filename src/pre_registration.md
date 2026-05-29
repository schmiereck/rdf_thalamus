# RDF Scientific Pre-Registration

*   **Iteration:** 027
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis

We conjecture that shared-parameter gradient competition contributes to collapse;
the experiment tests the observable consequence C_sep ≤ 0.10, not the mechanism
directly. A successful collapse-rate change is **consistent with** the mechanism,
not a demonstration of it.

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

**Amendment 1 — Readout Swap Anchor Fix:** The plan shifts `dyn_readout` from
`centroid_gated` (iter_026 A1, the ~30% anchor) to `mean` (Arm A). A FOURTH
arm — Arm A′ (shared backbone, `dyn_readout="centroid_gated"`, otherwise
identical to A) — is added as the true iter_026 anchor. This creates the
controlled chain: A′ → A → B. C_sep ≤ 0.10 is measured **against Arm A in this
iteration**, not against the iter_026 anchor, and the iter_026 number is
informal context only.

**Amendment 2 — Parameter Count as Alternative Explanation:** The exact
parameter count per arm is computed and logged BEFORE runs start. The
pre-registered interpretive rule is: if Arm B ≤10% AND Arm C ≤10%, the result
is *consistent with* gradient decoupling but *also consistent with* added
capacity. The report must NOT claim "the shared-backbone hypothesis is confirmed"
in that case — only "is not refuted, pending capacity control." A capacity-matched
shared-backbone control (e.g. widened conv channels) would be the mandatory
iter_028 follow-up.

## 2. Falsification Criterion

Three pre-registered outcome classes:

1. POSITIVE CONSTRUCTIVE: Arm B (separate backbone + JEPA+VICReg) collapse
   rate ≤10%. The architectural change is consistent with reducing collapse.

2. SECOND NULL (project pivots): Arm B collapse rate ≥20%. The shared backbone
   is NOT the primary structural cause of collapse. Per the Manager's
   instruction, the project pivots away from the shared-backbone hypothesis
   and does not iterate on this variable.

3. AMBIGUOUS MIDDLE (10%, 20%): The separate backbone partially helps but does
   not fully resolve collapse. Per the pre-committed default action, this is
   treated as a **soft null** that triggers the same pivot as ≥20%. The project
   does not iterate on this variable in the middle band.

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

## 3. Proposed Method

Step-by-step experimental protocol:

1. CREATE src/models_separate_dyn.py with:
   - SeparateDynEncoder(nn.Module): An encoder with TWO independent CNN
     backbones:
     (a) coord_backbone: conv1→conv2→conv3→conv4→conv_spatial (identical to
         NonParametricEncoder), producing z_coord via soft-argmax centroid.
     (b) dyn_backbone: conv1_dyn→conv2_dyn→conv3_dyn→conv4_dyn→conv_identity_dyn
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
   modified to use the new model class and four arms:

   COMMON TO ALL ARMS:
   - d_max=8, d_t=3, N=3
   - pos_encoding="none"
   - primary_objective="jepa", ccr_mode="covariance"
   - ccr_smooth_weight=10, ccr_spatial_weight=10
   - gdasr_log_only=True
   - lr=3e-4, gradient clipping max_norm=1.0
   - batch_size=64 (best from iter_026)
   - replay_buffer_capacity=4000
   - 8000 training steps, Adam optimizer
   - 10 seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
   - var_weight=25, cov_weight=25, sim_weight=25

   ARM A′ (iter_026 anchor, shared backbone, centroid_gated):
     Uses existing NonParametricJEPASpatial with dyn_readout="centroid_gated".
     This exactly matches iter_026 A1 configuration.
     mask_dyn_sim is not applicable (shared backbone model).

   ARM A (reference, shared backbone, mean readout):
     Uses existing NonParametricJEPASpatial with dyn_readout="mean".
     This re-anchors the shared-backbone collapse rate under mean readout.
     mask_dyn_sim=False (full JEPA+VICReg on both streams).

   ARM B (experimental, separate backbone + JEPA+VICReg):
     Uses NonParametricJEPASpatialSeparateDyn.
     mask_dyn_sim=False (full JEPA+VICReg on both streams).
     The independent dyn_backbone receives z_dyn's VICReg gradient without
     competition from z_coord's JEPA gradient.

   ARM C (separate-encoder control, no z_dyn task objective):
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
   - Hungarian-primary matching for semantic probes
   - Sanity disqualification: final_train_loss > 50 → counted as collapsed
   - Parameter count per arm (logged BEFORE runs start)

4. STOP RULE: ALL arms complete their full 10-seed runs. No early
   termination even if one arm passes the ≤10% gate.

5. The ONLY dependent variable for the pre-registered gate is collapse rate
   (dual criterion). Centroid MSE, delta_R2_color, and other downstream
   metrics are recorded for diagnostic reference but MUST NOT be used to
   select a winning arm.

6. PRE-REGISTERED OUTCOME CLASSIFICATION (stated before any seed runs):
   - If Arm B ≤10%: POSITIVE CONSTRUCTIVE — separate backbone is consistent
     with reducing collapse. Report whether Arm C <20% (backbone alone
     suffices) or Arm C ≥20% (backbone + task objective both needed).
   - If Arm B ≥20%: SECOND NULL — shared backbone is not the primary cause.
     Project pivots per Manager instruction.
   - If Arm B in (10%, 20%): SOFT NULL — partial improvement. Same pivot as
     ≥20% per pre-committed default action.
   - If Arm C ≥20%: ARM C FALSIFICATION — architectural change alone is
     insufficient even with VICReg; project pivots.

7. SANITY DISQUALIFICATION: A seed is disqualified (counted as collapsed)
   if mean total loss at step 8000 > 50 (same threshold as iter_026).

8. LANGUAGE AND FALSIFICATION DISCIPLINE:
   - Throughout the report, use "is consistent with / does not refute /
     provides evidence for"; do NOT use "proves," "demonstrates,"
     "stabilizes," or "resolves" without the capacity control.
   - Sub-agents must read the pre-registration and refuse to retune the
     gate threshold, seed list, buffer size, or dual-collapse threshold
     mid-run.

9. OUTPUT:
   - Per-seed CSV with all metrics including train and eval per_dim_std
   - Final analysis markdown with:
     (a) Per-arm collapse rates under dual, eval-only, train-only criteria
     (b) Per-seed train-vs-eval std gap table (co-equal with collapse rates)
     (c) Gate status per arm
     (d) Parameter count comparison
     (e) Pre-registered outcome classification
     (f) Readout effect report (A′ vs A)

FILES TO CREATE:
- src/models_separate_dyn.py (NEW): SeparateDynEncoder,
  NonParametricJEPASpatialSeparateDyn
- src/run_phase0_separate_dyn.py (NEW): experiment runner

FILES TO MODIFY:
- src/pre_registration.md (UPDATE with amendments)

FILES NOT TO MODIFY:
- src/models_dual_stream.py (leave untouched for backward compatibility)
- src/environment.py
- All other existing files

Total runs: 4 arms × 10 seeds = 40 runs × 8000 steps each.
Expected wall time: ~30-40 minutes with parallel workers (CPU).

---
*Updated with Manager Amendments 1, 2, and 3 prior to iteration execution.*
