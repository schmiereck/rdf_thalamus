# RDF Scientific Pre-Registration

*   **Iteration:** 028
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Removing the JEPA similarity loss gradient on z_dyn (mask_dyn_sim=True) from the
shared-backbone NonParametricJEPASpatial encoder does not destabilize the
VICReg-maintained per-dimension variance of z_dyn, AND z_dyn retains meaningful
identity encoding as measured by an independent (non-VICReg-enforced) readout.

Formally: Let C_C1 be the dual-criterion collapse rate of arm C1 (shared backbone,
mask_dyn_sim=True, weights 25/25/1) over 10 seeds. Let ΔR²_C1 be the mean
delta_R2_color of C1 across non-collapsed seeds. The hypothesis is:
(H1) C_C1 ≤ 0.10 (std-based collapse gate)
(H2) ΔR²_C1 ≥ D0_ΔR²_color + 0.05 AND mean_abs_corr_C1 ≤ mean_abs_corr_D0 + 0.05 (C1 outperforms D0 on independent readouts)

Both conditions must hold. This is the critical isolate that iter_027's Arm C
could not provide (it confounded separate-backbone with mask_dyn_sim).

Constructional acknowledgment: VICReg's variance hinge (γ - std, hinge at γ=1)
directly enforces per-dimension std ≥ 1, which trivially satisfies the collapse
gate (std < 0.5). Even a fully-confirmed C1+C2+C3 result will be reported as
"removing sim_loss_dyn from the z_dyn gradient path does not destabilize
VICReg-maintained variance" — NOT as "sim_loss_dyn causes collapse."

## 2. Falsification Criterion
Four independent falsification conditions, any one of which refutes the hypothesis
or downgrades its interpretation:

F1: C1 collapse rate ≥ 0.20 (dual criterion) → mask_dyn_sim alone is insufficient
    on the shared backbone; the separate-backbone architecture was load-bearing.
    The Arm C signal from iter_027 collapses to that confound.

F2: C2 collapse rate ≥ 0.20 (fresh seed bank) → the C1 result is seed-dependent,
    not general.

F3: C3 collapse rate ≥ 0.20 (±10% weight perturbation) → the C1 result is not
    robust to reasonable hyperparameter variation.

F4: C1 passes the std gate (C_C1 ≤ 0.10) BUT (ΔR²_C1 < D0_ΔR²_color + 0.05 OR mean_abs_corr_C1 > mean_abs_corr_D0 + 0.05) → VICReg-maintained variance is constructional; z_dyn has variance but no meaningful semantic content improvement over the collapsing baseline.

**Note:** Lower mean_abs_corr = better decorrelation, so "mean_abs_corr_C1 ≤ mean_abs_corr_D0 + 0.05" means C1 is at least as well decorrelated (within 0.05 tolerance) as the in-iteration null reference D0.

Additional guard: D0 (shared-backbone JEPA+VICReg baseline replication).
D0 serves as the in-iteration null reference for the independent readout comparison.
ΔR² and mean_abs_corr values from D0 are the baselines against which C1's relative
performance is assessed (F4 and H2 gates use the relative thresholds above).

## 3. Proposed Method
Step-by-step experimental protocol:

1. CREATE src/run_phase0_mask_dyn_sim_shared.py — the experiment runner.
   Based on run_phase0_separate_dyn.py, simplified to shared-backbone only.
   Implements mask_dyn_sim via loss adjustment after forward():
     adjusted_loss = loss_dict["loss"] - sim_weight * loss_dict["sim_loss_dyn"]
   This is identical to NonParametricJEPASpatialSeparateDyn's approach.
   No modification to models_dual_stream.py.

2. FOUR ARMS (10 seeds each, 40 total runs):

   D0 — Shared-backbone JEPA+VICReg baseline replication (shared backbone, mask_dyn_sim=False, weights 25/25/1):
     NonParametricJEPASpatial, dyn_readout="mean", d_max=8, d_t=3,
     pos_encoding="none", primary_objective="jepa", lr=3e-4, batch_size=64,
     buffer=4000, 8000 steps, gradient clipping max_norm=1.0,
     ccr_mode="covariance", ccr_smooth_weight=10, ccr_spatial_weight=10,
     gdasr_log_only=True, sim_weight=25, var_weight=25, cov_weight=1,
     seeds=[7, 17, 31, 53, 71, 83, 97, 113, 127, 149].
     Purpose: In-iteration null reference for the independent readout comparison.
     D0 is the canonical cov_weight=1 setting from iter_026/027 baselines; there is
     no weight delta to confound. Its ΔR² and mean_abs_corr serve as the in-iteration
     baselines for F4 and H2 relative-threshold gates.

   C1 — Primary arm (shared backbone, mask_dyn_sim=True, weights 25/25/1):
     Same as D0 but mask_dyn_sim=True. Same seeds.
     The critical isolate: does removing sim_loss_dyn prevent collapse
     without the separate-backbone confound?

   C2 — Seed robustness (shared backbone, mask_dyn_sim=True, weights 25/25/1):
     Same as C1 but fresh seed bank: [101, 103, 107, 109, 131, 137, 139,
     151, 157, 163]. Tests whether C1's result is seed-dependent.

   C3 — Weight robustness (shared backbone, mask_dyn_sim=True, weights 27.5/27.5/1.1):
     Same as C1 but var_weight=27.5, cov_weight=1.1 (+10% perturbation).
     Original seeds. sim_weight stays at 25. Tests sensitivity to weight variation.

3. EVALUATION at step 8000 (same protocol as iter_027):
   - Dual collapse criterion: collapsed_eval OR collapsed_train (per-dim std < 0.5)
   - Train-vs-eval std gap: report per-seed, co-equally with collapse rates
   - Hungarian-primary matching for semantic probes
   - Semantic probes: delta_R2_color (INDEPENDENT READOUT, relative gate vs D0),
     r2_dyn_color, r2_coord_color, r2_dyn_pos, r2_coord_pos,
     r2_dyn_identity, delta_r2_identity
   - VICReg health: per_dim_std, mean_abs_corr
   - Centroid MSE (reference only, NOT used for arm selection)
   - Training loss sanity: mean total loss at step 8000, loss > 50 → disqualified
   - Parameter count per arm (logged before runs start)

4. STOP RULE: All 40 runs complete. No early termination.

5. PRE-REGISTERED OUTCOME CLASSIFICATION:
   - If C1 ≤ 10% AND H2 passes (ΔR² relative threshold AND mean_abs_corr relative threshold): CONFIRMED — mask_dyn_sim on shared
     backbone does not destabilize VICReg-maintained variance AND preserves
     semantic encoding. Promotion to M2-style SFA (iter_029) is justified.
   - If C1 ≥ 20%: FALSIFIED — mask_dyn_sim alone insufficient; separate
     backbone was load-bearing.
   - If C1 ≤ 10% BUT F4 triggers (relative threshold on ΔR² and/or mean_abs_corr not met): DOWNGRADED — VICReg variance is
     constructional; no semantic content despite maintained variance.
   - If C2 ≥ 20%: SEED-DEPENDENT — C1 result does not generalize.
   - If C3 ≥ 20%: NOT ROBUST — C1 result sensitive to weight perturbation.

6. SAMPLE-SIZE CAVEAT:
   Fisher's exact test for 0/10 vs 3/10 gives p ≈ 0.21; the design cannot formally
   distinguish 0% from 10–20% at this sample size. Results are reported as point
   estimates with this limit explicitly noted.

7. LANGUAGE CONSTRAINTS (tracked failure mode from iter_027 overclaim):
   - Use "does not destabilize VICReg-maintained variance" or "is consistent
     with"; do NOT use "breakthrough", "causal driver", "eliminated", "BEST",
     "proves", "demonstrates", or "resolves."
   - Even a fully-confirmed C1+C2+C3 result is reported per the constructional
     framing in the hypothesis, not as "sim_loss_dyn causes collapse."

FILES TO CREATE:
- src/run_phase0_mask_dyn_sim_shared.py (NEW): experiment runner

FILES TO MODIFY:
- src/pre_registration.md (UPDATE with iter_028 plan)

FILES NOT TO MODIFY:
- src/models_dual_stream.py (mask_dyn_sim handled in runner, not model)
- src/models_separate_dyn.py (not needed for this experiment)
- src/environment.py

Total runs: 4 arms × 10 seeds = 40 runs × 8000 steps each.
Expected wall time: ~30-40 minutes with parallel workers (CPU).

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
