You are executing Phase 28 Sub-task 1 of the Thalamus collapse-elimination campaign.

## Context
Iter_027 found that Arm C (separate backbone, mask_dyn_sim=True, VICReg-only on z_dyn) achieved 0% collapse, but this confounded separate-backbone with mask_dyn_sim. The critical next isolate is: does mask_dyn_sim alone (on the shared backbone) prevent collapse? The Research Manager has given 3 required corrections to the proposed plan.

## Your Tasks

### Task 1: Update src/pre_registration.md with the following Manager-mandated corrections:

**Correction 1 — D0 re-label:** D0 is NOT a "weight-change anchor" because cov_weight=1 is already the canonical setting in iter_026/027 baselines. There is no weight delta to confound. Re-label D0 as "shared-backbone JEPA+VICReg baseline replication" and drop the cov_weight-confound rationale. State its purpose as: "in-iteration null reference for the independent readout comparison."

**Correction 2 — F4 relative threshold with mean_abs_corr:**
- Update F4 to use a RELATIVE threshold: "C1 passes the std gate (C_C1 ≤ 0.10) BUT (ΔR²_C1 < D0_ΔR²_color + 0.05 OR mean_abs_corr_C1 > mean_abs_corr_D0 + 0.05) → VICReg-maintained variance is constructional; z_dyn has variance but no meaningful semantic content improvement over the collapsing baseline."
- Update H2 similarly: "ΔR²_C1 ≥ D0_ΔR²_color + 0.05 AND mean_abs_corr_C1 ≤ mean_abs_corr_D0 + 0.05" (C1 outperforms D0 on independent readouts)
- Make clear that D0's ΔR² and mean_abs_corr are the in-iteration null references
- Note: lower mean_abs_corr = better decorrelation, so "C1 ≤ D0 + 0.05" means C1 is at least as well decorrelated (within 0.05 tolerance)

**Correction 3 — n=10 power caveat:** Add a section stating: "Fisher's exact test for 0/10 vs 3/10 gives p ≈ 0.21; the design cannot formally distinguish 0% from 10–20% at this sample size. Results are reported as point estimates with this limit explicitly noted."

Keep ALL other content from the existing pre_registration.md (hypothesis, arms, language constraints, constructional acknowledgment, etc.) but update the specific sections above.

### Task 2: Create src/run_phase0_mask_dyn_sim_shared.py

This is the experiment runner. It is based heavily on src/run_phase0_separate_dyn.py but simplified:
- Uses ONLY NonParametricJEPASpatial (shared backbone) — no separate-backbone model
- Implements mask_dyn_sim via loss adjustment AFTER forward(), exactly like models_separate_dyn.py does:
  ```
  loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(x_hist_t, x_target_t, ...)
  if mask_dyn_sim:
      loss_dict["loss"] = loss_dict["loss"] - sim_weight * loss_dict["sim_loss_dyn"]
  ```
  This is identical to what NonParametricJEPASpatialSeparateDyn does in its forward() override.

Four arms (10 seeds each, 40 total runs):

D0 — Baseline replication (shared backbone, mask_dyn_sim=False, weights 25/25/1):
  NonParametricJEPASpatial, dyn_readout="mean", d_max=8, d_t=3,
  pos_encoding="none", primary_objective="jepa", lr=3e-4, batch_size=64,
  buffer=4000, 8000 steps, gradient clipping max_norm=1.0,
  ccr_mode="covariance", ccr_smooth_weight=10, ccr_spatial_weight=10,
  gdasr_log_only=True, sim_weight=25, var_weight=25, cov_weight=1,
  seeds=[7, 17, 31, 53, 71, 83, 97, 113, 127, 149].

C1 — Primary arm (shared backbone, mask_dyn_sim=True, weights 25/25/1):
  Same as D0 but mask_dyn_sim=True in the runner. Same seeds.

C2 — Seed robustness (shared backbone, mask_dyn_sim=True, weights 25/25/1):
  Same as C1 but fresh seed bank: [101, 103, 107, 109, 131, 137, 139, 151, 157, 163].

C3 — Weight robustness (shared backbone, mask_dyn_sim=True, weights 27.5/27.5/1.1):
  Same as C1 but var_weight=27.5, cov_weight=1.1 (+10% perturbation).
  Original seeds. sim_weight stays at 25.

IMPORTANT implementation details:
1. The mask_dyn_sim adjustment must happen BEFORE backward(): compute total_loss from the adjusted loss_dict["loss"], then backward on total_loss
2. All evaluation functions (check_collapse, compute_vicreg_health, compute_semantic_probes, compute_centroid_mse, collect_multitraj_eval_data, evaluate_run, fit_linear_probe, etc.) should be copied from run_phase0_separate_dyn.py — they are identical since they only depend on the encoder output, not the model type
3. Output goes to archive/iter_028/results/ (runs/, checkpoints/, summary_iter_028.csv, final_analysis.md)
4. The analysis generation function must report per-arm collapse rates (dual, eval-only, train-only), per-seed train-vs-eval std gap table, gate checks for all arms, and the pre-registered outcome classification with language constraints
5. Include the D0 vs C1 comparison on independent readouts (ΔR², mean_abs_corr) in the analysis — this is the relative-threshold gate
6. Support --dry-run, --workers, --sequential flags as in the existing runner
7. Parameter count logging before runs

DO NOT modify src/models_dual_stream.py, src/models_separate_dyn.py, or src/environment.py.

## Success Criterion
Both files exist and are syntactically valid Python (for the .py file) and Markdown (for the .md file). The pre-registration incorporates all 3 Manager corrections.