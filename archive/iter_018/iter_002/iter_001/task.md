You are an AI executor sub-agent for iter_id 18.2.1. Your task is to implement the EG-MDL experiments, run the full sweeps across all 5 seeds [42, 123, 456, 789, 999], analyze the results, and generate the results directory.

### Detailed Requirements:

1. Create a new script `src/run_phase18_experiments.py` based on `src/run_phase17_experiments.py`.
2. Implement the three experimental arms:
   - Arm P (WUP-MDL, W=100) — Baseline, theta=None, gating="mdl"
   - Arm S (EG-MDL, W=100, theta=0.90) — gating="eg_mdl", theta=0.90
   - Arm S_alt (EG-MDL, W=100, theta=0.85) — gating="eg_mdl", theta=0.85
3. Support `gating="eg_mdl"` in `run_active_branch()`.
4. During the probationary period (when `probationary=True` and `gating` is "eg_mdl" or "mdl"), at each training step:
   - Run the forward pass to get loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn)
   - Compute the per-dimension prediction error for dimension index 3 (the 4th dimension):
     ```python
     e_coord_dim3 = F.mse_loss(z_pred_coord[:, 3], z_target_coord[:, 3])
     e_dyn_dim3 = F.mse_loss(z_pred_dyn[:, 3], z_target_dyn[:, 3])
     e_total_dim3 = (e_coord_dim3 + e_dyn_dim3).item()
     ```
   - Store these values in a list `wup_errors` initialized to `[]` when probation starts.
5. At the end of WUP probation (step == probation_end_step):
   - Compute the categorizer consistency ratio (MDL ratio) using `categorizer_consistency_ratio(...)`
   - Compute the prediction-trend ratio:
     ```python
     W = len(wup_errors)
     E_early = np.mean(wup_errors[:W//2]) if W > 0 else 0.0
     E_late = np.mean(wup_errors[W//2:]) if W > 0 else 0.0
     rho = E_late / max(E_early, 1e-8) if E_early > 0 else 1.0
     ```
   - If `gating == "eg_mdl"`:
     - Accepted if `ratio < 1.0` AND `rho < theta`.
   - If `gating == "mdl"`:
     - Accepted if `ratio < 1.0`.
6. Run both the Transition Sweep (N=3 -> 4 clean objects) and the Control Sweep (Noisy-TV distractor) on seeds: `[42, 123, 456, 789, 999]`.
7. Collect the required metrics per seed per arm:
   - `recruitment_accepted` (bool): whether the dimension was permanently recruited
   - `mse_cent` (float): centroid decoding MSE for the 4th object/distractor
   - `test_sim_loss` (float): test simulation loss on N=4 (or N=3 for control)
   - `mdl_ratio` (float): MDL consistency ratio at gate evaluation
   - `rho` (float): prediction-trend ratio at gate evaluation
   - `wup_errors` (list): per-step prediction errors during WUP
   - `attention_switch_rate` (float): post-recruitment attention stability
   - `centroid_tracking_error` (float): post-recruitment centroid tracking accuracy
8. Implement Welch's t-test statistical comparison and the pre-registered falsification audit:
   - C1: Recruitment rate < 80% on transition sweep (falsified if recruited < 4/5 seeds)
   - C2: False recruitment rate > 20% on control sweep (falsified if false recruited > 1/5 seeds)
   - C3: Mean centroid decoding MSE > 65.0 on transition sweep
   - C4: theta-sensitivity: if Arm S passes (fails no C1, C2, C3) but Arm S_alt fails (fails at least one of C1, C2, C3), or vice versa, report as theta-sensitive.
9. Save results under `archive/iter_018/results/`:
   - `summary_phase18.csv`
   - `audit_results_phase18.json`
   - `adaptation_curves_phase18.png`
   - A detailed markdown report at `archive/iter_018/results/phase18_report.md`.
10. Make sure to run `python src/run_phase18_experiments.py` and verify that it executes successfully, collects all results, and compiles everything perfectly. Ensure no print statements throw errors, handle potential NaNs in stats gracefully, and make sure directories are created. Let's do this!