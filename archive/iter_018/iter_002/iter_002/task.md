You are an AI executor sub-agent for iter_id 18.2.2. Your task is to implement the EG-MDL experiments, run the full sweeps across all 5 seeds [42, 123, 456, 789, 999], analyze the results, and generate the results directory.

### Core Implementation Guidelines:

1. Create a new script `src/run_phase18_experiments.py` based on `src/run_phase17_experiments.py`.
2. Add a disk-caching mechanism to `train_passive_cached(seed, device)` to avoid retraining the same seed passive model multiple times. Specifically, save it to `cache/passive_model_seed_{seed}.pt` and reuse it. This will save substantial execution time.
3. Modify the arms definition to match:
   ```python
   arms = [
       ("Arm P (WUP-MDL, W=100)",    100,  None, "mdl",    None),
       ("Arm S (EG-MDL, θ=0.90)",    100,  None, "eg_mdl", 0.90),
       ("Arm S_alt (EG-MDL, θ=0.85)",100,  None, "eg_mdl", 0.85),
   ]
   ```
4. Update `run_active_branch` signature and body to accept and handle `theta`.
5. Support `gating="eg_mdl"` alongside `"mdl"`. In both cases:
   - When probation starts at step 1800, set `probationary=True`, set `branch_model.d_t=4`, reset error buffer, and initialize `wup_errors = []`.
   - In each training step during WUP probation, calculate the per-dimension prediction error of the 4th dimension (index 3) on the training minibatch:
     ```python
     # Per-dimension prediction error for dimension 3
     e_coord_dim3 = F.mse_loss(z_pred_coord[:, 3], z_target_coord[:, 3])
     e_dyn_dim3 = F.mse_loss(z_pred_dyn[:, 3], z_target_dyn[:, 3])
     e_total_dim3 = (e_coord_dim3 + e_dyn_dim3).item()
     wup_errors.append(e_total_dim3)
     ```
   - At the end of probation, compute:
     ```python
     W = len(wup_errors)
     E_early = np.mean(wup_errors[:W//2]) if W > 0 else 0.0
     E_late = np.mean(wup_errors[W//2:]) if W > 0 else 0.0
     rho = E_late / max(E_early, 1e-8) if E_early > 0 else 1.0
     ```
   - Apply the gate:
     - For `gating="eg_mdl"`, accept if `ratio < 1.0` AND `rho < theta`.
     - For `gating="mdl"`, accept if `ratio < 1.0`.
   - Store both `ratio` (MDL ratio) and `rho` (prediction-trend ratio) in the final gating dict.
6. Record and return `"wup_errors": wup_errors` in `branch_results`.
7. For both Sweep 1 (N=3 -> 4 transition) and Sweep 2 (N=3 + Noisy-TV), collect `mdl_ratio` and `rho` per arm per seed.
8. Perform Welch's t-test comparing Arm P vs Arm S on `mse_cent` and `attention_switch_rate` (handling NaNs for unrecruited seeds properly).
9. Run the pre-registered falsification audit against the four criteria:
   - C1: Recruitment rate < 80% (falsified if recruited < 4/5 seeds)
   - C2: False recruitment rate > 20% on Noisy-TV control (falsified if false recruited > 1/5 seeds)
   - C3: Mean centroid decoding MSE > 65.0 on transition sweep (calculated across recruited seeds only or all seeds? Let's check both, but the primary pre-registered threshold is 65.0)
   - C4: theta-sensitivity: if Arm S passes (fails no C1, C2, C3) but Arm S_alt fails (fails at least one of C1, C2, C3), or vice versa.
10. Save all results under `archive/iter_018/results/`:
   - `summary_phase18.csv`
   - `audit_results_phase18.json`
   - `adaptation_curves_phase18.png`
   - Comprehensive markdown report `archive/iter_018/results/phase18_report.md`.
11. Run `python src/run_phase18_experiments.py` and print its complete execution output and verify that it is fully correct and does not raise any exceptions. Let's do this!