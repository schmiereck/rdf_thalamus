You are a high-fidelity research executor agent. Your goal is to implement and evaluate Phase 17 experiments: Encoder-only Smoothness-Uniqueness Gating (ESUG) vs WUP-MDL baseline.

Please follow these steps exactly:

Step 1: Modify `src/models_dual_stream.py`
Add `d_t_predict` (defaulting to None) to `NonParametricJEPASpatial.forward()`.
Inside `forward()`:
- Set `dt_pred = d_t_predict if d_t_predict is not None else self.d_t`.
- Pass `dt_pred` (instead of `self.d_t`) to `self.predictor`.
- Compute coordinate/dynamics prediction: `z_pred_coord[:, :dt_pred]`, `z_pred_dyn[:, :dt_pred]`.
- Compute target coordinates/dynamics for prediction matching: `z_target_coord[:, :dt_pred]`, `z_target_dyn[:, :dt_pred]`.
- Compute `sim_loss_coord` and `sim_loss_dyn` only over `dt_pred` dimensions.
- Keep the variance and covariance losses computed over all `self.d_t` dimensions of the encoder and predictor.

Step 2: Implement `src/run_phase17_experiments.py`
This script must perform a matched comparative study across 5 random seeds (42, 123, 456, 789, 999):
Compare 3 arms:
- Arm P (WUP-MDL, W=100)
- Arm Q (ESUG-100)
- Arm Q_fast (ESUG-30)

For each seed, execute TWO sweeps:
1. Transition Sweep (N=3 to N=4 clean objects):
   - Steps 1..1500: Passive pre-training on N=3 clean objects (standard variant, cached per seed).
   - At step 1500: Transition to N=4 clean objects.
   - At step 1800: Propose 4th dimension.
   - For Arm P: WUP starts at 1800, ends at 1900. Gating criteria: MDL ratio < 1.0.
   - For Arm Q: Evaluation starts at 1800, ends at 1900. Gating: R_unique > 0.15 and lambda < 0.5.
     During evaluation, keep the predictor 4th-dim untrained by passing `d_t_predict=3` to forward().
     Collect the consecutive sequence of target latent coordinates `zt_coord` over the B=100 steps to compute ESUG.
   - For Arm Q_fast: Evaluation starts at 1800, ends at 1830. Gating: R_unique > 0.15 and lambda < 0.5.
     During evaluation, keep the predictor 4th-dim untrained by passing `d_t_predict=3`.
     Collect consecutive `zt_coord` over B=30 steps to compute ESUG.
   - If accepted: Set d_t = 4, and train both encoder and predictor fully.
   - If rejected: Set d_t = 3, retry 50 steps later.
   - At step 3000: Run evaluation on N=4 test set, reporting centroid decoding MSE.

2. Control Sweep (Noisy-TV Distractor Control):
   - At step 1500: Transition to N=3 clean objects + 1 Noisy-TV distractor (`noisy_tv=True`).
   - At step 1800: Propose 4th dimension.
   - Evaluate false recruitment rate (fraction of seeds incorrectly accepted).

Step 3: Post-Recruitment Stability Auditing
For each seed where recruitment succeeds in the Transition Sweep:
- Monitor and log the Attention Token and CLTS motor control immediately following recruitment (for 100 steps).
- Calculate:
  - Attention Switch Rate: Fraction of steps where the attention token shifts channels over 100 steps post-recruitment.
  - Centroid Tracking Error: Average distance between pointer position and the centroid of the attended channel over the 100 steps post-recruitment.
Compare these metrics between Arm P (predictor trained/warm) and Arm Q/Q_fast (predictor completely untrained/cold) to evaluate the "representation-prediction temporal mismatch pathology."

Step 4: Save Results
- Save learning curves plot to `archive/iter_017/results/adaptation_curves_phase17.png`
- Save CSV summary of results to `archive/iter_017/results/summary_phase17.csv`
- Save JSON audit results to `archive/iter_017/results/audit_results_phase17.json`
- Generate scientific markdown report to `archive/iter_017/results/phase17_report.md`

Step 5: Run the entire script and verify it executes without errors. Let me know the results.