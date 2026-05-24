Please execute Phase 9: Implement and evaluate the Dynamic Surprise-Modulated Spatial Bottleneck Curriculum (DSMC) against the Gentle and Strong static bottlenecks, fully integrating the strategic adjustments requested by the Research Manager.

Specifically, perform the following steps:

1. **Pre-Registration Update**:
   First, update `src/pre_registration.md` to pre-register the updated hypothesis and the fifth falsification criterion (Temporal Prediction Safeguard) as outlined by the Research Manager.
   - Hypothesis point 4: The adaptive curriculum must not statistically degrade the final temporal prediction loss, achieving a final test L2/surprise loss ratio of < 1.15 compared to the static lambda = 0.01 baseline (Arm A).
   - Falsification Criterion 5: The final mean test temporal prediction loss (test L2/surprise loss) of the adaptive curriculum (Arm C) is statistically degraded (defined as a >15% increase, i.e., ratio >= 1.15) compared to the static lambda = 0.01 baseline (Arm A).

2. **Code Enhancement in `src/run_phase9_experiments.py`**:
   Refine the script to:
   - **Pre-Registered Formula**: Compute `lambda_target = 0.10 * max(0.0, 1.0 - ewma_surprise / 0.15)`.
   - **Controller Stability**: Guard against rapid oscillations by applying a step-to-step rate limit (clipping change to maximum +/-0.002 per step). For example:
     ```python
     if name == "Arm C":
         lambda_target = 0.10 * max(0.0, 1.0 - ewma_surprise / 0.15)
         if step == 1501:
             lambda_val = lambda_target
         else:
             lambda_val = lambda_val + np.clip(lambda_target - lambda_val, -0.002, 0.002)
     ```
   - **Temporal Prediction Safeguard**: In `evaluate_branch`, compute and return the test temporal prediction loss (`test_sim_loss`, i.e. the model's `sim_loss` evaluated on the fresh test sequence).
   - **Empirical Transparency**: Save `test_sim_loss` in the `results_list` and in `summary_phase9.csv`.
   - **Falsification Auditing**: Include the auditing of the fifth falsification criterion (the loss ratio Arm C / Arm A) in the text report output and console logs.

3. **Execution**:
   Run the 5-seed comparative sweep (seeds 42, 123, 456, 789, 999) across the three arms: Arm A (Gentle, lambda=0.01), Arm B (Strong, lambda=0.10), and Arm C (Experimental DSMC).
   Ensure that the results are saved in:
   - `archive/iter_009/results/summary_phase9.csv`
   - `archive/iter_009/results/dsmc_trajectories.png`
   - `archive/iter_009/results/performance_comparison_phase9.png`
   - `archive/iter_009/results/phase9_report.md`

Verify all 5 seeds successfully complete. Review the outputs and print the final aggregated averages.