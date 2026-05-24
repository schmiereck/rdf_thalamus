Please execute the core training and analysis for Phase 14:

1. Create `src/run_phase14_experiments.py` based on `src/run_phase13_experiments.py`.
   - Ensure it runs a matched 5-seed sweep (seeds: 42, 123, 456, 789, 999) on:
     - Arm G (Original RGB CLTS baseline) with `ccr_mode='none'`
     - Arm J (CCR-Hinge) with `ccr_mode='hinge'`
     - Arm K (CCR-Covariance) with `ccr_mode='covariance'`
   - Ensure both passive training on N=3 (steps 1..1500) and active training on N=4 (steps 1501..3000) use the appropriate `ccr_mode` for each arm.
   - For Arm J (hinge) and Arm K (covariance), ensure that the loss weights `ccr_smooth_weight` and `ccr_spatial_weight` are passed to `model()` forward calls as 10.0 (default).
   - Track evaluation metrics at checkpoints [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000] and the final step:
     - centroid decoding MSE (`mse_cent`),
     - test simulation loss (`test_sim_loss`),
     - soft spatial variance (`mean_var_3`),
     - pointer entropy,
     - coordinate velocity metrics (`std_vel_3` and `mean_abs_vel_3` of the novel object (channel 3) over the evaluation sequence).
   - Coordinate velocity is computed as:
     `vel_3 = x_mean_3[1:] - x_mean_3[:-1]`
     `std_vel_3 = np.std(vel_3)`
     `mean_abs_vel_3 = np.mean(np.abs(vel_3))`
     where `x_mean_3` is the soft centroid of channel 3 in pixel space [0, 127] over the 200 evaluation frames.

2. Run the experiments by executing `src/run_phase14_experiments.py`. Make sure it prints detailed progress logs.

3. In `src/run_phase14_experiments.py`, perform the statistical tests (Welch's t-test and Levene's test) on step 3000 offline test sim loss comparing Arm J and Arm K against Arm G.

4. Save the compiled summary of metrics to `archive/iter_14/results/summary_phase14.csv`, save audit logs to `archive/iter_14/results/audit_results_phase14.json`, and plot the offline test simulation loss adaptation trajectory over training steps, saving the plot to `archive/iter_14/results/adaptation_curves_phase14.png`.

5. Explicitly verify the pre-registered falsification criteria and the 'lazy encoder' failure mode (representational freezing when `std_vel_3 < 1.5` and centroid decoding MSE is high), providing a detailed scientific writeup of your findings.