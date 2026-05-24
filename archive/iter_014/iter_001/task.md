Please execute the Phase 14 research plan. Follow these sequential steps:
1. Read the pre-registration file `src/pre_registration.md`.
2. Update `src/pre_registration.md` to:
   - Tighten the falsification limit for the centroid decoding MSE of the novel object under active CLTS control to >= 70.0 (matching the hypothesis that CCR reduces it to < 70.0).
   - Document the physical/geometric grounding for the hinge margin epsilon = 0.15 (corresponding to 19.2 pixels in a 128-pixel canvas, which aligns with the maximum contact distance of two contacting objects of maximum radius r=8, i.e., 8+8=16 pixels, plus a small spatial buffer).
   - Define and document the tracking of the 'lazy encoder' failure mode using the coordinate velocity standard deviation (std_vel_3) and mean absolute velocity (mean_abs_vel_3) of the novel object (channel 3) over the evaluation sequence. Define a threshold of std_vel_3 < 1.5 as representational freezing (lazy encoder) if MSE remains high.
3. Modify `src/models_dual_stream.py` to add Contrastive Coordinate Regularization (CCR) inside `NonParametricJEPASpatial.forward`. Integrate:
   - Temporal smoothness (L_smooth): L2 distance of consecutive-frame coordinates over the 4-frame sequence (z_0, z_1, z_2, z_3 normalized to [0, 1] by dividing by 127.0).
   - Spatial separation:
     * Hinge-loss (for Arm J): Pairwise relu(epsilon - |z_i - z_j|) over all pairs of active coordinate channels up to d_t, averaged over all 4 frames of the transition sequence. Use epsilon = 0.15.
     * Covariance-loss (for Arm K): Off-diagonal covariance penalty on the normalized active coordinates z_0, z_1, z_2, z_3 across the batch, averaged over all 4 frames.
   - Return CCR loss components in the loss dictionary. Allow toggling ccr_mode ('none', 'hinge', 'covariance') and specifying ccr_smooth_weight and ccr_spatial_weight. Set default weights to 10.0.
4. Create `src/run_phase14_experiments.py` (basing it on `src/run_phase13_experiments.py`) to run a matched 5-seed sweep (seeds: 42, 123, 456, 789, 999) on Arm G (Original RGB CLTS baseline), Arm J (CCR-Hinge), and Arm K (CCR-Covariance).
5. In the training loop of `src/run_phase14_experiments.py`, ensure that:
   - Passive training on N=3 (steps 1..1500) and active training on N=4 (steps 1501..3000) use the appropriate ccr_mode for each arm.
   - Evaluation at checkpoints and final step tracks centroid decoding MSE, test simulation loss, soft spatial variance (mean_var_3), pointer entropy, and the coordinate velocity metrics (std_vel_3 and mean_abs_vel_3).
6. Perform Welch's t-test and Levene's test to compare the step 3000 test simulation loss of Arms J and K against Arm G.
7. Save the compiled summary of metrics to `archive/iter_14/results/summary_phase14.csv`, save audit logs to `archive/iter_14/results/audit_results_phase14.json`, and plot the offline test simulation loss adaptation trajectory over training steps, saving the plot to `archive/iter_14/results/adaptation_curves_phase14.png`.
8. Explicitly verify the falsification criteria and the lazy encoder velocity metrics, reporting your findings in detail.