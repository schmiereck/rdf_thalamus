Write the new experimental runner src/run_phase0_sfa_multistep.py.

This runner is based on src/run_phase0_sfa_sweep.py but implements the following changes:
1. ARMS:
   - Arm A (k=20, d_max=8): 5 seeds [42, 123, 456, 789, 999]
   - Arm B (k=50, d_max=8): 5 seeds [42, 123, 456, 789, 999]
   - Arm C (k=100, d_max=8): 5 seeds [42, 123, 456, 789, 999]
   - Arm D (Contrastive NT-Xent, d_max=8): 5 seeds [42, 123, 456, 789, 999]
   - Arm E (k=50, d_max=16): 5 seeds [42, 123, 456, 789, 999]
   - Arm F (Diagnostic, sim_weight=0, k=50, d_max=8): 1 seed [42]
2. All SFA weight ramps: 0.1 -> 10.0 over 500 steps.
3. Multi-trajectory evaluation collects data sequentially per trajectory to allow trajectory-structured analysis.
4. Implement the Invariance-vs-Discrimination diagnostics:
   - within_traj_var: average variance of z_dyn over timesteps within each trajectory.
   - between_traj_var: variance of the trajectory-wise z_dyn means.
   - shuffled_delta_r2_color and shuffled_r2_dyn_color: compute by randomly shuffling the rows of all collected semantic probe features (z_dyn, z_coord, pos, colors, radii) before performing the 50/50 train/test split.
5. Save results to archive/iter_024/results/
6. Checkpoint evaluations must run at step 2000 (monitoring only) and step 5000 (final).
7. Ensure that multi-step SFA computes SFA loss externally via collections.deque trajectory buffer (prefilled for first 110 steps), and gradient flows through z_dyn_current while z_past is detached.
8. The file must contain all functions needed to run, including:
   - ReplayBuffer
   - set_seed
   - fit_linear_probe
   - compute_slowness_metrics
   - check_collapse
   - compute_vicreg_health
   - fit_multivariate_probe_r2
   - compute_normalized_temporal_var
   - compute_tracking_quality
   - compute_sub_feature_probes
   - compute_centroid_mse
   - collect_multitraj_eval_data
   - compute_semantic_probes
   - compute_shuffled_semantic_probes (the shuffled version)
   - evaluate_run_with_diagnostics
   - run_single
   - main with argparse (accepting --dry-run and --seeds)

Check the file for any syntax errors after writing it and run a quick dry-run with 5 steps using `--dry-run` to verify it runs without crashing.