Create `src/run_iter032.py` based on `src/run_phase0_sfa_separate_backbone.py` with these key requirements:
1. Update `ARMS` to contain:
   - E1 (VICReg-only, mean-pool): readout="mean", sub_features=1, sfa_weight=0, mask_dyn_sim=True, primary_objective="sfa"
   - E1.5 (VICReg-only, centroid-gated scalar): readout="centroid_gated", sub_features=1, sfa_weight=0, mask_dyn_sim=True, primary_objective="sfa"
   - E2 (VICReg-only, centroid-gated rich K=4): readout="centroid_gated", sub_features=4, sfa_weight=0, mask_dyn_sim=True, primary_objective="sfa"
   - E3 (SFA+VICReg, centroid-gated rich K=4): readout="centroid_gated", sub_features=4, sfa_weight=5.0, mask_dyn_sim=True, primary_objective="sfa"
   Set other parameters like sim_weight=1.0, var_weight=25.0, cov_weight=25.0, lr=3e-4, batch_size=32, d_max=8, d_t=3, pos_encoding="none", replay_buffer_capacity=4000.
2. Define the exact seeds list:
   `SEEDS = [7, 17, 31, 53, 71, 83, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163]`
3. Ensure the output paths are:
   - results: `archive/iter_032/results/`
   - runs CSVs: `archive/iter_032/results/runs/`
   - checkpoints: `archive/iter_032/results/checkpoints/`
   - summary: `archive/iter_032/results/summary.csv`
   - analysis: `archive/iter_032/results/analysis.md`
4. Update `run_single` to correctly create `NonParametricJEPASpatialSeparateDyn` with the sub_features from the arm config.
5. In `run_single()`, pool the target representations correctly when printing / checking train collapse (using `sub_features` parameter).
6. In `evaluate_run()` and `compute_semantic_probes()`, use `sub_features` from the arm config (or pass it from `evaluate_run` to probe helpers) and ensure that `z_dyn` is pooled correctly across `sub_features` dimension as specified in the prompt:
   `z_dyn_pooled = z_dyn[:, :d_t * sub_features].reshape(N, d_t, sub_features).mean(axis=2)`
7. Implement CI computation using scipy.stats.t.interval (t-distribution) on non-collapsed E2 seeds.
8. Implement all 6 gate checks (F1 to F6) exactly as described in the pre-registration and prompt, and paired-seed comparisons (for F4, F5, F6) computing the per-seed difference first.
9. Verify that a dry run works correctly with 2 seeds: `--dry-run --seeds 7 17`
10. Run the dry run and confirm it works! Print the output of the dry run to stdout. Do not run the full 80 runs yet, just do the dry run.