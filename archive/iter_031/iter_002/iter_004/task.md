Create the training and evaluation script `src/run_iter031_partA.py` in the current project workspace.

Use `src/run_iter030_arm2.py` as a template, but adapt it fully for `ReconVICRegSeparateDyn`:
1. Import `ReconVICRegSeparateDyn` from `src.models_recon`.
2. Define the 20 seed bank: `[7, 17, 31, 53, 71, 83, 97, 113, 127, 149, 101, 103, 107, 109, 131, 137, 139, 151, 157, 163]`.
3. Define the three Arms as follows:
   - **Arm A**: "Arm A (d_max=8, trained)", d_max=8, freeze_encoder=False, recon_weight=25.0
   - **Arm B**: "Arm B (d_max=2, trained)", d_max=2, freeze_encoder=False, recon_weight=25.0
   - **Arm C**: "Arm C (d_max=8, random-encoder)", d_max=8, freeze_encoder=True, recon_weight=25.0
4. Configure all arms to use: steps=8000 (dry_run=5), batch_size=32, lr=3e-4, d_t=3 (frozen), pos_encoding="none", coord_vicreg=True, var_weight=25.0, cov_weight=25.0, sim_weight=1.0, and GDASR in log-only mode (which is handled by model.gdasr_log_only=True and no manual recruitment).
5. Ensure the training loop correctly updates the ReconVICReg model, calls backward on total loss, and handles encoder freezing for Arm C.
6. The evaluation pipeline (`evaluate_run`, `collect_multitraj_eval_data`, `_compute_semantic_probes_core`, `compute_semantic_probes`, `compute_centroid_mse`) must be identical to `src/run_iter030_arm2.py`, but you should also collect and report `recon_mse_mean` (average reconstruction MSE across the evaluation steps).
7. Ensure that results are saved to `archive/iter_031/results/` as CSV/JSON, and summary CSV and summary markdown analyses are written correctly. Let the output directory be `archive/iter_031/results/` (and subdirectories like `checkpoints/` inside).
8. Support `--workers` flag (defaults to 2), `--dry-run` flag, and `--seed-override` flag for easy testing.
9. Verify that running with `--dry-run` executes quickly and successfully across all 3 arms on a couple of seeds. Clean up dry run output files or prefix them clearly. Do NOT start the full 60 runs yet — just run a dry-run test to make sure everything works perfectly.