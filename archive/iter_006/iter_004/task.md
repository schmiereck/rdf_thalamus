Write and run the Phase 4 sweeps script `src/run_phase4_sweeps.py` that implements and runs the evaluations:

1. Dynamic Recruitment and Generalization Sweep ($N=3 \to N=4$ objects) across 5 seeds (42, 123, 456, 789, 999):
   - For each seed, trains `dynamic` (DynamicJEPA, starting at $d_t=3$, recruiting to 4), `b1` (FixedJEPA, $d_t=3$), and `b1_large` (FixedJEPA, $d_t=4$) under N=3.
   - Replay buffer size 2000, batch size 32, learning rate 1e-3, cov_weight=25.0, sim_weight=25.0, var_weight=25.0.
   - Warmup representation for 500 steps (no recruitment checks).
   - At step 501, reset `dynamic` error buffer and collect stable prediction errors in `error_buffer`.
   - At step 1500, transition the environment to $N=4$ objects.
   - Train until step 3000 (total 3000 steps).
   - During steps 1501-3000, call `update_recruitment_logic(sim_loss.item(), target_dim=3)` on `dynamic`.
   - Log the step of recruitment (when $d_t$ becomes 4) for `dynamic`.
   - At step 3000, evaluate each model on a test set of 100 transitions of $N=4$ and record:
     - Average prediction loss (sim_loss).
     - Pearson correlation coefficient $|r|$ between the recruited 4th latent channel `z_target[:, 3]` of `dynamic` and the physical position of the 4th object.
   - Save seed-by-seed CSV logs of steps 1501 to 3000 to `archive/iter_006/runs/gen_{model}_seed{seed}.csv`.

2. Noise Robustness and Attention Watchdog Resilience Sweep:
   - For each of the 5 seeds, load the pre-trained `M_active` model from `archive/iter_005/runs/M_active_seed{seed}.pt`.
   - Run 100-step test episodes under three environmental conditions using PhysicsSandbox with N=3:
     - Clean: `pixel_noise_std = 0.0`, `noisy_tv = False`.
     - Global noise: `pixel_noise_std = 0.15`, `noisy_tv = False`.
     - Noisy-TV entity: `pixel_noise_std = 0.0`, `noisy_tv = True`.
   - In each condition, measure:
     - Average prediction loss (L2 temporal surprise).
     - Average attention tracking overlap with the primary physical object `pos_0` (using `compute_physical_tracking_overlap`).
   - Compute the relative tracking overlap efficiency under noise: $Overlap_{noise} / Overlap_{clean}$.

3. Aggregation, Reporting, and Plotting:
   - Compute means and standard deviations across the 5 seeds.
   - Save complete compiled results to `archive/iter_006/results/summary_phase4.csv`.
   - Generate comparison learning curve plots of `dynamic` post-recruitment vs `b1_large` and save as `archive/iter_006/results/generalization_curves.png`.
   - Write a detailed scientific report evaluating each hypothesis and pre-registered falsification criterion to `archive/iter_006/results/phase4_report.md`.

Execute `python src/run_phase4_sweeps.py` to run the entire pipeline and verify that all outputs are generated cleanly. Ensure all logs, reports, and plots are written to the correct subdirectories under `archive/iter_006/`.