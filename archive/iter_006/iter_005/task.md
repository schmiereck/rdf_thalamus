Implement and execute Phase 4 sweeps and scientific reporting for Thalamus.

Task Steps:
1. Write a Python script `src/run_phase4_sweeps.py` to run:
   - Part 1: Generalization and Dimension Recruitment Sweep (transitioning $N=3 \to N=4$ objects) under clean conditions across 5 seeds:
     - Models: `b1` (FixedJEPA, $d_t=3$), `b1_large` (FixedJEPA, $d_t=4$), and `dynamic` (DynamicJEPA, starting at $d_t=3$, recruiting to 4).
     - Environment begins with $N=3$ objects.
     - Warmup representation for 500 steps.
     - At step 501, reset `dynamic` error buffer and start collecting stable N=3 prediction errors in `error_buffer`.
     - At step 1500, transition the environment to $N=4$ objects.
     - Train until step 3000 (total 3000 steps).
     - During steps 1501-3000, call `update_recruitment_logic(sim_loss.item(), target_dim=3)` on `dynamic`.
     - Log step of recruitment for `dynamic` (step when $d_t$ becomes 4).
     - At step 3000, evaluate each model on a test set of 100 transitions of $N=4$ and record:
       - Average L2 prediction loss.
       - Pearson correlation coefficient $|r|$ between the recruited 4th latent channel `z_target[:, 3]` of `dynamic` and the physical position of the 4th object.
     - Save the training logs as CSVs to `archive/iter_006/runs/gen_{model}_seed{seed}.csv`.

   - Part 2: Noise Robustness and Attention Watchdog Resilience Sweep:
     - For each of the 5 seeds (42, 123, 456, 789, 999), load the pre-trained `M_active` model from `archive/iter_005/runs/M_active_seed{seed}.pt`.
     - Run 100-step test episodes under three environmental noise conditions:
       - Clean: `pixel_noise_std = 0.0`, `noisy_tv = False`.
       - Global: `pixel_noise_std = 0.15`, `noisy_tv = False`.
       - Noisy-TV: `pixel_noise_std = 0.0`, `noisy_tv = True`.
     - In each condition, measure:
       - Average prediction loss (L2 temporal surprise).
       - Average attention tracking overlap with the primary physical object `pos_0`.
     - Compute the relative tracking overlap efficiency under noise: $Overlap_{noise} / Overlap_{clean}$.

   - Part 3: Aggregation, Plotting, and Reporting:
     - Compute means and standard deviations across seeds.
     - Save compiled metrics to `archive/iter_006/results/summary_phase4.csv`.
     - Generate learning curve plots of `dynamic` post-recruitment vs `b1_large` and save as `archive/iter_006/results/generalization_curves.png`.
     - Write a detailed scientific report evaluating each hypothesis and pre-registered falsification criterion to `archive/iter_006/results/phase4_report.md`.

2. Run `python src/run_phase4_sweeps.py` to execute the sweeps and generate all outputs under `archive/iter_006/`. Verify that all files compile and run successfully. Print out the summary metrics at the end.