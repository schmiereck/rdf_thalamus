1. Patch `src/models.py` to make `DynamicJEPA`'s `update_recruitment_logic` fully generic. Specifically, update it so that it can recruit from any dimension to the next (e.g. from `d_t = target_dim` to `d_t = target_dim + 1` where `target_dim` defaults to `self.d_t`). This allows recruiting from $d_t=3 \to 4$ when moving from $N=3 \to 4$.

2. Implement `src/run_phase4_sweeps.py` to execute the Phase 4 scientific evaluation:
   - Part 1: Dynamic Recruitment and Generalization Sweep ($N=3 \to N=4$ clean objects) across 5 seeds (42, 123, 456, 789, 999):
     - Train three models: `b1` (FixedJEPA, $d_t=3$), `b1_large` (FixedJEPA, $d_t=4$), and `dynamic` (DynamicJEPA, starting at $d_t=3$, recruiting to 4).
     - Environment begins with $N=3$ objects.
     - Warmup representation for 500 steps (do not update recruitment logic).
     - At step 501, reset `dynamic` error buffer and start collecting stable N=3 prediction errors in `error_buffer`.
     - At step 1500, transition to $N=4$ objects with parameterized physical variations.
     - Train until step 3000 (total 3000 steps).
     - During steps 1501-3000, call `update_recruitment_logic(sim_loss.item(), target_dim=3)` on `dynamic`.
     - Log step of recruitment for `dynamic` (the step when $d_t$ becomes 4).
     - Evaluate all models on a separate test set of 100 transitions of $N=4$ and record:
       - Overall prediction loss on $N=4$.
       - Pearson correlation coefficient $|r|$ between the recruited 4th latent dimension `z_target[:, 3]` of `dynamic` and the physical position of the 4th object.
     - Save the training logs as CSVs to `archive/iter_006/runs/gen_{model}_seed{seed}.csv`.

   - Part 2: Noise Robustness and Attention watchdog Resilience Sweep:
     - For each of the 5 seeds, load the pre-trained `M_active` model from `archive/iter_005/runs/M_active_seed{seed}.pt`.
     - Run 100-step test episodes under three environmental noise conditions:
       - Clean: `pixel_noise_std = 0.0`, `noisy_tv = False`.
       - Global: `pixel_noise_std = 0.15`, `noisy_tv = False`.
       - Noisy-TV: `pixel_noise_std = 0.0`, `noisy_tv = True`.
     - Measure and record for each condition:
       - Average prediction loss (L2 temporal surprise).
       - Average attention tracking overlap with the primary physical object `pos_0`.
     - Compute the relative tracking overlap efficiency under noise: $Overlap_{noise} / Overlap_{clean}$.

   - Part 3: Aggregation and Plotting:
     - Compute means and standard deviations across seeds.
     - Generate comparison learning curve plots of `dynamic` post-recruitment vs `b1_large` and save as `archive/iter_006/results/generalization_curves.png`.
     - Save the complete sweep metrics to `archive/iter_006/results/summary_phase4.csv`.
     - Write a detailed scientific report evaluating each hypothesis and pre-registered falsification criterion to `archive/iter_006/results/phase4_report.md`.

3. Run `python src/run_phase4_sweeps.py` and print the final compiled results. Ensure all files are written cleanly to `archive/iter_006/` subdirectories.