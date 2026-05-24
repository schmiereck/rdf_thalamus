Create and run src/run_phase13_experiments.py to evaluate the positional encodings hypothesis across 5 seeds.

Follow these instructions exactly:
1. Create `src/run_phase13_experiments.py` based on `src/run_phase12_experiments.py` but adapted for Phase 13.
2. In `src/run_phase13_experiments.py`:
- Use seeds: `[42, 123, 456, 789, 999]`.
- We will evaluate three arms:
  * "Arm G (RGB CLTS)" which uses `pos_encoding="none"`.
  * "Arm H (Linear CLTS)" which uses `pos_encoding="linear"`.
  * "Arm I (Sinusoidal CLTS)" which uses `pos_encoding="sinusoidal"`.
- Since each arm has a different input channel layout, they must be initialized and trained independently from step 1 onwards:
  * For each arm and seed:
    - Train passive base model for 1500 steps on N=3 env with the respective `pos_encoding` parameter using `train_base_model_passive(seed, device, pos_encoding)`.
    - Continue active training from steps 1501 to 3001 on N=4 env (with 2x mass perturbation on the novel 4th object) under Closed-Loop Thalamic Subsumption Motorics (CLTS) control (using CLTSMotorController).
    - Monitor surprises, track pointer positions for spatial coverage entropy, dynamically recruit dimensions, and evaluate test checkpoints at steps `[1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]`.
- Once all arms have finished training, run a statistical analysis:
  * Run a two-sample t-test (two-sided) comparing Arm G vs Arm H, and Arm G vs Arm I on step 3000 offline test simulation losses.
  * Run Levene's test for equality of variance on the step 3000 offline test simulation losses for Arm G vs Arm H, and Arm G vs Arm I.
  * Print p-values clearly in the logs.
- Audit results against the pre-registered falsification criteria:
  * Criterion 1 (Coordinate Accuracy): Falsified if centroid decoding MSE of the novel object under active control is >= 75.0 for Arm H or Arm I.
  * Criterion 2 (Spatial Tightness): Falsified if soft spatial variance of the coordinate encoder under active control is >= 10.0 for Arm H or Arm I.
  * Criterion 3 (Predictive Integrity): Falsified if post-collision test simulation loss is >= 0.050 for Arm H or Arm I.
  * Criterion 4 (Generalization / Robustness): Falsified if Arm H or Arm I test simulation loss is statistically worse than Arm G (Arm H/I simulation loss is significantly greater than Arm G, or t-test p <= 0.05).
- Save summary statistics to `archive/iter_013/results/summary_phase13.csv`.
- Plot offline test simulation losses at checkpoints for Arm G, Arm H, and Arm I (mean across seeds), and save to `archive/iter_013/results/auc_recovery_curves_phase13.png`.

3. Run the script `src/run_phase13_experiments.py` and print the logs and audit results. Make sure to capture the final statistics.