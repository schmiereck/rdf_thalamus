1. Append `CLTSMotorController` to `src/motor.py`. It should implement the 3-layer closed-loop thalamic subsumption motor controller as pre-registered in `src/pre_registration.md` (Layer 1 Reflexive PD tracking, Layer 2 Kinematic tracking, Layer 3 Epistemic surprise-modulated push override).
2. Create and run `src/run_phase12_experiments.py` to execute a 5-seed sweep ([42, 123, 456, 789, 999]) evaluating three arms: Arm F-Passive, Arm F-Random, and Arm G (CLTS).
3. The experiment must:
   - Train the base `NonParametricJEPASpatial` passively on N=3 for 1500 steps.
   - Clone into three branches at step 1501 and transition to N=4 with 2x mass perturbation on the 4th object (object 3).
   - Train from 1501 to 3000 with respective motor policies.
   - Track spatial coverage entropy of the pointer.
   - Evaluate model checkpoints at [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000] on a standardized passive test set (N=4, 2x mass perturbation) to compute adaptation curves and offline AUC.
   - Compute centroid decoding MSE, soft spatial variance, and Pearson correlation at step 3000.
   - Check the three falsification criteria from `src/pre_registration.md` and log the audit explicitly.
4. Save CSV results to `archive/iter_012/results/summary_phase12.csv` and generate adaptation plots to `archive/iter_012/results/auc_recovery_curves.png`.