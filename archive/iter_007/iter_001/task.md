You are a world-class AI researcher and planner. Your task is to implement and evaluate Phase 7 of the Thalamus research campaign, focusing on 'Active-Interaction-Driven Emergent Specialization' vs 'Passive Observation' during the N=3 -> N=4 generalization transition, adhering strictly to the Research Manager's critique on preventing the supervised 'Supervision Trap'.

Follow these sequential steps:

1. READ and REWRITE `src/pre_registration.md` to update the pre-registration hypothesis, falsification criteria, and experimental design. The new formulation must be 100% unsupervised, as follows:
   - **Hypothesis**: Under the dynamic, unsupervised JEPA representation-learning regime (GDASR), active physical interaction (Active Probing via Subsumption Motorics) with a newly introduced 4th object during the N=3 -> N=4 generalization transition will naturally force the newly recruited 4th latent dimension to represent the object's spatial coordinates more strongly than passive observation. Specifically:
     - A post-hoc linear readout probe trained on frozen latent representations of the Active Probing model to decode the 1D physical position of the 4th object will show a statistically significant correlation improvement (\Delta |r| >= 0.25) over the post-hoc linear readout probe trained on the Passive Observation model.
     - The Active Probing model will achieve a Pearson correlation |r| >= 0.40 between the activity of its recruited 4th dimension and the physical position of the 4th object, compared to |r| < 0.15 for the Passive Observation model.
     - This active-probing-driven spatial specialization will emerge without any backpropagation of coordinate/supervised loss gradients into the representation network (which remains 100% unsupervised via local temporal prediction and VICReg), and without causing representation collapse (cross-dimension correlation r_cross <= 0.30).
   - **Falsification Criteria**:
     - Falsification Criterion 1: The post-hoc linear readout of the 4th object's position from the Active Probing model's recruited dimension does NOT show a correlation improvement of at least \Delta |r| >= 0.25 over the Passive Observation model.
     - Falsification Criterion 2: The absolute Pearson correlation |r| between the recruited 4th dimension's activity and the physical position of the 4th object across the 5 evaluation seeds for the Active Probing model is less than 0.40.
     - Falsification Criterion 3: Active probing causes representation collapse (VICReg covariance/variance loss spikes, or cross-dimension correlation r_cross > 0.30).

2. DEVELOP/WRITE a new experimental runner `src/run_phase7_experiments.py` that implements a 5-seed sweep ([42, 123, 456, 789, 999]). For each seed, the runner must:
   - Train a single `DynamicJEPA` model passively on N=3 objects from step 1 to 1500 (using standard unsupervised loss: similarity + variance + covariance). Warmup/cooldown parameter tuning should match past iterations (e.g. `cooldown=300`, `stabilization_period=100`, `k=4`). Update recruitment logic using `model.update_recruitment_logic(sim_loss, target_dim=3)`.
   - At step 1500, checkpoint/clone this model into two identical branches:
     - **Branch A: Passive Observation (Control)**: Step 1501 to 3000 in N=4 objects environment. Take passive actions: `action = {"acc": 0.0, "push": False}`. Update model and recruitment logic normally.
     - **Branch B: Active Probing (Experimental)**: Step 1501 to 3000 in N=4 objects environment. Take active actions using a PD-controller + push mechanism targeting the 4th object's physical position from `info['positions'][3]`. E.g., `acc = Kp * (target_pos - pointer_pos) + Kd * dedt`, triggering a deliberate push when within 5 pixels, with a cooldown of 15 steps. The representation model MUST be trained 100% unsupervised using standard local temporal prediction loss + VICReg, with NO gradient flow from the controller or any coordinate target. Update recruitment logic normally.
   
3. POST-HOC EVALUATION (Linear Probe):
   - For both branches, freeze the model. Generate a fresh, independent test set of 200 transitions in a separate N=4 environment (seed + 5000) to collect:
     - The recruited 4th latent dimension activity (z_4).
     - The physical position of the 4th object (y_4).
   - Fit a 1D post-hoc linear regression probe (analytically using Ordinary Least Squares) mapping z_4 to y_4 on the first 100 test transitions.
   - Evaluate on the remaining 100 test transitions and measure:
     - The absolute Pearson correlation coefficient |r| between z_4 and y_4.
     - The MSE of the position prediction.
     - Check representation collapse metrics: cross-dimension correlation r_cross (correlation of recruited 4th dimension with the other 3 dimensions).

4. COMPILE METRICS & GENERATE PLOTS:
   - Save all evaluation metrics to `archive/iter_007/results/summary_phase7.csv`.
   - Create a plot comparing the post-hoc decoded vs ground-truth physical position curves of the 4th object for both the Passive and Active Probing models, and save it to `archive/iter_007/results/reconstruction_comparison.png`.
   - Generate a detailed markdown report in `archive/iter_007/results/phase7_report.md` analyzing the results, directly auditing them against the pre-registration falsification criteria, and reporting all findings with absolute scientific rigor.

Run this comprehensive pipeline and ensure it completes successfully. If any error occurs, debug and resolve it.