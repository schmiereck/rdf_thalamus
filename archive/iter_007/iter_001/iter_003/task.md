Implement and run the Phase 7 experiments, evaluate the models, save metrics, generate plots, and write the markdown report as follows:

1. Create a script `src/run_phase7_experiments.py` that implements the 5-seed sweep ([42, 123, 456, 789, 999]).
2. For each seed:
   - Initialize a PhysicsSandbox environment with N=3 objects, seed=seed.
   - Initialize a `DynamicJEPA(d_max=8, h=3, k=4, cooldown=300, stabilization_period=100)`. Set `model.d_t = 3`.
   - Train on N=3 passively (action = {"acc": 0.0, "push": False}) from step 1 to 1500 using standard unsupervised local temporal prediction + VICReg loss.
   - Enforce a 1000-step representation warmup: do NOT call `update_recruitment_logic` during steps 1 to 1000.
   - At step 1001, call `model.reset_error_buffer()`.
   - From step 1001 to 1500, call `model.update_recruitment_logic(sim_loss_val, target_dim=3)`.
   - At step 1500, clone/checkpoint the model into two identical branches:
     - **Branch A (Passive Observation Control)**:
       - Environment starts at step 1501 in N=4 objects, seed=seed + 1000.
       - Take passive action: `action = {"acc": 0.0, "push": False}`.
       - Clear replay buffer and history, then prefill buffer with 100 transitions using passive actions.
       - Train model passively from step 1501 to 3000.
       - Call `model_A.update_recruitment_logic(sim_loss_val, target_dim=3)` on each step.
     - **Branch B (Active Probing Experimental)**:
       - Environment starts at step 1501 in N=4 objects, seed=seed + 1000.
       - Take active actions using a PD-controller + push targeting the 4th object's position (info['positions'][3]). E.g., `acc = Kp * (target_pos - pointer_pos) + Kd * dedt`, with `Kp = 2.0`, `Kd = 0.5`. Trigger push when within 5 pixels, with a 15-step cooldown.
       - Clear replay buffer and history, then prefill buffer with 100 transitions using active actions.
       - Train model actively from step 1501 to 3000.
       - The training remains 100% unsupervised (temporal prediction + VICReg), with NO gradients backpropagated from the PD-controller or target position.
       - Call `model_B.update_recruitment_logic(sim_loss_val, target_dim=3)` on each step.
   
3. POST-HOC EVALUATION:
   - For each branch, freeze the model. Generate a fresh, independent test set of 200 transitions in a separate N=4 environment (seed + 5000) using passive actions to collect target physical positions (y_4) of the 4th object and target latent representations (z_target[:, 3]) of the recruited 4th dimension (z_4).
   - Fit a 1D post-hoc linear regression probe analytically using Ordinary Least Squares mapping z_4 to y_4 on the first 100 test transitions.
   - Evaluate on the remaining 100 test transitions to compute:
     - The absolute Pearson correlation coefficient |r| between z_4 and y_4.
     - The MSE of the position prediction.
     - Cross-dimension correlation r_cross: mean absolute correlation of recruited 4th dimension with the other 3 dimensions.
     - Standard deviation (variance) of the recruited 4th dimension (std_4).

4. COMPILE METRICS & GENERATE PLOTS:
   - Save all evaluation metrics across the 5 seeds and both branches to `archive/iter_007/results/summary_phase7.csv`.
   - Create a plot comparing the post-hoc decoded vs ground-truth physical position curves of the 4th object for both the Passive and Active Probing models of seed 42, and save it to `archive/iter_007/results/reconstruction_comparison.png`.
   - Generate a detailed markdown report in `archive/iter_007/results/phase7_report.md` analyzing the results, directly auditing them against the pre-registration falsification criteria, and reporting all findings with absolute scientific rigor.

Run `python src/run_phase7_experiments.py` and execute the entire pipeline. Verify that all results are saved correctly and provide a detailed analysis of the findings.