You are a highly specialised research agent. Your task is to execute the Phase 16 scientific plan.

Follow these instructions end-to-end:

### Step 1: Update Pre-Registration File
Open and edit `src/pre_registration.md` to:
1. Ensure both the hypothesis and falsification criteria align strictly on the centroid MSE threshold: Success is strictly `< 70.0`, and the hypothesis is falsified if the post-transition centroid decoding MSE is `>= 70.0` (eliminating the previous 75.0 discrepancy).
2. Explicitly define and formulate the predictability metric as a mathematically stable ratio with epsilon safety:
   $$U_{\\text{new}} = \\frac{\\text{MSE}_{\\text{new}}}{\\text{Var}_{\\text{new}} + \\epsilon}$$
   where $\\epsilon = 1e-5$, where $\\text{MSE}_{\\text{new}}$ is the mean-squared-error of the latent coordinate prediction for the newly proposed dimension across a validation batch of 100 transitions, and $\\text{Var}_{\\text{new}}$ is the variance of the target latent coordinate for that dimension across the same batch.
3. Save the updated `src/pre_registration.md` file.

### Step 2: Implement Phase 16 Experiments Script
Create `src/run_phase16_experiments.py` based on `src/run_phase15_experiments.py`. Implement the following features and optimizations:

1. **Passive Pre-Training Caching Optimization (CRITICAL for CPU speed)**:
   For each seed, do NOT perform a redundant 1500-step passive pre-training for every arm. Instead:
   - For each seed `s` in `[42, 123, 456, 789, 999]`:
     a. Train a standard passive pre-trained model where transition to `d_t = 3` happens unconditionally at step 600. Save/cache this pre-trained state. This state will be used to initialize the active CLTS phase (step 1501) for Arms K, O, O_big, P, and P_big.
     b. Train a passive pre-trained model for Arm N (where transition to `d_t = 3` is evaluated with the original immediate MDL gate and rejected, keeping it at `d_t = 2` until step 1500). Save/cache this state. This state will be used to initialize the active CLTS phase for Arm N.
   - This reduces the number of passive pre-trainings from 30 to 10, saving 30,000 training steps and speeding up CPU execution dramatically!

2. **Experimental Arms**:
   Evaluate 6 arms matched across the same 5 seeds:
   - **Arm K (Baseline)**: No probation, unconditional `3 -> 4` transition at step 1800.
   - **Arm N (Original MDL)**: Immediate MDL gating at step 1800+ without probation.
   - **Arm O (WUP-PVU, W=100)**: WUP of 100 steps, PVU criteria at probation end.
   - **Arm O_big (WUP-PVU, W=500)**: WUP of 500 steps, PVU criteria at probation end.
   - **Arm P (WUP-MDL, W=100)**: WUP of 100 steps, MDL criteria (consistency ratio < 1.0) at probation end.
   - **Arm P_big (WUP-MDL, W=500)**: WUP of 500 steps, MDL criteria (consistency ratio < 1.0) at probation end.

3. **Probation Gating Mechanics during Active CLTS Training**:
   - For arms with WUP (O, O_big, P, P_big):
     - At step 1800, propose `3 -> 4` transition. Set `probationary = True`, `probation_end_step = 1800 + W`, and change model's active dimensions `d_t` to 4.
     - During the probationary period (while `probationary` is True):
       - Train the model using `branch_model.d_t = 4` to update both the encoder and predictor weights for 4 dimensions using standard gradient descent.
       - But in the active motor loop, keep `d_t_effective = 3`. Pass only the first 3 dimensions of the coordinates, dynamics, and centroids to `clts_controller.get_action()`, protecting the active exploration from the un-converged probationary channel.
       - Skip/disable GDASR updates during probation to prevent secondary proposals.
     - At `step == probation_end_step`:
       - Sample a validation batch of 100 transitions from `branch_replay`.
       - Evaluate criteria:
         - **PVU (Arm O / O_big)**:
           - Calculate target latent coordinate `z_target = zt_coord[:, 3]` and prediction `z_pred = zp_coord[:, 3]`.
           - `mse_new = mean((z_target - z_pred)**2)`.
           - `var_new = var(z_target)`.
           - `u_new = mse_new / (var_new + 1e-5)`.
           - Uniqueness: compute absolute Pearson correlation of `z_target` with each of the first 3 dimensions `zt_coord[:, j]` (j=0,1,2). `max_corr = max(abs(corr_j))`.
           - Accept if: `var_new > 1e-3` AND `u_new < 0.5` AND `max_corr < 0.8`.
         - **MDL (Arm P / P_big)**:
           - Compute consistency ratio: `ratio = sim_new / sim_old`.
           - Accept if: `ratio < 1.0`.
       - If Accepted: set `probationary = False`, keep `branch_model.d_t = 4`.
       - If Rejected: revert `branch_model.d_t = 3`, set `probationary = False`, and schedule next proposal check 50 steps later (e.g. step + 50 will start a new probationary window).
       - Print detailed, clean log statements for acceptance/rejection events, including all computed parameters (e.g., var_new, u_new, max_corr, or ratio).

4. **Matched Sweeps & Metrics**:
   - Collect and compile:
     - Post-collision centroid decoding MSE (overall and post-collision evaluation frames).
     - Standardized test simulation loss.
     - Soft spatial variance (`mean_var_3`) of the coordinate bottleneck.
     - Pointer spatial coverage entropy.
     - Coordinate velocity standard deviation (`std_vel_3`) and mean absolute velocity (`mean_abs_vel_3`).
     - Gating diagnostics: final `d_t` reached, transition accepted step, and final gating criteria.
   - Conduct Welch's t-tests and Levene's tests comparing Arm O and Arm O_big vs Baseline Arm K and Arm N.
   - Save summaries to `archive/iter_016/results/summary_phase16.csv`.
   - Plot adaptation curves across evaluation steps `[1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]` for all 6 arms and save to `archive/iter_016/results/adaptation_curves_phase16.png`.
   - Save structured JSON to `archive/iter_016/results/audit_results_phase16.json`.

### Step 3: Scientific Verification & Clean Up
Ensure the script runs with no errors on CPU. Print clear progress and a final manager summary. Avoid any hyperbolic language. Verify whether all pre-registered falsification criteria are met.