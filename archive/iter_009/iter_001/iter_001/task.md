You are a highly skilled research engineer. Please execute Phase 9 of the project.

Follow these precise instructions:

1. Update `src/pre_registration.md` in-place to align with the Strategic Research Manager's Notes:
   - Adjust the falsification threshold for the mean centroid decoding MSE to <= 70.0 (instead of 72.0) under Section 2.
   - Add the curriculum activity sanity check mandate in Section 2:
     'Curriculum Activity Sanity Check: To assert that the curriculum successfully executed, the final average penalty weight across the 5 seeds must satisfy Mean(\lambda_T) >= 0.05. If the curriculum fails to ramp up to this level of regularization but passes the other metrics, it must be reported as a failure of the curriculum to activate, not a successful resolution of the trade-off.'

2. Write a Python script `src/run_phase9_experiments.py` to run a 5-seed comparative sweep (seeds: 42, 123, 456, 789, 999) across 3 arms:
   - Arm A (Gentle): Static bottleneck with fixed \lambda = 0.01.
   - Arm B (Strong): Static bottleneck with fixed \lambda = 0.10.
   - Arm C (Experimental DSMC): Dynamic bottleneck with DSMC (\lambda_{max}=0.10, \gamma=10.0, \alpha=0.95 smoothed surprise, initialized at 1.0).

   Specifically, for each seed:
   - Train on N=3 passively for 1500 steps (base model), with GDASR recruiting up to d_t = 3. Make sure to mirror the exact GDASR behavior, model training, and manual triggers from Phase 8.
   - Clone the base model for each arm using `clone_dynamic_jepa_spatial`.
   - Train each cloned model on N=4 for 1500 steps (steps 1501 to 3000) under closed-loop active probing targeting the recruited channel's centroid (exactly like Phase 8), with GDASR recruiting to d_t = 4.
   - For Arm C, apply DSMC:
     - Initialize EWMA surprise \bar{S}_{1500} = 1.0.
     - For each step t from 1501 to 3000:
       - \lambda_t = 0.10 * exp(-10.0 * \bar{S}_{t-1})
       - Pass \lambda_t as the lambda_spatial argument to the forward pass (with k_chan=3).
       - Do backward and step.
       - S_t = loss_dict["sim_loss"].item()
       - \bar{S}_t = 0.95 * \bar{S}_{t-1} + 0.05 * S_t
       - Log the step-by-step S_t, \bar{S}_t, and \lambda_t for tracking.
   - For Arm A and B, pass fixed \lambda = 0.01 or 0.10 as lambda_spatial (with k_chan=3).
   - At evaluation:
     - Run 203 steps of N=4 passive evaluation (exactly like Phase 8) to compute:
       - Pearson |r| Centroid & Pearson |r| Activation.
       - Post-hoc centroid decoding MSE (mse_cent).
       - Soft spatial variance of channel 3 (mean_var_3).
       - Collapse state (using the Criterion 5 check: E[|a_3|] >= 0.1 * E[|a_all|] and std(x_mean_3) > 5.0 pixels).
       - Log the final penalty weight \lambda_T = \lambda_{3001} at the end of training step 3000 for Arm C (which is \lambda_{3001} = 0.10 * exp(-10.0 * \bar{S}_{3000})), and 0.01 or 0.10 for Arm A/B.

3. Create directories and save results:
   - Save the raw data of all runs in a CSV `archive/iter_009/results/summary_phase9.csv`.
   - Log the training trajectory (surprise \bar{S}_t and \lambda_t) of Arm C for each seed, and save a plot of their mean trajectories over the N=4 phase to `archive/iter_009/results/dsmc_trajectories.png`.
   - Save a comparison plot showing the decoded positions vs ground truth for seed 42 across the three arms to `archive/iter_009/results/performance_comparison_phase9.png`.
   - Generate a comprehensive markdown report `archive/iter_009/results/phase9_report.md` reviewing the results against the pre-registered falsification criteria and curriculum sanity check.

4. Check your code carefully for bugs (device management, model copying, learning rates, correct logging of surprise/lambda). Run the script and compile the final summary metrics. Verify whether all hypotheses are validated or if any falsification criteria are triggered. Ensure everything is correctly placed in `archive/iter_009/results/`.