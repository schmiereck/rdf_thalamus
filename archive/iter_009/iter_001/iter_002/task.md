Write a Python script `src/run_phase9_experiments.py` to run the 5-seed comparative sweep (seeds: 42, 123, 456, 789, 999) across 3 arms:
   - Arm A (Gentle): Static bottleneck with fixed \lambda = 0.01.
   - Arm B (Strong): Static bottleneck with fixed \lambda = 0.10.
   - Arm C (Experimental DSMC): Dynamic bottleneck with DSMC (\lambda_{max}=0.10, \gamma=10.0, \alpha=0.95 smoothed surprise, initialized at 1.0).

Follow these steps:
1. Update `src/pre_registration.md` in-place:
   - Adjust falsification threshold for centroid decoding MSE to <= 70.0.
   - Add the curriculum activity sanity check mandate:
     'Curriculum Activity Sanity Check: To assert that the curriculum successfully executed, the final average penalty weight across the 5 seeds must satisfy Mean(\lambda_T) >= 0.05. If the curriculum fails to ramp up to this level of regularization but passes the other metrics, it must be reported as a failure of the curriculum to activate, not a successful resolution of the trade-off.'

2. Write and execute `src/run_phase9_experiments.py`:
   - Pass k_chan=3 to forward when lambda_spatial > 0.
   - Clone models using `clone_dynamic_jepa_spatial`.
   - Maintain the passive N=3 step training, active probing N=4 step training, and GDASR recruitment mechanisms from Phase 8.
   - For Arm C, apply DSMC:
     - Initialize EWMA surprise \bar{S}_{1500} = 1.0.
     - For each step t from 1501 to 3000:
       - \lambda_t = 0.10 * exp(-10.0 * \bar{S}_{t-1})
       - Pass \lambda_t as the lambda_spatial argument to the forward pass (with k_chan=3).
       - Do backward and step.
       - S_t = loss_dict["sim_loss"].item()
       - \bar{S}_t = 0.95 * \bar{S}_{t-1} + 0.05 * S_t
   - Compute all metrics on 203 passive test steps.
   - Save the CSV `archive/iter_009/results/summary_phase9.csv`.
   - Plot EWMA surprise and dynamic lambda trajectories for Arm C across seeds, saving to `archive/iter_009/results/dsmc_trajectories.png`.
   - Plot ground truth vs decoded positions for seed 42 across all arms, saving to `archive/iter_009/results/performance_comparison_phase9.png`.
   - Generate a markdown report `archive/iter_009/results/phase9_report.md`.