Please execute the Phase 15 research plan. Follow these sequential steps:

1. Read the pre-registered hypothesis and falsification criteria in `src/pre_registration.md` and strictly adhere to them.
2. Read `src/run_phase14_experiments.py` to understand the experimental setup, training loop, evaluation metrics, and plotting code.
3. Create `src/run_phase15_experiments.py` (basing it on `src/run_phase14_experiments.py`) to run a matched 5-seed comparative sweep (seeds: 42, 123, 456, 789, 999) across four experimental arms:
   - **Arm K (Baseline)**: Fixed covariance weight `ccr_spatial_weight = 10.0` with `ccr_mode='covariance'`.
   - **Arm L (Proportional SA-CCR)**: Dynamic proportional weight `ccr_spatial_weight(t) = 10.0 * (1.0 + 2.0 * \bar{S}(t))` with `ccr_mode='covariance'`.
   - **Arm M (Inverse SA-CCR)**: Dynamic inverse weight `ccr_spatial_weight(t) = 10.0 / (1.0 + 2.0 * \bar{S}(t))` with `ccr_mode='covariance'`.
   - **Arm N (Dual Control)**: Surprise Detector + slow Categorizer with consistency buffer (using `ccr_mode='covariance'`, fixed `ccr_spatial_weight=10.0`).
4. Implement dynamic surprise tracking for Arm L and Arm M:
   - Exponentially smoothed local surprise: `\bar{S}(t) = \alpha \cdot S(t) + (1 - \alpha) \cdot \bar{S}(t-1)` with smoothing factor `\alpha = 0.1` and `S(t)` is the online similarity loss (`sim_loss_val`), initialized to `0.10` at step 1501.
   - At each step `t` in passive training (steps 1..1500) and active training (steps 1501..3000), update `\bar{S}(t)` and scale the `ccr_spatial_weight` passed to `branch_model(...)`.
5. Implement the Dual Control (Arm N) slow Categorizer with consistency buffer:
   - For passive pre-training transition (at step 600): intercept the step. The Categorizer samples a validation batch of 100 transitions from the passive replay buffer, and computes the consistency ratio: `L_consistency = sim_loss_new / sim_loss_old` where `sim_loss_new` uses `d_t = 3` and `sim_loss_old` uses `d_t = 2`. If `L_consistency < 1.0`, accept the transition (`d_t = 3`), otherwise reject and suppress it.
   - For active training transition (at step 1800+): intercept the transition to `d_t = 4`. The Categorizer samples a validation batch of 100 transitions from the active replay buffer, and computes: `L_consistency = sim_loss_new / sim_loss_old` where `sim_loss_new` uses `d_t = 4` and `sim_loss_old` uses `d_t = 3`. If `L_consistency < 1.0`, accept (`d_t = 4`). If rejected, re-evaluate every 50 steps (e.g., step 1850, 1900, ...) until accepted.
6. Record and compile metrics for all four arms across 5 seeds:
   - Post-collision centroid decoding MSE (overall and post-collision evaluation frames).
   - Standardized test simulation loss.
   - Soft spatial variance (`mean_var_3`) of the coordinate bottleneck.
   - Pointer spatial coverage entropy.
   - Coordinate velocity standard deviation (`std_vel_3`) and mean absolute velocity (`mean_abs_vel_3`).
7. Apply Welch's t-test and Levene's test to compare:
   - Arm L vs Arm K.
   - Arm M vs Arm L.
   - Arm N vs Arm K.
8. Save the compiled summary to `archive/iter_015/results/summary_phase15.csv`, plot adaptation curves to `archive/iter_015/results/adaptation_curves_phase15.png`, and save audit results to `archive/iter_015/results/audit_results_phase15.json`.
9. Verify all pre-registered falsification criteria and the Strategic Manager's Notes, and write down an elegant, scientifically rigorous report of findings. Ensure there are no syntax or logical errors, and log all training progress clearly to stdout.