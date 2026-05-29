# RDF Scientific Pre-Registration

*   **Iteration:** 026
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
There exists a training-regime configuration within the swept parameter space
{batch_size ∈ {32, 64}, var_weight ∈ {25, 50}, sim_weight ∈ {25 constant, 0→25 ramp},
learning_rate ∈ {3e-4, 1e-4}} that reduces the z_dyn collapse rate of the
NonParametricJEPASpatial encoder (JEPA+VICReg, d_max=8, d_t=3, centroid_gated
readout) from the current 30% (iter_025 v2) to ≤10% over 10 seeds with 8000
training steps. The most likely candidate is batch_size=64, because VICReg's
variance and covariance estimation reliability scales inversely with
1/sqrt(B), and B=32 provides marginal statistical power for per-dimension
std estimation across d_t=3 dimensions.

## 2. Falsification Criterion
No single-knob variation in the sweep achieves ≤10% collapse rate (i.e., ≤1
collapsed seed out of 10) over the seed set [7, 17, 31, 53, 71, 83, 97, 113,
127, 149], where collapse is defined as: at the final evaluation (step 8000),
any of the first d_t=3 z_dyn dimensions has batch-std < 0.5 computed over 200
evaluation samples from a fresh PhysicsSandbox(N=3). Additionally, any
configuration where the mean training loss at step 8000 exceeds 100 (diverged)
or where the mean per-dimension z_dyn std at the final training log is < 0.1
(VICReg trivially satisfied / representation collapsed at training time) is
disqualified regardless of the evaluation collapse rate.

## 3. Proposed Method
Step-by-step experimental protocol:

1. Create src/run_phase0_collapse_sweep.py based on run_phase0_id_probe_v2.py,
   stripped to only the JEPA+VICReg control arm (no supervised/contrastive).
   
2. Sweep four regime knobs one-at-a-time against the canonical baseline:

   Arm A0 (canonical repeat): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=32. Identical to iter_025 v2 Arm A.
     10 seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149].
   
   Arm A1 (batch_size=64): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=64, replay_buffer_capacity=4000.
     Same 10 seeds.
   
   Arm A2 (var_weight=50): lr=3e-4, var_weight=50, cov_weight=25,
     sim_weight=25, batch_size=32. Same 10 seeds.
   
   Arm A3 (sim_weight warm-up): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight ramped 0→25 over 1000 steps, batch_size=32. Same 10 seeds.
   
   Arm A4 (lr=1e-4): lr=1e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=32. Same 10 seeds.

3. All arms share: gradient clipping max_norm=1.0, 8000 training steps,
   d_max=8, d_t=3, dyn_readout="centroid_gated", pos_encoding="none",
   CCR covariance mode (ccr_smooth_weight=10, ccr_spatial_weight=10),
   gdasr_log_only=True, Adam optimizer, replay_buffer pre-fill 200 transitions
   (doubled from 100 to ensure batch_size=64 is always achievable from step 1).

4. Evaluation at step 8000 only (no intermediate checkpoints to save compute):
   - Collapse check: per_dim_std < 0.5 on 200 eval samples from fresh env
   - VICReg health: per_dim_std, mean_abs_corr on eval samples
   - Training loss sanity: mean total loss, var_loss, sim_loss at final step
   - Centroid MSE (for reference only, NOT used for regime selection)
   - Semantic probes (for reference only, NOT used for regime selection)

5. Stop rule: Arms are evaluated in order A0, A1, A2, A3, A4. The first arm
   achieving ≤10% collapse rate (≤1/10 seeds collapsed) passes the gate and
   becomes the new canonical regime. If no arm clears, report the best achieved
   collapse rate as a measured null and do NOT declare a winner.

6. The ONLY dependent variable is collapse rate. delta_R2_color, centroid MSE,
   and other downstream metrics are recorded for reference but MUST NOT be used
   to pick a winning regime.

7. Hungarian matching remains the standing rule for any auxiliary semantic
   probe reporting (not relevant for collapse measurement but included for
   consistency).

Files to create/modify:
- src/run_phase0_collapse_sweep.py (NEW): collapse-rate sweep runner
- src/pre_registration.md (UPDATE): this plan

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
