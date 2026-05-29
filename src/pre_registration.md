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
127, 149], where collapse is defined as:

- **Dual Criterion:** A run is declared collapsed if **EITHER**:
  - (a) **Eval collapse:** At the final evaluation (step 8000), any of the first d_t=3 z_dyn dimensions has batch-std < 0.5 computed over 200 evaluation samples from a fresh PhysicsSandbox(N=3). This is the standing criterion used in iter_023–025.
  - (b) **Train collapse:** The per-dimension mean training-step z_dyn std at the final training log (step 8000) is < 0.5. This means VICReg has effectively given up on that dimension during training.

- **Sanity Disqualification:** A seed is disqualified (counted as collapsed) if **EITHER**:
  - Mean total loss at step 8000 exceeds 50. This divergence threshold is ~7× the healthy training mean from iter_025 v2 Arm A (≈6.05 ± 6.5) and well above any non-diverged seed.
  - Mean per-dimension z_dyn std at the final training log is < 0.5. This is half the VICReg variance target of std ≥ 1.0, meaning the representation has trivially satisfied / failed to train the variance constraint. (This is already captured by train collapse above.)

Report both collapsed_eval and collapsed_train per seed so the rate is reproducible under either rule alone.

## 3. Proposed Method
Step-by-step experimental protocol:

1. Create src/run_phase0_collapse_sweep.py based on run_phase0_id_probe_v2.py,
   stripped to only the JEPA+VICReg control arm (no supervised/contrastive losses).
   
2. Sweep four regime knobs one-at-a-time against the canonical baseline
   (all arms share the same 10 seeds and common hyperparameters):

   **Common to all arms:**
   - d_max=8, d_t=3
   - dyn_readout="centroid_gated", pos_encoding="none"
   - primary_objective="jepa", ccr_mode="covariance"
   - ccr_smooth_weight=10, ccr_spatial_weight=10
   - gdasr_log_only=True
   - gradient clipping max_norm=1.0
   - 8000 training steps, Adam optimizer
   - replay_buffer_capacity=4000, pre-fill 200 transitions

   Arm A0 (canonical repeat): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=32. Same 10 seeds.
   
   Arm A1 (batch_size=64): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=64. Same 10 seeds.
   
   Arm A2 (var_weight=50): lr=3e-4, var_weight=50, cov_weight=25,
     sim_weight=25, batch_size=32. Same 10 seeds.
   
   Arm A3 (sim_weight warm-up): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight ramped 0→25 over 1000 steps, batch_size=32. Same 10 seeds.
     Ramp formula: sim_weight = 25.0 * (step / 1000.0) for step <= 1000, then 25.0.
   
   Arm A4 (lr=1e-4): lr=1e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=32. Same 10 seeds.

3. Evaluation at step 8000 only (no intermediate checkpoints to save compute):
   - Collapse check (eval): per_dim_std < 0.5 on 200 eval samples from fresh env
   - Collapse check (train): per_dim_std at step 8000 from training log
   - VICReg health: per_dim_std, mean_abs_corr on eval samples
   - Training loss sanity: mean total loss, var_loss, sim_loss at final step
   - Centroid MSE (for REFERENCE only, NOT used for regime selection)
   - Semantic probes (for REFERENCE only, NOT used for regime selection)

4. Stop rule: ALL arms that are started must complete their full 10-seed runs.
   Do NOT early-terminate subsequent arms even if one passes the ≤10% gate.
   Report every arm's full collapse rate.

5. The ONLY dependent variable used for regime selection is collapse rate
   (under the dual criterion). delta_R2_color, centroid MSE, and other
   downstream metrics are recorded for reference but MUST NOT be used to pick
   a winning regime. This is explicitly prohibited.

6. Hungarian matching remains the standing rule for any auxiliary semantic
   probe reporting (not relevant for collapse measurement but included for
   consistency).

7. Output:
   - Aggregated CSV with per-seed results for all arms
   - Final analysis markdown with per-arm collapse rates under dual criterion,
     eval-only criterion, and train-only criterion; gate status; and any
     disqualified seeds flagged.
   - If no arm clears the ≤10% gate, explicitly state:
     "Measured null: no swept configuration cleared the ≤10% gate under the
     pre-registered protocol."

Files to create/modify:
- src/run_phase0_collapse_sweep.py (NEW): collapse-rate sweep runner
- src/pre_registration.md (UPDATE): this plan (corrected)

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
*Updated with Manager corrections: dual collapse criterion, VICReg sanity floor (std≥0.5, loss≤50), confound-free arms (all buffer=4000), no early termination.*
