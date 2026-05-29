You are implementing a collapse-rate sweep for iter_026 of the Thalamus project. 

## CRITICAL: Read the pre-registration first
Read `src/pre_registration.md` for context, then update it with the corrected protocol below BEFORE writing any code.

## Manager Corrections to the Pre-Registration

The pre-registration needs three corrections per the Research Manager:

### Correction 1: Dual collapse criterion
Collapse is declared if **EITHER**:
- (a) The existing `check_collapse` fires: any of the first d_t=3 z_dyn dimensions has batch-std < 0.5 computed over 200 eval samples from a fresh PhysicsSandbox(N=3). This is the standing criterion used in iter_023–025.
- (b) The per-dimension mean training-step z_dyn std at the final training log is < 0.5 (meaning VICReg has effectively given up on that dimension during training).

Report BOTH components per seed so the rate is reproducible under either rule alone.

### Correction 2: VICReg sanity floor and divergence threshold
- VICReg variance term targets std ≥ 1.0. A configuration where the mean per-dim z_dyn std at the final training log is < 0.5 is disqualified (not just "collapsed" but "trivially satisfied / failed to train"). This is half the VICReg target — anything below has effectively given up on the variance constraint.
- Divergence threshold: iter_025 v2 Arm A had mean final_train_loss ≈ 6.05 (std ≈ 6.5). Set divergence disqualification at mean total loss > 50 (roughly 7× the healthy training mean, well above any non-diverged seed in iter_025).

### Correction 3: Confound-free arm specifications and robustness check
(a) ALL arms share replay_buffer_capacity=4000 (the larger value needed for B=64). This eliminates the buffer-size confound from Arm A1. The pre-fill remains 200 transitions.
(b) If any arm passes the ≤10% gate, ALL arms that were started must complete their full 10-seed runs (do NOT early-terminate subsequent arms). Report every arm's full collapse rate.

## Arms (corrected)

All arms: d_max=8, d_t=3, dyn_readout="centroid_gated", pos_encoding="none",
primary_objective="jepa", ccr_mode="covariance", ccr_smooth_weight=10,
ccr_spatial_weight=10, gdasr_log_only=True, gradient clipping max_norm=1.0,
8000 training steps, Adam optimizer, replay_buffer_capacity=4000, pre-fill 200.

Seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]

Arm A0 (canonical repeat): lr=3e-4, var_weight=25, cov_weight=25, sim_weight=25, batch_size=32
Arm A1 (batch_size=64): lr=3e-4, var_weight=25, cov_weight=25, sim_weight=25, batch_size=64
Arm A2 (var_weight=50): lr=3e-4, var_weight=50, cov_weight=25, sim_weight=25, batch_size=32
Arm A3 (sim_weight warm-up): lr=3e-4, var_weight=25, cov_weight=25, sim_weight ramped 0→25 over 1000 steps, batch_size=32
Arm A4 (lr=1e-4): lr=1e-4, var_weight=25, cov_weight=25, sim_weight=25, batch_size=32

## Task: Create the sweep script

Create `src/run_phase0_collapse_sweep.py` based on `src/run_phase0_id_probe_v2.py`. The key changes:

1. STRIP to only the JEPA+VICReg control arm (no supervised/contrastive losses). Remove all supervised_weight, contrastive_weight, color_probe logic. Keep the core JEPA+VICReg training loop, the encoder, predictor, and evaluation.

2. The arms are A0–A4 as specified above. Each arm varies exactly ONE knob vs A0 (except A3 which ramps sim_weight).

3. Use the DUAL collapse criterion:
   - `collapsed_eval`: any d_t z_dyn dim has batch-std < 0.5 over 200 eval samples
   - `collapsed_train`: any d_t z_dyn dim has mean training-logged std < 0.5 (from the last per_dim_std training log entry at step 8000)
   - `collapsed`: collapsed_eval OR collapsed_train
   - Report all three per seed.

4. Sanity disqualification:
   - If mean total loss at final step > 50 → seed is disqualified (counted as collapsed)
   - If mean per-dim z_dyn std at final training log < 0.5 → already captured by collapsed_train

5. NO early termination of arms. Run all 5 arms × 10 seeds = 50 runs.

6. Evaluation at step 8000 only (no intermediate checkpoints to save compute). Record:
   - collapsed_eval, collapsed_train, collapsed (booleans per seed)
   - per_dim_std (from eval)
   - per_dim_std_train (from last training log)
   - final_train_loss, final_sim_loss, final_var_loss, final_cov_loss
   - centroid_mse_mean (for REFERENCE only)
   - delta_r2_color (Hungarian primary, for REFERENCE only)
   - vicreg_per_dim_std, vicreg_mean_abs_corr

7. Generate an aggregated CSV and a final analysis markdown file reporting:
   - Per-arm collapse rate (under the dual criterion)
   - Per-arm collapse rate under eval-only criterion (for backward comparison with iter_025)
   - Per-arm collapse rate under train-only criterion
   - Per-seed details table
   - Which arm(s), if any, clear the ≤10% gate
   - If none clear, explicitly state: "Measured null: no swept configuration cleared the ≤10% gate under the pre-registered protocol"
   - Sanity check: any disqualified seeds flagged
   - Reference-only downstream metrics (centroid_mse, delta_r2_color) clearly labeled as NOT used for regime selection

8. Save results to `archive/iter_026/results/`

## Implementation Details

- Use the existing `NonParametricJEPASpatial` from `src/models_dual_stream.py` (import it)
- Use the existing `PhysicsSandbox` from `src/environment.py`
- Use the existing `ExtendedReplayBuffer` class (copy from v2 script)
- Use the existing helper functions: `check_collapse`, `compute_vicreg_health`, `set_seed`
- For semantic probes, use the existing `compute_semantic_probes` with Hungarian matching only (simplified from v2)
- For centroid MSE, use the existing `compute_centroid_mse` function
- The sim_weight ramp in A3: sim_weight = 25.0 * (step / 1000.0) for step <= 1000, then 25.0

## Pre-Registration Update

After creating the script, update `src/pre_registration.md` to reflect the corrected protocol. Include:
1. The corrected hypothesis
2. The dual collapse criterion (eval + train)
3. The corrected sanity floors (std >= 0.5, loss <= 50)
4. The corrected arm specifications (all arms share replay_buffer_capacity=4000)
5. The corrected stop rule (all arms complete; no early termination)
6. Explicit prohibition on using delta_R2_color or any downstream metric to pick a winner

Then run the script: `cd /project && python src/run_phase0_collapse_sweep.py`

The script should auto-detect GPU/CPU and use whatever is available.