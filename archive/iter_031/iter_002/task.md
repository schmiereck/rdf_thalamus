You are executing Part A of iter_031: the Reconstruction+VICReg Ceiling Probe.

READ src/pre_registration.md FIRST — you must adhere to all pre-registered gates, arms, and configurations.

## Objective
Test whether Reconstruction+VICReg on the separate-backbone architecture achieves mean ΔR²_color ≥ 0.30 with lower 95% CI ≥ 0.18 (F1, F2), AND with non-trivial margins over d_max=2 (F3) and random-encoder (F4) controls.

## Three Arms (20 seeds each, union seed bank)
- **Arm A**: Reconstruction+VICReg, d_max=8, trained encoder, recon_weight=25.0
- **Arm B**: Reconstruction+VICReg, d_max=2, trained encoder, recon_weight=25.0 (under-capacity control)
- **Arm C**: Reconstruction+VICReg, d_max=8, RANDOM-ENCODER (frozen at init, only decoder trained), recon_weight=25.0

Seed bank: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149, 101, 103, 107, 109, 131, 137, 139, 151, 157, 163] (20 seeds)

## Implementation Plan

### Step 1: Create src/models_recon.py
Create `ReconVICRegSeparateDyn` model class:
- Encoder: `SeparateDynEncoder` from `src/models_separate_dyn.py` (existing, unchanged)
- Decoder head on dyn spatial features: deconv from (B, d_max, 8) → (B, 3, 128)
  Architecture: ConvTranspose1d(d_max, 128, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(128, 64, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(64, 32, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(32, 3, k=5, s=2, p=2, op=1)
- Forward pass:
  1. Encode x_target → z_coord, z_dyn, and also get a_dyn spatial features (B, d_max, 8) before mean pooling
  2. Decode: x_recon = decoder(a_dyn)
  3. Recon loss: MSE(x_recon, x_target)
  4. VICReg: var_loss + cov_loss on z_dyn (active dims), plus z_coord if coord_vicreg=True
  5. JEPA readout: predictor with stop-gradient on encoder output (surprise readout only, sim_weight=1.0)
  6. Total loss: recon_weight * recon_loss + var_weight * var_loss + cov_weight * cov_loss + sim_weight * sim_loss
- Must have all attributes needed for the evaluation pipeline: encoder, d_t, d_max, sub_features, color_probe_weight, color_probe_bias, id_contrastive_proj, gdasr_growth_points
- For Arm C (random-encoder): set requires_grad=False on ALL encoder parameters; only decoder + predictor + VICReg heads are trained

IMPORTANT: The SeparateDynEncoder forward() method returns (z_coord, z_dyn) via mean pooling. You need to also access the a_dyn spatial features (B, d_max, 8) BEFORE mean pooling. Either:
  a) Add a method to SeparateDynEncoder that returns intermediate features, or
  b) Modify the Recon model to call the backbone steps manually

### Step 2: Write training script src/run_iter031_partA.py
Follow the same pattern as src/run_iter030_arm2.py:
- Same replay buffer, environment, training loop structure
- For Arm C (random-encoder): after model creation, freeze ALL encoder parameters (set requires_grad=False on encoder.* params)
- Training: 8000 steps, batch_size=32, lr=3e-4, d_t=3 (frozen), pos_encoding="none"
- Loss weights: recon_weight=25.0, var_weight=25.0, cov_weight=25.0, sim_weight=1.0, coord_vicreg=True
- GDASR in log-only mode (no recruitment)
- Log reconstruction MSE, all loss components, per-dim std at regular intervals
- Evaluation: same pipeline as iter_029/030 (ΔR²_color linear probe, centroid MSE, collapse rate, VICReg health, reconstruction MSE)
- Save per-seed results CSV, summary CSV, and analysis markdown
- Support multiprocessing with --workers flag
- Support --dry-run for quick testing

### Step 3: Run the experiment
- Run all 3 arms × 20 seeds = 60 training runs
- Use multiprocessing (--workers) to parallelize
- Save checkpoints and results to archive/iter_031/results/

### Step 4: Evaluate and analyze
- Compute ΔR²_color for each seed (same pipeline as iter_029/030)
- Compute collapse rates
- Apply pre-registered gates F1-F4:
  - F1: Arm A mean ΔR²_color ≥ 0.30 (non-collapsed seeds)
  - F2: Arm A lower 95% CI ≥ 0.18
  - F3: Arm A mean ΔR²_color - Arm B mean ΔR²_color ≥ 0.10
  - F4: Arm A mean ΔR²_color - Arm C mean ΔR²_color ≥ 0.10
- Apply pre-committed mandate revision language based on which gates pass/fail
- Write full analysis to archive/iter_031/results/partA_analysis.md

## Key Files to Reference
- src/pre_registration.md — pre-registered gates and outcome language
- src/models_separate_dyn.py — SeparateDynEncoder, NonParametricJEPASpatialSeparateDyn
- src/models_dual_stream.py — VICReg loss functions, DualStreamPredictor, NonParametricJEPASpatial
- src/run_iter030_arm2.py — training/evaluation pipeline pattern (USE THIS AS TEMPLATE)
- src/environment.py — PhysicsSandbox

## Critical Reminders
- recon_weight=25.0 is LOCKED (no sweep)
- d_t=3 frozen, GDASR log-only
- coord_vicreg=True
- Use the EXACT same ΔR²_color evaluation pipeline as iter_029/030
- Report 95% CI across 20 seeds for all metrics
- The SeparateDynEncoder has separate conv1_dyn..conv4_dyn and conv_identity_dyn for the dyn backbone
- The decoder takes the dyn backbone's spatial features (BEFORE mean pooling) as input
- For Arm C: encoder is frozen, ONLY decoder + predictor + probe heads are trained