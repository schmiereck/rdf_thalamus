# RDF Scientific Pre-Registration

*   **Iteration:** 031
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Reconstruction+VICReg on the separate-backbone architecture achieves mean ΔR²_color ≥ 0.30
across 20 seeds (lower 95% CI ≥ 0.18), establishing that the architecture supports identity
encoding and that the M2 failure was objective-specific (all decoder-free objectives
insufficient for this architecture's mean-readout z_dyn stream), not architectural.
Additionally, reconstruction-trained models produce centroid MSE < 120 (improved over the
~160 baseline from VICReg-only and SFA+VICReg arms).

Specifically: a deconv decoder head on the dyn backbone's spatial features (B, d_max, 8)
→ (B, 3, 128) with loss = recon_weight × MSE(x_recon, x_input) + var_weight × VICReg(z_dyn)
+ cov_weight × VICReg(z_dyn) + (coord_vicreg=True) VICReg(z_coord) shapes z_dyn to carry
object identity information, as measured by the same ΔR²_color linear probe used across
iter_020-030.

## 2. Falsification Criterion
If Reconstruction+VICReg achieves mean ΔR²_color ≤ 0.275 (the best decoder-free result from
iter_029 Arm B, SFA+VICReg sfa_weight=5.0, 20 seeds), then the mean-readout z_dyn architecture
itself constrains identity encoding regardless of objective class, and the project must
redesign the z_dyn readout mechanism (e.g., centroid-gated readout from iter_027 Arm A') or
encoder architecture before any further objective work. This would be a fundamental architectural
finding, not an objective finding.

## 3. Proposed Method
## Part A: Reconstruction+VICReg Ceiling Probe (PRIMARY)

### A1: Model Implementation
Create `src/models_recon.py` containing `ReconVICRegSeparateDyn`:
- Encoder: `SeparateDynEncoder` (existing separate coord + dyn backbones, unchanged)
- Decoder: Deconv head on dyn spatial features a_dyn (B, d_max, 8) → (B, 3, 128)
  Architecture: ConvTranspose1d(d_max, 128, k=5, s=2, p=2, op=1) → ReLU → 
  ConvTranspose1d(128, 64, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(64, 32, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(32, 3, k=5, s=2, p=2, op=1)
- Loss: recon_weight × MSE(x_recon, x_target) + var_weight × [VICReg_var(z_dyn) + VICReg_var(z_coord)]
  + cov_weight × [VICReg_cov(z_dyn) + VICReg_cov(z_coord)] + sim_weight × predictor_loss
- Predictor: DualStreamPredictor with stop-gradient on encoder output (surprise readout only)
- All attributes needed for evaluation pipeline: encoder, d_t, d_max, sub_features,
  color_probe_weight, color_probe_bias, id_contrastive_proj, gdasr_growth_points

### A2: Quick Hyperparameter Scan
- recon_weight ∈ {10.0, 25.0, 50.0}
- 3 seeds per weight (7, 31, 97), 2000 steps each = 9 quick runs
- var_weight=25.0, cov_weight=25.0, sim_weight=1.0, coord_vicreg=True
- Select best recon_weight by ΔR²_color for full run

### A3: Full Training (20 seeds, union bank)
- Seeds: 10 original [7, 17, 31, 53, 71, 83, 97, 113, 127, 149] + 10 fresh [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]
- 8000 steps, batch_size=32, lr=3e-4, d_t=3, d_max=8
- pos_encoding="none", coord_vicreg=True
- GDASR in log-only mode (no recruitment)

### A4: Evaluation (identical pipeline to iter_029/030)
- ΔR²_color (primary): linear probe from z_dyn to object color, channel-object matched
- Centroid MSE: soft-argmax centroid decoding
- Collapse rate: per-dim std < 0.5 threshold
- VICReg health: per-dim std, mean absolute cross-correlation
- Reconstruction MSE
- All metrics reported with 95% CI across 20 seeds

### Comparison Baselines (from iter_029, no re-run)
- VICReg-only: ΔR²_color = 0.045 (20 seeds)
- SFA+VICReg sfa=5.0: ΔR²_color = 0.275 (20 seeds)

## Part B: Protocol Calibration (REQUIRED PREAMBLE)

### B1: N=2 Collision-Sparse CLTS Evaluation
- Using existing VICReg-only checkpoint from iter_029 (best tracking: 36.22 px)
- Environment: PhysicsSandbox(N=2), fewer collisions than N=3
- 3 conditions per seed (5 seeds): surprise-driven, frozen (locus=0), random
- 2000 evaluation steps per condition
- Measure FIRST: random/frozen baseline for tracking error, collision count/100 steps,
  collision attention selectivity (fraction of post-collision steps where colliding channel is attended)
- Report measured random baselines explicitly
- Define data-driven gate formulation: active condition must exceed random baseline by
  a pre-declared margin (e.g., tracking error ≤ random − 1σ, or collision selectivity ≥ random × 1.5)

### B2: Subtle Mass Perturbation Test
- In same N=2 environment, at step 1000: mass of object 0 changes by 1.5× (not 10×)
- Measure: perturbation attention selectivity (fraction of steps post-perturbation where
  changed object's channel is attended)
- Compare surprise-driven vs random baselines

## Mandate Revision (PRE-REGISTERED)
- If ceiling probe SUCCEEDS (ΔR²_color ≥ 0.30): M2 revised from "SFA+VICReg as primary"
  to "Reconstruction+VICReg as primary representation objective, decoder-free constraint
  relaxed as pragmatic compromise. SFA demoted to comparison baseline B1. Surprise readout
  retained via stop-gradient predictor. Future work may explore BYOL-style decoder-free
  alternatives approaching the reconstruction ceiling."
- If ceiling probe FAILS (ΔR²_color ≤ 0.275): "Mean-readout z_dyn architecture is
  insufficient for identity encoding under any tested objective class. M2 revision pending
  architectural redesign. Priority: centroid-gated z_dyn readout (iter_027 Arm A' showed
  directional improvement) or increased d_max."

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
