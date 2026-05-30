# RDF Scientific Pre-Registration

*   **Iteration:** 029
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
On the separate-backbone architecture (SeparateDynEncoder via NonParametricJEPASpatialSeparateDyn),
adding an explicit SFA slowness term ||z_dyn(t) - z_dyn(t-1)||² to the VICReg-only z_dyn objective
improves identity encoding as measured by ΔR²_color from a held-out linear probe, without introducing
collapse on the hard-seed seed bank [7, 17, 31, 53, 71, 83, 97, 113, 127, 149].
Specifically: Arm B (SFA+VICReg, sfa_weight=5.0, mask_dyn_sim=True) will show ΔR²_color > 0.1812
(the VICReg-only baseline from iter_027 Arm C) when computed over non-collapsed seeds, AND will
show collapse rate ≤ 10% on the same seed bank.

## 2. Falsification Criterion
The hypothesis is falsified if EITHER:
(F1) Arm B (SFA+VICReg) shows ΔR²_color ≤ 0.1812 (the VICReg-only baseline) when computed over
     non-collapsed seeds — meaning SFA adds no marginal identity-encoding benefit even on the
     gradient-isolated separate-backbone architecture; OR
(F2) Arm B shows collapse rate > 10% (≥2 of 10 seeds) on the original seed bank — meaning SFA
     destabilizes the previously stable VICReg-only regime.
Either outcome refutes the claim that SFA constructively shapes z_dyn on this architecture.

## 3. Proposed Method
Three-arm experiment on the original 10-seed bank [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]:

Arm A (Control — reuse iter_027 Arm C): Separate backbone, primary_objective="jepa",
  mask_dyn_sim=True, var_weight=25, cov_weight=25, sim_weight=1. This is VICReg-only on z_dyn.
  Known result: 0% collapse, ΔR²_color=0.1812. Data reused from iter_027.

Arm B (SFA+VICReg, primary): Separate backbone, primary_objective="sfa", mask_dyn_sim=True,
  sfa_weight=5.0, var_weight=25, cov_weight=25, sim_weight=1. This adds the SFA slowness term
  to the VICReg-only z_dyn objective, with JEPA prediction as a stop-gradient readout.
  10 new runs on the original seed bank.

Arm C (SFA+VICReg, conservative): Same as Arm B but sfa_weight=1.0.
  10 new runs on the original seed bank.
  
Total new runs: 20 (Arms B and C). Arm A data reused from iter_027.

Training: 8000 steps, batch_size=32, lr=3e-4, d_t=3 (frozen), buffer=4000.
Architecture: NonParametricJEPASpatialSeparateDyn (SeparateDynEncoder + DualStreamPredictor).

Evaluation metrics per run:
- Collapse check: per-dim std of z_dyn (threshold 0.5) on both eval and train batches
- ΔR²_color: held-out linear probe predicting RGB color from z_dyn (identity encoding quality)
- Centroid MSE: Arm F soft-argmax position decoding (spatial encoding quality)
- mean_abs_corr: VICReg health metric on z_dyn
- sfa_loss trajectory: verify slowness term is active and decreasing
- Per-seed outcomes with per-dim std magnitudes for any collapses

Key confounds to report:
- SFA mode sets var_loss_coord=0 and cov_loss_coord=0 (no VICReg on coord stream),
  while the JEPA-mode control gives coord stream full VICReg. This could affect z_coord
  quality but NOT z_dyn quality, so ΔR²_color comparison remains fair.
- "0% collapse" under SFA+VICReg is expected (SFA + VICReg both encourage high std)
  and is NOT the headline metric. The headline is ΔR²_color improvement.

Files to create:
- src/run_phase0_sfa_separate_backbone.py: experiment runner for Arms B and C
  (modeled after src/run_phase0_separate_dyn.py)

Files to reuse (unchanged):
- src/models_separate_dyn.py: NonParametricJEPASpatialSeparateDyn with mask_dyn_sim support
- src/models_dual_stream.py: NonParametricJEPASpatial with SFA mode
- src/environment.py: PhysicsSandbox

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
