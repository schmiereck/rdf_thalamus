# RDF Scientific Pre-Registration

*   **Iteration:** 029
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
On the separate-backbone architecture (SeparateDynEncoder via NonParametricJEPASpatialSeparateDyn),
adding an explicit SFA slowness term ||z_dyn(t) - z_dyn(t-1)||² to the VICReg-only z_dyn objective
is consistent with M2's predicted mechanism for improving identity encoding as measured by ΔR²_color
from a held-out linear probe, without introducing collapse on the union seed bank.

Specifically: Arm B (SFA+VICReg, sfa_weight=5.0, mask_dyn_sim=True, coord_vicreg=True) will show
ΔR²_color ≥ 0.30 (practical-significance threshold) when computed over non-collapsed seeds, AND will
show collapse rate ≤ 10% on the union seed bank.

NOTE: "0% collapse" under SFA+VICReg is expected (SFA + VICReg both encourage high std) and is NOT
the headline metric. The headline is ΔR²_color improvement above the practical-significance threshold.

## 2. Falsification Criterion
The hypothesis is falsified if ANY of the following hold:
(F1) Arm B shows mean ΔR²_color < 0.30 when computed over non-collapsed seeds — meaning SFA does not
     provide a practically significant identity-encoding benefit above the VICReg-only baseline;
(F2) Arm B shows collapse rate > 10% (>2 of 20 seeds) on the union seed bank — meaning SFA
     destabilizes the previously stable VICReg-only regime;
(F3) Arm B shows centroid_mse_mean exceeding the VICReg-only baseline (Arm A) by more than 1σ — meaning
     SFA degrades the spatial readout while potentially improving identity encoding.

The 0.30 threshold is derived from the Manager's instruction: max(0.1812 + 2σ, 0.30). The per-seed
σ from iter_027 Arm C is ≈0.35 (dominated by one outlier), making 0.1812 + 2σ ≈ 0.88 impractical.
The absolute floor of 0.30 is used as the practical-significance threshold. Results between 0.18 and
0.30 are reported as "no detectable marginal benefit."

Language constraint: positive results are reported as "consistent with M2's predicted mechanism,"
not as "demonstrates SFA improves identity."

## 3. Proposed Method
Three-arm experiment on a UNION seed bank of 20 seeds:
  Original seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
  Fresh seeds:   [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]

Arm A (VICReg-only Control): Separate backbone, primary_objective="jepa",
  mask_dyn_sim=True, var_weight=25, cov_weight=25, sim_weight=1.
  This is VICReg-only on z_dyn (the iter_027 Arm C configuration).
  20 NEW runs on the union seed bank.

Arm B (SFA+VICReg, primary): Separate backbone, primary_objective="sfa", mask_dyn_sim=True,
  sfa_weight=5.0, var_weight=25, cov_weight=25, sim_weight=1, coord_vicreg=True.
  This adds the SFA slowness term to the VICReg-only z_dyn objective, with JEPA prediction
  as a stop-gradient readout. Coord-stream VICReg is kept ON.
  20 runs on the union seed bank.

Arm C (SFA+VICReg, conservative): Same as Arm B but sfa_weight=1.0, coord_vicreg=True.
  20 runs on the union seed bank.

Total new runs: 60 (3 arms × 20 seeds).

Training: 8000 steps, batch_size=32, lr=3e-4, d_t=3 (frozen), buffer=4000.
Architecture: NonParametricJEPASpatialSeparateDyn (SeparateDynEncoder + DualStreamPredictor).

Evaluation metrics per run:
- Collapse check: per-dim std of z_dyn (threshold 0.5) on both eval and train batches
- ΔR²_color: held-out linear probe predicting RGB color from z_dyn (identity encoding quality)
- Centroid MSE: Arm F soft-argmax position decoding (spatial encoding quality)
- mean_abs_corr: VICReg health metric on z_dyn
- sfa_loss trajectory: verify slowness term is active and decreasing
- Per-seed outcomes with per-dim std magnitudes for any collapses
- Hard-seed table: explicit tabulation of seeds 53 and 71 outcomes across arms

Key design decisions (Manager's structural fixes):
1. Union seed bank includes both original hard seeds (53, 71) and fresh seeds to disambiguate
   SFA's identity-encoding benefit from hard-seed stabilization.
2. coord_vicreg=True: SFA mode now keeps coord-stream VICReg (var_loss_coord, cov_loss_coord)
   active, fixing the confound where SFA-mode previously zeroed coord VICReg while JEPA-mode
   kept it. This ensures the z_coord backbone receives consistent gradient signal.
3. centroid_mse degradation floor (F3): prevents a "win" on ΔR²_color that silently destroys
   the spatial readout.
