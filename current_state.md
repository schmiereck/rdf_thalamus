# Current Research State
Phase: SFA+VICReg on separate backbone fails F1 gate; M2 mandate not supported

## Goal
Achieve a training regime where the NonParametricJEPASpatial encoder collapses on ≤10% of seeds over ≥10 seeds. Ultimately: build a decoder-free, curiosity-driven representation agent.

## Confirmed (iter_029, agents 29.1 + 29.2)
- SFA+VICReg (sfa_weight=5.0) on separate backbone shows ΔR²_color=0.2749, FAILS the
  pre-registered F1 gate (0.30 threshold). The directional trend (6.2× over VICReg-only)
  is present but not robust.
- ZERO COLLAPSE across all 60 runs (3 arms × 20 seeds, union bank). SFA does not
  destabilize the separate-backbone architecture (F2 PASS).
- Centroid MSE is identical between SFA and VICReg-only (159.85 vs 159.83, F3 PASS).
  SFA does not degrade spatial readout.
- FRESH SEEDS: Arm B ΔR²_color = 0.3576 (above 0.30 threshold)
- ORIGINAL SEEDS: Arm B ΔR²_color = 0.1921 (below 0.30 threshold)
- HARD SEEDS 53, 71: Do not collapse under any arm, but show poor ΔR²_color
  under Arm B (seed 53: -0.0484, seed 71: 0.0413). Higher SFA weight HURTS
  hard seeds compared to conservative weight (1.0).
- CONSERVATIVE ARM C (sfa=1.0): ΔR²_color=0.1428, centroid_mse=167.13. Lower
  SFA weight produces weaker identity trend but better hard-seed behavior.
- COORD VICREG FIX: All arms now use coord_vicreg=True, eliminating the
  confound from iter_027 where SFA mode zeroed coord-stream VICReg.
- SFA LOSS IS ACTIVE: Mean final sfa_loss = 0.1408 (Arm B), confirming the
  slowness objective is optimized during training.

## Confirmed (prior iterations, still valid)
- SHARED BACKBONE: SFA falsified on shared backbone (iter_023-024), across all
  slowness formulations (single-step, multi-step, temporal contrastive)
- SEPARATE BACKBONE + VICReg-ONLY: 0% collapse (iter_027 Arm C)
- SEPARATE BACKBONE IS LOAD-BEARING: mask_dyn_sim alone on shared backbone
  insufficient (20% collapse on hard seeds, iter_028)
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027 Arm B vs C comparison)

## Refuted
- HYPOTHESIS (pre-registered): SFA+VICReg on separate backbone achieves ΔR²_color ≥ 0.30.
  FALSIFIED by Arm B mean ΔR²_color = 0.2749 (iter_029)
- M2 MANDATE (goal document): "SFA+VICReg as the primary representation objective."
  Not supported by evidence across either architecture.

## Best Result
- Arm B on fresh seeds only: ΔR²_color=0.3576, 0% collapse — but seed-dependent
- VICReg-only on separate backbone: 0% collapse across all tested seed banks (iter_027, 029)
- Centroid MSE ~160 across all arms (moderate; Phase-12 reference: CLTS 85.85, WUP-MDL clean 57.34)

## NOT Established
- Whether the SFA trend (0.27 vs 0.04) would reach significance with more seeds
- Whether reconstruction+VICReg would perform better on this architecture
- Whether the low ΔR²_color (0.04 for VICReg-only) indicates the representation
  is too weak for downstream tasks (attention, motor)
- Whether a higher sfa_weight (25.0+) would produce stronger effects or just
  more variance

## In Progress
- None. iter_029 complete.

## Open Questions (ordered by expected value)
1. Should M2 be revised from "SFA+VICReg as primary" to "VICReg-only as primary, SFA as optional auxiliary"? This is the central architectural decision.
2. Can the separate-backbone + VICReg-only architecture (0% collapse, but ΔR²_color=0.04) serve as the stable foundation for Phase 1 (passive observation), or is the identity encoding too weak?
3. Would reconstruction+VICReg (sml: 83% accuracy) work on the separate-backbone architecture, and would it require accepting a lightweight decoder (breaking the decoder-free principle)?
4. Is ΔR²_color the right metric? Many seeds show negative values (z_dyn encodes color WORSE than z_coord), suggesting the stream design may need fundamental revision.
5. Why do fresh seeds show much better ΔR²_color under SFA? Is this a training dynamics difference or a seed-distribution artifact?
6. Should the project accept that identity encoding in z_dyn is weak under all decoder-free objectives tested so far, and pivot to evaluating whether this matters for downstream tasks?
7. Can a higher sfa_weight (25.0) or a different SFA formulation (e.g., SFA on concatenated [z_coord, z_dyn]) produce a stronger effect?
