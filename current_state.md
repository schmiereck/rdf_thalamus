# Current Research State
Phase: ΔR²_color ≥ 0.30 comprehensively falsified; M2 mandate not supported; pivot to behavioral integration needed

## Goal
Achieve a training regime where the NonParametricJEPASpatial encoder collapses on ≤10% of seeds. Ultimately: build a decoder-free, curiosity-driven representation agent with thalamic gating and motor.

## Confirmed (iter_030, agents 30.2 + 30.3)
- ARM 1: All 3 pre-registered integration gates failed (0/3 pass)
  - G1: CLTS-SFA tracking error = 45.09 ± 9.82 px, CLTS-VICReg = 36.22 ± 2.83 px (threshold: 20 px)
  - G2: ALL conditions at 99-100% collision switch rate (ceiling effect — collisions too frequent)
  - G3: ALL conditions at 100% perturbation switch rate (ceiling effect — forced perturbation too aggressive)
  - CLTS-VICReg tracks BETTER than CLTS-SFA (36.22 vs 45.09 px) — identity encoding does not determine tracking quality
  - ARM 1 protocol confounded: G2/G3 uninformative, G1 threshold too tight
- ARM 2: Both D1 and D2 falsified on ΔR²_color ≥ 0.30 gate
  - D1 (batch-level temporal contrastive): ΔR²_color = 0.115, CI lower = 0.007
  - D2 (variance-ramped SFA): ΔR²_color = 0.189, CI lower = 0.074
  - Both: 0% collapse rate across 30 seeds
  - D2 WORSE than iter_029 static sfa_weight=5.0 (0.189 vs 0.275)

## Confirmed (prior iterations, still valid)
- ΔR²_color ≥ 0.30 NOT achieved by ANY decoder-free objective tested (iter_020-030):
  - SFA single-step (sfa=5.0): 0.275 (iter_029)
  - SFA variance-ramped: 0.189 (iter_030)
  - SFA multi-step (k=20,50,100): <0.10 (iter_024)
  - Temporal contrastive (batch-level): 0.115 (iter_030)
  - VICReg-only: 0.045 (iter_029)
- Separate-backbone + mask_dyn_sim=True + coord_vicreg=True: 0% collapse across 100+ seeds
- SHARED BACKBONE: SFA falsified (iter_023-024)
- SEPARATE BACKBONE IS LOAD-BEARING (iter_028)
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027)

## Refuted
- PRE-REGISTERED (iter_030): SFA+VICReg representation supports functional CLTS behavior per G1/G2/G3 gates. FALSIFIED — all gates failed.
- PRE-REGISTERED (iter_030): D1 or D2 achieves ΔR²_color ≥ 0.30 with CI lower ≥ 0.18. FALSIFIED — neither passes.
- M2 MANDATE (goal document): "SFA+VICReg as primary representation objective." Comprehensively not supported by evidence across iter_020-030.

## Best Result
- SFA+VICReg sfa_weight=5.0 on separate backbone: ΔR²_color = 0.275 (iter_029, 20 seeds)
- VICReg-only on separate backbone: ΔR²_color = 0.045, 0% collapse, best tracking (iter_030 ARM 1)
- Centroid MSE ~160 across all arms (moderate; Phase-12 reference: CLTS 85.85, WUP-MDL 57.34)

## NOT Established
- Whether the current representation supports the ACTUAL project goal (curiosity-driven agent with gating + motor) when evaluated with properly calibrated tests
- Whether object-level temporal contrastive (matching channels to physical objects) would significantly outperform batch-level NT-Xent
- Whether reconstruction+VICReg (sml: 83%) would work on this architecture
- Why VICReg-only tracks better than SFA+VICReg in closed-loop evaluation

## In Progress
- None. iter_030 complete.

## Open Questions (ordered by expected value)
1. Should the project accept weak identity encoding (ΔR²≈0.05-0.27) and test whether it matters for the actual project goal via properly-calibrated behavioral tests? This is the central decision.
2. Can a re-run of ARM 1 with collision-sparse environments (N=2, larger arena, or fewer substeps) and subtler perturbations distinguish surprise-driven from random/frozen attention?
3. Should M2 be formally demoted from "SFA+VICReg as primary" to "VICReg-only as primary" in goal.md?
4. Would object-level temporal contrastive (same-object-across-time positive pairs with channel-to-object matching) achieve ΔR²_color ≥ 0.30?
5. Should the project relax the decoder-free constraint to include reconstruction+VICReg as a pragmatic upper-bound reference?
6. Is the ΔR²_color metric fundamentally limited by the mean-over-spatial z_dyn readout, and would a centroid-gated readout improve it?
7. Why does SFA worsen tracking relative to VICReg-only? Does slowness constraint reduce representation responsiveness to sudden events?
