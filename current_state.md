# Current Research State
Phase: mask_dyn_sim on shared backbone falsified; hard-seed pattern identified

## Goal
Achieve a training regime where the NonParametricJEPASpatial encoder collapses on ≤10% of seeds over ≥10 seeds (dual criterion). Ultimately: build a decoder-free, curiosity-driven representation agent.

## Confirmed (iter_028, agents 28.1 + 28.2)
- MASK_DYN_SIM ALONE ON SHARED BACKBONE IS INSUFFICIENT: C1 (shared backbone, mask_dyn_sim=True) collapses at 20% (2/10: seeds 53, 71). Pre-registered F1 falsified. The post-forward loss subtraction does not by itself prevent z_dyn collapse (iter_028 28.2).
- HARD-SEED PATTERN: Seeds 53 and 71 collapse consistently across C1 and C3 (same hard seeds, different weight perturbation). Seed 53 also collapsed in D0; seed 71 did NOT collapse in D0 but did collapse in C1/C3. The collapse mode under mask_dyn_sim is more severe (std ~0.01) than under full JEPA+VICReg (std ~0.45) (iter_028 28.2).
- SEED-DEPENDENCE: C2 (fresh seeds, same config as C1) achieved 0% collapse (0/10). The original seed bank contains hard seeds; the fresh seed bank does not. The result is seed-dependent, not deterministic (iter_028 28.2).
- NOT WEIGHT-ROBUST: C3 (perturbed weights 27.5/27.5/1.1) also showed 20% collapse with the same hard seeds (53, 71). F3 triggered (iter_028 28.2).
- H2 RELATIVE GATE PASSED: C1 shows dramatically better semantic encoding than D0 when it does not collapse (ΔR²_color 0.23 vs 0.05, mean_abs_corr 0.52 vs 0.99). Removing sim_loss_dyn improves representation quality when collapse does not occur (iter_028 28.2).
- SEPARATE-BACKBONE IS LOAD-BEARING: The complete 2×2 table shows 30%→20% (shared backbone, sim_dyn ON→MASKED) vs 30%→0% (separate backbone, sim_dyn ON→MASKED). The separate-backbone architecture provides a structural benefit beyond the loss adjustment alone (iter_027 + iter_028).
- C2 BEST SEMANTIC ENCODING: C2 (fresh seeds, mask_dyn_sim) achieved ΔR²_color=0.514, mean_abs_corr=0.435, centroid_mse=99.68 — the best across all arms (iter_028 28.2).
- ZERO TIMEOUTS: All 40 runs completed with zero timeouts. Results are fully interpretable (iter_028 28.2).

## Confirmed (prior iterations, still valid)
- SHARED BACKBONE IS NOT THE PRIMARY COLLAPSE CAUSE (iter_027 Arm B, separate backbone + full JEPA+VICReg still collapses at 30%)
- VICReg-ONLY z_dyn ON SEPARATE BACKBONE: 0% collapse (iter_027 Arm C)
- ITER 023-026: Single-knob regime variations, SFA, multi-step SFA, temporal contrastive, supervised probes, and capacity increases all failed to reduce collapse below 10% on the shared backbone

## Refuted
- HYPOTHESIS (pre-registered): mask_dyn_sim alone on the shared backbone prevents z_dyn collapse. FALSIFIED by C1 (20% collapse, iter_028 28.2). F1 triggered.
- PRIOR: Shared backbone as primary structural cause (iter_027 Arm B)

## Best Result
- C2 (shared backbone, mask_dyn_sim, fresh seeds): 0% collapse, ΔR²_color=0.514, mean_abs_corr=0.435. But this is seed-dependent — the original seed bank shows 20% collapse under the same config.
- iter_027 Arm C (separate backbone, VICReg-only z_dyn): 0% collapse, ΔR²_color=0.18, robust across the original seed bank.

## NOT Established
- Whether the separate-backbone architecture provides gradient isolation (beyond capacity) that stabilizes VICReg
- Why seeds 53 and 71 are hard seeds under mask_dyn_sim but seed 71 survives under D0
- Whether increasing VICReg weight compensates for the loss of sim_loss_dyn on hard seeds
- Whether a stop-gradient on z_target_dyn achieves the same effect as mask_dyn_sim
- Whether C2's 0% collapse holds at longer training (16000+ steps)

## In Progress
- None. iter_028 complete.

## Open Questions (ordered by expected value)
1. Can the separate-backbone architecture be shown to provide gradient-isolation (not just capacity) that stabilizes VICReg? (This is the key remaining structural question.)
2. Why do seeds 53 and 71 collapse under mask_dyn_sim but seed 71 survives under D0? (Different collapse modes?)
3. Can increasing VICReg var_weight on the shared backbone compensate for sim_loss_dyn removal on hard seeds?
4. Does a stop-gradient on z_target_dyn (preserving predictor training) achieve the same collapse prevention as mask_dyn_sim?
5. What is the gradient magnitude ratio of sim_loss_dyn vs VICReg on z_dyn during early training on hard vs easy seeds?
6. Does C2's 0% result hold at longer training?
7. How does this finding relate to M2's SFA mandate? VICReg-only is a weaker objective than SFA+VICReg, and SFA was already refuted (iter_023-024).
