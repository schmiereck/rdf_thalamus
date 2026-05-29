# Current Research State
Phase: sim_loss_dyn identified as collapse driver; separate backbone refuted

## Goal
Achieve a training regime where the NonParametricJEPASpatial encoder collapses on ≤10% of seeds over ≥10 seeds (dual criterion). Ultimately: build a decoder-free, curiosity-driven representation agent.

## Confirmed (iter_027, agents 27.1 + 27.3)
- SHARED BACKBONE IS NOT THE PRIMARY COLLAPSE CAUSE: Arm B (separate backbone, full JEPA+VICReg) collapses at 30% (3/10), identical to shared-backbone arms (iter_027 27.3). Pre-registered SECOND NULL confirmed.
- SIM_LOSS_DYN IS THE CAUSAL DRIVER: Arm C (separate backbone, VICReg-only on z_dyn, mask_dyn_sim=True) achieves 0% collapse (0/10) across all seeds. The B-vs-C comparison (same architecture 135,608 params, same backbone, only mask_dyn_sim differs) provides controlled evidence that sim_loss_dyn causes collapse (iter_027 27.3).
- VICReg-ONLY z_dyn PRODUCES BEST SEMANTIC ENCODING: Arm C delta_R2_color = 0.1812, highest across all arms. Mean absolute correlation = 0.210, lowest across all arms. Removing sim_loss_dyn IMPROVES identity encoding (iter_027 27.3).
- NO TRAIN-vs-EVAL STD GAP IN ARM C: All 10 seeds have train stds > 0.98 and eval stds > 0.50. The representation is genuinely stable, not narrow (iter_027 27.3).
- CAPACITY CONFOUND ADDRESSED: Arm B and Arm C have identical parameter counts (135,608). Different collapse rates are NOT attributable to capacity (iter_027 27.3).
- READOUT EFFECT: centroid_gated (30% collapse) vs mean (40% collapse) on shared backbone — a +10pp difference suggesting centroid_gated provides slight protection (iter_027 27.3).
- PARAMETER COUNTS: Shared arms ~80K, separate arms ~136K (iter_027 27.3).

## Refuted
- HYPOTHESIS: The shared CNN backbone is the primary structural cause of z_dyn collapse. FALSIFIED by Arm B (separate backbone + full JEPA+VICReg still collapses at 30%) (iter_027 27.3).
- PRIOR ITERATIONS (023-026): Single-knob regime variations, SFA, multi-step SFA, temporal contrastive (NT-Xent), supervised probes, and capacity increases (d_max=16) all failed to reduce collapse below 10%. None identified the sim_loss_dyn pressure as the cause.

## Best Result
- Arm C (separate backbone, VICReg-only on z_dyn): 0% collapse rate (0/10), delta_R2_color=0.18, mean_abs_corr=0.21. First zero-collapse configuration in project history.

## NOT Established
- Whether mask_dyn_sim=True eliminates collapse on the SHARED backbone (the critical isolate — combines two interventions in Arm C)
- Whether VICReg-only z_dyn is sufficient for downstream tasks beyond color encoding
- Whether Arm C's 0% collapse rate holds at longer training (16000+ steps)
- Whether a stop-gradient on z_target_dyn (instead of removing sim_loss_dyn entirely) also prevents collapse

## In Progress
- None. iter_027 complete.

## Open Questions (ordered by expected value)
1. Does mask_dyn_sim=True eliminate collapse on the SHARED backbone? (Critical next test — isolates sim_loss_dyn from separate-backbone effect.)
2. Is VICReg-only z_dyn semantically useful for downstream tasks (causal sensitivity, generalization)?
3. Can stop-gradient on z_target_dyn (instead of full removal) prevent collapse while still training the predictor?
4. What is the gradient magnitude ratio of sim_loss_dyn vs VICReg on z_dyn? (Mechanism analysis.)
5. Does longer training reveal late collapse in Arm C?
6. How does this finding relate to M2's SFA mandate? VICReg-only is a weaker objective than SFA+VICReg.
7. Is the iter_010 DSDT failure explained by the absence of VICReg on the decoupled stream?
