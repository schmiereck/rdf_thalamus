# Current Research State
Phase: Mean-readout z_dyn identified as structural bottleneck; M2 mandate formally falsified

## Goal
Achieve a training regime where the encoder collapses on ≤10% of seeds. Ultimately: build a decoder-free, curiosity-driven representation agent with thalamic gating and motor.

## Confirmed (iter_031, agents 31.2 + 31.3)
- **FUNDAMENTAL ARCHITECTURAL FINDING**: Mean-pooling z_dyn = a_dyn.mean(dim=-1) destroys per-object color identity regardless of training objective. Even Reconstruction+VICReg with perfect pixel reconstruction (MSE=0.018) achieves only ΔR²_color = 0.063 (agent 31.2, 20 seeds, 0% collapse).
- F1 FAILED: mean ΔR²_color = 0.063 < 0.30 threshold
- F2 FAILED: lower 95% CI = -0.013 < 0.18 threshold
- F3 FAILED: Arm A − Arm B = 0.036 < 0.10 (channel count irrelevant — bottleneck is spatial)
- F4 FAILED: Arm C (random encoder) collapsed 100% (cannot distinguish training-for-identity from training-for-viability)
- d_max=2 control (Arm B) ΔR²_color = 0.027, close to d_max=8 — confirms spatial averaging, not channel capacity, is the bottleneck
- Random-encoder control: requires_grad=False on ALL encoder params → 100% collapse → VICReg gradient flow through encoder is load-bearing for representation existence

- **CLTS Protocol Calibration** (agent 31.3): N=2 collision-sparse environment eliminates iter_030 ceiling effects
  - G1 FAILED: surprise-driven tracking (37.61) ≈ random (38.75)
  - G2 FAILED: collision selectivity 0.59 vs random 0.44 (ratio 1.34, < 1.5× threshold)
  - G3 FAILED: perturbation selectivity 0.48 vs random 0.61 (surprise-driven WORSE)
  - Directional collision-sensitivity signal exists but insufficient for pre-registered gates
  - Underlying representation quality (VICReg-only, ΔR²≈0.04) constrains CLTS performance

## Confirmed (prior iterations, still valid)
- ΔR²_color ≥ 0.30 NOT achieved by ANY objective or architecture tested (iter_020-031):
  - SFA single-step (sfa=5.0): 0.275 (iter_029)
  - SFA variance-ramped: 0.189 (iter_030)
  - SFA multi-step (k=20,50,100): <0.10 (iter_024)
  - Temporal contrastive (batch-level): 0.115 (iter_030)
  - VICReg-only: 0.045 (iter_029)
  - Reconstruction+VICReg: 0.063 (iter_031) ← NEW: even supervised reconstruction fails
- Separate-backbone + mask_dyn_sim=True + coord_vicreg=True: 0% collapse across 100+ seeds
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027)
- SEPARATE BACKBONE IS LOAD-BEARING (iter_028)
- Mean-pooling z_dyn readout is the structural bottleneck for identity encoding (iter_031)

## Refuted
- M2 MANDATE (goal document): "SFA+VICReg as primary representation objective." Comprehensively falsified iter_023-030.
- Reconstruction+VICReg as ceiling for identity encoding via mean-readout z_dyn: falsified at ΔR²=0.063 (iter_031).
- Hypothesis that decoder-free constraint was the bottleneck: FALSIFIED — even supervised reconstruction fails through mean-readout.
- Hypothesis that channel capacity (d_max) was the bottleneck: FALSIFIED — d_max=2 ≈ d_max=8 (F3 fail, diff=0.036).

## Best Result
- SFA+VICReg sfa_weight=5.0 on separate backbone: ΔR²_color = 0.275 (iter_029, 20 seeds) — BEST across all objectives
- VICReg-only on separate backbone: ΔR²_color = 0.045, 0% collapse, best tracking (iter_030 ARM 1)
- Reconstruction+VICReg: ΔR²_color = 0.063, recon_MSE = 0.018 (iter_031) — pixels reconstructed well, identity lost in readout
- Centroid MSE ~160 across all arms (moderate; Phase-12 reference: CLTS 85.85, WUP-MDL 57.34)

## NOT Established
- Whether centroid-gated z_dyn readout (iter_027 Arm A' concept) breaks through ΔR²_color ≥ 0.30
- Whether object-level contrastive learning could work with a different readout mechanism
- Whether the reconstruction-trained encoder's spatial features produce better CLTS behavior than VICReg-only
- Whether increasing spatial resolution of a_dyn (from 8 to 16+) helps identity encoding

## In Progress
- None. iter_031 complete.

## Open Questions (ordered by expected value)
1. Will centroid-gated z_dyn readout (sample a_dyn at centroid positions instead of spatial mean) break through ΔR²_color ≥ 0.30? This is the most promising architectural fix — it directly addresses the demonstrated bottleneck.
2. Does the reconstruction-trained encoder produce better CLTS behavior than VICReg-only, even though ΔR²_color is low? The spatial features may carry useful information even if the mean-readout doesn't.
3. Can the N=2 protocol calibration environment + measured baselines from Part B be reused with a better representation to finally pass the CLTS gates?
4. Should the project formally revise M2 to "VICReg-only + centroid-gated readout" as the new primary architecture?
5. Is there a principled way to set ΔR²_color thresholds for centroid-gated readout that accounts for the changed architecture?
6. Does increasing a_dyn spatial resolution (interpolating from 8 to 16 or 32) help the centroid-gated readout?
