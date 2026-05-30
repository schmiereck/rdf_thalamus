# Current Research State
Phase: Branch (b) triggered — hard-pivot to behavioral evaluation

## Goal
Achieve a training regime where the encoder collapses on ≤10% of seeds. Build a decoder-free, curiosity-driven representation agent with thalamic gating and motor. Currently: the representation quality is insufficient for ΔR²_color ≥ 0.30 regardless of objective or readout architecture; pivot to evaluating the agent on behavioral metrics directly.

## Confirmed (iter_032, agents 32.1 + 32.2)
- **CENTROID-GATED READOUT CAUSES CATASTROPHIC COLLAPSE**: Cross-backbone attention coupling (using p_c from coord backbone to gate dyn backbone readout) produces 100% collapse for K=4 (both E2 and E3, 20/20 seeds each) and 10% collapse for K=1 (E1.5, 2/20 seeds). Mean-pool baseline (E1) remains 0% collapse. This is a distinct failure mode from the sim_loss_dyn driver (iter_027) or mean-pool bottleneck (iter_031).
- **PREDICTION FALSIFIED**: The pre-registered prediction that scalar centroid-sampling (E1.5) would yield ~+0.10 partial gain was falsified. E1.5 performed WORSE than E1 (ΔR²=0.078 vs 0.131) and introduced collapse.
- **ALL 6 GATES FAILED**: F1 (E2 ΔR²≥0.30 on non-collapsed): FAIL (all collapsed). F2 (lower CI≥0.18): FAIL. F3 (collapse≤10%): FAIL (max=1.0). F4 (E2−E1≥0.10): FAIL (-0.015). F5 (E2−E1.5≥0.10): FAIL (0.038). F6: 0.021 (informational only).
- **BRANCH (b) TRIGGERED**: Third convergent signal received → hard-pivot to behavioral evaluation.
- **NOVEL COLLAPSE MECHANISM**: Peaked softmax attention from the coord backbone creates an optimization bottleneck in the dyn backbone — only attended spatial positions receive strong VICReg gradient, leading to degenerate convergence with per_dim_std ~0.3-0.6.

## Confirmed (prior iterations, still valid)
- ΔR²_color ≥ 0.30 NOT achieved by ANY objective or architecture tested (iter_020-032)
- Best ΔR²_color = 0.275 (iter_029 Arm B, SFA+VICReg sfa=5.0, separate backbone)
- VICReg-only mean-pool = ΔR²≈0.045, 0% collapse (iter_029 Arm A)
- Reconstruction+VICReg mean-pool = ΔR²=0.063 (iter_031) — even supervised reconstruction fails through mean-pool
- Mean-pooling z_dyn readout is a structural bottleneck for identity encoding (iter_031)
- Cross-backbone attention coupling is ALSO a structural bottleneck — introduces collapse (iter_032)
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027); SEPARATE BACKBONE IS LOAD-BEARING (iter_028)

## Refuted
- Hypothesis that centroid-gated readout would fix the spatial bottleneck: FALSIFIED (iter_032) — it causes collapse instead
- Hypothesis that scalar centroid-sampling would yield partial ~+0.10 gain: FALSIFIED (iter_032, E1.5 ΔR²=0.078 < E1 ΔR²=0.131)
- M2 MANDATE (goal document): "SFA+VICReg as primary representation objective." Comprehensively falsified iter_023-030.
- Reconstruction+VICReg as ceiling for identity encoding via mean-readout: falsified at ΔR²=0.063 (iter_031)

## Best Available Representations
- **Stable baseline**: VICReg-only, mean-pool, separate backbone (ΔR²≈0.045, 0% collapse)
- **Strongest identity**: SFA+VICReg sfa_weight=5.0, mean-pool, separate backbone (ΔR²≈0.275, 0% collapse, iter_029 Arm B)
- Centroid MSE ~160 across all arms (moderate; Phase-12 reference: CLTS 85.85, WUP-MDL 57.34)

## In Progress
- None. iter_032 complete. Branch (b) triggered.

## Next: Behavioral Pivot Protocol (iter_033, pre-registered)
Per the iter_032 pre-registration Section 6, the behavioral-pivot protocol uses:
- G1 — Tracking: surprise-driven mean tracking error ≤ 0.75 × random baseline (38.75px → threshold = 29.06px)
- G2 — Collision selectivity: surprise-driven ≥ random × 1.5 (ratio) OR surprise-driven − random ≥ 0.20 (absolute Δ)
- G3 — Perturbation selectivity: surprise-driven ≥ random baseline + 0.10 (absolute)
- Environment: N=2 collision-sparse, calibrated in iter_031 Part B
- Representation: best available from iter_029 (SFA+VICReg sfa=5.0)

## Open Questions
1. Which representation should the behavioral pivot use? SFA+VICReg (ΔR²≈0.275) or VICReg-only (ΔR²≈0.045)?
2. Can the behavioral gates (G1-G3) pass with any existing representation?
3. Is the cross-backbone attention collapse inherent to all gating mechanisms, or specific to softmax-gated einsum?
4. Could gradient scaling or warm-up stabilize the centroid-gated readout?
5. Should the project revise its architecture away from the separate-backbone design toward something that natively supports per-object feature extraction?
6. Does increasing spatial resolution (8→16+) help either readout approach?
