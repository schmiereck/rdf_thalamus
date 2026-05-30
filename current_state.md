# Current Research State
Phase: Branch (c) confirmed — behavioral-pivot protocol invalidated

## Goal
Achieve a training regime where the encoder collapses on ≤10% of seeds. Build a decoder-free, curiosity-driven representation agent with thalamic gating and motor. Currently: the behavioral-pivot strategy is invalidated because the primary metric (post-collision selectivity V-B on N=2) does not discriminate perception quality.

## Confirmed (iter_033, agents 33.5 + 33.6)
- **BRANCH (c) FIRED (definitive, v3 full-physics ORACLE):** ORACLE selectivity_vb = 0.5044, RANDOM = 0.5043, gap = 0.0001 (< 0.10 threshold). Perfect perception with a near-perfect physics predictor does NOT improve post-collision attention selectivity over random locus selection.
- **ORACLE PREDICTOR BUG (v1/v2):** The original linear-extrapolation ORACLE predictor had (a) a timing bug where prev_positions was set from current info rather than saved before the step, and (b) cannot model wall bounces or elastic collisions, producing surprise ~146k-164k vs ~1-3 for LEARNED. Fixed in v3 with a full physics simulator.
- **METRIC CEILING FOR N=2:** With 2 objects, every collision involves both objects, so random attention has ~50% chance of matching the max-velocity-change object. The metric's random baseline IS its ceiling.
- **ORACLE TRACKING PARADOX:** ORACLE has HIGHER tracking error (58.08 px) than LEARNED-VICReg (33.26 px), despite perfect perception. The surprise distribution mismatch (clean zero between collisions for ORACLE vs. noisy baseline for LEARNED) causes different EMA calibration and attention switching.
- **VICReg PERTURBATION SELECTIVITY:** LEARNED-VICReg has the best perturbation selectivity (0.6075 vs 0.4308 for ORACLE, 0.4658 for RANDOM), suggesting noisy perception can provide useful behavioral signals for this secondary metric.
- **SFA STABILITY:** LEARNED-SFA has the most stable selectivity across seeds (std=0.011 vs 0.131 for VICReg, 0.068 for RANDOM, 0.015 for ORACLE).

## Confirmed (prior iterations, still valid)
- ΔR²_color ≥ 0.30 NOT achieved by ANY objective or architecture tested (iter_020-032)
- Best ΔR²_color = 0.275 (iter_029 Arm B, SFA+VICReg sfa=5.0, separate backbone)
- Mean-pooling z_dyn readout is a structural bottleneck for identity encoding (iter_031)
- Cross-backbone attention coupling causes collapse (iter_032)
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027); SEPARATE BACKBONE IS LOAD-BEARING (iter_028)

## Refuted
- M2 MANDATE: comprehensively falsified iter_023-030
- Behavioral-pivot strategy with current protocol: INVALIDATED by branch (c) (iter_033)
- Reconstruction+VICReg ceiling for identity encoding via mean-readout: falsified (iter_031)
- Centroid-gated readout: FALSIFIED — causes collapse (iter_032)

## Best Available Representations
- **Stable baseline**: VICReg-only, mean-pool, separate backbone (ΔR²≈0.045, 0% collapse)
- **Strongest identity**: SFA+VICReg sfa_weight=5.0, mean-pool, separate backbone (ΔR²≈0.275, 0% collapse)

## Next Steps
The behavioral-pivot protocol must be redesigned before another attempt. Options:
1. Increase N to 3+ objects so collisions don't involve all objects
2. Change primary metric (time-to-attend, surprise-ratio, or tracking-error differential)
3. Shorten attention cooldown (15→5 steps) to allow faster attention switching
4. Use a different environment or protocol entirely

## Open Questions
1. Can a redesigned behavioral metric discriminate perception quality?
2. Is the CLTSMotorController's 15-step attention cooldown appropriate?
3. Why does ORACLE have worse tracking error than LEARNED-VICReg?
4. Does noisy perception provide useful behavioral signals (perturbation selectivity)?
5. Should the project redesign the motor protocol or try a different environment (N=3)?
