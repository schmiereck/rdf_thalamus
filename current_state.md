# Current Research State
Phase: Pass-through environment null; foveated gaze escalation triggered

## Goal
Achieve a training regime where the encoder collapses on ≤10% of seeds. Build a
decoder-free, curiosity-driven representation agent with thalamic gating and motor.
Currently: behavioral benchmark validation — environment redesign phase.

## Confirmed (iter_035)
- **PASS-THROUGH ENVIRONMENT NULL (agent 35.3):** Removing object-object collisions
  does NOT make perception load-bearing for mass estimation. PASSIVE pointer gets
  12.27 valid collisions per object (gate threshold was 3.0). Root cause: in 1D
  bounded arena, a physical pointer inevitably collides with bouncing objects regardless
  of targeting policy.
- **ANALYTICAL CEILING GATE WORKED CORRECTLY:** Caught the fundamental environment-design
  failure before wasting compute on a full experiment. The gate is a useful methodological
  innovation for future benchmark designs.
- **1D FUNDAMENTAL CONSTRAINT:** No 1D physics modification that preserves the pointer
  as a physical entity can make perception load-bearing, because the pointer is always
  "in the way" of bouncing objects on the line.
- **FOVEATED GAZE ESCALATION TRIGGERED:** Per pre-committed escalation in pre-registration,
  the project must pull the foveated-gaze mechanism (goal.md Section 8.2) forward from
  deferred. Partial observation is the principled way to make perception necessary.

## Confirmed (prior iterations, still valid)
- v2 MALRE benchmark: validated as coverage-discrimination test (active-vs-passive gap=0.83)
  but ORACLE-vs-RANDOM gap=0.031 (iter_034)
- ORACLE ≈ RANDOM on behavioral-pivot protocol (iter_033)
- M2 MANDATE: comprehensively falsified iter_023-030
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027); SEPARATE BACKBONE IS LOAD-BEARING (iter_028)
- Mean-pooling z_dyn readout is a structural bottleneck for identity encoding (iter_031)
- Cross-backbone attention coupling causes collapse (iter_032)
- ΔR²_color ≥ 0.30 NOT achieved by ANY objective or architecture tested (iter_020-032)

## Refuted
- Pass-through physics as sufficient environment redesign for making perception load-bearing
- v1 MAPE benchmark: falsified due to pointer-object collision noise sensitivity
- Behavioral-pivot with N=2 selectivity: degenerate (iter_033)

## Best Available
- Representation: VICReg-only, mean-pool, separate backbone (ΔR²≈0.045, 0% collapse)
- Strongest identity: SFA+VICReg sfa=5.0, mean-pool, separate backbone (ΔR²≈0.275, 0% collapse)
- Behavioral benchmark: v2 MALRE (coverage discrimination only, not perception quality)
- Environment insight: 1D bounded arena prevents perception-gated information by collision structure alone

## Open Questions
1. What is the minimum gaze field width that makes perception load-bearing?
2. Should gaze use hard mask or soft attenuation?
3. Does foveated gaze naturally produce ORACLE>RANDOM>PASSIVE on POMLRE?
4. How does the CLTSMotorController EMA confound interact with gaze-directed attention?
5. Can the existing soft-argmax z_coord serve as the gaze direction signal?
6. Full-resolution-at-fixation with downsampled periphery (Section 8.2), or simple binary mask?
7. Should the pointer remain a physical entity in the environment, or become purely observational?
