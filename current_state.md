# Current Research State
Phase: Benchmark validation complete (v2 MALRE validated with caveats)

## Goal
Achieve a training regime where the encoder collapses on ≤10% of seeds. Build a
decoder-free, curiosity-driven representation agent with thalamic gating and motor.
Currently: the behavioral benchmark is validated as a coverage-discrimination test.

## Confirmed (iter_034)
- **v1 MAPE BENCHMARK FALSIFIED (agent 34.2):** Pointer-object collision mass estimates
  (m_i = 10*(-Δv_ptr)/Δv_obj) are extremely noise-sensitive, causing inverted ordering
  (PASSIVE=0.597 < RANDOM=0.999 < ORACLE=1.005). More active probing = worse estimates.
- **v2 MALRE BENCHMARK VALIDATED (agents 34.4/34.6):** All gates pass:
  ORACLE=0.503 < RANDOM=0.534 < PASSIVE=1.333 (MALRE).
  G1 gap=0.830 CI=[0.727,0.939], G2 gap=0.799 CI=[0.530,1.004],
  G3 ordering correct, G4 coverage gap=0.50.
  All sanity checks S1-S5 pass.
- **ORACLE-vs-RANDOM GAP NEGLIGIBLE:** MALRE gap=0.031, ORACLE wins only 3/8 seeds.
  The PASSIVE gap is a coverage artifact (no data → max penalty), not estimation quality.
- **BENCHMARK IS A COVERAGE DISCRIMINATION TEST:** Discriminates active-vs-passive
  exploration, NOT targeted-vs-random within active regime.

## Confirmed (prior iterations, still valid)
- ΔR²_color ≥ 0.30 NOT achieved by ANY objective or architecture tested (iter_020-032)
- Best ΔR²_color = 0.275 (iter_029 Arm B, SFA+VICReg sfa=5.0, separate backbone)
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027); SEPARATE BACKBONE IS LOAD-BEARING (iter_028)
- Mean-pooling z_dyn readout is a structural bottleneck for identity encoding (iter_031)
- Cross-backbone attention coupling causes collapse (iter_032)
- Behavioral-pivot protocol (N=2, post-collision selectivity V-B) is degenerate (iter_033)
- M2 MANDATE: comprehensively falsified iter_023-030
- CLTSMotorController EMA confound: different surprise distributions → different EMA
  calibration → different tracking behavior (iter_033 ORACLE 58px vs LEARNED 33px)

## Refuted
- v1 MAPE benchmark: falsified due to pointer-object collision noise sensitivity
- M2 MANDATE: comprehensively falsified iter_023-030
- Behavioral-pivot with N=2 selectivity: degenerate (iter_033)

## Best Available
- Representation: VICReg-only, mean-pool, separate backbone (ΔR²≈0.045, 0% collapse)
- Strongest identity: SFA+VICReg sfa=5.0, mean-pool, separate backbone (ΔR²≈0.275, 0% collapse)
- Behavioral benchmark: v2 MALRE (coverage discrimination, active-vs-passive gap=0.83)

## Open Questions
1. Can a coverage-time metric (steps to ≥3 pointer-collisions per object) discriminate
   ORACLE from RANDOM? This would complement MALRE.
2. Is the MALRE benchmark's active-vs-passive discrimination sufficient for iter_035,
   or does iter_035 need a metric that discriminates within the active regime?
3. For iter_035, should the LEARNED representation use a custom information-gain
   controller (avoiding CLTSMotorController EMA confound) or CLTSMotorController?
4. Can velocity-prediction MSE on held-out collisions complement MALRE by being more
   sensitive to data quality (better coverage → more diverse training → better prediction)?
5. Is the fundamental issue that N=3 objects in 1D naturally collide too much, making
   active exploration less discriminating than in sparser environments?
6. Should the project accept that the behavioral pivot (using motor behavior to evaluate
   perception) may require a fundamentally different environment or task structure?
