# Current Research State
Phase: Decision point — 2D gates measured, human go/no-go required

## Goal
Achieve a training regime where the encoder collapses on ≤10% of seeds. Build a
decoder-free, curiosity-driven representation agent with thalamic gating and motor.
Currently: environmental redesign decision point after 1D null chain and 2D gate measurement.

## Confirmed (iter_037)
- **2D GATE-1 PASSES (5/5 seeds):** PASSIVE pointer at center of 64×64 arena accumulates
  only 0.0-1.0 valid collisions/object vs 12.27 in 1D. 2D removes collision inevitability.
  All sanity checks pass (momentum/energy conservation verified, bounds respected, ~20 probes fired).
- **2D GATE-1b FAILS (3/5 seeds):** CV of per-object collision counts ≥0.30 in only 3/5
  seeds. Failure mode: collisions too rare for CV to be stable (0-1 events total per seed
  in most cases). Seeds 7,71 have 0 events; seeds 31,83 have 1 event; seed 53 has 3 events.
  This is a consequence of Gate-1 passing "too well."
- **2D GATE-2 FAILS (1/5 seeds):** CV of per-object probe counts ≥0.50 in only 1/5 seeds.
  CV values cluster around Poisson baseline (~0.39): 0.981, 0.202, 0.374, 0.408, 0.288.
  2D random walk with gaze_radius=8 in 64×64 still produces near-uniform coverage.
- **OVERALL: Path (i) blocked at tested parameterization.** Two of three gates fail.
- **1D NULL FINDING CRYSTALLIZED:** "The 1D × N=3 × 128px sandbox cannot make perception
  behaviorally load-bearing under an ORACLE-vs-RANDOM bracket across four mechanism-distinct
  redesigns" (iter_033-036). Documented as standalone citable finding.
- **OPTION (iii) REJECTED:** Decoder-free relaxation is mis-targeted — the binding constraint
  is environmental, not representational. Reconstruction+VICReg already tested (iter_031).

## Confirmed (prior iterations, still valid)
- Pass-through physics insufficient: PASSIVE gets 12.27 collisions/object (iter_035)
- Foveated gaze insufficient: RANDOM CV 0.36/0.46 vs 0.50 (iter_036)
- v2 MALRE: coverage-discrimination but ORACLE-vs-RANDOM gap=0.031 (iter_034)
- ORACLE ≈ RANDOM on behavioral-pivot protocol (iter_033)
- M2 MANDATE: comprehensively falsified iter_023-030
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027); SEPARATE BACKBONE IS LOAD-BEARING (iter_028)
- Mean-pooling z_dyn readout is a structural bottleneck (iter_031)
- Cross-backbone attention coupling causes collapse (iter_032)
- ΔR²_color ≥ 0.30 NOT achieved by ANY objective or architecture tested (iter_020-032)

## New Structural Insight
- **Gate-1/Gate-1b tension:** In 2D with a static central pointer, if collisions are rare
  enough to pass Gate-1, they tend to be uniformly rare, making Gate-1b unmeasurable.
  This tension may be fundamental to any static-pointer 2D setup and suggests the
  behavioral test design may need to shift from selective attention (which object to
  perceive?) to active navigation (where to go?).

## Best Available
- Representation: VICReg-only, mean-pool, separate backbone (ΔR²≈0.045, 0% collapse)
- Strongest identity: SFA+VICReg sfa=5.0, mean-pool, separate backbone (ΔR²≈0.275, 0% collapse)
- 1D behavioral: NULL — no 1D configuration makes perception load-bearing
- 2D behavioral: Gate-1 passes (collisions reduced), but heterogeneity gates fail

## Decision Required
Human must choose among:
(a) Explore different 2D parameterizations (with new pre-registered gates)
(b) Full 2D rebuild based on Gate-1 alone (~7-10 iters, ~4× FLOPs)
(c) Re-frame deliverable around representation + mechanism findings
(d) Another path not yet identified

## Open Questions
1. Is the Gate-1/Gate-1b tension fundamental to all 2D static-pointer setups?
2. Would active-navigation (not selective-attention) behavioral testing work in 2D?
3. Can different gaze radius/probe budget pass Gate-2, or is Poisson limit insurmountable?
4. Should the project accept the environmental null and re-frame?
5. Could a different behavioral task (not mass estimation) be load-bearing?
6. M2 remains untestable, not falsified
