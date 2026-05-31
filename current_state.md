# Current Research State
Phase: Meta-escalation triggered after F5 failure

## Goal
Achieve a training regime where the encoder collapses on ≤10% of seeds. Build a
decoder-free, curiosity-driven representation agent with thalamic gating and motor.
Currently: behavioral benchmark validation — meta-escalation decision required.

## Confirmed (iter_036)
- **F5 FIRED IN BOTH ARMS (agent 36.4):** Foveated gaze with GAZE_RADIUS=8 and
  ghostly pointer does NOT create sufficient coverage heterogeneity under RANDOM
  policy. Arm A CV=0.36, Arm B CV=0.46, both below 0.50 threshold.
- **FULL BRACKET NOT RUN:** Per pre-registered protocol, arms failing the CV gate
  are excluded from the bracket. Both arms failed, so no bracket was executed.
- **STRUCTURAL EXPLANATION:** With 3 objects in 128 pixels and gaze radius 8
  (covering 12.5% of the arena), random gaze motion encounters all objects with
  ~37.5% probability per probe, distributing coverage too evenly for targeted
  probing to improve upon.
- **IMPLEMENTATION CORRECTNESS:** Five pre-registered bugs were fixed (Δv
  across-substep, CV gate metric, sanity checks S1-S6, ORACLE gaze radius
  check, probe tracking). Three additional minor bugs fixed during execution.

## Confirmed (prior iterations, still valid)
- Pass-through physics insufficient: PASSIVE gets 12.27 collisions/object (iter_035)
- v2 MALRE: coverage-discrimination test (active-vs-passive gap=0.83) but
  ORACLE-vs-RANDOM gap=0.031 (iter_034)
- ORACLE ≈ RANDOM on behavioral-pivot protocol (iter_033)
- M2 MANDATE: comprehensively falsified iter_023-030
- SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027); SEPARATE BACKBONE IS LOAD-BEARING (iter_028)
- Mean-pooling z_dyn readout is a structural bottleneck (iter_031)
- Cross-backbone attention coupling causes collapse (iter_032)
- ΔR²_color ≥ 0.30 NOT achieved by ANY objective or architecture tested (iter_020-032)

## Chain of Environment-Design Failures
- iter_033: ORACLE ≈ RANDOM (motor protocol doesn't discriminate)
- iter_034: MALRE coverage gap strong but ORACLE-RANDOM gap negligible
- iter_035: Pass-through physics insufficient (PASSIVE gets too many collisions)
- iter_036: Foveated gaze insufficient (RANDOM coverage too even)

## Best Available
- Representation: VICReg-only, mean-pool, separate backbone (ΔR²≈0.045, 0% collapse)
- Strongest identity: SFA+VICReg sfa=5.0, mean-pool, separate backbone (ΔR²≈0.275, 0% collapse)
- Behavioral benchmark: v2 MALRE (coverage discrimination only)
- Environment: No 1D environment design produces ORACLE > RANDOM discrimination

## Meta-Escalation Options (pre-committed)
(i) 2D environment redesign — may create sufficient spatial structure for
    coverage heterogeneity and meaningful behavioral validation
(ii) Re-frame deliverable around representation + thalamic gating claims
    without behavioral validation — accept that the 1D sandbox cannot
    validate perception-driven behavior
(iii) Revisit decoder-free constraint — allow reconstruction to shape the
    representation, which sml showed matches VICReg performance

## Open Questions
1. Is the 1D sandbox itself the structural confound for behavioral validation?
2. Can a 2D environment create sufficient coverage heterogeneity?
3. Should the project re-frame around representation quality rather than behavior?
4. Would a smaller gaze radius pass the CV gate, or is 1D the fundamental limit?
5. Is mass estimation from elastic collisions too noise-sensitive for any 1D benchmark?
6. Can thalamic gating + motor claims be validated on a different task?
7. Should the decoder-free constraint be revisited given sml's reconstruction results?
