# Current Research State
Phase: Path (ii) re-frame — behavioral validation declared not tractable

## Goal
Achieve a training regime where the encoder collapses on ≤10% of seeds. Build a
decoder-free, curiosity-driven representation agent with thalamic gating and motor.
Currently: pivoting to re-frame after environment-design null chain (iter_033-038).

## Confirmed (iter_038)
- **2D NAVIGATION GATE-1 PASSES (5/5 seeds):** Mean per-object probe-event count
  under RANDOM navigation ≤ 3.0 in all seeds (range 0.67–2.67, overall mean 1.40).
  Navigation design successfully decouples event generation from passive collision
  inevitability.
- **2D NAVIGATION GATE-2 PASSES (5/5 seeds):** Per-object probe-event CV ≥ 0.50
  in all seeds (range 0.707–1.414). Random walk spatial clustering produces
  meaningful per-object heterogeneity under uniform-random object placement.
- **2D NAVIGATION GATE-1b FAILS:** Mean CV = 1.025 ≥ 0.30 (PASS) but std CV =
  0.320 > 0.25 (FAIL). Per-seed CVs [0.817, 1.414, 0.771, 0.707, 1.414] are
  bimodal — heterogeneity exists but is not reproducible across seeds. The
  interaction between random object placement and random walk trajectory dominates.
- **OVERALL: FAIL** — Gate-1b fails, pre-committed exit rule FAIL branch applied.
- **BEHAVIORAL-VALIDATION STRATEGY: NOT TRACTABLE** within project scope. This
  is the fifth consecutive failed environment design (iter_033-038), each failing
  at a different structural constraint.
- **S7 diagnostic note:** Static pointer under uniform-random placement has mean
  collision count 174.33 (vs 0.33 under iter_037 segment-based placement). This is
  expected and does not affect gate evaluation.

## Confirmed (prior iterations, still valid)
- **1D NULL CHAIN (iter_033-036):** "The 1D × N=3 × 128px sandbox cannot make
  perception behaviorally load-bearing under an ORACLE-vs-RANDOM bracket across
  four mechanism-distinct redesigns." Documented as standalone citable finding.
- **2D STATIC-POINTER OPPOSITION (iter_037):** Gate-1 passes (rare collisions)
  but Gate-1b/Gate-2 fail (too few events for meaningful CV; Poisson-limited
  coverage uniformity). Gate-1 and Gate-1b are in structural opposition.
- **M2 MANDATE:** Comprehensively falsified iter_023-030 (SFA on z_dyn does not
  produce identity encoding). M2 remains untestable, not falsified.
- **SIM_LOSS_DYN IS THE COLLAPSE DRIVER (iter_027):** Removing it reduces collapse
  from ≥30% to ≤10% on shared backbone.
- **SEPARATE BACKBONE IS LOAD-BEARING (iter_028):** Eliminates collapse regardless
  of sim_loss_dyn.
- **Mean-pool z_dyn readout is a structural bottleneck (iter_031):** ΔR²_color
  < 0.10 for ALL decoder-free objectives tested.
- **Cross-backbone attention coupling causes collapse (iter_032):** 100% for K=4,
  10% for K=1.
- **ΔR²_color ≥ 0.30 NOT achieved by ANY objective or architecture (iter_020-032).**

## Best Available
- Representation: VICReg-only, mean-pool, separate backbone (ΔR²≈0.045, 0% collapse)
- Strongest identity: SFA+VICReg sfa=5.0, mean-pool, separate backbone (ΔR²≈0.275, 0% collapse)
- Behavioral validation: NULL — no configuration (1D or 2D) makes perception
  load-bearing across five consecutive environment designs (iter_033-038)

## Environment Null Chain Summary (iter_033-038)
1. iter_033: ORACLE ≈ RANDOM on behavioral-pivot protocol (1D, N=3)
2. iter_034: v2 MALRE — coverage-discrimination but ORACLE-vs-RANDOM gap=0.031
3. iter_035: Pass-through physics — passive pointer gets 12.27 collisions/object
4. iter_036: Foveated gaze — CV=0.36/0.46 under RANDOM, both below 0.50
5. iter_037: 2D static pointer — Gate-1 passes, Gate-1b/Gate-2 fail (opposition)
6. iter_038: 2D navigating pointer — Gate-1/Gate-2 pass, Gate-1b fails (std CV=0.320>0.25)

## Next Phase: Path (ii) Re-frame (iter_039 scope)
The behavioral-validation strategy is declared not tractable within project scope.
The project pivots to a set of six pre-registered falsifiable claims:

1. **M1 pooled-VICReg necessity:** Without batch-level VICReg, the separate-
   backbone architecture collapses in ≥4/5 seeds. Gate: train without VICReg;
   check has_collapsed or per-dim std < 5.0.

2. **iter_028 collapse mechanism:** sim_loss_dyn is the causal driver of z_dyn
   collapse on the shared backbone; the separate backbone eliminates collapse
   regardless. Gate: (a) shared backbone without sim_loss_dyn collapses ≤1/5
   seeds; (b) separate backbone collapses 0/5 seeds.

3. **iter_031 mean-pool bottleneck:** Mean-pool spatial readout limits ΔR²_color
   < 0.10. No alternative readout achieves ΔR² ≥ 0.30 without collapse. Gate:
   test centroid-gated or max-pool; falsified if any achieves ΔR² ≥ 0.30 in
   ≥4/5 seeds.

4. **Structural-ceiling-gate primitive:** The analytical ceiling gate (measuring
   PASSIVE information access to predict bracket viability) is a valid research
   primitive. Gate: apply to a fresh task with known ground-truth discriminability;
   falsified if prediction is wrong.

5. **1D environmental null (documented, standing):** The four-iteration null
   chain (iter_033-036) is a standing negative finding — preserved as a citable
   result, not re-tested.

6. **2D navigation null (from iter_038):** Gate-1b std(CV) = 0.320 > 0.25 under
   2D navigation with uniform-random object placement. Parameterization and
   measured values documented as a data point.

## Open Questions
1. Is the Gate-1b failure fundamental to ALL random-walk + random-placement designs?
2. Can the six re-frame claims be tested within reasonable scope?
3. Should the structural-ceiling-gate primitive be applied to a task outside Thalamus?
4. Does the mean-pool bottleneck admit ANY spatial readout achieving ΔR² ≥ 0.30?
5. M2 remains untestable — how should this be stated in the re-frame?
6. Is the five-iteration null chain itself a sufficient deliverable for the environmental track?
