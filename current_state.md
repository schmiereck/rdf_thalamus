# Current Research State
Phase: M2 Definitively Refuted — Pivot to Object-Tracking-ID Contrastive

## Goal
Design and evaluate a neural architecture achieving hierarchical abstraction
without generative decoders, with an objective that makes z_dyn encode object
identity. The M2 mandate (SFA as primary representation objective) has been
empirically falsified across all tested variants.

## Confirmed
- **M2 FALSIFIED (iter_022-024)**: SFA on z_dyn does NOT produce identity-position
  separation. Tested across: CGIR (iter_022), SFA weight sweep 0.1-25.0 (iter_023),
  multi-step SFA k=20,50,100 (iter_024), temporal contrastive NT-Xent (iter_024).
  No variant achieves delta_R2_color ≥ 0.10.
- **MULTI-STEP SFA WORSENS COLLAPSE (iter_024)**: k=20,50,100 all show 100%
  collapse rate (5/5 seeds), worse than single-step SFA (2/5 at sfa=10, iter_023).
  Longer horizons amplify the SFA-VICReg gradient conflict.
- **SFA IS INDISCRIMINATE SMOOTHER (iter_024)**: As k increases, within-trajectory
  and between-trajectory variance drop proportionally (within/between ratio
  narrows from 1.96→1.25). SFA does not selectively preserve identity — it
  suppresses all temporal variation equally.
- **SHUFFLED-FRAME CONTROL CONFIRMS NULL (iter_024)**: delta_R2_color on shuffled
  frames is negative and similar to unshuffled values across all arms, confirming
  the minimal signal is encoder geometry, not SFA-driven semantics.
- **TEMPORAL CONTRASTIVE (NT-Xent) FAILS (iter_024)**: delta_R2_color = -0.013
  at d_max=8. NT-Xent at τ=0.1 fights VICReg, producing high-variance noisy
  representations (within_traj_var=0.630) with no semantic structure.
- **VICReg BATCH-LEVEL WORKS (iter_020-024)**: For K=1 d_max≤16, confirming M1.
- **d_max=16 BEST CAPACITY (iter_022-024)**: Consistent delta_R2_color improvement
  with more channels, but this is a capacity effect, not an objective effect.
- **RAMP STRATEGY HELPS STABILITY (iter_023-024)**: SFA weight ramp 0.1→target
  over 500 steps reduces immediate collapse but cannot prevent it at longer horizons.

## Refuted / Falsified
- **SFA AS IDENTITY ENCODING MECHANISM (iter_022-024, DEFINITIVE)**: Slowness
  on z_dyn does not produce identity-position separation under any tested
  formulation: single-step, multi-step, weight sweep, or temporal contrastive.
- **COMPOSITE M2 VIABILITY (iter_023-024)**: C5 structurally impossible; composite
  criterion never satisfied.
- **MULTI-STEP HORIZON HELPS (iter_024)**: Longer horizons worsen collapse
  rather than enabling slower feature extraction.
- **NT-Xent AS ALTERNATIVE OBJECTIVE (iter_024)**: Fights VICReg, no identity
  signal produced.

## Best Result
- Arm B (iter_024, k=50, d_max=8): delta_R2_color = 0.034, but 5/5 collapsed.
- Arm D (iter_024, contrastive): delta_R2_color = -0.013, 1/5 collapsed.
- Best absolute: iter_023 Arm B (d_max=16, sfa=10.0): delta_R2_color = 0.137,
  but this is a capacity effect, not an objective effect.

## In Progress
- None. The M2 mandate has been definitively tested and refuted.
- Pivot to object-tracking-ID contrastive learning is the next step.

## Open Questions
1. What objective can make z_dyn encode object identity? Object-tracking-ID
   contrastive is the leading candidate.
2. Does the shared CNN encoder fundamentally prevent identity encoding in z_dyn?
3. Can the NT-Xent + VICReg fight be resolved with different temperature or loss?
4. Is the 100% collapse in multi-step SFA caused by the trajectory buffer approach?
5. Should the dual-stream z_coord/z_dyn architecture be abandoned entirely?
6. Does the sml SFA+VICReg advantage simply not transfer to RGB inputs?
7. Would a supervised identity probe loss (using soft-argmax tracking to assign
   per-object labels) work as a direct training signal?
