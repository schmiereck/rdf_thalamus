# Current Research State
Phase: Phase 021 Complete — CGIR Hypothesis Falsified, Partial Effect Measured

## Goal
Design and evaluate a neural architecture that achieves hierarchical abstraction without
generative decoders, with SFA+VICReg shaping the identity stream (z_dyn) and soft-argmax
tracking position (z_coord). Current focus: achieving semantic disentanglement (delta_R2_color ≥ 0.10).

## Confirmed
- **CGIR PARTIAL EFFECT (iter_021, 20.2)**: Centroid-gated identity readout shifts
  delta_R2_color from -0.074 (mean-pooling baseline) to +0.050 — a +0.124 absolute
  improvement. This confirms spatial-mean pooling was a contributing factor but not
  the primary cause of the disentanglement failure.
- **CGIR+CCR SYNERGY (iter_021, 20.2)**: CCR amplifies CGIR's effect 6.7×
  (Arm A delta_R2_color=0.050 vs Arm D=0.007). CCR provides spatial decorrelation
  that enables CGIR's channel-specific readout to differentiate objects.
- **POSITION ENCODING HURTS CGIR (iter_021, 20.2)**: Sinusoidal pos encoding reduces
  CGIR effectiveness 4.5× (Arm A=0.050 vs Arm C=0.011). Consistent with iter_013.
- **C4 IDENTITY PROBE FAILS (iter_021, 20.2)**: delta_R2_identity is negative across
  ALL arms, meaning z_coord predicts compound identity (color+size) better than z_dyn.
  The separation is color-vs-position at best, not identity-vs-position.
- **SLOWNESS RATIO PATHOLOGY (iter_020–021)**: Across all SFA arms, z_dyn changes
  MORE than z_coord (ratio >> 1), contradicting SFA's intended effect. SFA does not
  effectively slow z_dyn.
- **ITER_020 NULL RESULT CONFIRMED (iter_021, 20.2)**: Arm B (Mean+SFA+CCR) replicates
  iter_020 Arm A1 results: delta_R2_color = -0.074, MSE = 121.86, 1/5 collapsed.
- **VICReg BATCH-LEVEL WORKS (iter_020–021)**: Collapse rates are low (1/5 for SFA arms),
  confirming M1 mandate. VICReg applied at batch level is effective.
- **JEPA BASELINE HEAVILY COLLAPSED (iter_020)**: JEPA+CCR still has 4/5 collapse,
  confirming M2's demotion of JEPA was correct.

## Refuted / Falsified
- **CGIR AS PRIMARY CAUSE (iter_021, C3)**: The hypothesis that spatial-mean pooling
  is the primary structural cause of semantic disentanglement failure is FALSIFIED.
  CGIR produces a +0.124 shift but does not achieve the 0.10 threshold (0.050 < 0.10).
- **IDENTITY-VS-POSITION SEPARATION (iter_021, C4)**: No arm achieves delta_R2_identity
  ≥ 0.10. Even the best CGIR arm has delta_R2_identity = -0.035. The "identity" label
  for z_dyn is misleading — at most it achieves partial "color readout."

## Best Result
- Arm A (CGIR+SFA+CCR): delta_R2_color = 0.050, MSE = 117.85, 1/5 collapsed.
  This is the best semantic disentanglement result to date, but still fails the
  pre-registered criterion of 0.10.

## In Progress
- Investigating the SFA slowness pathology (ratio >> 1) as a potential deeper root cause.
- Considering alternative objectives (contrastive, color-conditional) to replace or
  supplement SFA.

## Open Questions
1. What is the deeper root cause of semantic disentanglement failure beyond spatial-mean?
2. Can a contrastive objective (instead of SFA) produce identity separation in z_dyn?
3. Is SFA-on-consecutive-frames insufficient because color is already slow (trivially satisfied)?
4. Can stronger architectural interventions (slot attention, color bottleneck) achieve ≥ 0.10?
5. Should the disentanglement be reframed as color-vs-position given C4 failure?
6. Does increasing training steps beyond 5000 help CGIR converge further?
7. What role does the slowness ratio pathology play in blocking disentanglement?
