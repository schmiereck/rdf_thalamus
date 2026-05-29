# Current Research State
Phase: Phase 23 Complete — SFA Weight Sweep Falsified

## Goal
Design and evaluate a neural architecture achieving hierarchical abstraction
without generative decoders, with SFA+VICReg shaping z_dyn and soft-argmax
tracking position (z_coord). Current focus: finding an objective that makes
z_dyn encode object identity.

## Confirmed
- **SFA GRADIENT PROPAGATES (iter_023, all arms)**: Increasing sfa_weight from
  0.1 to 25.0 monotonically reduces normalized_dyn_var from 0.0086 to 0.0011.
  The 250x gradient imbalance was real; SFA at higher weights IS effective at
  slowing z_dyn.
- **C5 IS STRUCTURALLY IMPOSSIBLE (iter_023, 0/35 seeds)**: z_coord's
  normalized temporal variance (~1e-5) is 2-3 orders of magnitude below
  z_dyn's minimum (~1e-3). This is a metric artifact: soft-argmax centroids
  in [0,127] have O(127^2) spatial variance but O(1) temporal change, making
  the ratio tiny. VICReg's variance floor prevents z_dyn from reaching
  comparable slowness. The C5 criterion can NEVER be satisfied.
- **SLOWNESS DOES NOT PRODUCE IDENTITY ENCODING (iter_023)**: delta_R2_color
  is essentially flat across the entire sfa_weight sweep (0.040-0.064 for
  d_max=8 arms). Making z_dyn slower doesn't make it encode color/identity.
  This is the key falsification of M2's slowness-prior hypothesis.
- **RAMP STRATEGY WORKS FOR STABILITY (iter_023, A6)**: sfa_weight ramp
  0.1->25.0 over 1000 steps achieves 1/5 collapse (vs 2/5 for fixed sfa=25).
  SFA-VICReg gradient conflict is resolvable with proper initialization.
- **d_max=16 BEST FOR COLOR (iter_022-023, consistent)**: delta_R2_color =
  0.137 with d_max=16, sfa_weight=10.0. Best absolute color disentanglement
  but still below the 0.10 improvement threshold.
- **VICReg BATCH-LEVEL WORKS (iter_020-023)**: For K=1 d_max<=16, confirming M1.

## Refuted / Falsified
- **SFA AS IDENTITY ENCODING MECHANISM (iter_023, PRIMARY)**: FALSIFIED.
  Slowness on z_dyn does not produce identity-position separation. delta_R2_color
  improvement over sfa=0.1 baseline is +0.014 at best (threshold: 0.10).
- **COMPOSITE M2 VIABILITY (iter_023)**: FALSIFIED. C5 is never satisfied,
  making the composite criterion trivially impossible.
- **C5 AS SFA EFFECTIVENESS METRIC (iter_023)**: The criterion
  normalized_dyn_var < normalized_coord_var is structurally impossible in this
  architecture. It measures a metric artifact, not SFA effectiveness.

## Best Result
- Arm B (d_max=16, sfa=10.0): delta_R2_color = 0.137, delta_R2_identity =
  -0.027, 2/5 collapsed. Best color disentanglement across all iterations,
  but compound identity still fails and this is NOT an SFA effect.

## In Progress
- None active. The M2 mandate (SFA as primary representation objective) has
  been empirically tested and falsified for this architecture.

## Critical Architectural Insight
The dual-stream design creates a fundamental asymmetry:
- z_coord (soft-argmax centroid): Very low normalized temporal variance by
  construction (wide spatial range, small per-step changes)
- z_dyn (mean/CGIR pooling): Higher normalized temporal variance because
  VICReg forces per-dim std >= 1 (moderate spatial variance) while identity
  features change meaningfully across timesteps

SFA can reduce z_dyn's temporal variation but cannot overcome this structural
gap without violating VICReg's anti-collapse mandate. More importantly, even
when SFA successfully slows z_dyn, it doesn't cause identity encoding — it
just makes z_dyn less informative overall.

## Open Questions
1. What alternative objective would make z_dyn encode identity? Contrastive
   learning, supervised probe loss, or mutual information maximization are
   candidates. This is now the most important question.
2. Should the C5 metric be redefined? Using absolute temporal variance or
   raw delta_magnitude would avoid the spatial-variance normalization artifact.
3. Is identity encoding fundamentally impossible in z_dyn given the shared
   CNN encoder, or would a separate identity encoder succeed?
4. Would longer training (20k+ steps) eventually accumulate enough SFA
   slowness to produce identity separation?
5. Is d_max=24/32 the only viable path to better color disentanglement?
6. Should M2 (SFA as primary representation objective) be abandoned for this
   architecture, and if so, what replaces it?
7. Does the sml validation (SFA+VICReg at 82%) transfer to RGB inputs, or
   is the sml binary-task advantage specific to low-DOF inputs?
