# Current Research State
Phase: Architecture Ceiling Probe — Underpowered, Key Negative Results

## Goal
Determine whether the shared-CNN dual-stream NonParametricJEPASpatial encoder
can encode object identity in z_dyn under any objective, and whether ID-contrastive
is viable as a self-supervised proxy.

## Confirmed (iter_025 v2)
- d_max=16 CAPACITY EFFECT CONFIRMED (iter_025, Arm E): delta_R2_color = 0.14
  achieved by JEPA+VICReg at d_max=16 WITHOUT any identity objective. This
  confirms the iter_023 result (0.137) was a capacity effect, measured not
  post-hoc. The journal's "definitive M2 refutation" must be softened: this
  was always a capacity effect, now measured.
- SUPERVISED COLOR LOSS DOES NOT IMPROVE z_dyn IDENTITY (iter_025, Arm B):
  delta_R2_color = -0.024 under supervised probe, WORSE than control (+0.027).
  The loss converges in training but does not transfer to identity encoding.
- MATCHING DEPENDENCY IS REAL (iter_025, Arm B): 43% disagreement between
  sorted and Hungarian matching on pass/fail. Outcome is matching-dependent;
  falsification claim not earned for Arm B.
- COLLAPSE NOT FIXED BY LOWER LR (iter_025, v2): 30% collapse in control Arm A
  even at lr=3e-4 with gradient clipping (was 40-60% at lr=1e-3 in v1).
- ARM C (CONTRASTIVE) STABLE UNDER BOTH MATCHING SCHEMES: 0% disagreement
  between sorted and Hungarian; both agree on "fail" for all non-collapsed seeds.
  delta_R2_color = -0.028 with 50% collapse.
- EFFECT-SIZE THRESHOLD = 0.10 is defensible as "z_dyn explains ≥10% more
  color variance than z_coord" — not derived from invalid noise floor.

## Refuted / Falsified
- ID-CONTRASTIVE AS STANDALONE OBJECTIVE (Arm C): delta_R2_color = -0.028,
  50% collapse. Both matching schemes agree on fail. H2 is falsified.

## Best Result
- Arm E (d_max=16 JEPA+VICReg control): delta_R2_color = 0.14 (no identity
  objective needed — pure capacity effect)
- Arm A (d_max=8 JEPA+VICReg control): delta_R2_color = 0.027

## In Progress
- None. The architecture ceiling probe is blocked by collapse instability.

## NOT Established (honest assessment)
- Whether the architecture is or is not a bottleneck on identity encoding —
  the experiment is underpowered (30% control collapse). No architecture
  claim is earned.
- Whether fixing collapse would reveal latent identity capacity in z_dyn —
  this requires a stable training regime first.
- Whether a separate z_dyn encoder would solve the problem — this is the
  natural next architectural move but has not been tested.

## Open Questions (ordered by expected value)
1. Why does z_dyn collapse at 30% even with lower LR + gradient clipping?
2. Is the centroid_gated readout the bottleneck (attending at soft-argmax
   positions may not capture identity)?
3. Would a separate z_dyn encoder solve collapse + identity?
4. Would larger batch size improve VICReg stability?
5. Can the d_max=16 capacity effect be leveraged with identity objectives?
6. Should the project pivot to a different encoder architecture?
7. Is collapse a fundamental limitation of the dual-stream architecture?
