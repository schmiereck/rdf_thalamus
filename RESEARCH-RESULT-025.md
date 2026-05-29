# RDF Milestone Review — Iteration 025 (v2) — Null Result: Ceiling Probe Underpowered; Capacity Effect Confirmed

## 1. Pre-Declared Hypothesis and Falsification Criterion
Pre-declared (iter_025 v2, addressing five Research Manager criticisms of v1):
- **Primary hypothesis:** A supervised identity objective applied to z_dyn (Arm B)
  will produce delta_R2_color ≥ 0.10 above a matched control (Arm A) under a
  Hungarian-primary matching rule, with ≥10 seeds and a control collapse rate
  ≤20% (power threshold).
- **Falsification rule (pre-declared):** If Arm B does *not* exceed control by
  ≥0.10 under Hungarian matching, AND control collapse rate is ≤20%, the
  hypothesis "z_dyn can encode identity under a strong supervised signal in the
  shared CNN" is refuted. If control collapse exceeds 20%, the experiment is
  declared underpowered and no falsification claim is earned.
- **Auxiliary hypothesis (Arm E):** d_max=16 improvement is a capacity effect.
  Test: Arm E (d_max=16, JEPA+VICReg, no identity objective) should reach
  delta_R2_color comparable to prior d_max=16 identity-objective runs.

## 2. Experimental Protocol
- **Arms:** A (control, training regime only), B (supervised color regression
  on z_dyn), C (ID-contrastive), E (d_max=16 JEPA+VICReg, no identity term).
- **Training:** LR = 3e-4 (down from 1e-3 in v1), gradient clipping enabled,
  8000 steps (up from 5000), 10 seeds (up from 5).
- **Matching:** Hungarian-primary (pre-declared rule), sorted reported as
  sensitivity check.
- **Metric:** delta_R2_color (linear probe R² on z_dyn for object color,
  minus matched control).
- **Threshold:** 0.10 (defensible effect-size criterion, not derived from the
  invalid v1 noise floor).
- **Power gate (pre-declared):** Control collapse rate ≤20%.

## 3. Observed Quantities
- **Arm A (control) collapse rate:** 30% (3/10 seeds). **Above the 20% power
  threshold.** Reduced from 40-60% in v1, but insufficient.
- **Arm B (supervised) delta_R2_color:** -0.024 (worse than control).
- **Arm A (control) delta_R2_color:** +0.027.
- **Arm C (contrastive) delta_R2_color:** -0.028, with 50% collapse.
- **Arm E (d_max=16, no identity objective) delta_R2_color:** 0.14.
- **Matching disagreement (Arm B):** Hungarian vs sorted differ on pass/fail
  for 3/7 non-collapsed seeds (43% disagreement). Per pre-declared rule, this
  invalidates any falsification claim that depends on the matching procedure.
- **Supervised training loss (Arm B):** converges to near-zero (i.e., the
  supervised signal is being absorbed by the network — but not into z_dyn in
  a transferable form).

## 4. Verdict
- **Primary hypothesis: UNRESOLVED.** Control collapse (30%) exceeded the
  pre-declared 20% power threshold. Per the experiment's own falsification
  rule, no claim about "architecture refutes identity encoding" is earned.
  The Arm B < Arm A observation is suggestive but conditional on first
  stabilizing the regime.
- **Auxiliary hypothesis (capacity effect): CONSISTENT WITH HYPOTHESIS.**
  Arm E reached delta_R2_color = 0.14 with no identity objective. This is the
  cleanest possible demonstration that the d_max=16 improvement observed in
  iter_023 was attributable to representational capacity, not to the
  objective being tested.

## 5. Construction-vs-Empirical Note
- **Capacity-effect confirmation (Arm E) is empirical:** delta_R2_color
  depends on what the network actually learns; the 0.14 value is not fixed
  by construction. A different objective or LR could have failed to reach
  it. The result is a genuine measurement of what a d_max=16 JEPA+VICReg
  representation contains.
- **Underpowered-control verdict is structural:** the 30% > 20% comparison is
  just arithmetic against a pre-declared rule. It does not require new
  measurement to assert.
- **Arm B < Arm A is empirical but inconclusive:** the comparison is
  measured, but the survivor-bias confound under 30% collapse means the
  observed sign could flip in a stable regime. Do not promote.

## 6. Limitations
- The experiment cannot disambiguate objective-bottleneck from
  architecture-bottleneck for the identity-encoding question. That was its
  stated goal, and it failed to meet its own power requirement.
- The supervised arm's convergence-in-training with non-transfer-to-z_dyn is
  a real observation but admits multiple interpretations (information leaks
  into z_coord; collapse perturbs settling; matching procedure noise).
  Re-running under a stable regime is required.
- The 0.14 capacity baseline applies to JEPA+VICReg at d_max=16 with the
  current encoder; it should not be assumed transferable to other objectives
  without re-measurement.
- No claim is made about ID-contrastive or separate-encoder paths — they
  were not tested.
- **What is needed next:** a single iteration (iter_026) whose only job is to
  drive control collapse to ≤10% over ≥10 seeds. Without that substrate,
  no further objective falsification is interpretable.