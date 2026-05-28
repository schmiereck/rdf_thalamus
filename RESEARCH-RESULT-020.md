# RDF Milestone Review — Iteration 021 — Null Result: CGIR Insufficient for Semantic Disentanglement

## 1. Pre-Declared Hypothesis and Falsification Criterion
**Hypothesis (phase-021):** Replacing spatial-mean readout with centroid-gated identity readout (CGIR) on z_dyn will increase delta_R2_color by ≥ 0.10, resolving the semantic disentanglement failure (delta_R2_color < 0) observed since iter_020.
**Falsification criterion (C3):** delta_R2_color improvement < 0.10 between Arm A (CGIR) and Arm B (mean baseline).

## 2. Experimental Protocol
- **Arm A (CGIR):** z_dyn readout via centroid_gated (uses z_coord soft-argmax weights to extract channel-specific identity features). SFA+VICReg on z_dyn, CCR on z_coord, d_t=3 frozen, RGB-only input, lambda=25.
- **Arm B (Baseline):** z_dyn readout via spatial mean (standard). All other parameters identical to Arm A.
- **Arm C:** CGIR + positional encoding (sinusoidal position channel appended to RGB input).
- **Arm D:** CGIR without CCR (no spatial decorrelation on z_coord).
- **Identity Probe (C4):** Regresses z_dyn against color and size independently to decompose "identity."
- **Environment:** 1D physics sandbox, N=3 objects, varying colors/sizes/masses, elastic collisions. 5 seeds per arm.
- **Metrics:** delta_R2_color (R²_color_z_dyn - R²_color_z_coord), slowness ratio (z_coord_temporal_var / z_dyn_temporal_var), centroid decoding MSE.

## 3. Observed Quantities
- **Arm A (CGIR) delta_R2_color:** +0.050 (shifted from negative, but below 0.10 threshold)
- **Arm B (Mean baseline) delta_R2_color:** -0.074 (negative, confirming baseline failure)
- **CGIR directional effect:** +0.124 (positive shift, statistically consistent but below criterion)
- **Arm C (CGIR + pos encoding) delta_R2_color:** +0.011 (pos encoding hurts CGIR)
- **Arm D (CGIR, no CCR) delta_R2_color:** Negative (CCR essential for CGIR)
- **C4 Identity Probe:** Color partially captured by z_dyn; size NOT captured. "Identity" framing overstated.
- **Slowness ratio:** >> 1 across ALL arms (z_dyn is NOT slower than z_coord despite SFA objective).

## 4. Verdict
**Refuted.** CGIR produces a directional +0.124 shift in delta_R2_color but fails the pre-declared 0.10 threshold. The spatial-mean bottleneck is a contributing factor (~60% of the directional shift) but not the primary cause of semantic disentanglement failure.

## 5. Construction-vs-Empirical Note
The CGIR effect (+0.124) is genuinely empirical — it was not derivable from the architecture alone and required the controlled A/B comparison. The slowness ratio >> 1 finding is also genuinely empirical and not derivable from construction; one would expect SFA to make z_dyn slow, but it empirically doesn't. The position-encoding-hurts finding reproduces the iter_013 result under a different objective, strengthening it as a cross-objective empirical regularity.

## 6. Limitations
This null result does NOT show that spatial-mean readout is irrelevant — it contributes +0.124. It shows that resolving the semantic disentanglement gap requires addressing factors beyond the readout mechanism. The critical unresolved question is why SFA fails to make z_dyn slow (slowness ratio >> 1). This may indicate: (a) loss-weighting imbalance where VICReg overwhelms SFA gradients, (b) a fundamental limitation of frame-pair SFA on already-static signals, or (c) an implementation issue. Until this is resolved, no architectural intervention targeting z_dyn semantics can be properly evaluated, because the SFA foundation itself may be non-functional.