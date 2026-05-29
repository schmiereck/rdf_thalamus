# RDF Milestone Review — Iteration 023 — Null Result: SFA Slowness Prior Does Not Produce Identity Encoding

## 1. Pre-Declared Hypothesis and Falsification Criterion
**Hypothesis:** Increasing sfa_weight from 0.1 to parity with var_weight (25.0) would activate SFA, cause z_dyn to become slower than z_coord (C5 criterion: normalized_dyn_var < normalized_coord_var), and produce identity-position separation (delta_R2_color >= 0.10).

**Falsification criterion:** If delta_R2_color remains below 0.10 across the entire sfa_weight sweep, the hypothesis is refuted — SFA slowness does not produce identity encoding regardless of gradient strength.

**Secondary criterion (C5):** If normalized_dyn_var cannot drop below normalized_coord_var due to structural constraints, C5 is a metric artifact, not a valid test of SFA effectiveness.

**Tertiary criterion:** If sfa_weight=25.0 causes training collapse (SFA vs VICReg gradient conflict), test ramped initialization (0.1 → target over 1000 steps).

## 2. Experimental Protocol
- **Grid:** 1D physics sandbox, 128 RGB pixels, 3 objects with varying sizes/colors/masses.
- **Encoder:** NonParametricJEPASpatial (existing CNN backbone, Section 4.A).
- **Arms:**
  - A1-A6: d_max=8, CGIR+SFA+CCR, sfa_weight ∈ {0.1, 1.0, 5.0, 10.0, 25.0, 25.0-ramp}
  - B: d_max=16, CGIR+SFA+CCR, sfa_weight=25.0-ramp
- **d_t:** Frozen at 3 (M3). GDASR in log-only mode.
- **Training:** 5000 steps, batch_size=32, 5 seeds per arm.
- **VICReg:** var_weight=25.0, cov_weight=25.0 (batch-level, M1).
- **Other losses:** ccr_weight=1.0, sim_weight=25.0 (JEPA readout, stop-gradient).
- **Control:** Arm A1 (sfa_weight=0.1) replicates iter_022 Ctrl configuration.
- **Evaluation:** Same suite as iter_022: delta_R2_color, delta_R2_identity, normalized temporal variance, collapse rate, centroid MSE.

## 3. Observed Quantities
- **z_dyn normalized temporal variance (dose-response):**
  - sfa=0.1: 0.0086
  - sfa=1.0: ~0.005
  - sfa=5.0: ~0.003
  - sfa=10.0: ~0.002
  - sfa=25.0 (direct): ~0.0013 (4/5 collapse)
  - sfa=25.0 (ramp): 0.0011 (1/5 collapse)
- **z_coord normalized temporal variance:** ~1e-5 across ALL arms (structural constant).
- **delta_R2_color (d_max=8 arms):** 0.040–0.064 across the sweep. No monotonic trend with sfa_weight. Flat.
- **delta_R2_color (d_max=16, sfa=25 ramp):** 0.137 (consistent with iter_022 result of 0.130).
- **delta_R2_identity:** Negative across all arms (-0.027 to -0.055). SFA weight has no effect.
- **Collapse rate:** sfa=25.0 direct: 4/5; sfa=25.0 ramp: 1/5; all other arms: 0-1/5.

## 4. Verdict
**Refuted** on the primary criterion. Increasing sfa_weight successfully reduces z_dyn temporal variation (confirming SFA gradient propagates and the 250× imbalance was real), but delta_R2_color remains flat at 0.040–0.064 for d_max=8 — well below the 0.10 threshold. Making z_dyn slower does not make it encode object identity.

**C5 is a metric artifact** (definitional identity, not empirical finding): z_coord's normalized temporal variance (~1e-5) is structurally 2-3 orders of magnitude below z_dyn's reachable range, due to the soft-argmax centroid geometry in [0,127] combined with VICReg's std≥1 enforcement on z_dyn. C5 can never be satisfied in this architecture.

**Ramp strategy validated** (tertiary criterion): 1/5 collapse at sfa=25 ramp vs. 4/5 at sfa=25 direct. SFA-VICReg coexistence is achievable with proper initialization.

## 5. Construction-vs-Empirical Note
- **SFA gradient propagation (dose-response):** Genuinely empirical. The monotonic decrease in z_dyn variance with increasing sfa_weight confirms the gradient-imbalance diagnosis and validates that SFA is mechanically functional.
- **C5 impossibility:** Definitional identity. The soft-argmax centroid geometry (spatial variance O(127²), temporal change O(1²)) combined with VICReg's std≥1 constraint on z_dyn makes C5 structurally unsatisfiable. This is derivable from the construction, not a discovery about the system's behavior.
- **Slowness ≠ identity encoding:** Genuinely empirical and the most important finding. The SFA theoretical prediction (slowness → identity encoding) does not hold in practice on this task with consecutive-frame (k=1) SFA. On already-static features, the slowness prior provides no discriminative gradient — all representations produce equally small SFA loss, and VICReg determines the equilibrium.

## 6. Limitations
- This null result applies ONLY to consecutive-frame (k=1) SFA. Multi-step SFA (k>>1) remains untested and may provide the discriminative gradient that k=1 lacks.
- The experiment used a single architecture (NonParametricJEPASpatial CNN). Whether slowness produces identity encoding in other architectures (e.g., the deferred hierarchical pyramid, Section 8.6) is unknown.
- The delta_R2_color metric measures linear probe accuracy, which may underestimate nonlinear identity encoding. However, the flatness across the sweep (no trend) makes this unlikely to change the conclusion.
- Training was limited to 5000 steps. Whether longer training would eventually produce identity encoding is unknown, but the flat dose-response (no trend with sfa_weight) makes this unlikely.
- The JEPA readout (sim_weight=25.0) was active in all arms. Whether it interferes with SFA through shared encoder gradients is an open question (journal Open Question 7).
- This result does NOT show that slowness is irrelevant for representation learning in general — only that k=1 SFA on already-static features provides insufficient discriminative signal for identity encoding in this specific setup.