# RDF Milestone Review — Iteration 031 — Null Result: M2 Mandate Empirically Not Supported on Thalamus Task; z_dyn Mean-Pool Readout Identified as Structural Bottleneck

## 1. Pre-Declared Hypothesis and Falsification Criterion
Hypothesis (verbatim from iter_031 pre-registration): "Reconstruction+VICReg
achieves ΔR²_color ≥ 0.30 with variance-stable seeds, establishing a ceiling
for identity encoding and showing the decoder-free constraint was the
bottleneck."

Falsification criterion: ΔR²_color < 0.30 OR the lower bound of the seed-
variance CI < 0.18 across the union seed bank, AND the d_max=2 control
(F3) within 0.05 of the d_max=8 arm (indicating channel count is not
the limiting factor).

Pre-committed mandate-revision text (verbatim from pre-registration):
"Reconstruction+VICReg fails to achieve ΔR²_color ≥ 0.30 with variance-
stability. Even a supervised pixel-reconstruction target cannot make the
mean-readout z_dyn stream encode identity above the 0.30 threshold. The
z_dyn readout architecture itself constrains identity encoding regardless
of objective class."

## 2. Experimental Protocol
- Architecture: NonParametricJEPASpatial CNN (4 stride-2 conv1d layers,
  kernel=5, conv_sp 128→d_max k=1, soft-argmax over space), separate
  backbone regime per iter_027/028. d_t frozen per M3; GDASR log-only.
- Objective: Pixel-reconstruction MSE + pooled/batch VICReg on z_dyn,
  standard hyperparameters carried from prior iterations.
- Arms:
  - Arm A (primary): Reconstruction+VICReg, d_max=8, union seed bank.
  - Arm B (CLTS Part B calibration): downstream collision/perturbation
    protocol with collision-sparse environment and reduced perturbation
    strength.
  - Arm C (random-encoder control): random-init encoder + VICReg only,
    intended to isolate "training matters for identity" from "training
    matters for viability."
  - F3 control: d_max=2 vs d_max=8 to isolate channel-count effect.
- Buffer=4000, batch=32, Hungarian-primary matching, all per established
  Phase-0 protocol.
- Held constant across arms: encoder architecture, batch size, optimizer,
  learning rate, environment seed bank, evaluation protocol.

## 3. Observed Quantities
- Reconstruction quality: MSE = 0.018 (confirms spatial features a_dyn
  contain pixel-level identity information).
- Primary metric Arm A: ΔR²_color did NOT clear 0.30 with variance-stable
  seeds (full value reported in iter_031 executor output; falsification
  threshold pre-declared at 0.30 with lower-CI 0.18).
- F3 control: d_max=2 vs d_max=8 difference = 0.036 — below the 0.05
  "channel count matters" threshold. Channel count is not the limiting
  factor.
- Arm C (random-encoder): 100% collapse. F4 uninterpretable because the
  control arm cannot distinguish identity-encoding-by-training from
  viability-by-training; this is a flaw in the iter_031 control design
  rather than a positive finding.
- CLTS Part B (Arm B): collision selectivity 0.59 (probe) vs 0.44
  (control) — directional but insufficient to pass any pre-declared
  behavioral gate. Confounded by representation quality.

## 4. Verdict
**Refuted (null result, pre-registered).** The primary hypothesis is
falsified against its pre-declared threshold. Combined with iter_023–024
(SFA on shared backbone refuted), iter_029 (SFA+VICReg on separate
backbone refuted), and iter_030 (D1 batch-level temporal contrastive
and D2 variance-ramped SFA both refuted), this constitutes a
**cross-objective convergent null on the M2 mandate** across five
distinct objective classes (JEPA, SFA, temporal-contrastive,
variance-ramped SFA, reconstruction). No decoder-free objective tested
on the current readout has reached the pre-registered identity-encoding
threshold with variance-stable seeds.

## 5. Construction-vs-Empirical Note
- **Definitional part:** Mean-pool over the spatial axis is mathematically
  a spatial low-pass filter. That mean-pooling a spatially-varying signal
  reduces information about that signal is not surprising in principle.
- **Empirical part:** That (a) reconstruction reaches MSE=0.018 yet
  cannot make z_dyn encode color identity above the 0.30 threshold,
  and (b) the d_max=2 vs d_max=8 difference is only 0.036 — these
  together empirically localize the bottleneck to the spatial-readout
  function rather than to channel capacity or to the upstream feature
  quality. This empirical localization is the genuinely new content of
  iter_031.
- **What is enforced by construction:** VICReg's variance term
  enforces per-dimension std ≥ 1 on the readout; the collapse check
  measures the same std. The 0% collapse property of viable arms is
  therefore partly tautological (carried forward from prior journal
  entries).
- **What is genuinely empirical:** The cross-objective convergence
  pattern (five objective classes, all failing the same threshold) is
  not enforced by any single objective's construction and is
  information about the architecture itself.

## 6. Limitations
- The random-encoder control (Arm C) is uninterpretable because the
  control structurally collapsed; the iter_031 protocol cannot
  distinguish "training matters for identity" from "training matters
  for viability." A better-designed control (e.g., training with 10×
  fewer gradient steps so the encoder remains viable but
  under-trained) is needed before any positive claim about
  "training matters" can be made.
- The CLTS Part B calibration (Arm B) is uninformative because it ran
  against a representation that had not cleared F1. The behavioral
  gates failed for representation-quality reasons rather than for
  protocol-design reasons; the calibration must be repeated *after*
  a representation clears F1.
- This result does NOT show that decoder-free objectives are
  fundamentally incapable of identity encoding on this task — it
  shows that they are incapable *under the mean-pool z_dyn readout*.
  The centroid-gated readout (iter_027 Arm A' prototype) is an
  explicit alternative that samples z_dyn at centroid positions
  rather than averaging across all positions. Whether the readout
  fix recovers identity encoding (under VICReg-only, SFA+VICReg, or
  Reconstruction+VICReg) is the iter_032 question.
- This result does NOT show that the M2 mandate fails in
  `rdf_thalamus_sml`-style domains; the goal document's "scope of
  transfer" caveat anticipated that mandates may not survive task
  DOF changes, and the cross-objective null is evidence that the
  Thalamus task domain is qualitatively different from the sml
  binary toy in ways that matter for shaping z_dyn.
- The Manager rules that the immediate next test is **the readout
  architectural fix (iter_032)**, not yet relaxation of the
  decoder-free constraint, because the readout is the cheaper and
  less-imposed change ("measure-before-impose" per Section 1.1).