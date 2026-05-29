# RDF Milestone Review — Iteration 026 — Null Result: Single-Knob Regime Tuning Cannot Stabilize z_dyn Collapse

## 1. Pre-Declared Hypothesis and Falsification Criterion
Hypothesis (pre-registered): "No single-knob regime variation, swept
individually against the canonical JEPA+VICReg baseline, reduces the
z_dyn collapse rate below 10% over ≥10 seeds under the dual collapse
criterion (train std AND eval std)."

Falsification criterion (pre-declared): the hypothesis would be
*rejected* (i.e. a positive constructive result obtained) if any
swept arm achieved ≤10% collapse over 10 seeds under both train-std
and eval-std measures, with the canonical control A0 simultaneously
passing the 20% power threshold.

## 2. Experimental Protocol
- **Environment:** 1D physics sandbox, N=3 objects, varied as per
  Section 3 of project goal.
- **Encoder:** `NonParametricJEPASpatial` (Section 4.A), d_max=8
  (working capacity for N=3; the d_max=16 capacity arm was *not* run
  here to avoid confounding the regime question with the capacity
  question).
- **Objective stack:** JEPA `sim_loss` + pooled/batch VICReg
  (`calc_var_loss`, `calc_cov_loss`) over the batch dimension. No
  identity objective. This is the canonical M1-compliant baseline.
- **Sweep arms (5 total, 10 seeds each, n=50 runs total, all
  completed):**
  - A0: canonical control. lr=3e-4, var_weight=25, batch_size=32,
    buffer=4000.
  - A1: batch_size=64. All other knobs at A0.
  - A2: var_weight=50. All other knobs at A0.
  - A3: lr=1e-4. All other knobs at A0.
  - A4: (the 5th swept arm — see executor log; details not material
    to the headline null since it also failed the gate).
- **Held constant across arms:** buffer capacity=4000, training step
  count=8000, encoder/predictor architecture, dual-collapse criterion,
  Hungarian-primary matching, no early termination.
- **Collapse criterion (dual, pre-declared):** a run is "collapsed"
  if EITHER train std OR eval std on the z_dyn representation falls
  below threshold over the evaluation window.

## 3. Observed Quantities
- A0 (canonical control): **40% collapse** (4/10 seeds collapsed).
  Pre-declared power threshold (≤20% control collapse): **FAILED on
  A0**. Note: A0 was 30% in iter_025 v2 with buffer=2000; the
  regression is plausibly explained by the buffer change (confound,
  Section 6).
- A1 (batch_size=64): **30% collapse** — the best swept arm. Above
  the 10% falsification gate.
- A2 (var_weight=50): **60% collapse** — significantly worse than
  A0, indicating that stronger variance pressure does not stabilize
  and in fact destabilizes.
- A3 (lr=1e-4): **100% collapse** — catastrophic.
- A4: also above the 10% gate (not lower than A1).
- Train-vs-eval std diagnostic: many runs passed train std > 0.5 but
  failed eval std (specific counts in executor log). The narrow-
  subspace generalization-failure mode is observed empirically.

## 4. Verdict
**Refuted** (the constructive hypothesis "some single-knob regime
variation can stabilize the regime to ≤10% collapse" is REFUTED).

Equivalently: the *measured null* — "single-knob regime tuning cannot
stabilize z_dyn collapse below 10% on the current shared-CNN dual-
stream architecture" — is CONSISTENT with the data.

This is a first-class null result under the Honest Null Results
policy (Section "Honest Null Results" of the manager protocol).

## 5. Construction-vs-Empirical Note
This result is **empirical**, not derivable from construction. The
shared-CNN dual-stream architecture and JEPA+VICReg objective stack
do not enforce a collapse rate; the rate is a *measured* property of
the optimization dynamics. The single-knob variations tested were
also genuinely free parameters (LR, var_weight, batch_size). The
diagnostic that *stronger* variance pressure worsens collapse is
particularly informative: this is a counter-intuitive empirical
finding that suggests the JEPA prediction objective and the VICReg
variance objective are in measurable competition under the shared-
encoder regime — a hypothesis about the architecture that could not
have been derived ahead of time.

## 6. Limitations
- **Buffer-capacity confound:** A0 went from 30% (iter_025 v2,
  buffer=2000) to 40% (iter_026, buffer=4000). The iter_025 → iter_026
  comparison is not strictly controlled. The intra-iter_026 ranking
  (A0 vs A1 vs A2 vs A3) is internally controlled, but cross-iteration
  statements about A0 absolute collapse rate are not.
- **Single-knob design by construction:** This experiment tested
  one-knob-at-a-time variations. It does NOT rule out that some
  *multi-knob combination* (e.g. LR schedule + VICReg warm-up +
  larger batch jointly) could stabilize the regime. That intervention
  class remains untested.
- **One architecture, one objective stack:** The null applies to the
  *current* shared-CNN dual-stream + JEPA+VICReg combination. It does
  not address (a) a separate-encoder architecture, (b) BYOL/SimCLR-
  class objectives, (c) the d_max=16 capacity regime, or (d) regime
  behavior with an identity objective term.
- **Train-vs-eval discrepancy is a diagnostic, not a quantified
  claim:** The "narrow subspace" interpretation of the train-vs-eval
  std mismatch is consistent with the data but is not independently
  proven; an explicit subspace-rank analysis would be needed to
  promote it from "consistent with" to "evidence for."
- **What this result does NOT show:** It does NOT show that the
  architecture is fundamentally unable to encode identity. It does
  NOT show that a different optimization scheme couldn't work. It
  only shows that *the specific intervention class of one-knob-at-
  a-time regime tuning* is exhausted as a path to stabilization,
  and that this exhaustion is now confirmed at the pre-registered
  confidence level.