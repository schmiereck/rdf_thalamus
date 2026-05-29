# RDF Milestone Review — Iteration 024 — Null Result: M2 (SFA as Primary Representation Objective) Refuted Across All Slowness Formulations

## 1. Pre-Declared Hypothesis and Falsification Criterion
**Hypothesis (verbatim from iter_024 pre-registration as recorded in the iteration output):**
"Multi-step SFA (k=20, 50, 100) and temporal contrastive (NT-Xent) both fail to produce
identity encoding; M2 definitively refuted across all slowness formulations."

**Falsification criterion (carried forward from iter_022–023, applied consistently):**
Any tested objective is considered to support M2 if it achieves
`delta_R2_color ≥ 0.10` on the linear color probe over z_dyn, with collapse rate ≤ 1/5.
An objective is considered refuted if it does not clear this threshold under its declared
hyperparameters with the standard seed set.

Anticipated outcome (also declared in advance): a "clean double null" would justify
pivoting to object-tracking-ID contrastive in iter_025.

## 2. Experimental Protocol
*   **Encoder:** `NonParametricJEPASpatial` (Section 4.A), unchanged. d_max=8 unless noted;
    d_t frozen at 3 (M3); GDASR log-only.
*   **Input:** 1D RGB sandbox, N=3 objects, mandatory environmental variation (Section 3).
*   **Batch / training:** batch=32 with pooled/batch-level VICReg (M1, preserved).
*   **Arms tested:**
    - Multi-step SFA, k ∈ {20, 50, 100}, sfa_weight=10.0 with ramp 0.1→10.0 over 1000 steps.
    - Temporal contrastive (NT-Xent on consecutive frames as positive pairs) + VICReg.
*   **Controls:**
    - Shuffled-frame control (breaks temporal order while preserving frame distribution)
      to separate encoder-geometry signal from temporally-driven signal.
    - Invariance-vs-discrimination diagnostic: decomposition of probe variance into
      within-trajectory and between-trajectory components.
*   **Seeds:** 5 per arm (standard set).
*   **Reference points:**
    - iter_023 single-step SFA at sfa_weight=10.0: 2/5 collapse, delta_R2_color = 0.064.
    - iter_022–023 best d_max=16 arm: delta_R2_color = 0.137 (channel-capacity effect,
      not an SFA effect).
    - Falsification threshold: delta_R2_color ≥ 0.10.

## 3. Observed Quantities
*   **Multi-step SFA (k ∈ {20, 50, 100}, sfa_weight=10.0):**
    - Collapse rate: 5/5 seeds collapsed (all k values).
    - delta_R2_color: ≤ 0.034 across all arms.
    - within-traj vs. between-traj variance: proportionally reduced — indiscriminate
      smoothing, no identity-specific signal extracted.
*   **Temporal contrastive (NT-Xent + VICReg):**
    - Collapse rate: 1/5 collapsed (4/5 survive).
    - within_traj_var = 0.630 (high), indicating high-variance noise rather than structure.
    - No detectable color-probe signal above shuffled-frame control.
*   **Shuffled-frame control:** minimal probe signal present, attributable to encoder
    geometry (CNN + soft-argmax) alone, independent of temporal objective.

All measured values fall well below the 0.10 falsification threshold. The 0.034 maximum
is a factor of ~3 below threshold and a factor of ~4 below the d_max=16 channel-capacity
reference.

## 4. Verdict
**Refuted.** The hypothesis stated multi-step SFA and temporal contrastive would both fail
to produce identity encoding. Both did fail under the pre-declared protocol, with
delta_R2_color ≤ 0.034 (multi-step SFA) and no semantic structure (NT-Xent) against the
pre-declared threshold of 0.10. Combined with iter_022–023 results on single-step SFA and
the SFA weight sweep, this refutes the broader claim that slowness-prior objectives can
shape z_dyn into an identity-encoding stream under the current architecture (shared CNN +
dual-stream soft-argmax + batch VICReg).

The vacuum/null control (shuffled-frame) gave the predicted null, confirming the small
residual probe signal is encoder-geometric, not objective-driven.

## 5. Construction-vs-Empirical Note
This is genuinely empirical. The result is not derivable from the architecture in advance:
it was plausible that longer temporal horizons would supply SFA with the discriminative
gradient it lacks at k=1 (this is the standard SFA-literature mechanism, validated on
other tasks). The finding that the opposite occurs — k=20–100 amplifies, rather than
resolves, the SFA-VICReg gradient conflict — is novel information about how this
specific architecture responds to temporal-coherence objectives. Similarly, the
NT-Xent + VICReg incompatibility was not predictable from either component's individual
behavior.

What follows from construction (and should not be claimed as discovery): that z_coord's
normalized temporal variance is orders of magnitude below z_dyn's. This is a property of
the soft-argmax readout on a [0,127] spatial range, identified in iter_023 as a
definitional identity, and is unaffected by iter_024's results.

## 6. Limitations
*   **Architectural scope:** This refutes slowness-prior objectives **on the shared-CNN
    dual-stream backbone with soft-argmax centroid head**. It does not refute slowness as
    a representation prior in general, nor on a separate-encoder z_dyn, nor on a
    non-centroid readout.
*   **Hyperparameter scope:** Multi-step SFA was tested at sfa_weight=10.0 with the same
    ramp schedule as iter_023's best single-step result. Lower weights (e.g., 1.0, 3.0)
    were not swept; the 100% collapse result therefore strictly applies to the asymmetric
    "long horizon × high weight × short ramp" regime. The conclusion that "multi-step SFA
    fails" is robust to this only if a brief lower-weight sensitivity check confirms (see
    iter_025 pre-registration note).
*   **Threshold scope:** The 0.10 delta_R2_color threshold is itself a working choice, not
    a theoretically derived bound. Some downstream tasks may be served by smaller
    improvements. The refutation is specifically of "meaningful identity encoding"
    as operationalized by this threshold.
*   **Decoder-free constraint:** This result does not address whether allowing a small
    decoder on z_dyn would restore identity encoding. That remains an open architectural
    question, deliberately deferred.
*   **Seed reuse:** The 0.10 threshold has been applied across iter_021–024 with
    overlapping seed sets. The probability of a small-N false negative is non-zero; the
    iter_025 work should use a fresh seed batch for cross-validation of this refutation
    claim before it is treated as settled.