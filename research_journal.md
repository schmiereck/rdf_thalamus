# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — third pre-registered
    iteration in the diagnostic series. iter_029 produced a **clean
    pre-registered null on the M2 mandate itself**: SFA+VICReg on the
    separate-backbone regime did not reach the declared practical-significance
    threshold (ΔR²_color ≥ 0.30) and showed seed-dependent variance large
    enough that the trend cannot be relied upon. Combined with iter_023–024
    (SFA refuted on shared backbone), the explicit slowness objective is now
    empirically falsified across **both architectural regimes** in this task.
*   **Active Direction (revised):** The M2 mandate ("SFA + pooled VICReg as
    the primary representation objective on z_dyn") came into this project
    as a transferred result from `rdf_thalamus_sml`. That mandate is now
    under empirical challenge in the Thalamus task domain. Two
    interpretations remain open and must be distinguished in iter_030:
    (a) the sml transfer was scoped narrowly to the binary, low-DOF toy and
    does not generalize to the 1D RGB physics environment, or
    (b) the SFA term is correctly oriented but is being out-competed by
    another gradient (the VICReg variance hinge, the coord-stream JEPA
    loss, or both) in a way that prevents slowness from acquiring
    identity structure.
    Until that distinction is made, M2 cannot be treated as either
    validated *or* discarded for this project — it is a contested mandate.
*   **What we now have that is solid:** The **separate-backbone +
    VICReg-only z_dyn** configuration (iter_027 Arm C) has now been
    indirectly stress-tested twice (iter_028 C2 fresh seeds = 0% collapse;
    iter_029 60 runs = 0% collapse). It is the only configuration in the
    Phase-0 sweep that has reached zero collapse without invoking SFA.
    Its empirical content remains narrower than "stable identity encoding":
    the construction-vs-empirical caveat (Gate 1) still applies — VICReg's
    variance hinge directly enforces the std metric used in the collapse
    check, so 0% collapse is *partly* tautological. The genuinely empirical
    part is that ΔR²_color = 0.18 (iter_027 Arm C) > 0.04
    (iter_029 Arm A, VICReg-only with mask_dyn_sim+coord_vicreg) — a
    delta that is real but small.
*   **What we now have that is contested:** the iter_029 Arm B mean
    ΔR²_color = 0.27 trend is a 6.2× ratio over the matched control, but
    σ = 0.577 over 30 seeds is too large to count as established. Higher
    SFA weight (5.0) hurt compared to conservative weight (1.0); hard
    seeds (53, 71) showed no SFA benefit. **Gate 3 (Parameter-Tuning
    Hygiene) is failed**: the result is not stable under ±10% perturbation
    of the SFA weight and is not stable under reseeding. Per the goal
    document, this is "suggestive evidence at best."
*   **Next Priority (iter_030, pre-register tightly):** Disambiguate
    "M2 doesn't transfer to this task" from "M2 is being out-competed."
    Concrete arms:
    - D1: VICReg-only z_dyn separate-backbone (the iter_027 Arm C
      configuration replicated with the union seed bank including hard
      seeds 53, 71), as a re-confirmed anchor.
    - D2: SFA+VICReg with VICReg variance term **ramped down** on z_dyn
      (var_weight from 25 → 5 over training) to test the
      SFA-vs-VICReg competition hypothesis. If SFA only works when
      VICReg is weakened, the M2 transfer fails *because* the variance
      hinge dominates slowness on RGB inputs.
    - D3: Object-tracking-ID contrastive (positive pair = same object
      across time, negative = different object) — an augmentation-free
      identity objective that is *not* slowness. If D3 substantially
      beats D2, M2 should be replaced by an explicit identity-binding
      objective rather than slowness.
    - Pre-register: F1 = D3 mean ΔR²_color ≥ 0.30 across the union seed
      bank with σ such that the lower CI is also ≥ 0.18 (the Arm C
      anchor); F2 = collapse ≤ 10%; F3 = no centroid-MSE degradation
      beyond 110.
*   **Confidence Score:** 45% (down from 50%). One additional structural
    hypothesis (M2-as-stated) is now under empirical challenge — that is
    progress in the falsification sense, but it removes the mandate that
    was holding the Phase-0 plan together. The foundation is narrower
    than it looked; downstream Phase 1+ work is still not ready.

## 2. Strategic Insights & Lessons Learned
*   **EXPLICIT SLOWNESS DOES NOT RELIABLY PRODUCE IDENTITY ENCODING IN
    Z_DYN ON THIS TASK (iter_023–024 + iter_029, CROSS-ARCHITECTURE
    CONVERGENT NULL):** SFA on z_dyn now has two clean pre-registered
    tests on shared backbone (iter_023–024, refuted) and on separate
    backbone (iter_029, F1 not reached). The directional trend in
    iter_029 (6.2× over VICReg-only) is real but seed-dependent and
    sensitive to SFA weight; Gate 3 is failed. Treat as: explicit
    slowness is a *contributing* prior but not a *reliable shaper* of
    identity encoding for this task. This is the first cross-project
    finding that does **not** transfer cleanly from `rdf_thalamus_sml`,
    and the most likely reason is DOF: sml's binary toy had stationary
    object identity by construction, whereas RGB+motion has identity
    cues entangled with appearance variation across frames.
*   **M2 MANDATE IS UNDER EMPIRICAL CHALLENGE, NOT YET DISCARDED:** The
    goal document's M2 says SFA+VICReg is the primary representation
    objective. iter_029 is the first arm to test the *full M2
    configuration* on the separate backbone with the previously
    identified confounds (coord_vicreg, hard seeds) controlled — and
    it did not clear its pre-declared gate. This is not yet sufficient
    to overturn the mandate (Gate 3 failure means the result is
    suggestive), but it does forbid invoking M2 as a settled basis
    for Phase 1. The iter_030 D2/D3 arms are the discriminating tests.
*   **HARD SEEDS (53, 71) ARE A CONSISTENT DIAGNOSTIC, NOT JUST NOISE
    (iter_028 + iter_029, CROSS-ITERATION SYNTHESIS):** Seeds 53 and 71
    collapse under mask_dyn_sim (iter_028 C1, C3) and show no SFA
    benefit (iter_029). The pattern is: whatever the operative
    mechanism is, it fails on these seeds across multiple
    configurations. They are functioning as a stress test. Keep them
    in the union seed bank for iter_030; if a future objective passes
    F1 *including* those seeds, the result is robust.
*   **SEPARATE BACKBONE + VICReg-ONLY IS THE CURRENT BEST FOUNDATION
    (iter_027 Arm C + iter_028 C2 + iter_029 Arm A, INDIRECT
    CONVERGENT EVIDENCE):** 0% collapse across three independent runs
    under this configuration. The construction-vs-empirical caveat
    (Gate 1: VICReg variance ≈ collapse metric) caps how strongly we
    can claim this; ΔR²_color = 0.18 (iter_027 Arm C) is the
    empirical part. This is the working anchor for iter_030.
*   **SEPARATE-BACKBONE STRUCTURAL BENEFIT IS LOAD-BEARING (iter_028,
    CONFIRMED):** The 2×2 table in iter_028 (shared backbone 30%→20%
    collapse vs separate backbone 30%→0% under mask_dyn_sim ON→MASKED)
    establishes that backbone separation provides a real stability
    benefit beyond the loss adjustment alone. This refines the
    iter_027 finding that the shared backbone is not the *primary*
    cause: separating it is not *sufficient* on its own (Arm B still
    collapsed at 30%), but it is *necessary* in combination with loss
    restructuring.
*   **REPEATED PATTERN — TRANSFERRED OBJECTIVES DON'T SURVIVE INTACT
    ON THIS TASK:** sml's SFA result is the second transfer to break
    under the Thalamus task DOF (the first was the implicit assumption
    that VICReg+JEPA would behave the same way on the separate
    backbone as it did on the shared backbone; iter_027 Arm B
    refuted that). The pattern suggests the Thalamus task has a
    qualitatively different gradient landscape than sml's binary
    toy. The "scope of transfer" caveat in Section 1.1 of the goal
    document anticipated this; the data now reinforces it.
*   **PRE-REGISTERED NULLS REMAIN FIRST-CLASS RESULTS (iter_026, 027,
    028, 029, ENFORCED):** Four consecutive iterations have produced
    defensible nulls or partial nulls because they pre-declared their
    falsification criteria. The discipline is producing more
    information per iteration than the prior exploratory regime did.
*   **PRESERVED:** M1 (pooled VICReg) stands and is reinforced; M3
    (fixed dimensionality, GDASR log-only) stands; d_max=16 capacity
    baseline stands; Hungarian-primary matching stands; 20% control-
    collapse power threshold stands; buffer=4000 stands.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (ACTIVE, RE-LOCALIZED):** No
    objective tested so far — JEPA+VICReg, SFA+VICReg on either
    backbone, VICReg-only, mask_dyn_sim — reliably encodes identity
    above ΔR²_color = 0.30 across the union seed bank. The bottleneck
    is now characterized as: "in the dual-stream regime, no
    *slowness-or-prediction-based* objective produces stable identity
    encoding on RGB+motion inputs." This suggests the next move is
    qualitatively different (contrastive identity binding, D3 above),
    not a further sweep of slowness weights.
*   **Architectural-Cause Bottleneck (REVISED, iter_028):** Separate
    backbone is necessary-but-not-sufficient. Reclassified from
    "secondary" to "necessary structural prerequisite."
*   **M2-Transfer Bottleneck (NEW, iter_029):** M2 came in as a
    validated transferred result. It is not surviving cleanly in the
    Thalamus task. Tracked as an open mandate-revision question for
    iter_030–031.
*   **Variance/Seed-Dependence Bottleneck (NEW, iter_029):** σ = 0.577
    on a primary metric is too large for reliable downstream work.
    Even if a future arm crosses the F1 threshold in mean, an
    unstable result blocks Phase 1+. iter_030 must include a
    variance-stability gate (lower CI also above the anchor).
*   **Diagnostic-vs-Constructive Iteration Loop (CLEARED):** Three
    consecutive pre-registered diagnostic iterations have produced
    actionable nulls and a tighter localization of the bottleneck.
    Protocol is working.
*   **Overclaim Loop (TRACKED, MIXED):** iter_029 executor used
    "directional trend" and "not robust" appropriately, but also
    labeled the 6.2× ratio prominently without immediately flagging
    σ=0.577 — borderline. Manager continues to enforce that
    single-seed and high-variance results are not promoted to
    mandate revisions on their own.
*   **Objective-Swapping Loop (DORMANT, NOW UNDER REVIEW):** Has
    held for three iterations, but the iter_030 D3 arm (contrastive
    identity binding) is the first proposed jump out of the
    slowness/VICReg family. The Manager judgement is that the
    cross-architecture SFA refutation justifies this jump; the
    objective-swap discipline is being broken with cause, not
    drifting.
*   **Buffer-Capacity Confound (TRACKED):** buffer=4000 maintained
    through iter_029. Keep constant in iter_030.

## 4. Alternate Research Paths
*   **iter_030: M2-Transfer Disambiguation (IMMEDIATE PRIORITY,
    THREE-ARM PRE-REGISTERED):**
    - D1: VICReg-only z_dyn, separate backbone (iter_027 Arm C
      anchor, replicated on the union seed bank).
    - D2: SFA + VICReg with VICReg variance ramped down on z_dyn
      (var_weight 25 → 5 over training) — tests whether SFA was
      being out-competed by the variance hinge.
    - D3: Object-tracking-ID contrastive on z_dyn (positive = same
      object across time, negative = different object, anchor =
      Hungarian-matched track) — the first non-slowness identity
      objective.
    - Falsification: F1 = ΔR²_color ≥ 0.30 with lower CI ≥ 0.18 over
      union seed bank; F2 = collapse ≤ 10%; F3 = centroid_MSE not
      worse than 110.
    - Pre-registered prediction: D3 > D2 > D1, with D3 the only arm
      likely to clear F1 robustly. If D2 also clears F1, the
      M2-out-competed hypothesis is supported and the SFA mandate
      can be salvaged with a variance-ramp protocol; if only D3
      clears F1, M2 should be revised to a contrastive identity
      objective in the project mandate.
*   **iter_031 (CONDITIONAL): Either M2 Revision or Augmentation-Free
    Contrastive Foundation:** Conditional on iter_030 outcome.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (PROMOTED
    from CONDITIONAL):** The iter_029 result moves this from
    "candidate if iter_028 refutes Arm C" to "candidate if iter_030
    D3 also fails." BYOL-style identity targets without explicit
    slowness are now a serious contender, not a fallback.
*   **Accept Decoder-Free Constraint Relaxation (LAST RESORT,
    UNCHANGED):** Still last resort.
*   **Multi-Knob Regime Stabilization (DEFERRED):** Unchanged.
*   **Micro-Columns (DEFERRED per semantic caution):** Unchanged. The
    iter_029 high variance on identity encoding is *not* yet
    sufficient justification to invoke imposed micro-column
    disentanglement; D3 must be tried first.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.