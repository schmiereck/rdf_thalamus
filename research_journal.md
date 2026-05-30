# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — concluded with a
    **comprehensive cross-objective falsification of the M2 mandate**. After
    iter_023–024 (SFA refuted on shared backbone), iter_029 (SFA+VICReg
    refuted on separate backbone, σ too large), iter_030 (D1 batch-level
    temporal contrastive and D2 variance-ramped SFA both refuted), and now
    iter_031 (Reconstruction+VICReg ceiling probe also refuted), **no
    decoder-free objective tested on the current architecture has reached
    ΔR²_color ≥ 0.30 with variance-stable seeds**. iter_031 contributes the
    decisive structural diagnosis: the bottleneck is not the *objective
    class* but the *z_dyn readout itself* — mean-pooling across the spatial
    axis is a low-pass filter that destroys per-object color identity
    because color varies *spatially* (object 0 at pos 30, object 1 at pos
    70). The d_max=2 vs d_max=8 control (Δ=0.036) confirms channel count
    is irrelevant; the bottleneck is purely spatial.
*   **Active Direction (pivoted):** M2 is no longer the operative mandate.
    The project must pivot in one of two directions, and the Manager rules
    that the cheaper/more conservative path is taken first:
    - **Direction A (architectural fix, IMMEDIATE PRIORITY):** Replace
      mean-pool readout with a **centroid-gated readout** (iter_027 Arm A'
      prototype): sample z_dyn *at* each centroid position z_coord rather
      than averaging over all positions. This preserves per-object color
      because the sample comes from the spatial location of the object.
      This is the **measure-before-impose** path: it changes the readout,
      not the objective, and lets us re-test the existing objectives
      (VICReg-only, SFA+VICReg, Reconstruction+VICReg) under a
      non-destructive readout before declaring the objectives themselves
      failed.
    - **Direction B (pivot to behavioral evaluation, parallel):** Open
      Question 1 from the factual state — accept the current weak identity
      encoding (ΔR² ≈ 0.05–0.27) and test whether it matters for the actual
      project goal via properly-calibrated CLTS gates (collision-sparse
      environment, subtler perturbations, looser tracking threshold).
      iter_030/031 ARM 1 was confounded by collision/perturbation ceiling
      effects; that protocol is fixable.
*   **What is now solid:**
    - **Cross-objective null on the M2 mandate** is a first-class result
      (per goal document Section 9, failure modes are first-class
      deliverables). It is the cleanest pre-registered cross-iteration null
      the project has produced.
    - **The mean-pool readout is the structural bottleneck**, demonstrated
      empirically (d_max control) and consistent with a clear mechanistic
      story (spatial averaging destroys spatially-varying identity cues).
    - **VICReg-only on separate backbone** remains the working
      non-collapsing baseline. Centroid-MSE ≈ 160 across arms (vs Phase-12
      CLTS 85.85, WUP-MDL 57.34) is the gap that downstream work must
      narrow.
*   **What is now contested / disconfirmed:**
    - **M2 ("SFA+VICReg as primary representation objective") is
      empirically not supported in this task domain.** Per the goal
      document's own "scope of transfer" caveat, this is the second
      transfer from `rdf_thalamus_sml` that fails to survive intact on
      RGB+motion inputs. Formal mandate-revision text required in
      iter_032 before any Phase 1 work.
    - **The ΔR²_color ≥ 0.30 threshold may itself be unreachable under
      the mean-pool readout**, regardless of objective. This is iter_031's
      empirical contribution.
*   **Next Priority (iter_032, pre-register tightly):** Centroid-gated
    readout architectural fix as a single-variable change. Arms:
    - E1: Mean-pool readout + VICReg-only (the working baseline, control).
    - E2: Centroid-gated readout + VICReg-only.
    - E3: Centroid-gated readout + SFA+VICReg (re-runs M2 under the fixed
      readout — answers whether SFA was failing on its merits or because
      the readout downstream of it was destroying its signal).
    - Pre-register: F1 = E2 or E3 ΔR²_color ≥ 0.30 with lower CI ≥ 0.18
      across the union seed bank; F2 = collapse ≤ 10%; F3 = centroid_MSE
      not degraded beyond 110; **F4 = E2 vs E1 paired-seed Δ > 0.10
      (the readout fix is necessary for the gain, not coincidence).**
*   **Confidence Score:** 38% (down from 45%). The M2 falsification is
    progress in the falsification sense, but the iter_031 finding that the
    readout architecture caps achievable ΔR² regardless of objective means
    the project's foundation is narrower than even the post-iter_030
    assessment implied. Phase 1+ work remains blocked until a single
    configuration clears F1 with variance stability.

## 2. Strategic Insights & Lessons Learned
*   **THE Z_DYN READOUT IS THE STRUCTURAL BOTTLENECK, NOT THE OBJECTIVE
    (iter_031, ARCHITECTURAL FINDING):** Mean-pool over the spatial axis
    is a spatial low-pass filter; per-object identity cues that vary across
    spatial positions are destroyed at the readout regardless of how well
    the upstream features encode them. Evidence: (a) reconstruction MSE
    = 0.018 confirms spatial features `a_dyn` *do* contain pixel-level
    identity information; (b) ΔR²_color still fails F1 under supervised
    reconstruction; (c) d_max=2 vs d_max=8 control Δ=0.036 isolates the
    bottleneck as spatial, not channel-level. Mechanistic story is clean
    and consistent across the d_max sweep.
*   **M2 MANDATE IS EMPIRICALLY FALSIFIED FOR THIS TASK (iter_029–031,
    CROSS-OBJECTIVE CONVERGENT NULL):** Four pre-registered diagnostic
    iterations (023–024, 029, 030, 031) across two backbone regimes and
    five objective classes (JEPA, SFA, temporal-contrastive,
    variance-ramped SFA, reconstruction) all fail to reach ΔR²_color ≥
    0.30 with variance-stable seeds. The convergence across objective
    classes is what makes this a structural rather than an
    objective-selection failure. Per Section 9 of the goal document, this
    is a first-class result; per the "Honest Null Results" framing of the
    Manager prompt, it warrants a milestone report.
*   **CONTROLS THAT COLLAPSE STRUCTURALLY ARE UNINFORMATIVE (iter_031,
    PROTOCOL LESSON):** F4 (random-encoder control) was supposed to
    isolate "training matters for identity" from "training matters for
    viability." The random encoder collapsed 100% under VICReg, so the
    two effects cannot be separated. Lesson for future controls: if a
    control arm needs to be trained to a viable representation to be
    interpretable, do not use a frozen/random encoder as that control.
    Use instead a deliberately weakened training signal (e.g., 10× fewer
    gradient steps) that still produces a non-collapsed representation.
*   **DOWNSTREAM PROTOCOL CALIBRATION MUST FOLLOW REPRESENTATION
    VIABILITY (iter_031, PROTOCOL LESSON):** The CLTS Part B calibration
    ran in parallel with the representation probe and was confounded by
    representation quality. Gates failed for the wrong reason. Going
    forward, downstream behavioral evaluation (CLTS, motor) must be
    gated on a representation that clears F1 first; running both in the
    same iteration wastes the calibration.
*   **CARRIED FORWARD (still valid):**
    - M1 (pooled/batch VICReg) stands and is reinforced (random-encoder
      collapse shows variance hinge is load-bearing for *existence*).
    - M3 (fixed dimensionality, GDASR log-only) stands.
    - Separate backbone + mask_dyn_sim + coord_vicreg = 0% collapse,
      load-bearing combination (iter_028).
    - Hungarian-primary matching, d_max=16 capacity baseline,
      buffer=4000, 20% control-collapse power threshold all stand.
    - Pre-registered nulls are first-class results — four consecutive
      iterations now confirm the discipline produces more information
      than exploratory regime.
    - Hard seeds (53, 71) remain in the union seed bank.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (LOCALIZED to READOUT, iter_031):**
    Re-classified from "objective class" to "z_dyn readout architecture."
    This is a meaningful localization — the bottleneck moved from "what
    loss do we use" (where five objectives have now failed) to "what
    function maps spatial features to z_dyn" (where the centroid-gated
    readout is a concrete and untested alternative).
*   **M2-Transfer Bottleneck (RESOLVED to FALSIFIED, iter_031):** The
    mandate is empirically not supported for this task. Tracked as
    "mandate revision required in iter_032 pre-registration."
*   **Variance/Seed-Dependence Bottleneck (PERSISTS):** Still active.
    iter_032 F1 must include a variance-stability subclause (lower CI ≥
    0.18).
*   **Diagnostic-vs-Constructive Iteration Loop (RESOLVED):** Five
    consecutive pre-registered diagnostic iterations have produced
    actionable nulls and a tight architectural localization. The protocol
    is mature. iter_032 is the first opportunity to convert a localization
    into a constructive test.
*   **Overclaim Loop (DORMANT):** iter_031 executor used "comprehensive
    architectural null" appropriately, flagged F4 uninterpretability,
    acknowledged the centroid-gated readout is a hypothesis not a
    solution. Discipline holding.
*   **Objective-Swapping Loop (RESOLVED, REASSESSED):** Swapping
    objectives has now exhausted its useful range (five classes tested,
    all failed). Continuing to swap would be a true loop. iter_032
    changes the *readout* not the objective — a structural change, not
    a swap.
*   **Behavioral-Evaluation-Without-Representation-Foundation Loop
    (NEW):** iter_030 ARM 1 and iter_031 Part B both ran CLTS gates
    against representations that did not yet clear F1. Both produced
    uninformative results because the representation quality dominated.
    Tracking: do not run downstream behavioral evaluation again until
    F1 is cleared.
*   **Buffer-Capacity Confound (TRACKED):** buffer=4000 maintained
    through iter_031. Keep constant in iter_032.

## 4. Alternate Research Paths
*   **iter_032: Centroid-Gated Readout (IMMEDIATE PRIORITY, THREE-ARM
    PRE-REGISTERED):**
    - E1: Mean-pool readout + VICReg-only (control / current best baseline,
      union seed bank).
    - E2: Centroid-gated readout + VICReg-only (the architectural fix
      under the cheapest objective).
    - E3: Centroid-gated readout + SFA+VICReg (re-tests M2 under the
      fixed readout — necessary to determine whether the M2 falsification
      was readout-mediated).
    - Falsification: F1 = E2 or E3 ΔR²_color ≥ 0.30 with lower CI ≥ 0.18
      over union seed bank; F2 = collapse ≤ 10%; F3 = centroid_MSE
      ≤ 110; F4 = E2 − E1 paired-seed Δ > 0.10 (isolates the readout fix
      as causal).
    - Pre-registered prediction: E2 > E1 by Δ > 0.10. If E3 > E2, M2 may
      be partially recoverable; if E3 ≈ E2 or E3 < E2, M2 is definitively
      retired in the project mandate.
*   **iter_033 (CONDITIONAL): Either Behavioral Re-Evaluation OR
    Decoder-Free Constraint Relaxation:** Conditional on iter_032 outcome.
    If E2 clears F1: proceed to a properly-calibrated CLTS Part B
    (collision-sparse env, subtler perturbations) on the cleared
    representation. If E2 fails F1: the readout was not the bottleneck
    either, and the decoder-free + mean-pool architecture combination
    is the structural limit — relaxing one constraint becomes mandatory.
*   **Behavioral-Calibration Pivot (PROMOTED to PARALLEL PRIORITY):**
    Open Question 1 from the factual state asks whether the project goal
    can tolerate weak identity encoding. This is a strategic question
    that does not require waiting for iter_032. The CLTS protocol
    redesign (collision-sparse env, sparser perturbations, looser
    tracking threshold) can be prepared in parallel and applied
    immediately once iter_032 yields any non-collapsing representation
    worth evaluating.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (CONDITIONAL,
    DEMOTED):** Reserved as a fallback if iter_032 E2 and E3 both fail
    and the readout fix turns out to be insufficient. The centroid-gated
    readout is the cheaper test and must be exhausted first.
*   **Reconstruction+VICReg Constraint Relaxation (LAST RESORT):**
    iter_031 showed reconstruction does *not* rescue the mean-pool
    bottleneck. Reconstruction is no longer a plausible upper-bound
    reference under the current architecture; it would have to be
    combined with the readout fix to be informative.
*   **Micro-Columns (DEFERRED per semantic caution):** Unchanged. The
    iter_031 finding does NOT yet justify imposed micro-column
    disentanglement — the centroid-gated readout is a less imposed
    structural prior (it samples the existing feature map at the
    existing centroid positions, no new sub-networks) and must be
    tested first.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.
*   **Hierarchical Pyramid (Section 8.6) (DEFERRED):** Unchanged.
    Cannot be invoked until the flat-backbone foundation clears F1.