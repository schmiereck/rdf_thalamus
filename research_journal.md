# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — M2 DEFINITIVELY REFUTED;
    iter_025 v2 diagnostic also yielded a null due to underpowered training regime.
    Now in **stabilization sub-phase**: before testing any new objective, the
    base training regime itself must be made non-collapsing at the required
    confidence level (≤10% collapse over ≥10 seeds).
*   **Active Direction:** The iter_025 v2 ceiling probe was correctly executed
    against the Research Manager's prior critique but the result is *not earned*
    under its own falsification rule:
      - Control Arm A collapsed in 30% of seeds (>20% power threshold).
      - Supervised Arm B was *worse* than control (delta_R2_color: -0.024 vs +0.027),
        but with 30% control collapse and 43% Hungarian/sorted matching disagreement
        on the surviving seeds, this cannot be promoted to "architecture refutes
        identity encoding."
      - Arm E (d_max=16 JEPA+VICReg, no identity objective) reached 0.14
        delta_R2_color WITHOUT any identity term — confirming, as measured, that
        the iter_023 d_max=16 result was a **capacity effect**, not an objective
        effect.
    The honest synthesis: we still cannot disambiguate objective-bottleneck from
    architecture-bottleneck because the training regime is not stable enough to
    carry the probe. The supervised arm performing below control is *suggestive*
    that z_dyn may not carry identity-discriminative information through the
    shared CNN under the current regime, but this inference is **conditional on
    first eliminating collapse**.
*   **Next Priority (iter_026):** **Collapse-elimination sub-experiment** — single
    focused iteration whose only job is to drive Arm A (control / training regime
    only, no identity objective) collapse rate ≤10% over ≥10 seeds. Candidate
    interventions to sweep, independently and minimally:
      1. Learning-rate further reduction (1e-4, 3e-4 as anchor).
      2. VICReg variance-target re-scaling (current floor std≥1 may be the
         collapse driver under the new optimization regime).
      3. Warm-up schedule on the VICReg coefficients.
      4. Batch-size sensitivity (since pooled-VICReg gradient ~1/B; smaller B
         may already be the issue).
    No new objectives may be tested in iter_026. Until the regime is stable,
    every "objective falsification" claim is unearned. This is a deliberate,
    Manager-authorized scope reduction.
*   **Confidence Score:** 50% (reduced from 60%). Two consecutive iterations
    have failed to clear their own pre-declared gates due to training-regime
    instability rather than objective-level evidence. The project is now
    bottlenecked on baseline-stability, not on objective choice. This is a
    worse position than after iter_024 because the iter_025 v2 design was
    supposed to resolve the question and did not.

## 2. Strategic Insights & Lessons Learned
*   **A REGIME THAT COLLAPSES 30% OF SEEDS IS NOT A SUBSTRATE FOR FALSIFICATION
    (iter_025 v2, METHOD WIN):** When the control arm collapses at 30%, *any*
    negative claim about a tested arm is confounded by survivor bias on the
    non-collapsed seeds. The 20% power threshold pre-declared in iter_025 v2 is
    the correct rule and it correctly disqualified the experiment's primary
    claim. Maintain this rule going forward: no objective comparison is valid
    unless the control arm meets the collapse threshold first.
*   **THE d_max=16 EFFECT IS A CAPACITY EFFECT, MEASURED (iter_025 v2 Arm E,
    CONFIRMED):** Arm E reached delta_R2_color = 0.14 with *no* identity
    objective — only JEPA+VICReg at d_max=16. This is the cleanest disentangling
    possible: any future "objective X improved color decoding at d_max=16"
    claim must subtract the Arm E baseline (~0.14) before being interpreted.
    Update reference value: **d_max=16 capacity baseline ≈ 0.14**.
*   **SUPERVISED COLOR LOSS CONVERGES IN TRAINING BUT DOES NOT TRANSFER TO z_dyn
    (iter_025 v2 Arm B, SUGGESTIVE NOT CONCLUSIVE):** Arm B reached near-zero
    training loss yet produced delta_R2_color = -0.024 — *below* the no-identity
    control. Two interpretations remain open: (a) the supervised signal is
    absorbed by parameters outside z_dyn (e.g., the conv head or z_coord
    pathway leaks into solving the task without z_dyn carrying the
    information); (b) the training regime's instability prevents the supervised
    signal from settling into z_dyn. Disambiguating requires iter_026's stable
    regime as a prerequisite.
*   **MATCHING-CONFOUND IS REAL AND MUST BE PRE-DECLARED (iter_025 v2):** 43%
    disagreement between sorted and Hungarian matching on the surviving seeds
    means downstream metrics depend critically on the matching procedure. The
    pre-declared Hungarian-primary rule correctly invalidated post-hoc cherry-
    picking. Keep Hungarian-primary as the standing rule for all future
    delta_R2_* claims.
*   **LOWER LR + GRADIENT CLIPPING HELP BUT DO NOT SOLVE COLLAPSE (iter_025 v2):**
    The reduction from 40-60% (v1) to 30% (v2) is real progress but insufficient.
    Collapse mechanism likely has additional drivers (VICReg variance floor
    under low-LR regime, ramp duration, batch-level statistics) that single-knob
    tuning will not fix.
*   **PRESERVED FROM EARLIER ENTRIES:** M2 refutation across iter_022–024 stands;
    M1 (pooled VICReg) stands; sml transfer is partial at the objective level;
    decoder-free × identity × dual-stream × shared CNN conjunction still lacks
    a validated mechanism.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (ACTIVE, NOT YET ATTRIBUTABLE):** Still the
    dominant bottleneck. iter_025 v2 was the intended attribution probe; it did
    not earn its conclusion. Architecture-vs-objective question remains **open
    pending a stable regime**.
*   **Training-Regime-Stability Bottleneck (NEWLY PROMOTED TO PRIMARY):** The
    current regime collapses 30% of control seeds even after lower LR + gradient
    clipping. This is now the most immediate blocker — every downstream
    experiment requires a stable base. Promote to active focus for iter_026.
*   **Capacity-vs-Objective Confound (RESOLVED, iter_025 v2 Arm E):** d_max=16
    improvement attributed to capacity, not objective. Any future claim must
    subtract the ~0.14 baseline. Loop closed.
*   **Matching-Procedure Confound (RESOLVED, iter_025 v2):** Hungarian-primary
    is the standing rule. Loop closed.
*   **Diagnostic-vs-Constructive Iteration Loop (ACTIVE WARNING):** Two
    consecutive iterations (025 v1, v2) attempted diagnostic disambiguation and
    both produced unearned conclusions due to regime instability. The lesson:
    a diagnostic experiment is only as good as its baseline. iter_026 must be
    *purely* constructive on the baseline before any further diagnostic.
*   **Objective-Swapping Loop (DORMANT, ENFORCEABLE):** Resist the temptation
    to test a "next objective" (ID-contrastive, separate encoder, BYOL) until
    the regime is stable. The Manager will reject planning proposals that test
    a new objective in iter_026.
*   **Logistics:** Executor token limits persist. Tracked, not blocking.

## 4. Alternate Research Paths
*   **iter_026: Collapse-Elimination Sub-Experiment (IMMEDIATE, MANDATORY):**
    No new objective. Sweep regime knobs (LR, VICReg coefficients with warm-up,
    batch size, VICReg variance floor) on a single canonical control arm. Gate:
    ≤10% collapse rate over ≥10 seeds. Until this gate clears, all other paths
    are blocked.
*   **Object-Tracking-ID Contrastive (DEFERRED to iter_027+):** Still the
    leading objective candidate, but now conditional on the iter_026 baseline.
*   **Supervised Linear Probe on z_dyn (DEFERRED, requires stable regime):**
    Re-attempt only with a regime meeting the iter_026 gate. The iter_025 v2
    Arm B result is suggestive but not conclusive; re-run is mandatory.
*   **Separate Identity Encoder (HIGH PRIORITY, conditional):** If iter_027
    ID-contrastive on the shared CNN fails under a stable regime, this becomes
    the next architectural intervention.
*   **VICReg Variance Floor Re-Calibration (NEW, candidate for iter_026):** The
    current std≥1 target may be over-strict given the current latent norms; a
    lower or scheduled floor may be the missing piece for collapse-elimination.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (MEDIUM PRIORITY, on
    hold):** Unchanged.
*   **Accept Decoder-Free Constraint Relaxation (LAST RESORT):** Unchanged.
*   **Micro-Columns (DEFERRED per semantic caution):** Unchanged.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.