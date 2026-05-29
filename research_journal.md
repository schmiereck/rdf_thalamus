# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — stabilization sub-
    phase has produced a clean pre-registered NULL. **Single-knob regime
    tuning cannot drive collapse ≤10% under the current architecture.**
    The project is now at a decision branch: continue trying to stabilize
    the shared-CNN dual-stream regime via multi-knob / structural changes,
    or accept that collapse is a property of the architecture and pivot
    to a separate z_dyn encoder (or another structural change).
*   **Active Direction:** iter_026 executed a pre-registered 5-arm,
    50-run sweep with Manager-corrected protocol (dual collapse criterion,
    VICReg sanity floors anchored to iter_025 data, uniform buffer=4000,
    no early termination). The hypothesis ("no single-knob regime
    variation reduces z_dyn collapse below 10%") was the *measured
    outcome*: best arm (batch_size=64) achieved 30% collapse, the same
    as iter_025 v2; canonical A0 regressed to 40%; stronger VICReg
    variance weight (25→50) worsened collapse to 60%; lower LR
    (3e-4→1e-4) was catastrophic (100%). This is a first-class null
    result and removes single-knob regime tuning from the candidate
    intervention set.
*   **Diagnostic gains from iter_026 (secondary, suggestive):**
    - **Train-vs-eval std discrepancy:** many runs maintain train std
      > 0.5 but fail eval std. The representation is narrow on the
      train manifold but does not cover the eval state space —
      consistent with the shared CNN learning a partial subspace
      rather than a generalizing identity code.
    - **JEPA-vs-VICReg objective tension:** increasing variance weight
      worsens collapse, which is the opposite of the naive prior. The
      two objectives are competing under the dual-stream shared-CNN
      regime; pure variance pressure does not stabilize.
    - **Optimization budget is borderline:** 8000 steps at lr=1e-4
      is not enough to escape initialization; the optimization horizon
      and the collapse mechanism interact.
*   **Confound flagged (must be carried forward):** A0 regressed from
    30% (iter_025 v2, buffer=2000) to 40% (iter_026, buffer=4000).
    Any future cross-iteration comparison MUST control buffer capacity
    or treat iter_025/iter_026 baselines as different conditions.
*   **Next Priority (iter_027):** Architectural intervention, NOT
    another regime knob. Under the Manager-authorized scope-reduction
    rule, the candidate is the **separate z_dyn encoder** (the natural
    structural move flagged in iter_025 v2's Open Questions). Rationale:
    (a) we have ruled out single-knob regime tuning empirically;
    (b) the train-vs-eval discrepancy and the JEPA-vs-VICReg tension
    both point at shared-parameter competition between coord and dyn
    streams as the mechanism; (c) decoupling the streams structurally
    is the minimal intervention that addresses both diagnostics
    without re-introducing the failed DSDT pattern (which would only
    apply if the decoupled encoder *also* had no objective — z_dyn
    with SFA+VICReg has one). Any iter_027 plan MUST pre-register
    its collapse gate and its control arms.
*   **Confidence Score:** 40% (reduced from 50%). Two of three
    consecutive iterations failed gates; the third (iter_026) cleared
    its gate by producing a definitive null. We now have a more
    narrowly defined problem (architectural, not regime) but the
    foundation for downstream work is still not in place.

## 2. Strategic Insights & Lessons Learned
*   **SINGLE-KNOB REGIME TUNING CANNOT STABILIZE THE SHARED-CNN
    DUAL-STREAM REGIME (iter_026, NEW FINDING, CONFIRMED via pre-
    registered null):** Best swept configuration: 30% collapse.
    Canonical baseline: 40% collapse. The intervention class is
    exhausted. Future regime-tuning proposals are rejected by
    default unless they bundle ≥2 simultaneous structural changes
    with explicit interaction-effect rationale.
*   **STRONGER VICReg VARIANCE PRESSURE WORSENS COLLAPSE (iter_026,
    MECHANISTIC INSIGHT):** Doubling var_weight (25→50) increased
    collapse 40% → 60%. Interpretation: when variance pressure
    dominates, the JEPA prediction objective cannot shape useful
    representations, and the encoder/predictor co-adaptation
    breaks. This is *evidence of objective competition* in the
    shared-CNN regime — not yet proof that the shared CNN is the
    cause, but a strong update toward that hypothesis.
*   **TRAIN-vs-EVAL STD DISCREPANCY IS A REAL ARCHITECTURAL
    SIGNAL (iter_026):** Runs that pass train std > 0.5 still fail
    eval std. This is consistent with the encoder finding a narrow
    train-manifold subspace that does not generalize. Carry this
    diagnostic forward: any future "non-collapsed" arm must report
    both train AND eval std.
*   **BUFFER-CAPACITY IS A SILENT CONFOUND (iter_026):** The
    iter_025→026 buffer change (2000→4000) plausibly drove the A0
    regression. Future iterations must (a) hold buffer fixed, or
    (b) sweep it explicitly as a controlled variable.
*   **PRE-REGISTERED NULL IS A FIRST-CLASS RESULT (iter_026,
    METHOD WIN):** iter_026 followed the protocol; the protocol
    delivered a clean rejection of an intervention class. This
    is the discipline the Manager has been pushing for since
    iter_022. Carry forward as the standing methodology for all
    future single-claim diagnostic iterations.
*   **PRESERVED:** M2 refutation across iter_022–024 stands; M1
    (pooled VICReg) stands; the d_max=16 capacity baseline ≈ 0.14
    stands; Hungarian-primary matching rule stands; the 20% control-
    collapse power threshold stands.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (ACTIVE):** Unchanged in scope
    but now better-localized. The bottleneck is increasingly
    attributable to shared-parameter competition between z_coord
    and z_dyn pathways under JEPA+VICReg, not to objective choice
    alone.
*   **Training-Regime-Stability Bottleneck (PARTIALLY RESOLVED, now
    RECLASSIFIED as architectural):** iter_026 ruled out single-knob
    regime fixes. The "stability bottleneck" is now reframed as an
    *architectural* problem (shared CNN), not a hyperparameter
    problem.
*   **Capacity-vs-Objective Confound (RESOLVED, iter_025 v2):** Still
    closed.
*   **Matching-Procedure Confound (RESOLVED, iter_025 v2):** Still
    closed.
*   **Diagnostic-vs-Constructive Iteration Loop (PARTIALLY CLEARED):**
    iter_026 was diagnostic but its pre-registered structure delivered
    an actionable null. Continue to permit diagnostic iterations *only*
    when they carry a pre-registered falsification criterion and
    power-threshold check.
*   **Buffer-Capacity Confound (NEW, NOW TRACKED):** Open. Hold buffer
    constant in iter_027.
*   **Objective-Swapping Loop (DORMANT, ENFORCED):** Holds. iter_027
    tests an *architectural* change with the *current* objective set,
    not a new objective.
*   **Logistics:** Executor token limits persist. Tracked, not
    blocking.

## 4. Alternate Research Paths
*   **iter_027: Separate z_dyn Encoder (IMMEDIATE PRIORITY,
    ARCHITECTURAL):** Minimal structural change — give z_dyn its own
    CNN branch (shared low-level features optional), keeping objective
    stack (SFA+VICReg on z_dyn, JEPA on z_coord) unchanged. Pre-
    register: (a) collapse gate ≤10% over ≥10 seeds for the BASELINE
    *separate-encoder* control (no identity objective); (b) buffer
    held at 4000 to control for the iter_026 confound; (c)
    Hungarian-primary matching; (d) report train AND eval std. Falsification:
    if separate-encoder control also collapses ≥20%, the architecture
    hypothesis is rejected and the project must consider objective
    reformulation (BYOL/SimCLR class) or decoder-free constraint
    relaxation.
*   **Multi-Knob Regime Stabilization (DEFERRED, low priority):** If
    iter_027 fails, one fallback path is to revisit regime tuning
    with simultaneous multi-knob changes (e.g., LR schedule + VICReg
    warm-up + larger batch). Lower priority than the architectural
    probe.
*   **VICReg Variance Floor Re-Calibration (DEMOTED):** iter_026
    evidence that stronger variance pressure worsens collapse
    suggests the floor is not the issue. Lower priority.
*   **Object-Tracking-ID Contrastive (DEFERRED to iter_028+):**
    Conditional on iter_027 outcome.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (MEDIUM
    PRIORITY, conditional):** Becomes the lead candidate IF iter_027's
    separate-encoder architecture also collapses.
*   **Accept Decoder-Free Constraint Relaxation (LAST RESORT):**
    Unchanged.
*   **Micro-Columns (DEFERRED per semantic caution):** Unchanged.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.