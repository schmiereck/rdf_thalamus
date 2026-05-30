# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Representation Foundation) remains closed.
    iter_033 executed the pre-committed ORACLE-bracket behavioral pivot
    and triggered **branch (c): task / motor protocol is the bottleneck,
    not perception**. ORACLE (ground-truth positions + physics predictor)
    and RANDOM (ablation policy) produced indistinguishable post-collision
    selectivity (gap = 0.0001 on the primary metric, normalized gap g
    undefined because the bracket is degenerate). This is the cleanest
    outcome possible under the iter_032/033 pre-commit framework: the
    branch (a)/(b)/(c)/(d) rule eliminated the representation-vs-protocol
    confound that made iter_031 CLTS Part B uninterpretable, and the
    result is that the protocol simply does not discriminate perception
    quality at all.
*   **Mechanistic story (clean):** With N=2 objects, every collision
    involves *both* objects, so the "max-velocity-change object"
    identification target is one of only two candidates. Random
    attention has ~50% match probability by construction; this is
    simultaneously the random baseline AND the ceiling of the metric
    under perfect perception. No improvement in perception can move
    selectivity above ~0.50 in this regime. The iter_031 0.59-vs-0.44
    directional signal is therefore explained as within-noise variation
    around a structurally-saturated metric, not as evidence of
    representation-driven behavior. The user-hint warning about that
    signal being potentially noise (and not to be either dismissed nor
    embraced without calibration) was vindicated by the bracket.
*   **Active Direction (pivoted again, per pre-committed branch (c)
    rule):** Do NOT touch the representation. Fix the
    protocol/environment so that the behavioral metric is
    perception-discriminating. iter_034 must produce an environment +
    metric combination where ORACLE − RANDOM is empirically *non-zero*
    and large enough to support a meaningful normalized gap g. Until
    such a discriminating bracket exists, no claim about
    "representation sufficiency for behavior" can be made.
*   **What is now solid:**
    - **The ORACLE-bracket methodology works as a confound disambiguator.**
      It cleanly separated "representation limits behavior" (branch b),
      "representation suffices" (branch a), "protocol/task is the
      bottleneck" (branch c). This is the protocol the project should
      preserve for any future behavioral-evaluation iteration.
    - **The N=2 post-collision selectivity metric is structurally
      saturated** for any motor protocol that picks one object per
      collision. With both objects participating in every collision and
      only two candidates, ~50% match is both floor and ceiling. Future
      metric design must avoid this regime.
    - **ORACLE-v3 (full physics predictor) produces qualitatively
      different surprise statistics than LEARNED conditions** — sharp
      clean spikes at collisions vs continuous noise — which interacts
      with the EMA calibration of the surprise normalizer, producing
      paradoxical worse-tracking-under-perfect-perception (58 vs 33 px).
      This is an open mechanistic finding about CLTSMotorController, not
      about perception.
    - **15-step attention cooldown is suspect** at N=2 collision
      frequencies. Cooldown ≥ inter-collision interval means attention
      gets locked across collision events, which would hide any genuine
      post-collision selectivity even under perfect perception.
    - **iter_032 results remain valid** (cross-backbone attention
      collapse, retired CGIR-as-centroid-sample hypothesis).
    - **iter_028 substrate (separate backbone + mask_dyn_sim +
      coord_vicreg, ΔR²_color = 0.045)** remains the working
      representation; iter_033 confirmed it does not collapse and is
      usable behaviorally, but the bracket showed it cannot be
      evaluated *as such* on the current protocol.
*   **What is now retired or contested:**
    - **Behavioral pivot under N=2 post-collision selectivity is
      retired.** The protocol does not discriminate perception. The
      iter_031 partial signal is explained away by the bracket; do not
      re-cite 0.59-vs-0.44 as a directional finding.
    - **The "directional signals are not noise to be dismissed"
      principle from iter_032 journal is REFINED, not retracted:** a
      directional signal that exists is data, but data must be
      bracketed against ORACLE-and-RANDOM before being interpreted.
      iter_031's signal was real but interpretively empty without the
      bracket. Future directional signals must be bracketed before
      any strategic weight is attached.
    - **Branch (a) and Branch (b) of iter_033 are both untested**, not
      falsified, because the protocol degenerated to branch (c).
      M2's status remains "untestable on this architecture under
      currently-evaluated protocols," now widened to include CLTS
      post-collision selectivity at N=2.
    - **Constraint relaxation (decoder, higher d_t, VICReg-upstream)
      is NOT triggered.** The user-hint rule was explicit: relaxation
      is only justified under branch (b) (ORACLE − RANDOM large,
      LEARNED stuck near RANDOM). Branch (c) does not license it.
*   **Next Priority (iter_034, pre-register tightly):**
    Redesign the behavioral protocol so ORACLE − RANDOM is empirically
    non-zero on the primary metric. Three orthogonal axes to vary,
    each justified separately:
    1. **Object count:** N=3 or N=4. With N≥3 and only some objects
       involved per collision, random match drops below the
       "one-of-two" ceiling. Specifically: for the max-velocity-change-
       object metric with M of N objects collision-involved, random
       match probability is M/N; with N=4 and typically 2 colliding,
       random ≈ 0.5; with N=4 and rare 3-way events, random can drop
       further.
    2. **Metric granularity:** replace binary "did attention match the
       right object" with continuous "time-to-attend after collision
       event" or "fraction of N steps post-collision spent on the
       collision-involved object." Continuous metrics avoid the
       binary ceiling.
    3. **Motor protocol:** shorten or remove the 15-step attention
       cooldown for the bracket-calibration runs. If ORACLE−RANDOM
       opens up under shorter cooldown, the cooldown was the structural
       confound. Treat this as a calibration knob, not a feature
       change to the agent.
    Pre-register: run the ORACLE bracket on the *redesigned protocol
    first* (3 conditions × ≥5 seeds, hard seeds 53/71 included) and
    verify ORACLE − RANDOM ≥ 0.15 on the primary metric BEFORE
    running LEARNED. If the bracket itself doesn't open, the protocol
    is still degenerate and must be redesigned again before any
    perception claim is made.
    Pre-register the binding decision rule for iter_035: same
    branch (a)/(b)/(c)/(d) structure as iter_033, applied to the
    *new* bracketed protocol.
*   **Confidence Score:** 38% (recovered slightly from 35%). The
    iter_033 result is genuine progress: the project now knows that
    the iter_031 directional signal was structurally saturated noise,
    not weak evidence; that the ORACLE-bracket methodology
    successfully separates perception/protocol confounds; and that
    one more layer of "is the metric even right?" must be cleared
    before any sufficiency claim about the representation can be made.
    The confidence boost comes from the *methodological* gain (the
    bracket works), not from progress toward the project goal.
    Recovering further confidence requires iter_034 producing a
    non-degenerate bracket and iter_035 attaching LEARNED to it.

## 2. Strategic Insights & Lessons Learned
*   **ORACLE-BRACKET METHODOLOGY VALIDATED AS A CONFOUND
    DISAMBIGUATOR (iter_033, METHODOLOGICAL FINDING):** Constructing
    three conditions (RANDOM/LEARNED/ORACLE) on identical
    environment + motor logic + seed bank, with a pre-committed
    normalized-gap decision rule g = (LEARNED−RANDOM)/(ORACLE−RANDOM),
    cleanly separates the four possible interpretations of a
    behavioral result (sufficient / limiting / task-bottlenecked /
    partial). This protocol should be inherited by every future
    behavioral-evaluation iteration. The branch (c) outcome
    specifically — ORACLE indistinguishable from RANDOM — is a class
    of failure that no single-arm behavioral experiment can detect,
    and which iter_031 absorbed silently as "weak signal."
*   **METRIC SATURATION IS A FIRST-CLASS DESIGN CONCERN (iter_033,
    STRATEGIC FINDING):** N=2 post-collision selectivity has a
    structural ceiling at ~0.50 because both objects participate in
    every collision, so the "correct" attention target is one of two
    candidates, and random matching achieves the same rate. Metric
    design must explicitly compute the *random-baseline ceiling
    under the metric's own structure* before the metric is adopted.
    Generalization: any metric of the form "did the agent identify
    the right subset of size k from N candidates" has a structural
    random rate of k/N, and the metric's *useful range* is bounded
    above by 1 − k/N. If this useful range is small, the metric
    cannot discriminate behavioral quality regardless of perception.
*   **ORACLE IMPLEMENTATION IS NON-TRIVIAL AND ITERATIVE (iter_033,
    PROTOCOL LESSON):** Three ORACLE implementations were required
    to get the bracket right: v1 had a timing bug + linear
    extrapolation (surprise ~146k, ORACLE < RANDOM), v2 fixed timing
    but kept linear extrapolation (ORACLE ≈ RANDOM, gap 0.007), v3
    used a full physics simulator (ORACLE ≈ RANDOM, gap 0.0001).
    The v1/v2 results were *false-positive ORACLE failures* that
    could have been misinterpreted as branch (b) had the iteration
    stopped early. Lesson: ORACLE conditions must have an
    independent sanity check (e.g. expected surprise scale,
    cross-check against ground-truth event timing) before the
    bracket is interpreted. The v3 surprise scale (~310) vs v2
    (~164k) being 500× smaller is exactly the kind of sanity-check
    mismatch that flagged v1/v2 as broken.
*   **DIRECTIONAL-SIGNAL PRINCIPLE REFINED (iter_033, ITER_032
    PRINCIPLE UPDATE):** iter_032 journal cautioned against
    dismissing the iter_031 0.59-vs-0.44 directional signal as
    "noise." iter_033's bracket shows that signal *is* in-distribution
    noise around a structurally-saturated metric. The refined
    principle: directional signals are not noise to be dismissed,
    BUT they are also not evidence to be promoted until bracketed
    against both ORACLE and RANDOM controls. The bracket is the
    arbitrator, not prior intuition in either direction.
*   **CLTSMotorController BEHAVIORAL ARTIFACTS (iter_033, OPEN
    MECHANISTIC FINDING):** Under ORACLE perception, tracking error
    (58 px) is *worse* than under LEARNED (33 px). This is
    paradoxical at face value but consistent with the surprise-EMA
    calibrating differently when the surprise distribution is clean
    (sharp spikes at collisions, near-zero between) vs. noisy. The
    EMA-normalized attention switch is then triggered by different
    events under the two regimes. This means: the motor's
    surprise-attention coupling is itself a confound that depends on
    the noise statistics of the upstream signal, not just its
    semantic content. Implication: the motor protocol *must* be
    part of any bracketed evaluation, because changing the
    perception arm changes the motor's effective control law via
    the EMA calibration loop.
*   **15-STEP ATTENTION COOLDOWN IS SUSPECT (iter_033, OPEN
    MECHANISTIC FINDING):** At N=2 with frequent collisions, a
    15-step cooldown likely exceeds the inter-collision interval,
    preventing the agent from switching attention to the
    collision-involved object even when surprise correctly
    identifies it. This is a candidate explanation for why ORACLE's
    surprise (correct, sharp) doesn't translate to better
    selectivity — the controller's downstream dynamics block the
    response. iter_034 should treat cooldown as a tunable
    calibration parameter, not a fixed feature.
*   **CARRIED FORWARD (still valid):**
    - M1 (pooled/batch VICReg) stands.
    - M2 status unchanged: "untestable on this architecture under
      currently-evaluated protocols," now including N=2 CLTS
      post-collision selectivity in the list of degenerate protocols.
    - M3 (fixed dimensionality, GDASR log-only) stands.
    - Separate backbone + mask_dyn_sim + coord_vicreg = 0% collapse
      substrate (iter_028) confirmed usable in bracketed behavioral
      experiment.
    - Decoder-free constraint stands; not in scope for iter_034.
    - No positional encoding.
    - d_t=3 frozen.
    - Hard seeds (53, 71) remain in union seed bank.
    - Pre-registered decision rules continue to produce clean
      outcomes (now six consecutive iterations: 023–024, 029–033).
    - Cross-backbone attention coupling remains contraindicated
      without VICReg-upstream-of-gate (iter_032).
    - Constraint relaxation (decoder, higher d_t, alternative
      readout) remains BLOCKED until a non-degenerate ORACLE
      bracket produces branch (b).

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (STATUS: ORTHOGONAL, NOT BLOCKED):**
    iter_033 demonstrated that the ΔR²_color ≥ 0.30 target was
    indeed not the binding constraint, but ALSO that the project
    cannot yet establish what *is* binding because the behavioral
    protocol is degenerate. The bottleneck is currently *protocol
    design*, not representation or even behavior.
*   **Representation-Quality-Gate Loop (RESOLVED, iter_033):** The
    ORACLE bracket structurally prevents this loop from re-forming
    — it makes "representation sufficiency" a measurable concept
    rather than a moving target. iter_034 must preserve this.
*   **Metric-Saturation Loop (NEW, ACTIVE):** Two consecutive
    iterations (031, 033) have evaluated agents on a metric whose
    random-baseline ceiling happens to coincide with its empirical
    ceiling. iter_034 must explicitly compute and report the
    random-baseline ceiling under the chosen metric's structure
    BEFORE running the experiment. If the useful range
    (1 − random ceiling) is below 0.3, the metric is rejected
    as insufficiently discriminating.
*   **ORACLE-Implementation-Correctness Loop (NEW, TRACKED):**
    iter_033 required 3 ORACLE versions. Future ORACLE-bracket
    iterations should pre-register sanity checks (expected surprise
    scale, ground-truth-event-aligned firing rate) so a buggy
    ORACLE is detected before being interpreted as branch (b).
*   **Motor-Protocol-as-Confound Loop (NEW, TRACKED):** The
    surprise-EMA + attention-cooldown mechanism in CLTSMotor mixes
    with the perception noise statistics in non-trivial ways.
    iter_034 should bracket the motor protocol itself by running at
    multiple cooldown settings or with the cooldown removed, to
    isolate motor-protocol effects from perception effects.
*   **Behavioral-Evaluation-Without-Discriminating-Metric Loop
    (DISSOLVED, iter_033):** The bracket dissolves this loop, but
    only because the bracket actually fired branch (c) instead of
    being skipped. The hazard reappears only if iter_034 omits
    the bracket-first check.
*   **Diagnostic-vs-Constructive Iteration Loop (DORMANT):** Seven
    consecutive pre-registered iterations (023–024, 029–033) have
    produced tight findings, two strategic pivots, and a validated
    methodology. Protocol mature.
*   **Overclaim Loop (DORMANT):** iter_033 executor reported branch
    (c) with appropriate caveats (three ORACLE implementations,
    mechanistic explanation, explicit list of implications)
    without overclaiming.
*   **Objective-Swapping Loop (RESOLVED, RETIRED, unchanged).**
*   **Cross-Backbone Coupling Risk (TRACKED, unchanged).**

## 4. Alternate Research Paths
*   **iter_034: Discriminating-Protocol Calibration (IMMEDIATE
    PRIORITY, PRE-COMMITTED VIA ITER_033 BRANCH (c) RULE):**
    - Goal: produce a behavioral protocol where ORACLE − RANDOM ≥
      0.15 on the primary metric, so the bracket is non-degenerate
      and iter_035 can attach LEARNED to it.
    - Three orthogonal axes to vary (treat as a 3-arm calibration,
      not as separate experiments):
      * **Arm 1 — Object count:** redo the N=2 post-collision
        selectivity bracket at N=3 and N=4. Random baseline drops
        from ~0.5 to k/N where k is the average number of
        collision-involved objects.
      * **Arm 2 — Metric continuity:** replace binary
        "did-attention-match" with continuous "fraction of
        M-step-window post-collision spent on collision-involved
        objects" or "time-to-attend latency." Run at both N=2 and
        N=3 to see if continuity alone opens the bracket.
      * **Arm 3 — Motor-cooldown calibration:** at N=2 (the
        original setting), sweep cooldown ∈ {0, 3, 5, 15}. If
        ORACLE − RANDOM opens at shorter cooldowns, the cooldown
        was the structural confound and N=2 is salvageable.
    - For each arm, only RANDOM and ORACLE are run (LEARNED
      deferred to iter_035). Pre-register: at least one arm must
      yield ORACLE − RANDOM ≥ 0.15 with lower CI clear of zero
      over ≥5 seeds. If none do, the project enters a
      protocol-design phase, not constraint relaxation.
    - Pre-register the ORACLE sanity checks: surprise scale within
      2 orders of magnitude of expected physics-event amplitude,
      and at least 80% of surprise spikes within ±2 steps of a
      ground-truth collision event.
*   **iter_035 (CONDITIONAL on iter_034 success): Attach LEARNED to
    the discriminating protocol.** Re-run the full branch
    (a)/(b)/(c)/(d) rule from iter_033 with the new bracket.
    Substrate remains the iter_028 representation. Decision rule
    identical: g = (LEARNED − RANDOM)/(ORACLE − RANDOM).
*   **Constraint-Relaxation Phase (BLOCKED, unchanged):** Branch (b)
    from an iter_035 successful bracket is the only license for
    this. iter_033 produced branch (c), which does NOT license
    relaxation.
*   **Causal Sensitivity Probe (DEFERRED to iter_035):** Was
    iter_033 Arm 2 in the original plan; deferred because the
    primary bracket arm (V-B post-collision selectivity)
    degenerated. Re-attach in iter_035 if the bracket opens.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR), Micro-
    Columns, Hierarchical Pyramid, Phase-5 GDASR Reactivation:**
    all DEFERRED, unchanged.