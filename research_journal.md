# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Behavioral-benchmark validation under
    perception-gated environment redesign. iter_034 executed the
    pre-committed environment redesign (pass-through obj-obj
    physics) and hit the **analytical ceiling gate** at the
    first checkpoint: PASSIVE pointer accumulated 12.27 valid
    pointer-object collisions per object vs the pre-registered
    threshold of 3.0 — a 4× overshoot. The gate fired as
    designed, blocking the full experiment before any compute
    was wasted on what would have been a guaranteed
    ORACLE ≈ RANDOM outcome. This is a clean, pre-registered
    **null result on the "remove obj-obj collisions" lever**.
*   **Mechanistic story (clean):** In 1D physics, all entities
    share a single spatial axis. A passive pointer that exists
    *as a physical body* on that axis is on the trajectory of
    every bouncing object by geometric necessity. Removing
    obj-obj collisions (the iter_034 redesign) eliminates one
    source of "free" dynamics information but does not address
    the more fundamental source: the pointer itself is
    collision-inevitable. Therefore *no* 1D environment that
    preserves the pointer as a physical entity can gate
    dynamics information on the pointer's policy. The user
    hint's hypothesis ("make obj-obj collisions non-informative
    or rare") was correct at the obj-obj level but insufficient,
    because pointer-object collisions remain abundant and
    policy-independent. The iter_034 analytical ceiling gate
    was the right falsifier and it fired correctly.
*   **Implication (forced, not chosen):** The escalation rule
    pre-committed in iter_034's planning has been triggered:
    partial observation (foveated gaze, Section 8.2) is the
    principled way to make perception necessary, because under
    partial observation *you only learn what you look at*. This
    pulls Section 8.2 forward from deferred into the active
    research path. The geometric argument also implies the
    escalation is not optional within 1D — pointer-as-observer
    is the only remaining lever after pass-through.
*   **Active Direction (iter_035, pre-register tightly):**
    Redesign the environment to use a **foveated/gated
    observation window** instead of a physical pointer:
    - Pointer becomes a *gaze locus*, not a body. It does not
      collide with objects.
    - Information about an object's state is available to the
      agent *only* when the object is within the foveated
      window (or with a sharp distance-dependent attenuation).
    - Mass-estimation experiments require *agent-caused
      excitation* — replace pointer-object collisions with a
      gated "probe" action (e.g. a localized force pulse
      applied at the gaze locus) that the agent must aim.
    - Primary metric: per-object mass estimate from the
      *median* of repeated probe-induced collisions per object
      under a finite probe budget, NOT single-collision
      least-squares (which iter_034 v1 falsified as
      noise-amplifying for active policies).
    - Alternative metric: coverage-efficiency (steps to
      N>=3 probe-induced events per object).
    Run RANDOM and ORACLE only (LEARNED deferred to iter_036)
    with the iter_033 ORACLE sanity-checks as pass/fail
    preconditions: surprise scale within 2 orders of magnitude
    of expected physical event amplitude, ≥80% of surprise
    spikes within ±2 steps of ground-truth events. Gate:
    ORACLE − RANDOM ≥ 0.15 with lower CI clear of zero over
    ≥5 seeds.
*   **Pre-committed escalation (iter_036, conditional):** If
    foveated observation STILL cannot open the bracket
    (ORACLE − RANDOM < 0.15), that *is* the finding —
    perception is not behaviorally load-bearing under any
    tractable 1D observation regime — and the project must
    then confront whether the 1D sandbox itself is the
    structural confound (escalation to 2D or to a
    fundamentally different task formulation). Do not spend
    more than one foveated-gaze iteration before that meta-
    escalation.
*   **What is now solid:**
    - **The analytical ceiling gate is a high-leverage
      protocol primitive.** It detected an environment-design
      failure in one cheap measurement (counting passive
      collisions) before any model training. Inherit this
      primitive for every future environment redesign.
    - **1D + physical pointer + full observation is
      structurally incompatible with making perception
      load-bearing.** This is a geometric statement, not an
      empirical conjecture, and it eliminates an entire
      family of candidate fixes.
    - **iter_028 substrate** (separate backbone + mask_dyn_sim
      + coord_vicreg, ΔR²_color ≈ 0.045, 0% collapse) remains
      the working representation, unchanged.
    - **MALRE v2** remains validated as a coverage-
      discrimination test (active-vs-passive gap = 0.83);
      unchanged as the secondary metric, but no longer the
      primary benchmark for iter_035 because it does not
      discriminate within the active regime.
*   **What is now retired or contested:**
    - **Pass-through obj-obj physics as a sufficient
      environment redesign:** falsified by analytical
      ceiling gate (PASSIVE 12.27 >> threshold 3.0).
    - **The hypothesis that obj-obj collisions are the
      primary source of free dynamics information:** refined.
      Pointer-object collisions on a 1D axis with a physical
      pointer are an equally-or-more abundant source, and
      they are policy-independent for any policy that doesn't
      deliberately *avoid* objects.
    - **All metric-only redesigns on the full-observation
      environment** (the iter_034 open question 1–6 metric
      sweeps): retired without test, because the user-hint
      argument — confirmed by the geometric pointer-collision
      inevitability — establishes that no metric can
      discriminate perception in an environment where
      information is ungated by perception.
    - **Constraint relaxation (decoder, higher d_t,
      VICReg-upstream):** remains BLOCKED. The original rule
      (relaxation only under iter_033 branch (b)) is
      unaffected. iter_034 was a null on environment design,
      not a sufficiency result.
*   **Confidence Score:** 42% (up from 38%). The recovery is
    methodological, not goal-directed: a pre-registered gate
    caught a design failure at minimum cost, identifying a
    geometric constraint on the entire 1D-pointer experimental
    regime and forcing a principled — rather than ad-hoc —
    escalation to foveated observation. The project still has
    not validated the representation behaviorally; the gain
    is structural clarity about what regimes *cannot* validate
    it. Recovery beyond ~45% requires iter_035 producing an
    open bracket under foveated observation.

## 2. Strategic Insights & Lessons Learned
*   **ANALYTICAL CEILING GATE AS PROTOCOL PRIMITIVE (iter_034,
    METHODOLOGICAL FINDING):** Pre-registering a cheap-to-
    compute structural ceiling (here: count passive valid
    events vs threshold) as a hard gate before full experiment
    execution prevented an entire wasted run that would have
    almost certainly returned ORACLE ≈ RANDOM. The gate
    computed in passes that would otherwise have taken full
    training. This generalizes: for any
    bracketed-behavioral-evaluation iteration, the
    *prerequisite-for-discrimination* condition (e.g.
    "passive baseline must NOT already saturate information
    acquisition") should be expressed as a pre-registered
    analytical gate, evaluated first, with the experiment
    blocked on failure. Adopt as standard protocol.
*   **PERCEPTION-GATED-INFORMATION AS DESIGN AXIOM (iter_034,
    STRATEGIC FINDING):** For any experiment intended to
    validate that perception is *behaviorally load-bearing*,
    the environment must be constructed such that the
    information necessary for the downstream metric is
    *causally gated* on the agent's perception-driven action.
    If passive or random policies can accumulate the same
    information by virtue of environment geometry,
    ORACLE − RANDOM is bounded above by the *small*
    additional value of "smart" vs "any" action — typically
    noise-floor. Diagnostic question for every future
    environment design: "What unit of information does the
    agent need, and what action is required to acquire that
    unit?" If the answer is "no action" or "any action,"
    the environment cannot test perception sufficiency.
*   **1D + PHYSICAL POINTER IS A GEOMETRIC DEAD END FOR
    PERCEPTION-GATING (iter_034, STRUCTURAL CONSTRAINT):**
    Independent of any specific physics ruleset, an entity
    that physically occupies the same 1D spatial axis as the
    objects of interest will be involved in collisions at a
    rate proportional to total motion in the scene, not to
    the entity's own policy. This eliminates pass-through,
    collision-rare, low-density-N, and similar
    environment-modification levers as sufficient fixes.
    Only making the pointer a *non-physical observer*
    (foveated gaze, attention window) can break this. This
    is the structural argument for the iter_034 escalation
    to Section 8.2.
*   **MEDIAN-OF-REPEATED-EVENTS BEATS SINGLE-EVENT
    LEAST-SQUARES UNDER ACTIVE POLICIES (iter_034 carry-
    forward of iter_033/034 lesson):** The v1 MAPE benchmark
    (iter_034.2) using single pointer-object collision mass
    estimates inverted ordering (PASSIVE < RANDOM < ORACLE)
    because active probing increased per-collision noise.
    Any iter_035 mass-estimation metric must aggregate over
    *repeated* agent-caused events per object (median is the
    noise-robust default; mean is too sensitive to single-
    collision outliers in chaotic dynamics). Document this
    as the standard reduction for any future
    mass/parameter-estimation benchmark in this project.
*   **MOTOR CONFOUND PERSISTS, REQUIRES DESIGN AROUND, NOT
    THROUGH (iter_033 carry-forward, sharpened):** The
    iter_033 CLTSMotorController EMA-calibration confound
    remains live. iter_035's metric should be computed from
    *interaction outcomes* (mass-estimate accuracy or
    coverage-efficiency) rather than from tracking error.
    If the bracket arms must route through the motor
    controller, the surprise-EMA must be recalibrated
    per-condition (ORACLE / RANDOM / future LEARNED) so the
    benchmark does not silently measure controller
    calibration. Pre-register the per-condition EMA
    recalibration protocol in iter_035.
*   **CARRIED FORWARD (unchanged):**
    - M1 (pooled/batch VICReg) stands.
    - M2 status unchanged: "untestable under currently-
      evaluated protocols." Now widened: untestable under
      any 1D full-observation regime due to the structural
      gating argument.
    - M3 (fixed dimensionality d_t=3, GDASR log-only) stands.
    - Separate backbone + mask_dyn_sim + coord_vicreg = 0%
      collapse substrate.
    - Decoder-free constraint stands.
    - No positional encoding.
    - Pre-registered decision rules continue to produce
      clean outcomes (eight consecutive iterations:
      023–024, 029–034).
    - ORACLE-bracket methodology stands as the confound
      disambiguator for behavioral evaluation.
    - Metric saturation must be computed and reported
      before any metric is adopted (iter_033 lesson).
    - Constraint relaxation BLOCKED pending an open
      bracket producing branch (b).

## 3. Loop & Bottleneck Detection
*   **Environment-Design Bottleneck (NEW, ACTIVE):** The
    project's binding constraint has shifted from
    representation (resolved by iter_028 substrate +
    iter_033 bracket) to behavioral protocol (iter_033) to
    environment information-structure (iter_034). The
    currently-binding question: *can any 1D environment make
    perception load-bearing?* If foveated gaze (iter_035)
    opens the bracket: yes, partial observation suffices.
    If not: 1D itself may be the confound, and 2D escalation
    becomes the next forced move.
*   **Cheap-Analytical-Gate Loop (NEW, FAVORABLE):** iter_034
    validated that a cheap structural pre-check can catch a
    flawed experiment before execution. This is a *good*
    loop — adopt as standard. Future iterations should
    identify the cheapest computable necessary-condition for
    experimental success and gate execution on it.
*   **Geometric-Inevitability Loop (NEW, TRACKED):** Two
    consecutive environment-design attempts (iter_034
    pass-through; the metric sweeps that would have been
    iter_034 v1.x) have foundered on a geometric property of
    the test environment rather than on any modeled
    mechanism. Adopt the diagnostic: before proposing an
    environment modification, articulate the *geometric or
    topological* property that the modification changes, and
    argue that *that* property — not the mechanism the
    modification superficially targets — is the binding
    constraint.
*   **Metric-Saturation Loop (ACTIVE, unchanged):** Carry
    forward; applies to any iter_035 metric design. Compute
    and report the random-baseline ceiling structurally
    before adopting.
*   **ORACLE-Implementation-Correctness Loop (ACTIVE,
    unchanged):** Pre-register surprise-scale and event-
    alignment sanity checks for any ORACLE in iter_035.
*   **Motor-Protocol-as-Confound Loop (ACTIVE, unchanged):**
    Per-condition surprise-EMA recalibration required if
    iter_035 routes through CLTSMotorController; preferable
    to use a metric that bypasses the motor entirely.
*   **Representation-Quality-Gate Loop (RESOLVED, unchanged).**
    ORACLE bracket prevents re-formation.
*   **Diagnostic-vs-Constructive Iteration Loop (DORMANT,
    unchanged).** Eight consecutive clean pre-registered
    iterations; protocol mature. Honest null findings now
    routinely produced.
*   **Overclaim Loop (DORMANT, unchanged):** iter_034 reported
    its null result without overclaiming (correctly framed
    as "the gate fired, escalation triggered").

## 4. Alternate Research Paths
*   **iter_035: Foveated-Observation Environment + Bracket
    (IMMEDIATE PRIORITY, PRE-COMMITTED VIA ITER_034
    ESCALATION):**
    - Goal: produce a 1D environment in which dynamics
      information about each object is *causally gated* on
      the agent's gaze policy, and verify
      ORACLE − RANDOM ≥ 0.15 on a continuous, motor-
      independent metric over ≥5 seeds (hard seeds 53/71
      included).
    - Environment changes:
      * Pointer → non-physical gaze locus (no collision).
      * Object observation: full state available only when
        object within gaze window of radius r; outside,
        either unobserved (preferred — maximally gates
        information) or heavily attenuated.
      * Excitation: agent issues a localized "probe" force
        impulse at the gaze locus; this is the only way to
        induce object dynamics changes.
      * Object-object collisions: retain *or* remove —
        pre-register both as Arm A (keep, foveation alone)
        and Arm B (remove + foveate) to factor the
        contribution of each lever.
    - Primary metric: per-object mass estimate from the
      median of repeated probe-induced events per object
      under a fixed probe budget B (start B=20). Report
      per-object MAE.
    - Secondary metric: coverage-efficiency (number of
      probe events needed to reach ≥3 events per object).
    - Bracket: RANDOM and ORACLE only; LEARNED deferred.
      ORACLE = perfect knowledge of object positions and
      velocities, with policy that allocates probes to
      equalize per-object coverage and target moments when
      relative velocity is informative.
    - Analytical ceiling gate (pre-registered, computed
      first): under RANDOM policy, per-object event count
      must be sufficiently *unbalanced* across objects
      that ORACLE has room to improve. Specifically:
      coefficient of variation of per-object event counts
      under RANDOM ≥ 0.5, computed analytically or by
      single short rollout.
    - ORACLE sanity checks (pre-registered): surprise
      scale within 2 orders of magnitude of physical
      event amplitude; ≥80% of surprise spikes within ±2
      steps of ground-truth probe-induced events.
    - If gate or sanity checks fail: iteration reports
      the null and triggers iter_036 meta-escalation
      (see below). Do not relax constraints.
*   **iter_036: Meta-Escalation if Foveation Fails
    (PRE-COMMITTED, CONDITIONAL):** If iter_035 foveated
    observation cannot open the bracket, that establishes
    that 1D itself is structurally insufficient for
    perception-load-bearing experiments. The forced moves
    are: (i) 2D environment redesign — substantially more
    expensive but principled; (ii) accept that the
    project's behavioral validation goal is unreachable in
    a 1D sandbox and re-frame the project's deliverable
    around the *representation* + *thalamic gating* claims
    that can be validated without behavior; (iii) revisit
    whether the decoder-free constraint is itself part of
    the problem. Decision rule deferred to iter_036
    planning; explicitly noted that this is a
    meta-strategic decision, not a within-design choice.
*   **iter_037+ (CONDITIONAL on iter_035 bracket opening):
    Attach LEARNED, run branch (a)/(b)/(c)/(d) rule.**
    Substrate remains iter_028 + d_t=3 frozen + GDASR
    log-only. Decision rule identical to iter_033.
*   **Constraint-Relaxation Phase (BLOCKED, unchanged):**
    Only justified by branch (b) outcome from an
    open-bracket LEARNED run.
*   **Causal Sensitivity Probe (DEFERRED, unchanged):**
    Re-attach once a bracket opens.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR),
    Micro-Columns, Hierarchical Pyramid, Phase-5 GDASR
    Reactivation:** DEFERRED, unchanged.