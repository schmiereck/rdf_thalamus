# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** META-ESCALATION TRIGGERED. iter_036
    executed the pre-committed foveated-gaze redesign and
    hit the **coefficient-of-variation (CV) ceiling gate**
    in BOTH arms before any full bracket run:
    - Arm A (foveation only, obj-obj collisions retained):
      RANDOM per-object event-count CV = 0.36
    - Arm B (foveation + pass-through obj-obj):
      RANDOM per-object event-count CV = 0.46
    - Pre-registered threshold: CV ≥ 0.50
    The gate fired as designed. Under foveated gaze with
    GAZE_RADIUS=8 in a 128-pixel arena containing 3 objects,
    random gaze trajectories distribute probe events
    sufficiently evenly across objects that there is
    essentially no "underserved object" for an ORACLE to
    preferentially target. This is the structurally
    symmetric failure mode to iter_035's "PASSIVE already
    saturates": there, all policies acquired adequate
    collision information; here, all policies achieve
    adequate gaze coverage.
*   **Four-iteration null chain (clean, pre-registered,
    compute-conserving):** iter_033 (ORACLE ≈ RANDOM on
    behavioral pivot), iter_034 (v2 MALRE coverage-only,
    ORACLE-RANDOM=0.031), iter_035 (pass-through physics:
    PASSIVE 12.27 colls/obj vs 3.0 threshold), iter_036
    (foveation: RANDOM CV 0.36/0.46 vs 0.50 threshold).
    Each iteration applied a progressively more radical
    environment redesign and each was killed at an
    analytical/structural gate before any wasted training
    compute. **Cumulative finding:** no 1D-sandbox
    configuration tested has been able to make perception
    behaviorally load-bearing in the
    ORACLE-vs-RANDOM-bracket sense — full-observation
    regimes fail because passive policies acquire the
    information for free; partial-observation regimes fail
    because random policies cover space uniformly enough
    that selective allocation has nothing to gain.
*   **Mechanistic story (4-iteration synthesis):** Behavioral
    load-bearingness requires the *information rate per
    object* to be both (a) bounded — so that better
    allocation has discriminative value — and (b) unequal
    across objects under a non-perceptual baseline policy —
    so that there is "headroom" for perception-driven
    reallocation. The 1D × N=3 × 128-pixel × full-or-
    foveated regime fails one of these on every iteration:
    either bound (a) fails (full observation, pointer
    collisions abundant) or non-uniformity (b) fails
    (foveation in a small arena, random walks cover
    everything). These are not coincidences of parameter
    choice — they reflect the geometric fact that a small,
    low-dimensional, sparsely-populated arena does not
    naturally produce the information-allocation pressures
    that would make selective attention pay off.
*   **Implication (FORCED meta-decision, owed to iter_037
    planner):** The pre-committed meta-escalation has
    triggered. The three options on the journal are:
    (i) 2D environment redesign — principled (more spatial
        dimensions naturally create coverage heterogeneity
        because random walks in 2D do not cover area as
        uniformly as in 1D) but materially more expensive
        across all components (env, perception, motor);
    (ii) re-frame the project's behavioral-validation goal:
         accept the four-iteration null as evidence that
         behavioral validation in *any* 1D regime is
         structurally unreachable, and reduce the
         deliverable to representation-quality +
         thalamic-gating claims that can be evaluated
         without a bracketed behavioral metric;
    (iii) revisit whether the decoder-free constraint is
          itself binding: a decoder enables direct
          reconstruction-based evaluation that bypasses
          the bracket-discrimination problem entirely.
    This is a **meta-strategic decision**, not a
    within-design choice. The Manager's scope-reduction
    authority is in play and may have to be exercised again.
*   **Active Direction (iter_037, the meta-decision):**
    Iter_037 must NOT propose a new environment redesign.
    It must instead pre-register one of the three escalation
    paths with explicit cost/benefit/falsifiability criteria
    for each. Decision rule for iter_037:
    - Path (i) 2D: justified only if a concrete 2D design
      is sketched with the analytical ceiling gates
      pre-stated (passive-event bound, RANDOM-CV bound,
      ORACLE sanity checks). Estimate engineering cost in
      agent-iterations.
    - Path (ii) re-frame: justified only by enumerating
      the falsifiable representation + gating claims that
      REMAIN testable without a behavioral bracket, plus
      the gates that would validate each. Concretely:
      identity-disentanglement ΔR² thresholds, attention-
      token-trace properties, surprise-EMA calibration
      tests. Must NOT degenerate into "we have a nice
      representation, ship it."
    - Path (iii) decoder relaxation: justified only by
      an explicit argument for why the original decoder-
      free constraint was adopted, what it bought, and
      what is lost by relaxing it. Must include a
      falsifiable test that the relaxation buys
      behavioral discrimination that the constrained
      regime could not provide.
    Iter_037 deliverable: a pre-registered choice among
    (i)/(ii)/(iii) with the gates for the chosen path
    stated in advance.
*   **What is now solid:**
    - **Analytical/structural ceiling gates have saved
      four iterations of wasted training compute.**
      Adopt as standard protocol going forward; the
      primitive is mature.
    - **The 1D-sandbox is structurally insufficient for
      bracketed behavioral validation of perception under
      any tested observation regime.** This is now a
      four-iteration empirical finding, not a conjecture.
      Independent of any specific representation or
      motor controller.
    - **iter_028 substrate** (separate backbone +
      mask_dyn_sim + coord_vicreg, ΔR²_color ≈ 0.045,
      0% collapse) remains the working representation.
    - **MALRE v2 remains a valid coverage-discrimination
      test** but not a perception-quality test.
*   **What is now retired or contested:**
    - **Foveated-gaze observation in 1D × N=3 × 128px**
      as a sufficient environment redesign: falsified by
      the CV gate in both Arm A (collisions retained) and
      Arm B (pass-through). The two levers (foveation
      and pass-through) do not combine additively in a way
      that opens the bracket.
    - **The "find the right 1D environment" research
      path:** structurally retired. Four consecutive
      principled attempts have failed at the gate stage.
      Further 1D redesigns would be ad-hoc.
    - **Pass-through physics, low-density-N variants,
      single-collision LSQ mass estimation, and all
      metric-only redesigns on a full-observation
      environment:** remain retired per prior iterations.
    - **Constraint relaxation (decoder, higher d_t,
      VICReg-upstream):** REMAINS BLOCKED for path (i)
      and path (ii); becomes the explicit subject of
      path (iii) if iter_037 chooses it.
*   **Confidence Score:** 39% (down from 42%). The slight
    drop reflects that the foveation lever — which had
    been the principled escape from the iter_035
    pointer-geometry constraint — has now also failed at
    a structural gate. The gain in clarity (1D-sandbox is
    structurally insufficient) is real, but the project
    now faces a binding meta-decision with no within-1D
    options remaining. Methodological discipline remains
    high (nine consecutive clean pre-registered
    iterations, four consecutive analytical-gate saves).

## 2. Strategic Insights & Lessons Learned
*   **STRUCTURAL-CEILING GATE PRIMITIVE NOW VALIDATED
    ACROSS FOUR ITERATIONS (iter_036, METHODOLOGICAL
    FINDING, PROMOTED):** Pre-registered cheap-to-compute
    analytical/structural gates have now killed four
    consecutive flawed experiments before any wasted
    training run (iter_033 metric-saturation check,
    iter_034 MALRE active-passive gap, iter_035 PASSIVE
    collision count, iter_036 RANDOM CV). The pattern is
    stable enough to formalize as a protocol primitive:
    every iteration that proposes a bracketed-behavioral
    evaluation must include (a) the structural
    necessary-condition for discrimination, expressed as a
    single number computable from a short rollout or
    analytically, and (b) the threshold this number must
    meet, declared in advance. Block full execution on
    failure. This primitive is now the project's most
    reliable output.
*   **DUAL FAILURE MODES OF PERCEPTION-LOAD-BEARINGNESS
    (iter_036, STRATEGIC FINDING):** Behavioral validation
    of perception requires that under a non-perceptual
    baseline policy: (a) the information rate per object
    is BOUNDED (otherwise passive saturates — iter_035
    pattern), AND (b) the information rate per object is
    UNEQUAL across objects (otherwise random already
    allocates evenly — iter_036 pattern). The 1D × N=3 ×
    128-pixel arena fails one of these on every observation
    regime tested. Future environment designs (if any)
    must be evaluated against BOTH conditions with
    pre-registered analytical gates for each. The
    diagnostic question becomes: "Under a non-perceptual
    baseline, is per-object information acquisition both
    bounded and uneven?"
*   **GEOMETRIC-COVERAGE PROPERTY OF RANDOM WALKS IS A
    FIRST-ORDER ENVIRONMENT DESIGN CONSTRAINT (iter_036,
    STRUCTURAL CONSTRAINT):** In low-dimensional, small,
    sparsely-populated arenas, random-walk coverage is
    sufficiently uniform on relevant timescales that
    "smart" allocation gains little. This is the
    higher-dimensional analog of the iter_035 pointer
    geometry argument: the binding constraint is a
    geometric property of the *space* (here, dimensionality
    and density) rather than of the agent or the task.
    This is the structural argument for path (i) 2D over
    any further 1D redesign: not "2D is more interesting"
    but "2D random walks do not cover area as uniformly
    as 1D random walks cover line, restoring condition (b)."
*   **WHEN THE FOURTH PRINCIPLED REDESIGN FAILS, THE
    DESIGN SPACE IS EXHAUSTED, NOT UNLUCKY (iter_036,
    META-STRATEGIC FINDING):** With four consecutive
    pre-registered environment redesigns failing at
    structural gates, the prior on "the next 1D tweak
    will work" is now low enough that further 1D
    iterations would be motivated by sunk cost rather
    than evidence. The forced meta-decision is to change
    the design space (2D, path i), change the
    deliverable (path ii), or change the constraint
    (path iii). Continuing to iterate within 1D would
    violate the project's stated Manager discipline.
*   **CARRIED FORWARD (unchanged):**
    - M1 (pooled/batch VICReg) stands.
    - M2 status: "untestable under any tested 1D
      observation regime." Now four-iteration null.
    - M3 (fixed dimensionality d_t=3, GDASR log-only)
      stands.
    - Separate backbone + mask_dyn_sim + coord_vicreg =
      0% collapse substrate.
    - Decoder-free constraint stands for paths (i) and
      (ii); becomes the subject of debate under path (iii).
    - No positional encoding.
    - Pre-registered decision rules continue to produce
      clean outcomes (nine consecutive iterations:
      023–024, 029–036).
    - ORACLE-bracket methodology stands as the confound
      disambiguator for behavioral evaluation IF a
      bracket-able environment is ever found.
    - Metric saturation must be computed and reported
      before any metric is adopted.
    - Median-of-repeated-events beats single-event
      least-squares for active policies.
    - Per-condition surprise-EMA recalibration required
      for any motor-routed bracket.
    - Constraint relaxation BLOCKED for paths (i)/(ii);
      becomes path (iii)'s explicit subject.

## 3. Loop & Bottleneck Detection
*   **Environment-Design Bottleneck (NOW EXHAUSTED, FORCED
    META-DECISION):** Four pre-registered redesigns
    (iter_033, 034, 035, 036) have failed at structural
    gates. The bottleneck is no longer "find the right
    1D environment" — it is "decide whether to escalate
    dimensionality, scope, or constraint." Iter_037 must
    resolve this.
*   **Cheap-Analytical-Gate Loop (NOW INSTITUTIONALIZED):**
    Four-iteration validation of the primitive. Promoted
    to standard protocol. Every future behavioral-bracket
    proposal must include a structural ceiling gate.
*   **Geometric/Topological-Constraint Loop (CONFIRMED,
    PROMOTED):** Two consecutive iterations have foundered
    on geometric properties of 1D space (iter_035 pointer
    collisions; iter_036 random-walk coverage uniformity).
    The diagnostic — articulate the geometric/topological
    property the redesign changes — is now mandatory for
    any further environment design.
*   **Dual-Failure-Mode Loop (NEW):** Any future
    bracketed-behavioral evaluation must check BOTH
    "passive doesn't saturate" AND "random doesn't already
    cover" conditions with separate pre-registered gates.
    Past iterations checked one or the other but not both.
*   **Sunk-Cost-Avoidance Loop (NEW, FAVORABLE):** Iter_036
    is the trigger point for an explicit meta-decision
    rather than another 1D iteration. The project's
    discipline is being tested by the temptation to "try
    one more thing" — the journal pre-commitment (from
    iter_035) is what should hold the line.
*   **Metric-Saturation Loop (ACTIVE, unchanged):** Carry
    forward to any future bracket design.
*   **ORACLE-Implementation-Correctness Loop (DORMANT for
    iter_037 since no ORACLE will be implemented in a
    meta-decision iteration).**
*   **Motor-Protocol-as-Confound Loop (DORMANT for
    iter_037).**
*   **Representation-Quality-Gate Loop (RESOLVED, becomes
    path (ii)'s subject if chosen).**
*   **Diagnostic-vs-Constructive Iteration Loop (DORMANT):**
    Nine consecutive clean pre-registered iterations;
    protocol mature.
*   **Overclaim Loop (DORMANT):** iter_036 reported its
    null cleanly with the meta-escalation framed as
    pre-committed, not improvised.

## 4. Alternate Research Paths
*   **iter_037: META-DECISION ITERATION (IMMEDIATE,
    PRE-COMMITTED VIA ITER_035/036 ESCALATION):**
    No new experiment. Iter_037 produces a single
    deliverable: a pre-registered choice among paths
    (i)/(ii)/(iii) with full justification and gates.
    Required content:
    - Restate the four-iteration null chain with the
      gate values that fired.
    - For each of (i)/(ii)/(iii), pre-register the
      falsifiability criterion that would validate or
      kill that path.
    - For the chosen path, pre-register iter_038's
      first experiment with its structural ceiling
      gate(s).
    - Document the decision rule (why this path, why
      not the other two) so it is auditable.
    - Estimate engineering cost (agent-iterations)
      for the chosen path.
    Hard rule: NO new environment design within 1D.
    Soft rule: prefer the path with the lowest
    engineering cost given equal falsifiability.
*   **Path (i): 2D environment redesign (CANDIDATE).**
    Argument: random walks in 2D do not cover area as
    uniformly as in 1D cover line, restoring CV
    condition (b). Cost: substantial — env, perception
    (2D conv), motor, evaluation all need redesign.
    Falsifiability: must pre-register both PASSIVE
    bound and RANDOM CV gates analogous to iter_035/036.
*   **Path (ii): re-frame around representation +
    gating (CANDIDATE).** Argument: iter_028 substrate
    + iter_032 cross-backbone finding + iter_034
    coverage-test validation already produce a
    defensible non-behavioral story. Cost: low —
    consolidation, not new compute. Falsifiability:
    must pre-register the specific representation
    and gating claims and the gates that validate
    each, to avoid degeneration into "ship what we
    have."
*   **Path (iii): revisit decoder-free constraint
    (CANDIDATE).** Argument: a decoder enables
    reconstruction-based evaluation that bypasses the
    bracket-discrimination problem. Cost: medium —
    decoder design + retraining, but reuses
    substrate. Falsifiability: must pre-register a
    test that decoder-enabled evaluation buys
    discrimination the constrained regime could not.
    Risk: violates the project's foundational
    decoder-free principle; requires explicit
    acknowledgement.
*   **iter_038+ (CONDITIONAL on iter_037 choice):**
    First experiment along the chosen path. Substrate
    remains iter_028 + d_t=3 frozen + GDASR log-only
    unless path (iii) is chosen.
*   **Constraint-Relaxation Phase:** Status now
    contingent on iter_037's path choice rather than
    blocked outright.
*   **Causal Sensitivity Probe (DEFERRED, unchanged):**
    Re-attach only if a bracket-able environment is
    found (path i) or path (iii) provides a
    bracket-free evaluation.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR),
    Micro-Columns, Hierarchical Pyramid, Phase-5 GDASR
    Reactivation:** DEFERRED, unchanged.