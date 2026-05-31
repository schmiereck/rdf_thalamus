# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** META-DECISION AWAITING HUMAN
    GO/NO-GO. iter_037 executed exactly the de-risking
    and decision-support scope mandated by the
    iter_035/036 escalation: NO autonomous path
    selection, NO 2D rebuild, NO new representation
    work. Three deliverables completed:
    1. The four-iteration 1D environment-design null
       chain (iter_033 metric saturation, iter_034
       free autonomous information, iter_035
       collision-inevitable shared-axis pointer,
       iter_036 small-arena random-gaze coverage)
       crystallized as a standalone documented
       finding — a clean scientific result independent
       of whatever path is chosen next.
    2. A MINIMAL 2D cheap-gate experiment (rollouts
       only, no learning, no ORACLE bracket) at
       64×64 / N=3 / gaze_radius=8 / 5 seeds,
       applying the project's own validated
       structural-ceiling-gate primitive to the
       most expensive escalation option BEFORE any
       commitment. Gates pre-registered with
       thresholds; per-seed decision rules added per
       iter_036 Manager critique.
    3. A decision-support document scoping path (i),
       path (ii), explicitly rejecting path (iii) as
       mis-targeted (the blocker is environmental
       not representational; iter_031 already
       falsified mean-pool reconstruction), and
       estimating engineering cost for each remaining
       option.
*   **2D cheap-gate result (iter_037, NEW STRUCTURAL
    FINDING):** Of the three pre-registered 2D gates,
    one passes and two fail in an informative pattern:
    - **Gate-1 PASSES.** Per-object PASSIVE collision
      count is 0–1 over 5 seeds vs the 3.0 threshold
      and vs the 1D iter_035 measurement of 12.27.
      2D geometry removes collision inevitability:
      with a static central pointer and objects free
      in two dimensions, off-axis trajectories
      prevent the 1D collision saturation.
    - **Gate-1b FAILS.** Collisions are now so rare
      that the per-object collision-count CV is not
      stable across seeds — a sample-size-noise
      regime, not a heterogeneity regime.
    - **Gate-2 FAILS.** RANDOM gaze coverage CV at
      the tested parameterization clusters near the
      Poisson baseline (~0.39) rather than the
      pre-registered ≥0.50 threshold. 2D random
      walks DO not cover area as uniformly as 1D
      random walks cover line in principle, but at
      a 64×64 arena with gaze_radius=8 the
      non-uniformity is not yet of an order that
      opens the bracket.
*   **The Gate-1/Gate-1b tension (iter_037,
    STRUCTURAL FINDING):** The two failures are in
    structural opposition: Gate-1 demands rare
    collisions, Gate-1b demands enough collisions
    for CV to be meaningful. With a static central
    pointer in 2D, the parameterization that
    satisfies one tends to fail the other. This
    tension may be fundamental to the
    static-pointer behavioral-test design itself
    rather than a tuning problem — i.e. path (i)
    might require not just a 2D arena but a 2D
    task redesign (navigation vs selection)
    separate from coverage-by-attention. This is a
    novel, empirically measured constraint not
    anticipated in the iter_036 escalation; it
    tightens the path-(i) cost estimate
    substantially.
*   **Path-(i) implication:** path (i) is NOT
    blocked outright — 2D at the tested
    parameterization is blocked, which is different.
    A full path-(i) commitment would have to
    (a) widen the 2D parameter sweep, (b) redesign
    the behavioral test away from pointer-collision
    probing toward navigation/selection, or (c) both.
    Each adds engineering cost beyond the iter_036
    estimate. The cheap gates have done their job:
    a substantial scope/risk update is now on the
    table before any compute is spent.
*   **Path-(iii) explicitly rejected (iter_037,
    DOCUMENTED):** decoder relaxation does not
    address the current blocker. The four-iteration
    null is that perception is not behaviorally
    load-bearing in the tested environments — i.e.
    the agent does not need to look in order to act
    well. A decoder enables a different evaluation
    style (reconstruction quality) but does not
    make perception necessary for action. Moreover,
    iter_031 already established that mean-pool
    reconstruction fails as a representation
    shaper. Path (iii) is now off the table on
    principled, not aesthetic, grounds.
*   **Active Direction (HUMAN DECISION POINT,
    iter_038):** the project is now blocked on a
    human go/no-go between:
    - Path (i) 2D — now with explicit
      Gate-1/Gate-1b tension cost added, requiring
      either a wider 2D sweep, a task redesign, or
      both; estimated 7–10 agent-iterations and ~4×
      FLOPs of 1D work, with the rebuild scope
      documented (1D-conv → 2D-conv backbone,
      re-run iter_020–032 representation work,
      2D soft-argmax centroid).
    - Path (ii) re-frame — consolidate the
      representation-quality + thalamic-gating
      + analytical-ceiling-gate methodology
      deliverables that ARE testable without
      bracketed behavioral validation; low
      compute cost; must commit to specific
      falsifiable representation/gating claims
      rather than degenerating into
      "ship what we have."
    - Path (iii) — REJECTED.
    Iter_038 must NOT proceed until a human
    decision is in. If iter_038 is invoked
    autonomously before that decision, it should
    do CONSOLIDATION (writing up the
    four-iteration null + 2D cheap-gate finding
    + methodological primitives) rather than new
    experimental work — this is path-(ii)-flavored
    preparation that does not foreclose path (i)
    and is recoverable cost if path (i) is later
    chosen.
*   **What is now solid:**
    - **The four-iteration 1D null is a documented
      standalone finding,** not just a journal
      observation. It forecloses the 1D testbed
      for the curiosity-driven perception-action
      thesis under the ORACLE-vs-RANDOM bracket.
      This is the clean result of the iter_033–036
      chain regardless of what comes next.
    - **The structural-ceiling-gate primitive has
      now correctly killed FIVE experiments**
      (iter_033 metric saturation, iter_034 MALRE
      active-passive gap, iter_035 PASSIVE
      collision count, iter_036 RANDOM CV, and
      iter_037 2D gates). The primitive is
      production-ready as a project methodology.
    - **2D is not the cheap win it appeared to be
      in the iter_036 escalation.** Gate-1
      passes — the 1D collision constraint is
      genuinely removed — but Gate-1/Gate-1b
      opposition surfaces a NEW design problem.
      Path (i) cost is materially higher than
      previously estimated.
    - **Path (iii) is principled-out,** not just
      deferred.
    - **Substrate unchanged:** iter_028 separate
      backbone + mask_dyn_sim + coord_vicreg
      (ΔR²_color ≈ 0.045, 0% collapse) remains the
      working representation. M1 batch-VICReg,
      M3 frozen-dim d_t=3 + GDASR log-only,
      decoder-free constraint all hold for paths
      (i) and (ii).
*   **What is now retired or contested:**
    - **The naïve form of path (i) (drop-in 2D
      environment, reuse pointer-collision
      behavioral test):** falsified by Gate-1b/
      Gate-2 at the tested parameterization. A
      viable path (i) requires task redesign on
      top of dimensionality change.
    - **Path (iii):** retired on principled
      grounds (mis-targeted vs the actual blocker).
    - **All retirements from prior iterations
      carry forward unchanged.**
*   **Confidence Score:** 40% (+1 from iter_036's
    39%). The slight increase reflects net
    methodological gain: the 1D null is now a
    defended finding, path (iii) is cleanly
    eliminated, path (i)'s true cost is now visible
    before commitment, and the
    structural-ceiling-gate primitive has been
    validated on its fifth use. The score is not
    higher because the project still faces a
    binding meta-decision with the lowest-cost
    remaining option (path ii) being a deliverable
    change rather than a technical advance. The
    score will move once a human go/no-go is made
    and the chosen path produces measurable
    progress.

## 2. Strategic Insights & Lessons Learned
*   **DE-RISKING-BEFORE-COMMITMENT IS THE CORRECT
    USE OF THE STRUCTURAL-CEILING-GATE PRIMITIVE
    AT META-DECISION SCALE (iter_037,
    METHODOLOGICAL FINDING, PROMOTED):** The gate
    primitive that the project developed for
    within-iteration triage has now been applied
    one level up — to a between-iterations
    escalation choice. Spending ONE cheap iteration
    to measure whether path (i) actually delivers
    its theoretical benefit (random-walk coverage
    non-uniformity in 2D) BEFORE committing
    7–10 iterations of rebuild is a textbook
    application of the project's own discipline.
    The result (Gate-1 passes, Gate-1b/Gate-2 fail)
    materially changes the path-(i) cost estimate
    and surfaces a design problem (Gate-1/Gate-1b
    tension) that would otherwise have been
    discovered mid-rebuild at much higher cost.
    Adopt as standard protocol: any
    meta-escalation that requires substantial
    sunk cost must first pass a cheap-gate
    de-risking pass.
*   **STATIC-POINTER 2D HAS A FUNDAMENTAL
    GATE-1/GATE-1b TENSION (iter_037, STRUCTURAL
    FINDING):** Under a 2D arena with a static
    central pointer and pointer-collision probing,
    "rare collisions" (which Gate-1 requires) and
    "enough collisions for heterogeneity to be
    stable" (which Gate-1b requires) are in
    opposition at any single parameterization. The
    diagnostic prescription is that path (i)
    must either (a) widen the parameter sweep and
    hope to find a goldilocks zone, (b) abandon
    the pointer-collision test design in favor of
    a navigation or selection task, or (c) both.
    This is a real, measured constraint not
    anticipated in the iter_036 escalation.
*   **PATH (iii) IS NOT JUST DEFERRED, IT IS
    MIS-TARGETED (iter_037, STRATEGIC FINDING):**
    The four-iteration null is "perception is not
    behaviorally load-bearing in the tested
    environments." A decoder changes evaluation
    style; it does not make perception necessary
    for action. iter_031 separately falsified
    mean-pool reconstruction as a representation
    shaper, so the decoder route is doubly
    penalized. This is the first time in the
    project the decoder-free constraint has been
    defended on argumentative rather than
    stipulative grounds — a positive consequence
    of the four-iteration null.
*   **THE FOUR-ITERATION 1D NULL IS A STANDALONE
    DELIVERABLE (iter_037, STRATEGIC FINDING):**
    Independent of which path is chosen next,
    "1D × N=3 × 128px cannot make perception
    behaviorally load-bearing under an
    ORACLE-vs-RANDOM bracket because either
    passive saturates information acquisition
    (full-observation regimes) or random already
    covers space uniformly (partial-observation
    regimes)" is a publishable structural result
    about the design of behavioral-perception
    benchmarks. Documenting this carefully is
    genuine scientific output, not merely a record
    of failure.
*   **WHEN THE PROJECT'S OWN DISCIPLINE PROHIBITS
    THE OBVIOUS NEXT STEP, DISCIPLINE WINS
    (iter_037, META-METHODOLOGICAL FINDING):**
    The temptation in iter_037 was to autonomously
    pick path (i) and start rebuilding — "we know
    2D is more interesting, let's just go." The
    user hint enforced the discipline of cheap
    de-risking + decision support instead. That
    discipline produced a result (Gate-1/Gate-1b
    tension) that an autonomous pick would have
    hit weeks of work later, at much higher cost.
    Future Manager critique should default to this
    pattern when meta-escalation is in play.
*   **CARRIED FORWARD (unchanged):**
    - M1 (pooled/batch VICReg) stands.
    - M2 status: "untestable under any tested 1D
      observation regime" — four-iteration null,
      not falsified.
    - M3 (fixed dimensionality d_t=3, GDASR
      log-only) stands.
    - iter_028 substrate (separate backbone +
      mask_dyn_sim + coord_vicreg) = 0% collapse.
    - Decoder-free constraint stands AND is now
      defended on argumentative grounds.
    - No positional encoding.
    - Pre-registered decision rules continue to
      produce clean outcomes (ten consecutive
      iterations: 023–024, 029–037).
    - ORACLE-bracket methodology stands as the
      confound disambiguator for behavioral
      evaluation IF a bracket-able environment is
      ever found.
    - Metric saturation must be computed and
      reported before any metric is adopted.
    - Median-of-repeated-events beats single-event
      least-squares for active policies.
    - Per-condition surprise-EMA recalibration
      required for any motor-routed bracket.

## 3. Loop & Bottleneck Detection
*   **Human-Decision Bottleneck (NEW, BINDING):**
    The project is now blocked on a human go/no-go
    between path (i) (now more expensive than the
    iter_036 estimate, with task redesign added)
    and path (ii) (re-frame deliverable). This is
    not a within-Manager-authority decision —
    it is a scope/goals decision. Iter_038 must
    not autonomously resolve it. If autonomously
    triggered before a decision, default to
    path-(ii)-flavored consolidation (writing up
    the null + cheap-gate findings + methodology),
    which is recoverable if path (i) is later
    chosen.
*   **De-Risking-Before-Commitment Primitive
    (NEW, PROMOTED):** Cheap-gate de-risking
    applied to meta-escalations themselves is now
    validated and should be standard protocol for
    any future high-cost path proposal.
*   **Structural-Ceiling-Gate Primitive
    (INSTITUTIONALIZED, fifth successful use):**
    Continues to deliver. Carry forward.
*   **Gate-1/Gate-1b-Tension Loop (NEW):** Any
    future bracket design with a static probe
    element must check that rarity-of-events and
    stability-of-CV are simultaneously satisfiable
    at the chosen parameterization. Add to the
    Dual-Failure-Mode loop checklist.
*   **Dual-Failure-Mode Loop (ACTIVE, unchanged):**
    Any future bracketed-behavioral evaluation
    must check BOTH "passive doesn't saturate"
    AND "random doesn't already cover" with
    pre-registered gates, plus the new
    "events-are-stable" Gate-1b.
*   **Sunk-Cost-Avoidance Loop (ACTIVE, FAVORABLE):**
    Held. iter_037 did not start the 2D rebuild
    despite the temptation; the cheap gate found
    the design problem at fractional cost.
*   **Metric-Saturation Loop (ACTIVE):** Carry
    forward.
*   **Geometric/Topological-Constraint Loop
    (ACTIVE):** Still mandatory. iter_037 added a
    new geometric constraint (rare-events vs
    stable-CV at fixed parameterization).
*   **ORACLE-Implementation-Correctness Loop
    (DORMANT):** No ORACLE built in iter_037.
*   **Motor-Protocol-as-Confound Loop (DORMANT).**
*   **Diagnostic-vs-Constructive Iteration Loop
    (DORMANT):** Ten consecutive clean
    pre-registered iterations.
*   **Overclaim Loop (DORMANT):** iter_037
    reported the 2D cheap-gate result as
    "blocks path (i) at tested parameterization"
    rather than "2D doesn't work" — appropriately
    bounded.

## 4. Alternate Research Paths
*   **iter_038: HUMAN-DECISION-DEPENDENT.**
    - If human selects path (i) with task
      redesign: iter_038 pre-registers the wider
      2D parameter sweep and/or the
      navigation-or-selection task design, with
      cheap gates on each parameterization
      candidate before any training compute.
      Engineering cost estimate (revised):
      10–14 agent-iterations including task
      redesign.
    - If human selects path (ii) re-frame:
      iter_038 begins consolidation —
      formally write up (a) the four-iteration
      1D null, (b) the 2D cheap-gate result, (c)
      the iter_028 representation substrate and
      its non-collapse properties, (d) the
      analytical-ceiling-gate methodology, (e)
      what falsifiable representation + gating
      claims survive and how each would be
      validated without a behavioral bracket.
    - If no human decision: iter_038 defaults to
      consolidation (path-(ii)-flavored, no new
      compute) — recoverable cost if path (i) is
      later chosen, and produces durable artifacts
      regardless.
*   **Path (i) revised cost estimate:** 10–14
    agent-iterations (was 7–10 in iter_036).
    Increase reflects the Gate-1/Gate-1b tension
    requiring task redesign on top of arena
    dimensionality change.
*   **Path (ii) consolidation:** scope unchanged
    from iter_036. Must commit to specific
    falsifiable claims (identity-disentanglement
    ΔR² thresholds, attention-token-trace
    properties, surprise-EMA calibration tests,
    analytical-ceiling-gate primitive
    demonstration on a fresh task) to avoid
    degenerating into "ship what we have."
*   **Path (iii):** REJECTED. See iter_037
    strategic insights.
*   **Causal Sensitivity Probe (DEFERRED):**
    contingent on path (i).
*   **Augmentation-Based Self-Supervision,
    Micro-Columns, Hierarchical Pyramid,
    Phase-5 GDASR Reactivation:** DEFERRED,
    unchanged.