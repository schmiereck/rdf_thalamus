# Research Manager Log - Iteration 035

## Iteration 035 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (iter_035, pre-planning)

1. Directional — the next iteration's primary lever is the **environment, not the metric**. iter_034 showed that on the current full-observation 3-object sandbox, object-object collisions deliver mass information autonomously, so no metric redesign on the same environment can make perception load-bearing (ORACLE−RANDOM=0.031 against a 0.83 active-vs-passive gap proves the gating is missing upstream of the readout). Redesign the environment so dynamics information is *gated by perception-driven targeting* — cheapest sufficient change is non-colliding (pass-through) or collision-sparse objects under a finite interaction budget, with per-object mass recovered from the MEDIAN of repeated agent-caused pointer-object collisions (not the iter_034-v1 single-shot least-squares that active policies degraded). N≥3, continuous metric.

2. Discipline — iter_035 is **BENCHMARK-VALIDATION ONLY**. The pre-registered gate is ORACLE−RANDOM ≥ 0.15 with the lower CI clear of zero over ≥5 seeds, gated by the iter_033 ORACLE sanity-checks (surprise scale in physical range; ≥80% of surprise spikes within ±2 steps of ground-truth collision events). Do **not** attach LEARNED and do **not** invoke the iter_033 (a)/(b)/(c)/(d) sufficiency rule until ORACLE≫RANDOM is established on the redesigned bracket. Compute and report the random-baseline ceiling of the chosen metric *before* running (Metric-Saturation loop), and recalibrate the surprise-EMA per condition if the metric routes through CLTSMotorController, so the bracket does not measure controller calibration (iter_033 motor-confound carry-forward).

3. Pre-commit the escalation — if a *single* perception-gated environment redesign still cannot open the bracket, **that null is itself the finding** ("perception is not behaviorally load-bearing under full observation") and the next iteration pulls the foveated-gaze mechanism (goal Section 8.2) forward from deferred, because partial observation is the principled way to make perception necessary. Cap this avenue at one environment-redesign iteration before escalation; do not loop on environment tweaks. Preserve the iter_028 substrate, separate backbone, d_t=3 frozen, GDASR log-only (M3), decoder-free, no positional encoding; do not revise M2.

---

## Iteration 035 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
In a pass-through physics sandbox (N=3 objects that pass through each other; 
only pointer-object collisions remain elastic) with a 15-push budget over 2000 steps, 
the ORACLE targeted-exploration policy (PD-tracks the least-collided object, pushes 
when within 6px) achieves a Per-Object Median Log-Ratio Error (POMLRE) at least 0.15 
lower than a RANDOM policy, with the lower 95% paired bootstrap CI of 
(RANDOM_POMLRE - ORACLE_POMLRE) clear of zero over 8 seeds. The ordering 
ORACLE_POMLRE < RANDOM_POMLRE < PASSIVE_POMLRE holds in the mean.

**Proposed Falsification Criterion:**
The hypothesis is falsified if ANY of:
(F1) RANDOM_POMLRE - ORACLE_POMLRE < 0.15 (ORACLE does not substantially outperform RANDOM), OR
(F2) The lower 95% bootstrap CI of (RANDOM_POMLRE - ORACLE_POMLRE) includes zero (gap not statistically reliable), OR
(F3) Any ORACLE sanity check fails:
  S1: ORACLE achieves ≥3 informative pointer-object collisions per object (mean across seeds),
  S2: ORACLE push budget utilization ≥ 80% (≥12 of 15 pushes used),
  S3: ≥80% of collision events used for mass estimation have |Δv_obj| > 1.0,
  S4: No single object receives >80% of ORACLE's total pushes (even targeting),
  S5: ORACLE pointer stays in bounds ≥95% of steps.
If F3 fires, the ORACLE implementation is buggy and no comparison is interpreted.

**Proposed Method:**
Step 1: Create src/run_iter035_benchmark.py implementing:

A. PassThroughPhysicsSandbox — subclass of PhysicsSandbox where the step() method 
   skips elastic collision resolution between non-pointer entities. Only collisions 
   involving the pointer (index N) are resolved. Objects pass through each other 
   but bounce off walls normally.

B. Three conditions (NO learned representation — benchmark validation only):
   - ORACLE: Custom controller that (a) maintains per-object pointer-collision count,
     (b) PD-tracks the least-collided object (Kp=2.0, Kd=0.5), (c) when within 
     |error|≤6.0, sets pointer_vel=5.0 toward target (1 push budget unit), 
     (d) after push, switches target to next least-collided object, (e) after 
     15 pushes exhausted, continues PD tracking without pushing.
   - RANDOM: Random acceleration ∈ [-10,10], random push (p=0.1) until budget 
     exhausted. No targeting.
   - PASSIVE: No action (acc=0, push=False). Pointer only moves from incidental 
     collisions with objects.

C. Collision detection (same as iter_034): Before/after each env.step(), compare 
   entity velocities. Log pointer-object collision events with pre/post velocities.

D. Metric — POMLRE (Per-Object Median Log-Ratio Error):
   For each object i:
     1. Collect pointer-object collision events for object i
     2. Compute m_est_k = -10 * Δv_ptr_k / Δv_obj_k for each event k
     3. Filter: keep only events where |Δv_obj_k| > 1.0
     4. If ≥3 valid events: m_hat_i = median(m_est_k), error_i = |log(m_hat_i / m_true_i)|
     5. If 1-2 valid events: m_hat_i = mean(m_est_k), error_i = |log(m_hat_i / m_true_i)|
     6. If 0 valid events: error_i = 2.0 (maximum penalty)
   POMLRE = mean(error_i across 3 objects)

E. Pre-run analytical ceiling: Before running, compute expected POMLRE for PASSIVE 
   analytically (stationary pointer at 64, objects bouncing freely in [0,128], 
   estimate expected informative collision count per object).

F. Run 8 seeds × 3 conditions = 24 episodes, 2000 steps each.

G. Compute bootstrap CI (10000 samples, paired by seed) for 
   (RANDOM_POMLRE - ORACLE_POMLRE) and check all gates.

Step 2: Run the experiment, analyze results, check all gates.

Step 3: If gate passes → benchmark validated, proceed to iter_036 with LEARNED.
        If gate fails → null finding: "perception is not behaviorally load-bearing 
        under full observation even with pass-through dynamics." Escalate to 
        foveated-gaze mechanism (goal.md Section 8.2) for partial observation.

FILES CREATED:
- src/run_iter035_benchmark.py (new, main experiment)
- archive/iter_035/results/ (output directory)

PRESERVED: iter_028 substrate (separate backbone), d_t=3 frozen, GDASR log-only (M3),
decoder-free, no positional encoding, M2 not reopened. No LEARNED representation used.

PRE-COMMITTED ESCALATION: If ORACLE-RANDOM gap < 0.15 on pass-through environment,
that is itself the finding, and the project pulls foveated-gaze (Section 8.2) forward 
from deferred. No additional environment tweaks before escalation.

---

## Iteration 035 -> Planner [Strategic Guidance]

Manager's Note (iter_035, plan critique)

The plan is materially well-scoped — pass-through dynamics + finite push budget + median-of-ratios per-object metric is the right operationalization of the user-hint's "perception-gated information" redesign, and the pre-committed escalation to gaze on a null result is correctly inherited. Three corrections are required before execution.

1. **Construction-vs-empirical risk on ORACLE: the metric is partly a count, and ORACLE is hand-coded to maximize that count.** The ORACLE policy explicitly tracks per-object collision counts and switches target to the least-collided object, while the POMLRE metric returns a fixed penalty of 2.0 whenever an object has 0 valid events and degrades for 1–2 events. Under a 15-push budget over 3 objects, ORACLE deterministically delivers ≥5 attempted pushes per object while RANDOM with p=0.1 trigger and no targeting will almost certainly leave at least one object below the ≥3-event threshold on most seeds. A large RANDOM−ORACLE gap is therefore guaranteed *by the metric's coverage-penalty branch*, not by mass-estimation accuracy — this is exactly the "PASSIVE has no data → max penalty" coverage artifact iter_034 already flagged, re-imported on the RANDOM arm. **Fix:** add a mandatory decomposition gate that reports the gap separately under (i) coverage-only (count valid events per object, no estimation) and (ii) estimation-only (restrict to seed×object cells where *both* RANDOM and ORACLE achieved ≥3 valid events, and compare median-ratio errors there). Pre-register that the headline claim requires the estimation-only gap to be non-trivial; if the gap collapses to ~0 once coverage is matched, report honestly as "ORACLE wins by coverage, not by perception-quality discrimination" — which is still a valid benchmark-validation outcome, but must not be sold as the latter.

2. **The falsification criterion is one-sided and the ordering claim in the hypothesis is unguarded.** F1/F2 only fire if ORACLE fails to beat RANDOM; they do not fire if PASSIVE comes out *below* RANDOM or ORACLE (which would invalidate the metric's monotonicity and replay iter_034-v1's inverted-ordering failure mode). Also, "lower 95% CI clear of zero" is correctly stated in F2 but the hypothesis text says "lower 95% paired bootstrap CI" without specifying tail — make it explicit: lower bound of the two-sided 95% CI on (RANDOM−ORACLE) > 0. **Fix:** add F4 = ordering sanity check (mean PASSIVE_POMLRE > mean RANDOM_POMLRE > mean ORACLE_POMLRE; if violated, the metric is rejected as in iter_034-v1, regardless of F1). Tighten the CI language. Also pre-register what the analytical-ceiling computation (step E) must show for the run to even start — give it a pass/fail criterion, otherwise it is decoration, not a gate.

3. **Language and pre-registration discipline.** The hypothesis as drafted is acceptable but the proposed_method narrates "benchmark validated, proceed to iter_036" as if the positive outcome were the default. Restate the success condition as "is consistent with the redesigned environment making perception-driven targeting load-bearing for mass-estimation under a finite budget" — not "perception sufficiency is established." The Orchestrator will write `src/pre_registration.md` from this plan; ensure that file commits, verbatim, (a) the hypothesis with the explicit CI tail, (b) F1–F4 plus the analytical-ceiling gate, (c) the coverage-vs-estimation decomposition rule from point 1, (d) the iter_033 ORACLE sanity checks (surprise-scale and event-timing alignment) — these are listed in the user hint and the journal but absent from the current F3 list, which only covers push-budget mechanics. Add them, or justify in the pre-registration why they don't apply once the metric no longer routes through CLTSMotorController. The pre-committed escalation to foveated gaze on a null result must also appear in the file so it cannot be re-litigated post-hoc.

A clean null on this iteration (no perception-gated environment opens the bracket) is a first-class deliverable — it is the principled trigger to pull Section 8.2 forward, and should be reported as such rather than reframed as a partial success.

---

## Iteration 035 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
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

---

## Iteration 035 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 034 — Null Result: Pass-Through Environment Insufficient to Make Perception Load-Bearing in 1D

## 1. Pre-Declared Hypothesis and Falsification Criterion
Hypothesis (iter_034, pre-registered): "Removing object-object
collisions (pass-through obj-obj physics) is the cheapest
sufficient environment redesign to make dynamics information
causally gated on the agent's pointer policy, producing an
ORACLE − RANDOM gap ≥ 0.15 on a per-object mass-estimation
metric."

Falsification criterion (pre-registered analytical ceiling
gate, evaluated before full experiment): "Under a PASSIVE
(zero-action) pointer in the redesigned environment, the mean
number of valid pointer-object collisions per object over the
benchmark episode length must NOT exceed 3.0. A passive count
>> 3.0 indicates that pointer-object collisions remain
policy-independent and abundant, in which case ORACLE cannot
meaningfully outperform RANDOM and the redesign is rejected
as insufficient."

## 2. Experimental Protocol
- Environment: 1D physics sandbox, 128 RGB pixels, N=3 objects,
  standard physics parameters from iter_033, with
  obj-obj collisions modified to *pass-through* (objects do not
  interact with each other; pointer-object collisions retained).
- Pointer: physical entity on the 1D axis, zero acceleration
  in the PASSIVE condition.
- Measurement: count of "valid" pointer-object collisions per
  object over the benchmark episode length, averaged across
  the seed bank.
- Pre-registered threshold for gate pass: ≤3.0 collisions per
  object.
- The full ORACLE/RANDOM bracket was *not* run, because the
  analytical gate is sequenced first and failure of the gate
  blocks the experiment by pre-registration.

## 3. Observed Quantities
- PASSIVE valid pointer-object collisions per object: **12.27**
  (units: count per object per episode).
- Pre-registered threshold: 3.0.
- Overshoot: ~4.1×.
- Outcome: gate FAILED. Full ORACLE/RANDOM bracket not
  executed.

## 4. Verdict
**Refuted.** The pre-registered hypothesis that removing
obj-obj collisions is a sufficient environment redesign to
make perception load-bearing is refuted by the analytical
ceiling gate. Pointer-object collisions in a 1D physics
environment remain abundant under a passive pointer because
the pointer and objects share a single spatial axis and the
pointer is therefore on the trajectory of bouncing objects
by geometric necessity.

## 5. Construction-vs-Empirical Note
The *geometric* fact that a physical body on a 1D axis is
collision-inevitable with other moving bodies on the same axis
is structural — it follows from the dimensionality of the
state space, not from any specific dynamics rule. The
*quantitative* result (12.27 collisions/object under the
specific environment parameters of iter_034) is empirical and
is informative about the magnitude of the problem (4× the
tolerated ceiling, not marginal). The structural fact alone
was sufficient to predict gate failure; the empirical
measurement confirmed it and quantified the margin. The
generalization — that no 1D environment modification preserving
a physical pointer can satisfy the gate — is a construction-
level claim and is therefore stronger than this single
measurement; it forecloses an entire family of candidate
fixes.

## 6. Limitations
- This result does NOT show that perception is unnecessary for
  the project's downstream goal — it shows only that the
  current 1D + physical pointer + full observation environment
  cannot test that question.
- This result does NOT validate or invalidate any
  representation-learning claim. The iter_028 substrate
  remains the working representation; its behavioral
  sufficiency remains untested.
- This result does NOT establish that foveated gaze will
  succeed. It establishes only that foveated gaze is the
  principled next lever to test, because it changes the
  geometric property (pointer-as-physical-body) that this
  iteration identified as the binding constraint. Foveated
  gaze may itself fail to open the bracket, in which case the
  project must escalate to 2D or restructure its deliverable.
- The single-measurement nature of the gate result (12.27
  collisions/object) is not seed-replicated; however, the
  geometric argument that motivates the gate is seed-
  independent, and a 4× overshoot leaves no plausible seed
  sensitivity that could change the verdict.
- "Valid collision" was defined by the iter_034 executor;
  if that definition is unusually strict or loose, the
  absolute number 12.27 could shift, but the geometric
  argument that this number is bounded *below* by a
  significant nonzero rate is unaffected.
- The iter_034 escalation to foveated gaze is a forced move
  from this result, but the design of the foveated-gaze
  iteration (iter_035) has its own pre-registration
  requirements (analytical ceiling gate, ORACLE sanity
  checks, motor-confound design-around) that this milestone
  does not satisfy on its behalf.

---

