# Research Manager Log - Iteration 036

## Iteration 036 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints — iter_035

1. **Honor the pre-commitment, do not re-litigate it.** iter_034's escalation rule forces iter_035 to be the foveated/non-physical-gaze redesign with RANDOM and ORACLE only (LEARNED deferred). Do not let the Planner reopen "should we try one more full-observation tweak" or expand scope to include LEARNED in the same iteration — the journal pre-registered exactly one foveation iteration before meta-escalation, and that budget must be respected. The Planner's job is to specify the foveated environment, the probe action, and the gates; not to redesign the strategy.

2. **The analytical ceiling gate must be inverted for this iteration, and pre-registered before any training.** Under full observation the failure mode was "PASSIVE already saturates"; under foveation the symmetric failure mode is "RANDOM gaze already covers every object adequately, leaving ORACLE no room." Require the Planner to pre-declare (a) the coefficient-of-variation gate on per-object probe-event counts under RANDOM (journal: CV ≥ 0.5), (b) the ORACLE surprise-scale and event-alignment sanity checks from iter_033/034, and (c) the per-condition surprise-EMA recalibration protocol if any arm routes through CLTSMotorController. Gate evaluation must precede the full bracket, exactly as in iter_034.

3. **Hold the line on scientific discipline around the metric.** The primary metric must be the median of repeated probe-induced events per object under a fixed probe budget — single-event least-squares is already falsified (iter_034.2) and must not reappear. The bracket-opening criterion stays ORACLE − RANDOM ≥ 0.15 with the lower CI clear of zero over ≥5 seeds (hard seeds 53/71 included); any softer language ("trend toward", "approaching") in the plan is to be rejected. Also require the Planner to pre-commit Arm A (foveation only) vs Arm B (foveation + pass-through obj-obj) as the factorial design the journal already specified, so the contribution of each lever is identifiable rather than confounded.

---

## Iteration 036 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
In a foveated 1D physics sandbox (N=3 objects, ghostly gaze pointer with
GAZE_RADIUS=8 pixels, probe action applying 1D elastic collision between
gaze [M_gaze=10] and nearest object within gaze window, probe budget B=20
over 2000 steps) — under both Arm A (normal obj-obj collisions retained)
and Arm B (pass-through obj-obj, only probe reveals dynamics) — the ORACLE
targeted-exploration policy (PD-tracks least-probed object, probes when
target center is within GAZE_RADIUS of gaze and |error|≤6.0) achieves
Per-Object Median Log-Ratio Error (POMLRE) at least 0.15 lower than a
RANDOM gaze+probe policy, with the lower bound of the two-sided 95% paired
bootstrap CI of (RANDOM_POMLRE - ORACLE_POMLRE) clear of zero over 8 seeds
(including hard seeds 53, 71). The ordering
ORACLE_POMLRE < RANDOM_POMLRE < PASSIVE_POMLRE holds in the mean within
each arm. The success condition is: the result is consistent with foveated
gaze making perception-driven targeting load-bearing for mass estimation
under a finite probe budget — NOT "perception sufficiency is established."

**Proposed Falsification Criterion:**
The hypothesis is falsified if ANY of the following holds in EITHER arm:

(F1) RANDOM_POMLRE - ORACLE_POMLRE < 0.15 (ORACLE does not substantially
     outperform RANDOM on the primary metric).
(F2) The lower bound of the two-sided 95% paired bootstrap CI (10000 resamples)
     of (RANDOM_POMLRE - ORACLE_POMLRE) includes zero (gap not statistically
     reliable).
(F3) Any ORACLE sanity check fails (if F3 fires, the ORACLE implementation is
     buggy and no comparison is interpreted):
     S1: ORACLE achieves ≥3 probe-induced collision events per object
         (mean across seeds) in the arm.
     S2: ORACLE probe success rate ≥ 60% (≥12 of 20 probe attempts result
         in a collision event — an object was found within the gaze window).
     S3: ≥80% of ORACLE's probe-induced collision events have |Δv_obj| > 1.0
         (informative velocity change for mass estimation).
     S4: No single object receives >80% of ORACLE's total probe events
         (even targeting should distribute across objects).
     S5: ORACLE gaze stays in bounds ≥95% of steps.
     S6: Each of the 3 objects receives ≥10% of ORACLE's total probe events
         (targeting covers all objects, not just one).
(F4) Ordering sanity check violated: mean(PASSIVE_POMLRE) > mean(RANDOM_POMLRE)
     > mean(ORACLE_POMLRE) does NOT hold within the arm. If violated, the
     metric is rejected as in iter_034-v1, regardless of F1/F2.
(F5) CV gate fails: coefficient of variation (std/mean) of per-object
     probe-event counts under RANDOM < 0.5 in the arm. This means RANDOM
     distributes probes too evenly for ORACLE to improve on coverage.
(F6) Coverage-vs-estimation decomposition: if the estimation-only gap (on
     seed×object cells where BOTH ORACLE and RANDOM have ≥3 valid events)
     is < 0.05 in BOTH arms, the finding is "ORACLE wins by coverage, not
     by perception-quality discrimination" — which is still a valid
     benchmark-validation outcome, but must not be sold as the latter.

**Proposed Method:**
Step 1: Create src/run_iter036_benchmark.py implementing:

A. FoveatedGazeSandbox — subclass of PhysicsSandbox with:
   - Ghostly pointer: during the physics substep loop, the pointer passes
     through objects (no pointer-object collision resolution). The pointer
     still bounces off walls at 0 and 128, and still receives acceleration
     from action['acc']. This breaks the 1D geometric inevitability that
     caused iter_035's ceiling gate failure.
   - Probe action: when action['probe']=True and probe_budget > 0:
     1. Find the nearest object with center within GAZE_RADIUS=8 pixels
        of the gaze center.
     2. If found: apply 1D elastic collision between gaze (M_gaze=10.0)
        and that object using the standard formula. Update both gaze
        velocity and object velocity. Record pre-step and post-step
        velocities for mass estimation. Decrement probe_budget.
     3. If not found: probe is wasted (no collision). Decrement
        probe_budget.
   - Foveated render: override render() so that only pixels within
     [gaze_pos - GAZE_RADIUS, gaze_pos + GAZE_RADIUS] are visible.
     Outside this window, canvas is zero. (For future LEARNED condition;
     does not affect ORACLE/RANDOM benchmark metrics.)
   - pass_through parameter: if True, objects also pass through each
     other (same as PassThroughPhysicsSandbox from iter_035). This
     enables the Arm B factorial condition.
   - Collision event recording: for each step where a probe occurs,
     record (step, obj_idx, v_gaze_pre_step, v_obj_pre_step,
     v_gaze_post_step, v_obj_post_step). Pre-step velocities are
     recorded BEFORE the probe collision is applied. Post-step
     velocities are recorded AFTER the full physics substep loop
     completes. The Δv values include confounding physics (wall
     bounces, obj-obj collisions) — same noise structure as iter_035.

B. Three conditions (NO learned representation — benchmark validation only):
   - ORACLE: Perfect knowledge of all object positions and velocities.
     * Maintains per-object probe-event count.
     * PD-tracks the least-probed object (Kp=2.0, Kd=0.5).
     * When the target's center is within GAZE_RADIUS=8 of the gaze center
       AND |error| ≤ 6.0, probes.
     * After probing, switches target to next least-probed object.
     * After probe budget exhausted, continues PD tracking without probing.
   - RANDOM: Random gaze acceleration ∈ [-10, 10]. Random probe with
     probability p = 0.01 per step (expected ~20 probes over 2000 steps)
     until budget exhausted. No targeting.
   - PASSIVE: Gaze starts at 64.0, no acceleration, no probing.
     0 probe events per object → POMLRE = 2.0 (metric ceiling).

C. Metric — POMLRE (Per-Object Median Log-Ratio Error, same as iter_035):
   For each object i:
     1. Collect probe events for object i with |Δv_obj| > 1.0.
     2. m_est_k = -M_gaze * Δv_gaze_k / Δv_obj_k for each event k.
     3. If ≥3 valid events: m_hat_i = median(m_est_k),
        error_i = |log(m_hat_i / m_true_i)|.
     4. If 1-2 valid events: m_hat_i = mean(m_est_k),
        error_i = |log(m_hat_i / m_true_i)|.
     5. If 0 valid events: error_i = 2.0 (maximum penalty).
   POMLRE = mean(error_i across 3 objects).

D. Factorial design:
   - Arm A: Foveated gaze + normal obj-obj collisions (foveation lever alone).
   - Arm B: Foveated gaze + pass-through obj-obj collisions (both levers:
     foveation removes incidental pointer-object info; pass-through removes
     free obj-obj dynamics info).
   Primary metric computed separately per arm. Factorial comparison
   identifies the contribution of each lever to the ORACLE-RANDOM gap.

Step 2: Analytical ceiling gate (pre-registered, computed first).
   Run RANDOM for 5 seeds × 2 arms = 10 short episodes (2000 steps each).
   For each arm, compute:
   - Per-object probe-event counts under RANDOM.
   - CV = std(counts) / mean(counts) across the 3 objects, averaged over seeds.
   - Mean per-object event count.
   Gate criteria (BOTH arms must pass):
   - CV ≥ 0.5: RANDOM coverage is sufficiently uneven for ORACLE to improve.
   - Mean per-object count ≥ 0.5: RANDOM gets at least some events.
   If the gate fails in either arm, report: "Foveated gaze with GAZE_RADIUS=8
   does not create sufficient coverage imbalance under RANDOM policy in Arm
   [A/B]." Do NOT proceed to full bracket for that arm.

Step 3: If gate passes for both arms, run full bracket.
   8 seeds × 2 arms × 3 conditions = 48 episodes, 2000 steps each.
   Seeds: [7, 31, 53, 71, 83, 97, 113, 163] (hard seeds 53, 71 included).

Step 4: Analysis.
   - Per-seed POMLRE for each condition × arm.
   - Bootstrap CI (10000 paired resamples, paired by seed).
   - Sanity checks S1-S6.
   - Coverage-vs-estimation decomposition: restrict to seed×object cells
     where BOTH ORACLE and RANDOM have ≥3 valid events; compute estimation-
     only gap.
   - Factorial comparison: gap in Arm A vs gap in Arm B.

Step 5: Report with honest null-finding protocol.
   - If all gates pass (F1-F6): hypothesis supported — foveated gaze makes
     perception load-bearing.
   - If F1/F2 fail but F5 passes: null — foveated gaze creates coverage
     imbalance but this doesn't translate to a ≥0.15 POMLRE gap.
   - If F5 fails: null — GAZE_RADIUS=8 too wide for RANDOM to have uneven
     coverage; triggers meta-escalation.
   - If F6 fires: report "ORACLE wins by coverage, not perception-quality
     discrimination."
   - No softer language ("trend toward", "approaching") per manager directive.

FILES CREATED:
- src/run_iter036_benchmark.py (new, main experiment)
- archive/iter_036/results/ (output directory)

PRESERVED: iter_028 substrate (separate backbone, mask_dyn_sim, coord_vicreg,
0% collapse), d_t=3 frozen, GDASR log-only (M3), decoder-free, no positional
encoding, M2 not reopened. No LEARNED representation used. No CLTSMotorController
in the loop (per-condition surprise-EMA recalibration therefore not required;
the iter_033 surprise-scale and event-alignment sanity checks are replaced by
probe-mechanics checks S1-S6, which verify that ORACLE's probes produce
physically sensible collision events rather than verifying surprise-signal
properties).

PRE-COMMITTED ESCALATION:
If the experiment produces a null result (ORACLE-RANDOM gap < 0.15 in BOTH
arms, or CV gate fails in both arms), that finding triggers meta-escalation:
the project must confront whether the 1D sandbox itself is the structural
confound. Options: (i) 2D environment redesign, (ii) re-frame deliverable
around representation+thalamic gating claims without behavioral validation,
(iii) revisit decoder-free constraint. No additional foveated-gaze parameter
adjustments (e.g., different GAZE_RADIUS) before meta-escalation — this is
the one foveation iteration budgeted by the journal.

---

## Iteration 036 -> Planner [Strategic Guidance]

Manager's Note — iter_036 (Skeptic Review of Proposed Plan)

**1. The plan is broadly sound and honors the pre-commitment — endorse with three sharpenings, not a rewrite.** Foveated/non-physical gaze, RANDOM+ORACLE only (LEARNED deferred), POMLRE primary, factorial Arm A vs Arm B, analytical CV gate first, ≥5 seeds (you specified 8 including 53/71), and pre-committed meta-escalation on null — all required elements are present. Do not let this turn into another redesign cycle; lock the plan and run it. The pre-registration file `src/pre_registration.md` must be written and committed *before* any simulation code is executed (the Orchestrator will auto-write it from your hypothesis/falsification fields — confirm that fields F1–F6 and S1–S6 land in that file verbatim, including the 8 seeds, GAZE_RADIUS=8, B=20, 2000 steps, M_gaze=10, and the |Δv_obj|>1.0 cutoff).

**2. Construction-vs-empirical check on the probe mechanism — this is the highest-risk flaw I see.** The probe is defined as "apply 1D elastic collision between gaze and the nearest object within the window," and the metric `m_est = -M_gaze · Δv_gaze / Δv_obj` inverts that same collision formula. If pre-step velocities are read at the instant of the probe, that estimator is a definitional identity and ORACLE will trivially win by getting more clean events — a constructional, not empirical, result. The plan partially acknowledges this by recording *post-substep* velocities (so wall bounces and obj-obj events in Arm A add noise), but you must pre-register **explicitly** which Δv enters the estimator and confirm it is the *across-substep* delta, not the across-collision-only delta. Furthermore, the F6 coverage-vs-estimation decomposition is the right instrument here and must be reported on equal footing with F1/F2, not as a footnote: if the estimation-only gap collapses to <0.05 in both arms, the honest framing is "ORACLE wins by coverage allocation, foveated gaze validates as a coverage-discrimination benchmark" — which is a perfectly respectable finding, but it is **not** "perception is load-bearing for mass estimation." State this explicitly in the pre-registration and in the eventual report.

**3. Two tightenings before lock-in, and a language-hygiene flag.** (a) The CV gate threshold (≥0.5) and "mean per-object count ≥0.5" are reasonable but must be evaluated **only** on the analytical-gate seeds (the 5 short rollouts), with the gate decision made and logged *before* the 8-seed bracket is launched — no peeking at bracket data to retune the gate. (b) The ORACLE sanity check S1 (≥3 events/object) and the metric's "≥3 valid events → median" branch can interact: if RANDOM frequently falls into the "1–2 events → mean" or "0 events → 2.0" branches while ORACLE is in the median branch, the gap is partly a fallback-rule artifact. Pre-register the per-condition distribution over the three branches as a reported diagnostic, and require that the headline POMLRE gap be re-computed with all three conditions forced into the same branch (e.g., truncating to seed×object cells with ≥3 valid events for all three) — the F6 decomposition already partly does this; make it primary, not optional. (c) Language: drop "perception sufficiency is established" entirely from the hypothesis text — your draft already uses "is consistent with," which is correct; propagate that discipline to the eventual report. Avoid "validates," "demonstrates," "proves" regardless of outcome; the honest framings are "is consistent with foveated gaze making perception load-bearing under the declared protocol" or "does not refute the null that foveated gaze is insufficient." A clean null here, with the pre-committed meta-escalation to 2D / re-framing / decoder-revisit, is a first-class success of the method — treat it as such if it fires.

---

## Iteration 036 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
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

---

## Iteration 036 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 036 — Null Result: Foveated Gaze in 1D Cannot Make Perception Load-Bearing; Four-Iteration Environment-Design Null Chain Complete

## 1. Pre-Declared Hypothesis and Falsification Criterion
Verbatim from the iter_035/036 pre-registration recorded in
the prior journal:

- Hypothesis: A 1D environment in which the pointer is a
  non-physical gaze locus, object state is observable only
  within a foveated window of radius r, and excitation
  requires an agent-issued localized probe action, will
  cause coverage of objects to depend on gaze policy,
  enabling ORACLE − RANDOM ≥ 0.15 on a per-object
  mass-estimation metric.
- Analytical ceiling gate (pre-registered, computed first):
  under RANDOM policy, the coefficient of variation of
  per-object probe-event counts must be ≥ 0.50, computed
  analytically or by a short rollout. Below this threshold,
  RANDOM already distributes coverage too evenly for ORACLE
  to have headroom, and the bracket experiment is blocked.
- Two arms factor the two levers: Arm A retains obj-obj
  collisions (foveation alone); Arm B removes them
  (foveation + pass-through).
- Escalation rule: if the CV gate fails in both arms, the
  foveation lever is also insufficient and the
  pre-committed meta-escalation (path i/ii/iii in the
  journal) triggers without further 1D iteration.

## 2. Experimental Protocol
- Arena: 1D, 128 pixels, N=3 objects.
- Pointer: non-physical gaze locus, no collisions with objects.
- Observation: object state available iff object centroid
  within gaze window of radius GAZE_RADIUS=8 pixels.
- Excitation: localized probe-force impulse at the gaze
  locus is the only mechanism that produces object dynamics
  changes.
- Arms:
  - Arm A: foveation + obj-obj collisions retained.
  - Arm B: foveation + pass-through obj-obj physics.
- Policy under measurement: RANDOM gaze trajectory.
- Measurement: per-object probe-event counts under a fixed
  rollout; coefficient of variation across the 3 objects.
- Gate: CV ≥ 0.50 required for full-bracket execution.
- Compute scope: only the analytical ceiling gate was
  executed; the full ORACLE-vs-RANDOM bracket and any
  learned model training were blocked by the gate failure.

## 3. Observed Quantities
- Arm A RANDOM per-object event-count CV: **0.36**.
- Arm B RANDOM per-object event-count CV: **0.46**.
- Pre-registered threshold: **0.50**.
- Result: gate fails in both arms (Arm A by 0.14, Arm B by
  0.04). Arm B is closer to threshold but still below.
- Cumulative null chain across four iterations of
  environment redesign:
  - iter_033: ORACLE − RANDOM ≈ 0 on behavioral pivot.
  - iter_034: v2 MALRE ORACLE − RANDOM = 0.031 within
    active regime.
  - iter_035: PASSIVE pointer 12.27 valid collisions per
    object vs threshold 3.0 (saturation gate, 4× overshoot).
  - iter_036: RANDOM CV 0.36 / 0.46 vs threshold 0.50
    (heterogeneity gate, both arms below).

## 4. Verdict
**Refuted.** The pre-declared hypothesis that foveated gaze
in a 1D × N=3 × 128-pixel arena makes perception
behaviorally load-bearing under an ORACLE-vs-RANDOM bracket
is refuted: the necessary precondition (RANDOM coverage
heterogeneity) is not met under either tested arm. In
combination with the prior three iterations, the broader
hypothesis that *any* tested 1D-sandbox configuration can
produce ORACLE − RANDOM discrimination on a behavioral
metric is refuted across four pre-registered, mechanistically
distinct redesigns. The pre-committed meta-escalation to a
choice among (i) 2D environment, (ii) re-frame the
deliverable, or (iii) relax the decoder-free constraint is
triggered. Iter_037 will execute that meta-decision; no
further 1D environment redesign is sanctioned.

## 5. Construction-vs-Empirical Note
This result is empirical, not definitional. The CV
threshold of 0.50 is a pre-registered design choice
motivated by the need for ORACLE to have measurable
headroom over RANDOM; it was not set to guarantee any
particular outcome and could in principle have been
exceeded by a redesign that concentrated objects spatially
or used a more restrictive gaze radius. The measured CV
values (0.36, 0.46) are properties of how random gaze
trajectories of finite length cover a 1D arena of 128
pixels containing 3 objects — a geometric-statistical
property of the test environment under the chosen radius.
The four-iteration null chain itself is a stronger
empirical statement: each iteration tested a distinct
mechanism (collision pattern, environment density,
geometry, observation gating) and each failed at a
structurally different gate. The cumulative pattern is
empirical evidence for the constraint, not a derivation
from it.

## 6. Limitations
- The CV gate failure was tested only at GAZE_RADIUS=8
  pixels. A substantially smaller radius (e.g. 2 pixels)
  might push CV above 0.50, but would create a separate
  exploration problem and would not have been within the
  pre-registered parameter envelope. Whether *any* radius
  in (0, 8] could pass the gate has not been measured.
  This is acknowledged but not chased, because moving
  outside the pre-registered envelope to recover the
  hypothesis would violate Gate 3 hygiene.
- The result does not show that no 1D environment could
  ever produce an open bracket; it shows that the four
  principled, mechanism-distinct redesigns tried so far
  have not, and the prior on further 1D tweaks succeeding
  is now low.
- The result does not invalidate the iter_028 representation
  substrate or the MALRE v2 coverage test; both remain
  valid within their established scopes.
- The result does not pre-judge the iter_037 meta-decision.
  It only forces that decision to be made.
- The four-iteration sequence does not establish
  representation-quality conclusions of any kind; it
  establishes only environment-design conclusions. The M2
  mandate's "untestable" status is unchanged.
- The analytical ceiling gate primitive has now saved
  compute four times, but no positive validation of any
  behavioral claim has been produced; the project's
  methodological maturity is currently outpacing its
  scientific progress, which is itself the signal driving
  the meta-escalation.

---

