# Research Manager Log - Iteration 033

## Iteration 033 -> Planner [Pre-Planning Hints]

## Manager's Pre-Planning Hints — iter_033

**Hint 1 (directional, binding).** Execute the user-mandated three-condition oracle bracket exactly as specified: RANDOM (lower bound), LEARNED (iter_028 mean-pool E1 substrate — *not* the collapsed centroid variants), and ORACLE (ground-truth positions/identity fed as `z_coord`/`z_dyn`, with surprise from a known-dynamics physics predictor). All three conditions must share environment, seed bank (n≥10, including hard seeds 53 and 71), and `CLTSMotorController` logic — any deviation in motor code or environment between conditions invalidates the bracket. The normalized gap `g = (LEARNED - RANDOM) / (ORACLE - RANDOM)` with its four-branch decision rule (a/b/c/d) must be written into the pre-registration verbatim, before the run, and committed to before any seed executes.

**Hint 2 (directional, scope discipline).** Resist three predictable scope drifts: (i) do not re-open the M2 objective comparison — it stays "untestable" until constraint-relaxation provides a working non-mean-pool readout; (ii) do not introduce a new representation-side ΔR² gate alongside the behavioral gate, since reintroducing one revives the dissolved representation-quality-gate loop; (iii) do not let "ORACLE was hard to build" become a reason to weaken its construction — if the oracle predictor is approximate, that approximation must be quantified and reported as a bracket-ceiling caveat, not absorbed silently. The primary behavioral metric for the `g` computation must itself be pre-declared (recommend post-collision probing selectivity, since it directly calibrates the iter_031 1.34× signal); secondary metrics (tracking error, mass-change causal sensitivity) get reported but do not drive the gate.

**Hint 3 (scientific discipline).** The oracle bracket is precisely the right instrument because it makes the `g` ratio *relative to a measured ceiling*, not a guessed absolute — preserve that relativization in the language of the report. Use restrained phrasing throughout: "the representation is consistent with sufficiency for this behavior," not "the representation is sufficient"; "perception does not appear to limit behavior at the measured n," not "perception is solved." Treat branch (c) — small `(ORACLE - RANDOM)` — as a first-class possible outcome that would *invalidate* the whole behavioral-pivot strategy by showing the task itself does not discriminate perception quality; if (c) fires, the report must say so plainly rather than reinterpreting partial signals as evidence for branches (a) or (d).

---

## Iteration 033 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
The best available mean-pool representation (SFA+VICReg sfa_weight=5.0, separate backbone,
ΔR²≈0.275, 0% collapse, iter_029 Arm B) supports functional surprise-driven behavior
that accounts for at least 20% of the gap between random (lower bound) and oracle
(perfect-perception upper bound) baselines, as measured by post-collision attention
selectivity on N=2 collision-sparse environments. Specifically, the normalized gap
g = (LEARNED - RANDOM) / (ORACLE - RANDOM) on post-collision attention selectivity
(version B: attended object matches max-velocity-change object) will be > 0.20,
with the four-branch decision rule determining project direction.

**Proposed Falsification Criterion:**
F1 FAIL: g <= 0.20 — the learned representation accounts for less than 20% of the
oracle-random gap, meaning the representation PROVABLY limits behavior and
constraint-relaxation (decoder, higher d_t, or the VICReg-upstream-of-gate fix
for the iter_032 cross-backbone collapse) is justified.

Additionally, if branch (c) fires — (ORACLE - RANDOM) is itself small (|difference| < 0.10
on primary metric) — the task or motor protocol is the bottleneck, NOT perception,
and the behavioral-pivot strategy itself is invalidated for this protocol/environment.

The lower 95% CI of g (bootstrapped across seeds) must be reported; if it excludes
the boundary between adjacent branches, the branch assignment is confident; if not,
both branches are reported as possible.

**Proposed Method:**
## Three-Condition Oracle Bracket Experiment

### Conditions (all sharing identical environment, seed bank, and CLTSMotorController logic)

1. **RANDOM (lower bound):** CLTSMotorController with random token_locus selection
   each step (no encoder, no prediction). The controller's attention_cooldown is
   set to 0 so locus is re-randomized every step, and all surprise EMA values are
   zeroed to prevent any learned structure from influencing behavior.

2. **LEARNED (test condition):** Load iter_029 Arm B checkpoints (SFA+VICReg,
   sfa_weight=5.0, separate backbone, mean-pool readout, d_max=8, d_t=3 trained).
   Evaluate with d_t=2 (first 2 channels active for N=2 objects). Run encoder +
   predictor forward pass, feed z_pred_coord, z_target_coord, z_pred_dyn,
   z_target_dyn to the standard CLTSMotorController.get_action().

3. **ORACLE (upper bound):** Feed ground-truth positions (info['positions'][:d_t])
   as z_coord, ground-truth mean-color (np.mean(info['colors'][:d_t], axis=1))
   as z_dyn, linear-extrapolation predicted positions (prev_pos + prev_vel * dt)
   as z_pred_coord, and z_dyn as z_pred_dyn (identity is constant). Feed these
   tensors to the SAME CLTSMotorController.get_action() method — identical
   surprise computation (per-channel MSE + EMA normalization + attention cooldown),
   identical PD tracking, identical push logic.

### Environment
PhysicsSandbox(N=2) — collision-sparse by design. Mass perturbation at step 1000:
multiply object 0's mass by 1.5× (same as iter_031 Part B). 2000 evaluation steps
per seed.

### Seed Bank
12 seeds: [7, 17, 31, 53, 71, 83, 97, 101, 107, 113, 137, 163]
Includes hard seeds 53 and 71 (mandated by user/manager).

### Primary Behavioral Metric
Post-collision attention selectivity (version B): for each collision event,
within POST_COLLISION_WINDOW=15 steps after the collision, the fraction of steps
where the attended object (token_locus mapped to object index via closest-centroid)
matches the max-velocity-change object. This is the same metric as iter_031 Part B,
directly calibrating the 0.59-vs-0.44 signal.

### Secondary Metrics (reported but do NOT drive the gate)
- Mean tracking error (pointer position vs attended centroid, in pixels)
- Perturbation selectivity (fraction of steps 1000-1099 where attended object = object 0)

### Decision Rule (PRE-COMMITTED, VERBATIM from manager/user)
g = (LEARNED_primary - RANDOM_primary) / (ORACLE_primary - RANDOM_primary)

(a) g >= 0.70 AND lower bootstrapped 95% CI >= 0.50: representation is SUFFICIENT
    for the behavior. ΔR²_color >= 0.30 was a distraction. Project ADVANCES to
    Phase 2/3 integration on the mean-pool representation.

(b) g <= 0.20: representation PROVABLY limits behavior. ONLY THEN is
    constraint-relaxation justified (iter_034+: decoder, higher d_t, or the
    deferred VICReg-upstream-of-gate fix for iter_032 cross-backbone collapse),
    with the concrete target set by the measured (ORACLE - RANDOM) gap.

(c) |ORACLE_primary - RANDOM_primary| < 0.10: the TASK or MOTOR PROTOCOL is the
    bottleneck, NOT perception. Fix the protocol/environment. Do NOT touch the
    representation. This branch invalidates the behavioral-pivot strategy for
    this specific protocol.

(d) 0.20 < g < 0.70: partial sufficiency. Advance to Phase 2/3 but flag perception
    as a known secondary limiter to revisit.

### Implementation Details

**File: src/run_iter033.py** (new script)

1. OraclePhysicsPredictor class: predicts next-step positions via linear
   extrapolation (pos + vel * dt). Surprised at collisions (velocity changes)
   but correct for constant-velocity motion. This is "known dynamics" — the
   predictor knows about constant-velocity physics but is genuinely surprised
   by collision events (which change velocities unpredictably).

2. OracleController class: wraps ground-truth perception into the CLTSMotorController
   interface. Constructs z_coord (1, d_max) tensor from info['positions'],
   z_dyn (1, d_max) from mean color values, z_pred_coord from physics predictor,
   z_pred_dyn = z_dyn. Calls CLTSMotorController.get_action() with these tensors.
   This ensures IDENTICAL motor code across all three conditions.

3. For the RANDOM condition: uses CLTSMotorController but overrides token_locus
   to random each step and zeroes all EMA statistics, so no learned surprise
   structure influences behavior. Equivalent to the "random" condition from
   iter_031 Part B.

4. For the LEARNED condition: loads iter_029 Arm B checkpoints
   (archive/iter_029/results/checkpoints/b_sfavicreg,_sfa_5.0_seed{N}.pt),
   sets d_t=2 for evaluation, runs standard encoder+predictor forward pass,
   feeds outputs to CLTSMotorController.

5. Collision detection: same as iter_031 Part B (COLLISION_DIST_THRESHOLD=4.0,
   COLLISION_VELOCITY_CHANGE_THRESHOLD=1.0, POST_COLLISION_WINDOW=15).

6. Channel-to-object mapping: use closest-centroid matching (same as iter_031).

7. Bootstrap CI for g: resample seeds 10000 times, compute g for each resample,
   report 95% percentile CI.

### Preserved Constraints
- Separate backbone + iter_029 config (SFA+VICReg, sfa_weight=5.0)
- d_t=2 frozen for N=2 (GDASR log-only, M3)
- Decoder-free (no reconstruction)
- No positional encoding
- M2 mandate stays "untestable" until constraint-relaxation phase
- No new representation-side ΔR² gate alongside the behavioral gate

---

## Iteration 033 -> Planner [Strategic Guidance]

## Manager's Note — iter_033 Plan Critique

**Point 1 (binding — substrate deviates from user hint, must be reconciled).**
The user hint *explicitly names the substrate*: "LEARNED — the best VIABLE representation, i.e. mean-pool E1 (separate backbone, **iter_028 config**, **d_t=3** frozen)". The plan instead loads **iter_029 Arm B** (SFA+VICReg sfa=5.0) at **d_t=2**. Both deviations need explicit handling before execution:

- *Substrate choice:* iter_029 Arm B is also mean-pool / 0% collapse and has higher ΔR² (0.275 vs 0.045), so a "best viable" interpretation includes it — but silently substituting it changes what a failure means. If g<0.20 on Arm B, it does *not* rule out that E1-VICReg-only would have done the same (or better, since SFA's slowness on z_dyn shrinks err_dyn and could systematically degrade the surprise readout — the very M2-interaction warning in the goal doc, §4.D). Either (a) include both as parallel LEARNED arms (preferred — and far more diagnostic), or (b) drop Arm B and use the user-named iter_028 substrate. Do **not** silently swap.
- *d_t = 2 vs 3:* the user hint says "d_t=3 frozen". Using d_t=2 because the env is N=2 is defensible, but it must be flagged in the pre-registration as a *deliberate* deviation, not a quiet matching-to-task. Better: keep d_t=3 (one unused channel) so the architecture under test is identical to the named config; this also tests whether the unused channel collapses or stays VICReg-clean.

**Point 2 (binding — ORACLE-bracket integrity).** Three issues that, if unaddressed, will produce an uninterpretable bracket:

- *Same-code-different-input is not the same as same-conditions.* `CLTSMotorController`'s EMA statistics, attention-cooldown, and push threshold (`μ+σ`) were calibrated implicitly against noisy learned surprise. ORACLE will produce qualitatively different surprise (clean zero between collisions, sharp spikes at collisions); the SAME `get_action()` will behave differently because the surprise distribution shifts shape. Pre-commit that the ORACLE represents *"ceiling under the existing motor code,"* not the absolute behavioral ceiling — and report the per-condition surprise distributions so the reader can see the bracket is not deformed by EMA mismatch.
- *Bracket ordering is not guaranteed.* Pre-commit a sanity check **before** computing g: require `ORACLE > LEARNED > RANDOM` on the primary metric (or at minimum `ORACLE > RANDOM`). If LEARNED beats ORACLE, or RANDOM beats ORACLE, the metric or the oracle construction is broken and g is meaningless — report this outcome plainly rather than computing a ratio.
- *Channel-to-object mapping is asymmetric.* "Closest-centroid" mapping is trivially correct for ORACLE (its centroid *is* the position) but biased for LEARNED (its centroid is a learned soft-argmax). Either use the same imperfect mapping for ORACLE too (e.g. forward through the encoder of a random untrained net, or accept the asymmetry and quantify its size).
- z_pred_dyn = z_dyn under ORACLE means err_dyn ≡ 0; surprise is purely position-driven. Confirm in the plan that LEARNED's surprise is summed the same way and report the position/identity decomposition.

**Point 3 (scientific discipline).** The pre-registration file must be written and committed **before any seed executes** — the Orchestrator will write `src/pre_registration.md` from the plan's `hypothesis` and `falsification_criterion` fields, so both must be sharpened now. Specifically:

- Report the **raw triple** `(RANDOM, LEARNED, ORACLE)` with CIs, not only `g`. A ratio hides denominator instability — with n=12 and a possibly small `(ORACLE − RANDOM)` gap, `g` can swing wildly while the underlying values are stable.
- Promote **branch (c)** to a first-class hypothesis in the falsification text, not a footnote — it would invalidate the entire behavioral-pivot strategy and that is exactly the kind of outcome the pre-registration must explicitly own *before* seeing data. Pre-commit the (c)-threshold (currently `|ORACLE − RANDOM| < 0.10`) and its construction.
- Restrained language in the report: "is consistent with sufficiency under this protocol" / "does not refute the representation's adequacy for the measured behavior" — never "the representation is sufficient." A passed gate at n=12 with one primary metric on one environment is evidence, not proof.

---

## Iteration 033 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
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

---

## Iteration 033 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 033 — Null Result: Behavioral Pivot Protocol Degenerate Under N=2 Post-Collision Selectivity

## 1. Pre-Declared Hypothesis and Falsification Criterion
Pre-registered hypothesis (verbatim from iter_033 pre-registration,
per the user hint binding rule): the iter_028 separate-backbone +
mask_dyn_sim + coord_vicreg + VICReg-only representation (ΔR²_color
= 0.045) drives surprise-based behavior on CLTSMotorController. The
test was constructed as a three-condition bracket — RANDOM (ablation
policy), LEARNED (iter_028 substrate), ORACLE (ground-truth
positions + ground-truth identity + physics-based surprise
predictor) — on identical environment (N=2 collision-sparse,
calibrated in iter_031 Part B), seed bank (≥10 seeds including
hard seeds 53/71), and motor logic.

Pre-committed decision rule on the normalized gap
g = (LEARNED − RANDOM) / (ORACLE − RANDOM) on the primary
behavioral metric (post-collision selectivity):
- (a) g ≥ 0.70 with lower CI ≥ 0.50 → representation sufficient,
  advance to Phase 2/3.
- (b) g ≤ 0.20 → representation provably limits behavior, license
  constraint-relaxation.
- (c) (ORACLE − RANDOM) itself small (degenerate bracket) → task or
  motor protocol is the bottleneck, fix the protocol; do NOT touch
  the representation.
- (d) intermediate g (0.20–0.70) → partial sufficiency.

## 2. Experimental Protocol
- Environment: 1D RGB physics sandbox, 128 pixels, N=2 objects
  under elastic collision dynamics (iter_031 calibration).
- Conditions: 3 (RANDOM, LEARNED, ORACLE), seeds ≥10 each,
  including hard seeds 53 and 71.
- Motor: CLTSMotorController, default config (Kp=2.0, Kd=0.5,
  Kv=0.5, push trigger at mu+sigma + |err|≤6.0, 15-step attention
  cooldown).
- LEARNED representation: iter_028 separate-backbone CNN
  (d_t=3 frozen, GDASR log-only, decoder-free, no positional
  encoding), mean-pool z_dyn readout.
- ORACLE: ground-truth `info['positions']` as z_coord and
  color/size as z_dyn, with surprise computed from a known-dynamics
  physics predictor. Implemented in three successive versions
  (v1: timing bug + linear extrapolation; v2: timing fix + linear
  extrapolation; v3: full physics simulator).
- Primary metric: post-collision attention selectivity (binary
  match on the max-velocity-change object within an N-step window).
- Secondary metrics: centroid tracking error; causal sensitivity to
  mass change (planned, not reached because bracket degenerated).

## 3. Observed Quantities
- ORACLE-v1 (buggy timing): surprise scale ~146 000; ORACLE < RANDOM
  (flagged as buggy implementation, not a real branch (b) signal).
- ORACLE-v2 (timing fixed, linear extrapolation): surprise scale
  ~164 000; ORACLE − RANDOM ≈ 0.007 on primary metric.
- ORACLE-v3 (full physics predictor, definitive): surprise scale
  ~310; ORACLE − RANDOM = 0.0001 on primary metric.
- LEARNED tracking error: 33 px; ORACLE tracking error: 58 px
  (paradoxically worse under perfect perception; attributed to
  surprise-EMA calibrating differently under the qualitatively
  different ORACLE surprise distribution).
- Random baseline ceiling under the metric's structure: ~0.50,
  matching the empirical ORACLE rate (both objects participate in
  every collision; "correct" choice is one of two).
- Falsification threshold for branch (c): |ORACLE − RANDOM| < 0.10.
  Observed gap 0.0001 ≪ 0.10. Branch (c) fires unambiguously.

## 4. Verdict
**Refuted (the protocol-discriminates-perception assumption was
refuted; representation sufficiency is not adjudicated by this
experiment).** The bracket degenerated: ORACLE and RANDOM are
empirically indistinguishable on the primary metric. Per the
pre-committed branch (c) rule, the conclusion is that the task /
motor-protocol combination does not discriminate perception
quality, and the representation is therefore not under test. The
representation's behavioral sufficiency (branch a) and behavioral
insufficiency (branch b) are both UNTESTED, not refuted, by this
experiment. No license for constraint relaxation is created.

## 5. Construction-vs-Empirical Note
Part of the result follows from construction: with N=2 objects and
both participating in every collision, the random match rate on a
"pick the right object" metric is exactly 1/2 by combinatorics,
and this is also the metric's saturation point. The structural
ceiling could have been computed analytically before the run —
this is a *methodological failure* of the iter_031/033 protocol
design that the iter_033 bracket caught empirically.

What is genuinely empirical: (1) that the LEARNED representation
produces non-collapsing surprise statistics at all on the
iter_028 substrate, confirming the substrate is at least
behaviorally viable; (2) that ORACLE under sharp clean surprise
produces *worse* tracking than LEARNED under noisy surprise, which
is a non-obvious mechanistic finding about
CLTSMotorController's surprise-EMA coupling and is independent of
the metric saturation issue; (3) that three ORACLE implementations
were needed to get the bracket right, validating the practice of
ORACLE sanity-checking before interpretation.

## 6. Limitations
- This result does NOT show that the iter_028 representation is
  insufficient for behavior. It shows only that the iter_033
  protocol cannot tell. The representation may be sufficient,
  insufficient, or partial; the bracket degeneracy makes all
  three indistinguishable.
- This result does NOT show that surprise-driven attention is a
  bad mechanism. It shows that the chosen metric does not
  discriminate good from bad attention policies in the N=2 regime
  with the current cooldown.
- This result does NOT generalize beyond the tested protocol.
  N≥3 environments, continuous metrics, or shorter-cooldown
  motors may all open the bracket; iter_034 will test these.
- The "ORACLE under perfect perception produces worse tracking"
  finding is an observation about CLTSMotorController, not about
  perception, and requires its own mechanistic follow-up before
  it can be promoted to a strategic constraint.
- The iter_031 0.59-vs-0.44 directional signal is now reinterpreted
  as in-distribution noise around a saturated metric, not as
  evidence of representation-driven behavior. Earlier journal
  entries treating that signal as a candidate finding should be
  read with this correction in mind.
- What would be needed next: an iter_034 calibration run producing
  a non-degenerate bracket (ORACLE − RANDOM ≥ 0.15) on at least
  one redesigned protocol axis (N≥3, continuous metric, or
  motor-cooldown sweep), with ORACLE sanity-checks
  (surprise scale within expected range, surprise spikes aligned
  to ground-truth events). Only then can iter_035 attach LEARNED
  and run the branch (a)/(b)/(c)/(d) rule meaningfully.

---

