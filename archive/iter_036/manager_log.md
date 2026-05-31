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

