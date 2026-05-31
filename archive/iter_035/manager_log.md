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

