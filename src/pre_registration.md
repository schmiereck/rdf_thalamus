# RDF Scientific Pre-Registration

*   **Iteration:** 035
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
In a pass-through physics sandbox (N=3 objects that pass through each other; 
only pointer-object collisions remain elastic) with a 15-push budget over 2000 steps, 
the ORACLE targeted-exploration policy (PD-tracks the least-collided object, pushes 
when within 6px) achieves a Per-Object Median Log-Ratio Error (POMLRE) at least 0.15 
lower than a RANDOM policy, with the lower bound of the two-sided 95% paired bootstrap 
CI of (RANDOM_POMLRE - ORACLE_POMLRE) clear of zero over 8 seeds. The ordering 
ORACLE_POMLRE < RANDOM_POMLRE < PASSIVE_POMLRE holds in the mean.

The success condition is: the result is consistent with the redesigned environment 
making perception-driven targeting load-bearing for mass-estimation under a finite 
budget — NOT "perception sufficiency is established."

## 2. Falsification Criteria
The hypothesis is falsified if ANY of:

(F1) RANDOM_POMLRE - ORACLE_POMLRE < 0.15 (ORACLE does not substantially outperform RANDOM), OR
(F2) The lower bound of the two-sided 95% bootstrap CI of (RANDOM_POMLRE - ORACLE_POMLRE) includes zero (gap not statistically reliable), OR
(F3) Any ORACLE sanity check fails:
  S1: ORACLE achieves ≥3 informative pointer-object collisions per object (mean across seeds),
  S2: ORACLE push budget utilization ≥ 80% (≥12 of 15 pushes used),
  S3: ≥80% of collision events used for mass estimation have |Δv_obj| > 1.0,
  S4: No single object receives >80% of ORACLE's total pushes (even targeting),
  S5: ORACLE pointer stays in bounds ≥95% of steps.
  If F3 fires, the ORACLE implementation is buggy and no comparison is interpreted.
(F4) Ordering sanity check violated: mean PASSIVE_POMLRE > mean RANDOM_POMLRE > mean ORACLE_POMLRE. 
  If violated, the metric is rejected as in iter_034-v1, regardless of F1/F2.

**NEW — Coverage-vs-Estimation Decomposition (Correction 1):**
The headline claim additionally requires the estimation-only gap to be non-trivial (≥0.05).
If the full POMLRE gap ≥ 0.15 but the estimation-only gap (on matched-coverage cells) 
collapses to ~0, report honestly: "ORACLE wins by coverage, not by perception-quality 
discrimination" — which is still a valid benchmark-validation outcome, but must not be 
sold as the latter.

## 3. Coverage-vs-Estimation Decomposition (Mandatory)
For each condition×seed, compute and report:

**(i) Coverage-only:** Count valid events per object (no estimation). Compare mean 
valid-event counts across ORACLE, RANDOM, and PASSIVE. This isolates the coverage 
advantage of targeting.

**(ii) Estimation-only:** Restrict to seed×object cells where BOTH RANDOM and ORACLE 
achieved ≥3 valid events. Compute per-object POMLRE error on this restricted set only. 
Report the gap between ORACLE and RANDOM on this estimation-only subset. The headline 
claim requires this gap ≥ 0.05.

If the estimation-only gap < 0.05 but full POMLRE gap ≥ 0.15, the finding is: 
"ORACLE wins by coverage, not by perception-quality discrimination."

## 4. Analytical Ceiling Gate (Step E)
Before running the full experiment, compute expected POMLRE for PASSIVE:
- In the pass-through environment, the PASSIVE pointer starts at x=64 with no acceleration.
- Objects bounce between walls freely and pass through each other.
- Estimate the expected number of pointer-object collisions per object over 2000 steps 
  by simulating PASSIVE for a few seeds and counting collisions, then extrapolating.

**Gate:** If expected PASSIVE valid collisions per object ≥ 3, the environment redesign 
has failed (passive gets enough data without targeting). Do NOT proceed to full experiment. 
Report this as the finding.

If expected PASSIVE valid collisions per object < 3, proceed to the full experiment.

## 5. Why iter_033 ORACLE Sanity Checks Don't Apply Here
The iter_033 sanity checks for surprise-scale calibration and event-timing alignment 
verified properties of the CLTSMotorController (surprise EMA, event detection timing). 
In this iteration, the metric is computed directly from collision velocity deltas 
(m_est = -POINTER_MASS * Δv_ptr / Δv_obj). There is no CLTSMotorController in the loop, 
no surprise EMA, and no event-timing alignment issue. Therefore, the S1-S5 sanity checks 
in F3 cover all ORACLE implementation correctness requirements for this benchmark.

## 6. Proposed Method
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

C. Collision detection (pointer-object only): Before/after each env.step(), compare 
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
   by simulating PASSIVE for a few seeds and counting collisions.

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

## 7. Pre-Committed Escalation
If the experiment produces a null result (ORACLE-RANDOM gap < 0.15 on pass-through 
environment), that is itself the finding: "perception is not behaviorally load-bearing 
under full observation even with pass-through dynamics." The project must pull the 
foveated-gaze mechanism (goal.md Section 8.2) forward from deferred. No additional 
environment tweaks before escalation.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
*Updated with Research Manager corrections: Coverage-vs-Estimation Decomposition, 
F4 Ordering Sanity Check + Analytical Ceiling Gate, Language + Sanity Checks + Escalation.*
