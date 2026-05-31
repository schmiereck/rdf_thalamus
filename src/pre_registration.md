# RDF Scientific Pre-Registration

*   **Iteration:** 035
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
In a pass-through physics sandbox (N=3 objects that pass through each other; 
only pointer-object collisions remain elastic) with a 15-push budget over 2000 steps, 
the ORACLE targeted-exploration policy (PD-tracks the least-collided object, pushes 
when within 6px) achieves a Per-Object Median Log-Ratio Error (POMLRE) at least 0.15 
lower than a RANDOM policy, with the lower 95% paired bootstrap CI of 
(RANDOM_POMLRE - ORACLE_POMLRE) clear of zero over 8 seeds. The ordering 
ORACLE_POMLRE < RANDOM_POMLRE < PASSIVE_POMLRE holds in the mean.

## 2. Falsification Criterion
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

## 3. Proposed Method
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
*Created automatically by the RDF Orchestrator prior to iteration execution.*
