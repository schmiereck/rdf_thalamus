
# Task: Implement and Run iter_035 Pass-Through Benchmark Experiment

## Context
This is a benchmark-validation-only iteration. The project has shown (iter_033-034) that in the standard 3-object sandbox, object-object collisions deliver mass information autonomously, so perception-driven targeting is NOT load-bearing (ORACLE-RANDOM gap = 0.031). This iteration redesigns the ENVIRONMENT so that dynamics information is gated by perception-driven targeting.

## CRITICAL: Pre-Registration Corrections (from Research Manager)

Before running ANY code, you MUST update `src/pre_registration.md` to incorporate these three corrections. Read the current file first, then rewrite it to include:

### Correction 1: Coverage-vs-Estimation Decomposition
Add to the pre-registration:
- A mandatory decomposition gate that reports the ORACLE-vs-RANDOM gap separately under:
  (i) **Coverage-only**: count valid events per object (no estimation), compare across conditions
  (ii) **Estimation-only**: restrict to seed×object cells where BOTH RANDOM and ORACLE achieved ≥3 valid events, and compare POMLRE errors there only
- Pre-register that the headline claim requires the estimation-only gap to be non-trivial (≥0.05). If the gap collapses to ~0 once coverage is matched, report honestly as "ORACLE wins by coverage, not by perception-quality discrimination" — which is still a valid benchmark-validation outcome, but must not be sold as the latter.

### Correction 2: F4 Ordering Sanity Check + Analytical Ceiling Gate
- Add F4 = ordering sanity check: mean PASSIVE_POMLRE > mean RANDOM_POMLRE > mean ORACLE_POMLRE. If violated, the metric is rejected as in iter_034-v1, regardless of F1/F2.
- Tighten CI language: "lower bound of the two-sided 95% CI on (RANDOM−ORACLE) > 0"
- Add an analytical-ceiling gate (step E): PASSIVE expected POMLRE must be ≥1.0 (i.e., PASSIVE has essentially no informative collisions on most objects). If the analytical ceiling shows PASSIVE would get enough collisions incidentally, the environment redesign has failed and the run does not proceed.

### Correction 3: Language + Sanity Checks + Escalation
- Restate the success condition as: "is consistent with the redesigned environment making perception-driven targeting load-bearing for mass-estimation under a finite budget" — NOT "perception sufficiency is established."
- Add justification for why iter_033 ORACLE sanity checks (surprise-scale and event-timing alignment) don't apply: The metric is computed directly from collision velocity deltas (no CLTSMotorController, no surprise EMA), so there is no surprise-scale calibration to verify and no event-timing alignment issue. The S1-S5 sanity checks in F3 cover the ORACLE implementation correctness.
- Ensure the pre-committed escalation to foveated gaze (goal.md Section 8.2) on a null result appears explicitly in the pre-registration file.

## Implementation: src/run_iter035_benchmark.py

### A. PassThroughPhysicsSandbox
Subclass `PhysicsSandbox` from `src/environment.py`. Override the `step()` method:
- In the substep loop, when checking for elastic collisions between adjacent entities, SKIP any collision pair where BOTH entities are non-pointer (i < N_OBJECTS and j < N_OBJECTS). Only resolve collisions where at least one entity is the pointer (index N_OBJECTS).
- Objects still bounce off walls normally.
- Objects pass through each other silently (no velocity exchange, no position correction between non-pointer entities).
- All other behavior (rendering, reset, etc.) is inherited from PhysicsSandbox.

IMPORTANT IMPLEMENTATION DETAIL for PassThroughPhysicsSandbox.step():
The parent class step() has a substep loop. You need to override the entire step() method (copy it from the parent) and modify the collision resolution section. In the section that iterates over sorted adjacent pairs, add a condition: if both i and j are less than N_OBJECTS (i.e., neither is the pointer), skip the collision resolution entirely (no overlap correction, no velocity exchange). Only apply collision physics when at least one of i, j equals N_OBJECTS (the pointer index).

### B. Three Conditions (NO learned representation)

**ORACLE:**
- Maintains per-object pointer-collision count and a push budget (max 15 pushes)
- PD-tracks the least-collided object (Kp=2.0, Kd=0.5)
- When within |error|≤6.0 AND push_cooldown==0 AND pushes_remaining>0: sets pointer_vel=5.0 toward target (counts as 1 push), increments push_cooldown=15, switches target to next least-collided object
- After 15 pushes exhausted, continues PD tracking without pushing
- Tracks push count and refuses pushes after budget exhausted

**RANDOM:**
- Random acceleration ∈ [-10,10] each step
- Random push (p=0.1) until 15-push budget exhausted, then no more pushes
- Uses a seeded RNG for reproducibility
- No targeting

**PASSIVE:**
- No action (acc=0, push=False). Pointer only moves from incidental collisions with objects.

### C. Collision Detection
Same approach as iter_034: Before/after each env.step(), save pre-velocities and pre-positions. After step, detect collisions by comparing velocities. BUT in the pass-through environment, the ONLY collisions that occur are pointer-object and object-wall. There are NO object-object collisions by design.

For collision detection, focus on pointer-object collisions only:
- Before each step: save pointer velocity and position, and each object's velocity and position
- After each step: compare pointer velocity change and each object's velocity change
- A pointer-object collision is detected when:
  - |Δv_pointer| > some threshold (e.g., 0.5) AND |Δv_obj| > some threshold (e.g., 0.5)
  - The pointer and object were close enough (distance < pointer_radius + obj_radius + threshold)

Log all pointer-object collision events with: step, obj_idx, pre/post velocities of both pointer and object.

### D. Metric — POMLRE (Per-Object Median Log-Ratio Error)
For each object i:
  1. Collect pointer-object collision events for object i
  2. Compute m_est_k = -POINTER_MASS * Δv_ptr_k / Δv_obj_k for each event k  (POINTER_MASS = 10.0)
  3. Filter: keep only events where |Δv_obj_k| > 1.0
  4. If ≥3 valid events: m_hat_i = median(m_est_k), error_i = |log(m_hat_i / m_true_i)|
  5. If 1-2 valid events: m_hat_i = mean(m_est_k), error_i = |log(m_hat_i / m_true_i)|
  6. If 0 valid events: error_i = 2.0 (maximum penalty)
POMLRE = mean(error_i across 3 objects)

### D2. Coverage-vs-Estimation Decomposition (CRITICAL — Manager Correction 1)
Also compute and report:
- **Coverage-only metric**: For each condition×seed, mean valid-event count per object. Compare across ORACLE/RANDOM/PASSIVE.
- **Estimation-only POMLRE**: Restrict to seed×object cells where BOTH RANDOM and ORACLE achieved ≥3 valid events. Compute per-object error on this restricted set only. Report the gap between ORACLE and RANDOM on this estimation-only subset.
- The headline claim requires the estimation-only gap (RANDOM_est - ORACLE_est) ≥ 0.05. If this gap is < 0.05, report honestly: "ORACLE wins by coverage, not by perception-quality discrimination."

### E. Analytical Ceiling Computation (with gate)
Before running, compute expected POMLRE for PASSIVE:
- In the pass-through environment, the PASSIVE pointer starts at x=64 and has no acceleration.
- Objects bounce between walls freely and pass through each other.
- Estimate: with pointer stationary at 64 (but pointer CAN be moved by collisions), and 3 objects of radius ~5.5 uniformly moving in [0,128], estimate the expected number of pointer-object collisions per object over 2000 steps.
- A simpler approach: actually simulate PASSIVE for a few seeds and count collisions, then extrapolate.
- Gate: If expected PASSIVE valid collisions per object ≥ 3, the environment redesign has failed (passive gets enough data without targeting). Do NOT proceed to full experiment. Report this as the finding.
- If expected PASSIVE valid collisions per object < 3, proceed to the full experiment.

### F. Run
8 seeds = [7, 31, 53, 71, 83, 97, 113, 163]
3 conditions × 8 seeds = 24 episodes, 2000 steps each.

### G. Gates and Analysis
Compute bootstrap CI (10000 samples, paired by seed) for (RANDOM_POMLRE - ORACLE_POMLRE).

**Falsification criteria (ALL must pass for hypothesis to hold):**
- F1: RANDOM_POMLRE - ORACLE_POMLRE ≥ 0.15
- F2: Lower bound of two-sided 95% CI on (RANDOM - ORACLE) > 0
- F3: All ORACLE sanity checks pass (S1-S5 from pre-reg)
- F4: Ordering sanity: mean PASSIVE > mean RANDOM > mean ORACLE. If violated, metric rejected.
- **NEW (Correction 1):** Estimation-only gap (on matched-coverage cells) ≥ 0.05. If estimation-only gap < 0.05 but full POMLRE gap ≥ 0.15, report: "ORACLE wins by coverage, not by perception-quality discrimination."

**Analytical ceiling gate (before running):** PASSIVE expected valid collisions per object < 3.

### Sanity Checks (F3)
S1: ORACLE achieves ≥3 informative pointer-object collisions per object (mean across seeds)
S2: ORACLE push budget utilization ≥ 80% (≥12 of 15 pushes used)
S3: ≥80% of collision events used for mass estimation have |Δv_obj| > 1.0
S4: No single object receives >80% of ORACLE's total pushes (even targeting)
S5: ORACLE pointer stays in bounds ≥95% of steps

### Output Files
Save to archive/iter_035/results/:
- per_run.csv: one row per seed×condition with all metrics
- summary.csv: aggregated stats
- analysis.md: full analysis with all gates, decomposition, and conclusion
- sanity_checks.txt: pass/fail for each sanity check
- analytical_ceiling.txt: the pre-run analytical estimate

### Pre-Committed Escalation
If the experiment produces a null result (ORACLE-RANDOM gap < 0.15 on pass-through environment), that is itself the finding: "perception is not behaviorally load-bearing under full observation even with pass-through dynamics." The project must pull the foveated-gaze mechanism (goal.md Section 8.2) forward from deferred. No additional environment tweaks before escalation.

## Files to Read
- src/environment.py — the PhysicsSandbox class to subclass
- src/run_iter034_v2.py — prior benchmark code for reference on collision detection and metric computation
- src/pre_registration.md — current pre-registration to update

## Files to Create/Modify
- src/pre_registration.md — UPDATE with Manager's three corrections
- src/run_iter035_benchmark.py — NEW, the main experiment
- archive/iter_035/results/ — output directory

## PRESERVED CONSTRAINTS
- iter_028 substrate (separate backbone), d_t=3 frozen, GDASR log-only (M3)
- decoder-free, no positional encoding, M2 not reopened
- No LEARNED representation used — benchmark validation only

## IMPORTANT NOTES
- Use `import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` to import from the src directory
- Set torch.set_num_threads(4) if using torch
- Use numpy for all computation (no torch needed for this benchmark)
- Seed the RNG properly for each condition×seed combination
- The PassThroughPhysicsSandbox MUST correctly skip object-object collision resolution in the substep loop while preserving wall bouncing and pointer-object elastic collisions
- Test the PassThroughPhysicsSandbox briefly before running the full experiment to ensure objects pass through each other
- Remember: masses in PhysicsSandbox are set to radii (mass proportional to size), so m_true_i = env.radii[i]
