## Task: Implement and Run iter_036 Foveated Gaze Benchmark

You must implement `src/run_iter036_benchmark.py` and run the full experiment according to the pre-registration at `src/pre_registration.md`. Read it first.

### Context

The Thalamus project has been trying to validate a behavioral benchmark where perception-driven targeting is load-bearing for mass estimation. iter_035 showed that in 1D with a physical pointer, even pass-through physics doesn't help because a passive pointer inevitably collides with bouncing objects (12.27 collisions/object). The solution: make the pointer **ghostly** (non-physical — passes through objects), and use an explicit **probe action** that applies a 1D elastic collision between the gaze pointer and a nearby object. Combined with **foveated gaze** (only a window of GAZE_RADIUS=8 pixels is "visible"), this should make perception load-bearing.

### Implementation Specification

Create `src/run_iter036_benchmark.py` implementing:

#### A. FoveatedGazeSandbox — subclass of PhysicsSandbox with:

1. **Ghostly pointer**: During the physics substep loop, the gaze pointer passes through objects (NO pointer-object collision resolution). The pointer still bounces off walls at 0 and 128, and still receives acceleration from action['acc']. This breaks the 1D geometric inevitability from iter_035.

2. **Probe action**: When `action['probe']=True` and `probe_budget > 0`:
   - Find the nearest object with center within GAZE_RADIUS=8 pixels of the gaze center.
   - If found: apply 1D elastic collision between gaze (M_gaze=10.0) and that object. Update both gaze velocity and object velocity using the standard elastic collision formula. Decrement probe_budget.
   - If not found: probe is wasted (no collision). Decrement probe_budget.
   - Record: (step, obj_idx, v_gaze_pre_step, v_obj_pre_step, v_gaze_post_step, v_obj_post_step)
   - **Critical Δv definition (Manager directive):** Pre-step velocities are recorded BEFORE the probe collision is applied AND before the substep loop runs. Post-step velocities are recorded AFTER the full physics substep loop completes. The Δv values therefore include confounding physics (wall bounces, obj-obj collisions) — this ensures the mass estimator is NOT a definitional identity.

3. **Foveated render**: Override render() so that only pixels within [gaze_pos - GAZE_RADIUS, gaze_pos + GAZE_RADIUS] are visible. Outside this window, canvas is zero.

4. **pass_through parameter**: If True, objects also pass through each other (same as PassThroughPhysicsSandbox from iter_035). This enables the Arm B factorial condition.

5. **Constructor**: `FoveatedGazeSandbox(N=3, substeps=10, seed=None, pass_through=False, gaze_radius=8)`

#### B. Three Conditions (NO learned representation):

1. **ORACLE**: Perfect knowledge of all object positions and velocities.
   - Maintains per-object probe-event count.
   - PD-tracks the least-probed object (Kp=2.0, Kd=0.5).
   - When the target's center is within GAZE_RADIUS of gaze center AND |error| ≤ 6.0, fires probe.
   - After probing, switches target to next least-probed object.
   - After probe budget exhausted, continues PD tracking without probing.
   - Probe budget B=20 over 2000 steps.

2. **RANDOM**: Random gaze acceleration ∈ [-10, 10]. Random probe with probability p = 0.01 per step (expected ~20 probes over 2000 steps) until budget exhausted. No targeting. Probe budget B=20.

3. **PASSIVE**: Gaze starts at 64.0, no acceleration, no probing. 0 probe events → POMLRE = 2.0 (ceiling).

#### C. Metric — POMLRE (Per-Object Median Log-Ratio Error):

For each object i:
1. Collect probe events for object i with |Δv_obj| > 1.0.
2. m_est_k = -M_gaze * Δv_gaze_k / Δv_obj_k for each event k.
3. If ≥3 valid events: m_hat_i = median(m_est_k), error_i = |log(m_hat_i / m_true_i)|.
4. If 1-2 valid events: m_hat_i = mean(m_est_k), error_i = |log(m_hat_i / m_true_i)|.
5. If 0 valid events: error_i = 2.0 (maximum penalty).
POMLRE = mean(error_i across 3 objects).

**Manager directive — branch distribution diagnostic:** Report the per-condition distribution over the three branches (0 events → 2.0, 1-2 events → mean, ≥3 events → median) as a diagnostic table. Also re-compute the headline POMLRE gap restricting to seed×object cells where BOTH ORACLE and RANDOM have ≥3 valid events (this is the F6 decomposition, but compute it as a PRIMARY output, not a footnote).

#### D. Factorial Design:

- **Arm A**: Foveated gaze + normal obj-obj collisions (foveation lever alone).
- **Arm B**: Foveated gaze + pass-through obj-obj collisions (both levers).
Primary metric computed separately per arm. Factorial comparison identifies the contribution of each lever.

#### E. Analytical Ceiling Gate (computed FIRST, before bracket):

Run RANDOM for 5 seeds × 2 arms = 10 short episodes (2000 steps each).
Gate seeds: [7, 31, 53, 71, 83]
For each arm, compute:
- Per-object probe-event counts under RANDOM.
- CV = std(counts) / mean(counts) across the 3 objects, averaged over seeds.
- Mean per-object event count.
Gate criteria (BOTH arms must pass):
- CV ≥ 0.5: RANDOM coverage is sufficiently uneven for ORACLE to improve.
- Mean per-object count ≥ 0.5: RANDOM gets at least some events.
**Manager directive:** The gate decision is made and logged BEFORE the 8-seed bracket launches. No peeking at bracket data.

If gate fails in either arm, report: "Foveated gaze with GAZE_RADIUS=8 does not create sufficient coverage imbalance under RANDOM policy in Arm [A/B]." Do NOT proceed to full bracket for that arm.

#### F. Full Bracket (if gate passes):

8 seeds × 2 arms × 3 conditions = 48 episodes, 2000 steps each.
Seeds: [7, 31, 53, 71, 83, 97, 113, 163] (hard seeds 53, 71 included).

#### G. Falsification Criteria (from pre-registration):

F1: RANDOM_POMLRE - ORACLE_POMLRE < 0.15 in either arm → falsified
F2: Lower bound of 95% paired bootstrap CI of gap includes zero → falsified
F3: ORACLE sanity checks fail (S1-S6 below) → buggy, no comparison interpreted
F4: Ordering PASSIVE > RANDOM > ORACLE violated → metric rejected
F5: CV gate fails (RANDOM coverage too even) → null
F6: Estimation-only gap < 0.05 in BOTH arms → "ORACLE wins by coverage, not perception-quality discrimination"

**ORACLE Sanity Checks (S1-S6):**
S1: ORACLE achieves ≥3 probe-induced collision events per object (mean across seeds)
S2: ORACLE probe success rate ≥ 60% (≥12 of 20 probe attempts result in a collision)
S3: ≥80% of ORACLE's probe-induced collision events have |Δv_obj| > 1.0
S4: No single object receives >80% of ORACLE's total probe events
S5: ORACLE gaze stays in bounds ≥95% of steps
S6: Each of 3 objects receives ≥10% of ORACLE's total probe events

#### H. Analysis and Reporting:

1. Per-seed POMLRE for each condition × arm.
2. Bootstrap CI (10000 paired resamples, paired by seed).
3. Sanity checks S1-S6.
4. Coverage-vs-estimation decomposition (F6) — **reported on equal footing with F1/F2, not as footnote**.
5. Branch distribution diagnostic (how many seed×object cells fall into each POMLRE branch per condition).
6. Factorial comparison: gap in Arm A vs gap in Arm B.

#### I. Output Files:

Save to `archive/iter_036/results/`:
- `per_run.csv`: condition, seed, arm, pomlre, per-obj valid counts, per-obj errors, true masses, probe stats, branch info
- `cv_gate.txt`: gate results per arm
- `sanity_checks.txt`: S1-S6 per arm
- `analysis.md`: full analysis with all tables, gates, conclusion
- `branch_distribution.txt`: per-condition branch distribution

#### J. Language Discipline (Manager directive):

- NO soft language: "trend toward", "approaching", "validates", "demonstrates", "proves"
- Correct framing: "is consistent with foveated gaze making perception load-bearing under the declared protocol" or "does not refute the null that foveated gaze is insufficient"
- A clean null with pre-committed meta-escalation is a first-class success of the method

#### K. Pre-Committed Escalation:

If null result (ORACLE-RANDOM gap < 0.15 in BOTH arms, or CV gate fails in both arms), that triggers meta-escalation: the project must confront whether the 1D sandbox itself is the structural confound. No additional foveated-gaze parameter adjustments before meta-escalation.

### Reference Files

- `src/environment.py` — PhysicsSandbox base class
- `src/run_iter035_benchmark.py` — PassThroughPhysicsSandbox and POMLRE implementation (reuse patterns)
- `src/pre_registration.md` — Full pre-registration (hypothesis, falsification criteria, method)

### Execution

1. Create `src/run_iter036_benchmark.py`
2. Run: `cd /home/user && python src/run_iter036_benchmark.py`
3. Verify output files are produced in `archive/iter_036/results/`
4. Read and report the key results

The script should be self-contained and runnable. Use `numpy` only (no torch needed for this benchmark). Set `np.random.seed()` appropriately for reproducibility.