# RDF Scientific Pre-Registration

*   **Iteration:** 034
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
A dynamics-learning benchmark with N=3 objects, where the primary metric is
mass-estimation MAPE (Mean Absolute Percentage Error) computed from ground-truth
collision logs with injected velocity observation noise (σ_vel = 0.5 px/step), is
non-degenerate: an ORACLE-TARGETED policy that actively probes under-observed
objects achieves mass-estimation MAPE at least 0.15 lower than a RANDOM policy
(RANDOM_MAPE - ORACLE_MAPE ≥ 0.15), with the lower 95% bootstrap CI of this gap
≥ 0.05 over ≥8 seeds including hard seeds 53 and 71. Additionally, PASSIVE_MAPE -
RANDOM_MAPE ≥ 0.05 with lower 95% bootstrap CI > 0, establishing the ordering
ORACLE_MAPE < RANDOM_MAPE < PASSIVE_MAPE. A passed gate establishes "the metric
responds to targeted vs untargeted coverage of the collision manifold, which is a
necessary precondition for measuring perception in iter_035" — NOT "measures the
Pillar-E end-goal capability" or "measures hidden-parameter inference."

## 2. Falsification Criterion
The benchmark is falsified if ANY of the following hold:
(F1) RANDOM_MAPE - ORACLE_MAPE < 0.15 (the ORACLE-TARGETED policy does not
     substantially outperform random action on mass estimation), OR
(F2) The lower 95% bootstrap CI of (RANDOM_MAPE - ORACLE_MAPE) includes zero
     (the gap is not statistically reliable), OR
(F2b) PASSIVE_MAPE - RANDOM_MAPE < 0.05 (natural inter-object collisions
      saturate mass inference and action contributes nothing), OR
(F2c) The lower 95% bootstrap CI of (PASSIVE_MAPE - RANDOM_MAPE) ≤ 0, OR
(F2d) The ordering ORACLE_MAPE < RANDOM_MAPE < PASSIVE_MAPE is violated
      (mean across seeds), OR
(F3) Any ORACLE sanity-check precondition fails:
     S1: ORACLE achieves ≥3 pointer-object collisions per object (mean across seeds),
     S2: ORACLE pointer-object collision count ≥ PASSIVE pointer-object collision count (per seed, paired),
     S3: ≥90% of logged collision events show |Δv| > 0.5 px/step,
     S4: ORACLE achieves ≥1 pointer-object collision per object per seed (pointer reaches all objects),
     S5: ORACLE pointer stays in bounds [0, 128] for ≥95% of steps (no runaway).
If F3 fires, the ORACLE implementation is buggy and no comparison is interpreted.

## 3. Proposed Method
Step 1: Create src/run_iter034_benchmark.py implementing the full experiment.

ENVIRONMENT: PhysicsSandbox(N=3), 2000 interaction steps, 8 seeds
[7, 31, 53, 71, 83, 97, 113, 163]. No pixel noise, no noisy TV, no structured distractor.
Substeps=10 (default).

THREE CONDITIONS (no LEARNED representation — that is iter_035):

(A) ORACLE-TARGETED: Custom policy that:
    - Maintains per-object collision count (from ground-truth collision detection)
    - Targets the object with FEWEST observed pointer-object collisions (information-gain maximization)
    - Moves pointer toward target using PD control (Kp=2.0, Kd=0.5)
    - Pushes when within |error| ≤ 6.0 of target (push gives pointer_vel = 5.0 toward target)
    - After pushing, switches target to next least-observed object
    - Does NOT use CLTSMotorController — implements targeting directly

(B) RANDOM: Uniform random acceleration ∈ [-10, 10], random push with p=0.1, no motor controller.

(C) PASSIVE: No pointer action (acc=0, push=False). Pointer starts at 64.0 with zero velocity
    and stays put (only moves if hit by objects). Only natural object-object and object-pointer
    collisions provide mass info.

COLLISION DETECTION:
- Before each env.step(), save all entity velocities (objects + pointer)
- After env.step(), compare velocities
- For each pair of adjacent entities (sorted by position), check:
  - Distance between them < radii_sum + threshold (threshold = 4.0)
  - Both have velocity changes |Δv| > 0.5 px/step
- Log: (step, entity_i, entity_j, v_i_pre, v_j_pre, v_i_post, v_j_post)
- Entity indices: 0..N-1 = objects, N = pointer

MASS ESTIMATION WITH VELOCITY NOISE:
- For each logged collision, add Gaussian noise to observed velocities:
  v_observed = v_true + N(0, σ_vel²) where σ_vel = 0.5
- From elastic collision physics (momentum + energy conservation):
  For a collision between entities i and j:
    m_i * (v_i_pre - v_i_post) + m_j * (v_j_pre - v_j_post) = 0  (momentum)
  This gives: m_i * Δv_i = -m_j * Δv_j  →  m_i/m_j = -Δv_j/Δv_i
- For pointer-object collisions (entity j = N, m_N = 10):
  m_i = -10 * Δv_N / Δv_i  (absolute mass)
- Build linear system A * m = b where each collision gives one row
  - For pointer-object: m_i = known value → equation: m_i = -10 * Δv_N / Δv_i
  - For object-object: m_i/m_j = ratio → equation: m_i - ratio * m_j = 0
- Solve via np.linalg.lstsq
- Objects with 0 observed collisions: m_hat = 5.5 (prior mean, as radii are uniform [3,8], mass = radius)
- MAPE = mean(|m_hat_i - m_true_i| / m_true_i) across 3 objects

PRIMARY METRIC: MAPE (lower is better).
- Gate G1 (non-degeneracy): RANDOM_MAPE - ORACLE_MAPE ≥ 0.15, lower 95% bootstrap CI ≥ 0.05
- Gate G2 (coverage gradient): PASSIVE_MAPE - RANDOM_MAPE ≥ 0.05, lower 95% bootstrap CI > 0
- Gate G3 (ordering): ORACLE_MAPE < RANDOM_MAPE < PASSIVE_MAPE (mean across seeds)

SECONDARY METRICS:
- Per-object pointer collision count (coverage)
- Per-object mass estimation error (breakdown by object)
- Held-out velocity prediction MSE (split collisions 80/20 by time, fit masses from train, predict post-collision velocities on test)

SANITY CHECKS (all must pass before interpreting G1/G2/G3):
- S1: ORACLE achieves ≥3 pointer-object collisions per object (mean across seeds)
- S2: ORACLE pointer-object collision count ≥ PASSIVE pointer-object collision count (per seed, paired)
- S3: ≥90% of logged collision events show |Δv| > 0.5 px/step
- S4: ORACLE achieves ≥1 pointer-object collision per object per seed (pointer reaches all objects)
- S5: ORACLE pointer stays in bounds [0, 128] for ≥95% of steps (no runaway)

ANALYTICAL CEILING:
- Compute expected PASSIVE MAPE analytically: with stationary pointer at 64, objects moving at 0.5-2.0 px/step in [0,128], estimate probability of each object hitting the pointer in 2000 steps
- If PASSIVE_MAPE is close to 0 (most objects get hit anyway), the benchmark has no useful range
- Report this BEFORE running the experiment; if useful range < 0.3, flag but still run (the analytical estimate may be wrong)

BOOTSTRAP CI:
- 10000 bootstrap samples for all CI computations
- Use paired bootstrap (resample seeds) for gap estimates

EMA CONFOUND NEUTRALIZATION:
- The MAPE metric is computed entirely from ground-truth collision logs + noisy velocity observations
- No surprise signal, no CLTSMotorController, no EMA calibration involved
- The iter_033 artifact (ORACLE 58 px tracking vs LEARNED 33 px) is structurally impossible here

## 4. iter_035 Attachment Sketch
In iter_035, the LEARNED representation will be attached via a custom information-gain controller (NOT CLTSMotorController) that uses per-channel raw surprise from the encoder to determine which object to target (replacing ORACLE's collision-count-based targeting). The metric remains MAPE from ground-truth collision logs. The bracket becomes: LEARNED-SURPRISE vs ORACLE-COUNT vs RANDOM vs PASSIVE. This avoids the iter_033 EMA confound because: (a) the metric is from ground-truth collision logs, not from the representation; (b) the controller uses raw/z-scored surprise with a fixed window, not EMA-calibrated surprise.

Step 2: Run the experiment (8 seeds × 3 conditions = 24 runs).

Step 3: Analyze results. Report per-seed MAPE for each condition, compute
gap and CI, check all sanity preconditions, apply gates.

FILES CREATED/MODIFIED:
- src/run_iter034_benchmark.py (new, main experiment)
- src/pre_registration.md (updated with corrections)
- archive/iter_034/results/ (output directory)

PRESERVED (per directive): M1/M3, iter_028 substrate, d_t=3 frozen,
GDASR log-only, decoder-free, no positional encoding, M2 not reopened.
No LEARNED representation is used in this iteration.

---
*Updated with Manager Corrections for iter_034 benchmark validation.*
