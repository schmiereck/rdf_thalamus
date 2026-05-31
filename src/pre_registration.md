# RDF Scientific Pre-Registration

*   **Iteration:** 034
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
A dynamics-learning benchmark with N=3 objects, where the primary metric is
mass-estimation MAPE (Mean Absolute Percentage Error) computed from ground-truth
collision logs, is non-degenerate: an ORACLE-TARGETED policy that actively probes
under-observed objects achieves mass-estimation MAPE at least 0.15 lower than a
RANDOM policy (RANDOM_MAPE - ORACLE_MAPE ≥ 0.15), with the lower 95% bootstrap
CI of this gap ≥ 0.05 over ≥8 seeds including hard seeds 53 and 71. The metric
directly measures the Pillar-E end-goal capability (hidden-parameter inference
through active interaction) without routing through the EMA-coupled motor
controller, thereby neutralizing the 33-vs-58 px tracking artifact from iter_033.

## 2. Falsification Criterion
The benchmark is falsified if EITHER:
(F1) RANDOM_MAPE - ORACLE_MAPE < 0.15 (the ORACLE-TARGETED policy does not
     substantially outperform random action on mass estimation), OR
(F2) The lower 95% bootstrap CI of (RANDOM_MAPE - ORACLE_MAPE) includes zero
     (the gap is not statistically reliable), OR
(F3) Any ORACLE sanity-check precondition fails:
     S1: ORACLE achieves ≥3 collisions per object (targeting works),
     S2: ORACLE total collision count ≥ PASSIVE total collision count,
     S3: ≥90% of logged collision events show |Δv| > 0.5 px/step,
     S4: ORACLE mean surprise per step ∈ [0.01, 100] (physical range,
         not 146k as in the buggy iter_033 v1/v2),
     S5: ≥80% of ORACLE surprise spikes align (±2 steps) with ground-truth
         collision events.
If F3 fires, the ORACLE implementation is buggy and no comparison is interpreted.

## 3. Proposed Method
Step 1: Create src/run_iter034_benchmark.py implementing the full experiment.

ENVIRONMENT: PhysicsSandbox(N=3), 2000 interaction steps, 8 seeds
[7, 31, 53, 71, 83, 97, 113, 163].

THREE CONDITIONS (no LEARNED representation — that is iter_035):

(A) ORACLE-TARGETED: Custom policy that (i) maintains per-object collision
    count, (ii) moves pointer toward the object with fewest observed collisions
    using PD control (Kp=2.0, Kd=0.5), (iii) pushes when within |error| ≤ 6.0
    of target, (iv) uses the full-physics simulate_physics() from iter_033 v3
    for surprise computation. Does NOT use CLTSMotorController — avoids the
    EMA confound entirely by implementing a clean information-gain-maximizing
    policy directly.

(B) RANDOM: Uniform random acceleration ∈ [-10, 10], random push with p=0.1,
    no motor controller.

(C) PASSIVE: No pointer action (acc=0, push=False). Pointer acts as a
    passive object. Only natural object-object collisions provide mass info.

MASS ESTIMATION PROCEDURE:
- During interaction, log all collision events: (step, obj_i, obj_j, v_i_pre,
  v_j_pre, v_i_post, v_j_post). Collision detection: |pos_i - pos_j| <
  radii_i + radii_j + threshold AND |Δv| > 0.5 for either object.
- From elastic collision physics, each collision (i,j) gives:
  m_i * (v_i - v_i') = m_j * (v_j' - v_j)
  → linear constraint on mass vector [m_0, m_1, m_2].
- Pointer-object collisions give absolute mass (pointer mass = 10).
- Solve overdetermined system via least-squares (np.linalg.lstsq).
- Objects with 0 observed collisions: m_hat = 5.5 (prior mean).
- MAPE = mean(|m_hat_i - m_true_i| / m_true_i) across 3 objects.

PRIMARY METRIC: MAPE (lower is better).
Gate 1 (non-degeneracy): RANDOM_MAPE - ORACLE_MAPE ≥ 0.15, lower 95% CI ≥ 0.05.
Gate 2 (end-goal validity): The metric measures hidden-parameter inference
through active interaction, which IS the Pillar-E artificial-curiosity
capability, not a proxy for it. Justified because mass is only observable
through collisions, and active probing systematically increases collision
coverage.

SECONDARY METRIC: Held-out velocity prediction MSE.
Split collision data 80/20 by time. Fit masses from training collisions,
predict post-collision velocities on test collisions, compute MSE.

ORACLE SANITY CHECKS (pre-conditions, must ALL pass before interpreting F1/F2):
S1: ORACLE achieves ≥3 collisions per object (mean across seeds).
S2: ORACLE total collision count ≥ PASSIVE total collision count (per seed).
S3: ≥90% of logged collision events show |Δv| > 0.5 px/step.
S4: ORACLE mean surprise per step ∈ [0.01, 100].
S5: ≥80% of ORACLE surprise spikes align (±2 steps) with collisions.

EMA CONFOUND NEUTRALIZATION (stated explicitly per Manager hint):
The mass-estimation metric is computed entirely from ground-truth collision
logs (positions, velocities from env.info), NOT from the agent's surprise
signal or motor controller state. The ORACLE-TARGETED policy uses a custom
information-gain-maximizing controller (not CLTSMotorController), so the
different surprise distributions between conditions cannot affect the metric.
The iter_033 tracking artifact (ORACLE 58 px vs LEARNED 33 px) arose because
the CLTSMotorController's EMA calibrated differently under qualitatively
different surprise distributions, causing different attention switching rates.
By not routing the metric through the EMA-coupled motor, this artifact is
structurally impossible in the new benchmark.

Step 2: Run the experiment (8 seeds × 3 conditions = 24 runs).

Step 3: Analyze results. Report per-seed MAPE for each condition, compute
gap and CI, check all sanity preconditions, apply gates.

FILES CREATED/MODIFIED:
- src/run_iter034_benchmark.py (new, main experiment)
- src/pre_registration.md (auto-generated from plan)

PRESERVED (per directive): separate backbone architecture, d_t=3 frozen,
GDASR log-only (M3), decoder-free, no positional encoding, M2 not reopened.
No LEARNED representation is used in this iteration.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
