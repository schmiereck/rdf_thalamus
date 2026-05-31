## Task: Implement and Run the iter_034 Dynamics-Learning Benchmark Validation

You MUST first read `src/pre_registration.md` and adhere to its pre-registered hypothesis and falsification criteria. Then apply the corrections below (which refine, not contradict, the pre-registration).

### Context
This is a BENCHMARK VALIDATION iteration only. No LEARNED representation is used. We are validating that a mass-estimation MAPE metric on N=3 objects is non-degenerate (ORACLE-TARGETED beats RANDOM beats PASSIVE). This will be used in iter_035 to test LEARNED representations.

### Manager Corrections to Pre-Registration (MUST incorporate)

**CRITICAL Correction 1 — Inject velocity observation noise:**
With noiseless `env.info` velocities and elastic collision physics, a single pointer-object collision gives EXACT mass (pointer mass = 10 is known). MAPE becomes ~0 for any touched object, collapsing to a pure coverage metric. **Inject Gaussian noise σ_vel = 0.5 px/step on ALL observed velocities** (both pre and post) before mass estimation. This makes inference non-trivial and gives the benchmark genuine estimator variance.

**CRITICAL Correction 2 — Add PASSIVE gate:**
The pre-registration only has ORACLE vs RANDOM. Add: PASSIVE_MAPE - RANDOM_MAPE ≥ 0.05, with lower 95% bootstrap CI > 0. Without this, we cannot detect the failure mode where natural inter-object collisions saturate mass inference and action contributes nothing. Also require the ordering: ORACLE_MAPE < RANDOM_MAPE < PASSIVE_MAPE.

**CRITICAL Correction 3 — Tighten language:**
A passed gate establishes "the metric responds to targeted vs untargeted coverage of the collision manifold, which is a necessary precondition for measuring perception in iter_035" — NOT "measures the Pillar-E end-goal capability" or "measures hidden-parameter inference."

**CRITICAL Correction 4 — Tighten S4 sanity check:**
The pre-registration says S4: ORACLE mean surprise per step ∈ [0.01, 100]. But in this benchmark, the ORACLE-TARGETED policy does NOT use a surprise-based controller — it targets the least-observed object by collision count. There is no surprise signal driving behavior. Replace S4 with: "ORACLE achieves ≥1 pointer-object collision per object per seed (pointer actually reaches all objects)." Replace S5 with: "ORACLE pointer stays in bounds [0, 128] for ≥95% of steps (no runaway)."

**CRITICAL Correction 5 — Add iter_035 attachment sketch:**
In iter_035, the LEARNED representation will be attached via a custom information-gain controller (NOT CLTSMotorController) that uses per-channel raw surprise from the encoder to determine which object to target (replacing ORACLE's collision-count-based targeting). The metric remains MAPE from ground-truth collision logs. The bracket becomes: LEARNED-SURPRISE vs ORACLE-COUNT vs RANDOM vs PASSIVE. This avoids the iter_033 EMA confound because: (a) the metric is from ground-truth collision logs, not from the representation; (b) the controller uses raw/z-scored surprise with a fixed window, not EMA-calibrated surprise.

**CRITICAL Correction 6 — Compute analytical random-baseline ceiling:**
Before running, compute and report: with N=3 objects in a 128-pixel space, pointer starting at 64, what fraction of the MAPE range is below the PASSIVE ceiling? If useful range (1 - PASSIVE_MAPE/maximum_MAPE) < 0.3, reject the metric before running.

### Implementation Details

**Step 1: Update `src/pre_registration.md`** with all corrections above.

**Step 2: Create `src/run_iter034_benchmark.py`**

ENVIRONMENT: `PhysicsSandbox(N=3)`, 2000 interaction steps, 8 seeds [7, 31, 53, 71, 83, 97, 113, 163]. No pixel noise, no noisy TV, no structured distractor. Substeps=10 (default).

THREE CONDITIONS (no LEARNED representation):

**(A) ORACLE-TARGETED**: Custom policy that:
- Maintains per-object collision count (from ground-truth collision detection)
- Targets the object with FEWEST observed pointer-object collisions (information-gain maximization)
- Moves pointer toward target using PD control (Kp=2.0, Kd=0.5)
- Pushes when within |error| ≤ 6.0 of target (push gives pointer_vel = 5.0 toward target)
- After pushing, switches target to next least-observed object
- Does NOT use CLTSMotorController — implements targeting directly

**(B) RANDOM**: Uniform random acceleration ∈ [-10, 10], random push with p=0.1, no motor controller.

**(C) PASSIVE**: No pointer action (acc=0, push=False). Pointer starts at 64.0 with zero velocity and stays put (only moves if hit by objects). Only natural object-object and object-pointer collisions provide mass info.

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

**Step 3: Run the experiment** (8 seeds × 3 conditions = 24 runs)
Each run: 2000 steps with N=3 objects. Should take ~30 seconds per run. Total ~12 minutes.

**Step 4: Write results to `archive/iter_034/results/`**
- `per_run.csv`: per-seed, per-condition results
- `summary.csv`: aggregated results
- `analysis.md`: full analysis with gate checks, sanity checks, ceiling computation
- `sanity_checks.txt`: pass/fail for each sanity check

**Important implementation notes:**
- Use `src/environment.py` as-is (PhysicsSandbox)
- The `simulate_physics` function from `src/run_iter033_v3.py` is NOT needed (no ORACLE predictor)
- The ORACLE-TARGETED policy targets by collision COUNT, not surprise
- Set torch.set_num_threads(4) to avoid CPU oversubscription
- Use numpy random state seeded per-run for reproducibility
- The velocity noise for mass estimation uses a FIXED random seed per (condition, seed) pair so results are reproducible
- When pointer is at position 64 with zero velocity and PASSIVE mode, objects can still collide with it (the pointer is a physical entity in the environment)
- In PASSIVE mode, the pointer may drift slightly from object collisions (the physics engine moves it), but no active acceleration is applied

PRESERVE: M1/M3, iter_028 substrate, d_t=3 frozen, GDASR log-only, decoder-free, no positional encoding, M2 not reopened. No LEARNED representation is used.