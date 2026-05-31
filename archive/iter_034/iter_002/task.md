## Task: Create and Run the iter_034 Dynamics-Learning Benchmark

Read `src/pre_registration.md` first — this is the frozen pre-registration. Then create and run `src/run_iter034_benchmark.py`.

### Key Implementation Requirements

The pre-registration is already written. You must implement exactly what it specifies. Here are the critical details:

**ENVIRONMENT**: `PhysicsSandbox(N=3)`, 2000 steps, 8 seeds [7, 31, 53, 71, 83, 97, 113, 163]. Import from `src/environment.py`.

**THREE CONDITIONS (no LEARNED representation):**

**(A) ORACLE-TARGETED**: 
- Before starting the loop, initialize `obj_pointer_collisions = [0, 0, 0]` (per-object count of pointer collisions)
- At each step, find the object with the fewest pointer collisions: `target_idx = argmin(obj_pointer_collisions)`
- Compute PD control toward that object's position: `error = positions[target_idx] - pointer_pos`, `acc = Kp * error + Kd * d(error)/dt` with Kp=2.0, Kd=0.5
- Push when `abs(error) <= 6.0` and push cooldown is 0: set `pointer_vel = 5.0 * sign(error)`, increment cooldown to 15 steps
- After a push, switch target to next least-observed object (the next argmin after current target)
- When a pointer-object collision is detected, increment `obj_pointer_collisions[obj_idx]`

**(B) RANDOM**: `acc = uniform(-10, 10)`, `push = rand() < 0.1`

**(C) PASSIVE**: `acc = 0`, `push = False`. Pointer starts at 64.0, zero velocity. Objects can still collide with it.

**COLLISION DETECTION (critical — implement carefully):**
- Before `env.step()`, save: `pre_velocities = np.concatenate([env.velocities.copy(), [env.pointer_vel]])` and `pre_positions = np.concatenate([env.positions.copy(), [env.pointer_pos]])`
- After `env.step()`, save: `post_velocities = np.concatenate([info["velocities"].copy(), [info["pointer_vel"]]])` and `post_positions = np.concatenate([info["positions"].copy(), [info["pointer_pos"]]])`
- For each pair of ADJACENT entities (sorted by position after step), check:
  - Distance between them < sum of their radii + threshold (4.0)
  - Both have |Δv| > 0.5 (where Δv = v_post - v_pre for each entity)
- A collision event is: `(step, entity_i, entity_j, v_i_pre, v_j_pre, v_i_post, v_j_post)`
- IMPORTANT: entity N (=3 for N=3 objects) is the pointer. Its mass is 10.

**MASS ESTIMATION WITH VELOCITY NOISE (σ_vel = 0.5):**
- For each collision event, add noise: `v_obs_pre_i = v_true_pre_i + np.random.normal(0, 0.5)`, same for all 4 velocities
- Use a FIXED noise seed per (condition, seed) pair for reproducibility: `rng = np.random.RandomState(seed * 100 + hash(condition) % 10000)`
- From momentum conservation: `m_i * (v_i_pre - v_i_post) = m_j * (v_j_post - v_j_pre)`
  → `m_i * Δv_i = -m_j * Δv_j`
- For pointer-object collisions (one entity is pointer, index N=3, mass=10):
  If entity i is the object and entity j is the pointer: `m_i = 10 * (-Δv_pointer) / Δv_object`
  If entity i is the pointer and entity j is the object: `m_j = 10 * (-Δv_pointer) / Δv_object` (same formula, just identify which is the object)
- For object-object collisions: `m_i / m_j = -Δv_j / Δv_i`
- Build the linear system:
  - For pointer-object collisions involving object k: add row [0...1...0] * m = estimated_m_k (direct estimate)
  - For object-object collisions between objects i,j: add row [0...1...-ratio...0] * m = 0 (where ratio = -Δv_j/Δv_i)
- Solve via `np.linalg.lstsq(A, b)` with rcond=None
- Objects with 0 observed collisions (of any type): m_hat = 5.5 (prior mean)
- MAPE = mean(|m_hat_i - m_true_i| / m_true_i) for i in 0..2

**VELOCITY PREDICTION (secondary metric):**
- Split collisions by time: first 80% = train, last 20% = test
- From train collisions, estimate masses (same procedure)
- For test collisions: predict post-collision velocities using elastic collision formulas with estimated masses
- Compute MSE between predicted and actual post-collision velocities

**GATES:**
- G1: RANDOM_MAPE - ORACLE_MAPE ≥ 0.15, lower 95% bootstrap CI ≥ 0.05
- G2: PASSIVE_MAPE - RANDOM_MAPE ≥ 0.05, lower 95% bootstrap CI > 0
- G3: ORACLE_MAPE < RANDOM_MAPE < PASSIVE_MAPE (means)

**SANITY CHECKS:**
- S1: ORACLE achieves ≥3 pointer-object collisions per object (mean across seeds)
- S2: ORACLE pointer-object collision count ≥ PASSIVE (per seed, paired)
- S3: ≥90% of logged collision events show |Δv| > 0.5
- S4: ORACLE achieves ≥1 pointer-object collision per object per seed
- S5: ORACLE pointer stays in bounds [0, 128] for ≥95% of steps

**ANALYTICAL CEILING:**
Before running the experiment, compute analytically: with a stationary pointer at position 64 and radius 4, and 3 objects with random radii ∈ [3,8], random velocities ∈ [0.5, 2.0], random starting positions uniformly in [0,128] (with segment constraint), what is the expected probability that each object collides with the pointer in 2000 steps? The pointer is at 64, radius 4. An object with radius r at position x and velocity v (bouncing between walls) will hit the pointer when it passes through [64-4-r, 64+4+r]. For a random walk between walls, the expected crossing time can be estimated. But a SIMPLER approach: just simulate 1000 passive runs first and report the empirical PASSIVE MAPE distribution, then check if useful range > 0.3.

Actually, the simplest valid approach: just run the experiment and report whether PASSIVE_MAPE is high enough for the benchmark to have useful range. The analytical computation is optional.

**OUTPUT:**
Write results to `archive/iter_034/results/`:
- `per_run.csv`
- `summary.csv`
- `analysis.md`
- `sanity_checks.txt`

**IMPORTANT: The script should be efficient.** Each run is 2000 steps with N=3 objects. No torch models are used (no LEARNED representation). This is pure numpy. Should complete all 24 runs in under 5 minutes.

**Set `torch.set_num_threads(4)` even though we don't use torch heavily.**

After running the experiment, print a clear summary of:
1. All sanity check pass/fail
2. Per-condition MAPE (mean, std, per-seed)
3. Gate results (G1, G2, G3)
4. Whether the benchmark is validated or falsified