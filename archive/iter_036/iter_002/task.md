## Task: Create src/run_iter036_benchmark.py (WRITE CODE ONLY, DO NOT RUN)

Create the file `src/run_iter036_benchmark.py` implementing the foveated gaze benchmark. DO NOT run it — just write the code.

### Key Reference Files (read first):
- `src/environment.py` — PhysicsSandbox base class
- `src/run_iter035_benchmark.py` — PassThroughPhysicsSandbox + POMLRE implementation patterns

### Specification

#### 1. FoveatedGazeSandbox (subclass of PhysicsSandbox)

Constructor: `__init__(self, N=3, substeps=10, seed=None, pass_through=False, gaze_radius=8)`

The key difference from PhysicsSandbox: the pointer is **ghostly** — it passes through objects (no pointer-object collision resolution). The pointer still bounces off walls at 0 and 128, and still receives acceleration from action['acc'].

Override `step(self, action=None)`:
- action can now include `action['probe']` (bool) in addition to `action['acc']`
- The step method needs a `probe_budget` attribute (initialized to 20 in reset())
- When `action['probe']=True` and `self.probe_budget > 0`:
  1. Find the nearest object with center within gaze_radius pixels of the gaze center (self.pointer_pos)
  2. If found: apply 1D elastic collision between gaze (mass=10.0) and that object. Update both gaze velocity and object velocity. Decrement probe_budget. Record the probe event.
  3. If not found: probe is wasted. Decrement probe_budget.
- In the physics substep loop: only resolve obj-obj collisions (skip pointer-object pairs). If `self.pass_through=True`, also skip obj-obj collisions.
- Record pre-step velocities BEFORE probe collision + substep loop, post-step velocities AFTER full substep loop.

Override `render(self)`:
- Only pixels within [gaze_pos - gaze_radius, gaze_pos + gaze_radius] are visible. Outside this window, canvas is zero. (For future LEARNED condition; does not affect ORACLE/RANDOM metrics.)

Add `probe_events` list attribute, reset each episode. Each event is a tuple: `(step, obj_idx, v_gaze_pre, v_obj_pre, v_gaze_post, v_obj_post)`

IMPORTANT: Look at how the existing step() method works in PhysicsSandbox. The ghostly pointer means: during the collision resolution loop (sorted by position), SKIP any collision pair where one entity is the pointer (index N, i.e. the last element in the concatenated arrays). This means the pointer never collides with objects — it just passes through them.

For the probe mechanism: the probe is applied OUTSIDE the substep loop, before or after it. The standard approach:
1. Record pre-step velocities
2. If probe=True and budget>0 and an object is within gaze_radius: apply elastic collision immediately (updating self.pointer_vel and self.velocities[obj_idx])
3. Then run the normal substep loop (which will NOT further collide the pointer with objects, since we skip pointer-object pairs)
4. Record post-step velocities after the loop

#### 2. Episode Runner: run_episode(condition, seed, arm)

arm is 'A' (normal obj-obj) or 'B' (pass-through obj-obj).

Create environment:
```python
env = FoveatedGazeSandbox(N=3, substeps=10, seed=seed, pass_through=(arm=='B'), gaze_radius=8)
```

Conditions:
- **ORACLE**: Perfect knowledge of positions/velocities from env info.
  - Maintains per-object probe-event count.
  - PD-tracks least-probed object (Kp=2.0, Kd=0.5).
  - When target center within GAZE_RADIUS=8 of gaze center AND |error| ≤ 6.0: fire probe.
  - After probing, switch target to next least-probed.
  - After probe budget exhausted, continue PD tracking without probing.
  - Track: probe_attempts, probe_hits, per-object probe counts, steps in bounds.

- **RANDOM**: Random gaze acceleration ∈ [-10, 10]. Random probe with p=0.01 per step until budget exhausted. No targeting.

- **PASSIVE**: Gaze starts at 64.0, no acceleration, no probing. 0 probe events → POMLRE = 2.0.

All conditions: 2000 steps, probe_budget=20.

Return dict with: condition, seed, arm, pomlre, per_obj_n_valid, per_obj_errors, true_masses, probe_attempts (ORACLE), probe_hits (ORACLE), per_obj_probe_counts (ORACLE), steps_in_bounds_frac (ORACLE), n_probe_events, branch per (seed, obj) cell.

#### 3. POMLRE Metric (reuse from iter_035 with branch tracking)

For each object i:
1. Collect probe events with |Δv_obj| > 1.0.
2. m_est_k = -M_gaze * Δv_gaze_k / Δv_obj_k
3. Branch: 0 events → error=2.0, 1-2 events → median→mean of estimates, ≥3 events → median of estimates
4. error_i = |log(m_hat_i / m_true_i)| (or 2.0 if no valid events)

Track which branch each (seed, obj_idx) falls into.

#### 4. Main Flow

```python
def main():
    # 1. CV Gate (5 seeds × 2 arms)
    GATE_SEEDS = [7, 31, 53, 71, 83]
    SEEDS = [7, 31, 53, 71, 83, 97, 113, 163]
    
    # Run RANDOM only for gate
    # Compute per-object probe-event counts, CV, mean
    # Log gate decision BEFORE bracket
    # If gate fails in an arm, skip that arm in bracket
    
    # 2. Full Bracket (8 seeds × remaining arms × 3 conditions)
    
    # 3. Analysis: F1-F6, S1-S6, branch distribution, coverage-vs-estimation
    
    # 4. Save output files
```

Bootstrap CI function: 10000 paired resamples.

#### 5. Output Files

Save to `archive/iter_036/results/`:
- `per_run.csv`
- `cv_gate.txt`
- `sanity_checks.txt`
- `analysis.md`
- `branch_distribution.txt`

#### CRITICAL IMPLEMENTATION DETAILS:

1. **Δv definition**: Pre-step = recorded BEFORE probe + substep loop. Post-step = recorded AFTER full substep loop. The Δv includes confounding physics.

2. **Ghostly pointer in substep loop**: When iterating sorted pairs for collision resolution, SKIP any pair (i,j) where i==N or j==N (pointer index). This is the core mechanism.

3. **Probe collision formula**: Standard 1D elastic collision:
   ```
   v1_new = (v1*(m1-m2) + 2*m2*v2) / (m1+m2)
   v2_new = (v2*(m2-m1) + 2*m1*v1) / (m1+m2)
   ```
   where m1=M_gaze=10.0, v1=gaze_vel, m2=obj_mass, v2=obj_vel.

4. **Seed handling**: Use `np.random.RandomState(seed * 1000 + hash(condition + arm) % 10000)` for policy randomness within episodes, so conditions are comparable.

5. **No torch needed** — numpy only.

Write the complete file. Do NOT run it.