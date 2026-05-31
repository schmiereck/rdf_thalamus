
# Task: Implement and Run iter_035 Pass-Through Benchmark

## What to do
Create `src/run_iter035_benchmark.py` and run it. The pre-registration at `src/pre_registration.md` has already been updated. Read it first for the exact spec. Then implement and execute.

## Key Implementation Details

### 1. PassThroughPhysicsSandbox
Subclass `PhysicsSandbox` from `src/environment.py`. Override `step()` to skip object-object collision resolution. The easiest approach: copy the entire step() method from the parent class, and in the collision resolution loop, add a check: if both i and j are less than self.N (neither is the pointer), skip that pair entirely (no overlap resolution, no velocity exchange). Only apply collision physics when at least one entity is the pointer.

```python
class PassThroughPhysicsSandbox(PhysicsSandbox):
    def step(self, action=None):
        # Same as parent, but skip obj-obj collision resolution
        # Copy parent step() and modify the collision loop:
        # In the "Resolve elastic collisions between adjacent objects" section,
        # add: if i < self.N and j < self.N: continue
```

### 2. Collision Detection
Since objects pass through each other, only pointer-object collisions occur. Detect them by:
- Before each env.step(): save pre-velocities for pointer and all objects
- After each env.step(): check if pointer velocity changed significantly AND an object's velocity changed significantly
- For each object, check if |pointer_pos - obj_pos| < pointer_radius + obj_radius + threshold

Specifically, for each object i:
```python
d = abs(env.pointer_pos - env.positions[i])
if d < env.pointer_radius + env.radii[i] + THRESHOLD:
    dv_ptr = post_pointer_vel - pre_pointer_vel
    dv_obj = post_obj_vel[i] - pre_obj_vel[i]
    if abs(dv_ptr) > DV_THRESHOLD and abs(dv_obj) > DV_THRESHOLD:
        # This is a pointer-object collision with object i
        log_collision(step, i, pre/post velocities)
```

Actually, a simpler approach: just check for velocity changes on pointer and each object after each step. If both change significantly AND they're close, it's a collision. The threshold-based detection from iter_034 works fine.

### 3. Three Conditions

**ORACLE:**
```python
# State: obj_ptr_colls = [0,0,0], pushes_remaining=15, push_cooldown=0, prev_error=None, target=None
# Each step:
# 1. Select target = argmin(obj_ptr_colls)
# 2. PD control toward target
# 3. If |error| <= 6.0 and push_cooldown == 0 and pushes_remaining > 0:
#    pointer_vel = 5.0 * sign(error); push_cooldown = 15; pushes_remaining -= 1
#    Switch target to next least-collided
# 4. Decrement push_cooldown
```

**RANDOM:**
```python
# State: pushes_remaining=15, rng
# Each step:
# acc = rng.uniform(-10, 10)
# push = rng.rand() < 0.1 and pushes_remaining > 0
# if push: pushes_remaining -= 1
```

**PASSIVE:**
```python
# action = {'acc': 0.0, 'push': False}
```

### 4. POMLRE Metric
```python
POINTER_MASS = 10.0
for each object i:
    events = pointer-object collisions for object i where |Δv_obj| > 1.0
    if len(events) >= 3:
        m_estimates = [-POINTER_MASS * dv_ptr_k / dv_obj_k for each event k]
        m_hat = median(m_estimates)
        error_i = |log(m_hat / m_true_i)|
    elif len(events) >= 1:
        m_estimates = [-POINTER_MASS * dv_ptr_k / dv_obj_k for each event k]
        m_hat = mean(m_estimates)
        error_i = |log(m_hat / m_true_i)|
    else:
        error_i = 2.0
POMLRE = mean(error_i across 3 objects)
```

### 5. Coverage-vs-Estimation Decomposition
After computing POMLRE for all runs, compute:
- **Coverage-only**: For each condition×seed, mean valid-event count per object
- **Estimation-only POMLRE**: For seed×object cells where BOTH RANDOM and ORACLE have ≥3 valid events, compute per-object error using the same formula, and report the gap. If the estimation-only gap < 0.05, note this in the analysis.

### 6. Analytical Ceiling Gate
Before running the full experiment, simulate PASSIVE for 5 seeds with the PassThroughPhysicsSandbox and count pointer-object collisions per object. If the mean per-object collision count ≥ 3, report the environment redesign as failed and STOP.

### 7. Sanity Checks
S1: ORACLE achieves ≥3 informative pointer-object collisions per object (mean across seeds)
S2: ORACLE push budget utilization ≥ 80% (≥12 of 15 pushes used)
S3: ≥80% of collision events used for mass estimation have |Δv_obj| > 1.0
S4: No single object receives >80% of ORACLE's total pushes
S5: ORACLE pointer stays in bounds ≥95% of steps

### 8. Gates
F1: RANDOM_POMLRE - ORACLE_POMLRE ≥ 0.15
F2: Lower bound of two-sided 95% CI on (RANDOM - ORACLE) > 0
F3: All sanity checks pass
F4: mean PASSIVE > mean RANDOM > mean ORACLE
Estimation-only gap ≥ 0.05

### 9. Output
Save to `archive/iter_035/results/`:
- per_run.csv
- summary.csv
- analysis.md
- sanity_checks.txt
- analytical_ceiling.txt

### 10. Pre-Committed Escalation
If ORACLE-RANDOM gap < 0.15, report as null finding and note that foveated-gaze (Section 8.2) must be pulled forward.

## Constants
SEEDS = [7, 31, 53, 71, 83, 97, 113, 163]
N_OBJECTS = 3
N_STEPS = 2000
THRESHOLD = 4.0  # for collision proximity detection
DV_THRESHOLD = 0.5  # for velocity change detection
POINTER_MASS = 10.0
KP = 2.0
KD = 0.5
PUSH_DISTANCE = 6.0
PUSH_VEL = 5.0
PUSH_COOLDOWN_STEPS = 15
MAX_PUSHES = 15
N_BOOTSTRAP = 10000

## Files to read
- src/environment.py — parent class
- src/pre_registration.md — exact specification

## IMPORTANT
- The PassThroughPhysicsSandbox is the critical new component. Test it briefly (1-2 steps) to confirm objects pass through each other before running the full experiment.
- Print progress to console for each condition×seed combination.
- If the analytical ceiling gate FAILS (PASSIVE gets ≥3 collisions per object), stop and report. Do not proceed to the full experiment.
- Use numpy only, no torch needed.
