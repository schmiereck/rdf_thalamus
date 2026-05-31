You are implementing and running the 2D cheap gate experiment for the Thalamus project. This is the core de-risking experiment for a potential 2D environment migration.

**CRITICAL: Read `src/pre_registration.md` FIRST.** All parameters, gate definitions, and decision rules are pre-registered there. You must strictly adhere to them. Do not modify the pre-registration.

## Task

Create `src/run_iter037_2d_gates.py` implementing the full 2D cheap gate experiment, then RUN IT and save results to `archive/iter_037/results/`.

### A. Implement PhysicsSandbox2D

A minimal 2D physics sandbox for gate tests. Key requirements:
- 2D positions (N×2 numpy array), 2D velocities (N×2), radii (N,), masses (N,), colors (N,3)
- Pointer: position (2,), velocity (2,), radius=4.0, mass=10.0, color=white
- 2D elastic circle-circle collision: standard physics — when two circles overlap (2D distance < r1+r2), compute the normal along the line connecting centers, decompose velocities into normal/tangential, exchange normal components weighted by mass. Formula:
  ```
  n = (pos2 - pos1) / |pos2 - pos1|  # unit normal
  v1n = dot(v1, n), v2n = dot(v2, n)
  v1n_new = (v1n*(m1-m2) + 2*m2*v2n) / (m1+m2)
  v2n_new = (v2n*(m2-m1) + 2*m1*v1n) / (m1+m2)
  v1_new = v1 + (v1n_new - v1n) * n
  v2_new = v2 + (v2n_new - v2n) * n
  ```
- Wall bounces at [0, 64]² boundaries (reflect position and velocity component)
- 10 substeps per step (dt = 1/10)
- Reset: randomized positions (non-overlapping — use rejection sampling or grid segments), randomized velocities per component ∈ [-2.0,-0.5]∪[0.5,2.0]
- Collision event recording for pointer-object collisions: (step, obj_idx, pre_ptr_vel, pre_obj_vel, post_ptr_vel, post_obj_vel) — 2D velocities stored as (vx, vy) tuples or 2-element arrays
- No rendering needed — this is physics-only, no pixel rendering

For the RESET method:
- Divide the 64×64 arena into N segments (e.g., 3 columns or a grid)
- Place each object in its segment with some margin
- This prevents initial overlaps
- Randomize velocities per component: sample from [-2.0, -0.5] ∪ [0.5, 2.0] (ensure non-zero speed)
- Pointer starts at (32, 32) with zero velocity for Gate-1

For COLLISION DETECTION in the substep loop:
- Check ALL pairs (object-object AND pointer-object) for overlap
- Resolve overlapping pairs: separate them along the center-connecting line proportional to inverse mass, then exchange velocity components along the normal
- A "valid collision event" for recording purposes: the pointer and an object overlap AND |Δvx| > 0.5 OR |Δvy| > 0.5 (pre vs post velocity change). Record ALL pointer-object collision events (with and without the velocity filter) — the velocity filter is only for the gate metric, not for physics.

### B. Gate-1: PASSIVE Boundedness

- 5 seeds × 2000 steps
- Seeds: [7, 31, 53, 71, 83]
- Pointer starts at (32, 32) with zero velocity, zero acceleration (no action)
- Objects move normally (with obj-obj collisions and wall bounces)
- Count per-object valid collisions (2D distance < r_ptr + r_obj + 4.0 AND |Δvx| > 0.5 OR |Δvy| > 0.5)
- Also count raw proximity collisions (without the Δv filter) for diagnostics
- Report per-seed: (mean per-object valid count, per-object counts, raw counts)
- Gate threshold: mean per-object valid count ≤ 3.0
- Per-seed decision: ≥4/5 seeds must individually have mean per-object count ≤ 3.0

### C. Gate-1b: PASSIVE Collision Heterogeneity

- Uses the SAME collision data from Gate-1 (no separate run needed)
- For each seed, compute CV = std(per-object valid collision counts) / mean(per-object valid collision counts)
- Threshold: CV ≥ 0.30
- Per-seed decision: ≥4/5 seeds must individually have CV ≥ 0.30
- Also compute CV for raw (unfiltered) collision counts for comparison

### D. Gate-2: RANDOM Gaze Heterogeneity

- 5 seeds × 2000 steps
- Seeds: [7, 31, 53, 71, 83]
- Ghostly gaze pointer: during substep loop, the gaze pointer moves but does NOT physically collide with objects (skip all pointer-object collision resolution in the substep loop). The gaze pointer bounces off walls. Objects collide with each other normally.
- Each step: 2D random acceleration, each component ∈ [-10, 10], applied to gaze velocity
- Probe mechanism: probability p=0.01 per step, gaze_radius=8 (2D Euclidean distance between gaze center and object center), probe_budget=20
- On probe: find the nearest object with center within gaze_radius of gaze center; if found, apply a 2D elastic collision between gaze (mass=10) and that object (along center-connecting line), record the event; decrement probe_budget regardless of hit/miss
- Count per-object probe events across the 2000 steps
- Compute CV = std(per-object probe counts) / mean(per-object probe counts)
- Threshold: CV ≥ 0.50
- Per-seed decision: ≥4/5 seeds must individually have CV ≥ 0.50
- Also report: mean per-object probe count, total probes fired

### E. Sanity Checks

- S1: Physics conservation test: create a controlled 2D collision (two objects moving directly at each other along x-axis), verify total momentum and kinetic energy are conserved before/after collision. Print values.
- S2: Pointer stays within [0, 64]² throughout all Gate-1 runs (check every step)
- S3: Objects stay within [0, 64]² throughout all Gate-1 runs (check every step)
- S4: Gate-1 PASSIVE pointer actually has approximately zero velocity throughout (it gets hit by objects, so it WILL acquire velocity from collisions — this check should verify the pointer receives no ACTIVE acceleration, not that it stays at zero. Clarify in the check: verify pointer acc = 0 and initial vel = 0; velocity changes come only from collisions.)
- S5: Gate-2 RANDOM gaze fires approximately 20 probes over 2000 steps (expected 0.01 × 2000 = 20, check within [15, 25])

### F. Output Files

Save to `archive/iter_037/results/`:

1. `gate1_results.csv` — columns: seed, obj_0_valid, obj_1_valid, obj_2_valid, mean_valid, cv_valid, obj_0_raw, obj_1_raw, obj_2_raw, mean_raw, cv_raw, gate1_pass, gate1b_pass
2. `gate2_results.csv` — columns: seed, obj_0_probes, obj_1_probes, obj_2_probes, mean_probes, cv, total_probes_fired, gate2_pass
3. `sanity_checks.txt` — detailed sanity check output
4. `gate_summary.txt` — overall gate pass/fail summary with per-seed details

### Important Implementation Notes

- This is a NEW file `src/run_iter037_2d_gates.py` — do NOT modify any existing source files
- The 2D physics must be correct — use standard 2D elastic collision formulas
- The collision event recording must capture BOTH the velocity-filtered (valid) counts AND the raw proximity counts
- For Gate-2, the gaze pointer is ghostly during substeps but can physically collide during probe actions
- Print progress to stdout for debugging
- Use numpy for all array operations
- Run the script to completion and verify output files are created
- Pre-registered parameters are FROZEN — do not change them

Run the script after creating it and report all results.