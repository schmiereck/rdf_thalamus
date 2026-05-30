# RDF Scientific Pre-Registration

*   **Iteration:** 033
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
The best available mean-pool representation (SFA+VICReg sfa_weight=5.0, separate backbone,
ΔR²≈0.275, 0% collapse, iter_029 Arm B) supports functional surprise-driven behavior
that accounts for at least 20% of the gap between random (lower bound) and oracle
(perfect-perception upper bound) baselines, as measured by post-collision attention
selectivity on N=2 collision-sparse environments. Specifically, the normalized gap
g = (LEARNED - RANDOM) / (ORACLE - RANDOM) on post-collision attention selectivity
(version B: attended object matches max-velocity-change object) will be > 0.20,
with the four-branch decision rule determining project direction.

## 2. Falsification Criterion
F1 FAIL: g <= 0.20 — the learned representation accounts for less than 20% of the
oracle-random gap, meaning the representation PROVABLY limits behavior and
constraint-relaxation (decoder, higher d_t, or the VICReg-upstream-of-gate fix
for the iter_032 cross-backbone collapse) is justified.

Additionally, if branch (c) fires — (ORACLE - RANDOM) is itself small (|difference| < 0.10
on primary metric) — the task or motor protocol is the bottleneck, NOT perception,
and the behavioral-pivot strategy itself is invalidated for this protocol/environment.

The lower 95% CI of g (bootstrapped across seeds) must be reported; if it excludes
the boundary between adjacent branches, the branch assignment is confident; if not,
both branches are reported as possible.

## 3. Proposed Method
## Three-Condition Oracle Bracket Experiment

### Conditions (all sharing identical environment, seed bank, and CLTSMotorController logic)

1. **RANDOM (lower bound):** CLTSMotorController with random token_locus selection
   each step (no encoder, no prediction). The controller's attention_cooldown is
   set to 0 so locus is re-randomized every step, and all surprise EMA values are
   zeroed to prevent any learned structure from influencing behavior.

2. **LEARNED (test condition):** Load iter_029 Arm B checkpoints (SFA+VICReg,
   sfa_weight=5.0, separate backbone, mean-pool readout, d_max=8, d_t=3 trained).
   Evaluate with d_t=2 (first 2 channels active for N=2 objects). Run encoder +
   predictor forward pass, feed z_pred_coord, z_target_coord, z_pred_dyn,
   z_target_dyn to the standard CLTSMotorController.get_action().

3. **ORACLE (upper bound):** Feed ground-truth positions (info['positions'][:d_t])
   as z_coord, ground-truth mean-color (np.mean(info['colors'][:d_t], axis=1))
   as z_dyn, linear-extrapolation predicted positions (prev_pos + prev_vel * dt)
   as z_pred_coord, and z_dyn as z_pred_dyn (identity is constant). Feed these
   tensors to the SAME CLTSMotorController.get_action() method — identical
   surprise computation (per-channel MSE + EMA normalization + attention cooldown),
   identical PD tracking, identical push logic.

### Environment
PhysicsSandbox(N=2) — collision-sparse by design. Mass perturbation at step 1000:
multiply object 0's mass by 1.5× (same as iter_031 Part B). 2000 evaluation steps
per seed.

### Seed Bank
12 seeds: [7, 17, 31, 53, 71, 83, 97, 101, 107, 113, 137, 163]
Includes hard seeds 53 and 71 (mandated by user/manager).

### Primary Behavioral Metric
Post-collision attention selectivity (version B): for each collision event,
within POST_COLLISION_WINDOW=15 steps after the collision, the fraction of steps
where the attended object (token_locus mapped to object index via closest-centroid)
matches the max-velocity-change object. This is the same metric as iter_031 Part B,
directly calibrating the 0.59-vs-0.44 signal.

### Secondary Metrics (reported but do NOT drive the gate)
- Mean tracking error (pointer position vs attended centroid, in pixels)
- Perturbation selectivity (fraction of steps 1000-1099 where attended object = object 0)

### Decision Rule (PRE-COMMITTED, VERBATIM from manager/user)
g = (LEARNED_primary - RANDOM_primary) / (ORACLE_primary - RANDOM_primary)

(a) g >= 0.70 AND lower bootstrapped 95% CI >= 0.50: representation is SUFFICIENT
    for the behavior. ΔR²_color >= 0.30 was a distraction. Project ADVANCES to
    Phase 2/3 integration on the mean-pool representation.

(b) g <= 0.20: representation PROVABLY limits behavior. ONLY THEN is
    constraint-relaxation justified (iter_034+: decoder, higher d_t, or the
    deferred VICReg-upstream-of-gate fix for iter_032 cross-backbone collapse),
    with the concrete target set by the measured (ORACLE - RANDOM) gap.

(c) |ORACLE_primary - RANDOM_primary| < 0.10: the TASK or MOTOR PROTOCOL is the
    bottleneck, NOT perception. Fix the protocol/environment. Do NOT touch the
    representation. This branch invalidates the behavioral-pivot strategy for
    this specific protocol.

(d) 0.20 < g < 0.70: partial sufficiency. Advance to Phase 2/3 but flag perception
    as a known secondary limiter to revisit.

### Implementation Details

**File: src/run_iter033.py** (new script)

1. OraclePhysicsPredictor class: predicts next-step positions via linear
   extrapolation (pos + vel * dt). Surprised at collisions (velocity changes)
   but correct for constant-velocity motion. This is "known dynamics" — the
   predictor knows about constant-velocity physics but is genuinely surprised
   by collision events (which change velocities unpredictably).

2. OracleController class: wraps ground-truth perception into the CLTSMotorController
   interface. Constructs z_coord (1, d_max) tensor from info['positions'],
   z_dyn (1, d_max) from mean color values, z_pred_coord from physics predictor,
   z_pred_dyn = z_dyn. Calls CLTSMotorController.get_action() with these tensors.
   This ensures IDENTICAL motor code across all three conditions.

3. For the RANDOM condition: uses CLTSMotorController but overrides token_locus
   to random each step and zeroes all EMA statistics, so no learned surprise
   structure influences behavior. Equivalent to the "random" condition from
   iter_031 Part B.

4. For the LEARNED condition: loads iter_029 Arm B checkpoints
   (archive/iter_029/results/checkpoints/b_sfavicreg,_sfa_5.0_seed{N}.pt),
   sets d_t=2 for evaluation, runs standard encoder+predictor forward pass,
   feeds outputs to CLTSMotorController.

5. Collision detection: same as iter_031 Part B (COLLISION_DIST_THRESHOLD=4.0,
   COLLISION_VELOCITY_CHANGE_THRESHOLD=1.0, POST_COLLISION_WINDOW=15).

6. Channel-to-object mapping: use closest-centroid matching (same as iter_031).

7. Bootstrap CI for g: resample seeds 10000 times, compute g for each resample,
   report 95% percentile CI.

### Preserved Constraints
- Separate backbone + iter_029 config (SFA+VICReg, sfa_weight=5.0)
- d_t=2 frozen for N=2 (GDASR log-only, M3)
- Decoder-free (no reconstruction)
- No positional encoding
- M2 mandate stays "untestable" until constraint-relaxation phase
- No new representation-side ΔR² gate alongside the behavioral gate

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
