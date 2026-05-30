ARM 1 — Integration Smoke-Test for Iteration 030.

FIRST: Read src/pre_registration.md to understand the full pre-registered hypothesis and falsification criteria. You MUST adhere strictly to these criteria.

## Overview
Load pre-trained checkpoints from iter_029 and run them through CLTSMotorController closed-loop evaluation. This tests whether the current representation quality (ΔR²_color ≈ 0.27 for SFA+VICReg, ≈ 0.04 for VICReg-only) is sufficient for downstream thalamic gating and motor behavior.

## Experiment Design

### Checkpoints
- SFA+VICReg checkpoints: `archive/iter_029/results/checkpoints/b_sfavicreg,_sfa_5.0_seed{SEED}.pt`
- VICReg-only checkpoints: `archive/iter_029/results/checkpoints/a_vicreg-only_control_seed{SEED}.pt`

### Seeds
10 fresh seeds: [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]
Hard seeds (reported separately): [53, 71]

### Four Conditions Per Seed
1. **CLTS-SFA**: Load SFA+VICReg checkpoint, run CLTSMotorController with surprise-driven attention
2. **CLTS-VICReg**: Load VICReg-only checkpoint, run CLTSMotorController with surprise-driven attention
3. **CLTS-Frozen**: Load SFA+VICReg checkpoint, but override token_locus to always be channel 0 (frozen attention — control for PD tracking)
4. **CLTS-Random**: Load SFA+VICReg checkpoint, but override token_locus to be uniformly random from [0, d_t) at each step (random attention — control for chance)

### Closed-Loop Evaluation Protocol (per seed per condition, 2000 steps)
1. Create PhysicsSandbox(N=3, seed=seed)
2. Initialize model from checkpoint, set d_t=3
3. Initialize CLTSMotorController(Kp=2.0, Kd=0.5, Kv=0.5)
4. Run 2000 steps with the controller providing actions
5. First 200 steps: warm-up with fixed attention at channel 0 (let EMA stats converge), no metrics collected
6. Steps 200-1000: collect tracking metrics, detect collisions
7. At step 1000: Apply mass perturbation — set object 0 mass to 3× current mass. Force pointer near object 0 (pointer_pos = obj_0_pos ± 5.0). Issue a push action.
8. Steps 1000-2000: continue evaluation, measure response to perturbation

### Model Forward Pass Adaptation
The NonParametricJEPASpatialSeparateDyn model in SFA mode needs x_hist (B, H, 3, 128) and x_target (B, 3, 128). You need to maintain a history deque of 4 frames. The forward pass returns: loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn).

To compute surprise for CLTS, use z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn. Take the active dims [:d_t]:
```python
err_coord_c = MSE(z_pred_coord[:, c] - z_target_coord[:, c]) for c in range(d_t)
err_dyn_c = MSE(z_pred_dyn[:, c] - z_target_dyn[:, c]) for c in range(d_t)
surprise_c = err_coord_c + err_dyn_c
```

For CLTS-Frozen: after computing surprises, override token_locus to 0.
For CLTS-Random: after computing surprises, set token_locus to random.randint(0, d_t-1).

### Collision Detection
At each step, check environment info["velocities"]. A collision between objects i and j is flagged if:
- |velocity_i(t) - velocity_i(t-1)| > 2.0 (velocity change threshold)
Record the previous velocities to detect changes.

When a collision is detected, record the objects involved. Check whether CLTS attention (token_locus) switches to a collision-involved channel within 15 steps. To map channels to objects, use the z_coord centroids: channel c's centroid should be near the position of the object it tracks. Sort channels by centroid value and sort objects by position to get a mapping.

### Mass Perturbation
At step 1000:
- env.masses[0] *= 3.0
- env.pointer_pos = env.positions[0] + (5.0 if env.positions[0] < 64 else -5.0)
- Issue push action
- Record whether CLTS attention switches to object 0's channel within 20 steps

### Metrics to Collect Per Seed Per Condition
1. **tracking_error_mean**: mean |pointer_pos - attended_centroid| over steps 200-2000
2. **collision_switch_rate**: fraction of collision events where attention was on a collision-involved channel within 15 steps
3. **collision_switch_count**: number of collision events detected
4. **perturbation_switch**: 1 if attention switched to object 0's channel within 20 steps after mass perturbation, 0 otherwise
5. **perturbation_switch_step**: step at which switch occurred (if any)
6. **mean_surprise**: mean surprise signal over evaluation

### Gate Evaluation (from pre-registration)
- **G1 PASS**: CLTS-SFA tracking_error_mean < 20 pixels AND CLTS-VICReg tracking_error_mean < 20 pixels
- **G2 PASS**: CLTS-SFA collision_switch_rate ≥ max(CLTS-Random collision_switch_rate, CLTS-Frozen collision_switch_rate) + 0.15 (15pp)
  - ALSO: CLTS-VICReg collision_switch_rate ≥ max(CLTS-Random, CLTS-Frozen) + 0.15 (for the M2 decision)
- **G3 PASS**: CLTS-SFA perturbation_switch_rate ≥ max(CLTS-Random perturbation_switch_rate, CLTS-Frozen perturbation_switch_rate) + 0.15

Note: For G2 and G3, compute the mean switch rate across seeds for each condition, then check if CLTS-SFA ≥ max(control) + 0.15. Also check CLTS-VICReg ≥ max(control) + 0.15 for the M2 decision hierarchy.

### Decision Rules (from pre-registration)
- ≥2 of 3 gates pass for CLTS-SFA → representation sufficient → project advances
- 0-1 gates pass → representation insufficient
- CLTS-SFA passes AND CLTS-VICReg passes → M2 demoted
- CLTS-SFA passes AND CLTS-VICReg fails → identity encoding matters, 0.30 search justified
- Both fail → representation truly insufficient

### Output Files
Write results to: `archive/iter_030/results/arm1_integration_smoke_test.csv`
Write per-seed per-condition results to: `archive/iter_030/results/arm1_per_seed.csv`
Write analysis to: `archive/iter_030/results/arm1_analysis.md`

### Implementation Notes
- Use src/motor.py CLTSMotorController — it already implements surprise-driven attention with EMA normalization
- Use src/models_separate_dyn.py NonParametricJEPASpatialSeparateDyn — create the model, set d_t=3, load checkpoint
- Use src/environment.py PhysicsSandbox
- The CLTSMotorController.get_action() takes model, obs, info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, d_t, centroids
- To get centroids from the model, use model.encoder(x_target) which returns (z_coord, z_dyn)
- For the closed loop: at each step, encode current obs to get z_coord and z_dyn, maintain a history of z_coord and z_dyn for the predictor, run predictor to get z_pred_coord and z_pred_dyn, compute surprise, and pass to CLTS controller
- IMPORTANT: Run on CPU. Use torch.set_num_threads(2) to avoid CPU thrashing. Process seeds sequentially.
- Report mean ± std for all metrics. Flag hard seeds 53, 71 separately.
- If checkpoint doesn't exist for a seed, skip and report.
