
# Phase 0 Recovery: Fix SFA Implementation and Re-run with Adjusted Parameters

## Context
The initial Phase 0 experiment (run_phase0_sfa.py) produced results showing:
- 2/5 SFA seeds collapsed (per-dim std < 0.5)  
- JEPA baseline also unhealthy (4/5 seeds with per-dim std < 0.5)
- Slowness ratio extremely high (805.2) in SFA arm — z_dyn collapsed to constant, making ratio meaningless
- Semantic probes showed negative R² (worse than mean prediction)

Root cause analysis identified TWO issues:
1. **SFA implementation bug**: z_prev_dyn is NOT detached in the SFA loss, causing gradient to flow through BOTH z_target and z_prev, effectively doubling the SFA gradient signal. Fix: add `.detach()` on z_prev_dyn_active.
2. **Training regime too simplified**: No CCR, only 3000 steps, no recruitment — this is insufficient even for the JEPA baseline. The prior successful baselines used 5000 steps + CCR covariance mode.

## Task: Fix, Adjust, and Re-run

### Step 1: Fix the SFA implementation in src/models_dual_stream.py

In the SFA branch of `NonParametricJEPASpatial.forward()`, find this line:
```python
sfa_loss = F.mse_loss(z_target_dyn_active, z_prev_dyn_active)
```

Change it to:
```python
sfa_loss = F.mse_loss(z_target_dyn_active, z_prev_dyn_active.detach())
```

This ensures the SFA gradient only flows through z_target (pushing it toward z_prev), not also through z_prev (which would double the effective SFA gradient strength).

### Step 2: Create src/run_phase0_sfa_recovery.py

A new experiment runner with these adjustments:

**Recovery arms (4 arms × 5 seeds):**
- **Arm A1 (SFA w=0.1)**: sfa_weight=0.1, 5000 steps, CCR covariance mode (smooth=10, spatial=10), pos_encoding="none", d_t=3, gdasr_log_only=True
- **Arm A2 (SFA w=1.0 fixed)**: sfa_weight=1.0, 5000 steps, CCR covariance mode, pos_encoding="none", d_t=3, gdasr_log_only=True — this tests whether the detach fix alone resolves collapse
- **Arm B (JEPA+CCR baseline)**: primary_objective="jepa", 5000 steps, CCR covariance mode, pos_encoding="none", d_t=3, gdasr_log_only=True — same extended regime for fair comparison
- **Arm C (SFA w=0.1+pos)**: sfa_weight=0.1, 5000 steps, CCR covariance mode, pos_encoding="sinusoidal", d_t=3

**Training protocol:**
- Seeds: [42, 123, 456, 789, 999]
- Environment: PhysicsSandbox(N=3, seed=seed) for 5000 steps (increased from 3000)
- Passive observation only (no motor)
- History: deque(maxlen=4)
- Replay buffer: capacity 2000, prefill 100
- Optimizer: Adam, lr=1e-3
- d_t=3 frozen, GDASR log-only
- CCR mode: 'covariance', ccr_smooth_weight=10.0, ccr_spatial_weight=10.0
- sim_weight=25.0, var_weight=25.0, cov_weight=25.0 (same as before)

**Evaluation protocol (same as original, at step 5000 and checkpoint at 2500):**
1. Non-collapse check: has_collapsed + per-dim std >= 0.5
2. Centroid decoding MSE (200 test frames, 3 objects)
3. Slowness metrics (sanity check)
4. Semantic disentanglement probes (linear probes for color R² from z_dyn vs z_coord)
5. VICReg health (per-dim std, mean abs correlation)
6. GDASR growth-point log
7. Prediction error

**Key comparison:** 
- C1: SFA arm A1 collapse rate vs original A collapse rate (2/5)
- C2: MSE_A1 vs MSE_B (need MSE_A1 <= 1.10 * MSE_B)
- C3 semantic: delta_R2_color for A1 >= 0.10

### Step 3: Run the experiment

Execute `python src/run_phase0_sfa_recovery.py` and save all results to `archive/iter_020/results/`.

### Step 4: Generate comparison report

After all runs complete, output a clear comparison between the original Phase 0 results and the recovery results, including:
- Per-arm mean ± std for: centroid_mse_mean, slowness_ratio, delta_R2_color, per_dim_std
- Falsification audit for each SFA arm against C1, C2, C3
- Whether the detach fix and/or sfa_weight reduction resolved the collapse

### Important:
- Use CPU (torch device "cpu") unless CUDA is available
- Set torch.set_num_threads(2) to prevent CPU thrashing
- The evaluation functions (compute_centroid_mse, compute_semantic_probes, etc.) from run_phase0_sfa.py can be reused
- Document any parameter adjustments and their rationale in the output
