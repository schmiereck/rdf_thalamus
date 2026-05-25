## Phase 18: EG-MDL (Entropy-Gated MDL) Implementation and Experimentation

You must implement and run the full EG-MDL experiment, then analyze the results against the pre-registered falsification criteria. Read `src/pre_registration.md` FIRST to understand the exact hypothesis, arms, and falsification criteria.

### Context

The Thalamus project has been iteratively solving the problem of dimension recruitment gating. Phase 16 established WUP-MDL (Probationary Warm-Up Period + MDL consistency gate) which achieves 100% recruitment rate on N=3→4 transition but 100% false recruitment on Noisy-TV distractors. Phase 17 falsified ESUG (encoder-only gating without warm-up) because cold-started encoder projections produce rough temporal dynamics.

The Phase 18 hypothesis: Adding a prediction-trend gate to WUP-MDL will maintain ≥80% recruitment while reducing Noisy-TV false recruitment from 100% to ≤20%.

### Arms to Implement

1. **Arm P (WUP-MDL, W=100)** — Baseline from Phase 17. Re-implement identically for matched comparison.
2. **Arm S (EG-MDL, W=100, θ=0.90)** — During the WUP period, record per-step prediction error e[t] for the proposed 4th dimension. At end of WUP, compute ρ = E_late / E_early where E_early = mean(e[0:W/2]) and E_late = mean(e[W/2:W]). Gate accepts if MDL ratio < 1.0 AND ρ < 0.90.
3. **Arm S_alt (EG-MDL, W=100, θ=0.85)** — Same as Arm S but θ=0.85. Robustness arm to test θ-sensitivity.

### Implementation Details

**Reference code:** `src/run_phase17_experiments.py` provides the complete experimental pipeline. Create `src/run_phase18_experiments.py` based on this, with the following modifications:

#### Prediction-Trend Gate Computation

During the WUP probationary period, at each training step, compute the **per-dimension prediction error for the 4th dimension** (index 3):

From the forward pass outputs `(z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn)`:
```python
# Per-dimension prediction error for dimension 3
e_coord_dim3 = F.mse_loss(z_pred_coord[:, 3], z_target_coord[:, 3])
e_dyn_dim3 = F.mse_loss(z_pred_dyn[:, 3], z_target_dyn[:, 3])
e_total_dim3 = (e_coord_dim3 + e_dyn_dim3).item()
```

Store these in a list `wup_errors` throughout the WUP period.

At the end of WUP (step = probation_end_step):
```python
W = len(wup_errors)
E_early = np.mean(wup_errors[:W//2])
E_late = np.mean(wup_errors[W//2:])
rho = E_late / max(E_early, 1e-8)
```

**Composite gate for Arm S:**
- MDL criterion: ratio < 1.0 (same as Arm P)
- Prediction-trend: ρ < 0.90
- BOTH must pass for acceptance

**Composite gate for Arm S_alt:**
- Same but ρ < 0.85

#### Arm-specific Branch Logic

Modify `run_active_branch()` to support a new `gating="eg_mdl"` option alongside the existing "mdl" and "esug":

For Arms S and S_alt:
1. Same WUP period as Arm P (d_t=4, train both encoder and predictor for W steps)
2. During WUP, record `wup_errors` (per-step dim-3 prediction error)
3. At end of WUP, compute MDL ratio AND prediction-trend ratio ρ
4. Apply composite gate

The arm definitions should be:
```python
arms = [
    ("Arm P (WUP-MDL, W=100)",    100,  None, "mdl",    None),
    ("Arm S (EG-MDL, θ=0.90)",    100,  None, "eg_mdl", 0.90),
    ("Arm S_alt (EG-MDL, θ=0.85)",100,  None, "eg_mdl", 0.85),
]
```

### Experimental Protocol

**Seeds:** [42, 123, 456, 789, 999] (same as Phase 17 for matched comparison)

**Sweep 1 — Transition Sweep:** N=3→4 clean objects. 
- Passive training on N=3 for 1500 steps (shared across arms per seed)
- Active CLTS training for steps 1501-3000 with 4th object introduced at step 1500
- WUP proposal triggered at step 1800 (same as Phase 17)
- Evaluate at steps [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]

**Sweep 2 — Control Sweep:** N=3 clean + 1 Noisy-TV distractor.
- Same passive pre-training on N=3
- Active CLTS with Noisy-TV replacing the 4th object
- Same WUP proposal at step 1800
- Measure false recruitment rate

### Metrics to Collect

Per arm per seed:
- `recruitment_accepted` (bool): whether the dimension was permanently recruited
- `mse_cent` (float): centroid decoding MSE for the 4th object/distractor
- `test_sim_loss` (float): test simulation loss on N=4 (or N=3 for control)
- `mdl_ratio` (float): MDL consistency ratio at gate evaluation
- `rho` (float): prediction-trend ratio at gate evaluation
- `wup_errors` (list): per-step prediction errors during WUP
- `attention_switch_rate` (float): post-recruitment attention stability
- `centroid_tracking_error` (float): post-recruitment centroid tracking accuracy

### Analysis Required

After all sweeps complete:

1. **Recruitment rates** per arm (transition sweep): count of seeds where accepted
2. **False recruitment rates** per arm (control sweep): count of seeds where accepted
3. **Mean ± std** of mse_cent per arm (transition sweep, only recruited seeds)
4. **Welch's t-tests**: Arm P vs Arm S on mse_cent, attention_switch_rate
5. **Falsification audit** against the pre-registered criteria:
   - C1: Recruitment rate < 80% on transition sweep
   - C2: False recruitment rate > 20% on control sweep
   - C3: Mean centroid decoding MSE > 65.0 on transition sweep
   - C4 (NEW): θ-sensitivity: if Arm S passes but Arm S_alt fails (or vice versa), report as θ-sensitive finding
6. **Save results** to `archive/iter_018/results/` as CSV, JSON, and markdown report

### Important Implementation Notes

- The model class is `NonParametricJEPASpatial` from `src/models_dual_stream.py`
- The environment class is `PhysicsSandbox` from `src/environment.py` 
- The motor controller is `CLTSMotorController` from `src/motor.py`
- Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- The passive training function `train_passive_cached()` from Phase 17 can be reused as-is
- The evaluation function `evaluate_branch()` can be reused
- The `categorizer_consistency_ratio()` function can be reused for MDL gate computation

### Output Requirements

1. Create `src/run_phase18_experiments.py` with the full experiment code
2. Run the experiment and save all results to `archive/iter_018/results/`
3. Produce a comprehensive markdown report at `archive/iter_018/results/phase18_report.md`
4. The report must include: hypothesis, method, per-seed results table, mean±std summary, falsification audit, and interpretation

### CRITICAL: Pre-Registration Compliance

Before running ANY code, read `src/pre_registration.md` and ensure your implementation exactly matches the pre-registered method. Do NOT deviate from the specified arms, thresholds, or evaluation protocol. The Manager will audit your results against this file.