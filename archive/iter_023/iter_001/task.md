Create the SFA weight sweep experiment runner and update pre-registration for iteration 023.

## Context

This is a Thalamus project experiment. The key finding from iter_022 is that sfa_weight=0.1 provides 250× less gradient than var_weight=25.0, making SFA functionally disabled. This iteration sweeps sfa_weight across [0.1, 1.0, 5.0, 10.0, 25.0] to find where SFA becomes effective.

## Task 1: Update src/pre_registration.md

Update the pre-registration with the Manager's directives:

1. **Elevate delta_R2_color ≥ 0.10 improvement over sfa_weight=0.1 baseline to PRIMARY falsification.** The reasoning: C5 (normalized_dyn_var < normalized_coord_var) is constructional — once sfa_weight is large enough, z_dyn MUST become slower by design. Passing C5 only confirms gradient propagation, not identity encoding.

2. **Reframe C5 as "gradient-propagation verification" (necessary but not sufficient).** 

3. **Add the Composite M2 Viability Criterion explicitly:** There exists an sfa_weight ∈ [0.1, 25.0] such that ≥ 3/5 seeds simultaneously satisfy: (a) C5 (SFA gradient reaches z_dyn), (b) delta_R2_color improvement ≥ 0.10 over A1 baseline, and (c) per-dim std > 0.5 (non-collapse). If no such sfa_weight exists, SFA + batch VICReg cannot jointly shape z_dyn into a slow identity representation in this architecture.

4. **Fix hypothesis language:** Replace "will begin to separate" with "we test whether SFA at parity produces measurable identity–position separation."

5. **Note Arm B pre-commitment:** Arm B is pre-set to sfa_weight=10.0. If optimal_sfa is between 5.0 and 10.0, Arm B may fail for the wrong reason. Do not interpret Arm B failure alone as falsifying the compound hypothesis.

## Task 2: Create src/run_phase0_sfa_sweep.py

Base this on `src/run_phase0_sfa_archceiling.py` (read it for the full structure). Key changes:

### Arm Configurations (7 arms × 5 seeds × 5000 steps):

```
Arm A1 (Ctrl sfa=0.1): d_max=8, CGIR, CCR, d_t=3, sfa_weight=0.1
Arm A2 (sfa=1.0):      d_max=8, CGIR, CCR, d_t=3, sfa_weight=1.0
Arm A3 (sfa=5.0):      d_max=8, CGIR, CCR, d_t=3, sfa_weight=5.0
Arm A4 (sfa=10.0):     d_max=8, CGIR, CCR, d_t=3, sfa_weight=10.0
Arm A5 (sfa=25.0 fix): d_max=8, CGIR, CCR, d_t=3, sfa_weight=25.0
Arm A6 (sfa=25.0 ramp):d_max=8, CGIR, CCR, d_t=3, sfa_weight ramp 0.1→25.0 over 1000 steps
Arm B  (d_max=16 sfa=10): d_max=16, CGIR, CCR, d_t=3, sfa_weight=10.0
```

All arms preserve:
- primary_objective="sfa"
- sim_weight=25.0 (JEPA readout, stop-gradient per M2)
- var_weight=25.0, cov_weight=25.0 (batch VICReg, M1)
- dyn_readout="centroid_gated" (CGIR)
- ccr_mode="covariance", ccr_smooth_weight=10.0, ccr_spatial_weight=10.0
- pos_encoding="none"
- sub_features=1, dyn_source="spatial"
- d_t=3 frozen, gdasr_log_only=True
- Adam lr=1e-3, batch=32, replay_buffer=2000
- 5000 training steps
- Seeds: [42, 123, 456, 789, 999]

### Arm A6 Ramp Implementation:
In the training loop, compute:
```python
effective_sfa_weight = 0.1 + (25.0 - 0.1) * min(1.0, step / 1000)
```
And pass it to model.forward() via the `sfa_weight=effective_sfa_weight` parameter override.

### Output Directory:
Save results to `archive/iter_023/results/`

### Evaluation Suite:
Same metrics as iter_022, directly comparable:
1. C5 (SFA effective): normalized_dyn_var < normalized_coord_var per seed
2. C1 (Collapse): per_dim_std < 0.5 in < 2/5 seeds per arm
3. Centroid MSE
4. delta_R2_color (primary criterion)
5. delta_R2_identity
6. Normalized temporal variance (dyn and coord)
7. Slowness ratio
8. Per-dim std, collapse counts
9. Tracking quality
10. GDASR growth-point logs

### Falsification Audit:
The final audit section must evaluate:

**PRIMARY**: delta_R2_color improvement over A1 baseline ≥ 0.10 for at least one arm.
**COMPOSITE M2 VIABILITY**: Existence of sfa_weight where ≥ 3/5 seeds simultaneously pass C5 + delta_R2_color ≥ 0.10 over A1 + per-dim std > 0.5.
**TERTIARY**: If all sfa_weight ≥ 5.0 arms collapse ≥ 3/5 seeds (even with ramping), SFA-VICReg conflict is unresolvable.

Use the same evaluation functions as run_phase0_sfa_archceiling.py (ReplayBuffer, set_seed, fit_linear_probe, compute_slowness_metrics, check_collapse, compute_vicreg_health, compute_normalized_temporal_var, compute_tracking_quality, compute_semantic_probes, compute_centroid_mse, evaluate_run, run_single).

The key changes from the archceiling runner are:
1. Different ARMS configuration (7 arms with varying sfa_weight)
2. Arm A6 needs ramp logic in run_single
3. The audit section evaluates the new falsification criteria
4. Output directory is archive/iter_023/results/
5. The comparison report should focus on SFA weight vs. metrics relationship

Make sure the code is complete, runnable, and handles all 7 arms correctly. The `run_single` function needs to accept the sfa_weight ramp configuration for Arm A6.

IMPORTANT: Read src/run_phase0_sfa_archceiling.py fully to understand the complete structure, then adapt it. Do NOT modify models_dual_stream.py — the sfa_weight parameter already supports per-forward-call override.