## Task: Create and run the CGIR Phase 0 experiment

Read src/pre_registration.md FIRST before doing anything else. You must strictly adhere to the pre-registered hypothesis and falsification criteria.

### Overview
Create `src/run_phase0_sfa_cgir.py` — an experiment runner that tests the CGIR (Centroid-Gated Identity Readout) architectural change against the pre-registered falsification criteria C1-C4.

### Architecture Changes (already made in src/models_dual_stream.py)
- `NonParametricEncoder` now accepts `dyn_readout="centroid_gated"` which:
  - Adds `conv_identity = nn.Conv1d(128, d_max, kernel_size=1)` 
  - Computes z_dyn by pooling identity features at soft-argmax-attended positions with stop-gradient on attention
  - `p_c = F.softmax(a_spatial, dim=-1)` then `z_dyn = torch.sum(a_identity * p_c.detach(), dim=-1)`
- `NonParametricJEPASpatial` passes `dyn_readout` parameter to the encoder

### Four Arms × 5 Seeds × 5000 Steps

```
Arm A (CGIR+SFA+CCR):      dyn_readout="centroid_gated", sfa_weight=0.1, pos_encoding="none", 
                            CCR=covariance (ccr_smooth=10, ccr_spatial=10), var=25, cov=25
Arm B (Mean+SFA+CCR):      dyn_readout="mean",           sfa_weight=0.1, pos_encoding="none",
                            CCR=covariance (ccr_smooth=10, ccr_spatial=10), var=25, cov=25
Arm C (CGIR+SFA+CCR+pos):  dyn_readout="centroid_gated", sfa_weight=0.1, pos_encoding="sinusoidal",
                            CCR=covariance (ccr_smooth=10, ccr_spatial=10), var=25, cov=25
Arm D (CGIR+SFA no CCR):   dyn_readout="centroid_gated", sfa_weight=0.1, pos_encoding="none",
                            CCR=none, var=25, cov=25
```

Seeds: [42, 123, 456, 789, 999]
Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000
d_t=3, gdasr_log_only=True

### Experiment Runner Structure

Use `src/run_phase0_sfa_recovery.py` as a template. The runner should:

1. **Training loop**: Same as recovery runner (ReplayBuffer, PhysicsSandbox with N=3, etc.)
   - Use `NonParametricJEPASpatial(primary_objective="sfa", ...)` with appropriate `dyn_readout`
   - Pass CCR params through the forward call

2. **Evaluation protocol** (same as iter_020):
   - Collapse check: per-dim std < 0.5
   - VICReg health: per-dim std and mean absolute correlation
   - Centroid decoding MSE: linear probe of z_coord → position
   - Slowness metrics: mean_dyn_delta / mean_coord_delta ratio
   - Semantic disentanglement probes: delta_R2_color
   - GDASR growth-point logging
   - Checkpoint evaluation at step 2500

3. **C4 Identity Probe** (NEW — Manager's recommended addition):
   Extend `compute_semantic_probes()` to also probe for object identity = compound label encoding both color AND size (radius).
   
   For each dimension d matched to object o:
   - Compute R²_dyn_identity: R² of z_dyn[d] predicting a compound identity label
   - Compute R²_coord_identity: R² of z_coord[d] predicting the same compound label
   - The compound label should combine color and radius information. Since color is 3D (RGB) and radius is 1D, create a 4D compound: `identity = [R, G, B, radius_normalized]`. Normalize radius to [0, 1] range (divide by max radius ~20).
   - Fit a MULTIVARIATE linear probe: predict the 4D identity vector from z_dyn[d] or z_coord[d] using a single linear layer. Report R² as 1 - SS_res/SS_tot.
   - Compute delta_R2_identity = R²_dyn_identity - R²_coord_identity
   - This is C4: delta_R2_identity ≥ 0.10 means z_dyn predicts identity better than z_coord

   Implementation: For multivariate R², use:
   ```python
   # y is (N, 4), z is (N,)
   # Fit: y_pred = z @ W + b where W is (1, 4), b is (4,)
   # Use least-squares
   Z_aug = np.stack([z, np.ones_like(z)], axis=1)  # (N, 2)
   theta = np.linalg.pinv(Z_aug.T @ Z_aug) @ Z_aug.T @ y  # (2, 4)
   y_pred = Z_aug @ theta  # (N, 4)
   ss_res = np.sum((y - y_pred) ** 2)
   ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
   r2 = 1.0 - ss_res / (ss_tot + 1e-12)
   ```

4. **Output files** — save to `archive/iter_020/results/` (same directory structure as iter_020 recovery):
   - Per-run CSVs and JSONs in `runs/` subdirectory
   - Training logs in `runs/`
   - Model checkpoints in `checkpoints/`
   - Summary CSV: `summary_phase0_cgir.csv` (final step=5000 only)
   - Checkpoint summary: `summary_phase0_cgir_cp2500.csv`
   - Aggregated stats: `aggregated_phase0_cgir.csv`
   - Audit JSON: `audit_phase0_cgir.json`
   - **Comparison report**: `CGIR_COMPARISON_REPORT.md` — must include:
     - Table comparing all 4 arms on C1-C4
     - Per-dimension probe details for the CGIR arm
     - Comparison with iter_020 results (Arm A1 from iter_020 = replication of Arm B here)
     - Honest falsification audit
     - If C3 passes for Arm A but not Arm B, frame as: "CGIR structurally routes per-object appearance information into z_dyn, consistent with the hypothesis that mean-pooling prevented this routing." NOT "SFA enables emergent identity-position disentanglement."
     - If C3 passes but C4 fails, clarify: the separation is color-vs-position, not identity-vs-position.

### CRITICAL Implementation Notes

1. The `NonParametricJEPASpatial` constructor signature is:
   ```python
   NonParametricJEPASpatial(d_max=8, h=3, k=4, cooldown=300, stabilization_period=100, 
       pos_encoding="none", primary_objective="jepa", sfa_weight=25.0, 
       gdasr_log_only=True, dyn_readout="mean")
   ```
   Use `primary_objective="sfa"` for all SFA arms.

2. For Arm D (no CCR), pass `ccr_mode='none'` to the forward call.

3. Make sure the `collect_multitraj_eval_data()` function also collects `radii` from `info["radii"]` for the C4 probe.

4. Save the comparison report to `archive/iter_020/results/CGIR_COMPARISON_REPORT.md`.

5. Run the full experiment (not dry-run). Expect ~20 minutes per arm (5 seeds × 5000 steps each).

6. After all runs complete, generate the falsification audit JSON comparing all arms against C1-C4.

7. IMPORTANT: The evaluation functions (collapse check, centroid MSE, semantic probes) should work with the CGIR encoder just like they did with the mean-pooling encoder. The encoder returns (z_coord, z_dyn) with the same shapes regardless of dyn_readout mode.

8. Make sure the runner script can be executed standalone: `python src/run_phase0_sfa_cgir.py`
