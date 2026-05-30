ARM 2 — M2 Decisive Test for Iteration 030.

FIRST: Read src/pre_registration.md to understand the full pre-registered hypothesis and falsification criteria. You MUST adhere strictly to these criteria.

## Overview
Train two new representation objectives on the separate-backbone architecture and test whether either achieves mean ΔR²_color ≥ 0.30 with variance-stability (lower 95% CI ≥ 0.18) on a 30-seed union bank.

## Two Arms

### D1 — Temporal Identity Contrastive Binding (NT-Xent)
- **Architecture**: NonParametricJEPASpatialSeparateDyn with primary_objective="contrastive"
- **Loss**: Same-object-across-consecutive-timesteps as positive pairs, different-objects-as-negatives
- **Implementation**: 
  1. Encode x_target and x_hist[:,-1] to get z_dyn_target (B, d_dyn) and z_dyn_hist (B, H, d_dyn), taking z_dyn_hist[:,-1] as z_dyn_prev
  2. Use sorted matching: sort z_coord channels by centroid value, sort object positions by position, map channel→object
  3. For each matched (channel c, object o): z_dyn_target[:, c] and z_dyn_prev[:, c] form a positive pair; z_dyn_target[:, c] and z_dyn_prev[:, other_c] for other_c ≠ c form negatives
  4. NT-Xent loss: contrastive_loss = cross_entropy(sim(z_pos, z_anchor) / temperature, labels)
  5. The contrastive loss replaces SFA as the primary identity-shaping objective on z_dyn
  6. Total loss = contrastive_weight * contrastive_loss + var_weight * VICReg_var + cov_weight * VICReg_cov + sim_weight * JEPA_readout_loss
- **Key params**: contrastive_weight=5.0, temperature=0.1, mask_dyn_sim=True, coord_vicreg=True, sfa_weight=0 (SFA disabled)

### D2 — Variance-Ramped SFA
- **Architecture**: NonParametricJEPASpatialSeparateDyn with primary_objective="sfa"  
- **Loss**: Same as iter_029 Arm B but sfa_weight is LINEARLY RAMPED from 0 to 5.0 over the first 4000 training steps, then held constant at 5.0 for steps 4001-8000
- **Implementation**: In the training loop, compute: sfa_weight_current = min(5.0, 5.0 * step / 4000) and pass sfa_weight=sfa_weight_current to model.forward()
- **Key params**: mask_dyn_sim=True, coord_vicreg=True, sfa_weight=5.0 (target, ramped)

### Shared Configuration
- d_max=8, d_t=3 (frozen), h=3, k=4
- Separate backbone (NonParametricJEPASpatialSeparateDyn)
- GDASR log-only (gdasr_log_only=True)
- pos_encoding="none"
- lr=3e-4, batch_size=32
- buffer_capacity=4000
- sim_weight=1.0, var_weight=25.0, cov_weight=25.0
- total_steps=8000
- eval_steps=200 for semantic probe evaluation

### Seed Bank (30 seeds)
Original: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
Fresh: [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]
New: [173, 179, 181, 191, 193, 197, 199, 211, 223, 227]
Hard seeds 53, 71 must be included but flagged separately.

## Contrastive Loss Implementation Detail

The existing model's forward method with primary_objective="contrastive" already has a basic NT-Xent implementation using batch-level positive pairs (z_target and z_hist[:,-1]). However, this is NOT object-level matching — it treats each batch element as its own positive pair, which doesn't enforce same-object identity across time.

You need to implement OBJECT-LEVEL temporal contrastive binding. Here's how:

1. In the forward pass, after encoding x_target and x_hist, extract z_dyn_target (B, d_t) and z_dyn_prev (B, d_t) — the latter from z_hist_dyn[:, -1, :d_t]
2. Match channels to objects using sorted matching on z_coord:
   - Sort z_coord channels by centroid value (ascending)
   - Sort object positions (ascending)  
   - Map sorted channel index → sorted object index
3. For each batch element, for each channel c matched to object o:
   - Positive pair: (z_dyn_target[b, c], z_dyn_prev[b, c]) — same object at consecutive timesteps
   - Negatives: (z_dyn_target[b, c], z_dyn_prev[b, c']) for all c' ≠ c — different objects at t-1
4. Compute NT-Xent loss per (batch, channel) pair, then average

This requires access to the object positions and colors from the replay buffer (which stores them). In the training loop, when sampling a batch, extract positions to compute the matching.

If implementing object-level contrastive is too complex, you may use the EXISTING batch-level NT-Xent already implemented in the model (primary_objective="contrastive"), which uses (z_target, z_prev) from the same batch element as positive pairs. This is weaker but simpler. Document which approach was used.

## Evaluation Protocol (same as iter_029)
For each trained model, compute:
1. **ΔR²_color**: linear probe R² for color from z_dyn minus R² from z_coord (using Hungarian matching)
2. **collapse_rate**: whether per-dim std < 0.5 on eval data
3. **centroid_mse_mean**: linear probe MSE for centroid-to-position mapping
4. **r2_dyn_color**, **r2_coord_color**: per-stream color decodability
5. **r2_dyn_pos**, **r2_coord_pos**: per-stream position decodability
6. **mean_abs_corr**: inter-channel correlation

Use the same evaluation functions from src/run_phase0_sfa_separate_backbone.py:
- evaluate_run()
- compute_semantic_probes()
- compute_centroid_mse()
- check_collapse()
- compute_vicreg_health()

## Falsification Criteria
- **D1 falsified**: mean ΔR²_color < 0.30 OR lower 95% CI < 0.18 on 30-seed bank
- **D2 falsified**: mean ΔR²_color < 0.30 OR lower 95% CI < 0.18 on 30-seed bank
- **Both falsified**: ΔR²_color ≥ 0.30 is unachievable by any decoder-free objective tested

For the 95% CI lower bound: compute mean and std of ΔR²_color across non-collapsed seeds, then CI_lower = mean - 1.96 * std / sqrt(n). If n < 5, cannot compute CI → flag as underpowered.

## Output Files
- `archive/iter_030/results/arm2_per_seed.csv` — per-seed per-arm results
- `archive/iter_030/results/arm2_summary.csv` — arm-level summary
- `archive/iter_030/results/arm2_analysis.md` — analysis with gate evaluation
- `archive/iter_030/results/arm2_checkpoints/` — saved model checkpoints

## Implementation Notes
- Base your script on src/run_phase0_sfa_separate_backbone.py which has all the evaluation infrastructure
- For D2 (variance-ramped SFA): modify the training loop to compute sfa_weight_current = min(5.0, 5.0 * step / 4000) and pass it as sfa_weight=sfa_weight_current to model.forward()
- For D1 (temporal contrastive): use primary_objective="contrastive" in the model config. The existing forward method already has a batch-level NT-Xent implementation. You can use this as-is or extend it for object-level matching.
- Run on CPU. Use torch.set_num_threads(2). Process sequentially or with ProcessPoolExecutor (max 4 workers).
- Report mean ± std for all metrics. Flag hard seeds 53, 71 separately.
- For each arm, also report: the 95% CI lower bound for ΔR²_color, and whether it exceeds 0.18.
