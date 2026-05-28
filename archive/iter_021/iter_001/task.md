You are implementing the Architectural Ceiling experiment for the Thalamus project (iter 022 / phase 21). Read `src/pre_registration.md` first for context, then read `src/models_dual_stream.py` and `src/run_phase0_sfa_cgir.py` for the current code.

Your task has three parts:

## PART A: Update Pre-Registration

Write `src/pre_registration.md` with updated content. Key changes from existing pre-reg (based on Strategic Research Manager's 3 directives):

1. **UNAMBIGUOUS FALSIFICATION FORMULA** (Directive 1):
   The primary falsification criterion must be written as pseudocode:
   ```
   PRIMARY: If mean_over_seeds(delta_R2_identity[Arm_C] - delta_R2_identity[Ctrl]) < 0.10,
   then the single-scalar bottleneck is NOT the primary cause, and the hypothesis is falsified.
   ```
   Similarly for secondary criteria using improvement notation.

2. **NORMALIZED TEMPORAL VARIANCE** (Directive 2):
   Add to metrics: normalized_temporal_var(z) = mean(frame-to-frame Δz²) / mean(z²)
   SFA effectiveness = normalized_temporal_var(z_dyn) < normalized_temporal_var(z_coord)
   If Arm C passes identity threshold BUT normalized_dyn_var ≥ normalized_coord_var,
   the result is "capacity enables encoding; SFA is along for the ride" — a WEAKER claim.

3. **CAPACITY VS SFA DISTINCTION** (Directive 3):
   Pre-commit: If Arm C passes delta_R2_identity ≥ 0.10 improvement but per-sub-feature
   probes show no selective encoding (each sub-feature has similar R² across all 4 identity
   dimensions), then report as "capacity enables identity encoding, but SFA does not produce
   disentangled sub-feature specialization" — NOT as "SFA shapes disentanglement."

## PART B: Model Changes in src/models_dual_stream.py

### B1: NonParametricEncoder modifications

Add two new parameters to __init__:
- `sub_features` (K, int, default=1): number of sub-features per channel for z_dyn
- `dyn_source` ("spatial"|"conv4", default="spatial"): source of features for z_dyn computation

**When dyn_source="conv4" and K=1 (Arm A):**
- Add `self.dyn_proj = nn.Linear(128, 1)` in __init__
- In forward(): after computing a_spatial and p_c (centroid attention), interpolate the conv4 backbone features to width 128, then use p_c to soft-attend:
  ```python
  features_interp = F.interpolate(features, size=128, mode='linear', align_corners=False)  # (B, 128, 128)
  p_c = F.softmax(a_spatial, dim=-1)  # (B, d_max, 128) — same as centroid_gated
  attended = torch.bmm(p_c, features_interp.transpose(1, 2))  # (B, d_max, 128)
  attended_flat = attended.reshape(B * d_max, 128)
  z_dyn = self.dyn_proj(attended_flat).reshape(B, d_max)  # (B, d_max)
  ```
  Stop-gradient on p_c just like centroid_gated: use `p_c.detach()`.

**When sub_features=K > 1 (Arm C):**
- `conv_identity` outputs `d_max * K` channels: `nn.Conv1d(128, d_max * K, kernel_size=1)`
- After interpolation to 128 width, reshape:
  ```python
  a_identity = self.conv_identity(features)  # (B, d_max*K, 8)
  a_identity = F.interpolate(a_identity, size=128, mode='linear', align_corners=False)  # (B, d_max*K, 128)
  a_identity = a_identity.reshape(B, d_max, K, 128)  # (B, d_max, K, 128)
  p_c = F.softmax(a_spatial, dim=-1).detach()  # (B, d_max, 128) — stop-gradient on attention
  z_dyn = torch.einsum('bcs,bcks->bck', p_c, a_identity)  # (B, d_max, K)
  z_dyn = z_dyn.reshape(B, d_max * K)  # (B, d_max * K)
  ```

**For existing centroid_gated with K=1 (Ctrl):** keep exactly as-is. The existing `dyn_readout="centroid_gated"` path should remain unchanged.

**Important**: In the forward() method, return z_dyn with shape (B, d_max * K). Add a property `d_dyn` that returns `d_max * sub_features`.

**For d_max=16 (Arm B)**: Just pass d_max=16. conv_spatial and conv_identity will be 128→16. sub_features=1, dyn_source="spatial", dyn_readout="centroid_gated". Works naturally.

### B2: DualStreamPredictor modifications

Add `d_dyn` parameter (default=None, falls back to d_max):
```python
class DualStreamPredictor(nn.Module):
    def __init__(self, d_max=8, d_dyn=None, h=3):
        self.d_max = d_max
        self.d_dyn = d_dyn if d_dyn is not None else d_max
        self.h = h
        total_in = h * (d_max + self.d_dyn)
        total_out = d_max + self.d_dyn
        self.net = nn.Sequential(
            nn.Linear(total_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, total_out)
        )
```

In forward(), handle the masking differently for coord vs dyn:
- coord: active dims = d_t (of d_max total), mask dims >= d_t
- dyn: active dims = d_t * K where K = d_dyn / d_max (but pass K separately or compute from d_dyn/d_max)
  Actually simpler: pass `d_t_dyn` as a parameter = d_t * sub_features, defaulting to d_t.
  Or just use `d_t * (self.d_dyn // self.d_max)` if they're cleanly divisible.
  
  Simplest approach: forward signature becomes:
  ```python
  def forward(self, z_coord_history, z_dyn_history, d_t, mask_coord=False, d_t_dyn=None):
  ```
  where d_t_dyn defaults to d_t (backward compatible) but for K>1 it's d_t * K.

The split of the output into pred_coord and pred_dyn should use d_max for the split point:
```python
pred_coord, pred_dyn = torch.split(pred, self.d_max, dim=-1)
```

And the masking of inactive dims:
- coord: mask dims >= d_t → 0
- dyn: mask dims >= d_t_dyn → 0

### B3: NonParametricJEPASpatial modifications

Add `sub_features` and `dyn_source` params to __init__. Pass them through to encoder and set up predictor with correct d_dyn:
```python
def __init__(self, ..., sub_features=1, dyn_source="spatial"):
    self.sub_features = sub_features
    self.dyn_source = dyn_source
    self.encoder = NonParametricEncoder(d_max=d_max, pos_encoding=pos_encoding, 
        dyn_readout=dyn_readout, sub_features=sub_features, dyn_source=dyn_source)
    self.predictor = DualStreamPredictor(d_max=d_max, d_dyn=d_max * sub_features, h=h)
```

In the SFA mode of forward():
- z_dyn from encoder has shape (B, d_max * K). Active features = d_t * K.
- SFA loss: `F.mse_loss(z_target_dyn[:, :self.d_t * self.sub_features], z_prev_dyn[:, :self.d_t * self.sub_features].detach())`
- VICReg variance + covariance on z_target_dyn[:, :self.d_t * self.sub_features]
- Predictor receives z_coord_history (B, H, d_max) and z_dyn_history (B, H, d_max*K)
- In predictor call, pass d_t_dyn=self.d_t * self.sub_features

For the JEPA readout with sub-features:
- z_hist_sfa_dyn: (B, H, d_max*K) → detach
- Predictor output: pred_coord (B, d_max), pred_dyn (B, d_max*K)
- Prediction losses on active dims only (d_t for coord, d_t*K for dyn)

CCR remains on z_coord only — UNCHANGED.

Also update clone() method to pass sub_features and dyn_source.

## PART C: Create Experiment Runner

Create `src/run_phase0_sfa_archceiling.py` based on `src/run_phase0_sfa_cgir.py` but with these differences:

### Arms Configuration:
```python
ARMS = [
    {
        "name": "Ctrl (CGIR+SFA+CCR, d_max=8, K=1)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "Arm A (Conv4 CGIR, d_max=8, K=1)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "conv4",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "Arm B (Expanded d_max=16, K=1)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 16, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "Arm C (Sub-Features K=4, d_max=8)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 4, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
]
SEEDS = [42, 123, 456, 789, 999]
```

### Model Creation in run_single:
```python
model = NonParametricJEPASpatial(
    d_max=arm_config["d_max"],
    h=3, k=4, cooldown=300, stabilization_period=100,
    pos_encoding=arm_config["pos_encoding"],
    primary_objective=arm_config["primary_objective"],
    sfa_weight=arm_config["sfa_weight"],
    gdasr_log_only=True,
    dyn_readout=arm_config["dyn_readout"],
    sub_features=arm_config["sub_features"],
    dyn_source=arm_config["dyn_source"],
)
model.d_t = arm_config["d_t"]
```

### Updated Evaluation Metrics (add to evaluate_run):

1. **Normalized temporal variance**:
   - temporal_var(z) = mean((z[t+1] - z[t])²) over consecutive frames
   - spatial_var(z) = mean(z²) over all frames (proxy for magnitude)
   - normalized_var = temporal_var / (spatial_var + 1e-8)
   - Report separately for z_dyn (active dims only: d_t * K) and z_coord (d_t)
   - SFA effective iff normalized_dyn_var < normalized_coord_var

2. **Centroid tracking quality**:
   - For each channel c, compute corr(Δz_coord[c], Δtrue_pos[matched_obj[c]])
   - Also compute corr(z_coord[c], true_pos[matched_obj[c]]) (level tracking)
   - High correlation = good tracking. Low = tracking failure.
   - Match channels to objects using the same dim_to_obj mapping as semantic probes.

3. **Per-sub-feature identity probes** (Arm C with K=4 only):
   - For each channel c in [0, d_t) and sub-feature k in [0, K):
     compute R² of z_dyn[c*K+k] against [R, G, B, radius_normalized] individually
   - Also compute per-sub-feature multivariate R² against the full identity vector
   - Test: if disentangled, some (c,k) pairs should have high R² for specific identity
     dimensions. If distributed, all (c,k) have similar R² across all dims.

### Semantic Probes Adaptation for Arm C:

For the standard delta_R2_color/identity metrics that are compared across all arms:
- Pool the K sub-features per channel: z_dyn_pooled[c] = mean(z_dyn[c*K:(c+1)*K])
- Then use z_dyn_pooled[c] for the standard probe, just like the K=1 case
- This ensures the comparison is fair: each arm produces the same number of probed dimensions

For per-sub-feature probes (Arm C only):
- Probe each z_dyn[c*K+k] individually against identity components

### Output directory: archive/iter_022/results/

### Falsification Audit:
Use IMPROVEMENT-based criteria throughout:
```
C1 (Collapse): Ctrl collapsed seeds < 2 AND Arm C collapsed seeds < 2
C2 (Tracking): Arm C centroid MSE ≤ 1.10 × Ctrl centroid MSE
C3 (Color):   mean_over_seeds(delta_R2_color[Arm C] - delta_R2_color[Ctrl]) ≥ 0.10
C4 (Identity): mean_over_seeds(delta_R2_identity[Arm C] - delta_R2_identity[Ctrl]) ≥ 0.10
C5 (SFA effective, advisory): normalized_dyn_var[Arm C] < normalized_coord_var[Arm C]

OVERALL: C1 AND C2 AND C4 → hypothesis validated
```

### TESTING:
After writing all code, run these tests:

1. Model forward-pass test:
```bash
cd /home/user && python -c "
import sys; sys.path.insert(0, '.')
from src.models_dual_stream import NonParametricJEPASpatial
import torch

# Test default (K=1, dyn_source=spatial) - should match existing behavior
m1 = NonParametricJEPASpatial(d_max=8, h=3, k=4, pos_encoding='none', primary_objective='sfa', sfa_weight=0.1, gdasr_log_only=True, dyn_readout='centroid_gated', sub_features=1, dyn_source='spatial')
m1.d_t = 3
x_h = torch.randn(2, 3, 3, 128)
x_t = torch.randn(2, 3, 128)
out = m1(x_h, x_t, ccr_mode='covariance')
print('K=1 spatial: loss =', out[0]['loss'].item())

# Test K=4
m2 = NonParametricJEPASpatial(d_max=8, h=3, k=4, pos_encoding='none', primary_objective='sfa', sfa_weight=0.1, gdasr_log_only=True, dyn_readout='centroid_gated', sub_features=4, dyn_source='spatial')
m2.d_t = 3
out2 = m2(x_h, x_t, ccr_mode='covariance')
print('K=4 spatial: loss =', out2[0]['loss'].item())

# Test dyn_source=conv4
m3 = NonParametricJEPASpatial(d_max=8, h=3, k=4, pos_encoding='none', primary_objective='sfa', sfa_weight=0.1, gdasr_log_only=True, dyn_readout='centroid_gated', sub_features=1, dyn_source='conv4')
m3.d_t = 3
out3 = m3(x_h, x_t, ccr_mode='covariance')
print('K=1 conv4: loss =', out3[0]['loss'].item())

# Test d_max=16
m4 = NonParametricJEPASpatial(d_max=16, h=3, k=4, pos_encoding='none', primary_objective='sfa', sfa_weight=0.1, gdasr_log_only=True, dyn_readout='centroid_gated', sub_features=1, dyn_source='spatial')
m4.d_t = 3
out4 = m4(x_h, x_t, ccr_mode='covariance')
print('d_max=16: loss =', out4[0]['loss'].item())

print('All model tests PASSED')
"
```

2. Dry-run of experiment script:
```bash
cd /home/user && python src/run_phase0_sfa_archceiling.py --dry-run --seeds 42
```

WRITE ALL FILES. Do not just describe what to write — actually write the complete code to the files.