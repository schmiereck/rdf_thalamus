You need to create the experiment runner script and verify everything works. The model changes in src/models_dual_stream.py have already been made (adding sub_features, dyn_source to NonParametricEncoder, DualStreamPredictor, NonParametricJEPASpatial).

## STEP 1: Quick model forward-pass verification

Run this test:
```bash
cd /home/user && python -c "
import sys; sys.path.insert(0, '.')
from src.models_dual_stream import NonParametricJEPASpatial
import torch

# Test default (K=1, dyn_source=spatial)
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

# Test backward pass for all
for i, (m, o) in enumerate([(m1, out), (m2, out2), (m3, out3), (m4, out4)]):
    o[0]['loss'].backward()
    print(f'Model {i+1} backward: OK')

print('All model tests PASSED')
"
```

If any test fails, fix the model code in src/models_dual_stream.py before proceeding.

## STEP 2: Create the experiment runner

Create `src/run_phase0_sfa_archceiling.py` based on `src/run_phase0_sfa_cgir.py`. Read the CGIR runner first to understand the structure. The new runner has:

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

### Key differences from CGIR runner:

1. **Model creation** in run_single():
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

2. **Updated evaluate_run function** with these new metrics:

a) **Normalized temporal variance**:
```python
def compute_normalized_temporal_var(z_dyn_arr, z_coord_arr, d_t, sub_features=1):
    """z_dyn_arr: (N, d_max*K), z_coord_arr: (N, d_max)"""
    d_t_dyn = d_t * sub_features
    # Temporal variance: mean(Δz²)
    dyn_diffs = np.diff(z_dyn_arr[:, :d_t_dyn], axis=0)
    coord_diffs = np.diff(z_coord_arr[:, :d_t], axis=0)
    temporal_var_dyn = np.mean(dyn_diffs ** 2)
    temporal_var_coord = np.mean(coord_diffs ** 2)
    # Spatial variance: mean(z²)
    spatial_var_dyn = np.mean(z_dyn_arr[:, :d_t_dyn] ** 2)
    spatial_var_coord = np.mean(z_coord_arr[:, :d_t] ** 2)
    # Normalized
    norm_dyn = temporal_var_dyn / (spatial_var_dyn + 1e-8)
    norm_coord = temporal_var_coord / (spatial_var_coord + 1e-8)
    sfa_effective = norm_dyn < norm_coord
    return {
        "temporal_var_dyn": temporal_var_dyn,
        "temporal_var_coord": temporal_var_coord,
        "spatial_var_dyn": spatial_var_dyn,
        "spatial_var_coord": spatial_var_coord,
        "normalized_dyn_var": norm_dyn,
        "normalized_coord_var": norm_coord,
        "sfa_effective": bool(sfa_effective),
    }
```

b) **Centroid tracking quality**:
```python
def compute_tracking_quality(z_coord_arr, pos_arr, d_t):
    """Compute correlation between z_coord changes and true position changes."""
    # Match channels to objects by minimum distance
    dim_to_obj = {}
    used_objs = set()
    for d in range(d_t):
        best_obj = None
        best_dist = np.inf
        for o in range(pos_arr.shape[1]):
            if o in used_objs:
                continue
            dist = np.mean(np.abs(z_coord_arr[:, d] - pos_arr[:, o]))
            if dist < best_dist:
                best_dist = dist
                best_obj = o
        if best_obj is not None:
            dim_to_obj[d] = best_obj
            used_objs.add(best_obj)
    
    delta_corr = []
    level_corr = []
    for d in range(d_t):
        if d not in dim_to_obj:
            continue
        o = dim_to_obj[d]
        dz = np.diff(z_coord_arr[:, d])
        dp = np.diff(pos_arr[:, o])
        if len(dz) > 2 and np.std(dz) > 1e-8 and np.std(dp) > 1e-8:
            delta_corr.append(np.corrcoef(dz, dp)[0, 1])
        else:
            delta_corr.append(0.0)
        
        z_c = z_coord_arr[:, d]
        p = pos_arr[:, o]
        if np.std(z_c) > 1e-8 and np.std(p) > 1e-8:
            level_corr.append(np.corrcoef(z_c, p)[0, 1])
        else:
            level_corr.append(0.0)
    
    return {
        "dim_to_obj": dim_to_obj,
        "delta_corr_mean": float(np.mean(delta_corr)) if delta_corr else 0.0,
        "level_corr_mean": float(np.mean(level_corr)) if level_corr else 0.0,
        "delta_corr_per_dim": delta_corr,
        "level_corr_per_dim": level_corr,
    }
```

c) **Per-sub-feature identity probes** (only when sub_features > 1):
```python
def compute_sub_feature_probes(z_dyn_arr, colors_arr, radii_arr, d_t, sub_features, num_samples, train_ratio=0.5):
    """
    For each (channel c, sub-feature k), fit R² against [R, G, B, radius_norm] individually.
    Returns a (d_t, sub_features, 4) array of R² values.
    """
    max_radius = 20.0
    n_train = int(num_samples * train_ratio)
    
    # Per-sub-feature R² against individual identity dimensions
    r2_matrix = np.zeros((d_t, sub_features, 4))
    
    for c in range(d_t):
        for k in range(sub_features):
            idx = c * sub_features + k
            z = z_dyn_arr[n_train:, idx]
            
            for id_dim, (id_data,) in enumerate([
                (colors_arr[n_train:, c, 0],),  # R
                (colors_arr[n_train:, c, 1],),  # G
                (colors_arr[n_train:, c, 2],),  # B
                (radii_arr[n_train:, c] / max_radius,),  # radius_norm
            ]):
                if np.std(z) < 1e-8 or np.std(id_data) < 1e-8:
                    r2_matrix[c, k, id_dim] = 0.0
                else:
                    # Use training data for fit, test for R²
                    z_train = z_dyn_arr[:n_train, idx]
                    y_train = id_data_train  # Need to construct properly
                    # Actually simpler: just use scipy or manual linear regression
                    # Let's use numpy
                    from numpy.polynomial import polynomial as P
                    # Fit on train
                    z_tr = z_dyn_arr[:n_train, idx]
                    y_tr = id_data_tr  # train targets
                    if id_dim < 3:
                        y_tr = colors_arr[:n_train, c, id_dim]
                    else:
                        y_tr = radii_arr[:n_train, c] / max_radius
                    
                    # Linear regression
                    A = np.vstack([z_tr, np.ones_like(z_tr)]).T
                    try:
                        theta = np.linalg.lstsq(A, y_tr, rcond=None)[0]
                        y_pred = z * theta[0] + theta[1]
                        ss_res = np.sum((id_data - y_pred) ** 2)
                        ss_tot = np.sum((id_data - np.mean(y_tr)) ** 2)
                        if ss_tot < 1e-12:
                            r2_matrix[c, k, id_dim] = 0.0
                        else:
                            r2_matrix[c, k, id_dim] = float(1.0 - ss_res / ss_tot)
                    except:
                        r2_matrix[c, k, id_dim] = 0.0
    
    return r2_matrix
```

Actually, this per-sub-feature probe needs to match the same dim_to_obj mapping as the semantic probes. Let me simplify:

For the sub-feature probes, match each channel c to its closest object (same as semantic probes), then for that object, probe each sub-feature against that object's identity features.

d) **Updated evaluate_run**: collects z_dyn, z_coord, pos, colors, radii during evaluation, then computes all the new metrics. The key is to collect enough data for tracking quality (needs true positions per step).

3. **Semantic probes adaptation for K>1**: For the standard delta_R2 metrics that compare across all arms, pool K sub-features per channel:
```python
if sub_features > 1:
    z_dyn_pooled = z_dyn_arr.reshape(num_samples, d_max, sub_features).mean(axis=2)  # (N, d_max)
else:
    z_dyn_pooled = z_dyn_arr
```
Then use z_dyn_pooled in the standard semantic probe. This ensures fair comparison.

4. **Falsification audit** using improvement-based criteria:
```
C1 (Collapse): Ctrl collapsed < 2 AND Arm C collapsed < 2
C2 (Tracking): Arm C MSE ≤ 1.10 × Ctrl MSE
C3 (Color):   mean_over_seeds(delta_R2_color[Arm C] - delta_R2_color[Ctrl]) ≥ 0.10
C4 (Identity): mean_over_seeds(delta_R2_identity[Arm C] - delta_R2_identity[Ctrl]) ≥ 0.10
C5 (SFA effective): normalized_dyn_var[Arm C] < normalized_coord_var[Arm C]

OVERALL: C1 AND C2 AND C4 → hypothesis validated
```

5. **Output directory**: archive/iter_022/results/

6. **Training**: Same as CGIR runner — 5000 steps, Adam lr=1e-3, batch=32, replay_buffer=2000. 
   BUT in the training loop, pass `d_t_predict=arm_config["d_t"]` to model.forward() and set `d_t_dyn` correctly.

7. **Important**: The training loop should pass the correct d_t_predict. For all arms, d_t=3. The predictor internally uses d_t_dyn = d_t * sub_features for masking.

## STEP 3: Dry-run test

After creating the runner, test it:
```bash
cd /home/user && python src/run_phase0_sfa_archceiling.py --dry-run --seeds 42
```

This should complete quickly (5 training steps) and produce output files in archive/iter_022/results/.

WRITE THE COMPLETE RUNNER FILE. Actually write it to disk — don't just describe it.