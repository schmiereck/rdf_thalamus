You are implementing and running the iter_025 Architecture Ceiling Probe experiment. Read src/pre_registration.md FIRST for the full specification, then implement and run the experiment.

## CONTEXT
This is a localization probe to determine whether the failure of identity encoding in z_dyn (delta_R2_color < 0.10 across iter_021-024) is due to insufficient objective or architecture ceiling.

Previous iterations (021-024) tested SFA and temporal contrastive on z_dyn. ALL FAILED. This iteration adds supervised color probe (Arm B) and ID-contrastive (Arm C) as stronger objectives, with careful matching protocol.

## KEY FILES TO READ FIRST
1. src/pre_registration.md — Full specification with matching confound protocol, noise floor, language hygiene
2. src/models_dual_stream.py — Current model code (NonParametricJEPASpatial class)
3. src/run_phase0_sfa_multistep.py — Previous experiment runner (basis for new runner)
4. src/environment.py — PhysicsSandbox (provides positions, colors, radii in info dict)

## IMPLEMENTATION PLAN

### Step 1: Add methods to src/models_dual_stream.py

Add to NonParametricJEPASpatial class:

a) Color probe head parameters in __init__:
```python
self.color_probe_weight = nn.Parameter(torch.randn(d_max, 3) * 0.01)
self.color_probe_bias = nn.Parameter(torch.zeros(d_max, 3))
```

b) Static method for channel-to-object matching (sorted position):
```python
@staticmethod
def match_channels_sorted(z_coord, positions, d_t, N):
    """Match z_coord channels to objects by sorting both by position.
    Returns: assignment dict {dim_idx: obj_idx}
    """
    z_sorted_idx = torch.argsort(z_coord[:, :d_t], dim=1)  # (B, d_t)
    pos_sorted_idx = torch.argsort(positions[:, :N], dim=1)  # (B, N)
    # For each batch sample, channel d_t_sorted[d] corresponds to obj N_sorted[d]
    return z_sorted_idx, pos_sorted_idx
```

c) Static method for Hungarian matching:
```python
@staticmethod
def match_channels_hungarian(z_coord, positions, d_t, N):
    """Match z_coord channels to objects using optimal assignment (Hungarian).
    Returns assignment dict per sample.
    """
    from scipy.optimize import linear_sum_assignment
    B = z_coord.shape[0]
    assignments = []
    for b in range(B):
        cost = torch.cdist(z_coord[b, :d_t].unsqueeze(0), positions[b, :N].unsqueeze(0)).squeeze(0)  # (d_t, N)
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        assignments.append(list(zip(row_ind, col_ind)))
    return assignments
```

d) Method compute_supervised_color_loss(z_coord, z_dyn, positions, colors, d_t, N, matching_mode="sorted"):
- Uses the matching to assign each z_dyn channel to an object
- Computes color_pred = z_dyn_matched * color_probe_weight + color_probe_bias  
- Returns MSE(color_pred, colors_matched) and the assignment info

e) Method compute_id_contrastive_loss(z_coord, z_dyn, positions, colors, d_t, N, matching_mode="sorted"):
- Match channels to objects
- Discretize colors into 8 bins (RGB ordering: which channel is dominant)
- Compute SupCon-style contrastive loss with these labels
- Returns the contrastive loss

f) Method compute_mismatch_rate(z_coord, positions, d_t, N):
- Run both sorted and Hungarian matching
- Return fraction of (batch, channel) pairs where assignments disagree

### Step 2: Create src/run_phase0_id_probe.py

Based on run_phase0_sfa_multistep.py with these key modifications:

1. **Extended ReplayBuffer** that stores positions and colors alongside observations:
```python
class ExtendedReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, x_hist, x_target, positions, colors, radii):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (x_hist, x_target, positions, colors, radii)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        x_hist_b, x_target_b, pos_b, colors_b, radii_b = zip(*batch)
        return (np.stack(x_hist_b), np.stack(x_target_b), 
                np.stack(pos_b), np.stack(colors_b), np.stack(radii_b))
```

2. **Noise floor runs** (3 runs, frozen encoder, 1000 steps):
   - Create model with random init
   - Freeze all encoder parameters
   - Only train the linear probe head
   - Measure delta_R2_color → compute floor_mean

3. **Main experiment arms** (use same training loop structure as iter_024):

Arm A: JEPA+VICReg Control
```python
{"name": "A (JEPA+VICReg Control)", "primary_objective": "jepa", 
 "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
 "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
 "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
 "supervised_weight": 0.0, "contrastive_weight": 0.0}
```

Arm B: Supervised Color Probe + VICReg (d_max=8)
```python
{"name": "B (Supervised Color Probe d_max=8)", "primary_objective": "jepa",
 "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
 "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
 "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
 "supervised_weight": 25.0, "contrastive_weight": 0.0}
```

Arm C: ID-Contrastive + VICReg (d_max=8)
```python
{"name": "C (ID-Contrastive d_max=8)", "primary_objective": "jepa",
 "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
 "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
 "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
 "supervised_weight": 0.0, "contrastive_weight": 25.0}
```

Arm D: Supervised Color Probe + VICReg (d_max=16)
```python
{"name": "D (Supervised Color Probe d_max=16)", "primary_objective": "jepa",
 "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
 "d_max": 16, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
 "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
 "supervised_weight": 25.0, "contrastive_weight": 0.0}
```

4. **Training loop modifications** for Arms B, C, D:
After model forward pass, compute additional losses:
- For Arm B/D: supervised_color_loss using sorted matching (primary) + also log Hungarian
- For Arm C: id_contrastive_loss using sorted matching (primary) + also log Hungarian
- Add these to total_loss before backward()

5. **Evaluation with matching protocol:**
- At each evaluation checkpoint, compute delta_R2_color under BOTH matching schemes
- Compute mismatch rate
- Report both sets of results

6. **Seeds:** [7, 17, 31, 53, 71] — FRESH, disjoint from iter_021-024

7. **Results directory:** archive/iter_025/results/

### Step 3: Run the experiment

```bash
cd /home/user && python src/run_phase0_id_probe.py --sequential
```

Use --sequential for reliability. If it's too slow, use --workers 4.

### Step 4: Analyze results and compile report

After all runs complete, read the results CSV and compute:
- Per-arm mean delta_R2_color (sorted vs Hungarian matching)
- Collapse rates
- Mismatch rates
- Noise floor value
- Compare against thresholds: max(0.10, floor_mean + 0.08)
- Arm A drift check vs iter_022-024 reference
- Assign to the four outcome quadrants
- Save a comprehensive analysis to archive/iter_025/results/analysis.md

## CRITICAL REMINDERS
- Read src/pre_registration.md FIRST
- Fresh seeds [7, 17, 31, 53, 71] — NOT the old seeds
- The JEPA loss remains as the base objective for ALL arms; supervised/contrastive are ADDITIONAL
- supervised_weight ramp 0.1→25.0 over 500 steps if collapse occurs, otherwise 25.0 from start
- d_t=3 frozen, gdasr_log_only=True (M3 preserved)
- Report language: "compatible with sufficient architectural capacity under direct supervision" NOT "demonstrates the architecture can encode identity"
- Any Arm D result alone does NOT confirm H1
- If Arm A drifts > 0.03 from reference, flag it

## EXPECTED OUTPUTS
1. Modified src/models_dual_stream.py with color_probe, contrastive, and matching methods
2. New src/run_phase0_id_probe.py experiment runner
3. archive/iter_025/results/ with all run data, summaries, and analysis