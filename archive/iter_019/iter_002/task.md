## Phase 19 Experiment: ITAG (Input-Level Temporal AutoCorrelation Gating) with Structured Distractor

### CRITICAL: Read Pre-Registration First
Before writing ANY code, read `src/pre_registration.md` and strictly adhere to the pre-registered hypothesis, falsification criteria, and experimental protocol. Do NOT deviate from the pre-registered plan.

### Context
This is Phase 19 of the Thalamus project. Phases 16-18 all failed to solve the dimension recruitment gating problem:
- WUP-MDL (Phase 16): 100% genuine recruitment, 100% false recruitment on Noisy-TV
- ESUG (Phase 17): Encoder cold-start caused rejection of genuine objects
- EG-MDL (Phase 18): Predictor cold-start optimization transient rendered ρ useless

ITAG is the fourth attempt. The Manager has explicitly noted that ITAG trivially succeeds on Noisy-TV (by definition of white noise), so the real test is a STRUCTURED DISTRACTOR that has high temporal autocorrelation but is not a physics object.

### Task: Implement and Run Full Phase 19 Experiments

You need to create 3 files and modify 1 file, then run the experiments.

#### 1. CREATE src/itag.py
Implement ITAG (Input-level Temporal AutoCorrelation Gating) and ISAG (Input-level Spatial AutoCorrelation Gating):

```python
import numpy as np
import torch

def identify_surprising_positions(prediction_error_map, top_k=16):
    """
    Identify top-K spatial positions with highest prediction error.
    Args:
        prediction_error_map: (B, 128) or (128,) - per-position prediction error from the encoder
        top_k: number of positions to select
    Returns:
        positions: list of indices into the 128-pixel array
    """
    # If batched, take mean across batch
    if prediction_error_map.dim() == 2:
        prediction_error_map = prediction_error_map.mean(dim=0)  # (128,)
    elif prediction_error_map.dim() == 3:
        # (B, C, 128) -> take norm across C, mean across B
        prediction_error_map = prediction_error_map.norm(dim=1).mean(dim=0)  # (128,)
    
    _, top_indices = torch.topk(prediction_error_map, min(top_k, prediction_error_map.shape[0]))
    return top_indices.sort()[0].tolist()

def compute_itag(pixel_buffer, surprising_positions, window=20):
    """
    Compute ITAG score: mean lag-1 temporal autocorrelation of raw pixel values 
    at surprising positions over a sliding window.
    
    Args:
        pixel_buffer: list of length W_t, each element is (3, 128) numpy array
        surprising_positions: list of spatial indices
        window: number of timesteps to use (uses last `window` from buffer)
    Returns:
        itag_score: float in [-1, 1]
    """
    if len(pixel_buffer) < 2 or len(surprising_positions) == 0:
        return 0.0
    
    # Use last `window` timesteps
    buffer_slice = pixel_buffer[-window:]
    if len(buffer_slice) < 2:
        return 0.0
    
    # Stack into array: (T, 3, 128)
    pixel_array = np.stack(buffer_slice, axis=0)
    T = pixel_array.shape[0]
    
    # Compute per-position temporal autocorrelation
    autocorrs = []
    for pos in surprising_positions:
        if pos < 0 or pos >= 128:
            continue
        # Get pixel intensity at this position across time (sum across RGB channels for robustness)
        # Shape: (T,)
        pixel_vals = pixel_array[:, :, pos].sum(axis=1)  # (T,)
        
        # Compute lag-1 autocorrelation
        if T < 2:
            autocorrs.append(0.0)
            continue
        
        x_t = pixel_vals[:-1]  # (T-1,)
        x_t1 = pixel_vals[1:]  # (T-1,)
        
        # Pearson correlation between x_t and x_{t+1}
        std_t = np.std(x_t)
        std_t1 = np.std(x_t1)
        
        if std_t < 1e-8 or std_t1 < 1e-8:
            # Constant signal: high autocorrelation
            autocorrs.append(1.0 if np.mean(np.abs(x_t - x_t[0])) < 1e-6 else 0.0)
            continue
        
        corr = np.corrcoef(x_t, x_t1)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        autocorrs.append(corr)
    
    if len(autocorrs) == 0:
        return 0.0
    
    return float(np.mean(autocorrs))

def compute_isag(pixel_buffer, surprising_positions):
    """
    Compute ISAG score: mean lag-1 spatial autocorrelation of raw pixel values 
    at adjacent surprising positions, computed per frame.
    
    Args:
        pixel_buffer: list of (3, 128) numpy arrays (single frame used from last)
        surprising_positions: list of spatial indices (sorted)
    Returns:
        isag_score: float in [-1, 1]
    """
    if len(pixel_buffer) == 0 or len(surprising_positions) < 2:
        return 0.0
    
    # Use last frame
    frame = pixel_buffer[-1]  # (3, 128)
    
    # Sort positions
    sorted_pos = sorted(surprising_positions)
    
    # Compute spatial autocorrelation between adjacent surprising positions
    spatial_corrs = []
    for i in range(len(sorted_pos) - 1):
        pos_a = sorted_pos[i]
        pos_b = sorted_pos[i + 1]
        
        if pos_a >= 128 or pos_b >= 128:
            continue
        
        # Pixel intensity (sum across RGB)
        val_a = frame[:, pos_a].sum()
        val_b = frame[:, pos_b].sum()
        
        # For spatial autocorrelation we need multiple adjacent pairs
        # Since we only have one frame, compute correlation across the spatial positions
        # using a simpler approach: check if adjacent surprising positions have similar intensity
        # This is a proxy for spatial smoothness
        spatial_corrs.append(val_a * val_b)  # unnormalized similarity
    
    if len(spatial_corrs) == 0:
        return 0.0
    
    # Normalize: compare actual similarity to expected if positions were independent
    all_vals = [frame[:, p].sum() for p in sorted_pos if p < 128]
    if len(all_vals) == 0:
        return 0.0
    
    mean_val = np.mean(all_vals)
    var_val = np.var(all_vals)
    
    if var_val < 1e-8:
        return 1.0  # All positions same intensity → perfectly smooth
    
    # Spatial autocorrelation (Moran's I style)
    n = len(sorted_pos)
    w_sum = n - 1  # number of adjacent pairs
    numerator = sum((all_vals[i] - mean_val) * (all_vals[i+1] - mean_val) 
                    for i in range(len(all_vals)-1))
    denominator = var_val * w_sum
    
    if abs(denominator) < 1e-8:
        return 0.0
    
    isag = numerator / denominator
    return float(np.clip(isag, -1.0, 1.0))
```

#### 2. MODIFY src/environment.py
Add a `structured_distractor` parameter to PhysicsSandbox and implement a Sinusoidal Oscillator entity.

Add `structured_distractor=False` parameter to `__init__`. When `structured_distractor=True`:
- Add state variables: `self.sd_center`, `self.sd_amplitude`, `self.sd_omega`, `self.sd_phase`, `self.sd_color`, `self.sd_radius`, `self.sd_t` (timestep counter)
- Initialize in `reset()`: randomly initialize all parameters as specified in pre-registration
- Update in `step()`: increment `self.sd_t`, compute position as `center + amplitude * sin(omega * t + phase)`
- Render the structured distractor as a soft-edged circle (same rendering as other objects) — add it to the render list alongside objects and pointer, but AFTER them (so it appears on top)
- In `info` dict, add: `sd_pos`, `sd_color`, `sd_radius`
- The structured distractor does NOT participate in collision physics — it passes through other objects

#### 3. CREATE src/run_phase19_experiments.py
This is the main experiment runner. It should closely follow the structure of `src/run_phase18_experiments.py` but with these modifications:

**Experimental Arms (5 seeds each):**
- Arm A: WUP-MDL (W=100), no ITAG pre-filter — identical to Phase 18 Arm P
- Arm B: ITAG pre-filter (τ=0.3, W_t=20) + WUP-MDL (W=100)
- Arm C: ITAG-only gating (τ=0.3, W_t=20), no WUP — immediate recruitment if ITAG > τ

**Three Sweeps per Arm (3 × 5 × 3 = 45 branches):**
1. Transition sweep: N=3→4 (genuine 4th physics object)
2. Noisy-TV control sweep: N=3 + Noisy-TV
3. Structured Distractor control sweep: N=3 + Sinusoidal Oscillator

**ITAG Logic in run_active_branch():**
When the model detects that prediction error has exceeded the recruitment threshold (at the proposal check step, currently step 1800), BEFORE initiating WUP:

For Arms B and C:
1. Compute the per-position prediction error map from the pre-trained encoder
   - Use `model.encoder.forward_spatial(target_t)` to get a_spatial of shape (B, d_max, 128)
   - Compute per-position error as the L2 norm across d_max dimensions: (B, 128)
2. Identify top-K=16 surprising positions using `identify_surprising_positions()`
3. Collect the last W_t=20 raw pixel frames from `branch_history` at those positions
4. Compute ITAG score using `compute_itag()`
5. For Arm B: if ITAG > τ=0.3, proceed with WUP-MDL; otherwise, reject
6. For Arm C: if ITAG > τ=0.3, immediately recruit (set d_t=4); otherwise, reject

**ITAG Score Logging:**
Log ITAG and ISAG scores per timestep during the evaluation window (the 20 steps before the proposal check). This allows computing Cohen's d between distributions.

**Metrics to collect:**
- All metrics from Phase 18 (recruitment rate, MSE, sim loss, etc.)
- ITAG score distributions (per timestep, per seed, per condition)
- ISAG score distributions
- Cohen's d between: (genuine vs Noisy-TV), (genuine vs structured distractor)
- ROC analysis
- For each arm: false recruitment rate on each distractor type

**Falsification Audit:**
At the end of the script, compute:
- C1: Cohen's d between ITAG distributions for genuine vs Noisy-TV
- C2: Cohen's d between ITAG distributions for genuine vs structured distractor
- C3: Gating performance on Noisy-TV (false recruitment ≤ 20%, genuine ≥ 80%)
- C4: Gating performance on structured distractor (false recruitment ≤ 20%, genuine ≥ 80%)
- C5: Scope-reduction trigger evaluation

**Important Implementation Details:**
- Use `from src.itag import identify_surprising_positions, compute_itag, compute_isag`
- Use `from src.models_dual_stream import NonParametricJEPASpatial`
- Use `from src.motor import CLTSMotorController`
- Use `from src.environment import PhysicsSandbox`
- Follow the same training protocol as Phase 18: passive training on N=3 for 1500 steps (with caching), then active CLTS training for 1500 steps
- The structured distractor environment should use `PhysicsSandbox(N=3, seed=..., structured_distractor=True)`
- Seeds: [42, 123, 456, 789, 999]
- Save all results to `archive/iter_019/results/`

**Key: The ITAG computation must happen BEFORE the WUP period, using raw pixel data from the environment's history, NOT from the model's latent representations. This is what makes it immune to cold-start pathologies.**

#### 4. Run the experiments
Execute: `python src/run_phase19_experiments.py`

Save results to `archive/iter_019/results/`. Expected runtime: ~15-30 minutes depending on hardware.

### Expected Outcomes (from pre-registration)
- ITAG trivially separates genuine objects from Noisy-TV (Cohen's d >> 1.5)
- ITAG fails to separate genuine objects from structured distractors (Cohen's d < 1.5)
- Arm B will successfully filter Noisy-TV but will falsely recruit structured distractors
- Arm C will show similar behavior
- The scope-reduction trigger C5 will be activated

### Success Criterion
All 45 experiment branches complete, ITAG distributions are collected, falsification audit is performed, and results are saved to archive/iter_019/results/.
