Create and run a script `src/run_iter031_partB.py` to calibrate the CLTS evaluation protocol on a collision-sparse N=2 environment.

Follow these specifications precisely:

### Step 1: Model Loading
- Checkpoints exist at: `archive/iter_029/results/checkpoints/a_vicreg-only_control_seed{}.pt` for seeds `[7, 31, 97, 113, 137]`.
- Load them using the model architecture: `NonParametricJEPASpatialSeparateDyn` from `src/models_separate_dyn.py`.
- Instantiate the model with:
  ```python
  model = NonParametricJEPASpatialSeparateDyn(
      d_max=8, h=3, k=4, cooldown=300, stabilization_period=100,
      pos_encoding="none", primary_objective="jepa",
      sfa_weight=25.0, gdasr_log_only=True,
      dyn_readout="mean", sub_features=1, dyn_source="spatial",
      mask_dyn_sim=True, coord_vicreg=True,
  )
  ```
- Set `model.d_t = 3` (matching the checkpoint).
- Use `torch.device("cpu")` for execution.

### Step 2: Environment Settings & Mass Perturbation
- Use `PhysicsSandbox(N=2, seed=seed)` for each seed.
- Run for exactly 2000 steps total (step indices 0 to 1999).
- Warm up the environment first by pre-filling a deque of length 4 with observations, taking 3 steps of `{"acc": 0.0, "push": False}` before starting the 2000 evaluation steps loop.
- At step 1000 of the 2000 steps loop: multiply the mass of Object 0 by 1.5× (`env.masses[0] *= 1.5`).

### Step 3: Run Conditions
For each seed in `[7, 31, 97, 113, 137]`, run 3 conditions:
1. **surprise-driven**: Standard surrogate-readout attention using CLTSMotorController.
2. **frozen**: Fixed attention locus = 0.
3. **random**: At each step, choose attention locus uniformly at random from the active channels `0` to `d_t - 1`.

To implement the locus override for `frozen` and `random` conditions inside the loop, use the logic:
```python
# Before controller.get_action:
if condition == "frozen":
    controller.token_locus = 0
    controller.attention_cooldown = controller.attention_cooldown_max
elif condition == "random":
    controller.token_locus = int(np.random.randint(0, d_t))
    controller.attention_cooldown = controller.attention_cooldown_max

# Call controller.get_action:
action, locus, surprises = controller.get_action(
    model, history[3], info,
    z_pred_coord, z_target_coord,
    z_pred_dyn, z_target_dyn,
    d_t, centroids,
)
```

We will run evaluations for two distinct settings of `d_t`:
- **Setting A**: `d_t = 2` (Evaluating channels 0 and 1, mapping precisely to the 2 physical objects).
- **Setting B**: `d_t = 3` (Evaluating channels 0, 1, and 2, matching the model's training dimensionality).
Run the full 5 seeds × 3 conditions sweep for BOTH `d_t = 2` and `d_t = 3` to find the most calibrated and discriminative protocol setting.

### Step 4: Robust Channel Mapping
Create a robust channel-to-object mapping function to handle mismatch between `d_t` and `N`:
```python
def get_channel_to_obj_mapping_robust(centroids, positions, d_t):
    mapping = {}
    centroid_vals = centroids[0, :d_t].cpu().numpy()
    for ch in range(d_t):
        val = centroid_vals[ch]
        # Map channel to the physical object closest to its centroid
        closest_obj = int(np.argmin(np.abs(positions - val)))
        mapping[ch] = closest_obj
    return mapping
```

### Step 5: Metrics and Post-Collision Analysis
Collect the following metrics for each of the 15 runs per setting of `d_t`:
1. **Tracking Error**: Mean absolute error between `info["pointer_pos"]` and `centroids[0, locus].item()`.
2. **Collision Count per 100 steps**: Count how many object-object collision events occur and divide by 20.
   Detect collisions when `abs(info["positions"][0] - info["positions"][1]) < (info["radii"][0] + info["radii"][1] + 4.0)` AND the velocity change of any object is > 1.0 compared to the previous step.
   Record:
   - `coll_step`: the step at which collision occurred
   - `max_change_obj`: the object index (0 or 1) that had the larger absolute velocity change.
3. **Collision Attention Selectivity**:
   A step `s` is post-collision if `1 <= s - t <= 15` for any collision step `t`.
   Compute:
   - **Version A (Any colliding object)**: Fraction of post-collision steps where the attended object is in `[0, 1]` (i.e. is any colliding object).
   - **Version B (Max velocity-change object)**: Fraction of post-collision steps where the attended object matches the `max_change_obj` of the most recent collision.
4. **Perturbation Attention Selectivity**: Fraction of steps in the 100-step window post-perturbation (steps 1000-1099) where the changed object (Object 0) is attended.

### Step 6: Apply Gates and Write Results
- Create `archive/iter_031/results/` directory.
- Apply pre-registered gates G1-G3 using the measured mean and std of the `random` condition as baselines:
  - **G1_tracking**: surprise-driven tracking error ≤ random_tracking_mean − 1*random_tracking_std
  - **G2_collision_selectivity (Version B)**: surprise-driven collision selectivity (Version B) ≥ random_selectivity_max × 1.5
  - **G3_perturbation_selectivity**: surprise-driven perturbation selectivity ≥ random_selectivity × 1.5
- Write all detailed results to `archive/iter_031/results/partB_per_seed.csv` and `archive/iter_031/results/partB_summary.csv`.
- Write a clear, comprehensive scientific analysis report to `archive/iter_031/results/partB_analysis.md`, explicitly detailing:
  - The measured frozen and random baselines (mean ± std)
  - Surprise-driven performance for both `d_t = 2` and `d_t = 3` settings
  - Gate evaluation results and final protocol recommendation.

Write, run, and complete the script now. Do not hesitate to print rich logs.