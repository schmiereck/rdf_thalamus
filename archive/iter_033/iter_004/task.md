Write and execute the iter_033 oracle bracket experiment. Create the file `src/run_iter033.py` with the complete experiment code, then run it.

IMPORTANT: Read `src/pre_registration.md` first for the exact experiment specification. Also read `src/motor.py` (the CLTSMotorController class), `src/environment.py` (PhysicsSandbox), `src/models_separate_dyn.py` (the model class), and `src/run_iter031_partB.py` (reference implementation). These are the ONLY files you need to read. Do NOT read any other files.

The experiment has 4 conditions × 12 seeds = 48 runs, each 2000 evaluation steps.

### Conditions:
1. **RANDOM**: CLTSMotorController with random locus each step (zero EMA, cooldown=0 before each action, then override token_locus to random)
2. **LEARNED-VICReg**: Load iter_029 Arm A checkpoints, run encoder+predictor, feed to CLTSMotorController
3. **LEARNED-SFA**: Load iter_029 Arm B checkpoints, run encoder+predictor, feed to CLTSMotorController
4. **ORACLE**: Feed ground-truth positions/colors as z_coord/z_dyn, linear-extrapolation positions as z_pred_coord, z_dyn as z_pred_dyn, to the SAME CLTSMotorController

### Key Specs:
- Seeds: [7, 17, 31, 53, 71, 83, 97, 101, 107, 113, 137, 163]
- d_t=3 (frozen), d_max=8
- N=2 objects, 2000 steps, mass perturbation at step 1000 (1.5x on object 0)
- Primary metric: post-collision attention selectivity (version B) — fraction of post-collision-window steps where attended object matches max-velocity-change object
- Secondary: mean tracking error, perturbation selectivity (steps 1000-1099, attended=obj0)
- Collision detection: COLLISION_DIST_THRESHOLD=4.0, COLLISION_VELOCITY_CHANGE_THRESHOLD=1.0, POST_COLLISION_WINDOW=15
- Channel-to-object mapping: closest-centroid (centroids[0,c] vs info['positions'])
- Bootstrap CI: 10000 resamples of seeds for g ratio

### Model Construction (for LEARNED conditions):
```python
model = NonParametricJEPASpatialSeparateDyn(
    d_max=8, h=3, k=4, cooldown=300, stabilization_period=100,
    pos_encoding="none", primary_objective="jepa",
    sfa_weight=25.0, gdasr_log_only=True,
    dyn_readout="mean", sub_features=1, dyn_source="spatial",
    mask_dyn_sim=True, coord_vicreg=True,
)
model.d_t = 3
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval()
```

### Checkpoint Paths:
- Arm A: `archive/iter_029/results/checkpoints/a_vicreg-only_control_seed{N}.pt`
- Arm B: `archive/iter_029/results/checkpoints/b_sfavicreg,_sfa_5.0_seed{N}.pt`

### ORACLE Condition Implementation:
Maintain `prev_positions` and `prev_velocities` arrays (initialized from env after warmup).

Each step:
```python
# Ground-truth tensors
positions = info['positions']  # shape (2,)
velocities = info['velocities']  # shape (2,)
colors = info['colors']  # shape (2, 3)

z_coord = torch.zeros(1, 8)
z_coord[0, :d_t] = torch.tensor(positions[:d_t], dtype=torch.float32)

z_dyn = torch.zeros(1, 8)
for i in range(d_t):
    z_dyn[0, i] = torch.tensor(np.mean(colors[i]), dtype=torch.float32)

z_pred_coord = torch.zeros(1, 8)
z_pred_coord[0, :d_t] = torch.tensor(prev_positions[:d_t] + prev_velocities[:d_t] * 1.0, dtype=torch.float32)

z_pred_dyn = z_dyn.clone()  # identity is constant

centroids = z_coord.clone()  # ground-truth positions as centroids
```

Then call controller.get_action(None, obs, info, z_pred_coord, z_coord, z_pred_dyn, z_dyn, d_t, centroids).
Note: the `model` param in get_action is unused by CLTSMotorController.

### RANDOM Condition Implementation:
Each step, BEFORE calling get_action:
```python
controller.mu[:] = 0.0
controller.sigma[:] = 1.0
controller.attention_cooldown = 0
```
Then run the encoder+predictor normally (to get z tensors and centroids for tracking), call get_action, but AFTER the call override:
```python
random_locus = np.random.randint(0, d_t)
locus = random_locus
controller.token_locus = random_locus
```

### LEARNED Condition Forward Pass:
```python
x_hist = torch.from_numpy(np.stack(list(history)[:3], axis=0)).float().unsqueeze(0)  # (1, 3, 3, 128)
x_target = torch.from_numpy(history[3]).float().unsqueeze(0)  # (1, 3, 128)
with torch.no_grad():
    z_coord, z_dyn = model.encoder(x_target)
    centroids = z_coord
    loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(x_hist, x_target, d_t_predict=min(d_t, model.d_max))
```

### Output:
1. `archive/iter_033/results/per_run.csv` — one row per run with all metrics
2. `archive/iter_033/results/summary.csv` — condition-level summary
3. `archive/iter_033/results/analysis.md` — full analysis including:
   - Raw triple (RANDOM, LEARNED-VICReg, LEARNED-SFA, ORACLE) means +/- stds
   - Ordering sanity check
   - Branch (c) check: |ORACLE - RANDOM| < 0.10?
   - g_vr and g_sfa with bootstrap 95% CI
   - Branch assignment per decision rule
   - Surprise decomposition
   - Per-seed primary metric table

### Bootstrap CI:
```python
def bootstrap_g(learned_vals, random_vals, oracle_vals, n_bootstrap=10000):
    n = len(learned_vals)
    rng = np.random.RandomState(42)
    gs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        l = np.mean(learned_vals[idx])
        r = np.mean(random_vals[idx])
        o = np.mean(oracle_vals[idx])
        denom = o - r
        if abs(denom) < 1e-10:
            gs.append(np.nan)
        else:
            gs.append((l - r) / denom)
    gs = np.array(gs)
    gs_valid = gs[~np.isnan(gs)]
    return np.nanmean(gs), np.percentile(gs_valid, 2.5), np.percentile(gs_valid, 97.5)
```

Write the complete script, run it, and produce all output files. Print a summary of key results at the end.