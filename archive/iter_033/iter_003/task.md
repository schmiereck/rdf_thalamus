## Task: Implement and Execute the Iter_033 Three-Condition Oracle Bracket Experiment

Read the pre-registration file at `src/pre_registration.md` first. You MUST adhere to every specification in it. Do NOT deviate from the pre-registered design.

### Overview
Implement `src/run_iter033.py` that runs a 4-condition × 12-seed oracle bracket experiment on the Thalamus project. The 4 conditions are RANDOM, LEARNED-VICReg, LEARNED-SFA, and ORACLE. All share the same environment, seed bank, and CLTSMotorController logic.

### Key Files to Reference
- `src/motor.py` — CLTSMotorController class (you will use this directly)
- `src/environment.py` — PhysicsSandbox class (N=2)
- `src/models_separate_dyn.py` — NonParametricJEPASpatialSeparateDyn model class
- `src/models_dual_stream.py` — DualStreamPredictor, calculate_centroid_and_variance
- `src/run_iter031_partB.py` — Reference implementation for CLTS evaluation (very similar structure, can use as template)
- `archive/iter_029/results/checkpoints/` — Checkpoint files for both arms

### Checkpoint Files
- **LEARNED-VICReg (Arm A):** `archive/iter_029/results/checkpoints/a_vicreg-only_control_seed{N}.pt` where N is each seed
- **LEARNED-SFA (Arm B):** `archive/iter_029/results/checkpoints/b_sfavicreg,_sfa_5.0_seed{N}.pt` where N is each seed

### Seeds
[7, 17, 31, 53, 71, 83, 97, 101, 107, 113, 137, 163]

### Conditions Specification

**1. RANDOM (lower bound):**
- Create a CLTSMotorController
- Each step: override token_locus to a random integer in [0, d_t-1]
- Zero all EMA statistics (mu, sigma) each step BEFORE computing action so no learned structure influences behavior
- Set attention_cooldown to 0 each step (so it re-randomizes next step)
- For centroids: run the encoder on the current observation just to get centroids for tracking-error computation, but ignore the surprise signal entirely. The surprise-driven attention is replaced by random.

Actually, the simplest correct implementation: use CLTSMotorController normally, but AFTER calling get_action(), override the token_locus and reset all EMA. But this is wrong because get_action would still use accumulated EMA. Better approach:

Create the controller. Each step:
1. Reset controller.mu[:] = 0, controller.sigma[:] = 1.0
2. Set controller.attention_cooldown = 0
3. Run encoder forward pass to get z_coord, z_dyn, and predictor to get z_pred_coord, z_pred_dyn, z_target_coord, z_target_dyn
4. Call controller.get_action(model, obs, info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, d_t, centroids) — but this will update EMA internally
5. AFTER get_action, override controller.token_locus = random int in [0, d_t-1]
6. Use this random locus for tracking and all metrics

Wait — the pre-registration says "random token_locus selection each step, EMA zeroed, cooldown=0". The cleanest implementation is:

- Before each step's get_action call, reset controller state:
  controller.mu[:] = 0
  controller.sigma[:] = 1.0  
  controller.attention_cooldown = 0
- Run get_action normally — it will compute surprise, update EMA, select argmax locus
- But since EMA was zeroed, the z-score will just be (s - 0) / (1.0) = s, so argmax of raw surprise
- Then OVERRIDE: token_locus = random int in [0, d_t-1]
- This is NOT purely random. It's random locus but still using surprise for some internal state.

Actually the CORRECT implementation per the pre-registration is:
- "random token_locus selection each step" — the locus is random, NOT surprise-driven
- "EMA zeroed to prevent any learned structure from influencing behavior" — EMA doesn't accumulate
- "attention_cooldown = 0 so locus is re-randomized every step"

So: each step, set token_locus to a random channel. Zero EMA before the step. The surprise values computed inside get_action are discarded (not used for locus selection). The tracking error is still computed against the random locus's centroid.

Implementation:
```python
# RANDOM condition
controller.mu[:] = 0.0
controller.sigma[:] = 1.0
controller.attention_cooldown = 0
random_locus = np.random.randint(0, d_t)
# Run get_action to get action dict and surprises (for logging)
action, locus, surprises = controller.get_action(...)
# Override locus
locus = random_locus
controller.token_locus = random_locus
```

**2. LEARNED-VICReg (Arm A):**
- Load model from `a_vicreg-only_control_seed{N}.pt`
- Build model with: NonParametricJEPASpatialSeparateDyn(d_max=8, h=3, k=4, cooldown=300, stabilization_period=100, pos_encoding="none", primary_objective="jepa", sfa_weight=25.0, gdasr_log_only=True, dyn_readout="mean", sub_features=1, dyn_source="spatial", mask_dyn_sim=True, coord_vicreg=True)
- Set model.d_t = 3
- Create CLTSMotorController (fresh for each seed)
- Run standard evaluation loop

**3. LEARNED-SFA (Arm B):**
- Load model from `b_sfavicreg,_sfa_5.0_seed{N}.pt`
- Same model config as Arm A
- Set model.d_t = 3
- Same evaluation loop

**4. ORACLE (upper bound):**
This is the critical new condition. Do NOT use the encoder or predictor. Instead:

- Create CLTSMotorController (fresh for each seed)
- Maintain previous positions and velocities: `prev_positions`, `prev_velocities`
- Each step:
  1. Get ground-truth info from environment: positions = info['positions'], velocities = info['velocities'], colors = info['colors']
  2. Construct z_coord tensor: shape (1, d_max), first d_t channels = positions (normalized to pixel space 0-128), remaining channels = 0
  3. Construct z_dyn tensor: shape (1, d_max), first d_t channels = mean of each object's RGB color, remaining channels = 0
  4. Construct z_pred_coord: shape (1, d_max), linear extrapolation = prev_positions + prev_velocities * dt (dt=1.0). First step: z_pred_coord = z_coord.
  5. Construct z_pred_dyn = z_dyn (identity is constant)
  6. Construct centroids = z_coord (for the controller's tracking)
  7. Call controller.get_action(model=None, obs=obs, info=info, z_pred_coord, z_target_coord=z_coord, z_pred_dyn, z_target_dyn=z_dyn, d_t=3, centroids)

Wait — get_action signature is:
```python
def get_action(self, model, obs, info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, d_t, centroids):
```

For the ORACLE condition, we don't have a model. We need to adapt. The simplest approach: create a dummy model object that has the necessary attributes (or pass None and handle it in the controller call). Actually, looking at the CLTSMotorController code, it doesn't use `model` at all! It only uses z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, d_t, and centroids. The `model` parameter is not used inside get_action.

So for ORACLE: we can pass model=None (it won't be used) and just construct the z tensors correctly.

IMPORTANT: The ORACLE controller uses the SAME EMA, SAME cooldown, SAME push logic as LEARNED. The only difference is the INPUT (ground-truth vs. learned).

### Evaluation Loop (per run)
```python
env = PhysicsSandbox(N=2, seed=seed)
controller = CLTSMotorController()
# ... condition-specific setup ...

obs = env.reset()
history = collections.deque(maxlen=4)
history.append(obs)

# Warmup: 3 steps of zero action
prev_velocities = env.velocities.copy()
for _ in range(3):
    obs, info = env.step({"acc": 0.0, "push": False})
    history.append(obs)
    prev_velocities = info["velocities"].copy()

for step in range(2000):
    # Mass perturbation at step 1000
    if step == 1000:
        env.masses[0] *= 1.5
    
    # ... condition-specific z_coord, z_dyn, z_pred_coord, z_pred_dyn construction ...
    
    # Call controller.get_action(...)
    action, locus, surprises = controller.get_action(...)
    
    # Override for RANDOM condition
    if condition == "random":
        random_locus = np.random.randint(0, d_t)
        locus = random_locus
        controller.token_locus = random_locus
    
    # Take action
    obs, info = env.step(action)
    history.append(obs)
    
    # Compute tracking error, collision detection, etc.
    # ...
```

### Metrics to Compute Per Run
1. **Post-collision attention selectivity (Version B)** — PRIMARY metric
   - For each collision event, within POST_COLLISION_WINDOW=15 steps, fraction where attended object = max-velocity-change object
   - Collision detection: COLLISION_DIST_THRESHOLD=4.0, COLLISION_VELOCITY_CHANGE_THRESHOLD=1.0
   
2. **Mean tracking error** — SECONDARY
   - Mean |pointer_pos - centroid of attended channel| across all steps
   
3. **Perturbation selectivity** — SECONDARY
   - Fraction of steps 1000-1099 where attended object = object 0

4. **Surprise distributions** (for characterization)
   - Per-channel mean and std of surprise values across all steps
   - err_coord and err_dyn separately (decomposition)

### Channel-to-Object Mapping
```python
def get_channel_to_obj_mapping(centroids, positions, d_t):
    mapping = {}
    centroid_vals = centroids[0, :d_t].cpu().numpy() if torch.is_tensor(centroids) else centroids[0, :d_t]
    for ch in range(d_t):
        val = centroid_vals[ch]
        closest_obj = int(np.argmin(np.abs(positions - val)))
        mapping[ch] = closest_obj
    return mapping
```

### ORACLE Centroid Construction
For the ORACLE condition, centroids should be constructed from ground-truth positions:
```python
centroids = torch.zeros(1, 8)  # d_max=8
centroids[0, :d_t] = torch.tensor(info['positions'][:d_t], dtype=torch.float32)
```

### Model Construction for LEARNED Conditions
```python
def build_model():
    model = NonParametricJEPASpatialSeparateDyn(
        d_max=8, h=3, k=4, cooldown=300, stabilization_period=100,
        pos_encoding="none", primary_objective="jepa",
        sfa_weight=25.0, gdasr_log_only=True,
        dyn_readout="mean", sub_features=1, dyn_source="spatial",
        mask_dyn_sim=True, coord_vicreg=True,
    )
    model.d_t = 3
    return model
```

### Output
1. Write per-run results to `archive/iter_033/results/per_run.csv`
2. Write summary statistics to `archive/iter_033/results/summary.csv`
3. Write full analysis to `archive/iter_033/results/analysis.md`

The analysis MUST include:
- Raw triple (RANDOM, LEARNED-VICReg, LEARNED-SFA, ORACLE) with mean +/- std for ALL three metrics
- Ordering sanity check: confirm ORACLE >= RANDOM on primary metric
- Branch (c) check: |ORACLE - RANDOM| < 0.10?
- g_vr and g_sfa with bootstrapped 95% CI (10000 resamples of seeds)
- Branch assignment per the pre-registered decision rule
- Surprise distributions per condition (mean, std)
- err_coord vs err_dyn decomposition for LEARNED conditions
- Per-seed table for primary metric
- Restrained language throughout

### Bootstrap CI Implementation
```python
def bootstrap_g(learned_vals, random_vals, oracle_vals, n_bootstrap=10000):
    n = len(learned_vals)
    gs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        l = np.mean(learned_vals[idx])
        r = np.mean(random_vals[idx])
        o = np.mean(oracle_vals[idx])
        denom = o - r
        if abs(denom) < 1e-10:
            gs.append(np.nan)
        else:
            gs.append((l - r) / denom)
    gs = np.array(gs)
    gs = gs[~np.isnan(gs)]
    ci_lower = np.percentile(gs, 2.5)
    ci_upper = np.percentile(gs, 97.5)
    return np.nanmean(gs), ci_lower, ci_upper
```

### Important Implementation Notes
- d_t = 3 for ALL conditions (per pre-registration)
- For the LEARNED conditions, the model was trained with d_t=3 but with N=3 objects. When evaluating with N=2, the model will only track 2 of its 3 channels — this is expected. The third channel will have some activation but correspond to no physical object.
- The LEARNED condition forward pass requires x_hist of shape (1, H=3, 3, 128) and x_target of shape (1, 3, 128). Build these from the history deque.
- For the RANDOM condition, you still need to run the encoder to get centroids (for tracking error computation and channel-to-object mapping). But the locus selection is random.
- The ORACLE condition does NOT run the encoder at all. It constructs z_coord from ground truth.
- Use torch.no_grad() for all model forward passes.
- Use device="cpu" and torch.set_num_threads(4) for efficiency.
- Environment: PhysicsSandbox(N=2, seed=seed). No pixel noise, no Noisy-TV, no structured distractor.

### CRITICAL: Pre-Registration Adherence
Before running ANY seed, confirm that `src/pre_registration.md` exists and has been read. The experiment MUST match every specification in that file. Key checks:
- 4 conditions (not 3): RANDOM, LEARNED-VICReg, LEARNED-SFA, ORACLE
- d_t=3 (not d_t=2)
- 12 seeds including 53 and 71
- Same CLTSMotorController code across all conditions
- Ordering sanity check before computing g
- Branch (c) check with |ORACLE - RANDOM| < 0.10
- Bootstrap CI with 10000 resamples
- Report raw triple with CIs
- Surprise decomposition (err_coord vs err_dyn)

Execute the script after writing it. Save all outputs to `archive/iter_033/results/`. Print a summary of the results at the end.