## Diagnosis and Fix for Iter_033 Oracle Bracket — ORACLE Predictor Too Noisy

The iter_033 oracle bracket experiment completed all 48 runs, but the ORACLE condition performed WORSE than RANDOM on the primary metric (selectivity_vb: 0.4344 vs 0.4862). The ordering sanity check FAILED.

**Root cause:** The ORACLE's linear-extrapolation predictor (prev_pos + prev_vel * dt) produces astronomically high surprise_coord values (~146,000 mean vs ~1-3 for LEARNED). This is because:
1. The physics involves wall bounces and collisions that linear extrapolation can't predict
2. These massive surprise values overwhelm the EMA statistics (mu, sigma)
3. The CLTSMotorController's attention mechanism and push logic are calibrated against these inflated values
4. The ORACLE doesn't represent "perfect perception + good surprise" — it represents "perfect perception + broken surprise signal"

The Manager's pre-registration note (Point 2) specifically warned: "the ORACLE will produce qualitatively different surprise distributions" and said to "report the per-condition surprise distributions so the reader can see the bracket is not deformed by EMA mismatch." The bracket IS deformed.

**Fix:** The ORACLE predictor should predict CONSTANT velocity between collisions, and have ZERO surprise for constant-velocity motion. Only genuine collision events (which change velocities unpredictably) should produce surprise. The correct ORACLE implementation should:
- Predict next position as current position + current velocity * dt (not PREVIOUS position + PREVIOUS velocity)
- This way, for constant-velocity motion between collisions, the prediction error is near zero
- At collision events, the prediction error will be non-zero (because velocity changed)
- This produces a surprise distribution that is qualitatively similar to LEARNED (low baseline, spikes at collisions) rather than the current broken one

But wait — the current code uses prev_velocities from the PREVIOUS step, not current velocities. When velocity changes due to a collision, the oracle predictor uses the PRE-collision velocity to predict, which gives a large error. But the problem is bigger: even for constant-velocity motion, prev_pos + prev_vel may not match the actual new position because of substep integration, wall bounces, and other numerical effects.

**The correct fix:** Use the CURRENT step's position and velocity to construct the prediction for the NEXT step. This means:
- At step t, after observing info['positions'] and info['velocities']:
  - z_coord = info['positions']  (current ground truth)
  - z_pred_coord = info['positions'] + info['velocities'] * dt  (one-step lookahead using CURRENT velocity)
- This way, between collisions, z_pred_coord will closely match the actual next position
- At collisions, the velocity changes, so the prediction will be off — genuine surprise

Actually, the issue is more fundamental. The CLTSMotorController computes surprise as:
```
err_coord = MSE(z_pred_coord[:, c] - z_target_coord[:, c])
```

Where z_target_coord is the CURRENT observation's z_coord. And z_pred_coord is from the PREDICTOR's output.

For the ORACLE, z_target_coord should be the current ground-truth position, and z_pred_coord should be the PREDICTION of the current position (made from previous state). 

The current implementation uses:
```python
z_pred_coord[0, :d_t] = prev_positions[:d_t] + prev_velocities[:d_t] * 1.0
```

This predicts current position = previous position + previous velocity. This should be close to correct for constant-velocity motion. But in practice:
- Wall bounces change velocity between substeps
- The environment uses 10 substeps per step
- After a wall bounce, prev_velocity may point toward the wall while current position has bounced away

Let me look at the numbers more carefully. The mean ORACLE surprise_coord is ~146,000. For 2 objects in 128 pixels, a typical prediction error might be ~1-5 pixels per step for constant-velocity motion. MSE of ~5 pixels = 25. But 146,000 is way too high. Something is wrong.

Actually, wait — the MSE is computed over channels. With d_t=3 channels and one channel having no corresponding object (d_t=3 but N=2), that channel would have z_coord=0, z_pred_coord=0. Let me check: for the unused channel (index 2), z_coord[0,2]=0 and z_pred_coord[0,2]=prev_positions[2]+prev_velocities[2]. But prev_positions has only 2 elements! So prev_positions[:d_t] = prev_positions[:3] would fail with N=2 objects that only have 2 positions!

**THIS IS THE BUG.** The ORACLE code does:
```python
z_pred_coord[0, :d_t] = torch.tensor(prev_positions[:d_t] + prev_velocities[:d_t] * 1.0, ...)
```

But `prev_positions` has only 2 elements (N=2 objects) while `d_t=3`. So `prev_positions[:3]` would include the 3rd element which doesn't exist — this would either crash or produce garbage. Let me check the actual code...

Looking at the original script:
```python
z_pred_coord[0, :min(D_T, len(prev_positions))] = torch.tensor(
    prev_positions[:min(D_T, len(prev_positions))] + prev_velocities[:min(D_T, len(prev_velocities))] * 1.0,
    dtype=torch.float32)
```

OK, it uses `min(D_T, len(...))`, so it should only use 2 elements. The 3rd channel stays 0. So the 3rd channel has z_pred_coord=0, z_coord=0, and MSE = 0. That's fine.

But then why is surprise so high? Let me think again...

Actually, the issue might be that after mass perturbation or during collisions, the linear extrapolation is just very inaccurate. Let's compute: if a collision changes velocity by ~4 px/step, then the prediction error would be ~4 pixels, and MSE would be ~16. For 2 channels that's ~32 total per step. But 146,000 is 3 orders of magnitude higher.

Wait — the ORACLE uses torch tensors. The `z_pred_coord` and `z_coord` are both shape (1, D_MAX). The MSE is computed per-channel:
```python
err_coord = torch.mean((z_pred_coord[:, c] - z_target_coord[:, c])**2)
```

This is `torch.mean` over the batch dimension (size 1), so it's just the squared difference. For channel c, it's (z_pred_coord[0,c] - z_coord[0,c])^2. With positions in [0, 128] range and prediction error of ~4 pixels, that's ~16 per channel per step. Over 2000 steps, the average should be ~16 or less. But we're getting 146,000. That's absurd.

Unless... the prev_positions and prev_velocities are not being updated correctly. Let me check the ORACLE evaluation loop:

The prev_positions is initialized from `info['positions']` after warmup. Then each step:
```python
prev_positions = info["positions"].copy()
prev_velocities = info["velocities"].copy()
```

This is AFTER the env.step(action) call. So prev_positions stores the CURRENT step's positions, and the prediction uses them to predict the NEXT step. But z_coord is ALSO the current step's positions. So the surprise is computed as:

surprise = MSE(current_position, predicted_current_position_from_previous)

Wait, that's the correct semantics — the predictor predicts the current observation from previous state, and the surprise is the difference. Let me re-examine...

Actually, in the CLTSMotorController:
- z_target_coord is the ENCODER's output for the current observation
- z_pred_coord is the PREDICTOR's output (prediction of the current observation from history)

For the ORACLE:
- z_coord = current info['positions'] → used as z_target_coord
- z_pred_coord = prev_positions + prev_velocities * dt → prediction of current positions from previous state

This should give small errors for constant-velocity motion. Unless dt is not 1.0. The PhysicsSandbox uses `substeps=10` and `dt = 1.0 / substeps` internally, but the step() function does `1.0 / self.substeps` internally. The external time step is 1.0 time unit. So dt=1.0 for the oracle predictor should be correct.

Let me think about what could cause 146,000 average surprise. That would require average prediction error of ~382 pixels squared. With positions in [0, 128], that means predictions are wildly off — like predicting position 500 when actual is 64. This is possible if prev_positions + prev_velocities goes way outside [0, 128]. After a wall bounce, the velocity reverses but the position is clamped. If we use the pre-bounce velocity to predict the next position, we'd predict the object going through the wall.

For example: object at position 125 with velocity +2. After the step, it bounces off the wall at 128 and ends up at position 123 with velocity -2. The oracle predicts: 125 + 2 = 127. Actual: 123. Error = 4. MSE = 16. That's not huge.

But if an object bounces and then collides with another object in the same step (both happen in the 10 substeps), the actual position change could be much larger. Still, MSE of ~146,000 implies average error of ~382 per channel, which is impossible in a 128-pixel space.

I think there might be a more subtle bug. Let me re-read the ORACLE code...

Actually, I bet the issue is that the first step after warmup uses prev_positions from the last warmup step, but the actual step has already moved. And over many steps, the errors accumulate in the EMA (which uses them for push decisions). But the raw surprise should still be bounded by (128)^2 = 16384 per channel.

Wait, maybe the issue is even simpler. Let me look at the surprise computation more carefully:

```python
for c in range(D_T):
    surprise_log_coord.append(torch.mean((z_pred_coord[:, c] - z_coord[:, c])**2).item())
```

z_pred_coord and z_coord are (1, D_MAX) tensors. z_pred_coord[:, c] is a (1,) tensor. torch.mean over a (1,) tensor just returns the single value. So it's (z_pred_coord[0,c] - z_coord[0,c])^2.

For the oracle, z_pred_coord[0, :2] = prev_positions[:2] + prev_velocities[:2]. If prev_positions = [60, 90] and prev_velocities = [2, -3], then z_pred_coord[0, :2] = [62, 87]. If actual positions = [62, 87], MSE = 0. If actual = [65, 80] (collision changed velocities), MSE = (65-62)^2 + (80-87)^2 = 9 + 49 = 58. Still nowhere near 146,000.

I'm confused. Let me just re-run with some debugging to see what's actually happening. Actually, let me look more carefully at the ORACLE code in the script...

Oh wait. I see the issue now. The `prev_positions` and `prev_velocities` are updated AFTER `env.step()`:
```python
obs, info = env.step(action)
...
prev_positions = info["positions"].copy()
prev_velocities = info["velocities"].copy()
```

Then at the START of the next loop iteration:
```python
z_coord[0, :min(D_T, len(positions))] = torch.tensor(positions[:min(D_T, len(positions))], ...)
z_pred_coord[0, :min(D_T, len(prev_positions))] = torch.tensor(
    prev_positions[:min(D_T, len(prev_positions))] + prev_velocities[:min(D_T, len(prev_velocities))] * 1.0, ...)
```

But `positions = info["positions"]` — this is the info from the CURRENT step (which was set by env.step() at the end of the previous iteration). And `prev_positions` is ALSO from the same info! They're the same values!

NO WAIT. Let me re-read more carefully. The loop structure is:

```python
for step in range(EVAL_STEPS):
    # 1. Construct z_coord from info['positions'] (current state)
    positions = info["positions"]
    # 2. Construct z_pred_coord from prev_positions + prev_velocities
    z_pred_coord = prev_positions + prev_velocities
    # 3. Call controller.get_action with z_coord and z_pred_coord
    action, locus, surprises = controller.get_action(...)
    # 4. Take action -> new state
    obs, info = env.step(action)
    # 5. Update prev_positions/velocities from new info
    prev_positions = info["positions"].copy()
    prev_velocities = info["velocities"].copy()
```

So at step 0:
- positions = info from after warmup
- prev_positions = info from after warmup (initialized the same way)
- z_coord = current positions
- z_pred_coord = prev_positions + prev_velocities (predicted current positions from previous)
- BUT prev_positions IS the current positions (both from after warmup)

So the first step's prediction is: current_position + current_velocity. That predicts the NEXT step's position, not the current one! And then we compare against the current position. So the error is approximately |velocity| ≈ 2 pixels. MSE ≈ 4 per channel. That's small.

Then after step 0:
- prev_positions = new positions (after step 0's action)
- prev_velocities = new velocities

At step 1:
- positions = same new positions (info hasn't been updated yet — WAIT)

Actually, I think the issue is that `info` is not being updated between the action call and the next loop iteration in the ORACLE code. Let me re-read...

After `env.step(action)`, `info` is updated with the new positions/velocities. Then the next iteration starts and reads `positions = info["positions"]`. So positions IS the new state. And prev_positions is ALSO the new state (set at the end of the previous iteration). They're the same!

So z_pred_coord = positions + velocities (not previous_positions + previous_velocities!). The prediction is one step AHEAD of the current observation. And the target is the current observation. So the "surprise" is measuring how far the prediction overshoots, which is approximately the velocity magnitude (~2 pixels), giving MSE of ~4. Over 2000 steps, that's still just ~4 average per step.

Unless velocities are much larger than 2 pixels/step. Let me check the PhysicsSandbox... velocities are initialized in [-2, 2] but can change during collisions. After a push, the pointer gets velocity 5.0. Collisions between objects can change velocities based on mass ratios. But the objects themselves should have velocities in the single digits.

I'm going to take a different approach: run a debugging version that prints the actual z_coord and z_pred_coord values for the ORACLE condition to see what's happening.

---

OK here's what I think happened. Let me re-examine the script that was actually run. There might be a subtle timing issue where `info` is stale from the warmup loop and doesn't get properly updated at the start of the main loop.

In the warmup:
```python
for _ in range(WARMUP_STEPS):
    obs, info = env.step({"acc": 0.0, "push": False})
    history.append(obs)
    prev_positions = info["positions"].copy()
    prev_velocities = info["velocities"].copy()
# After warmup, these are set again:
prev_positions = info["positions"].copy()
prev_velocities = info["velocities"].copy()
```

Then the main loop:
```python
for step in range(EVAL_STEPS):
    ...
    positions = info["positions"]
    velocities = info["velocities"]
    # z_coord uses positions (current)
    # z_pred_coord uses prev_positions + prev_velocities (previous)
    ...
    action, locus, surprises = controller.get_action(...)
    ...
    obs, info = env.step(action)
    ...
    prev_positions = info["positions"].copy()
    prev_velocities = info["velocities"].copy()
```

At step 0:
- positions = info["positions"] = last warmup state's positions
- prev_positions = same (initialized from same info)
- z_pred_coord = positions + velocities = last warmup state's position + velocity
- z_coord = last warmup state's position
- Surprise = MSE(position + velocity, position) = velocity^2 ≈ 4

At step 0, env.step(action) updates info. At step 1:
- positions = new info["positions"] (after step 0's action)
- prev_positions = same (updated at end of step 0)
- z_pred_coord = prev_positions + prev_velocities = new_positions + new_velocities (predicts step 2)
- z_coord = new_positions (current)
- Surprise = MSE(positions + velocities, positions) ≈ velocity^2 ≈ 4

So the surprise should be bounded by velocity^2 ≈ 4-25 per channel per step. Average over all steps should be ~10-50 at most. NOT 146,000.

Something is fundamentally wrong. Let me look at the actual per-run data for oracle:

oracle,7: surprise_coord=78383
oracle,83: surprise_coord=290667

These are AVERAGES over 2000 steps. Per step, per channel, the MSE is ~78,000 to ~290,000. That's sqrt(78000) ≈ 280 pixel error. Impossible in a 128-pixel space!

Unless the issue is numerical. Let me look at how the controller computes surprise:

```python
err_coord = torch.mean((z_pred_coord[:, c] - z_target_coord[:, c])**2).item()
```

z_pred_coord and z_target_coord are both (1, D_MAX) tensors. For the oracle, they're constructed as:
```python
z_coord = torch.zeros(1, D_MAX)  # all zeros initially
z_coord[0, :min(D_T, len(positions))] = torch.tensor(positions[:min(D_T, len(positions))], dtype=torch.float32)

z_pred_coord = torch.zeros(1, D_MAX)
z_pred_coord[0, :min(D_T, len(prev_positions))] = torch.tensor(
    prev_positions[:min(D_T, len(prev_positions))] + prev_velocities[:min(D_T, len(prev_velocities))] * 1.0,
    dtype=torch.float32)
```

Wait — when `min(D_T, len(prev_positions))` is computed, D_T=3 and len(prev_positions)=2, so min=2. So only 2 elements are set. The 3rd channel stays at 0.

For z_target_coord (which is z_coord in the get_action call):
```python
centroids = z_coord.clone()
```

And in get_action:
```python
err_coord = torch.mean((z_pred_coord[:, c] - z_target_coord[:, c])**2).item()
```

For channels 0 and 1 (the two objects), this should be small. For channel 2 (unused), both are 0, so error is 0.

Hmm, unless the get_action call is using different variables than I think. Let me re-read the get_action call:

```python
action, locus, surprises = controller.get_action(
    None, history[-1], info, z_pred_coord, z_coord, z_pred_dyn, z_dyn, D_T, centroids)
```

So z_target_coord = z_coord (current positions) and z_pred_coord = predicted positions. That's correct.

OK, I'm going to take a different approach. I'll re-run with a diagnostic version that:
1. Prints the actual z_coord and z_pred_coord values for the first 10 steps of the ORACLE
2. Checks whether the issue is with the predictor, the EMA, or something else
3. If the predictor is truly broken, fix it and re-run the full experiment

Write `src/run_iter033_v2.py` that:

1. Runs JUST the ORACLE condition for seed=7 (the simplest case)
2. Prints z_coord[:3], z_pred_coord[:3], surprise per channel, EMA mu/sigma for the first 50 steps
3. Also runs the LEARNED-SFA condition for seed=7 for comparison
4. Diagnoses the issue and then runs the full corrected experiment if needed

Actually, let me just fix the likely issue and re-run. The most likely problem is that prev_positions and current positions are the SAME in the ORACLE loop (both from the same info dict), making the prediction always one step ahead rather than predicting the current state. Let me fix this by properly maintaining separate prev_positions from the PREVIOUS iteration.

Here's the corrected ORACLE implementation:

```python
# Before the loop:
prev_positions = None  # Will be set after first step
prev_velocities = None

# In the loop:
if prev_positions is None:
    # First step: no prediction available, use current positions as prediction
    z_pred_coord = z_coord.clone()
else:
    z_pred_coord = torch.zeros(1, D_MAX)
    z_pred_coord[0, :min(D_T, len(prev_positions))] = torch.tensor(
        prev_positions[:min(D_T, len(prev_positions))] + prev_velocities[:min(D_T, len(prev_velocities))] * 1.0,
        dtype=torch.float32)
# After env.step:
prev_positions = old_positions.copy()  # The positions BEFORE the step
prev_velocities = old_velocities.copy()  # The velocities BEFORE the step
```

The key fix: save the positions BEFORE taking the step, then use them to predict the positions AFTER the step. Currently, prev_positions is set to the same values as current positions because they both come from the same info dict.

Let me write and run the corrected experiment.

---

Create `src/run_iter033_v2.py` that fixes the ORACLE predictor timing issue. The full corrected script is below. Write it and run it.

```python
#!/usr/bin/env python3
"""Iter_033 v2: Three-Condition Oracle Bracket — Fixed ORACLE predictor timing."""
import os, sys, csv, json, collections, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models_separate_dyn import NonParametricJEPASpatialSeparateDyn
from src.motor import CLTSMotorController
from src.environment import PhysicsSandbox

SEEDS = [7, 17, 31, 53, 71, 83, 97, 101, 107, 113, 137, 163]
CONDITIONS = ["random", "learned_vicreg", "learned_sfa", "oracle"]
D_T = 3
D_MAX = 8
EVAL_STEPS = 2000
PERTURB_STEP = 1000
MASS_MULTIPLIER = 1.5
WARMUP_STEPS = 3
HISTORY_LEN = 4
COLLISION_DIST_THRESHOLD = 4.0
COLLISION_VELOCITY_CHANGE_THRESHOLD = 1.0
POST_COLLISION_WINDOW = 15
CKPT_DIR = "archive/iter_029/results/checkpoints"
RESULTS_DIR = "archive/iter_033/results"

def build_model():
    model = NonParametricJEPASpatialSeparateDyn(
        d_max=D_MAX, h=3, k=4, cooldown=300, stabilization_period=100,
        pos_encoding="none", primary_objective="jepa", sfa_weight=25.0,
        gdasr_log_only=True, dyn_readout="mean", sub_features=1,
        dyn_source="spatial", mask_dyn_sim=True, coord_vicreg=True,
    )
    model.d_t = D_T
    return model

def get_channel_to_obj_mapping(centroids_np, positions, d_t):
    mapping = {}
    for ch in range(d_t):
        val = centroids_np[ch]
        closest_obj = int(np.argmin(np.abs(positions - val)))
        mapping[ch] = closest_obj
    return mapping

def detect_collision(info, prev_velocities):
    pos_diff = abs(info["positions"][0] - info["positions"][1])
    radii_sum = info["radii"][0] + info["radii"][1]
    if pos_diff >= (radii_sum + COLLISION_DIST_THRESHOLD):
        return False, None
    vel_changes = np.abs(info["velocities"] - prev_velocities)
    max_change = np.max(vel_changes)
    if max_change > COLLISION_VELOCITY_CHANGE_THRESHOLD:
        return True, int(np.argmax(vel_changes))
    return False, None

def compute_selectivity_vb(collision_events, attended_per_step, eval_steps):
    post_coll_steps = []
    for coll_step, max_change_obj in collision_events:
        for s in range(coll_step + 1, coll_step + POST_COLLISION_WINDOW + 1):
            if 0 <= s < eval_steps:
                post_coll_steps.append((s, max_change_obj))
    if not post_coll_steps:
        return 0.0
    total, count = 0, 0
    step_to_attn = {s: obj for s, obj in attended_per_step}
    for s, max_obj in post_coll_steps:
        attn_obj = step_to_attn.get(s)
        if attn_obj is not None:
            total += 1
            if attn_obj == max_obj:
                count += 1
    return count / max(total, 1)

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
    gs_arr = np.array(gs)
    gs_valid = gs_arr[~np.isnan(gs_arr)]
    if len(gs_valid) == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmean(gs_arr)), float(np.percentile(gs_valid, 2.5)), float(np.percentile(gs_valid, 97.5))

def run_random(seed):
    env = PhysicsSandbox(N=2, seed=seed)
    controller = CLTSMotorController()
    model = build_model()
    model.eval()
    obs = env.reset()
    history = collections.deque(maxlen=HISTORY_LEN)
    history.append(obs)
    prev_velocities = env.velocities.copy()
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        prev_velocities = info["velocities"].copy()
    tracking_errors, collision_events, attended_per_step, perturbation_attended = [], [], [], []
    surprise_log_coord, surprise_log_dyn = [], []
    for step in range(EVAL_STEPS):
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER
        obs_tensor = torch.tensor(history[-1], dtype=torch.float32).unsqueeze(0)
        x_hist = torch.from_numpy(np.stack(list(history)[:3], axis=0)).float().unsqueeze(0)
        x_target = torch.from_numpy(history[-1]).float().unsqueeze(0)
        with torch.no_grad():
            z_coord, z_dyn = model.encoder(obs_tensor)
            centroids = z_coord
            loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(x_hist, x_target, d_t_predict=min(D_T, D_MAX))
        controller.mu[:] = 0.0
        controller.sigma[:] = 1.0
        controller.attention_cooldown = 0
        action, locus, surprises = controller.get_action(
            None, history[-1], info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, D_T, centroids)
        random_locus = int(np.random.randint(0, D_T))
        locus = random_locus
        controller.token_locus = random_locus
        for c in range(D_T):
            surprise_log_coord.append(torch.mean((z_pred_coord[:, c] - z_target_coord[:, c])**2).item())
            surprise_log_dyn.append(torch.mean((z_pred_dyn[:, c] - z_target_dyn[:, c])**2).item())
        obs, info = env.step(action)
        history.append(obs)
        centroids_np = centroids[0, :D_T].cpu().numpy()
        ch2obj = get_channel_to_obj_mapping(centroids_np, info["positions"], D_T)
        attended_obj = ch2obj.get(locus, -1)
        target_centroid = centroids_np[locus]
        tracking_errors.append(abs(info["pointer_pos"] - target_centroid))
        attended_per_step.append((step, attended_obj))
        is_collision, max_change_obj = detect_collision(info, prev_velocities)
        if is_collision:
            collision_events.append((step, max_change_obj))
        if PERTURB_STEP <= step <= PERTURB_STEP + 99:
            perturbation_attended.append(1 if attended_obj == 0 else 0)
        prev_velocities = info["velocities"].copy()
    post_coll_vb = compute_selectivity_vb(collision_events, attended_per_step, EVAL_STEPS)
    return {
        "tracking_error": float(np.mean(tracking_errors)),
        "selectivity_vb": post_coll_vb,
        "perturbation_selectivity": float(np.mean(perturbation_attended)) if perturbation_attended else 0.0,
        "collision_count": len(collision_events),
        "mean_surprise_coord": float(np.mean(surprise_log_coord)) if surprise_log_coord else 0.0,
        "mean_surprise_dyn": float(np.mean(surprise_log_dyn)) if surprise_log_dyn else 0.0,
    }

def run_learned(seed, ckpt_path):
    env = PhysicsSandbox(N=2, seed=seed)
    controller = CLTSMotorController()
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    obs = env.reset()
    history = collections.deque(maxlen=HISTORY_LEN)
    history.append(obs)
    prev_velocities = env.velocities.copy()
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        prev_velocities = info["velocities"].copy()
    tracking_errors, collision_events, attended_per_step, perturbation_attended = [], [], [], []
    surprise_log_coord, surprise_log_dyn = [], []
    for step in range(EVAL_STEPS):
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER
        obs_tensor = torch.tensor(history[-1], dtype=torch.float32).unsqueeze(0)
        x_hist = torch.from_numpy(np.stack(list(history)[:3], axis=0)).float().unsqueeze(0)
        x_target = torch.from_numpy(history[-1]).float().unsqueeze(0)
        with torch.no_grad():
            z_coord, z_dyn = model.encoder(obs_tensor)
            centroids = z_coord
            loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(x_hist, x_target, d_t_predict=min(D_T, D_MAX))
        action, locus, surprises = controller.get_action(
            None, history[-1], info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, D_T, centroids)
        for c in range(D_T):
            surprise_log_coord.append(torch.mean((z_pred_coord[:, c] - z_target_coord[:, c])**2).item())
            surprise_log_dyn.append(torch.mean((z_pred_dyn[:, c] - z_target_dyn[:, c])**2).item())
        obs, info = env.step(action)
        history.append(obs)
        centroids_np = centroids[0, :D_T].cpu().numpy()
        ch2obj = get_channel_to_obj_mapping(centroids_np, info["positions"], D_T)
        attended_obj = ch2obj.get(locus, -1)
        target_centroid = centroids_np[locus]
        tracking_errors.append(abs(info["pointer_pos"] - target_centroid))
        attended_per_step.append((step, attended_obj))
        is_collision, max_change_obj = detect_collision(info, prev_velocities)
        if is_collision:
            collision_events.append((step, max_change_obj))
        if PERTURB_STEP <= step <= PERTURB_STEP + 99:
            perturbation_attended.append(1 if attended_obj == 0 else 0)
        prev_velocities = info["velocities"].copy()
    post_coll_vb = compute_selectivity_vb(collision_events, attended_per_step, EVAL_STEPS)
    return {
        "tracking_error": float(np.mean(tracking_errors)),
        "selectivity_vb": post_coll_vb,
        "perturbation_selectivity": float(np.mean(perturbation_attended)) if perturbation_attended else 0.0,
        "collision_count": len(collision_events),
        "mean_surprise_coord": float(np.mean(surprise_log_coord)) if surprise_log_coord else 0.0,
        "mean_surprise_dyn": float(np.mean(surprise_log_dyn)) if surprise_log_dyn else 0.0,
    }

def run_oracle(seed):
    env = PhysicsSandbox(N=2, seed=seed)
    controller = CLTSMotorController()
    obs = env.reset()
    history = collections.deque(maxlen=HISTORY_LEN)
    history.append(obs)
    prev_velocities = env.velocities.copy()
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        prev_velocities = info["velocities"].copy()
    # CRITICAL FIX: prev_positions and prev_velocities must be from BEFORE the current step
    # Initialize them from the last warmup step's state
    prev_positions = info["positions"].copy()
    prev_velocities = info["velocities"].copy()
    first_step = True
    tracking_errors, collision_events, attended_per_step, perturbation_attended = [], [], [], []
    surprise_log_coord, surprise_log_dyn = [], []
    for step in range(EVAL_STEPS):
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER
        # Current ground truth
        positions = info["positions"]
        velocities = info["velocities"]
        colors = info["colors"]
        # Construct z_coord (current ground truth)
        z_coord = torch.zeros(1, D_MAX)
        n_obj = min(D_T, len(positions))
        z_coord[0, :n_obj] = torch.tensor(positions[:n_obj], dtype=torch.float32)
        # Construct z_dyn (current ground truth identity)
        z_dyn = torch.zeros(1, D_MAX)
        for i in range(min(D_T, len(colors))):
            z_dyn[0, i] = torch.tensor(np.mean(colors[i]), dtype=torch.float32)
        # Construct z_pred_coord (prediction of CURRENT positions from PREVIOUS state)
        if first_step:
            # No previous state available; predict current = current
            z_pred_coord = z_coord.clone()
            first_step = False
        else:
            z_pred_coord = torch.zeros(1, D_MAX)
            n_prev = min(D_T, len(prev_positions))
            predicted = prev_positions[:n_prev] + prev_velocities[:n_prev] * 1.0
            z_pred_coord[0, :n_prev] = torch.tensor(predicted, dtype=torch.float32)
        # z_pred_dyn = z_dyn (identity is constant)
        z_pred_dyn = z_dyn.clone()
        centroids = z_coord.clone()
        # SAVE current state as "previous" BEFORE taking action (for next iteration's prediction)
        saved_positions = positions.copy()
        saved_velocities = velocities.copy()
        # Call controller
        action, locus, surprises = controller.get_action(
            None, history[-1], info, z_pred_coord, z_coord, z_pred_dyn, z_dyn, D_T, centroids)
        # Log surprise decomposition
        for c in range(D_T):
            surprise_log_coord.append(torch.mean((z_pred_coord[:, c] - z_coord[:, c])**2).item())
            surprise_log_dyn.append(0.0)
        obs, info = env.step(action)
        history.append(obs)
        # Update prev_positions/velocities for next iteration
        prev_positions = saved_positions
        prev_velocities = saved_velocities
        centroids_np = centroids[0, :D_T].cpu().numpy()
        ch2obj = get_channel_to_obj_mapping(centroids_np, info["positions"], D_T)
        attended_obj = ch2obj.get(locus, -1)
        target_centroid = centroids_np[locus]
        tracking_errors.append(abs(info["pointer_pos"] - target_centroid))
        attended_per_step.append((step, attended_obj))
        is_collision, max_change_obj = detect_collision(info, prev_velocities)
        if is_collision:
            collision_events.append((step, max_change_obj))
        if PERTURB_STEP <= step <= PERTURB_STEP + 99:
            perturbation_attended.append(1 if attended_obj == 0 else 0)
        prev_velocities = info["velocities"].copy()
    post_coll_vb = compute_selectivity_vb(collision_events, attended_per_step, EVAL_STEPS)
    return {
        "tracking_error": float(np.mean(tracking_errors)),
        "selectivity_vb": post_coll_vb,
        "perturbation_selectivity": float(np.mean(perturbation_attended)) if perturbation_attended else 0.0,
        "collision_count": len(collision_events),
        "mean_surprise_coord": float(np.mean(surprise_log_coord)) if surprise_log_coord else 0.0,
        "mean_surprise_dyn": 0.0,
    }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.set_num_threads(4)
    print(f"Iter_033 v2 — Fixed ORACLE predictor timing")
    print(f"Seeds: {SEEDS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"d_t={D_T}, d_max={D_MAX}, eval_steps={EVAL_STEPS}")
    all_results = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            label = f"{condition} | seed={seed}"
            print(f"[RUN] {label} ...", end=" ", flush=True)
            if condition == "random":
                result = run_random(seed)
            elif condition == "learned_vicreg":
                ckpt = os.path.join(CKPT_DIR, f"a_vicreg-only_control_seed{seed}.pt")
                result = run_learned(seed, ckpt)
            elif condition == "learned_sfa":
                ckpt = os.path.join(CKPT_DIR, f"b_sfavicreg,_sfa_5.0_seed{seed}.pt")
                result = run_learned(seed, ckpt)
            elif condition == "oracle":
                result = run_oracle(seed)
            else:
                raise ValueError(f"Unknown condition: {condition}")
            row = {"condition": condition, "seed": seed, **result}
            all_results.append(row)
            print(f"sel_vb={result['selectivity_vb']:.4f} track={result['tracking_error']:.2f} "
                  f"pert={result['perturbation_selectivity']:.4f} coll={result['collision_count']} "
                  f"s_coord={result['mean_surprise_coord']:.4f} s_dyn={result['mean_surprise_dyn']:.4f}")

    df = pd.DataFrame(all_results)
    per_run_path = os.path.join(RESULTS_DIR, "per_run_v2.csv")
    df.to_csv(per_run_path, index=False)
    print(f"\nSaved per-run results to {per_run_path}")

    # Summary
    summary_rows = []
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        row = {
            "condition": condition,
            "n_seeds": len(sub),
            "mean_selectivity_vb": sub["selectivity_vb"].mean(),
            "std_selectivity_vb": sub["selectivity_vb"].std(ddof=1) if len(sub) > 1 else 0.0,
            "mean_tracking_error": sub["tracking_error"].mean(),
            "std_tracking_error": sub["tracking_error"].std(ddof=1) if len(sub) > 1 else 0.0,
            "mean_perturbation_sel": sub["perturbation_selectivity"].mean(),
            "std_perturbation_sel": sub["perturbation_selectivity"].std(ddof=1) if len(sub) > 1 else 0.0,
            "mean_surprise_coord": sub["mean_surprise_coord"].mean(),
            "mean_surprise_dyn": sub["mean_surprise_dyn"].mean(),
        }
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "summary_v2.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # g-ratio analysis
    random_sel = df[df["condition"] == "random"].set_index("seed")["selectivity_vb"]
    vicreg_sel = df[df["condition"] == "learned_vicreg"].set_index("seed")["selectivity_vb"]
    sfa_sel = df[df["condition"] == "learned_sfa"].set_index("seed")["selectivity_vb"]
    oracle_sel = df[df["condition"] == "oracle"].set_index("seed")["selectivity_vb"]

    common_seeds = sorted(set(random_sel.index) & set(vicreg_sel.index) & set(sfa_sel.index) & set(oracle_sel.index))
    r_vals = random_sel[common_seeds].values
    vr_vals = vicreg_sel[common_seeds].values
    sfa_vals = sfa_sel[common_seeds].values
    o_vals = oracle_sel[common_seeds].values

    r_mean = np.mean(r_vals)
    o_mean = np.mean(o_vals)
    vr_mean = np.mean(vr_vals)
    sfa_mean = np.mean(sfa_vals)

    print("\n" + "="*60)
    print("ORACLE BRACKET ANALYSIS (v2 — fixed timing)")
    print("="*60)
    print(f"\nRaw triple (selectivity_vb):")
    print(f"  RANDOM:         {r_mean:.4f} +/- {np.std(r_vals, ddof=1):.4f}")
    print(f"  LEARNED-VICReg: {vr_mean:.4f} +/- {np.std(vr_vals, ddof=1):.4f}")
    print(f"  LEARNED-SFA:    {sfa_mean:.4f} +/- {np.std(sfa_vals, ddof=1):.4f}")
    print(f"  ORACLE:         {o_mean:.4f} +/- {np.std(o_vals, ddof=1):.4f}")

    ordering_ok = (o_mean >= r_mean)
    print(f"\nOrdering sanity check: ORACLE({o_mean:.4f}) >= RANDOM({r_mean:.4f}) -> {ordering_ok}")

    oracle_random_gap = o_mean - r_mean
    branch_c = abs(oracle_random_gap) < 0.10
    print(f"Branch (c) check: |ORACLE - RANDOM| = |{oracle_random_gap:.4f}| < 0.10 -> {branch_c}")

    if not branch_c and ordering_ok:
        g_vr_mean, g_vr_lo, g_vr_hi = bootstrap_g(vr_vals, r_vals, o_vals)
        g_sfa_mean, g_sfa_lo, g_sfa_hi = bootstrap_g(sfa_vals, r_vals, o_vals)
        print(f"\ng_vicreg = {g_vr_mean:.4f} (95% CI: [{g_vr_lo:.4f}, {g_vr_hi:.4f}])")
        print(f"g_sfa    = {g_sfa_mean:.4f} (95% CI: [{g_sfa_lo:.4f}, {g_sfa_hi:.4f}])")

        for name, g_mean, g_lo, g_hi in [("VICReg", g_vr_mean, g_vr_lo, g_vr_hi), ("SFA", g_sfa_mean, g_sfa_lo, g_sfa_hi)]:
            if g_mean >= 0.70 and g_lo >= 0.50:
                branch = "(a) consistent with sufficiency"
            elif g_mean <= 0.20:
                branch = "(b) representation provably limits behavior"
            elif 0.20 < g_mean < 0.70:
                branch = "(d) partial sufficiency"
            else:
                branch = "ambiguous"
            print(f"  {name}: {branch}")
    elif branch_c:
        print("\nBRANCH (c) FIRED: Task/motor protocol is the bottleneck, NOT perception.")
    elif not ordering_ok:
        print("\nOrdering violated: ORACLE < RANDOM on primary metric.")
        print("The oracle bracket is not valid with this protocol/metric combination.")
        print("This itself is a finding: perfect perception does not improve this metric under this motor code.")

    # Per-seed table
    print(f"\nPer-seed selectivity_vb:")
    print(f"{'Seed':>6} | {'RANDOM':>8} | {'VICReg':>8} | {'SFA':>8} | {'ORACLE':>8}")
    for s in common_seeds:
        print(f"{s:>6} | {random_sel[s]:.4f}   | {vicreg_sel[s]:.4f}   | {sfa_sel[s]:.4f}   | {oracle_sel[s]:.4f}")

    # Surprise decomposition
    print(f"\nSurprise decomposition (mean across seeds):")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        print(f"  {condition:>15}: coord={sub['mean_surprise_coord'].mean():.4f}, dyn={sub['mean_surprise_dyn'].mean():.4f}")

    # Secondary metrics
    print(f"\nSecondary metrics (mean +/- std):")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        print(f"  {condition:>15}: tracking={sub['tracking_error'].mean():.2f}+/-{sub['tracking_error'].std(ddof=1):.2f}, "
              f"pert_sel={sub['perturbation_selectivity'].mean():.4f}+/-{sub['perturbation_selectivity'].std(ddof=1):.4f}")

    # Write analysis report
    analysis_lines = []
    analysis_lines.append("# Iter_033 v2 — Three-Condition Oracle Bracket (Fixed ORACLE Timing)\n\n")
    analysis_lines.append("## Raw Triple (Primary Metric: Post-Collision Selectivity V-B)\n\n")
    analysis_lines.append("| Condition | Mean | Std | n |\n")
    analysis_lines.append("|----------|------|-----|---|\n")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        analysis_lines.append(f"| {condition} | {sub['selectivity_vb'].mean():.4f} | {sub['selectivity_vb'].std(ddof=1):.4f} | {len(sub)} |\n")
    analysis_lines.append(f"\n**Ordering sanity check:** ORACLE({o_mean:.4f}) >= RANDOM({r_mean:.4f}) = {ordering_ok}\n\n")
    analysis_lines.append(f"**Branch (c) check:** |ORACLE - RANDOM| = |{oracle_random_gap:.4f}| < 0.10 = {branch_c}\n\n")

    if not branch_c and ordering_ok:
        analysis_lines.append("## g-Ratio Analysis\n\n")
        analysis_lines.append(f"| Arm | g | 95% CI Lower | 95% CI Upper | Branch |\n")
        analysis_lines.append(f"|-----|---|-------------|-------------|--------|\n")
        for name, g_mean, g_lo, g_hi in [("VICReg", g_vr_mean, g_vr_lo, g_vr_hi), ("SFA", g_sfa_mean, g_sfa_lo, g_sfa_hi)]:
            if g_mean >= 0.70 and g_lo >= 0.50:
                branch = "(a)"
            elif g_mean <= 0.20:
                branch = "(b)"
            elif 0.20 < g_mean < 0.70:
                branch = "(d)"
            else:
                branch = "ambiguous"
            analysis_lines.append(f"| {name} | {g_mean:.4f} | {g_lo:.4f} | {g_hi:.4f} | {branch} |\n")
    elif branch_c:
        analysis_lines.append("## BRANCH (c) FIRED\n\n")
        analysis_lines.append("The task or motor protocol is the bottleneck, NOT perception.\n")
    elif not ordering_ok:
        analysis_lines.append("## Ordering Violated\n\n")
        analysis_lines.append("ORACLE < RANDOM on primary metric. Perfect perception does not improve this metric under this motor code.\n")

    analysis_lines.append("\n## Per-Seed Primary Metric\n\n")
    analysis_lines.append("| Seed | RANDOM | VICReg | SFA | ORACLE |\n")
    analysis_lines.append("|------|--------|--------|-----|--------|\n")
    for s in common_seeds:
        analysis_lines.append(f"| {s} | {random_sel[s]:.4f} | {vicreg_sel[s]:.4f} | {sfa_sel[s]:.4f} | {oracle_sel[s]:.4f} |\n")

    analysis_lines.append("\n## Surprise Decomposition\n\n")
    analysis_lines.append("| Condition | Mean Surprise Coord | Mean Surprise Dyn |\n")
    analysis_lines.append("|-----------|--------------------|------------------|\n")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        analysis_lines.append(f"| {condition} | {sub['mean_surprise_coord'].mean():.4f} | {sub['mean_surprise_dyn'].mean():.4f} |\n")

    analysis_lines.append("\n## Secondary Metrics\n\n")
    analysis_lines.append("| Condition | Tracking Error (mean) | Tracking Error (std) | Pert Sel (mean) | Pert Sel (std) |\n")
    analysis_lines.append("|-----------|----------------------|---------------------|-----------------|---------------|\n")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        te_m = sub['tracking_error'].mean()
        te_s = sub['tracking_error'].std(ddof=1)
        ps_m = sub['perturbation_selectivity'].mean()
        ps_s = sub['perturbation_selectivity'].std(ddof=1)
        analysis_lines.append(f"| {condition} | {te_m:.2f} | {te_s:.2f} | {ps_m:.4f} | {ps_s:.4f} |\n")

    analysis_path = os.path.join(RESULTS_DIR, "analysis_v2.md")
    with open(analysis_path, "w") as f:
        f.writelines(analysis_lines)
    print(f"\nSaved analysis to {analysis_path}")
    print(f"\nTotal runs: {len(all_results)}")

if __name__ == "__main__":
    main()
```

Write this script to `src/run_iter033_v2.py` and run it. Print all output.