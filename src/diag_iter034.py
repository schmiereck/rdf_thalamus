"""Quick diagnostic to understand mass estimation behavior."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import PhysicsSandbox

# Replicate a single ORACLE run with diagnostics
seed = 31
env = PhysicsSandbox(N=3, substeps=10, seed=seed)
np.random.seed(seed)

obj_pointer_collisions = [0, 0, 0]
push_cooldown = 0
prev_error = None
forced_next_target = None

collisions = []

for step in range(2000):
    pre_velocities = np.concatenate([env.velocities.copy(), [env.pointer_vel]])
    pre_positions = np.concatenate([env.positions.copy(), [env.pointer_pos]])

    if forced_next_target is not None:
        target_idx = forced_next_target
        forced_next_target = None
    else:
        target_idx = int(np.argmin(obj_pointer_collisions))

    error = env.positions[target_idx] - env.pointer_pos
    d_error = error - prev_error if prev_error is not None else 0.0
    acc = 2.0 * error + 0.5 * d_error
    prev_error = error

    if push_cooldown > 0:
        push_cooldown -= 1

    if abs(error) <= 6.0 and push_cooldown == 0:
        env.pointer_vel = 5.0 * np.sign(error)
        push_cooldown = 15
        counts = np.array(obj_pointer_collisions, dtype=float)
        counts[target_idx] = np.inf
        forced_next_target = int(np.argmin(counts))

    action = {'acc': acc, 'push': False}
    obs, info = env.step(action)

    post_velocities = np.concatenate([info["velocities"].copy(), [info["pointer_vel"]]])
    post_positions = np.concatenate([info["positions"].copy(), [info["pointer_pos"]]])

    radii = np.concatenate([env.radii.copy(), [env.pointer_radius]])
    sort_idx = np.argsort(post_positions)
    for k in range(len(sort_idx) - 1):
        i = int(sort_idx[k])
        j = int(sort_idx[k + 1])
        dist = post_positions[j] - post_positions[i]
        radii_sum = radii[i] + radii[j]
        if dist < radii_sum + 4.0:
            dv_i = post_velocities[i] - pre_velocities[i]
            dv_j = post_velocities[j] - pre_velocities[j]
            if abs(dv_i) > 0.5 and abs(dv_j) > 0.5:
                collisions.append((step, i, j, pre_velocities[i], pre_velocities[j],
                                   post_velocities[i], post_velocities[j],
                                   float(dv_i), float(dv_j)))
                if i == 3 and j < 3:
                    obj_pointer_collisions[j] += 1
                elif j == 3 and i < 3:
                    obj_pointer_collisions[i] += 1

print("True masses:", env.masses)
print("Per-object pointer collisions:", obj_pointer_collisions)
print("Total collisions:", len(collisions))

# Analyze pointer-object collisions
po_collisions = [c for c in collisions if c[1] == 3 or c[2] == 3]
print("Pointer-object collisions:", len(po_collisions))

m_estimates = {0: [], 1: [], 2: []}
for c in po_collisions:
    step, i, j, vi_pre, vj_pre, vi_post, vj_post, dvi, dvj = c
    if i == 3:
        obj_idx = j
        dv_pointer = dvi
        dv_obj = dvj
    else:
        obj_idx = i
        dv_pointer = dvj
        dv_obj = dvi
    if abs(dv_obj) > 1e-8:
        m_est = 10.0 * (-dv_pointer) / dv_obj
        m_estimates[obj_idx].append(m_est)

for obj_idx in range(3):
    ests = m_estimates[obj_idx]
    if ests:
        ests = np.array(ests)
        print(f"\nObject {obj_idx} (true mass={env.masses[obj_idx]:.2f}):")
        print(f"  N estimates: {len(ests)}")
        print(f"  Mean: {np.mean(ests):.2f}, Median: {np.median(ests):.2f}")
        print(f"  Std: {np.std(ests):.2f}")
        print(f"  Min: {np.min(ests):.2f}, Max: {np.max(ests):.2f}")
        print(f"  P5: {np.percentile(ests, 5):.2f}, P95: {np.percentile(ests, 95):.2f}")
        # Check for outliers
        n_neg = np.sum(ests < 0)
        n_large = np.sum(ests > 50)
        print(f"  Negative estimates: {n_neg}, >50: {n_large}")

# Now do the same with noise
noise_rng = np.random.RandomState(seed * 100 + hash('ORACLE') % 10000)
m_estimates_noisy = {0: [], 1: [], 2: []}
for c in po_collisions:
    step, i, j, vi_pre, vj_pre, vi_post, vj_post, dvi, dvj = c
    vi_pre_n = vi_pre + noise_rng.normal(0, 0.5)
    vj_pre_n = vj_pre + noise_rng.normal(0, 0.5)
    vi_post_n = vi_post + noise_rng.normal(0, 0.5)
    vj_post_n = vj_post + noise_rng.normal(0, 0.5)
    dvi_n = vi_post_n - vi_pre_n
    dvj_n = vj_post_n - vj_pre_n
    if i == 3:
        obj_idx = j
        dv_pointer = dvi_n
        dv_obj = dvj_n
    else:
        obj_idx = i
        dv_pointer = dvj_n
        dv_obj = dvi_n
    if abs(dv_obj) > 1e-8:
        m_est = 10.0 * (-dv_pointer) / dv_obj
        m_estimates_noisy[obj_idx].append(m_est)

print("\n--- WITH NOISE ---")
for obj_idx in range(3):
    ests = m_estimates_noisy[obj_idx]
    if ests:
        ests = np.array(ests)
        print(f"\nObject {obj_idx} (true mass={env.masses[obj_idx]:.2f}):")
        print(f"  N estimates: {len(ests)}")
        print(f"  Mean: {np.mean(ests):.2f}, Median: {np.median(ests):.2f}")
        print(f"  Std: {np.std(ests):.2f}")
        print(f"  Min: {np.min(ests):.2f}, Max: {np.max(ests):.2f}")
        print(f"  P5: {np.percentile(ests, 5):.2f}, P95: {np.percentile(ests, 95):.2f}")
        n_neg = np.sum(ests < 0)
        n_large = np.sum(ests > 50)
        print(f"  Negative estimates: {n_neg}, >50: {n_large}")
