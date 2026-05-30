#!/usr/bin/env python3
"""Diagnostic: full physics predictor including pointer."""
import os, sys, collections
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment import PhysicsSandbox
from src.motor import CLTSMotorController

D_T = 3
D_MAX = 8
WARMUP_STEPS = 3
EVAL_STEPS = 2000
PERTURB_STEP = 1000
MASS_MULTIPLIER = 1.5


def simulate_physics(prev_info, dt=1.0, substeps=10):
    """Simulate all entities forward by dt with no actions."""
    # Object state
    pos = np.concatenate([prev_info["positions"].copy(), [prev_info["pointer_pos"]]])
    vel = np.concatenate([prev_info["velocities"].copy(), [prev_info["pointer_vel"]]])
    radii = np.concatenate([prev_info["radii"].copy(), [prev_info["pointer_radius"]]])
    masses = np.concatenate([prev_info["masses"].copy(), [prev_info["pointer_mass"]]])
    
    sub_dt = dt / substeps
    
    for _ in range(substeps):
        pos += vel * sub_dt
        
        # Wall bounces
        for i in range(len(pos)):
            if pos[i] - radii[i] < 0.0:
                pos[i] = radii[i]
                if vel[i] < 0.0:
                    vel[i] = -vel[i]
            elif pos[i] + radii[i] > 128.0:
                pos[i] = 128.0 - radii[i]
                if vel[i] > 0.0:
                    vel[i] = -vel[i]
        
        # Resolve collisions between adjacent entities
        sort_idx = np.argsort(pos)
        for idx_in_sort in range(len(pos) - 1):
            i = sort_idx[idx_in_sort]
            j = sort_idx[idx_in_sort + 1]
            
            dist = pos[j] - pos[i]
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                m_inv_i = 1.0 / masses[i]
                m_inv_j = 1.0 / masses[j]
                sum_inv_m = m_inv_i + m_inv_j
                
                pos[i] -= overlap * (m_inv_i / sum_inv_m)
                pos[j] += overlap * (m_inv_j / sum_inv_m)
                
                if vel[i] > vel[j]:
                    v1, v2 = vel[i], vel[j]
                    m1, m2 = masses[i], masses[j]
                    vel[i] = (v1 * (m1 - m2) + 2.0 * m2 * v2) / (m1 + m2)
                    vel[j] = (v2 * (m2 - m1) + 2.0 * m1 * v1) / (m1 + m2)
            
            # Additional boundary check
            for k in range(len(pos)):
                if pos[k] - radii[k] < 0.0:
                    pos[k] = radii[k]
                    if vel[k] < 0.0:
                        vel[k] = -vel[k]
                elif pos[k] + radii[k] > 128.0:
                    pos[k] = 128.0 - radii[k]
                    if vel[k] > 0.0:
                        vel[k] = -vel[k]
    
    # Return predicted object positions (exclude pointer)
    return pos[:-1]


def run_variant(use_physics=False):
    env = PhysicsSandbox(N=2, seed=7)
    controller = CLTSMotorController()
    obs = env.reset()
    history = collections.deque(maxlen=4)
    history.append(obs)
    
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
    
    prev_info = {
        "positions": info["positions"].copy(),
        "velocities": info["velocities"].copy(),
        "radii": info["radii"].copy(),
        "masses": info["masses"].copy(),
        "pointer_pos": info["pointer_pos"],
        "pointer_vel": info["pointer_vel"],
        "pointer_radius": info["pointer_radius"],
        "pointer_mass": info["pointer_mass"],
    }
    
    surprises_all = []
    
    for step in range(EVAL_STEPS):
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER
        
        saved_info = {
            "positions": info["positions"].copy(),
            "velocities": info["velocities"].copy(),
            "radii": info["radii"].copy(),
            "masses": info["masses"].copy(),
            "pointer_pos": info["pointer_pos"],
            "pointer_vel": info["pointer_vel"],
            "pointer_radius": info["pointer_radius"],
            "pointer_mass": info["pointer_mass"],
        }
        
        positions = info["positions"]
        colors = info["colors"]
        
        z_coord = torch.zeros(1, D_MAX)
        n_obj = min(D_T, len(positions))
        z_coord[0, :n_obj] = torch.tensor(positions[:n_obj], dtype=torch.float32)
        
        z_dyn = torch.zeros(1, D_MAX)
        for i in range(min(D_T, len(colors))):
            z_dyn[0, i] = torch.tensor(np.mean(colors[i]), dtype=torch.float32)
        
        z_pred_coord = torch.zeros(1, D_MAX)
        if use_physics:
            predicted = simulate_physics(prev_info, dt=1.0, substeps=10)
        else:
            n_prev = min(D_T, len(prev_info["positions"]))
            predicted = prev_info["positions"][:n_prev] + prev_info["velocities"][:n_prev] * 1.0
        n_pred = min(D_T, len(predicted))
        z_pred_coord[0, :n_pred] = torch.tensor(predicted[:n_pred], dtype=torch.float32)
        
        z_pred_dyn = z_dyn.clone()
        centroids = z_coord.clone()
        
        step_surprises = []
        for c in range(D_T):
            surprise = torch.mean((z_pred_coord[:, c] - z_coord[:, c])**2).item()
            step_surprises.append(surprise)
        surprises_all.append(step_surprises)
        
        action, locus, surprises = controller.get_action(
            None, history[-1], info, z_pred_coord, z_coord, z_pred_dyn, z_dyn, D_T, centroids)
        
        obs, info = env.step(action)
        history.append(obs)
        
        prev_info = saved_info
    
    surprises_all = np.array(surprises_all)
    return surprises_all


print("Testing predictor variants...")

s_naive = run_variant(use_physics=False)
print(f"\nNAIVE predictor:")
print(f"  mean={s_naive.mean():.2f}, max={s_naive.max():.2f}, ch0+1_mean={s_naive[:, :2].mean():.2f}")
for c in range(D_T):
    count = np.sum(s_naive[:, c] > 1000)
    print(f"  ch{c}: {count} steps with surprise > 1000")

s_physics = run_variant(use_physics=True)
print(f"\nFULL PHYSICS predictor:")
print(f"  mean={s_physics.mean():.2f}, max={s_physics.max():.2f}, ch0+1_mean={s_physics[:, :2].mean():.2f}")
for c in range(D_T):
    count = np.sum(s_physics[:, c] > 1000)
    print(f"  ch{c}: {count} steps with surprise > 1000")

# Show top 10 highest surprise steps for physics
flat_surprises = s_physics.flatten()
flat_indices = np.argsort(flat_surprises)[-10:]
print("\nTop 10 highest surprise values (physics):")
for idx in reversed(flat_indices):
    step = idx // D_T
    ch = idx % D_T
    print(f"  step={step}, ch={ch}, surprise={flat_surprises[idx]:.4f}")

# Compare histograms
print("\nSurprise histograms (physics, ch0+1):")
combined = s_physics[:, :2].flatten()
print(f"  <1:   {np.sum(combined < 1)}")
print(f"  <10:  {np.sum(combined < 10)}")
print(f"  <100: {np.sum(combined < 100)}")
print(f"  <1000: {np.sum(combined < 1000)}")
print(f"  >=1000: {np.sum(combined >= 1000)}")
