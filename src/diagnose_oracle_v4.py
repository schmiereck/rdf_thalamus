#!/usr/bin/env python3
"""Diagnostic: compare predictor variants for ORACLE."""
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


def predict_naive(prev_positions, prev_velocities, d_t):
    """v2: prev_pos + prev_vel * dt"""
    n_prev = min(d_t, len(prev_positions))
    predicted = prev_positions[:n_prev] + prev_velocities[:n_prev] * 1.0
    return predicted


def predict_physics(prev_positions, prev_velocities, radii, masses, dt=1.0, substeps=10):
    """Simulate object physics (no pointer) for dt."""
    pos = prev_positions.copy()
    vel = prev_velocities.copy()
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
        
        # Object-object collisions
        if len(pos) == 2:
            dist = pos[1] - pos[0]
            min_dist = radii[0] + radii[1]
            if dist < min_dist:
                overlap = min_dist - dist
                m_inv_0 = 1.0 / masses[0]
                m_inv_1 = 1.0 / masses[1]
                sum_inv = m_inv_0 + m_inv_1
                pos[0] -= overlap * (m_inv_0 / sum_inv)
                pos[1] += overlap * (m_inv_1 / sum_inv)
                if vel[0] > vel[1]:
                    v0, v1 = vel[0], vel[1]
                    m0, m1 = masses[0], masses[1]
                    vel[0] = (v0 * (m0 - m1) + 2.0 * m1 * v1) / (m0 + m1)
                    vel[1] = (v1 * (m1 - m0) + 2.0 * m0 * v0) / (m0 + m1)
            # Additional boundary check after collision
            for i in range(len(pos)):
                if pos[i] - radii[i] < 0.0:
                    pos[i] = radii[i]
                    if vel[i] < 0.0:
                        vel[i] = -vel[i]
                elif pos[i] + radii[i] > 128.0:
                    pos[i] = 128.0 - radii[i]
                    if vel[i] > 0.0:
                        vel[i] = -vel[i]
    
    return pos


def run_variant(predictor_fn, label):
    env = PhysicsSandbox(N=2, seed=7)
    controller = CLTSMotorController()
    obs = env.reset()
    history = collections.deque(maxlen=4)
    history.append(obs)
    
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
    
    prev_positions = info["positions"].copy()
    prev_velocities = info["velocities"].copy()
    prev_radii = info["radii"].copy()
    prev_masses = info["masses"].copy()
    
    surprises_all = []
    
    for step in range(EVAL_STEPS):
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER
        
        saved_positions = info["positions"].copy()
        saved_velocities = info["velocities"].copy()
        saved_radii = info["radii"].copy()
        saved_masses = info["masses"].copy()
        
        positions = info["positions"]
        colors = info["colors"]
        
        z_coord = torch.zeros(1, D_MAX)
        n_obj = min(D_T, len(positions))
        z_coord[0, :n_obj] = torch.tensor(positions[:n_obj], dtype=torch.float32)
        
        z_dyn = torch.zeros(1, D_MAX)
        for i in range(min(D_T, len(colors))):
            z_dyn[0, i] = torch.tensor(np.mean(colors[i]), dtype=torch.float32)
        
        z_pred_coord = torch.zeros(1, D_MAX)
        if predictor_fn == "naive":
            predicted = predict_naive(prev_positions, prev_velocities, D_T)
        else:
            predicted = predict_physics(prev_positions, prev_velocities, prev_radii, prev_masses, dt=1.0, substeps=10)
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
        
        prev_positions = saved_positions
        prev_velocities = saved_velocities
        prev_radii = saved_radii
        prev_masses = saved_masses
    
    surprises_all = np.array(surprises_all)
    return surprises_all


print("Testing predictor variants...")

s_naive = run_variant("naive", "naive")
print(f"\nNAIVE predictor:")
print(f"  mean={s_naive.mean():.2f}, max={s_naive.max():.2f}, ch0+1_mean={s_naive[:, :2].mean():.2f}")
for c in range(D_T):
    count = np.sum(s_naive[:, c] > 1000)
    print(f"  ch{c}: {count} steps with surprise > 1000")

s_physics = run_variant("physics", "physics")
print(f"\nPHYSICS predictor:")
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
