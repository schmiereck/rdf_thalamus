#!/usr/bin/env python3
"""Diagnostic script to trace ORACLE surprise values with v2 logic over full run."""
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

surprises_all = []
max_surprise_steps = []

for step in range(EVAL_STEPS):
    if step == PERTURB_STEP:
        env.masses[0] *= MASS_MULTIPLIER

    saved_positions = info["positions"].copy()
    saved_velocities = info["velocities"].copy()

    positions = info["positions"]
    velocities = info["velocities"]
    colors = info["colors"]

    z_coord = torch.zeros(1, D_MAX)
    n_obj = min(D_T, len(positions))
    z_coord[0, :n_obj] = torch.tensor(positions[:n_obj], dtype=torch.float32)

    z_dyn = torch.zeros(1, D_MAX)
    for i in range(min(D_T, len(colors))):
        z_dyn[0, i] = torch.tensor(np.mean(colors[i]), dtype=torch.float32)

    z_pred_coord = torch.zeros(1, D_MAX)
    n_prev = min(D_T, len(prev_positions))
    predicted = prev_positions[:n_prev] + prev_velocities[:n_prev] * 1.0
    z_pred_coord[0, :n_prev] = torch.tensor(predicted, dtype=torch.float32)

    z_pred_dyn = z_dyn.clone()
    centroids = z_coord.clone()

    step_surprises = []
    for c in range(D_T):
        surprise = torch.mean((z_pred_coord[:, c] - z_coord[:, c])**2).item()
        step_surprises.append(surprise)
    surprises_all.append(step_surprises)

    max_s = max(step_surprises)
    if max_s > 1000:
        max_surprise_steps.append((step, positions, velocities, prev_positions, prev_velocities, predicted, step_surprises))

    action, locus, surprises = controller.get_action(
        None, history[-1], info, z_pred_coord, z_coord, z_pred_dyn, z_dyn, D_T, centroids)

    obs, info = env.step(action)
    history.append(obs)

    prev_positions = saved_positions
    prev_velocities = saved_velocities

surprises_all = np.array(surprises_all)
print(f"Total steps: {len(surprises_all)}")
print(f"Shape: {surprises_all.shape}")
print(f"Mean per channel: {surprises_all.mean(axis=0)}")
print(f"Max per channel: {surprises_all.max(axis=0)}")
print(f"Min per channel: {surprises_all.min(axis=0)}")
print(f"Std per channel: {surprises_all.std(axis=0)}")
print(f"Overall mean: {surprises_all.mean():.4f}")
print(f"Overall max: {surprises_all.max():.4f}")

# Count steps with surprise > 1000
for c in range(D_T):
    count = np.sum(surprises_all[:, c] > 1000)
    print(f"Channel {c}: {count} steps with surprise > 1000")

# Show top 10 highest surprise steps
flat_surprises = surprises_all.flatten()
flat_indices = np.argsort(flat_surprises)[-10:]
print("\nTop 10 highest surprise values:")
for idx in reversed(flat_indices):
    step = idx // D_T
    ch = idx % D_T
    print(f"  step={step}, ch={ch}, surprise={flat_surprises[idx]:.4f}")

# Show first 5 high-surprise steps in detail
print(f"\nFirst 5 high-surprise steps (detail):")
for i, (step, pos, vel, prev_pos, prev_vel, pred, surp) in enumerate(max_surprise_steps[:5]):
    print(f"  step={step}:")
    print(f"    positions={pos}, velocities={vel}")
    print(f"    prev_positions={prev_pos}, prev_velocities={prev_vel}")
    print(f"    predicted={pred}")
    print(f"    surprises={surp}")
