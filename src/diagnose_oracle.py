#!/usr/bin/env python3
"""Diagnostic script to trace ORACLE surprise values."""
import os, sys, collections
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment import PhysicsSandbox
from src.motor import CLTSMotorController

D_T = 3
D_MAX = 8
WARMUP_STEPS = 3

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

print("After warmup:")
print(f"  prev_positions = {prev_positions}")
print(f"  prev_velocities = {prev_velocities}")
print(f"  info positions = {info['positions']}")
print(f"  info velocities = {info['velocities']}")
print()

for step in range(20):
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
    
    print(f"Step {step}:")
    print(f"  positions = {positions}")
    print(f"  velocities = {velocities}")
    print(f"  prev_positions = {prev_positions}")
    print(f"  prev_velocities = {prev_velocities}")
    print(f"  predicted = {predicted}")
    
    for c in range(D_T):
        surprise = torch.mean((z_pred_coord[:, c] - z_coord[:, c])**2).item()
        print(f"  c={c}: z_coord={z_coord[0,c].item():.4f}, z_pred={z_pred_coord[0,c].item():.4f}, surprise={surprise:.4f}")
    
    action, locus, surprises = controller.get_action(
        None, history[-1], info, z_pred_coord, z_coord, z_pred_dyn, z_dyn, D_T, centroids)
    print(f"  controller locus={locus}, surprises={surprises}")
    print(f"  controller mu={controller.mu[:D_T]}, sigma={controller.sigma[:D_T]}")
    
    obs, info = env.step(action)
    history.append(obs)
    prev_positions = info["positions"].copy()
    prev_velocities = info["velocities"].copy()
    print()
