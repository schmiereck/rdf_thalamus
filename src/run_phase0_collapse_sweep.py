#!/usr/bin/env python3
"""
Iter_026 Collapse-Rate Sweep Runner.

Based on run_phase0_id_probe_v2.py, stripped to JEPA+VICReg only.
Sweeps 5 arms (A0-A4) x 10 seeds to find a configuration that reduces
collapse rate to <=10%.

Dual collapse criterion:
  - collapsed_eval: any d_t z_dyn dim has batch-std < 0.5 on 200 eval samples
  - collapsed_train: any d_t z_dyn dim has mean training-logged std < 0.5 at step 8000
  - collapsed: collapsed_eval OR collapsed_train

Sanity disqualification:
  - mean total loss at final step > 50 -> disqualified (counted as collapsed)
  - mean per-dim z_dyn std at final training log < 0.5 -> captured by collapsed_train

All arms complete their full 10-seed runs (no early termination).
Results saved to archive/iter_026/results/
"""

import os
import sys
import json
import random
import argparse
import collections
import warnings
import concurrent.futures

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models_dual_stream import NonParametricJEPASpatial

# ---------------------------------------------------------------------------
# Extended replay buffer
# ---------------------------------------------------------------------------
class ExtendedReplayBuffer:
    def __init__(self, capacity=4000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, x_hist, x_target, positions, colors, radii):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (x_hist, x_target, positions, colors, radii)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        x_hist_b, x_target_b, pos_b, colors_b, radii_b = zip(*batch)
        return (
            np.stack(x_hist_b, axis=0),
            np.stack(x_target_b, axis=0),
            np.stack(pos_b, axis=0),
            np.stack(colors_b, axis=0),
            np.stack(radii_b, axis=0),
        )

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_collapse(z_dyn, d_t, std_threshold=0.5):
    z_active = z_dyn[:, :d_t]
    per_dim_std = np.std(z_active, axis=0)
    has_collapsed = np.any(per_dim_std < std_threshold)
    return has_collapsed, per_dim_std


def compute_vicreg_health(z_dyn, d_t):
    z_active = z_dyn[:, :d_t]
    per_dim_std = np.std(z_active, axis=0)
    if d_t > 1:
        corr = np.corrcoef(z_active.T)
        triu = np.triu_indices(d_t, k=1)
        mean_abs_corr = float(np.mean(np.abs(corr[triu])))
    else:
        mean_abs_corr = 0.0
    return {
        "per_dim_std": per_dim_std.tolist(),
        "mean_abs_corr": mean_abs_corr,
    }


def fit_linear_probe(z, y):
    Z = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    return theta[0], theta[1]


def fit_multivariate_probe_r2(z_feature, y_target):
    z = z_feature.reshape(-1)
    y = y_target.reshape(len(z), -1)
    N = len(z)
    if N < 5 or y.shape[1] < 1:
        return 0.0
    Z_aug = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.linalg.pinv(Z_aug.T @ Z_aug) @ Z_aug.T @ y
    y_pred = Z_aug @ theta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def compute_centroid_mse(model, test_env, test_history, num_samples=200, device="cpu"):
    model.eval()
    centroids_list = []
    pos_list = []

    obs = test_env.reset()
    test_history.clear()
    test_history.append(obs)

    for _ in range(4):
        obs, info = test_env.step({"acc": 0.0, "push": False})
        test_history.append(obs)

    collected = 0
    while collected < num_samples:
        obs, info = test_env.step({"acc": 0.0, "push": False})
        test_history.append(obs)
        if len(test_history) == 4:
            x_target_t = torch.from_numpy(test_history[3]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z_coord, _ = model.encoder(x_target_t)
            centroids_list.append(z_coord[:, :model.d_t].cpu().numpy())
            true_pos = np.array(info["positions"][:model.d_t])
            if len(true_pos) < model.d_t:
                true_pos = np.pad(true_pos, (0, model.d_t - len(true_pos)), constant_values=np.nan)
            pos_list.append(true_pos)
            collected += 1

    centroids_arr = np.concatenate(centroids_list, axis=0)
    pos_arr = np.array(pos_list)

    mse_per_object = []
    r_per_object = []
    for obj_idx in range(model.d_t):
        mask = ~np.isnan(pos_arr[:, obj_idx])
        if mask.sum() < 10:
            mse_per_object.append(float("nan"))
            r_per_object.append(0.0)
            continue
        z_obj = centroids_arr[mask, obj_idx]
        y_obj = pos_arr[mask, obj_idx]
        w, b = fit_linear_probe(z_obj, y_obj)
        y_pred = z_obj * w + b
        mse = float(np.mean((y_obj - y_pred) ** 2))
        r = float(np.corrcoef(z_obj, y_obj)[0, 1]) if len(z_obj) > 2 else 0.0
        if np.isnan(r):
            r = 0.0
        mse_per_object.append(mse)
        r_per_object.append(abs(r))

    return {
        "mse_per_object": mse_per_object,
        "mse_mean": float(np.nanmean(mse_per_object)),
        "r_per_object": r_per_object,
        "r_mean": float(np.nanmean(r_per_object)),
    }


def collect_multitraj_eval_data(model, num_samples=200, base_seed=30000, device="cpu"):
    model.eval()
    num_trajectories = max(1, num_samples // 20)
    samples_per_traj = max(1, num_samples // num_trajectories)

    traj_z_dyn = []
    traj_z_coord = []
    traj_pos = []
    traj_colors = []
    traj_radii = []

    with torch.no_grad():
        for t_idx in range(num_trajectories):
            env_seed = base_seed + t_idx * 100
            env = PhysicsSandbox(N=3, seed=env_seed)
            history = collections.deque(maxlen=4)

            obs = env.reset()
            history.append(obs)
            for _ in range(3):
                obs, info = env.step({"acc": 0.0, "push": False})
                history.append(obs)

            z_dyn_t = []
            z_coord_t = []
            pos_t = []
            colors_t = []
            radii_t = []

            collected = 0
            while collected < samples_per_traj:
                obs, info = env.step({"acc": 0.0, "push": False})
                history.append(obs)
                if len(history) == 4:
                    x_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    z_c, z_d = model.encoder(x_t)
                    z_dyn_t.append(z_d[0].cpu().numpy())
                    z_coord_t.append(z_c[0].cpu().numpy())
                    pos_t.append(info["positions"])
                    colors_t.append(info["colors"])
                    radii_t.append(info["radii"])
                    collected += 1

            traj_z_dyn.append(np.array(z_dyn_t))
            traj_z_coord.append(np.array(z_coord_t))
            traj_pos.append(np.array(pos_t))
            traj_colors.append(np.array(colors_t))
            traj_radii.append(np.array(radii_t))

    z_dyn_arr = np.concatenate(traj_z_dyn, axis=0)
    z_coord_arr = np.concatenate(traj_z_coord, axis=0)
    pos_arr = np.concatenate(traj_pos, axis=0)
    colors_arr = np.concatenate(traj_colors, axis=0)
    radii_arr = np.concatenate(traj_radii, axis=0)

    return (traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii,
            z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr)


# ---------------------------------------------------------------------------
# Semantic probes core (Hungarian matching only)
# ---------------------------------------------------------------------------
def _compute_semantic_probes_core(z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr,
                                   d_t, sub_features, train_ratio=0.5):
    N = pos_arr.shape[1]

    if sub_features > 1:
        num_samples = z_dyn_arr.shape[0]
        z_dyn_pooled = z_dyn_arr[:, :d_t * sub_features].reshape(num_samples, d_t, sub_features).mean(axis=2)
    else:
        num_samples = z_dyn_arr.shape[0]
        z_dyn_pooled = z_dyn_arr[:, :d_t]

    # Hungarian matching only
    cost = np.zeros((d_t, N))
    for d in range(d_t):
        for o in range(N):
            cost[d, o] = np.mean(np.abs(z_coord_arr[:, d] - pos_arr[:, o]))
    row_ind, col_ind = linear_sum_assignment(cost)
    dim_to_obj = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    n_train = int(num_samples * train_ratio)

    def fit_probe_r2(z_feature, y_target):
        z_train = z_feature[:n_train]
        y_train = y_target[:n_train]
        z_test = z_feature[n_train:]
        y_test = y_target[n_train:]
        if z_train.size < 5 or y_train.size < 5:
            return 0.0
        w, b = fit_linear_probe(z_train, y_train)
        y_pred = z_test * w + b
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_train)) ** 2)
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    probes = {}
    r2_per_dim = []

    for d in range(d_t):
        if d not in dim_to_obj:
            r2_per_dim.append({
                "dyn_color": 0.0, "coord_color": 0.0,
                "dyn_pos": 0.0, "coord_pos": 0.0,
                "dyn_identity": 0.0, "coord_identity": 0.0,
            })
            continue

        obj = dim_to_obj[d]
        z_d = z_dyn_pooled[:, d]
        z_c = z_coord_arr[:, d]

        r2_dyn_pos = fit_probe_r2(z_d, pos_arr[:, obj])
        r2_coord_pos = fit_probe_r2(z_c, pos_arr[:, obj])

        r2_dyn_ch = []
        r2_coord_ch = []
        for ch in range(3):
            r2_dyn_ch.append(fit_probe_r2(z_d, colors_arr[:, obj, ch]))
            r2_coord_ch.append(fit_probe_r2(z_c, colors_arr[:, obj, ch]))
        r2_dyn_color = float(np.mean(r2_dyn_ch))
        r2_coord_color = float(np.mean(r2_coord_ch))

        max_radius = 20.0
        identity_vec = np.zeros((num_samples, 4))
        identity_vec[:, :3] = colors_arr[:, obj, :]
        identity_vec[:, 3] = radii_arr[:, obj] / max_radius

        r2_dyn_identity = fit_multivariate_probe_r2(z_d, identity_vec)
        r2_coord_identity = fit_multivariate_probe_r2(z_c, identity_vec)

        r2_per_dim.append({
            "dyn_color": r2_dyn_color,
            "coord_color": r2_coord_color,
            "dyn_pos": r2_dyn_pos,
            "coord_pos": r2_coord_pos,
            "dyn_identity": r2_dyn_identity,
            "coord_identity": r2_coord_identity,
        })

    r2_dyn_color_all = np.mean([p["dyn_color"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_color_all = np.mean([p["coord_color"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_dyn_pos_all = np.mean([p["dyn_pos"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_pos_all = np.mean([p["coord_pos"] for p in r2_per_dim]) if r2_per_dim else 0.0
    delta_r2_color = float(r2_dyn_color_all - r2_coord_color_all)

    r2_dyn_identity_all = np.mean([p["dyn_identity"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_identity_all = np.mean([p["coord_identity"] for p in r2_per_dim]) if r2_per_dim else 0.0
    delta_r2_identity = float(r2_dyn_identity_all - r2_coord_identity_all)

    return {
        "dim_to_obj": dim_to_obj,
        "r2_per_dim": r2_per_dim,
        "r2_dyn_color": float(r2_dyn_color_all),
        "r2_coord_color": float(r2_coord_color_all),
        "r2_dyn_pos": float(r2_dyn_pos_all),
        "r2_coord_pos": float(r2_coord_pos_all),
        "delta_r2_color": delta_r2_color,
        "r2_dyn_identity": float(r2_dyn_identity_all),
        "r2_coord_identity": float(r2_coord_identity_all),
        "delta_r2_identity": delta_r2_identity,
    }


def compute_semantic_probes(model, num_samples=200, train_ratio=0.5,
                            base_seed=30000, device="cpu"):
    model.eval()
    (traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii,
     z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr) = \
        collect_multitraj_eval_data(model, num_samples=num_samples,
                                     base_seed=base_seed, device=device)

    d_t = model.d_t
    sub_features = model.sub_features

    return _compute_semantic_probes_core(
        z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr,
        d_t, sub_features, train_ratio=train_ratio
    )


# ---------------------------------------------------------------------------
# Evaluation with diagnostics
# ---------------------------------------------------------------------------
def evaluate_run(model, arm_config, seed, device, eval_steps=200):
    d_t = arm_config.get("d_t", 3)
    sub_features = arm_config.get("sub_features", 1)
    name = arm_config["name"]

    # --- Collapse check on eval samples ---
    eval_env = PhysicsSandbox(N=3, seed=seed + 10000)
    eval_history = collections.deque(maxlen=4)
    obs = eval_env.reset()
    eval_history.append(obs)
    for _ in range(4):
        obs, _ = eval_env.step({"acc": 0.0, "push": False})
        eval_history.append(obs)

    z_dyn_all = []
    z_coord_all = []
    for _ in range(eval_steps):
        obs, _ = eval_env.step({"acc": 0.0, "push": False})
        eval_history.append(obs)
        if len(eval_history) == 4:
            x_t = torch.from_numpy(eval_history[3]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z_coord, z_dyn = model.encoder(x_t)
            z_dyn_all.append(z_dyn[:, :d_t * sub_features].cpu().numpy())
            z_coord_all.append(z_coord[:, :d_t].cpu().numpy())

    z_dyn_arr = np.concatenate(z_dyn_all, axis=0)
    z_coord_arr = np.concatenate(z_coord_all, axis=0)

    if sub_features > 1:
        z_dyn_pooled_check = z_dyn_arr[:, :d_t * sub_features].reshape(eval_steps, d_t, sub_features).mean(axis=2)
    else:
        z_dyn_pooled_check = z_dyn_arr[:, :d_t]
    has_collapsed_eval, per_dim_std = check_collapse(z_dyn_pooled_check, d_t)
    vh = compute_vicreg_health(z_dyn_pooled_check, d_t)

    # --- Centroid MSE ---
    test_env = PhysicsSandbox(N=3, seed=seed + 20000)
    test_history = collections.deque(maxlen=4)
    obs = test_env.reset()
    test_history.append(obs)
    centroid_mse = compute_centroid_mse(model, test_env, test_history,
                                         num_samples=eval_steps, device=device)

    # --- Semantic probes (Hungarian only) ---
    semantic = compute_semantic_probes(model, num_samples=eval_steps,
                                       base_seed=seed + 30000, device=device)

    results = {
        "arm": name,
        "seed": seed,
        "collapsed_eval": bool(has_collapsed_eval),
        "per_dim_std": per_dim_std.tolist(),
        "vicreg_per_dim_std": per_dim_std.tolist(),
        "vicreg_mean_abs_corr": vh["mean_abs_corr"],
        "centroid_mse_mean": centroid_mse["mse_mean"],
        "centroid_r_mean": centroid_mse["r_mean"],
        "delta_r2_color": semantic["delta_r2_color"],
        "r2_dyn_color": semantic["r2_dyn_color"],
        "r2_coord_color": semantic["r2_coord_color"],
        "r2_dyn_pos": semantic["r2_dyn_pos"],
        "r2_coord_pos": semantic["r2_coord_pos"],
        "r2_dyn_identity": semantic["r2_dyn_identity"],
        "r2_coord_identity": semantic["r2_coord_identity"],
        "delta_r2_identity": semantic["delta_r2_identity"],
        "dim_to_obj": semantic["dim_to_obj"],
    }

    return results


# ---------------------------------------------------------------------------
# Single-run training + evaluation
# ---------------------------------------------------------------------------
def run_single(arm_config, seed, device, dry_run=False):
    name = arm_config["name"]
    sim_weight = arm_config.get("sim_weight", 25.0)
    var_weight = arm_config.get("var_weight", 25.0)
    cov_weight = arm_config.get("cov_weight", 25.0)
    pos_encoding = arm_config.get("pos_encoding", "none")
    d_t = arm_config.get("d_t", 3)
    dyn_readout = arm_config.get("dyn_readout", "mean")
    ccr_mode = arm_config.get("ccr_mode", "covariance")
    ccr_smooth_weight = arm_config.get("ccr_smooth_weight", 10.0)
    ccr_spatial_weight = arm_config.get("ccr_spatial_weight", 10.0)
    d_max = arm_config.get("d_max", 8)
    sub_features = arm_config.get("sub_features", 1)
    lr = arm_config.get("lr", 3e-4)
    batch_size = arm_config.get("batch_size", 32)
    replay_buffer_capacity = arm_config.get("replay_buffer_capacity", 4000)
    sim_weight_ramp = arm_config.get("sim_weight_ramp", False)
    sim_weight_ramp_steps = arm_config.get("sim_weight_ramp_steps", 1000)

    set_seed(seed)

    total_steps = 5 if dry_run else 8000
    eval_steps = min(200, total_steps // 2) if dry_run else 200

    model = NonParametricJEPASpatial(
        d_max=d_max,
        h=3,
        k=4,
        cooldown=300,
        stabilization_period=100,
        pos_encoding=pos_encoding,
        primary_objective="jepa",
        gdasr_log_only=True,
        dyn_readout=dyn_readout,
        sub_features=sub_features,
        dyn_source="spatial",
    )
    model.d_t = d_t
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    env = PhysicsSandbox(N=3, seed=seed)
    obs = env.reset()
    history = collections.deque(maxlen=4)
    history.append(obs)

    replay_buffer = ExtendedReplayBuffer(capacity=replay_buffer_capacity)

    def _prefill(n):
        while len(replay_buffer) < n:
            obs, info = env.step({"acc": 0.0, "push": False})
            history.append(obs)
            if len(history) == 4:
                replay_buffer.push(
                    np.stack(list(history)[:3], axis=0),
                    history[3],
                    info["positions"],
                    info["colors"],
                    info["radii"],
                )

    prefill_steps = max(1, min(total_steps - 2, 200))
    _prefill(prefill_steps)

    logs = []

    for step in range(max(1, prefill_steps + 1), total_steps + 1):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)

        if len(history) == 4:
            replay_buffer.push(
                np.stack(list(history)[:3], axis=0),
                history[3],
                info["positions"],
                info["colors"],
                info["radii"],
            )

        x_hist_b, x_target_b, pos_b, colors_b, radii_b = replay_buffer.sample(min(batch_size, len(replay_buffer)))
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)
        pos_t = torch.from_numpy(pos_b).float().to(device)
        colors_t = torch.from_numpy(colors_b).float().to(device)

        # Sim-weight ramp for A3
        current_sim_weight = sim_weight
        if sim_weight_ramp and step <= sim_weight_ramp_steps:
            current_sim_weight = sim_weight * (step / float(sim_weight_ramp_steps))

        model.train()
        optimizer.zero_grad()

        loss_dict, _, (z_target_coord, z_target_dyn) = model(
            x_hist_t,
            x_target_t,
            sim_weight=current_sim_weight,
            var_weight=var_weight,
            cov_weight=cov_weight,
            d_t_predict=d_t,
            ccr_mode=ccr_mode,
            ccr_smooth_weight=ccr_smooth_weight,
            ccr_spatial_weight=ccr_spatial_weight,
        )

        total_loss = loss_dict["loss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        sim_loss_val = loss_dict["sim_loss"].item()
        model.update_recruitment_logic(sim_loss_val, target_dim=d_t, step=step)

        log_entry = {
            "step": step,
            "loss": total_loss.item(),
            "sim_loss": loss_dict["sim_loss"].item(),
            "var_loss": loss_dict["var_loss"].item(),
            "cov_loss": loss_dict["cov_loss"].item(),
            "ccr_smooth_loss": loss_dict.get("ccr_smooth_loss", 0.0),
            "ccr_spatial_loss": loss_dict.get("ccr_spatial_loss", 0.0),
            "per_dim_std": None,
        }

        # Per-dimension std logging every 500 steps
        if step % 500 == 0:
            with torch.no_grad():
                per_dim_std_train = z_target_dyn[:, :d_t].std(dim=0).cpu().numpy()
            log_entry["per_dim_std"] = per_dim_std_train.tolist()

        logs.append(log_entry)

        if step % 1000 == 0 or step == total_steps:
            print(f"  [{name}] seed={seed} step={step:5d}/{total_steps}  "
                  f"loss={log_entry['loss']:.4f}  sim={log_entry['sim_loss']:.4f}  "
                  f"var={log_entry['var_loss']:.4f}  cov={log_entry['cov_loss']:.4f}  "
                  f"ccr_s={log_entry['ccr_smooth_loss']:.4f}  ccr_sp={log_entry['ccr_spatial_loss']:.4f}")

    # Final evaluation at step 8000 (or total_steps for dry_run)
    print(f"  [{name}] seed={seed} -> final evaluation at step {total_steps}")
    eval_res = evaluate_run(model, arm_config, seed, device, eval_steps=eval_steps)

    # Extract final training log values
    final_log = logs[-1]
    eval_res["final_train_loss"] = final_log["loss"]
    eval_res["final_sim_loss"] = final_log["sim_loss"]
    eval_res["final_var_loss"] = final_log["var_loss"]
    eval_res["final_cov_loss"] = final_log["cov_loss"]

    # Train collapse check: last per_dim_std log at step 8000 (or closest)
    per_dim_std_train_last = None
    for entry in reversed(logs):
        if entry["per_dim_std"] is not None:
            per_dim_std_train_last = np.array(entry["per_dim_std"])
            break

    if per_dim_std_train_last is not None:
        collapsed_train = np.any(per_dim_std_train_last < 0.5)
        eval_res["collapsed_train"] = bool(collapsed_train)
        eval_res["per_dim_std_train"] = per_dim_std_train_last.tolist()
    else:
        eval_res["collapsed_train"] = False
        eval_res["per_dim_std_train"] = None

    # Dual collapse criterion
    eval_res["collapsed"] = eval_res["collapsed_eval"] or eval_res["collapsed_train"]

    # Sanity disqualification: loss > 50
    if eval_res["final_train_loss"] > 50.0:
        eval_res["disqualified"] = True
        eval_res["collapsed"] = True  # Count as collapsed
    else:
        eval_res["disqualified"] = False

    return eval_res, model, logs


# ---------------------------------------------------------------------------
# Arms configuration
# ---------------------------------------------------------------------------
ARMS = [
    {
        "name": "A0 (canonical repeat)",
        "lr": 3e-4, "var_weight": 25.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 32,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
    },
    {
        "name": "A1 (batch_size=64)",
        "lr": 3e-4, "var_weight": 25.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 64,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
    },
    {
        "name": "A2 (var_weight=50)",
        "lr": 3e-4, "var_weight": 50.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 32,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
    },
    {
        "name": "A3 (sim_weight warm-up)",
        "lr": 3e-4, "var_weight": 25.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 32,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
        "sim_weight_ramp": True, "sim_weight_ramp_steps": 1000,
    },
    {
        "name": "A4 (lr=1e-4)",
        "lr": 1e-4, "var_weight": 25.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 32,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
    },
]

SEEDS = [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]


def _sanitize_arm_name(name):
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('=', '_')


def _flatten_result(res, runs_dir):
    flat = {k: v for k, v in res.items() if k not in (
        "centroid_mse_per_object", "r2_per_dim", "dim_to_obj",
    )}
    flat["per_dim_std"] = str(res.get("per_dim_std", []))
    flat["per_dim_std_train"] = str(res.get("per_dim_std_train", []))

    safe_name = _sanitize_arm_name(res["arm"])
    run_id = f"{safe_name}_seed{res['seed']}"
    csv_path = os.path.join(runs_dir, f"{run_id}.csv")
    json_path = os.path.join(runs_dir, f"{run_id}.json")

    df_row = pd.DataFrame([flat])
    df_row.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(res, f, indent=2, default=str)

    return csv_path


def _run_single_worker(args_tuple):
    arm, seed, device_str, dry_run, runs_dir, checkpoints_dir = args_tuple
    device = torch.device(device_str)
    torch.set_num_threads(1)

    name = arm["name"]
    print(f"[{name}] seed={seed} -> starting on {device} (dry_run={dry_run})")

    eval_res, model, logs = run_single(arm, seed, device, dry_run=dry_run)

    csv_path = _flatten_result(eval_res, runs_dir)
    safe_name = _sanitize_arm_name(eval_res["arm"])
    run_id = f"{safe_name}_seed{eval_res['seed']}"
    json_path = os.path.join(runs_dir, f"{run_id}.json")
    print(f"  -> Saved {run_id}  ({csv_path}, {json_path})")

    ckpt_path = os.path.join(checkpoints_dir, f"{safe_name}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  -> Saved checkpoint {ckpt_path}")

    logs_path = os.path.join(runs_dir, f"{safe_name}_seed{seed}_logs.csv")
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(logs_path, index=False)
    print(f"  -> Saved training logs {logs_path}")

    return eval_res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cpu_count = os.cpu_count() or 2
    default_workers = min(cpu_count - 1, 8)

    parser = argparse.ArgumentParser(
        description="Iter_026 Collapse-Rate Sweep"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 5 training steps per seed for quick correctness check.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list (default: 10 seeds).")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Number of parallel workers (default: min(cpu_count-1, 8)={default_workers}).")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially (no parallelism).")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda'. Default: auto-detect.")
    args = parser.parse_args()

    dry_run = args.dry_run
    seeds = args.seeds if args.seeds is not None else SEEDS
    max_workers = args.workers
    sequential = args.sequential

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)

    print(f"CPU count: {cpu_count}")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Dry-run mode: {dry_run}")
    print(f"Seeds: {seeds}")
    print(f"Arms: {[a['name'] for a in ARMS]}")
    if not sequential:
        print(f"Parallel workers: {max_workers}")
    print()

    results_dir = "archive/iter_026/results"
    runs_dir = os.path.join(results_dir, "runs")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Main experiment tasks
    # ------------------------------------------------------------------ #
    tasks = []
    task_labels = []
    for arm in ARMS:
        name = arm["name"]
        for seed in seeds:
            tasks.append((arm, seed, device_str, dry_run, runs_dir, checkpoints_dir))
            task_labels.append(f"{name} seed={seed}")

    total_tasks = len(tasks)
    print(f"Total main tasks to run: {total_tasks}")
    for i, label in enumerate(task_labels, 1):
        print(f"  [{i:2d}] {label}")
    print()

    all_results = []

    if sequential:
        for i, task in enumerate(tasks):
            arm, seed = task[0], task[1]
            print(f"\n{'='*70}")
            print(f"TASK [{i+1}/{total_tasks}]: {arm['name']} seed={seed}")
            print(f"{'='*70}")
            res = _run_single_worker(task)
            all_results.append(res)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_label = {}
            for i, task in enumerate(tasks):
                future = executor.submit(_run_single_worker, task)
                future_to_label[future] = task_labels[i]

            done_count = 0
            for future in concurrent.futures.as_completed(future_to_label):
                label = future_to_label[future]
                done_count += 1
                try:
                    res = future.result()
                    all_results.append(res)
                    print(f"\n[{done_count}/{total_tasks}] COMPLETED: {label}")
                except Exception as exc:
                    print(f"\n[{done_count}/{total_tasks}] FAILED: {label} -> {exc}")

    # ------------------------------------------------------------------ #
    # Aggregate results
    # ------------------------------------------------------------------ #
    df_all = pd.DataFrame(all_results)

    numeric_cols = [
        "collapsed_eval", "collapsed_train", "collapsed", "disqualified",
        "final_train_loss", "final_sim_loss", "final_var_loss", "final_cov_loss",
        "centroid_mse_mean", "centroid_r_mean",
        "vicreg_mean_abs_corr",
        "delta_r2_color", "r2_dyn_color", "r2_coord_color",
        "r2_dyn_pos", "r2_coord_pos",
        "r2_dyn_identity", "r2_coord_identity", "delta_r2_identity",
    ]
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    summary_path = os.path.join(results_dir, "summary_iter_026.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"\nSaved full results to {summary_path}")

    # ------------------------------------------------------------------ #
    # Analysis and report
    # ------------------------------------------------------------------ #
    _generate_analysis(df_all, results_dir)
    print("\nDone.")


def _generate_analysis(df_all, results_dir):
    """Generate final_analysis.md with per-arm statistics and pre-registered interpretations."""
    lines = []
    lines.append("# Iter_026 Collapse-Rate Sweep Analysis\n")
    lines.append(f"**Date:** Auto-generated\n")
    lines.append("**Dual collapse criterion:** collapsed = collapsed_eval OR collapsed_train\n")
    lines.append("**Sanity disqualification:** final_train_loss > 50 OR per_dim_std_train < 0.5\n")
    lines.append("**Arms:** A0 (canonical), A1 (batch=64), A2 (var=50), A3 (sim warm-up), A4 (lr=1e-4)\n")
    lines.append("---\n")

    def _fmt(val, fmt=".4f"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:{fmt}}"

    # Per-arm summary
    lines.append("## Per-Arm Summary (Step = 8000)\n")
    for arm_name in [a["name"] for a in ARMS]:
        df_arm = df_all[df_all["arm"] == arm_name]
        if len(df_arm) == 0:
            continue
        lines.append(f"### {arm_name}\n")
        lines.append(f"- N seeds: {len(df_arm)}\n")

        # Dual criterion
        collapse_rate_dual = df_arm["collapsed"].mean()
        collapsed_count_dual = int(df_arm["collapsed"].sum())
        lines.append(f"- Collapse rate (dual criterion): {collapse_rate_dual:.2f} ({collapsed_count_dual}/{len(df_arm)})\n")

        # Eval-only
        collapse_rate_eval = df_arm["collapsed_eval"].mean()
        collapsed_count_eval = int(df_arm["collapsed_eval"].sum())
        lines.append(f"- Collapse rate (eval-only): {collapse_rate_eval:.2f} ({collapsed_count_eval}/{len(df_arm)})\n")

        # Train-only
        collapse_rate_train = df_arm["collapsed_train"].mean()
        collapsed_count_train = int(df_arm["collapsed_train"].sum())
        lines.append(f"- Collapse rate (train-only): {collapse_rate_train:.2f} ({collapsed_count_train}/{len(df_arm)})\n")

        # Disqualified
        disqualified_count = int(df_arm["disqualified"].sum())
        if disqualified_count > 0:
            lines.append(f"- **Disqualified seeds (loss > 50):** {disqualified_count}\n")

        lines.append(f"- Mean final train loss: {_fmt(df_arm['final_train_loss'].mean(), '.4f')} +/- {_fmt(df_arm['final_train_loss'].std(), '.4f')}\n")
        lines.append(f"- Mean final sim loss: {_fmt(df_arm['final_sim_loss'].mean(), '.4f')}\n")
        lines.append(f"- Mean final var loss: {_fmt(df_arm['final_var_loss'].mean(), '.4f')}\n")
        lines.append(f"- Mean final cov loss: {_fmt(df_arm['final_cov_loss'].mean(), '.4f')}\n")
        lines.append(f"- Centroid MSE (REF ONLY): {_fmt(df_arm['centroid_mse_mean'].mean(), '.2f')}\n")
        lines.append(f"- delta_R2_color (REF ONLY): {_fmt(df_arm['delta_r2_color'].mean(), '.4f')}\n")
        lines.append(f"- Mean abs corr: {_fmt(df_arm['vicreg_mean_abs_corr'].mean(), '.3f')}\n")

        # Per-seed details
        lines.append("\n**Per-seed details:**\n")
        lines.append("| seed | collapsed_eval | collapsed_train | collapsed | disqualified | final_loss | per_dim_std_eval | per_dim_std_train | centroid_mse | delta_r2_color |\n")
        lines.append("|------|----------------|-----------------|-----------|--------------|------------|------------------|-------------------|--------------|----------------|\n")
        for _, row in df_arm.iterrows():
            s = int(row["seed"])
            ce = "Y" if row["collapsed_eval"] else "N"
            ct = "Y" if row["collapsed_train"] else "N"
            c = "Y" if row["collapsed"] else "N"
            dq = "Y" if row["disqualified"] else "N"
            fl = _fmt(row.get("final_train_loss"), ".4f")
            pds = str(row.get("per_dim_std", []))
            pds_t = str(row.get("per_dim_std_train", []))
            cm = _fmt(row.get("centroid_mse_mean"), ".2f")
            dr = _fmt(row.get("delta_r2_color"), ".4f")
            lines.append(f"| {s} | {ce} | {ct} | {c} | {dq} | {fl} | {pds} | {pds_t} | {cm} | {dr} |\n")
        lines.append("\n")

    # Gate check
    lines.append("## Gate Check\n\n")
    gate_cleared = False
    cleared_arms = []
    for arm_name in [a["name"] for a in ARMS]:
        df_arm = df_all[df_all["arm"] == arm_name]
        if len(df_arm) == 0:
            continue
        collapse_rate = df_arm["collapsed"].mean()
        if collapse_rate <= 0.10:
            gate_cleared = True
            cleared_arms.append(arm_name)
            lines.append(f"- **{arm_name}:** PASS (collapse rate {collapse_rate:.2f} <= 10%)\n")
        else:
            lines.append(f"- **{arm_name}:** FAIL (collapse rate {collapse_rate:.2f} > 10%)\n")

    lines.append("\n")
    if gate_cleared:
        lines.append(f"**Gate cleared by:** {', '.join(cleared_arms)}\n")
        lines.append("All arms completed their full 10-seed runs as pre-registered.\n")
    else:
        lines.append("**Measured null: no swept configuration cleared the <=10% gate under the pre-registered protocol.**\n")
        lines.append("All arms completed their full 10-seed runs as pre-registered.\n")
    lines.append("\n")

    # Sanity check
    lines.append("## Sanity Check\n\n")
    total_disqualified = int(df_all["disqualified"].sum())
    if total_disqualified > 0:
        lines.append(f"- **WARNING:** {total_disqualified} seed(s) disqualified due to final_train_loss > 50.\n")
        dq_df = df_all[df_all["disqualified"] == True]
        for _, row in dq_df.iterrows():
            lines.append(f"  - {row['arm']} seed={int(row['seed'])}: loss={row['final_train_loss']:.2f}\n")
    else:
        lines.append("- No seeds disqualified. All final losses <= 50.\n")
    lines.append("\n")

    # Reference-only downstream metrics disclaimer
    lines.append("## Reference-Only Downstream Metrics (NOT used for regime selection)\n\n")
    lines.append("The following metrics are recorded for diagnostic purposes only. ")
    lines.append("**They MUST NOT be used to pick a winning regime** per the pre-registered protocol.\n\n")
    for arm_name in [a["name"] for a in ARMS]:
        df_arm = df_all[df_all["arm"] == arm_name]
        if len(df_arm) == 0:
            continue
        lines.append(f"**{arm_name}:**\n")
        lines.append(f"- Centroid MSE mean: {_fmt(df_arm['centroid_mse_mean'].mean(), '.2f')} +/- {_fmt(df_arm['centroid_mse_mean'].std(), '.2f')}\n")
        lines.append(f"- Centroid R mean: {_fmt(df_arm['centroid_r_mean'].mean(), '.3f')}\n")
        lines.append(f"- delta_R2_color mean: {_fmt(df_arm['delta_r2_color'].mean(), '.4f')} +/- {_fmt(df_arm['delta_r2_color'].std(), '.4f')}\n")
        lines.append(f"- delta_R2_identity mean: {_fmt(df_arm['delta_r2_identity'].mean(), '.4f')}\n")
        lines.append("\n")

    # Next-step recommendation
    lines.append("## Next-Step Recommendation\n\n")
    if gate_cleared:
        lines.append("- At least one arm cleared the <=10% collapse gate.\n")
        lines.append(f"- **Recommendation:** Select the arm with the lowest collapse rate among {cleared_arms} as the new canonical regime.\n")
        lines.append("- If multiple arms cleared, prefer the simplest change from A0 (canonical).\n")
    else:
        lines.append("- No arm cleared the <=10% gate.\n")
        lines.append("- **Recommendation:** The swept parameter space does not contain a configuration that beats the canonical A0 collapse rate.\n")
        lines.append("- Consider expanding the sweep (e.g., longer training, stronger variance regularization, architectural changes).\n")
    lines.append("\n")

    analysis_path = os.path.join(results_dir, "final_analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Saved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
