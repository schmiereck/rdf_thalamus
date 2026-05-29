#!/usr/bin/env python3
"""
Phase 0 Architecture Ceiling Probe Runner v2 (iter_025 re-run).

Addresses five critical methodological flaws from iter_025:
1. Lower learning rate (3e-4) + gradient clipping to fix collapse
2. Defensible effect-size threshold = 0.10 (no noise floor)
3. Hungarian matching as sole primary scheme; sorted reported as secondary
4. 10 seeds for adequate power (>=5 non-collapsed target)
5. Arm E (d_max=16 control) to audit iter_023 capacity claim

Arms:
  A: JEPA+VICReg Control (d_max=8)
  B: Supervised Color Probe + VICReg (d_max=8)
  C: ID-Contrastive + VICReg (d_max=8)
  D: Supervised Color Probe + VICReg (d_max=16)
  E: JEPA+VICReg Control (d_max=16) [capacity audit]

Results saved to archive/iter_025/results_v2/
"""

import os
import sys
import csv
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
    def __init__(self, capacity=2000):
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
def fit_linear_probe(z, y):
    Z = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    return theta[0], theta[1]


def compute_slowness_metrics(z_dyn_log, z_coord_log):
    dyn_deltas = []
    coord_deltas = []
    for z_dyn, z_coord in zip(z_dyn_log, z_coord_log):
        dyn_diff = np.diff(z_dyn, axis=0)
        coord_diff = np.diff(z_coord, axis=0)
        if len(dyn_diff) > 0:
            dyn_deltas.append(np.mean(np.sum(dyn_diff ** 2, axis=1)))
            coord_deltas.append(np.mean(np.sum(coord_diff ** 2, axis=1)))
    mean_dyn_delta = float(np.mean(dyn_deltas)) if dyn_deltas else 0.0
    mean_coord_delta = float(np.mean(coord_deltas)) if coord_deltas else 0.0
    ratio = mean_dyn_delta / (mean_coord_delta + 1e-12)
    return {
        "mean_dyn_delta": mean_dyn_delta,
        "mean_coord_delta": mean_coord_delta,
        "ratio": ratio,
    }


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


def compute_normalized_temporal_var(z_dyn_arr, z_coord_arr, d_t, sub_features=1):
    d_t_dyn = d_t * sub_features
    dyn_diffs = np.diff(z_dyn_arr[:, :d_t_dyn], axis=0)
    coord_diffs = np.diff(z_coord_arr[:, :d_t], axis=0)
    temporal_var_dyn = np.mean(dyn_diffs ** 2)
    temporal_var_coord = np.mean(coord_diffs ** 2)
    spatial_var_dyn = np.mean(z_dyn_arr[:, :d_t_dyn] ** 2)
    spatial_var_coord = np.mean(z_coord_arr[:, :d_t] ** 2)
    norm_dyn = temporal_var_dyn / (spatial_var_dyn + 1e-8)
    norm_coord = temporal_var_coord / (spatial_var_coord + 1e-8)
    sfa_effective = norm_dyn < norm_coord
    return {
        "temporal_var_dyn": temporal_var_dyn,
        "temporal_var_coord": temporal_var_coord,
        "spatial_var_dyn": spatial_var_dyn,
        "spatial_var_coord": spatial_var_coord,
        "normalized_dyn_var": norm_dyn,
        "normalized_coord_var": norm_coord,
        "sfa_effective": bool(sfa_effective),
    }


def compute_tracking_quality(z_coord_arr, pos_arr, d_t):
    dim_to_obj = {}
    used_objs = set()
    for d in range(d_t):
        best_obj = None
        best_dist = np.inf
        for o in range(pos_arr.shape[1]):
            if o in used_objs:
                continue
            dist = np.mean(np.abs(z_coord_arr[:, d] - pos_arr[:, o]))
            if dist < best_dist:
                best_dist = dist
                best_obj = o
        if best_obj is not None:
            dim_to_obj[d] = best_obj
            used_objs.add(best_obj)

    delta_corr = []
    level_corr = []
    for d in range(d_t):
        if d not in dim_to_obj:
            continue
        o = dim_to_obj[d]
        dz = np.diff(z_coord_arr[:, d])
        dp = np.diff(pos_arr[:, o])
        if len(dz) > 2 and np.std(dz) > 1e-8 and np.std(dp) > 1e-8:
            delta_corr.append(np.corrcoef(dz, dp)[0, 1])
        else:
            delta_corr.append(0.0)

        z_c = z_coord_arr[:, d]
        p = pos_arr[:, o]
        if np.std(z_c) > 1e-8 and np.std(p) > 1e-8:
            level_corr.append(np.corrcoef(z_c, p)[0, 1])
        else:
            level_corr.append(0.0)

    return {
        "dim_to_obj": dim_to_obj,
        "delta_corr_mean": float(np.mean(delta_corr)) if delta_corr else 0.0,
        "level_corr_mean": float(np.mean(level_corr)) if level_corr else 0.0,
        "delta_corr_per_dim": delta_corr,
        "level_corr_per_dim": level_corr,
    }


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
# Semantic probes core (with matching_mode support)
# ---------------------------------------------------------------------------
def _compute_semantic_probes_core(z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr,
                                   d_t, sub_features, train_ratio=0.5, rng_seed=None,
                                   matching_mode="hungarian"):
    N = pos_arr.shape[1]

    if sub_features > 1:
        num_samples = z_dyn_arr.shape[0]
        z_dyn_pooled = z_dyn_arr[:, :d_t * sub_features].reshape(num_samples, d_t, sub_features).mean(axis=2)
    else:
        num_samples = z_dyn_arr.shape[0]
        z_dyn_pooled = z_dyn_arr[:, :d_t]

    if matching_mode == "greedy":
        dim_to_obj = {}
        used_objs = set()
        for d in range(d_t):
            best_obj = None
            best_dist = np.inf
            for o in range(N):
                if o in used_objs:
                    continue
                dist = np.mean(np.abs(z_coord_arr[:, d] - pos_arr[:, o]))
                if dist < best_dist:
                    best_dist = dist
                    best_obj = o
            if best_obj is not None:
                dim_to_obj[d] = best_obj
                used_objs.add(best_obj)
    elif matching_mode == "sorted":
        avg_z = np.mean(z_coord_arr[:, :d_t], axis=0)
        avg_pos = np.mean(pos_arr[:, :N], axis=0)
        z_sort_idx = np.argsort(avg_z)
        pos_sort_idx = np.argsort(avg_pos)
        dim_to_obj = {}
        for rank in range(min(d_t, N)):
            dim_to_obj[int(z_sort_idx[rank])] = int(pos_sort_idx[rank])
    elif matching_mode == "hungarian":
        cost = np.zeros((d_t, N))
        for d in range(d_t):
            for o in range(N):
                cost[d, o] = np.mean(np.abs(z_coord_arr[:, d] - pos_arr[:, o]))
        row_ind, col_ind = linear_sum_assignment(cost)
        dim_to_obj = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    else:
        raise ValueError(f"Unknown matching_mode: {matching_mode}")

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
                            base_seed=30000, device="cpu", matching_mode="hungarian"):
    model.eval()
    (traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii,
     z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr) = \
        collect_multitraj_eval_data(model, num_samples=num_samples,
                                     base_seed=base_seed, device=device)

    d_t = model.d_t
    sub_features = model.sub_features

    return _compute_semantic_probes_core(
        z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr,
        d_t, sub_features, train_ratio=train_ratio, matching_mode=matching_mode
    )


def compute_shuffled_semantic_probes(model, num_samples=200, train_ratio=0.5,
                                      base_seed=30000, device="cpu", matching_mode="hungarian"):
    model.eval()
    (traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii,
     z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr) = \
        collect_multitraj_eval_data(model, num_samples=num_samples,
                                     base_seed=base_seed, device=device)

    N = len(z_dyn_arr)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)

    z_dyn_s = z_dyn_arr[perm]
    z_coord_s = z_coord_arr[perm]
    pos_s = pos_arr[perm]
    colors_s = colors_arr[perm]
    radii_s = radii_arr[perm]

    d_t = model.d_t
    sub_features = model.sub_features

    return _compute_semantic_probes_core(
        z_dyn_s, z_coord_s, pos_s, colors_s, radii_s,
        d_t, sub_features, train_ratio=train_ratio, matching_mode=matching_mode
    )


def compute_traj_variance_diagnostics(traj_z_dyn, d_t, sub_features):
    d_t_dyn = d_t * sub_features
    within_vars = []
    traj_means = []

    for z_dyn_traj in traj_z_dyn:
        z_active = z_dyn_traj[:, :d_t_dyn]
        if z_active.shape[0] < 2:
            continue
        traj_var_per_dim = np.var(z_active, axis=0)
        within_vars.append(np.mean(traj_var_per_dim))
        traj_means.append(np.mean(z_active, axis=0))

    within_traj_var = float(np.mean(within_vars)) if within_vars else 0.0

    if len(traj_means) > 1:
        traj_means_arr = np.array(traj_means)
        between_traj_var = float(np.mean(np.var(traj_means_arr, axis=0)))
    else:
        between_traj_var = 0.0

    return {
        "within_traj_var": within_traj_var,
        "between_traj_var": between_traj_var,
    }


# ---------------------------------------------------------------------------
# Evaluation with diagnostics (Hungarian primary, sorted secondary)
# ---------------------------------------------------------------------------
def evaluate_run_with_diagnostics(model, arm_config, seed, device, checkpoint_step=None, eval_steps=200):
    d_t = arm_config.get("d_t", 3)
    sub_features = arm_config.get("sub_features", 1)
    name = arm_config["name"]

    # --- Collapse check ---
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
    has_collapsed, per_dim_std = check_collapse(z_dyn_pooled_check, d_t)
    vh = compute_vicreg_health(z_dyn_pooled_check, d_t)

    slowness = compute_slowness_metrics([z_dyn_arr], [z_coord_arr])
    norm_tv = compute_normalized_temporal_var(z_dyn_arr, z_coord_arr, d_t, sub_features)

    test_env = PhysicsSandbox(N=3, seed=seed + 20000)
    test_history = collections.deque(maxlen=4)
    obs = test_env.reset()
    test_history.append(obs)
    centroid_mse = compute_centroid_mse(model, test_env, test_history,
                                         num_samples=eval_steps, device=device)

    (traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii,
     z_dyn_eval, z_coord_eval, pos_eval, colors_eval, radii_eval) = \
        collect_multitraj_eval_data(model, num_samples=eval_steps,
                                     base_seed=seed + 40000, device=device)

    tracking_quality = compute_tracking_quality(z_coord_eval, pos_eval, d_t)
    traj_diagnostics = compute_traj_variance_diagnostics(traj_z_dyn, d_t, sub_features)

    # Semantic probes: PRIMARY Hungarian, SECONDARY sorted
    semantic_hungarian = compute_semantic_probes(model, num_samples=eval_steps,
                                               base_seed=seed + 30000, device=device, matching_mode="hungarian")
    shuffled_hungarian = compute_shuffled_semantic_probes(model, num_samples=eval_steps,
                                                         base_seed=seed + 30000, device=device, matching_mode="hungarian")
    semantic_sorted = compute_semantic_probes(model, num_samples=eval_steps,
                                             base_seed=seed + 30000, device=device, matching_mode="sorted")

    # Compute mismatch rate between sorted and Hungarian dim_to_obj
    dim_to_obj_sorted = semantic_sorted["dim_to_obj"]
    dim_to_obj_hungarian = semantic_hungarian["dim_to_obj"]
    mismatch_count = 0
    for d in range(d_t):
        obj_s = dim_to_obj_sorted.get(d, -1)
        obj_h = dim_to_obj_hungarian.get(d, -1)
        if obj_s != obj_h:
            mismatch_count += 1
    eval_mismatch_rate = mismatch_count / d_t if d_t > 0 else 0.0

    threshold = arm_config.get("eval_threshold", 0.10)
    pass_sorted = semantic_sorted["delta_r2_color"] >= threshold
    pass_hungarian = semantic_hungarian["delta_r2_color"] >= threshold
    scheme_agreement = pass_sorted == pass_hungarian

    results = {
        "arm": name,
        "seed": seed,
        "checkpoint_step": checkpoint_step,
        "has_collapsed": int(has_collapsed),
        "per_dim_std": per_dim_std.tolist(),
        "mean_abs_corr": vh["mean_abs_corr"],
        "mean_dyn_delta": slowness["mean_dyn_delta"],
        "mean_coord_delta": slowness["mean_coord_delta"],
        "slowness_ratio": slowness["ratio"],
        "centroid_mse_mean": centroid_mse["mse_mean"],
        "centroid_mse_per_object": centroid_mse["mse_per_object"],
        "centroid_r_mean": centroid_mse["r_mean"],
        "centroid_r_per_object": centroid_mse["r_per_object"],
        "vicreg_per_dim_std": per_dim_std.tolist(),
        "vicreg_mean_abs_corr": vh["mean_abs_corr"],
        "temporal_var_dyn": norm_tv["temporal_var_dyn"],
        "temporal_var_coord": norm_tv["temporal_var_coord"],
        "spatial_var_dyn": norm_tv["spatial_var_dyn"],
        "spatial_var_coord": norm_tv["spatial_var_coord"],
        "normalized_dyn_var": norm_tv["normalized_dyn_var"],
        "normalized_coord_var": norm_tv["normalized_coord_var"],
        "sfa_effective": norm_tv["sfa_effective"],
        "tracking_delta_corr": tracking_quality["delta_corr_mean"],
        "tracking_level_corr": tracking_quality["level_corr_mean"],
        "tracking_delta_per_dim": tracking_quality["delta_corr_per_dim"],
        "tracking_level_per_dim": tracking_quality["level_corr_per_dim"],
        "within_traj_var": traj_diagnostics["within_traj_var"],
        "between_traj_var": traj_diagnostics["between_traj_var"],
        # PRIMARY Hungarian metrics
        "r2_dyn_color": semantic_hungarian["r2_dyn_color"],
        "r2_coord_color": semantic_hungarian["r2_coord_color"],
        "r2_dyn_pos": semantic_hungarian["r2_dyn_pos"],
        "r2_coord_pos": semantic_hungarian["r2_coord_pos"],
        "delta_r2_color": semantic_hungarian["delta_r2_color"],
        "r2_dyn_identity": semantic_hungarian["r2_dyn_identity"],
        "r2_coord_identity": semantic_hungarian["r2_coord_identity"],
        "delta_r2_identity": semantic_hungarian["delta_r2_identity"],
        "r2_per_dim": semantic_hungarian["r2_per_dim"],
        "dim_to_obj": semantic_hungarian["dim_to_obj"],
        "shuffled_r2_dyn_color": shuffled_hungarian["r2_dyn_color"],
        "shuffled_delta_r2_color": shuffled_hungarian["delta_r2_color"],
        "shuffled_r2_dyn_identity": shuffled_hungarian["r2_dyn_identity"],
        "shuffled_delta_r2_identity": shuffled_hungarian["delta_r2_identity"],
        "gdasr_growth_point_count": len(model.gdasr_growth_points) if hasattr(model, "gdasr_growth_points") else 0,
        "gdasr_growth_points": list(model.gdasr_growth_points) if hasattr(model, "gdasr_growth_points") else [],
        # Secondary sorted and mismatch fields
        "delta_r2_color_sorted": semantic_sorted["delta_r2_color"],
        "delta_r2_color_hungarian": semantic_hungarian["delta_r2_color"],
        "eval_mismatch_rate": eval_mismatch_rate,
        "scheme_agreement": scheme_agreement,
    }

    return results


# ---------------------------------------------------------------------------
# Single-run training + evaluation
# ---------------------------------------------------------------------------
def run_single(arm_config, seed, device, dry_run=False):
    name = arm_config["name"]
    primary_obj = arm_config["primary_objective"]
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
    dyn_source = arm_config.get("dyn_source", "spatial")
    supervised_weight = arm_config.get("supervised_weight", 0.0)
    contrastive_weight = arm_config.get("contrastive_weight", 0.0)

    # Ramp supervised weight 0.1 -> target over first 500 steps to prevent collapse
    def _ramped_supervised_weight(step):
        if supervised_weight <= 0.0:
            return 0.0
        if step <= 500:
            return 0.1 + (supervised_weight - 0.1) * (step / 500.0)
        return supervised_weight

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
        primary_objective=primary_obj,
        gdasr_log_only=True,
        dyn_readout=dyn_readout,
        sub_features=sub_features,
        dyn_source=dyn_source,
    )
    model.d_t = d_t
    model = model.to(device)

    # CORRECTED: lower lr, gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    env = PhysicsSandbox(N=3, seed=seed)
    obs = env.reset()
    history = collections.deque(maxlen=4)
    history.append(obs)

    replay_buffer = ExtendedReplayBuffer(capacity=2000)

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

    prefill_steps = max(1, min(total_steps - 2, 100))
    _prefill(prefill_steps)

    logs = []
    results_list = []

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

        x_hist_b, x_target_b, pos_b, colors_b, radii_b = replay_buffer.sample(min(32, len(replay_buffer)))
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)
        pos_t = torch.from_numpy(pos_b).float().to(device)
        colors_t = torch.from_numpy(colors_b).float().to(device)

        model.train()
        optimizer.zero_grad()

        loss_dict, _, (z_target_coord, z_target_dyn) = model(
            x_hist_t,
            x_target_t,
            sim_weight=sim_weight,
            var_weight=var_weight,
            cov_weight=cov_weight,
            d_t_predict=d_t,
            ccr_mode=ccr_mode,
            ccr_smooth_weight=ccr_smooth_weight,
            ccr_spatial_weight=ccr_spatial_weight,
        )

        total_loss = loss_dict["loss"]
        log_entry = {
            "step": step,
            "loss": 0.0,
            "loss_internal": loss_dict["loss"].item(),
            "sim_loss": loss_dict["sim_loss"].item(),
            "var_loss": loss_dict["var_loss"].item(),
            "cov_loss": loss_dict["cov_loss"].item(),
            "ccr_smooth_loss": loss_dict.get("ccr_smooth_loss", torch.tensor(0.0)).item(),
            "ccr_spatial_loss": loss_dict.get("ccr_spatial_loss", torch.tensor(0.0)).item(),
            "supervised_loss": 0.0,
            "supervised_loss_sorted": 0.0,
            "contrastive_loss": 0.0,
            "contrastive_loss_sorted": 0.0,
            "mismatch_rate": 0.0,
            "supervised_weight": _ramped_supervised_weight(step),
            "contrastive_weight": contrastive_weight,
            "per_dim_std": None,
        }

        # Additional supervised/contrastive losses (Hungarian primary)
        current_supervised_weight = _ramped_supervised_weight(step)
        if current_supervised_weight > 0.0:
            sup_loss_hungarian, _ = model.compute_supervised_color_loss(
                z_target_coord, z_target_dyn, pos_t, colors_t,
                d_t=model.d_t, N=3, matching_mode="hungarian"
            )
            total_loss = total_loss + current_supervised_weight * sup_loss_hungarian

            with torch.no_grad():
                sup_loss_sorted, _ = model.compute_supervised_color_loss(
                    z_target_coord, z_target_dyn, pos_t, colors_t,
                    d_t=model.d_t, N=3, matching_mode="sorted"
                )
                mismatch = model.compute_mismatch_rate(z_target_coord, pos_t, model.d_t, 3)
            log_entry["supervised_loss"] = sup_loss_hungarian.item()
            log_entry["supervised_loss_sorted"] = sup_loss_sorted.item()
            log_entry["mismatch_rate"] = mismatch
        elif contrastive_weight > 0.0:
            cont_loss_hungarian, _ = model.compute_id_contrastive_loss(
                z_target_coord, z_target_dyn, pos_t, colors_t,
                d_t=model.d_t, N=3, matching_mode="hungarian"
            )
            total_loss = total_loss + contrastive_weight * cont_loss_hungarian

            with torch.no_grad():
                cont_loss_sorted, _ = model.compute_id_contrastive_loss(
                    z_target_coord, z_target_dyn, pos_t, colors_t,
                    d_t=model.d_t, N=3, matching_mode="sorted"
                )
                mismatch = model.compute_mismatch_rate(z_target_coord, pos_t, model.d_t, 3)
            log_entry["contrastive_loss"] = cont_loss_hungarian.item()
            log_entry["contrastive_loss_sorted"] = cont_loss_sorted.item()
            log_entry["mismatch_rate"] = mismatch

        total_loss.backward()
        # CORRECTED: gradient clipping before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        sim_loss_val = loss_dict["sim_loss"].item()
        model.update_recruitment_logic(sim_loss_val, target_dim=d_t, step=step)

        log_entry["loss"] = total_loss.item()

        # Per-dimension std logging every 500 steps
        if step % 500 == 0:
            with torch.no_grad():
                per_dim_std_train = z_target_dyn[:, :d_t].std(dim=0).cpu().numpy()
            log_entry["per_dim_std"] = per_dim_std_train.tolist()

        logs.append(log_entry)

        if step % 1000 == 0 or step == total_steps:
            extra = ""
            if current_supervised_weight > 0.0:
                extra = f"  sup_h={log_entry['supervised_loss']:.4f}  sup_s={log_entry['supervised_loss_sorted']:.4f}  mm={log_entry['mismatch_rate']:.3f}"
            elif contrastive_weight > 0.0:
                extra = f"  cont_h={log_entry['contrastive_loss']:.4f}  cont_s={log_entry['contrastive_loss_sorted']:.4f}  mm={log_entry['mismatch_rate']:.3f}"
            print(f"  [{name}] seed={seed} step={step:5d}/{total_steps}  "
                  f"loss={log_entry['loss']:.4f}  sim={log_entry['sim_loss']:.4f}  "
                  f"var={log_entry['var_loss']:.4f}  cov={log_entry['cov_loss']:.4f}  "
                  f"ccr_s={log_entry['ccr_smooth_loss']:.4f}  ccr_sp={log_entry['ccr_spatial_loss']:.4f}{extra}")

        # Checkpoint evaluations at 2000, 4000, 6000, 8000
        if step in (2000, 4000, 6000, 8000) and not dry_run:
            print(f"  [{name}] seed={seed} -> checkpoint evaluation at step {step}")
            cp_results = evaluate_run_with_diagnostics(model, arm_config, seed, device,
                                                        checkpoint_step=step, eval_steps=eval_steps)
            cp_results["final_train_loss"] = log_entry["loss"]
            cp_results["final_train_sim_loss"] = log_entry["sim_loss"]
            results_list.append(cp_results)

    # Final evaluation if no checkpoint was saved (e.g. dry-run or very short run)
    if len(results_list) == 0 or results_list[-1]["checkpoint_step"] != total_steps:
        print(f"  [{name}] seed={seed} -> final evaluation at step {total_steps}")
        final_res = evaluate_run_with_diagnostics(model, arm_config, seed, device,
                                                   checkpoint_step=total_steps, eval_steps=eval_steps)
        final_res["final_train_loss"] = logs[-1]["loss"]
        final_res["final_train_sim_loss"] = logs[-1]["sim_loss"]
        results_list.append(final_res)

    return results_list, model, logs


# ---------------------------------------------------------------------------
# Arms configuration
# ---------------------------------------------------------------------------
ARMS = [
    {
        "name": "A (JEPA+VICReg Control)",
        "primary_objective": "jepa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "supervised_weight": 0.0, "contrastive_weight": 0.0,
    },
    {
        "name": "B (Supervised Color Probe d_max=8)",
        "primary_objective": "jepa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "supervised_weight": 25.0, "contrastive_weight": 0.0,
    },
    {
        "name": "C (ID-Contrastive d_max=8)",
        "primary_objective": "jepa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "supervised_weight": 0.0, "contrastive_weight": 25.0,
    },
    {
        "name": "D (Supervised Color Probe d_max=16)",
        "primary_objective": "jepa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "d_max": 16, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "supervised_weight": 25.0, "contrastive_weight": 0.0,
    },
    {
        "name": "E (JEPA+VICReg Control d_max=16)",
        "primary_objective": "jepa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "d_max": 16, "d_t": 3, "dyn_readout": "centroid_gated", "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "supervised_weight": 0.0, "contrastive_weight": 0.0,
    },
]

SEEDS = [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]


def _sanitize_arm_name(name):
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('=', '_')


def _flatten_result(res, runs_dir):
    flat = {k: v for k, v in res.items() if k not in (
        "gdasr_growth_points", "centroid_mse_per_object",
        "centroid_r_per_object", "r2_per_dim", "dim_to_obj",
        "tracking_delta_per_dim", "tracking_level_per_dim",
    )}
    flat["gdasr_growth_point_count"] = len(res.get("gdasr_growth_points", []))
    flat["centroid_mse_obj0"] = res.get("centroid_mse_per_object", [])[0] if len(res.get("centroid_mse_per_object", [])) > 0 else None
    flat["centroid_mse_obj1"] = res.get("centroid_mse_per_object", [])[1] if len(res.get("centroid_mse_per_object", [])) > 1 else None
    flat["centroid_mse_obj2"] = res.get("centroid_mse_per_object", [])[2] if len(res.get("centroid_mse_per_object", [])) > 2 else None
    flat["centroid_r_obj0"] = res.get("centroid_r_per_object", [])[0] if len(res.get("centroid_r_per_object", [])) > 0 else None
    flat["centroid_r_obj1"] = res.get("centroid_r_per_object", [])[1] if len(res.get("centroid_r_per_object", [])) > 1 else None
    flat["centroid_r_obj2"] = res.get("centroid_r_per_object", [])[2] if len(res.get("centroid_r_per_object", [])) > 2 else None
    flat["per_dim_std"] = str(res.get("per_dim_std", []))
    flat["tracking_delta_per_dim"] = str(res.get("tracking_delta_per_dim", []))
    flat["tracking_level_per_dim"] = str(res.get("tracking_level_per_dim", []))

    r2pd = res.get("r2_per_dim", [])
    dim_map = res.get("dim_to_obj", {})
    for d_idx, probe_d in enumerate(r2pd):
        flat[f"r2_dim{d_idx}_dyn_color"] = probe_d.get("dyn_color", None)
        flat[f"r2_dim{d_idx}_coord_color"] = probe_d.get("coord_color", None)
        flat[f"r2_dim{d_idx}_dyn_pos"] = probe_d.get("dyn_pos", None)
        flat[f"r2_dim{d_idx}_coord_pos"] = probe_d.get("coord_pos", None)
        flat[f"r2_dim{d_idx}_dyn_identity"] = probe_d.get("dyn_identity", None)
        flat[f"r2_dim{d_idx}_coord_identity"] = probe_d.get("coord_identity", None)
        flat[f"dim{d_idx}_matched_obj"] = dim_map.get(d_idx, None)
    flat["dim_to_obj"] = json.dumps(dim_map, default=str)

    cp_label = f"_cp{res['checkpoint_step']}" if res.get("checkpoint_step") else ""
    safe_name = _sanitize_arm_name(res["arm"])
    run_id = f"{safe_name}_seed{res['seed']}{cp_label}"
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

    results_list, model, logs = run_single(arm, seed, device, dry_run=dry_run)

    result_entries = []
    for res in results_list:
        csv_path = _flatten_result(res, runs_dir)
        result_entries.append((res, csv_path))
        safe_name = _sanitize_arm_name(res["arm"])
        cp_label = f"_cp{res['checkpoint_step']}" if res.get("checkpoint_step") else ""
        run_id = f"{safe_name}_seed{res['seed']}{cp_label}"
        json_path = os.path.join(runs_dir, f"{run_id}.json")
        print(f"  -> Saved {run_id}  ({csv_path}, {json_path})")

    safe_name = _sanitize_arm_name(arm["name"])
    ckpt_path = os.path.join(checkpoints_dir, f"{safe_name}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  -> Saved checkpoint {ckpt_path}")

    logs_path = os.path.join(runs_dir, f"{safe_name}_seed{seed}_logs.csv")
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(logs_path, index=False)
    print(f"  -> Saved training logs {logs_path}")

    return result_entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cpu_count = os.cpu_count() or 2
    default_workers = min(cpu_count - 1, 8)

    parser = argparse.ArgumentParser(
        description="Phase 0 Architecture Ceiling Probe Runner v2 (iter_025 re-run)"
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

    results_dir = "archive/iter_025/results_v2"
    runs_dir = os.path.join(results_dir, "runs")
    figs_dir = os.path.join(results_dir, "figs")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    all_result_entries = []

    # Pre-declared effect-size threshold (no noise floor)
    threshold = 0.10
    print(f"Pre-declared effect-size threshold (delta_R2_color): {threshold:.4f}")
    print()

    # ------------------------------------------------------------------ #
    # Main experiment tasks
    # ------------------------------------------------------------------ #
    tasks = []
    task_labels = []
    for arm in ARMS:
        name = arm["name"]
        for seed in seeds:
            # Inject eval threshold into arm config for evaluation
            arm_copy = dict(arm)
            arm_copy["eval_threshold"] = threshold
            tasks.append((arm_copy, seed, device_str, dry_run, runs_dir, checkpoints_dir))
            task_labels.append(f"{name} seed={seed}")

    total_tasks = len(tasks)
    print(f"Total main tasks to run: {total_tasks}")
    for i, label in enumerate(task_labels, 1):
        print(f"  [{i:2d}] {label}")
    print()

    if sequential:
        for i, task in enumerate(tasks):
            arm, seed = task[0], task[1]
            print(f"\n{'='*70}")
            print(f"TASK [{i+1}/{total_tasks}]: {arm['name']} seed={seed}")
            print(f"{'='*70}")
            entries = _run_single_worker(task)
            all_result_entries.extend(entries)
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
                    entries = future.result()
                    all_result_entries.extend(entries)
                    print(f"\n[{done_count}/{total_tasks}] COMPLETED: {label}")
                except Exception as exc:
                    print(f"\n[{done_count}/{total_tasks}] FAILED: {label} -> {exc}")

    # ------------------------------------------------------------------ #
    # Aggregate results
    # ------------------------------------------------------------------ #
    all_results = []
    checkpoint_results = []
    for res, _csv_path in all_result_entries:
        all_results.append(res)
        if res["checkpoint_step"] in (2000, 4000, 6000):
            checkpoint_results.append(res)

    final_results = [r for r in all_results if r.get("checkpoint_step") not in (2000, 4000, 6000)]
    df_all = pd.DataFrame(final_results)

    numeric_cols = [
        "has_collapsed", "mean_dyn_delta", "mean_coord_delta", "slowness_ratio",
        "centroid_mse_mean", "centroid_r_mean", "mean_abs_corr",
        "final_train_loss", "final_train_sim_loss",
        "gdasr_growth_point_count",
        "r2_dyn_color", "r2_coord_color", "r2_dyn_pos", "r2_coord_pos",
        "delta_r2_color",
        "r2_dyn_identity", "r2_coord_identity", "delta_r2_identity",
        "normalized_dyn_var", "normalized_coord_var",
        "tracking_delta_corr", "tracking_level_corr",
        "temporal_var_dyn", "spatial_var_dyn",
        "within_traj_var", "between_traj_var",
        "shuffled_r2_dyn_color", "shuffled_delta_r2_color",
        "shuffled_r2_dyn_identity", "shuffled_delta_r2_identity",
        "delta_r2_color_sorted", "delta_r2_color_hungarian",
        "eval_mismatch_rate",
    ]
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    summary_path = os.path.join(results_dir, "summary_phase0_id_probe_v2.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"\nSaved full results to {summary_path}")

    if len(checkpoint_results) > 0:
        df_cp = pd.DataFrame(checkpoint_results)
        for col in numeric_cols:
            if col in df_cp.columns:
                df_cp[col] = pd.to_numeric(df_cp[col], errors="coerce")
        cp_summary_path = os.path.join(results_dir, "summary_phase0_id_probe_v2_cp.csv")
        df_cp.to_csv(cp_summary_path, index=False)
        print(f"Saved checkpoint results to {cp_summary_path}")

    agg_cols = [
        "has_collapsed", "mean_dyn_delta", "mean_coord_delta", "slowness_ratio",
        "centroid_mse_mean", "centroid_r_mean", "mean_abs_corr",
        "final_train_loss", "final_train_sim_loss", "gdasr_growth_point_count",
        "r2_dyn_color", "r2_coord_color", "delta_r2_color",
        "r2_dyn_identity", "r2_coord_identity", "delta_r2_identity",
        "normalized_dyn_var", "normalized_coord_var",
        "tracking_delta_corr", "tracking_level_corr",
        "temporal_var_dyn", "spatial_var_dyn",
        "within_traj_var", "between_traj_var",
        "shuffled_r2_dyn_color", "shuffled_delta_r2_color",
        "shuffled_r2_dyn_identity", "shuffled_delta_r2_identity",
        "delta_r2_color_sorted", "delta_r2_color_hungarian", "eval_mismatch_rate",
    ]
    present_agg_cols = [c for c in agg_cols if c in df_all.columns]
    if present_agg_cols and "arm" in df_all.columns and len(df_all) > 0:
        agg = df_all.groupby("arm")[present_agg_cols].agg(["mean", "std"]).reset_index()
        agg_path = os.path.join(results_dir, "aggregated_phase0_id_probe_v2.csv")
        agg.to_csv(agg_path)
        print(f"\nAggregated stats saved to {agg_path}")

    # ------------------------------------------------------------------ #
    # Analysis and report
    # ------------------------------------------------------------------ #
    _generate_analysis(df_all, threshold, results_dir)
    print("\nDone.")


def _generate_analysis(df_all, threshold, results_dir):
    """Generate final_analysis.md with per-arm statistics and pre-registered interpretations."""
    lines = []
    lines.append("# Iter_025 Architecture Ceiling Probe Analysis v2\n")
    lines.append(f"**Date:** Auto-generated\n")
    lines.append(f"**Pre-declared effect-size threshold:** {threshold:.4f}\n")
    lines.append("**Matching scheme:** Hungarian (primary); sorted (secondary check)\n")
    lines.append("---\n")

    # Helper for mean/std
    def _agg(col, df):
        vals = df[col].dropna().values
        if len(vals) == 0:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    def _fmt(val, fmt=".4f"):
        if val is None:
            return "N/A"
        return f"{val:{fmt}}"

    # Per-arm summary
    lines.append("## Per-Arm Summary (Final Step = 8000)\n\n")
    for arm_name in sorted(df_all["arm"].unique()):
        df_arm = df_all[df_all["arm"] == arm_name]
        lines.append(f"### {arm_name}\n")
        lines.append(f"- N seeds: {len(df_arm)}\n")

        collapse_rate = df_arm["has_collapsed"].mean()
        collapsed_count = int(df_arm["has_collapsed"].sum())
        lines.append(f"- Collapse rate: {collapse_rate:.2f} ({collapsed_count}/{len(df_arm)})\n")

        mean_drc_h, std_drc_h = _agg("delta_r2_color_hungarian", df_arm)
        lines.append(f"- delta_R2_color (Hungarian primary): {_fmt(mean_drc_h)} +/- {_fmt(std_drc_h)}\n")

        mean_drc_s, std_drc_s = _agg("delta_r2_color_sorted", df_arm)
        lines.append(f"- delta_R2_color (sorted secondary): {_fmt(mean_drc_s)} +/- {_fmt(std_drc_s)}\n")

        mean_mm, std_mm = _agg("eval_mismatch_rate", df_arm)
        lines.append(f"- Eval mismatch rate: {_fmt(mean_mm, '.3f')} +/- {_fmt(std_mm, '.3f')}\n")

        lines.append(f"- Centroid MSE: {_fmt(_agg('centroid_mse_mean', df_arm)[0], '.2f')}\n")
        lines.append(f"- Tracking level corr: {_fmt(_agg('tracking_level_corr', df_arm)[0], '.3f')}\n")
        lines.append(f"- Mean abs corr: {_fmt(_agg('mean_abs_corr', df_arm)[0], '.3f')}\n")

        # Per-seed details
        lines.append("\n**Per-seed details:**\n")
        lines.append("| seed | collapsed | delta_R2_color_HU | delta_R2_color_sorted | mismatch_rate | scheme_agree |\n")
        lines.append("|------|-----------|-------------------|-----------------------|---------------|--------------|\n")
        for _, row in df_arm.iterrows():
            s = int(row["seed"])
            c = int(row["has_collapsed"])
            d_h = _fmt(row.get("delta_r2_color_hungarian"), ".4f")
            d_s = _fmt(row.get("delta_r2_color_sorted"), ".4f")
            mm = _fmt(row.get("eval_mismatch_rate"), ".3f")
            agree = "Y" if row.get("scheme_agreement", False) else "N"
            lines.append(f"| {s} | {c} | {d_h} | {d_s} | {mm} | {agree} |\n")
        lines.append("\n")

    # Power check
    lines.append("## Experiment Power Check\n\n")
    df_a = df_all[df_all["arm"] == "A (JEPA+VICReg Control)"]
    arm_a_collapse_rate = df_a["has_collapsed"].mean() if len(df_a) > 0 else 1.0
    lines.append(f"- Arm A collapse rate: {arm_a_collapse_rate:.2f} ({int(df_a['has_collapsed'].sum())}/{len(df_a)})\n")
    if arm_a_collapse_rate > 0.20:
        lines.append("- **RESULT: UNDERPOWERED.** Arm A collapse rate > 2/10. No architecture claim can be valid.\n")
        powered = False
    else:
        lines.append("- Adequately powered (Arm A collapse rate <= 2/10).\n")
        powered = True
    lines.append("\n")

    # Matching dependency report
    lines.append("## Matching Dependency Report\n\n")
    for arm_name in ["B (Supervised Color Probe d_max=8)", "C (ID-Contrastive d_max=8)"]:
        df_arm = df_all[df_all["arm"] == arm_name]
        if len(df_arm) == 0:
            lines.append(f"**{arm_name}:** No data.\n\n")
            continue
        non_collapsed = df_arm[df_arm["has_collapsed"] == 0]
        if len(non_collapsed) == 0:
            lines.append(f"**{arm_name}:** All seeds collapsed; matching check irrelevant.\n\n")
            continue
        disagree = non_collapsed["scheme_agreement"] == False
        disagree_rate = disagree.mean()
        lines.append(f"**{arm_name}:**\n")
        lines.append(f"- Non-collapsed seeds: {len(non_collapsed)}\n")
        lines.append(f"- Sorted vs Hungarian pass/fail disagreement rate: {disagree_rate:.2f} ({int(disagree.sum())}/{len(non_collapsed)})\n")
        if disagree_rate > 0.25:
            lines.append("- **Matching-dependent outcome.** Falsification claim NOT earned for this arm.\n")
        else:
            lines.append("- Matching schemes agree on pass/fail for >=75% of non-collapsed seeds. Verdict is stable.\n")
        lines.append("\n")

    # H1 / H2 falsification verdicts
    lines.append("## Falsification Verdicts\n\n")

    # H1
    df_b = df_all[df_all["arm"] == "B (Supervised Color Probe d_max=8)"]
    h1_verdict = "N/A"
    h1_pass = False
    if len(df_b) > 0:
        non_collapsed_b = df_b[df_b["has_collapsed"] == 0]
        collapse_rate_b = df_b["has_collapsed"].mean()
        if len(non_collapsed_b) > 0:
            mean_delta_hu = non_collapsed_b["delta_r2_color_hungarian"].mean()
            lines.append(f"**H1 (Architecture Capacity — Arm B):**\n")
            lines.append(f"- Collapse rate: {collapse_rate_b:.2f} ({int(df_b['has_collapsed'].sum())}/{len(df_b)})\n")
            lines.append(f"- Mean delta_R2_color (non-collapsed, Hungarian): {mean_delta_hu:.4f}\n")
            lines.append(f"- Threshold: {threshold:.4f}\n")
            if not powered:
                lines.append(f"- **Verdict:** UNDERPOWERED. No architecture claim valid.\n")
                h1_verdict = "Underpowered"
            elif collapse_rate_b > 0.20:
                lines.append(f"- **Verdict:** FAILED (collapse rate > 2/10).\n")
                h1_verdict = "Failed (collapse)"
            else:
                if mean_delta_hu >= threshold:
                    lines.append(f"- **Verdict:** PASS. Compatible with sufficient architectural capacity under direct supervision.\n")
                    h1_pass = True
                    h1_verdict = "Pass"
                else:
                    lines.append(f"- **Verdict:** FAIL. Consistent with an architecture-level bottleneck on identity encoding.\n")
                    h1_verdict = "Fail"
        else:
            lines.append(f"**H1 (Arm B):** ALL RUNS COLLAPSED. H1 FAILED.\n")
            h1_verdict = "Failed (all collapsed)"
    else:
        lines.append("**H1 (Arm B):** No data.\n")
    lines.append("\n")

    # H2
    df_c = df_all[df_all["arm"] == "C (ID-Contrastive d_max=8)"]
    h2_verdict = "N/A"
    h2_pass = False
    if len(df_c) > 0:
        non_collapsed_c = df_c[df_c["has_collapsed"] == 0]
        collapse_rate_c = df_c["has_collapsed"].mean()
        if len(non_collapsed_c) > 0:
            mean_delta_hu = non_collapsed_c["delta_r2_color_hungarian"].mean()
            lines.append(f"**H2 (ID-Contrastive Viability — Arm C):**\n")
            lines.append(f"- Collapse rate: {collapse_rate_c:.2f} ({int(df_c['has_collapsed'].sum())}/{len(df_c)})\n")
            lines.append(f"- Mean delta_R2_color (non-collapsed, Hungarian): {mean_delta_hu:.4f}\n")
            lines.append(f"- Threshold: {threshold:.4f}\n")
            if not powered:
                lines.append(f"- **Verdict:** UNDERPOWERED. No architecture claim valid.\n")
                h2_verdict = "Underpowered"
            elif collapse_rate_c > 0.20:
                lines.append(f"- **Verdict:** FAILED (collapse rate > 2/10).\n")
                h2_verdict = "Failed (collapse)"
            else:
                if mean_delta_hu >= threshold:
                    lines.append(f"- **Verdict:** PASS. Supervised (slot IDs are privileged), not evidence decoder-free self-supervision is solved.\n")
                    h2_pass = True
                    h2_verdict = "Pass"
                else:
                    lines.append(f"- **Verdict:** FAIL. ID-contrastive formulation insufficient.\n")
                    h2_verdict = "Fail"
        else:
            lines.append(f"**H2 (Arm C):** ALL RUNS COLLAPSED. H2 FAILED.\n")
            h2_verdict = "Failed (all collapsed)"
    else:
        lines.append("**H2 (Arm C):** No data.\n")
    lines.append("\n")

    # Capacity audit (Arm E)
    lines.append("## Capacity Audit (Arm E vs iter_023 d_max=16 claim)\n\n")
    df_e = df_all[df_all["arm"] == "E (JEPA+VICReg Control d_max=16)"]
    if len(df_e) > 0:
        non_collapsed_e = df_e[df_e["has_collapsed"] == 0]
        if len(non_collapsed_e) > 0:
            mean_delta_e = non_collapsed_e["delta_r2_color_hungarian"].mean()
            lines.append(f"- Arm E non-collapsed mean delta_R2_color (Hungarian): {mean_delta_e:.4f}\n")
            if mean_delta_e >= threshold:
                lines.append("- **Audit result:** d_max=16 improvement CONFIRMED as capacity effect (occurs without identity objective).\n")
            else:
                lines.append("- **Audit result:** d_max=16 improvement NOT replicated without identity objective. The iter_023 claim needs revision.\n")
        else:
            lines.append("- Arm E: ALL RUNS COLLAPSED. Audit inconclusive.\n")
    else:
        lines.append("- Arm E: No data.\n")
    lines.append("\n")

    # Outcome quadrant
    lines.append("## Outcome Quadrant\n\n")
    lines.append("| Arm B | Arm C | Interpretation |\n")
    lines.append("|-------|-------|----------------|\n")

    if not powered:
        lines.append("| N/A | N/A | Experiment underpowered (Arm A collapse > 2/10). No architecture claim valid. |\n")
    elif h1_pass and h2_pass:
        lines.append("| PASS | PASS | Architecture CAN encode identity; ID-contrastive viable. Continue developing ID-contrastive. |\n")
    elif h1_pass and not h2_pass:
        lines.append("| PASS | FAIL | Architecture CAN encode; contrastive formulation insufficient. Try direct supervised objective. |\n")
    elif not h1_pass and h2_pass:
        lines.append("| FAIL | PASS | Unexpected: supervised < contrastive. Check implementation. |\n")
    else:
        lines.append("| FAIL | FAIL | Consistent with an architecture-level bottleneck on identity encoding (conditional on mismatch rate). Next: separate z_dyn encoder. |\n")
    lines.append("\n")

    # Next-step recommendation
    lines.append("## Next-Step Recommendation\n\n")
    if not powered:
        lines.append("- The experiment is underpowered due to high collapse rate in the control arm.\n")
        lines.append("- **Recommendation:** Investigate and fix the underlying collapse issue before running another architecture probe. Consider further reducing LR, increasing batch size, or stronger variance regularization.\n")
    elif h1_pass and h2_pass:
        lines.append("- Both H1 and H2 pass. The architecture is capable of encoding identity under direct supervision, and the contrastive proxy works.\n")
        lines.append("- **Recommendation:** Continue refining the ID-contrastive objective (e.g., stronger augmentations, harder negatives) and test decoder-free self-supervision without privileged slot IDs.\n")
    elif h1_pass and not h2_pass:
        lines.append("- H1 passes but H2 fails. The architecture has capacity, but the contrastive formulation is insufficient.\n")
        lines.append("- **Recommendation:** Use the direct supervised color loss as a training objective (not just probe) and explore stronger contrastive variants.\n")
    elif not h1_pass and h2_pass:
        lines.append("- H1 fails but H2 passes. This is unexpected and suggests a possible implementation issue with the supervised loss or matching.\n")
        lines.append("- **Recommendation:** Debug the supervised color loss implementation and verifying gradient flow.\n")
    else:
        lines.append("- Both H1 and H2 fail (and experiment is powered). This is consistent with an architecture-level bottleneck.\n")
        lines.append("- **Recommendation:** Modify the architecture to relax the decoder-free constraint or give z_dyn a separate encoder pathway.\n")
    lines.append("\n")

    analysis_path = os.path.join(results_dir, "final_analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Saved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
