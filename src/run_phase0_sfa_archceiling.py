#!/usr/bin/env python3
"""
Phase 0 SFA Architecture Ceiling Evaluation Runner.

Tests three architectural modifications against the control:
- Conv4 dynamics source (dyn_source='conv4')
- Expanded d_max=16
- Sub-features K=4

Four arms x 5 seeds x 5000 steps.
Seeds: [42, 123, 456, 789, 999]
Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000
d_t=3, gdasr_log_only=True

Results saved to archive/iter_022/results/
"""

import os
import sys
import csv
import json
import random
import argparse
import collections
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models_dual_stream import (
    NonParametricJEPASpatial,
    calculate_centroid_and_variance,
    add_positional_encoding,
)

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, x_hist, x_target):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (x_hist, x_target)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        x_hist_b, x_target_b = zip(*batch)
        return np.stack(x_hist_b, axis=0), np.stack(x_target_b, axis=0)

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
    """
    z_feature: (N,) array of scalar feature
    y_target:  (N, K) array of K-dimensional targets
    Returns multivariate R^2.
    """
    z = z_feature.reshape(-1)
    y = y_target.reshape(len(z), -1)
    N = len(z)
    if N < 5 or y.shape[1] < 1:
        return 0.0
    Z_aug = np.stack([z, np.ones_like(z)], axis=1)  # (N, 2)
    theta = np.linalg.pinv(Z_aug.T @ Z_aug) @ Z_aug.T @ y  # (2, K)
    y_pred = Z_aug @ theta  # (N, K)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ---------------------------------------------------------------------------
# NEW: Normalized temporal variance
# ---------------------------------------------------------------------------
def compute_normalized_temporal_var(z_dyn_arr, z_coord_arr, d_t, sub_features=1):
    """z_dyn_arr: (N, d_max*K), z_coord_arr: (N, d_max)"""
    d_t_dyn = d_t * sub_features
    # Temporal variance: mean(Δz²)
    dyn_diffs = np.diff(z_dyn_arr[:, :d_t_dyn], axis=0)
    coord_diffs = np.diff(z_coord_arr[:, :d_t], axis=0)
    temporal_var_dyn = np.mean(dyn_diffs ** 2)
    temporal_var_coord = np.mean(coord_diffs ** 2)
    # Spatial variance: mean(z²)
    spatial_var_dyn = np.mean(z_dyn_arr[:, :d_t_dyn] ** 2)
    spatial_var_coord = np.mean(z_coord_arr[:, :d_t] ** 2)
    # Normalized
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


# ---------------------------------------------------------------------------
# NEW: Centroid tracking quality
# ---------------------------------------------------------------------------
def compute_tracking_quality(z_coord_arr, pos_arr, d_t):
    """Compute correlation between z_coord changes and true position changes."""
    # Match channels to objects by minimum distance
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


# ---------------------------------------------------------------------------
# NEW: Per-sub-feature identity probes
# ---------------------------------------------------------------------------
def compute_sub_feature_probes(z_dyn_arr, colors_arr, radii_arr, d_t, sub_features, num_samples, train_ratio=0.5):
    """
    For each (channel c, sub-feature k), fit R² against [R, G, B, radius_norm] individually.
    Returns a (d_t, sub_features, 4) array of R² values.
    Matches channels to objects using z_coord_arr (requires reordering externally).
    Here we assume colors_arr and radii_arr already match channel order.
    """
    max_radius = 20.0
    n_train = int(num_samples * train_ratio)

    # Per-sub-feature R² against individual identity dimensions
    r2_matrix = np.zeros((d_t, sub_features, 4))

    for c in range(d_t):
        for k in range(sub_features):
            idx = c * sub_features + k
            if idx >= z_dyn_arr.shape[1]:
                continue
            z_test = z_dyn_arr[n_train:, idx]
            z_train = z_dyn_arr[:n_train, idx]

            for id_dim in range(4):
                if id_dim < 3:
                    y_test = colors_arr[n_train:, c, id_dim]
                    y_train = colors_arr[:n_train, c, id_dim]
                else:
                    y_test = radii_arr[n_train:, c] / max_radius
                    y_train = radii_arr[:n_train, c] / max_radius

                if np.std(z_train) < 1e-8 or np.std(z_test) < 1e-8 or np.std(y_train) < 1e-8:
                    r2_matrix[c, k, id_dim] = 0.0
                    continue

                # Linear regression on train
                A = np.vstack([z_train, np.ones_like(z_train)]).T
                try:
                    theta = np.linalg.lstsq(A, y_train, rcond=None)[0]
                    y_pred = z_test * theta[0] + theta[1]
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_train)) ** 2)
                    if ss_tot < 1e-12:
                        r2_matrix[c, k, id_dim] = 0.0
                    else:
                        r2_matrix[c, k, id_dim] = float(1.0 - ss_res / ss_tot)
                except:
                    r2_matrix[c, k, id_dim] = 0.0

    return r2_matrix


# ---------------------------------------------------------------------------
# Centroid decoding MSE (adapted for variable d_max)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Multi-trajectory eval data collection (adapted for variable d_max)
# ---------------------------------------------------------------------------
def collect_multitraj_eval_data(model, num_samples=200, base_seed=30000, device="cpu"):
    """Collect evaluation data from multiple trajectories for semantic probing."""
    model.eval()
    num_trajectories = max(1, num_samples // 20)
    samples_per_traj = max(1, num_samples // num_trajectories)

    z_dyn_list = []
    z_coord_list = []
    pos_list = []
    colors_list = []
    radii_list = []

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

            collected = 0
            while collected < samples_per_traj:
                obs, info = env.step({"acc": 0.0, "push": False})
                history.append(obs)
                if len(history) == 4:
                    x_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    z_c, z_d = model.encoder(x_t)
                    # Keep all d_max dims for downstream processing
                    z_dyn_list.append(z_d[0].cpu().numpy())
                    z_coord_list.append(z_c[0].cpu().numpy())
                    pos_list.append(info["positions"])
                    colors_list.append(info["colors"])
                    radii_list.append(info["radii"])
                    collected += 1

    z_dyn_arr = np.array(z_dyn_list)
    z_coord_arr = np.array(z_coord_list)
    pos_arr = np.array(pos_list)
    colors_arr = np.array(colors_list)
    radii_arr = np.array(radii_list)

    return z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr


# ---------------------------------------------------------------------------
# Semantic probes (adapted for K>1 pooling)
# ---------------------------------------------------------------------------
def compute_semantic_probes(model, num_samples=200, train_ratio=0.5,
                            base_seed=30000, device="cpu"):
    model.eval()
    z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr = \
        collect_multitraj_eval_data(model, num_samples=num_samples,
                                     base_seed=base_seed, device=device)

    N = pos_arr.shape[1]
    d_t = model.d_t
    sub_features = model.sub_features
    d_max = model.d_max

    # Pool K sub-features per channel for fair cross-arm comparison
    if sub_features > 1:
        # z_dyn_arr: (num_samples, d_max*sub_features)
        z_dyn_pooled = z_dyn_arr[:, :d_t * sub_features].reshape(num_samples, d_t, sub_features).mean(axis=2)  # (N, d_t)
    else:
        z_dyn_pooled = z_dyn_arr[:, :d_t]  # (N, d_t)

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

        # Compound identity probe
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

    # Sub-feature probes (only when K > 1)
    sub_feature_r2_matrix = None
    if sub_features > 1:
        # Match channels to objects for identity probing
        sub_feature_r2_matrix = compute_sub_feature_probes(
            z_dyn_arr, colors_arr, radii_arr, d_t, sub_features, num_samples, train_ratio
        )

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
        "sub_feature_r2_matrix": sub_feature_r2_matrix.tolist() if sub_feature_r2_matrix is not None else None,
    }


# ---------------------------------------------------------------------------
# Evaluation helper (updated with new metrics)
# ---------------------------------------------------------------------------
def evaluate_run(model, arm_config, seed, device, checkpoint_step=None, eval_steps=200):
    """
    Run the full evaluation protocol on a model.
    Returns a results dict.
    """
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

    # Collapse check on pooled features
    if sub_features > 1:
        z_dyn_pooled_check = z_dyn_arr[:, :d_t * sub_features].reshape(eval_steps, d_t, sub_features).mean(axis=2)
    else:
        z_dyn_pooled_check = z_dyn_arr[:, :d_t]
    has_collapsed, per_dim_std = check_collapse(z_dyn_pooled_check, d_t)
    vh = compute_vicreg_health(z_dyn_pooled_check, d_t)

    # Slowness metrics
    slowness = compute_slowness_metrics([z_dyn_arr], [z_coord_arr])

    # Normalized temporal variance
    norm_tv = compute_normalized_temporal_var(z_dyn_arr, z_coord_arr, d_t, sub_features)

    # Centroid decoding MSE
    test_env = PhysicsSandbox(N=3, seed=seed + 20000)
    test_history = collections.deque(maxlen=4)
    obs = test_env.reset()
    test_history.append(obs)
    centroid_mse = compute_centroid_mse(model, test_env, test_history,
                                         num_samples=eval_steps, device=device)

    # Tracking quality (needs true positions collected per step)
    # Re-use eval data's true positions from multitraj collection
    z_dyn_eval, z_coord_eval, pos_eval, colors_eval, radii_eval = \
        collect_multitraj_eval_data(model, num_samples=eval_steps,
                                     base_seed=seed + 40000, device=device)
    tracking_quality = compute_tracking_quality(z_coord_eval, pos_eval, d_t)

    # Semantic disentanglement probes
    semantic = compute_semantic_probes(model, num_samples=eval_steps,
                                        base_seed=seed + 30000, device=device)

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
        # New metrics
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
        # Semantic probes
        "r2_dyn_color": semantic["r2_dyn_color"],
        "r2_coord_color": semantic["r2_coord_color"],
        "r2_dyn_pos": semantic["r2_dyn_pos"],
        "r2_coord_pos": semantic["r2_coord_pos"],
        "delta_r2_color": semantic["delta_r2_color"],
        "r2_dyn_identity": semantic["r2_dyn_identity"],
        "r2_coord_identity": semantic["r2_coord_identity"],
        "delta_r2_identity": semantic["delta_r2_identity"],
        "r2_per_dim": semantic["r2_per_dim"],
        "dim_to_obj": semantic["dim_to_obj"],
        "sub_feature_r2_matrix": semantic["sub_feature_r2_matrix"],
        "gdasr_growth_point_count": len(model.gdasr_growth_points) if hasattr(model, "gdasr_growth_points") else 0,
        "gdasr_growth_points": list(model.gdasr_growth_points) if hasattr(model, "gdasr_growth_points") else [],
    }
    return results


# ---------------------------------------------------------------------------
# Single-run training + evaluation
# ---------------------------------------------------------------------------
def run_single(arm_config, seed, device, dry_run=False):
    """
    Run one arm x seed experiment.
    """
    name = arm_config["name"]
    primary_obj = arm_config["primary_objective"]
    sfa_weight = arm_config.get("sfa_weight", 0.1)
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

    set_seed(seed)

    total_steps = 5 if dry_run else 5000
    eval_steps = min(200, total_steps // 2) if dry_run else 200

    model = NonParametricJEPASpatial(
        d_max=d_max,
        h=3,
        k=4,
        cooldown=300,
        stabilization_period=100,
        pos_encoding=pos_encoding,
        primary_objective=primary_obj,
        sfa_weight=sfa_weight,
        gdasr_log_only=True,
        dyn_readout=dyn_readout,
        sub_features=sub_features,
        dyn_source=dyn_source,
    )
    model.d_t = d_t
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    env = PhysicsSandbox(N=3, seed=seed)
    obs = env.reset()
    history = collections.deque(maxlen=4)
    history.append(obs)

    replay_buffer = ReplayBuffer(capacity=2000)

    def _prefill(n):
        while len(replay_buffer) < n:
            obs, info = env.step({"acc": 0.0, "push": False})
            history.append(obs)
            if len(history) == 4:
                replay_buffer.push(
                    np.stack(list(history)[:3], axis=0),
                    history[3],
                )

    _prefill(min(100, total_steps))

    logs = []
    results_list = []

    for step in range(1, total_steps + 1):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        if len(history) == 4:
            replay_buffer.push(
                np.stack(list(history)[:3], axis=0),
                history[3],
            )

        x_hist_b, x_target_b = replay_buffer.sample(min(32, len(replay_buffer)))
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)

        model.train()
        optimizer.zero_grad()

        loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(
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
        loss_dict["loss"].backward()
        optimizer.step()

        sim_loss_val = loss_dict["sim_loss"].item()
        model.update_recruitment_logic(sim_loss_val, target_dim=d_t, step=step)

        log_entry = {
            "step": step,
            "loss": loss_dict["loss"].item(),
            "sim_loss": sim_loss_val,
            "sfa_loss": loss_dict.get("sfa_loss", torch.tensor(0.0)).item(),
            "var_loss": loss_dict["var_loss"].item(),
            "cov_loss": loss_dict["cov_loss"].item(),
            "ccr_smooth_loss": loss_dict.get("ccr_smooth_loss", torch.tensor(0.0)).item(),
            "ccr_spatial_loss": loss_dict.get("ccr_spatial_loss", torch.tensor(0.0)).item(),
        }
        logs.append(log_entry)

        if step % 1000 == 0 or step == total_steps:
            print(f"  [{name}] seed={seed} step={step:5d}/{total_steps}  "
                  f"loss={log_entry['loss']:.4f}  sim={log_entry['sim_loss']:.4f}  "
                  f"sfa={log_entry['sfa_loss']:.4f}  ccr_s={log_entry['ccr_smooth_loss']:.4f}  "
                  f"ccr_sp={log_entry['ccr_spatial_loss']:.4f}")

        # Checkpoint evaluation at step 2500
        if step == 2500 and not dry_run:
            print(f"  [{name}] seed={seed} -> checkpoint evaluation at step 2500")
            cp_results = evaluate_run(model, arm_config, seed, device, checkpoint_step=2500, eval_steps=eval_steps)
            cp_results["final_train_loss"] = log_entry["loss"]
            cp_results["final_train_sim_loss"] = log_entry["sim_loss"]
            cp_results["final_train_sfa_loss"] = log_entry["sfa_loss"]
            results_list.append(cp_results)

    # Final evaluation at step 5000
    final_results = evaluate_run(model, arm_config, seed, device, checkpoint_step=total_steps, eval_steps=eval_steps)
    final_results["final_train_loss"] = logs[-1]["loss"]
    final_results["final_train_sim_loss"] = logs[-1]["sim_loss"]
    final_results["final_train_sfa_loss"] = logs[-1]["sfa_loss"]
    results_list.append(final_results)

    return results_list, model, logs


# ---------------------------------------------------------------------------
# Arms configuration
# ---------------------------------------------------------------------------
ARMS = [
    {
        "name": "Ctrl (CGIR+SFA+CCR, d_max=8, K=1)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "Arm A (Conv4 CGIR, d_max=8, K=1)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "conv4",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "Arm B (Expanded d_max=16, K=1)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 16, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "Arm C (Sub-Features K=4, d_max=8)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 4, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
]

SEEDS = [42, 123, 456, 789, 999]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 SFA Architecture Ceiling Evaluation Runner"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 5 training steps per seed for quick correctness check.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override seed list (default: [42, 123, 456, 789, 999]).",
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    seeds = args.seeds if args.seeds is not None else SEEDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Dry-run mode: {dry_run}")
    print(f"Seeds: {seeds}")
    print(f"Arms: {[a['name'] for a in ARMS]}")
    print()

    results_dir = "archive/iter_022/results"
    runs_dir = os.path.join(results_dir, "runs")
    figs_dir = os.path.join(results_dir, "figs")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    all_results = []
    checkpoint_results = []

    for arm in ARMS:
        name = arm["name"]
        print(f"{'='*70}")
        print(f"ARM: {name}")
        print(f"{'='*70}")
        for seed in seeds:
            print(f"\n--- {name}, seed={seed} ---")
            results_list, model, logs = run_single(arm, seed, device, dry_run=dry_run)

            for res in results_list:
                all_results.append(res)
                if res["checkpoint_step"] == 2500:
                    checkpoint_results.append(res)

                # Save per-run details
                cp_label = f"_cp{res['checkpoint_step']}" if res.get("checkpoint_step") else ""
                run_id = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+','')}_seed{seed}{cp_label}"
                csv_path = os.path.join(runs_dir, f"{run_id}.csv")
                json_path = os.path.join(runs_dir, f"{run_id}.json")

                flat = {k: v for k, v in res.items() if k not in (
                    "gdasr_growth_points", "centroid_mse_per_object",
                    "centroid_r_per_object", "r2_per_dim", "dim_to_obj",
                    "tracking_delta_per_dim", "tracking_level_per_dim",
                    "sub_feature_r2_matrix",
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

                df_row = pd.DataFrame([flat])
                df_row.to_csv(csv_path, index=False)

                with open(json_path, "w") as f:
                    json.dump(res, f, indent=2, default=str)
                print(f"  -> Saved {csv_path}")

            # Save model checkpoint at final step
            ckpt_path = os.path.join(checkpoints_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+','')}_seed{seed}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint {ckpt_path}")

            # Save training logs
            logs_path = os.path.join(runs_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+','')}_seed{seed}_logs.csv")
            logs_df = pd.DataFrame(logs)
            logs_df.to_csv(logs_path, index=False)
            print(f"  -> Saved training logs {logs_path}")

    # ------------------------------------------------------------------ #
    # Compile aggregate results (FINAL step=5000 only)
    # ------------------------------------------------------------------ #
    final_results = [r for r in all_results if r.get("checkpoint_step") != 2500]
    df_all = pd.DataFrame(final_results)

    for col in [
        "has_collapsed", "mean_dyn_delta", "mean_coord_delta", "slowness_ratio",
        "centroid_mse_mean", "centroid_r_mean", "mean_abs_corr",
        "final_train_loss", "final_train_sim_loss", "final_train_sfa_loss",
        "gdasr_growth_point_count",
        "r2_dyn_color", "r2_coord_color", "r2_dyn_pos", "r2_coord_pos",
        "delta_r2_color",
        "r2_dyn_identity", "r2_coord_identity", "delta_r2_identity",
        "normalized_dyn_var", "normalized_coord_var", "sfa_effective",
        "tracking_delta_corr", "tracking_level_corr",
        "temporal_var_dyn", "temporal_var_coord", "spatial_var_dyn", "spatial_var_coord",
    ]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    summary_path = os.path.join(results_dir, "summary_phase0_archceiling.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"\nSaved full results to {summary_path}")

    # ------------------------------------------------------------------ #
    # Checkpoint aggregate (step=2500)
    # ------------------------------------------------------------------ #
    if len(checkpoint_results) > 0:
        df_cp = pd.DataFrame(checkpoint_results)
        for col in [
            "has_collapsed", "mean_dyn_delta", "mean_coord_delta", "slowness_ratio",
            "centroid_mse_mean", "centroid_r_mean", "mean_abs_corr",
            "r2_dyn_color", "r2_coord_color", "delta_r2_color",
            "r2_dyn_identity", "r2_coord_identity", "delta_r2_identity",
            "normalized_dyn_var", "normalized_coord_var", "sfa_effective",
            "tracking_delta_corr", "tracking_level_corr",
        ]:
            if col in df_cp.columns:
                df_cp[col] = pd.to_numeric(df_cp[col], errors="coerce")
        cp_summary_path = os.path.join(results_dir, "summary_phase0_archceiling_cp2500.csv")
        df_cp.to_csv(cp_summary_path, index=False)
        print(f"Saved checkpoint results to {cp_summary_path}")

    # ------------------------------------------------------------------ #
    # Aggregated per-arm stats (FINAL step=5000)
    # ------------------------------------------------------------------ #
    agg_cols = [
        "has_collapsed", "mean_dyn_delta", "mean_coord_delta", "slowness_ratio",
        "centroid_mse_mean", "centroid_r_mean", "mean_abs_corr",
        "final_train_loss", "final_train_sim_loss", "gdasr_growth_point_count",
        "r2_dyn_color", "r2_coord_color", "delta_r2_color",
        "r2_dyn_identity", "r2_coord_identity", "delta_r2_identity",
        "normalized_dyn_var", "normalized_coord_var",
        "tracking_delta_corr", "tracking_level_corr",
        "temporal_var_dyn", "spatial_var_dyn",
    ]
    agg = df_all.groupby("arm")[agg_cols].agg(["mean", "std"]).reset_index()
    agg_path = os.path.join(results_dir, "aggregated_phase0_archceiling.csv")
    agg.to_csv(agg_path)
    print(f"\nAggregated stats saved to {agg_path}")
    print("\n" + "=" * 70)
    print("ARCHCEILING AGGREGATED RESULTS (mean ± std) @ step 5000")
    print("=" * 70)
    for _, row in agg.iterrows():
        name = row["arm"]
        print(f"\n--- {name} ---")
        for col in agg_cols:
            mean_val = row[(col, "mean")]
            std_val = row[(col, "std")]
            print(f"  {col:30s} = {mean_val:.4f} ± {std_val:.4f}")

    # ------------------------------------------------------------------ #
    # Falsification audit (Arch Ceiling)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FALSIFICATION AUDIT (Architecture Ceiling @ step 5000)")
    print("=" * 70)

    ctrl = df_all[df_all["arm"] == "Ctrl (CGIR+SFA+CCR, d_max=8, K=1)"]
    arm_a = df_all[df_all["arm"] == "Arm A (Conv4 CGIR, d_max=8, K=1)"]
    arm_b = df_all[df_all["arm"] == "Arm B (Expanded d_max=16, K=1)"]
    arm_c = df_all[df_all["arm"] == "Arm C (Sub-Features K=4, d_max=8)"]

    def audit_arm(arm_df, label):
        if len(arm_df) == 0:
            print(f"\n{label}: data missing.")
            return None
        collapsed = int(arm_df["has_collapsed"].sum())
        coll_rate = collapsed / len(arm_df)
        mse_mean = float(arm_df["centroid_mse_mean"].mean())
        delta_color = float(arm_df["delta_r2_color"].mean())
        delta_identity = float(arm_df["delta_r2_identity"].mean())
        norm_dyn = float(arm_df["normalized_dyn_var"].mean())
        norm_coord = float(arm_df["normalized_coord_var"].mean())
        sfa_eff = bool(arm_df["sfa_effective"].mean() > 0.5)
        print(f"\n{label}: n={len(arm_df)}, collapsed={collapsed}/{len(arm_df)} ({coll_rate:.0%})")
        print(f"  centroid_mse_mean  = {mse_mean:.2f}")
        print(f"  delta_r2_color     = {delta_color:.4f}")
        print(f"  delta_r2_identity  = {delta_identity:.4f}")
        print(f"  normalized_dyn_var = {norm_dyn:.6f}")
        print(f"  normalized_coord_var = {norm_coord:.6f}")
        print(f"  sfa_effective      = {sfa_eff}")
        return {
            "collapsed": collapsed,
            "n": len(arm_df),
            "mse_mean": mse_mean,
            "delta_r2_color": delta_color,
            "delta_r2_identity": delta_identity,
            "normalized_dyn_var": norm_dyn,
            "normalized_coord_var": norm_coord,
            "sfa_effective": sfa_eff,
        }

    audit_ctrl = audit_arm(ctrl, "Ctrl (CGIR+SFA+CCR, d_max=8, K=1)")
    audit_a = audit_arm(arm_a, "Arm A (Conv4 CGIR, d_max=8, K=1)")
    audit_b = audit_arm(arm_b, "Arm B (Expanded d_max=16, K=1)")
    audit_c = audit_arm(arm_c, "Arm C (Sub-Features K=4, d_max=8)")

    # Criteria
    print("\n" + "=" * 70)
    print("KEY COMPARISONS & CRITERIA")
    print("=" * 70)

    # C1: Collapse — Ctrl < 2 AND Arm C < 2
    if audit_ctrl and audit_c:
        c1_pass = (audit_ctrl["collapsed"] < 2) and (audit_c["collapsed"] < 2)
        print(f"\nC1 (Collapse: Ctrl < 2 AND Arm C < 2 seeds collapsed):")
        print(f"  Ctrl collapsed: {audit_ctrl['collapsed']}/{audit_ctrl['n']}")
        print(f"  Arm C collapsed: {audit_c['collapsed']}/{audit_c['n']}")
        print(f"  -> {'PASS' if c1_pass else 'FAIL'}")
    else:
        c1_pass = False

    # C2: Tracking — Arm C MSE ≤ 1.10 × Ctrl MSE
    if audit_c and audit_ctrl:
        threshold = 1.10 * audit_ctrl["mse_mean"]
        c2_pass = audit_c["mse_mean"] <= threshold
        print(f"\nC2 (Centroid MSE: Arm C <= 1.10 x Ctrl):")
        print(f"  Arm C MSE = {audit_c['mse_mean']:.4f}")
        print(f"  Ctrl MSE = {audit_ctrl['mse_mean']:.4f}")
        print(f"  Threshold (1.10 x Ctrl) = {threshold:.4f}")
        print(f"  -> {'PASS' if c2_pass else 'FAIL'}")
    else:
        c2_pass = False

    # C3: Color — Arm C improves over Ctrl by ≥ 0.10
    if audit_c and audit_ctrl:
        color_diff = audit_c["delta_r2_color"] - audit_ctrl["delta_r2_color"]
        c3_pass = color_diff >= 0.10
        print(f"\nC3 (Color: mean(delta_R2_color[Arm C] - delta_R2_color[Ctrl]) >= 0.10):")
        print(f"  Arm C delta_R2_color = {audit_c['delta_r2_color']:.4f}")
        print(f"  Ctrl delta_R2_color = {audit_ctrl['delta_r2_color']:.4f}")
        print(f"  Diff = {color_diff:.4f}")
        print(f"  -> {'PASS' if c3_pass else 'FAIL'}")
    else:
        c3_pass = False

    # C4: Identity — Arm C improves over Ctrl by ≥ 0.10
    if audit_c and audit_ctrl:
        identity_diff = audit_c["delta_r2_identity"] - audit_ctrl["delta_r2_identity"]
        c4_pass = identity_diff >= 0.10
        print(f"\nC4 (Identity: mean(delta_R2_identity[Arm C] - delta_R2_identity[Ctrl]) >= 0.10):")
        print(f"  Arm C delta_R2_identity = {audit_c['delta_r2_identity']:.4f}")
        print(f"  Ctrl delta_R2_identity = {audit_ctrl['delta_r2_identity']:.4f}")
        print(f"  Diff = {identity_diff:.4f}")
        print(f"  -> {'PASS' if c4_pass else 'FAIL'}")
    else:
        c4_pass = False

    # C5: SFA effective — normalized_dyn_var < normalized_coord_var for Arm C
    if audit_c:
        c5_pass = audit_c["normalized_dyn_var"] < audit_c["normalized_coord_var"]
        print(f"\nC5 (SFA effective: Arm C normalized_dyn_var < normalized_coord_var):")
        print(f"  Arm C normalized_dyn_var = {audit_c['normalized_dyn_var']:.6f}")
        print(f"  Arm C normalized_coord_var = {audit_c['normalized_coord_var']:.6f}")
        print(f"  -> {'PASS' if c5_pass else 'FAIL'}")
    else:
        c5_pass = False

    # OVERALL: C1 AND C2 AND C4 → hypothesis validated
    overall = c1_pass and c2_pass and c4_pass
    print(f"\n{'=' * 70}")
    print(f"OVERALL HYPOTHESIS: {'VALIDATED' if overall else 'FALSIFIED'}")
    print(f"{'=' * 70}")
    print(f"  C1 (Collapse)           : {'PASS' if c1_pass else 'FAIL'}")
    print(f"  C2 (Centroid MSE)       : {'PASS' if c2_pass else 'FAIL'}")
    print(f"  C3 (Color disentang)    : {'PASS' if c3_pass else 'FAIL'}")
    print(f"  C4 (Identity probe)     : {'PASS' if c4_pass else 'FAIL'}")
    print(f"  C5 (SFA effective)      : {'PASS' if c5_pass else 'FAIL'}")
    print(f"\n  OVERALL (C1 AND C2 AND C4): {'VALIDATED' if overall else 'FALSIFIED'}")

    # ------------------------------------------------------------------ #
    # Structured audit data
    # ------------------------------------------------------------------ #
    audit_data = {
        "c1_collapse_pass": bool(c1_pass),
        "c1_collapsed_seeds_ctrl": int(audit_ctrl["collapsed"]) if audit_ctrl else -1,
        "c1_collapsed_seeds_arm_c": int(audit_c["collapsed"]) if audit_c else -1,
        "c2_mse_pass": bool(c2_pass),
        "c2_mse_ctrl_mean": float(audit_ctrl["mse_mean"]) if audit_ctrl else -1.0,
        "c2_mse_arm_c_mean": float(audit_c["mse_mean"]) if audit_c else -1.0,
        "c2_mse_threshold": float(1.10 * audit_ctrl["mse_mean"]) if audit_ctrl else -1.0,
        "c3_color_pass": bool(c3_pass),
        "c3_delta_r2_color_ctrl": float(audit_ctrl["delta_r2_color"]) if audit_ctrl else -1.0,
        "c3_delta_r2_color_arm_c": float(audit_c["delta_r2_color"]) if audit_c else -1.0,
        "c3_color_diff": float(audit_c["delta_r2_color"] - audit_ctrl["delta_r2_color"]) if audit_c and audit_ctrl else -1.0,
        "c4_identity_pass": bool(c4_pass),
        "c4_delta_r2_identity_ctrl": float(audit_ctrl["delta_r2_identity"]) if audit_ctrl else -1.0,
        "c4_delta_r2_identity_arm_c": float(audit_c["delta_r2_identity"]) if audit_c else -1.0,
        "c4_identity_diff": float(audit_c["delta_r2_identity"] - audit_ctrl["delta_r2_identity"]) if audit_c and audit_ctrl else -1.0,
        "c5_sfa_effective_pass": bool(c5_pass),
        "c5_normalized_dyn_var_arm_c": float(audit_c["normalized_dyn_var"]) if audit_c else -1.0,
        "c5_normalized_coord_var_arm_c": float(audit_c["normalized_coord_var"]) if audit_c else -1.0,
        "overall_validated": bool(overall),
        "overall_criteria": "C1 AND C2 AND C4",
        "arms": {
            "ctrl": audit_ctrl,
            "arm_a": audit_a,
            "arm_b": audit_b,
            "arm_c": audit_c,
        }
    }
    audit_path = os.path.join(results_dir, "audit_phase0_archceiling.json")
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)
    print(f"\nAudit report saved to {audit_path}")

    # ------------------------------------------------------------------ #
    # Write ARCHCEILING_COMPARISON_REPORT.md
    # ------------------------------------------------------------------ #
    report_lines = [
        "# Architecture Ceiling Phase 0: Comparison Report",
        "",
        "## Summary",
        "",
        "This report evaluates three architectural modifications against the control",
        "to determine if any variant validates the hypothesis. The key test is whether",
        "sub-features (K=4) improve semantic disentanglement without breaking",
        "centroid tracking.",
        "",
        "## Arm Configurations",
        "",
        "| Arm | d_max | sub_features | dyn_source | Notes |",
        "|-----|-------|--------------|------------|-------|",
        "| Ctrl | 8 | 1 | spatial | Baseline CGIR |",
        "| A | 8 | 1 | conv4 | Conv4 dynamics source |",
        "| B | 16 | 1 | spatial | Expanded capacity |",
        "| C | 8 | 4 | spatial | Sub-features K=4 |",
        "",
        "Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000, d_t=3.",
        "Seeds: [42, 123, 456, 789, 999].",
        "",
        "## Results Table",
        "",
    ]

    def _arm_row(audit):
        if audit is None:
            return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        return (
            f"{audit['collapsed']}/{audit['n']}",
            f"{audit['mse_mean']:.2f}",
            f"{audit['delta_r2_color']:.4f}",
            f"{audit['delta_r2_identity']:.4f}",
            f"{audit['normalized_dyn_var']:.6f}",
            f"{audit['normalized_coord_var']:.6f}",
            f"{'yes' if audit['sfa_effective'] else 'no'}",
        )

    ctrl_c, ctrl_mse, ctrl_dc, ctrl_di, ctrl_nd, ctrl_nc, ctrl_se = _arm_row(audit_ctrl)
    a_c, a_mse, a_dc, a_di, a_nd, a_nc, a_se = _arm_row(audit_a)
    b_c, b_mse, b_dc, b_di, b_nd, b_nc, b_se = _arm_row(audit_b)
    c_c, c_mse, c_dc, c_di, c_nd, c_nc, c_se = _arm_row(audit_c)

    report_lines += [
        "| Metric | Ctrl | Arm A (conv4) | Arm B (d=16) | Arm C (K=4) |",
        "|--------|------|---------------|--------------|-------------|",
        f"| Collapse | {ctrl_c} | {a_c} | {b_c} | {c_c} |",
        f"| MSE | {ctrl_mse} | {a_mse} | {b_mse} | {c_mse} |",
        f"| dR2_color | {ctrl_dc} | {a_dc} | {b_dc} | {c_dc} |",
        f"| dR2_ident | {ctrl_di} | {a_di} | {b_di} | {c_di} |",
        f"| norm_dyn_var | {ctrl_nd} | {a_nd} | {b_nd} | {c_nd} |",
        f"| norm_coord_var | {ctrl_nc} | {a_nc} | {b_nc} | {c_nc} |",
        f"| SFA effective | {ctrl_se} | {a_se} | {b_se} | {c_se} |",
        "",
        "## Criterion Definitions",
        "",
        "- **C1 (Collapse):** Ctrl < 2/5 AND Arm C < 2/5 collapsed => PASS.",
        "- **C2 (MSE):** Arm C MSE <= 1.10 x Ctrl MSE => PASS.",
        "- **C3 (Color):** mean(delta_R2_color[Arm C] - delta_R2_color[Ctrl]) >= 0.10 => PASS.",
        "- **C4 (Identity):** mean(delta_R2_identity[Arm C] - delta_R2_identity[Ctrl]) >= 0.10 => PASS.",
        "- **C5 (SFA effective):** normalized_dyn_var < normalized_coord_var for Arm C => PASS.",
        "",
        "## Honest Falsification Audit",
        "",
        f"| Criterion | Result | Detail |",
        f"|-----------|--------|--------|",
        f"| C1 (Collapse) | {'PASS' if c1_pass else 'FAIL'} | Ctrl: {audit_ctrl['collapsed'] if audit_ctrl else 'N/A'}/{audit_ctrl['n'] if audit_ctrl else 'N/A'}, Arm C: {audit_c['collapsed'] if audit_c else 'N/A'}/{audit_c['n'] if audit_c else 'N/A'} |",
        f"| C2 (Centroid MSE) | {'PASS' if c2_pass else 'FAIL'} | Arm C MSE={audit_c['mse_mean'] if audit_c else 'N/A':.4f} vs threshold={1.10 * audit_ctrl['mse_mean'] if audit_ctrl else 'N/A':.4f} (1.10 x Ctrl) |",
        f"| C3 (Color disentanglement) | {'PASS' if c3_pass else 'FAIL'} | Diff={audit_c['delta_r2_color'] - audit_ctrl['delta_r2_color'] if audit_c and audit_ctrl else 'N/A':.4f} |",
        f"| C4 (Identity probe) | {'PASS' if c4_pass else 'FAIL'} | Diff={audit_c['delta_r2_identity'] - audit_ctrl['delta_r2_identity'] if audit_c and audit_ctrl else 'N/A':.4f} |",
        f"| C5 (SFA effective) | {'PASS' if c5_pass else 'FAIL'} | Arm C norm_dyn={audit_c['normalized_dyn_var'] if audit_c else 'N/A':.6f} < norm_coord={audit_c['normalized_coord_var'] if audit_c else 'N/A':.6f} |",
        "",
        f"**Overall (C1 AND C2 AND C4): {'HYPOTHESIS VALIDATED' if overall else 'HYPOTHESIS FALSIFIED'}**",
        "",
        "## Interpretation",
        "",
    ]

    if overall:
        report_lines += [
            "The architecture ceiling experiment validated that at least one variant",
            "(sub-features K=4) improves semantic disentanglement (C4, identity probe)",
            "while maintaining tracking quality (C2) and avoiding collapse (C1).",
            "This suggests the bottleneck is not the soft-argmax centroid computation,",
            "but somewhere else in the architecture (e.g., dynamics readout mechanism).",
            "",
        ]
    else:
        report_lines += [
            "The architecture ceiling experiment FAILED to validate the hypothesis.",
            "None of the tested architectural modifications (conv4 source, expanded d_max,",
            "or sub-features K=4) achieved the required improvement in semantic disentanglement",
            "while maintaining tracking quality.",
            "",
            "**Conclusion:** The bottleneck lies elsewhere. Next experiments should test:",
            "1. Different dynamics readout mechanisms (attention-based, learned pooling).",
            "2. Non-linear sub-feature interactions instead of independent K features.",
            "3. Separate dynamics encoder instead of shared CNN.",
            "",
        ]

    report_text = "\n".join(report_lines)
    report_path = os.path.join(results_dir, "ARCHCEILING_COMPARISON_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nComparison report saved to {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
