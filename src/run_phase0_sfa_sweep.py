#!/usr/bin/env python3
"""
Phase 0 SFA Weight Sweep Runner.

Sweeps sfa_weight across [0.1, 1.0, 5.0, 10.0, 25.0] to find where SFA
becomes effective. Tests whether gradient parity between SFA and VICReg
produces measurable identity-position separation.

7 arms x 5 seeds x 5000 steps.
Seeds: [42, 123, 456, 789, 999]
Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000
d_t=3, gdasr_log_only=True

Arm A1 (Ctrl sfa=0.1):     d_max=8, CGIR, CCR, d_t=3, sfa_weight=0.1
Arm A2 (sfa=1.0):           d_max=8, CGIR, CCR, d_t=3, sfa_weight=1.0
Arm A3 (sfa=5.0):           d_max=8, CGIR, CCR, d_t=3, sfa_weight=5.0
Arm A4 (sfa=10.0):          d_max=8, CGIR, CCR, d_t=3, sfa_weight=10.0
Arm A5 (sfa=25.0 fixed):    d_max=8, CGIR, CCR, d_t=3, sfa_weight=25.0
Arm A6 (sfa=25.0 ramp):     d_max=8, CGIR, CCR, d_t=3, sfa_weight ramp 0.1->25.0 over 1000 steps
Arm B  (d_max=16 sfa=10.0): d_max=16, CGIR, CCR, d_t=3, sfa_weight=10.0

Results saved to archive/iter_023/results/
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
# Normalized temporal variance
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
# Centroid tracking quality
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
# Per-sub-feature identity probes
# ---------------------------------------------------------------------------
def compute_sub_feature_probes(z_dyn_arr, colors_arr, radii_arr, d_t, sub_features, num_samples, train_ratio=0.5):
    """
    For each (channel c, sub-feature k), fit R² against [R, G, B, radius_norm] individually.
    Returns a (d_t, sub_features, 4) array of R² values.
    """
    max_radius = 20.0
    n_train = int(num_samples * train_ratio)

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
        z_dyn_pooled = z_dyn_arr[:, :d_t * sub_features].reshape(num_samples, d_t, sub_features).mean(axis=2)
    else:
        z_dyn_pooled = z_dyn_arr[:, :d_t]

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

    sub_feature_r2_matrix = None
    if sub_features > 1:
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
# Evaluation helper
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

    # Tracking quality
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
        # Normalized temporal variance
        "temporal_var_dyn": norm_tv["temporal_var_dyn"],
        "temporal_var_coord": norm_tv["temporal_var_coord"],
        "spatial_var_dyn": norm_tv["spatial_var_dyn"],
        "spatial_var_coord": norm_tv["spatial_var_coord"],
        "normalized_dyn_var": norm_tv["normalized_dyn_var"],
        "normalized_coord_var": norm_tv["normalized_coord_var"],
        "sfa_effective": norm_tv["sfa_effective"],
        # Tracking
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
    Handles sfa_weight ramping for Arm A6.
    """
    name = arm_config["name"]
    primary_obj = arm_config["primary_objective"]
    sfa_weight = arm_config.get("sfa_weight", 0.1)
    sfa_ramp = arm_config.get("sfa_ramp", False)
    sfa_ramp_start = arm_config.get("sfa_ramp_start", 0.1)
    sfa_ramp_end = arm_config.get("sfa_ramp_end", 25.0)
    sfa_ramp_steps = arm_config.get("sfa_ramp_steps", 1000)
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

    # For model init, use the base sfa_weight (0.1 for ramp arms)
    init_sfa = sfa_ramp_start if sfa_ramp else sfa_weight

    model = NonParametricJEPASpatial(
        d_max=d_max,
        h=3,
        k=4,
        cooldown=300,
        stabilization_period=100,
        pos_encoding=pos_encoding,
        primary_objective=primary_obj,
        sfa_weight=init_sfa,
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

        # Compute effective sfa_weight for ramp arms
        if sfa_ramp:
            effective_sfa_weight = sfa_ramp_start + (sfa_ramp_end - sfa_ramp_start) * min(1.0, step / sfa_ramp_steps)
        else:
            effective_sfa_weight = None  # use model's config sfa_weight

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
            sfa_weight=effective_sfa_weight,  # forward-call override (None = use config)
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
        if sfa_ramp:
            log_entry["effective_sfa_weight"] = effective_sfa_weight
        logs.append(log_entry)

        if step % 1000 == 0 or step == total_steps:
            sfa_log = f"  sfa_eff={effective_sfa_weight:.3f}" if sfa_ramp else f"  sfa={effective_sfa_weight if effective_sfa_weight is not None else sfa_weight:.1f}"
            print(f"  [{name}] seed={seed} step={step:5d}/{total_steps}  "
                  f"loss={log_entry['loss']:.4f}  sim={log_entry['sim_loss']:.4f}  "
                  f"sfa={log_entry['sfa_loss']:.4f}  ccr_s={log_entry['ccr_smooth_loss']:.4f}  "
                  f"ccr_sp={log_entry['ccr_spatial_loss']:.4f}" + sfa_log)

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
# Arms configuration — SFA weight sweep
# ---------------------------------------------------------------------------
ARMS = [
    {
        "name": "A1 (Ctrl sfa=0.1)",
        "primary_objective": "sfa", "sfa_weight": 0.1,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "A2 (sfa=1.0)",
        "primary_objective": "sfa", "sfa_weight": 1.0,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "A3 (sfa=5.0)",
        "primary_objective": "sfa", "sfa_weight": 5.0,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "A4 (sfa=10.0)",
        "primary_objective": "sfa", "sfa_weight": 10.0,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "A5 (sfa=25.0 fixed)",
        "primary_objective": "sfa", "sfa_weight": 25.0,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "A6 (sfa=25.0 ramp)",
        "primary_objective": "sfa", "sfa_weight": 0.1,  # init value, ramp overrides
        "sfa_ramp": True,
        "sfa_ramp_start": 0.1,
        "sfa_ramp_end": 25.0,
        "sfa_ramp_steps": 1000,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "B (d_max=16 sfa=10.0)",
        "primary_objective": "sfa", "sfa_weight": 10.0,
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 16, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
]

SEEDS = [42, 123, 456, 789, 999]


def _sanitize_arm_name(name):
    """Convert arm name to a filesystem-safe identifier."""
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 SFA Weight Sweep Runner"
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

    results_dir = "archive/iter_023/results"
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
                safe_name = _sanitize_arm_name(name)
                run_id = f"{safe_name}_seed{seed}{cp_label}"
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
            safe_name = _sanitize_arm_name(name)
            ckpt_path = os.path.join(checkpoints_dir, f"{safe_name}_seed{seed}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint {ckpt_path}")

            # Save training logs
            logs_path = os.path.join(runs_dir, f"{safe_name}_seed{seed}_logs.csv")
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

    summary_path = os.path.join(results_dir, "summary_phase0_sfa_sweep.csv")
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
        cp_summary_path = os.path.join(results_dir, "summary_phase0_sfa_sweep_cp2500.csv")
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
    agg_path = os.path.join(results_dir, "aggregated_phase0_sfa_sweep.csv")
    agg.to_csv(agg_path)
    print(f"\nAggregated stats saved to {agg_path}")
    print("\n" + "=" * 70)
    print("SFA WEIGHT SWEEP AGGREGATED RESULTS (mean ± std) @ step 5000")
    print("=" * 70)
    for _, row in agg.iterrows():
        name = row["arm"]
        print(f"\n--- {name} ---")
        for col in agg_cols:
            mean_val = row[(col, "mean")]
            std_val = row[(col, "std")]
            print(f"  {col:30s} = {mean_val:.4f} ± {std_val:.4f}")

    # ------------------------------------------------------------------ #
    # Falsification Audit — SFA Weight Sweep
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FALSIFICATION AUDIT (SFA Weight Sweep @ step 5000)")
    print("=" * 70)

    a1 = df_all[df_all["arm"] == "A1 (Ctrl sfa=0.1)"]
    a2 = df_all[df_all["arm"] == "A2 (sfa=1.0)"]
    a3 = df_all[df_all["arm"] == "A3 (sfa=5.0)"]
    a4 = df_all[df_all["arm"] == "A4 (sfa=10.0)"]
    a5 = df_all[df_all["arm"] == "A5 (sfa=25.0 fixed)"]
    a6 = df_all[df_all["arm"] == "A6 (sfa=25.0 ramp)"]
    arm_b = df_all[df_all["arm"] == "B (d_max=16 sfa=10.0)"]

    a_arms = [a2, a3, a4, a5, a6]
    a_arm_names = ["A2 (sfa=1.0)", "A3 (sfa=5.0)", "A4 (sfa=10.0)", "A5 (sfa=25.0 fixed)", "A6 (sfa=25.0 ramp)"]

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
        sfa_eff_count = int(arm_df["sfa_effective"].sum())
        sfa_eff_rate = sfa_eff_count / len(arm_df)
        print(f"\n{label}: n={len(arm_df)}, collapsed={collapsed}/{len(arm_df)} ({coll_rate:.0%})")
        print(f"  centroid_mse_mean      = {mse_mean:.2f}")
        print(f"  delta_r2_color         = {delta_color:.4f}")
        print(f"  delta_r2_identity      = {delta_identity:.4f}")
        print(f"  normalized_dyn_var     = {norm_dyn:.6f}")
        print(f"  normalized_coord_var   = {norm_coord:.6f}")
        print(f"  sfa_effective (C5)     = {sfa_eff_count}/{len(arm_df)} ({sfa_eff_rate:.0%})")
        return {
            "collapsed": collapsed,
            "n": len(arm_df),
            "mse_mean": mse_mean,
            "delta_r2_color": delta_color,
            "delta_r2_identity": delta_identity,
            "normalized_dyn_var": norm_dyn,
            "normalized_coord_var": norm_coord,
            "sfa_effective_count": sfa_eff_count,
            "sfa_effective_rate": sfa_eff_rate,
        }

    audit_a1 = audit_arm(a1, "A1 (Ctrl sfa=0.1)")
    audit_a2 = audit_arm(a2, "A2 (sfa=1.0)")
    audit_a3 = audit_arm(a3, "A3 (sfa=5.0)")
    audit_a4 = audit_arm(a4, "A4 (sfa=10.0)")
    audit_a5 = audit_arm(a5, "A5 (sfa=25.0 fixed)")
    audit_a6 = audit_arm(a6, "A6 (sfa=25.0 ramp)")
    audit_b = audit_arm(arm_b, "B (d_max=16 sfa=10.0)")

    # ------------------------------------------------------------------ #
    # PRIMARY: delta_R2_color improvement over A1 baseline >= 0.10
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("PRIMARY FALSIFICATION")
    print("delta_R2_color improvement over A1 baseline >= 0.10 for at least one arm")
    print("=" * 70)

    a1_delta_color = float(audit_a1["delta_r2_color"]) if audit_a1 else 0.0
    primary_pass = False
    best_delta_color_arm = None
    best_delta_color_diff = -float("inf")

    for audit_obj, arm_name in zip(
        [audit_a2, audit_a3, audit_a4, audit_a5, audit_a6],
        a_arm_names
    ):
        if audit_obj is None:
            continue
        diff = audit_obj["delta_r2_color"] - a1_delta_color
        meets_threshold = diff >= 0.10
        if meets_threshold:
            primary_pass = True
        print(f"  {arm_name:25s} dR2_color={audit_obj['delta_r2_color']:.4f}  "
              f"Δ={diff:+.4f}  {'** MEETS THRESHOLD **' if meets_threshold else ''}")
        if diff > best_delta_color_diff:
            best_delta_color_diff = diff
            best_delta_color_arm = arm_name

    print(f"\n  A1 baseline delta_R2_color = {a1_delta_color:.4f}")
    if primary_pass:
        print(f"  -> PASS: {best_delta_color_arm} achieves Δ={best_delta_color_diff:.4f}")
    else:
        print(f"  -> FAIL: No arm achieves Δ >= 0.10. Best is {best_delta_color_arm} at Δ={best_delta_color_diff:.4f}")

    # ------------------------------------------------------------------ #
    # COMPOSITE M2 VIABILITY
    # Existence of sfa_weight where >= 3/5 seeds simultaneously:
    #   (a) C5: sfa_effective
    #   (b) delta_R2_color improvement >= 0.10 over A1
    #   (c) per_dim_std > 0.5 (non-collapse)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("COMPOSITE M2 VIABILITY CRITERION")
    print("Exists sfa_weight where >= 3/5 seeds satisfy C5 + dR2_color >= 0.10 + non-collapse")
    print("=" * 70)

    composites = {}
    composite_pass = False
    best_composite_arm = None
    best_composite_count = 0

    for arm_idx, arm_df in enumerate(a_arms):
        arm_name = a_arm_names[arm_idx]
        if len(arm_df) == 0:
            continue

        pass_count = 0
        per_seed_passes = []
        for i, (_, row) in enumerate(arm_df.iterrows()):
            c5 = bool(row.get("sfa_effective", False))
            delta_color_impr = row.get("delta_r2_color", 0.0) - a1_delta_color
            delta_color_ok = delta_color_impr >= 0.10
            # Non-collapse: all per-dim std > 0.5
            per_dim_std = row.get("per_dim_std", [])
            if isinstance(per_dim_std, str):
                try:
                    per_dim_std = json.loads(per_dim_std)
                except:
                    per_dim_std = []
            non_collapse = all(s > 0.5 for s in per_dim_std) if per_dim_std else True

            passes_all = c5 and delta_color_ok and non_collapse
            if passes_all:
                pass_count += 1
            per_seed_passes.append({
                "seed": row.get("seed", i),
                "c5": c5,
                "delta_color_improved": delta_color_ok,
                "delta_color_val": row.get("delta_r2_color", 0.0),
                "non_collapse": non_collapse,
                "passes_all": passes_all,
            })

        passes_composite = pass_count >= 3
        if passes_composite:
            composite_pass = True
        if pass_count > best_composite_count:
            best_composite_count = pass_count
            best_composite_arm = arm_name

        print(f"\n  {arm_name}: {pass_count}/5 seeds pass all three criteria")
        for sp in per_seed_passes:
            markers = []
            if sp["c5"]: markers.append("C5✓")
            else: markers.append("C5✗")
            if sp["delta_color_improved"]: markers.append("dR2✓")
            else: markers.append(f"dR2✗({sp['delta_color_val']:.3f})")
            if sp["non_collapse"]: markers.append("NC✓")
            else: markers.append("NC✗")
            pass_str = "PASS" if sp["passes_all"] else "FAIL"
            print(f"    seed={sp['seed']:3d}  {' '.join(markers):15s}  -> {pass_str}")

        composites[arm_name] = {
            "pass_count": pass_count,
            "passes_composite": passes_composite,
            "per_seed": per_seed_passes,
        }

    if composite_pass:
        print(f"\n  -> PASS: {best_composite_arm} has {best_composite_count}/5 seeds simultaneously passing C5 + dR2>=0.10 + non-collapse")
    else:
        print(f"\n  -> FAIL: No sfa_weight achieves >= 3/5 composite seeds. Best is {best_composite_arm} at {best_composite_count}/5.")

    # ------------------------------------------------------------------ #
    # TERTIARY: If all sfa_weight >= 5.0 arms collapse >= 3/5 seeds
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("TERTIARY FALSIFICATION")
    print("If all sfa_weight >= 5.0 arms collapse >= 3/5 seeds (even with ramping)")
    print("=" * 70)

    # Arms with sfa_weight >= 5.0: A3, A4, A5, A6
    heavy_arms = [(audit_a3, "A3 (sfa=5.0)"), (audit_a4, "A4 (sfa=10.0)"),
                   (audit_a5, "A5 (sfa=25.0 fixed)"), (audit_a6, "A6 (sfa=25.0 ramp)")]

    all_heavy_collapse = True
    heavy_any_pass = False
    for haudit, hname in heavy_arms:
        if haudit is None:
            continue
        heavy_collapse_rate = haudit["collapsed"] / haudit["n"]
        heavy_pass = haudit["collapsed"] >= 3
        status = "COLLAPSE >= 3/5" if heavy_pass else f"collapse {haudit['collapsed']}/{haudit['n']} (< 3)"
        if not heavy_pass:
            all_heavy_collapse = False
            heavy_any_pass = True
        print(f"  {hname:25s}: {status}")

    if all_heavy_collapse:
        print(f"\n  -> TERTIARY TRIGGERED: All high-sfa_weight arms collapse >= 3/5 seeds.")
        print(f"     SFA-VICReg conflict appears unresolvable at effective SFA strengths.")
    else:
        print(f"\n  -> Not triggered: At least one high-sfa-weight arm has < 3/5 collapsed seeds.")

    # ------------------------------------------------------------------ #
    # OVERALL CONCLUSION
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("OVERALL CONCLUSION")
    print("=" * 70)
    print(f"  PRIMARY   (delta_R2_color >= 0.10 over A1)    : {'PASS' if primary_pass else 'FAIL'}")
    print(f"  COMPOSITE (>= 3/5 seeds pass C5+dR2+non-coll) : {'PASS' if composite_pass else 'FAIL'}")
    print(f"  TERTIARY  (all heavy arms collapse)           : {'TRIGGERED' if all_heavy_collapse else 'NOT TRIGGERED'}")

    if primary_pass:
        print(f"\n  HYPOTHESIS: NOT FALSIFIED — at least one sfa_weight produces measurable identity-position separation.")
    else:
        print(f"\n  HYPOTHESIS: FALSIFIED — no sfa_weight in the sweep produces delta_R2_color >= 0.10 over A1 baseline.")
        if composite_pass:
            print(f"  NOTE: Composite M2 viability passes, but delta_R2_color improvement is insufficient.")

    # ------------------------------------------------------------------ #
    # Structured audit data
    # ------------------------------------------------------------------ #
    audit_data = {
        "primary_delta_r2_color_pass": bool(primary_pass),
        "primary_a1_baseline": float(a1_delta_color),
        "best_delta_r2_color_arm": best_delta_color_arm,
        "best_delta_r2_color_diff": float(best_delta_color_diff),
        "composite_m2_viable_pass": bool(composite_pass),
        "best_composite_arm": best_composite_arm,
        "best_composite_seed_count": best_composite_count,
        "tertiary_all_heavy_collapse": bool(all_heavy_collapse),
        "a1_baseline": audit_a1,
        "arms": {
            "a2_sfa1": audit_a2,
            "a3_sfa5": audit_a3,
            "a4_sfa10": audit_a4,
            "a5_sfa25_fixed": audit_a5,
            "a6_sfa25_ramp": audit_a6,
            "b_d16_sfa10": audit_b,
        },
        "composite_details": {
            arm_name: composite.get("per_seed", [])
            for arm_name, composite in composites.items()
        },
    }
    audit_path = os.path.join(results_dir, "audit_phase0_sfa_sweep.json")
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)
    print(f"\nAudit report saved to {audit_path}")

    # ------------------------------------------------------------------ #
    # SFA_WEIGHT_COMPARISON_REPORT.md
    # ------------------------------------------------------------------ #
    report_lines = [
        "# SFA Weight Sweep: Comparison Report",
        "",
        "## Summary",
        "",
        "This report evaluates the effect of varying sfa_weight across [0.1, 1.0,",
        "5.0, 10.0, 25.0] to find where SFA becomes effective for identity-position",
        "separation. Arm A6 tests a ramp schedule; Arm B tests expanded capacity.",
        "",
        "## Arm Configurations",
        "",
        "| Arm | d_max | sfa_weight | Mode | Notes |",
        "|-----|-------|------------|------|-------|",
        "| A1 | 8 | 0.1 | fixed | Baseline (Ctrl iter_022) |",
        "| A2 | 8 | 1.0 | fixed | 10x increase, gradient sensitivity |",
        "| A3 | 8 | 5.0 | fixed | 20% of VICReg strength |",
        "| A4 | 8 | 10.0 | fixed | 40% of VICReg strength |",
        "| A5 | 8 | 25.0 | fixed | Full parity with VICReg |",
        "| A6 | 8 | 0.1→25.0 | ramp | Linear ramp over 1000 steps |",
        "| B | 16 | 10.0 | fixed | Expanded channels, pre-set |",
        "",
        "Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000, d_t=3.",
        "Seeds: [42, 123, 456, 789, 999].",
        "",
        "## Results Table",
        "",
    ]

    def _arm_row_swa(audit):
        if audit is None:
            return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        return (
            f"{audit['collapsed']}/{audit['n']}",
            f"{audit['mse_mean']:.2f}",
            f"{audit['delta_r2_color']:.4f}",
            f"{audit['delta_r2_identity']:.4f}",
            f"{audit['normalized_dyn_var']:.6f}",
            f"{audit['normalized_coord_var']:.6f}",
            f"{audit['sfa_effective_count']}/{audit['n']}",
        )

    a1_c, a1_mse, a1_dc, a1_di, a1_nd, a1_nc, a1_sc = _arm_row_swa(audit_a1)
    a2_c, a2_mse, a2_dc, a2_di, a2_nd, a2_nc, a2_sc = _arm_row_swa(audit_a2)
    a3_c, a3_mse, a3_dc, a3_di, a3_nd, a3_nc, a3_sc = _arm_row_swa(audit_a3)
    a4_c, a4_mse, a4_dc, a4_di, a4_nd, a4_nc, a4_sc = _arm_row_swa(audit_a4)
    a5_c, a5_mse, a5_dc, a5_di, a5_nd, a5_nc, a5_sc = _arm_row_swa(audit_a5)
    a6_c, a6_mse, a6_dc, a6_di, a6_nd, a6_nc, a6_sc = _arm_row_swa(audit_a6)
    b_c, b_mse, b_dc, b_di, b_nd, b_nc, b_sc = _arm_row_swa(audit_b)

    report_lines += [
        "| Metric | A1 (0.1) | A2 (1.0) | A3 (5.0) | A4 (10.0) | A5 (25.0) | A6 (ramp) | B (d16) |",
        "|--------|----------|----------|----------|-----------|-----------|-----------|---------|",
        f"| Collapse | {a1_c} | {a2_c} | {a3_c} | {a4_c} | {a5_c} | {a6_c} | {b_c} |",
        f"| MSE | {a1_mse} | {a2_mse} | {a3_mse} | {a4_mse} | {a5_mse} | {a6_mse} | {b_mse} |",
        f"| dR2_color | {a1_dc} | {a2_dc} | {a3_dc} | {a4_dc} | {a5_dc} | {a6_dc} | {b_dc} |",
        f"| dR2_ident | {a1_di} | {a2_di} | {a3_di} | {a4_di} | {a5_di} | {a6_di} | {b_di} |",
        f"| norm_dyn_var | {a1_nd} | {a2_nd} | {a3_nd} | {a4_nd} | {a5_nd} | {a6_nd} | {b_nd} |",
        f"| norm_coord_var | {a1_nc} | {a2_nc} | {a3_nc} | {a4_nc} | {a5_nc} | {a6_nc} | {b_nc} |",
        f"| C5 (sfa_eff) | {a1_sc} | {a2_sc} | {a3_sc} | {a4_sc} | {a5_sc} | {a6_sc} | {b_sc} |",
        "",
        "## Criterion Definitions",
        "",
        "- **PRIMARY:** delta_R2_color improvement over A1 >= 0.10 for at least one arm.",
        "- **COMPOSITE M2:** Exists sfa_weight where >= 3/5 seeds pass C5 + dR2>=0.10 + non-collapse.",
        "- **TERTIARY:** All sfa>=5.0 arms collapse >= 3/5 seeds => unresolvable SFA-VICReg conflict.",
        "- **C5 (reframed):** Gradient propagation check — necessary but not sufficient.",
        "",
        "## Honest Falsification Audit",
        "",
        f"| Criterion | Result | Detail |",
        f"|-----------|--------|--------|",
        f"| PRIMARY (dR2_color >= 0.10 over A1) | {'PASS' if primary_pass else 'FAIL'} | Best: {best_delta_color_arm} Δ={best_delta_color_diff:.4f} |",
        f"| COMPOSITE M2 Viability | {'PASS' if composite_pass else 'FAIL'} | Best: {best_composite_arm} {best_composite_count}/5 seeds |",
        f"| TERTIARY (all heavy collapse) | {'TRIGGERED' if all_heavy_collapse else 'NOT TRIGGERED'} | SFA-VICRes conflict {'unresolvable' if all_heavy_collapse else 'resolvable in some arms'} |",
        "",
        f"**Overall: {'HYPOTHESIS NOT FALSIFIED' if primary_pass else 'HYPOTHESIS FALSIFIED'}**",
        "",
        "## Interpretation",
        "",
    ]

    if primary_pass:
        report_lines += [
            f"The SFA weight sweep found that {best_delta_color_arm} achieves",
            f"delta_R2_color improvement of {best_delta_color_diff:.4f} over the A1 baseline,",
            "meeting the primary criterion. SFA at sufficient strength produces",
            "measurable identity-position separation.",
            "",
            "Next steps:",
            "1. Fine-tune around the best sfa_weight value.",
            "2. If composite M2 also passes, the weight is viable for broader use.",
            "3. Investigate whether Arm B's expanded capacity further improves results.",
            "",
        ]
    else:
        report_lines += [
            "The SFA weight sweep FAILED to validate the hypothesis.",
            "No sfa_weight in the sweep produced delta_R2_color improvement >= 0.10",
            "over the A1 (sfa=0.1) baseline.",
            "",
            "Possible conclusions:",
            "1. SFA's slowness prior does not naturally separate identity from position",
            "   in this architecture, even at gradient parity.",
            "2. The VICReg objectives (variance+invariance+covariance) dominate the",
            "   dynamics at all sfa_weight levels tested.",
            "3. The shared CNN encoder cannot simultaneously support position tracking",
            "   (z_coord) and identity/speed encoding (z_dyn).",
            "",
            "If composite M2 viability passes despite failing the primary criterion,",
            "this confirms that SFA's gradient propagation works (C5 passes) and",
            "representations remain non-degenerate, but the separation is simply not",
            "large enough to matter for the identity probe.",
            "",
            "Recommended next directions:",
            "1. Separate dynamics encoder instead of shared architecture.",
            "2. Learnable SFA vs VICReg weight balance (meta-optimization).",
            "3. Different slowness formulation (e.g., slow-feature analysis on temporal",
            "   differences rather than direct z_dyn comparison).",
            "",
        ]

    # Arm B pre-commitment note
    report_lines += [
        "## Arm B Pre-Commitment Note",
        "",
        "Arm B was pre-set to sfa_weight=10.0. If the optimal sfa_weight (from A-arms)",
        "is between 5.0 and 10.0, Arm B may fail for the wrong reason. Arm B failure",
        "should not be interpreted as falsifying the compound hypothesis on its own;",
        "the A-arm sweep data supersedes it.",
        "",
    ]

    # Training log analysis
    if not dry_run:
        report_lines += [
            "## Training Dynamics",
            "",
            "See training logs in `runs/` for per-step loss curves.",
            "Arm A6 (ramp) should show a transition in sfa_loss around step 1000",
            "as effective_sfa_weight reaches 25.0.",
            "",
        ]

    report_text = "\n".join(report_lines)
    report_path = os.path.join(results_dir, "SFA_WEIGHT_COMPARISON_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nComparison report saved to {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
