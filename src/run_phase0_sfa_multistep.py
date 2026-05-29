#!/usr/bin/env python3
"""
Phase 0 Multi-Step SFA Runner (iter_024).

Arms:
  A: k=20,  d_max=8  (5 seeds)
  B: k=50,  d_max=8  (5 seeds)
  C: k=100, d_max=8  (5 seeds)
  D: Contrastive NT-Xent, d_max=8  (5 seeds)
  E: k=50,  d_max=16 (5 seeds)
  F: Diagnostic (sim_weight=0, k=50, d_max=8)  (1 seed)

All SFA arms ramp sfa_weight 0.1 -> 10.0 over 500 steps.
Multi-step SFA is computed externally via a trajectory buffer (prefilled for
first 110 steps) with z_past detached and gradient flowing through z_current.

Checkpoint evaluations at step 2000 (monitoring) and step 5000 (final).
Includes Invariance-vs-Discrimination diagnostics:
  - within_traj_var / between_traj_var
  - shuffled semantic probes

Results saved to archive/iter_024/results/
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

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models_dual_stream import NonParametricJEPASpatial

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


# ---------------------------------------------------------------------------
# Normalized temporal variance
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Centroid tracking quality
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Per-sub-feature identity probes
# ---------------------------------------------------------------------------
def compute_sub_feature_probes(z_dyn_arr, colors_arr, radii_arr, d_t, sub_features, num_samples, train_ratio=0.5):
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
# Centroid decoding MSE
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
# Multi-trajectory eval data collection (trajectory-structured)
# ---------------------------------------------------------------------------
def collect_multitraj_eval_data(model, num_samples=200, base_seed=30000, device="cpu"):
    """Collect evaluation data from multiple trajectories, keeping per-trajectory structure."""
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

    # Flattened arrays
    z_dyn_arr = np.concatenate(traj_z_dyn, axis=0)
    z_coord_arr = np.concatenate(traj_z_coord, axis=0)
    pos_arr = np.concatenate(traj_pos, axis=0)
    colors_arr = np.concatenate(traj_colors, axis=0)
    radii_arr = np.concatenate(traj_radii, axis=0)

    return (traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii,
            z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr)


# ---------------------------------------------------------------------------
# Semantic probes core
# ---------------------------------------------------------------------------
def _compute_semantic_probes_core(z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr,
                                   d_t, sub_features, train_ratio=0.5, rng_seed=None):
    N = pos_arr.shape[1]
    d_max = d_t  # for pooled comparison

    if sub_features > 1:
        num_samples = z_dyn_arr.shape[0]
        z_dyn_pooled = z_dyn_arr[:, :d_t * sub_features].reshape(num_samples, d_t, sub_features).mean(axis=2)
    else:
        num_samples = z_dyn_arr.shape[0]
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
# Semantic probes (normal)
# ---------------------------------------------------------------------------
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
# Shuffled semantic probes
# ---------------------------------------------------------------------------
def compute_shuffled_semantic_probes(model, num_samples=200, train_ratio=0.5,
                                      base_seed=30000, device="cpu"):
    """Compute semantic probes after randomly shuffling rows of all features."""
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
        d_t, sub_features, train_ratio=train_ratio
    )


# ---------------------------------------------------------------------------
# Trajectory invariance diagnostics
# ---------------------------------------------------------------------------
def compute_traj_variance_diagnostics(traj_z_dyn, d_t, sub_features):
    """
    Compute within-trajectory and between-trajectory variance of z_dyn.
    """
    d_t_dyn = d_t * sub_features
    within_vars = []
    traj_means = []

    for z_dyn_traj in traj_z_dyn:
        z_active = z_dyn_traj[:, :d_t_dyn]
        if z_active.shape[0] < 2:
            continue
        # Average variance across active dimensions for this trajectory
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
# Evaluation with diagnostics
# ---------------------------------------------------------------------------
def evaluate_run_with_diagnostics(model, arm_config, seed, device, checkpoint_step=None, eval_steps=200):
    """
    Run the full evaluation protocol with invariance-discrimination diagnostics.
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

    # Multi-trajectory eval data (structured + flattened)
    (traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii,
     z_dyn_eval, z_coord_eval, pos_eval, colors_eval, radii_eval) = \
        collect_multitraj_eval_data(model, num_samples=eval_steps,
                                     base_seed=seed + 40000, device=device)

    # Tracking quality
    tracking_quality = compute_tracking_quality(z_coord_eval, pos_eval, d_t)

    # Invariance vs Discrimination diagnostics
    traj_diagnostics = compute_traj_variance_diagnostics(traj_z_dyn, d_t, sub_features)

    # Semantic probes (normal)
    semantic = compute_semantic_probes(model, num_samples=eval_steps,
                                        base_seed=seed + 30000, device=device)

    # Semantic probes (shuffled)
    shuffled = compute_shuffled_semantic_probes(model, num_samples=eval_steps,
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
        # Trajectory invariance diagnostics
        "within_traj_var": traj_diagnostics["within_traj_var"],
        "between_traj_var": traj_diagnostics["between_traj_var"],
        # Semantic probes (normal)
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
        # Semantic probes (shuffled)
        "shuffled_r2_dyn_color": shuffled["r2_dyn_color"],
        "shuffled_delta_r2_color": shuffled["delta_r2_color"],
        "shuffled_r2_dyn_identity": shuffled["r2_dyn_identity"],
        "shuffled_delta_r2_identity": shuffled["delta_r2_identity"],
        # GDASR
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
    contrastive_weight = arm_config.get("contrastive_weight", 25.0)
    temperature = arm_config.get("temperature", 0.1)

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
        sfa_weight=0.0,  # SFA computed externally
        gdasr_log_only=True,
        dyn_readout=dyn_readout,
        sub_features=sub_features,
        dyn_source=dyn_source,
        contrastive_weight=contrastive_weight,
        temperature=temperature,
    )
    model.d_t = d_t
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    env = PhysicsSandbox(N=3, seed=seed)
    obs = env.reset()
    history = collections.deque(maxlen=4)
    history.append(obs)

    replay_buffer = ReplayBuffer(capacity=2000)
    traj_buffer = collections.deque(maxlen=2)

    def _prefill(n):
        while len(replay_buffer) < n:
            obs, info = env.step({"acc": 0.0, "push": False})
            history.append(obs)
            traj_buffer.append(obs)
            if len(history) == 4:
                replay_buffer.push(
                    np.stack(list(history)[:3], axis=0),
                    history[3],
                )

    prefill_steps = max(1, min(total_steps - 2, 100))
    _prefill(prefill_steps)

    logs = []
    results_list = []

    # SFA weight ramp config (applies to all SFA arms)
    sfa_ramp_start = 0.1
    sfa_ramp_end = 10.0
    sfa_ramp_steps = 500
    is_sfa_arm = primary_obj in ("sfa",)
    is_contrastive_arm = primary_obj == "contrastive"

    for step in range(max(1, prefill_steps + 1), total_steps + 1):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        traj_buffer.append(obs)

        if len(history) == 4:
            replay_buffer.push(
                np.stack(list(history)[:3], axis=0),
                history[3],
            )

        # Compute external SFA loss via trajectory buffer (prefilled after step 110)
        sfa_loss = torch.tensor(0.0, device=device)
        if is_sfa_arm and step > 110 and len(traj_buffer) == 2:
            x_curr = torch.from_numpy(traj_buffer[-1]).float().unsqueeze(0).to(device)
            x_past = torch.from_numpy(traj_buffer[-2]).float().unsqueeze(0).to(device)

            z_coord_curr, z_dyn_curr = model.encoder(x_curr)
            with torch.no_grad():
                z_coord_past, z_dyn_past = model.encoder(x_past)

            d_t_dyn = model.d_t * model.sub_features
            sfa_loss = F.mse_loss(z_dyn_curr[:, :d_t_dyn], z_dyn_past[:, :d_t_dyn])

        # JEPA / VICReg / Contrastive from replay buffer
        x_hist_b, x_target_b = replay_buffer.sample(min(32, len(replay_buffer)))
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)

        model.train()
        optimizer.zero_grad()

        # Effective SFA weight (ramp for all SFA arms)
        if is_sfa_arm:
            effective_sfa_weight = sfa_ramp_start + (sfa_ramp_end - sfa_ramp_start) * min(1.0, step / sfa_ramp_steps)
        else:
            effective_sfa_weight = 0.0

        loss_dict, _, _ = model(
            x_hist_t,
            x_target_t,
            sim_weight=sim_weight,
            var_weight=var_weight,
            cov_weight=cov_weight,
            d_t_predict=d_t,
            ccr_mode=ccr_mode,
            ccr_smooth_weight=ccr_smooth_weight,
            ccr_spatial_weight=ccr_spatial_weight,
            sfa_weight=0.0,  # suppress internal SFA; we add external manually
            contrastive_weight=contrastive_weight if is_contrastive_arm else None,
            temperature=temperature if is_contrastive_arm else None,
        )

        total_loss = loss_dict["loss"]
        if is_sfa_arm:
            total_loss = total_loss + effective_sfa_weight * sfa_loss
        total_loss.backward()
        optimizer.step()

        sim_loss_val = loss_dict["sim_loss"].item()
        model.update_recruitment_logic(sim_loss_val, target_dim=d_t, step=step)

        log_entry = {
            "step": step,
            "loss": total_loss.item(),
            "loss_internal": loss_dict["loss"].item(),
            "sim_loss": sim_loss_val,
            "sfa_loss": sfa_loss.item(),
            "var_loss": loss_dict["var_loss"].item(),
            "cov_loss": loss_dict["cov_loss"].item(),
            "ccr_smooth_loss": loss_dict.get("ccr_smooth_loss", torch.tensor(0.0)).item(),
            "ccr_spatial_loss": loss_dict.get("ccr_spatial_loss", torch.tensor(0.0)).item(),
            "effective_sfa_weight": effective_sfa_weight,
        }
        if is_contrastive_arm:
            log_entry["contrastive_loss"] = loss_dict.get("contrastive_loss", torch.tensor(0.0)).item()
        logs.append(log_entry)

        if step % 1000 == 0 or step == total_steps:
            extra = f"  sfa_eff={effective_sfa_weight:.3f}" if is_sfa_arm else ""
            if is_contrastive_arm:
                extra = f"  contrastive={log_entry.get('contrastive_loss', 0):.4f}"
            print(f"  [{name}] seed={seed} step={step:5d}/{total_steps}  "
                  f"loss={log_entry['loss']:.4f}  sim={log_entry['sim_loss']:.4f}  "
                  f"sfa={log_entry['sfa_loss']:.4f}  ccr_s={log_entry['ccr_smooth_loss']:.4f}  "
                  f"ccr_sp={log_entry['ccr_spatial_loss']:.4f}{extra}")

        # Checkpoint evaluation at step 2000 (monitoring only)
        if step == 2000 and not dry_run:
            print(f"  [{name}] seed={seed} -> checkpoint evaluation at step 2000")
            cp_results = evaluate_run_with_diagnostics(model, arm_config, seed, device,
                                                        checkpoint_step=2000, eval_steps=eval_steps)
            cp_results["final_train_loss"] = log_entry["loss"]
            cp_results["final_train_sim_loss"] = log_entry["sim_loss"]
            cp_results["final_train_sfa_loss"] = log_entry["sfa_loss"]
            results_list.append(cp_results)

    # Final evaluation at step 5000
    final_results = evaluate_run_with_diagnostics(model, arm_config, seed, device,
                                                   checkpoint_step=total_steps, eval_steps=eval_steps)
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
        "name": "A (k=20 d_max=8)",
        "primary_objective": "sfa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 20, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "B (k=50 d_max=8)",
        "primary_objective": "sfa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 50, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "C (k=100 d_max=8)",
        "primary_objective": "sfa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 100, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "D (Contrastive d_max=8)",
        "primary_objective": "contrastive",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 1, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "contrastive_weight": 25.0, "temperature": 0.1,
    },
    {
        "name": "E (k=50 d_max=16)",
        "primary_objective": "sfa",
        "sim_weight": 25.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 16, "sub_features": 50, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
    {
        "name": "F (Diagnostic sim=0 k=50 d_max=8)",
        "primary_objective": "sfa",
        "sim_weight": 0.0, "var_weight": 25.0, "cov_weight": 25.0,
        "pos_encoding": "none", "dyn_readout": "centroid_gated",
        "d_t": 3, "d_max": 8, "sub_features": 50, "dyn_source": "spatial",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
    },
]

SEEDS = [42, 123, 456, 789, 999]


def _sanitize_arm_name(name):
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('=', '_')


def _flatten_result(res, runs_dir):
    """Flatten a result dict and save per-run CSV."""
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
    """
    Worker function for ProcessPoolExecutor.

    Each worker:
      1. Sets torch.set_num_threads(1) to prevent thread oversubscription.
      2. Runs run_single(arm, seed, device, dry_run).
      3. Saves per-run CSV, JSON, checkpoint, and log files.
      4. Returns collected result dicts.
    """
    arm, seed, device_str, dry_run, runs_dir, checkpoints_dir = args_tuple
    device = torch.device(device_str)

    # Prevent PyTorch thread oversubscription inside each worker
    torch.set_num_threads(1)

    name = arm["name"]
    print(f"[{name}] seed={seed} -> starting on {device} (dry_run={dry_run})")

    results_list, model, logs = run_single(arm, seed, device, dry_run=dry_run)

    # Return value containers
    result_entries = []  # list of (res_dict, csv_path)

    for res in results_list:
        csv_path = _flatten_result(res, runs_dir)
        result_entries.append((res, csv_path))
        name = res["arm"]
        safe_name = _sanitize_arm_name(name)
        cp_label = f"_cp{res['checkpoint_step']}" if res.get("checkpoint_step") else ""
        run_id = f"{safe_name}_seed{res['seed']}{cp_label}"
        json_path = os.path.join(runs_dir, f"{run_id}.json")
        print(f"  -> Saved {run_id}  ({csv_path}, {json_path})")

    # Save model checkpoint
    safe_name = _sanitize_arm_name(arm["name"])
    ckpt_path = os.path.join(checkpoints_dir, f"{safe_name}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  -> Saved checkpoint {ckpt_path}")

    # Save training logs
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
        description="Phase 0 Multi-Step SFA Runner"
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
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of parallel workers (default: min(cpu_count-1, 8)={default_workers}).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially (no parallelism). Useful for debugging.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: 'cpu' or 'cuda'. Default: auto-detect.",
    )
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

    results_dir = "archive/iter_024/results"
    runs_dir = os.path.join(results_dir, "runs")
    figs_dir = os.path.join(results_dir, "figs")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build flat task list: (arm, seed, device, dry_run, runs_dir, checkpoints_dir)
    # ------------------------------------------------------------------ #
    tasks = []
    task_labels = []
    for arm in ARMS:
        name = arm["name"]
        arm_seeds = seeds if name != "F (Diagnostic sim=0 k=50 d_max=8)" else [42]
        for seed in arm_seeds:
            tasks.append((arm, seed, device_str, dry_run, runs_dir, checkpoints_dir))
            task_labels.append(f"{name} seed={seed}")

    total_tasks = len(tasks)
    print(f"Total tasks to run: {total_tasks}")
    for i, label in enumerate(task_labels, 1):
        print(f"  [{i:2d}] {label}")
    print()

    all_result_entries = []  # list of (res_dict, csv_path) tuples from all workers

    if sequential:
        # Sequential execution (for debugging)
        for i, task in enumerate(tasks):
            arm, seed = task[0], task[1]
            print(f"\n{'='*70}")
            print(f"TASK [{i+1}/{total_tasks}]: {arm['name']} seed={seed}")
            print(f"{'='*70}")
            entries = _run_single_worker(task)
            all_result_entries.extend(entries)
    else:
        # Parallel execution via ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_label = {}
            for i, task in enumerate(tasks):
                future = executor.submit(_run_single_worker, task)
                future_to_label[future] = task_labels[i]

            # Collect results as they complete
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
    # Flatten results
    # ------------------------------------------------------------------ #
    all_results = []
    checkpoint_results = []
    for res, _csv_path in all_result_entries:
        all_results.append(res)
        if res["checkpoint_step"] == 2000:
            checkpoint_results.append(res)

    # ------------------------------------------------------------------ #
    # Compile aggregate results (FINAL step=5000 only)
    # ------------------------------------------------------------------ #
    final_results = [r for r in all_results if r.get("checkpoint_step") != 2000]
    df_all = pd.DataFrame(final_results)

    numeric_cols = [
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
        "within_traj_var", "between_traj_var",
        "shuffled_r2_dyn_color", "shuffled_delta_r2_color",
        "shuffled_r2_dyn_identity", "shuffled_delta_r2_identity",
    ]
    for col in numeric_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    summary_path = os.path.join(results_dir, "summary_phase0_sfa_multistep.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"\nSaved full results to {summary_path}")

    # ------------------------------------------------------------------ #
    # Checkpoint aggregate (step=2000)
    # ------------------------------------------------------------------ #
    if len(checkpoint_results) > 0:
        df_cp = pd.DataFrame(checkpoint_results)
        for col in numeric_cols:
            if col in df_cp.columns:
                df_cp[col] = pd.to_numeric(df_cp[col], errors="coerce")
        cp_summary_path = os.path.join(results_dir, "summary_phase0_sfa_multistep_cp2000.csv")
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
        "within_traj_var", "between_traj_var",
        "shuffled_r2_dyn_color", "shuffled_delta_r2_color",
        "shuffled_r2_dyn_identity", "shuffled_delta_r2_identity",
    ]
    present_agg_cols = [c for c in agg_cols if c in df_all.columns]
    if present_agg_cols and "arm" in df_all.columns and len(df_all) > 0:
        agg = df_all.groupby("arm")[present_agg_cols].agg(["mean", "std"]).reset_index()
        agg_path = os.path.join(results_dir, "aggregated_phase0_sfa_multistep.csv")
        agg.to_csv(agg_path)
        print(f"\nAggregated stats saved to {agg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
