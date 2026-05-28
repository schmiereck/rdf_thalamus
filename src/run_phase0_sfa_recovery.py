#!/usr/bin/env python3
"""
Phase 0 Recovery: Fix SFA Implementation and Re-run with Adjusted Parameters.

Key changes from original Phase 0:
1. SFA bug fix: z_prev_dyn_active detached in SFA loss (models_dual_stream.py)
2. Extended training: 5000 steps (was 3000)
3. CCR covariance mode (was 'none'): ccr_smooth_weight=10.0, ccr_spatial_weight=10.0
4. Four recovery arms × 5 seeds:
   - Arm A1 (SFA w=0.1):        sfa_weight=0.1,  pos_encoding="none"
   - Arm A2 (SFA w=1.0 fixed):   sfa_weight=1.0,  pos_encoding="none"
   - Arm B  (JEPA+CCR baseline): primary_objective="jepa", pos_encoding="none"
   - Arm C  (SFA w=0.1+pos):    sfa_weight=0.1,  pos_encoding="sinusoidal"
5. Evaluation at step 2500 (checkpoint) and step 5000 (final)
6. GDASR log-only mode, d_t=3 frozen

Results saved to archive/iter_020/results/
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
# Helpers: same as original run_phase0_sfa.py
# ---------------------------------------------------------------------------
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


def fit_linear_probe(z, y):
    Z = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    return theta[0], theta[1]


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


def collect_multitraj_eval_data(model, num_samples=200, base_seed=30000, device="cpu"):
    model.eval()
    num_trajectories = max(1, num_samples // 20)
    samples_per_traj = max(1, num_samples // num_trajectories)

    z_dyn_list = []
    z_coord_list = []
    pos_list = []
    colors_list = []

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
                    collected += 1

    z_dyn_arr = np.array(z_dyn_list)
    z_coord_arr = np.array(z_coord_list)
    pos_arr = np.array(pos_list)
    colors_arr = np.array(colors_list)

    return z_dyn_arr, z_coord_arr, pos_arr, colors_arr


def compute_semantic_probes(model, num_samples=200, train_ratio=0.5,
                            base_seed=30000, device="cpu"):
    model.eval()
    z_dyn_arr, z_coord_arr, pos_arr, colors_arr = \
        collect_multitraj_eval_data(model, num_samples=num_samples,
                                     base_seed=base_seed, device=device)

    N = pos_arr.shape[1]
    d_t = model.d_t

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
            r2_per_dim.append({"dyn_color": 0.0, "coord_color": 0.0,
                               "dyn_pos": 0.0, "coord_pos": 0.0})
            continue

        obj = dim_to_obj[d]
        z_d = z_dyn_arr[:, d]
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

        r2_per_dim.append({
            "dyn_color": r2_dyn_color,
            "coord_color": r2_coord_color,
            "dyn_pos": r2_dyn_pos,
            "coord_pos": r2_coord_pos,
        })

    r2_dyn_color_all = np.mean([p["dyn_color"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_color_all = np.mean([p["coord_color"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_dyn_pos_all = np.mean([p["dyn_pos"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_pos_all = np.mean([p["coord_pos"] for p in r2_per_dim]) if r2_per_dim else 0.0
    delta_r2_color = float(r2_dyn_color_all - r2_coord_color_all)

    return {
        "dim_to_obj": dim_to_obj,
        "r2_per_dim": r2_per_dim,
        "r2_dyn_color": float(r2_dyn_color_all),
        "r2_coord_color": float(r2_coord_color_all),
        "r2_dyn_pos": float(r2_dyn_pos_all),
        "r2_coord_pos": float(r2_coord_pos_all),
        "delta_r2_color": delta_r2_color,
    }


# ---------------------------------------------------------------------------
# Evaluation helper (run eval protocol for a model and return results dict)
# ---------------------------------------------------------------------------
def evaluate_run(model, arm_config, seed, device, checkpoint_step=None, eval_steps=200):
    """
    Run the full evaluation protocol on a model.
    Returns a results dict and a label.
    """
    d_t = arm_config.get("d_t", 3)
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
            z_dyn_all.append(z_dyn[:, :d_t].cpu().numpy())
            z_coord_all.append(z_coord[:, :d_t].cpu().numpy())

    z_dyn_arr = np.concatenate(z_dyn_all, axis=0)
    z_coord_arr = np.concatenate(z_coord_all, axis=0)

    has_collapsed, per_dim_std = check_collapse(z_dyn_arr, d_t)
    vh = compute_vicreg_health(z_dyn_arr, d_t)

    # --- Slowness metrics ---
    slowness = compute_slowness_metrics([z_dyn_arr], [z_coord_arr])

    # --- Centroid decoding MSE ---
    test_env = PhysicsSandbox(N=3, seed=seed + 20000)
    test_history = collections.deque(maxlen=4)
    obs = test_env.reset()
    test_history.append(obs)
    centroid_mse = compute_centroid_mse(model, test_env, test_history,
                                         num_samples=eval_steps, device=device)

    # --- Semantic disentanglement probes ---
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
        "r2_dyn_color": semantic["r2_dyn_color"],
        "r2_coord_color": semantic["r2_coord_color"],
        "r2_dyn_pos": semantic["r2_dyn_pos"],
        "r2_coord_pos": semantic["r2_coord_pos"],
        "delta_r2_color": semantic["delta_r2_color"],
        "r2_per_dim": semantic["r2_per_dim"],
        "dim_to_obj": semantic["dim_to_obj"],
        "gdasr_growth_point_count": len(model.gdasr_growth_points) if hasattr(model, "gdasr_growth_points") else 0,
        "gdasr_growth_points": list(model.gdasr_growth_points) if hasattr(model, "gdasr_growth_points") else [],
    }
    return results


# ---------------------------------------------------------------------------
# Single-run training + evaluation (recovery protocol)
# ---------------------------------------------------------------------------
def run_single(arm_config, seed, device, dry_run=False):
    """
    Run one arm × seed experiment with recovery protocol.
    """
    name = arm_config["name"]
    primary_obj = arm_config["primary_objective"]
    sfa_weight = arm_config.get("sfa_weight", 0.1)
    sim_weight = arm_config.get("sim_weight", 25.0)
    var_weight = arm_config.get("var_weight", 25.0)
    cov_weight = arm_config.get("cov_weight", 25.0)
    pos_encoding = arm_config.get("pos_encoding", "none")
    d_t = arm_config.get("d_t", 3)
    ccr_mode = arm_config.get("ccr_mode", "covariance")
    ccr_smooth_weight = arm_config.get("ccr_smooth_weight", 10.0)
    ccr_spatial_weight = arm_config.get("ccr_spatial_weight", 10.0)

    set_seed(seed)

    total_steps = 5 if dry_run else 5000
    eval_steps = min(200, total_steps // 2) if dry_run else 200

    model = NonParametricJEPASpatial(
        d_max=8,
        h=3,
        k=4,
        cooldown=300,
        stabilization_period=100,
        pos_encoding=pos_encoding,
        primary_objective=primary_obj,
        sfa_weight=sfa_weight,
        gdasr_log_only=True,
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
# Arms configuration (recovery)
# ---------------------------------------------------------------------------
ARMS = [
    {
        "name": "Arm A1 (SFA w=0.1)",
        "primary_objective": "sfa",
        "sfa_weight": 0.1,
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "pos_encoding": "none",
        "d_t": 3,
        "ccr_mode": "covariance",
        "ccr_smooth_weight": 10.0,
        "ccr_spatial_weight": 10.0,
        "gdasr_log_only": True,
    },
    {
        "name": "Arm A2 (SFA w=1.0 fixed)",
        "primary_objective": "sfa",
        "sfa_weight": 1.0,
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "pos_encoding": "none",
        "d_t": 3,
        "ccr_mode": "covariance",
        "ccr_smooth_weight": 10.0,
        "ccr_spatial_weight": 10.0,
        "gdasr_log_only": True,
    },
    {
        "name": "Arm B (JEPA+CCR baseline)",
        "primary_objective": "jepa",
        "sfa_weight": 0.1,
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "pos_encoding": "none",
        "d_t": 3,
        "ccr_mode": "covariance",
        "ccr_smooth_weight": 10.0,
        "ccr_spatial_weight": 10.0,
        "gdasr_log_only": True,
    },
    {
        "name": "Arm C (SFA w=0.1+pos)",
        "primary_objective": "sfa",
        "sfa_weight": 0.1,
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "pos_encoding": "sinusoidal",
        "d_t": 3,
        "ccr_mode": "covariance",
        "ccr_smooth_weight": 10.0,
        "ccr_spatial_weight": 10.0,
        "gdasr_log_only": True,
    },
]

SEEDS = [42, 123, 456, 789, 999]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 SFA Recovery Experiment Runner"
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

    results_dir = "archive/iter_020/results"
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
                run_id = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_seed{seed}{cp_label}"
                csv_path = os.path.join(runs_dir, f"{run_id}.csv")
                json_path = os.path.join(runs_dir, f"{run_id}.json")

                flat = {k: v for k, v in res.items() if k not in (
                    "gdasr_growth_points", "centroid_mse_per_object",
                    "centroid_r_per_object", "r2_per_dim", "dim_to_obj",
                )}
                flat["gdasr_growth_point_count"] = len(res.get("gdasr_growth_points", []))
                flat["centroid_mse_obj0"] = res.get("centroid_mse_per_object", [])[0] if len(res.get("centroid_mse_per_object", [])) > 0 else None
                flat["centroid_mse_obj1"] = res.get("centroid_mse_per_object", [])[1] if len(res.get("centroid_mse_per_object", [])) > 1 else None
                flat["centroid_mse_obj2"] = res.get("centroid_mse_per_object", [])[2] if len(res.get("centroid_mse_per_object", [])) > 2 else None
                flat["centroid_r_obj0"] = res.get("centroid_r_per_object", [])[0] if len(res.get("centroid_r_per_object", [])) > 0 else None
                flat["centroid_r_obj1"] = res.get("centroid_r_per_object", [])[1] if len(res.get("centroid_r_per_object", [])) > 1 else None
                flat["centroid_r_obj2"] = res.get("centroid_r_per_object", [])[2] if len(res.get("centroid_r_per_object", [])) > 2 else None
                flat["per_dim_std"] = str(res.get("per_dim_std", []))

                r2pd = res.get("r2_per_dim", [])
                dim_map = res.get("dim_to_obj", {})
                for d_idx, probe_d in enumerate(r2pd):
                    flat[f"r2_dim{d_idx}_dyn_color"] = probe_d.get("dyn_color", None)
                    flat[f"r2_dim{d_idx}_coord_color"] = probe_d.get("coord_color", None)
                    flat[f"r2_dim{d_idx}_dyn_pos"] = probe_d.get("dyn_pos", None)
                    flat[f"r2_dim{d_idx}_coord_pos"] = probe_d.get("coord_pos", None)
                    flat[f"dim{d_idx}_matched_obj"] = dim_map.get(d_idx, None)
                flat["dim_to_obj"] = json.dumps(dim_map, default=str)

                df_row = pd.DataFrame([flat])
                df_row.to_csv(csv_path, index=False)

                with open(json_path, "w") as f:
                    json.dump(res, f, indent=2, default=str)
                print(f"  -> Saved {csv_path}")

            # Save model checkpoint at final step
            ckpt_path = os.path.join(checkpoints_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_seed{seed}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint {ckpt_path}")

            # Save training logs
            logs_path = os.path.join(runs_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_seed{seed}_logs.csv")
            logs_df = pd.DataFrame(logs)
            logs_df.to_csv(logs_path, index=False)
            print(f"  -> Saved training logs {logs_path}")

    # ------------------------------------------------------------------ #
    # Compile aggregate results (FINAL step=5000 only)
    # ------------------------------------------------------------------ #
    final_results = [r for r in all_results if r.get("checkpoint_step") == 5000 or r.get("checkpoint_step") is None]
    df_all = pd.DataFrame(final_results)

    for col in [
        "has_collapsed", "mean_dyn_delta", "mean_coord_delta", "slowness_ratio",
        "centroid_mse_mean", "centroid_r_mean", "mean_abs_corr",
        "final_train_loss", "final_train_sim_loss", "final_train_sfa_loss",
        "gdasr_growth_point_count",
        "r2_dyn_color", "r2_coord_color", "r2_dyn_pos", "r2_coord_pos",
        "delta_r2_color",
    ]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    summary_path = os.path.join(results_dir, "summary_phase0_recovery.csv")
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
        ]:
            if col in df_cp.columns:
                df_cp[col] = pd.to_numeric(df_cp[col], errors="coerce")
        cp_summary_path = os.path.join(results_dir, "summary_phase0_recovery_cp2500.csv")
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
    ]
    agg = df_all.groupby("arm")[agg_cols].agg(["mean", "std"]).reset_index()
    agg_path = os.path.join(results_dir, "aggregated_phase0_recovery.csv")
    agg.to_csv(agg_path)
    print(f"\nAggregated stats saved to {agg_path}")
    print("\n" + "=" * 70)
    print("RECOVERY AGGREGATED RESULTS (mean ± std) @ step 5000")
    print("=" * 70)
    for _, row in agg.iterrows():
        name = row["arm"]
        print(f"\n--- {name} ---")
        for col in agg_cols:
            mean_val = row[(col, "mean")]
            std_val = row[(col, "std")]
            print(f"  {col:30s} = {mean_val:.4f} ± {std_val:.4f}")

    # ------------------------------------------------------------------ #
    # Falsification audit (recovery vs original criteria)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FALSIFICATION AUDIT (Recovery @ step 5000)")
    print("=" * 70)

    arm_a1 = df_all[df_all["arm"] == "Arm A1 (SFA w=0.1)"]
    arm_a2 = df_all[df_all["arm"] == "Arm A2 (SFA w=1.0 fixed)"]
    arm_b = df_all[df_all["arm"] == "Arm B (JEPA+CCR baseline)"]
    arm_c = df_all[df_all["arm"] == "Arm C (SFA w=0.1+pos)"]

    def audit_arm(arm_df, label):
        if len(arm_df) == 0:
            print(f"\n{label}: data missing.")
            return None
        collapsed = int(arm_df["has_collapsed"].sum())
        coll_rate = collapsed / len(arm_df)
        mse_mean = float(arm_df["centroid_mse_mean"].mean())
        delta_color = float(arm_df["delta_r2_color"].mean())
        per_dim_std_mean = float(np.mean(arm_df["per_dim_std"].apply(lambda x: np.mean(eval(x) if isinstance(x, str) else x))))
        slowness = float(arm_df["slowness_ratio"].mean())
        print(f"\n{label}: n={len(arm_df)}, collapsed={collapsed}/{len(arm_df)} ({coll_rate:.0%})")
        print(f"  centroid_mse_mean = {mse_mean:.2f}")
        print(f"  delta_r2_color    = {delta_color:.4f}")
        print(f"  slowness_ratio    = {slowness:.4f}")
        print(f"  avg per_dim_std   = {per_dim_std_mean:.4f}")
        return {
            "collapsed": collapsed,
            "n": len(arm_df),
            "mse_mean": mse_mean,
            "delta_r2_color": delta_color,
            "slowness_ratio": slowness,
            "per_dim_std_mean": per_dim_std_mean,
        }

    audit_a1 = audit_arm(arm_a1, "Arm A1 (SFA w=0.1)")
    audit_a2 = audit_arm(arm_a2, "Arm A2 (SFA w=1.0 fixed)")
    audit_b = audit_arm(arm_b, "Arm B (JEPA+CCR baseline)")
    audit_c = audit_arm(arm_c, "Arm C (SFA w=0.1+pos)")

    # Key comparisons
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    # C1: Collapse rate for A1
    if audit_a1:
        c1_pass = audit_a1["collapsed"] < 2
        print(f"\nC1 (Collapse rate A1 < 2/5):")
        print(f"  A1 collapsed: {audit_a1['collapsed']}/{audit_a1['n']}")
        print(f"  -> {'PASS' if c1_pass else 'FAIL'}")
    else:
        c1_pass = False

    # C2: MSE_A1 <= 1.10 * MSE_B
    if audit_a1 and audit_b:
        threshold = 1.10 * audit_b["mse_mean"]
        c2_pass = audit_a1["mse_mean"] <= threshold
        print(f"\nC2 (Centroid MSE A1 <= 1.10 x B):")
        print(f"  A1 MSE = {audit_a1['mse_mean']:.4f}")
        print(f"  B MSE  = {audit_b['mse_mean']:.4f}")
        print(f"  Threshold (1.10 x B) = {threshold:.4f}")
        print(f"  -> {'PASS' if c2_pass else 'FAIL'}")
    else:
        c2_pass = False

    # C3: delta_R2_color for A1 >= 0.10
    if audit_a1:
        c3_pass = audit_a1["delta_r2_color"] >= 0.10
        print(f"\nC3 (Semantic: A1 delta_R2_color >= 0.10):")
        print(f"  A1 delta_R2_color = {audit_a1['delta_r2_color']:.4f}")
        print(f"  -> {'PASS' if c3_pass else 'FAIL'}")
    else:
        c3_pass = False

    overall = c1_pass and c2_pass and c3_pass
    print(f"\n{'=' * 70}")
    print(f"OVERALL RECOVERY: {'VALIDATED' if overall else 'FALSIFIED'}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------ #
    # Comparison with Original Phase 0
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL Phase 0 vs RECOVERY Phase 0")
    print("=" * 70)
    print("\nOriginal Phase 0 (from pre-run):")
    print("  Arm A (SFA w=1.0): 2/5 collapsed, slowness ratio=805.2, delta_R2_color=-0.087")
    print("  Arm B (JEPA):      4/5 collapsed (per-dim std<0.5), MSE=111.70")
    print("\nRecovery Phase 0 @ step 5000:")
    if audit_a1:
        print(f"  Arm A1 (SFA w=0.1): {audit_a1['collapsed']}/{audit_a1['n']} collapsed, "
              f"slowness_ratio={audit_a1['slowness_ratio']:.2f}, delta_R2_color={audit_a1['delta_r2_color']:.4f}")
    if audit_a2:
        print(f"  Arm A2 (SFA w=1.0): {audit_a2['collapsed']}/{audit_a2['n']} collapsed, "
              f"slowness_ratio={audit_a2['slowness_ratio']:.2f}, delta_R2_color={audit_a2['delta_r2_color']:.4f}")
    if audit_b:
        print(f"  Arm B (JEPA+CCR):   {audit_b['collapsed']}/{audit_b['n']} collapsed, "
              f"slowness_ratio={audit_b['slowness_ratio']:.2f}, MSE={audit_b['mse_mean']:.2f}")
    if audit_c:
        print(f"  Arm C (SFA+pos):    {audit_c['collapsed']}/{audit_c['n']} collapsed, "
              f"slowness_ratio={audit_c['slowness_ratio']:.2f}, delta_R2_color={audit_c['delta_r2_color']:.4f}")

    print("\n" + "-" * 70)
    print("Rationale for adjustments:")
    print("  1. Fixed .detach() bug on z_prev in SFA loss: prevents doubled gradient signal")
    print("  2. Increased steps 3000 -> 5000: more training time for convergence")
    print("  3. Added CCR covariance mode: prevents coordinate collapse / inter-dim correlation")
    print("  4. Reduced sfa_weight 1.0 -> 0.1 (A1, C): weaker slowness regularization")
    print("  5. Arm A2 keeps sfa_weight=1.0: isolate effect of detach fix alone")
    print("-" * 70)

    # Save audit report
    audit_data = {
        "c1_collapse_pass": bool(c1_pass),
        "c1_collapsed_seeds_a1": int(audit_a1["collapsed"]) if audit_a1 else -1,
        "c2_mse_pass": bool(c2_pass),
        "c2_mse_a1_mean": float(audit_a1["mse_mean"]) if audit_a1 else -1.0,
        "c2_mse_b_mean": float(audit_b["mse_mean"]) if audit_b else -1.0,
        "c2_mse_threshold": float(1.10 * audit_b["mse_mean"]) if audit_b else -1.0,
        "c3_semantic_disentanglement_pass": bool(c3_pass),
        "c3_semantic_delta_r2_color_a1": float(audit_a1["delta_r2_color"]) if audit_a1 else -1.0,
        "overall_validated": bool(overall),
        "arms": {
            "a1": audit_a1,
            "a2": audit_a2,
            "b": audit_b,
            "c": audit_c,
        }
    }
    audit_path = os.path.join(results_dir, "audit_phase0_recovery.json")
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)
    print(f"\nAudit report saved to {audit_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
