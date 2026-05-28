#!/usr/bin/env python3
"""
Phase 0: SFA (Slowness) as Primary Representation Objective.

Evaluates three arms × 5 seeds:
  Arm A (SFA+VICReg)     : sfa_weight=1.0, var=25, cov=25, pos_encoding='none'
  Arm B (JEPA+VICReg)    : baseline, sim=25, var=25, cov=25, pos_encoding='none'
  Arm C (SFA+pos_enc)    : sfa_weight=1.0, var=25, cov=25, pos_encoding='sinusoidal'

Pre-registration: src/pre_registration.md (Iteration 020)
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
# Replay buffer (same as existing phases)
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
# Helpers: slowness, collapse check, linear probe, centroid MSE, VICReg health
# ---------------------------------------------------------------------------
def compute_slowness_metrics(z_dyn_log, z_coord_log):
    """
    Both: list of (T, d_t) arrays over N test frames.
    Returns dict with mean temporal deltas and ratio.
    """
    dyn_deltas = []
    coord_deltas = []
    for z_dyn, z_coord in zip(z_dyn_log, z_coord_log):
        # Temporal differences
        dyn_diff = np.diff(z_dyn, axis=0)  # (T-1, d_t)
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
    """
    Returns has_collapsed (bool) and per-dimension std array.
    Uses z_dyn of shape (N, d_max) or (N, d_t).
    """
    z_active = z_dyn[:, :d_t]
    per_dim_std = np.std(z_active, axis=0)  # (d_t,)
    has_collapsed = np.any(per_dim_std < std_threshold)
    return has_collapsed, per_dim_std


def fit_linear_probe(z, y):
    """Simple linear probe via least squares: y_pred = w * z + b."""
    Z = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    return theta[0], theta[1]


def compute_centroid_mse(model, test_env, test_history, num_samples=200, device="cpu"):
    """
    Evaluates centroid-decoding MSE: linear probe on soft-argmax centroids
    vs. true object positions for all 3 objects.
    """
    model.eval()
    centroids_list = []
    pos_list = []

    # Collect centroids and positions from passive rollout
    obs = test_env.reset()
    test_history.clear()
    test_history.append(obs)

    for _ in range(4):  # warm up history for H=3
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
            centroids_list.append(z_coord[:, :model.d_t].cpu().numpy())  # (1, d_t)
            # True positions (only first d_t objects, pad if fewer)
            true_pos = np.array(info["positions"][:model.d_t])
            if len(true_pos) < model.d_t:
                true_pos = np.pad(true_pos, (0, model.d_t - len(true_pos)), constant_values=np.nan)
            pos_list.append(true_pos)
            collected += 1

    centroids_arr = np.concatenate(centroids_list, axis=0)  # (N, d_t)
    pos_arr = np.array(pos_list)  # (N, d_t)

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
    """
    Returns per-dimension std and mean absolute correlation between active dimensions.
    """
    z_active = z_dyn[:, :d_t]
    per_dim_std = np.std(z_active, axis=0)
    if d_t > 1:
        corr = np.corrcoef(z_active.T)
        # upper triangle off-diagonal
        triu = np.triu_indices(d_t, k=1)
        mean_abs_corr = float(np.mean(np.abs(corr[triu])))
    else:
        mean_abs_corr = 0.0
    return {
        "per_dim_std": per_dim_std.tolist(),
        "mean_abs_corr": mean_abs_corr,
    }


def compute_semantic_probes(model, test_env, test_history, num_samples=200,
                             train_ratio=0.5, device="cpu"):
    """
    Semantic disentanglement probes.

    Collects z_dyn, z_coord along with ground-truth positions and colors over
    num_samples frames.  Matches each latent dimension to the nearest object
    (by centroid position).  Fits linear probes on the first half and evaluates
    R² on the second half for two tasks:

      * color  regression  (3-D RGB target → 3 scalar R² values, then mean)
      * position regression  (1-D scalar target → 1 R² value)

    Probes are fit separately for z_dyn and z_coord, giving four R² scores:
      R2_dyn_color,  R2_coord_color,  R2_dyn_pos,  R2_coord_pos

    The key disentanglement metric is:
      delta_R2_color = R2_dyn_color - R2_coord_color
    If SFA separates identity from position, z_dyn should predict color well
    and z_coord should predict color poorly, yielding a positive delta.
    """
    model.eval()

    # Collect data
    obs = test_env.reset()
    test_history.clear()
    test_history.append(obs)

    # Warm up
    for _ in range(4):
        obs, info = test_env.step({"acc": 0.0, "push": False})
        test_history.append(obs)

    z_dyn_list = []
    z_coord_list = []
    pos_list = []  # positions for all N objects
    colors_list = []  # colors for all N objects

    with torch.no_grad():
        for _ in range(num_samples):
            obs, info = test_env.step({"acc": 0.0, "push": False})
            test_history.append(obs)
            if len(test_history) == 4:
                x_t = torch.from_numpy(test_history[3]).float().unsqueeze(0).to(device)
                z_c, z_d = model.encoder(x_t)
                z_dyn_list.append(z_d[0].cpu().numpy())       # (d_max,)
                z_coord_list.append(z_c[0].cpu().numpy())      # (d_max,)
                pos_list.append(info["positions"])              # (N,)
                colors_list.append(info["colors"])              # (N, 3)

    z_dyn_arr = np.array(z_dyn_list)   # (num_samples, d_max)
    z_coord_arr = np.array(z_coord_list)
    pos_arr = np.array(pos_list)       # (num_samples, N)
    colors_arr = np.array(colors_list) # (num_samples, N, 3)

    N = pos_arr.shape[1]  # number of objects
    d_t = model.d_t

    # Match each latent dimension d in [0, d_t) to the nearest object
    # by average distance between centroid z_coord and true position.
    dim_to_obj = {}  # dim_idx -> obj_idx
    used_objs = set()
    for d in range(d_t):
        best_obj = None
        best_dist = np.inf
        for o in range(N):
            if o in used_objs:
                continue
            # average absolute distance between centroid and position
            dist = np.mean(np.abs(z_coord_arr[:, d] - pos_arr[:, o]))
            if dist < best_dist:
                best_dist = dist
                best_obj = o
        if best_obj is not None:
            dim_to_obj[d] = best_obj
            used_objs.add(best_obj)

    # Split into train and test
    n_train = int(num_samples * train_ratio)

    def fit_probe_r2(z_feature, y_target):
        """Fit linear probe: y = w*z + b. Return R² on test portion."""
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

        # Position probe
        r2_dyn_pos = fit_probe_r2(z_d, pos_arr[:, obj])
        r2_coord_pos = fit_probe_r2(z_c, pos_arr[:, obj])

        # Color probe (3 channels, average R²)
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

    # Averages across matched dimensions
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
# Single-run training + evaluation
# ---------------------------------------------------------------------------
def run_single(arm_config, seed, device, dry_run=False):
    """
    Run one arm × seed experiment.

    Parameters
    ----------
    arm_config : dict
        Keys: 'name', 'primary_objective', 'sfa_weight', 'sim_weight',
              'var_weight', 'cov_weight', 'pos_encoding', 'd_t'
    seed : int
    device : torch.device
    dry_run : bool  — if True, run only 5 training steps per seed for quick verification.

    Returns
    -------
    results : dict  (metrics for this run)
    """
    name = arm_config["name"]
    primary_obj = arm_config["primary_objective"]
    sfa_weight = arm_config.get("sfa_weight", 1.0)
    sim_weight = arm_config.get("sim_weight", 25.0)
    var_weight = arm_config.get("var_weight", 25.0)
    cov_weight = arm_config.get("cov_weight", 25.0)
    pos_encoding = arm_config.get("pos_encoding", "none")
    d_t = arm_config.get("d_t", 3)

    set_seed(seed)

    # Determine total training steps (5 for dry run, 3000 otherwise)
    total_steps = 5 if dry_run else 3000
    eval_steps = min(200, total_steps // 2) if dry_run else 200

    # Create model
    model = NonParametricJEPASpatial(
        d_max=8,
        h=3,
        k=4,
        cooldown=300,
        stabilization_period=100,
        pos_encoding=pos_encoding,
        primary_objective=primary_obj,
        sfa_weight=sfa_weight,
        gdasr_log_only=True,  # log-only mode for Phase 0
    )
    model.d_t = d_t  # freeze at 3
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Environment
    env = PhysicsSandbox(N=3, seed=seed)
    obs = env.reset()
    history = collections.deque(maxlen=4)
    history.append(obs)

    replay_buffer = ReplayBuffer(capacity=2000)

    # Prefill buffer
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

    # Training loop
    logs = []
    gdasr_growth_points = []

    for step in range(1, total_steps + 1):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        if len(history) == 4:
            replay_buffer.push(
                np.stack(list(history)[:3], axis=0),
                history[3],
            )

        # Sample batch
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
            d_t_predict=d_t,  # predictor always sees d_t dimensions
        )
        loss_dict["loss"].backward()
        optimizer.step()

        # GDASR log-only update
        sim_loss_val = loss_dict["sim_loss"].item()
        model.update_recruitment_logic(sim_loss_val, target_dim=d_t, step=step)

        # Log
        log_entry = {
            "step": step,
            "loss": loss_dict["loss"].item(),
            "sim_loss": sim_loss_val,
            "sfa_loss": loss_dict.get("sfa_loss", torch.tensor(0.0)).item(),
            "var_loss": loss_dict["var_loss"].item(),
            "cov_loss": loss_dict["cov_loss"].item(),
        }
        logs.append(log_entry)

        if step % 1000 == 0 or step == total_steps:
            print(f"  [{name}] seed={seed} step={step:5d}/{total_steps}  "
                  f"loss={log_entry['loss']:.4f}  sim={log_entry['sim_loss']:.4f}  "
                  f"sfa={log_entry['sfa_loss']:.4f}")

    # Collect GDASR growth points
    gdasr_growth_points = list(model.gdasr_growth_points)

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    model.eval()

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

    z_dyn_arr = np.concatenate(z_dyn_all, axis=0)   # (eval_steps, d_t)
    z_coord_arr = np.concatenate(z_coord_all, axis=0)

    has_collapsed, per_dim_std = check_collapse(z_dyn_arr, d_t)
    vh = compute_vicreg_health(z_dyn_arr, d_t)

    # --- Slowness metrics ---
    # Use the collected z_dyn and z_coord arrays for temporal deltas
    slowness = compute_slowness_metrics(
        [z_dyn_arr], [z_coord_arr]
    )

    # --- Centroid decoding MSE on a separate fresh test set ---
    test_env = PhysicsSandbox(N=3, seed=seed + 20000)
    test_history = collections.deque(maxlen=4)
    obs = test_env.reset()
    test_history.append(obs)
    centroid_mse = compute_centroid_mse(model, test_env, test_history,
                                         num_samples=eval_steps, device=device)

    # --- Semantic disentanglement probes ---
    semantic = compute_semantic_probes(model, test_env, test_history,
                                        num_samples=eval_steps, device=device)

    # --- Summary results ---
    results = {
        "arm": name,
        "seed": seed,

        # Collapse
        "has_collapsed": int(has_collapsed),
        "per_dim_std": per_dim_std.tolist(),
        "mean_abs_corr": vh["mean_abs_corr"],

        # Slowness
        "mean_dyn_delta": slowness["mean_dyn_delta"],
        "mean_coord_delta": slowness["mean_coord_delta"],
        "slowness_ratio": slowness["ratio"],

        # Centroid MSE
        "centroid_mse_mean": centroid_mse["mse_mean"],
        "centroid_mse_per_object": centroid_mse["mse_per_object"],
        "centroid_r_mean": centroid_mse["r_mean"],
        "centroid_r_per_object": centroid_mse["r_per_object"],

        # VICReg health
        "vicreg_per_dim_std": per_dim_std.tolist(),
        "vicreg_mean_abs_corr": vh["mean_abs_corr"],

        # Semantic disentanglement probes
        "r2_dyn_color": semantic["r2_dyn_color"],
        "r2_coord_color": semantic["r2_coord_color"],
        "r2_dyn_pos": semantic["r2_dyn_pos"],
        "r2_coord_pos": semantic["r2_coord_pos"],
        "delta_r2_color": semantic["delta_r2_color"],
        "r2_per_dim": semantic["r2_per_dim"],
        "dim_to_obj": semantic["dim_to_obj"],

        # GDASR
        "gdasr_growth_point_count": len(gdasr_growth_points),
        "gdasr_growth_points": gdasr_growth_points,

        # Final training loss
        "final_train_loss": logs[-1]["loss"],
        "final_train_sim_loss": logs[-1]["sim_loss"],
        "final_train_sfa_loss": logs[-1]["sfa_loss"],
    }
    return results


# ---------------------------------------------------------------------------
# Arms configuration
# ---------------------------------------------------------------------------
ARMS = [
    {
        "name": "Arm A (SFA+VICReg)",
        "primary_objective": "sfa",
        "sfa_weight": 1.0,
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "pos_encoding": "none",
        "d_t": 3,
    },
    {
        "name": "Arm B (JEPA+VICReg)",
        "primary_objective": "jepa",
        "sfa_weight": 1.0,  # unused in jepa mode
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "pos_encoding": "none",
        "d_t": 3,
    },
    {
        "name": "Arm C (SFA+pos_enc)",
        "primary_objective": "sfa",
        "sfa_weight": 1.0,
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "pos_encoding": "sinusoidal",
        "d_t": 3,
    },
]

SEEDS = [42, 123, 456, 789, 999]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 SFA Experiment Runner"
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

    suffix = "_dryrun" if dry_run else ""
    results_dir = "archive/iter_020"
    runs_dir = os.path.join(results_dir, "runs")
    figs_dir = os.path.join(results_dir, "figs")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    all_results = []

    for arm in ARMS:
        name = arm["name"]
        print(f"{'='*70}")
        print(f"ARM: {name}")
        print(f"{'='*70}")
        for seed in seeds:
            print(f"\n--- {name}, seed={seed} ---")
            results = run_single(arm, seed, device, dry_run=dry_run)
            all_results.append(results)

            # Save per-run CSV with detailed logs
            run_id = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_seed{seed}{suffix}"
            csv_path = os.path.join(runs_dir, f"{run_id}.csv")
            json_path = os.path.join(runs_dir, f"{run_id}.json")

            # Flatten per-object / per-dimension lists for CSV
            flat = {k: v for k, v in results.items() if k not in (
                "gdasr_growth_points", "centroid_mse_per_object",
                "centroid_r_per_object", "r2_per_dim", "dim_to_obj",
            )}
            flat["gdasr_growth_point_count"] = len(results.get("gdasr_growth_points", []))
            flat["centroid_mse_obj0"] = results.get("centroid_mse_per_object", [])[0] if len(results.get("centroid_mse_per_object", [])) > 0 else None
            flat["centroid_mse_obj1"] = results.get("centroid_mse_per_object", [])[1] if len(results.get("centroid_mse_per_object", [])) > 1 else None
            flat["centroid_mse_obj2"] = results.get("centroid_mse_per_object", [])[2] if len(results.get("centroid_mse_per_object", [])) > 2 else None
            flat["centroid_r_obj0"] = results.get("centroid_r_per_object", [])[0] if len(results.get("centroid_r_per_object", [])) > 0 else None
            flat["centroid_r_obj1"] = results.get("centroid_r_per_object", [])[1] if len(results.get("centroid_r_per_object", [])) > 1 else None
            flat["centroid_r_obj2"] = results.get("centroid_r_per_object", [])[2] if len(results.get("centroid_r_per_object", [])) > 2 else None
            flat["per_dim_std"] = str(results.get("per_dim_std", []))

            # Semantic probe per-dimension details
            r2pd = results.get("r2_per_dim", [])
            dim_map = results.get("dim_to_obj", {})
            for d_idx, probe_d in enumerate(r2pd):
                flat[f"r2_dim{d_idx}_dyn_color"] = probe_d.get("dyn_color", None)
                flat[f"r2_dim{d_idx}_coord_color"] = probe_d.get("coord_color", None)
                flat[f"r2_dim{d_idx}_dyn_pos"] = probe_d.get("dyn_pos", None)
                flat[f"r2_dim{d_idx}_coord_pos"] = probe_d.get("coord_pos", None)
                flat[f"dim{d_idx}_matched_obj"] = dim_map.get(d_idx, None)
            flat["dim_to_obj"] = json.dumps(dim_map, default=str)

            df_row = pd.DataFrame([flat])
            df_row.to_csv(csv_path, index=False)

            # Save full JSON (includes growth points)
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"  -> Saved {csv_path}")

    # ------------------------------------------------------------------ #
    # Compile aggregate results
    # ------------------------------------------------------------------ #
    df_all = pd.DataFrame(all_results)

    # Ensure numeric columns
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

    summary_path = os.path.join(results_dir, f"summary_phase0{suffix}.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"\nSaved full results to {summary_path}")

    # ------------------------------------------------------------------ #
    # Aggregated per-arm stats
    # ------------------------------------------------------------------ #
    agg_cols = [
        "has_collapsed", "mean_dyn_delta", "mean_coord_delta", "slowness_ratio",
        "centroid_mse_mean", "centroid_r_mean", "mean_abs_corr",
        "final_train_loss", "final_train_sim_loss", "gdasr_growth_point_count",
    ]
    agg = df_all.groupby("arm")[agg_cols].agg(["mean", "std"]).reset_index()
    agg_path = os.path.join(results_dir, f"aggregated_phase0{suffix}.csv")
    agg.to_csv(agg_path)
    print(f"\nAggregated stats saved to {agg_path}")
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (mean ± std)")
    print("=" * 70)
    for _, row in agg.iterrows():
        name = row["arm"]
        print(f"\n--- {name} ---")
        for col in agg_cols:
            mean_val = row[(col, "mean")]
            std_val = row[(col, "std")]
            print(f"  {col:30s} = {mean_val:.4f} ± {std_val:.4f}")

    # ------------------------------------------------------------------ #
    # Falsification audit
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("FALSIFICATION AUDIT (per pre-registration)")
    print("=" * 70)

    # Separate arms
    arm_a = df_all[df_all["arm"] == "Arm A (SFA+VICReg)"]
    arm_b = df_all[df_all["arm"] == "Arm B (JEPA+VICReg)"]
    arm_c = df_all[df_all["arm"] == "Arm C (SFA+pos_enc)"]

    # --- C1: Collapse ---
    if len(arm_a) > 0:
        collapsed_a = int(arm_a["has_collapsed"].sum())
        print(f"\nC1 (Collapse): Arm A collapsed seeds: {collapsed_a} / {len(arm_a)}")
        c1_pass = collapsed_a < 2  # fail if >= 2 seeds collapsed
        print(f"  -> {'PASS' if c1_pass else 'FAIL'} (threshold: < 2 collapsed seeds)")
    else:
        c1_pass = False
        print("\nC1: Arm A data missing.")

    # --- C2: Centroid MSE ---
    if len(arm_a) > 0 and len(arm_b) > 0:
        mse_a_mean = float(arm_a["centroid_mse_mean"].mean())
        mse_b_mean = float(arm_b["centroid_mse_mean"].mean())
        print(f"\nC2 (Centroid MSE):")
        print(f"  Arm A (SFA)  MSE = {mse_a_mean:.4f}")
        print(f"  Arm B (JEPA) MSE = {mse_b_mean:.4f}")
        if np.isnan(mse_a_mean) or np.isnan(mse_b_mean):
            print("  [WARNING] centroid_mse_mean contains NaN — possibly too few eval samples.")
            threshold = None
            c2_pass = False
            print(f"  -> FAIL (NaN MSE values)")
        else:
            threshold = 1.10 * mse_b_mean
            print(f"  Threshold (1.10 x JEPA) = {threshold:.4f}")
            c2_pass = mse_a_mean <= threshold
            print(f"  -> {'PASS' if c2_pass else 'FAIL'} (SFA MSE {'' if c2_pass else 'NOT '}<= 1.10 x JEPA MSE)")

            # Welch's t-test
            mse_a_vals = arm_a["centroid_mse_mean"].values.astype(float)
            mse_b_vals = arm_b["centroid_mse_mean"].values.astype(float)
            if len(mse_a_vals) > 1 and len(mse_b_vals) > 1:
                t_stat, p_val = scipy_stats.ttest_ind(mse_a_vals, mse_b_vals, equal_var=False)
                n1, n2 = len(mse_a_vals), len(mse_b_vals)
                dof = (mse_a_vals.var() / n1 + mse_b_vals.var() / n2) ** 2 / \
                      ((mse_a_vals.var() / n1) ** 2 / (n1 - 1) + (mse_b_vals.var() / n2) ** 2 / (n2 - 1))
                cohens_d = (mse_a_mean - mse_b_mean) / np.sqrt(
                    ((n1 - 1) * mse_a_vals.var() + (n2 - 1) * mse_b_vals.var()) / (n1 + n2 - 2)
                )
                print(f"  Welch's t({dof:.1f}) = {t_stat:.4f}, p = {p_val:.4f}")
                print(f"  Cohen's d = {cohens_d:.4f}")
    else:
        c2_pass = False
        mse_a_mean = float("nan")
        mse_b_mean = float("nan")
        threshold = None
        print("\nC2: Arm A or Arm B data missing.")

    # --- C3 (Slowness separation — sanity check) ---
    if len(arm_a) > 0:
        ratios_a = arm_a["slowness_ratio"].values.astype(float)
        mean_ratio_a = float(np.mean(ratios_a))
        print(f"\nC3_sanity (Slowness separation — sanity check):")
        print(f"  Arm A mean slowness ratio = {mean_ratio_a:.4f}")
        print(f"  Threshold: ratio < 0.6")
        c3_sanity_pass = mean_ratio_a < 0.6
        print(f"  -> {'PASS' if c3_sanity_pass else 'FAIL'} (ratio {'<' if c3_sanity_pass else '>='} 0.6)")

        # One-sample t-test against 0.6
        if len(ratios_a) > 1:
            t_stat3, p_val3 = scipy_stats.ttest_1samp(ratios_a, 0.6, alternative="less")
            print(f"  One-sample t({len(ratios_a)-1}) vs 0.6: t = {t_stat3:.4f}, p = {p_val3:.4f} (one-sided less)")
    else:
        c3_sanity_pass = False
        mean_ratio_a = -1.0
        print("\nC3_sanity: Arm A data missing.")

    # --- C3 (Semantic disentanglement — delta_R2_color >= 0.10) ---
    if len(arm_a) > 0:
        deltas_a = arm_a["delta_r2_color"].values.astype(float)
        mean_delta_a = float(np.mean(deltas_a))
        print(f"\nC3_semantic (Semantic disentanglement — delta_R2_color):")
        print(f"  Arm A mean delta_R2_color = {mean_delta_a:.4f}")
        print(f"  Threshold: delta_R2_color >= 0.10")
        c3_semantic_pass = mean_delta_a >= 0.10
        print(f"  -> {'PASS' if c3_semantic_pass else 'FAIL'} (delta {'>= ' if c3_semantic_pass else '< '} 0.10)")

        # One-sample t-test against 0.10
        if len(deltas_a) > 1:
            t_stat_sem, p_val_sem = scipy_stats.ttest_1samp(deltas_a, 0.10, alternative="greater")
            print(f"  One-sample t({len(deltas_a)-1}) vs 0.10: t = {t_stat_sem:.4f}, p = {p_val_sem:.4f} (one-sided greater)")
    else:
        c3_semantic_pass = False
        mean_delta_a = -1.0
        print("\nC3_semantic: Arm A data missing.")

    overall = c1_pass and c2_pass and c3_sanity_pass and c3_semantic_pass
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'VALIDATED' if overall else 'FALSIFIED'}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------ #
    # Save audit report
    # ------------------------------------------------------------------ #
    audit_data = {
        "c1_collapse_pass": bool(c1_pass),
        "c1_collapsed_seeds_a": int(collapsed_a) if len(arm_a) > 0 else -1,
        "c2_mse_pass": bool(c2_pass),
        "c2_mse_sfa_mean": float(mse_a_mean) if len(arm_a) > 0 else -1.0,
        "c2_mse_jepa_mean": float(mse_b_mean) if len(arm_b) > 0 else -1.0,
        "c2_mse_threshold": float(threshold) if len(arm_a) > 0 and len(arm_b) > 0 else -1.0,
        "c3_sanity_slowness_pass": bool(c3_sanity_pass),
        "c3_sanity_slowness_ratio_mean": float(mean_ratio_a),
        "c3_semantic_disentanglement_pass": bool(c3_semantic_pass),
        "c3_semantic_delta_r2_color_mean": float(mean_delta_a),
        "overall_validated": bool(overall),
    }
    audit_path = os.path.join(results_dir, f"audit_phase0{suffix}.json")
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nAudit report saved to {audit_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()