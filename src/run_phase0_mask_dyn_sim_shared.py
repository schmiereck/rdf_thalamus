#!/usr/bin/env python3
"""
Iter_028 Shared-Backbone mask_dyn_sim Probe.

4 arms x 10 seeds to test whether mask_dyn_sim alone (on the shared backbone)
prevents z_dyn collapse.

Arms:
  D0 — Baseline replication (shared backbone, mask_dyn_sim=False, weights 25/25/1)
  C1 — Primary arm (shared backbone, mask_dyn_sim=True, weights 25/25/1)
  C2 — Seed robustness (shared backbone, mask_dyn_sim=True, weights 25/25/1, fresh seeds)
  C3 — Weight robustness (shared backbone, mask_dyn_sim=True, weights 27.5/27.5/1.1)

mask_dyn_sim is implemented via loss adjustment AFTER forward(), exactly like
models_separate_dyn.py does:
    loss_dict["loss"] = loss_dict["loss"] - sim_weight * loss_dict["sim_loss_dyn"]
This is identical to NonParametricJEPASpatialSeparateDyn's approach.

No modification to models_dual_stream.py, models_separate_dyn.py, or environment.py.
"""

import os, sys, json, random, argparse, collections, warnings, concurrent.futures
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment import PhysicsSandbox
from src.models_dual_stream import NonParametricJEPASpatial


# ---------------------------------------------------------------------------
# Replay Buffer
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
        xh, xt, p, c, r = zip(*batch)
        return np.stack(xh), np.stack(xt), np.stack(p), np.stack(c), np.stack(r)

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Seeding
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
# Collapse check & VICReg health
# ---------------------------------------------------------------------------
def check_collapse(z_dyn, d_t, std_threshold=0.5):
    z_active = z_dyn[:, :d_t]
    per_dim_std = np.std(z_active, axis=0)
    return np.any(per_dim_std < std_threshold), per_dim_std


def compute_vicreg_health(z_dyn, d_t):
    z_active = z_dyn[:, :d_t]
    per_dim_std = np.std(z_active, axis=0)
    if d_t > 1:
        corr = np.corrcoef(z_active.T)
        triu = np.triu_indices(d_t, k=1)
        mean_abs_corr = float(np.mean(np.abs(corr[triu])))
    else:
        mean_abs_corr = 0.0
    return {"per_dim_std": per_dim_std.tolist(), "mean_abs_corr": mean_abs_corr}


# ---------------------------------------------------------------------------
# Linear probe helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Centroid MSE
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
# Multi-trajectory data collection
# ---------------------------------------------------------------------------
def collect_multitraj_eval_data(model, num_samples=200, base_seed=30000, device="cpu"):
    model.eval()
    num_trajectories = max(1, num_samples // 20)
    samples_per_traj = max(1, num_samples // num_trajectories)
    traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii = [], [], [], [], []
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
            z_dyn_t, z_coord_t, pos_t, colors_t, radii_t = [], [], [], [], []
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
    return (
        traj_z_dyn,
        traj_z_coord,
        traj_pos,
        traj_colors,
        traj_radii,
        np.concatenate(traj_z_dyn, axis=0),
        np.concatenate(traj_z_coord, axis=0),
        np.concatenate(traj_pos, axis=0),
        np.concatenate(traj_colors, axis=0),
        np.concatenate(traj_radii, axis=0),
    )


# ---------------------------------------------------------------------------
# Semantic probes core
# ---------------------------------------------------------------------------
def _compute_semantic_probes_core(z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr, d_t, sub_features, train_ratio=0.5):
    N = pos_arr.shape[1]
    num_samples = z_dyn_arr.shape[0]
    if sub_features > 1:
        z_dyn_pooled = z_dyn_arr[:, :d_t * sub_features].reshape(num_samples, d_t, sub_features).mean(axis=2)
    else:
        z_dyn_pooled = z_dyn_arr[:, :d_t]

    cost = np.zeros((d_t, N))
    for d in range(d_t):
        for o in range(N):
            cost[d, o] = np.mean(np.abs(z_coord_arr[:, d] - pos_arr[:, o]))
    row_ind, col_ind = linear_sum_assignment(cost)
    dim_to_obj = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    n_train = int(num_samples * train_ratio)

    def fit_probe_r2(z_feature, y_target):
        z_train, y_train = z_feature[:n_train], y_target[:n_train]
        z_test, y_test = z_feature[n_train:], y_target[n_train:]
        if z_train.size < 5 or y_train.size < 5:
            return 0.0
        w, b = fit_linear_probe(z_train, y_train)
        y_pred = z_test * w + b
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_train)) ** 2)
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

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
        r2_dyn_ch = [fit_probe_r2(z_d, colors_arr[:, obj, ch]) for ch in range(3)]
        r2_coord_ch = [fit_probe_r2(z_c, colors_arr[:, obj, ch]) for ch in range(3)]
        max_radius = 20.0
        identity_vec = np.zeros((num_samples, 4))
        identity_vec[:, :3] = colors_arr[:, obj, :]
        identity_vec[:, 3] = radii_arr[:, obj] / max_radius
        r2_per_dim.append({
            "dyn_color": float(np.mean(r2_dyn_ch)),
            "coord_color": float(np.mean(r2_coord_ch)),
            "dyn_pos": r2_dyn_pos,
            "coord_pos": r2_coord_pos,
            "dyn_identity": fit_multivariate_probe_r2(z_d, identity_vec),
            "coord_identity": fit_multivariate_probe_r2(z_c, identity_vec),
        })

    r2_dyn_color_all = np.mean([p["dyn_color"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_color_all = np.mean([p["coord_color"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_dyn_pos_all = np.mean([p["dyn_pos"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_pos_all = np.mean([p["coord_pos"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_dyn_identity_all = np.mean([p["dyn_identity"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_identity_all = np.mean([p["coord_identity"] for p in r2_per_dim]) if r2_per_dim else 0.0

    return {
        "dim_to_obj": dim_to_obj,
        "r2_per_dim": r2_per_dim,
        "r2_dyn_color": float(r2_dyn_color_all),
        "r2_coord_color": float(r2_coord_color_all),
        "delta_r2_color": float(r2_dyn_color_all - r2_coord_color_all),
        "r2_dyn_pos": float(r2_dyn_pos_all),
        "r2_coord_pos": float(r2_coord_pos_all),
        "r2_dyn_identity": float(r2_dyn_identity_all),
        "r2_coord_identity": float(r2_coord_identity_all),
        "delta_r2_identity": float(r2_dyn_identity_all - r2_coord_identity_all),
    }


def compute_semantic_probes(model, num_samples=200, train_ratio=0.5, base_seed=30000, device="cpu"):
    model.eval()
    (_, _, _, _, _, z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr) = collect_multitraj_eval_data(
        model, num_samples=num_samples, base_seed=base_seed, device=device
    )
    return _compute_semantic_probes_core(
        z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr,
        model.d_t, model.sub_features, train_ratio=train_ratio
    )


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate_run(model, arm_config, seed, device, eval_steps=200):
    d_t = arm_config.get("d_t", 3)
    sub_features = arm_config.get("sub_features", 1)
    name = arm_config["name"]

    eval_env = PhysicsSandbox(N=3, seed=seed + 10000)
    eval_history = collections.deque(maxlen=4)
    obs = eval_env.reset()
    eval_history.append(obs)
    for _ in range(4):
        obs, _ = eval_env.step({"acc": 0.0, "push": False})
        eval_history.append(obs)

    z_dyn_all, z_coord_all = [], []
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

    test_env = PhysicsSandbox(N=3, seed=seed + 20000)
    test_history = collections.deque(maxlen=4)
    centroid_mse = compute_centroid_mse(model, test_env, test_history, num_samples=eval_steps, device=device)

    semantic = compute_semantic_probes(model, num_samples=eval_steps, base_seed=seed + 30000, device=device)

    return {
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


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Single run (core logic)
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
    lr = arm_config.get("lr", 3e-4)
    batch_size = arm_config.get("batch_size", 64)
    replay_buffer_capacity = arm_config.get("replay_buffer_capacity", 4000)
    mask_dyn_sim = arm_config.get("mask_dyn_sim", False)
    gdasr_log_only = arm_config.get("gdasr_log_only", True)

    set_seed(seed)
    total_steps = 5 if dry_run else 8000
    eval_steps = min(200, total_steps // 2) if dry_run else 200

    # Shared backbone only — always NonParametricJEPASpatial
    model = NonParametricJEPASpatial(
        d_max=d_max, h=3, k=4, cooldown=300, stabilization_period=100,
        pos_encoding=pos_encoding, primary_objective="jepa",
        gdasr_log_only=gdasr_log_only,
        dyn_readout=dyn_readout, sub_features=1, dyn_source="spatial"
    )
    model.d_t = d_t
    model = model.to(device)
    param_count = count_parameters(model)
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
                    info["positions"], info["colors"], info["radii"],
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
                info["positions"], info["colors"], info["radii"],
            )

        x_hist_b, x_target_b, pos_b, colors_b, radii_b = replay_buffer.sample(
            min(batch_size, len(replay_buffer))
        )
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)

        model.train()
        optimizer.zero_grad()

        # Forward pass
        loss_dict, _, (z_target_coord, z_target_dyn) = model(
            x_hist_t, x_target_t,
            sim_weight=sim_weight,
            var_weight=var_weight,
            cov_weight=cov_weight,
            d_t_predict=d_t,
            ccr_mode=ccr_mode,
            ccr_smooth_weight=ccr_smooth_weight,
            ccr_spatial_weight=ccr_spatial_weight,
        )

        # mask_dyn_sim adjustment BEFORE backward()
        # Subtract sim_loss_dyn from the gradient path (identical to separate_dyn approach)
        if mask_dyn_sim:
            loss_dict["loss"] = loss_dict["loss"] - sim_weight * loss_dict["sim_loss_dyn"]

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
        if step % 500 == 0:
            with torch.no_grad():
                per_dim_std_train = z_target_dyn[:, :d_t].std(dim=0).cpu().numpy()
            log_entry["per_dim_std"] = per_dim_std_train.tolist()
        logs.append(log_entry)

        if step % 1000 == 0 or step == total_steps:
            print(
                f"  [{name}] seed={seed} step={step:5d}/{total_steps}  "
                f"loss={log_entry['loss']:.4f}  sim={log_entry['sim_loss']:.4f}  "
                f"var={log_entry['var_loss']:.4f}  cov={log_entry['cov_loss']:.4f}"
            )

    eval_res = evaluate_run(model, arm_config, seed, device, eval_steps=eval_steps)

    final_log = logs[-1]
    eval_res["final_train_loss"] = final_log["loss"]
    eval_res["final_sim_loss"] = final_log["sim_loss"]
    eval_res["final_var_loss"] = final_log["var_loss"]
    eval_res["final_cov_loss"] = final_log["cov_loss"]
    eval_res["param_count"] = param_count

    per_dim_std_train_last = None
    for entry in reversed(logs):
        if entry["per_dim_std"] is not None:
            per_dim_std_train_last = np.array(entry["per_dim_std"])
            break
    if per_dim_std_train_last is not None:
        eval_res["collapsed_train"] = bool(np.any(per_dim_std_train_last < 0.5))
        eval_res["per_dim_std_train"] = per_dim_std_train_last.tolist()
    else:
        eval_res["collapsed_train"] = False
        eval_res["per_dim_std_train"] = None

    eval_res["collapsed"] = eval_res["collapsed_eval"] or eval_res["collapsed_train"]
    if eval_res["final_train_loss"] > 50.0:
        eval_res["disqualified"] = True
        eval_res["collapsed"] = True
    else:
        eval_res["disqualified"] = False

    return eval_res, model, logs


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
ORIGINAL_SEEDS = [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
FRESH_SEEDS = [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]

ARMS = [
    {
        "name": "D0 (baseline replication)",
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 1.0,
        "batch_size": 64,
        "d_max": 8,
        "d_t": 3,
        "dyn_readout": "mean",
        "pos_encoding": "none",
        "primary_objective": "jepa",
        "ccr_mode": "covariance",
        "ccr_smooth_weight": 10.0,
        "ccr_spatial_weight": 10.0,
        "lr": 3e-4,
        "replay_buffer_capacity": 4000,
        "gdasr_log_only": True,
        "mask_dyn_sim": False,
        "seeds": ORIGINAL_SEEDS,
    },
    {
        "name": "C1 (primary, mask_dyn_sim)",
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 1.0,
        "batch_size": 64,
        "d_max": 8,
        "d_t": 3,
        "dyn_readout": "mean",
        "pos_encoding": "none",
        "primary_objective": "jepa",
        "ccr_mode": "covariance",
        "ccr_smooth_weight": 10.0,
        "ccr_spatial_weight": 10.0,
        "lr": 3e-4,
        "replay_buffer_capacity": 4000,
        "gdasr_log_only": True,
        "mask_dyn_sim": True,
        "seeds": ORIGINAL_SEEDS,
    },
    {
        "name": "C2 (seed robustness)",
        "sim_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 1.0,
        "batch_size": 64,
        "d_max": 8,
        "d_t": 3,
        "dyn_readout": "mean",
        "pos_encoding": "none",
        "primary_objective": "jepa",
        "ccr_mode": "covariance",
        "ccr_smooth_weight": 10.0,
        "ccr_spatial_weight": 10.0,
        "lr": 3e-4,
        "replay_buffer_capacity": 4000,
        "gdasr_log_only": True,
        "mask_dyn_sim": True,
        "seeds": FRESH_SEEDS,
    },
    {
        "name": "C3 (weight robustness)",
        "sim_weight": 25.0,
        "var_weight": 27.5,
        "cov_weight": 1.1,
        "batch_size": 64,
        "d_max": 8,
        "d_t": 3,
        "dyn_readout": "mean",
        "pos_encoding": "none",
        "primary_objective": "jepa",
        "ccr_mode": "covariance",
        "ccr_smooth_weight": 10.0,
        "ccr_spatial_weight": 10.0,
        "lr": 3e-4,
        "replay_buffer_capacity": 4000,
        "gdasr_log_only": True,
        "mask_dyn_sim": True,
        "seeds": ORIGINAL_SEEDS,
    },
]


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------
def _sanitize_arm_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "")
        .replace("=", "_")
        .replace("'", "")
    )


def _flatten_result(res, runs_dir):
    flat = {k: v for k, v in res.items() if k not in ("centroid_mse_per_object", "r2_per_dim", "dim_to_obj")}
    flat["per_dim_std"] = str(res.get("per_dim_std", []))
    flat["per_dim_std_train"] = str(res.get("per_dim_std_train", []))
    safe_name = _sanitize_arm_name(res["arm"])
    run_id = f"{safe_name}_seed{res['seed']}"
    csv_path = os.path.join(runs_dir, f"{run_id}.csv")
    json_path = os.path.join(runs_dir, f"{run_id}.json")
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
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
    run_id = f"{safe_name}_seed{seed}"
    ckpt_path = os.path.join(checkpoints_dir, f"{safe_name}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    logs_path = os.path.join(runs_dir, f"{safe_name}_seed{seed}_logs.csv")
    pd.DataFrame(logs).to_csv(logs_path, index=False)
    return eval_res


def _run_single_worker_with_timeout(args_tuple, timeout_sec=600):
    """
    Wrapper for _run_single_worker that adds per-seed timeout.
    If timeout occurs, return a timeout result dict.
    """
    arm, seed, device_str, dry_run, runs_dir, checkpoints_dir = args_tuple
    name = arm["name"]
    print(f"[{name}] seed={seed} -> starting on {device_str} (dry_run={dry_run}, timeout={timeout_sec}s)")
    
    device = torch.device(device_str)
    torch.set_num_threads(1)

    def _inner():
        eval_res, model, logs = run_single(arm, seed, device, dry_run=dry_run)
        csv_path = _flatten_result(eval_res, runs_dir)
        safe_name = _sanitize_arm_name(eval_res["arm"])
        run_id = f"{safe_name}_seed{seed}"
        ckpt_path = os.path.join(checkpoints_dir, f"{safe_name}_seed{seed}.pt")
        torch.save(model.state_dict(), ckpt_path)
        logs_path = os.path.join(runs_dir, f"{safe_name}_seed{seed}_logs.csv")
        pd.DataFrame(logs).to_csv(logs_path, index=False)
        return eval_res

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_inner)
        try:
            eval_res = future.result(timeout=timeout_sec)
            eval_res["timeout"] = False
            return eval_res
        except concurrent.futures.TimeoutError:
            print(f"[{name}] seed={seed} TIMED OUT after {timeout_sec}s (engineering failure, not collapse)")
            timeout_result = {
                "arm": name,
                "seed": seed,
                "collapsed": False,
                "collapsed_eval": False,
                "collapsed_train": False,
                "timeout": True,
                "disqualified": False,
                "per_dim_std": [],
                "per_dim_std_train": [],
                "vicreg_per_dim_std": [],
                "vicreg_mean_abs_corr": 0.0,
                "centroid_mse_mean": float("nan"),
                "centroid_r_mean": 0.0,
                "delta_r2_color": 0.0,
                "r2_dyn_color": 0.0,
                "r2_coord_color": 0.0,
                "r2_dyn_pos": 0.0,
                "r2_coord_pos": 0.0,
                "r2_dyn_identity": 0.0,
                "r2_coord_identity": 0.0,
                "delta_r2_identity": 0.0,
                "final_train_loss": float("nan"),
                "final_sim_loss": float("nan"),
                "final_var_loss": float("nan"),
                "final_cov_loss": float("nan"),
                "param_count": None,
                "dim_to_obj": {},
            }
            # Still write the timeout result to disk
            csv_path = _flatten_result(timeout_result, runs_dir)
            return timeout_result


def _run_single_worker_with_timeout_wrapper(args_tuple, timeout_sec=600):
    """
    Top-level function for parallel pool. Delegates to _run_single_worker_with_timeout.
    """
    return _run_single_worker_with_timeout(args_tuple, timeout_sec=timeout_sec)


# ---------------------------------------------------------------------------
# Analysis generation
# ---------------------------------------------------------------------------
def _generate_analysis(df_all, results_dir):
    lines = []
    lines.append("# Iter_028 Shared-Backbone mask_dyn_sim Probe — Analysis\n")
    lines.append("**Dual collapse criterion:** collapsed = collapsed_eval OR collapsed_train\n")
    lines.append("**Sanity disqualification:** final_train_loss > 50 counted as collapsed\n")
    lines.append("**Arms:** D0 (baseline), C1 (primary), C2 (seed robustness), C3 (weight robustness)\n")
    lines.append("**Gate threshold:** ≤10% collapse rate (dual criterion)\n")
    lines.append("**Relative threshold (F4/H2):** D0's ΔR² and mean_abs_corr are in-iteration null references\n\n")
    lines.append("---\n")

    arm_order = ["D0 (baseline replication)", "C1 (primary, mask_dyn_sim)", "C2 (seed robustness)", "C3 (weight robustness)"]

    # ---- Per-Arm Summary ----
    lines.append("## Per-Arm Summary\n")
    for arm_name in arm_order:
        df_arm = df_all[df_all["arm"] == arm_name]
        if len(df_arm) == 0:
            continue
        lines.append(f"### {arm_name}\n")
        lines.append(f"- N seeds: {len(df_arm)}\n")

        # Timeout count
        if "timeout" in df_arm.columns:
            timeout_count = int(df_arm["timeout"].sum())
        else:
            timeout_count = 0
        not_timeout = df_arm[~df_arm["timeout"]] if "timeout" in df_arm.columns else df_arm

        cr_dual = df_arm["collapsed"].mean()
        cc_dual = int(df_arm["collapsed"].sum())
        cr_eval = df_arm["collapsed_eval"].mean()
        cc_eval = int(df_arm["collapsed_eval"].sum())
        cr_train = df_arm["collapsed_train"].mean()
        cc_train = int(df_arm["collapsed_train"].sum())
        lines.append(f"- Collapse rate (dual): {cr_dual:.2f} ({cc_dual}/{len(df_arm)})\n")
        lines.append(f"- Collapse rate (eval-only): {cr_eval:.2f} ({cc_eval}/{len(df_arm)})\n")
        lines.append(f"- Collapse rate (train-only): {cr_train:.2f} ({cc_train}/{len(df_arm)})\n")

        # PRIMARY (excluding timeouts)
        if len(not_timeout) > 0:
            cr_primary = not_timeout["collapsed"].mean()
            cc_primary = int(not_timeout["collapsed"].sum())
            lines.append(f"- **PRIMARY collapse rate (excl. timeouts):** {cr_primary:.2f} ({cc_primary}/{len(not_timeout)})\n")

            # SENSITIVITY (including timeouts as failures)
            if timeout_count > 0:
                n_total = len(df_arm)
                cc_sensitivity = cc_primary + timeout_count
                cr_sensitivity = cc_sensitivity / n_total
                lines.append(f"- **SENSITIVITY collapse rate (incl. timeouts as failures):** {cr_sensitivity:.2f} ({cc_sensitivity}/{n_total})\n")
        else:
            lines.append(f"- **PRIMARY collapse rate (excl. timeouts):** N/A (all timed out)\n")

        lines.append(f"- **Timeout count:** {timeout_count}\n")
        dq_count = int(df_arm["disqualified"].sum()) if "disqualified" in df_arm.columns else 0
        if dq_count > 0:
            lines.append(f"- **Disqualified seeds (loss > 50):** {dq_count}\n")
        lines.append(f"- Mean final train loss: {df_arm['final_train_loss'].mean():.4f} +/- {df_arm['final_train_loss'].std():.4f}\n")
        lines.append(f"- Centroid MSE (REF ONLY): {df_arm['centroid_mse_mean'].mean():.2f}\n")
        lines.append(f"- delta_R2_color (REF ONLY): {df_arm['delta_r2_color'].mean():.4f}\n")
        lines.append(f"- Mean abs corr: {df_arm['vicreg_mean_abs_corr'].mean():.3f}\n")
        if "param_count" in df_arm.columns:
            lines.append(f"- Parameter count: {int(df_arm['param_count'].iloc[0])}\n")
        lines.append("\n")

    # ---- Per-Seed Train-vs-Eval Std Gap Table ----
    lines.append("## Per-Seed Train-vs-Eval Std Gap Table (CO-EQUAL reporting)\n\n")
    lines.append("| seed | arm | collapsed_eval | collapsed_train | collapsed | per_dim_std_eval | per_dim_std_train |\n")
    lines.append("|------|-----|----------------|-----------------|-----------|------------------|-------------------|\n")
    for _, row in df_all.iterrows():
        s = int(row["seed"])
        arm = row["arm"]
        ce = "Y" if row["collapsed_eval"] else "N"
        ct = "Y" if row["collapsed_train"] else "N"
        c = "Y" if row["collapsed"] else "N"
        pds_e = str(row.get("per_dim_std", []))
        pds_t = str(row.get("per_dim_std_train", []))
        lines.append(f"| {s} | {arm} | {ce} | {ct} | {c} | {pds_e} | {pds_t} |\n")
    lines.append("\n")

    # ---- Gate Check ----
    lines.append("## Gate Check (PRIMARY — excluding timeouts)\n\n")
    for arm_name in arm_order:
        df_arm = df_all[df_all["arm"] == arm_name]
        if len(df_arm) == 0:
            continue
        # Use PRIMARY rate (excluding timeouts)
        if "timeout" in df_arm.columns:
            not_timeout = df_arm[~df_arm["timeout"]]
        else:
            not_timeout = df_arm
        if len(not_timeout) > 0:
            cr = not_timeout["collapsed"].mean()
            status = "PASS" if cr <= 0.10 else "FAIL"
            lines.append(f"- **{arm_name}:** {status} (PRIMARY collapse rate {cr:.2f}, n={len(not_timeout)})\n")
        else:
            lines.append(f"- **{arm_name}:** SKIP (all timed out)\n")
    lines.append("\n")

    # ---- D0 vs C1 Relative Threshold Comparison ----
    lines.append("## D0 vs C1 — Independent Readout Comparison (Relative-Threshold Gate)\n\n")
    df_D0 = df_all[df_all["arm"] == "D0 (baseline replication)"]
    df_C1 = df_all[df_all["arm"] == "C1 (primary, mask_dyn_sim)"]

    if len(df_D0) > 0 and len(df_C1) > 0:
        d0_delta_r2 = df_D0["delta_r2_color"].mean()
        c1_delta_r2 = df_C1["delta_r2_color"].mean()
        d0_mac = df_D0["vicreg_mean_abs_corr"].mean()
        c1_mac = df_C1["vicreg_mean_abs_corr"].mean()

        lines.append(f"**D0 (null reference):**\n")
        lines.append(f"- Mean ΔR²_color: {d0_delta_r2:.4f}\n")
        lines.append(f"- Mean mean_abs_corr: {d0_mac:.4f}\n\n")

        lines.append(f"**C1 (primary):**\n")
        lines.append(f"- Mean ΔR²_color: {c1_delta_r2:.4f}\n")
        lines.append(f"- Mean mean_abs_corr: {c1_mac:.4f}\n\n")

        lines.append(f"**Relative thresholds:**\n")
        lines.append(f"- ΔR² gate: C1 ≥ D0 + 0.05 = {d0_delta_r2 + 0.05:.4f}; C1 = {c1_delta_r2:.4f} → ")
        delta_gate_pass = c1_delta_r2 >= (d0_delta_r2 + 0.05)
        lines.append(f"{'PASS' if delta_gate_pass else 'FAIL'}\n")

        lines.append(f"- mean_abs_corr gate: C1 ≤ D0 + 0.05 = {d0_mac + 0.05:.4f}; C1 = {c1_mac:.4f} → ")
        mac_gate_pass = c1_mac <= (d0_mac + 0.05)
        lines.append(f"{'PASS' if mac_gate_pass else 'FAIL'}\n\n")

        lines.append(f"H2 relative gate (ΔR² AND mean_abs_corr): {'PASS' if (delta_gate_pass and mac_gate_pass) else 'FAIL'}\n")
    else:
        lines.append("*Missing D0 or C1 data — relative threshold comparison skipped.*\n\n")

    # ---- Pre-Registered Outcome Classification ----
    lines.append("## Pre-Registered Outcome Classification\n\n")
    df_C1_a = df_all[df_all["arm"] == "C1 (primary, mask_dyn_sim)"]
    df_C2_a = df_all[df_all["arm"] == "C2 (seed robustness)"]
    df_C3_a = df_all[df_all["arm"] == "C3 (weight robustness)"]

    cr_C1 = df_C1_a["collapsed"].mean() if len(df_C1_a) > 0 else 1.0
    cr_C2 = df_C2_a["collapsed"].mean() if len(df_C2_a) > 0 else 1.0
    cr_C3 = df_C3_a["collapsed"].mean() if len(df_C3_a) > 0 else 1.0

    # Use PRIMARY rates (excluding timeouts) for gate evaluation
    if "timeout" in df_C1_a.columns:
        c1_not_to = df_C1_a[~df_C1_a["timeout"]]
        cr_C1 = c1_not_to["collapsed"].mean() if len(c1_not_to) > 0 else 1.0
    if "timeout" in df_C2_a.columns:
        c2_not_to = df_C2_a[~df_C2_a["timeout"]]
        cr_C2 = c2_not_to["collapsed"].mean() if len(c2_not_to) > 0 else 1.0
    if "timeout" in df_C3_a.columns:
        c3_not_to = df_C3_a[~df_C3_a["timeout"]]
        cr_C3 = c3_not_to["collapsed"].mean() if len(c3_not_to) > 0 else 1.0

    lines.append(f"C1 **PRIMARY** collapse rate (excl. timeouts): {cr_C1:.2f}\n")
    lines.append(f"C2 **PRIMARY** collapse rate (excl. timeouts): {cr_C2:.2f}\n")
    lines.append(f"C3 **PRIMARY** collapse rate (excl. timeouts): {cr_C3:.2f}\n\n")

    # F1
    if cr_C1 >= 0.20:
        lines.append("**F1:** FALSIFIED — mask_dyn_sim alone insufficient on the shared backbone.\n")

    # F2
    if cr_C2 >= 0.20:
        lines.append("**F2:** SEED-DEPENDENT — C1 result does not generalize to fresh seeds.\n")

    # F3
    if cr_C3 >= 0.20:
        lines.append("**F3:** NOT ROBUST — C1 result sensitive to weight perturbation.\n")

    # F4 / CONFIRMED / DOWNGRADED
    if cr_C1 <= 0.10:
        c1_delta_r2_val = df_C1_a["delta_r2_color"].mean() if len(df_C1_a) > 0 else 0.0
        d0_delta_r2_val = df_D0["delta_r2_color"].mean() if len(df_D0) > 0 else 0.0
        c1_mac_val = df_C1_a["vicreg_mean_abs_corr"].mean() if len(df_C1_a) > 0 else 1.0
        d0_mac_val = df_D0["vicreg_mean_abs_corr"].mean() if len(df_D0) > 0 else 0.0
        f4_trigger = (c1_delta_r2_val < d0_delta_r2_val + 0.05) or (c1_mac_val > d0_mac_val + 0.05)
        if f4_trigger:
            lines.append("**F4:** DOWNGRADED — VICReg-maintained variance is constructional; "
                         "z_dyn has variance but no meaningful semantic content improvement over D0.\n")
        else:
            lines.append("**H1+H2:** CONFIRMED — mask_dyn_sim on shared backbone does not destabilize "
                         "VICReg-maintained variance AND preserves semantic encoding (relative to D0).\n")
    lines.append("\n")

    # Sample-size caveat
    lines.append("## Sample-Size Caveat\n\n")
    lines.append("Fisher's exact test for 0/10 vs 3/10 gives p ≈ 0.21; the design cannot formally "
                 "distinguish 0% from 10–20% at this sample size. Results are reported as point "
                 "estimates with this limit explicitly noted.\n\n")

    # ---- Parameter Count Comparison ----
    lines.append("## Parameter Count Comparison\n\n")
    if "param_count" in df_all.columns:
        for arm_name in arm_order:
            df_arm = df_all[df_all["arm"] == arm_name]
            if len(df_arm) > 0 and "param_count" in df_arm.columns:
                lines.append(f"- {arm_name}: {int(df_arm['param_count'].iloc[0])}\n")
    lines.append("\n")

    # ---- Timeout Audit ----
    lines.append("## Timeout Audit\n\n")
    lines.append("Timeouts are ENGINEERING failures, NOT representation failures.\n")
    lines.append("The PRIMARY collapse rate (used for gates) excludes timeouts.\n\n")
    interpretable = True
    for arm_name in arm_order:
        df_arm = df_all[df_all["arm"] == arm_name]
        if len(df_arm) == 0:
            continue
        if "timeout" in df_arm.columns:
            to_count = int(df_arm["timeout"].sum())
        else:
            to_count = 0
        ok = "OK" if to_count <= 1 else "TOO MANY — re-launch with longer budget"
        if to_count > 1:
            interpretable = False
        lines.append(f"- **{arm_name}:** {to_count} timeout(s) → {ok}\n")
    lines.append(f"\n**Overall interpretable:** {'Yes' if interpretable else 'No — re-launch required'}\n")
    lines.append("\n")

    # ---- Language Constraints Reminder ----
    lines.append("## Language Constraints (from pre-registration)\n\n")
    lines.append("Results are reported using 'does not destabilize VICReg-maintained variance' or "
                 "'is consistent with.' Terms such as 'breakthrough', 'causal driver', 'eliminated', "
                 "'BEST', 'proves', 'demonstrates', or 'resolves' are NOT used.\n")
    lines.append("Even a fully-confirmed C1+C2+C3 result is reported per the constructional framing "
                 "in the hypothesis, not as 'sim_loss_dyn causes collapse.'\n")

    analysis_path = os.path.join(results_dir, "final_analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Saved analysis to {analysis_path}")


# ---------------------------------------------------------------------------
# Resume helper — scan for existing JSON results
# ---------------------------------------------------------------------------
def load_existing_results(runs_dir):
    """
    Scan runs_dir for .json result files.
    Returns a dict mapping (arm_name, seed) -> result_dict for valid results
    (files that contain both "arm" and "seed" keys).
    """
    results = {}
    if not os.path.isdir(runs_dir):
        return results
    for fname in os.listdir(runs_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(runs_dir, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            if "arm" in data and "seed" in data:
                key = (data["arm"], data["seed"])
                # Ensure timeout flag exists for merged results
                if "timeout" not in data:
                    data["timeout"] = False
                if "disqualified" not in data:
                    data["disqualified"] = False
                results[key] = data
                print(f"  Loaded existing result: {data['arm']} seed={data['seed']}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping invalid JSON file {fname}: {e}")
    return results


# ---------------------------------------------------------------------------
# Per-seed timeout wrapper
# ---------------------------------------------------------------------------
def run_single_with_timeout(arm_config, seed, device, dry_run=False, timeout_sec=600):
    """
    Wrap run_single with a per-seed timeout using ProcessPoolExecutor.
    If timeout occurs, return a timeout result dict (collapsed=False, timeout=True).
    Timeout is an ENGINEERING failure, not a representation failure.
    """
    import concurrent.futures
    name = arm_config["name"]

    def _wrapper():
        return run_single(arm_config, seed, device, dry_run=dry_run)

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_wrapper)
        try:
            eval_res, model, logs = future.result(timeout=timeout_sec)
            eval_res["timeout"] = False
            return eval_res, model, logs
        except concurrent.futures.TimeoutError:
            print(f"[{name}] seed={seed} TIMED OUT after {timeout_sec}s (engineering failure, not collapse)")
            timeout_result = {
                "arm": name,
                "seed": seed,
                "collapsed": False,
                "collapsed_eval": False,
                "collapsed_train": False,
                "timeout": True,
                "disqualified": False,
                "per_dim_std": [],
                "per_dim_std_train": [],
                "vicreg_per_dim_std": [],
                "vicreg_mean_abs_corr": 0.0,
                "centroid_mse_mean": float("nan"),
                "centroid_r_mean": 0.0,
                "delta_r2_color": 0.0,
                "r2_dyn_color": 0.0,
                "r2_coord_color": 0.0,
                "r2_dyn_pos": 0.0,
                "r2_coord_pos": 0.0,
                "r2_dyn_identity": 0.0,
                "r2_coord_identity": 0.0,
                "delta_r2_identity": 0.0,
                "final_train_loss": float("nan"),
                "final_sim_loss": float("nan"),
                "final_var_loss": float("nan"),
                "final_cov_loss": float("nan"),
                "param_count": None,
                "dim_to_obj": {},
            }
            return timeout_result, None, []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cpu_count = os.cpu_count() or 2
    default_workers = min(cpu_count - 1, 8)

    parser = argparse.ArgumentParser(description="Iter_028 Shared-Backbone mask_dyn_sim Probe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=lambda x: (str(x).lower() in ("true", "1", "yes")), default=True)
    args = parser.parse_args()

    dry_run = args.dry_run
    seeds = args.seeds  # if None, each arm uses its own seeds
    max_workers = args.workers
    resume = args.resume

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)

    print(f"Device: {device} | Dry-run: {dry_run} | Seeds override: {seeds} | Workers: {max_workers if not args.sequential else 1}")

    results_dir = "archive/iter_028/results"
    runs_dir = os.path.join(results_dir, "runs")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Log parameter counts BEFORE runs
    print("\n=== Parameter Counts ===")
    for arm in ARMS:
        m = NonParametricJEPASpatial(
            d_max=arm["d_max"], h=3, k=4, pos_encoding="none",
            primary_objective="jepa", gdasr_log_only=True,
            dyn_readout=arm["dyn_readout"], sub_features=1, dyn_source="spatial",
        )
        m.d_t = arm["d_t"]
        pc = count_parameters(m)
        print(f"  {arm['name']}: {pc} parameters (sim_weight={arm['sim_weight']}, var_weight={arm['var_weight']}, mask_dyn_sim={arm['mask_dyn_sim']})")
    print()

    # Build task list — each arm has its own seed set (unless --seeds overrides)
    tasks = []
    for arm in ARMS:
        arm_seeds = seeds if seeds is not None else arm["seeds"]
        for seed in arm_seeds:
            tasks.append((arm, seed, device_str, dry_run, runs_dir, checkpoints_dir))

    # ---- Resume logic ----
    existing_results = {}
    if resume:
        existing_results = load_existing_results(runs_dir)
    tasks_to_run = [t for t in tasks if (t[0]["name"], t[1]) not in existing_results]
    total_new = len(tasks_to_run)
    total_existing = len(tasks) - total_new
    if total_existing > 0:
        print(f"Resuming: {total_existing} existing results loaded, {total_new} new tasks to run.")
    else:
        print(f"Total tasks to run: {total_new}")

    # Merge existing results first
    all_results = list(existing_results.values())

    if args.sequential:
        for i, task in enumerate(tasks_to_run):
            arm_name = task[0]["name"]
            seed = task[1]
            print(f"\n{'=' * 70}\nTASK [{i + 1}/{total_new}]: {arm_name} seed={seed}\n{'=' * 70}")
            res = _run_single_worker_with_timeout(task, timeout_sec=600)
            all_results.append(res)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_label = {
                executor.submit(_run_single_worker_with_timeout_wrapper, task, 600): f"{task[0]['name']} seed={task[1]}"
                for task in tasks_to_run
            }
            for future in concurrent.futures.as_completed(future_to_label):
                label = future_to_label[future]
                try:
                    res = future.result(timeout=720)  # slightly more than per-seed timeout to allow graceful handling
                    all_results.append(res)
                    print(f"COMPLETED: {label}")
                except concurrent.futures.TimeoutError:
                    print(f"TIMEOUT (wrapper): {label}")
                    arm_name = task[0]["name"] if hasattr(task, '__getitem__') else label.split()[0]
                    seed = task[1] if hasattr(task, '__getitem__') else 0
                    all_results.append({
                        "arm": arm_name,
                        "seed": seed,
                        "collapsed": False,
                        "collapsed_eval": False,
                        "collapsed_train": False,
                        "timeout": True,
                        "disqualified": False,
                    })
                except Exception as exc:
                    print(f"FAILED: {label} -> {exc}")

    df_all = pd.DataFrame(all_results)
    summary_path = os.path.join(results_dir, "summary_iter_028.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    _generate_analysis(df_all, results_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
