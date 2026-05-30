#!/usr/bin/env python3
"""
Iter_031 Part A: Reconstruction + VICReg Ceiling Probe on Separate Backbone.
"""

import os, sys, json, random, argparse, collections, warnings, concurrent.futures, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.environment import PhysicsSandbox
from src.models_recon import ReconVICRegSeparateDyn


class ExtendedReplayBuffer:
    def __init__(self, capacity=4000):
        self.capacity = capacity; self.buffer = []; self.position = 0
    def push(self, x_hist, x_target, positions, colors, radii):
        if len(self.buffer) < self.capacity: self.buffer.append(None)
        self.buffer[self.position] = (x_hist, x_target, positions, colors, radii)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        xh, xt, p, c, r = zip(*batch)
        return np.stack(xh), np.stack(xt), np.stack(p), np.stack(c), np.stack(r)
    def clear(self): self.buffer = []; self.position = 0
    def __len__(self): return len(self.buffer)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def check_collapse(z_dyn, d_t, std_threshold=0.5):
    z_active = z_dyn[:, :d_t]; per_dim_std = np.std(z_active, axis=0)
    return np.any(per_dim_std < std_threshold), per_dim_std


def compute_vicreg_health(z_dyn, d_t):
    z_active = z_dyn[:, :d_t]; per_dim_std = np.std(z_active, axis=0)
    if d_t > 1:
        corr = np.corrcoef(z_active.T); triu = np.triu_indices(d_t, k=1)
        mean_abs_corr = float(np.mean(np.abs(corr[triu])))
    else: mean_abs_corr = 0.0
    return {"per_dim_std": per_dim_std.tolist(), "mean_abs_corr": mean_abs_corr}


def fit_linear_probe(z, y):
    Z = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y; return theta[0], theta[1]


def fit_multivariate_probe_r2(z_feature, y_target):
    z = z_feature.reshape(-1); y = y_target.reshape(len(z), -1); N = len(z)
    if N < 5 or y.shape[1] < 1: return 0.0
    Z_aug = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.linalg.pinv(Z_aug.T @ Z_aug) @ Z_aug.T @ y
    y_pred = Z_aug @ theta; ss_res = np.sum((y - y_pred)**2); ss_tot = np.sum((y - np.mean(y, axis=0))**2)
    if ss_tot < 1e-12: return 0.0
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def compute_centroid_mse(model, test_env, test_history, num_samples=200, device="cpu"):
    model.eval(); centroids_list = []; pos_list = []
    # Effective dimension: model.d_coord = model.encoder.d_max (the actual output size)
    effective_d_coord = model.encoder.d_max
    obs = test_env.reset(); test_history.clear(); test_history.append(obs)
    for _ in range(4): obs, info = test_env.step({"acc": 0.0, "push": False}); test_history.append(obs)
    collected = 0
    while collected < num_samples:
        obs, info = test_env.step({"acc": 0.0, "push": False}); test_history.append(obs)
        if len(test_history) == 4:
            x_target_t = torch.from_numpy(test_history[3]).float().unsqueeze(0).to(device)
            with torch.no_grad(): z_coord, _ = model.encoder(x_target_t)
            centroids_list.append(z_coord[:, :effective_d_coord].cpu().numpy())
            true_pos = np.array(info["positions"][:effective_d_coord])
            if len(true_pos) < effective_d_coord: true_pos = np.pad(true_pos, (0, effective_d_coord - len(true_pos)), constant_values=np.nan)
            pos_list.append(true_pos); collected += 1
    centroids_arr = np.concatenate(centroids_list, axis=0); pos_arr = np.array(pos_list)
    mse_per_object = []; r_per_object = []
    for obj_idx in range(effective_d_coord):
        mask = ~np.isnan(pos_arr[:, obj_idx])
        if mask.sum() < 10: mse_per_object.append(float("nan")); r_per_object.append(0.0); continue
        z_obj = centroids_arr[mask, obj_idx]; y_obj = pos_arr[mask, obj_idx]
        w, b = fit_linear_probe(z_obj, y_obj); y_pred = z_obj * w + b
        mse = float(np.mean((y_obj - y_pred)**2)); r = float(np.corrcoef(z_obj, y_obj)[0, 1]) if len(z_obj) > 2 else 0.0
        if np.isnan(r): r = 0.0
        mse_per_object.append(mse); r_per_object.append(abs(r))
    return {"mse_per_object": mse_per_object, "mse_mean": float(np.nanmean(mse_per_object)), "r_per_object": r_per_object, "r_mean": float(np.nanmean(r_per_object))}


def collect_multitraj_eval_data(model, num_samples=200, base_seed=30000, device="cpu"):
    model.eval(); num_trajectories = max(1, num_samples // 20); samples_per_traj = max(1, num_samples // num_trajectories)
    traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii = [], [], [], [], []
    with torch.no_grad():
        for t_idx in range(num_trajectories):
            env_seed = base_seed + t_idx * 100; env = PhysicsSandbox(N=3, seed=env_seed)
            history = collections.deque(maxlen=4); obs = env.reset(); history.append(obs)
            for _ in range(3): obs, info = env.step({"acc": 0.0, "push": False}); history.append(obs)
            z_dyn_t, z_coord_t, pos_t, colors_t, radii_t = [], [], [], [], []
            collected = 0
            while collected < samples_per_traj:
                obs, info = env.step({"acc": 0.0, "push": False}); history.append(obs)
                if len(history) == 4:
                    x_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    z_c, z_d = model.encoder(x_t)
                    z_dyn_t.append(z_d[0].cpu().numpy()); z_coord_t.append(z_c[0].cpu().numpy())
                    pos_t.append(info["positions"]); colors_t.append(info["colors"]); radii_t.append(info["radii"])
                    collected += 1
            traj_z_dyn.append(np.array(z_dyn_t)); traj_z_coord.append(np.array(z_coord_t))
            traj_pos.append(np.array(pos_t)); traj_colors.append(np.array(colors_t)); traj_radii.append(np.array(radii_t))
    return (traj_z_dyn, traj_z_coord, traj_pos, traj_colors, traj_radii,
            np.concatenate(traj_z_dyn, axis=0), np.concatenate(traj_z_coord, axis=0),
            np.concatenate(traj_pos, axis=0), np.concatenate(traj_colors, axis=0), np.concatenate(traj_radii, axis=0))


def _compute_semantic_probes_core(z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr, d_t, sub_features, train_ratio=0.5):
    N = pos_arr.shape[1]; num_samples = z_dyn_arr.shape[0]
    if sub_features > 1: z_dyn_pooled = z_dyn_arr[:, :d_t * sub_features].reshape(num_samples, d_t, sub_features).mean(axis=2)
    else: z_dyn_pooled = z_dyn_arr[:, :d_t]
    cost = np.zeros((d_t, N))
    for d in range(d_t):
        for o in range(N): cost[d, o] = np.mean(np.abs(z_coord_arr[:, d] - pos_arr[:, o]))
    row_ind, col_ind = linear_sum_assignment(cost)
    dim_to_obj = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    n_train = int(num_samples * train_ratio)
    def fit_probe_r2(z_feature, y_target):
        z_train, y_train = z_feature[:n_train], y_target[:n_train]; z_test, y_test = z_feature[n_train:], y_target[n_train:]
        if z_train.size < 5 or y_train.size < 5: return 0.0
        w, b = fit_linear_probe(z_train, y_train); y_pred = z_test * w + b
        ss_res = np.sum((y_test - y_pred)**2); ss_tot = np.sum((y_test - np.mean(y_train))**2)
        if ss_tot < 1e-12: return 0.0
        return float(1.0 - ss_res / ss_tot)
    r2_per_dim = []
    for d in range(d_t):
        if d not in dim_to_obj:
            r2_per_dim.append({"dyn_color": 0.0, "coord_color": 0.0, "dyn_pos": 0.0, "coord_pos": 0.0, "dyn_identity": 0.0, "coord_identity": 0.0}); continue
        obj = dim_to_obj[d]; z_d = z_dyn_pooled[:, d]; z_c = z_coord_arr[:, d]
        r2_dyn_pos = fit_probe_r2(z_d, pos_arr[:, obj]); r2_coord_pos = fit_probe_r2(z_c, pos_arr[:, obj])
        r2_dyn_ch = [fit_probe_r2(z_d, colors_arr[:, obj, ch]) for ch in range(3)]
        r2_coord_ch = [fit_probe_r2(z_c, colors_arr[:, obj, ch]) for ch in range(3)]
        max_radius = 20.0; identity_vec = np.zeros((num_samples, 4))
        identity_vec[:, :3] = colors_arr[:, obj, :]; identity_vec[:, 3] = radii_arr[:, obj] / max_radius
        r2_per_dim.append({
            "dyn_color": float(np.mean(r2_dyn_ch)), "coord_color": float(np.mean(r2_coord_ch)),
            "dyn_pos": r2_dyn_pos, "coord_pos": r2_coord_pos,
            "dyn_identity": fit_multivariate_probe_r2(z_d, identity_vec), "coord_identity": fit_multivariate_probe_r2(z_c, identity_vec),
        })
    r2_dyn_color_all = np.mean([p["dyn_color"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_color_all = np.mean([p["coord_color"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_dyn_pos_all = np.mean([p["dyn_pos"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_pos_all = np.mean([p["coord_pos"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_dyn_identity_all = np.mean([p["dyn_identity"] for p in r2_per_dim]) if r2_per_dim else 0.0
    r2_coord_identity_all = np.mean([p["coord_identity"] for p in r2_per_dim]) if r2_per_dim else 0.0
    return {"dim_to_obj": dim_to_obj, "r2_per_dim": r2_per_dim,
            "r2_dyn_color": float(r2_dyn_color_all), "r2_coord_color": float(r2_coord_color_all),
            "delta_r2_color": float(r2_dyn_color_all - r2_coord_color_all),
            "r2_dyn_pos": float(r2_dyn_pos_all), "r2_coord_pos": float(r2_coord_pos_all),
            "r2_dyn_identity": float(r2_dyn_identity_all), "r2_coord_identity": float(r2_coord_identity_all),
            "delta_r2_identity": float(r2_dyn_identity_all - r2_coord_identity_all)}


def compute_semantic_probes(model, num_samples=200, train_ratio=0.5, base_seed=30000, device="cpu"):
    model.eval()
    # Clamp d_t to actual model capacity
    effective_d_t = min(model.d_t, model.encoder.d_max)
    (_, _, _, _, _, z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr) = collect_multitraj_eval_data(model, num_samples=num_samples, base_seed=base_seed, device=device)
    return _compute_semantic_probes_core(z_dyn_arr, z_coord_arr, pos_arr, colors_arr, radii_arr, effective_d_t, model.sub_features, train_ratio=train_ratio)


def evaluate_run(model, arm_config, seed, device, eval_steps=200):
    d_t = arm_config.get("d_t", 3); sub_features = arm_config.get("sub_features", 1); name = arm_config["name"]
    d_max = arm_config.get("d_max", 8)
    # Effective dyn/coord dimension is bounded by both d_t and actual model capacity
    effective_d_t = min(d_t, d_max)
    eval_env = PhysicsSandbox(N=3, seed=seed + 10000); eval_history = collections.deque(maxlen=4)
    obs = eval_env.reset(); eval_history.append(obs)
    for _ in range(4): obs, _ = eval_env.step({"acc": 0.0, "push": False}); eval_history.append(obs)
    z_dyn_all, z_coord_all, recon_losses = [], [], []
    for _ in range(eval_steps):
        obs, _ = eval_env.step({"acc": 0.0, "push": False}); eval_history.append(obs)
        if len(eval_history) == 4:
            x_hist_t = torch.from_numpy(np.stack(list(eval_history)[:3], axis=0)).float().unsqueeze(0).to(device)
            x_target_t = torch.from_numpy(eval_history[3]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                loss_dict, _, (z_coord, z_dyn) = model(x_hist_t, x_target_t)
            z_dyn_all.append(z_dyn[:, :effective_d_t * sub_features].cpu().numpy()); z_coord_all.append(z_coord[:, :effective_d_t].cpu().numpy())
            recon_losses.append(loss_dict["recon_loss"].cpu().item())
    z_dyn_arr = np.concatenate(z_dyn_all, axis=0); z_coord_arr = np.concatenate(z_coord_all, axis=0)
    if sub_features > 1: z_dyn_pooled_check = z_dyn_arr[:, :effective_d_t * sub_features].reshape(eval_steps, effective_d_t, sub_features).mean(axis=2)
    else: z_dyn_pooled_check = z_dyn_arr[:, :effective_d_t]
    has_collapsed_eval, per_dim_std = check_collapse(z_dyn_pooled_check, effective_d_t); vh = compute_vicreg_health(z_dyn_pooled_check, effective_d_t)
    
    test_env = PhysicsSandbox(N=3, seed=seed + 20000); test_history = collections.deque(maxlen=4)
    obs = test_env.reset(); test_history.append(obs)
    centroid_mse = compute_centroid_mse(model, test_env, test_history, num_samples=eval_steps, device=device)
    semantic = compute_semantic_probes(model, num_samples=eval_steps, base_seed=seed + 30000, device=device)
    return {"arm": name, "seed": seed, "collapsed_eval": bool(has_collapsed_eval),
            "per_dim_std": per_dim_std.tolist(), "vicreg_per_dim_std": per_dim_std.tolist(),
            "vicreg_mean_abs_corr": vh["mean_abs_corr"],
            "centroid_mse_mean": centroid_mse["mse_mean"], "centroid_r_mean": centroid_mse["r_mean"],
            "delta_r2_color": semantic["delta_r2_color"], "r2_dyn_color": semantic["r2_dyn_color"],
            "r2_coord_color": semantic["r2_coord_color"], "r2_dyn_pos": semantic["r2_dyn_pos"],
            "r2_coord_pos": semantic["r2_coord_pos"], "r2_dyn_identity": semantic["r2_dyn_identity"],
            "r2_coord_identity": semantic["r2_coord_identity"], "delta_r2_identity": semantic["delta_r2_identity"],
            "dim_to_obj": semantic["dim_to_obj"],
            "recon_mse_mean": float(np.mean(recon_losses))}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sanitize_result_for_mp(res):
    clean = {}
    for k, v in res.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
        elif isinstance(v, (np.ndarray, np.generic)):
            clean[k] = v.tolist() if v.ndim > 0 else float(v)
        else:
            clean[k] = v
    return clean


def run_single(arm_config, seed, device, dry_run=False):
    name = arm_config["name"]
    sim_weight = arm_config.get("sim_weight", 1.0)
    var_weight = arm_config.get("var_weight", 25.0)
    cov_weight = arm_config.get("cov_weight", 25.0)
    recon_weight_val = arm_config.get("recon_weight", 25.0)
    pos_encoding = arm_config.get("pos_encoding", "none")
    d_t = arm_config.get("d_t", 3)
    d_max = arm_config.get("d_max", 8)
    lr = arm_config.get("lr", 3e-4)
    batch_size = arm_config.get("batch_size", 32)
    replay_buffer_capacity = arm_config.get("replay_buffer_capacity", 4000)
    coord_vicreg = arm_config.get("coord_vicreg", True)

    set_seed(seed)
    total_steps = 5 if dry_run else 8000
    eval_steps = min(200, total_steps // 2) if dry_run else 200

    model = ReconVICRegSeparateDyn(
        d_max=d_max, h=3, k=4, cooldown=300, stabilization_period=100,
        pos_encoding=pos_encoding, dyn_readout="mean", sub_features=1,
        dyn_source="spatial", coord_vicreg=coord_vicreg,
        recon_weight=recon_weight_val, var_weight=var_weight,
        cov_weight=cov_weight, sim_weight=sim_weight)
    
    # Freeze encoder parameters if specified
    if arm_config.get("freeze_encoder", False):
        for param in model.encoder.parameters():
            param.requires_grad = False

    model.d_t = d_t; model = model.to(device)
    param_count = count_parameters(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    env = PhysicsSandbox(N=3, seed=seed); obs = env.reset()
    history = collections.deque(maxlen=4); history.append(obs)
    replay_buffer = ExtendedReplayBuffer(capacity=replay_buffer_capacity)
    
    def _prefill(n):
        while len(replay_buffer) < n:
            obs, info = env.step({"acc": 0.0, "push": False}); history.append(obs)
            if len(history) == 4:
                replay_buffer.push(np.stack(list(history)[:3], axis=0), history[3],
                                   info["positions"], info["colors"], info["radii"])
    
    prefill_steps = max(1, min(total_steps - 2, 200)); _prefill(prefill_steps)
    logs = []
    
    for step in range(max(1, prefill_steps + 1), total_steps + 1):
        obs, info = env.step({"acc": 0.0, "push": False}); history.append(obs)
        if len(history) == 4:
            replay_buffer.push(np.stack(list(history)[:3], axis=0), history[3],
                               info["positions"], info["colors"], info["radii"])
        x_hist_b, x_target_b, pos_b, colors_b, radii_b = replay_buffer.sample(min(batch_size, len(replay_buffer)))
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)
        
        model.train(); optimizer.zero_grad()

        loss_dict, _, (z_target_coord, z_target_dyn) = model(x_hist_t, x_target_t)
        total_loss = loss_dict["loss"]; total_loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        optimizer.step()
        
        sim_loss_val = loss_dict["sim_loss"].item()
        model.update_recruitment_logic(sim_loss_val, target_dim=d_t, step=step)
        
        log_entry = {
            "step": step, "loss": total_loss.item(),
            "recon_loss": loss_dict["recon_loss"].item(),
            "sim_loss": loss_dict["sim_loss"].item(),
            "var_loss": loss_dict["var_loss"].item(),
            "cov_loss": loss_dict["cov_loss"].item(),
            "per_dim_std": None,
        }
        if step % 500 == 0:
            with torch.no_grad():
                per_dim_std_train = z_target_dyn[:, :d_t].std(dim=0).cpu().numpy()
            log_entry["per_dim_std"] = per_dim_std_train.tolist()
        logs.append(log_entry)
        
        if step % 1000 == 0 or step == total_steps:
            print(f"  [{name}] seed={seed} step={step:5d}/{total_steps}"
                  f"  loss={log_entry['loss']:.4f}  recon={log_entry['recon_loss']:.4f}"
                  f"  sim={log_entry['sim_loss']:.4f}  var={log_entry['var_loss']:.4f}"
                  f"  cov={log_entry['cov_loss']:.4f}")
                  
    eval_res = evaluate_run(model, arm_config, seed, device, eval_steps=eval_steps)
    final_log = logs[-1]
    
    eval_res["final_train_loss"] = final_log["loss"]
    eval_res["final_recon_loss"] = final_log["recon_loss"]
    eval_res["final_sim_loss"] = final_log["sim_loss"]
    eval_res["final_var_loss"] = final_log["var_loss"]
    eval_res["final_cov_loss"] = final_log["cov_loss"]
    eval_res["param_count"] = param_count
    
    per_dim_std_train_last = None
    for entry in reversed(logs):
        if entry["per_dim_std"] is not None: per_dim_std_train_last = np.array(entry["per_dim_std"]); break
    if per_dim_std_train_last is not None:
        eval_res["collapsed_train"] = bool(np.any(per_dim_std_train_last < 0.5))
        eval_res["per_dim_std_train"] = per_dim_std_train_last.tolist()
    else:
        eval_res["collapsed_train"] = False; eval_res["per_dim_std_train"] = None
    eval_res["collapsed"] = eval_res["collapsed_eval"] or eval_res["collapsed_train"]
    eval_res["disqualified"] = False
    return eval_res, model, logs


# ------------------------------------------------------------------
# ARM definitions
# ------------------------------------------------------------------

ARMS = [
    {
        "name": "Arm A (d_max=8, trained)",
        "d_max": 8,
        "freeze_encoder": False,
        "recon_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "sim_weight": 1.0,
        "lr": 3e-4,
        "batch_size": 32,
        "d_t": 3,
        "pos_encoding": "none",
        "replay_buffer_capacity": 4000,
        "coord_vicreg": True,
    },
    {
        "name": "Arm B (d_max=2, trained)",
        "d_max": 2,
        "freeze_encoder": False,
        "recon_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "sim_weight": 1.0,
        "lr": 3e-4,
        "batch_size": 32,
        "d_t": 3,
        "pos_encoding": "none",
        "replay_buffer_capacity": 4000,
        "coord_vicreg": True,
    },
    {
        "name": "Arm C (d_max=8, random-encoder)",
        "d_max": 8,
        "freeze_encoder": True,
        "recon_weight": 25.0,
        "var_weight": 25.0,
        "cov_weight": 25.0,
        "sim_weight": 1.0,
        "lr": 3e-4,
        "batch_size": 32,
        "d_t": 3,
        "pos_encoding": "none",
        "replay_buffer_capacity": 4000,
        "coord_vicreg": True,
    }
]

SEEDS = [7, 17, 31, 53, 71, 83, 97, 113, 127, 149, 101, 103, 107, 109, 131, 137, 139, 151, 157, 163]


def _sanitize_arm_name(name):
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('=', '_').replace("'", "").replace(',', '')


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
    device = torch.device(device_str); torch.set_num_threads(1)
    name = arm["name"]
    print(f"[{name}] seed={seed} -> starting on {device} (dry_run={dry_run})")
    eval_res, model, logs = run_single(arm, seed, device, dry_run=dry_run)
    eval_res = _sanitize_result_for_mp(eval_res)
    csv_path = _flatten_result(eval_res, runs_dir)
    safe_name = _sanitize_arm_name(eval_res["arm"])
    ckpt_path = os.path.join(checkpoints_dir, f"{safe_name}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    logs_path = os.path.join(runs_dir, f"{safe_name}_seed{seed}_logs.csv")
    pd.DataFrame(logs).to_csv(logs_path, index=False)
    return eval_res


def _generate_analysis(df_all, results_dir):
    lines = []
    lines.append("# Iter_031 Part A — Reconstruction+VICReg Ceiling Probe Analysis\n\n")
    lines.append("This analysis tests whether Reconstruction+VICReg on the separate-backbone architecture "
                 "achieves mean ΔR²_color ≥ 0.30 (lower 95% CI ≥ 0.18) on non-collapsed seeds, "
                 "with non-trivial margins over capacity and training controls.\n\n")
    
    lines.append("## Pre-Registered Falsification Gates (F1–F4)\n\n")
    lines.append("| Gate | Criterion | Meaning | Result / Status |\n")
    lines.append("|------|-----------|---------|-----------------|\n")
    
    arm_names = [a["name"] for a in ARMS]
    df_a = df_all[df_all["arm"] == arm_names[0]]
    df_b = df_all[df_all["arm"] == arm_names[1]]
    df_c = df_all[df_all["arm"] == arm_names[2]]
    
    df_a_nc = df_a[~df_a["collapsed"]] if len(df_a) > 0 else df_a
    n_nc_a = len(df_a_nc)
    
    mean_a, std_a, ci_lower_a = 0.0, 0.0, 0.0
    if n_nc_a > 0:
        mean_a = df_a_nc["delta_r2_color"].mean()
        std_a = df_a_nc["delta_r2_color"].std(ddof=1) if n_nc_a > 1 else 0.0
        se_a = std_a / math.sqrt(n_nc_a)
        ci_lower_a = mean_a - 1.96 * se_a
        
    mean_b = df_b[~df_b["collapsed"]]["delta_r2_color"].mean() if len(df_b) > 0 else 0.0
    mean_c = df_c[~df_c["collapsed"]]["delta_r2_color"].mean() if len(df_c) > 0 else 0.0
    
    f1_pass = mean_a >= 0.30
    f2_pass = ci_lower_a >= 0.18 if n_nc_a >= 5 else False
    f3_pass = (mean_a - mean_b) >= 0.10
    f4_pass = (mean_a - mean_c) >= 0.10
    
    lines.append(f"| **F1** | mean ΔR²_color (Arm A) ≥ 0.30 | Ceiling clearance | {'PASS' if f1_pass else 'FAIL'} (mean={mean_a:.4f}) |\n")
    lines.append(f"| **F2** | Lower 95% CI of mean ΔR²_color (Arm A) ≥ 0.18 | Variance stability | {'PASS' if f2_pass else 'FAIL'} (lower CI={ci_lower_a:.4f}, n={n_nc_a}) |\n")
    lines.append(f"| **F3** | mean ΔR²_color (Arm A) — mean ΔR²_color (Arm B) ≥ 0.10 | Capacity-matters | {'PASS' if f3_pass else 'FAIL'} (diff={mean_a - mean_b:.4f}) |\n")
    lines.append(f"| **F4** | mean ΔR²_color (Arm A) — mean ΔR²_color (Arm C) ≥ 0.10 | Training-matters | {'PASS' if f4_pass else 'FAIL'} (diff={mean_a - mean_c:.4f}) |\n")
    lines.append("\n\n")
    
    lines.append("## Detailed Summary per Arm (all seeds)\n\n")
    for name in arm_names:
        df_arm = df_all[df_all["arm"] == name]
        if len(df_arm) == 0: continue
        lines.append(f"### {name}\n")
        lines.append(f"- N seeds: {len(df_arm)}\n")
        cr = df_arm["collapsed"].mean(); cc = int(df_arm["collapsed"].sum())
        lines.append(f"- Collapse rate: {cr:.2f} ({cc}/{len(df_arm)})\n")
        lines.append(f"- Mean ΔR²_color (all): {df_arm['delta_r2_color'].mean():.4f} ± {df_arm['delta_r2_color'].std():.4f}\n")
        lines.append(f"- Mean reconstruction MSE: {df_arm['recon_mse_mean'].mean():.6f}\n")
        lines.append(f"- Mean centroid MSE: {df_arm['centroid_mse_mean'].mean():.2f}\n")
        lines.append(f"- Mean abs corr: {df_arm['vicreg_mean_abs_corr'].mean():.3f}\n")
        lines.append("\n")

    lines.append("## Pre-Committed Mandate Revision Language\n\n")
    if f1_pass and f2_pass and f3_pass and f4_pass:
        lines.append("> **ALL GATES PASSED: Hypothesis Supported**\n"
                     "> \"Reconstruction+VICReg on the separate-backbone architecture achieves mean ΔR²_color ≥ 0.30 with robust variance. "
                     "The result is consistent with the architecture having sufficient capacity to preserve color information under a supervised pixel-MSE target. "
                     "The d_max=2 control (F3) and random-encoder control (F4) confirm that the finding is non-trivial: both capacity and training are required.\n"
                     ">\n"
                     "> **M2 is revised** from 'SFA+VICReg as primary' to 'Reconstruction+VICReg as primary representation objective, decoder-free constraint relaxed as pragmatic compromise. "
                     "SFA demoted to comparison baseline B1. Surprise readout retained via stop-gradient predictor. Future work may explore BYOL-style decoder-free alternatives approaching the reconstruction ceiling.'\"\n")
    elif not f1_pass or not f2_pass:
        lines.append("> **F1 or F2 FAILED: Ceiling not cleared or variance-unstable**\n"
                     "> \"Reconstruction+VICReg fails to achieve ΔR²_color ≥ 0.30 with variance-stability. Even a supervised pixel-reconstruction target cannot make the mean-readout z_dyn stream encode identity above the 0.30 threshold. "
                     "The z_dyn readout architecture itself constrains identity encoding regardless of objective class.\n"
                     ">\n"
                     "> **M2 revision pending architectural redesign:** priority is centroid-gated z_dyn readout or increased d_max.'\"\n")
    elif not f3_pass:
        lines.append("> **F3 FAILED: Capacity does not matter**\n"
                     "> \"d_max=2 under-capacity control achieves ΔR²_color within 0.10 of d_max=8. The finding is trivially explained by bottleneck capacity — "
                     "even severe under-capacity preserves color via reconstruction gradient. The result does not support a meaningful training effect.'\"\n")
    elif not f4_pass:
        lines.append("> **F4 FAILED: Training does not matter**\n"
                     "> \"Random-encoder control achieves ΔR²_color within 0.10 of trained-encoder. The finding is constructional — "
                     "random features with a trained decoder are sufficient to preserve color information. The result does not support a meaningful training effect on the encoder.'\"\n")
                     
    analysis_path = os.path.join(results_dir, "partA_analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Saved analysis to {analysis_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-override", type=int, default=None)
    args = parser.parse_args()

    results_dir = "archive/iter_031/results"
    runs_dir = os.path.join(results_dir, "runs")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    seeds_to_run = SEEDS
    if args.seed_override is not None:
        seeds_to_run = [args.seed_override]
    elif args.dry_run:
        seeds_to_run = [7, 17]

    # Build queue
    tasks = []
    device_list = ["cuda" if torch.cuda.is_available() else "cpu"]
    for arm in ARMS:
        for seed in seeds_to_run:
            device = device_list[len(tasks) % len(device_list)]
            tasks.append((arm, seed, device, args.dry_run, runs_dir, checkpoints_dir))

    print(f"Launching {len(tasks)} jobs with {args.workers} workers (dry_run={args.dry_run})")
    
    results = []
    if args.workers > 1 and len(tasks) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_single_worker, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    print(f"Error executing worker: {e}")
    else:
        for t in tasks:
            res = _run_single_worker(t)
            results.append(res)

    df_all = pd.DataFrame(results)
    summary_path = os.path.join(results_dir, "summary.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"Saved all results to {summary_path}")

    _generate_analysis(df_all, results_dir)


if __name__ == "__main__":
    main()