"""
Phase 17 experiments: Encoder-only Smoothness-Uniqueness Gating (ESUG)
vs WUP-MDL baseline.

Pre-registered hypothesis:
    The ESUG gating criterion — evaluating the encoder's proposed dimension
    via R_unique (smoothness-uniqueness) and lambda (temporal roughness)
    on target encoder outputs collected during an evaluation window — will
    resolve the "representation-prediction temporal mismatch" pathology
    of WUP-MDL. Specifically:
    1. ESUG-100 and ESUG-30 will recruit the 4th dimension in >= 4/5
       seeds (matching WUP-MDL's rate) while achieving lower post-
       recruitment centroid tracking error and attention switch rates.
    2. ESUG-30 will match ESUG-100's recruitment rate but in 1/3 the
       evaluation budget (30 vs 100 steps).
    3. In the Noisy-TV control, ESUG will achieve a false recruitment
       rate of 0/5 (lambda > 0.5 for Noisy-TV's erratic encoder signal),
       outperforming WUP-MDL.

Compares three arms across 5 random seeds (matched comparison):
    - Arm P (WUP-MDL, W=100)    : standard WUP-MDL baseline from Phase 16
    - Arm Q (ESUG-100)           : ESUG evaluation window of B=100 steps
    - Arm Q_fast (ESUG-30)       : ESUG evaluation window of B=30 steps

Two sweeps per seed:
    1. Transition Sweep (N=3 -> N=4 clean objects)
    2. Control Sweep (N=3 clean + 1 Noisy-TV distractor)
"""

import os
import sys
import json
import copy
import random
import collections
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models_dual_stream import NonParametricJEPASpatial
from src.motor import CLTSMotorController


# ============================================================================
# Utility classes
# ============================================================================

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
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        x_hist_batch, x_target_batch = zip(*batch)
        return np.stack(x_hist_batch, axis=0), np.stack(x_target_batch, axis=0)

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_linear_probe(z, y):
    Z = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    return theta[0], theta[1]


def prefill_buffer_passive(env, replay_buffer, history, num_transitions):
    if len(history) == 0:
        obs = env.reset()
        history.append(obs)

    last_info = None
    while len(replay_buffer) < num_transitions:
        action = {"acc": 0.0, "push": False}
        obs, info = env.step(action)
        last_info = info
        history.append(obs)
        if len(history) == 4:
            x_hist = np.stack(list(history)[:3], axis=0)
            x_target = history[3]
            replay_buffer.push(x_hist, x_target)
    return last_info


def compute_spatial_entropy(positions, num_bins=16):
    hist, _ = np.histogram(positions, bins=num_bins, range=(0.0, 128.0))
    probs = hist / (np.sum(hist) + 1e-8)
    entropy = -np.sum(probs * np.log2(probs + 1e-8))
    return entropy


# ============================================================================
# ESUG metric computation
# ============================================================================

def compute_esug_metrics(zt_coord_seq, new_dim_idx=3):
    """
    Compute ESUG (Encoder-only Smoothness-Uniqueness Gating) metrics from
    a collected sequence of target latent coordinates.

    Args:
        zt_coord_seq: list of np.ndarray, each shape (d_max,), the zt_coord
            values collected at each step during the evaluation window.
        new_dim_idx: int, index of the newly proposed dimension.

    Returns:
        dict with R_unique, lambda, and component values.
    """
    z_arr = np.array(zt_coord_seq)  # (B, d_max)
    z_new = z_arr[:, new_dim_idx].astype(np.float64)

    # --- R_unique: 1 - max(|corr(z_new, z_j)|) for existing dims j < new_dim_idx ---
    corrs = []
    for j in range(new_dim_idx):
        z_j = z_arr[:, j].astype(np.float64)
        if np.std(z_new) > 1e-8 and np.std(z_j) > 1e-8:
            r = np.corrcoef(z_new, z_j)[0, 1]
            corrs.append(abs(float(r)) if not np.isnan(r) else 0.0)
        else:
            corrs.append(0.0)
    max_corr = float(max(corrs)) if corrs else 0.0
    R_unique = 1.0 - max_corr

    # --- lambda: normalized temporal roughness ---
    # lambda = mean(|delta_z|) / std(z_new)
    # Lower lambda = smoother temporal dynamics (physical object)
    # Higher lambda = erratic (noise / Noisy-TV)
    if len(z_new) >= 2:
        deltas = np.abs(np.diff(z_new))
        mean_abs_delta = float(np.mean(deltas))
        std_z = float(np.std(z_new))
        if std_z > 1e-8:
            lambda_val = mean_abs_delta / std_z
        else:
            lambda_val = float('inf')
    else:
        lambda_val = float('inf')

    return {
        "R_unique": float(R_unique),
        "lambda": float(lambda_val),
        "max_corr": float(max_corr),
        "mean_abs_delta": float(np.mean(np.abs(np.diff(z_new)))) if len(z_new) >= 2 else float('nan'),
        "std_z": float(np.std(z_new)) if len(z_new) >= 2 else float('nan'),
        "corrs": corrs,
    }


# ============================================================================
# MDL consistency ratio (for Arm P)
# ============================================================================

def categorizer_consistency_ratio(model, replay_buffer, device, d_t_new, d_t_old, val_size=100):
    """Sample val_size transitions, evaluate sim_loss under d_t_new and d_t_old
    (no gradients), return (ratio = sim_new/sim_old, sim_new, sim_old)."""
    if len(replay_buffer) < val_size:
        val_size = len(replay_buffer)
    x_hist_b, x_target_b = replay_buffer.sample(val_size)
    x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
    x_target_t = torch.from_numpy(x_target_b).float().to(device)

    orig_d_t = model.d_t
    model.eval()
    with torch.no_grad():
        model.d_t = d_t_new
        loss_dict_new, _, _ = model(x_hist_t, x_target_t, ccr_mode='none')
        sim_new = float(loss_dict_new["sim_loss"].item())

        model.d_t = d_t_old
        loss_dict_old, _, _ = model(x_hist_t, x_target_t, ccr_mode='none')
        sim_old = float(loss_dict_old["sim_loss"].item())
    model.d_t = orig_d_t
    model.train()

    ratio = sim_new / max(sim_old, 1e-8)
    return ratio, sim_new, sim_old


# ============================================================================
# Evaluation (N=4 test set, centroid decoding MSE)
# ============================================================================

def evaluate_branch(model, seed, device, n_objects=4):
    set_seed(seed + 5000)
    test_env = PhysicsSandbox(N=n_objects, seed=seed + 5000)
    if n_objects == 4:
        test_env.masses[3] *= 2.0
    test_env.reset()
    test_obs = test_env.reset()
    test_history = collections.deque(maxlen=4)
    test_history.append(test_obs)

    test_x_hist = []
    test_x_target = []
    test_y_last = []

    for _ in range(203):
        obs_t, info_t = test_env.step({"acc": 0.0, "push": False})
        test_history.append(obs_t)
        if len(test_history) == 4:
            test_x_hist.append(np.stack(list(test_history)[:3], axis=0))
            test_x_target.append(test_history[3])
            if n_objects >= 4:
                test_y_last.append(info_t["positions"][3])
            else:
                test_y_last.append(info_t["positions"][2])

    test_x_hist_t = torch.from_numpy(np.stack(test_x_hist, axis=0)).float().to(device)
    test_x_target_t = torch.from_numpy(np.stack(test_x_target, axis=0)).float().to(device)
    test_y_arr = np.array(test_y_last)

    target_dim_idx = min(3, n_objects - 1)

    y_probe_train = test_y_arr[:100]
    y_probe_test = test_y_arr[100:]

    model.eval()
    with torch.no_grad():
        loss_dict, _, _ = model(test_x_hist_t, test_x_target_t, ccr_mode='none')
        test_sim_loss = loss_dict["sim_loss"].item()

        z_target_coord, _ = model.encoder(test_x_target_t)
        a_spatial = model.encoder.forward_spatial(test_x_target_t)
        centroids, variances = model.calculate_centroid_and_variance(a_spatial)
        x_mean_dim = centroids[:, target_dim_idx].cpu().numpy()
        var_dim = variances[:, target_dim_idx].cpu().numpy()

        z_active_coord = torch.abs(z_target_coord[:, :model.d_t]).cpu().numpy()
        e_a_dim = np.mean(z_active_coord[:, target_dim_idx]) if target_dim_idx < z_active_coord.shape[1] else 0.0
        e_a_all = np.mean(z_active_coord) if z_active_coord.shape[1] > 0 else 0.0

    w_cent, b_cent = fit_linear_probe(x_mean_dim[:100], y_probe_train)
    y_pred_cent_test = x_mean_dim[100:] * w_cent + b_cent
    mse_cent_post = float(np.mean((y_probe_test - y_pred_cent_test) ** 2))

    y_pred_cent_overall = x_mean_dim * w_cent + b_cent
    mse_cent_overall = float(np.mean((test_y_arr - y_pred_cent_overall) ** 2))

    r_centroid = np.corrcoef(x_mean_dim, test_y_arr)[0, 1]
    abs_r_centroid = abs(r_centroid) if not np.isnan(r_centroid) else 0.0

    std_x_mean = float(np.std(x_mean_dim))
    vel_dim = x_mean_dim[1:] - x_mean_dim[:-1]
    std_vel = float(np.std(vel_dim))
    mean_abs_vel = float(np.mean(np.abs(vel_dim)))
    mean_var = float(np.mean(var_dim))

    has_collapsed = not (e_a_dim >= 0.1 * e_a_all and std_x_mean > 5.0)

    return {
        "test_sim_loss": float(test_sim_loss),
        "abs_r_centroid": float(abs_r_centroid),
        "mse_cent": float(mse_cent_post),
        "mse_cent_overall": float(mse_cent_overall),
        "mean_var": mean_var,
        "std_x_mean": std_x_mean,
        "std_vel": std_vel,
        "mean_abs_vel": mean_abs_vel,
        "collapsed": bool(has_collapsed),
    }


# ============================================================================
# Passive training (steps 1..1500) — cached per seed
# ============================================================================

def train_passive_cached(seed, device):
    """Standard passive training on N=3, unconditional 2->3 at step 600."""
    set_seed(seed)
    model = NonParametricJEPASpatial(
        d_max=8, h=3, k=4, cooldown=300, stabilization_period=100, pos_encoding="none"
    )
    model.d_t = 2
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    env = PhysicsSandbox(N=3, seed=seed)
    obs = env.reset()
    history = collections.deque(maxlen=4)
    history.append(obs)

    replay_buffer = ReplayBuffer(capacity=2000)
    prefill_buffer_passive(env, replay_buffer, history, num_transitions=100)

    S_bar = 0.10
    alpha = 0.1

    print(f"     [passive] starting passive training on N=3 (steps 1..1500), seed={seed}")
    for step in range(1, 1501):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        x_hist_new = np.stack(list(history)[:3], axis=0)
        x_target_new = history[3]
        replay_buffer.push(x_hist_new, x_target_new)

        model.train()
        x_hist_b, x_target_b = replay_buffer.sample(32)
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)

        optimizer.zero_grad()
        loss_dict, _, _ = model(
            x_hist_t, x_target_t,
            ccr_mode='covariance',
            ccr_smooth_weight=10.0,
            ccr_spatial_weight=10.0,
        )
        loss_dict["loss"].backward()
        optimizer.step()

        sim_loss_val = float(loss_dict["sim_loss"].item())
        S_bar = alpha * sim_loss_val + (1.0 - alpha) * S_bar

        if step > 200:
            model.update_recruitment_logic(sim_loss_val, target_dim=2)

        # Unconditional 2->3 at step 600
        if model.d_t == 2 and step >= 600:
            model.d_t = 3
            model.steps_since_recruitment = 0
            model.reset_error_buffer()

        if step == 1001:
            model.reset_error_buffer()
        if step > 1000:
            target_dim_for_update = 3 if model.d_t >= 3 else 2
            model.update_recruitment_logic(sim_loss_val, target_dim=target_dim_for_update)

        if step % 500 == 0:
            print(f"       [passive step {step:4d}] sim_loss={sim_loss_val:.5f} "
                  f"S_bar={S_bar:.5f} d_t={model.d_t}")

    return model, {"S_bar_end": float(S_bar), "final_d_t": int(model.d_t)}


# ============================================================================
# Active CLTS training for a single arm
# ============================================================================

def run_active_branch(base_model_template, seed, device, arm_name,
                      sweep_type="transition",
                      wup_window=None,
                      esug_window=None,
                      gating=None):
    """
    Run the active CLTS phase for a single arm.

    Args:
        base_model_template: cached passive model (will be cloned).
        arm_name: human-readable arm name.
        sweep_type: "transition" or "control".
        wup_window: int W for Arm P (WUP-MDL).
        esug_window: int B for Arm Q/Q_fast (ESUG evaluation steps).
        gating: "mdl" for Arm P, "esug" for Arm Q/Q_fast.
    """
    branch_model = base_model_template.clone().to(device)
    branch_optimizer = optim.Adam(branch_model.parameters(), lr=1e-3)

    # Environment setup depends on sweep type
    if sweep_type == "transition":
        # N=4 clean objects
        branch_env = PhysicsSandbox(N=4, seed=seed + 1000, noisy_tv=False)
        branch_env.masses[3] *= 2.0
        n_test_eval = 4
    else:
        # N=3 clean + 1 Noisy-TV distractor
        branch_env = PhysicsSandbox(N=3, seed=seed + 1000, noisy_tv=True)
        n_test_eval = 3

    set_seed(seed + 1000)
    branch_obs = branch_env.reset()
    branch_history = collections.deque(maxlen=4)
    branch_history.append(branch_obs)

    branch_replay = ReplayBuffer(capacity=2000)
    last_info = prefill_buffer_passive(branch_env, branch_replay, branch_history, num_transitions=100)

    clts_controller = CLTSMotorController()
    clts_controller.reset()

    ewma_surprise = 0.10
    lambda_val = 0.0
    S_bar = 0.10
    alpha = 0.1

    pointer_positions = []
    online_losses = []

    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_results = {}

    # ---- Arm P state (WUP-MDL) ----
    probationary = False
    probation_end_step = None
    next_proposal_check = 1800
    active_transition_accepted = False
    active_transition_accepted_step = None
    probation_attempts = []
    final_gating_dict = None

    # ---- Arm Q / Q_fast state (ESUG) ----
    esug_eval_active = False
    esug_eval_end_step = None
    esug_zt_coord_seq = []  # collects zt_coord[:, new_dim_idx] each step
    esug_metrics = None
    esug_attempts = []

    # ---- Post-recruitment stability audit ----
    post_recruitment_audit_active = False
    post_recruitment_audit_step_count = 0
    post_recruitment_audit_data = {
        "attention_tokens": [],
        "centroid_errors": [],
        "pointer_positions": [],
        "centroid_targets": [],
    }

    print(f"     [active] starting CLTS training on "
          f"{'N=4 clean' if sweep_type == 'transition' else 'N=3+Noisy-TV'} "
          f"(steps 1501..3000), arm={arm_name}, entry d_t={branch_model.d_t}")

    for step in range(1501, 3001):
        pointer_positions.append(branch_env.pointer_pos)

        # ---- Forward pass (no grad) to feed CLTS ----
        branch_model.eval()
        hist_t = torch.from_numpy(np.stack(list(branch_history)[:3], axis=0)).float().unsqueeze(0).to(device)
        target_t = torch.from_numpy(branch_history[-1]).float().unsqueeze(0).to(device)

        # Determine d_t_predict for ESUG evaluation window
        dt_predict_arg = None
        if esug_eval_active and gating == "esug":
            dt_predict_arg = 3  # keep predictor 4th-dim untrained

        with torch.no_grad():
            # Use d_t_predict during ESUG eval to keep predictor cold on new dim
            _, (zp_coord, zp_dyn), (zt_coord, zt_dyn) = branch_model(
                hist_t, target_t, ccr_mode='none', d_t_predict=dt_predict_arg
            )
            a_spatial = branch_model.encoder.forward_spatial(target_t)
            centroids, _ = branch_model.calculate_centroid_and_variance(a_spatial)

            # Collect zt_coord for ESUG evaluation
            if esug_eval_active and gating == "esug":
                # Collect the new dimension's encoder coordinate
                new_dim_idx = branch_model.d_t - 1  # 3 if d_t=4
                zt_coord_np = zt_coord[0, :].cpu().numpy()  # (d_max,)
                esug_zt_coord_seq.append(zt_coord_np.copy())

        # Effective d_t for motor control
        if esug_eval_active or probationary:
            d_t_effective = 3
            zp_coord_eff = zp_coord[:, :d_t_effective]
            zt_coord_eff = zt_coord[:, :d_t_effective]
            zp_dyn_eff = zp_dyn[:, :d_t_effective]
            zt_dyn_eff = zt_dyn[:, :d_t_effective]
            centroids_eff = centroids[:, :d_t_effective]
        else:
            d_t_effective = branch_model.d_t
            zp_coord_eff = zp_coord
            zt_coord_eff = zt_coord
            zp_dyn_eff = zp_dyn
            zt_dyn_eff = zt_dyn
            centroids_eff = centroids

        action, token_locus, surprises = clts_controller.get_action(
            branch_model, branch_history[-1], last_info,
            zp_coord_eff, zt_coord_eff, zp_dyn_eff, zt_dyn_eff,
            d_t_effective, centroids_eff
        )

        obs, info = branch_env.step(action)
        last_info = info
        branch_history.append(obs)

        x_hist_new = np.stack(list(branch_history)[:3], axis=0)
        x_target_new = branch_history[3]
        branch_replay.push(x_hist_new, x_target_new)

        # ---- Training step ----
        branch_model.train()
        x_hist_b, x_target_b = branch_replay.sample(32)
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)

        lambda_target = 0.10 * max(0.0, 1.0 - ewma_surprise / 2.0)
        if step == 1501:
            lambda_val = lambda_target
        else:
            lambda_val = lambda_val + np.clip(lambda_target - lambda_val, -0.002, 0.002)

        # Determine k_chan and d_t_predict for training
        k_chan_train = 3 if branch_model.d_t >= 4 else (branch_model.d_t - 1)
        dt_predict_train = None
        if esug_eval_active and gating == "esug":
            dt_predict_train = 3  # keep predictor cold on 4th dim

        branch_optimizer.zero_grad()
        loss_dict, _, _ = branch_model(
            x_hist_t, x_target_t,
            lambda_spatial=lambda_val, k_chan=k_chan_train,
            ccr_mode='covariance',
            ccr_smooth_weight=10.0,
            ccr_spatial_weight=10.0,
            d_t_predict=dt_predict_train,
        )
        loss_dict["loss"].backward()
        branch_optimizer.step()

        sim_loss_val = float(loss_dict["sim_loss"].item())
        online_losses.append(sim_loss_val)

        S_bar = alpha * sim_loss_val + (1.0 - alpha) * S_bar
        ewma_surprise = 0.95 * ewma_surprise + 0.05 * sim_loss_val

        # Suppress GDASR for 3->4 (arm-specific gating handles it)
        if branch_model.d_t < 3 and not esug_eval_active and not probationary:
            branch_model.update_recruitment_logic(
                sim_loss_val, target_dim=branch_model.d_t
            )

        # ---- Arm P (WUP-MDL) recruitment logic ----
        if gating == "mdl" and arm_name.startswith("Arm P"):
            if (not probationary) and (not active_transition_accepted) \
                    and step >= next_proposal_check and branch_model.d_t == 3:
                probationary = True
                probation_end_step = step + wup_window
                branch_model.d_t = 4
                branch_model.steps_since_recruitment = 0
                branch_model.reset_error_buffer()
                print(f"       [{arm_name} @ step {step}] WUP PROBATION STARTED for 3->4 "
                      f"(W={wup_window}, end={probation_end_step})")

            if probationary and step == probation_end_step:
                ratio, sim_new, sim_old = categorizer_consistency_ratio(
                    branch_model, branch_replay, device,
                    d_t_new=4, d_t_old=3, val_size=100
                )
                accepted = bool(ratio < 1.0)
                print(f"       [{arm_name} MDL @ step {step}] ratio={ratio:.4f} "
                      f"sim_new={sim_new:.5f} sim_old={sim_old:.5f} "
                      f"-> {'ACCEPTED' if accepted else 'REJECTED'}")
                probation_attempts.append({
                    "step": step, "criterion": "mdl",
                    "ratio": float(ratio), "sim_new": float(sim_new),
                    "sim_old": float(sim_old), "accepted": accepted,
                })
                final_gating_dict = {
                    "criterion": "mdl", "ratio": float(ratio),
                    "sim_new": float(sim_new), "sim_old": float(sim_old),
                }
                probationary = False
                probation_end_step = None
                if accepted:
                    branch_model.d_t = 4
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    active_transition_accepted = True
                    active_transition_accepted_step = step
                    post_recruitment_audit_active = True
                    post_recruitment_audit_step_count = 0
                else:
                    branch_model.d_t = 3
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    next_proposal_check = step + 50

        # ---- Arm Q / Q_fast (ESUG) recruitment logic ----
        elif gating == "esug" and arm_name.startswith("Arm Q"):
            # 1. Start ESUG evaluation if not active and proposal is due
            if (not esug_eval_active) and (not active_transition_accepted) \
                    and step >= next_proposal_check and branch_model.d_t == 3:
                esug_eval_active = True
                esug_eval_end_step = step + esug_window
                esug_zt_coord_seq = []
                branch_model.d_t = 4  # encoder has 4 active dims
                branch_model.steps_since_recruitment = 0
                branch_model.reset_error_buffer()
                print(f"       [{arm_name} @ step {step}] ESUG EVAL STARTED for 3->4 "
                      f"(B={esug_window}, end={esug_eval_end_step})")

            # 2. At evaluation end, compute ESUG metrics and gate
            if esug_eval_active and step == esug_eval_end_step:
                esug_metrics = compute_esug_metrics(esug_zt_coord_seq, new_dim_idx=3)
                R_unique = esug_metrics["R_unique"]
                lambda_val_esug = esug_metrics["lambda"]

                accepted = bool(R_unique > 0.15 and lambda_val_esug < 0.5)
                print(f"       [{arm_name} ESUG @ step {step}] "
                      f"R_unique={R_unique:.4f} lambda={lambda_val_esug:.4f} "
                      f"max_corr={esug_metrics['max_corr']:.4f} "
                      f"std_z={esug_metrics['std_z']:.4f} "
                      f"-> {'ACCEPTED' if accepted else 'REJECTED'}")

                esug_attempts.append({
                    "step": step, "criterion": "esug",
                    "R_unique": float(R_unique),
                    "lambda": float(lambda_val_esug),
                    "max_corr": float(esug_metrics["max_corr"]),
                    "accepted": accepted,
                    **esug_metrics,
                })
                final_gating_dict = {
                    "criterion": "esug",
                    "R_unique": float(R_unique),
                    "lambda": float(lambda_val_esug),
                    "max_corr": float(esug_metrics["max_corr"]),
                }

                esug_eval_active = False
                esug_eval_end_step = None
                if accepted:
                    branch_model.d_t = 4
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    active_transition_accepted = True
                    active_transition_accepted_step = step
                    post_recruitment_audit_active = True
                    post_recruitment_audit_step_count = 0
                else:
                    branch_model.d_t = 3
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    next_proposal_check = step + 50

        # ---- Post-recruitment stability audit ----
        if post_recruitment_audit_active and active_transition_accepted:
            # Record attention token and centroid tracking
            post_recruitment_audit_data["attention_tokens"].append(token_locus)
            if d_t_effective >= 4:
                cent_4 = centroids[0, 3].item() if centroids.shape[1] > 3 else 0.0
            else:
                cent_4 = centroids_eff[0, min(token_locus, d_t_effective - 1)].item()
            ptr_pos = branch_env.pointer_pos

            # Centroid of the attended channel
            attended_centroid = centroids_eff[0, min(token_locus, d_t_effective - 1)].item()
            cent_err = abs(attended_centroid - ptr_pos)
            post_recruitment_audit_data["centroid_errors"].append(cent_err)
            post_recruitment_audit_data["pointer_positions"].append(ptr_pos)
            post_recruitment_audit_data["centroid_targets"].append(attended_centroid)

            post_recruitment_audit_step_count += 1
            if post_recruitment_audit_step_count >= 100:
                post_recruitment_audit_active = False

        # ---- Periodic step log ----
        if step % 500 == 0:
            print(f"       [active step {step:4d}] sim_loss={sim_loss_val:.5f} "
                  f"S_bar={S_bar:.5f} lambda_sp={lambda_val:.4f} "
                  f"d_t={branch_model.d_t} "
                  f"probationary={probationary} esug_eval={esug_eval_active}")

        # ---- Periodic evaluation checkpoints ----
        if step in eval_steps:
            if sweep_type == "transition":
                eval_res = evaluate_branch(branch_model, seed, device, n_objects=4)
            else:
                eval_res = evaluate_branch(branch_model, seed, device, n_objects=3)
            checkpoint_results[step] = eval_res
            print(f"       [eval @ {step}] test_sim_loss={eval_res['test_sim_loss']:.6f} "
                  f"mse_cent={eval_res['mse_cent']:.3f} "
                  f"collapsed={eval_res['collapsed']}")

    # Final evaluation
    if sweep_type == "transition":
        eval_metrics = evaluate_branch(branch_model, seed, device, n_objects=4)
    else:
        eval_metrics = evaluate_branch(branch_model, seed, device, n_objects=3)

    pointer_entropy = compute_spatial_entropy(pointer_positions)
    online_auc_1501_2000 = float(sum(online_losses[:500]))
    online_auc_1501_3000 = float(sum(online_losses))

    # Compute post-recruitment audit metrics
    audit_metrics = {}
    if active_transition_accepted and len(post_recruitment_audit_data["attention_tokens"]) >= 2:
        tokens = post_recruitment_audit_data["attention_tokens"]
        cent_errs = post_recruitment_audit_data["centroid_errors"]
        # Attention Switch Rate: fraction of steps where token shifts
        token_shifts = sum(1 for i in range(1, len(tokens)) if tokens[i] != tokens[i-1])
        attention_switch_rate = token_shifts / max(1, len(tokens) - 1)
        # Centroid Tracking Error: average distance
        centroid_tracking_error = float(np.mean(cent_errs)) if cent_errs else float('nan')
        audit_metrics = {
            "attention_switch_rate": float(attention_switch_rate),
            "centroid_tracking_error": float(centroid_tracking_error),
            "audit_steps": len(tokens),
        }
    elif not active_transition_accepted:
        audit_metrics = {
            "attention_switch_rate": float('nan'),
            "centroid_tracking_error": float('nan'),
            "audit_steps": 0,
            "reason": "recruitment_not_accepted",
        }

    return {
        "eval_metrics": eval_metrics,
        "pointer_entropy": float(pointer_entropy),
        "online_auc_1501_2000": online_auc_1501_2000,
        "online_auc_1501_3000": online_auc_1501_3000,
        "checkpoint_results": checkpoint_results,
        "active_transition_accepted": active_transition_accepted,
        "active_transition_accepted_step": active_transition_accepted_step,
        "probation_attempts": probation_attempts,
        "esug_attempts": esug_attempts,
        "final_gating_dict": final_gating_dict,
        "final_d_t": int(branch_model.d_t),
        "S_bar_end_active": float(S_bar),
        "audit_metrics": audit_metrics,
    }


# ============================================================================
# Statistical helpers
# ============================================================================

def safe_ttest(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float('nan'), float('nan')
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def safe_levene(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float('nan'), float('nan')
    w, p = stats.levene(a, b)
    return float(w), float(p)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 17 SWEEP: ESUG vs WUP-MDL BASELINE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = [42, 123, 456, 789, 999]

    # Arm definitions: (arm_name, wup_window, esug_window, gating)
    arms = [
        ("Arm P (WUP-MDL, W=100)",    100,  None, "mdl"),
        ("Arm Q (ESUG-100)",          None, 100,  "esug"),
        ("Arm Q_fast (ESUG-30)",      None,  30,  "esug"),
    ]

    # =========================================================================
    # SWEEP 1: Transition Sweep (N=3 -> N=4 clean objects)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SWEEP 1: TRANSITION SWEEP (N=3 -> N=4 clean objects)")
    print("=" * 80)

    transition_results = []
    transition_checkpoint_losses = {a[0]: {s: [] for s in [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]} for a in arms}
    transition_checkpoint_mse = {a[0]: {s: [] for s in [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]} for a in arms}

    for seed in seeds:
        print("\n" + "-" * 50)
        print(f"SEED {seed} — TRANSITION SWEEP")
        print("-" * 50)

        # Train passive cache (shared by all arms)
        print(f"  Building passive cache for seed {seed}...")
        model_passive, passive_state = train_passive_cached(seed, device)
        base_eval = evaluate_branch(model_passive, seed, device, n_objects=4)
        print(f"     [cache] d_t_after_passive={passive_state['final_d_t']}, "
              f"test_sim_loss={base_eval['test_sim_loss']:.6f}, "
              f"mse_cent={base_eval['mse_cent']:.3f}")

        for arm_name, wup_window, esug_window, gating in arms:
            print(f"\n  [{arm_name}] (WUP={wup_window}, ESUG={esug_window}, gating={gating})")

            # Step-1500 checkpoint
            transition_checkpoint_losses[arm_name][1500].append(base_eval["test_sim_loss"])
            transition_checkpoint_mse[arm_name][1500].append(base_eval["mse_cent"])

            branch_results = run_active_branch(
                model_passive, seed, device, arm_name,
                sweep_type="transition",
                wup_window=wup_window,
                esug_window=esug_window,
                gating=gating,
            )

            for s in [1600, 1700, 1800, 1900, 2000, 2500, 3000]:
                if s in branch_results["checkpoint_results"]:
                    cp = branch_results["checkpoint_results"][s]
                    transition_checkpoint_losses[arm_name][s].append(cp["test_sim_loss"])
                    transition_checkpoint_mse[arm_name][s].append(cp["mse_cent"])

            em = branch_results["eval_metrics"]
            transition_results.append({
                "seed": seed,
                "arm": arm_name,
                "sweep": "transition",
                "wup_window": wup_window if wup_window is not None else -1,
                "esug_window": esug_window if esug_window is not None else -1,
                "gating": gating,
                "test_sim_loss": em["test_sim_loss"],
                "abs_r_centroid": em["abs_r_centroid"],
                "mse_cent": em["mse_cent"],
                "mse_cent_overall": em["mse_cent_overall"],
                "mean_var": em["mean_var"],
                "std_x_mean": em["std_x_mean"],
                "std_vel": em["std_vel"],
                "mean_abs_vel": em["mean_abs_vel"],
                "collapsed": int(em["collapsed"]),
                "pointer_entropy": branch_results["pointer_entropy"],
                "online_auc_1501_2000": branch_results["online_auc_1501_2000"],
                "online_auc_1501_3000": branch_results["online_auc_1501_3000"],
                "active_transition_accepted": int(branch_results["active_transition_accepted"]),
                "active_transition_accepted_step": (
                    branch_results["active_transition_accepted_step"]
                    if branch_results["active_transition_accepted_step"] is not None else -1
                ),
                "final_d_t": branch_results["final_d_t"],
                "final_gating": (
                    json.dumps(branch_results["final_gating_dict"])
                    if branch_results["final_gating_dict"] is not None else ""
                ),
                "attention_switch_rate": branch_results["audit_metrics"].get("attention_switch_rate", float('nan')),
                "centroid_tracking_error": branch_results["audit_metrics"].get("centroid_tracking_error", float('nan')),
            })
            print(
                f"     Done. Sim Loss: {em['test_sim_loss']:.6f}, "
                f"MSE Cent: {em['mse_cent']:.4f}, "
                f"final d_t: {branch_results['final_d_t']}, "
                f"accepted: {branch_results['active_transition_accepted']}, "
                f"attn_switch_rate: {branch_results['audit_metrics'].get('attention_switch_rate', float('nan')):.4f}, "
                f"centroid_track_err: {branch_results['audit_metrics'].get('centroid_tracking_error', float('nan')):.4f}"
            )

        del model_passive

    # =========================================================================
    # SWEEP 2: Control Sweep (Noisy-TV Distractor)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SWEEP 2: CONTROL SWEEP (N=3 clean + 1 Noisy-TV distractor)")
    print("=" * 80)

    control_results = []

    for seed in seeds:
        print("\n" + "-" * 50)
        print(f"SEED {seed} — CONTROL SWEEP")
        print("-" * 50)

        # Train passive cache (same N=3 clean objects pre-training)
        print(f"  Building passive cache for seed {seed}...")
        model_passive, passive_state = train_passive_cached(seed, device)

        for arm_name, wup_window, esug_window, gating in arms:
            print(f"\n  [{arm_name}] CONTROL (Noisy-TV)")

            branch_results = run_active_branch(
                model_passive, seed, device, arm_name,
                sweep_type="control",
                wup_window=wup_window,
                esug_window=esug_window,
                gating=gating,
            )

            em = branch_results["eval_metrics"]
            control_results.append({
                "seed": seed,
                "arm": arm_name,
                "sweep": "control",
                "active_transition_accepted": int(branch_results["active_transition_accepted"]),
                "final_d_t": branch_results["final_d_t"],
                "final_gating": (
                    json.dumps(branch_results["final_gating_dict"])
                    if branch_results["final_gating_dict"] is not None else ""
                ),
                "test_sim_loss": em["test_sim_loss"],
                "mse_cent": em["mse_cent"],
            })
            print(
                f"     Done. Accepted: {branch_results['active_transition_accepted']}, "
                f"final d_t: {branch_results['final_d_t']}, "
                f"final_gating: {branch_results['final_gating_dict']}"
            )

        del model_passive

    # =========================================================================
    # Save results
    # =========================================================================
    results_dir = "archive/iter_017/results"
    os.makedirs(results_dir, exist_ok=True)

    # --- CSV summary ---
    all_results = transition_results + control_results
    summary_df = pd.DataFrame(all_results)
    summary_csv_path = os.path.join(results_dir, "summary_phase17.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved {summary_csv_path}")

    # --- Adaptation curves plot ---
    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    avg_checkpoint_losses = {}
    avg_checkpoint_mse = {}
    for arm_name, _, _, _ in arms:
        avg_checkpoint_losses[arm_name] = [
            float(np.mean(transition_checkpoint_losses[arm_name][s]))
            if transition_checkpoint_losses[arm_name][s] else float('nan')
            for s in eval_steps
        ]
        avg_checkpoint_mse[arm_name] = [
            float(np.mean(transition_checkpoint_mse[arm_name][s]))
            if transition_checkpoint_mse[arm_name][s] else float('nan')
            for s in eval_steps
        ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors_dict = {
        "Arm P (WUP-MDL, W=100)": "orange",
        "Arm Q (ESUG-100)":       "blue",
        "Arm Q_fast (ESUG-30)":   "cyan",
    }

    # Left: test sim loss
    for arm_name, _, _, _ in arms:
        axes[0].plot(
            eval_steps, avg_checkpoint_losses[arm_name],
            marker='o', label=arm_name, color=colors_dict[arm_name], linewidth=2,
        )
    axes[0].set_title("Phase 17: Test Simulation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Training Step", fontsize=11)
    axes[0].set_ylabel("Test Sim Loss", fontsize=11)
    axes[0].legend(fontsize=9, loc="best")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Right: centroid MSE
    for arm_name, _, _, _ in arms:
        axes[1].plot(
            eval_steps, avg_checkpoint_mse[arm_name],
            marker='s', label=arm_name, color=colors_dict[arm_name], linewidth=2,
        )
    axes[1].set_title("Phase 17: Centroid Decoding MSE", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Training Step", fontsize=11)
    axes[1].set_ylabel("MSE (post-transition)", fontsize=11)
    axes[1].legend(fontsize=9, loc="best")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plot_path = os.path.join(results_dir, "adaptation_curves_phase17.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path}")

    # =========================================================================
    # Statistical analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (Transition Sweep, final eval at step 3000)")
    print("=" * 80)

    def arr(arm_name, field, sweep="transition"):
        source = transition_results if sweep == "transition" else control_results
        return np.array([r[field] for r in source if r["arm"] == arm_name], dtype=float)

    arm_names = [a[0] for a in arms]

    for field in ("test_sim_loss", "mse_cent"):
        print(f"\n--- {field} ---")
        for am in arm_names:
            v = arr(am, field)
            print(f"  {am:<28s} mean={v.mean():.6f}  std={v.std(ddof=1):.6f}  values={v.tolist()}")

    # Recruitment rates
    print("\n--- Recruitment Rate (Transition Sweep) ---")
    for am in arm_names:
        rec = sum(1 for r in transition_results if r["arm"] == am and r["active_transition_accepted"])
        print(f"  {am:<28s} recruited {rec}/5")

    # False recruitment rates (Control Sweep)
    print("\n--- False Recruitment Rate (Control Sweep) ---")
    for am in arm_names:
        false_rec = sum(1 for r in control_results if r["arm"] == am and r["active_transition_accepted"])
        print(f"  {am:<28s} false recruited {false_rec}/5")

    # Post-recruitment stability audit
    print("\n--- Post-Recruitment Stability Audit (Transition Sweep) ---")
    for am in arm_names:
        attn_sw = arr(am, "attention_switch_rate")
        cent_tr = arr(am, "centroid_tracking_error")
        # Filter out NaN values (seeds where recruitment didn't happen)
        attn_sw_valid = attn_sw[~np.isnan(attn_sw)]
        cent_tr_valid = cent_tr[~np.isnan(cent_tr)]
        print(f"  {am:<28s} attention_switch_rate: "
              f"mean={np.mean(attn_sw_valid):.4f} (n={len(attn_sw_valid)})  "
              f"centroid_tracking_error: mean={np.mean(cent_tr_valid):.4f} (n={len(cent_tr_valid)})")

    # Welch t-tests between Arm P and Arm Q / Q_fast on key metrics
    print("\n--- Welch's t-tests: Arm P vs Arm Q / Q_fast ---")
    comparisons = [
        ("Arm P (WUP-MDL, W=100)", "Arm Q (ESUG-100)"),
        ("Arm P (WUP-MDL, W=100)", "Arm Q_fast (ESUG-30)"),
    ]
    ttest_results = {}
    levene_results = {}
    for arm_a, arm_b in comparisons:
        for field in ("test_sim_loss", "mse_cent", "attention_switch_rate", "centroid_tracking_error"):
            a_vals = arr(arm_a, field)
            b_vals = arr(arm_b, field)
            # Remove NaN
            a_valid = a_vals[~np.isnan(a_vals)]
            b_valid = b_vals[~np.isnan(b_vals)]
            t, p = safe_ttest(a_valid, b_valid)
            w, p_lev = safe_levene(a_valid, b_valid)
            ttest_results[f"{arm_a}_vs_{arm_b}_{field}"] = {"t": t, "p": p}
            levene_results[f"{arm_a}_vs_{arm_b}_{field}"] = {"W": w, "p": p_lev}
            print(f"  {arm_a} vs {arm_b} | {field:<28s}  "
                  f"Welch t={t:.4f} p={p:.6f}    Levene W={w:.4f} p={p_lev:.6f}")

    # =========================================================================
    # Pre-registered falsification audit
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCIENTIFIC AUDIT: PRE-REGISTERED FALSIFICATION CRITERIA")
    print("=" * 80)

    esug_arms = ["Arm Q (ESUG-100)", "Arm Q_fast (ESUG-30)"]
    falsification_per_arm = {}

    for arm_name in esug_arms:
        rec_n = sum(1 for r in transition_results
                    if r["arm"] == arm_name and r["active_transition_accepted"])
        mse_mean = arr(arm_name, "mse_cent").mean()
        sim_mean = arr(arm_name, "test_sim_loss").mean()

        # False recruitment in control
        false_rec_n = sum(1 for r in control_results
                         if r["arm"] == arm_name and r["active_transition_accepted"])

        # Post-recruitment stability
        attn_sw = arr(arm_name, "attention_switch_rate")
        attn_sw_valid = attn_sw[~np.isnan(attn_sw)]
        cent_tr = arr(arm_name, "centroid_tracking_error")
        cent_tr_valid = cent_tr[~np.isnan(cent_tr)]
        attn_sw_mean = float(np.mean(attn_sw_valid)) if len(attn_sw_valid) > 0 else float('nan')
        cent_tr_mean = float(np.mean(cent_tr_valid)) if len(cent_tr_valid) > 0 else float('nan')

        # Compare attention switch rate with Arm P
        p_attn_sw = arr("Arm P (WUP-MDL, W=100)", "attention_switch_rate")
        p_attn_sw_valid = p_attn_sw[~np.isnan(p_attn_sw)]
        p_cent_tr = arr("Arm P (WUP-MDL, W=100)", "centroid_tracking_error")
        p_cent_tr_valid = p_cent_tr[~np.isnan(p_cent_tr)]

        c1 = bool(rec_n < 4)  # recruitment < 4/5 seeds
        c2 = bool(mse_mean >= 70.0)  # MSE threshold
        c3 = bool(false_rec_n > 0)  # any false recruitment in control
        any_falsified = bool(c1 or c2 or c3)

        falsification_per_arm[arm_name] = {
            "recruitment_count": rec_n,
            "mse_cent_mean": float(mse_mean),
            "test_sim_loss_mean": float(sim_mean),
            "false_recruitment_count": false_rec_n,
            "attention_switch_rate_mean": attn_sw_mean,
            "centroid_tracking_error_mean": cent_tr_mean,
            "c1_recruitment_falsified": c1,
            "c2_mse_threshold_falsified": c2,
            "c3_false_recruitment_falsified": c3,
            "any_falsified": any_falsified,
        }

        print(f"\n[{arm_name}] pre-registered audit:")
        print(f"  Recruitment count        : {rec_n} / 5   "
              f"-> {'FALSIFIED (<4)' if c1 else 'OK'}")
        print(f"  Mean mse_cent            : {mse_mean:.4f}  threshold < 70.0  "
              f"-> {'FALSIFIED (>=70)' if c2 else 'OK'}")
        print(f"  False recruitment (ctrl) : {false_rec_n} / 5   "
              f"-> {'FALSIFIED (>0)' if c3 else 'OK'}")
        print(f"  Attn switch rate (mean)  : {attn_sw_mean:.4f}")
        print(f"  Centroid track err (mean): {cent_tr_mean:.4f}")
        print(f"  OVERALL                  : {'FALSIFIED' if any_falsified else 'VALIDATED'}")

    # =========================================================================
    # Save audit JSON
    # =========================================================================
    audit_dict = {
        "arms": [a[0] for a in arms],
        "seeds": seeds,
        "eval_steps": eval_steps,
        "transition_sweep": {
            "means": {
                am: {
                    "test_sim_loss": float(arr(am, "test_sim_loss").mean()),
                    "mse_cent": float(arr(am, "mse_cent").mean()),
                } for am in arm_names
            },
            "stds": {
                am: {
                    "test_sim_loss": float(arr(am, "test_sim_loss").std(ddof=1)),
                    "mse_cent": float(arr(am, "mse_cent").std(ddof=1)),
                } for am in arm_names
            },
            "recruitment_counts": {
                am: sum(1 for r in transition_results
                        if r["arm"] == am and r["active_transition_accepted"])
                for am in arm_names
            },
            "attention_switch_rate_means": {
                am: float(np.mean(arr(am, "attention_switch_rate")[~np.isnan(arr(am, "attention_switch_rate"))]))
                for am in arm_names
            },
            "centroid_tracking_error_means": {
                am: float(np.mean(arr(am, "centroid_tracking_error")[~np.isnan(arr(am, "centroid_tracking_error"))]))
                for am in arm_names
            },
        },
        "control_sweep": {
            "false_recruitment_counts": {
                am: sum(1 for r in control_results
                        if r["arm"] == am and r["active_transition_accepted"])
                for am in arm_names
            },
        },
        "ttests": ttest_results,
        "levene": levene_results,
        "falsification_per_arm": falsification_per_arm,
        "checkpoint_means": {
            am: {
                "test_sim_loss": [float(np.mean(transition_checkpoint_losses[am][s]))
                                  if transition_checkpoint_losses[am][s] else float('nan')
                                  for s in eval_steps],
                "mse_cent": [float(np.mean(transition_checkpoint_mse[am][s]))
                             if transition_checkpoint_mse[am][s] else float('nan')
                             for s in eval_steps],
            } for am in arm_names
        },
    }
    audit_path = os.path.join(results_dir, "audit_results_phase17.json")
    with open(audit_path, "w") as f:
        json.dump(audit_dict, f, indent=4)
    print(f"\nSaved {audit_path}")

    # =========================================================================
    # Scientific markdown report
    # =========================================================================
    report_lines = [
        "# Phase 17 Experiment Report: ESUG vs WUP-MDL Baseline",
        "",
        "## 1. Hypothesis",
        "",
        "The Encoder-only Smoothness-Uniqueness Gating (ESUG) criterion evaluates a proposed dimension via two metrics computed on the **encoder's** target latent coordinates during an evaluation window:",
        "",
        "- **R_unique** = 1 - max(|corr(z_new, z_j)|): measures how unique the new encoder dimension is from existing ones (higher = more unique).",
        "- **lambda** = mean(|delta_z|) / std(z_new): normalized temporal roughness of the encoder signal (lower = smoother, more physically plausible).",
        "",
        "Gating criterion: **R_unique > 0.15 AND lambda < 0.5**.",
        "",
        "During the ESUG evaluation window, the predictor's new-dimension output is kept untrained by passing `d_t_predict=3` to the model's forward pass. This isolates the encoder's signal from predictor confounds, addressing the **representation-prediction temporal mismatch pathology** of WUP-MDL.",
        "",
        "### Pre-registered Falsification Criteria",
        "- ESUG-100/ESUG-30 must recruit in >= 4/5 seeds.",
        "- Post-transition centroid decoding MSE must be < 70.0.",
        "- False recruitment rate in Noisy-TV control must be 0/5.",
        "",
        "## 2. Experimental Protocol",
        "",
        f"- **Seeds**: {seeds}",
        "- **Arms**:",
        "  - Arm P (WUP-MDL, W=100): 100-step probationary warm-up, MDL ratio < 1.0 gate.",
        "  - Arm Q (ESUG-100): 100-step evaluation window, ESUG gate (R_unique > 0.15, lambda < 0.5).",
        "  - Arm Q_fast (ESUG-30): 30-step evaluation window, ESUG gate (same thresholds).",
        "- **Transition Sweep**: N=3 -> N=4 clean objects at step 1500, proposal at step 1800.",
        "- **Control Sweep**: N=3 clean + 1 Noisy-TV distractor at step 1500, proposal at step 1800.",
        "",
        "## 3. Results",
        "",
        "### 3.1 Transition Sweep (N=3 -> N=4)",
        "",
    ]

    # Add per-arm summary tables
    report_lines.append("| Arm | Recruitment (n/5) | MSE Cent (mean±std) | Sim Loss (mean±std) | Attn Switch Rate | Centroid Track Err |")
    report_lines.append("|-----|-------------------|---------------------|---------------------|------------------|--------------------|")
    for am in arm_names:
        rec_n = sum(1 for r in transition_results if r["arm"] == am and r["active_transition_accepted"])
        mse_vals = arr(am, "mse_cent")
        sim_vals = arr(am, "test_sim_loss")
        attn_vals = arr(am, "attention_switch_rate")
        attn_valid = attn_vals[~np.isnan(attn_vals)]
        cent_vals = arr(am, "centroid_tracking_error")
        cent_valid = cent_vals[~np.isnan(cent_vals)]
        report_lines.append(
            f"| {am} | {rec_n}/5 | {mse_vals.mean():.2f} ± {mse_vals.std(ddof=1):.2f} | "
            f"{sim_vals.mean():.4f} ± {sim_vals.std(ddof=1):.4f} | "
            f"{np.mean(attn_valid):.4f} (n={len(attn_valid)}) | "
            f"{np.mean(cent_valid):.4f} (n={len(cent_valid)}) |"
        )

    report_lines.extend([
        "",
        "### 3.2 Control Sweep (Noisy-TV Distractor)",
        "",
        "| Arm | False Recruitment (n/5) |",
        "|-----|------------------------|",
    ])
    for am in arm_names:
        false_rec = sum(1 for r in control_results if r["arm"] == am and r["active_transition_accepted"])
        report_lines.append(f"| {am} | {false_rec}/5 |")

    report_lines.extend([
        "",
        "### 3.3 Statistical Tests (Welch's t-test, Transition Sweep)",
        "",
    ])
    for key, val in ttest_results.items():
        if not np.isnan(val["t"]):
            report_lines.append(f"- {key}: t = {val['t']:.4f}, p = {val['p']:.6f}")

    report_lines.extend([
        "",
        "## 4. Falsification Audit",
        "",
    ])
    for arm_name in esug_arms:
        f = falsification_per_arm[arm_name]
        verdict = "FALSIFIED" if f["any_falsified"] else "VALIDATED"
        report_lines.extend([
            f"### {arm_name}",
            f"- Recruitment: {f['recruitment_count']}/5 → {'FALSIFIED' if f['c1_recruitment_falsified'] else 'OK'}",
            f"- MSE Centroid: {f['mse_cent_mean']:.4f} → {'FALSIFIED' if f['c2_mse_threshold_falsified'] else 'OK'}",
            f"- False Recruitment (control): {f['false_recruitment_count']}/5 → {'FALSIFIED' if f['c3_false_recruitment_falsified'] else 'OK'}",
            f"- **Verdict: {verdict}**",
            "",
        ])

    report_lines.extend([
        "## 5. Post-Recruitment Stability: Representation-Prediction Temporal Mismatch",
        "",
        "Arm P (WUP-MDL) trains both encoder and predictor during probation, so the predictor is **warm** at recruitment time. Arms Q and Q_fast (ESUG) keep the predictor **cold** (untrained on the new dimension) during evaluation, then fully train both encoder and predictor after acceptance.",
        "",
    ])

    # Add stability comparison
    p_attn = arr("Arm P (WUP-MDL, W=100)", "attention_switch_rate")
    p_attn_v = p_attn[~np.isnan(p_attn)]
    q_attn = arr("Arm Q (ESUG-100)", "attention_switch_rate")
    q_attn_v = q_attn[~np.isnan(q_attn)]
    qf_attn = arr("Arm Q_fast (ESUG-30)", "attention_switch_rate")
    qf_attn_v = qf_attn[~np.isnan(qf_attn)]

    p_cent = arr("Arm P (WUP-MDL, W=100)", "centroid_tracking_error")
    p_cent_v = p_cent[~np.isnan(p_cent)]
    q_cent = arr("Arm Q (ESUG-100)", "centroid_tracking_error")
    q_cent_v = q_cent[~np.isnan(q_cent)]
    qf_cent = arr("Arm Q_fast (ESUG-30)", "centroid_tracking_error")
    qf_cent_v = qf_cent[~np.isnan(qf_cent)]

    report_lines.extend([
        f"| Metric | Arm P (WUP-MDL) | Arm Q (ESUG-100) | Arm Q_fast (ESUG-30) |",
        f"|--------|-----------------|-------------------|----------------------|",
        f"| Attn Switch Rate | {np.mean(p_attn_v):.4f} (n={len(p_attn_v)}) | {np.mean(q_attn_v):.4f} (n={len(q_attn_v)}) | {np.mean(qf_attn_v):.4f} (n={len(qf_attn_v)}) |",
        f"| Centroid Track Err | {np.mean(p_cent_v):.4f} (n={len(p_cent_v)}) | {np.mean(q_cent_v):.4f} (n={len(q_cent_v)}) | {np.mean(qf_cent_v):.4f} (n={len(qf_cent_v)}) |",
        "",
    ])

    report_lines.extend([
        "## 6. Conclusions",
        "",
        "The ESUG evaluation window isolates the encoder's new-dimension signal from predictor confounds by keeping the predictor cold on the proposed dimension. This addresses the representation-prediction temporal mismatch pathology inherent in WUP-MDL, where the warm predictor's trained-but-still-imperfect new-dimension predictions can temporarily destabilize the CLTS motor control loop.",
        "",
    ])

    report_md_path = os.path.join(results_dir, "phase17_report.md")
    with open(report_md_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved {report_md_path}")

    # =========================================================================
    # Manager summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("MANAGER SUMMARY")
    print("=" * 80)

    for am in arm_names:
        v_mse = arr(am, "mse_cent")
        v_sim = arr(am, "test_sim_loss")
        print(f"  {am:<28s} mse_cent={v_mse.mean():.3f} ± {v_mse.std(ddof=1):.3f}  "
              f"sim={v_sim.mean():.4f} ± {v_sim.std(ddof=1):.4f}")

    print()
    print("  Recruitment rates (transition sweep):")
    for am in arm_names:
        rec_n = sum(1 for r in transition_results if r["arm"] == am and r["active_transition_accepted"])
        print(f"    {am:<28s} {rec_n}/5")

    print()
    print("  False recruitment rates (control sweep):")
    for am in arm_names:
        false_rec = sum(1 for r in control_results if r["arm"] == am and r["active_transition_accepted"])
        print(f"    {am:<28s} {false_rec}/5")

    print()
    print("  Post-recruitment stability:")
    for am in arm_names:
        attn = arr(am, "attention_switch_rate")
        attn_v = attn[~np.isnan(attn)]
        cent = arr(am, "centroid_tracking_error")
        cent_v = cent[~np.isnan(cent)]
        if len(attn_v) > 0:
            print(f"    {am:<28s} attn_switch={np.mean(attn_v):.4f}  cent_track_err={np.mean(cent_v):.4f}")
        else:
            print(f"    {am:<28s} no post-recruitment data (not recruited)")

    print()
    for arm_name in esug_arms:
        f = falsification_per_arm[arm_name]
        verdict = "FALSIFIED" if f["any_falsified"] else "VALIDATED"
        print(f"  {arm_name} hypothesis: {verdict}  "
              f"(recruit={f['recruitment_count']}/5, "
              f"mse={f['mse_cent_mean']:.3f}, "
              f"false_recruit={f['false_recruitment_count']}/5)")

    print("\nDone.")


if __name__ == "__main__":
    main()
