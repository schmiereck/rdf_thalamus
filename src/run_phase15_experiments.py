"""
Phase 15 experiments: Surprise-Adaptive Contrastive Coordinate Regularization
(SA-CCR) vs Fixed Regularization vs Dual Control Architecture.

Pre-registered hypothesis (see src/pre_registration.md):
    Proportional surprise scaling of the covariance regularization weight on
    the non-parametric soft-argmax bottleneck (Arm L) will stabilize
    coordinate representations during high-surprise transitions more
    effectively than a fixed weight (Arm K) and an inverse surprise scaling
    (Arm M), without degrading physics prediction.

This script also evaluates an Arm N (Dual Control) as proposed by the
Strategic Manager: a slow Categorizer with a consistency buffer (Minimum
Description Length-style) that gates dimension recruitment via a
sim_loss_new / sim_loss_old ratio, computed from a held-out batch of replay
transitions.

Compares four arms across the same 5 random seeds (matched comparison):
    - Arm K (Baseline)              : ccr_spatial_weight = 10.0 fixed
    - Arm L (Proportional SA-CCR)   : 10.0 * (1 + 2 * S_bar(t))
    - Arm M (Inverse SA-CCR)        : 10.0 / (1 + 2 * S_bar(t))
    - Arm N (Dual Control)          : Surprise Detector + slow Categorizer
                                      with consistency buffer
"""

import os
import sys
import json
import random
import collections
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models_dual_stream import NonParametricJEPASpatial
from src.motor import CLTSMotorController


# ----------------------------------------------------------------------------
# Replay buffer (unchanged from Phase 14)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# SA-CCR scheduler
# ----------------------------------------------------------------------------
def ccr_weight_for_arm(arm_name, S_bar, base=10.0, gamma=2.0):
    """Map (arm, smoothed surprise) -> ccr_spatial_weight."""
    if arm_name == "Arm L (Proportional SA-CCR)":
        return float(base * (1.0 + gamma * S_bar))
    if arm_name == "Arm M (Inverse SA-CCR)":
        return float(base / (1.0 + gamma * S_bar))
    return float(base)


# ----------------------------------------------------------------------------
# Dual Control Categorizer (Arm N)
# ----------------------------------------------------------------------------
def categorizer_consistency_ratio(model, replay_buffer, device, d_t_new, d_t_old, val_size=100):
    """Sample val_size transitions, evaluate sim_loss under d_t_new and d_t_old
    (without gradients), and return L_consistency = sim_loss_new / sim_loss_old.

    Lower is better: a ratio < 1.0 means recruiting the extra dimension
    actually reduced the held-out predictive error and is therefore
    'minimum description length'-justified.
    """
    if len(replay_buffer) < val_size:
        val_size = len(replay_buffer)
    x_hist_b, x_target_b = replay_buffer.sample(val_size)
    x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
    x_target_t = torch.from_numpy(x_target_b).float().to(device)

    orig_d_t = model.d_t
    model.eval()
    with torch.no_grad():
        # Evaluate under the proposed d_t (new)
        model.d_t = d_t_new
        loss_dict_new, _, _ = model(x_hist_t, x_target_t, ccr_mode='none')
        sim_new = float(loss_dict_new["sim_loss"].item())

        # Evaluate under the previous d_t (old)
        model.d_t = d_t_old
        loss_dict_old, _, _ = model(x_hist_t, x_target_t, ccr_mode='none')
        sim_old = float(loss_dict_old["sim_loss"].item())
    model.d_t = orig_d_t
    model.train()

    ratio = sim_new / max(sim_old, 1e-8)
    return ratio, sim_new, sim_old


# ----------------------------------------------------------------------------
# Evaluation: matches Phase 14, with an additional 'overall' MSE that uses
# the same probe but evaluates on the full 200-frame window.
# ----------------------------------------------------------------------------
def evaluate_branch_phase15(model, seed, device):
    set_seed(seed + 5000)
    test_env = PhysicsSandbox(N=4, seed=seed + 5000)
    test_env.masses[3] *= 2.0
    test_env.reset()
    test_obs = test_env.reset()
    test_history = collections.deque(maxlen=4)
    test_history.append(test_obs)

    test_x_hist = []
    test_x_target = []
    test_y_4 = []

    for _ in range(203):
        obs_t, info_t = test_env.step({"acc": 0.0, "push": False})
        test_history.append(obs_t)
        if len(test_history) == 4:
            test_x_hist.append(np.stack(list(test_history)[:3], axis=0))
            test_x_target.append(test_history[3])
            test_y_4.append(info_t["positions"][3])

    test_x_hist_t = torch.from_numpy(np.stack(test_x_hist, axis=0)).float().to(device)
    test_x_target_t = torch.from_numpy(np.stack(test_x_target, axis=0)).float().to(device)
    test_y_4_arr = np.array(test_y_4)

    y_probe_train = test_y_4_arr[:100]
    y_probe_test = test_y_4_arr[100:]

    model.eval()
    with torch.no_grad():
        loss_dict, _, _ = model(test_x_hist_t, test_x_target_t, ccr_mode='none')
        test_sim_loss = loss_dict["sim_loss"].item()

        z_target_coord, z_target_dyn = model.encoder(test_x_target_t)
        z_3 = z_target_coord[:, 3].cpu().numpy()

        a_spatial = model.encoder.forward_spatial(test_x_target_t)
        centroids, variances = model.calculate_centroid_and_variance(a_spatial)
        x_mean_3 = centroids[:, 3].cpu().numpy()
        var_3 = variances[:, 3].cpu().numpy()

        z_active_coord = torch.abs(z_target_coord[:, :4]).cpu().numpy()
        e_a_3 = np.mean(z_active_coord[:, 3])
        e_a_all = np.mean(z_active_coord)

    w_cent, b_cent = fit_linear_probe(x_mean_3[:100], y_probe_train)
    y_pred_cent_test = x_mean_3[100:] * w_cent + b_cent
    mse_cent_post = float(np.mean((y_probe_test - y_pred_cent_test) ** 2))

    # Overall MSE on the full 200-frame window using the same probe
    y_pred_cent_overall = x_mean_3 * w_cent + b_cent
    mse_cent_overall = float(np.mean((test_y_4_arr - y_pred_cent_overall) ** 2))

    r_centroid = np.corrcoef(x_mean_3, test_y_4_arr)[0, 1]
    abs_r_centroid = abs(r_centroid) if not np.isnan(r_centroid) else 0.0

    mean_var_3 = float(np.mean(var_3))
    std_x_mean_3 = float(np.std(x_mean_3))

    vel_3 = x_mean_3[1:] - x_mean_3[:-1]
    std_vel_3 = float(np.std(vel_3))
    mean_abs_vel_3 = float(np.mean(np.abs(vel_3)))

    has_collapsed = not (e_a_3 >= 0.1 * e_a_all and std_x_mean_3 > 5.0)

    return {
        "test_sim_loss": float(test_sim_loss),
        "abs_r_centroid": float(abs_r_centroid),
        "mse_cent": float(mse_cent_post),
        "mse_cent_overall": float(mse_cent_overall),
        "mean_var_3": mean_var_3,
        "std_x_mean_3": std_x_mean_3,
        "std_vel_3": std_vel_3,
        "mean_abs_vel_3": mean_abs_vel_3,
        "collapsed": bool(has_collapsed),
    }


# ----------------------------------------------------------------------------
# Passive training (steps 1..1500), with arm-specific weight schedule.
# Returns (trained_model, dual_control_passive_state) where state tracks
# whether the d_t=2 -> d_t=3 transition was accepted by the Categorizer
# (only relevant for Arm N; for other arms it is always accepted).
# ----------------------------------------------------------------------------
def train_base_model_passive(seed, device, arm_name, ccr_mode):
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

    # Surprise tracker for adaptive arms (initialized to 0.10)
    S_bar = 0.10
    alpha = 0.1

    # Arm N tracking
    passive_transition_accepted = True
    passive_transition_ratio = None
    passive_attempted = False

    print(f"     [passive] starting passive training on N=3 (steps 1..1500), "
          f"arm={arm_name}, ccr_mode={ccr_mode}")
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

        # Arm-specific weight schedule (Arm N keeps base = 10.0)
        ccr_w = ccr_weight_for_arm(arm_name, S_bar)

        optimizer.zero_grad()
        loss_dict, _, _ = model(
            x_hist_t, x_target_t,
            ccr_mode=ccr_mode,
            ccr_smooth_weight=10.0,
            ccr_spatial_weight=ccr_w,
        )
        loss_dict["loss"].backward()
        optimizer.step()

        sim_loss_val = float(loss_dict["sim_loss"].item())
        # Update smoothed surprise
        S_bar = alpha * sim_loss_val + (1.0 - alpha) * S_bar

        if step > 200:
            model.update_recruitment_logic(sim_loss_val, target_dim=2)

        # ---- d_t=2 -> d_t=3 transition at step 600 ----
        if (not passive_attempted) and model.d_t == 2 and step >= 600:
            passive_attempted = True
            if arm_name == "Arm N (Dual Control)":
                # Slow Categorizer: validate the transition via consistency buffer
                ratio, sim_new, sim_old = categorizer_consistency_ratio(
                    model, replay_buffer, device, d_t_new=3, d_t_old=2, val_size=100
                )
                passive_transition_ratio = ratio
                if ratio < 1.0:
                    print(f"       [Arm N Categorizer @ step {step}] PASSIVE transition "
                          f"d_t 2->3 ACCEPTED (ratio={ratio:.4f}, sim_new={sim_new:.5f}, "
                          f"sim_old={sim_old:.5f})")
                    model.d_t = 3
                    model.steps_since_recruitment = 0
                    model.reset_error_buffer()
                    passive_transition_accepted = True
                else:
                    print(f"       [Arm N Categorizer @ step {step}] PASSIVE transition "
                          f"d_t 2->3 REJECTED + SUPPRESSED (ratio={ratio:.4f}, "
                          f"sim_new={sim_new:.5f}, sim_old={sim_old:.5f})")
                    passive_transition_accepted = False
            else:
                model.d_t = 3
                model.steps_since_recruitment = 0
                model.reset_error_buffer()

        if step == 1001:
            model.reset_error_buffer()
        if step > 1000:
            # Only update recruitment toward target_dim=3 if we are at d_t=3 (or above).
            # For Arm N where the transition was suppressed, we keep updating toward d_t=2.
            target_dim_for_update = 3 if model.d_t >= 3 else 2
            model.update_recruitment_logic(sim_loss_val, target_dim=target_dim_for_update)

        if step % 250 == 0:
            print(f"       [passive step {step:4d}] sim_loss={sim_loss_val:.5f} "
                  f"S_bar={S_bar:.5f} ccr_w={ccr_w:.3f} "
                  f"ccr_smooth={loss_dict['ccr_smooth_loss'].item():.5f} "
                  f"ccr_spatial={loss_dict['ccr_spatial_loss'].item():.5f} "
                  f"d_t={model.d_t}")

    dual_state = {
        "passive_transition_accepted": passive_transition_accepted,
        "passive_transition_ratio": passive_transition_ratio,
        "S_bar_end_passive": float(S_bar),
    }
    return model, dual_state


# ----------------------------------------------------------------------------
# Active CLTS training (steps 1501..3000) with arm-specific weight schedule
# and Arm N's slow Categorizer for the d_t=3 -> d_t=4 transition.
# ----------------------------------------------------------------------------
def run_active_branch(branch_model, seed, device, arm_name, ccr_mode, dual_state):
    branch_optimizer = optim.Adam(branch_model.parameters(), lr=1e-3)

    set_seed(seed + 1000)
    branch_env = PhysicsSandbox(N=4, seed=seed + 1000)
    branch_env.masses[3] *= 2.0

    branch_obs = branch_env.reset()
    branch_history = collections.deque(maxlen=4)
    branch_history.append(branch_obs)

    branch_replay = ReplayBuffer(capacity=2000)
    last_info = prefill_buffer_passive(branch_env, branch_replay, branch_history, num_transitions=100)

    clts_controller = CLTSMotorController()
    clts_controller.reset()

    # Spatial-bottleneck schedule (same EWMA as Phase 14, kept for compatibility)
    ewma_surprise = 0.10
    lambda_val = 0.0

    # Smoothed surprise for SA-CCR (initialized to 0.10 at step 1501 per protocol)
    S_bar = 0.10
    alpha = 0.1

    pointer_positions = []
    online_losses = []

    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_results = {}

    # Arm N active transition state (only used by Arm N)
    active_transition_accepted = False
    active_transition_accepted_step = None
    active_transition_ratios = []
    next_categorizer_check = 1800

    # If Arm N rejected the passive transition, the base d_t is still 2 entering
    # the active phase. We log that condition.
    print(f"     [active] starting active CLTS training on N=4 "
          f"(steps 1501..3000), arm={arm_name}, ccr_mode={ccr_mode}, "
          f"entry d_t={branch_model.d_t}")

    for step in range(1501, 3001):
        pointer_positions.append(branch_env.pointer_pos)

        branch_model.eval()
        hist_t = torch.from_numpy(np.stack(list(branch_history)[:3], axis=0)).float().unsqueeze(0).to(device)
        target_t = torch.from_numpy(branch_history[-1]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            _, (zp_coord, zp_dyn), (zt_coord, zt_dyn) = branch_model(
                hist_t, target_t, ccr_mode='none'
            )
            a_spatial = branch_model.encoder.forward_spatial(target_t)
            centroids, _ = branch_model.calculate_centroid_and_variance(a_spatial)

        action, _, _ = clts_controller.get_action(
            branch_model, branch_history[-1], last_info,
            zp_coord, zt_coord, zp_dyn, zt_dyn,
            branch_model.d_t, centroids
        )

        obs, info = branch_env.step(action)
        last_info = info
        branch_history.append(obs)

        x_hist_new = np.stack(list(branch_history)[:3], axis=0)
        x_target_new = branch_history[3]
        branch_replay.push(x_hist_new, x_target_new)

        branch_model.train()
        x_hist_b, x_target_b = branch_replay.sample(32)
        x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
        x_target_t = torch.from_numpy(x_target_b).float().to(device)

        # Existing soft spatial-bottleneck schedule from Phase 14
        lambda_target = 0.10 * max(0.0, 1.0 - ewma_surprise / 2.0)
        if step == 1501:
            lambda_val = lambda_target
        else:
            lambda_val = lambda_val + np.clip(lambda_target - lambda_val, -0.002, 0.002)

        # Arm-specific CCR weight
        ccr_w = ccr_weight_for_arm(arm_name, S_bar)

        branch_optimizer.zero_grad()
        loss_dict, _, _ = branch_model(
            x_hist_t, x_target_t,
            lambda_spatial=lambda_val, k_chan=3,
            ccr_mode=ccr_mode,
            ccr_smooth_weight=10.0,
            ccr_spatial_weight=ccr_w,
        )
        loss_dict["loss"].backward()
        branch_optimizer.step()

        sim_loss_val = float(loss_dict["sim_loss"].item())
        online_losses.append(sim_loss_val)

        # Update both smoothed surprise (SA-CCR) and EWMA (legacy)
        S_bar = alpha * sim_loss_val + (1.0 - alpha) * S_bar
        ewma_surprise = 0.95 * ewma_surprise + 0.05 * sim_loss_val

        branch_model.update_recruitment_logic(sim_loss_val, target_dim=branch_model.d_t)

        # ---- d_t -> d_t+1 transition at step 1800+ ----
        if arm_name == "Arm N (Dual Control)":
            # Determine which target d_t we are attempting next. If passive was
            # rejected, we need to first attempt 2->3 here, then 3->4 once 3 is
            # accepted. If passive was accepted (or non-Arm-N), we attempt 3->4.
            if step >= next_categorizer_check and step <= 3000 and not active_transition_accepted:
                if branch_model.d_t == 2:
                    d_old, d_new = 2, 3
                elif branch_model.d_t == 3:
                    d_old, d_new = 3, 4
                else:
                    d_old, d_new = None, None

                if d_new is not None:
                    ratio, sim_new, sim_old = categorizer_consistency_ratio(
                        branch_model, branch_replay, device,
                        d_t_new=d_new, d_t_old=d_old, val_size=100
                    )
                    active_transition_ratios.append((step, d_old, d_new, ratio))
                    if ratio < 1.0:
                        print(f"       [Arm N Categorizer @ step {step}] ACTIVE transition "
                              f"d_t {d_old}->{d_new} ACCEPTED (ratio={ratio:.4f}, "
                              f"sim_new={sim_new:.5f}, sim_old={sim_old:.5f})")
                        branch_model.d_t = d_new
                        branch_model.steps_since_recruitment = 0
                        branch_model.reset_error_buffer()
                        # If we just promoted to 4, we are done.  If we just
                        # promoted from 2 to 3 (because passive was rejected),
                        # schedule the next check at step+50 so we can later
                        # promote 3->4.
                        if d_new == 4:
                            active_transition_accepted = True
                            active_transition_accepted_step = step
                            next_categorizer_check = step + 50
                        else:
                            next_categorizer_check = step + 50
                    else:
                        print(f"       [Arm N Categorizer @ step {step}] ACTIVE transition "
                              f"d_t {d_old}->{d_new} REJECTED (ratio={ratio:.4f}, "
                              f"sim_new={sim_new:.5f}, sim_old={sim_old:.5f}); "
                              f"retry at step {step + 50}")
                        next_categorizer_check = step + 50
        else:
            # Original Phase 14 unconditional transition
            if branch_model.d_t == 3 and step >= 1800:
                branch_model.d_t = 4
                branch_model.steps_since_recruitment = 0
                branch_model.reset_error_buffer()

        if step % 250 == 0:
            print(f"       [active step {step:4d}] sim_loss={sim_loss_val:.5f} "
                  f"S_bar={S_bar:.5f} ccr_w={ccr_w:.3f} "
                  f"ccr_smooth={loss_dict['ccr_smooth_loss'].item():.5f} "
                  f"ccr_spatial={loss_dict['ccr_spatial_loss'].item():.5f} "
                  f"lambda_sp={lambda_val:.4f} d_t={branch_model.d_t}")

        if step in eval_steps:
            eval_res = evaluate_branch_phase15(branch_model, seed, device)
            checkpoint_results[step] = eval_res
            print(f"       [eval @ {step}] test_sim_loss={eval_res['test_sim_loss']:.6f} "
                  f"mse_cent={eval_res['mse_cent']:.3f} "
                  f"mse_overall={eval_res['mse_cent_overall']:.3f} "
                  f"mean_var_3={eval_res['mean_var_3']:.3f} "
                  f"std_vel_3={eval_res['std_vel_3']:.3f}")

    eval_metrics = evaluate_branch_phase15(branch_model, seed, device)
    pointer_entropy = compute_spatial_entropy(pointer_positions)
    online_auc_1501_2000 = float(sum(online_losses[:500]))
    online_auc_1501_3000 = float(sum(online_losses))

    return {
        "eval_metrics": eval_metrics,
        "pointer_entropy": float(pointer_entropy),
        "online_auc_1501_2000": online_auc_1501_2000,
        "online_auc_1501_3000": online_auc_1501_3000,
        "checkpoint_results": checkpoint_results,
        "active_transition_accepted": active_transition_accepted,
        "active_transition_accepted_step": active_transition_accepted_step,
        "active_transition_ratios": active_transition_ratios,
        "S_bar_end_active": float(S_bar),
    }


# ----------------------------------------------------------------------------
# Welch / Levene test helpers
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PHASE 15 SWEEP: SURPRISE-ADAPTIVE CCR vs FIXED CCR vs DUAL CONTROL")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = [42, 123, 456, 789, 999]
    arms = [
        ("Arm K (Baseline)",                "covariance"),
        ("Arm L (Proportional SA-CCR)",     "covariance"),
        ("Arm M (Inverse SA-CCR)",          "covariance"),
        ("Arm N (Dual Control)",            "covariance"),
    ]

    results_list = []
    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_losses = {arm_name: {step: [] for step in eval_steps} for arm_name, _ in arms}
    checkpoint_mse_cent = {arm_name: {step: [] for step in eval_steps} for arm_name, _ in arms}
    checkpoint_std_vel = {arm_name: {step: [] for step in eval_steps} for arm_name, _ in arms}

    for seed in seeds:
        print("\n" + "-" * 50)
        print(f"SEED {seed}")
        print("-" * 50)

        for arm_name, ccr_mode in arms:
            print(f"\n[{arm_name}] (ccr_mode={ccr_mode})")
            print(f"  1. Training passive base model on N=3 with arm={arm_name}, "
                  f"ccr_mode={ccr_mode}...")
            base_model, dual_state = train_base_model_passive(
                seed, device, arm_name, ccr_mode
            )
            print(f"     Finished passive pretraining. d_t = {base_model.d_t}, "
                  f"S_bar_end={dual_state['S_bar_end_passive']:.5f}, "
                  f"passive_transition_accepted={dual_state['passive_transition_accepted']}, "
                  f"passive_transition_ratio={dual_state['passive_transition_ratio']}")

            base_eval = evaluate_branch_phase15(base_model, seed, device)
            checkpoint_losses[arm_name][1500].append(base_eval["test_sim_loss"])
            checkpoint_mse_cent[arm_name][1500].append(base_eval["mse_cent"])
            checkpoint_std_vel[arm_name][1500].append(base_eval["std_vel_3"])
            print(f"     Step 1500 base eval: "
                  f"test_sim_loss={base_eval['test_sim_loss']:.6f} "
                  f"mse_cent={base_eval['mse_cent']:.3f} "
                  f"mse_cent_overall={base_eval['mse_cent_overall']:.3f} "
                  f"std_vel_3={base_eval['std_vel_3']:.3f}")

            print(f"  2. Active CLTS training (steps 1501..3000) on N=4 with 2x mass "
                  f"perturbation, arm={arm_name}, ccr_mode={ccr_mode}...")
            branch_results = run_active_branch(
                base_model, seed, device, arm_name, ccr_mode, dual_state
            )

            for step in eval_steps:
                if step == 1500:
                    continue
                if step in branch_results["checkpoint_results"]:
                    cp = branch_results["checkpoint_results"][step]
                    checkpoint_losses[arm_name][step].append(cp["test_sim_loss"])
                    checkpoint_mse_cent[arm_name][step].append(cp["mse_cent"])
                    checkpoint_std_vel[arm_name][step].append(cp["std_vel_3"])

            em = branch_results["eval_metrics"]
            results_list.append({
                "seed": seed,
                "arm": arm_name,
                "ccr_mode": ccr_mode,
                "test_sim_loss": em["test_sim_loss"],
                "abs_r_centroid": em["abs_r_centroid"],
                "mse_cent": em["mse_cent"],
                "mse_cent_overall": em["mse_cent_overall"],
                "mean_var_3": em["mean_var_3"],
                "std_x_mean_3": em["std_x_mean_3"],
                "std_vel_3": em["std_vel_3"],
                "mean_abs_vel_3": em["mean_abs_vel_3"],
                "collapsed": int(em["collapsed"]),
                "pointer_entropy": branch_results["pointer_entropy"],
                "online_auc_1501_2000": branch_results["online_auc_1501_2000"],
                "online_auc_1501_3000": branch_results["online_auc_1501_3000"],
                "passive_transition_accepted": int(dual_state["passive_transition_accepted"]),
                "active_transition_accepted_step": (
                    branch_results["active_transition_accepted_step"]
                    if branch_results["active_transition_accepted_step"] is not None else -1
                ),
                "final_d_t": int(base_model.d_t),
            })
            print(
                f"     Done. Sim Loss: {em['test_sim_loss']:.6f}, "
                f"MSE Centroid (post): {em['mse_cent']:.4f}, "
                f"MSE Centroid (overall): {em['mse_cent_overall']:.4f}, "
                f"Soft Var: {em['mean_var_3']:.4f}, "
                f"std_vel_3: {em['std_vel_3']:.4f}, "
                f"mean_abs_vel_3: {em['mean_abs_vel_3']:.4f}, "
                f"Pointer Entropy: {branch_results['pointer_entropy']:.4f}, "
                f"final d_t: {base_model.d_t}"
            )

    # ------------------------------------------------------------------
    # Save CSV summary
    # ------------------------------------------------------------------
    results_dir = "archive/iter_015/results"
    os.makedirs(results_dir, exist_ok=True)
    summary_df = pd.DataFrame(results_list)
    summary_csv_path = os.path.join(results_dir, "summary_phase15.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved {summary_csv_path}")

    # ------------------------------------------------------------------
    # Plot adaptation curves
    # ------------------------------------------------------------------
    avg_checkpoint_losses = {}
    for arm_name, _ in arms:
        avg_checkpoint_losses[arm_name] = [
            float(np.mean(checkpoint_losses[arm_name][step])) for step in eval_steps
        ]

    plt.figure(figsize=(10, 6))
    colors_dict = {
        "Arm K (Baseline)":              "green",
        "Arm L (Proportional SA-CCR)":   "blue",
        "Arm M (Inverse SA-CCR)":        "orange",
        "Arm N (Dual Control)":          "purple",
    }
    for arm_name, _ in arms:
        plt.plot(
            eval_steps, avg_checkpoint_losses[arm_name],
            marker='o', label=arm_name, color=colors_dict[arm_name], linewidth=2,
        )
    plt.title("Phase 15: Adaptation Trajectory (Offline Test Sim Loss over Steps)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Offline Test Simulation Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plot_path = os.path.join(results_dir, "adaptation_curves_phase15.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path}")

    # ------------------------------------------------------------------
    # Statistical analysis (Welch's t-test + Levene's test)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (step 3000 final eval)")
    print("=" * 80)

    def arr(arm_name, field):
        return np.array([r[field] for r in results_list if r["arm"] == arm_name], dtype=float)

    arm_k_sim = arr("Arm K (Baseline)", "test_sim_loss")
    arm_l_sim = arr("Arm L (Proportional SA-CCR)", "test_sim_loss")
    arm_m_sim = arr("Arm M (Inverse SA-CCR)", "test_sim_loss")
    arm_n_sim = arr("Arm N (Dual Control)", "test_sim_loss")

    arm_k_mse = arr("Arm K (Baseline)", "mse_cent")
    arm_l_mse = arr("Arm L (Proportional SA-CCR)", "mse_cent")
    arm_m_mse = arr("Arm M (Inverse SA-CCR)", "mse_cent")
    arm_n_mse = arr("Arm N (Dual Control)", "mse_cent")

    arm_k_var = arr("Arm K (Baseline)", "mean_var_3")
    arm_l_var = arr("Arm L (Proportional SA-CCR)", "mean_var_3")
    arm_m_var = arr("Arm M (Inverse SA-CCR)", "mean_var_3")
    arm_n_var = arr("Arm N (Dual Control)", "mean_var_3")

    for label, k, l, m, n in [
        ("test_sim_loss", arm_k_sim, arm_l_sim, arm_m_sim, arm_n_sim),
        ("mse_cent",      arm_k_mse, arm_l_mse, arm_m_mse, arm_n_mse),
        ("mean_var_3",    arm_k_var, arm_l_var, arm_m_var, arm_n_var),
    ]:
        print(f"\n--- {label} ---")
        print(f"  Arm K mean={k.mean():.6f}  std={k.std(ddof=1):.6f}  values={k.tolist()}")
        print(f"  Arm L mean={l.mean():.6f}  std={l.std(ddof=1):.6f}  values={l.tolist()}")
        print(f"  Arm M mean={m.mean():.6f}  std={m.std(ddof=1):.6f}  values={m.tolist()}")
        print(f"  Arm N mean={n.mean():.6f}  std={n.std(ddof=1):.6f}  values={n.tolist()}")

    # Pre-registered comparisons
    t_LK_sim, p_LK_sim = safe_ttest(arm_l_sim, arm_k_sim)
    t_LK_mse, p_LK_mse = safe_ttest(arm_l_mse, arm_k_mse)
    t_ML_sim, p_ML_sim = safe_ttest(arm_m_sim, arm_l_sim)
    t_ML_mse, p_ML_mse = safe_ttest(arm_m_mse, arm_l_mse)
    t_NK_sim, p_NK_sim = safe_ttest(arm_n_sim, arm_k_sim)
    t_NK_mse, p_NK_mse = safe_ttest(arm_n_mse, arm_k_mse)

    w_LK_sim, p_lev_LK_sim = safe_levene(arm_l_sim, arm_k_sim)
    w_LK_mse, p_lev_LK_mse = safe_levene(arm_l_mse, arm_k_mse)
    w_ML_sim, p_lev_ML_sim = safe_levene(arm_m_sim, arm_l_sim)
    w_ML_mse, p_lev_ML_mse = safe_levene(arm_m_mse, arm_l_mse)
    w_NK_sim, p_lev_NK_sim = safe_levene(arm_n_sim, arm_k_sim)
    w_NK_mse, p_lev_NK_mse = safe_levene(arm_n_mse, arm_k_mse)

    print("\nWelch's two-sample two-sided t-tests:")
    print(f"  L vs K  | test_sim_loss : t={t_LK_sim:.4f}, p={p_LK_sim:.6f}")
    print(f"  L vs K  | mse_cent      : t={t_LK_mse:.4f}, p={p_LK_mse:.6f}")
    print(f"  M vs L  | test_sim_loss : t={t_ML_sim:.4f}, p={p_ML_sim:.6f}")
    print(f"  M vs L  | mse_cent      : t={t_ML_mse:.4f}, p={p_ML_mse:.6f}")
    print(f"  N vs K  | test_sim_loss : t={t_NK_sim:.4f}, p={p_NK_sim:.6f}")
    print(f"  N vs K  | mse_cent      : t={t_NK_mse:.4f}, p={p_NK_mse:.6f}")

    print("\nLevene's test for equality of variances:")
    print(f"  L vs K  | test_sim_loss : W={w_LK_sim:.4f}, p={p_lev_LK_sim:.6f}")
    print(f"  L vs K  | mse_cent      : W={w_LK_mse:.4f}, p={p_lev_LK_mse:.6f}")
    print(f"  M vs L  | test_sim_loss : W={w_ML_sim:.4f}, p={p_lev_ML_sim:.6f}")
    print(f"  M vs L  | mse_cent      : W={w_ML_mse:.4f}, p={p_lev_ML_mse:.6f}")
    print(f"  N vs K  | test_sim_loss : W={w_NK_sim:.4f}, p={p_lev_NK_sim:.6f}")
    print(f"  N vs K  | mse_cent      : W={w_NK_mse:.4f}, p={p_lev_NK_mse:.6f}")

    # ------------------------------------------------------------------
    # Pre-registered falsification audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCIENTIFIC AUDIT: PRE-REGISTERED FALSIFICATION CRITERIA")
    print("=" * 80)

    agg = summary_df.groupby("arm").mean(numeric_only=True).reset_index()
    print(agg.to_string(index=False))

    # Helper: lower MSE is better, lower sim_loss is better.
    # Criterion 1: Arm L does NOT achieve a statistically significant reduction
    # in post-collision centroid decoding MSE compared to Arm K (p >= 0.05).
    #   -> mean(L) >= mean(K) OR p_LK_mse >= 0.05
    c1_falsified = bool(
        (arm_l_mse.mean() >= arm_k_mse.mean()) or (p_LK_mse >= 0.05)
    )

    # Criterion 2: Arm L's post-collision test simulation loss is statistically
    # inferior (i.e. higher) to Arm K (p < 0.05, mean(L) > mean(K)).
    c2_falsified = bool(
        (arm_l_sim.mean() > arm_k_sim.mean()) and (p_LK_sim < 0.05)
    )

    # Criterion 3: Soft spatial variance of Arm L's coordinate bottleneck
    # exceeds 8.5 pixels^2 during high-surprise phases (use mean_var_3).
    c3_falsified = bool(arm_l_var.mean() > 8.5)

    # Criterion 4: Arm M achieves lower centroid decoding MSE than Arm L while
    # maintaining equal or better test simulation loss.
    c4_falsified = bool(
        (arm_m_mse.mean() < arm_l_mse.mean()) and (arm_m_sim.mean() <= arm_l_sim.mean())
    )

    print(f"\nCriterion 1 (Significant MSE improvement L vs K, falsified if not):")
    print(f"  Arm K mse_cent mean: {arm_k_mse.mean():.4f}")
    print(f"  Arm L mse_cent mean: {arm_l_mse.mean():.4f}")
    print(f"  Welch p (L vs K):    {p_LK_mse:.6f}")
    print(f"  -> {'FALSIFIED' if c1_falsified else 'VALIDATED'}")

    print(f"\nCriterion 2 (L sim_loss NOT statistically inferior to K, falsified if it is):")
    print(f"  Arm K sim_loss mean: {arm_k_sim.mean():.6f}")
    print(f"  Arm L sim_loss mean: {arm_l_sim.mean():.6f}")
    print(f"  Welch p (L vs K):    {p_LK_sim:.6f}")
    print(f"  -> {'FALSIFIED' if c2_falsified else 'VALIDATED'}")

    print(f"\nCriterion 3 (Arm L soft variance <= 8.5):")
    print(f"  Arm L mean_var_3 mean: {arm_l_var.mean():.4f}")
    print(f"  -> {'FALSIFIED' if c3_falsified else 'VALIDATED'}")

    print(f"\nCriterion 4 (M not strictly better than L on both MSE and sim_loss):")
    print(f"  Arm L mse_cent: {arm_l_mse.mean():.4f}  sim_loss: {arm_l_sim.mean():.6f}")
    print(f"  Arm M mse_cent: {arm_m_mse.mean():.4f}  sim_loss: {arm_m_sim.mean():.6f}")
    print(f"  -> {'FALSIFIED' if c4_falsified else 'VALIDATED'}")

    hypothesis_falsified = bool(c1_falsified or c2_falsified or c3_falsified or c4_falsified)

    print("\n" + "=" * 80)
    print(f"OVERALL SA-CCR HYPOTHESIS: "
          f"{'FALSIFIED' if hypothesis_falsified else 'VALIDATED'}")
    print("=" * 80)

    # Strategic Manager's empirical comparison: does the slow Categorizer
    # (Arm N) deliver comparable or better stability than the fixed-CCR
    # baseline (Arm K) *without* hand-coding the weight schedule?
    print("\n--- Strategic Manager's empirical comparison (Arm N vs Arm K) ---")
    print(f"  Arm K mse_cent: {arm_k_mse.mean():.4f}  Arm N mse_cent: {arm_n_mse.mean():.4f}  "
          f"Welch p (N vs K): {p_NK_mse:.6f}")
    print(f"  Arm K sim_loss: {arm_k_sim.mean():.6f}  Arm N sim_loss: {arm_n_sim.mean():.6f}  "
          f"Welch p (N vs K): {p_NK_sim:.6f}")

    # Save audit JSON
    audit_dict = {
        "arms": [a for a, _ in arms],
        "seeds": seeds,
        "means": {
            "arm_k_test_sim_loss": float(arm_k_sim.mean()),
            "arm_l_test_sim_loss": float(arm_l_sim.mean()),
            "arm_m_test_sim_loss": float(arm_m_sim.mean()),
            "arm_n_test_sim_loss": float(arm_n_sim.mean()),
            "arm_k_mse_cent": float(arm_k_mse.mean()),
            "arm_l_mse_cent": float(arm_l_mse.mean()),
            "arm_m_mse_cent": float(arm_m_mse.mean()),
            "arm_n_mse_cent": float(arm_n_mse.mean()),
            "arm_k_mean_var_3": float(arm_k_var.mean()),
            "arm_l_mean_var_3": float(arm_l_var.mean()),
            "arm_m_mean_var_3": float(arm_m_var.mean()),
            "arm_n_mean_var_3": float(arm_n_var.mean()),
        },
        "ttests": {
            "L_vs_K_sim": {"t": t_LK_sim, "p": p_LK_sim},
            "L_vs_K_mse": {"t": t_LK_mse, "p": p_LK_mse},
            "M_vs_L_sim": {"t": t_ML_sim, "p": p_ML_sim},
            "M_vs_L_mse": {"t": t_ML_mse, "p": p_ML_mse},
            "N_vs_K_sim": {"t": t_NK_sim, "p": p_NK_sim},
            "N_vs_K_mse": {"t": t_NK_mse, "p": p_NK_mse},
        },
        "levene": {
            "L_vs_K_sim": {"W": w_LK_sim, "p": p_lev_LK_sim},
            "L_vs_K_mse": {"W": w_LK_mse, "p": p_lev_LK_mse},
            "M_vs_L_sim": {"W": w_ML_sim, "p": p_lev_ML_sim},
            "M_vs_L_mse": {"W": w_ML_mse, "p": p_lev_ML_mse},
            "N_vs_K_sim": {"W": w_NK_sim, "p": p_lev_NK_sim},
            "N_vs_K_mse": {"W": w_NK_mse, "p": p_lev_NK_mse},
        },
        "falsification": {
            "c1_falsified": c1_falsified,
            "c2_falsified": c2_falsified,
            "c3_falsified": c3_falsified,
            "c4_falsified": c4_falsified,
            "hypothesis_falsified": hypothesis_falsified,
        },
    }
    audit_path = os.path.join(results_dir, "audit_results_phase15.json")
    with open(audit_path, "w") as f:
        json.dump(audit_dict, f, indent=4)
    print(f"\nSaved {audit_path}")


if __name__ == "__main__":
    main()
