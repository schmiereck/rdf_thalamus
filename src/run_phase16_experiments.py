"""
Phase 16 experiments: Probationary Warm-Up Period (WUP) for dimension
recruitment under CLTS active control, with PVU vs MDL gating criteria.

Pre-registered hypothesis (see src/pre_registration.md):
    A Probationary Warm-Up Period of W steps, combined with a
    Predictability-Variance-Uniqueness (PVU) gating metric, will resolve
    the cold-start rejection bias of the immediate MDL gate. Arms O / O_big
    are expected to recruit the 4th dimension in >= 4/5 seeds, reduce the
    post-transition centroid decoding MSE strictly below 70.0, and avoid
    inflating the test simulation loss above 0.15.

Compares six arms across the same 5 random seeds (matched comparison):
    - Arm K (Baseline)         : no probation, unconditional 3->4 at step 1800
    - Arm N (Original MDL)     : immediate MDL gate at step 1800+ (no WUP)
    - Arm O (WUP-PVU, W=100)   : WUP=100, PVU criteria at probation end
    - Arm O_big (WUP-PVU, W=500): WUP=500, PVU criteria at probation end
    - Arm P (WUP-MDL, W=100)   : WUP=100, MDL ratio < 1.0 at probation end
    - Arm P_big (WUP-MDL, W=500): WUP=500, MDL ratio < 1.0 at probation end

CPU-speed optimization: the passive pre-training phase (steps 1..1500) is
cached. For each seed we train exactly TWO passive states (one standard and
one "Arm N"), and clone them into the active phase for the appropriate arms.
This reduces the passive training cost from 30 runs to 10 runs.
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
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models_dual_stream import NonParametricJEPASpatial
from src.motor import CLTSMotorController


# ----------------------------------------------------------------------------
# Replay buffer (unchanged from Phase 14/15)
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
# MDL-style consistency ratio (used by Arm N and Arm P / P_big)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# PVU evaluation at the end of a probation window (used by Arm O / O_big)
# ----------------------------------------------------------------------------
def pvu_evaluation(model, replay_buffer, device, new_dim_idx=3, val_size=100,
                   pvu_eps=1e-5):
    """Evaluate PVU on the newly-proposed dimension after the probation window.

    Returns a dict with var_new, mse_new, u_new, max_corr, accepted (bool),
    and the per-existing-dim absolute correlations.
    """
    if len(replay_buffer) < val_size:
        val_size = len(replay_buffer)
    x_hist_b, x_target_b = replay_buffer.sample(val_size)
    x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
    x_target_t = torch.from_numpy(x_target_b).float().to(device)

    model.eval()
    with torch.no_grad():
        _, (zp_coord, _zp_dyn), (zt_coord, _zt_dyn) = model(
            x_hist_t, x_target_t, ccr_mode='none'
        )
    model.train()

    z_target = zt_coord[:, new_dim_idx].detach().cpu().numpy().astype(np.float64)
    z_pred   = zp_coord[:, new_dim_idx].detach().cpu().numpy().astype(np.float64)

    mse_new = float(np.mean((z_target - z_pred) ** 2))
    var_new = float(np.var(z_target))
    u_new = mse_new / (var_new + pvu_eps)

    # Uniqueness: max absolute Pearson correlation with each existing dim
    corrs = []
    for j in range(new_dim_idx):
        z_j = zt_coord[:, j].detach().cpu().numpy().astype(np.float64)
        if np.std(z_target) > 1e-8 and np.std(z_j) > 1e-8:
            r = np.corrcoef(z_target, z_j)[0, 1]
            corrs.append(abs(float(r)) if not np.isnan(r) else 0.0)
        else:
            corrs.append(0.0)
    max_corr = float(max(corrs)) if corrs else 0.0

    accepted = bool((var_new > 1e-3) and (u_new < 0.5) and (max_corr < 0.8))

    return {
        "var_new": var_new,
        "mse_new": mse_new,
        "u_new": u_new,
        "max_corr": max_corr,
        "corrs": corrs,
        "accepted": accepted,
    }


# ----------------------------------------------------------------------------
# Evaluation (matched to Phase 15)
# ----------------------------------------------------------------------------
def evaluate_branch(model, seed, device):
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

        z_target_coord, _ = model.encoder(test_x_target_t)
        _ = z_target_coord[:, 3].cpu().numpy()

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
# Passive training (steps 1..1500). Two variants:
#   variant = "standard"  : unconditional 2->3 transition at step 600
#   variant = "arm_n"     : MDL-gated 2->3 transition at step 600 (Phase-15
#                           Arm-N semantics; typically REJECTED -> stays at 2)
# Each is trained ONCE per seed and cached, then cloned by all arms that
# share the variant.
# ----------------------------------------------------------------------------
def train_passive_cached(seed, device, variant):
    assert variant in ("standard", "arm_n")
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

    passive_transition_accepted = True
    passive_transition_ratio = None
    passive_attempted = False

    print(f"     [passive/{variant}] starting passive training on N=3 (steps 1..1500)")
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

        # ---- d_t=2 -> d_t=3 transition at step 600 ----
        if (not passive_attempted) and model.d_t == 2 and step >= 600:
            passive_attempted = True
            if variant == "arm_n":
                ratio, sim_new, sim_old = categorizer_consistency_ratio(
                    model, replay_buffer, device, d_t_new=3, d_t_old=2, val_size=100
                )
                passive_transition_ratio = ratio
                if ratio < 1.0:
                    print(f"       [arm_n MDL @ step {step}] PASSIVE 2->3 ACCEPTED "
                          f"(ratio={ratio:.4f}, sim_new={sim_new:.5f}, sim_old={sim_old:.5f})")
                    model.d_t = 3
                    model.steps_since_recruitment = 0
                    model.reset_error_buffer()
                    passive_transition_accepted = True
                else:
                    print(f"       [arm_n MDL @ step {step}] PASSIVE 2->3 REJECTED + SUPPRESSED "
                          f"(ratio={ratio:.4f}, sim_new={sim_new:.5f}, sim_old={sim_old:.5f})")
                    passive_transition_accepted = False
            else:
                model.d_t = 3
                model.steps_since_recruitment = 0
                model.reset_error_buffer()

        if step == 1001:
            model.reset_error_buffer()
        if step > 1000:
            target_dim_for_update = 3 if model.d_t >= 3 else 2
            model.update_recruitment_logic(sim_loss_val, target_dim=target_dim_for_update)

        if step % 500 == 0:
            print(f"       [passive/{variant} step {step:4d}] sim_loss={sim_loss_val:.5f} "
                  f"S_bar={S_bar:.5f} d_t={model.d_t}")

    dual_state = {
        "variant": variant,
        "passive_transition_accepted": passive_transition_accepted,
        "passive_transition_ratio": passive_transition_ratio,
        "S_bar_end_passive": float(S_bar),
        "final_d_t_after_passive": int(model.d_t),
    }
    return model, dual_state


# ----------------------------------------------------------------------------
# Active CLTS training (steps 1501..3000) per arm. Operates on a CLONED
# passive base model so the cached state is not mutated.
# ----------------------------------------------------------------------------
def run_active_branch(base_model_template, seed, device, arm_name, dual_state,
                      wup_window=None, gating=None):
    """Run the active CLTS phase for a single arm.

    Args:
        base_model_template: cached passive model (must NOT be mutated).
        arm_name: human-readable arm name.
        wup_window: None for arms K and N. Integer W for arms O/O_big/P/P_big.
        gating: "pvu" or "mdl" when wup_window is set.
    """
    # Clone the cached passive model so the cache is preserved.
    branch_model = base_model_template.clone().to(device)
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

    ewma_surprise = 0.10
    lambda_val = 0.0
    S_bar = 0.10
    alpha = 0.1

    pointer_positions = []
    online_losses = []

    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_results = {}

    # ---- Arm N (immediate MDL) active state ----
    active_transition_accepted = False
    active_transition_accepted_step = None
    active_transition_ratios = []
    next_categorizer_check = 1800

    # ---- Probationary arms state ----
    probationary = False
    probation_end_step = None
    probation_attempts = []   # list of dicts: {step, criteria_dict, accepted}
    next_proposal_check = 1800
    final_gating_dict = None  # final accept/reject diagnostics

    print(f"     [active] starting active CLTS training on N=4 "
          f"(steps 1501..3000), arm={arm_name}, entry d_t={branch_model.d_t}")

    for step in range(1501, 3001):
        pointer_positions.append(branch_env.pointer_pos)

        # ---- Forward pass (no grad) to feed CLTS ----
        branch_model.eval()
        hist_t = torch.from_numpy(np.stack(list(branch_history)[:3], axis=0)).float().unsqueeze(0).to(device)
        target_t = torch.from_numpy(branch_history[-1]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            _, (zp_coord, zp_dyn), (zt_coord, zt_dyn) = branch_model(
                hist_t, target_t, ccr_mode='none'
            )
            a_spatial = branch_model.encoder.forward_spatial(target_t)
            centroids, _ = branch_model.calculate_centroid_and_variance(a_spatial)

        # Effective d_t for motor: during probation we PROTECT exploration by
        # capping the controller to the first 3 dimensions, regardless of the
        # fact that branch_model.d_t == 4 internally.
        if probationary:
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

        action, _, _ = clts_controller.get_action(
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

        branch_optimizer.zero_grad()
        loss_dict, _, _ = branch_model(
            x_hist_t, x_target_t,
            lambda_spatial=lambda_val, k_chan=3,
            ccr_mode='covariance',
            ccr_smooth_weight=10.0,
            ccr_spatial_weight=10.0,
        )
        loss_dict["loss"].backward()
        branch_optimizer.step()

        sim_loss_val = float(loss_dict["sim_loss"].item())
        online_losses.append(sim_loss_val)

        S_bar = alpha * sim_loss_val + (1.0 - alpha) * S_bar
        ewma_surprise = 0.95 * ewma_surprise + 0.05 * sim_loss_val

        # Recruitment-logic policy in the active phase:
        #   * Arm N: GDASR remains active (matches Phase-15 Arm-N protocol);
        #     the MDL gate still controls 3->4 via categorizer_consistency_ratio.
        #   * Arms K / O / O_big / P / P_big: GDASR is suppressed for the
        #     3->4 transition. We still update the EMA / error buffer (so the
        #     statistics are populated), but we clamp recruitment by passing
        #     target_dim = current d_t only when d_t < 3, otherwise we skip
        #     update_recruitment_logic entirely. This ensures the 3->4
        #     transition is governed exclusively by the arm's gating mechanism
        #     (unconditional step-1800 for Arm K; probationary WUP for the
        #     others), as specified in the Phase-16 protocol.
        if arm_name == "Arm N (Original MDL)":
            branch_model.update_recruitment_logic(
                sim_loss_val, target_dim=branch_model.d_t
            )
        else:
            if branch_model.d_t < 3 and not probationary:
                branch_model.update_recruitment_logic(
                    sim_loss_val, target_dim=branch_model.d_t
                )
            # Otherwise: leave d_t alone; arm-specific mechanism handles 3->4.

        # ---- Arm K (baseline, unconditional 3->4 at step 1800) ----
        if arm_name == "Arm K (Baseline)":
            if branch_model.d_t == 3 and step >= 1800:
                branch_model.d_t = 4
                branch_model.steps_since_recruitment = 0
                branch_model.reset_error_buffer()
                if not active_transition_accepted:
                    active_transition_accepted = True
                    active_transition_accepted_step = step

        # ---- Arm N (immediate MDL gate, no probation) ----
        elif arm_name == "Arm N (Original MDL)":
            if step >= next_categorizer_check and not active_transition_accepted:
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
                        print(f"       [Arm N MDL @ step {step}] {d_old}->{d_new} ACCEPTED "
                              f"(ratio={ratio:.4f}, sim_new={sim_new:.5f}, sim_old={sim_old:.5f})")
                        branch_model.d_t = d_new
                        branch_model.steps_since_recruitment = 0
                        branch_model.reset_error_buffer()
                        if d_new == 4:
                            active_transition_accepted = True
                            active_transition_accepted_step = step
                            final_gating_dict = {
                                "criterion": "mdl",
                                "ratio": float(ratio),
                                "sim_new": float(sim_new),
                                "sim_old": float(sim_old),
                            }
                        next_categorizer_check = step + 50
                    else:
                        print(f"       [Arm N MDL @ step {step}] {d_old}->{d_new} REJECTED "
                              f"(ratio={ratio:.4f}, sim_new={sim_new:.5f}, sim_old={sim_old:.5f}); "
                              f"retry @ step {step + 50}")
                        next_categorizer_check = step + 50

        # ---- Arms O / O_big / P / P_big (WUP-PVU / WUP-MDL) ----
        elif arm_name in ("Arm O (WUP-PVU, W=100)", "Arm O_big (WUP-PVU, W=500)",
                          "Arm P (WUP-MDL, W=100)", "Arm P_big (WUP-MDL, W=500)"):
            # 1. Propose probation if not currently probating, not yet accepted,
            #    and the proposal check has come due.
            if (not probationary) and (not active_transition_accepted) \
                    and step >= next_proposal_check and branch_model.d_t == 3:
                probationary = True
                probation_end_step = step + wup_window
                branch_model.d_t = 4  # let the 4th-dim heads receive gradients
                branch_model.steps_since_recruitment = 0
                branch_model.reset_error_buffer()
                print(f"       [{arm_name} @ step {step}] PROBATION STARTED for 3->4 "
                      f"(W={wup_window}, end={probation_end_step}, gating={gating})")

            # 2. Evaluate at probation end.
            if probationary and step == probation_end_step:
                if gating == "pvu":
                    crit = pvu_evaluation(branch_model, branch_replay, device,
                                          new_dim_idx=3, val_size=100)
                    accepted = crit["accepted"]
                    print(f"       [{arm_name} PVU @ step {step}] "
                          f"var_new={crit['var_new']:.6f} mse_new={crit['mse_new']:.6f} "
                          f"u_new={crit['u_new']:.4f} max_corr={crit['max_corr']:.4f} "
                          f"-> {'ACCEPTED' if accepted else 'REJECTED'}")
                    probation_attempts.append({"step": step, "criterion": "pvu", **crit})
                    diag = {"criterion": "pvu", **crit}
                elif gating == "mdl":
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
                    diag = {
                        "criterion": "mdl", "ratio": float(ratio),
                        "sim_new": float(sim_new), "sim_old": float(sim_old),
                    }
                else:
                    raise ValueError(f"Unknown gating: {gating}")

                probationary = False
                probation_end_step = None
                if accepted:
                    branch_model.d_t = 4
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    active_transition_accepted = True
                    active_transition_accepted_step = step
                    final_gating_dict = diag
                else:
                    # Revert to d_t = 3, schedule next probation 50 steps later.
                    branch_model.d_t = 3
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    next_proposal_check = step + 50
                    final_gating_dict = diag  # keep latest diagnostics

        # ---- Periodic step log ----
        if step % 500 == 0:
            print(f"       [active step {step:4d}] sim_loss={sim_loss_val:.5f} "
                  f"S_bar={S_bar:.5f} lambda_sp={lambda_val:.4f} "
                  f"d_t={branch_model.d_t} probationary={probationary}")

        # ---- Periodic evaluation checkpoints ----
        if step in eval_steps:
            eval_res = evaluate_branch(branch_model, seed, device)
            checkpoint_results[step] = eval_res
            print(f"       [eval @ {step}] test_sim_loss={eval_res['test_sim_loss']:.6f} "
                  f"mse_cent={eval_res['mse_cent']:.3f} "
                  f"mse_overall={eval_res['mse_cent_overall']:.3f} "
                  f"mean_var_3={eval_res['mean_var_3']:.3f} "
                  f"std_vel_3={eval_res['std_vel_3']:.3f}")

    eval_metrics = evaluate_branch(branch_model, seed, device)
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
        "probation_attempts": probation_attempts,
        "final_gating_dict": final_gating_dict,
        "final_d_t": int(branch_model.d_t),
        "S_bar_end_active": float(S_bar),
    }


# ----------------------------------------------------------------------------
# Welch / Levene helpers
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
    print("PHASE 16 SWEEP: PROBATIONARY WARM-UP (PVU / MDL) vs BASELINE vs ORIGINAL MDL")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = [42, 123, 456, 789, 999]

    # Arm definitions. variant tells us which cached passive state to clone.
    # (arm_name, variant, wup_window, gating)
    arms = [
        ("Arm K (Baseline)",              "standard", None, None),
        ("Arm N (Original MDL)",          "arm_n",    None, None),
        ("Arm O (WUP-PVU, W=100)",        "standard", 100,  "pvu"),
        ("Arm O_big (WUP-PVU, W=500)",    "standard", 500,  "pvu"),
        ("Arm P (WUP-MDL, W=100)",        "standard", 100,  "mdl"),
        ("Arm P_big (WUP-MDL, W=500)",    "standard", 500,  "mdl"),
    ]

    results_list = []
    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_losses    = {a[0]: {s: [] for s in eval_steps} for a in arms}
    checkpoint_mse_cent  = {a[0]: {s: [] for s in eval_steps} for a in arms}
    checkpoint_std_vel   = {a[0]: {s: [] for s in eval_steps} for a in arms}

    for seed in seeds:
        print("\n" + "-" * 50)
        print(f"SEED {seed}")
        print("-" * 50)

        # ----- Train + cache the TWO passive states needed for this seed -----
        print(f"  Building passive cache for seed {seed} (2 variants)...")
        passive_cache = {}
        passive_state = {}
        for variant in ("standard", "arm_n"):
            model_passive, dual_state = train_passive_cached(seed, device, variant)
            passive_cache[variant] = model_passive
            passive_state[variant] = dual_state
            base_eval = evaluate_branch(model_passive, seed, device)
            print(f"     [cache/{variant}] d_t_after_passive="
                  f"{dual_state['final_d_t_after_passive']}, "
                  f"passive_accepted={dual_state['passive_transition_accepted']}, "
                  f"passive_ratio={dual_state['passive_transition_ratio']}, "
                  f"test_sim_loss={base_eval['test_sim_loss']:.6f}, "
                  f"mse_cent={base_eval['mse_cent']:.3f}")

        # ----- Run the active phase for each arm, cloning from the cache -----
        for arm_name, variant, wup_window, gating in arms:
            print(f"\n  [{arm_name}] (variant={variant}, "
                  f"WUP={wup_window}, gating={gating})")

            base_model_template = passive_cache[variant]
            dual_state = passive_state[variant]

            # Step-1500 checkpoint is the cached state (same for all arms using
            # the same variant, so cheap to re-evaluate):
            base_eval = evaluate_branch(base_model_template, seed, device)
            checkpoint_losses[arm_name][1500].append(base_eval["test_sim_loss"])
            checkpoint_mse_cent[arm_name][1500].append(base_eval["mse_cent"])
            checkpoint_std_vel[arm_name][1500].append(base_eval["std_vel_3"])

            branch_results = run_active_branch(
                base_model_template, seed, device, arm_name, dual_state,
                wup_window=wup_window, gating=gating,
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
                "variant": variant,
                "wup_window": wup_window if wup_window is not None else -1,
                "gating": gating if gating is not None else "none",
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
            })
            print(
                f"     Done. Sim Loss: {em['test_sim_loss']:.6f}, "
                f"MSE Cent (post): {em['mse_cent']:.4f}, "
                f"MSE Cent (overall): {em['mse_cent_overall']:.4f}, "
                f"Soft Var: {em['mean_var_3']:.4f}, "
                f"std_vel_3: {em['std_vel_3']:.4f}, "
                f"mean_abs_vel_3: {em['mean_abs_vel_3']:.4f}, "
                f"Pointer Entropy: {branch_results['pointer_entropy']:.4f}, "
                f"final d_t: {branch_results['final_d_t']}"
            )

        # Free the cache for this seed before moving on.
        del passive_cache
        del passive_state

    # ------------------------------------------------------------------
    # Save CSV summary
    # ------------------------------------------------------------------
    results_dir = "archive/iter_016/results"
    os.makedirs(results_dir, exist_ok=True)
    summary_df = pd.DataFrame(results_list)
    summary_csv_path = os.path.join(results_dir, "summary_phase16.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved {summary_csv_path}")

    # ------------------------------------------------------------------
    # Plot adaptation curves (test_sim_loss vs step) across all 6 arms
    # ------------------------------------------------------------------
    avg_checkpoint_losses = {}
    for arm_name, _, _, _ in arms:
        avg_checkpoint_losses[arm_name] = [
            float(np.mean(checkpoint_losses[arm_name][step])) for step in eval_steps
        ]

    plt.figure(figsize=(11, 6))
    colors_dict = {
        "Arm K (Baseline)":              "green",
        "Arm N (Original MDL)":          "purple",
        "Arm O (WUP-PVU, W=100)":        "blue",
        "Arm O_big (WUP-PVU, W=500)":    "navy",
        "Arm P (WUP-MDL, W=100)":        "orange",
        "Arm P_big (WUP-MDL, W=500)":    "red",
    }
    for arm_name, _, _, _ in arms:
        plt.plot(
            eval_steps, avg_checkpoint_losses[arm_name],
            marker='o', label=arm_name, color=colors_dict[arm_name], linewidth=2,
        )
    plt.title("Phase 16: Adaptation Trajectory (Offline Test Sim Loss over Steps)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Offline Test Simulation Loss", fontsize=12)
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    plot_path = os.path.join(results_dir, "adaptation_curves_phase16.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path}")

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (final eval at step 3000)")
    print("=" * 80)

    def arr(arm_name, field):
        return np.array([r[field] for r in results_list if r["arm"] == arm_name], dtype=float)

    arm_names = [a[0] for a in arms]
    metrics_by_arm = {}
    for am in arm_names:
        metrics_by_arm[am] = {
            "test_sim_loss": arr(am, "test_sim_loss"),
            "mse_cent":      arr(am, "mse_cent"),
            "mean_var_3":    arr(am, "mean_var_3"),
        }

    for field in ("test_sim_loss", "mse_cent", "mean_var_3"):
        print(f"\n--- {field} ---")
        for am in arm_names:
            v = metrics_by_arm[am][field]
            print(f"  {am:<32s} mean={v.mean():.6f}  std={v.std(ddof=1):.6f}  values={v.tolist()}")

    # Pre-registered Welch + Levene comparisons:
    # Arm O   vs Arm K, Arm O   vs Arm N
    # Arm O_big vs Arm K, Arm O_big vs Arm N
    comparisons = [
        ("Arm O (WUP-PVU, W=100)",     "Arm K (Baseline)"),
        ("Arm O (WUP-PVU, W=100)",     "Arm N (Original MDL)"),
        ("Arm O_big (WUP-PVU, W=500)", "Arm K (Baseline)"),
        ("Arm O_big (WUP-PVU, W=500)", "Arm N (Original MDL)"),
    ]
    ttest_results = {}
    levene_results = {}
    print("\nWelch's two-sample two-sided t-tests + Levene tests:")
    for arm_a, arm_b in comparisons:
        for field in ("test_sim_loss", "mse_cent", "mean_var_3"):
            a = metrics_by_arm[arm_a][field]
            b = metrics_by_arm[arm_b][field]
            t, p = safe_ttest(a, b)
            w, p_lev = safe_levene(a, b)
            ttest_results[f"{arm_a}_vs_{arm_b}_{field}"] = {"t": t, "p": p}
            levene_results[f"{arm_a}_vs_{arm_b}_{field}"] = {"W": w, "p": p_lev}
            print(f"  {arm_a} vs {arm_b} | {field:<14s}  "
                  f"Welch t={t:.4f} p={p:.6f}    Levene W={w:.4f} p={p_lev:.6f}")

    # ------------------------------------------------------------------
    # Pre-registered falsification audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCIENTIFIC AUDIT: PRE-REGISTERED FALSIFICATION CRITERIA")
    print("=" * 80)

    agg = summary_df.groupby("arm").mean(numeric_only=True).reset_index()
    print(agg.to_string(index=False))

    def recruitment_count(arm_name):
        return int(sum(
            1 for r in results_list
            if r["arm"] == arm_name and r["final_d_t"] >= 4
        ))

    pvu_arms = ["Arm O (WUP-PVU, W=100)", "Arm O_big (WUP-PVU, W=500)"]
    falsification_per_arm = {}
    for arm_name in pvu_arms:
        rec_n = recruitment_count(arm_name)
        mse_mean = metrics_by_arm[arm_name]["mse_cent"].mean()
        var_mean = metrics_by_arm[arm_name]["mean_var_3"].mean()
        sim_mean = metrics_by_arm[arm_name]["test_sim_loss"].mean()
        sim_k = metrics_by_arm["Arm K (Baseline)"]["test_sim_loss"]
        sim_a = metrics_by_arm[arm_name]["test_sim_loss"]
        _, p_sim_vs_k = safe_ttest(sim_a, sim_k)

        # max_corr per seed: read back from the last probation attempt of each
        # seed run that was a PVU criterion. We approximate via results_list's
        # final_gating field (json).
        max_corrs = []
        for r in results_list:
            if r["arm"] != arm_name or not r["final_gating"]:
                continue
            try:
                d = json.loads(r["final_gating"])
                if d.get("criterion") == "pvu":
                    max_corrs.append(float(d.get("max_corr", np.nan)))
            except Exception:
                pass
        max_corr_mean = float(np.nanmean(max_corrs)) if max_corrs else float('nan')

        c1 = bool(rec_n < 3)  # recruitment < 60%
        c2 = bool(mse_mean >= 70.0)
        c3 = bool((not np.isnan(max_corr_mean) and max_corr_mean >= 0.8) or var_mean <= 1e-3)
        c4 = bool((p_sim_vs_k < 0.05) and (sim_mean > 0.15))

        any_falsified = bool(c1 or c2 or c3 or c4)
        falsification_per_arm[arm_name] = {
            "recruitment_count": rec_n,
            "mse_cent_mean":     float(mse_mean),
            "mean_var_3_mean":   float(var_mean),
            "max_corr_mean":     max_corr_mean,
            "test_sim_loss_mean": float(sim_mean),
            "welch_sim_vs_K_p":  float(p_sim_vs_k),
            "c1_recruitment_falsified": c1,
            "c2_mse_threshold_falsified": c2,
            "c3_redundancy_or_collapse_falsified": c3,
            "c4_sim_loss_inflation_falsified": c4,
            "any_falsified": any_falsified,
        }

        print(f"\n[{arm_name}] pre-registered audit:")
        print(f"  Recruitment count    : {rec_n} / 5   "
              f"-> {'FALSIFIED (<3)' if c1 else 'OK'}")
        print(f"  Mean mse_cent        : {mse_mean:.4f}  threshold < 70.0  "
              f"-> {'FALSIFIED (>=70)' if c2 else 'OK'}")
        print(f"  Mean mean_var_3      : {var_mean:.6f}, mean max_corr: {max_corr_mean:.4f}  "
              f"-> {'FALSIFIED (collapse/redundant)' if c3 else 'OK'}")
        print(f"  Mean test_sim_loss   : {sim_mean:.6f}  Welch p vs K: {p_sim_vs_k:.6f}  "
              f"-> {'FALSIFIED (sim loss inflated)' if c4 else 'OK'}")
        print(f"  OVERALL              : {'FALSIFIED' if any_falsified else 'VALIDATED'}")

    # ------------------------------------------------------------------
    # Save audit JSON
    # ------------------------------------------------------------------
    audit_dict = {
        "arms": [a[0] for a in arms],
        "seeds": seeds,
        "eval_steps": eval_steps,
        "means": {
            am: {
                "test_sim_loss": float(metrics_by_arm[am]["test_sim_loss"].mean()),
                "mse_cent":      float(metrics_by_arm[am]["mse_cent"].mean()),
                "mean_var_3":    float(metrics_by_arm[am]["mean_var_3"].mean()),
            } for am in arm_names
        },
        "stds": {
            am: {
                "test_sim_loss": float(metrics_by_arm[am]["test_sim_loss"].std(ddof=1)),
                "mse_cent":      float(metrics_by_arm[am]["mse_cent"].std(ddof=1)),
                "mean_var_3":    float(metrics_by_arm[am]["mean_var_3"].std(ddof=1)),
            } for am in arm_names
        },
        "ttests": ttest_results,
        "levene": levene_results,
        "falsification_per_arm": falsification_per_arm,
        "checkpoint_means": {
            am: {
                "test_sim_loss": [float(np.mean(checkpoint_losses[am][s])) for s in eval_steps],
                "mse_cent":      [float(np.mean(checkpoint_mse_cent[am][s])) for s in eval_steps],
                "std_vel_3":     [float(np.mean(checkpoint_std_vel[am][s])) for s in eval_steps],
            } for am in arm_names
        },
    }
    audit_path = os.path.join(results_dir, "audit_results_phase16.json")
    with open(audit_path, "w") as f:
        json.dump(audit_dict, f, indent=4)
    print(f"\nSaved {audit_path}")

    # ------------------------------------------------------------------
    # Manager summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MANAGER SUMMARY")
    print("=" * 80)
    for am in arm_names:
        v_mse = metrics_by_arm[am]["mse_cent"]
        v_sim = metrics_by_arm[am]["test_sim_loss"]
        v_var = metrics_by_arm[am]["mean_var_3"]
        print(f"  {am:<32s} mse_cent={v_mse.mean():.3f} ± {v_mse.std(ddof=1):.3f}  "
              f"sim={v_sim.mean():.4f} ± {v_sim.std(ddof=1):.4f}  "
              f"var={v_var.mean():.3f} ± {v_var.std(ddof=1):.3f}")
    print()
    for arm_name in pvu_arms:
        f = falsification_per_arm[arm_name]
        verdict = "FALSIFIED" if f["any_falsified"] else "VALIDATED"
        print(f"  {arm_name} hypothesis: {verdict}  "
              f"(recruit={f['recruitment_count']}/5, "
              f"mse={f['mse_cent_mean']:.3f}, "
              f"sim={f['test_sim_loss_mean']:.4f}, "
              f"max_corr={f['max_corr_mean']:.4f}, "
              f"var={f['mean_var_3_mean']:.3f})")


if __name__ == "__main__":
    main()
