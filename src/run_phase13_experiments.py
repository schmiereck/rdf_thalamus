import os
import sys
import csv
import json
import random
import collections
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models_dual_stream import NonParametricJEPASpatial
from src.motor import CLTSMotorController


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


def evaluate_branch_phase13(model, seed, device):
    set_seed(seed + 5000)
    test_env = PhysicsSandbox(N=4, seed=seed + 5000)
    test_env.masses[3] *= 2.0  # 2x mass perturbation on the novel 4th object
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
        loss_dict, _, _ = model(test_x_hist_t, test_x_target_t)
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
    y_pred_cent = x_mean_3[100:] * w_cent + b_cent
    mse_cent = np.mean((y_probe_test - y_pred_cent) ** 2)

    r_centroid = np.corrcoef(x_mean_3, test_y_4_arr)[0, 1]
    abs_r_centroid = abs(r_centroid) if not np.isnan(r_centroid) else 0.0

    mean_var_3 = np.mean(var_3)
    std_x_mean_3 = np.std(x_mean_3)

    has_collapsed = not (e_a_3 >= 0.1 * e_a_all and std_x_mean_3 > 5.0)

    return {
        "test_sim_loss": test_sim_loss,
        "abs_r_centroid": abs_r_centroid,
        "mse_cent": mse_cent,
        "mean_var_3": mean_var_3,
        "std_x_mean_3": std_x_mean_3,
        "collapsed": has_collapsed,
    }


def train_base_model_passive(seed, device, pos_encoding):
    set_seed(seed)
    model = NonParametricJEPASpatial(
        d_max=8, h=3, k=4, cooldown=300, stabilization_period=100, pos_encoding=pos_encoding
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
        loss_dict, _, _ = model(x_hist_t, x_target_t)
        loss_dict["loss"].backward()
        optimizer.step()

        sim_loss_val = loss_dict["sim_loss"].item()

        if step > 200:
            model.update_recruitment_logic(sim_loss_val, target_dim=2)

        if model.d_t == 2 and step >= 600:
            model.d_t = 3
            model.steps_since_recruitment = 0
            model.reset_error_buffer()

        if step == 1001:
            model.reset_error_buffer()
        if step > 1000:
            model.update_recruitment_logic(sim_loss_val, target_dim=3)

    return model


def run_active_branch(branch_model, seed, device, arm_name):
    """Run the active CLTS training phase from step 1501 to 3000 for a given branch model."""
    branch_optimizer = optim.Adam(branch_model.parameters(), lr=1e-3)

    set_seed(seed + 1000)
    branch_env = PhysicsSandbox(N=4, seed=seed + 1000)
    branch_env.masses[3] *= 2.0  # Apply 2x mass perturbation

    branch_obs = branch_env.reset()
    branch_history = collections.deque(maxlen=4)
    branch_history.append(branch_obs)

    branch_replay = ReplayBuffer(capacity=2000)
    last_info = prefill_buffer_passive(branch_env, branch_replay, branch_history, num_transitions=100)

    clts_controller = CLTSMotorController()
    clts_controller.reset()

    ewma_surprise = 0.10
    lambda_val = 0.0

    pointer_positions = []
    online_losses = []

    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_results = {}

    for step in range(1501, 3001):
        pointer_positions.append(branch_env.pointer_pos)

        branch_model.eval()
        hist_t = torch.from_numpy(np.stack(list(branch_history)[:3], axis=0)).float().unsqueeze(0).to(device)
        target_t = torch.from_numpy(branch_history[-1]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            loss_dict, (zp_coord, zp_dyn), (zt_coord, zt_dyn) = branch_model(hist_t, target_t)
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

        lambda_target = 0.10 * max(0.0, 1.0 - ewma_surprise / 2.0)
        if step == 1501:
            lambda_val = lambda_target
        else:
            lambda_val = lambda_val + np.clip(lambda_target - lambda_val, -0.002, 0.002)

        branch_optimizer.zero_grad()
        loss_dict, _, _ = branch_model(x_hist_t, x_target_t, lambda_spatial=lambda_val, k_chan=3)
        loss_dict["loss"].backward()
        branch_optimizer.step()

        sim_loss_val = loss_dict["sim_loss"].item()
        online_losses.append(sim_loss_val)
        ewma_surprise = 0.95 * ewma_surprise + 0.05 * sim_loss_val

        branch_model.update_recruitment_logic(sim_loss_val, target_dim=3)
        if branch_model.d_t == 3 and step >= 1800:
            branch_model.d_t = 4
            branch_model.steps_since_recruitment = 0
            branch_model.reset_error_buffer()

        if step in eval_steps:
            eval_res = evaluate_branch_phase13(branch_model, seed, device)
            checkpoint_results[step] = eval_res

    eval_metrics = evaluate_branch_phase13(branch_model, seed, device)
    pointer_entropy = compute_spatial_entropy(pointer_positions)
    online_auc_1501_2000 = sum(online_losses[:500])
    online_auc_1501_3000 = sum(online_losses)

    return {
        "eval_metrics": eval_metrics,
        "pointer_entropy": pointer_entropy,
        "online_auc_1501_2000": online_auc_1501_2000,
        "online_auc_1501_3000": online_auc_1501_3000,
        "checkpoint_results": checkpoint_results,
    }


def main():
    print("=" * 80)
    print("PHASE 13 SWEEP: EVALUATING POSITIONAL ENCODINGS UNDER CLTS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = [42, 123, 456, 789, 999]
    arms = [
        ("Arm G (RGB CLTS)", "none"),
        ("Arm H (Linear CLTS)", "linear"),
        ("Arm I (Sinusoidal CLTS)", "sinusoidal"),
    ]

    results_list = []
    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_losses = {arm_name: {step: [] for step in eval_steps} for arm_name, _ in arms}

    # Evaluate base models at step 1500 inside the loop (they exist after passive pretraining)
    for seed in seeds:
        print(f"\n" + "-" * 50)
        print(f"SEED {seed}")
        print("-" * 50)

        for arm_name, pos_encoding in arms:
            print(f"\n[{arm_name}] (pos_encoding={pos_encoding})")
            print(f"  1. Training passive base model on N=3 with pos_encoding={pos_encoding}...")
            base_model = train_base_model_passive(seed, device, pos_encoding)
            print(f"     Finished passive pretraining. d_t = {base_model.d_t}")

            base_eval = evaluate_branch_phase13(base_model, seed, device)
            checkpoint_losses[arm_name][1500].append(base_eval["test_sim_loss"])
            print(f"     Step 1500 test sim loss: {base_eval['test_sim_loss']:.6f}")

            print(f"  2. Active CLTS training (steps 1501..3000) on N=4 with 2x mass perturbation...")
            branch_results = run_active_branch(base_model, seed, device, arm_name)

            for step in eval_steps:
                if step == 1500:
                    continue
                if step in branch_results["checkpoint_results"]:
                    checkpoint_losses[arm_name][step].append(
                        branch_results["checkpoint_results"][step]["test_sim_loss"]
                    )

            em = branch_results["eval_metrics"]
            results_list.append({
                "seed": seed,
                "arm": arm_name,
                "pos_encoding": pos_encoding,
                "test_sim_loss": em["test_sim_loss"],
                "abs_r_centroid": em["abs_r_centroid"],
                "mse_cent": em["mse_cent"],
                "mean_var_3": em["mean_var_3"],
                "std_x_mean_3": em["std_x_mean_3"],
                "collapsed": int(em["collapsed"]),
                "pointer_entropy": branch_results["pointer_entropy"],
                "online_auc_1501_2000": branch_results["online_auc_1501_2000"],
                "online_auc_1501_3000": branch_results["online_auc_1501_3000"],
            })
            print(
                f"     Done. Sim Loss: {em['test_sim_loss']:.6f}, "
                f"MSE Centroid: {em['mse_cent']:.4f}, "
                f"Soft Var: {em['mean_var_3']:.4f}, "
                f"Pointer Entropy: {branch_results['pointer_entropy']:.4f}"
            )

    # Save CSV summary
    results_dir = "archive/iter_013/results"
    os.makedirs(results_dir, exist_ok=True)
    summary_df = pd.DataFrame(results_list)
    summary_csv_path = os.path.join(results_dir, "summary_phase13.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved {summary_csv_path}")

    # Compute average test losses at checkpoints across seeds
    avg_checkpoint_losses = {}
    for arm_name, _ in arms:
        avg_checkpoint_losses[arm_name] = [np.mean(checkpoint_losses[arm_name][step]) for step in eval_steps]

    # Plot adaptation curves
    plt.figure(figsize=(10, 6))
    colors_dict = {
        "Arm G (RGB CLTS)": "red",
        "Arm H (Linear CLTS)": "blue",
        "Arm I (Sinusoidal CLTS)": "green",
    }
    for arm_name, _ in arms:
        plt.plot(
            eval_steps, avg_checkpoint_losses[arm_name],
            marker='o', label=arm_name, color=colors_dict[arm_name], linewidth=2,
        )
    plt.title("Phase 13: Adaptation Trajectory (Offline Test Sim Loss over Steps)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Standardized Test Simulation Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plot_path = os.path.join(results_dir, "auc_recovery_curves_phase13.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path}")

    # Statistical analysis on step 3000 offline test sim losses
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (step 3000 offline test sim loss)")
    print("=" * 80)

    arm_g_losses = np.array(checkpoint_losses["Arm G (RGB CLTS)"][3000])
    arm_h_losses = np.array(checkpoint_losses["Arm H (Linear CLTS)"][3000])
    arm_i_losses = np.array(checkpoint_losses["Arm I (Sinusoidal CLTS)"][3000])

    print(f"\nArm G (RGB CLTS)        step 3000 losses: {arm_g_losses.tolist()}")
    print(f"  mean = {arm_g_losses.mean():.6f}, std = {arm_g_losses.std(ddof=1):.6f}")
    print(f"Arm H (Linear CLTS)     step 3000 losses: {arm_h_losses.tolist()}")
    print(f"  mean = {arm_h_losses.mean():.6f}, std = {arm_h_losses.std(ddof=1):.6f}")
    print(f"Arm I (Sinusoidal CLTS) step 3000 losses: {arm_i_losses.tolist()}")
    print(f"  mean = {arm_i_losses.mean():.6f}, std = {arm_i_losses.std(ddof=1):.6f}")

    # Two-sample two-sided t-tests
    t_stat_gh, p_value_gh = stats.ttest_ind(arm_g_losses, arm_h_losses, equal_var=False)
    t_stat_gi, p_value_gi = stats.ttest_ind(arm_g_losses, arm_i_losses, equal_var=False)
    print(f"\nTwo-sample two-sided Welch's t-test:")
    print(f"  Arm G vs Arm H: t = {t_stat_gh:.4f}, p-value = {p_value_gh:.6f}")
    print(f"  Arm G vs Arm I: t = {t_stat_gi:.4f}, p-value = {p_value_gi:.6f}")

    # Levene's test for equality of variances
    lev_stat_gh, lev_p_gh = stats.levene(arm_g_losses, arm_h_losses)
    lev_stat_gi, lev_p_gi = stats.levene(arm_g_losses, arm_i_losses)
    print(f"\nLevene's test (equality of variances):")
    print(f"  Arm G vs Arm H: W = {lev_stat_gh:.4f}, p-value = {lev_p_gh:.6f}")
    print(f"  Arm G vs Arm I: W = {lev_stat_gi:.4f}, p-value = {lev_p_gi:.6f}")

    # Pre-registered falsification audit
    print("\n" + "=" * 80)
    print("SCIENTIFIC AUDIT: PRE-REGISTERED FALSIFICATION CRITERIA")
    print("=" * 80)

    agg = summary_df.groupby("arm").mean(numeric_only=True).reset_index()
    print(agg.to_string(index=False))

    g_row = agg[agg["arm"] == "Arm G (RGB CLTS)"].iloc[0]
    h_row = agg[agg["arm"] == "Arm H (Linear CLTS)"].iloc[0]
    i_row = agg[agg["arm"] == "Arm I (Sinusoidal CLTS)"].iloc[0]

    # Criterion 1: Centroid decoding MSE of novel object < 75.0 for Arm H or Arm I
    c1_h_falsified = h_row["mse_cent"] >= 75.0
    c1_i_falsified = i_row["mse_cent"] >= 75.0
    c1_falsified = c1_h_falsified or c1_i_falsified
    print(f"\nCriterion 1 (Coordinate Accuracy):")
    print(f"  Arm G MSE Centroid: {g_row['mse_cent']:.4f} (baseline)")
    print(f"  Arm H MSE Centroid: {h_row['mse_cent']:.4f} (threshold: < 75.0) -> "
          f"{'FALSIFIED' if c1_h_falsified else 'OK'}")
    print(f"  Arm I MSE Centroid: {i_row['mse_cent']:.4f} (threshold: < 75.0) -> "
          f"{'FALSIFIED' if c1_i_falsified else 'OK'}")
    print(f"  Overall: {'FALSIFIED' if c1_falsified else 'VALIDATED'}")

    # Criterion 2: Soft spatial variance < 10.0 for Arm H or Arm I
    c2_h_falsified = h_row["mean_var_3"] >= 10.0
    c2_i_falsified = i_row["mean_var_3"] >= 10.0
    c2_falsified = c2_h_falsified or c2_i_falsified
    print(f"\nCriterion 2 (Spatial Tightness):")
    print(f"  Arm G Soft Var: {g_row['mean_var_3']:.4f} (baseline)")
    print(f"  Arm H Soft Var: {h_row['mean_var_3']:.4f} (threshold: < 10.0) -> "
          f"{'FALSIFIED' if c2_h_falsified else 'OK'}")
    print(f"  Arm I Soft Var: {i_row['mean_var_3']:.4f} (threshold: < 10.0) -> "
          f"{'FALSIFIED' if c2_i_falsified else 'OK'}")
    print(f"  Overall: {'FALSIFIED' if c2_falsified else 'VALIDATED'}")

    # Criterion 3: Post-collision test sim loss < 0.050 for Arm H or Arm I
    c3_h_falsified = h_row["test_sim_loss"] >= 0.050
    c3_i_falsified = i_row["test_sim_loss"] >= 0.050
    c3_falsified = c3_h_falsified or c3_i_falsified
    print(f"\nCriterion 3 (Predictive Integrity):")
    print(f"  Arm G Test Sim Loss: {g_row['test_sim_loss']:.6f} (baseline)")
    print(f"  Arm H Test Sim Loss: {h_row['test_sim_loss']:.6f} (threshold: < 0.050) -> "
          f"{'FALSIFIED' if c3_h_falsified else 'OK'}")
    print(f"  Arm I Test Sim Loss: {i_row['test_sim_loss']:.6f} (threshold: < 0.050) -> "
          f"{'FALSIFIED' if c3_i_falsified else 'OK'}")
    print(f"  Overall: {'FALSIFIED' if c3_falsified else 'VALIDATED'}")

    # Criterion 4: Statistically non-inferior to Arm G
    # Falsified if Arm H/I loss is significantly greater than Arm G (worse) at p <= 0.05
    h_significantly_worse = (h_row["test_sim_loss"] > g_row["test_sim_loss"]) and (p_value_gh <= 0.05)
    i_significantly_worse = (i_row["test_sim_loss"] > g_row["test_sim_loss"]) and (p_value_gi <= 0.05)
    c4_falsified = h_significantly_worse or i_significantly_worse
    print(f"\nCriterion 4 (Generalization / Robustness):")
    print(f"  Arm G mean test sim loss: {g_row['test_sim_loss']:.6f}")
    print(f"  Arm H mean test sim loss: {h_row['test_sim_loss']:.6f}, "
          f"t-test p={p_value_gh:.6f} -> "
          f"{'FALSIFIED (Arm H significantly worse)' if h_significantly_worse else 'OK'}")
    print(f"  Arm I mean test sim loss: {i_row['test_sim_loss']:.6f}, "
          f"t-test p={p_value_gi:.6f} -> "
          f"{'FALSIFIED (Arm I significantly worse)' if i_significantly_worse else 'OK'}")
    print(f"  Overall: {'FALSIFIED' if c4_falsified else 'VALIDATED'}")

    hypothesis_falsified = c1_falsified or c2_falsified or c3_falsified or c4_falsified
    print("\n" + "=" * 80)
    print(f"OVERALL HYPOTHESIS: {'FALSIFIED' if hypothesis_falsified else 'VALIDATED'}")
    print("=" * 80)

    # Save audit results JSON
    audit_dict = {
        "arm_g_test_sim_loss_mean": float(g_row["test_sim_loss"]),
        "arm_h_test_sim_loss_mean": float(h_row["test_sim_loss"]),
        "arm_i_test_sim_loss_mean": float(i_row["test_sim_loss"]),
        "arm_g_mse_cent_mean": float(g_row["mse_cent"]),
        "arm_h_mse_cent_mean": float(h_row["mse_cent"]),
        "arm_i_mse_cent_mean": float(i_row["mse_cent"]),
        "arm_g_mean_var_3_mean": float(g_row["mean_var_3"]),
        "arm_h_mean_var_3_mean": float(h_row["mean_var_3"]),
        "arm_i_mean_var_3_mean": float(i_row["mean_var_3"]),
        "arm_g_pointer_entropy_mean": float(g_row["pointer_entropy"]),
        "arm_h_pointer_entropy_mean": float(h_row["pointer_entropy"]),
        "arm_i_pointer_entropy_mean": float(i_row["pointer_entropy"]),
        "ttest_gh_t": float(t_stat_gh),
        "ttest_gh_p": float(p_value_gh),
        "ttest_gi_t": float(t_stat_gi),
        "ttest_gi_p": float(p_value_gi),
        "levene_gh_W": float(lev_stat_gh),
        "levene_gh_p": float(lev_p_gh),
        "levene_gi_W": float(lev_stat_gi),
        "levene_gi_p": float(lev_p_gi),
        "c1_falsified": bool(c1_falsified),
        "c2_falsified": bool(c2_falsified),
        "c3_falsified": bool(c3_falsified),
        "c4_falsified": bool(c4_falsified),
        "hypothesis_falsified": bool(hypothesis_falsified),
    }
    audit_path = os.path.join(results_dir, "audit_results_phase13.json")
    with open(audit_path, "w") as f:
        json.dump(audit_dict, f, indent=4)
    print(f"\nSaved {audit_path}")


if __name__ == "__main__":
    main()
