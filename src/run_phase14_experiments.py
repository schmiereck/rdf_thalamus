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


def evaluate_branch_phase14(model, seed, device):
    """
    Evaluate a (trained) branch model on a 200-frame N=4 test sequence with a 2x mass perturbation
    on the novel object (channel 3). Returns:
      - test_sim_loss
      - mse_cent (centroid decoding MSE on the 100-frame held-out portion)
      - mean_var_3 (mean soft spatial variance of channel 3)
      - std_x_mean_3
      - std_vel_3, mean_abs_vel_3 (coordinate velocity metrics of novel object)
      - abs_r_centroid (kept for legacy)
      - collapsed flag (kept for legacy)
    """
    set_seed(seed + 5000)
    test_env = PhysicsSandbox(N=4, seed=seed + 5000)
    test_env.masses[3] *= 2.0  # 2x mass perturbation on the novel 4th object
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
        # Use ccr_mode='none' for evaluation (we only care about sim_loss as the predictive metric)
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
    y_pred_cent = x_mean_3[100:] * w_cent + b_cent
    mse_cent = np.mean((y_probe_test - y_pred_cent) ** 2)

    r_centroid = np.corrcoef(x_mean_3, test_y_4_arr)[0, 1]
    abs_r_centroid = abs(r_centroid) if not np.isnan(r_centroid) else 0.0

    mean_var_3 = float(np.mean(var_3))
    std_x_mean_3 = float(np.std(x_mean_3))

    # Coordinate velocity metrics for the 'lazy encoder' diagnosis
    vel_3 = x_mean_3[1:] - x_mean_3[:-1]
    std_vel_3 = float(np.std(vel_3))
    mean_abs_vel_3 = float(np.mean(np.abs(vel_3)))

    has_collapsed = not (e_a_3 >= 0.1 * e_a_all and std_x_mean_3 > 5.0)

    return {
        "test_sim_loss": float(test_sim_loss),
        "abs_r_centroid": float(abs_r_centroid),
        "mse_cent": float(mse_cent),
        "mean_var_3": mean_var_3,
        "std_x_mean_3": std_x_mean_3,
        "std_vel_3": std_vel_3,
        "mean_abs_vel_3": mean_abs_vel_3,
        "collapsed": bool(has_collapsed),
    }


def train_base_model_passive(seed, device, ccr_mode):
    """Passive training on N=3 for steps 1..1500, applying the given ccr_mode."""
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

    print(f"     [passive] starting passive training on N=3 (steps 1..1500), ccr_mode={ccr_mode}")
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
            ccr_mode=ccr_mode,
            ccr_smooth_weight=10.0,
            ccr_spatial_weight=10.0,
        )
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

        if step % 250 == 0:
            print(f"       [passive step {step:4d}] sim_loss={sim_loss_val:.5f} "
                  f"ccr_smooth={loss_dict['ccr_smooth_loss'].item():.5f} "
                  f"ccr_spatial={loss_dict['ccr_spatial_loss'].item():.5f} "
                  f"d_t={model.d_t}")

    return model


def run_active_branch(branch_model, seed, device, arm_name, ccr_mode):
    """Active CLTS training (steps 1501..3000) on N=4 with the given ccr_mode."""
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

    pointer_positions = []
    online_losses = []

    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_results = {}

    print(f"     [active] starting active CLTS training on N=4 (steps 1501..3000), ccr_mode={ccr_mode}")
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

        lambda_target = 0.10 * max(0.0, 1.0 - ewma_surprise / 2.0)
        if step == 1501:
            lambda_val = lambda_target
        else:
            lambda_val = lambda_val + np.clip(lambda_target - lambda_val, -0.002, 0.002)

        branch_optimizer.zero_grad()
        loss_dict, _, _ = branch_model(
            x_hist_t, x_target_t,
            lambda_spatial=lambda_val, k_chan=3,
            ccr_mode=ccr_mode,
            ccr_smooth_weight=10.0,
            ccr_spatial_weight=10.0,
        )
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

        if step % 250 == 0:
            print(f"       [active step {step:4d}] sim_loss={sim_loss_val:.5f} "
                  f"ccr_smooth={loss_dict['ccr_smooth_loss'].item():.5f} "
                  f"ccr_spatial={loss_dict['ccr_spatial_loss'].item():.5f} "
                  f"lambda_sp={lambda_val:.4f} d_t={branch_model.d_t}")

        if step in eval_steps:
            eval_res = evaluate_branch_phase14(branch_model, seed, device)
            checkpoint_results[step] = eval_res
            print(f"       [eval @ {step}] test_sim_loss={eval_res['test_sim_loss']:.6f} "
                  f"mse_cent={eval_res['mse_cent']:.3f} mean_var_3={eval_res['mean_var_3']:.3f} "
                  f"std_vel_3={eval_res['std_vel_3']:.3f} mean_abs_vel_3={eval_res['mean_abs_vel_3']:.3f}")

    eval_metrics = evaluate_branch_phase14(branch_model, seed, device)
    pointer_entropy = compute_spatial_entropy(pointer_positions)
    online_auc_1501_2000 = float(sum(online_losses[:500]))
    online_auc_1501_3000 = float(sum(online_losses))

    return {
        "eval_metrics": eval_metrics,
        "pointer_entropy": float(pointer_entropy),
        "online_auc_1501_2000": online_auc_1501_2000,
        "online_auc_1501_3000": online_auc_1501_3000,
        "checkpoint_results": checkpoint_results,
    }


def main():
    print("=" * 80)
    print("PHASE 14 SWEEP: CONTRASTIVE COORDINATE REGULARIZATION (CCR) UNDER CLTS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = [42, 123, 456, 789, 999]
    arms = [
        ("Arm G (RGB CLTS)", "none"),
        ("Arm J (CCR-Hinge)", "hinge"),
        ("Arm K (CCR-Covariance)", "covariance"),
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
            print(f"  1. Training passive base model on N=3 with ccr_mode={ccr_mode}...")
            base_model = train_base_model_passive(seed, device, ccr_mode)
            print(f"     Finished passive pretraining. d_t = {base_model.d_t}")

            base_eval = evaluate_branch_phase14(base_model, seed, device)
            checkpoint_losses[arm_name][1500].append(base_eval["test_sim_loss"])
            checkpoint_mse_cent[arm_name][1500].append(base_eval["mse_cent"])
            checkpoint_std_vel[arm_name][1500].append(base_eval["std_vel_3"])
            print(f"     Step 1500 base eval: "
                  f"test_sim_loss={base_eval['test_sim_loss']:.6f} "
                  f"mse_cent={base_eval['mse_cent']:.3f} "
                  f"std_vel_3={base_eval['std_vel_3']:.3f}")

            print(f"  2. Active CLTS training (steps 1501..3000) on N=4 with 2x mass perturbation, "
                  f"ccr_mode={ccr_mode}...")
            branch_results = run_active_branch(base_model, seed, device, arm_name, ccr_mode)

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
                "mean_var_3": em["mean_var_3"],
                "std_x_mean_3": em["std_x_mean_3"],
                "std_vel_3": em["std_vel_3"],
                "mean_abs_vel_3": em["mean_abs_vel_3"],
                "collapsed": int(em["collapsed"]),
                "pointer_entropy": branch_results["pointer_entropy"],
                "online_auc_1501_2000": branch_results["online_auc_1501_2000"],
                "online_auc_1501_3000": branch_results["online_auc_1501_3000"],
            })
            print(
                f"     Done. Sim Loss: {em['test_sim_loss']:.6f}, "
                f"MSE Centroid: {em['mse_cent']:.4f}, "
                f"Soft Var: {em['mean_var_3']:.4f}, "
                f"std_vel_3: {em['std_vel_3']:.4f}, "
                f"mean_abs_vel_3: {em['mean_abs_vel_3']:.4f}, "
                f"Pointer Entropy: {branch_results['pointer_entropy']:.4f}"
            )

    # Save CSV summary
    results_dir = "archive/iter_14/results"
    os.makedirs(results_dir, exist_ok=True)
    summary_df = pd.DataFrame(results_list)
    summary_csv_path = os.path.join(results_dir, "summary_phase14.csv")
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
        "Arm J (CCR-Hinge)": "blue",
        "Arm K (CCR-Covariance)": "green",
    }
    for arm_name, _ in arms:
        plt.plot(
            eval_steps, avg_checkpoint_losses[arm_name],
            marker='o', label=arm_name, color=colors_dict[arm_name], linewidth=2,
        )
    plt.title("Phase 14: Adaptation Trajectory (Offline Test Sim Loss over Steps)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Offline Test Simulation Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plot_path = os.path.join(results_dir, "adaptation_curves_phase14.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved {plot_path}")

    # Statistical analysis on step 3000 offline test sim losses
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (step 3000 offline test sim loss)")
    print("=" * 80)

    arm_g_losses = np.array(checkpoint_losses["Arm G (RGB CLTS)"][3000])
    arm_j_losses = np.array(checkpoint_losses["Arm J (CCR-Hinge)"][3000])
    arm_k_losses = np.array(checkpoint_losses["Arm K (CCR-Covariance)"][3000])

    print(f"\nArm G (RGB CLTS)         step 3000 losses: {arm_g_losses.tolist()}")
    print(f"  mean = {arm_g_losses.mean():.6f}, std = {arm_g_losses.std(ddof=1):.6f}")
    print(f"Arm J (CCR-Hinge)        step 3000 losses: {arm_j_losses.tolist()}")
    print(f"  mean = {arm_j_losses.mean():.6f}, std = {arm_j_losses.std(ddof=1):.6f}")
    print(f"Arm K (CCR-Covariance)   step 3000 losses: {arm_k_losses.tolist()}")
    print(f"  mean = {arm_k_losses.mean():.6f}, std = {arm_k_losses.std(ddof=1):.6f}")

    # Welch's two-sample two-sided t-test
    t_stat_gj, p_value_gj = stats.ttest_ind(arm_g_losses, arm_j_losses, equal_var=False)
    t_stat_gk, p_value_gk = stats.ttest_ind(arm_g_losses, arm_k_losses, equal_var=False)
    print(f"\nWelch's two-sample two-sided t-test:")
    print(f"  Arm G vs Arm J: t = {t_stat_gj:.4f}, p-value = {p_value_gj:.6f}")
    print(f"  Arm G vs Arm K: t = {t_stat_gk:.4f}, p-value = {p_value_gk:.6f}")

    # Levene's test for equality of variances
    lev_stat_gj, lev_p_gj = stats.levene(arm_g_losses, arm_j_losses)
    lev_stat_gk, lev_p_gk = stats.levene(arm_g_losses, arm_k_losses)
    print(f"\nLevene's test (equality of variances):")
    print(f"  Arm G vs Arm J: W = {lev_stat_gj:.4f}, p-value = {lev_p_gj:.6f}")
    print(f"  Arm G vs Arm K: W = {lev_stat_gk:.4f}, p-value = {lev_p_gk:.6f}")

    # Pre-registered falsification audit
    print("\n" + "=" * 80)
    print("SCIENTIFIC AUDIT: PRE-REGISTERED FALSIFICATION CRITERIA")
    print("=" * 80)

    agg = summary_df.groupby("arm").mean(numeric_only=True).reset_index()
    print(agg.to_string(index=False))

    g_row = agg[agg["arm"] == "Arm G (RGB CLTS)"].iloc[0]
    j_row = agg[agg["arm"] == "Arm J (CCR-Hinge)"].iloc[0]
    k_row = agg[agg["arm"] == "Arm K (CCR-Covariance)"].iloc[0]

    # Identify the best CCR arm by mean centroid decoding MSE (lower is better)
    best_ccr_arm_name = "Arm J (CCR-Hinge)" if j_row["mse_cent"] <= k_row["mse_cent"] else "Arm K (CCR-Covariance)"
    best_row = j_row if best_ccr_arm_name == "Arm J (CCR-Hinge)" else k_row
    print(f"\nBest CCR arm (by mse_cent): {best_ccr_arm_name}")

    # Criterion 1: Best CCR arm centroid MSE >= 70.0 -> falsified
    c1_falsified = bool(best_row["mse_cent"] >= 70.0)
    print(f"\nCriterion 1 (Coordinate Accuracy):")
    print(f"  Arm G MSE Centroid: {g_row['mse_cent']:.4f} (baseline)")
    print(f"  Arm J MSE Centroid: {j_row['mse_cent']:.4f}")
    print(f"  Arm K MSE Centroid: {k_row['mse_cent']:.4f}")
    print(f"  Best CCR arm MSE: {best_row['mse_cent']:.4f} (falsification threshold: >= 70.0) -> "
          f"{'FALSIFIED' if c1_falsified else 'VALIDATED'}")

    # Criterion 2: Best CCR arm test sim loss > 0.050 -> falsified
    c2_falsified = bool(best_row["test_sim_loss"] > 0.050)
    print(f"\nCriterion 2 (Predictive Integrity at step 3000):")
    print(f"  Arm G Test Sim Loss: {g_row['test_sim_loss']:.6f} (baseline)")
    print(f"  Arm J Test Sim Loss: {j_row['test_sim_loss']:.6f}")
    print(f"  Arm K Test Sim Loss: {k_row['test_sim_loss']:.6f}")
    print(f"  Best CCR arm Loss: {best_row['test_sim_loss']:.6f} (falsification threshold: > 0.050) -> "
          f"{'FALSIFIED' if c2_falsified else 'VALIDATED'}")

    # Criterion 3: Pointer spatial entropy drops below 3.5 for the best CCR arm -> falsified
    c3_falsified = bool(best_row["pointer_entropy"] < 3.5)
    print(f"\nCriterion 3 (Exploratory Behavior / Pointer Entropy):")
    print(f"  Arm G Pointer Entropy: {g_row['pointer_entropy']:.4f} (baseline)")
    print(f"  Arm J Pointer Entropy: {j_row['pointer_entropy']:.4f}")
    print(f"  Arm K Pointer Entropy: {k_row['pointer_entropy']:.4f}")
    print(f"  Best CCR arm Entropy: {best_row['pointer_entropy']:.4f} (falsification threshold: < 3.5) -> "
          f"{'FALSIFIED' if c3_falsified else 'VALIDATED'}")

    # Criterion 4: Soft spatial variance of the coordinate encoder > 10.0 for the best CCR arm -> falsified
    c4_falsified = bool(best_row["mean_var_3"] > 10.0)
    print(f"\nCriterion 4 (Spatial Tightness):")
    print(f"  Arm G Soft Var: {g_row['mean_var_3']:.4f} (baseline)")
    print(f"  Arm J Soft Var: {j_row['mean_var_3']:.4f}")
    print(f"  Arm K Soft Var: {k_row['mean_var_3']:.4f}")
    print(f"  Best CCR arm Soft Var: {best_row['mean_var_3']:.4f} (falsification threshold: > 10.0) -> "
          f"{'FALSIFIED' if c4_falsified else 'VALIDATED'}")

    # Criterion 5: Lazy encoder failure mode (std_vel_3 < 1.5 AND mse_cent >= 70.0) for the best CCR arm
    c5_falsified = bool(best_row["std_vel_3"] < 1.5 and best_row["mse_cent"] >= 70.0)
    print(f"\nCriterion 5 (Lazy Encoder Failure Mode):")
    print(f"  Arm G std_vel_3: {g_row['std_vel_3']:.4f}, mean_abs_vel_3: {g_row['mean_abs_vel_3']:.4f}")
    print(f"  Arm J std_vel_3: {j_row['std_vel_3']:.4f}, mean_abs_vel_3: {j_row['mean_abs_vel_3']:.4f}")
    print(f"  Arm K std_vel_3: {k_row['std_vel_3']:.4f}, mean_abs_vel_3: {k_row['mean_abs_vel_3']:.4f}")
    print(f"  Best CCR arm std_vel_3: {best_row['std_vel_3']:.4f} (threshold for lazy: < 1.5), "
          f"mse_cent: {best_row['mse_cent']:.4f} (threshold for high MSE: >= 70.0) -> "
          f"{'FALSIFIED (lazy encoder detected)' if c5_falsified else 'OK (no lazy encoder)'}")

    hypothesis_falsified = c1_falsified or c2_falsified or c3_falsified or c4_falsified or c5_falsified
    print("\n" + "=" * 80)
    print(f"OVERALL HYPOTHESIS: {'FALSIFIED' if hypothesis_falsified else 'VALIDATED'}")
    print("=" * 80)

    # Save audit results JSON
    audit_dict = {
        "best_ccr_arm": best_ccr_arm_name,
        "arm_g_test_sim_loss_mean": float(g_row["test_sim_loss"]),
        "arm_j_test_sim_loss_mean": float(j_row["test_sim_loss"]),
        "arm_k_test_sim_loss_mean": float(k_row["test_sim_loss"]),
        "arm_g_mse_cent_mean": float(g_row["mse_cent"]),
        "arm_j_mse_cent_mean": float(j_row["mse_cent"]),
        "arm_k_mse_cent_mean": float(k_row["mse_cent"]),
        "arm_g_mean_var_3_mean": float(g_row["mean_var_3"]),
        "arm_j_mean_var_3_mean": float(j_row["mean_var_3"]),
        "arm_k_mean_var_3_mean": float(k_row["mean_var_3"]),
        "arm_g_pointer_entropy_mean": float(g_row["pointer_entropy"]),
        "arm_j_pointer_entropy_mean": float(j_row["pointer_entropy"]),
        "arm_k_pointer_entropy_mean": float(k_row["pointer_entropy"]),
        "arm_g_std_vel_3_mean": float(g_row["std_vel_3"]),
        "arm_j_std_vel_3_mean": float(j_row["std_vel_3"]),
        "arm_k_std_vel_3_mean": float(k_row["std_vel_3"]),
        "arm_g_mean_abs_vel_3_mean": float(g_row["mean_abs_vel_3"]),
        "arm_j_mean_abs_vel_3_mean": float(j_row["mean_abs_vel_3"]),
        "arm_k_mean_abs_vel_3_mean": float(k_row["mean_abs_vel_3"]),
        "ttest_gj_t": float(t_stat_gj),
        "ttest_gj_p": float(p_value_gj),
        "ttest_gk_t": float(t_stat_gk),
        "ttest_gk_p": float(p_value_gk),
        "levene_gj_W": float(lev_stat_gj),
        "levene_gj_p": float(lev_p_gj),
        "levene_gk_W": float(lev_stat_gk),
        "levene_gk_p": float(lev_p_gk),
        "c1_falsified": bool(c1_falsified),
        "c2_falsified": bool(c2_falsified),
        "c3_falsified": bool(c3_falsified),
        "c4_falsified": bool(c4_falsified),
        "c5_falsified": bool(c5_falsified),
        "hypothesis_falsified": bool(hypothesis_falsified),
    }
    audit_path = os.path.join(results_dir, "audit_results_phase14.json")
    with open(audit_path, "w") as f:
        json.dump(audit_dict, f, indent=4)
    print(f"\nSaved {audit_path}")


if __name__ == "__main__":
    main()
