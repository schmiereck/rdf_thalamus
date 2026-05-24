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

def evaluate_branch_phase12(model, seed, device):
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
            
    # Fit linear probes
    w_cent, b_cent = fit_linear_probe(x_mean_3[:100], y_probe_train)
    y_pred_cent = x_mean_3[100:] * w_cent + b_cent
    mse_cent = np.mean((y_probe_test - y_pred_cent)**2)
    
    # Calculate Pearson correlations
    r_centroid = np.corrcoef(x_mean_3, test_y_4_arr)[0, 1]
    abs_r_centroid = abs(r_centroid) if not np.isnan(r_centroid) else 0.0
    
    mean_var_3 = np.mean(var_3)
    std_x_mean_3 = np.std(x_mean_3)
    
    # Standard collapse criterion (from Phase 10/11)
    has_collapsed = not (e_a_3 >= 0.1 * e_a_all and std_x_mean_3 > 5.0)
    
    return {
        "test_sim_loss": test_sim_loss,
        "abs_r_centroid": abs_r_centroid,
        "mse_cent": mse_cent,
        "mean_var_3": mean_var_3,
        "std_x_mean_3": std_x_mean_3,
        "collapsed": has_collapsed
    }

def train_base_model_passive(seed, device):
    set_seed(seed)
    model = NonParametricJEPASpatial(d_max=8, h=3, k=4, cooldown=300, stabilization_period=100)
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

def main():
    print("=" * 80)
    print("PHASE 12 SWEEP: EVALUATING CLOSED-LOOP THALAMIC SUBSUMPTION (CLTS)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    seeds = [42, 123, 456, 789, 999]
    results_list = []
    
    # Track standardized test losses at checkpoints across seeds to compute AUC adaptation curve
    eval_steps = [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]
    checkpoint_losses = {
        "Arm F-Passive": {step: [] for step in eval_steps},
        "Arm F-Random": {step: [] for step in eval_steps},
        "Arm G (CLTS)": {step: [] for step in eval_steps}
    }
    
    for seed in seeds:
        print(f"\n" + "-"*50)
        print(f"SEED {seed}")
        print("-"*50)
        
        # 1. Train passive base model on N=3
        print("1. Training passive base model on N=3...")
        base_model = train_base_model_passive(seed, device)
        print(f"   Finished. Final d_t = {base_model.d_t}")
        
        # Save step 1500 model to evaluate as initial checkpoint for all arms
        base_eval = evaluate_branch_phase12(base_model, seed, device)
        for arm in checkpoint_losses.keys():
            checkpoint_losses[arm][1500].append(base_eval["test_sim_loss"])
            
        # 2. Clone into 3 branches and train on N=4 (with 2x mass perturbation)
        arms = ["Arm F-Passive", "Arm F-Random", "Arm G (CLTS)"]
        
        for arm in arms:
            print(f"\nTraining {arm}...")
            branch_model = base_model.clone()
            branch_optimizer = optim.Adam(branch_model.parameters(), lr=1e-3)
            
            # Initialize environment with N=4
            set_seed(seed + 1000)
            branch_env = PhysicsSandbox(N=4, seed=seed + 1000)
            branch_env.masses[3] *= 2.0  # Apply 2x mass perturbation
            
            branch_obs = branch_env.reset()
            branch_history = collections.deque(maxlen=4)
            branch_history.append(branch_obs)
            
            branch_replay = ReplayBuffer(capacity=2000)
            last_info = prefill_buffer_passive(branch_env, branch_replay, branch_history, num_transitions=100)
            
            clts_controller = None
            if arm == "Arm G (CLTS)":
                clts_controller = CLTSMotorController()
                clts_controller.reset()
                
            ewma_surprise = 0.10
            lambda_val = 0.0
            
            # Log pointer positions to track spatial coverage entropy
            pointer_positions = []
            
            # Log online losses to compute training AUC
            online_losses = []
            
            for step in range(1501, 3001):
                # 1. Action Selection
                pointer_positions.append(branch_env.pointer_pos)
                
                if arm == "Arm F-Passive":
                    action = {"acc": 0.0, "push": False}
                elif arm == "Arm F-Random":
                    action = {
                        "acc": float(np.random.uniform(-10.0, 10.0)),
                        "push": bool(np.random.rand() < 0.1)
                    }
                elif arm == "Arm G (CLTS)":
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
                
                # Take environment step
                obs, info = branch_env.step(action)
                last_info = info
                branch_history.append(obs)
                
                # Push transition to replay buffer
                x_hist_new = np.stack(list(branch_history)[:3], axis=0)
                x_target_new = branch_history[3]
                branch_replay.push(x_hist_new, x_target_new)
                
                # Train model step
                branch_model.train()
                x_hist_b, x_target_b = branch_replay.sample(32)
                x_hist_t = torch.from_numpy(x_hist_b).float().to(device)
                x_target_t = torch.from_numpy(x_target_b).float().to(device)
                
                # Dynamic lambda curriculum
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
                
                # Dynamic GDASR dimension recruitment
                branch_model.update_recruitment_logic(sim_loss_val, target_dim=3)
                if branch_model.d_t == 3 and step >= 1800:
                    branch_model.d_t = 4
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    
                # Evaluate intermediate checkpoints
                if step in eval_steps:
                    eval_res = evaluate_branch_phase12(branch_model, seed, device)
                    checkpoint_losses[arm][step].append(eval_res["test_sim_loss"])
                    
            # Compute step 3000 evaluation metrics
            eval_metrics = evaluate_branch_phase12(branch_model, seed, device)
            pointer_entropy = compute_spatial_entropy(pointer_positions)
            online_auc_1501_2000 = sum(online_losses[:500])
            online_auc_1501_3000 = sum(online_losses)
            
            results_list.append({
                "seed": seed,
                "arm": arm,
                "test_sim_loss": eval_metrics["test_sim_loss"],
                "abs_r_centroid": eval_metrics["abs_r_centroid"],
                "mse_cent": eval_metrics["mse_cent"],
                "mean_var_3": eval_metrics["mean_var_3"],
                "std_x_mean_3": eval_metrics["std_x_mean_3"],
                "collapsed": int(eval_metrics["collapsed"]),
                "pointer_entropy": pointer_entropy,
                "online_auc_1501_2000": online_auc_1501_2000,
                "online_auc_1501_3000": online_auc_1501_3000
            })
            
            print(f"   Done. Sim Loss: {eval_metrics['test_sim_loss']:.5f}, MSE Centroid: {eval_metrics['mse_cent']:.4f}, Soft Var: {eval_metrics['mean_var_3']:.4f}, Pointer Entropy: {pointer_entropy:.4f}")
            
    # Save CSV summary
    os.makedirs("archive/iter_012/results", exist_ok=True)
    summary_df = pd.DataFrame(results_list)
    summary_df.to_csv("archive/iter_012/results/summary_phase12.csv", index=False)
    print("\nSaved summary_phase12.csv successfully!")
    
    # Compute average test losses at checkpoints across seeds
    avg_checkpoint_losses = {}
    for arm in arms:
        avg_checkpoint_losses[arm] = [np.mean(checkpoint_losses[arm][step]) for step in eval_steps]
        
    # Plot adaptation curves (AUC Recovery Curves)
    plt.figure(figsize=(10, 6))
    colors_dict = {
        "Arm F-Passive": "blue",
        "Arm F-Random": "green",
        "Arm G (CLTS)": "red"
    }
    for arm in arms:
        plt.plot(eval_steps, avg_checkpoint_losses[arm], marker='o', label=arm, color=colors_dict[arm], linewidth=2)
        
    plt.title("Adaptation Trajectory (Offline Test Sim Loss over Steps)", fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Standardized Test Simulation Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig("archive/iter_012/results/auc_recovery_curves.png", dpi=150)
    plt.close()
    print("Saved auc_recovery_curves.png successfully!")
    
    # Perform pre-registered falsification audit
    print("\n" + "="*80)
    print("SCIENTIFIC AUDIT: PRE-REGISTERED FALSIFICATION CRITERIA")
    print("=" * 80)
    
    # Aggregate statistics
    agg = summary_df.groupby("arm").mean().reset_index()
    print(agg)
    
    clts_row = agg[agg["arm"] == "Arm G (CLTS)"].iloc[0]
    random_row = agg[agg["arm"] == "Arm F-Random"].iloc[0]
    passive_row = agg[agg["arm"] == "Arm F-Passive"].iloc[0]
    
    loss_reduction = (random_row["test_sim_loss"] - clts_row["test_sim_loss"]) / random_row["test_sim_loss"] * 100.0
    c1_falsified = loss_reduction < 20.0
    print(f"\nCriterion 1 (Predictive Performance):")
    print(f"  - Arm F-Random Sim Loss: {random_row['test_sim_loss']:.6f}")
    print(f"  - Arm G (CLTS) Sim Loss: {clts_row['test_sim_loss']:.6f}")
    print(f"  - Reduction: {loss_reduction:.2f}% (Falsification threshold: < 20.0%)")
    print(f"  - RESULT: {'FALSIFIED' if c1_falsified else 'VALIDATED'}")
    
    c2_var_falsified = clts_row["mean_var_3"] > 20.0
    c2_mse_falsified = clts_row["mse_cent"] > 85.0
    c2_falsified = c2_var_falsified or c2_mse_falsified
    print(f"\nCriterion 2 (Representational Instability):")
    print(f"  - Arm G (CLTS) Soft Spatial Variance: {clts_row['mean_var_3']:.4f} (Falsification threshold: > 20.0)")
    print(f"  - Arm G (CLTS) Centroid Decoding MSE: {clts_row['mse_cent']:.4f} (Falsification threshold: > 85.0)")
    print(f"  - RESULT: {'FALSIFIED' if c2_falsified else 'VALIDATED'}")
    
    clts_offline_auc = sum(avg_checkpoint_losses["Arm G (CLTS)"])
    random_offline_auc = sum(avg_checkpoint_losses["Arm F-Random"])
    auc_reduction = (random_offline_auc - clts_offline_auc) / random_offline_auc * 100.0
    c3_falsified = auc_reduction < 15.0
    print(f"\nCriterion 3 (Adaptation Efficiency):")
    print(f"  - Arm F-Random Checkpoint Test Loss AUC: {random_offline_auc:.4f}")
    print(f"  - Arm G (CLTS) Checkpoint Test Loss AUC: {clts_offline_auc:.4f}")
    print(f"  - AUC Reduction: {auc_reduction:.2f}% (Falsification threshold: < 15.0%)")
    print(f"  - RESULT: {'FALSIFIED' if c3_falsified else 'VALIDATED'}")
    
    print(f"\nSpatial Coverage Entropy Analysis:")
    print(f"  - Arm F-Passive Pointer Entropy: {passive_row['pointer_entropy']:.4f}")
    print(f"  - Arm F-Random Pointer Entropy: {random_row['pointer_entropy']:.4f}")
    print(f"  - Arm G (CLTS) Pointer Entropy: {clts_row['pointer_entropy']:.4f}")
    
    summary_dict = {
        "clts_test_sim_loss": float(clts_row["test_sim_loss"]),
        "random_test_sim_loss": float(random_row["test_sim_loss"]),
        "passive_test_sim_loss": float(passive_row["test_sim_loss"]),
        "loss_reduction_pct": float(loss_reduction),
        "clts_soft_spatial_variance": float(clts_row["mean_var_3"]),
        "clts_centroid_decoding_mse": float(clts_row["mse_cent"]),
        "clts_offline_auc": float(clts_offline_auc),
        "random_offline_auc": float(random_offline_auc),
        "auc_reduction_pct": float(auc_reduction),
        "clts_pointer_entropy": float(clts_row["pointer_entropy"]),
        "random_pointer_entropy": float(random_row["pointer_entropy"]),
        "c1_falsified": bool(c1_falsified),
        "c2_falsified": bool(c2_falsified),
        "c3_falsified": bool(c3_falsified),
        "hypothesis_falsified": bool(c1_falsified or c2_falsified or c3_falsified)
    }
    with open("archive/iter_012/results/audit_results.json", "w") as f:
        json.dump(summary_dict, f, indent=4)
    print("\nSaved audit_results.json successfully!")

if __name__ == "__main__":
    main()
