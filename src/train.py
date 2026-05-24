import os
import sys
import csv
import time
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# Ensure src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models import FixedJEPA, DynamicJEPA

# Replay Buffer storing (x_hist, x_target)
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
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prefill_buffer(env, replay_buffer, history, num_transitions):
    # Ensure there's at least one frame in history to start stepping
    if len(history) == 0:
        obs = env.reset()
        history.append(obs)
        
    while len(replay_buffer) < num_transitions:
        obs, info = env.step()
        history.append(obs)
        if len(history) == 4:
            x_hist = np.stack(list(history)[:3], axis=0) # shape (3, 3, 128)
            x_target = history[3] # shape (3, 128)
            replay_buffer.push(x_hist, x_target)

def run_experiment(model_type, seed, device):
    print(f"\n" + "="*60)
    print(f"RUNNING EXPERIMENT: Model = {model_type:<10} | Seed = {seed}")
    print("="*60)
    
    set_seed(seed)
    
    # Initialize Model
    if model_type == 'b1':
        model = FixedJEPA(d_t=2, d_max=8, h=3)
    elif model_type == 'b1_large':
        model = FixedJEPA(d_t=3, d_max=8, h=3)
    elif model_type == 'dynamic':
        model = DynamicJEPA(d_max=8, h=3, k=4, cooldown=500, stabilization_period=200)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Environment (starts with N=2)
    env = PhysicsSandbox(N=2, seed=seed)
    obs = env.reset()
    
    history = collections.deque(maxlen=4)
    history.append(obs)
    
    replay_buffer = ReplayBuffer(capacity=2000)
    
    # Pre-fill replay buffer before training loop starts to ensure immediate training
    prefill_buffer(env, replay_buffer, history, num_transitions=100)
    
    logs = []
    recruitment_step = None
    
    # Training loop: 5000 steps total
    for step in range(1, 5001):
        if step == 1501:
            # Transition to PhysicsSandbox(N=3)
            print(f"[Step {step}] Transitioning to PhysicsSandbox(N=3)...")
            env = PhysicsSandbox(N=3, seed=seed)
            obs = env.reset()
            history.clear()
            history.append(obs)
            replay_buffer.clear()
            
            # Pre-fill buffer with N=3 transitions before starting training again
            prefill_buffer(env, replay_buffer, history, num_transitions=100)
            
        # Environment interaction
        obs, info = env.step()
        history.append(obs)
        if len(history) == 4:
            x_hist = np.stack(list(history)[:3], axis=0)
            x_target = history[3]
            replay_buffer.push(x_hist, x_target)
            
        # Draw a batch from the replay buffer
        batch_x_hist_np, batch_x_target_np = replay_buffer.sample(32)
        batch_x_hist = torch.tensor(batch_x_hist_np, dtype=torch.float32).to(device)
        batch_x_target = torch.tensor(batch_x_target_np, dtype=torch.float32).to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        loss_dict, z_pred, z_target = model(batch_x_hist, batch_x_target, cov_weight=25.0)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        
        sim_loss_val = loss_dict["sim_loss"].item()
        
        # If 'dynamic', update GDASR recruitment logic with 1000-step representation-warmup
        if model_type == 'dynamic':
            if step == 1001:
                model.reset_error_buffer()
            if step > 1000:
                model.update_recruitment_logic(sim_loss_val)
                
            if model.d_t == 3 and recruitment_step is None:
                recruitment_step = step
                print(f" >>> [GDASR] Seed {seed} recruited 3rd dimension at Step {step}! <<<")
                
        # Calculate diagnostics on the training batch
        active_dims = model.d_t
        z_target_np = z_target.detach().cpu().numpy()
        stds = np.std(z_target_np, axis=0, ddof=0)
        if len(stds) < 8:
            stds = np.pad(stds, (0, 8 - len(stds)), 'constant', constant_values=0.0)
            
        # Correlation of active dimensions
        z_active = z_target_np[:, :active_dims]
        if active_dims > 1:
            corr_matrix = np.corrcoef(z_active, rowvar=False)
            corr_matrix = np.atleast_2d(corr_matrix)
            triu_indices = np.triu_indices(active_dims, k=1)
            abs_corrs = np.abs(corr_matrix[triu_indices])
            mean_abs_corr = np.mean(abs_corrs) if len(abs_corrs) > 0 else 0.0
        else:
            mean_abs_corr = 0.0
            
        # Store log
        step_log = {
            "step": step,
            "loss": loss.item(),
            "sim_loss": sim_loss_val,
            "var_loss": loss_dict["var_loss"].item(),
            "cov_loss": loss_dict["cov_loss"].item(),
            "active_dims": active_dims,
            "mean_abs_corr": mean_abs_corr
        }
        for i in range(8):
            step_log[f"std_dim{i}"] = stds[i]
            
        logs.append(step_log)
        
        if step % 500 == 0:
            print(f"Step {step:4d}/5000 | Loss: {loss.item():.4f} | Sim Loss: {sim_loss_val:.4f} | Active Dims: {active_dims} | Mean Abs Corr: {mean_abs_corr:.4f}")
            
    # Save step-by-step logs for each step
    os.makedirs("archive/iter_003/runs", exist_ok=True)
    csv_path = f"archive/iter_003/runs/{model_type}_seed{seed}.csv"
    headers = ["step", "loss", "sim_loss", "var_loss", "cov_loss", "active_dims"] + [f"std_dim{i}" for i in range(8)] + ["mean_abs_corr"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for log in logs:
            writer.writerow(log)
            
    # Evaluation at the end of training on N=3 (seed+10000)
    print(f"Evaluating model on separate test environment (seed={seed+10000})...")
    test_env = PhysicsSandbox(N=3, seed=seed+10000)
    test_obs = test_env.reset(seed=seed+10000)
    test_history = collections.deque(maxlen=4)
    test_history.append(test_obs)
    
    test_x_hist_list = []
    test_x_target_list = []
    while len(test_x_hist_list) < 100:
        obs_t, info_t = test_env.step()
        test_history.append(obs_t)
        if len(test_history) == 4:
            x_hist_t = np.stack(list(test_history)[:3], axis=0)
            x_target_t = test_history[3]
            test_x_hist_list.append(x_hist_t)
            test_x_target_list.append(x_target_t)
            
    test_x_hist = torch.tensor(np.stack(test_x_hist_list, axis=0), dtype=torch.float32).to(device)
    test_x_target = torch.tensor(np.stack(test_x_target_list, axis=0), dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        test_loss_dict, z_pred_eval, z_target_eval = model(test_x_hist, test_x_target, cov_weight=25.0)
        
    final_test_sim_loss = test_loss_dict["sim_loss"].item()
    z_target_eval_np = z_target_eval.detach().cpu().numpy()
    final_active_dims = model.d_t
    
    # Compute stds for active dimensions
    eval_active_stds = np.std(z_target_eval_np[:, :final_active_dims], axis=0, ddof=0).tolist()
    
    # Compute mean absolute correlation and cross-correlation matrix
    if final_active_dims > 1:
        corr_matrix = np.corrcoef(z_target_eval_np[:, :final_active_dims], rowvar=False)
        corr_matrix = np.atleast_2d(corr_matrix)
        triu_indices = np.triu_indices(final_active_dims, k=1)
        abs_corrs = np.abs(corr_matrix[triu_indices])
        final_mean_abs_corr = np.mean(abs_corrs) if len(abs_corrs) > 0 else 0.0
    else:
        corr_matrix = np.ones((final_active_dims, final_active_dims))
        final_mean_abs_corr = 0.0
        
    r_0_2 = None
    r_1_2 = None
    if model_type == 'dynamic' and final_active_dims >= 3:
        r_0_2 = corr_matrix[0, 2]
        r_1_2 = corr_matrix[1, 2]
        
    print(f"Evaluation Results | Test Sim Loss: {final_test_sim_loss:.4f} | Final Active Dims: {final_active_dims}")
    print(f"Active Stds: {eval_active_stds} | Mean Abs Corr: {final_mean_abs_corr:.4f}")
    if r_0_2 is not None:
        print(f"Orthogonality: r_0_2 = {r_0_2:.4f}, r_1_2 = {r_1_2:.4f}")
        
    return {
        "model_type": model_type,
        "seed": seed,
        "final_test_sim_loss": final_test_sim_loss,
        "recruitment_step": recruitment_step if recruitment_step is not None else np.nan,
        "active_dims": final_active_dims,
        "active_stds": eval_active_stds,
        "mean_abs_corr": final_mean_abs_corr,
        "r_0_2": r_0_2 if r_0_2 is not None else np.nan,
        "r_1_2": r_1_2 if r_1_2 is not None else np.nan,
        "logs": logs
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    seeds = [42, 123, 456, 789, 999]
    model_types = ['b1', 'b1_large', 'dynamic']
    
    all_results = []
    all_run_logs = {model_type: [] for model_type in model_types}
    
    start_time = time.time()
    
    # Run the 15 experiments
    for model_type in model_types:
        for seed in seeds:
            res = run_experiment(model_type, seed, device)
            all_results.append(res)
            all_run_logs[model_type].append(res["logs"])
            
    total_duration = time.time() - start_time
    print(f"\nTraining of all 15 models completed in {total_duration/60:.2f} minutes.")
    
    # Aggregation and Analysis
    df_results = pd.DataFrame(all_results)
    
    summary_data = []
    
    print("\n" + "="*80)
    print("AGGREGATED RESULTS ACROSS SEEDS (Mean ± Std)")
    print("="*80)
    
    for model_type in model_types:
        sub_df = df_results[df_results["model_type"] == model_type]
        
        # Test Sim Loss
        mean_loss = sub_df["final_test_sim_loss"].mean()
        std_loss = sub_df["final_test_sim_loss"].std()
        
        # Recruitment Step
        valid_recruits = sub_df["recruitment_step"].dropna()
        if len(valid_recruits) > 0:
            mean_rec_step = valid_recruits.mean()
            std_rec_step = valid_recruits.std()
            rec_step_str = f"{mean_rec_step:.1f} ± {std_rec_step:.1f}"
        else:
            mean_rec_step, std_rec_step = np.nan, np.nan
            rec_step_str = "N/A"
            
        # Mean Absolute Correlation
        mean_mac = sub_df["mean_abs_corr"].mean()
        std_mac = sub_df["mean_abs_corr"].std()
        
        # Orthogonality r_0_2
        valid_r02 = sub_df["r_0_2"].dropna()
        if len(valid_r02) > 0:
            mean_r02 = valid_r02.mean()
            std_r02 = valid_r02.std()
            r02_str = f"{mean_r02:.4f} ± {std_r02:.4f}"
        else:
            mean_r02, std_r02 = np.nan, np.nan
            r02_str = "N/A"
            
        # Orthogonality r_1_2
        valid_r12 = sub_df["r_1_2"].dropna()
        if len(valid_r12) > 0:
            mean_r12 = valid_r12.mean()
            std_r12 = valid_r12.std()
            r12_str = f"{mean_r12:.4f} ± {std_r12:.4f}"
        else:
            mean_r12, std_r12 = np.nan, np.nan
            r12_str = "N/A"
            
        # Representation collapse checking
        collapse_count = 0
        for idx, row in sub_df.iterrows():
            stds = row["active_stds"]
            mac = row["mean_abs_corr"]
            # Collapse if any active std <= 0.1 or mean absolute corr >= 0.3
            has_collapsed = any(s <= 0.1 for s in stds) or mac >= 0.3
            if has_collapsed:
                collapse_count += 1
                
        collapse_rate = collapse_count / len(sub_df)
        
        print(f"Model: {model_type:<10}")
        print(f"  Final Test Sim Loss       : {mean_loss:.5f} ± {std_loss:.5f}")
        print(f"  Recruitment Step          : {rec_step_str}")
        print(f"  Mean Abs Correlation     : {mean_mac:.5f} ± {std_mac:.5f}")
        print(f"  Orthogonality r_0_2       : {r02_str}")
        print(f"  Orthogonality r_1_2       : {r12_str}")
        print(f"  Representation Collapse   : {collapse_count}/{len(sub_df)} runs ({collapse_rate*100:.1f}%)")
        print("-" * 50)
        
        summary_data.append({
            "model_type": model_type,
            "mean_test_sim_loss": mean_loss,
            "std_test_sim_loss": std_loss,
            "mean_recruitment_step": mean_rec_step,
            "std_recruitment_step": std_rec_step,
            "mean_abs_corr": mean_mac,
            "std_abs_corr": std_mac,
            "mean_r_0_2": mean_r02,
            "std_r_0_2": std_r02,
            "mean_r_1_2": mean_r12,
            "std_r_1_2": std_r12,
            "collapse_rate": collapse_rate,
        })
        
    # Save summary.csv
    os.makedirs("archive/iter_003/results", exist_ok=True)
    pd.DataFrame(summary_data).to_csv("archive/iter_003/results/summary.csv", index=False)
    print("Summary saved to archive/iter_003/results/summary.csv")
    
    # ----------------------------------------------------
    # Preregistered Falsification Criteria Checklist
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("PREREGISTERED FALSIFICATION CRITERIA CHECKLIST")
    print("="*80)
    
    dynamic_df = df_results[df_results["model_type"] == 'dynamic']
    b1_df = df_results[df_results["model_type"] == 'b1']
    
    # Criterion 1: Did 'dynamic' always recruit a 3rd dimension (d_t=3) within 3500 steps of N=3?
    # Exposure of N=3 starts at step 1500, training ends at step 5000. So recruitment step must be within [1501, 5000]
    recruited_count = dynamic_df["recruitment_step"].notna().sum()
    all_recruited = recruited_count == len(seeds)
    c1_passed = all_recruited
    c1_status = "SUPPORTED" if c1_passed else "FALSIFIED"
    print(f"Criterion 1: Always recruit 3rd dimension (d_t=3) within 3500 steps of N=3?")
    print(f"  - Status: {c1_status}")
    print(f"  - Detail: Dynamic model recruited in {recruited_count}/{len(seeds)} runs.")
    if recruited_count > 0:
        steps_list = dynamic_df["recruitment_step"].dropna().tolist()
        print(f"  - Recruitment steps: {steps_list}")
        
    # Criterion 2: Is the final N=3 prediction error of 'dynamic' at least 30% lower than 'b1'?
    mean_loss_dyn = dynamic_df["final_test_sim_loss"].mean()
    mean_loss_b1 = b1_df["final_test_sim_loss"].mean()
    pct_improvement = (mean_loss_b1 - mean_loss_dyn) / mean_loss_b1 * 100
    c2_passed = pct_improvement >= 30.0
    c2_status = "SUPPORTED" if c2_passed else "FALSIFIED"
    print(f"Criterion 2: Is the final N=3 prediction error of 'dynamic' at least 30% lower than 'b1'?")
    print(f"  - Status: {c2_status}")
    print(f"  - Detail: 'b1' Loss = {mean_loss_b1:.5f}, 'dynamic' Loss = {mean_loss_dyn:.5f} ({pct_improvement:.1f}% lower)")
    
    # Criterion 3: Did representation collapse occur (any active dim std <= 0.1 or mean absolute corr >= 0.3)?
    # We check if *any* active dimension standard deviation in the evaluation batch is <= 0.1
    # or if the average absolute correlation between active dimensions is >= 0.3.
    # We'll check this for 'dynamic' first, and also generally.
    dyn_collapsed_count = 0
    for idx, row in dynamic_df.iterrows():
        stds = row["active_stds"]
        mac = row["mean_abs_corr"]
        if any(s <= 0.1 for s in stds) or mac >= 0.3:
            dyn_collapsed_count += 1
            
    c3_passed = dyn_collapsed_count == 0
    c3_status = "SUPPORTED" if c3_passed else "FALSIFIED"
    print(f"Criterion 3: No representation collapse (any active dim std <= 0.1 or mean absolute corr >= 0.3)?")
    print(f"  - Status: {c3_status}")
    print(f"  - Detail: Dynamic model collapsed in {dyn_collapsed_count}/{len(seeds)} runs.")
    for idx, row in dynamic_df.iterrows():
        print(f"    * Seed {row['seed']}: active stds = {[round(s, 3) for s in row['active_stds']]}, mean abs corr = {row['mean_abs_corr']:.4f}")
        
    # Overall Hypothesis Status
    overall_supported = c1_passed and c2_passed and c3_passed
    print("\n" + "="*80)
    print(f"OVERALL HYPOTHESIS STATUS: " + ("SUPPORTED" if overall_supported else "FALSIFIED"))
    print("="*80)
    
    # ----------------------------------------------------
    # Plotting Learning Curves
    # ----------------------------------------------------
    print("\nGenerating learning curve plots...")
    
    # Helper for smoothing
    def smooth_curve(y, box_pts=50):
        return pd.Series(y).rolling(window=box_pts, min_periods=1).mean().values
        
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    
    colors = {
        'b1': '#1f77b4',       # Muted blue
        'b1_large': '#2ca02c', # Muted green
        'dynamic': '#ff7f0e'   # Muted orange
    }
    
    labels = {
        'b1': 'Baseline B1 (Fixed $d_t=2$)',
        'b1_large': 'Baseline B1 Large (Fixed $d_t=3$)',
        'dynamic': 'Thalamus Dynamic (GDASR)'
    }
    
    steps = np.arange(1, 5001)
    
    # Panel 1: Prediction Error (sim_loss)
    for model_type in model_types:
        sim_losses = np.array([[log['sim_loss'] for log in run] for run in all_run_logs[model_type]])
        mean_loss = np.mean(sim_losses, axis=0)
        std_loss = np.std(sim_losses, axis=0)
        
        # Smooth both mean and std
        mean_loss_smooth = smooth_curve(mean_loss, box_pts=50)
        std_loss_smooth = smooth_curve(std_loss, box_pts=50)
        
        axes[0].plot(steps, mean_loss_smooth, color=colors[model_type], label=labels[model_type], linewidth=2.0)
        axes[0].fill_between(
            steps, 
            np.maximum(mean_loss_smooth - std_loss_smooth, 0), 
            mean_loss_smooth + std_loss_smooth, 
            color=colors[model_type], 
            alpha=0.15
        )
        
    axes[0].axvline(x=1500, color='#d62728', linestyle='--', linewidth=1.5, label='Transition to N=3 Objects')
    axes[0].set_ylabel('Prediction Error ($sim\_loss$)', fontsize=12)
    axes[0].set_title('A. Temporal Prediction Error over Training Steps', fontsize=14, fontweight='bold', loc='left')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Panel 2: Active Dimensions (d_t)
    for model_type in model_types:
        active_dims = np.array([[log['active_dims'] for log in run] for run in all_run_logs[model_type]])
        mean_active = np.mean(active_dims, axis=0)
        std_active = np.std(active_dims, axis=0)
        
        axes[1].plot(steps, mean_active, color=colors[model_type], label=labels[model_type], linewidth=2.0)
        if np.max(std_active) > 0.0:
            axes[1].fill_between(
                steps, 
                mean_active - std_active, 
                mean_active + std_active, 
                color=colors[model_type], 
                alpha=0.15
            )
            
    axes[1].axvline(x=1500, color='#d62728', linestyle='--', linewidth=1.5, label='Transition to N=3 Objects')
    axes[1].set_xlabel('Training Steps', fontsize=12)
    axes[1].set_ylabel('Active Latent Dimensions ($d_t$)', fontsize=12)
    axes[1].set_title('B. Active Latent Space Size ($d_t$) over Training Steps', fontsize=14, fontweight='bold', loc='left')
    axes[1].legend(fontsize=10, loc='lower right')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("archive/iter_003/results/learning_curves.png", dpi=200)
    plt.close()
    print("Learning curve plot generated and saved to: archive/iter_003/results/learning_curves.png")

if __name__ == "__main__":
    main()