Create the training script 'src/train_thalamus.py'. Use '.venv/Scripts/python.exe' to run it.

The script must contain the following code:

```python
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

# Set threads to prevent CPU thrashing
torch.set_num_threads(2)

# Ensure src directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models import FixedJEPA
from src.thalamus import ThalamusNet, NonGatedControlNet

# 1. Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, x_hist, x_target, color_0, pos_0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (x_hist, x_target, color_0, pos_0)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        x_hist, x_target, color_0, pos_0 = zip(*batch)
        return (np.stack(x_hist, axis=0), 
                np.stack(x_target, axis=0), 
                np.stack(color_0, axis=0), 
                np.array(pos_0, dtype=np.float32))

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

def prefill_buffer(env, replay_buffer, history, num_transitions):
    if len(history) == 0:
        obs = env.reset()
        history.append(obs)
        
    while len(replay_buffer) < num_transitions:
        obs, info = env.step()
        history.append(obs)
        if len(history) == 4:
            x_hist = np.stack(list(history)[:3], axis=0)
            x_target = history[3]
            color_0 = info["colors"][0]
            pos_0 = info["positions"][0]
            replay_buffer.push(x_hist, x_target, color_0, pos_0)

def find_stable_step(losses, threshold=0.08, window=100):
    raw_step = -1
    for i in range(len(losses)):
        if all(l < threshold for l in losses[i:]):
            raw_step = i + 1
            break
            
    roll_step = -1
    if len(losses) >= window:
        rolling_means = []
        for i in range(len(losses)):
            if i < window - 1:
                rolling_means.append(np.mean(losses[:i+1]))
            else:
                rolling_means.append(np.mean(losses[i-window+1:i+1]))
        for i in range(len(rolling_means)):
            if all(rm < threshold for rm in rolling_means[i:]):
                roll_step = i + 1
                break
                
    return raw_step, roll_step

def run_experiment(model_type, seed, device):
    print(f"\n" + "="*60)
    print(f"RUNNING EXPERIMENT: Model = {model_type:<10} | Seed = {seed}")
    print("="*60)
    
    set_seed(seed)
    
    # Initialize Model
    if model_type == 'gated':
        model = ThalamusNet(d_max=8, h=3, cooldown=200)
    elif model_type == 'nongated':
        model = NonGatedControlNet(d_max=8, h=3)
    elif model_type == 'b1':
        model = FixedJEPA(d_t=2, d_max=8, h=3)
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
    prefill_buffer(env, replay_buffer, history, num_transitions=100)
    
    logs = []
    
    # Training loop: 5000 steps
    for step in range(1, 5001):
        if step == 1501:
            # Transition to PhysicsSandbox(N=3)
            print(f"[Step {step}] Transitioning to PhysicsSandbox(N=3)...")
            env = PhysicsSandbox(N=3, seed=seed)
            obs = env.reset()
            history.clear()
            history.append(obs)
            replay_buffer.clear()
            prefill_buffer(env, replay_buffer, history, num_transitions=100)
            
        # Environment interaction
        obs, info = env.step()
        history.append(obs)
        if len(history) == 4:
            x_hist = np.stack(list(history)[:3], axis=0)
            x_target = history[3]
            color_0 = info["colors"][0]
            pos_0 = info["positions"][0]
            replay_buffer.push(x_hist, x_target, color_0, pos_0)
            
        # Draw batch
        batch_x_hist_np, batch_x_target_np, batch_color_0_np, batch_pos_0_np = replay_buffer.sample(32)
        batch_x_hist = torch.tensor(batch_x_hist_np, dtype=torch.float32).to(device)
        batch_x_target = torch.tensor(batch_x_target_np, dtype=torch.float32).to(device)
        batch_color_0 = torch.tensor(batch_color_0_np, dtype=torch.float32).to(device)
        batch_pos_0 = torch.tensor(batch_pos_0_np, dtype=torch.float32).to(device)
        
        # Determine priming mode
        priming_mode = "external" if step <= 1500 else "self"
        
        model.train()
        optimizer.zero_grad()
        
        if model_type in ['gated', 'nongated']:
            loss_dict, _, _ = model(
                batch_x_hist, 
                batch_x_target, 
                external_query=batch_color_0, 
                priming_mode=priming_mode
            )
            loss = loss_dict["loss"]
            loss.backward()
            model.zero_inactive_gradients()
            optimizer.step()
            
            l2_sim_loss_val = loss_dict["l2_sim_loss"].item()
            token_locus_val = loss_dict.get("token_locus", -1)
            l2_locked_val = int(loss_dict.get("l2_locked", False))
        else:
            # B1 (FixedJEPA)
            loss_dict, _, _ = model(batch_x_hist, batch_x_target)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            
            l2_sim_loss_val = loss_dict["sim_loss"].item()
            token_locus_val = -1
            l2_locked_val = -1
            
        # Compute tracking overlap for steps 1501-5000
        overlap_val = 0.0
        if step > 1500 and model_type in ['gated', 'nongated']:
            overlap_tensor = model.compute_physical_tracking_overlap(batch_pos_0, batch_x_target, external_query=batch_color_0)
            overlap_val = overlap_tensor.mean().item()
            
        logs.append({
            "step": step,
            "loss": loss.item(),
            "l2_sim_loss": l2_sim_loss_val,
            "token_locus": token_locus_val,
            "l2_locked": l2_locked_val,
            "overlap": overlap_val
        })
        
        if step % 1000 == 0:
            print(f"Step {step:4d}/5000 | Loss: {loss.item():.4f} | L2 Sim Loss: {l2_sim_loss_val:.4f} | Overlap: {overlap_val:.4f} | Locus: {token_locus_val}")
            
    # Save training runs
    os.makedirs("archive/iter_004/runs", exist_ok=True)
    df_run = pd.DataFrame(logs)
    df_run.to_csv(f"archive/iter_004/runs/{model_type}_seed{seed}.csv", index=False)
    
    # 2. Evaluation on Separate Environment (seed + 10000) with N=3 for 100 steps
    print(f"Evaluating {model_type} on separate test environment (seed={seed+10000})...")
    test_env = PhysicsSandbox(N=3, seed=seed+10000)
    test_obs = test_env.reset()
    test_history = collections.deque(maxlen=4)
    test_history.append(test_obs)
    
    test_x_hist_list = []
    test_x_target_list = []
    test_color_0_list = []
    test_target_pos_list = []
    
    while len(test_x_hist_list) < 100:
        obs_t, info_t = test_env.step()
        test_history.append(obs_t)
        if len(test_history) == 4:
            x_hist_t = np.stack(list(test_history)[:3], axis=0)
            x_target_t = test_history[3]
            test_x_hist_list.append(x_hist_t)
            test_x_target_list.append(x_target_t)
            test_color_0_list.append(info_t["colors"][0])
            test_target_pos_list.append(info_t["positions"][0])
            
    test_x_hist = torch.tensor(np.stack(test_x_hist_list, axis=0), dtype=torch.float32).to(device)
    test_x_target = torch.tensor(np.stack(test_x_target_list, axis=0), dtype=torch.float32).to(device)
    test_color_0 = torch.tensor(np.stack(test_color_0_list, axis=0), dtype=torch.float32).to(device)
    test_target_pos = torch.tensor(np.array(test_target_pos_list), dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        if model_type == 'b1':
            test_loss_dict, _, _ = model(test_x_hist, test_x_target)
            final_test_l2_loss = test_loss_dict["sim_loss"].item()
            final_test_overlap = 0.0
        elif model_type in ['gated', 'nongated']:
            test_loss_dict, _, _ = model(test_x_hist, test_x_target, priming_mode="self")
            final_test_l2_loss = test_loss_dict["l2_sim_loss"].item()
            overlap_tensor = model.compute_physical_tracking_overlap(test_target_pos, test_x_target, external_query=test_color_0)
            final_test_overlap = overlap_tensor.mean().item()
            
    print(f"Evaluation results | Test L2 Loss: {final_test_l2_loss:.4f} | Test Tracking Overlap: {final_test_overlap:.4f}")
    
    return {
        "model_type": model_type,
        "seed": seed,
        "final_test_l2_loss": final_test_l2_loss,
        "final_test_overlap": final_test_overlap,
        "logs": logs
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    seeds = [42, 123, 456, 789, 999]
    model_types = ['gated', 'nongated', 'b1']
    
    results = []
    
    start_time = time.time()
    for model_type in model_types:
        for seed in seeds:
            res = run_experiment(model_type, seed, device)
            results.append(res)
            
    print(f"\nAll experiments finished in {(time.time() - start_time)/60:.2f} minutes.")
    
    # 3. Analyze and Aggregate
    df_results = pd.DataFrame(results)
    
    # Let's save a summary dataframe
    summary_list = []
    for model_type in model_types:
        sub_df = df_results[df_results["model_type"] == model_type]
        mean_l2 = sub_df["final_test_l2_loss"].mean()
        std_l2 = sub_df["final_test_l2_loss"].std()
        mean_overlap = sub_df["final_test_overlap"].mean()
        std_overlap = sub_df["final_test_overlap"].std()
        
        # Calculate stable step
        stable_steps_raw = []
        stable_steps_roll = []
        overlaps_train = []
        for index, row in sub_df.iterrows():
            losses = [step_log["l2_sim_loss"] for step_log in row["logs"]]
            raw_s, roll_s = find_stable_step(losses, threshold=0.08, window=100)
            stable_steps_raw.append(raw_s)
            stable_steps_roll.append(roll_s)
            
            # Average training tracking overlap (steps 1501-5000)
            overlaps_t = [step_log["overlap"] for step_log in row["logs"][1500:]]
            if len(overlaps_t) > 0:
                overlaps_train.append(np.mean(overlaps_t))
            else:
                overlaps_train.append(0.0)
                
        summary_list.append({
            "model_type": model_type,
            "test_l2_loss_mean": mean_l2,
            "test_l2_loss_std": std_l2,
            "test_overlap_mean": mean_overlap,
            "test_overlap_std": std_overlap,
            "stable_step_raw_mean": np.mean(stable_steps_raw),
            "stable_step_roll_mean": np.mean(stable_steps_roll),
            "train_overlap_mean": np.mean(overlaps_train)
        })
        
    df_summary = pd.DataFrame(summary_list)
    os.makedirs("archive/iter_004/results", exist_ok=True)
    df_summary.to_csv("archive/iter_004/results/summary.csv", index=False)
    print("\nSummary results saved successfully to archive/iter_004/results/summary.csv")
    print(df_summary.to_string())
    
    # Levene's test between gated and nongated
    gated_test_losses = df_results[df_results["model_type"] == "gated"]["final_test_l2_loss"].tolist()
    nongated_test_losses = df_results[df_results["model_type"] == "nongated"]["final_test_l2_loss"].tolist()
    
    # Manual Levene's test fallback
    def manual_levene(g1, g2):
        g1 = np.array(g1)
        g2 = np.array(g2)
        n1 = len(g1)
        n2 = len(g2)
        med1 = np.median(g1)
        med2 = np.median(g2)
        z1 = np.abs(g1 - med1)
        z2 = np.abs(g2 - med2)
        
        all_z = np.concatenate([z1, z2])
        grand_mean = np.mean(all_z)
        mean1 = np.mean(z1)
        mean2 = np.mean(z2)
        
        ss_between = n1 * (mean1 - grand_mean)**2 + n2 * (mean2 - grand_mean)**2
        ss_within = np.sum((z1 - mean1)**2) + np.sum((z2 - mean2)**2)
        
        df_between = 1
        df_within = n1 + n2 - 2
        
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        f_stat = ms_between / (ms_within + 1e-12)
        
        try:
            from scipy.stats import f
            p_val = f.sf(f_stat, df_between, df_within)
        except ImportError:
            p_val = float('nan')
            
        return f_stat, p_val

    f_stat, levene_p = manual_levene(gated_test_losses, nongated_test_losses)
    print(f"\nLevene's Test on Test L2 Loss Variance (Gated vs Non-Gated): F-statistic = {f_stat:.4f}, p-value = {levene_p:.4f}")
    
    # Falsification Criteria Check
    print("\n" + "="*80)
    print("FALSIFICATION CRITERIA EVALUATION")
    print("="*80)
    
    gated_summary = df_summary[df_summary["model_type"] == "gated"].iloc[0]
    nongated_summary = df_summary[df_summary["model_type"] == "nongated"].iloc[0]
    b1_summary = df_summary[df_summary["model_type"] == "b1"].iloc[0]
    
    # Criterion 1: Does gated have a lower standard deviation of L2 test loss than nongated?
    c1_passed = gated_summary["test_l2_loss_std"] < nongated_summary["test_l2_loss_std"]
    c1_stat_sig = levene_p < 0.05 if not np.isnan(levene_p) else True
    print(f"Criterion 1: Gated std ({gated_summary['test_l2_loss_std']:.6f}) < Non-Gated std ({nongated_summary['test_l2_loss_std']:.6f})? -> {c1_passed}")
    print(f"            Levene p-value = {levene_p:.4f} (Statistically significant p < 0.05? -> {c1_stat_sig})")
    
    # Criterion 2: Does gated reach stable L2 loss < 0.08 in fewer steps than nongated?
    c2_passed = gated_summary["stable_step_roll_mean"] < nongated_summary["stable_step_roll_mean"]
    print(f"Criterion 2: Gated stable step ({gated_summary['stable_step_roll_mean']:.1f}) < Non-Gated stable step ({nongated_summary['stable_step_roll_mean']:.1f})? -> {c2_passed}")
    
    # Criterion 3: Does gated maintain target tracking overlap > 0.85 and reduce prediction loss on the target object by >= 15% compared to B1?
    c3_overlap_ok = gated_summary["test_overlap_mean"] > 0.85
    l2_loss_gated = gated_summary["test_l2_loss_mean"]
    l2_loss_b1 = b1_summary["test_l2_loss_mean"]
    reduction_pct = (l2_loss_b1 - l2_loss_gated) / l2_loss_b1 * 100.0
    c3_reduction_ok = reduction_pct >= 15.0
    c3_passed = c3_overlap_ok and c3_reduction_ok
    print(f"Criterion 3: Gated test overlap ({gated_summary['test_overlap_mean']:.4f}) > 0.85? -> {c3_overlap_ok}")
    print(f"            Gated reduces prediction loss vs B1 by {reduction_pct:.2f}% (>= 15%? -> {c3_reduction_ok})")
    print(f"            Overall Criterion 3 -> {c3_passed}")
    
    overall_passed = c1_passed and c2_passed and c3_passed
    print(f"\nOVERALL HYPOTHESIS VALIDATION RESULT: {'PASSED' if overall_passed else 'FALSIFIED'}")
    
    # Write execution log
    log_path = "archive/iter_004/results/execution_log.txt"
    with open(log_path, "w") as f:
        f.write("=== PHASE 2 SYSTEMATIC EVALUATION LOG ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write("--- AGGREGATED SUMMARY ---\n")
        f.write(df_summary.to_string())
        f.write("\n\n--- STATISTICAL TESTS ---\n")
        f.write(f"Levene F-statistic: {f_stat:.4f}, p-value: {levene_p:.4f}\n\n")
        f.write("--- FALSIFICATION CRITERIA CHECKS ---\n")
        f.write(f"Criterion 1 (lower L2 test variance): {'PASSED' if c1_passed else 'FAILED'} (std: gated={gated_summary['test_l2_loss_std']:.6f}, nongated={nongated_summary['test_l2_loss_std']:.6f}, p={levene_p:.4f})\n")
        f.write(f"Criterion 2 (sample efficiency): {'PASSED' if c2_passed else 'FAILED'} (stable step: gated={gated_summary['stable_step_roll_mean']:.1f}, nongated={nongated_summary['stable_step_roll_mean']:.1f})\n")
        f.write(f"Criterion 3 (stable tracking and loss reduction vs B1): {'PASSED' if c3_passed else 'FAILED'} (test overlap: gated={gated_summary['test_overlap_mean']:.4f}, loss reduction vs B1: {reduction_pct:.2f}%)\n")
        f.write(f"\nOverall hypothesis: {'PASSED' if overall_passed else 'FALSIFIED'}\n")
        
    print(f"Execution log saved to {log_path}")
    
    # 4. Plots Generation
    print("\nGenerating visual plots...")
    
    # A. Learning Curves
    plt.figure(figsize=(10, 6))
    for m in model_types:
        m_runs = [df_results[(df_results["model_type"] == m) & (df_results["seed"] == s)]["logs"].iloc[0] for s in seeds]
        steps = [l["step"] for l in m_runs[0]]
        l2_losses = np.array([[l["l2_sim_loss"] for l in r] for r in m_runs]) # (5, 5000)
        mean_losses = np.mean(l2_losses, axis=0)
        std_losses = np.std(l2_losses, axis=0)
        
        plt.plot(steps, mean_losses, label=f"{m} (mean)")
        plt.fill_between(steps, mean_losses - std_losses, mean_losses + std_losses, alpha=0.15)
        
    plt.axvline(x=1500, color='gray', linestyle='--', label='N=2 -> N=3 Transition')
    plt.xlabel("Training Step")
    plt.ylabel("L2 Prediction Similarity Loss")
    plt.title("Phase 2: L2 Prediction Loss Learning Curves (Mean +/- Std across 5 seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("archive/iter_004/results/learning_curves.png", dpi=150)
    plt.close()
    
    # B. Tracking Overlap (steps 1501-5000)
    plt.figure(figsize=(10, 6))
    for m in ['gated', 'nongated']:
        m_runs = [df_results[(df_results["model_type"] == m) & (df_results["seed"] == s)]["logs"].iloc[0] for s in seeds]
        steps = [l["step"] for l in m_runs[0]][1500:]
        overlaps = np.array([[l["overlap"] for l in r][1500:] for r in m_runs]) # (5, 3500)
        
        # Apply 100-step rolling average for smoother visualization
        window = 100
        smooth_overlaps = []
        for r_idx in range(len(overlaps)):
            r_over = overlaps[r_idx]
            smoothed = pd.Series(r_over).rolling(window=window, min_periods=1).mean().values
            smooth_overlaps.append(smoothed)
        smooth_overlaps = np.array(smooth_overlaps)
        
        mean_over = np.mean(smooth_overlaps, axis=0)
        std_over = np.std(smooth_overlaps, axis=0)
        
        plt.plot(steps, mean_over, label=f"{m} (smoothed mean)")
        plt.fill_between(steps, mean_over - std_over, mean_over + std_over, alpha=0.15)
        
    plt.xlabel("Training Step")
    plt.ylabel("Tracking Overlap (100-step smoothed)")
    plt.title("Phase 2: Physical Tracking Overlap under Self-Generated Attention (Steps 1501-5000)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("archive/iter_004/results/tracking_overlap.png", dpi=150)
    plt.close()
    
    # C. Token Traces
    plt.figure(figsize=(10, 6))
    g_run_s42 = df_results[(df_results["model_type"] == "gated") & (df_results["seed"] == 42)]["logs"].iloc[0]
    steps = [l["step"] for l in g_run_s42]
    loci_s42 = [l["token_locus"] for l in g_run_s42]
    
    all_gated_runs = [df_results[(df_results["model_type"] == "gated") & (df_results["seed"] == s)]["logs"].iloc[0] for s in seeds]
    all_loci = np.array([[l["token_locus"] for l in r] for r in all_gated_runs]) # (5, 5000)
    
    smoothed_locus_s42 = pd.Series(loci_s42).rolling(window=100, min_periods=1).mean().values
    mean_loci_across_seeds = np.mean(all_loci, axis=0)
    smoothed_mean_loci = pd.Series(mean_loci_across_seeds).rolling(window=100, min_periods=1).mean().values
    
    plt.scatter(steps[::10], loci_s42[::10], alpha=0.2, color='blue', s=5, label='Seed 42 Raw Token Locus')
    plt.plot(steps, smoothed_locus_s42, color='blue', linewidth=2, label='Seed 42 (100-step rolling avg)')
    plt.plot(steps, smoothed_mean_loci, color='orange', linewidth=2, linestyle='--', label='All 5 Seeds (mean rolling avg)')
    
    plt.axvline(x=1500, color='gray', linestyle='--', label='N=2 -> N=3 Transition')
    plt.yticks([0, 1, 2, 3, 4], ['L1-Seg0', 'L1-Seg1', 'L1-Seg2', 'L1-Seg3', 'L2-Global'])
    plt.xlabel("Training Step")
    plt.ylabel("Attention Locus")
    plt.title("Phase 2: Attention Token Locus and Gating/Curriculum Behavior")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("archive/iter_004/results/token_traces.png", dpi=150)
    plt.close()
    
    print("All plots generated and saved successfully under archive/iter_004/results/")

if __name__ == "__main__":
    main()
```

Write the file `src/train_thalamus.py`, run it using `.venv/Scripts/python.exe`, and monitor execution. Show the full printout when done.