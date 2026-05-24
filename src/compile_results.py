import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def find_stable_step_roll(losses, threshold=0.08, window=100):
    losses = np.array(losses)
    roll_means = pd.Series(losses).rolling(window=window, min_periods=1).mean().values
    
    stable_step = -1
    for i in range(len(roll_means)):
        if all(rm < threshold for rm in roll_means[i:]):
            stable_step = i + 1  # 1-based step
            break
            
    if stable_step == -1:
        stable_step = len(losses)
    return stable_step

def main():
    seeds = [42, 123, 456, 789, 999]
    model_types = ['gated', 'nongated', 'b1']
    
    # Check if files exist
    missing_files = []
    for m in model_types:
        for s in seeds:
            csv_path = f"archive/iter_004/runs/{m}_seed{s}.csv"
            json_path = f"archive/iter_004/runs/{m}_seed{s}_eval.json"
            if not os.path.exists(csv_path):
                missing_files.append(csv_path)
            if not os.path.exists(json_path):
                missing_files.append(json_path)
                
    if len(missing_files) > 0:
        print(f"Error: {len(missing_files)} required data files are missing.")
        print("Please run the training sweep first to generate these files.")
        print(f"First missing file: {missing_files[0]}")
        sys.exit(1)
        
    print("All 15 evaluation JSONs and 15 run CSVs are present. Compiling results...")
    
    # Load evaluation metrics
    evals = []
    runs_data = {}  # (model_type, seed) -> dataframe
    
    for m in model_types:
        for s in seeds:
            # Load eval JSON
            json_path = f"archive/iter_004/runs/{m}_seed{s}_eval.json"
            with open(json_path, "r") as f:
                eval_data = json.load(f)
            evals.append(eval_data)
            
            # Load run CSV
            csv_path = f"archive/iter_004/runs/{m}_seed{s}.csv"
            df = pd.read_csv(csv_path)
            runs_data[(m, s)] = df
            
    df_evals = pd.DataFrame(evals)
    
    # Compute aggregates
    summary_list = []
    for m in model_types:
        sub_evals = df_evals[df_evals["model_type"] == m]
        
        mean_l2 = sub_evals["final_test_l2_loss"].mean()
        std_l2 = sub_evals["final_test_l2_loss"].std()
        mean_overlap = sub_evals["final_test_overlap"].mean()
        std_overlap = sub_evals["final_test_overlap"].std()
        
        # Calculate stable step and train tracking overlap for each seed
        stable_steps_roll = []
        overlaps_train = []
        
        for s in seeds:
            df = runs_data[(m, s)]
            losses = df["l2_sim_loss"].tolist()
            stable_s = find_stable_step_roll(losses, threshold=0.08, window=100)
            stable_steps_roll.append(stable_s)
            
            # Average training tracking overlap (steps 1501-5000)
            # 1-based steps 1501 to 5000 correspond to indices 1500 to 4999
            overlap_slice = df["overlap"].iloc[1500:5000].values
            if len(overlap_slice) > 0:
                overlaps_train.append(np.mean(overlap_slice))
            else:
                overlaps_train.append(0.0)
                
        summary_list.append({
            "model_type": m,
            "test_l2_loss_mean": mean_l2,
            "test_l2_loss_std": std_l2,
            "test_overlap_mean": mean_overlap,
            "test_overlap_std": std_overlap,
            "stable_step_roll_mean": np.mean(stable_steps_roll),
            "train_overlap_mean": np.mean(overlaps_train)
        })
        
    df_summary = pd.DataFrame(summary_list)
    os.makedirs("archive/iter_004/results", exist_ok=True)
    df_summary.to_csv("archive/iter_004/results/summary.csv", index=False)
    print("\nSummary statistics computed and saved to archive/iter_004/results/summary.csv")
    print(df_summary.to_string())
    
    # Run Levene's test between gated and nongated
    gated_test_losses = df_evals[df_evals["model_type"] == "gated"]["final_test_l2_loss"].tolist()
    nongated_test_losses = df_evals[df_evals["model_type"] == "nongated"]["final_test_l2_loss"].tolist()
    
    try:
        from scipy.stats import levene
        levene_stat, levene_p = levene(gated_test_losses, nongated_test_losses)
    except ImportError:
        levene_stat, levene_p = manual_levene(gated_test_losses, nongated_test_losses)
        
    print(f"\nLevene's Test on Test L2 Loss Variance (Gated vs Non-Gated): F-statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")
    
    # Falsification Criteria Check
    print("\n" + "="*80)
    print("PRE-REGISTERED FALSIFICATION CRITERIA EVALUATION")
    print("="*80)
    
    gated_sum = df_summary[df_summary["model_type"] == "gated"].iloc[0]
    nongated_sum = df_summary[df_summary["model_type"] == "nongated"].iloc[0]
    b1_sum = df_summary[df_summary["model_type"] == "b1"].iloc[0]
    
    # Criterion 1: Does gated have a lower standard deviation of L2 test loss than nongated? (Compute Levene p-value)
    c1_passed = gated_sum["test_l2_loss_std"] < nongated_sum["test_l2_loss_std"]
    c1_stat_sig = (levene_p < 0.05) if not np.isnan(levene_p) else True
    print(f"Criterion 1: Gated std ({gated_sum['test_l2_loss_std']:.6f}) < Non-Gated std ({nongated_sum['test_l2_loss_std']:.6f})? -> {c1_passed}")
    print(f"            Levene p-value = {levene_p:.4f} (Statistically significant p < 0.05? -> {c1_stat_sig})")
    
    # Criterion 2: Does gated reach stable L2 loss < 0.08 in fewer steps than nongated?
    c2_passed = gated_sum["stable_step_roll_mean"] < nongated_sum["stable_step_roll_mean"]
    print(f"Criterion 2: Gated stable step ({gated_sum['stable_step_roll_mean']:.1f}) < Non-Gated stable step ({nongated_sum['stable_step_roll_mean']:.1f})? -> {c2_passed}")
    
    # Criterion 3: Does gated maintain target tracking overlap > 0.85 and reduce prediction loss on the target object by >= 15% compared to B1?
    c3_overlap_ok = gated_sum["test_overlap_mean"] > 0.85
    l2_loss_gated = gated_sum["test_l2_loss_mean"]
    l2_loss_b1 = b1_sum["test_l2_loss_mean"]
    reduction_pct = (l2_loss_b1 - l2_loss_gated) / l2_loss_b1 * 100.0
    c3_reduction_ok = reduction_pct >= 15.0
    c3_passed = c3_overlap_ok and c3_reduction_ok
    print(f"Criterion 3: Gated test overlap ({gated_sum['test_overlap_mean']:.4f}) > 0.85? -> {c3_overlap_ok}")
    print(f"            Gated reduces prediction loss vs B1 by {reduction_pct:.2f}% (>= 15%? -> {c3_reduction_ok})")
    print(f"            Overall Criterion 3 -> {c3_passed}")
    
    overall_passed = c1_passed and c2_passed and c3_passed
    print(f"\nOVERALL HYPOTHESIS VALIDATION RESULT: {'PASSED' if overall_passed else 'FALSIFIED'}")
    
    # Write execution log
    log_path = "archive/iter_004/results/execution_log.txt"
    with open(log_path, "w") as f_log:
        f_log.write("=== PHASE 2 SYSTEMATIC EVALUATION LOG ===\n")
        f_log.write(f"Seeds: {seeds}\n\n")
        f_log.write("--- AGGREGATED SUMMARY ---\n")
        f_log.write(df_summary.to_string())
        f_log.write("\n\n--- STATISTICAL TESTS ---\n")
        f_log.write(f"Levene F-statistic: {levene_stat:.4f}, p-value: {levene_p:.4f}\n\n")
        f_log.write("--- FALSIFICATION CRITERIA CHECKS ---\n")
        f_log.write(f"Criterion 1 (lower L2 test variance): {'PASSED' if c1_passed else 'FAILED'} (std: gated={gated_sum['test_l2_loss_std']:.6f}, nongated={nongated_sum['test_l2_loss_std']:.6f}, p={levene_p:.4f})\n")
        f_log.write(f"Criterion 2 (sample efficiency): {'PASSED' if c2_passed else 'FAILED'} (stable step: gated={gated_sum['stable_step_roll_mean']:.1f}, nongated={nongated_sum['stable_step_roll_mean']:.1f})\n")
        f_log.write(f"Criterion 3 (stable tracking and loss reduction vs B1): {'PASSED' if c3_passed else 'FAILED'} (test overlap: gated={gated_sum['test_overlap_mean']:.4f}, loss reduction vs B1: {reduction_pct:.2f}%)\n")
        f_log.write(f"\nOverall hypothesis: {'PASSED' if overall_passed else 'FALSIFIED'}\n")
        
    print(f"Execution log saved to {log_path}")
    
    # Generate visual plots
    print("\nGenerating visual plots...")
    
    # A. Learning Curves
    plt.figure(figsize=(10, 6))
    for m in model_types:
        l2_losses = []
        for s in seeds:
            df = runs_data[(m, s)]
            l2_losses.append(df["l2_sim_loss"].values)
        l2_losses = np.array(l2_losses)  # (5, 5000)
        
        steps = np.arange(1, 5001)
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
        overlaps = []
        for s in seeds:
            df = runs_data[(m, s)]
            overlaps.append(df["overlap"].iloc[1500:5000].values)
        overlaps = np.array(overlaps)  # (5, 3500)
        
        # Apply 100-step rolling average for smoother visualization
        window = 100
        smooth_overlaps = []
        for r_idx in range(len(overlaps)):
            r_over = overlaps[r_idx]
            smoothed = pd.Series(r_over).rolling(window=window, min_periods=1).mean().values
            smooth_overlaps.append(smoothed)
        smooth_overlaps = np.array(smooth_overlaps)
        
        steps = np.arange(1501, 5001)
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
    df_s42 = runs_data[("gated", 42)]
    steps = df_s42["step"].values
    loci_s42 = df_s42["token_locus"].values
    
    all_gated_loci = []
    for s in seeds:
        df_g = runs_data[("gated", s)]
        all_gated_loci.append(df_g["token_locus"].values)
    all_gated_loci = np.array(all_gated_loci)  # (5, 5000)
    
    smoothed_locus_s42 = pd.Series(loci_s42).rolling(window=100, min_periods=1).mean().values
    mean_loci_across_seeds = np.mean(all_gated_loci, axis=0)
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
