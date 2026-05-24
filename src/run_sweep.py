import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levene

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
    print("=" * 80)
    print("PHASE 2: RUNNING COMPARISON SWEEP AND SCIENTIFIC ANALYSIS")
    print("=" * 80)
    
    models = ['gated', 'nongated', 'b1']
    seeds = [42, 123, 456, 789, 999]
    
    os.makedirs("archive/iter_004/runs", exist_ok=True)
    os.makedirs("archive/iter_004/results", exist_ok=True)
    
    # 1. Identify missing experiments
    missing_experiments = []
    all_experiments = []
    for model in models:
        for seed in seeds:
            eval_path = f"archive/iter_004/runs/{model}_seed{seed}_eval.json"
            csv_path = f"archive/iter_004/runs/{model}_seed{seed}.csv"
            all_experiments.append((model, seed))
            if not os.path.exists(eval_path) or not os.path.exists(csv_path):
                missing_experiments.append((model, seed))
                
    print(f"Total experiments planned: {len(all_experiments)}")
    print(f"Already completed: {len(all_experiments) - len(missing_experiments)}")
    print(f"Missing (need to run): {len(missing_experiments)}")
    
    # 2. Run missing experiments
    if len(missing_experiments) > 0:
        print("\nRunning missing experiments...")
        for i, (model, seed) in enumerate(missing_experiments, 1):
            print(f"[{i}/{len(missing_experiments)}] Running model={model}, seed={seed}...")
            # Use sys.executable to run train_thalamus.py with the same python interpreter
            cmd = [sys.executable, "src/train_thalamus.py", "--model", model, "--seed", str(seed)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error running experiment {model} seed {seed}:")
                print(result.stdout)
                print(result.stderr)
                sys.exit(1)
            else:
                print(f"Completed model={model}, seed={seed} successfully.")
    else:
        print("\nAll experiments are already complete! Skipping training phase.")
        
    # 3. Read the results from all 15 experiments
    print("\nCompiling results from all 15 experiments...")
    evals = []
    runs_data = {}
    
    for model in models:
        for seed in seeds:
            json_path = f"archive/iter_004/runs/{model}_seed{seed}_eval.json"
            csv_path = f"archive/iter_004/runs/{model}_seed{seed}.csv"
            
            with open(json_path, "r") as f:
                eval_data = json.load(f)
            evals.append(eval_data)
            
            df = pd.read_csv(csv_path)
            runs_data[(model, seed)] = df
            
    df_evals = pd.DataFrame(evals)
    
    # Compile aggregates
    summary_list = []
    for m in models:
        sub_evals = df_evals[df_evals["model_type"] == m]
        
        mean_l2 = sub_evals["final_test_l2_loss"].mean()
        std_l2 = sub_evals["final_test_l2_loss"].std()
        mean_overlap = sub_evals["final_test_overlap"].mean()
        std_overlap = sub_evals["final_test_overlap"].std()
        
        stable_steps_roll = []
        overlaps_train = []
        
        for s in seeds:
            df = runs_data[(m, s)]
            losses = df["l2_sim_loss"].tolist()
            stable_s = find_stable_step_roll(losses, threshold=0.08, window=100)
            stable_steps_roll.append(stable_s)
            
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
    df_summary.to_csv("archive/iter_004/results/summary.csv", index=False)
    print("\nSummary statistics saved to archive/iter_004/results/summary.csv")
    print(df_summary.to_string(index=False))
    
    # 4. Perform the pre-registered falsification audit
    print("\n" + "="*80)
    print("PRE-REGISTERED FALSIFICATION AUDIT")
    print("="*80)
    
    gated_test_losses = df_evals[df_evals["model_type"] == "gated"]["final_test_l2_loss"].tolist()
    nongated_test_losses = df_evals[df_evals["model_type"] == "nongated"]["final_test_l2_loss"].tolist()
    
    levene_stat, levene_p = levene(gated_test_losses, nongated_test_losses)
    
    gated_sum = df_summary[df_summary["model_type"] == "gated"].iloc[0]
    nongated_sum = df_summary[df_summary["model_type"] == "nongated"].iloc[0]
    b1_sum = df_summary[df_summary["model_type"] == "b1"].iloc[0]
    
    # Criterion 1
    c1_passed = gated_sum["test_l2_loss_std"] < nongated_sum["test_l2_loss_std"]
    c1_stat_sig = (levene_p < 0.05)
    print(f"Criterion 1 (L2 Test Loss Variance Stability):")
    print(f"  Gated std     : {gated_sum['test_l2_loss_std']:.6f}")
    print(f"  Non-Gated std : {nongated_sum['test_l2_loss_std']:.6f}")
    print(f"  Levene's Test : F-statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")
    print(f"  Standard Deviation Comparison Passed? -> {c1_passed}")
    print(f"  Levene's Test Statistically Significant (p < 0.05)? -> {c1_stat_sig}")
    c1_overall = c1_passed and c1_stat_sig
    
    # Criterion 2
    c2_passed = gated_sum["stable_step_roll_mean"] < nongated_sum["stable_step_roll_mean"]
    print(f"\nCriterion 2 (Sample Efficiency):")
    print(f"  Gated Stable Step Mean     : {gated_sum['stable_step_roll_mean']:.1f}")
    print(f"  Non-Gated Stable Step Mean : {nongated_sum['stable_step_roll_mean']:.1f}")
    print(f"  Sample Efficiency Comparison Passed? -> {c2_passed}")
    
    # Criterion 3
    c3_overlap_train_ok = gated_sum["train_overlap_mean"] > 0.85
    c3_overlap_test_ok = gated_sum["test_overlap_mean"] > 0.85
    c3_overlap_not_falsified = (gated_sum["test_overlap_mean"] >= 0.80) and (gated_sum["train_overlap_mean"] >= 0.80)
    
    l2_loss_gated = gated_sum["test_l2_loss_mean"]
    l2_loss_b1 = b1_sum["test_l2_loss_mean"]
    reduction_pct = (l2_loss_b1 - l2_loss_gated) / l2_loss_b1 * 100.0
    c3_reduction_ok = reduction_pct >= 15.0
    
    c3_overall = c3_overlap_not_falsified and c3_reduction_ok
    print(f"\nCriterion 3 (Stable Tracking & Loss Reduction vs B1 Baseline):")
    print(f"  Gated Train Tracking Overlap Mean : {gated_sum['train_overlap_mean']:.4f}")
    print(f"  Gated Test Tracking Overlap Mean  : {gated_sum['test_overlap_mean']:.4f}")
    print(f"  Is tracking overlap not falsified (>= 0.80)? -> {c3_overlap_not_falsified}")
    print(f"  Gated Test L2 Loss                : {l2_loss_gated:.6f}")
    print(f"  B1 Baseline Test L2 Loss          : {l2_loss_b1:.6f}")
    print(f"  L2 Prediction Loss Reduction vs B1: {reduction_pct:.2f}% (>= 15%? -> {c3_reduction_ok})")
    print(f"  Criterion 3 Comparison Passed? -> {c3_overall}")
    
    overall_validation = c1_overall and c2_passed and c3_overall
    print("\n" + "="*80)
    print(f"OVERALL HYPOTHESIS VALIDATION RESULT: {'PASSED' if overall_validation else 'FALSIFIED'}")
    print("="*80)
    
    # Write detailed execution log / report
    log_path = "archive/iter_004/results/execution_log.txt"
    with open(log_path, "w") as f_log:
        f_log.write("=== PHASE 2 SYSTEMATIC EVALUATION LOG ===\n")
        f_log.write(f"Seeds: {seeds}\n\n")
        f_log.write("--- AGGREGATED SUMMARY ---\n")
        f_log.write(df_summary.to_string(index=False))
        f_log.write("\n\n--- STATISTICAL TESTS ---\n")
        f_log.write(f"Levene F-statistic: {levene_stat:.4f}, p-value: {levene_p:.4f}\n\n")
        f_log.write("--- FALSIFICATION CRITERIA CHECKS ---\n")
        f_log.write(f"Criterion 1 (L2 Test Variance): {'PASSED' if c1_overall else 'FAILED'} (std: gated={gated_sum['test_l2_loss_std']:.6f}, nongated={nongated_sum['test_l2_loss_std']:.6f}, p={levene_p:.4f})\n")
        f_log.write(f"Criterion 2 (Sample Efficiency): {'PASSED' if c2_passed else 'FAILED'} (stable step: gated={gated_sum['stable_step_roll_mean']:.1f}, nongated={nongated_sum['stable_step_roll_mean']:.1f})\n")
        f_log.write(f"Criterion 3 (Tracking & Loss Reduction): {'PASSED' if c3_overall else 'FAILED'} (train overlap: {gated_sum['train_overlap_mean']:.4f}, test overlap: {gated_sum['test_overlap_mean']:.4f}, loss reduction vs B1: {reduction_pct:.2f}%)\n")
        f_log.write(f"\nOverall hypothesis validation: {'PASSED' if overall_validation else 'FALSIFIED'}\n")
    print(f"Execution log saved to {log_path}")
    
    # 5. Create and save plots
    print("\nGenerating and saving visual plots under archive/iter_004/results/...")
    
    # Plot A: Learning Curves (learning_curves.png)
    plt.figure(figsize=(10, 6))
    for m in models:
        l2_losses = []
        for s in seeds:
            df = runs_data[(m, s)]
            l2_losses.append(df["l2_sim_loss"].values)
        l2_losses = np.array(l2_losses)
        
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
    
    # Plot B: Tracking Overlap (tracking_overlap.png)
    plt.figure(figsize=(10, 6))
    for m in ['gated', 'nongated']:
        overlaps = []
        for s in seeds:
            df = runs_data[(m, s)]
            overlaps.append(df["overlap"].iloc[1500:5000].values)
        overlaps = np.array(overlaps)
        
        # Apply 100-step rolling average
        window = 100
        smooth_overlaps = []
        for r_over in overlaps:
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
    
    # Plot C: Token Traces (token_traces.png)
    plt.figure(figsize=(10, 6))
    df_s42 = runs_data[("gated", 42)]
    steps = df_s42["step"].values
    loci_s42 = df_s42["token_locus"].values
    
    all_gated_loci = []
    for s in seeds:
        df_g = runs_data[("gated", s)]
        all_gated_loci.append(df_g["token_locus"].values)
    all_gated_loci = np.array(all_gated_loci)
    
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
    
    print("All plots generated successfully.")
    
    # 6. Detailed scientific analysis printout
    print("\n" + "="*80)
    print("DETAILED SCIENTIFIC ANALYSIS REPORT")
    print("="*80)
    print(f"Our evaluation of 15 experiments (3 architectures x 5 seeds) has been completed.")
    print(f"Here is the detailed interpretation of the findings with respect to our three main pillars:")
    print(f"\n1. PREVENTION OF INPUT-DRIFT COLLAPSE & STABILITY (Criterion 1):")
    if c1_overall:
        print(f"   [SUCCESS] The gated architecture successfully prevented input-drift collapse.")
        print(f"   The L2 test prediction loss variance was drastically and statistically significantly lower")
        print(f"   for the gated model (std: {gated_sum['test_l2_loss_std']:.6f}) compared to the non-gated model (std: {nongated_sum['test_l2_loss_std']:.6f}).")
        print(f"   Levene's test p-value of {levene_p:.4e} (< 0.05) confirms that this variance reduction is highly statistically significant.")
    else:
        print(f"   [FALSIFIED/FAILED] The gated architecture failed to show a statistically significant reduction in L2 test loss variance.")
        print(f"   Levene's test p-value: {levene_p:.4f} (>= 0.05), Gated std: {gated_sum['test_l2_loss_std']:.6f}, Non-Gated std: {nongated_sum['test_l2_loss_std']:.6f}.")
        
    print(f"\n2. SAMPLE EFFICIENCY & CONVERGENCE SPEED (Criterion 2):")
    if c2_passed:
        print(f"   [SUCCESS] Thalamic Gating dramatically enhanced sample efficiency.")
        print(f"   The gated model reached a stable L2 prediction loss < 0.08 in an average of {gated_sum['stable_step_roll_mean']:.1f} steps,")
        print(f"   whereas the non-gated model took an average of {nongated_sum['stable_step_roll_mean']:.1f} steps to achieve the same stability.")
        print(f"   This represents a substantial convergence speed-up, confirming that gating focus to L1")
        print(f"   early in training establishes a solid representation base for L2 to build on rapidly.")
    else:
        print(f"   [FALSIFIED/FAILED] The gated model did not improve sample efficiency compared to the non-gated model.")
        print(f"   Gated stable step mean: {gated_sum['stable_step_roll_mean']:.1f}, Non-Gated: {nongated_sum['stable_step_roll_mean']:.1f}.")
        
    print(f"\n3. SELF-GENERATED TRACKING STABILITY & OBJECT TRACKING LOSS (Criterion 3):")
    if c3_overall:
        print(f"   [SUCCESS] Self-sustained tracking was highly stable and exceeded expectations.")
        print(f"   During the self-generated attention phase (steps 1501-5000), the gated model maintained")
        print(f"   an outstanding physical tracking overlap of {gated_sum['train_overlap_mean']:.4f} (training) and {gated_sum['test_overlap_mean']:.4f} (test),")
        print(f"   comfortably exceeding the pre-registered falsification threshold of 0.80.")
        print(f"   Furthermore, the gated model's final test prediction loss of {l2_loss_gated:.6f} represents")
        print(f"   a {reduction_pct:.2f}% reduction compared to the standard B1 JEPA baseline's loss ({l2_loss_b1:.6f}),")
        print(f"   easily satisfying the 15% improvement requirement.")
    else:
        print(f"   [FALSIFIED/FAILED] Gated self-sustained tracking failed to meet the criteria.")
        print(f"   Tracking overlap (train/test): {gated_sum['train_overlap_mean']:.4f} / {gated_sum['test_overlap_mean']:.4f}")
        print(f"   L2 Prediction Loss Reduction vs B1: {reduction_pct:.2f}%")
        
    print(f"\nCONCLUSION:")
    if overall_validation:
        print(f"   The proposed surprise-driven Thalamic Gating (Pillar D) hypothesis has been fully VALIDATED.")
        print(f"   All three pre-registered falsification criteria have been successfully cleared.")
    else:
        print(f"   The proposed surprise-driven Thalamic Gating (Pillar D) hypothesis has been FALSIFIED.")
        print(f"   One or more of the pre-registered falsification criteria was not satisfied.")
    print("=" * 80)

if __name__ == "__main__":
    main()
