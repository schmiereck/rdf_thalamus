import pandas as pd

df = pd.read_csv("archive/iter_008/results/summary_phase8.csv")

print("Columns in CSV:", df.columns.tolist())

# Group by branch
branches = df["branch"].unique()

for branch in branches:
    b_df = df[df["branch"] == branch]
    avg_r_centroid = b_df["abs_r_centroid"].mean()
    avg_r_activation = b_df["abs_r_activation"].mean()
    avg_mse_centroid = b_df["mse_cent"].mean()
    avg_mse_activation = b_df["mse_act"].mean()
    avg_spatial_var = b_df["mean_var_3"].mean()
    
    # Recruitment rate: 
    # Let's count how many had d_t == 4 (which means recruited, either during N=3 passive training or during N=4 training)
    # Actually, in all seeds, the final d_t is 4. Let's verify:
    # (Since d_t is not in summary_phase8.csv, we can check recruitment_step. If it is != -1 OR the dimension was recruited in N=3 passive)
    # The recruitment step is -1 for seed 456 because it was recruited in N=3.
    # So recruitment rate is 100% for all of them if we count d_t == 4.
    # Let's print both: the percentage of seeds where recruitment_step != -1, and the actual 100% if we consider that it was recruited in N=3.
    rec_step_count = sum(b_df["recruitment_step"] != -1)
    rec_rate_by_step = (rec_step_count / len(b_df)) * 100
    
    collapsed_count = b_df["collapsed"].sum()
    non_collapsed_count = len(b_df) - collapsed_count
    
    print(f"\nConfiguration: {branch}")
    print(f"  Pearson |r| (Centroid)   : {avg_r_centroid:.4f}")
    print(f"  Pearson |r| (Activation) : {avg_r_activation:.4f}")
    print(f"  Decoding MSE (Centroid)  : {avg_mse_centroid:.4f}")
    print(f"  Decoding MSE (Activation): {avg_mse_activation:.4f}")
    print(f"  Soft Spatial Variance   : {avg_spatial_var:.4f}")
    print(f"  Recruitment Rate (Step)  : {rec_rate_by_step:.1f}% ({rec_step_count}/5 seeds recruited during Phase 2)")
    print(f"  Collapse Count           : {collapsed_count} seeds collapsed")
    print(f"  Non-Collapse Count       : {non_collapsed_count} seeds did not collapse")
