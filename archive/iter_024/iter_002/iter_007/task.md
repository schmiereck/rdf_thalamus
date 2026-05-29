Create a Python script src/fix_report.py that correctly parses the aggregated CSV `archive/iter_024/results/aggregated_phase0_sfa_multistep.csv` and regenerates both `archive/iter_024/results/phase0_multistep_sfa_report.md` and `archive/iter_024/results/audit_results.json` using the correct "mean" and "std" column indices.

Specifically:
- Arm A (k=20 d_max=8) actual mean delta_R2_color: -0.01116, std: 0.20369
- Arm B (k=50 d_max=8) actual mean delta_R2_color: 0.03365, std: 0.07118
- Arm C (k=100 d_max=8) actual mean delta_R2_color: -0.07390, std: 0.05959
- Arm D (Contrastive d_max=8) actual mean delta_R2_color: -0.01301, std: 0.17784
- Arm E (k=50 d_max=16) actual mean delta_R2_color: 0.03031, std: 0.06039
- Arm F (Diagnostic sim=0 k=50 d_max=8) actual mean delta_R2_color: -0.07740

Parse all columns by using plain CSV reader:
- row[1] = arm name
- row[2] = has_collapsed mean, row[3] = has_collapsed std
- row[10] = centroid_mse_mean mean, row[11] = centroid_mse_mean std
- row[26] = delta_r2_color mean, row[27] = delta_r2_color std
- row[46] = within_traj_var mean, row[47] = within_traj_var std
- row[48] = between_traj_var mean, row[49] = between_traj_var std
- row[52] = shuffled_delta_r2_color mean, row[53] = shuffled_delta_r2_color std

Write the Markdown report and JSON audit files with the correct means and stds in the tables and discussion! Then run the python script and print the updated Markdown file to stdout.