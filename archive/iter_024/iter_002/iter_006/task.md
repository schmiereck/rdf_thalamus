Write a Python script to parse `archive/iter_024/results/aggregated_phase0_sfa_multistep.csv` and compile a comprehensive scientific report at `archive/iter_024/results/phase0_multistep_sfa_report.md` along with a structured JSON audit `archive/iter_024/results/audit_results.json`.

The markdown report must include:
1. Introduction & Executive Summary of Iteration 024.
2. The pre-registered Hypotheses (Part A: Multi-step SFA, Part B: Temporal Contrastive NT-Xent) and Falsification Criteria.
3. Detailed results table with Arm, Seeds, delta_R2_color, within_traj_var, between_traj_var, shuffled_delta_r2_color, and centroid_mse_mean.
4. Falsification Evaluation:
   - Part A: SFA (Arms A-C and Arm E). Check if M2 is refuted.
   - Part B: Temporal Contrastive (Arm D). Check if Arm D is refuted or consistent.
   - Variance analysis: how multi-step horizon affects within/between trajectory variance and whether it acts purely mechanically or shapes representation semantically.
   - Diagnostic validation: Shuffled-frame control findings.
5. Narrative on Pivot to Object-Tracking-ID Contrastive in Iteration 025 (as suggested in pre-registration).

Run the python script to generate these files, then print the Markdown report to stdout so we can review it.