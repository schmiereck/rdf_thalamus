DIAGNOSE COLLAPSE PROBLEM in iter_025

The iter_025 experiment had 40-60% collapse rates across ALL arms including the control (JEPA+VICReg). The Research Manager identified this as a setup failure — we cannot draw conclusions when the control itself collapses at 60%.

YOUR TASK: Analyze the existing iter_025 training logs to diagnose WHY the model collapses so frequently. Look at:

1. Read the per-seed JSON result files from archive/iter_025/results/runs/ — focus on the per_dim_std and collapse status for each seed×arm combination
2. Read the training logs (CSV) for at least one collapsed and one non-collapsed run to understand the training trajectory (when does collapse happen — early, mid, late?)
3. Check whether collapse is correlated with specific seeds or happens randomly
4. Look at whether the supervised/contrastive arms collapse MORE than the control (suggesting the added loss causes it) or at the SAME rate (suggesting the base JEPA+VICReg setup is unstable)

Specifically examine:
- archive/iter_025/results/runs/a_jepavicreg_control_seed*.json (5 files) — for the control arm
- archive/iter_025/results/runs/b_supervised_color_probe_d_max_8_seed*.json (5 files) — for the supervised arm
- At least 2 training log CSVs (one collapsed, one non-collapsed) to see loss trajectories

Write your diagnostic findings to src/collapse_diagnosis.md including:
- Which seeds collapse in which arms
- Whether there's a pattern (early vs late collapse)
- Whether the added loss in arms B/C/D makes collapse worse vs the control
- The most likely root cause (learning rate, VICReg weight, gradient conflict, etc.)
- A concrete recommendation for how to stabilize training for the corrected experiment