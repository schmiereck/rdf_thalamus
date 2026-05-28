Run the full Architectural Ceiling experiment. This is the primary data-collection phase for iter_022.

## INSTRUCTIONS

1. First, clean up any dry-run results from archive/iter_022/results/ (they only have 5-step data, which is useless):
```bash
rm -rf /home/user/archive/iter_022/results/*
```

2. Run the full experiment:
```bash
cd /home/user && python src/run_phase0_sfa_archceiling.py
```

This will run 4 arms × 5 seeds × 5000 steps = 20 training runs. Each run takes about 2-3 minutes. Total expected time: ~60-90 minutes.

3. After the experiment completes, read the output files and report:
- archive/iter_022/results/aggregated_phase0_archceiling.csv (per-arm mean/std)
- archive/iter_022/results/audit_phase0_archceiling.json (falsification audit)
- archive/iter_022/results/ARCHCEILING_COMPARISON_REPORT.md

4. For each arm, report these key metrics (mean over 5 seeds):
   - collapsed seeds count
   - centroid_mse_mean
   - delta_r2_color (absolute)
   - delta_r2_identity (absolute)
   - improvement in delta_r2_color vs Ctrl
   - improvement in delta_r2_identity vs Ctrl
   - normalized_dyn_var
   - normalized_coord_var
   - sfa_effective (bool)
   - slowness_ratio (legacy, for comparison with iter_021)
   - delta_corr_mean (centroid tracking quality)
   - per_dim_std_mean

5. For Arm C (K=4), also report the per-sub-feature identity probe results if available:
   - Whether sub-features show selective encoding (one k encodes R, another G, etc.)
   - Or distributed encoding (all k have similar R² across identity dims)

6. Apply the falsification audit per the pre-registered criteria:
   C1: Ctrl collapsed < 2 AND Arm C collapsed < 2
   C2: Arm C MSE ≤ 1.10 × Ctrl MSE
   C3: improvement(delta_R2_color, Arm C vs Ctrl) ≥ 0.10
   C4: improvement(delta_R2_identity, Arm C vs Ctrl) ≥ 0.10
   C5: normalized_dyn_var[Arm C] < normalized_coord_var[Arm C] (advisory)
   
   OVERALL: C1 AND C2 AND C4 → hypothesis validated

7. Apply the Manager's interpretive guardrails:
   - If C4 passes but C5 fails: "capacity enables encoding; SFA is along for the ride"
   - If C4 passes but sub-feature probes show no selective encoding: "capacity enables identity encoding, but SFA does not produce disentangled sub-feature specialization"

IMPORTANT: This is a long-running computation. Just launch it and wait. The script handles everything including evaluation and audit. Once it's done, read the results and report them.

If there's an error in the script, fix it and re-run. But the dry-run already passed, so it should work.