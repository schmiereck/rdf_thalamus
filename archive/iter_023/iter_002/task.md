Fix Unicode encoding issues in src/run_phase0_sfa_sweep.py and then execute the full experiment.

## Step 1: Fix Unicode Issues

In src/run_phase0_sfa_sweep.py, replace all non-ASCII characters with ASCII equivalents to avoid Windows cp1252 encoding errors:

1. Line 171: Replace "Δz²" with "dz^2" in the comment
2. Line 1151: Replace "Δ=" with "dR2_diff="
3. Line 1158: Replace "Δ=" with "dR2_diff="
4. Line 1160: Replace "Δ=" with "dR2_diff="
5. Line 1221: Replace "C5✓" with "C5[Y]" and "C5✗" with "C5[N]"
6. Line 1223: Replace "dR2✓" with "dR2[Y]" and "dR2✗" with "dR2[N]"
7. Line 1225: Replace "NC✓" with "NC[Y]" and "NC✗" with "NC[N]"
8. Line 1394: Replace "Δ=" with "dR2_diff="

Use sed or Python to do these replacements. Verify with a quick grep that no non-ASCII chars remain.

## Step 2: Run the Full Experiment

Execute:
```
cd /d %~dp0..
python src/run_phase0_sfa_sweep.py
```

This runs 7 arms × 5 seeds × 5000 steps. Expect ~30-60 minutes runtime.

The results should be saved to archive/iter_023/results/.

## Step 3: Report Results

After the experiment completes, read the key output files:
- archive/iter_023/results/audit_phase0_sfa_sweep.json
- archive/iter_023/results/summary_phase0_sfa_sweep.csv (look at the aggregated data)

Report the following metrics for each arm:
- Collapse rate (seeds with per_dim_std < 0.5)
- Mean delta_R2_color
- Mean delta_R2_identity
- Mean normalized_dyn_var and normalized_coord_var
- C5 pass rate (fraction of seeds where normalized_dyn_var < normalized_coord_var)
- Mean centroid MSE

And report the falsification audit results:
- PRIMARY: Does any arm achieve delta_R2_color improvement >= 0.10 over A1 baseline?
- COMPOSITE M2: Does any sfa_weight achieve >= 3/5 seeds passing C5 + dR2>=0.10 + non-collapse?
- TERTIARY: Do all sfa_weight >= 5.0 arms collapse >= 3/5 seeds?

IMPORTANT: If the run takes too long and you hit timeout, report whatever partial results are available.