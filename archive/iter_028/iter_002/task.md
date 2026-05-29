You are executing Phase 28 Sub-task 2 of the Thalamus collapse-elimination campaign.

## Your Task
Execute the iter_028 experiment by running src/run_phase0_mask_dyn_sim_shared.py.

## Execution Command
From the project root directory, run:
```
python src/run_phase0_mask_dyn_sim_shared.py --sequential
```

Use `--sequential` to avoid parallel execution issues. If the sequential run takes too long, you may try with `--workers 4` instead.

If there are any import errors or syntax errors, fix them in the runner file and re-run. Do NOT modify models_dual_stream.py, models_separate_dyn.py, or environment.py.

## What to Expect
- 4 arms × 10 seeds = 40 runs, each 8000 steps
- Each run prints progress every 1000 steps
- Total expected wall time: ~30-60 minutes depending on hardware
- Output goes to archive/iter_028/results/

## After Runs Complete
1. Read the final_analysis.md from archive/iter_028/results/
2. Read the summary CSV from archive/iter_028/results/summary_iter_028.csv
3. Verify all 40 runs completed (no crashes)
4. Extract and report the key metrics:
   - Per-arm collapse rates (dual, eval-only, train-only)
   - Per-arm mean delta_R2_color (non-collapsed seeds only)
   - Per-arm mean_abs_corr (non-collapsed seeds only)
   - D0 vs C1 comparison on independent readouts
   - Whether F1/F2/F3/F4 gates triggered

## Success Criterion
All 40 runs complete. Results file exists with all 40 rows. Key metrics extracted for analysis.