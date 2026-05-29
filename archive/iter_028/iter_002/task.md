You are executing the iter_028 Thalamus collapse-elimination experiment. The script has already been modified with resume logic, per-seed timeout, and corrected timeout semantics. 13/40 runs already completed (D0: all 10, C1: 3 of 10). The script will detect and skip these.

## Execution Command

Run the experiment using PARALLEL workers (NOT --sequential):

```bash
cd /project && python src/run_phase0_mask_dyn_sim_shared.py --workers 4
```

The default --resume=True will automatically skip the 13 already-completed seeds.

## Expected Behavior

- The script should detect 13 existing JSON results and skip those seeds
- It should run the remaining 27 seeds:
  - C1 remaining: seeds 53, 71, 83, 97, 113, 127, 149 (7 seeds)
  - C2 all: seeds 101, 103, 107, 109, 131, 137, 139, 151, 157, 163 (10 seeds)
  - C3 all: seeds 7, 17, 31, 53, 71, 83, 97, 113, 127, 149 (10 seeds)
- Each seed takes ~60-90 seconds for 8000 training steps + evaluation
- Total expected wall time: ~25-35 minutes with 4 parallel workers

## Important Notes

1. Do NOT use --sequential flag (that caused the previous taskkill issue)
2. Do NOT add any new arms or change hyperparameters
3. If the script fails, check the error message carefully and fix any issues
4. The script should produce:
   - Individual JSON result files in archive/iter_028/results/runs/ for each seed
   - Checkpoint files in archive/iter_028/results/checkpoints/
   - A summary CSV: archive/iter_028/results/summary_iter_028.csv
   - A final analysis: archive/iter_028/results/final_analysis.md
5. After the script completes, read the final_analysis.md file and report the key results

## After Completion

Read and report:
1. archive/iter_028/results/final_analysis.md (the complete analysis)
2. The key metrics: per-arm collapse rates (PRIMARY, excluding timeouts), gate check results, F1-F4 outcome classification

## Troubleshooting

If the parallel execution stalls (common issue with PyTorch multiprocessing):
- Try running with --workers 2 instead of 4
- If that also fails, try --workers 1 (sequential but with timeout per seed)
- NEVER use taskkill — if a seed hangs, the per-seed timeout (600s) should handle it
- If you need to, you can also add --sequential (but with timeout) as a last resort

If any import errors occur, make sure you're running from the project root (/project).
