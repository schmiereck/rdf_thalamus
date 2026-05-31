## Task: Run the iter_036 foveated gaze benchmark

Execute the benchmark script and report results.

1. Run: `cd /home/user && python src/run_iter036_benchmark.py 2>&1`
2. If there are errors, read the relevant code sections, fix them, and re-run.
3. Once it completes successfully, read the output files in `archive/iter_036/results/` and report:
   - CV gate results per arm
   - Per-condition POMLRE (mean, std) per arm
   - All gate results (F1-F4, F6)
   - Sanity checks (S1-S6)
   - Branch distribution
   - Overall conclusion

The script should take about 5-10 minutes to run (48 episodes × 2000 steps each, plus CV gate).

If there are Python errors, read the relevant source sections and fix them. Common issues might be:
- Missing attributes or methods
- Type mismatches in the compute functions
- Import issues

After fixing, re-run until it completes successfully.