Inside `src/run_iter031_partB.py`, do the following:
1. Replace all occurrences of the Unicode arrow symbol `→` in print statements with ASCII `->`.
2. Make sure any file writing (such as saving `partB_analysis.md`) uses `open(..., encoding="utf-8")` to ensure compatibility with Windows console and filesystem.
3. Run `python src/run_iter031_partB.py` to execute the full evaluation sweep (15 runs × 2 settings = 30 evaluations in total).
4. After running, verify that `archive/iter_031/results/partB_per_seed.csv`, `archive/iter_031/results/partB_summary.csv`, and `archive/iter_031/results/partB_analysis.md` are all correctly created and contain non-empty data.
5. Print out the summary of the gate evaluations from your run.