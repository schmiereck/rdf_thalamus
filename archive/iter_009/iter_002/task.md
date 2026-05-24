Please run the Phase 9 experimental sweep using the virtual environment's python executable:
`C:\Users\thomas\Projekte\uroboros-rdf\.venv\Scripts\python.exe src/run_phase9_experiments.py`

This will execute the 5-seed comparative sweep across the three arms:
1. Arm A (Gentle, lambda=0.01)
2. Arm B (Strong, lambda=0.10)
3. Arm C (Experimental DSMC)

The script already implements the pre-registered formula, controller stability rate limiting, and the temporal prediction safeguard.

Please verify that:
- The script runs successfully to completion without errors (this may take 10-15 minutes).
- All 5 seeds complete successfully.
- The results are saved in the `archive/iter_009/results/` directory, including:
  - `summary_phase9.csv`
  - `dsmc_trajectories.png`
  - `performance_comparison_phase9.png`
  - `phase9_report.md`
- Print the final averages from the aggregated table.