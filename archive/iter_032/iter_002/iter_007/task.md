Now, please execute the FULL 4-arm × 20-seed experiment (80 runs, 8000 steps each) using the runner script `src/run_iter032.py`.
Run the command:
`python src/run_iter032.py --workers 6` (or adapt the worker count based on CPU/GPU capacity to ensure maximum speed and safety).

This is a large job, so make sure to monitor the terminal outputs to ensure there are no CUDA out-of-memory errors or other parallel execution issues.
Once all 80 runs complete, make sure the script writes:
1. `archive/iter_032/results/summary.csv`
2. `archive/iter_032/results/analysis.md`
Print the summary and the final analysis file contents to stdout so we can inspect them.