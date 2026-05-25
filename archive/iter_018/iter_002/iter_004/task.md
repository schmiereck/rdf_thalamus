You are an AI executor sub-agent for iter_id 18.2.4. Your task is to run the existing experiment script `src/run_phase18_experiments.py`, verify that it runs to completion without any encoding or execution errors, and check that all results are saved to `archive/iter_018/results/`.

### Steps:

1. Execute the existing experiment script:
   ```bash
   python src/run_phase18_experiments.py
   ```
2. Monitor the output. If there is any print statement or file write raising a `UnicodeEncodeError` (e.g. from math symbols in `report_lines` or elsewhere), fix it immediately. (Hint: make sure any console print statements are strictly ASCII and any file opens with unicode characters use `encoding="utf-8"`).
3. Confirm that the following files are produced and correctly populated under `archive/iter_018/results/`:
   - `summary_phase18.csv`
   - `audit_results_phase18.json`
   - `adaptation_curves_phase18.png`
   - `phase18_report.md`
4. Print a summary of the falsification audit results from the run. Let's do this!