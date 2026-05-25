You are an AI executor sub-agent for iter_id 18.2.6. Your task is to run the corrected experiment script `src/run_phase18_experiments.py` and save all results.

The critical fix (`model.d_t = 3` upon cache load) has been applied to `src/run_phase18_experiments.py`. We need to run it now to obtain the correct experimental results!

### Steps:

1. Execute the corrected experiment script:
   ```bash
   python src/run_phase18_experiments.py
   ```
2. Monitor the output closely. Ensure it runs to completion and produces the following files under `archive/iter_018/results/`:
   - `summary_phase18.csv`
   - `audit_results_phase18.json`
   - `adaptation_curves_phase18.png`
   - `phase18_report.md`
3. Verify that the control sweep (Noisy-TV distractor) correctly triggers WUP probation at step 1800, computes the trend error and MDL ratio at step 1900, and performs the correct gating logic (either accepts or rejects, with EG-MDL arms S/S_alt rejecting and standard WUP-MDL baseline Arm P hopefully also showing the registered behavior).
4. After completion, print the final falsification audit table and the θ-sensitivity check. Let's do this!