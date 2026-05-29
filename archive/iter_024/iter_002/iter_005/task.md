Run the full 5000-step training for all 6 arms and seeds using the parallelized runner src/run_phase0_sfa_multistep.py.

Do NOT pass `--dry-run`. Just run:
`python src/run_phase0_sfa_multistep.py --workers 8` (or use the default number of workers).

Make sure:
1. It runs all 26 seed-arm configurations (Arms A-E: 5 seeds each, Arm F: 1 seed) for 5000 steps to completion.
2. It monitors progress and prints status updates.
3. Once finished, it verifies that the output files are successfully written under `archive/iter_024/results/`:
   - `summary_phase0_sfa_multistep.csv`
   - `summary_phase0_sfa_multistep_cp2000.csv`
   - `aggregated_phase0_sfa_multistep.csv`
4. Verify that each file contains 26 rows of valid numerical data.
5. Print a summary of the compiled average metrics per arm (e.g. delta_R2_color, within_traj_var, between_traj_var, shuffled_delta_r2_color) to stdout.