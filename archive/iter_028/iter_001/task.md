You are modifying files for the iter_028 Thalamus collapse-elimination experiment. TWO files need modification. Do NOT run any experiments — only modify the files.

## Task 1: Update src/pre_registration.md

Add two sections to the existing pre-registration file. Keep ALL existing content intact and append these additions at the end, before the "Created automatically" line:

### Addition A: Resumed Runs Code-Equivalence Declaration

Add a section declaring that the 13/40 already-completed runs (D0: 10 seeds, C1: 3 seeds) were produced with IDENTICAL code to the remaining 27 runs. Specifically state:
- The model class (NonParametricJEPASpatial), loss computation, evaluation, and matching logic are unchanged between the existing 13 runs and the 27 remaining runs.
- The only code changes in this iteration are: (a) resume logic that skips existing JSON result files, (b) per-seed timeout wrapper, (c) updated analysis generation to handle timeout results separately.
- If there is ANY doubt about code equivalence, the 13 runs must be re-run, not reused. The pre-registration declares there is no doubt based on the JSON result file audit (they contain the expected arm names, seed values, parameter counts, and metric structure).

### Addition B: Timeout Semantics Protocol

Add a section specifying timeout handling per the Research Manager's requirement:
- A per-seed timeout is an ENGINEERING failure, NOT a representation failure.
- Timeout handling: If a seed exceeds the per-seed timeout (600 seconds), it is logged with a "timeout" flag but is NOT counted as collapsed for the primary collapse rate.
- Three reporting tiers:
  (a) PRIMARY: collapse rate excluding timeouts (only genuine std-based collapse counts)
  (b) SENSITIVITY: collapse rate including timeouts as failures (upper bound)
  (c) TIMEOUT COUNT: number of timed-out seeds per arm, reported separately
- If timeouts exceed 1 per arm, the run is not interpretable and must be re-launched with a longer budget.
- The pre-registered gates (F1-F4, H1-H2) are evaluated on the PRIMARY (excluding timeouts) collapse rate only.

## Task 2: Modify src/run_phase0_mask_dyn_sim_shared.py

Make these specific modifications:

### Modification A: Add resume logic to main()

Before building the task list, add code that scans archive/iter_028/results/runs/ for existing JSON result files. For each arm+seed combination, check if a JSON file exists with matching arm name and seed. If it exists and contains both "arm" and "seed" keys, skip that seed and load the result from the JSON.

Specifically:
1. Add a function `load_existing_results(runs_dir)` that:
   - Scans all .json files in runs_dir
   - For each file, loads it and checks for "arm" and "seed" keys
   - Returns a dict mapping (arm_name, seed) -> result_dict
2. In main(), after creating the tasks list, filter out tasks whose (arm_name, seed) already exists in the loaded results
3. After all new runs complete, merge the loaded existing results with the new results before creating the DataFrame

### Modification B: Add per-seed timeout with correct semantics

1. Add a `run_single_with_timeout` function that wraps `run_single` with a per-seed timeout (600 seconds). Since `run_single` involves PyTorch training, use `concurrent.futures.ProcessPoolExecutor` with `future.result(timeout=600)` for each seed. If timeout occurs:
   - Log a result dict with: arm, seed, collapsed=False (NOT collapsed — timeout is engineering failure), collapsed_eval=False, collapsed_train=False, timeout=True, disqualified=False
   - The timeout flag distinguishes this from genuine collapse
   - Add "timeout" column to the DataFrame

2. In the sequential execution path, also use timeout handling per seed.
3. In the parallel execution path (ProcessPoolExecutor), use `future.result(timeout=600)` for each future, but note that with parallel workers, individual futures already have their own processes. Add timeout handling there too.

### Modification C: Update _generate_analysis() for timeout handling

1. In the per-arm summary section, add:
   - Timeout count per arm
   - PRIMARY collapse rate (excluding timeouts)
   - SENSITIVITY collapse rate (including timeouts as failures)
2. The gate check should use the PRIMARY collapse rate (excluding timeouts)
3. Add a "Timeout Audit" section showing timeout count per arm and whether the run is interpretable (≤1 timeout per arm)

### Modification D: Add --resume flag (default True)

Add a `--resume` flag (default True). When True, the script checks for and skips existing results. When False, it re-runs all seeds from scratch.

## Important constraints:
- Do NOT modify src/models_dual_stream.py, src/models_separate_dyn.py, or src/environment.py
- Do NOT add new arms or change hyperparameters
- Keep all existing functionality intact
- Make the script runnable with `python src/run_phase0_mask_dyn_sim_shared.py` (defaults should work)
- After making all modifications, verify the script can be imported without errors by running: `cd /project && python -c "import ast; ast.parse(open('src/run_phase0_mask_dyn_sim_shared.py').read()); print('Syntax OK')"`
