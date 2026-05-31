## Task: Run the iter_038 2D Navigation Gate Probe experiment

### Context
The experiment script has been created at `src/run_iter038_nav_gates.py` and the pre-registration updated at `src/pre_registration.md`. This is a PURE PHYSICS ROLLOUT experiment — no training, no learned model, no representation work. We are testing whether a physically navigating pointer in a 2D arena resolves structural constraints that blocked behavioral validation.

### CRITICAL: Read the pre-registration FIRST
Before running anything, read `src/pre_registration.md` to understand the hypothesis, falsification criteria, gate definitions, and exit rules. You MUST adhere to these strictly.

### Instructions

1. **Run the experiment:**
   ```
   cd /home/user && python src/run_iter038_nav_gates.py
   ```
   
2. **Read ALL output files** from `archive/iter_038/results/`:
   - `probe_events.csv`
   - `diagnostic_collisions.csv`
   - `sanity_checks.txt`
   - `gate_summary.txt`
   - `trajectory_stats.csv`

3. **Apply the pre-committed exit rule EXACTLY as specified in the pre-registration:**

   **PASS (all three gates clear):** Write up the result using APPROVED reporting language ONLY:
   - "is consistent with bracket-admission"
   - "does not refute the null"
   - Report exact measured values vs thresholds
   - NEVER use: "demonstrates", "proves", "shows that 2D navigation works", "validates"
   - Hand off to HUMAN go/no-go on full 2D rebuild
   - Do NOT begin any 2D rebuild work
   - Document: specific gate measurements, estimated cost/scope of full rebuild, what carries over (M1, M3, iter_028 substrate, decoder-free), what must be rebuilt, caveat that gates passing does NOT guarantee bracket discriminates

   **FAIL (any gate fails):** Declare the behavioral-validation strategy not tractable within project scope.
   - Report the specific gate that failed, its measured value vs threshold
   - The six re-frame claims become the iter_039 scope
   - Do NOT suggest exploring different 2D parameterizations (this was the LAST sanctioned environment-design iteration)
   - Do NOT suggest going back to 1D (the 1D null is crystallized)

4. **Write a comprehensive exit-rule document** to `archive/iter_038/results/exit_rule_application.txt` with:
   - Per-gate results with exact values
   - The exit rule being applied
   - The outcome (PASS or FAIL branch)
   - If FAIL: the six re-frame claims that become iter_039 scope, quoted from the pre-registration
   - If PASS: the handoff document for the human go/no-go decision

5. **Report the following metrics in your final result:**
   - gate1_pass: bool (overall)
   - gate1b_pass: bool (overall)
   - gate2_pass: bool (overall)
   - all_gates_pass: bool
   - gate1_mean_probe_count: float (mean across seeds)
   - gate1b_mean_cv: float
   - gate1b_std_cv: float
   - gate2_per_seed_cvs: list of 5 floats
   - gate2_n_pass_seeds: int
   - sanity_all_pass: bool
   - per_seed_probe_counts: list of lists (5 seeds × 3 objects)
   - per_seed_collision_counts: list of lists (5 seeds × 3 objects, diagnostic)
   - exit_branch: "PASS" or "FAIL"
   - trajectory_coverage_mean: float (mean arena coverage across seeds)

Run the script and report all results. Do NOT modify any parameters or the script.