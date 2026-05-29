# RDF Scientific Pre-Registration

*   **Iteration:** 028
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Removing the JEPA similarity loss gradient on z_dyn (mask_dyn_sim=True) from the
shared-backbone NonParametricJEPASpatial encoder does not destabilize the
VICReg-maintained per-dimension variance of z_dyn, AND z_dyn retains meaningful
identity encoding as measured by an independent (non-VICReg-enforced) readout.

Formally: Let C_C1 be the dual-criterion collapse rate of arm C1 (shared backbone,
mask_dyn_sim=True, weights 25/25/1) over 10 seeds. Let ΔR²_C1 be the mean
delta_R2_color of C1 across non-collapsed seeds. The hypothesis is:
(H1) C_C1 ≤ 0.10 (std-based collapse gate)
(H2) ΔR²_C1 ≥ D0_ΔR²_color + 0.05 AND mean_abs_corr_C1 ≤ mean_abs_corr_D0 + 0.05
     (C1 outperforms D0 on independent readouts relative to in-iteration null)

Both conditions must hold. This is the critical isolate that iter_027's Arm C
could not provide (it confounded separate-backbone with mask_dyn_sim).

Constructional acknowledgment: VICReg's variance hinge (γ - std, hinge at γ=1)
directly enforces per-dimension std ≥ 1, which trivially satisfies the collapse
gate (std < 0.5). Even a fully-confirmed C1+C2+C3 result will be reported as
"removing sim_loss_dyn from the z_dyn gradient path does not destabilize
VICReg-maintained variance" — NOT as "sim_loss_dyn causes collapse."

Pre-declared 2×2 prediction table (loss-competition hypothesis):
|                      | sim_dyn ON       | sim_dyn MASKED       |
|----------------------|------------------|----------------------|
| Shared backbone      | ~30% (D0, conf.) | ≤10% (C1, predicted) |
| Separate backbone    | ~30% (027 B)     | ~0% (027 C, conf.)   |

If C1 ≥ 20%: the separate backbone was load-bearing after all; hypothesis refuted.

## 2. Falsification Criterion
Four independent falsification conditions, any one of which refutes the hypothesis
or downgrades its interpretation:

F1: C1 collapse rate ≥ 0.20 (dual criterion) → mask_dyn_sim alone is insufficient
    on the shared backbone; the separate-backbone architecture was load-bearing.
    The Arm C signal from iter_027 collapses to that confound.

F2: C2 collapse rate ≥ 0.20 (fresh seed bank) → the C1 result is seed-dependent,
    not general.

F3: C3 collapse rate ≥ 0.20 (±10% weight perturbation) → the C1 result is not
    robust to reasonable hyperparameter variation.

F4: C1 passes the std gate (C_C1 ≤ 0.10) BUT (ΔR²_C1 < D0_ΔR²_color + 0.05 OR
    mean_abs_corr_C1 > mean_abs_corr_D0 + 0.05) → VICReg-maintained variance is
    constructional; z_dyn has variance but no meaningful semantic content
    improvement over the collapsing baseline.

Additional guard: D0 (shared-backbone JEPA+VICReg baseline replication) serves
as the in-iteration null reference. If D0 collapse rate < 0.20, the report must
note that the cov_weight=1 setting may contribute to reduced collapse (confound
awareness). D0 ΔR² and mean_abs_corr are the baselines for F4 and H2 gates.

Sample-size caveat: Fisher's exact test for 0/10 vs 3/10 gives p ≈ 0.21; the
design cannot formally distinguish 0% from 10–20% at this sample size. Results
are reported as point estimates with this limit explicitly noted.

Language constraints: Use "does not destabilize VICReg-maintained variance" or
"is consistent with"; do NOT use "breakthrough", "causal driver", "eliminated",
"BEST", "proves", "demonstrates", or "resolves."

## 3. Proposed Method
RESUME the pre-registered iter_028 experiment (same C1/C2/C3/D0 matrix, same
hyperparameters). Do NOT redesign. 13/40 runs are already complete; 27 remain.

Step 1: MODIFY src/run_phase0_mask_dyn_sim_shared.py to add resume logic and
per-seed timeouts:
(a) Before running each (arm, seed), check if the corresponding JSON result file
    already exists in archive/iter_028/results/runs/. If it exists and is valid
    (contains "arm" and "seed" keys), skip that seed — load the result from the
    existing JSON instead.
(b) Wrap each seed's training+evaluation in a per-seed timeout (600 seconds = 10
    minutes, generous for 8000 steps). If a seed times out, log it as a failed
    seed (mark collapsed=True, disqualified=True, with a timeout flag) and move on.
(c) Collect both new results and existing results into the final DataFrame.
(d) The existing results_dir is archive/iter_028/results/ — keep writing there.

Step 2: RUN remaining 27 seeds:
- C1 remaining: seeds 53, 71, 83, 97, 113, 127, 149 (7 seeds)
- C2 all: seeds 101, 103, 107, 109, 131, 137, 139, 151, 157, 163 (10 seeds)
- C3 all: seeds 7, 17, 31, 53, 71, 83, 97, 113, 127, 149 (10 seeds)
Use parallel execution (--workers flag, default min(cpu_count-1, 4)) for speed,
not --sequential (which was the source of the taskkill problem).

Step 3: After all 40 runs are accounted for (new + resumed), generate the final
analysis using the existing _generate_analysis() function. The analysis includes:
- Per-arm collapse rates (dual criterion: eval OR train per-dim std < 0.5)
- Per-seed train-vs-eval std gap table
- Gate check (C1, C2, C3 ≤ 10%)
- D0 vs C1 relative-threshold comparison (ΔR² and mean_abs_corr)
- Pre-registered outcome classification (F1-F4 / H1+H2)
- Parameter count comparison
- Sample-size caveat

Step 4: UPDATE src/pre_registration.md with the final plan (including 2×2 prediction
table and resume details).

FILES TO MODIFY:
- src/run_phase0_mask_dyn_sim_shared.py: add resume logic + per-seed timeout
- src/pre_registration.md: update with resume plan and 2×2 prediction table

FILES NOT TO MODIFY:
- src/models_dual_stream.py (mask_dyn_sim handled in runner, not model)
- src/models_separate_dyn.py (not needed for this experiment)
- src/environment.py

Total runs: 4 arms × 10 seeds = 40 (13 existing + 27 new).
Existing D0 results: 10/10, collapse rate 30% (seeds 17, 53, 83 collapsed).
Existing C1 results: 3/10, collapse rate 0% (seeds 7, 17, 31 — all non-collapsed).
Expected wall time for 27 new runs: ~25-35 minutes with parallel workers.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
