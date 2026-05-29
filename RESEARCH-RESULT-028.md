# RDF Research Result — Iteration 028

## 1. Hypothesis
phase-28: mask_dyn_sim on shared backbone does NOT eliminate z_dyn collapse (F1 falsified); hard-seed pattern (seeds 53, 71) identified; separate-backbone architecture is load-bearing

## 2. Status
ok

## 3. Analysis
This phase completed the pre-registered iter_028 experiment: a 4-arm × 10-seed
matrix testing whether mask_dyn_sim alone (on the shared backbone) prevents
z_dyn collapse. The experiment was a direct continuation of iter_027, which
identified sim_loss_dyn as the collapse driver on the separate backbone but
could not isolate the loss-adjustment effect from the architecture effect.

Sub-agent 28.1 modified the script to add resume logic (skipping 13 existing
results), per-seed timeout with correct semantics (engineering failure ≠
representation failure), and updated the pre-registration with code-equivalence
declaration and timeout protocol. Sub-agent 28.2 ran the remaining 27 seeds
with parallel workers, completing all 40 runs with zero timeouts.

Key findings:
1. F1 FALSIFIED: C1 collapse rate = 20% (2/10), exceeding the ≤10% gate.
   mask_dyn_sim alone on the shared backbone does not prevent collapse.
2. Seed-dependence: C2 (fresh seeds) achieved 0% collapse. The same
   configuration with different seeds produces different outcomes. Seeds
   53 and 71 are "hard seeds" that collapse under both C1 and C3.
3. H2 PASSED: When C1 does not collapse, its semantic encoding is
   substantially better than D0 (ΔR² 0.23 vs 0.05, mean_abs_corr 0.52 vs 0.99).
4. The 2×2 table now shows that the separate-backbone architecture contributes
   a structural benefit beyond the loss adjustment. The path from 30% (shared,
   sim_dyn ON) to 0% (separate, sim_dyn MASKED) requires BOTH interventions.

The hard-seed pattern is the most actionable finding. Seeds 53 and 71 collapse
consistently across C1 and C3 (same architecture, different weight perturbation),
but seed 53 also collapsed in D0 and seed 71 did NOT collapse in D0. This
suggests the collapse mode under mask_dyn_sim is different from the collapse
mode under full JEPA+VICReg — the former is more severe (std ~0.01 vs ~0.45)
and may involve a different failure mechanism.

The constructional caveat remains: VICReg's variance hinge is the mechanism
that prevents collapse when it works, and the question is why it fails on
certain seeds. The sim_loss_dyn gradient appears to be a competing force
that can push VICReg below its operating point, but removing it does not
guarantee VICReg can maintain its guarantee on all seeds.


## 4. Metrics
{'total_seeds': 40, 'existing_resumed': 13, 'new_completed': 27, 'timeouts': 0, 'd0_collapse_rate_primary': 0.3, 'c1_collapse_rate_primary': 0.2, 'c2_collapse_rate_primary': 0.0, 'c3_collapse_rate_primary': 0.2, 'd0_mean_abs_corr': 0.999, 'c1_mean_abs_corr': 0.521, 'c2_mean_abs_corr': 0.435, 'c3_mean_abs_corr': 0.474, 'd0_delta_r2_color': 0.054, 'c1_delta_r2_color': 0.231, 'c2_delta_r2_color': 0.514, 'c3_delta_r2_color': 0.168, 'h2_relative_gate': 'PASS', 'f1_outcome': 'FALSIFIED', 'f2_outcome': 'NOT_TRIGGERED', 'f3_outcome': 'NOT_ROBUST', 'param_count': 80336, 'collapsed_seeds_C1': [53, 71], 'collapsed_seeds_D0': [17, 53, 83], 'collapsed_seeds_C3': [53, 71]}

## 5. Notes
F1 falsified; mask_dyn_sim on shared backbone insufficient. Hard-seed pattern identified. C2 at 0% reveals seed-dependence.

---
*Note: This is an automated summary as the Research Manager did not provide a full milestone report.*
