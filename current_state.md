# Current Research State
Phase: Collapse-Elimination Sweep — Measured Null

## Goal
Achieve a training regime where the NonParametricJEPASpatial encoder (JEPA+VICReg, d_max=8, d_t=3, centroid_gated readout) collapses on ≤10% of seeds over ≥10 seeds. This is a prerequisite for any further architectural or objective experiments.

## Confirmed (iter_026, agent 26.1)
- MEASURED NULL: No single-knob regime variation in {batch_size ∈ {32,64}, var_weight ∈ {25,50}, sim_weight ∈ {25, 0→25 ramp}, lr ∈ {3e-4, 1e-4}} achieves ≤10% collapse rate over 10 seeds under the dual criterion (eval-std < 0.5 OR train-std < 0.5).
- A0 (canonical repeat, lr=3e-4, B=32, buffer=4000): 40% collapse rate (dual). Regressed from iter_025's 30%, possibly due to buffer-size change.
- A1 (batch_size=64): 30% collapse rate (dual). Best arm, matches iter_025 v2. Insufficient for the gate.
- A2 (var_weight=50): 60% collapse rate (dual). Doubling VICReg variance weight WORSENED collapse — JEPA-vs-VICReg objective tension.
- A3 (sim_weight 0→25 ramp): 50% collapse rate (dual). JEPA warm-up worsened collapse.
- A4 (lr=1e-4): 100% collapse rate. LR too low for 8000-step training budget.
- TRAIN-VS-EVAL DISCREPANCY: Many runs maintain train-std > 0.5 but fail eval-std < 0.5 (e.g. A2: 10% train collapse vs 60% eval collapse). Representation is narrow, not fully collapsed.
- BUFFER SENSITIVITY: A0 regression from 30%→40% with buffer change (2000→4000) suggests replay buffer composition affects collapse probability.
- SANITY CHECK: No seeds disqualified (all losses ≤ 50, all met the VICReg floor at training time for non-collapsed seeds).

## Refuted
- HYPOTHESIS (iter_026 pre-registered): There exists a single-knob regime variation within the swept space that achieves ≤10% collapse. Falsified.

## Best Result
- A1 (batch_size=64): 30% collapse rate (dual). Same as iter_025 v2. No improvement found.

## In Progress
- None. Collapse-elimination sweep complete.

## NOT Established
- Whether multi-knob combinations (e.g. batch_size=64 + longer training) could reach ≤10%
- Whether the centroid_gated readout is the collapse bottleneck
- Whether a separate z_dyn encoder would solve collapse
- Whether longer training would help or hurt collapse rates
- Whether the buffer-size effect is a genuine confound or noise

## Open Questions (ordered by expected value)
1. Is the centroid_gated z_dyn readout the collapse bottleneck? (Comparing with mean readout is a clean, single-knob test within collapse-elimination scope.)
2. Would a combined regime (batch_size=64 + 16000 steps) overcome individual-arm weaknesses?
3. Why does A0 regress from 30% to 40% with the buffer change? Is this a genuine confound?
4. Is the train-vs-eval collapse discrepancy fixable by architectural changes, or is it fundamental?
5. Should the project pivot to a separate z_dyn encoder?
6. Is collapse a fundamental limitation of JEPA+VICReg on this architecture?
7. Would longer training (20000+ steps) help or is the 8000-step budget too short?
