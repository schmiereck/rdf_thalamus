# RDF Research Result — Iteration 034

## 1. Hypothesis
phase-34: v1 MAPE benchmark falsified (pointer-object noise sensitivity); v2 MALRE benchmark validated as coverage-discrimination test (active-vs-passive gap=0.83) but underpowered for ORACLE-vs-RANDOM discrimination (gap=0.031, 3/8 seeds)

## 2. Status
ok

## 3. Analysis
Phase 34 set out to validate a behavioral benchmark for iter_035's perception
sufficiency test. Two metric designs were tested:

v1 (MAPE): Used least-squares mass estimation from all collision types with
velocity noise. Falsified because pointer-object collisions are too noisy —
the formula m_i = 10*(-Δv_ptr)/Δv_obj has extreme sensitivity, and with
hundreds of such rows, the least-squares system is overwhelmed. Active policies
that create MORE pointer-object collisions get WORSE mass estimates.

v2 (MALRE): Used MEDIAN of mass-ratio estimates from object-object collisions
only. Validated with all gates passing. However, the ORACLE-vs-RANDOM gap is
negligible (0.031, ORACLE wins only 3/8 seeds). The PASSIVE gap is a coverage
artifact (PASSIVE has no data for most pairs → max penalty).

The fundamental issue is that in 1D elastic collisions with 3 objects and a
movable pointer, object-object collisions happen naturally and abundantly
regardless of the pointer policy. Targeting specific objects only changes the
pointer-object collision distribution, which doesn't affect object-object
collision quality. The MALRE metric (based on object-object ratios) therefore
can't discriminate targeting quality.

For iter_035, the options are:
1. Use the v2 MALRE benchmark as-is, accepting it only discriminates
   active-vs-passive (not targeting quality)
2. Design a metric that uses pointer-object collision data more robustly
   (e.g., per-object coverage as a primary metric instead of mass estimation)
3. Use a different approach entirely: instead of measuring how well the agent
   estimates hidden parameters, measure how quickly it achieves coverage of
   the collision manifold (time-to-full-coverage as the metric)


## 4. Metrics
{'v1_mape_oracle': 1.005, 'v1_mape_random': 0.999, 'v1_mape_passive': 0.597, 'v1_result': 'FALSIFIED', 'v2_malre_oracle': 0.503, 'v2_malre_random': 0.534, 'v2_malre_passive': 1.333, 'v2_oracle_random_gap': 0.031, 'v2_passive_oracle_gap': 0.83, 'v2_g1_pass': True, 'v2_g2_pass': True, 'v2_g3_pass': True, 'v2_g4_pass': True, 'v2_all_sanity_pass': True, 'v2_result': 'VALIDATED_with_caveats', 'oracle_wins_vs_random': '3/8 seeds', 'n_runs': 24}

## 5. Notes
Benchmark validated as coverage discrimination test; active-vs-passive gap is strong but ORACLE-vs-RANDOM gap is negligible.

---
*Note: This is an automated summary as the Research Manager did not provide a full milestone report.*
