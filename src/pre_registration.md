# RDF Scientific Pre-Registration

*   **Iteration:** 023
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis

The SFA objective has been functionally disabled by a 250× gradient imbalance
(sfa_weight=0.1 vs var_weight=25.0). Increasing sfa_weight to parity with
var_weight will activate SFA, causing normalized_dyn_var to drop below
normalized_coord_var (C5 criterion: z_dyn becomes slower than z_coord).

**Manager's key directive:** Once sfa_weight is large enough that C5 passes,
z_dyn *must* become slower by construction. Therefore, passing C5 only confirms
gradient propagation — it does not confirm identity encoding. The meaningful
test is whether functional SFA produces measurable identity–position separation.

We test whether SFA at parity produces measurable identity–position separation,
as evidenced by delta_R2_color improvement over the sfa_weight=0.1 baseline.
Delta_R2_color ≥ 0.10 over baseline is the primary falsification criterion.

Specifically:
- At sfa_weight ≥ 5.0, normalized_dyn_var < normalized_coord_var (SFA gradient
  reaches z_dyn). This is a prerequisite (gradient propagation), not a sufficient
  condition for the hypothesis.
- At sfa_weight = 25.0 (parity with var_weight), SFA-VICReg gradient conflict
  may cause instability or collapse; if so, a linear ramp from 0.1 to 25.0
  over the first 1000 steps will resolve it.
- At the best functional sfa_weight, d_max=16 will compound with SFA to
  achieve delta_R2_color ≥ 0.15 and delta_R2_identity ≥ 0.0 (breaking the
  negative identity trend from iter 022).

## 2. Falsification Criteria

**PRIMARY FALSIFICATION (delta_R2_color):** If no sfa_weight value in the sweep
[0.1, 1.0, 5.0, 10.0, 25.0] produces delta_R2_color improvement over the
sfa_weight=0.1 (A1) baseline of ≥ 0.10, the hypothesis is falsified. This means
that even when SFA is functionally active (C5 passes), the slowness prior does
not separate identity from position in this architecture.

**COMPOSITE M2 VIABILITY CRITERION:** There exists an sfa_weight ∈ [0.1, 25.0]
such that ≥ 3/5 seeds simultaneously satisfy:
  (a) C5: normalized_dyn_var < normalized_coord_var (SFA gradient reaches z_dyn),
  (b) delta_R2_color improvement ≥ 0.10 over A1 baseline,
  (c) per-dim std > 0.5 (non-collapse).

If no such sfa_weight exists, SFA + batch VICReg cannot jointly shape z_dyn into
a slow identity representation in this architecture.

**TERTIARY FALSIFICATION (unresolvable conflict):** If all sfa_weight ≥ 5.0 arms
collapse in ≥ 3/5 seeds (even with ramping), the SFA-VICReg gradient conflict is
unresolvable at effective SFA strengths, and the hypothesis that SFA can coexist
with batch VICReg at parity is falsified.

**C5 REINTERPRETED:** C5 (normalized_dyn_var < normalized_coord_var) is a
gradient-propagation verification check — necessary but **not sufficient** for
hypothesis support. Passing C5 only confirms that the SFA gradient reaches z_dyn;
it does not confirm that SFA encodes identity. The C5 criterion has been demoted
from primary to tertiary status.

## 3. Proposed Method

EXPERIMENT DESIGN: 7 arms × 5 seeds × 5000 steps. Seeds: [42, 123, 456, 789, 999].

ARM CONFIGURATIONS:

Arm A1 (Ctrl sfa=0.1): d_max=8, CGIR, CCR, d_t=3, sfa_weight=0.1
  → Direct replication of iter_022 Ctrl for within-run comparison.

Arm A2 (sfa=1.0): d_max=8, CGIR, CCR, d_t=3, sfa_weight=1.0
  → 10× increase; still below VICReg parity but tests gradient sensitivity.

Arm A3 (sfa=5.0): d_max=8, CGIR, CCR, d_t=3, sfa_weight=5.0
  → Intermediate; 5/25 = 20% of VICReg strength. First test of functional SFA.

Arm A4 (sfa=10.0): d_max=8, CGIR, CCR, d_t=3, sfa_weight=10.0
  → 40% of VICReg strength. Likely functional SFA with less conflict risk.

Arm A5 (sfa=25.0 fixed): d_max=8, CGIR, CCR, d_t=3, sfa_weight=25.0
  → Full parity with var_weight. Tests SFA effectiveness at maximum strength.
  Expected risk: gradient conflict with VICReg causing instability/collapse.

Arm A6 (sfa=25.0 ramp): d_max=8, CGIR, CCR, d_t=3, sfa_weight ramp 0.1→25.0 over 1000 steps
  → Contingency arm: linearly ramp sfa_weight from 0.1 to 25.0 over the first
  1000 steps, then hold at 25.0. Tests whether gradual SFA introduction avoids
  the collapse predicted for Arm A5. Implementation: in the training loop,
  compute effective_sfa_weight = 0.1 + (25.0 - 0.1) * min(1.0, step / 1000)
  and pass it via the sfa_weight forward-call override.

Arm B (d_max=16 sfa=10.0): d_max=16, CGIR, CCR, d_t=3, sfa_weight=10.0
  → Secondary arm pre-set to sfa_weight=10.0. Tests whether expanded channels
  + functional SFA compound positively. **Pre-commitment note:** This arm is
  pre-set to sfa_weight=10.0. If the optimal sfa_weight (from A-arms) is between
  5.0 and 10.0, Arm B may fail for the wrong reason. Do not interpret Arm B
  failure alone as falsifying the compound hypothesis.

ALL ARMS PRESERVE:
- primary_objective="sfa"
- sim_weight=25.0 (JEPA readout, stop-gradient per M2)
- var_weight=25.0, cov_weight=25.0 (batch VICReg, M1)
- dyn_readout="centroid_gated" (CGIR)
- ccr_mode="covariance", ccr_smooth_weight=10.0, ccr_spatial_weight=10.0
- pos_encoding="none"
- sub_features=1, dyn_source="spatial"
- d_t=3 frozen, gdasr_log_only=True
- Adam lr=1e-3, batch=32, replay_buffer=2000
- 5000 training steps

METRICS (same as iter_022, directly comparable):
1. C5 (gradient propagation): normalized_dyn_var < normalized_coord_var per seed
2. C1 (Collapse): per_dim_std < 0.5 in < 2/5 seeds per arm
3. Centroid MSE
4. delta_R2_color (primary criterion — improvement over A1 baseline)
5. delta_R2_identity
6. Normalized temporal variance (dyn and coord)
7. Slowness ratio
8. Per-dim std, collapse counts
9. Tracking quality
10. GDASR growth-point logs

CODE CHANGES:
1. src/run_phase0_sfa_sweep.py (NEW): Main experiment runner.
   - Based on run_phase0_sfa_archceiling.py structure.
   - 7 arms × 5 seeds × 5000 steps.
   - Ramp schedule for Arm A6: compute effective sfa_weight per step,
     pass via forward() sfa_weight parameter override.
   - Arm B pre-set to sfa_weight=10.0 (single-phase, no sequential dependency).
   - Same evaluation suite: normalized temporal variance, semantic probes,
     collapse checks, centroid MSE, tracking quality.
   - Results saved to archive/iter_023/results/.

2. src/models_dual_stream.py: NO CHANGES needed.
   The sfa_weight parameter already supports per-forward-call override.
   The ramp schedule is implemented in the training loop, not the model.

3. src/pre_registration.md: Updated with this plan.

SINGLE-PHASE EXECUTION: All 7 arms run simultaneously. Arm B is pre-set to
sfa_weight=10.0. If sfa_weight=10.0 turns out suboptimal, the A-arms
provide the data to identify the correct weight for a follow-up.

**Arm B pre-commitment caveat:** If optimal_sfa ∈ (5.0, 10.0), Arm B may fail
for the wrong reason. Arm B failure should not be interpreted as falsifying
the compound hypothesis on its own; the A-arm data supersedes it.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
