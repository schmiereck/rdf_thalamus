# Research Manager Log - Iteration 023

## Iteration 023 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
The SFA objective has been functionally disabled by a 250× gradient imbalance
(sfa_weight=0.1 vs var_weight=25.0). Increasing sfa_weight to parity with
var_weight will activate SFA, causing normalized_dyn_var to drop below
normalized_coord_var (C5 criterion: z_dyn becomes slower than z_coord). With
functional SFA, the slow identity stream will begin to separate from the fast
position stream, producing delta_R2_color ≥ 0.10 (improvement over
sfa_weight=0.1 baseline of +0.050).

Specifically:
- At sfa_weight ≥ 5.0, normalized_dyn_var < normalized_coord_var (SFA effective).
- At sfa_weight = 25.0 (parity with var_weight), SFA-VICReg gradient conflict
  may cause instability or collapse; if so, a linear ramp from 0.1 to 25.0
  over the first 1000 steps will resolve it.
- At the best functional sfa_weight, d_max=16 will compound with SFA to
  achieve delta_R2_color ≥ 0.15 and delta_R2_identity ≥ 0.0 (breaking the
  negative identity trend from iter 022).

**Proposed Falsification Criterion:**
PRIMARY FALSIFICATION: If no sfa_weight value in the sweep [0.1, 1.0, 5.0,
10.0, 25.0] produces normalized_dyn_var < normalized_coord_var across ≥ 3/5
seeds, then SFA cannot shape z_dyn even at gradient parity, and the
hypothesis is falsified. This would mean the SFA objective itself (not its
weight) is structurally incompatible with the CNN architecture or the
z_dyn readout.

SECONDARY FALSIFICATION: If sfa_weight achieves C5 (SFA effective) but the
best delta_R2_color improvement over the sfa_weight=0.1 baseline is < 0.05,
then functional SFA does not translate to semantic disentanglement — the
slowness prior does not separate identity from position in this architecture.

TERTIARY FALSIFICATION: If all sfa_weight ≥ 5.0 arms collapse in ≥ 3/5 seeds
(even with ramping), the SFA-VICReg gradient conflict is unresolvable at
effective SFA strengths, and the hypothesis that SFA can coexist with
batch VICReg at parity is falsified.

**Proposed Method:**
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

Arm B (d_max=16 at best sfa): d_max=16, CGIR, CCR, d_t=3, sfa_weight=<BEST>
  → Secondary arm using the best sfa_weight from A2–A6 (determined by C5 pass
  rate + lowest collapse rate + best delta_R2_color). If multiple pass, use
  the lowest effective weight. If none pass C5, use sfa_weight=10.0 as the
  most likely candidate. Tests whether expanded channels + functional SFA
  compound positively.

ALL ARMS PRESERVE:
- primary_objective="sfa"
- sim_weight=25.0 (JEPA readout, stop-gradient per M2)
- var_weight=25.0, cov_weight=25.0 (batch VICReg, M1)
- dyn_readout="centroid_gated" (CGIR)
- ccr_mode="covariance", ccr_smooth=10, ccr_spatial=10
- pos_encoding="none"
- sub_features=1, dyn_source="spatial"
- d_t=3 frozen, gdasr_log_only=True
- Adam lr=1e-3, batch=32, replay_buffer=2000
- 5000 training steps

METRICS (same as iter_022, directly comparable):
1. C5 (SFA effective): normalized_dyn_var < normalized_coord_var per seed
2. C1 (Collapse): per_dim_std < 0.5 in < 2/5 seeds per arm
3. C2 (Centroid MSE): arm MSE ≤ 1.10 × A1 MSE
4. C3 (Color): delta_R2_color improvement(Arm - A1) ≥ 0.05
5. C4 (Identity): delta_R2_identity improvement(Arm - A1) ≥ 0.05
6. Normalized temporal variance (dyn and coord) — primary C5 indicator
7. Slowness ratio (dyn_delta / coord_delta)
8. Per-dim std, collapse counts
9. Tracking quality (delta_corr, level_corr)
10. GDASR growth-point logs (log-only per M3)

CODE CHANGES:
1. src/run_phase0_sfa_sweep.py (NEW): Main experiment runner.
   - Based on run_phase0_sfa_archceiling.py structure.
   - 7 arms × 5 seeds × 5000 steps.
   - Ramp schedule for Arm A6: compute effective sfa_weight per step,
     pass via forward() sfa_weight parameter override.
   - Arm B determined after A-arms complete (two-phase execution),
     or pre-set to sfa_weight=10.0 if single-phase preferred.
   - Same evaluation suite: normalized temporal variance, semantic probes,
     collapse checks, centroid MSE, tracking quality.
   - Results saved to archive/iter_023/results/.

2. src/models_dual_stream.py: NO CHANGES needed.
   The sfa_weight parameter already supports per-forward-call override.
   The ramp schedule is implemented in the training loop, not the model.

3. src/pre_registration.md: Update with this plan.

TWO-PHASE vs SINGLE-PHASE:
Single-phase preferred (all 7 arms run simultaneously with Arm B pre-set
to sfa_weight=10.0). This avoids sequential dependency and produces all
results in one run. If sfa_weight=10.0 turns out suboptimal, the A-arms
provide the data to identify the correct weight for a follow-up.

---

## Iteration 023 -> Planner [Strategic Guidance]

# Manager's Note — Iteration 023

## 1. C5 Is Constructional, Not Empirical — Reframe or Deprioritize

The C5 criterion (normalized_dyn_var < normalized_coord_var) is the primary falsification criterion in your plan. **This is a constructional result, not an empirical one.** By design, SFA penalizes z_dyn's temporal variation and does not penalize z_coord's. Once sfa_weight is large enough, z_dyn *must* become slower than z_coord — this follows directly from the loss asymmetry, not from any emergent property. Confirming C5 at sfa_weight ≥ 5.0 tells us the SFA gradient is propagating; it does **not** tell us SFA is encoding identity.

The actual empirical test is whether slowness shapes z_dyn into an *identity-encoding* representation, not merely a *slow* one. A slow-but-uninformative z_dyn (constant noise that satisfies VICReg's std ≥ 1 but carries no object identity) would pass C5 while completely failing M2's intent. The **delta_R2_color criterion** is the real test.

**Directive:** Elevate delta_R2_color ≥ 0.10 improvement over the sfa_weight=0.1 baseline to **PRIMARY falsification**. Reframe C5 as a **gradient-propagation verification** (necessary but not sufficient). The 0.05 threshold in your secondary criterion is too low — it matches neither the Phase 0 gate nor the sml evidence suggesting SFA should be transformative. If SFA at gradient parity only produces +0.05 color improvement, that is evidence *against* M2's relevance to this architecture, not for it.

## 2. The SFA–VICReg Equilibrium Is the Central Scientific Question

The plan's tertiary criterion (collapse at sfa_weight ≥ 5.0) captures the failure mode, but the more important characterization is the **joint satisfaction region**: at what sfa_weight does z_dyn achieve *all three* of slowness (C5), identity encoding (delta_R2_color ≥ 0.10), and non-collapse (C1)? Your sweep will map this, but the pre-registration must name the composite criterion explicitly:

> **Composite M2 Viability Criterion:** There exists an sfa_weight ∈ [0.1, 25.0] such that ≥ 3/5 seeds simultaneously satisfy: (a) C5 (SFA gradient reaches z_dyn), (b) delta_R2_color improvement ≥ 0.10 over A1, and (c) per-dim std > 0.5 (non-collapse).

If no such sfa_weight exists, then SFA + batch VICReg cannot jointly shape z_dyn into a slow identity representation in this architecture — and the M2 mandate, while validated on sml's binary task, is falsified for the RGB CNN. This is an honest negative result and should be reported as such, not reframed as partial success.

## 3. Pre-Registration Precision and Language Hygiene

Before the Orchestrator commits `src/pre_registration.md`, ensure:

- **Hypothesis language:** Replace "the slow identity stream *will begin to separate*" with "we test whether SFA at parity produces measurable identity–position separation." The current phrasing assumes the outcome.
- **Quantitative thresholds:** State the composite criterion above explicitly, with the 0.10 color threshold (not 0.05). The secondary criterion at 0.05 can remain as a directional-check tier, but the primary claim tier must match the Phase 0 gate.
- **Arm B pre-commitment:** Since Arm B is pre-set to sfa_weight=10.0 in single-phase mode, explicitly note that if 5.0 < optimal_sfa < 10.0, Arm B may fail for the wrong reason and a follow-up is warranted. Do not interpret Arm B failure alone as falsifying the compound hypothesis.

The sub-agents executing this plan must read `src/pre_registration.md` and adhere strictly to its stated criteria and thresholds during evaluation. No post-hoc threshold relaxation.

---

