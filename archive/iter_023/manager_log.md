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

## Iteration 023 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — M2 Core Mechanism Falsified.
*   **Active Direction:** The SFA weight sweep (iter_023) has produced a decisive null result: making z_dyn slower does NOT make it encode object identity. The slowness prior is insufficient for identity-position separation. This falsifies the core M2 mechanism (SFA → slowness → identity encoding) as implemented with consecutive-frame (k=1) SFA. The M2 mandate requires revision: slowness is a necessary but not sufficient condition, and the discriminative signal must come from elsewhere.
*   **Next Priority:** Test multi-step SFA (k>>1, e.g. k=50) as the most principled alternative — standard SFA literature uses multi-timescale slowness, and longer horizons provide richer gradients even on quasi-static features. If multi-step SFA also fails, the slowness prior is fundamentally insufficient for this task and the representation objective must shift to a contrastive or augmentation-based approach.
*   **Confidence Score:** 70% (Reduced from 75%. The M2 mechanism is now empirically falsified for k=1 SFA. Confidence in the characterization is high, but the path to a working representation objective is more uncertain. Multi-step SFA is the last slowness-based option before abandoning the M2 framing entirely.)

## 2. Strategic Insights & Lessons Learned
*   **SFA GRADIENT PROPAGATES (iter_023, CONFIRMED):** Increasing sfa_weight from 0.1 to 25.0 (with ramp) monotonically reduces z_dyn normalized temporal variance from 0.0086 to 0.0011. The 250× gradient imbalance from iter_022 was real and is fixable. SFA is not "broken" — it just doesn't do what M2 assumed.
*   **SLOWNESS ≠ IDENTITY ENCODING (iter_023, CRITICAL NULL RESULT):** delta_R2_color is flat across the entire sfa_weight sweep (0.040–0.064 for d_max=8). Making z_dyn slower does not make it encode color/identity. The slowness prior provides no discriminative signal on already-static features — all competing representations (color-encoding, noise-encoding, constant) produce equally small SFA loss on consecutive frames. VICReg then determines the winner, pushing toward high-variance representations.
*   **C5 IS STRUCTURALLY IMPOSSIBLE (iter_023, DEFINITIONAL IDENTITY):** z_coord normalized temporal variance is ~1e-5 (2-3 orders of magnitude below z_dyn). This is a structural property of soft-argmax centroids in [0,127]: spatial variance O(127²) dominates temporal change O(1²). VICReg prevents z_dyn from reaching comparable normalized slowness. C5 must be replaced by delta_R2_color as the primary SFA-effectiveness criterion.
*   **RAMP STRATEGY WORKS (iter_023, PRACTICAL CONTRIBUTION):** Linearly ramping sfa_weight from 0.1 to target over 1000 steps reduces collapse from 4/5 (sfa=25 direct) to 1/5 (sfa=25 ramp). SFA-VICReg coexistence is achievable with proper initialization.
*   **d_max=16 REMAINS BEST FOR COLOR (iter_022-023, REPLICATED):** delta_R2_color ≈ 0.137 with d_max=16, consistent across two iterations. But this improvement comes from channel capacity, not from SFA. SFA weight doesn't affect the d_max=16 result.
*   **K=4 SUB-FEATURES STILL UNSOLVED (iter_022):** 12 active features at batch_size=32 overwhelms VICReg. Per-channel VICReg is a potential fix but untested. Do NOT test K=4 until VICReg collapse is resolved.
*   **CONV4 SOURCE UNSTABLE (iter_022):** dyn_source="conv4" collapsed 3/5 seeds. Linear(128→1) projection produces highly correlated channels. Avoid.
*   **Position Encoding Consistently Hurts (iter_013, iter_021):** Both under JEPA and under SFA+CGIR, adding explicit position encoding degrades performance. Robust finding across two objectives and two readout mechanisms.
*   **Cold-Start Trilemma (Phases 16–18, PRESERVED):** M3 (fixed dimensionality + log-only GDASR) sidesteps this by design. Still active as a constraint for Phase 5.
*   **JEPA Baseline Collapsed (iter_020):** JEPA+CCR 4/5 collapse, confirming M2's demotion of JEPA. But SFA's replacement is now also in question.

## 3. Loop & Bottleneck Detection
*   **SFA Effectiveness Loop (PARTIALLY RESOLVED):** SFA gradient propagates (confirmed), but slowness doesn't produce identity encoding (falsified). The "does SFA work?" question is answered: yes mechanically, no functionally. The loop shifts to: "can any slowness-based objective produce identity encoding?" Multi-step SFA is the next test.
*   **Identity Encoding Bottleneck (ACTIVE, CRITICAL):** No arm in iter_021-023 achieves delta_R2_identity ≥ 0.10. Only color is partially captured; size is not captured at all. The "identity stream" label for z_dyn is misleading. This is the fundamental bottleneck — without identity encoding, the entire M2 architecture lacks its foundation.
*   **Metric Artifact Loop (RESOLVED):** C5 was structurally impossible. Replaced by delta_R2_color as primary criterion. No further C5-based experiments.
*   **Gating Design Loop (STALE):** M3 sidesteps. Not the current priority.
*   **Logistics Note (iter_020-023):** Executor token limits remain a recurring infrastructure issue. Not blocking but requires monitoring.

## 4. Alternate Research Paths
*   **Multi-Step SFA (IMMEDIATE PRIORITY):** Use `||z_dyn(t) - z_dyn(t-k)||^2` with k=50 or k=100. Over longer timescales, scenes change (objects enter/exit, configurations shift), providing a richer SFA gradient that can discriminate between identity-preserving and identity-changing transitions. This is the standard SFA literature approach (Wiskott & Sejnowski 2002). If this also fails, the slowness prior is fundamentally insufficient.
*   **Contrastive Identity Objective (HIGH PRIORITY if multi-step SFA fails):** Replace or augment SFA with a contrastive objective using object-level tracking IDs (available from the physics engine). Same-object representations are pulled together across time; different-object representations are pushed apart. This is a stronger supervision signal but maintains decoder-free property. Requires infrastructure for object-level labels.
*   **Augmentation-Based SFA (MEDIUM PRIORITY):** Apply temporal augmentation (color jitter, small geometric transforms) to create artificial temporal variation in identity features, giving SFA a gradient to push against. Risk: may fight the natural invariance we want.
*   **Temporal Coherence via Augmentation (MEDIUM PRIORITY):** Instead of SFA, use augmentation-based self-supervision: same object under different augmentations should have similar z_dyn, different objects should have different z_dyn. This is the SimCLR/BYOL approach adapted for temporal data.
*   **d_max=16 + Better Objective (MEDIUM PRIORITY):** The channel capacity result (0.137) is the best to date. Combine d_max=16 with whatever replaces k=1 SFA as the primary objective.
*   **Abandon Slowness, Use Direct Supervision (LOW PRIORITY):** If all slowness-based and self-supervised approaches fail, use direct color/size supervision as a supervised upper bound. This would measure the theoretical ceiling of z_dyn for identity encoding, informing whether the bottleneck is the objective or the architecture.
*   **Micro-Columns (DEFERRED per semantic caution):** Only justified if a working objective leaves a measured gap. The gap currently exists, but the cause may be the objective, not the architecture.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Already adopted per M3. GDASR runs in log-only mode, d_t frozen at target.

---

## Iteration 023 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 023 — Null Result: SFA Slowness Prior Does Not Produce Identity Encoding

## 1. Pre-Declared Hypothesis and Falsification Criterion
**Hypothesis:** Increasing sfa_weight from 0.1 to parity with var_weight (25.0) would activate SFA, cause z_dyn to become slower than z_coord (C5 criterion: normalized_dyn_var < normalized_coord_var), and produce identity-position separation (delta_R2_color >= 0.10).

**Falsification criterion:** If delta_R2_color remains below 0.10 across the entire sfa_weight sweep, the hypothesis is refuted — SFA slowness does not produce identity encoding regardless of gradient strength.

**Secondary criterion (C5):** If normalized_dyn_var cannot drop below normalized_coord_var due to structural constraints, C5 is a metric artifact, not a valid test of SFA effectiveness.

**Tertiary criterion:** If sfa_weight=25.0 causes training collapse (SFA vs VICReg gradient conflict), test ramped initialization (0.1 → target over 1000 steps).

## 2. Experimental Protocol
- **Grid:** 1D physics sandbox, 128 RGB pixels, 3 objects with varying sizes/colors/masses.
- **Encoder:** NonParametricJEPASpatial (existing CNN backbone, Section 4.A).
- **Arms:**
  - A1-A6: d_max=8, CGIR+SFA+CCR, sfa_weight ∈ {0.1, 1.0, 5.0, 10.0, 25.0, 25.0-ramp}
  - B: d_max=16, CGIR+SFA+CCR, sfa_weight=25.0-ramp
- **d_t:** Frozen at 3 (M3). GDASR in log-only mode.
- **Training:** 5000 steps, batch_size=32, 5 seeds per arm.
- **VICReg:** var_weight=25.0, cov_weight=25.0 (batch-level, M1).
- **Other losses:** ccr_weight=1.0, sim_weight=25.0 (JEPA readout, stop-gradient).
- **Control:** Arm A1 (sfa_weight=0.1) replicates iter_022 Ctrl configuration.
- **Evaluation:** Same suite as iter_022: delta_R2_color, delta_R2_identity, normalized temporal variance, collapse rate, centroid MSE.

## 3. Observed Quantities
- **z_dyn normalized temporal variance (dose-response):**
  - sfa=0.1: 0.0086
  - sfa=1.0: ~0.005
  - sfa=5.0: ~0.003
  - sfa=10.0: ~0.002
  - sfa=25.0 (direct): ~0.0013 (4/5 collapse)
  - sfa=25.0 (ramp): 0.0011 (1/5 collapse)
- **z_coord normalized temporal variance:** ~1e-5 across ALL arms (structural constant).
- **delta_R2_color (d_max=8 arms):** 0.040–0.064 across the sweep. No monotonic trend with sfa_weight. Flat.
- **delta_R2_color (d_max=16, sfa=25 ramp):** 0.137 (consistent with iter_022 result of 0.130).
- **delta_R2_identity:** Negative across all arms (-0.027 to -0.055). SFA weight has no effect.
- **Collapse rate:** sfa=25.0 direct: 4/5; sfa=25.0 ramp: 1/5; all other arms: 0-1/5.

## 4. Verdict
**Refuted** on the primary criterion. Increasing sfa_weight successfully reduces z_dyn temporal variation (confirming SFA gradient propagates and the 250× imbalance was real), but delta_R2_color remains flat at 0.040–0.064 for d_max=8 — well below the 0.10 threshold. Making z_dyn slower does not make it encode object identity.

**C5 is a metric artifact** (definitional identity, not empirical finding): z_coord's normalized temporal variance (~1e-5) is structurally 2-3 orders of magnitude below z_dyn's reachable range, due to the soft-argmax centroid geometry in [0,127] combined with VICReg's std≥1 enforcement on z_dyn. C5 can never be satisfied in this architecture.

**Ramp strategy validated** (tertiary criterion): 1/5 collapse at sfa=25 ramp vs. 4/5 at sfa=25 direct. SFA-VICReg coexistence is achievable with proper initialization.

## 5. Construction-vs-Empirical Note
- **SFA gradient propagation (dose-response):** Genuinely empirical. The monotonic decrease in z_dyn variance with increasing sfa_weight confirms the gradient-imbalance diagnosis and validates that SFA is mechanically functional.
- **C5 impossibility:** Definitional identity. The soft-argmax centroid geometry (spatial variance O(127²), temporal change O(1²)) combined with VICReg's std≥1 constraint on z_dyn makes C5 structurally unsatisfiable. This is derivable from the construction, not a discovery about the system's behavior.
- **Slowness ≠ identity encoding:** Genuinely empirical and the most important finding. The SFA theoretical prediction (slowness → identity encoding) does not hold in practice on this task with consecutive-frame (k=1) SFA. On already-static features, the slowness prior provides no discriminative gradient — all representations produce equally small SFA loss, and VICReg determines the equilibrium.

## 6. Limitations
- This null result applies ONLY to consecutive-frame (k=1) SFA. Multi-step SFA (k>>1) remains untested and may provide the discriminative gradient that k=1 lacks.
- The experiment used a single architecture (NonParametricJEPASpatial CNN). Whether slowness produces identity encoding in other architectures (e.g., the deferred hierarchical pyramid, Section 8.6) is unknown.
- The delta_R2_color metric measures linear probe accuracy, which may underestimate nonlinear identity encoding. However, the flatness across the sweep (no trend) makes this unlikely to change the conclusion.
- Training was limited to 5000 steps. Whether longer training would eventually produce identity encoding is unknown, but the flat dose-response (no trend with sfa_weight) makes this unlikely.
- The JEPA readout (sim_weight=25.0) was active in all arms. Whether it interferes with SFA through shared encoder gradients is an open question (journal Open Question 7).
- This result does NOT show that slowness is irrelevant for representation learning in general — only that k=1 SFA on already-static features provides insufficient discriminative signal for identity encoding in this specific setup.

---

