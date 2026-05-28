# Research Manager Log - Iteration 020

## Iteration 020 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
On the existing NonParametricJEPASpatial CNN encoder with frozen d_t=3 and no dynamic
recruitment, replacing the JEPA prediction objective with SFA (slowness) on z_dyn as the
primary representation objective, combined with batch VICReg on z_dyn, will:
(1) Train without representation collapse (all per-dimension std >= 0.5, no has_collapsed trigger),
(2) Achieve centroid-decoding MSE via Arm F (soft-argmax on z_coord) at most 10% higher
    than the JEPA+VICReg baseline (i.e., MSE_SFA <= 1.10 * MSE_JEPA, where MSE_JEPA ~ 55.6),
(3) Produce z_dyn representations that are significantly slower (mean ||z_dyn(t) - z_dyn(t-1)||^2
    at least 40% lower) than z_coord temporal variation, indicating that SFA separates slow
    identity from fast position by construction.

**Proposed Falsification Criterion:**
The hypothesis is falsified if ANY of the following hold across the 5-seed sweep:
C1 (Collapse): SFA+VICReg arm collapses (has_collapsed=True OR any active dimension std < 0.5)
    in >= 2 out of 5 seeds.
C2 (Centroid MSE): Mean centroid-decoding MSE of the SFA arm exceeds 1.10 * mean MSE of the
    JEPA baseline arm (i.e., SFA degrades spatial readout by more than 10%).
C3 (Slowness separation): The ratio mean(||z_dyn(t)-z_dyn(t-1)||^2) / mean(||z_coord(t)-z_coord(t-1)||^2)
    is >= 0.6 for the SFA arm (i.e., z_dyn is not substantially slower than z_coord,
    failing the identity-position separation hypothesis).

**Proposed Method:**
Step 1: Create SFA training infrastructure.
- MODIFY src/models_dual_stream.py: Add an SFA loss computation to NonParametricJEPASpatial.forward()
  that computes L_sfa = ||z_dyn_target - z_dyn_prev||^2 over the batch, where z_dyn_prev is
  z_dyn from the previous frame in the history. Also add batch VICReg on z_dyn_target (the 
  target frame's dynamics representation). The total representation loss becomes:
  L_repr = sfa_weight * L_sfa + var_weight * VICReg_var(z_dyn) + cov_weight * VICReg_cov(z_dyn)
  The JEPA sim_loss is retained ONLY as a readout/surprise signal (stop-gradient detached from
  the representation), not as a training objective for the encoder.
- Keep the predictor's forward pass for surprise computation, but detach its output from the
  encoder's gradient path when SFA is the primary objective.

Step 2: Create the Phase 0 experiment runner.
- CREATE src/run_phase0_sfa.py: A 5-seed sweep comparing three arms:
  Arm A (SFA+VICReg): Primary objective = SFA on z_dyn + batch VICReg on z_dyn.
    - sfa_weight=1.0, var_weight=25.0, cov_weight=25.0
    - Predictor is retained for surprise readout but gradients do NOT flow to encoder from sim_loss.
    - z_coord is NOT slowed (M2 mandate).
    - d_t frozen at 3, no recruitment.
  Arm B (JEPA+VICReg Baseline, B1): Same as current NonParametricJEPASpatial with
    sim_weight=25.0, var_weight=25.0, cov_weight=25.0. d_t frozen at 3, no recruitment.
    This is the direct comparison baseline.
  Arm C (SFA+VICReg with pos_encoding="sinusoidal"): Same as Arm A but with sinusoidal
    position encoding added to the input. Tests whether explicit position channels
    narrow the Spatial Specialization Gap under SFA.

Step 3: Training protocol (shared across arms).
- Environment: PhysicsSandbox(N=3, seed=seed) for 3000 training steps, batch_size=32.
- History buffer: deque(maxlen=4) for H=3 history + 1 target.
- Replay buffer: capacity 2000, prefill 100 transitions.
- Optimizer: Adam, lr=1e-3.
- d_t = 3 frozen from start (no recruitment logic).
- GDASR runs in log-only mode (computes and logs growth-point signals without modifying d_t).
- Seeds: [42, 123, 456, 789, 999].

Step 4: Evaluation protocol (at step 3000 and checkpoint at 1500).
- Non-collapse check: has_collapsed criterion + per-dimension std >= 0.5.
- Centroid decoding MSE: linear probe on soft-argmax centroids vs. true object positions
  (all 3 objects, report mean and per-object).
- Slowness metrics: mean ||z_dyn(t) - z_dyn(t-1)||^2 and mean ||z_coord(t) - z_coord(t-1)||^2
  over 200 test frames, and their ratio.
- VICReg health: per-dimension std, mean absolute correlation between active dimensions.
- GDASR growth-point log: count and timing of would-have-recruited events.

Step 5: Statistical analysis.
- Welch's t-test comparing SFA arm vs JEPA arm on centroid MSE.
- Welch's t-test comparing SFA arm z_dyn slowness ratio vs 0.6 threshold.
- Cohen's d for effect sizes.
- Report with confidence intervals per reporting standards (Section 9).

Step 6: Falsification audit against C1, C2, C3.
- C1: Count collapsed seeds per arm.
- C2: Compare mean MSE_SFA vs 1.10 * mean MSE_JEPA.
- C3: Compute slowness ratio for SFA arm; check >= 0.6.

Files to create/modify:
- MODIFY: src/models_dual_stream.py (add SFA objective option to NonParametricJEPASpatial.forward)
- CREATE: src/run_phase0_sfa.py (experiment runner with 3 arms × 5 seeds)
- UPDATE: src/pre_registration.md (with Phase 0 pre-registration)

---

## Iteration 020 -> Planner [Strategic Guidance]

# Manager's Note – Phase 0 Objective Migration

## 1. Construction-vs-Empirical Red Flag on Claim (3) [CRITICAL]

The hypothesis claim (3) and falsification criterion C3 suffer a **constructional confound**. You imposed an explicit slowness penalty on `z_dyn` and none on `z_coord`. The optimizer minimizing what you asked it to minimize — yielding `||Δz_dyn||² / ||Δz_coord||² < 0.6` — is a **verification of the gradient path**, not empirical evidence that identity and position are semantically disentangled. The ratio tells us "the loss term worked," not "the representation factorized meaningfully."

**Required action:**
- **Reframe claim (3) as a sanity check** (verify the SFA objective is active and VICReg prevents collapse to a constant). State this explicitly in the pre-registration.
- **Add a genuinely empirical disentanglement metric:** linear probes predicting object identity (color/size class) from `z_dyn` vs. `z_coord`, and object centroid position from `z_coord` vs. `z_dyn`. If SFA creates semantic separation, `z_dyn` should predict identity well and position poorly, and vice versa for `z_coord`. *This* is the test of the Spatial-Gap hypothesis — not the slowness ratio.
- **Replace or augment C3** with a criterion on the semantic probes, e.g.: "C3 is falsified if the linear-probe accuracy for object identity from `z_dyn` does not exceed that from `z_coord` by at least 10 percentage points."

## 2. Pre-Registration and Parameter Hygiene

The Orchestrator will automatically write `src/pre_registration.md` from this plan. Sub-agents **must read and strictly adhere** to it during execution.

The initial weight `sfa_weight=1.0` is a declared starting point. If the first run shows SFA is too weak relative to VICReg (slowness ratio near 1.0) or too strong (collapse toward constant despite VICReg), adjusting it is legitimate — but **document the adjustment and its rationale** in the run log. Do not silently tune after viewing results and report as if the final value was the plan all along.

## 3. The Centroid MSE Bar Is Asymmetric

C2 tests that SFA doesn't *degrade* by more than 10% vs. JEPA. This is the right gate for a bridge phase. But per the sml findings (SFA+VICReg 82% vs. JEPA+VICReg 61%), SFA may *improve* the representation. Report both directions honestly: if MSE_SFA < MSE_JEPA, that is compatible with the sml transfer and should be noted; if MSE_SFA > MSE_JEPA but within 10%, that is a successful bridge but a signal that RGB physics may differ from binary sml in ways worth characterizing. Do not frame a non-improvement as a failure if it passes C2.

---

## Iteration 020 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
The spatial-mean z_dyn computation (z_dyn = a_spatial.mean(dim=-1)) in
NonParametricEncoder is the primary structural cause of the semantic
disentanglement failure (delta_R2_color = -0.074, target ≥ 0.10) observed
across all arms in iter_020. Replacing it with a Centroid-Gated Identity
Readout (CGIR) — a separate 1x1 convolution (conv_identity) producing an
identity feature map, pooled at the soft-argmax-attended spatial positions
with stop-gradient on the spatial attention — will enable SFA+VICReg to
produce z_dyn representations that encode object identity (color) while
z_coord encodes object position. Specifically:

(1) CGIR+SFA will train without collapse (per-dim std ≥ 0.5 in ≥ 4/5 seeds).
(2) CGIR+SFA centroid MSE will be within 10% of the mean-pooling SFA baseline
    (MSE_CGIR ≤ 1.10 × MSE_mean, where MSE_mean ≈ 121.9 from iter_020 Arm A1).
(3) CGIR+SFA will achieve semantic disentanglement: delta_R2_color ≥ 0.10,
    where delta_R2_color = R²_dyn_color - R²_coord_color. This is the metric
    that persistently failed in iter_020 (all arms had negative delta_R2_color).

The CGIR mechanism works because: (a) the soft-argmax attention naturally
localizes each channel d to a specific object, (b) pooling identity features
at the attended position reads out the object's local appearance (color),
(c) SFA on z_dyn then correctly encourages this readout to be slow (color
IS slow across frames), and (d) VICReg prevents the trivial constant solution.
The stop-gradient on the spatial attention prevents SFA from distorting
position tracking. This is structurally different from the mean-pooling
z_dyn, which averages over ALL spatial positions and cannot isolate per-object
identity information.

**Proposed Falsification Criterion:**
The hypothesis is falsified if ANY of the following hold across the 5-seed
sweep for the CGIR+SFA arm:

C1 (Collapse): Per-dimension std < 0.5 in ≥ 2 out of 5 seeds (i.e., the
    CGIR architecture causes representation collapse no better than the
    mean-pooling baseline).

C2 (Centroid MSE degradation): Mean centroid-decoding MSE of the CGIR+SFA
    arm exceeds 1.10 × mean MSE of the mean-pooling SFA baseline (i.e.,
    CGIR degrades spatial readout by more than 10% relative to iter_020
    Arm A1 result of MSE ≈ 121.9).

C3 (Semantic disentanglement failure): delta_R2_color < 0.10 for the
    CGIR+SFA arm, where delta_R2_color = R²_dyn_color - R²_coord_color.
    This is the same criterion that failed in iter_020. If CGIR does not
    fix it, the root cause is NOT the spatial-mean computation and the
    hypothesis is falsified. A delta_R2_color ≥ 0.10 would mean z_dyn
    predicts object color meaningfully better than z_coord, confirming
    identity-position separation.

**Proposed Method:**
Step 1: Modify NonParametricEncoder (src/models_dual_stream.py).
- Add conv_identity = nn.Conv1d(128, d_max, kernel_size=1) as a separate
  identity head alongside the existing conv_spatial.
- Add a dyn_readout parameter ("mean" or "centroid_gated") to control
  z_dyn computation (default "mean" for backward compatibility).
- In "centroid_gated" mode:
  a) Compute backbone features via existing conv1-4 → (B, 128, 8).
  b) Compute position spatial map: conv_spatial(features) → (B, d_max, 8)
     → interpolate → (B, d_max, 128) → soft-argmax → z_coord.
  c) Compute identity spatial map: conv_identity(features) → (B, d_max, 8)
     → interpolate → (B, d_max, 128) → a_identity.
  d) Compute z_dyn via centroid-gated readout:
     p_c = softmax(a_spatial, dim=-1)  # spatial attention, shape (B, d_max, 128)
     z_dyn = sum(a_identity * p_c.detach(), dim=-1)  # (B, d_max)
     The p_c.detach() prevents SFA gradient from distorting position tracking.
- Keep forward_spatial() unchanged for backward compatibility.
- Parameter increase: conv_identity adds 128×8 + 8 = 1,032 params (< 0.5% of total).

Step 2: Update NonParametricJEPASpatial.forward() SFA mode.
- The existing SFA mode computes sfa_loss on z_dyn_target vs z_dyn_prev.
  With CGIR, z_dyn is now computed differently but has the same shape (B, d_max).
  No changes to the loss computation are needed — only the z_dyn values change.
- The VICReg var/cov losses on z_dyn remain unchanged.
- The predictor (surprise readout) remains unchanged (takes detached z_dyn and z_coord).
- The CCR losses on z_coord remain unchanged.

Step 3: Create src/run_phase0_sfa_cgir.py.
Three arms × 5 seeds:

Arm A (CGIR+SFA w=0.1): Centroid-gated identity readout, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="none", CCR=none,
  d_t=3, gdasr_log_only=True. This is the primary test of the hypothesis.

Arm B (Mean+SFA w=0.1+CCR): Original mean-pooling z_dyn, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="none", CCR=covariance
  (ccr_smooth=10, ccr_spatial=10), d_t=3, gdasr_log_only=True.
  Direct replication of iter_020 Arm A1 for comparison — confirms that
  the only difference is the CGIR architecture change.

Arm C (CGIR+SFA w=0.1+pos): Centroid-gated identity readout, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="sinusoidal", CCR=none,
  d_t=3, gdasr_log_only=True. Tests whether positional encoding further
  improves CGIR (it helped collapse in iter_020 Arm C).

Seeds: [42, 123, 456, 789, 999].
Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000.
Evaluation: same protocol as iter_020 (collapse check, centroid MSE via
  linear probe, slowness metrics, VICReg health, semantic disentanglement
  probes with delta_R2_color, GDASR growth-point logging).
Checkpoint evaluation at step 2500 and final evaluation at step 5000.

Step 4: Falsification audit.
Compare CGIR+SFA arm vs Mean+SFA baseline on all three criteria.
The primary claim is that CGIR enables delta_R2_color ≥ 0.10 (C3).
If CGIR achieves C3 but degrades C1 or C2, report the trade-off honestly.
If CGIR fails C3, the root cause is NOT the spatial-mean computation and
the hypothesis is falsified — future work would need to explore other
architectural changes (slot attention, explicit color bottlenecks, etc.).

Files to create/modify:
- MODIFY: src/models_dual_stream.py (add CGIR to NonParametricEncoder,
  add dyn_readout parameter, implement centroid-gated readout)
- CREATE: src/run_phase0_sfa_cgir.py (experiment runner, 3 arms × 5 seeds)
- UPDATE: src/pre_registration.md (with Phase 0 CGIR pre-registration)

---

## Iteration 020 -> Planner [Strategic Guidance]

# Manager's Note — Strategic Guidance for Iteration 021

## 1. Construction-vs-Empirical Audit on C3: The Positive Result May Be Near-Tautological

The CGIR mechanism reads `a_identity` at spatial positions already localized by the soft-argmax attention `p_c`. Since the backbone processes RGB input and `conv_identity` is a 1×1 conv on backbone features, a positive `delta_R2_color ≥ 0.10` largely follows from: (a) attention localizes to objects, (b) backbone already encodes RGB, (c) the 1×1 conv preserves relevant features at the attended location. This is **not** SFA "discovering" identity-position separation — it is the architecture *structurally enforcing* that `z_dyn` reads from object positions while `z_coord` tracks those positions.

**Required action:** In the pre-registration and any subsequent reporting, a positive C3 result must be framed as: *"CGIR structurally routes per-object appearance information into `z_dyn`, consistent with the hypothesis that the mean-pooling bottleneck prevented this routing."* Do **not** frame it as "SFA enables emergent identity-position disentanglement" — the disentanglement is in the wiring, not the learning dynamics.

**Stronger test (optional but recommended):** Add a fourth criterion — does `z_dyn` from CGIR predict object **identity** (e.g., a compound label encoding both color and size) better than `z_coord`? If CGIR only captures color (the most salient RGB feature), but not size or mass, then the "identity" label is misleading — it should be called "color readout." This would clarify whether the separation is identity-vs-position or merely color-vs-position.

## 2. Arm A Has No Objective Shaping z_coord — Potential Tracking Degradation

Arm A (the primary test arm) specifies `CCR=none`. Under M2, the JEPA sim_loss is demoted to a readout with stop-gradient. The predictor's surprise signal also has stop-gradient. So in Arm A, **nothing directly shapes z_coord** beyond indirect backbone gradients flowing through `z_dyn`'s SFA+VICReg losses and the CGIR readout. The centroid-MSE evaluation depends on `z_coord` tracking ability. If tracking degrades in Arm A, it will be unclear whether this is a CGIR side-effect or simply a missing CCR objective.

**Required action:** Either (a) add at least `ccr_smooth` to Arm A so that `z_coord` has a direct temporal-smoothness objective (which is legitimate — position should change smoothly, just not be forced to be *constant*), or (b) add a fourth arm: CGIR+SFA+CCR, which is the fully-armed variant. The current Arm B (Mean+SFA+CCR) cannot distinguish whether CGIR or CCR is responsible for any observed difference.

## 3. Honest Accounting of iter_020's Null Result

The iter_020 finding that `delta_R2_color = -0.074` (i.e., `z_coord` predicts color *better* than `z_dyn`) across all arms is an honest, clearly-documented null result under the original architecture. It is a **success of the method** that this was measured. Do not retroactively reframe it as a "failure of mean-pooling" without the evidence this iteration is designed to produce. The pre-registration must record the prior null result and state that this iteration tests whether a specific architectural change (CGIR) changes the outcome — not that the prior outcome was "wrong."

**Pre-registration reminder:** The Orchestrator will write `src/pre_registration.md` from this plan. Ensure the sub-agents read it and strictly adhere to the declared hypothesis and falsification criteria during execution. No post-hoc criterion relaxation is permitted.

---

## Iteration 020 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — SFA Effectiveness Crisis Identified.
*   **Active Direction:** The CGIR falsification (iter_021) has uncovered a deeper problem than the spatial-mean bottleneck: **SFA on consecutive frames is not effectively shaping `z_dyn` into a slow representation**. The slowness ratio remains >> 1 across all arms, meaning the SFA loss is not driving `z_dyn` toward temporal stability. The root cause appears to be that object identity features (color, size) are *already constant across consecutive frames*, so the SFA gradient provides minimal directional learning signal — it merely says "keep z_dyn constant," which VICReg opposes, producing a tug-of-war that doesn't converge to identity encoding. This strikes at the heart of the M2 mandate and must be resolved before Phase 0 can be considered complete.
*   **Next Priority:** Diagnose why SFA fails to shape `z_dyn` and determine whether this is (a) an implementation bug, (b) a loss-weighting issue (SFA loss scale vs. VICReg loss scale), (c) a fundamental limitation of frame-pair SFA on already-static signals, or (d) an architectural issue requiring contrastive or augmentation-based objectives. This must be addressed before any further architectural interventions (CGIR variants, micro-columns, etc.).
*   **Confidence Score:** 75% (Reduced from 88%. The core M2 mechanism—SFA as the primary representation shaper—is now empirically questionable on this task. Confidence in the *characterization* of the problem remains high, but confidence in the path to solution has dropped significantly.)

## 2. Strategic Insights & Lessons Learned
*   **CGIR is a contributing factor, not the primary cause (iter_021, FALSIFIED):** Centroid-Gated Identity Readout provides a directional +0.124 shift in delta_R2_color but fails the 0.10 threshold. The spatial-mean bottleneck accounts for ~60% of a gap that is itself insufficient. CCR on z_coord is essential for CGIR to function. Position encoding hurts CGIR (0.011 vs 0.050), consistent with iter_013.
*   **SFA Slowness Ratio Crisis (iter_021, CRITICAL FINDING):** The slowness ratio (z_coord_temporal_var / z_dyn_temporal_var) remains >> 1 across all arms, indicating that SFA is NOT making z_dyn slower than z_coord. This contradicts the M2 design assumption. Root cause hypothesis: on consecutive frames, object identity features are already constant (a red blob stays red), so the SFA loss gradient `||z_dyn(t) - z_dyn(t-1)||^2` provides zero discriminative learning signal — all competing representations (color-encoding, noise-encoding, constant) produce equally small SFA loss. The VICReg variance term then determines the winner, pushing toward high-variance representations rather than identity-encoding ones.
*   **The SFA-VICReg Tug-of-War (iter_021, NEW CONSTRAINT):** SFA says "be constant," VICReg says "be variable." On already-static signals, neither objective provides directional pressure toward encoding color. The equilibrium is a representation that fluctuates at the VICReg margin (std ≈ 1) with minimal temporal correlation, not one that encodes stable object identity. This is a structural problem, not a tuning problem.
*   **"Identity" is misleading — only color is partially captured (iter_021, C4 probe):** The identity probe reveals that size is not captured by z_dyn at all. Only color shows partial encoding. The "identity stream" framing overstates what z_dyn actually represents.
*   **Cold-Start Trilemma (Phases 16–18, PRESERVED):** Three distinct cold-start pathologies constrain gating designs: (1) encoder cold-start → smoothness gates reject genuine objects, (2) predictor cold-start → MDL gates accept noise, (3) optimization transient → prediction-trend gates accept everything. M3 (fixed dimensionality + log-only GDASR) sidesteps this trilemma by design.
*   **Position Encoding Consistently Hurts (iter_013, iter_021):** Both under JEPA and under SFA+CGIR, adding explicit position encoding degrades performance. This is now a robust finding across two objectives and two readout mechanisms.

## 3. Loop & Bottleneck Detection
*   **SFA Effectiveness Loop (ACTIVE, CRITICAL):** The M2 mandate assumes SFA shapes z_dyn into a slow identity representation. Empirically, it doesn't. This is the current bottleneck — all downstream interventions (CGIR, micro-columns, spatial specialization gap) depend on z_dyn actually encoding identity. Must resolve before proceeding.
*   **Gating Design Loop (Phases 15–18, STALE):** Four consecutive gating failures mapped the cold-start trilemma. M3 (fixed dimensionality) sidesteps this loop. De-activated in favor of M3; not the current priority.
*   **Logistics Note (iter_020/021):** Executor 20.2 hit token limit while writing COMPARISON_REPORT.md. This is a recurring infrastructure issue that may require chunking output or reducing report verbosity in future iterations.

## 4. Alternate Research Paths
*   **SFA Loss-Weighting Diagnosis (IMMEDIATE PRIORITY):** Measure the actual magnitude of the SFA loss vs. VICReg loss during training. If VICReg dominates by orders of magnitude, the SFA gradient is swamped and z_dyn is shaped by VICReg alone. This is the cheapest diagnostic and must be run first.
*   **Multi-Step SFA (Medium priority):** Instead of `||z_dyn(t) - z_dyn(t-1)||^2`, use `||z_dyn(t) - z_dyn(t-k)||^2` with k >> 1 (e.g., k=50). Over longer timescales, even slow-varying features exhibit some drift (objects swap positions, scenes change), providing a richer SFA gradient. This is the standard SFA literature approach (Wiskott & Sejnowski 2002 use slow features at multiple timescales).
*   **Contrastive Identity Objective (Medium priority):** Replace or augment SFA with a contrastive objective that explicitly pulls same-object representations together and pushes different-object representations apart. This requires object-level labels or tracking IDs (available from the physics engine). This is a stronger supervision signal than SFA but maintains decoder-free property.
*   **Augmentation-Based SFA (Low priority):** Apply temporal augmentation (color jitter, small geometric transforms) to create artificial temporal variation in identity features, giving SFA a gradient to push against. However, this may fight the natural invariance we want.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Already adopted per M3. GDASR runs in log-only mode, d_t frozen at target. This unblocks non-gating work.
*   **Micro-Columns (DEFERRED per semantic caution):** Only justified if SFA-on-z_dyn leaves a measured gap after the SFA effectiveness crisis is resolved. The gap currently exists, but the cause may be SFA failure, not architectural inadequacy.

---

## Iteration 020 -> Project Archive [Milestone Report]

# RDF Milestone Review — Iteration 021 — Null Result: CGIR Insufficient for Semantic Disentanglement

## 1. Pre-Declared Hypothesis and Falsification Criterion
**Hypothesis (phase-021):** Replacing spatial-mean readout with centroid-gated identity readout (CGIR) on z_dyn will increase delta_R2_color by ≥ 0.10, resolving the semantic disentanglement failure (delta_R2_color < 0) observed since iter_020.
**Falsification criterion (C3):** delta_R2_color improvement < 0.10 between Arm A (CGIR) and Arm B (mean baseline).

## 2. Experimental Protocol
- **Arm A (CGIR):** z_dyn readout via centroid_gated (uses z_coord soft-argmax weights to extract channel-specific identity features). SFA+VICReg on z_dyn, CCR on z_coord, d_t=3 frozen, RGB-only input, lambda=25.
- **Arm B (Baseline):** z_dyn readout via spatial mean (standard). All other parameters identical to Arm A.
- **Arm C:** CGIR + positional encoding (sinusoidal position channel appended to RGB input).
- **Arm D:** CGIR without CCR (no spatial decorrelation on z_coord).
- **Identity Probe (C4):** Regresses z_dyn against color and size independently to decompose "identity."
- **Environment:** 1D physics sandbox, N=3 objects, varying colors/sizes/masses, elastic collisions. 5 seeds per arm.
- **Metrics:** delta_R2_color (R²_color_z_dyn - R²_color_z_coord), slowness ratio (z_coord_temporal_var / z_dyn_temporal_var), centroid decoding MSE.

## 3. Observed Quantities
- **Arm A (CGIR) delta_R2_color:** +0.050 (shifted from negative, but below 0.10 threshold)
- **Arm B (Mean baseline) delta_R2_color:** -0.074 (negative, confirming baseline failure)
- **CGIR directional effect:** +0.124 (positive shift, statistically consistent but below criterion)
- **Arm C (CGIR + pos encoding) delta_R2_color:** +0.011 (pos encoding hurts CGIR)
- **Arm D (CGIR, no CCR) delta_R2_color:** Negative (CCR essential for CGIR)
- **C4 Identity Probe:** Color partially captured by z_dyn; size NOT captured. "Identity" framing overstated.
- **Slowness ratio:** >> 1 across ALL arms (z_dyn is NOT slower than z_coord despite SFA objective).

## 4. Verdict
**Refuted.** CGIR produces a directional +0.124 shift in delta_R2_color but fails the pre-declared 0.10 threshold. The spatial-mean bottleneck is a contributing factor (~60% of the directional shift) but not the primary cause of semantic disentanglement failure.

## 5. Construction-vs-Empirical Note
The CGIR effect (+0.124) is genuinely empirical — it was not derivable from the architecture alone and required the controlled A/B comparison. The slowness ratio >> 1 finding is also genuinely empirical and not derivable from construction; one would expect SFA to make z_dyn slow, but it empirically doesn't. The position-encoding-hurts finding reproduces the iter_013 result under a different objective, strengthening it as a cross-objective empirical regularity.

## 6. Limitations
This null result does NOT show that spatial-mean readout is irrelevant — it contributes +0.124. It shows that resolving the semantic disentanglement gap requires addressing factors beyond the readout mechanism. The critical unresolved question is why SFA fails to make z_dyn slow (slowness ratio >> 1). This may indicate: (a) loss-weighting imbalance where VICReg overwhelms SFA gradients, (b) a fundamental limitation of frame-pair SFA on already-static signals, or (c) an implementation issue. Until this is resolved, no architectural intervention targeting z_dyn semantics can be properly evaluated, because the SFA foundation itself may be non-functional.

---

