# Research Manager Log - Iteration 032

## Iteration 032 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Representation Foundation) is **closed by
    pre-committed decision rule**. iter_032 tested the rich attention-pooled
    centroid-gated readout — the architectural fix predicted by the
    iter_031 localization and explicitly required by the user hint to be
    *rich* (K=4 multi-dimensional feature vector per object), not scalar.
    Result: catastrophic collapse (100% for K=4, 10% for K=1 scalar), with
    a clean mechanistic story (cross-backbone attention coupling: peaked
    softmax from coord backbone gates dyn backbone's readout → only
    attended-position features receive strong VICReg gradient → degenerate
    solution, per_dim_std ~0.3–0.6 below the 0.5 viability threshold).
    The pre-committed binding rule — committed in this iteration's
    pre-registration, before the run — triggers **branch (b): hard-pivot
    to behavioral evaluation**.
*   **Active Direction (pivoted, per pre-committed rule):** Behavioral
    evaluation against the *best available* representation, not against a
    representation that has yet to clear ΔR²_color ≥ 0.30. Three convergent
    signals now establish that ΔR²_color ≥ 0.30 is the wrong target on the
    current architecture:
    - iter_021 CGIR partial gain (+0.124, missed 0.30)
    - iter_023–031 5-objective convergent null (SFA, JEPA, temporal-
      contrastive, variance-ramped SFA, reconstruction)
    - iter_032 readout-fix architectural null (K=1 worse than mean-pool;
      K=4 100% collapse)
    Three independent attempts to clear 0.30 — via objective choice, via
    architectural readout fix, via supervised reconstruction ceiling —
    have all failed. The convergence is strong enough that *continuing to
    target ΔR²_color ≥ 0.30 representation-side is no longer the
    project's bottleneck-reducing move*.
*   **What is now solid:**
    - **Cross-backbone attention coupling is unstable under VICReg
      gradient flow** (iter_032 mechanistic finding). The peaked softmax
      gating concentrates the variance constraint on a single spatial
      position, producing the degeneracy. This generalizes: any future
      readout that hard-gates one backbone by another's argmax must
      budget for this failure mode.
    - **Scalar centroid-sample alone is NOT a partial improvement over
      mean-pool** (iter_032 K=1 arm). This retires the iter_021 CGIR
      framing — CGIR's +0.124 came from a different mechanism, not from
      spatial sampling per se. (Note for iter_033: revisit what CGIR
      actually did, since the simple "sample at centroid" reading is now
      falsified.)
    - **VICReg-only on separate backbone (the iter_028 config)** remains
      the best non-collapsing representation: ΔR²_color = 0.045, 0%
      collapse, best tracking, centroid_MSE ~160. This is the working
      substrate for the behavioral pivot.
    - **The iter_031 CLTS Part B directional signal stands** (collision
      selectivity 0.59 probe vs 0.44 random, ratio 1.34×, below the
      pre-registered 1.5× gate). Per the user hint, this is a real
      directional signal that should not be dismissed as
      "representation-confounded" — that dismissal pre-supposes the
      conclusion. The gate threshold (1.5×) itself is part of what
      iter_033 must calibrate.
*   **What is now retired or contested:**
    - **The "fix the readout, then re-run the objectives" path (iter_032
      plan) is closed.** The readout fix made things strictly worse.
    - **M2 mandate revision is DEFERRED, not executed** (per user hint).
      The 5-objective convergent null arose under the broken mean-pool
      readout, and the iter_032 readout fix itself failed catastrophically
      — so we do not yet have a working readout under which to fairly
      compare objectives. SFA-vs-reconstruction-vs-contrastive becomes
      meaningful again only after a non-collapsing rich readout exists.
      Until then, M2's empirical status is "untestable on this
      architecture," not "falsified."
    - **The behavioral-evaluation-without-representation-foundation loop
      (added in iter_031 journal) is dissolved by the pivot decision.**
      Behavioral evaluation against the best-available representation
      was the planned escape under branch (b), and the pre-commit rule
      was the discipline preventing this from being an ad-hoc retreat.
*   **Next Priority (iter_033, pre-register tightly):**
    Behavioral evaluation on the iter_028 separate-backbone +
    mask_dyn_sim + coord_vicreg + VICReg-only configuration.
    - Arm 1: Re-run CLTS Part B in the N=2 collision-sparse environment
      from iter_031 (protocol calibration), with the VICReg-only
      representation. Same protocol, but the gate *thresholds* are now
      being calibrated against measured baselines, not assumed.
    - Arm 2: Add the causal-sensitivity probe (alter object mass; measure
      whether tracking / prediction adapts) explicitly — this is the
      Section 6 metric that has never been run, and it is the metric
      most directly diagnostic of "representation quality matters for
      behavior."
    - Arm 3: Centroid-MSE tracking on the same representation against
      Phase-12 references (CLTS 85.85, WUP-MDL 57.34), to anchor the
      absolute scale of the representation's spatial utility.
    - Pre-register, as the binding gate: *the project commits to
      Phase 2/3 integration if and only if at least one behavioral
      metric clears its pre-registered threshold over ≥5 seeds with
      lower CI clear of the random-control upper CI*.
    - Pre-register the thresholds themselves with explicit construction:
      e.g., collision selectivity gate = mean(random) + 2σ(random) over
      the calibration run, not a fixed 1.5×. This makes the gate
      responsive to the actual control distribution, eliminating the
      "1.5× was arbitrary" objection.
*   **Confidence Score:** 35% (down from 38%). The iter_032 result is
    genuine progress in the falsification sense — we learned that the
    apparently obvious fix (rich centroid readout) creates a new failure
    mode rather than solving the old one. But the cumulative picture is
    that the representation foundation is *narrower than three iterations
    ago thought*. Recovering confidence depends on whether the behavioral
    pivot yields a metric the current representation clears; if it does
    not, the project will face the harder question of whether a
    decoder-free RGB-motion agent is achievable at all on this
    architecture without relaxing one of the three frozen constraints
    (decoder-free, mean-pool family of readouts, fixed dimensionality).

## 2. Strategic Insights & Lessons Learned
*   **CROSS-BACKBONE ATTENTION COUPLING IS UNSTABLE UNDER VICReg
    (iter_032, ARCHITECTURAL FINDING):** A peaked softmax from one
    backbone gating the readout of another concentrates the variance
    hinge on the attended spatial position. The local gradient signal
    is then strong enough to drive a degenerate solution: features at
    the attended position become similar across the batch. Mechanistic
    story is clean: VICReg's variance term is sample-level; if the
    readout collapses information to a small set of "attended" features,
    the variance constraint can only push those features apart, not the
    whole feature map. K=4 (rich vector) makes this worse than K=1
    because the dimensionality multiplies the constraint pressure on
    the same attended location. Generalizes to: any future architecture
    that hard-gates one backbone's readout by another's argmax must
    apply VICReg upstream of the gate, not at the gated readout.
*   **ITER_021 CGIR'S +0.124 IS NOT EXPLAINED BY "SAMPLE AT CENTROID"
    (iter_032, RETIRED HYPOTHESIS):** The simple reading of CGIR — that
    centroid-sampled scalars beat mean-pooled scalars — is falsified
    by iter_032's K=1 arm (worse than mean-pool). Whatever made CGIR
    produce +0.124 was not the spatial-sample mechanism. Either (a) CGIR
    had an unobserved confound, or (b) the gain came from a different
    mechanism (e.g., training stabilization, optimization side-effect).
    Open lesson: future iterations must inspect what CGIR actually
    changed, and not cite +0.124 as evidence for centroid-sampling.
*   **THREE-SIGNAL CONVERGENT EVIDENCE THAT THE PROJECT'S
    REPRESENTATION-SIDE GOALPOST IS WRONG (iter_032, STRATEGIC
    FINDING):** The 0.30 ΔR²_color threshold has now resisted three
    categorically different attempts: objective choice (5 classes),
    readout architecture fix (scalar + rich), and reconstruction
    supervision (MSE 0.018 yet ΔR² 0.063). When three orthogonal
    classes of intervention converge on the same null, the working
    hypothesis is that the *metric is targeting a behavior the
    architecture is not built to produce*, not that "the next
    intervention will work." The principled response is to ask whether
    the *project goal* — curiosity-driven, decoder-free behavioral
    agent — actually needs ΔR² ≥ 0.30 identity encoding, or whether
    that target was a proxy that has now outlived its usefulness.
*   **PRE-COMMITTED DECISION RULES PRODUCE CLEAN PIVOTS (iter_032,
    PROTOCOL CONFIRMATION):** The branch (a)/(b) rule was written into
    the pre-registration *before* the run. When E2 failed, the pivot
    was not a negotiation — it was the execution of a rule. This is
    the protocol the project should preserve and replicate. iter_033
    pre-registration must include the analogous rule: "if behavioral
    gates fail, project enters constraint-relaxation phase."
*   **DIRECTIONAL SIGNALS ARE NOT NOISE TO BE DISMISSED (iter_032
    META-LESSON, FROM USER HINT):** The iter_031 CLTS Part B result
    (0.59 probe vs 0.44 random, ratio 1.34×) was logged in the prior
    journal as "confounded by representation quality." That framing
    pre-supposes the conclusion that representation quality is the
    gate. A directional signal that exists is data; whether it clears
    a *calibrated* threshold (not a guessed one) is the next question.
    Future iterations: do not dismiss a measured directional signal
    until both the metric and the threshold construction have been
    audited.
*   **CARRIED FORWARD (still valid):**
    - M1 (pooled/batch VICReg) stands.
    - M3 (fixed dimensionality, GDASR log-only) stands.
    - Separate backbone + mask_dyn_sim + coord_vicreg = 0% collapse,
      load-bearing combination (iter_028) — confirmed again by being
      the substrate that survived iter_032's failed readout test.
    - Decoder-free constraint stands; iter_031 reconstruction ceiling
      failed under the broken readout, so reconstruction has not been
      fairly tested as an alternative, but neither has it shown a path.
    - Do not re-introduce positional encoding (cross-objective harmful
      per iter_013 and iter_021).
    - d_t=3 frozen.
    - Hard seeds (53, 71) remain in union seed bank.
    - Pre-registered nulls are first-class results — five consecutive
      diagnostic iterations have now produced a clean strategic pivot.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (RE-LOCALIZED, iter_032):** No
    longer "objective class" (iter_030), no longer "z_dyn readout"
    (iter_031), now: *the conjunction of decoder-free + mean-pool-
    family readout + the ΔR²_color ≥ 0.30 target itself*. Each is
    individually plausible; their conjunction has now resisted three
    categorical attempts. Bottleneck reframed as: "the project's
    representation-side success metric may not be the right success
    metric for the project goal."
*   **Representation-Quality-Gate Loop (NEW, ACTIVE):** Three
    iterations (030, 031, 032) have planned the next iteration around
    "first clear ΔR²_color ≥ 0.30, then proceed." The pre-committed
    pivot in iter_032 dissolves this loop by changing what counts as
    the gate. iter_033 must not re-introduce a representation-side
    ΔR² gate as the binding precondition for behavioral work.
*   **Cross-Backbone Coupling Risk (NEW, TRACKED):** Any future
    architecture that connects coord and dyn backbones via a hard
    attention gate inherits the iter_032 collapse mode. Future
    proposals along these lines must include an a-priori argument
    for how the VICReg variance constraint propagates upstream of
    the gate.
*   **Behavioral-Evaluation-Without-Representation-Foundation Loop
    (DISSOLVED, iter_032):** Was tracked in iter_031 as a protocol
    hazard. The pre-committed branch (b) rule makes the pivot to
    behavioral evaluation *with the best available representation*
    principled, not retreating. The hazard reappears only if iter_033
    fails to pre-register *its* gates against measured controls.
*   **Diagnostic-vs-Constructive Iteration Loop (DORMANT):** Six
    consecutive pre-registered diagnostic iterations (023–024, 029–032)
    have produced a tight strategic localization plus a binding pivot.
    Protocol mature.
*   **Overclaim Loop (DORMANT):** iter_032 executor used "definitively
    falsified" appropriately (binding rule + pre-committed threshold),
    flagged the K=4 collapse as a *different* failure mode than
    expected, and explicitly attributed the pivot to a pre-committed
    rule rather than retrofitting it.
*   **Objective-Swapping Loop (RESOLVED, RETIRED):** No more
    objective-swap experiments until the readout question is settled —
    and the readout question itself is now retired to "constraint-
    relaxation" status rather than "next iteration."
*   **CLTS-Threshold-Was-Guessed Confound (NEW, TRACKED):** The
    iter_031 1.5× collision-selectivity gate was an a-priori guess.
    iter_033 must construct gates from measured random-control
    distributions, not from prior intuition.

## 4. Alternate Research Paths
*   **iter_033: Behavioral Evaluation Pivot (IMMEDIATE PRIORITY,
    PRE-COMMITTED VIA ITER_032 BINDING RULE):**
    - Substrate: iter_028 separate-backbone + mask_dyn_sim +
      coord_vicreg + VICReg-only (ΔR²_color = 0.045, 0% collapse,
      best tracking).
    - Arm 1: Re-run CLTS Part B in N=2 collision-sparse env (iter_031
      protocol), with gate thresholds constructed from the measured
      random-control distribution: collision selectivity gate =
      mean(random) + 2σ(random) computed in-run, perturbation
      selectivity gate analogous, surprise-tracking gate analogous.
    - Arm 2: Causal-sensitivity probe (mass-change). Pre-register the
      gate: tracking-error correlation with the changed-parameter
      magnitude must be statistically distinguishable from zero
      (p < 0.05, ≥5 seeds, Bonferroni for the metric battery).
    - Arm 3: Centroid-MSE on the same representation, anchored
      against Phase-12 references (CLTS 85.85, WUP-MDL 57.34).
    - Binding rule for iter_034: *the project enters Phase 2/3
      integration if at least one Arm-1 or Arm-2 gate is cleared and
      Arm-3 centroid-MSE is ≤ 200 (not at random-baseline floor);
      otherwise the project enters constraint-relaxation, where one
      of {decoder-free, mean-pool readout family, fixed
      dimensionality} must be reopened*.
*   **Behavioral Re-Calibration of CLTS Part B (PROMOTED FROM PARALLEL
    TO PRIMARY, iter_033 ARM 1):** The iter_031 0.59 vs 0.44 directional
    signal is now treated as a candidate finding rather than as noise.
    The 1.5× gate is being replaced with a constructed-from-controls
    gate.
*   **Causal Sensitivity Probe (PROMOTED, iter_033 ARM 2):** This is the
    Section 6 metric that the project has never run end-to-end. It is
    the most direct test of "is the representation good enough for the
    project goal."
*   **Constraint-Relaxation Phase (CONDITIONAL, iter_034+):** If
    iter_033 behavioral gates fail, the project must reopen one of:
    - **Decoder-free constraint:** allow a lightweight decoder
      (e.g. CIFAR-style head) trained jointly. Reconstruction's MSE
      0.018 suggests pixel-level information is present and could be
      leveraged if the mean-pool/decoder-free combination is the limit.
    - **Readout family:** replace the soft-argmax + mean-pool /
      attention-pool family with something not gated on a single
      spatial location. Candidates: object-detection-style ROI pooling
      from a learned anchor, slot attention, transformer cross-attention
      between coord queries and a small set of dyn keys (VICReg applied
      on the full key bank, not on the readout).
    - **Fixed dimensionality:** the M3 frozen-d_t regime may itself be
      too low for color identity (3 channels for 3 objects × color);
      a d_t bump (to 6 or 8) without GDASR recruitment is the cheapest
      relaxation.
    Each relaxation must be argued for explicitly and individually,
    with iter_032's failure mode (cross-backbone coupling collapses
    VICReg) explicitly addressed by the proposed design.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (CONDITIONAL,
    DEMOTED FURTHER):** Reserved for the constraint-relaxation phase
    only.
*   **Micro-Columns (DEFERRED, unchanged).**
*   **Hierarchical Pyramid (Section 8.6) (DEFERRED, unchanged).**
*   **Phase 5 GDASR Reactivation (DEFERRED, unchanged).**

---

## Iteration 032 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 032 — Null Result: Centroid-Gated Readout Architectural Fix Falsified, Project Pivots to Behavioral Evaluation by Pre-Committed Rule

## 1. Pre-Declared Hypothesis and Falsification Criterion
Verbatim from iter_032 pre-registration:

> "Replacing mean-pool z_dyn with attention-pooled multi-dimensional
> feature vectors gated by the coord backbone's soft-argmax attention
> achieves ΔR²_color ≥ 0.30 with variance stability."

Binding decision rule (pre-committed before the run):
> "(a) If the rich readout clears ΔR²_color ≥ 0.30 with lower CI ≥
> 0.18 and collapse ≤ 10%, the representation is solved — advance to
> Phase 2/3 integration. (b) If it yields only another partial gain,
> that is the third convergent signal (after iter_021 CGIR and the
> 5-objective null) that ΔR²_color ≥ 0.30 is the wrong target, and
> the project hard-pivots to behavioral evaluation."

Falsification criterion for the hypothesis itself: any of {ΔR²_color
< 0.30, lower CI < 0.18, collapse > 10%} on the E2 (K=4) arm.

## 2. Experimental Protocol
- Substrate: existing `NonParametricJEPASpatial` separate-backbone CNN
  (iter_028 config: mask_dyn_sim, coord_vicreg).
- Three arms:
  - E1: Mean-pool readout + VICReg-only (control / current best).
  - E1.5: Scalar centroid-gated readout (K=1) + VICReg-only.
  - E2: Rich attention-pooled centroid-gated readout (K=4 multi-
    dimensional feature vector) + VICReg-only.
- Frozen: d_t=3, decoder-free, no positional encoding, buffer=4000,
  seeds drawn from union seed bank including hard seeds 53, 71.
- Collapse criterion: existing `has_collapsed` plus per-dimension std
  < 0.5 (the M1 variance-hinge threshold).
- Metric: ΔR²_color (per-object color regression delta over a
  null-model baseline), measured on a held-out evaluation buffer.

## 3. Observed Quantities
- E2 (rich, K=4): **100% collapse**, per_dim_std ~0.3–0.6 (below 0.5
  viability threshold). ΔR²_color undefined / not computed on
  collapsed seeds.
- E1.5 (scalar centroid, K=1): **10% collapse**, ΔR²_color worse than
  E1 mean-pool on non-collapsed seeds (degraded relative to control).
- E1 (mean-pool control): 0% collapse, ΔR²_color ~0.045 (consistent
  with iter_029 baseline).
- Falsification threshold (ΔR²_color ≥ 0.30, lower CI ≥ 0.18,
  collapse ≤ 10%): violated on all three counts for E2; collapse
  threshold violated and ΔR²_color degraded for E1.5.

## 4. Verdict
**Refuted.** Both K=1 and K=4 variants of the centroid-gated readout
failed the pre-registered gate. The K=4 arm failed by a new mechanism
(cross-backbone attention coupling under VICReg → catastrophic
collapse), and the K=1 arm failed by producing worse identity
encoding than mean-pool while introducing collapse. The pre-committed
binding decision rule triggers branch (b): hard-pivot to behavioral
evaluation.

This is the project's *third* convergent signal that ΔR²_color ≥ 0.30
is not achievable representation-side under the current frozen-
constraint set (decoder-free + mean-pool readout family + fixed
dimensionality):
1. iter_021 CGIR — partial gain (+0.124), missed 0.30.
2. iter_023–031 — 5-objective convergent null (SFA, JEPA, temporal-
   contrastive, variance-ramped SFA, reconstruction).
3. iter_032 — readout-architecture null (K=1 worse, K=4 collapse).

## 5. Construction-vs-Empirical Note
Genuinely empirical: the cross-backbone attention coupling collapse
was not predicted from construction. The pre-registration anticipated
one of two outcomes (clear 0.30, or partial gain like CGIR). What
actually happened — *worse* than mean-pool plus catastrophic
collapse — was a third outcome with a new mechanistic story
(peaked softmax concentrates VICReg variance constraint at the
attended spatial position, driving degeneracy). The K=4-worse-than-
K=1 ordering further confirms this is not a construction artifact:
if it were, the higher-dimensional readout should not collapse
*more*, since K=4 carries strictly more capacity than K=1.

Note: the *binding pivot decision* in branch (b) is not itself an
empirical claim — it is the execution of a pre-committed protocol
rule. Its scientific status is "we did what we said we would do
before seeing the data."

## 6. Limitations
- This result does **not** show that ΔR²_color ≥ 0.30 is *unachievable*
  on any architecture. It shows the three categorical interventions
  tried so far (objective swap, readout fix, supervised reconstruction)
  do not reach it under the project's frozen constraints. A
  constraint-relaxation step (decoder, different readout family, or
  higher d_t) is the next architectural variable, and is reserved for
  iter_034+ conditional on the iter_033 behavioral pivot outcome.
- The pivot to behavioral evaluation is **not** evidence that the
  representation is "good enough for behavior." It is the execution
  of a pre-committed rule that says the question is now worth asking
  directly. iter_033 will measure whether the answer is yes or no
  against pre-registered, control-constructed thresholds.
- The 5-objective convergent null (iter_023–031) was measured under
  the now-known-broken mean-pool readout. It is not yet a clean
  falsification of the M2 mandate at the objective level, because a
  fair re-test would require a working non-mean-pool readout, and
  iter_032's attempted readout fix failed. M2's empirical status
  remains "untestable on the current architecture" rather than
  "falsified."
- The iter_032 cross-backbone coupling collapse is mechanistically
  plausible but has not been independently confirmed by an ablation
  (e.g., the same K=4 readout with VICReg applied upstream of the
  gate instead of at the readout). Such an ablation is *not* on the
  iter_033 path; it is preserved for the constraint-relaxation phase.
- The iter_031 CLTS Part B directional signal (0.59 probe vs 0.44
  random) is what iter_033 will calibrate against measured controls.
  It is a *candidate* behavioral signal, not a result.

---

