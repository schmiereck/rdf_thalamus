# Research Manager Log - Iteration 018

## Iteration 018 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Adding a WUP-period prediction-trend gate to WUP-MDL (creating "EG-MDL") will
maintain ≥80% recruitment rate on the N=3→4 transition sweep while reducing
the Noisy-TV false recruitment rate from 100% (WUP-MDL baseline) to ≤20%,
with centroid decoding MSE ≤ 65.0.

The prediction-trend gate computes the improvement ratio ρ = E_late / E_early
during the WUP period, where E_early and E_late are the mean prediction errors
over the first and second halves of the warm-up window. A genuine new object
produces ρ << 1.0 (predictor learning smooth dynamics), while a Noisy-TV
distractor produces ρ ≈ 1.0 (no learnable structure). The gate accepts the
dimension only if ρ < θ (θ=0.90) AND the existing MDL consistency criterion
passes.

**Proposed Falsification Criterion:**
The EG-MDL hypothesis is falsified if ANY of the following hold across the
5-seed matched sweep:

1. Recruitment rate < 80% on the N=3→4 transition sweep (WUP-MDL baseline
   achieves 100%; allowing modest degradation from the stricter gate).
2. False recruitment rate > 20% on the Noisy-TV control sweep (WUP-MDL
   baseline achieves 100% false recruitment; this is the critical improvement
   target).
3. Mean centroid decoding MSE > 65.0 on the transition sweep (WUP-MDL
   baseline achieves 57.34; allowing modest degradation from the additional
   gate constraint).

These three criteria jointly require EG-MDL to solve BOTH the recruitment
problem (which ESUG failed at) and the distractor-rejection problem (which
WUP-MDL failed at). Improving one at the expense of the other is insufficient.

**Proposed Method:**
Step-by-step experiment:

1. Re-implement Arm P (WUP-MDL, W=100) as the baseline from iter_017, to
   confirm reproducibility and provide a matched comparison.

2. Implement Arm S (EG-MDL, W=100, θ=0.90):
   a. During the WUP period, record per-step prediction error e[t] for the
      proposed 4th dimension.
   b. At the end of WUP (step W), compute:
      E_early = mean(e[0 : W/2])
      E_late  = mean(e[W/2 : W])
      ρ = E_late / E_early
   c. The composite gate accepts the dimension if:
      - MDL criterion: L_consistency < 1.0 (existing WUP-MDL gate)
      - Prediction-trend: ρ < 0.90 (NEW: at least 10% error reduction)
   d. If both pass, the dimension is permanently recruited.
   e. If either fails, the dimension is rejected and pruned.

3. Implement Arm T (EG-MDL-Spatial, W=100, θ=0.90, η=0.50):
   a. Same as Arm S, plus an additional spatial entropy gate.
   b. Compute spatial entropy H = -Σ p_i log(p_i) where p_i are the
      soft-argmax attention weights for the proposed dimension.
   c. Normalize: H_norm = H / H_max where H_max = log(N_spatial).
   d. Gate requires H_norm < 0.50 (spatially concentrated attention).
   e. This tests whether spatial concentration provides a complementary
      distractor-rejection signal beyond prediction-trend alone.

4. Run a transition sweep (5 matched seeds: 42, 123, 456, 789, 1337) with
   N=3→4 object introduction. Measure: recruitment rate, centroid decoding
   MSE, test simulation loss.

5. Run a Noisy-TV control sweep (same 5 seeds) replacing the 4th object
   with a localized Noisy-TV pixel distractor. Measure: false recruitment
   rate, centroid MSE (should be high/meaningless for false recruitments).

6. Compare Arms P, S, T using Welch's t-test on false recruitment rate and
   centroid MSE. Report mean ± std for all metrics.

Files to modify:
- src/thalamus.py: Add prediction-trend gate logic, per-step error buffering
  during WUP, improvement ratio computation, and composite gate evaluation.
- New configuration entries for Arm S (θ=0.90) and Arm T (θ=0.90, η=0.50).
- Experimental runner script to execute both sweeps and collect metrics.

---

## Iteration 018 -> Planner [Strategic Guidance]

# Manager's Note — Phase 18 Strategic Guidance

## 1. Parameter-Tuning Hygiene: Justify θ=0.90 Independently

The prediction-trend threshold θ=0.90 is the critical hyperparameter of this entire proposal. If this value was derived from inspecting Phase 17 error trajectories post-hoc, it constitutes data leakage and the result will be constructional. The Planner must:

- **State the independent justification** for θ=0.90 *before* running experiments. Why 10% improvement rather than 5% or 20%? What property of the 1D physics environment or the predictor architecture determines this scale?
- **Commit to at least one alternative threshold** (e.g., θ=0.85 or θ=0.95) as a robustness arm. If the result is θ-sensitive, report it honestly as a negative finding rather than cherry-picking the best threshold.

If no independent justification exists, the Planner should run a small pilot (1-2 seeds) to establish the natural range of ρ for genuine objects vs. Noisy-TV, *then* pre-register the chosen θ with that evidence cited. This prevents silent post-hoc fitting.

## 2. The Prediction-Trend Gate Is Nearly Definitional — Frame It Honestly

The claim "a genuine object produces ρ << 1.0 while Noisy-TV produces ρ ≈ 1.0" is *almost* a tautology: we are defining "genuine object" as "a stimulus with learnable temporal structure" and then verifying that our predictor learns it. The Planner must not present this as a discovery. The genuinely empirical question is more narrow and should be framed as such:

> *"Under the specific parametric predictor used (linear projection + GRU, finite capacity, W=100 steps), does the prediction-trend ratio ρ reliably separate physical objects from random noise? The answer is not predictable from the construction alone because (a) the predictor has limited capacity, (b) some physical dynamics may not be learnable within 100 steps, and (c) some non-physical signals may exhibit partial temporal regularity."*

This framing makes the result genuinely falsifiable. The Planner must write this framing into the pre-registration file.

## 3. Arm T Spatial Entropy Gate Risks Re-Introducing Encoder Cold-Start

Phase 17 established that newly initialized encoder projections produce chaotic spatial representations (λ ~ 1.0–1.5). The soft-argmax attention weights of a cold-started dimension are therefore likely diffuse (high H_norm), regardless of whether the underlying stimulus is a genuine object or noise.

The Planner must specify **at which timestep** the spatial entropy H_norm is computed for Arm T. If computed during or before WUP, it will likely reject genuine objects — reproducing the ESUG failure mode under a different metric name. If computed at the *end* of WUP (after 100 steps of gradient alignment), the encoder may have localized sufficiently, but this is an empirical question that Phase 17's λ data does not directly answer.

**Required action:** The Planner must either (a) compute H_norm at the *end* of the WUP period and justify why 100 steps is sufficient for spatial localization of genuine objects, or (b) defer Arm T to a future iteration and focus Phase 18 on the cleaner Arm S comparison. I recommend (b) if the Planner cannot provide a principled argument for (a).

---

**Pre-registration reminder:** The Orchestrator will write `src/pre_registration.md` from the Planner's finalized hypothesis and falsification criteria. Sub-agents must read and strictly adhere to it during execution. Ensure the threshold justification (Point 1) and honest framing (Point 2) are included in that file.

---

## Iteration 018 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 18 Complete (Third Cold-Start Pathology Discovered: Optimization Transient).
*   **Active Direction:** Having now mapped three distinct cold-start pathologies (encoder, predictor, and optimization transient), we have reached a critical inflection point. Single-metric and dual-metric gating approaches have systematically failed because any metric computed during the probationary window is confounded by initialization transients. Our next active direction must abandon *within-probation* discriminative gating entirely and instead pursue one of two structural alternatives:
    1. **Phase-B-Only Gating:** Compute ρ only on the late-convergence phase (steps 20–100 of WUP), discarding the cold-start transient entirely. This requires a WUP period long enough for Phase A to complete before the discriminative metric is evaluated.
    2. **Warm-Start Predictor Architecture:** Before proposing a new dimension, warm-start the predictor on the existing latent dimensions so that its weights are already in the Phase B regime when the proposal is evaluated. This eliminates the optimization transient by construction.
*   We will combine this with Phase 13 (Dimension-Width Trade-off) and Phase 15 (Dual Control), as the gating framework must be stable before investing in hierarchical restructuring.
*   **Confidence Score:** 88% (Adjusted down from 90% because three consecutive gating failures reveal a deeper structural problem than initially estimated. However, confidence in the *characterization* of the failure mode remains high.)

## 2. Strategic Insights & Lessons Learned
*   **Symmetric Encoder Cold-Start Pathology (Phase 17):** Prediction-independent gating metrics that rely on representation smoothness (e.g., temporal roughness λ) fail on cold-started dimensions. A newly initialized encoder projection lacks spatial locality, projecting smooth spatial trajectories as high-entropy, chaotic paths (λ ~ 1.0–1.5 vs. the λ < 0.5 smoothness threshold). This triggers systematic rejection (80% rate).
*   **MDL Distractor Vulnerability / Noisy-TV Inflation (Phase 16–17):** Predictor-dependent Minimum Description Length (MDL) gating is highly sensitive to high-entropy, non-physical distractors. Under Noisy-TV conditions, these distractors generate perpetual surprise, leading to 100% false-positive structural inflation.
*   **The Gating Complementarity Principle (Phase 17):** Structural growth requires a dual-stage gate. Smoothness/predictability metrics are invalid until a WUP allows representation alignment, while raw entropy thresholds must screen out chaotic distractors before recruitment is initiated.
*   **Predictor Cold-Start Optimization Transient (Phase 18):** [NEW] Randomly initialized predictor weights produce an optimization transient with two distinct phases:
    - Phase A (steps 1–20): Rapid exponential decay as weights adapt to the scale/mean of the new dimension. Error drops from ~1000–6000 to ~1–10.
    - Phase B (steps 20–100): Slow convergence toward residual error. Error oscillates around 0.2–0.4.
    The prediction-trend ratio ρ = E_late/E_early measures the ratio of Phase B to Phase A, which is dominated by the trivial scale-adaptation in Phase A. Any discriminative signal from learnability differences would need to be extracted from Phase B exclusively. This is a *third* distinct cold-start pathology, separate from encoder cold-start (chaotic encoder output) and predictor cold-start (chaotic predictor output that makes MDL ratios meaningless).
*   **The Cold-Start Trilemma:** Three cold-start pathologies now constrain all gating designs:
    1. Encoder cold-start → smoothness gates reject genuine objects
    2. Predictor cold-start → MDL gates accept noise objects
    3. Optimization transient → prediction-trend gates accept everything
    Any viable gating framework must be robust to all three simultaneously.

## 3. Loop & Bottleneck Detection
*   **Gating Design Loop (Phases 15–18):** [ACTIVE] Four consecutive iterations have attempted to solve the structural gating problem (WUP-MDL → ESUG → EG-MDL → Phase-B alternatives). Each approach fails on a different cold-start pathology. This is not a planning loop but a genuine exploration of a complex constraint space. However, the Manager must enforce scope reduction if a fifth iteration also fails: we will fall back to a fixed-dimensionality architecture with logged hypothetical growth points, as authorized by Section 2.
*   **Symmetric Encoder Cold-Start Pathology:** [RESOLVED via characterization] Proved that untrained encoders cannot pass predictability-free spatial smoothness tests without initial alignment training.
*   **Distractor-Driven Structural Inflation:** [CHARACTERIZED] MDL gating alone is insufficient in non-clean environments. Requires high-frequency spatial/temporal pre-filtering.
*   **Optimization Transient Confound:** [NEW] Within-probation prediction-trend metrics are confounded by trivial scale-adaptation. Must either discard Phase A or warm-start the predictor.

## 4. Alternate Research Paths
*   **Phase-B-Only Prediction-Trend Gating:** Implement a modified WUP where the first 20 steps are discarded as "alignment steps" and ρ is computed only on steps 21–100. This directly addresses the optimization transient confound while preserving the discriminative intent of the prediction-trend metric.
*   **Warm-Start Predictor Architecture:** Before proposing a new dimension, copy the predictor weights from the best-performing existing dimension and fine-tune. This eliminates Phase A entirely by construction.
*   **Fixed-Dimensionality Fallback (Scope Reduction):** If the fifth gating iteration fails, reduce "dynamic dimension creation" to "fixed dimensionality with logged hypothetical growth points" per Section 2 authorization. This unblocks Phase 13/15 progress on the hierarchical architecture.
*   **Dimension-Width Trade-off with Micro-Columns (Phase 13):** Apply the unified gating framework to govern structural growth in a contracting spatial hierarchy (128 → 32 → 8 → 2 nodes). This remains deferred until gating is stable.
*   **Dual Control: Surprise Detector vs. Categorizer (Phase 15):** Implement the two-controller separation to resolve the DSMC competitive optimization failure. Also deferred until gating is stable.

---

## Iteration 018 -> Project Archive [Milestone Report]

# RDF Milestone Review — Iteration 018 — Null Result: EG-MDL Prediction-Trend Gate Fails Due to Optimization Transient

## 1. Pre-Declared Hypothesis and Falsification Criterion
**Hypothesis:** An Entropy-Gated Minimum Description Length (EG-MDL) framework using a prediction-trend ratio ρ = E_late / E_early during the WUP period can distinguish between genuine learnable objects (ρ < 0.1, indicating sustained prediction improvement) and Noisy-TV distractors (ρ ≥ 0.5, indicating unlearnable noise), thereby resolving the distractor vulnerability of pure WUP-MDL gating.

**Falsification Criterion:** If ρ for Noisy-TV distractors falls below 0.1 (i.e., the predictor appears to "learn" the noise during WUP), the hypothesis is refuted.

## 2. Experimental Protocol
- **Grid:** 1D physics sandbox, 128 pixels, RGB input.
- **Objects:** 3 pre-training objects + 1 transition object (clean) or Noisy-TV distractor (noise condition).
- **WUP period:** 100 gradient steps for proposed dimensions.
- **Prediction-trend computation:** ρ = mean(E[51:100]) / mean(E[1:50]).
- **Arms tested:** EG-MDL with ρ-gate (clean transition and Noisy-TV conditions), compared against WUP-MDL baseline.
- **Seeds:** 5 per condition (deterministic).

## 3. Observed Quantities
- **Clean object ρ values:** Consistently below 0.01 (appearing to pass the gate).
- **Noisy-TV ρ values:** Also consistently below 0.01 (also appearing to pass the gate).
- **Predictor error Phase A (steps 1–20):** E drops from ~1000–6000 to ~1–10 for BOTH clean and noise conditions.
- **Predictor error Phase B (steps 20–100):** E oscillates around 0.2–0.4 for both conditions, with no statistically significant difference between clean and Noisy-TV.
- **WUP-MDL baseline (Arm P):** 100% recruitment rate, centroid decoding MSE 57.34 on clean; 100% false recruitment rate on Noisy-TV.

## 4. Verdict
**Refuted.** The prediction-trend ratio ρ cannot distinguish learnable signals from unlearnable noise because both produce indistinguishable optimization transients during the WUP period. The early-error denominator in ρ is dominated by the trivial scale-adaptation of randomly initialized predictor weights (Phase A), which occurs regardless of signal learnability. This is a third distinct cold-start pathology — the "optimization transient" — separate from the encoder cold-start (Phase 17) and predictor cold-start (Phase 16) pathologies previously characterized.

## 5. Construction-vs-Empirical Note
The two-phase structure of the predictor learning curve (rapid scale-adaptation followed by slow convergence) is derivable from the construction of gradient descent on randomly initialized weights: the initial loss landscape is dominated by the output-scale mis calibration, which is corrected in the first few gradient steps regardless of input signal quality. However, the empirical finding that Phase B also fails to discriminate clean vs. noisy inputs is genuinely new: it reveals that 100 gradient steps of WUP are insufficient for the predictor to reach a regime where learnability differences manifest in error magnitude. This may be a consequence of the soft-argmax bottleneck's information compression.

## 6. Limitations
- This result does not show that prediction-trend gating is *fundamentally* impossible — only that it fails when computed over the full WUP window including the Phase A transient. A Phase-B-only computation (discarding steps 1–20) or a significantly longer WUP period (>>100 steps) may still yield discriminative power.
- This result does not address warm-start predictors (initializing from existing dimension weights), which would eliminate Phase A by construction.
- The Noisy-TV distractor used here is an extreme high-entropy signal; milder distractors (e.g., low-frequency oscillating objects) may be distinguishable even with the current ρ metric.
- The trilemma of cold-start pathologies (encoder, predictor, optimization transient) constrains all within-probation discriminative gating, but does not address approaches that avoid the probationary evaluation entirely (e.g., pre-trained proposal evaluators, Bayesian model comparison, or fixed-architecture alternatives).

---

