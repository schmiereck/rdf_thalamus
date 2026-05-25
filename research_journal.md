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