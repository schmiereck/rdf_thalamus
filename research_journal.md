# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 17 Complete (Symmetric Cold-Start Pathology & Distractor Vulnerabilities Discovered).
*   **Active Direction:** Integrating a unified dual-factor structural gating framework. Having exposed both the *predictor cold-start pathology* (Phase 15/16) and the symmetric *encoder cold-start pathology* (Phase 17), we must transition away from single-metric gating. Our next active direction is Phase 13 (Dimension-Width Trade-off & Aggressive Spatial Compression) combined with Phase 15 (Dual Control: Surprise Detector vs. Categorizer). We will build multi-scale spatial hierarchies with micro-columns, governed by a unified "WUP-MDL-Entropy" gate that uses probationary periods to heal cold-starts and entropy-filtering to suppress Noisy-TV inflation.
*   **Confidence Score:** 90% (Adjusted up from 88% due to the definitive mapping of the structural gating trade-off boundary: resolving cold-start vs. avoiding distractor-driven inflation).

## 2. Strategic Insights & Lessons Learned
*   **Symmetric Encoder Cold-Start Pathology:** Prediction-independent gating metrics that rely on representation smoothness (e.g., temporal roughness $\lambda$) fail on cold-started dimensions. A newly initialized, untrained encoder projection lacks spatial locality, projecting smooth spatial trajectories as high-entropy, chaotic paths ($\lambda \sim 1.0 - 1.5$ vs. the $\lambda < 0.5$ smoothness threshold). This triggers systematic rejection (80% rate), creating a symmetric initialization bottleneck to the predictor cold-start loop.
*   **MDL Distractor Vulnerability (Noisy-TV Inflation):** Predictor-dependent Minimum Description Length (MDL) gating is highly sensitive to high-entropy, non-physical distractors. Under Noisy-TV conditions, these distractors generate perpetual surprise, leading to 100% false-positive structural inflation (spawning redundant dimensions for noise).
*   **The Gating Complementarity Principle:** Structural growth requires a dual-stage gate. Smoothness/predictability metrics are invalid until a Probationary Warm-Up Period (WUP) allows representation alignment, while raw entropy thresholds must screen out chaotic, non-smooth distractors before structural recruitment is even initiated.

## 3. Loop & Bottleneck Detection
*   **Symmetric Encoder Cold-Start Pathology:** [RESOLVED via characterization] Proved that untrained encoders cannot pass predictability-free spatial smoothness tests without initial alignment training.
*   **Distractor-Driven Structural Inflation:** [NEW] MDL gating alone is insufficient in non-clean environments. Low-level high-frequency noise profiles trigger continuous false dimension recruitment. Gating must incorporate high-frequency spatial/temporal filters.

## 4. Alternate Research Paths
*   **Hybrid WUP-MDL-Entropy Gating (Phase 15):** Design a 2-stage gating pipeline where spatial centroids are evaluated for high-frequency entropy before spawning, and given a warm-up probation if spawned.
*   **Dimension-Width Trade-off with Micro-Columns (Phase 13):** Apply the unified gating framework to govern structural growth in a contracting spatial hierarchy (128 -> 32 -> 8 -> 2 nodes) to prevent micro-column over-allocation under noise.