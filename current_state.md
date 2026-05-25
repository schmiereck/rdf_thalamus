# Current Research State
Phase: Phase 17 Complete (ESUG Falsified, Encoder Cold-Start Pathology Discovered)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 17 evaluated the prediction-independent Encoder-only Smoothness-Uniqueness Gating (ESUG) framework to bypass predictor cold-starts.

## Confirmed
- FALSIFICATION OF ESUG HYPOTHESIS (iter_17.1): ESUG gating (Arms Q & Q_fast) is resoundingly falsified. Recruitment rate was only 20% (1/5 seeds), and mean post-transition centroid decoding MSE was 82.83 (Arm Q) and 185.95 (Arm Q_fast), violating the success threshold of < 55.0.
- ENCODER COLD-START PATHOLOGY (iter_17.1): Randomly initialized proposed dimensions produce highly rough temporal trajectories (lambda ~ 1.0-1.5), triggering immediate ESUG gate rejections. This establishes a symmetric "encoder cold-start" pathology to Phase 15's "predictor cold-start."
- WUP-MDL PERFORMANCE (iter_17.1): WUP-MDL (Arm P, W=100) remains highly robust for coordinate tracking under drift, achieving 100% recruitment (5/5 seeds) and a centroid decoding MSE of 57.34.
- DISTRACTOR VULNERABILITY (iter_17.1): Arm P exhibits a 100% false recruitment rate (5/5 seeds) in the Noisy-TV control group, indicating that predictor-dependent gates are easily fooled by high-entropy, localized noise. ESUG has superior noise specificity (rejecting Noisy-TV in 4/5 seeds), but fails due to its severe cold-start bias.

## Refuted / Falsified
- PREDICTION-INDEPENDENT ESUG GATING (iter_17.1): Encoder-only gating without a warm-up period is fundamentally unsuitable for structural growth due to the rough temporal dynamics of newly spawned encoder projections.

## Best Result
- Arm P (WUP-MDL, W=100, transition sweep): Centroid Decoding MSE: 57.34, Test Sim Loss: 0.0791 (iter_17.1).
- Arm Q (ESUG-100, seed 42, recruited): Centroid Decoding MSE: 58.24, Test Sim Loss: 0.0198 (iter_17.1).

## In Progress
- Designing an Entropy-Gated Minimum Description Length (EG-MDL) framework to combine WUP-MDL's coordinate tracking robustness with active distractor suppression.

## Open Questions
- Can an adaptive temporal smoothness threshold lambda(t) that decays during evaluation resolve the encoder cold-start pathology?
- How can we modify MDL gating to distinguish a true clean object from a highly unmodelable distractor like Noisy-TV?
- Is there an information-theoretic gating metric (like mutual information or spatial entropy of centroids) that is robust to both encoder and predictor cold-starts?
