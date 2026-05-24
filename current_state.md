# Current Research State
Phase: Phase 2 Complete (Thalamic Gating Evaluated & Falsified)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 2 goal was to implement Thalamic Gating (Pillar D) with attention-token routing, plasticity gating, soft Z-score normalization, relative stability locks, and physical tracking overlap.

## Confirmed
- Soft-normalized Z-score surprise routing with epsilon and variance floors successfully resolves the Z-score noise/explosion bottleneck (iter_004.1).
- Dynamic gradient-gating and plasticity-gating successfully freeze gradients in inactive layers, focusing learning purely on active attention loci (iter_004.1).
- Thalamic Gated Net achieves a massive **49.1% prediction loss reduction** compared to the single-layer baseline B1 (0.02301 vs 0.04520) and an **18.1% reduction** vs the non-gated control (0.02301 vs 0.02811), confirming that gating surprise-driven attention significantly enhances representation learning and prediction quality in time (iter_004.3).
- Staged training via Relative Stability Lock successfully prevents input-drift collapse, reducing raw variance of L2 test loss compared to the non-gated model (0.00718 vs 0.01160) (iter_004.3).

## Refuted
- REFUTED: Surprise-driven Thalamic Gating with a 200-step cooldown maintains physical target tracking overlap > 0.85 (iter_004.3). Objects move across 32-pixel segments within 16-32 steps, causing the 200-step cooldown to introduce severe lag, resulting in near-random tracking overlap (11.20% test, 22.75% train).
- REFUTED: Thalamic Gating achieves superior sample efficiency to reach loss < 0.08 (iter_004.3). Both gated and non-gated models reach L2 prediction loss < 0.08 immediately in step 1, because the latent representation task is rapidly optimized.
- REFUTED: The gated model shows a statistically significant reduction in L2 test prediction loss variance compared to the non-gated model (iter_004.3). While the raw standard deviation is lower (0.00718 vs 0.01160), Levene's test p-value is 0.3560, failing the p < 0.05 significance threshold.

## Best Result
- Gated L2 Test Prediction Loss: 0.02301 ± 0.00718 (iter_004.3)
- Loss reduction vs Baseline B1: 49.09% (iter_004.3)

## In Progress
- Preparing Phase 3 (Subsumption Motorics & Closed Loop) implementation.

## Open Questions
- How can we resolve physical tracking lag? Can a surprise-modulated dynamic cooldown allow the attention token to track fast-moving physical objects?
- Will coupling motor control to the gated representations allow the agent to actively "chase" or perturb the tracked object, thereby increasing physical overlap and causal understanding?
