# Current Research State
Phase: Phase 10 Complete (DSDT Sweep Evaluated and Audited)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 10 goal was to evaluate whether decoupling representation channels into a dual-stream latent space (Spatial Coordinate Stream $z^{coord}$ and Temporal Dynamics Stream $z^{dyn}$) with stop-gradients could resolve the spatial localization-prediction trade-off.

## Confirmed
- MATHEMATICAL GRADIENT DECOUPLING (iter_010.1): Specificity unit tests confirmed that spatial bottleneck gradients only update the coordinate stream head, while dynamics prediction gradients only flow back to the dynamics head and shared conv backbone.
- ACTIVE MULTI-STREAM INTEGRATION (iter_010.3): The Information Flow Control Test confirmed that the predictor actively utilizes the coordinate stream; zero-masking $z^{coord}$ during inference caused prediction loss to spike from 13.44 to 3952.18 (a 29292.3% error increase).

## Refuted / Falsified
- REFUTED (iter_010.3): Decoupling via complete stop-gradient isolation resolves the localization-prediction trade-off. Instead, it triggered all three pre-registered falsification criteria:
  1. Spatial variance of $z^{coord}$ rose to 1356.52 (Falsified: $> 75.0$).
  2. Prediction loss was 13.44, a massive penalty compared to Arm C (Falsified: ratio vs Arm A $\ge 1.10$).
  3. Collapse rate was 100% across all 5 seeds (Falsified: collapse rate $> 0.0\%$).
- Complete stop-gradient isolation prevents coordinate representations from grounding. Without prediction or visual gradients, the coordinate stream becomes "semantically blind"—representing static background noise or arbitrary static visual spikes.

## Best Result
- Gentle Bottleneck (Arm A) Centroid Decoding MSE: 69.11, Spatial Variance: 118.76, Test Sim Loss: 0.081522
- Single-Stream DSMC (Arm C) Centroid Decoding MSE: 73.46, Spatial Variance: 70.61, Test Sim Loss: 0.100149
- Dual-Stream DSDT (Arm D) Centroid Decoding MSE: 75.88, Spatial Variance: 1356.52, Test Sim Loss: 13.446293 (Collapsed)

## In Progress
- Devising a multi-stage curriculum or auxiliary spatial reconstruction loss to ground decoupled coordinate representations.

## Open Questions
- How can we ground the coordinate stream without letting prediction gradients dilute the spatial bottleneck?
- Does a multi-stage training curriculum (first training jointly with gradients, then decoupling) prevent semantic blindness?
- Can we use an auxiliary self-supervised spatial reconstruction loss to ground coordinate channels?
