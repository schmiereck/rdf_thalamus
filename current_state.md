# Current Research State
Phase: Phase 15 Complete (Surprise-Adaptive Covariance and Dual Control Evaluated)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 15 goal was to evaluate Surprise-Adaptive Contrastive Coordinate Regularization (SA-CCR) and implement the structural separation of the Surprise Detector and slow Categorizer (Dual Control).

## Confirmed
- FALSIFICATION OF SURPRISE-PROPORTIONAL REGULARIZATION (iter_15.1): Dynamic proportional scaling of the covariance weight (Arm L) did not significantly reduce post-collision centroid decoding MSE compared to the fixed baseline (Arm K: 62.64 vs Arm L: 65.53; Welch's p = 0.895), resoundingly falsifying the SA-CCR hypothesis.
- VOLATILITY OF PROPORTIONAL SCALING (iter_15.1): Arm L introduced high optimization volatility, triggering an instability in seed 456 where sim_loss exploded to 14.88, demonstrating that hand-coded surprise modulation curves are unstable.
- COLD-DIMENSION REJECTION BIAS IN MDL GATES (iter_15.1): In Arm N, the proposed consistency ratio L_consistency = sim_new / sim_old was systematically biased against recruitment. Because the newly proposed dimension's predictor head was "cold" (randomly initialized), its validation loss was extremely high, yielding ratios of 1.3e4 to 3.4e4. This forced the Categorizer to reject 100% of recruitment proposals (5/5 passive and 31/31 active retries), leaving Arm N stuck at d_t = 3 and unable to track the novel 4th object (mse_cent = 130.39 vs Arm K = 62.64).

## Refuted / Falsified
- SUPERIORITY OF PROPORTIONAL SCALING VS INVERSE (iter_15.1): Arm M (Inverse SA-CCR) achieved a lower mean centroid decoding MSE (64.08) and test simulation loss (0.148) compared to Arm L (65.53 and 14.88 respectively), directly falsifying pre-registered Criterion 4.
- IMMEDIATE PREDICTION-BASED RECRUITMENT GATING (iter_15.1): The structural hypothesis that a vanilla MDL gate based on immediate predictive error can cleanly regulate dimensionality growth is refuted due to cold-start predictor bias.

## Best Result
- Arm K (Baseline): Centroid Decoding MSE: 62.64, Test Sim Loss: 0.0901, Soft Spatial Variance: 8.28, Pointer Spatial Entropy: 3.96 (iter_15.1).

## In Progress
- Investigating mitigations for cold-start predictor bias in MDL consistency gating, specifically asymmetric warm-up training for proposed channels or encoder-only spatial consistency metrics.

## Open Questions
- How can we implement an asymmetric warm-up phase to train proposed dimensions before the MDL consistency gate evaluates them?
- Can we define a purely encoder-based consistency metric (e.g., spatial compactness or mutual information of centroids) that avoids prediction-head cold-start bias?
- Does staggering learning rates (slower for the encoder, faster for the predictor) mitigate cold-dimension rejection?
