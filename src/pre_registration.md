# RDF Scientific Pre-Registration

*   **Iteration:** 008
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Introducing an unsupervised spatial localization bottleneck—implemented by minimizing the soft spatial variance of the active/recruited latent channel's activations over the 1D space—and coupling it with the output-as-input attention loop (where the attention query targets the channel's spatial centroid for closed-loop active probing) will stabilize the spatial alignment of the recruited dimension. This combined mechanism will:
1. Increase the average absolute Pearson correlation |r| between the recruited dimension's spatial centroid (and activation) and the true physical coordinate of the novel object to >= 0.60 across a 5-seed sweep, while stabilizing seed-to-seed variance such that no single seed falls below |r| = 0.45 (reversing the high variance of Phase 7, which ranged from 0.005 to 0.562).
2. Achieve a further reduction in post-hoc linear decoding MSE of the novel object's coordinate by at least 25% compared to the Phase 7 Active Probing baseline (reducing average MSE from 73.65 to < 55.0).
3. Reduce the soft spatial variance (spread) of the recruited dimension's activation on the novel object by at least 40% compared to the unconstrained Phase 7 active probing baseline.

## 2. Falsification Criterion
The hypothesis will be proven FALSE if any of the following outcomes are observed over the 5-seed evaluation sweep:
1. The average absolute Pearson correlation |r| between the recruited dimension's activation/centroid and the novel object's position is < 0.60, or if any individual seed exhibits |r| < 0.45.
2. The average post-hoc linear decoding MSE of the novel object's coordinate is >= 55.0 (failing to achieve a 25% improvement over the 73.65 baseline).
3. The average soft spatial variance of the active/recruited channel is not reduced by at least 40% compared to the unconstrained active probing baseline.
4. The recruitment rate of the 4th dimension during active probing drops below 100% (5 out of 5 seeds).
5. Criterion 5 (Non-Collapse & Activity Threshold): The hypothesis is falsified if the recruited channel k undergoes representation collapse. Specifically, its mean activation magnitude must remain active (e.g., E[|a_k|] >= 0.1 * 1/C sum_c E[|a_c|]) and its temporal standard deviation must be non-trivial (std(x_mean_k) > 5.0 pixels), proving it has not collapsed into an inactive channel or a static, non-responsive spatial spike.

## 3. Proposed Method
1. Spatial Centroid & Variance Computation: For each channel c of the 1D latent representation, compute the spatial centroid x_mean_c = sum_i(i * softmax(a_{c, i})) and the soft spatial variance Var_c = sum_i((i - x_mean_c)^2 * softmax(a_{c, i})), where a_{c, i} is the activation of channel c at spatial index i of the 1D feature map.
2. Spatial Bottleneck Loss: Add a regularization term to the unsupervised loss objective of the plastic layer: L_spatial = lambda * Var_k, where k is the channel holding the Attention Token (the recruited/active channel representing the novel object) and lambda is a scaling hyperparameter. Stop gradients from updating other channels to prevent collapse.
3. Output-as-Input Active Probing: Modify the motor controller to receive the spatial centroid x_mean_k of the attended channel as the target coordinate, driving the continuous physical pointer to target and probe this localized entity.
4. Control vs. Experimental Sweep: Run a 5-seed comparison sweep comparing this "Spatial Bottleneck + Output-as-Input Active Probing" model (Experimental) against the "Unconstrained Active Probing" model from Phase 7 (Control).
5. Measurement: Measure and report the Pearson correlation |r| of the recruited channel with the true coordinate, the post-hoc linear coordinate decoding MSE, the spatial variance of activations, and the dimension recruitment rate.

## 4. Hyperparameter Search Space
We pre-register the hyperparameter search space for lambda to be:
*   $\lambda \in \{0.01, 0.1, 1.0\}$
*   Default selection: $\lambda = 0.1$

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
