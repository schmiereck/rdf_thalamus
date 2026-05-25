# RDF Scientific Pre-Registration

*   **Iteration:** 015
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Surprise-Adaptive Contrastive Coordinate Regularization (SA-CCR)—where the covariance regularization weight on the non-parametric soft-argmax bottleneck is dynamically scaled proportionally to the local temporal prediction error (surprise)—will stabilize coordinate representations during high-surprise transitions (collisions) more effectively than a fixed regularization weight. 

Formally, we define the adaptive weight as:
\lambda_{cov}(t) = \lambda_{cov, 0} \cdot (1 + \gamma \cdot \bar{S}(t))
where \lambda_{cov, 0} is the baseline regularization weight, \bar{S}(t) is the exponentially smoothed temporal prediction error (surprise), and \gamma > 0 is the surprise scaling rate. We hypothesize that proportional surprise scaling (Arm L, \gamma = 2.0) will outperform fixed regularization (Arm K, \gamma = 0.0) and inverse surprise scaling (Arm M, \lambda_{cov}(t) = \lambda_{cov, 0} / (1 + \gamma \cdot \bar{S}(t))) in tracking accuracy and coordinate stability under active closed-loop motor control, without degrading overall physics prediction.

## 2. Falsification Criterion
Our hypothesis is falsified if any of the following occur:
1. Arm L (Proportional SA-CCR) does not achieve a statistically significant reduction in post-collision centroid decoding MSE compared to Arm K (fixed CCR-Covariance) across 5 random seeds (Welch's t-test, p >= 0.05).
2. Arm L's post-collision test simulation loss is statistically inferior to Arm K's (Welch's t-test, p < 0.05).
3. The soft spatial variance of Arm L's coordinate bottleneck exceeds 8.5 pixels^2 during high-surprise phases.
4. Arm M (Inverse SA-CCR) achieves lower centroid decoding MSE than Arm L while maintaining equal or better test simulation loss.

## 3. Proposed Method
Step-by-step description of the experiment:
1. Modify the loss computation in `src/thalamus.py` or the training pipeline to compute the exponentially smoothed local surprise \bar{S}(t) = \alpha \cdot S(t) + (1 - \alpha) \cdot \bar{S}(t-1) with smoothing factor \alpha = 0.1.
2. Implement three experimental arms inside the training loop:
   - Arm K (Baseline): Fixed covariance weight \lambda_{cov, 0} = 0.1 (as validated in Phase 14).
   - Arm L (Proportional SA-CCR): Dynamic weight \lambda_{cov}(t) = \lambda_{cov, 0} * (1 + 2.0 * \bar{S}(t)).
   - Arm M (Inverse SA-CCR): Dynamic weight \lambda_{cov}(t) = \lambda_{cov, 0} / (1 + 2.0 * \bar{S}(t)).
3. Run a matched 5-seed sweep across all three arms under Closed-Loop Thalamic Subsumption (CLTS) control (N=3 objects during passive pre-training, transitioning to N=4 objects during active training).
4. Record and evaluate:
   - Post-collision centroid decoding MSE (overall and post-collision specific frames).
   - Standardized test simulation loss.
   - Soft spatial variance of the coordinate bottleneck.
   - Pointer spatial coverage entropy (to monitor exploration).
5. Apply Welch's t-test and Levene's test on the results across the 5 seeds to validate statistical significance.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
