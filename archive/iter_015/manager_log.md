# Research Manager Log - Iteration 015

## Iteration 015 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Surprise-Adaptive Contrastive Coordinate Regularization (SA-CCR)—where the covariance regularization weight on the non-parametric soft-argmax bottleneck is dynamically scaled proportionally to the local temporal prediction error (surprise)—will stabilize coordinate representations during high-surprise transitions (collisions) more effectively than a fixed regularization weight. 

Formally, we define the adaptive weight as:
\lambda_{cov}(t) = \lambda_{cov, 0} \cdot (1 + \gamma \cdot \bar{S}(t))
where \lambda_{cov, 0} is the baseline regularization weight, \bar{S}(t) is the exponentially smoothed temporal prediction error (surprise), and \gamma > 0 is the surprise scaling rate. We hypothesize that proportional surprise scaling (Arm L, \gamma = 2.0) will outperform fixed regularization (Arm K, \gamma = 0.0) and inverse surprise scaling (Arm M, \lambda_{cov}(t) = \lambda_{cov, 0} / (1 + \gamma \cdot \bar{S}(t))) in tracking accuracy and coordinate stability under active closed-loop motor control, without degrading overall physics prediction.

**Proposed Falsification Criterion:**
Our hypothesis is falsified if any of the following occur:
1. Arm L (Proportional SA-CCR) does not achieve a statistically significant reduction in post-collision centroid decoding MSE compared to Arm K (fixed CCR-Covariance) across 5 random seeds (Welch's t-test, p >= 0.05).
2. Arm L's post-collision test simulation loss is statistically inferior to Arm K's (Welch's t-test, p < 0.05).
3. The soft spatial variance of Arm L's coordinate bottleneck exceeds 8.5 pixels^2 during high-surprise phases.
4. Arm M (Inverse SA-CCR) achieves lower centroid decoding MSE than Arm L while maintaining equal or better test simulation loss.

**Proposed Method:**
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

## Iteration 015 -> Planner [Strategic Guidance]

# Manager's Note: Strategic Guidance for Phase 15

## 1. Architectural Avoidance: The DSMC Loop vs. True Dual Control
The proposed **Surprise-Adaptive Contrastive Coordinate Regularization (SA-CCR)** is a regression to parameter-tuning heuristics. It is conceptually almost identical to the **Dynamic Surprise-Modulated Spatial Bottleneck Curriculum (DSMC)** evaluated in Iteration 009, merely shifting the target of the surprise-modulation from spatial variance to the covariance regularization parameter ($\lambda_{cov}$). 

While adjusting regularization scaling dynamically is a valid engineering tool, it is a **soft patch** that avoids the core structural bottleneck identified in our research goals: **the competitive optimization interference caused by the triple role of surprise** (as learning gradient, attention router, and motor drive). 

We must proceed to the structural transition mandated for this phase: the **Dual Control Architecture** (separating the fast, reactive **Surprise Detector** from the slow, deliberative **Categorizer** operating on a consistency buffer). Do not spend this entire iteration tuning another adaptive lambda curve. Implement the structural separation.

## 2. The Construction-vs-Empirical Test on Surprise-Modulation
If you evaluate a model where $\lambda_{cov}(t)$ is explicitly programmed to scale up when surprise is high (e.g., during collisions), and you observe "increased coordinate stability and decorrelation during high-surprise frames," **you have demonstrated a definitional identity, not an empirical discovery.** The behavior follows directly from the mathematical construction you built into the training loop.

To achieve a genuine empirical finding:
*   In the **Dual Control** paradigm, the slow Categorizer must decide *whether* to recruit dimensions or update weights at the attended locus based on whether the change reduces variance across a *multi-scenario consistency buffer* (Minimum Description Length principle). 
*   An empirical success would be showing that this slow, buffer-validated Categorizer naturally ignores transient, non-generalizable collision noise *without* requiring an explicit, hand-coded surprise-proportional weight-scaling formula.

## 3. Pre-Registration Mandate & Metric Hygiene
Your proposed falsification criteria are headed in the right direction by incorporating Welch's t-test. However, to maintain high scientific rigor:
1.  **Avoid Absolute Thresholds:** As learned in Phase 14, absolute prediction loss limits (e.g., $0.050$) are highly vulnerable to seed-specific environmental chaos (such as multi-body elastic collisions). Define all performance criteria **relative to the baseline** (e.g., Arm K) using non-inferiority or superiority margins with statistical significance ($p < 0.05$).
2.  **Pre-Registration File:** You must ensure that your exact hypotheses, mathematical formulations of the controller interaction, and quantitative falsification criteria are fully written to `src/pre_registration.md` before execution. Your sub-agents must read and strictly adhere to this file. 
3.  **Language Discipline:** In your final evaluation, describe your findings using restrained, falsifiable language (e.g., "is consistent with," "provides evidence for") and avoid hyperbolic descriptors. An honest null result is a successful validation of our method.

---

