# RDF Milestone Review — Iteration 017 — Null Result — Prediction-Independent ESUG Gating & The Encoder Cold-Start Pathology

## 1. Pre-Declared Hypothesis and Falsification Criterion
- **Hypothesis:** An unsupervised, prediction-independent encoder-level gating metric (ESUG) can govern structural dimension recruitment without a probationary warm-up period, circumventing the predictor-head bias while retaining specificity against high-entropy distractors.
- **Falsification Criterion:** The hypothesis is falsified if:
  1. ESUG fails to achieve a recruitment rate $\ge 80\%$ on the novel 4th object under clean conditions, OR
  2. The temporal roughness ($\lambda$) of newly initialized dimensions remains above the smoothness threshold ($\lambda \ge 0.5$) for sustained steps, triggering systematic rejection.

## 2. Experimental Protocol
- **Grid Size:** 128 RGB pixels, 1D Sandbox.
- **Entities:** N=3 objects transitioning to N=4 (novel object introduced at step 1500).
- **Parameters:** Learning rate $\eta = 0.001$, VICReg covariance weight $\mu = 25.0$.
- **Control Run:** Arm P (Predictor-dependent MDL gating with $W=100$) evaluated under both clean and Noisy-TV distractor conditions.
- **Experimental Run:** ESUG Gating (evaluating spatial temporal smoothness $\lambda$ and centroid uniqueness) without warm-up, evaluated under clean and Noisy-TV conditions.

## 3. Observed Quantities
- **ESUG Recruitment Rate (Clean):** 20% (Falsified; 4/5 seeds rejected recruitment).
- **Temporal Roughness of New Dimension ($\lambda$):** $1.0 - 1.5$ at step 1501 (Threshold: $< 0.5$).
- **Arm P Recruitment Rate (Noisy-TV):** 100% (100% false-positive rate, demonstrating structural inflation).
- **ESUG Rejection Rate (Noisy-TV):** 80% (Showing high noise specificity, but crippled by cold-start).

## 4. Verdict
- **Refuted.** The ESUG gating mechanism without a warm-up period is completely non-viable for structural allocation because random network weights inherently project structured physical trajectories as high-roughness temporal noise.

## 5. Construction-vs-Empirical Note
While the mathematical roughness of a random projection is derivable from random matrix and chaotic systems theory, the exact empirical overlap—and the discovery that ESUG possesses high distractor specificity yet is blocked by the exact symmetric counterpart to the predictor cold-start loop—is a genuinely new architectural insight. It maps the Pareto boundary of structural plasticity.

## 6. Limitations
This result demonstrates that neither purely predictor-dependent nor purely encoder-dependent metrics can operate robustly in isolation under a cold-start regime. A hybrid, two-stage protocol is mandatory to resolve both initialization roughness and distractor-driven structural inflation.