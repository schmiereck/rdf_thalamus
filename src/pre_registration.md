# RDF Scientific Pre-Registration

*   **Iteration:** 017
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
An encoder-only, prediction-independent gating mechanism based on Encoder-only Smoothness-Uniqueness Gating (ESUG)—which combines linear residual variance (Uniqueness Ratio R_unique > 0.15) and first-difference temporal variance (Smoothness Ratio lambda < 0.5)—can successfully identify and recruit a newly required representation dimension during the N=3 to N=4 object transition without requiring any probationary warm-up period (W=0) for a temporal predictor head. Specifically, this prediction-independent gate will achieve 100% recruitment rate across 5 seeds, reduce post-transition centroid decoding MSE to < 55.0, and decrease decision latency by at least 50% compared to the prediction-dependent WUP-MDL baseline (Arm P), while maintaining 0% false recruitment under high-frequency noise distractors.

## 2. Falsification Criterion
The hypothesis will be falsified if any of the following occur:
1. The recruitment rate of Arm Q (ESUG-100) or Arm Q_fast (ESUG-30) is less than 100% (5/5 seeds) during the N=3 to N=4 transition.
2. The mean post-transition centroid decoding MSE of Arm Q or Arm Q_fast is >= 55.0.
3. The decision latency (steps from transition to recruitment) of Arm Q_fast is not significantly lower than Arm P (WUP-MDL, W=100), or is > 40 steps.
4. The false recruitment rate of Arm Q or Arm Q_fast in the N=3 noise-distractor control group is > 0% (i.e., it incorrectly recruits a noisy dimension).

## 3. Proposed Method
1. Modify `src/thalamus.py` to implement the ESUG gating mechanism. ESUG calculates the uniqueness ratio R_unique (using linear projection residuals of the proposed dimension onto the active dimensions over a sliding buffer B) and the temporal smoothness ratio lambda (first-difference variance normalized by total variance).
2. Implement Arm Q (ESUG with B=100) and Arm Q_fast (ESUG with B=30).
3. Run a 5-seed comparative sweep comparing Arm P (WUP-MDL baseline), Arm Q, and Arm Q_fast under the N=3 to N=4 transition.
4. Run a parallel control sweep where the environment remains at N=3, but a noisy-TV distractor is introduced as a proposed dimension, evaluating the false recruitment rate of Arm P, Arm Q, and Arm Q_fast.
5. Measure and report: recruitment rate, centroid decoding MSE, decision latency, and false recruitment rate across all arms and seeds.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
