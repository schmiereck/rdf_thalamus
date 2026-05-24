# RDF Milestone Review — Iteration 006 — Phase 4: Generalization & Noise Robustness

## 1. Pre-Declared Hypothesis and Falsification Criterion
*   **Hypothesis 4.1 (Noise Robustness):** The Z-score normalized surprise Watchdog prevents attention-trapping by Noisy-TV distractors, maintaining a relative tracking efficiency of target objects of $\ge 80.0\%$ compared to a noise-free environment.
*   **Hypothesis 4.2 (Generalization - N=3 to N=4):** Upon introducing a 4th physical object, the GDASR mechanism recruits a 4th latent dimension (recruitment rate $\ge 80\%$), resulting in a $\ge 30\%$ prediction loss reduction over the fixed-dimension B1 baseline, and the activity of the recruited dimension correlates significantly ($|r| \ge 0.4$) with the spatial position of the 4th object.

## 2. Experimental Protocol
*   **Environment:** 1D Physics Sandbox, 128 RGB pixels.
*   **Condition 1 (Noise Robustness):** High-frequency global Gaussian noise ($\sigma = 0.1$) and a structured "Noisy-TV" distractor (a localized 10-pixel segment displaying random independent states each step). Control: Unnormalized surprise watchdog vs. Z-score normalized watchdog.
*   **Condition 2 (Generalization):** Model trained on $N=3$ objects is exposed to $N=4$ objects. Under GDASR, a new dimension is recruited when the residual error exceeds the adaptive threshold.
*   **Evaluations:** 5 independent random seeds. Baselines: B1 (Standard JEPA, fixed dimensions) and B1_large (equivalent parameter count to the recruited model).

## 3. Observed Quantities
*   **Watchdog Tracking Efficiency:** $93.86\%$ relative tracking efficiency under Noisy-TV distractors (Falsification threshold: $< 80.0\%$).
*   **Dimension Recruitment Rate:** $80.0\%$ (4 out of 5 seeds recruited a 4th dimension).
*   **Prediction Loss Reduction over B1:** $6.0\%$ reduction (Falsification threshold: $< 30.0\%$).
*   **Recruited Dimension Spatial Correlation:** $|r| = 0.0456 \pm 0.012$ (Falsification threshold: $< 0.4$).

## 4. Verdict
*   **Hypothesis 4.1 (Noise Robustness):** Consistent. The Z-score normalization successfully prevented attention trapping, maintaining high tracking efficiency ($93.86\%$).
*   **Hypothesis 4.2 (Generalization):** Refuted. While dimension recruitment occurred at the expected rate ($80\%$) and provided a minor prediction benefit ($6.0\%$), it fell far short of the $30\%$ target. Crucially, the recruited dimension did not encode the spatial position of the novel object ($|r| \approx 0.0456$).

## 5. Construction-vs-Empirical Note
*   **Noisy-TV Avoidance:** This is a **definitional identity**. By defining the watchdog routing over $z_i = (S_i - \mu_i) / \sigma_i$, any stationary high-variance noise source naturally inflates $\sigma_i$, driving the steady-state $z_i$ toward zero. This is a mathematical consequence of Z-score scaling, not an emergent cognitive property of the network.
*   **Dimension Specialization:** The lack of spatial correlation shows that bottom-up residual recruitment does *not* automatically self-organize into localized coordinate representation. The recruited dimension acts as a generic residual capacity booster rather than a dedicated "object tracker," which is an empirical discovery of a limitation in unsupervised dimension recruitment.

## 6. Limitations
*   Unsupervised, prediction-error-driven dimension recruitment (GDASR) allocates variance capacity but lacks the spatial inductive bias to associate new dimensions with specific physical objects.
*   Without downstream task-driven gradients (e.g., motor target tracking) or explicit spatial routing, recruited dimensions display diffuse representations.