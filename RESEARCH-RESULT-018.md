# RDF Milestone Review — Iteration 018 — Null Result: EG-MDL Prediction-Trend Gate Fails Due to Optimization Transient

## 1. Pre-Declared Hypothesis and Falsification Criterion
**Hypothesis:** An Entropy-Gated Minimum Description Length (EG-MDL) framework using a prediction-trend ratio ρ = E_late / E_early during the WUP period can distinguish between genuine learnable objects (ρ < 0.1, indicating sustained prediction improvement) and Noisy-TV distractors (ρ ≥ 0.5, indicating unlearnable noise), thereby resolving the distractor vulnerability of pure WUP-MDL gating.

**Falsification Criterion:** If ρ for Noisy-TV distractors falls below 0.1 (i.e., the predictor appears to "learn" the noise during WUP), the hypothesis is refuted.

## 2. Experimental Protocol
- **Grid:** 1D physics sandbox, 128 pixels, RGB input.
- **Objects:** 3 pre-training objects + 1 transition object (clean) or Noisy-TV distractor (noise condition).
- **WUP period:** 100 gradient steps for proposed dimensions.
- **Prediction-trend computation:** ρ = mean(E[51:100]) / mean(E[1:50]).
- **Arms tested:** EG-MDL with ρ-gate (clean transition and Noisy-TV conditions), compared against WUP-MDL baseline.
- **Seeds:** 5 per condition (deterministic).

## 3. Observed Quantities
- **Clean object ρ values:** Consistently below 0.01 (appearing to pass the gate).
- **Noisy-TV ρ values:** Also consistently below 0.01 (also appearing to pass the gate).
- **Predictor error Phase A (steps 1–20):** E drops from ~1000–6000 to ~1–10 for BOTH clean and noise conditions.
- **Predictor error Phase B (steps 20–100):** E oscillates around 0.2–0.4 for both conditions, with no statistically significant difference between clean and Noisy-TV.
- **WUP-MDL baseline (Arm P):** 100% recruitment rate, centroid decoding MSE 57.34 on clean; 100% false recruitment rate on Noisy-TV.

## 4. Verdict
**Refuted.** The prediction-trend ratio ρ cannot distinguish learnable signals from unlearnable noise because both produce indistinguishable optimization transients during the WUP period. The early-error denominator in ρ is dominated by the trivial scale-adaptation of randomly initialized predictor weights (Phase A), which occurs regardless of signal learnability. This is a third distinct cold-start pathology — the "optimization transient" — separate from the encoder cold-start (Phase 17) and predictor cold-start (Phase 16) pathologies previously characterized.

## 5. Construction-vs-Empirical Note
The two-phase structure of the predictor learning curve (rapid scale-adaptation followed by slow convergence) is derivable from the construction of gradient descent on randomly initialized weights: the initial loss landscape is dominated by the output-scale mis calibration, which is corrected in the first few gradient steps regardless of input signal quality. However, the empirical finding that Phase B also fails to discriminate clean vs. noisy inputs is genuinely new: it reveals that 100 gradient steps of WUP are insufficient for the predictor to reach a regime where learnability differences manifest in error magnitude. This may be a consequence of the soft-argmax bottleneck's information compression.

## 6. Limitations
- This result does not show that prediction-trend gating is *fundamentally* impossible — only that it fails when computed over the full WUP window including the Phase A transient. A Phase-B-only computation (discarding steps 1–20) or a significantly longer WUP period (>>100 steps) may still yield discriminative power.
- This result does not address warm-start predictors (initializing from existing dimension weights), which would eliminate Phase A by construction.
- The Noisy-TV distractor used here is an extreme high-entropy signal; milder distractors (e.g., low-frequency oscillating objects) may be distinguishable even with the current ρ metric.
- The trilemma of cold-start pathologies (encoder, predictor, optimization transient) constrains all within-probation discriminative gating, but does not address approaches that avoid the probationary evaluation entirely (e.g., pre-trained proposal evaluators, Bayesian model comparison, or fixed-architecture alternatives).