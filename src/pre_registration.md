# RDF Scientific Pre-Registration

*   **Iteration:** 018
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Adding a WUP-period prediction-trend gate to WUP-MDL (creating "EG-MDL") will
maintain ≥80% recruitment rate on the N=3→4 transition sweep while reducing
the Noisy-TV false recruitment rate from 100% (WUP-MDL baseline) to ≤20%,
with centroid decoding MSE ≤ 65.0.

The prediction-trend gate computes the improvement ratio ρ = E_late / E_early
during the WUP period, where E_early and E_late are the mean prediction errors
over the first and second halves of the warm-up window. A genuine new object
produces ρ << 1.0 (predictor learning smooth dynamics), while a Noisy-TV
distractor produces ρ ≈ 1.0 (no learnable structure). The gate accepts the
dimension only if ρ < θ (θ=0.90) AND the existing MDL consistency criterion
passes.

## 2. Falsification Criterion
The EG-MDL hypothesis is falsified if ANY of the following hold across the
5-seed matched sweep:

1. Recruitment rate < 80% on the N=3→4 transition sweep (WUP-MDL baseline
   achieves 100%; allowing modest degradation from the stricter gate).
2. False recruitment rate > 20% on the Noisy-TV control sweep (WUP-MDL
   baseline achieves 100% false recruitment; this is the critical improvement
   target).
3. Mean centroid decoding MSE > 65.0 on the transition sweep (WUP-MDL
   baseline achieves 57.34; allowing modest degradation from the additional
   gate constraint).
4. For Arm S_alt (θ=0.85, robustness arm): if the result is θ-sensitive —
   i.e., Arm S (θ=0.90) passes but Arm S_alt fails, or vice versa — this
   constitutes a sensitivity finding that must be reported honestly. It does
   not falsify Arm S per se, but it undermines confidence in the threshold
   choice and indicates that the separation margin between genuine objects
   and distractors is narrower than anticipated.

These three primary criteria (with the sensitivity caveat for Arm S_alt) jointly
require EG-MDL to solve BOTH the recruitment problem (which ESUG failed at) and
the distractor-rejection problem (which WUP-MDL failed at). Improving one at the
expense of the other is insufficient.

## 3. Proposed Method
Step-by-step experiment:

1. Re-implement Arm P (WUP-MDL, W=100) as the baseline from iter_017, to
   confirm reproducibility and provide a matched comparison.

2. Implement Arm S (EG-MDL, W=100, θ=0.90):
   a. During the WUP period, record per-step prediction error e[t] for the
      proposed 4th dimension.
   b. At the end of WUP (step W), compute:
      E_early = mean(e[0 : W/2])
      E_late  = mean(e[W/2 : W])
      ρ = E_late / E_early
   c. The composite gate accepts the dimension if:
      - MDL criterion: L_consistency < 1.0 (existing WUP-MDL gate)
      - Prediction-trend: ρ < 0.90 (NEW: at least 10% error reduction)
   d. If both pass, the dimension is permanently recruited.
   e. If either fails, the dimension is rejected and pruned.

3. Implement Arm S_alt (EG-MDL, W=100, θ=0.85):
   a. Same as Arm S but with θ=0.85 (requiring 15% error reduction during WUP).
   b. This robustness arm tests whether results are sensitive to the choice of θ.

> Arm T (EG-MDL-Spatial) is DEFERRED to a future iteration. The Manager's guidance notes that computing spatial entropy H_norm on a cold-started encoder projection is likely to reject genuine objects — reproducing the ESUG failure mode under a different metric name. Phase 17 established that newly initialized encoder projections produce chaotic spatial representations (λ ~ 1.0-1.5). Even at the end of a 100-step WUP period, the encoder's spatial attention for the new dimension may remain diffuse for genuine objects, making H_norm an unreliable gate without further evidence. If Arm S succeeds, Arm T can be revisited with H_norm computed after a longer stabilization period.

4. Run a transition sweep (5 matched seeds: 42, 123, 456, 789, 1337) with
   N=3→4 object introduction. Measure: recruitment rate, centroid decoding
   MSE, test simulation loss.

5. Run a Noisy-TV control sweep (same 5 seeds) replacing the 4th object
   with a localized Noisy-TV pixel distractor. Measure: false recruitment
   rate, centroid MSE (should be high/meaningless for false recruitments).

6. Compare Arms P, S, S_alt using Welch's t-test on false recruitment rate and
   centroid MSE. Report mean ± std for all metrics.

Files to modify:
- src/thalamus.py: Add prediction-trend gate logic, per-step error buffering
  during WUP, improvement ratio computation, and composite gate evaluation.
- New configuration entries for Arm S (θ=0.90) and Arm S_alt (θ=0.85).
- Experimental runner script to execute both sweeps and collect metrics.

## 4. Threshold Justification
The predictor architecture is a 2-layer MLP (128 hidden units) operating on
3-step latent histories. During 100 WUP steps at lr=1e-3 with batch_size=32,
genuine physical objects in the 1D sandbox exhibit smooth dynamics
(constant-velocity trajectories with occasional elastic collisions). The
predictor should learn constant-velocity extrapolation within ~50 steps,
producing a measurable reduction in prediction error between the first and
second halves of the WUP window. A 10% error reduction threshold (ρ < 0.90)
is conservative because:

- For genuine objects, even a partially-trained predictor should reduce error
  by well over 10% in 50 additional gradient steps.
- For Noisy-TV, which has no learnable temporal structure, the error should
  remain approximately constant (ρ ≈ 1.0).
- The threshold is intentionally lenient to avoid false rejections of genuine
  objects whose dynamics are harder to learn (e.g., post-collision trajectories).

Additionally, we commit to θ=0.85 as a robustness arm (Arm S_alt). If the
result is θ-sensitive (i.e., Arm S passes but S_alt fails, or vice versa),
this will be reported honestly rather than cherry-picked.

## 5. Honest Framing: Empirical Falsifiability
> Under the specific parametric predictor used (2-layer MLP, 128 hidden units, 3-step history, finite capacity, W=100 steps), does the prediction-trend ratio ρ reliably separate physical objects from random noise? The answer is not predictable from the construction alone because (a) the predictor has limited capacity, (b) some physical dynamics may not be learnable within 100 steps (e.g., rare collision sequences), and (c) some non-physical signals may exhibit partial temporal regularity (e.g., a Noisy-TV with correlated noise). The prediction-trend gate is nearly definitional — we are verifying that our predictor can learn smooth physics faster than random noise within a fixed budget. The genuinely empirical question is whether the separation is clean enough at the specific threshold θ=0.90 to produce ≥80% recruitment and ≤20% false recruitment simultaneously.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
