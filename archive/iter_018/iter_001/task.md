Update the pre-registration file at src/pre_registration.md with the Manager's three required additions. Read the current file first, then rewrite it with these additions:

## 1. Threshold Justification for θ=0.90

Add a section "## 4. Threshold Justification" that provides an independent, pre-experimental justification for θ=0.90:

The predictor architecture is a 2-layer MLP (128 hidden units) operating on 3-step latent histories. During 100 WUP steps at lr=1e-3 with batch_size=32, genuine physical objects in the 1D sandbox exhibit smooth dynamics (constant-velocity trajectories with occasional elastic collisions). The predictor should learn constant-velocity extrapolation within ~50 steps, producing a measurable reduction in prediction error between the first and second halves of the WUP window. A 10% error reduction threshold (ρ < 0.90) is conservative because:
- For genuine objects, even a partially-trained predictor should reduce error by well over 10% in 50 additional gradient steps
- For Noisy-TV, which has no learnable temporal structure, the error should remain approximately constant (ρ ≈ 1.0)
- The threshold is intentionally lenient to avoid false rejections of genuine objects whose dynamics are harder to learn (e.g., post-collision trajectories)

Additionally, we commit to θ=0.85 as a robustness arm (Arm S_alt). If the result is θ-sensitive (i.e., Arm S passes but S_alt fails, or vice versa), this will be reported honestly rather than cherry-picked.

## 2. Honest Framing of the Prediction-Trend Gate

Add a section "## 5. Honest Framing: Empirical Falsifiability" that includes this exact paragraph:

> Under the specific parametric predictor used (2-layer MLP, 128 hidden units, 3-step history, finite capacity, W=100 steps), does the prediction-trend ratio ρ reliably separate physical objects from random noise? The answer is not predictable from the construction alone because (a) the predictor has limited capacity, (b) some physical dynamics may not be learnable within 100 steps (e.g., rare collision sequences), and (c) some non-physical signals may exhibit partial temporal regularity (e.g., a Noisy-TV with correlated noise). The prediction-trend gate is nearly definitional — we are verifying that our predictor can learn smooth physics faster than random noise within a fixed budget. The genuinely empirical question is whether the separation is clean enough at the specific threshold θ=0.90 to produce ≥80% recruitment and ≤20% false recruitment simultaneously.

## 3. Arm T Decision: Deferred

Modify Section 3 (Proposed Method) to DEFER Arm T. Replace Arm T with the following note:

> Arm T (EG-MDL-Spatial) is DEFERRED to a future iteration. The Manager's guidance notes that computing spatial entropy H_norm on a cold-started encoder projection is likely to reject genuine objects — reproducing the ESUG failure mode under a different metric name. Phase 17 established that newly initialized encoder projections produce chaotic spatial representations (λ ~ 1.0-1.5). Even at the end of a 100-step WUP period, the encoder's spatial attention for the new dimension may remain diffuse for genuine objects, making H_norm an unreliable gate without further evidence. If Arm S succeeds, Arm T can be revisited with H_norm computed after a longer stabilization period.

Replace Arm T in the method with Arm S_alt:
> 3. Implement Arm S_alt (EG-MDL, W=100, θ=0.85):
>    a. Same as Arm S but with θ=0.85 (requiring 15% error reduction during WUP).
>    b. This robustness arm tests whether results are sensitive to the choice of θ.

Also update the falsification criteria if needed to account for Arm S_alt.

Keep all other sections of the pre-registration file intact. The file should be well-structured markdown.