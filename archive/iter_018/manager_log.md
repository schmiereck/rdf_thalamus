# Research Manager Log - Iteration 018

## Iteration 018 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
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

**Proposed Falsification Criterion:**
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

These three criteria jointly require EG-MDL to solve BOTH the recruitment
problem (which ESUG failed at) and the distractor-rejection problem (which
WUP-MDL failed at). Improving one at the expense of the other is insufficient.

**Proposed Method:**
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

3. Implement Arm T (EG-MDL-Spatial, W=100, θ=0.90, η=0.50):
   a. Same as Arm S, plus an additional spatial entropy gate.
   b. Compute spatial entropy H = -Σ p_i log(p_i) where p_i are the
      soft-argmax attention weights for the proposed dimension.
   c. Normalize: H_norm = H / H_max where H_max = log(N_spatial).
   d. Gate requires H_norm < 0.50 (spatially concentrated attention).
   e. This tests whether spatial concentration provides a complementary
      distractor-rejection signal beyond prediction-trend alone.

4. Run a transition sweep (5 matched seeds: 42, 123, 456, 789, 1337) with
   N=3→4 object introduction. Measure: recruitment rate, centroid decoding
   MSE, test simulation loss.

5. Run a Noisy-TV control sweep (same 5 seeds) replacing the 4th object
   with a localized Noisy-TV pixel distractor. Measure: false recruitment
   rate, centroid MSE (should be high/meaningless for false recruitments).

6. Compare Arms P, S, T using Welch's t-test on false recruitment rate and
   centroid MSE. Report mean ± std for all metrics.

Files to modify:
- src/thalamus.py: Add prediction-trend gate logic, per-step error buffering
  during WUP, improvement ratio computation, and composite gate evaluation.
- New configuration entries for Arm S (θ=0.90) and Arm T (θ=0.90, η=0.50).
- Experimental runner script to execute both sweeps and collect metrics.

---

## Iteration 018 -> Planner [Strategic Guidance]

# Manager's Note — Phase 18 Strategic Guidance

## 1. Parameter-Tuning Hygiene: Justify θ=0.90 Independently

The prediction-trend threshold θ=0.90 is the critical hyperparameter of this entire proposal. If this value was derived from inspecting Phase 17 error trajectories post-hoc, it constitutes data leakage and the result will be constructional. The Planner must:

- **State the independent justification** for θ=0.90 *before* running experiments. Why 10% improvement rather than 5% or 20%? What property of the 1D physics environment or the predictor architecture determines this scale?
- **Commit to at least one alternative threshold** (e.g., θ=0.85 or θ=0.95) as a robustness arm. If the result is θ-sensitive, report it honestly as a negative finding rather than cherry-picking the best threshold.

If no independent justification exists, the Planner should run a small pilot (1-2 seeds) to establish the natural range of ρ for genuine objects vs. Noisy-TV, *then* pre-register the chosen θ with that evidence cited. This prevents silent post-hoc fitting.

## 2. The Prediction-Trend Gate Is Nearly Definitional — Frame It Honestly

The claim "a genuine object produces ρ << 1.0 while Noisy-TV produces ρ ≈ 1.0" is *almost* a tautology: we are defining "genuine object" as "a stimulus with learnable temporal structure" and then verifying that our predictor learns it. The Planner must not present this as a discovery. The genuinely empirical question is more narrow and should be framed as such:

> *"Under the specific parametric predictor used (linear projection + GRU, finite capacity, W=100 steps), does the prediction-trend ratio ρ reliably separate physical objects from random noise? The answer is not predictable from the construction alone because (a) the predictor has limited capacity, (b) some physical dynamics may not be learnable within 100 steps, and (c) some non-physical signals may exhibit partial temporal regularity."*

This framing makes the result genuinely falsifiable. The Planner must write this framing into the pre-registration file.

## 3. Arm T Spatial Entropy Gate Risks Re-Introducing Encoder Cold-Start

Phase 17 established that newly initialized encoder projections produce chaotic spatial representations (λ ~ 1.0–1.5). The soft-argmax attention weights of a cold-started dimension are therefore likely diffuse (high H_norm), regardless of whether the underlying stimulus is a genuine object or noise.

The Planner must specify **at which timestep** the spatial entropy H_norm is computed for Arm T. If computed during or before WUP, it will likely reject genuine objects — reproducing the ESUG failure mode under a different metric name. If computed at the *end* of WUP (after 100 steps of gradient alignment), the encoder may have localized sufficiently, but this is an empirical question that Phase 17's λ data does not directly answer.

**Required action:** The Planner must either (a) compute H_norm at the *end* of the WUP period and justify why 100 steps is sufficient for spatial localization of genuine objects, or (b) defer Arm T to a future iteration and focus Phase 18 on the cleaner Arm S comparison. I recommend (b) if the Planner cannot provide a principled argument for (a).

---

**Pre-registration reminder:** The Orchestrator will write `src/pre_registration.md` from the Planner's finalized hypothesis and falsification criteria. Sub-agents must read and strictly adhere to it during execution. Ensure the threshold justification (Point 1) and honest framing (Point 2) are included in that file.

---

