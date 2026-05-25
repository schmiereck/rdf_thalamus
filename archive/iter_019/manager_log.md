# Research Manager Log - Iteration 019

## Iteration 019 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
The temporal autocorrelation of raw pixel values at spatial positions identified
as "surprising" by the pre-trained encoder's prediction error map provides a
discriminative signal between genuine object transitions and Noisy-TV distractors
that is immune to all three known cold-start pathologies.

Formally, when overall prediction error exceeds the recruitment threshold:
1. Identify the top-K spatial positions S with highest per-position prediction
   error from the pre-trained encoder (NOT from any new/cold dimension).
2. Compute ITAG = (1/|S|) Σ_{x∈S} Corr[pixel(x,t), pixel(x,t+1)]
   over W_t=20 consecutive timesteps (lag-1 temporal autocorrelation of raw
   pixel values at each surprising position, averaged across positions).
3. Gating decision: if ITAG > τ=0.3, initiate WUP-MDL recruitment; if ITAG ≤ τ,
   reject the dimension proposal as noise.

For genuine N=3→4 transitions: ITAG > 0.5 (object pixels persist and move
smoothly, producing temporally autocorrelated pixel trajectories).
For Noisy-TV distractors: ITAG < 0.1 (noise pixels are independent each frame,
producing zero temporal autocorrelation by definition of white noise).

This metric avoids cold-start pathologies because:
(a) It operates on raw pixel values, not encoder output → no encoder cold-start.
(b) It requires no predictor → no predictor cold-start.
(c) It requires no learning during evaluation → no optimization transient.

Additionally, the spatial autocorrelation ISAG = Corr[pixel(x,t), pixel(x+1,t)]
for adjacent surprising positions provides a complementary per-frame discriminative
signal (genuine objects produce spatially smooth pixel patches; noise produces
spatially independent pixels).

**Proposed Falsification Criterion:**
The hypothesis is falsified if EITHER of the following holds across a 5-seed
experimental sweep:

C1 (Discriminative Power): Cohen's d between the ITAG score distributions for
genuine N=3→4 transitions and Noisy-TV controls is less than 1.5. This would
indicate that temporal autocorrelation of raw pixels at surprising positions
does not carry sufficient discriminative information to separate structured
from unstructured signals.

C2 (Gating Performance): The ITAG pre-filter (threshold τ=0.3) applied before
WUP-MDL recruitment fails to achieve BOTH:
  - False recruitment rate on Noisy-TV ≤ 20% (currently 100% without ITAG)
  - Genuine recruitment rate ≥ 80% (currently 100% without ITAG)
If ITAG pre-filtering rejects too many genuine objects OR accepts too many
Noisy-TV distractors, the metric is insufficient for practical gating.

**Proposed Method:**
Step-by-step experimental protocol:

1. CODE: Add ITAG computation module (src/itag.py):
   - Function compute_itag(pixel_array, surprising_positions, window=20):
     For each position x in surprising_positions, compute lag-1 temporal
     autocorrelation of pixel values over the window. Return mean across
     positions.
   - Function compute_isag(pixel_array, surprising_positions):
     For each frame, compute lag-1 spatial autocorrelation of pixel values
     at adjacent surprising positions. Return mean across frames.
   - Function identify_surprising_positions(prediction_error_map, top_k=16):
     Return indices of top-K positions by prediction error norm.

2. CODE: Modify the dimension recruitment gating logic:
   - When prediction error exceeds recruitment threshold, BEFORE initiating WUP:
     a. Compute surprising positions from pre-trained encoder's error map
     b. Collect W_t=20 timesteps of raw pixel data at those positions
     c. Compute ITAG score
     d. If ITAG > τ=0.3: proceed with WUP-MDL (Arm B) or immediate
        recruitment (Arm C)
     e. If ITAG ≤ τ: reject the dimension proposal, log the rejection

3. EXPERIMENTAL ARMS (5 seeds each, matched random seeds):

   Arm A (Baseline): WUP-MDL (W=100) with no ITAG pre-filter.
     - Current best performer: 100% genuine recruitment, 100% false recruitment
     - Serves as the comparison baseline
  
   Arm B (ITAG+MDL): ITAG pre-filter (τ=0.3, W_t=20) + WUP-MDL (W=100).
     - ITAG pre-filter blocks Noisy-TV from entering WUP
     - WUP-MDL handles cold-start predictor for genuine objects that pass ITAG
     - Expected: high genuine recruitment, low false recruitment
  
   Arm C (ITAG-only): ITAG-only gating (τ=0.3, W_t=20), no WUP.
     - If ITAG > τ, immediately recruit the dimension (skip WUP entirely)
     - Tests whether ITAG alone is sufficient, or whether WUP is still needed
     - Expected: faster recruitment, but potentially lower quality dimensions
       if cold-start predictor is problematic without WUP

4. SWEEPS per arm:
   - Transition sweep: N=3→4 (genuine 4th object introduced)
   - Control sweep: N=3 + Noisy-TV distractor (noise region added)

5. METRICS:
   - ITAG score distribution for genuine vs Noisy-TV (per-timestep, across
     the evaluation window) — for Cohen's d computation
   - ISAG score distribution (secondary analysis)
   - Genuine recruitment rate (target ≥ 80%)
   - False recruitment rate on Noisy-TV (target ≤ 20%)
   - Centroid decoding MSE
   - Test simulation loss
   - Time-to-recruitment (steps from introduction to recruitment decision)

6. ANALYSIS:
   - Compute Cohen's d between ITAG distributions → test C1
   - Compute optimal τ via ROC analysis on ITAG distributions
   - Compare Arm B vs Arm A on false recruitment rate → test C2
   - Compare Arm C vs Arm B on genuine recruitment quality → determine
     whether WUP is still needed after ITAG pre-filtering

7. FILES TO CREATE/MODIFY:
   - CREATE: src/itag.py — ITAG and ISAG metric computation
   - MODIFY: Dimension recruitment gating module — add ITAG pre-filter
   - MODIFY: Experiment runner — add Arms B and C with ITAG logic
   - MODIFY: Logging — record ITAG/ISAG scores per timestep for analysis

---

## Iteration 019 -> Planner [Strategic Guidance]

# Manager's Note — Phase 19 Strategic Guidance

## 1. CONSTRUCTION-vs-EMPIRICAL TEST: ITAG IS TRIVIALLY TRUE BY CONSTRUCTION

Before endorsing this plan, I applied the discipline mandated by my role:

> *"If the Planner gets the 'expected' result (ITAG separates genuine from Noisy-TV), will it tell us anything we did not already build into the construction?"*

**Answer: No.** This result is constructional, not empirical.

- Noisy-TV is **defined** as white noise (independent across frames). Temporal autocorrelation ≈ 0 is a **mathematical identity** of this definition, not an empirical discovery.
- Physical objects in the simulation **by construction** persist across frames with smooth motion. Temporal autocorrelation > 0 follows directly from the physics engine's continuity.

ITAG is detecting a **definitional difference** between two signal classes, not an emergent discriminative property. Cohen's d > 1.5 is virtually guaranteed — you are measuring the difference between correlated and uncorrelated signals, which is a statistical certainty.

**Instruction:** The Planner must either (a) reframe ITAG as a *verification of a definitional identity* (white noise has zero autocorrelation — confirmed), or (b) propose an experiment whose outcome is **not predictable from the construction alone**.

## 2. THE REAL SCIENTIFIC QUESTION: STRUCTURED DISTRACTORS

The genuinely non-trivial test is: **what happens when the distractor is NOT white noise?**

If the distractor is a slowly moving, colored, temporally correlated but task-irrelevant pattern (e.g., a sliding textured surface, a periodic oscillation, a structured visual distractor like a waving flag), ITAG will fail identically to all previous approaches because the distractor also has high temporal autocorrelation.

The Noisy-TV
<|SPACE|>
scenario is the **easiest possible case** for this metric. It must be supplemented with a **structured distractor** condition — a signal that is temporally autocorrelated but semantically irrelevant — before any claim of discriminative power is justified.

**Instruction:** Add at minimum one experimental arm with a **temporally correlated distractor** (e.g., a slowly drifting colored patch, a sinusoidal intensity modulation, or a physically simulated but task-irrelevant background object). If ITAG also rejects this distractor, that would be a genuinely empirical result. If it doesn't, then ITAG has the same fundamental limitation as all previous approaches — it cannot distinguish *structured but irrelevant* from *structured and relevant*.

## 3. SCOPE REDUCTION WARNING: FIFTH ITERATION THRESHOLD

Per the Research Journal (Section 3), four consecutive iterations have attempted to solve the structural gating problem (WUP-MDL → ESUG → EG-MDL → now ITAG). Each fails on a different pathology. This is the fifth attempt.

I am authorized and expected by Section 2 to enact scope reduction if this iteration also fails. The fallback is: **"fixed dimensionality with logged hypothetical growth points."** This would:
- Set dimensionality to a pre-allocated maximum (e.g., 8 dimensions)
- Disable dynamic recruitment entirely
- Log the *conditions under which recruitment would have been triggered* as observational data
- Unblock progress on Phase 13 (Dimension-Width Trade-off) and Phase 15 (Dual Control), which have been deferred for 4+ iterations

**Instruction:** The Planner must include, in the pre-registration file, an explicit **scope-reduction trigger**: if ITAG (a) achieves Cohen's d > 1.5 on Noisy-TV but (b) fails on a structured-distractor condition, OR if the Planner otherwise cannot demonstrate discriminative power on a non-trivial distractor within this iteration, then the project will fall back to fixed dimensionality with logged growth points. This trigger must be stated *before* execution begins.

---

**Summary of mandatory modifications to the plan before execution:**
1. Reframe the Noisy-TV discrimination as a verification of a definitional identity, not a discovery
2. Add a structured-distractor experimental condition (temporally correlated but task-irrelevant signal)
3. Include an explicit scope-reduction trigger in the pre-registration file
4. The Orchestrator will write `src/pre_registration.md` before execution — the Planner's sub-agents must read and strictly adhere to it

---

