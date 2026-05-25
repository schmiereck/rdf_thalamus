# RDF Scientific Pre-Registration — Phase 19

## 1. Hypothesis (Modified per Manager Review)

### 1.1 Original ITAG Hypothesis (Reframed)
The temporal autocorrelation of raw pixel values at spatial positions identified as "surprising" by the pre-trained encoder provides a signal that can distinguish genuine N=3→4 object transitions from Noisy-TV distractors.

**Manager's Reframing:** This is a **verification of a definitional identity**, not an empirical discovery. Noisy-TV is defined as white noise (independent across frames), so zero temporal autocorrelation is a mathematical identity. Cohen's d > 1.5 is virtually guaranteed by construction. This arm serves as a **sanity check** confirming the metric computation is correct — not as evidence of discriminative power.

### 1.2 Primary Scientific Question (New)
The genuinely non-trivial test is: **can ITAG distinguish structured-but-task-irrelevant distractors from genuine objects?**

If the distractor is temporally correlated but not a genuine physics object (e.g., a sinusoidal oscillator, a slowly drifting colored patch), ITAG will also produce high scores because the distractor has high temporal autocorrelation. This is the **failure regime** that would falsify the hypothesis.

### 1.3 Formal Definition
When overall prediction error exceeds the recruitment threshold:
1. Identify the top-K spatial positions S with highest per-position prediction error from the pre-trained encoder's error map.
2. Compute ITAG = (1/|S|) Σ_{x∈S} Corr[pixel(x,t), pixel(x,t+1)] over W_t=20 consecutive timesteps.
3. Gating decision: if ITAG > τ=0.3, initiate WUP-MDL recruitment; if ITAG ≤ τ, reject.

**Predictions:**
- Genuine N=3→4 transitions: ITAG > 0.5 (verification of definitional identity)
- Noisy-TV distractors: ITAG < 0.1 (verification of definitional identity)
- Structured distractors (sinusoidal oscillator): ITAG > 0.5 (CONJECTURE — this is the empirical test)

If ITAG > 0.5 for structured distractors, they will pass the ITAG gate, causing false recruitment identical to WUP-MDL without ITAG. This would falsify the hypothesis that ITAG provides discriminative power beyond the trivial Noisy-TV case.

### 1.4 Cold-Start Immunity Claim (Verified by Construction)
ITAG avoids all three known cold-start pathologies because:
(a) It operates on raw pixel values, not encoder output → no encoder cold-start
(b) It requires no predictor → no predictor cold-start
(c) It requires no learning during evaluation → no optimization transient

This claim is verified by construction and does not require empirical validation.

## 2. Falsification Criteria

### C1 — Noisy-TV Discrimination (Sanity Check)
Cohen's d between ITAG score distributions for genuine transitions and Noisy-TV controls is less than 1.5. **Expected: NOT falsified** (trivially true by construction). If falsified, the ITAG computation has a bug.

### C2 — Structured Distractor Discrimination (Primary Test)
Cohen's d between ITAG score distributions for genuine transitions and **structured distractor** controls is less than 1.5. This would indicate that temporal autocorrelation of raw pixels does NOT distinguish physics-based objects from non-physics temporally correlated patterns. **Expected: FALSIFIED** — both classes have high temporal autocorrelation, so ITAG cannot separate them.

### C3 — Gating Performance on Noisy-TV
The ITAG pre-filter (τ=0.3) applied before WUP-MDL recruitment achieves:
  - False recruitment rate on Noisy-TV ≤ 20%
  - Genuine recruitment rate ≥ 80%
**Expected: ACHIEVED** (trivially, since Noisy-TV has near-zero ITAG).

### C4 — Gating Performance on Structured Distractors
The ITAG pre-filter (τ=0.3) applied before WUP-MDL recruitment achieves:
  - False recruitment rate on structured distractors ≤ 20%
  - Genuine recruitment rate ≥ 80%
**Expected: NOT ACHIEVED** — structured distractors will pass ITAG with high scores, causing false recruitment ≥ 80%.

### C5 — Scope-Reduction Trigger (Mandatory)
If criteria C2 AND C4 are both failed (ITAG cannot distinguish structured distractors from genuine objects), the project will fall back to **fixed dimensionality with logged growth points** as specified in the Research Manager's guidance. Specifically:
  - Set dimensionality to a pre-allocated maximum (d_max=8)
  - Disable dynamic recruitment entirely
  - Log the conditions under which recruitment would have been triggered as observational data
  - Unblock progress on Phase 13 (Dimension-Width Trade-off) and Phase 15 (Dual Control)

## 3. Structured Distractor Definition

The structured distractor is a **Sinusoidal Oscillator** entity:
- Position: x(t) = center + amplitude × sin(ω×t + φ)
  - center ∈ [32, 96] (randomly initialized)
  - amplitude ∈ [5, 15] (randomly initialized)
  - ω ∈ [0.02, 0.05] (slow oscillation period ~125-314 steps)
  - φ ∈ [0, 2π] (randomly initialized)
- Color: persistent, randomly initialized from [0.3, 1.0] per channel
- Radius: randomly initialized from [3.0, 8.0]
- **No collision physics**: passes through other objects (non-interacting)
- Renders as a soft-edged circle (same rendering as other objects)

This entity has:
- High temporal autocorrelation in position (smooth sinusoidal motion)
- High temporal autocorrelation in color (persistent)
- High spatial autocorrelation (smooth edges)
- BUT: does not follow Newtonian collision physics
- BUT: is not semantically relevant to the agent's physical interaction task

## 4. Proposed Method

### 4.1 Code Changes
- CREATE: src/itag.py — ITAG and ISAG metric computation functions
- MODIFY: src/environment.py — add `structured_distractor` mode with Sinusoidal Oscillator
- CREATE: src/run_phase19_experiments.py — full experiment runner

### 4.2 Experimental Arms (5 seeds each, matched seeds = [42, 123, 456, 789, 999])

Arm A (Baseline): WUP-MDL (W=100) with no ITAG pre-filter.
  - Serves as the comparison baseline (same as Phase 18 Arm P)

Arm B (ITAG+MDL): ITAG pre-filter (τ=0.3, W_t=20) + WUP-MDL (W=100).
  - ITAG pre-filter blocks low-autocorrelation signals from entering WUP
  - WUP-MDL handles cold-start predictor for signals that pass ITAG

Arm C (ITAG-only): ITAG-only gating (τ=0.3, W_t=20), no WUP.
  - If ITAG > τ, immediately recruit (skip WUP)
  - Tests whether ITAG alone is sufficient

### 4.3 Sweeps per Arm (3 conditions × 5 seeds = 15 per arm, 45 total)

Sweep 1 (Transition): N=3→4 (genuine 4th physics object) — ITAG should be high
Sweep 2 (Noisy-TV Control): N=3 + Noisy-TV — ITAG should be low (sanity check)
Sweep 3 (Structured Distractor Control): N=3 + Sinusoidal Oscillator — ITAG should be high (falsification test)

### 4.4 Metrics
- ITAG score distribution per timestep for each condition (for Cohen's d)
- ISAG score distribution (secondary analysis)
- Genuine recruitment rate (target ≥ 80%)
- False recruitment rate per distractor type (target ≤ 20%)
- Centroid decoding MSE
- Test simulation loss
- Time-to-recruitment

### 4.5 Analysis
- Compute Cohen's d between ITAG distributions: genuine vs Noisy-TV (C1), genuine vs structured (C2)
- ROC analysis on ITAG distributions
- Compare Arm B vs Arm A on false recruitment rates
- Evaluate scope-reduction trigger (C5)

## 5. Scope-Reduction Trigger (Pre-Registered)

**IF** C2 is falsified (Cohen's d < 1.5 between genuine and structured distractor ITAG distributions) **AND** C4 is failed (false recruitment rate on structured distractors > 20%), **THEN**:

The project will enact the following scope reduction:
1. Disable dynamic dimension recruitment (GDASR) entirely
2. Pre-allocate d_max=8 dimensions from initialization
3. Log hypothetical recruitment events (timestamp, error level, would-have-recruited) as observational data
4. Resume Phase 13 (Dimension-Width Trade-off) and Phase 15 (Dual Control) with fixed dimensionality

This scope reduction is pre-committed and will be enacted regardless of any post-hoc rationalization about why ITAG "should have worked" on structured distractors.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
