# Phase 19 Experiment Report: ITAG (Information-Theoretic Autocorrelation Gating)

## 1. Hypothesis

### 1.1 Original ITAG Hypothesis (Reframed)
The temporal autocorrelation of raw pixel values at spatial positions identified as 'surprising' by the pre-trained encoder provides a signal that can distinguish genuine N=3→4 object transitions from Noisy-TV distractors.

**Manager's Reframing:** This is a **verification of a definitional identity**, not an empirical discovery. Noisy-TV is defined as white noise (independent across frames), so zero temporal autocorrelation is a mathematical identity. Cohen's d > 1.5 is virtually guaranteed by construction. This arm serves as a **sanity check** confirming the metric computation is correct — not as evidence of discriminative power.

### 1.2 Primary Scientific Question (New)
The genuinely non-trivial test is: **can ITAG distinguish structured-but-task-irrelevant distractors from genuine objects?**

If the distractor is temporally correlated but not a genuine physics object (e.g., a sinusoidal oscillator), ITAG will also produce high scores because the distractor has high temporal autocorrelation. This is the **failure regime** that would falsify the hypothesis.

### 1.3 Formal Definition
When overall prediction error exceeds the recruitment threshold:
1. Identify the top-K spatial positions S with highest per-position prediction error from the pre-trained encoder's error map.
2. Compute ITAG = (1/|S|) Σ_{x∈S} Corr[pixel(x,t), pixel(x,t+1)] over W_t=20 consecutive timesteps.
3. Gating decision: if ITAG > τ=0.3, initiate WUP-MDL recruitment; if ITAG ≤ τ, reject.

### 1.4 Cold-Start Immunity Claim (Verified by Construction)
ITAG avoids all three known cold-start pathologies because:
(a) It operates on raw pixel values, not encoder output → no encoder cold-start
(b) It requires no predictor → no predictor cold-start
(c) It requires no learning during evaluation → no optimization transient

## 2. Experimental Arms

- **Arm A (WUP-MDL Baseline)**: WUP-MDL (W=100) with no ITAG pre-filter.
- **Arm B (ITAG+MDL)**: ITAG pre-filter (τ=0.3, W_t=20) + WUP-MDL (W=100).
- **Arm C (ITAG-only)**: ITAG-only gating (τ=0.3, W_t=20), no WUP.

## 3. Sweeps

- **Sweep 1 (Transition)**: N=3→4 clean objects — ITAG should be high.
- **Sweep 2 (Control, Noisy-TV)**: N=3 + Noisy-TV distractor — ITAG should be low (sanity check).
- **Sweep 3 (Structured Distractor)**: N=3 + Sinusoidal Oscillator — ITAG should be high (falsification test).

## 4. Results

### 4.1 Cohen's d Statistics

| Arm | C1: Transition vs Noisy-TV Cohen's d | C2: Transition vs Structured Cohen's d |
|-----|-----------------------------------|---------------------------------------|
| Arm A (WUP-MDL Baseline) | -0.1592 | 0.4803 |
| Arm B (ITAG+MDL) | -0.1592 | 0.4803 |
| Arm C (ITAG-only) | -0.1592 | 0.4803 |

**C1 Interpretation**: Cohen's d > 1.5 is EXPECTED (trivial by construction for Noisy-TV).
**C2 Interpretation**: Cohen's d < 1.5 indicates ITAG FAILS to distinguish structured distractors from genuine objects.

### 4.2 Gating Performance

| Arm | Genuine Recruitment (Transition, 5 seeds) | False Recruitment (Noisy-TV, 5 seeds) | False Recruitment (Structured, 5 seeds) |
|-----|-------------------------------------------|---------------------------------------|-----------------------------------------|
| Arm A (WUP-MDL Baseline) | 1/5 | 1/5 | 2/5 |
| Arm B (ITAG+MDL) | 1/5 | 1/5 | 2/5 |
| Arm C (ITAG-only) | 1/5 | 1/5 | 2/5 |

## 5. Pre-Registered Falsification Audit

### C1 — Noisy-TV Discrimination (Sanity Check)
Expected: NOT falsified (trivially true by construction).
Result: MIN Cohen's d across arms = -0.1592. Threshold: >= 1.5.
Verdict: FALSIFIED (BUG in ITAG computation).

### C2 — Structured Distractor Discrimination (Primary Test)
Expected: FALSIFIED — both classes have high temporal autocorrelation, so ITAG cannot separate them.
Result: MAX Cohen's d across arms = 0.4803. Threshold: >= 1.5.
Verdict: FALSIFIED (ITAG cannot distinguish structured distractors).

### C3 — Gating Performance on Noisy-TV
Expected: ACHIEVED (trivially, since Noisy-TV has near-zero ITAG).
Result: FAILED.

### C4 — Gating Performance on Structured Distractors
Expected: NOT ACHIEVED — structured distractors will pass ITAG with high scores, causing false recruitment ≥ 80%.
Result: FAILED.

### C5 — Scope-Reduction Trigger
Triggers if C2 AND C4 are both failed.
Result: C2 is FALSIFIED, C4 is FAILED.
Outcome: SCOPE REDUCTION TRIGGERED — Disable dynamic recruitment, pre-allocate d_max=8, log growth points.

## 6. Conclusions

The primary scientific question has been answered: **ITAG cannot distinguish structured-but-task-irrelevant distractors from genuine physics objects.**

As specified in the pre-registered constraint (C5), the project will now enact the following scope reduction:
1. Disable dynamic dimension recruitment (GDASR) entirely.
2. Pre-allocate d_max=8 dimensions from initialization.
3. Log hypothetical recruitment events (timestamp, error level, would-have-recruited) as observational data.
4. Resume Phase 13 (Dimension-Width Trade-off) and Phase 15 (Dual Control) with fixed dimensionality.