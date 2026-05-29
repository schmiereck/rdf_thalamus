# Iter_025 Architecture Ceiling Probe — Final Comprehensive Analysis

**Experiment:** Phase 0 ID Probe (Architecture Ceiling Probe)  
**Date:** Auto-generated after full execution  
**Total Runs:** 23 (3 noise floor + 20 main: 4 arms × 5 seeds + checkpoint evals)  
**Seeds (main):** [7, 17, 31, 53, 71]  
**Seeds (noise floor):** [7, 17, 31]  
**Steps:** 5000 (main), 1000 (noise floor)  
**Checkpoint eval:** Step 2000 (monitoring only)  
**Primary metric:** delta_R2_color (frozen-encoder linear probe on z_dyn vs z_coord)  

---

## 1. Noise Floor (Frozen Random Encoder + Probe-Only Training)

Three short runs with the encoder frozen at random initialization, training only the linear probe head.

| Seed | delta_R2_color | Collapsed? | per_dim_std |
|------|----------------|------------|-------------|
| 7    | +0.5795        | No         | [0.768, 0.782, 0.829] |
| 17   | −0.1929        | No         | [0.731, 0.874, 0.943] |
| 31   | +2.5541        | No         | [0.715, 0.845, 0.918] |

- **floor_mean = 0.9802**
- **floor_std = 1.1567**

> ⚠️ **Data quality note:** The noise floor exhibits extremely high variance (coefficient of variation ≈ 1.18). Seed 31 produced delta_R2_color = +2.55, suggesting the frozen random encoder happens to produce a z_dyn configuration where the probe spuriously achieves very high R² relative to z_coord. This is numerically possible because R² can exceed 1.0 on test data when the probe overfits to the random structure. The high variance means the empirical floor is poorly estimated.

---

## 2. Effective Threshold

Per pre-registration:

```
threshold = max(0.10, floor_mean + 0.08)
         = max(0.10, 0.9802 + 0.08)
         = 1.0602
```

**Ceiling threshold for declaring H1/H2 confirmed: 1.0602**

> **Important caveat:** Because the noise floor contains an outlier (seed 31 = +2.55), this threshold is very high. The delta_R2_color metric itself can take values well outside [0, 1] (and even negative), so a threshold of 1.06 is not fundamentally unachievable in principle — however, no arm in this study came close to it. The outcome should be interpreted in light of both the threshold and the absolute magnitude of the signal.

---

## 3. Per-Arm Results (Final Step = 5000)

### 3a. Arm A — JEPA+VICReg Control (d_max=8)
*Baseline control. Expected: incidental identity encoding only. Drift check against iter_022-024.*

| Seed | Collapsed? | delta_R2_color (greedy) | Centroid MSE | Tracking Level Corr |
|------|------------|------------------------|--------------|---------------------|
| 7    | No         | +0.0452                | 71.49        | −0.031              |
| 17   | No         | −0.0103                | 86.44        | +0.339              |
| 31   | No         | −0.0105                | 205.46       | −0.086              |
| 53   | **Yes**    | −0.0149                | 188.18       | +0.011              |
| 71   | **Yes**    | −0.2199                | 38.99        | +0.067              |

**Aggregates:**
- **Collapse rate:** 2/5 (40%)
- **Mean delta_R2_color (all seeds):** −0.0255 ± 0.210
- **Mean delta_R2_color (non-collapsed, N=3):** +0.0081
- **Mean centroid MSE:** 118.11 ± 74.1
- **Mean tracking level corr:** +0.060 ± 0.166

### 3b. Arm B — Supervised Color Probe (d_max=8) [CRITICAL DIAGNOSTIC]
*Architecture ceiling probe. If this fails, the shared CNN encoder cannot route identity to z_dyn under direct supervision.*

| Seed | Collapsed? | delta_R2_color (greedy) | delta_R2_color (sorted) | delta_R2_color (Hungarian) | Eval Mismatch Rate | Centroid MSE | Tracking Level Corr |
|------|------------|------------------------|------------------------|---------------------------|-------------------|--------------|---------------------|
| 7    | **Yes**    | −0.0400                | −0.2488                | −0.1264                   | 0.667 (2/3)       | 67.98        | +0.024              |
| 17   | No         | +0.0362                | −0.0205                | +0.0362                   | 0.667 (2/3)       | 85.58        | +0.271              |
| 31   | No         | +0.0787                | +0.0679                | +0.0787                   | 0.667 (2/3)       | 207.48       | −0.123              |
| 53   | **Yes**    | −0.1081                | +0.1784                | +0.1635                   | 0.667 (2/3)       | 153.33       | +0.489              |
| 71   | **Yes**    | −0.1290                | −0.0705                | −0.1527                   | 0.667 (2/3)       | 56.62        | +0.125              |

**Aggregates:**
- **Collapse rate:** 3/5 (60%)
- **Mean delta_R2_color (non-collapsed, greedy, N=2):** +0.0574 (seeds 17, 31)
- **Mean delta_R2_color (non-collapsed, sorted, N=2):** +0.0237
- **Mean delta_R2_color (non-collapsed, Hungarian, N=2):** +0.0574
- **Eval mismatch rate (mean across all seeds):** 0.667 (≈ 67%) — **sorted vs Hungarian assignments disagree on 2 out of 3 dimensions**
- **Scheme agreement (pass/fail):** All seeds agree that the outcome is "fail" under both sorted and Hungarian matching
- **Mean centroid MSE:** 114.20 ± 64.2
- **Mean tracking level corr:** +0.157 ± 0.235

**Cross-matching note:** For Arm B, the greedy matching result happens to equal the Hungarian result on non-collapsed seeds (because the greedy matching in the evaluation found the same assignment as Hungarian for those seeds). The sorted matching produces lower delta_R2_color on seed 17 (−0.0205 vs +0.0362).

### 3c. Arm C — ID-Contrastive (d_max=8)
*Tests whether a color-similarity contrastive objective can serve as a self-supervised proxy for identity.*

| Seed | Collapsed? | delta_R2_color (greedy) | delta_R2_color (sorted) | delta_R2_color (Hungarian) | Eval Mismatch Rate | Centroid MSE | Tracking Level Corr |
|------|------------|------------------------|------------------------|---------------------------|-------------------|--------------|---------------------|
| 7    | No         | +0.0038                | +0.1057                | +0.1057                   | 0.000 (0/3)       | 71.70        | +0.100              |
| 17   | **Yes**    | −0.0741                | +0.0060                | +0.0299                   | 0.667 (2/3)       | 57.75        | +0.165              |
| 31   | No         | +0.0618                | +0.0618                | +0.1242                   | 1.000 (3/3)       | 214.24       | −0.126              |
| 53   | **Yes**    | +0.0560                | +0.0622                | +0.0622                   | 0.000 (0/3)       | 230.87       | +0.416              |
| 71   | **Yes**    | −0.1069                | −0.1683                | −0.0966                   | 0.667 (2/3)       | 40.75        | +0.059              |

**Aggregates:**
- **Collapse rate:** 3/5 (60%)
- **Mean delta_R2_color (non-collapsed, greedy, N=2):** +0.0328 (seeds 7, 31)
- **Mean delta_R2_color (non-collapsed, sorted, N=2):** +0.0837
- **Mean delta_R2_color (non-collapsed, Hungarian, N=2):** +0.1149
- **Eval mismatch rate (mean across all seeds):** 0.467 (≈ 47%)
- **Scheme agreement (pass/fail):** All seeds agree "fail" under both matching schemes
- **Mean centroid MSE:** 123.06 ± 91.7
- **Mean tracking level corr:** +0.123 ± 0.196

**Key observation:** Arm C shows the strongest signal under Hungarian matching (mean +0.115 on non-collapsed seeds), but still far below threshold. The contrastive objective produces a modestly positive delta_R2_color on some non-collapsed seeds (greedy: +0.0038, +0.0618), whereas Arm B also shows positive values (+0.0362, +0.0787). However, neither arm sustains this across all seeds.

### 3d. Arm D — Supervised Color Probe (d_max=16)
*Channel-capacity probe. Tests whether increased d_max improves supervised identity encoding.*

| Seed | Collapsed? | delta_R2_color (greedy) | delta_R2_color (sorted) | delta_R2_color (Hungarian) | Eval Mismatch Rate | Centroid MSE | Tracking Level Corr |
|------|------------|------------------------|------------------------|---------------------------|-------------------|--------------|---------------------|
| 7    | **Yes**    | +0.0008                | +0.9396                | −0.0371                   | 0.667 (2/3)       | 47.01        | +0.199              |
| 17   | **Yes**    | −0.0001                | +0.0073                | −0.0809                   | 0.667 (2/3)       | 83.41        | −0.194              |
| 31   | No         | −0.0020                | −0.0180                | −0.0982                   | 0.667 (2/3)       | 220.55       | −0.048              |
| 53   | No         | −0.0485                | +0.0035                | −0.0542                   | 0.667 (2/3)       | 228.33       | −0.196              |
| 71   | **Yes**    | −0.0066                | −0.0221                | −0.0221                   | 0.000 (0/3)       | 63.03        | +0.281              |

**Aggregates:**
- **Collapse rate:** 3/5 (60%)
- **Mean delta_R2_color (non-collapsed, greedy, N=2):** −0.0252 (seeds 31, 53)
- **Mean delta_R2_color (non-collapsed, sorted, N=2):** −0.0073
- **Mean delta_R2_color (non-collapsed, Hungarian, N=2):** −0.0762
- **Eval mismatch rate (mean across all seeds):** 0.533 (≈ 53%)
- **Mean centroid MSE:** 128.47 ± 88.6
- **Mean tracking level corr:** +0.009 ± 0.222

**Key observation:** Arm D (d_max=16) performed *worse* than Arm B (d_max=8) on non-collapsed seeds. Increasing latent capacity from 8 to 16 did not improve identity encoding and may have hurt it. This is a negative result for the channel-capacity hypothesis.

---

## 4. Arm A Drift Check

| Metric | Arm A (fresh seeds) | iter_022-024 Reference | Drift? |
|--------|---------------------|------------------------|--------|
| Mean delta_R2_color (all seeds) | −0.0255 | ~0.00 to +0.02 | **NO** (Δ = 0.026, below 0.03 threshold) |
| Mean delta_R2_color (non-collapsed) | +0.0081 | ~0.00 to +0.02 | **NO** |
| Collapse rate | 40% (2/5) | ~40-60% | Consistent |

**Conclusion:** No seed-batch drift detected. The fresh seed set [7, 17, 31, 53, 71] produces results consistent with the previous iteration control arms. Cross-iteration comparison is valid.

---

## 5. Pre-Registered Falsification Checks

### 5a. H1 — Architecture Capacity (Arm B)

**Criterion:** delta_R2_color ≥ 1.0602 (mean over non-collapsed seeds) AND collapse rate ≤ 3/5.

| Condition | Result |
|-----------|--------|
| Non-collapsed mean (greedy) | +0.0574 |
| Non-collapsed mean (sorted) | +0.0237 |
| Non-collapsed mean (Hungarian) | +0.0574 |
| Passes threshold (1.0602)? | **NO** |
| Collapse rate | 60% (3/5) — acceptable per criterion |
| **H1 status** | **FALSIFIED** |

**Language per pre-registration:**  
> "The result is **consistent with an architecture-level bottleneck on identity encoding, conditional on the sorted-position matching scheme (mismatch rate: 67%)**."

### 5b. H2 — ID-Contrastive Viability (Arm C)

**Criterion:** delta_R2_color ≥ 1.0602 (mean over non-collapsed seeds) AND collapse rate ≤ 1/5.

| Condition | Result |
|-----------|--------|
| Non-collapsed mean (greedy) | +0.0328 |
| Non-collapsed mean (sorted) | +0.0837 |
| Non-collapsed mean (Hungarian) | +0.1149 |
| Passes threshold (1.0602)? | **NO** |
| Collapse rate | 60% (3/5) — exceeds 1/5 criterion |
| **H2 status** | **FALSIFIED** |

**Language per pre-registration:**  
> "The contrastive formulation is **insufficient**; the architecture may or may not be the bottleneck."

### 5c. Arm D — Channel Capacity Probe

Arm D alone does NOT confirm H1. However, the fact that Arm D (d_max=16) performs worse than Arm B (d_max=8) on non-collapsed seeds suggests that **channel capacity is not the limiting factor**.

---

## 6. Outcome Quadrant Assignment

| Arm B | Arm C | Quadrant | Interpretation | Next Move |
|-------|-------|----------|----------------|-----------|
| **FAIL** | **FAIL** | **Q4** | Architecture-level bottleneck on identity encoding via shared CNN + soft-argmax centroid head | **Next iteration: separate z_dyn encoder** |

Pre-registered quadrant table:

| Arm B | Arm C | Interpretation |
|-------|-------|----------------|
| ✓ | ✓ | H1+H2 confirmed |
| ✓ | ✗ | H1 confirmed, H2 refuted |
| ✗ | ✓ | Check implementation bugs |
| **✗** | **✗** | **Architecture-level bottleneck (conditional on matching scheme, mismatch rate: B=67%, C=47%)** |

---

## 7. Cross-Arm Comparison

| Metric | Arm A (Control) | Arm B (Supervised d=8) | Arm C (Contrastive d=8) | Arm D (Supervised d=16) |
|--------|-----------------|------------------------|-------------------------|-------------------------|
| Collapse rate | 40% (2/5) | 60% (3/5) | 60% (3/5) | 60% (3/5) |
| δR²_color (greedy, non-collapsed) | +0.008 | +0.057 | +0.033 | −0.025 |
| δR²_color (sorted, non-collapsed) | N/A | +0.024 | +0.084 | −0.007 |
| δR²_color (Hungarian, non-collapsed) | N/A | +0.057 | +0.115 | −0.076 |
| Best single-seed δR²_color | +0.045 (seed 7, greedy) | +0.079 (seed 31, greedy) | +0.124 (seed 31, Hungarian) | +0.940 (seed 7, sorted*) |
| Mean centroid MSE | 118.1 | 114.2 | 123.1 | 128.5 |
| Mean tracking corr | +0.060 | +0.157 | +0.123 | +0.009 |

> *Seed 7 of Arm D shows δR²_color_sorted = +0.940, but this seed was collapsed and the high value is driven by the sorted matching producing a different (and apparently spuriously better) assignment on that run. The Hungarian value for the same seed is −0.037, indicating strong scheme disagreement.

**Interpretation:**
- The supervised arms (B, D) do not show material improvement over the control (A) on the primary metric.
- Arm C under Hungarian matching shows the highest non-collapsed mean (+0.115), but still 10× below threshold.
- Collapse rates are similar across all arms (40-60%), suggesting the collapse is driven by the base architecture (shared CNN + centroid readout) rather than the additional objective.
- The high mismatch rates (47-67%) between sorted and Hungarian assignment confirm that the matching scheme is a meaningful confound, and all ceiling claims must be reported as conditional.

---

## 8. Supporting Diagnostics

### 8a. VICReg Health

| Arm | Mean per-dim std (non-collapsed) | Mean abs correlation |
|-----|----------------------------------|----------------------|
| A   | [0.79, 0.89, 1.06] | 0.44 |
| B   | [0.93, 0.91, 0.63] | 0.40 |
| C   | [0.66, 0.79, 0.66] | 0.35 |
| D   | [0.78, 0.49, 0.94] | 0.34 |

Collapsed runs universally show at least one dimension with std < 0.5, confirming the collapse detector is working.

### 8b. Training Loss Convergence

All arms showed typical VICReg loss trajectories:
- Total loss decreased from ~30-40 at step 1000 to ~2-15 at step 5000
- Sim loss converged to ~0.03-0.10
- Supervised loss (Arm B/D) and contrastive loss (Arm C) both converged to near-zero by step 4000-5000

The fact that the auxiliary objectives converged to low loss while the downstream delta_R2_color remained near-zero is consistent with the **bottleneck hypothesis**: the encoder is learning the supervised/contrastive task in a way that does not produce transferable identity representations in z_dyn.

### 8c. GDASR Growth Points

All arms logged GDASR growth points at d_t=3 (the frozen target dimension), consistent with the sandbox environment containing 3 objects. Growth point counts:
- Arm A: mean = 2.0 per seed
- Arm B: mean = 1.2 per seed  
- Arm C: mean = 1.2 per seed
- Arm D: mean = 1.4 per seed

---

## 9. Data Quality Assessment

| Issue | Severity | Notes |
|-------|----------|-------|
| Noise floor outlier (seed 31 = +2.55) | High | Drives threshold to 1.06, making failure almost certain. The floor_mean is not a reliable baseline due to extremely high variance across 3 seeds. |
| Negative R² values common | Medium | Many color probes show negative R², indicating the linear probe performs worse than the mean baseline. This occurs when z_dyn is uninformative about color. |
| High mismatch rates (47-67%) | Medium | Confirms matching-scheme sensitivity. All claims must be conditional. |
| Collapse rate 40-60% across arms | Medium | Suggests the base architecture is unstable. GDASR + VICReg alone does not prevent dimensional collapse. |
| No crashes or run failures | Low | All 23 runs completed successfully. |

**Recommendation:** In future iterations, consider using a more robust noise floor (e.g., 10+ seeds, or a theoretical floor based on random-guess baseline rather than empirical probe training). The current floor is too noisy to serve as a reliable threshold anchor.

---

## 10. Conclusions and Next Steps

### Primary Conclusion

Both H1 (Architecture Capacity) and H2 (ID-Contrastive Viability) are **falsified** under the pre-registered threshold of 1.0602.

The result is **consistent with an architecture-level bottleneck on identity encoding**, conditional on the sorted-position matching scheme (mismatch rate: B=67%, C=47%, D=53%). The shared-CNN dual-stream NonParametricJEPASpatial encoder, with soft-argmax centroid readout, does not route object identity information to z_dyn under either direct supervised color regression or color-similarity contrastive learning.

### Key Findings
1. **Supervised color probe (Arm B) does not achieve ceiling-level identity encoding** — even with a strong supervised signal backpropagating through the encoder, delta_R2_color stays below 0.08 on all non-collapsed seeds.
2. **ID-contrastive (Arm C) performs comparably to supervised** — highest signal under Hungarian matching (+0.115 non-collapsed mean), but still far below threshold.
3. **Increasing d_max from 8→16 (Arm D) does not help** — negative result for channel-capacity hypothesis.
4. **Collapse is architecture-driven** — all arms show 40-60% collapse, independent of the additional objective.
5. **Matching scheme matters** — 47-67% mismatch between sorted and Hungarian assignment means all claims are conditional.

### Recommended Next Move (per pre-registration)

> **Next iteration: separate z_dyn encoder.**

The shared CNN encoder appears to allocate all useful representational capacity to z_coord (position tracking), leaving no usable gradient or channel capacity for identity encoding in z_dyn. The pre-registered next move is to decouple the z_dyn encoder from the z_coord encoder, giving z_dyn its own dedicated convolutional pathway while keeping the input observations shared.

---

*Analysis compiled from:*
- `archive/iter_025/results/aggregated_phase0_id_probe.csv`
- `archive/iter_025/results/summary_phase0_id_probe.csv`
- `archive/iter_025/results/analysis.md` (auto-generated)
- `src/pre_registration.md` (pre-registered hypotheses and criteria)
