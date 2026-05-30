# Iter_030 ARM 2 — M2 Decisive Test Analysis

This analysis tests whether D1 (batch-level temporal contrastive NT-Xent) or D2 (variance-ramped SFA) achieves mean ΔR²_color ≥ 0.30 with variance-stability (lower 95% CI ≥ 0.18) on a 30-seed union bank.

**Dual collapse criterion:** collapsed = collapsed_eval OR collapsed_train
**Contrastive implementation:** Existing batch-level NT-Xent in model.forward (same batch index as positive pair, cross-batch as negatives). Object-level temporal matching NOT implemented — uses weaker batch-level signal.
**D2 ramp:** sfa_weight linearly 0 → 5.0 over steps 1–4000, then held at 5.0 for steps 4001–8000.

---

## Per-Arm Summary (all 30 seeds)

### D1 (Contrastive, batch-level)
- N seeds: 30
- Collapse rate (dual): 0.00 (0/30)
- Collapse rate (eval-only): 0.00 (0/30)
- Collapse rate (train-only): 0.00 (0/30)
- Mean ΔR²_color (all): 0.1149 ± 0.3029
- Mean centroid MSE: 159.61
- Mean abs corr: 0.183
- Parameter count: 135608

### D2 (SFA, variance-ramped)
- N seeds: 30
- Collapse rate (dual): 0.00 (0/30)
- Collapse rate (eval-only): 0.00 (0/30)
- Collapse rate (train-only): 0.00 (0/30)
- Mean ΔR²_color (all): 0.1894 ± 0.3218
- Mean centroid MSE: 165.70
- Mean abs corr: 0.143
- Parameter count: 135608

## Non-Collapsed Seed Analysis (preregistered gate)

Falsification criterion per arm:
- **FALSIFIED** if mean ΔR²_color < 0.30 OR lower 95% CI < 0.18 on non-collapsed seeds.
- If n_non-collapsed < 5, flagged as **underpowered** (cannot compute stable CI).

### D1 (Contrastive, batch-level) — non-collapsed
- N non-collapsed: 30 / 30
- Mean ΔR²_color: 0.1149 ± 0.3029
- SE: 0.0553
- 95% CI lower bound: 0.0065
- **VERDICT: FALSIFIED** — mean < 0.30 AND lower CI < 0.18.

### D2 (SFA, variance-ramped) — non-collapsed
- N non-collapsed: 30 / 30
- Mean ΔR²_color: 0.1894 ± 0.3218
- SE: 0.0587
- 95% CI lower bound: 0.0742
- **VERDICT: FALSIFIED** — mean < 0.30 AND lower CI < 0.18.

## Hard-Seed Table (Seeds 53, 71)

| seed | arm | collapsed | ΔR²_color | centroid_mse | mean_abs_corr | per_dim_std |
|------|-----|-----------|-----------|--------------|---------------|-------------|
| 53 | D1 (Contrastive, batch-level) | N | 0.0639 | 149.98 | 0.199 | [0.8759780526161194, 0.8004746437072754, 1.2054208517074585] |
| 53 | D2 (SFA, variance-ramped) | N | 0.0306 | 138.78 | 0.105 | [0.9768287539482117, 1.2222081422805786, 0.9918580651283264] |
| 71 | D1 (Contrastive, batch-level) | N | -0.1677 | 64.33 | 0.179 | [0.9921647310256958, 1.41298246383667, 1.0400313138961792] |
| 71 | D2 (SFA, variance-ramped) | N | -0.0680 | 67.46 | 0.042 | [1.2802647352218628, 1.2101740837097168, 1.2162635326385498] |

## Seed-Bank Breakdown

| arm | bank | n | mean_ΔR²_color | collapse_rate | mean_centroid_mse |
|-----|------|---|----------------|---------------|-------------------|
| D1 (Contrastive, batch-level) | original (10) | 10 | 0.0188 | 0.00 | 161.02 |
| D1 (Contrastive, batch-level) | fresh (10) | 10 | 0.2241 | 0.00 | 164.10 |
| D1 (Contrastive, batch-level) | new (10) | 10 | 0.1018 | 0.00 | 153.72 |
| D2 (SFA, variance-ramped) | original (10) | 10 | 0.1630 | 0.00 | 174.08 |
| D2 (SFA, variance-ramped) | fresh (10) | 10 | 0.2585 | 0.00 | 163.60 |
| D2 (SFA, variance-ramped) | new (10) | 10 | 0.1466 | 0.00 | 159.43 |

## Pre-Registered Outcome Classification

**Hypothesis:** Temporal identity contrastive binding (D1) or variance-ramped SFA (D2) achieves mean ΔR²_color ≥ 0.30 on 30-seed union with lower 95% CI ≥ 0.18.

**Both D1 and D2 are FALSIFIED.**

Mean ΔR²_color < 0.30 OR lower 95% CI < 0.18 on all sufficiently-powered arms.
This implies ΔR²_color ≥ 0.30 is unachievable by any decoder-free objective tested to date. Per the pre-registration, the project accepts the current representation quality per ARM 1 verdict.

