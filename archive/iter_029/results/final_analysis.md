# Iter_029 SFA+VICReg on Separate Backbone — Analysis
**Dual collapse criterion:** collapsed = collapsed_eval OR collapsed_train
**Arms:** A (VICReg-only control), B (SFA+VICReg, sfa=5.0), C (SFA+VICReg, sfa=1.0)
**Seed bank:** Union of original (10) + fresh (10) = 20 seeds
**Gate thresholds:** F1: ΔR²_color ≥ 0.30 (Arm B, non-collapsed), F2: collapse rate ≤ 10% (Arm B), F3: centroid MSE within 1σ of Arm A

---
## Per-Arm Summary
### A (VICReg-only control)
- N seeds: 1
- Collapse rate (dual): 1.00 (1/1)
- Collapse rate (eval-only): 1.00 (1/1)
- Collapse rate (train-only): 0.00 (0/1)
- **Disqualified seeds (loss > 50):** 1
- Mean ΔR²_color: 0.0000 ± nan
- Mean centroid MSE: nan
- Mean abs corr: 1.000
- Parameter count: 135608

### B (SFA+VICReg, sfa=5.0)
- N seeds: 1
- Collapse rate (dual): 1.00 (1/1)
- Collapse rate (eval-only): 1.00 (1/1)
- Collapse rate (train-only): 0.00 (0/1)
- **Disqualified seeds (loss > 50):** 1
- Mean ΔR²_color: 0.0000 ± nan
- Mean centroid MSE: nan
- Mean abs corr: 1.000
- Mean final SFA loss: 0.0000
- Parameter count: 135608

### C (SFA+VICReg, sfa=1.0)
- N seeds: 1
- Collapse rate (dual): 1.00 (1/1)
- Collapse rate (eval-only): 1.00 (1/1)
- Collapse rate (train-only): 0.00 (0/1)
- **Disqualified seeds (loss > 50):** 1
- Mean ΔR²_color: 0.0000 ± nan
- Mean centroid MSE: nan
- Mean abs corr: 1.000
- Mean final SFA loss: 0.0000
- Parameter count: 135608

## Hard-Seed Table (Seeds 53, 71)

| seed | arm | collapsed | ΔR²_color | centroid_mse | mean_abs_corr | per_dim_std |
|------|-----|-----------|-----------|--------------|---------------|-------------|

## Original vs Fresh Seed Bank Comparison

| arm | seed_bank | mean_ΔR²_color | collapse_rate | mean_centroid_mse |
|-----|-----------|----------------|---------------|-------------------|
| A (VICReg-only control) | original (1) | 0.0000 | 1.00 | nan |
| B (SFA+VICReg, sfa=5.0) | original (1) | 0.0000 | 1.00 | nan |
| C (SFA+VICReg, sfa=1.0) | original (1) | 0.0000 | 1.00 | nan |

## Gate Check

- **F1:** ALL seeds collapsed — cannot evaluate
- **F2 (collapse rate ≤ 10%):** FAIL (1.00 = 1/1)
- **F3 (centroid MSE ≤ Arm A + 1σ):** FAIL (Arm B = nan, Arm A mean+1σ = nan)

## Pre-Registered Outcome Classification

**HYPOTHESIS FALSIFIED**

Reasons: F1: ΔR²_color = -inf < 0.30; F2: collapse rate = 1.00 > 0.10

SFA+VICReg is **not consistent with M2's predicted mechanism** for improving
identity encoding on the separate-backbone architecture.

## Arm C (SFA+VICReg, sfa=1.0, conservative)

- Collapse rate: 1.00
- Mean ΔR²_color: 0.0000
- Mean centroid MSE: nan

