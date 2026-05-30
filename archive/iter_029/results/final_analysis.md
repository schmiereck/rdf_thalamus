# Iter_029 SFA+VICReg on Separate Backbone — Analysis
**Dual collapse criterion:** collapsed = collapsed_eval OR collapsed_train
**Arms:** A (VICReg-only control), B (SFA+VICReg, sfa=5.0), C (SFA+VICReg, sfa=1.0)
**Seed bank:** Union of original (10) + fresh (10) = 20 seeds
**Gate thresholds:** F1: ΔR²_color ≥ 0.30 (Arm B, non-collapsed), F2: collapse rate ≤ 10% (Arm B), F3: centroid MSE within 1σ of Arm A

---
## Per-Arm Summary
### A (VICReg-only control)
- N seeds: 20
- Collapse rate (dual): 0.00 (0/20)
- Collapse rate (eval-only): 0.00 (0/20)
- Collapse rate (train-only): 0.00 (0/20)
- Mean ΔR²_color: 0.0445 ± 0.2108
- Mean centroid MSE: 159.83
- Mean abs corr: 0.219
- Parameter count: 135608

### B (SFA+VICReg, sfa=5.0)
- N seeds: 20
- Collapse rate (dual): 0.00 (0/20)
- Collapse rate (eval-only): 0.00 (0/20)
- Collapse rate (train-only): 0.00 (0/20)
- Mean ΔR²_color: 0.2749 ± 0.5772
- Mean centroid MSE: 159.85
- Mean abs corr: 0.149
- Mean final SFA loss: 0.1408
- Parameter count: 135608

### C (SFA+VICReg, sfa=1.0)
- N seeds: 20
- Collapse rate (dual): 0.00 (0/20)
- Collapse rate (eval-only): 0.00 (0/20)
- Collapse rate (train-only): 0.00 (0/20)
- Mean ΔR²_color: 0.1428 ± 0.2634
- Mean centroid MSE: 167.13
- Mean abs corr: 0.126
- Mean final SFA loss: 0.2052
- Parameter count: 135608

## Hard-Seed Table (Seeds 53, 71)

| seed | arm | collapsed | ΔR²_color | centroid_mse | mean_abs_corr | per_dim_std |
|------|-----|-----------|-----------|--------------|---------------|-------------|
| 53 | A (VICReg-only control) | N | -0.0391 | 125.27 | 0.295 | [1.1486736536026, 0.679221510887146, 1.0565036535263062] |
| 53 | B (SFA+VICReg, sfa=5.0) | N | -0.0484 | 137.15 | 0.328 | [0.8680583834648132, 1.1542837619781494, 0.911353588104248] |
| 53 | C (SFA+VICReg, sfa=1.0) | N | 0.0403 | 135.27 | 0.117 | [1.2237608432769775, 1.0696364641189575, 1.0441347360610962] |
| 71 | A (VICReg-only control) | N | -0.0662 | 54.10 | 0.306 | [0.9214791059494019, 1.387037992477417, 1.3052146434783936] |
| 71 | B (SFA+VICReg, sfa=5.0) | N | 0.0413 | 66.60 | 0.049 | [1.3406065702438354, 1.226110577583313, 1.2482761144638062] |
| 71 | C (SFA+VICReg, sfa=1.0) | N | 0.1391 | 68.37 | 0.047 | [1.339755654335022, 1.1678798198699951, 1.090072751045227] |

## Original vs Fresh Seed Bank Comparison

| arm | seed_bank | mean_ΔR²_color | collapse_rate | mean_centroid_mse |
|-----|-----------|----------------|---------------|-------------------|
| A (VICReg-only control) | original (10) | 0.0557 | 0.00 | 154.48 |
| A (VICReg-only control) | fresh (10) | 0.0333 | 0.00 | 165.18 |
| B (SFA+VICReg, sfa=5.0) | original (10) | 0.1921 | 0.00 | 161.84 |
| B (SFA+VICReg, sfa=5.0) | fresh (10) | 0.3576 | 0.00 | 157.86 |
| C (SFA+VICReg, sfa=1.0) | original (10) | 0.0960 | 0.00 | 172.27 |
| C (SFA+VICReg, sfa=1.0) | fresh (10) | 0.1896 | 0.00 | 161.98 |

## Gate Check

- **F1 (ΔR²_color ≥ 0.30, non-collapsed):** FAIL (Arm B mean ΔR²_color = 0.2749)
- **F2 (collapse rate ≤ 10%):** PASS (0.00 = 0/20)
- **F3 (centroid MSE ≤ Arm A + 1σ):** PASS (Arm B = 159.85, Arm A mean+1σ = 241.88)

## Pre-Registered Outcome Classification

**HYPOTHESIS FALSIFIED**

Reasons: F1: ΔR²_color = 0.2749 < 0.30

SFA+VICReg is **not consistent with M2's predicted mechanism** for improving
identity encoding on the separate-backbone architecture.

## Arm C (SFA+VICReg, sfa=1.0, conservative)

- Collapse rate: 0.00
- Mean ΔR²_color: 0.1428
- Mean centroid MSE: 167.13

