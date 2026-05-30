# Iter_031 Part A — Reconstruction+VICReg Ceiling Probe Analysis

This analysis tests whether Reconstruction+VICReg on the separate-backbone architecture achieves mean ΔR²_color ≥ 0.30 (lower 95% CI ≥ 0.18) on non-collapsed seeds, with non-trivial margins over capacity and training controls.

## Pre-Registered Falsification Gates (F1–F4)

| Gate | Criterion | Meaning | Result / Status |
|------|-----------|---------|-----------------|
| **F1** | mean ΔR²_color (Arm A) ≥ 0.30 | Ceiling clearance | FAIL (mean=0.0631) |
| **F2** | Lower 95% CI of mean ΔR²_color (Arm A) ≥ 0.18 | Variance stability | FAIL (lower CI=-0.0127, n=20) |
| **F3** | mean ΔR²_color (Arm A) — mean ΔR²_color (Arm B) ≥ 0.10 | Capacity-matters | FAIL (diff=0.0361) |
| **F4** | mean ΔR²_color (Arm A) — mean ΔR²_color (Arm C) ≥ 0.10 | Training-matters | FAIL (diff=nan) |


## Detailed Summary per Arm (all seeds)

### Arm A (d_max=8, trained)
- N seeds: 20
- Collapse rate: 0.00 (0/20)
- Mean ΔR²_color (all): 0.0631 ± 0.1729
- Mean reconstruction MSE: 0.018496
- Mean centroid MSE: 160.32
- Mean abs corr: 0.134

### Arm B (d_max=2, trained)
- N seeds: 20
- Collapse rate: 0.00 (0/20)
- Mean ΔR²_color (all): 0.0271 ± 0.2029
- Mean reconstruction MSE: 0.035154
- Mean centroid MSE: 165.98
- Mean abs corr: 0.142

### Arm C (d_max=8, random-encoder)
- N seeds: 20
- Collapse rate: 1.00 (20/20)
- Mean ΔR²_color (all): 1705348.0261 ± 7626543.7590
- Mean reconstruction MSE: 0.033462
- Mean centroid MSE: 1949.11
- Mean abs corr: 0.285

## Pre-Committed Mandate Revision Language

> **F1 or F2 FAILED: Ceiling not cleared or variance-unstable**
> "Reconstruction+VICReg fails to achieve ΔR²_color ≥ 0.30 with variance-stability. Even a supervised pixel-reconstruction target cannot make the mean-readout z_dyn stream encode identity above the 0.30 threshold. The z_dyn readout architecture itself constrains identity encoding regardless of objective class.
>
> **M2 revision pending architectural redesign:** priority is centroid-gated z_dyn readout or increased d_max.'"
