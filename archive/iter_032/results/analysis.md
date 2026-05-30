# Iter_032: Rich Dyn Readout + Run 4-Arm Experiment — Analysis

**Dual collapse criterion:** collapsed = collapsed_eval OR collapsed_train

**Arms:** E1, E1.5, E2, E3

**Seed bank:** 20 seeds: [7, 17, 31, 53, 71, 83, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163]

## Per-Arm Summary

### E1 (VICReg-only, mean-pool)
- N seeds: 20
- Collapse rate: 0.00 (0/20)
- Mean ΔR²_color: 0.1309 ± 0.2960
- Mean centroid MSE: 165.79
- Mean abs corr: 0.115
- Parameter count: 135608

### E1.5 (VICReg-only, centroid-gated scalar)
- N seeds: 20
- Collapse rate: 0.10 (2/20)
- Mean ΔR²_color: 0.0784 ± 0.1681
- Mean centroid MSE: 166.55
- Mean abs corr: 0.154
- Parameter count: 135608

### E2 (VICReg-only, centroid-gated rich K=4)
- N seeds: 20
- Collapse rate: 1.00 (20/20)
- Mean ΔR²_color: 0.1164 ± 0.2714
- Mean centroid MSE: 156.99
- Mean abs corr: 0.140
- Parameter count: 151016

### E3 (SFA+VICReg, centroid-gated rich K=4)
- N seeds: 20
- Collapse rate: 1.00 (20/20)
- Mean ΔR²_color: 0.1378 ± 0.2958
- Mean centroid MSE: 162.64
- Mean abs corr: 0.152
- Parameter count: 151016

## Gate Evaluation (CRITICAL)

- **F1 (mean E2 ΔR²_color (non-collapsed) >= 0.30):** FAIL (ALL seeds collapsed)
- **F2 (lower 95% CI of E2 mean ΔR²_color >= 0.18):** FAIL (insufficient non-collapsed seeds)
- **F3 (collapse rate <= 0.10 across ALL arms):** FAIL (max_cr=1.00)
- **F4 (E2 - E1 paired-seed mean ΔR² >= 0.10):** FAIL (value=-0.0145)
- **F5 (E2 - E1.5 paired-seed mean ΔR² >= 0.10):** FAIL (value=0.0380)
- **F6 (E3 - E2 paired-seed mean ΔR² (informational only)):** (value=0.0214)

## Binding Decision Rule Outcome

**Outcome:** BRANCH (b) — Hard-pivot to behavioral evaluation (centroid tracking, collision selectivity, causal sensitivity) with the best available representation. No further representation-only iterations.

## Pre-Registered Outcome Classification

- **Outcome classification:** F1 FAIL — the rich readout does not break through the identity-encoding threshold. HARD-PIVOT TRIGGERED.
