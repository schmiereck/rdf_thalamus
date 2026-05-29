# Iter_025 Architecture Ceiling Probe Analysis
**Date:** Auto-generated
**Ceiling threshold:** 1.0602
---
## Noise Floor (Frozen Random Encoder + Probe-Only Training)
- Seeds: [7, 17, 31]
- delta_R2_color per run: ['0.5795', '-0.1929', '2.5541']
- floor_mean = 0.9802  (std = 1.1567)
- **Threshold used:** max(0.10, floor_mean + 0.08) = 1.0602

## Per-Arm Summary (Final Step = 5000)

### A (JEPA+VICReg Control)
- N seeds: 5
- Collapse rate: 0.40 (std=0.49)
- delta_R2_color (greedy): -0.0255 +/- 0.2096
- delta_R2_color (sorted): N/A +/- N/A
- delta_R2_color (Hungarian): N/A +/- N/A
- Eval mismatch rate: N/A +/- N/A
- Centroid MSE: 118.11
- Tracking level corr: 0.060

### B (Supervised Color Probe d_max=8)
- N seeds: 5
- Collapse rate: 0.60 (std=0.49)
- delta_R2_color (greedy): 0.0262 +/- 0.1146
- delta_R2_color (sorted): -0.0187 +/- 0.1428
- delta_R2_color (Hungarian): -0.0001 +/- 0.1213
- Eval mismatch rate: 0.667 +/- 0.000
- Centroid MSE: 114.20
- Tracking level corr: 0.157

### C (ID-Contrastive d_max=8)
- N seeds: 5
- Collapse rate: 0.60 (std=0.49)
- delta_R2_color (greedy): -0.0119 +/- 0.0681
- delta_R2_color (sorted): 0.0135 +/- 0.0962
- delta_R2_color (Hungarian): 0.0451 +/- 0.0782
- Eval mismatch rate: 0.467 +/- 0.400
- Centroid MSE: 123.06
- Tracking level corr: 0.123

### D (Supervised Color Probe d_max=16)
- N seeds: 5
- Collapse rate: 0.60 (std=0.49)
- delta_R2_color (greedy): -0.0205 +/- 0.0981
- delta_R2_color (sorted): 0.1913 +/- 0.3750
- delta_R2_color (Hungarian): -0.0564 +/- 0.0285
- Eval mismatch rate: 0.533 +/- 0.267
- Centroid MSE: 128.47
- Tracking level corr: 0.009

## Falsification Checks

**Arm B (Supervised d_max=8):**
- Collapse rate: 0.60 (3/5)
- Mean delta_R2_color (non-collapsed, greedy): 0.0574
- Passes threshold (1.0602)? NO
- **H1 conclusion:** Architecture capacity INSUFFICIENT under direct supervision — architecture-level bottleneck suspected
  *Language:* Consistent with an architecture-level bottleneck on identity encoding

**Arm C (ID-Contrastive d_max=8):**
- Collapse rate: 0.60 (3/5)
- Mean delta_R2_color (non-collapsed, greedy): 0.0328
- Passes threshold (1.0602)? NO
- **H2 conclusion:** ID-contrastive formulation insufficient; architecture may or may not be the bottleneck
  *Language:* Contrastive formulation insufficient

**Arm D (Supervised d_max=16):**
- Mean delta_R2_color (non-collapsed): -0.0524
- Note: Arm D alone does NOT confirm H1. Only meaningful in conjunction with Arm B.

## Arm A Drift Check (Fresh Seeds vs Previous Iterations)

- Arm A mean delta_R2_color (fresh seeds): -0.0255
- Iter_022-024 reference (typical control): ~0.00 to 0.02
- Drift flagged: NO

## Outcome Quadrant

| Arm B | Arm C | Interpretation |
|-------|-------|----------------|
|  NO   |  NO   | Architecture-level bottleneck suspected (conditional on matching scheme). Next: separate z_dyn encoder. |

