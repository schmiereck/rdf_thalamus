# Iter_025 Architecture Ceiling Probe Analysis
**Date:** Auto-generated
**Ceiling threshold:** 0.1000
---
## Noise Floor (Frozen Random Encoder + Probe-Only Training)
- Seeds: [7, 17, 31]
- delta_R2_color per run: ['0.0000', '0.0000', '0.0000']
- floor_mean = 0.0000  (std = 0.0000)
- **Threshold used:** max(0.10, floor_mean + 0.08) = 0.1000

## Per-Arm Summary (Final Step = 5000)

### A (JEPA+VICReg Control)
- N seeds: 1
- Collapse rate: 1.00 (std=0.00)
- delta_R2_color (greedy): 0.0000 +/- 0.0000
- delta_R2_color (sorted): N/A +/- N/A
- delta_R2_color (Hungarian): N/A +/- N/A
- Eval mismatch rate: N/A +/- N/A
- Centroid MSE: N/A
- Tracking level corr: -1.000

### B (Supervised Color Probe d_max=8)
- N seeds: 1
- Collapse rate: 1.00 (std=0.00)
- delta_R2_color (greedy): 0.0000 +/- 0.0000
- delta_R2_color (sorted): 0.0000 +/- 0.0000
- delta_R2_color (Hungarian): 0.0000 +/- 0.0000
- Eval mismatch rate: 0.000 +/- 0.000
- Centroid MSE: N/A
- Tracking level corr: -1.000

### C (ID-Contrastive d_max=8)
- N seeds: 1
- Collapse rate: 1.00 (std=0.00)
- delta_R2_color (greedy): 0.0000 +/- 0.0000
- delta_R2_color (sorted): 0.0000 +/- 0.0000
- delta_R2_color (Hungarian): 0.0000 +/- 0.0000
- Eval mismatch rate: 0.000 +/- 0.000
- Centroid MSE: N/A
- Tracking level corr: -1.000

### D (Supervised Color Probe d_max=16)
- N seeds: 1
- Collapse rate: 1.00 (std=0.00)
- delta_R2_color (greedy): 0.0000 +/- 0.0000
- delta_R2_color (sorted): 0.0000 +/- 0.0000
- delta_R2_color (Hungarian): 0.0000 +/- 0.0000
- Eval mismatch rate: 0.000 +/- 0.000
- Centroid MSE: N/A
- Tracking level corr: -1.000

## Falsification Checks

**Arm B:** ALL RUNS COLLAPSED. H1 FALSIFIED.

**Arm C:** ALL RUNS COLLAPSED. H2 FALSIFIED.

**Arm D:** ALL RUNS COLLAPSED.

## Arm A Drift Check (Fresh Seeds vs Previous Iterations)

- Arm A mean delta_R2_color (fresh seeds): 0.0000
- Iter_022-024 reference (typical control): ~0.00 to 0.02
- Drift flagged: NO

## Outcome Quadrant

| Arm B | Arm C | Interpretation |
|-------|-------|----------------|
|  NO   |  NO   | Architecture-level bottleneck suspected (conditional on matching scheme). Next: separate z_dyn encoder. |

