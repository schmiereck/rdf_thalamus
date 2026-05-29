# Iter_025 Architecture Ceiling Probe Analysis v2
**Date:** Auto-generated
**Pre-declared effect-size threshold:** 0.1000
**Matching scheme:** Hungarian (primary); sorted (secondary check)
---
## Per-Arm Summary (Final Step = 8000)

### A (JEPA+VICReg Control)
- N seeds: 10
- Collapse rate: 0.30 (3/10)
- delta_R2_color (Hungarian primary): 0.0267 +/- 0.1005
- delta_R2_color (sorted secondary): 0.0301 +/- 0.1646
- Eval mismatch rate: 0.600 +/- 0.327
- Centroid MSE: 156.55
- Tracking level corr: 0.051
- Mean abs corr: 0.368

**Per-seed details:**
| seed | collapsed | delta_R2_color_HU | delta_R2_color_sorted | mismatch_rate | scheme_agree |
|------|-----------|-------------------|-----------------------|---------------|--------------|
| 17 | 0 | -0.0206 | -0.0843 | 1.000 | Y |
| 31 | 0 | -0.1288 | 0.0598 | 0.667 | Y |
| 7 | 0 | 0.2519 | 0.4615 | 0.667 | Y |
| 53 | 1 | 0.1101 | 0.1024 | 0.667 | Y |
| 71 | 1 | 0.0157 | 0.0157 | 0.000 | Y |
| 83 | 0 | 0.0133 | -0.0180 | 0.667 | Y |
| 113 | 1 | -0.0887 | -0.0887 | 0.000 | Y |
| 97 | 0 | -0.0053 | -0.1809 | 0.667 | Y |
| 149 | 0 | 0.0530 | 0.0626 | 0.667 | Y |
| 127 | 0 | 0.0661 | -0.0289 | 1.000 | Y |

### B (Supervised Color Probe d_max=8)
- N seeds: 10
- Collapse rate: 0.30 (3/10)
- delta_R2_color (Hungarian primary): -0.0257 +/- 0.1327
- delta_R2_color (sorted secondary): -0.0840 +/- 0.1421
- Eval mismatch rate: 0.667 +/- 0.258
- Centroid MSE: 162.98
- Tracking level corr: 0.047
- Mean abs corr: 0.295

**Per-seed details:**
| seed | collapsed | delta_R2_color_HU | delta_R2_color_sorted | mismatch_rate | scheme_agree |
|------|-----------|-------------------|-----------------------|---------------|--------------|
| 17 | 0 | 0.1676 | -0.0368 | 0.667 | N |
| 7 | 0 | -0.1925 | -0.0932 | 0.667 | Y |
| 53 | 1 | 0.0893 | 0.0893 | 0.000 | Y |
| 83 | 0 | -0.0499 | 0.0145 | 0.667 | Y |
| 71 | 1 | -0.1679 | -0.1926 | 0.667 | Y |
| 31 | 0 | -0.2521 | -0.1559 | 0.667 | Y |
| 97 | 0 | 0.1148 | -0.3833 | 0.667 | N |
| 113 | 0 | -0.0026 | 0.1030 | 1.000 | N |
| 149 | 1 | -0.0100 | 0.0033 | 0.667 | Y |
| 127 | 0 | 0.0462 | -0.1880 | 1.000 | Y |

### C (ID-Contrastive d_max=8)
- N seeds: 10
- Collapse rate: 0.50 (5/10)
- delta_R2_color (Hungarian primary): 0.0106 +/- 0.1005
- delta_R2_color (sorted secondary): 0.0442 +/- 0.1994
- Eval mismatch rate: 0.500 +/- 0.428
- Centroid MSE: 159.56
- Tracking level corr: -0.051
- Mean abs corr: 0.331

**Per-seed details:**
| seed | collapsed | delta_R2_color_HU | delta_R2_color_sorted | mismatch_rate | scheme_agree |
|------|-----------|-------------------|-----------------------|---------------|--------------|
| 7 | 0 | 0.1314 | 0.5969 | 1.000 | Y |
| 17 | 0 | -0.0842 | -0.0167 | 1.000 | Y |
| 31 | 0 | -0.2284 | -0.0666 | 0.667 | Y |
| 53 | 1 | 0.1047 | 0.1047 | 0.000 | Y |
| 71 | 1 | 0.0225 | 0.0225 | 0.000 | Y |
| 83 | 0 | 0.0218 | 0.0218 | 0.000 | Y |
| 97 | 1 | 0.0456 | 0.0456 | 0.000 | Y |
| 113 | 1 | -0.0285 | -0.1536 | 1.000 | Y |
| 127 | 0 | 0.0196 | 0.0212 | 0.667 | Y |
| 149 | 1 | 0.1021 | -0.1340 | 0.667 | N |

### D (Supervised Color Probe d_max=16)
- N seeds: 10
- Collapse rate: 0.40 (4/10)
- delta_R2_color (Hungarian primary): -0.0989 +/- 0.1784
- delta_R2_color (sorted secondary): -0.0016 +/- 0.0819
- Eval mismatch rate: 0.500 +/- 0.342
- Centroid MSE: 167.70
- Tracking level corr: 0.015
- Mean abs corr: 0.344

**Per-seed details:**
| seed | collapsed | delta_R2_color_HU | delta_R2_color_sorted | mismatch_rate | scheme_agree |
|------|-----------|-------------------|-----------------------|---------------|--------------|
| 7 | 1 | -0.6109 | 0.2052 | 0.667 | N |
| 17 | 0 | 0.0152 | -0.0412 | 0.667 | Y |
| 53 | 0 | -0.0460 | -0.0097 | 0.667 | Y |
| 31 | 1 | -0.1018 | -0.1018 | 0.000 | Y |
| 71 | 0 | -0.1630 | -0.0391 | 1.000 | Y |
| 83 | 0 | -0.0151 | 0.0534 | 0.667 | Y |
| 97 | 0 | 0.0234 | 0.0234 | 0.000 | Y |
| 113 | 1 | -0.0236 | -0.0236 | 0.000 | Y |
| 127 | 0 | -0.0301 | -0.0862 | 0.667 | Y |
| 149 | 1 | -0.0373 | 0.0035 | 0.667 | Y |

### E (JEPA+VICReg Control d_max=16)
- N seeds: 10
- Collapse rate: 0.30 (3/10)
- delta_R2_color (Hungarian primary): 0.0691 +/- 0.2943
- delta_R2_color (sorted secondary): 0.1813 +/- 0.3407
- Eval mismatch rate: 0.433 +/- 0.367
- Centroid MSE: 171.37
- Tracking level corr: -0.077
- Mean abs corr: 0.309

**Per-seed details:**
| seed | collapsed | delta_R2_color_HU | delta_R2_color_sorted | mismatch_rate | scheme_agree |
|------|-----------|-------------------|-----------------------|---------------|--------------|
| 7 | 0 | -0.1286 | 0.7742 | 1.000 | N |
| 17 | 0 | -0.0331 | -0.0331 | 0.000 | Y |
| 31 | 0 | 0.0009 | 0.0009 | 0.000 | Y |
| 71 | 1 | -0.1779 | 0.0154 | 0.667 | Y |
| 53 | 0 | 0.0369 | 0.0141 | 0.667 | Y |
| 83 | 1 | 0.0060 | 0.0060 | 0.000 | Y |
| 113 | 1 | -0.1161 | -0.1161 | 0.000 | Y |
| 97 | 0 | 0.9107 | 0.9139 | 0.667 | Y |
| 127 | 0 | 0.1282 | 0.1945 | 0.667 | Y |
| 149 | 0 | 0.0637 | 0.0436 | 0.667 | Y |

## Experiment Power Check

- Arm A collapse rate: 0.30 (3/10)
- **RESULT: UNDERPOWERED.** Arm A collapse rate > 2/10. No architecture claim can be valid.

## Matching Dependency Report

**B (Supervised Color Probe d_max=8):**
- Non-collapsed seeds: 7
- Sorted vs Hungarian pass/fail disagreement rate: 0.43 (3/7)
- **Matching-dependent outcome.** Falsification claim NOT earned for this arm.

**C (ID-Contrastive d_max=8):**
- Non-collapsed seeds: 5
- Sorted vs Hungarian pass/fail disagreement rate: 0.00 (0/5)
- Matching schemes agree on pass/fail for >=75% of non-collapsed seeds. Verdict is stable.

## Falsification Verdicts

**H1 (Architecture Capacity — Arm B):**
- Collapse rate: 0.30 (3/10)
- Mean delta_R2_color (non-collapsed, Hungarian): -0.0241
- Threshold: 0.1000
- **Verdict:** UNDERPOWERED. No architecture claim valid.

**H2 (ID-Contrastive Viability — Arm C):**
- Collapse rate: 0.50 (5/10)
- Mean delta_R2_color (non-collapsed, Hungarian): -0.0280
- Threshold: 0.1000
- **Verdict:** UNDERPOWERED. No architecture claim valid.

## Capacity Audit (Arm E vs iter_023 d_max=16 claim)

- Arm E non-collapsed mean delta_R2_color (Hungarian): 0.1398
- **Audit result:** d_max=16 improvement CONFIRMED as capacity effect (occurs without identity objective).

## Outcome Quadrant

| Arm B | Arm C | Interpretation |
|-------|-------|----------------|
| N/A | N/A | Experiment underpowered (Arm A collapse > 2/10). No architecture claim valid. |

## Next-Step Recommendation

- The experiment is underpowered due to high collapse rate in the control arm.
- **Recommendation:** Investigate and fix the underlying collapse issue before running another architecture probe. Consider further reducing LR, increasing batch size, or stronger variance regularization.

