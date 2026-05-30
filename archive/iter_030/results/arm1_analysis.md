# ARM 1 — Integration Smoke-Test Analysis (Iter 030)
## Protocol
- Seeds: fresh [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]
- Hard seeds (reported separately): [53, 71]
- Conditions: ['CLTS-SFA', 'CLTS-VICReg', 'CLTS-Frozen', 'CLTS-Random']
- Evaluation steps per run: 200–2000
- Total seeds evaluated (fresh only for gates): 10

## Summary Statistics (Fresh Seeds)

| Condition | n | Tracking Error (mean±std) | Collision Switch (mean±std) | Perturbation Switch Rate | Mean Surprise (mean±std) |
|-----------|---|---------------------------|----------------------------|--------------------------|--------------------------|
| CLTS-SFA | 10 | 45.09±9.82 | 1.00±0.00 | 1.00 | 99.24±159.29 |
| CLTS-VICReg | 10 | 36.22±2.83 | 1.00±0.00 | 1.00 | 1949.18±1344.45 |
| CLTS-Frozen | 10 | 47.16±12.05 | 1.00±0.00 | 1.00 | 129.52±198.48 |
| CLTS-Random | 10 | 46.76±9.59 | 1.00±0.00 | 1.00 | 90.61±135.54 |

## Gate Evaluation (Fresh Seeds Only)

**G1 (Tracking Functionality):** mean tracking error < 20 pixels
- CLTS-SFA: 45.09 pixels → FAIL
- CLTS-VICReg: 36.22 pixels → FAIL

**G2 (Attention Validity — Collision):** switch-rate ≥ max(control) + 0.15
- CLTS-SFA: 1.00 vs control max 1.00 → FAIL
- CLTS-VICReg: 1.00 vs control max 1.00 → FAIL
- CLTS-Frozen: 1.00
- CLTS-Random: 1.00

**G3 (Causal Sensitivity — Mass Perturbation):** switch-rate ≥ max(control) + 0.15
- CLTS-SFA: 1.00 vs control max 1.00 → FAIL
- CLTS-VICReg: 1.00 vs control max 1.00 → FAIL
- CLTS-Frozen: 1.00
- CLTS-Random: 1.00

## Decision Rules

- CLTS-SFA gates passed: 0/3
- CLTS-VICReg gates passed: 0/3

**CLTS-SFA verdict: REPRESENTATION INSUFFICIENT** — objective hunt justified.

- Both fail → **representation truly insufficient**.

## Per-Seed Raw Results

| seed | cond | hard | track_err | coll_rate | coll_count | pert_switch | pert_step | surprise |
|------|------|------|-----------|-----------|------------|-------------|-----------|----------|
| 101 | CLTS-SFA | False | 55.81 | 1.00 | 764 | 1 | 1002 | 16.7078 |
| 101 | CLTS-VICReg | False | 38.44 | 1.00 | 789 | 1 | 1001 | 1486.8275 |
| 101 | CLTS-Frozen | False | 55.99 | 1.00 | 727 | 1 | 1003 | 32.1747 |
| 101 | CLTS-Random | False | 57.23 | 1.00 | 791 | 1 | 1001 | 16.9467 |
| 103 | CLTS-SFA | False | 29.19 | 1.00 | 882 | 1 | 1002 | 28.7353 |
| 103 | CLTS-VICReg | False | 33.92 | 1.00 | 673 | 1 | 1002 | 1918.3842 |
| 103 | CLTS-Frozen | False | 34.18 | 1.00 | 672 | 1 | 1018 | 6.0371 |
| 103 | CLTS-Random | False | 33.23 | 1.00 | 678 | 1 | 1002 | 46.4442 |
| 107 | CLTS-SFA | False | 32.96 | 1.00 | 564 | 1 | 1001 | 245.9735 |
| 107 | CLTS-VICReg | False | 33.70 | 1.00 | 513 | 1 | 1001 | 2161.6830 |
| 107 | CLTS-Frozen | False | 36.26 | 1.00 | 485 | 1 | 1001 | 473.9073 |
| 107 | CLTS-Random | False | 33.61 | 1.00 | 530 | 1 | 1002 | 234.1144 |
| 109 | CLTS-SFA | False | 54.83 | 1.00 | 697 | 1 | 1002 | 39.6848 |
| 109 | CLTS-VICReg | False | 36.94 | 1.00 | 734 | 1 | 1001 | 1734.9771 |
| 109 | CLTS-Frozen | False | 55.12 | 1.00 | 775 | 1 | 1004 | 45.5480 |
| 109 | CLTS-Random | False | 53.97 | 1.00 | 717 | 1 | 1009 | 39.3524 |
| 131 | CLTS-SFA | False | 47.44 | 1.00 | 611 | 1 | 1001 | 11.0264 |
| 131 | CLTS-VICReg | False | 34.52 | 1.00 | 574 | 1 | 1003 | 1147.6461 |
| 131 | CLTS-Frozen | False | 47.36 | 1.00 | 639 | 1 | 1001 | 8.0326 |
| 131 | CLTS-Random | False | 49.12 | 1.00 | 603 | 1 | 1002 | 10.5417 |
| 137 | CLTS-SFA | False | 52.76 | 1.00 | 717 | 1 | 1003 | 7.4350 |
| 137 | CLTS-VICReg | False | 34.08 | 1.00 | 497 | 1 | 1002 | 5480.2236 |
| 137 | CLTS-Frozen | False | 52.67 | 1.00 | 712 | 1 | 1004 | 4.5315 |
| 137 | CLTS-Random | False | 54.16 | 1.00 | 541 | 1 | 1002 | 5.6283 |
| 139 | CLTS-SFA | False | 39.72 | 1.00 | 234 | 1 | 1001 | 70.7467 |
| 139 | CLTS-VICReg | False | 36.16 | 1.00 | 518 | 1 | 1006 | 1860.4718 |
| 139 | CLTS-Frozen | False | 73.58 | 1.00 | 239 | 1 | 1007 | 161.6243 |
| 139 | CLTS-Random | False | 43.82 | 1.00 | 544 | 1 | 1002 | 66.8584 |
| 151 | CLTS-SFA | False | 33.73 | 1.00 | 700 | 1 | 1010 | 531.7401 |
| 151 | CLTS-VICReg | False | 32.74 | 1.00 | 648 | 1 | 1002 | 2632.1308 |
| 151 | CLTS-Frozen | False | 34.26 | 1.00 | 687 | 1 | 1001 | 554.7381 |
| 151 | CLTS-Random | False | 34.00 | 1.00 | 684 | 1 | 1003 | 448.8204 |
| 157 | CLTS-SFA | False | 48.34 | 1.00 | 587 | 1 | 1002 | 32.3858 |
| 157 | CLTS-VICReg | False | 41.01 | 1.00 | 276 | 1 | 1007 | 231.8911 |
| 157 | CLTS-Frozen | False | 35.83 | 1.00 | 356 | 1 | 1001 | 4.2100 |
| 157 | CLTS-Random | False | 48.76 | 1.00 | 600 | 1 | 1001 | 30.1888 |
| 163 | CLTS-SFA | False | 56.12 | 1.00 | 674 | 1 | 1005 | 7.9965 |
| 163 | CLTS-VICReg | False | 40.74 | 1.00 | 662 | 1 | 1004 | 837.5827 |
| 163 | CLTS-Frozen | False | 46.39 | 1.00 | 594 | 1 | 1004 | 4.3863 |
| 163 | CLTS-Random | False | 59.68 | 1.00 | 685 | 1 | 1001 | 7.1790 |
| 53 | CLTS-SFA | True | 38.57 | 1.00 | 630 | 1 | 1001 | 16.2268 |
| 53 | CLTS-VICReg | True | 32.96 | 1.00 | 606 | 1 | 1006 | 2540.8197 |
| 53 | CLTS-Frozen | True | 23.76 | 1.00 | 609 | 1 | 1002 | 4.8212 |
| 53 | CLTS-Random | True | 37.67 | 1.00 | 607 | 1 | 1001 | 15.7967 |
| 71 | CLTS-SFA | True | 34.06 | 1.00 | 635 | 1 | 1004 | 20.3225 |
| 71 | CLTS-VICReg | True | 33.08 | 1.00 | 656 | 1 | 1002 | 4354.2267 |
| 71 | CLTS-Frozen | True | 33.42 | 1.00 | 576 | 1 | 1001 | 36.4326 |
| 71 | CLTS-Random | True | 35.61 | 1.00 | 419 | 1 | 1008 | 17.4090 |
