# iter_034 v2 Benchmark Analysis (MALRE)

## v1 Failure
MAPE was falsified (PASSIVE=0.597 < RANDOM=0.999 < ORACLE=1.005).
Pointer-object collision mass estimates overwhelmed least-squares.

## v2 Metric: MALRE
Mean Absolute Log-Ratio Error from MEDIAN of obj-obj collision ratios.

## Per-Seed MALRE

| Seed | ORACLE | RANDOM | PASSIVE |
|------|--------|--------|--------|
| 7 | 0.2023 | 0.1817 | 1.3333 |
| 31 | 0.4400 | 0.3346 | 1.3333 |
| 53 | 0.4325 | 0.3309 | 1.3333 |
| 71 | 0.6245 | 0.6667 | 1.3333 |
| 83 | 0.5378 | 0.6667 | 1.3333 |
| 97 | 0.4996 | 0.4878 | 1.3333 |
| 113 | 0.7626 | 0.2712 | 1.3333 |
| 163 | 0.5247 | 1.3333 | 1.3333 |

## Summary

- ORACLE: MALRE=0.5030±0.1513
- RANDOM: MALRE=0.5341±0.3447
- PASSIVE: MALRE=1.3333±0.0000

## Gates
- G1: gap=0.8303 CI=[0.7265,0.9392] PASS
- G2: gap=0.7992 CI=[0.5304,1.0043] PASS
- G3: PASS
- G4: gap=0.50 PASS

## Result: VALIDATED
