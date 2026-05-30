# Iter_033 v2 — Three-Condition Oracle Bracket (Fixed ORACLE Timing)

## Raw Triple (Primary Metric: Post-Collision Selectivity V-B)

| Condition | Mean | Std | n |
|----------|------|-----|---|
| random | 0.4712 | 0.1378 | 12 |
| learned_vicreg | 0.4708 | 0.1311 | 12 |
| learned_sfa | 0.4993 | 0.0109 | 12 |
| oracle | 0.4782 | 0.0307 | 12 |

**Ordering sanity check:** ORACLE(0.4782) >= RANDOM(0.4712) = True

**Branch (c) check:** |ORACLE - RANDOM| = |0.0070| < 0.10 = True

## BRANCH (c) FIRED

The task or motor protocol is the bottleneck, NOT perception.
The behavioral-pivot strategy is invalidated for this protocol.

## Per-Seed Primary Metric

| Seed | RANDOM | VICReg | SFA | ORACLE |
|------|--------|--------|-----|--------|
| 7 | 0.4930 | 0.4939 | 0.4879 | 0.5093 |
| 17 | 0.6471 | 0.5133 | 0.5022 | 0.4122 |
| 31 | 0.4963 | 0.4696 | 0.5223 | 0.4456 |
| 53 | 0.0672 | 0.4735 | 0.4788 | 0.4885 |
| 71 | 0.5091 | 0.7320 | 0.5040 | 0.4965 |
| 83 | 0.4545 | 0.5086 | 0.5058 | 0.4972 |
| 97 | 0.4928 | 0.4583 | 0.4977 | 0.5114 |
| 101 | 0.5175 | 0.5082 | 0.4920 | 0.4590 |
| 107 | 0.5263 | 0.4829 | 0.5075 | 0.4882 |
| 113 | 0.4990 | 0.3398 | 0.4934 | 0.4982 |
| 137 | 0.5273 | 0.1600 | 0.4983 | 0.4447 |
| 163 | 0.4242 | 0.5097 | 0.5015 | 0.4874 |

## Surprise Decomposition

| Condition | Mean Surprise Coord | Mean Surprise Dyn |
|-----------|--------------------|------------------|
| random | 4050.0697 | 6.0309 |
| learned_vicreg | 1.3033 | 2314.1137 |
| learned_sfa | 3.3244 | 31.2094 |
| oracle | 163851.3572 | 0.0000 |

## Secondary Metrics

| Condition | Tracking Error (mean) | Tracking Error (std) | Pert Sel (mean) | Pert Sel (std) |
|-----------|----------------------|---------------------|-----------------|---------------|
| random | 43.69 | 13.94 | 0.5492 | 0.4582 |
| learned_vicreg | 33.26 | 22.98 | 0.6075 | 0.1922 |
| learned_sfa | 45.71 | 10.68 | 0.4750 | 0.1836 |
| oracle | 46.68 | 6.47 | 0.4458 | 0.1957 |
