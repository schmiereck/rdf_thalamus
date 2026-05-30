# Iter_033 v3 — Three-Condition Oracle Bracket (Full-Physics ORACLE Predictor)

## Raw Triple (Primary Metric: Post-Collision Selectivity V-B)

| Condition | Mean | Std | n |
|----------|------|-----|---|
| random | 0.5043 | 0.0683 | 12 |
| learned_vicreg | 0.4708 | 0.1311 | 12 |
| learned_sfa | 0.4993 | 0.0109 | 12 |
| oracle | 0.5044 | 0.0148 | 12 |

**Ordering sanity check:** ORACLE(0.5044) >= RANDOM(0.5043) = True

**Branch (c) check:** |ORACLE - RANDOM| = |0.0001| < 0.10 = True

## BRANCH (c) FIRED

The task or motor protocol is the bottleneck, NOT perception.
The behavioral-pivot strategy is invalidated for this protocol.

## Per-Seed Primary Metric

| Seed | RANDOM | VICReg | SFA | ORACLE |
|------|--------|--------|-----|--------|
| 7 | 0.5022 | 0.4939 | 0.4879 | 0.5000 |
| 17 | 0.6365 | 0.5133 | 0.5022 | 0.5102 |
| 31 | 0.4893 | 0.4696 | 0.5223 | 0.5265 |
| 53 | 0.4970 | 0.4735 | 0.4788 | 0.5140 |
| 71 | 0.5317 | 0.7320 | 0.5040 | 0.5028 |
| 83 | 0.4673 | 0.5086 | 0.5058 | 0.4822 |
| 97 | 0.5360 | 0.4583 | 0.4977 | 0.4992 |
| 101 | 0.5242 | 0.5082 | 0.4920 | 0.5075 |
| 107 | 0.5414 | 0.4829 | 0.5075 | 0.5102 |
| 113 | 0.4846 | 0.3398 | 0.4934 | 0.5030 |
| 137 | 0.5058 | 0.1600 | 0.4983 | 0.4748 |
| 163 | 0.3361 | 0.5097 | 0.5015 | 0.5227 |

## Surprise Decomposition

| Condition | Mean Surprise Coord | Mean Surprise Dyn |
|-----------|--------------------|------------------|
| random | 4106.9761 | 7.4768 |
| learned_vicreg | 1.3033 | 2314.1137 |
| learned_sfa | 3.3244 | 31.2094 |
| oracle | 309.5067 | 0.0000 |

## Secondary Metrics

| Condition | Tracking Error (mean) | Tracking Error (std) | Pert Sel (mean) | Pert Sel (std) |
|-----------|----------------------|---------------------|-----------------|---------------|
| random | 32.86 | 11.54 | 0.4658 | 0.2831 |
| learned_vicreg | 33.26 | 22.98 | 0.6075 | 0.1922 |
| learned_sfa | 45.71 | 10.68 | 0.4750 | 0.1836 |
| oracle | 58.08 | 21.03 | 0.4308 | 0.2774 |
