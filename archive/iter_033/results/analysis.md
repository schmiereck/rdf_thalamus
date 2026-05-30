# Iter_033 — Three-Condition Oracle Bracket Analysis

## Raw Triple (Primary Metric: Post-Collision Selectivity V-B)

| Condition | Mean | Std | n |
|----------|------|-----|---|
| random | 0.4862 | 0.0460 | 12 |
| learned_vicreg | 0.4708 | 0.1311 | 12 |
| learned_sfa | 0.4993 | 0.0109 | 12 |
| oracle | 0.4344 | 0.1452 | 12 |

**Ordering sanity check:** ORACLE(0.4344) >= RANDOM(0.4862) = False

**Branch (c) check:** |ORACLE - RANDOM| = |-0.0518| < 0.10 = True

## BRANCH (c) FIRED

The task or motor protocol is the bottleneck, NOT perception.
The behavioral-pivot strategy is invalidated for this protocol.

## Per-Seed Primary Metric

| Seed | RANDOM | VICReg | SFA | ORACLE |
|------|--------|--------|-----|--------|
| 7 | 0.4500 | 0.4939 | 0.4879 | 0.5103 |
| 17 | 0.3667 | 0.5133 | 0.5022 | 0.5287 |
| 31 | 0.5192 | 0.4696 | 0.5223 | 0.4972 |
| 53 | 0.5270 | 0.4735 | 0.4788 | 0.2770 |
| 71 | 0.5351 | 0.7320 | 0.5040 | 0.5472 |
| 83 | 0.4737 | 0.5086 | 0.5058 | 0.0523 |
| 97 | 0.4794 | 0.4583 | 0.4977 | 0.5063 |
| 101 | 0.4892 | 0.5082 | 0.4920 | 0.4267 |
| 107 | 0.5235 | 0.4829 | 0.5075 | 0.4982 |
| 113 | 0.5095 | 0.3398 | 0.4934 | 0.3464 |
| 137 | 0.4932 | 0.1600 | 0.4983 | 0.4911 |
| 163 | 0.4684 | 0.5097 | 0.5015 | 0.5316 |

## Surprise Decomposition

| Condition | Mean Surprise Coord | Mean Surprise Dyn |
|-----------|--------------------|------------------|
| random | 4009.2924 | 7.5861 |
| learned_vicreg | 1.3033 | 2314.1137 |
| learned_sfa | 3.3244 | 31.2094 |
| oracle | 146383.1381 | 0.0000 |

## Secondary Metrics

| Condition | Tracking Error (mean) | Tracking Error (std) | Pert Sel (mean) | Pert Sel (std) |
|-----------|----------------------|---------------------|-----------------|---------------|
| random | 41.00 | 12.40 | 0.2808 | 0.2294 |
| learned_vicreg | 33.26 | 22.98 | 0.6075 | 0.1922 |
| learned_sfa | 45.71 | 10.68 | 0.4750 | 0.1836 |
| oracle | 42.53 | 16.46 | 0.5417 | 0.2517 |
