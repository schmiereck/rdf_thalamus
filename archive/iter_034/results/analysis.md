# iter_034 Dynamics-Learning Benchmark Analysis

## Experimental Setup

- Environment: PhysicsSandbox(N=3), 2000 steps
- Seeds: [7, 31, 53, 71, 83, 97, 113, 163]
- Conditions: ['ORACLE', 'RANDOM', 'PASSIVE']
- Velocity noise: sigma_vel=0.5
- Collision threshold: 4.0, dv_threshold: 0.5

## Per-Seed MAPE Results

| Seed | ORACLE | RANDOM | PASSIVE |
|------|--------|--------|---------|
|    7 | 1.1653 | 1.0110 | 0.4895 |
|   31 | 0.6185 | 0.9456 | 0.9680 |
|   53 | 1.0008 | 0.8434 | 0.7225 |
|   71 | 0.9941 | 0.9502 | 0.7437 |
|   83 | 1.0315 | 1.3508 | 0.2566 |
|   97 | 0.9810 | 0.9989 | 0.4943 |
|  113 | 1.2469 | 1.0435 | 0.4184 |
|  163 | 1.0022 | 0.8477 | 0.6830 |

## Summary Statistics

| Condition | Mean MAPE | Std MAPE | Min MAPE | Max MAPE |
|-----------|-----------|----------|----------|----------|
| ORACLE    | 1.0050 | 0.1714 | 0.6185 | 1.2469 |
| RANDOM    | 0.9989 | 0.1493 | 0.8434 | 1.3508 |
| PASSIVE   | 0.5970 | 0.2098 | 0.2566 | 0.9680 |

## Gates

**G1** (RANDOM - ORACLE >= 0.15, CI lower >= 0.05):
- Mean gap: -0.0061
- 95% Bootstrap CI: [-0.1374, 0.1429]
- Result: FAIL

**G2** (PASSIVE - RANDOM >= 0.05, CI lower > 0):
- Mean gap: -0.4019
- 95% Bootstrap CI: [-0.6486, -0.1865]
- Result: FAIL

**G3** (ORACLE < RANDOM < PASSIVE means):
- ORACLE mean: 1.0050
- RANDOM mean: 0.9989
- PASSIVE mean: 0.5970
- Result: FAIL

## Sanity Checks

- **S1**: PASS — Mean per-object pointer collisions: [324.75, 464.375, 106.375]
- **S2**: PASS — ORACLE po=[207, 1846, 165, 977, 1611, 483, 1744, 131], PASSIVE po=[49, 83, 94, 78, 111, 77, 96, 124]
- **S3**: PASS — Min fraction across all runs: 1.0000
- **S4**: PASS — seed=7: [61, 65, 81]; seed=31: [1763, 39, 44]; seed=53: [75, 52, 38]; seed=71: [303, 374, 300]; seed=83: [77, 1442, 92]; seed=97: [164, 155, 164]; seed=113: [116, 1545, 83]; seed=163: [39, 43, 49]
- **S5**: PASS — Max OOB fraction across seeds: 0.000000

## Conclusion

**Benchmark FALSIFIED.** Sanity checks pass but one or more gates failed.
