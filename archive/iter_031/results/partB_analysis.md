# Iter_031 Part B — CLTS Protocol Calibration Analysis

## Overview

This report calibrates the CLTS evaluation protocol on a collision-sparse N=2 environment
with a subtle mass perturbation at step 1000 (1.5× multiplier on object 0).
Results cover 5 seeds (`[7, 31, 97, 113, 137]`) and 3 conditions (surprise-driven, frozen, random)
for both `d_t = 2` and `d_t = 3` settings.

Checkpoints: `archive/iter_029/results/checkpoints/a_vicreg-only_control_seed{}.pt`

## Baselines: Frozen and Random Conditions

### d_t = 2

**frozen** (13 seeds):
- Tracking error: 21.35 ± 18.68 px
- Collisions per 100 steps: 25.57 ± 36.60
- Collision selectivity (Version A): 1.0000 ± 0.0000
- Collision selectivity (Version B): 0.6126 ± 0.2168
- Perturbation selectivity: 0.3120 ± 0.2959

**random** (13 seeds):
- Tracking error: 38.75 ± 17.04 px
- Collisions per 100 steps: 10.33 ± 6.58
- Collision selectivity (Version A): 1.0000 ± 0.0000
- Collision selectivity (Version B): 0.4397 ± 0.1495
- Perturbation selectivity: 0.6060 ± 0.2373

### d_t = 3

**frozen** (13 seeds):
- Tracking error: 35.88 ± 16.77 px
- Collisions per 100 steps: 11.11 ± 13.75
- Collision selectivity (Version A): 1.0000 ± 0.0000
- Collision selectivity (Version B): 0.4269 ± 0.1533
- Perturbation selectivity: 0.5100 ± 0.3646

**random** (13 seeds):
- Tracking error: 33.80 ± 17.02 px
- Collisions per 100 steps: 17.05 ± 11.16
- Collision selectivity (Version A): 1.0000 ± 0.0000
- Collision selectivity (Version B): 0.5444 ± 0.0738
- Perturbation selectivity: 0.4800 ± 0.0822

## Surprise-Driven Performance

### d_t = 2

- Tracking error: 37.61 ± 11.30 px
- Collisions per 100 steps: 19.76 ± 26.09
- Collision selectivity (Version A): 1.0000 ± 0.0000
- Collision selectivity (Version B): 0.5900 ± 0.1868
- Perturbation selectivity: 0.4820 ± 0.2920

| Seed | Tracking Error | Collisions | Sel (V-A) | Sel (V-B) | Pert Sel |
|------|---------------|------------|------------|------------|----------|
| 7 | 25.74 | 64.60 | 1.0000 | 0.9236 | 0.0000 |
| 31 | 50.48 | 0.65 | 1.0000 | 0.5231 | 0.6700 |
| 97 | 30.15 | 12.20 | 1.0000 | 0.5087 | 0.5200 |
| 113 | 48.84 | 2.75 | 1.0000 | 0.4994 | 0.7500 |
| 137 | 32.85 | 18.60 | 1.0000 | 0.4952 | 0.4700 |

### d_t = 3

- Tracking error: 38.10 ± 26.75 px
- Collisions per 100 steps: 23.85 ± 20.07
- Collision selectivity (Version A): 1.0000 ± 0.0000
- Collision selectivity (Version B): 0.3843 ± 0.1388
- Perturbation selectivity: 0.5580 ± 0.0814

| Seed | Tracking Error | Collisions | Sel (V-A) | Sel (V-B) | Pert Sel |
|------|---------------|------------|------------|------------|----------|
| 7 | 78.77 | 3.80 | 1.0000 | 0.4939 | 0.5000 |
| 31 | 22.55 | 40.80 | 1.0000 | 0.4696 | 0.5000 |
| 97 | 10.53 | 1.20 | 1.0000 | 0.4583 | 0.5000 |
| 113 | 49.42 | 30.50 | 1.0000 | 0.3398 | 0.6700 |
| 137 | 29.23 | 42.95 | 1.0000 | 0.1600 | 0.6200 |


## Gate Evaluation (G1–G3)

| Gate | d_t | Criterion | Measured (SD) | Threshold | Result |
|------|-----|-----------|----------------|-----------|--------|
| G1_tracking | d_t=2 | SD tracking ≤ random_mean − 1*random_std | 37.61 | 21.72 | FAIL |
| G2_collision_sel(B) | d_t=2 | SD sel_B ≥ random × 1.5 | 0.5900 | 0.6595 | FAIL |
| G3_perturbation_sel | d_t=2 | SD pert_sel ≥ random × 1.5 | 0.4820 | 0.9090 | FAIL |
| G1_tracking | d_t=3 | SD tracking ≤ random_mean − 1*random_std | 38.10 | 16.79 | FAIL |
| G2_collision_sel(B) | d_t=3 | SD sel_B ≥ random × 1.5 | 0.3843 | 0.8167 | FAIL |
| G3_perturbation_sel | d_t=3 | SD pert_sel ≥ random × 1.5 | 0.5580 | 0.7200 | FAIL |

### Gate Pass Summary

| d_t | G1 | G2 | G3 | Total Passed |
|-----|----|----|----|-------------|
| d_t=2 | ✗ | ✗ | ✗ | 0 |
| d_t=3 | ✗ | ✗ | ✗ | 0 |

## Protocol Recommendation

**No setting passes all gates.**

Best performance: d_t = 2 with 0/3 gates passing.

### Analysis for d_t=2

❌ Not all gates pass at d_t=2.
  - G1 fails: surprise-driven tracking (37.61 px) is not better than random (mean ± std = 38.75 ± 17.04). This suggests the CLTS mechanism is not providing a tracking advantage over random.
  - G2 fails: post-collision selectivity (V-B) for surprise-driven (0.5900) is not ≥ 1.5× random baseline (0.4397). The mechanism does not preferentially attend to the max-velocity-change colliding object.
  - G3 fails: post-perturbation selectivity for surprise-driven (0.4820) is not ≥ 1.5× random baseline (0.6060). The mechanism does not preferentially attend to the perturbed object.

### Analysis for d_t=3

❌ Not all gates pass at d_t=3.
  - G1 fails: surprise-driven tracking (38.10 px) is not better than random (mean ± std = 33.80 ± 17.02). This suggests the CLTS mechanism is not providing a tracking advantage over random.
  - G2 fails: post-collision selectivity (V-B) for surprise-driven (0.3843) is not ≥ 1.5× random baseline (0.5444). The mechanism does not preferentially attend to the max-velocity-change colliding object.
  - G3 fails: post-perturbation selectivity for surprise-driven (0.5580) is not ≥ 1.5× random baseline (0.4800). The mechanism does not preferentially attend to the perturbed object.

---
*Analysis generated by `src/run_iter031_partB.py`
