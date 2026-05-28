# CGIR Phase 0: Comparison Report

## Summary

This report evaluates the Centroid-Gated Identity Readout (CGIR) architectural
change against the pre-registered falsification criteria C1-C4.

## Arm Configurations

| Arm | dyn_readout | pos_encoding | CCR | Notes |
|-----|-------------|--------------|-----|-------|
| A   | centroid_gated | none      | covariance (smooth=10, spatial=10) | Primary test |
| B   | mean        | none         | covariance (smooth=10, spatial=10) | Baseline replication (iter_020 A1) |
| C   | centroid_gated | sinusoidal | covariance (smooth=10, spatial=10) | With positional encoding |
| D   | centroid_gated | none      | none | CGIR without CCR |

Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000, d_t=3.
Seeds: [42, 123, 456, 789, 999].

## C1-C4 Comparison Table

| Criterion | Arm A (CGIR+CCR) | Arm B (Mean+CCR) | Arm C (CGIR+pos) | Arm D (CGIR no CCR) |
|-----------|------------------|------------------|------------------|---------------------|
| Collapse  | 1/5 | 1/5 | 1/5 | 1/5 |
| MSE       | 117.85 | 121.86 | 121.30 | 122.76 |
| dR2_color | 0.0498 | -0.0736 | 0.0110 | 0.0074 |
| dR2_ident | -0.0346 | 0.0028 | -0.0384 | -0.0588 |
| avg std   | 0.8929 | 0.8125 | 1.0569 | 0.8539 |
| slowness  | 208.8003 | 338.0072 | 1277.1550 | 203.6734 |

### Criterion Definitions

- **C1 (Collapse):** Per-dim std < 0.5 in < 2 out of 5 seeds => PASS.
- **C2 (MSE):** Arm A MSE <= 1.10 x Arm B MSE => PASS.
- **C3 (Color disentanglement):** delta_R2_color >= 0.10 for Arm A => PASS.
- **C4 (Identity disentanglement):** delta_R2_identity >= 0.10 for Arm A => PASS (advisory).

## Per-Dimension Probe Details (Arm A, representative seed)

Seed 42, step=5000:

| dim | matched_obj | dyn_color | coord_color | dyn_pos | coord_pos | dyn_identity | coord_identity |
|-----|-------------|-----------|-------------|---------|-----------|--------------|----------------|
| 0 | 1 | -0.1358 | 0.0290 | 0.0509 | 0.3275 | 0.0189 | 0.0538 |
| 1 | 0 | -0.2863 | 0.1064 | -0.3971 | -0.1446 | 0.0217 | 0.2668 |
| 2 | 2 | -0.0533 | -0.0258 | 0.0914 | 0.1988 | 0.0162 | 0.0297 |

## Comparison with iter_020 (Recovery)

- **Iter_020 Arm A1** = Mean+SFA+CCR, replicated here as **Arm B**.
- The only difference between Arm A and Arm B in this evaluation is `dyn_readout`
  (centroid_gated vs mean), isolating the CGIR architectural change.

### C3 Interpretation

**C3 FAILED.** The root cause is NOT the spatial-mean computation.
If CGIR does not fix semantic disentanglement, the hypothesis is falsified.

### C4 Interpretation

**C4 FAILED (or irrelevant because C3 failed).** No evidence that CGIR
routes identity information above the position stream.

## Honest Falsification Audit

| Criterion | Result | Detail |
|-----------|--------|--------|
| C1 (Collapse) | PASS | Arm A: 1/5 seeds collapsed |
| C2 (Centroid MSE) | PASS | Arm A MSE=117.8463 vs threshold=134.0444 (1.10 x B) |
| C3 (Color disentanglement) | FAIL | Arm A delta_R2_color=0.0498 |
| C4 (Identity probe) | FAIL (advisory) | Arm A delta_R2_identity=-0.0346 |

**Overall: HYPOTHESIS FALSIFIED**