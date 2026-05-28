# Architecture Ceiling Phase 0: Comparison Report

## Summary

This report evaluates three architectural modifications against the control
to determine if any variant validates the hypothesis. The key test is whether
sub-features (K=4) improve semantic disentanglement without breaking
centroid tracking.

## Arm Configurations

| Arm | d_max | sub_features | dyn_source | Notes |
|-----|-------|--------------|------------|-------|
| Ctrl | 8 | 1 | spatial | Baseline CGIR |
| A | 8 | 1 | conv4 | Conv4 dynamics source |
| B | 16 | 1 | spatial | Expanded capacity |
| C | 8 | 4 | spatial | Sub-features K=4 |

Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000, d_t=3.
Seeds: [42, 123, 456, 789, 999].

## Results Table

| Metric | Ctrl | Arm A (conv4) | Arm B (d=16) | Arm C (K=4) |
|--------|------|---------------|--------------|-------------|
| Collapse | 1/5 | 3/5 | 1/5 | 5/5 |
| MSE | 117.85 | 111.57 | 119.49 | 112.52 |
| dR2_color | 0.0498 | 0.0904 | 0.1299 | 0.1003 |
| dR2_ident | -0.0346 | -0.0168 | -0.0270 | -0.0555 |
| norm_dyn_var | 0.008557 | 0.008891 | 0.010455 | 0.017713 |
| norm_coord_var | 0.000017 | 0.000023 | 0.000030 | 0.000019 |
| SFA effective | no | no | no | no |

## Criterion Definitions

- **C1 (Collapse):** Ctrl < 2/5 AND Arm C < 2/5 collapsed => PASS.
- **C2 (MSE):** Arm C MSE <= 1.10 x Ctrl MSE => PASS.
- **C3 (Color):** mean(delta_R2_color[Arm C] - delta_R2_color[Ctrl]) >= 0.10 => PASS.
- **C4 (Identity):** mean(delta_R2_identity[Arm C] - delta_R2_identity[Ctrl]) >= 0.10 => PASS.
- **C5 (SFA effective):** normalized_dyn_var < normalized_coord_var for Arm C => PASS.

## Honest Falsification Audit

| Criterion | Result | Detail |
|-----------|--------|--------|
| C1 (Collapse) | FAIL | Ctrl: 1/5, Arm C: 5/5 |
| C2 (Centroid MSE) | PASS | Arm C MSE=112.5158 vs threshold=129.6309 (1.10 x Ctrl) |
| C3 (Color disentanglement) | FAIL | Diff=0.0504 |
| C4 (Identity probe) | FAIL | Diff=-0.0208 |
| C5 (SFA effective) | FAIL | Arm C norm_dyn=0.017713 < norm_coord=0.000019 |

**Overall (C1 AND C2 AND C4): HYPOTHESIS FALSIFIED**

## Interpretation

The architecture ceiling experiment FAILED to validate the hypothesis.
None of the tested architectural modifications (conv4 source, expanded d_max,
or sub-features K=4) achieved the required improvement in semantic disentanglement
while maintaining tracking quality.

**Conclusion:** The bottleneck lies elsewhere. Next experiments should test:
1. Different dynamics readout mechanisms (attention-based, learned pooling).
2. Non-linear sub-feature interactions instead of independent K features.
3. Separate dynamics encoder instead of shared CNN.
