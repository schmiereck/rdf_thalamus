# Current Research State
Phase: Phase 2 Complete (Representation Base Evaluated)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 1 goal was to validate passive representation learning (Pillars A, B, C) and GDASR recruitment.

## Confirmed
- Workspace environment and CPU-only PyTorch execution validated (iter_001.2).
- 1D Physics Sandbox successfully simulates continuous elastic boundary and inter-object collisions (iter_002.1).
- VICReg-style temporal-prediction JEPA pipeline successfully implemented with deterministic seeding (iter_002.3).
- Stop-gradient stabilization successfully blocks gradients flowing to stable dimensions during recruitment (iter_002.1).

## Refuted
- REFUTED: GDASR with a cumulative error buffer triggers dimension recruitment upon N=3 transition (iter_002.3). High initialization errors inflate the threshold.
- REFUTED: VICReg with cov_weight=1.0 prevents representation collapse; representations collapse to collinear 1D manifolds with r > 0.99 (iter_002.3).

## Best Result
- Final Temporal Prediction Sim Loss: 0.06662 (achieved by B1 and non-recruiting DynamicJEPA) (iter_002.3).

## In Progress
- Rebalancing VICReg loss weights and designing a sliding-window error buffer to resolve threshold inflation and representation collapse.

## Open Questions
- Will increasing the VICReg covariance weight to 25.0 successfully prevent representation collapse and maintain r < 0.3?
- Will a rolling sliding-window error buffer (cleared at the N=3 transition) enable sensitive and reliable GDASR recruitment?
- Does a representation-warmup phase prevent early-stage dimensionality collapse?
