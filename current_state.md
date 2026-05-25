# Current Research State
Phase: Phase 16 Complete (Dual Control Cold-Start Resolved via WUP-MDL)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 16 goal was to resolve the "cold-start" pathological reject loop in the Dual Control Categorizer using a Probationary Warm-Up Period (WUP).

## Confirmed
- SUCCESS OF PROBATIONARY WARM-UP (iter_16.1): Introducing a Probationary Warm-Up Period (WUP) of W = 100 or W = 500 steps completely resolved the cold-start predictor bias. Arm P (WUP-MDL, W=100) and Arm P_big (WUP-MDL, W=500) achieved a perfect 100% recruitment rate (5/5 seeds), promoting cleanly to d_t=4 at steps 1900 and 2300 respectively.
- CENTROID DECODING MSE RECOVERY (iter_16.1): By recruiting the 4th dimension once the predictor head was warmed up, Arm P and Arm P_big successfully tracked the novel 4th object under active perturbation, reducing the post-transition centroid decoding MSE to 52.68 (W=100) and 49.57 (W=500), well below the pre-registered success threshold of 70.0.
- FALSIFICATION OF PVU GATING HYPOTHESIS (iter_16.1): The pre-registered PVU gating hypothesis (Arm O & Arm O_big) was resoundingly falsified. In 100% of seeds, the PVU gate rejected the 4th dimension (0% recruitment rate) due to high redundancy (correlation 0.85–0.99, violating the max_corr < 0.8 threshold) and poor predictability ratio (U_new 0.57–1.77, violating the U_new < 0.5 threshold).
- PHYSICAL CORRELATION IN 1D PHYSICS (iter_16.1): The failure of PVU gating revealed that in a 1D physics sandbox, all object coordinates are highly correlated by construction. A strict absolute correlation threshold of < 0.8 is physically incompatible with coordinate tracking in a shared 1D workspace, whereas the MDL consistency ratio (sim_new / sim_old) is highly robust.

## Refuted / Falsified
- PREDICTABILITY-VARIANCE-UNIQUENESS (PVU) IMMEDIATE SUITABILITY (iter_16.1): Direct PVU gating with hard hand-coded thresholds fails due to physical coordinate correlation and slow relative predictor convergence compared to encoder coordinates.

## Best Result
- Arm O_big (WUP-PVU, W=500, active during probation): Centroid Decoding MSE: 48.25, Test Sim Loss: 0.2071 (iter_16.1).
- Arm P_big (WUP-MDL, W=500, fully accepted): Centroid Decoding MSE: 49.57, Test Sim Loss: 0.2166 (iter_16.1).
- Arm P (WUP-MDL, W=100, fully accepted): Centroid Decoding MSE: 52.68, Test Sim Loss: 0.1959 (iter_16.1).

## In Progress
- Investigating local, prediction-independent gating metrics (e.g., spatial entropy or mutual information of centroids) to enable self-regulated growth without prediction-head dependency.

## Open Questions
- Can we define a local, prediction-independent MDL gating metric that operates directly on spatial entropy or mutual information of the encoder outputs?
- How does the WUP framework scale when transitioning to a larger d_max (e.g., d_max = 16) with multiple distractor objects?
- Can we dynamically adjust the probationary window W based on the convergence rate of the shadow dimension's predictive loss?
