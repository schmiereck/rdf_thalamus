# Current Research State
Phase: Phase 9 Complete (DSMC Sweep Evaluated and Audited)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 9 goal was to implement, audit, and evaluate an adaptive, surprise-modulated spatial variance regularization weight $\lambda(t)$ under a 5-seed sweep with strict controller stability rate limiting and a temporal prediction safeguard.

## Confirmed
- PARETO-LIKE COGNITIVE BALANCE (iter_009.2): The DSMC curriculum (Arm C) successfully bridges the gap between gentle and strong bottlenecks. It achieves a highly localized spatial coordinate representation (soft spatial variance of 70.61 vs. 118.76 for Gentle Arm A) while preserving most of the predictive capacity (decoding MSE of 73.46 vs. 106.87 for Strong Arm B).
- STABLE CONTROLLER DYNAMICS (iter_009.2): Implementing a step-to-step clipping of $\pm 0.002$ on $\lambda(t)$ successfully prevents controller oscillations and yields 0.0% representation collapse across all 15 training runs (100% stability).
- HIGHLY SYSTEMATIC SURPRISE COUPLING (iter_009.2): As hypothesized, surprise spikes during the $N=3 \to N=4$ object transition event, suppressing $\lambda(t) \to 0$ to allocate unconstrained prediction learning capacity. As the environment becomes predictable, $\lambda(t)$ smoothly ramps up to compress and localize the representation.

## Refuted
- REFUTED (iter_009.2): A surprise-modulated curriculum can resolve the localization-prediction trade-off without any prediction accuracy penalty. Instead, Arm C's test simulation loss was 22.8% higher than Arm A's, triggering the Temporal Prediction Safeguard falsification criterion. Spatial localization always incurs a minor predictive capacity cost.
- REFUTED (iter_009.2): The local temporal surprise of a newly recruited channel in a dynamic environment can easily decay to near-zero. Instead, a steady-state level of surprise remained, which restricted the mean final penalty weight $\lambda_T$ to 0.0380 (failing the sanity check threshold of $\ge 0.05$).

## Best Result
- Gentle Bottleneck (Arm A) Centroid Decoding MSE: 69.11, Spatial Variance: 118.76
- Strong Bottleneck (Arm B) Centroid Decoding MSE: 106.87, Spatial Variance: 60.29
- DSMC Curriculum (Arm C) Centroid Decoding MSE: 73.46, Spatial Variance: 70.61
- Representation Collapse Rate (Arm C): 0.0%

## In Progress
- Decoupling spatial coordinate representation from temporal dynamic prediction using multi-scale networks.

## Open Questions
- Can we improve the curriculum by dynamically updating the surprise baseline $S_0$ based on running environmental entropy?
- Does a multi-scale temporal prediction architecture (recurrent and linear delay lines) completely decouple coordinate representation from temporal modeling?
- Does integrating spatial coordinate feedback into the policy's action generation naturally guide exploration to accelerate coordinate learning?
