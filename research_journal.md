# Research Manager Journal & Strategic State

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 9 (Dynamic Surprise-Modulated Curriculum & Rate-Limiter) Complete.
*   **Active Direction:** Transitioning from single-channel spatial-regularized latents to a Dual-Channel Latent Space (Architectural Split) to resolve the fundamental trade-off between spatial localization and temporal predictive capacity.
*   **Confidence Score:** 95% (Bolstered by rigorous systematic sweeps, explicit falsification of pre-registered thresholds, and a clear architectural path forward).

## 2. Strategic Insights & Lessons Learned
*   **The Inviolable Trade-Off of Single-Channel Latents:** Phase 9 results have definitively established that in a single-channel latent space, raw temporal prediction accuracy and spatial coordinate localization are mutually opposing optimization forces. Even with a dynamic, surprise-modulated curriculum (DSMC), forcing spatial localization degraded test prediction accuracy by 22.8%, failing the <15% safeguard.
*   **Curriculum-Driven Stability:** DSMC with a step-clipping rate limiter (+/-0.002) successfully prevents the optimization oscillations common to active feedback loops. It achieved a 0.0% representation collapse rate across all seeds, demonstrating that dynamically adjusting regularization pressure in response to temporal surprise spikes is a robust way to guide structural representation changes.
*   **The Limits of Adaptive Schedules:** While DSMC prevents collapse and provides a smoother Pareto-like trade-off than static aggressive penalties, it could not beat a statically tuned gentle bottleneck (decoding MSE of 73.46 vs 69.11). This indicates that dynamic coupling of temporal surprise to spatial regularization parameter schedules introduces sub-optimal local minima during joint optimization.

## 3. Loop & Bottleneck Detection
*   **The "Disentanglement-Accuracy" Loop:** Attempts to resolve this loop via dynamic curricula (DSMC) have reached their mathematical limit within a single-channel architecture. The representational capacity required for precise temporal dynamics modeling is fundamentally compromised when forced to compress into low-entropy spatial coordinates.
*   **Mitigation Strategy:** Decouple the pathways. We must physically separate the latent space into:
    1. A low-capacity, highly bottlenecked *spatial tracking channel* (mapping coordinates).
    2. A high-capacity, unregularized *predictive dynamics channel* (mapping temporal transitions).

## 4. Alternate Research Paths
*   **Dual-Channel Latent Space (Architectural Split):** (Now promoted to Primary Next Path) Partition the latent space into a soft-argmax spatial coordinate tracker and a recurrent temporal predictive model.
*   **Surprise-Gated Dynamic Routing:** Instead of using surprise to modulate a global regularization parameter, use it as a routing gate to determine which channel (spatial vs. temporal) receives the attention token and plasticity updates.