# Research Manager Journal & Strategic State

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 3 Complete (Motor & Closed Loop). Moving to Phase 4 (Generalization & Reporting).
*   **Active Direction:** Evaluating cross-environment generalization, causal sensitivity, and parameter-tuning stability under Phase 4. Transitioning from raw spatial tracking metrics to information-theoretic evaluation metrics due to the active-perception entropy trade-off.
*   **Confidence Score:** 85% (Up from 80%, reflecting the successful execution of the 5-seed closed-loop verification and the honest, rigorous falsification of the spatial tracking overlap metric).

## 2. Strategic Insights & Lessons Learned
*   **The Active-Perception Entropy Trade-off:** Active physical intervention via Subsumption Motorics (pointer acceleration and push actions) drastically improves causal world modeling, reducing post-collision prediction error by 75.0%. However, this active exploration inherently increases environmental entropy. Actively perturbing physical entities displaces them, causing low spatial overlap between the controller and the entities.
*   **Falsification of Spatial Overlap as an Active Metric:** High spatial overlap is a valid metric only for passive observation or highly constrained tracking. In a closed-loop interactive system, forcing the agent to maintain high spatial overlap restricts its exploratory capacity. Causal predictive accuracy must be decoupled from spatial proximity metrics.

## 3. Loop & Bottleneck Detection
*   **Identified Bottleneck:** Standard spatial evaluation metrics fail when the agent acts upon the environment.
*   **Mitigation Strategy for Next Phase:** For Phase 4 evaluation, we will substitute simple spatial tracking overlap with transfer entropy or mutual information between the agent's action history and the entity's subsequent state transitions, isolating true causal modeling from physical co-location.

## 4. Alternate Research Paths
*   **Causal Sensitivity Analysis (Priority):** Systematically perturbing hidden environmental variables (e.g., object mass and friction coefficients) in Phase 4 to verify if the latent representations have truly encoded mechanical invariants or merely memorized trajectory kinematics.
*   **Information-Theoretic Active Metrics:** Formulating an attention-to-entropy ratio to quantify how efficiently the agent targets high-surprise areas without inducing chaotic, unmodelable environmental states.