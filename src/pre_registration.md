# RDF Scientific Pre-Registration

*   **Iteration:** 004
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
In a hierarchical two-layer representation network (Layer 1: local spatial features; Layer 2: global object trajectory prediction), implementing surprise-driven Thalamic Gating (Pillar D) with Z-score normalized surprise routing, a token-holding cooldown (C = 200 steps), and a lower-layer Stability Lock (L2 eligibility gated by L1 convergence threshold \theta_conv = 0.25) will:
1. Prevent input-drift collapse, resulting in a statistically significant reduction in the variance (standard deviation) of Layer 2 test prediction loss across seeds compared to a non-gated baseline (where both layers train continuously).
2. Achieve superior sample efficiency, reaching a stable Layer 2 prediction loss < 0.08 in fewer training steps than the non-gated baseline.
3. Enable stable self-sustained tracking: when transitioning from externally primed queries (biased towards a target object color) to self-generated queries (using L2's own previous state prediction), the system will maintain target tracking overlap > 0.85 and reduce prediction loss on the target object by at least 15% compared to the non-gated standard JEPA.

## 2. Falsification Criterion
The hypothesis will be falsified if any of the following occur over a 5-seed evaluation suite:
1. The gated model fails to show a lower standard deviation of L2 test prediction loss across seeds compared to the non-gated model (p >= 0.05 via Levene's test, or higher raw standard deviation).
2. The gated model requires more or equal training steps than the non-gated model to reach L2 prediction loss < 0.08, or fails to reach this error level entirely.
3. During the self-generated attention phase, target tracking overlap falls below 0.80, or the test prediction loss on the target object is not at least 15% lower than that of the standard non-gated JEPA baseline.

## 3. Proposed Method
1. Architecture Extension: In `src/models.py` (or a new `src/thalamus.py`), implement a two-layer DynamicJEPA:
   - L1 (Local Segment Predictor): Processes visual segments, predicts its next latent state, and uses VICReg regularization (cov_weight=25.0) with GDASR recruitment.
   - L2 (Object Tracker): Aggregates L1 latents, predicts global entity trajectories, and uses identical collapse prevention and recruitment.
2. Thalamic Gating Mechanism:
   - Surprise Watchdog: Compute local surprise E_l(t) = ||z_l(t) - \hat{z}_l(t)||^2 for l \in {L1, L2}.
   - Z-Score Normalization: Maintain EMA of mean and variance for each layer's surprise to compute normalized surprise \bar{E}_l(t).
   - Token Routing: Route Attention Token T(t) = argmax \bar{E}_l(t) with a 200-step cooldown timer.
   - Stability Lock: Prevent L2 from receiving the token or updating weights until L1's running surprise falls below \theta_conv = 0.25.
   - Plasticity Gating: Freeze gradients and disable dimension recruitment for the layer not holding the token.
3. Priming & Self-Generation:
   - Implement external priming (biasing surprise weight towards a specific object's color) for the first 1500 steps.
   - Transition to self-generated priming by feeding L2's trajectory predictions back as the query vector.
4. Control & Evaluation:
   - Run 5 seeds for: (a) proposed Thalamic Gated model, (b) non-gated multi-layer control, and (c) single-layer baseline (B1).
   - Log test prediction loss, loss variance, attention tracking accuracy, and token transition histories.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
