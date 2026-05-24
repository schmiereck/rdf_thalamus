# Research Manager Log - Iteration 004

## Iteration 004 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
In a hierarchical two-layer representation network (Layer 1: local spatial features; Layer 2: global object trajectory prediction), implementing surprise-driven Thalamic Gating (Pillar D) with Z-score normalized surprise routing, a token-holding cooldown (C = 200 steps), and a lower-layer Stability Lock (L2 eligibility gated by L1 convergence threshold \theta_conv = 0.25) will:
1. Prevent input-drift collapse, resulting in a statistically significant reduction in the variance (standard deviation) of Layer 2 test prediction loss across seeds compared to a non-gated baseline (where both layers train continuously).
2. Achieve superior sample efficiency, reaching a stable Layer 2 prediction loss < 0.08 in fewer training steps than the non-gated baseline.
3. Enable stable self-sustained tracking: when transitioning from externally primed queries (biased towards a target object color) to self-generated queries (using L2's own previous state prediction), the system will maintain target tracking overlap > 0.85 and reduce prediction loss on the target object by at least 15% compared to the non-gated standard JEPA.

**Proposed Falsification Criterion:**
The hypothesis will be falsified if any of the following occur over a 5-seed evaluation suite:
1. The gated model fails to show a lower standard deviation of L2 test prediction loss across seeds compared to the non-gated model (p >= 0.05 via Levene's test, or higher raw standard deviation).
2. The gated model requires more or equal training steps than the non-gated model to reach L2 prediction loss < 0.08, or fails to reach this error level entirely.
3. During the self-generated attention phase, target tracking overlap falls below 0.80, or the test prediction loss on the target object is not at least 15% lower than that of the standard non-gated JEPA baseline.

**Proposed Method:**
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

## Iteration 004 -> Planner [Strategic Guidance]

### Strategic Guidance: Thalamic Gating & Inter-Layer Stability

We are entering Phase 2 (Thalamic Gating), which is a critical transition. While the representation base from Phase 1 is solid, moving to a closed-loop attention and plasticity-gating regime introduces serious risks of numerical instability and self-referential evaluation. 

Review the following strategic corrections before committing the hypothesis to `src/pre_registration.md`.

---

### 1. The Z-Score Noise Bottleneck (Numerical Stability)
As layers converge, their local temporal-prediction errors (surprise) will decrease. If a layer achieves high prediction accuracy, its surprise variance will approach zero. Standard Z-score normalization ($\frac{E - \mu}{\sigma}$) will explode or become dominated by micro-fluctuations (computational noise at machine epsilon), leading to chaotic attention-token routing.
*   **Directive:** You must reject a naive Z-score. Demand a soft-normalization scheme that incorporates a safety epsilon ($\epsilon \ge 1e-4$) and a minimum variance floor. 
*   **Falsification Guard:** If your model's attention routing becomes chaotic or uniform as training progresses, treat this as a failure of the normalization scheme rather than a feature of the gating.

### 2. Fragility of the Absolute Stability Lock Threshold
Your proposed Stability Lock gates L2 training on $L1$ surprise falling below $\theta_{conv} = 0.25$. Because our sandbox features continuous environmental variation (varying object counts, masses, and velocities), the absolute scale of prediction loss is not a physical constant. An absolute threshold of $0.25$ is highly fragile and prone to locking L2 out indefinitely under complex environments, or releasing it too early under simple ones.
*   **Directive:** Reframe the Stability Lock threshold. It should be a relative metric (e.g., L1's surprise trend stabilizing, such that its sliding-window variance falls below a fraction of its initial variance, or a ratio-based convergence criterion). If you retain an absolute threshold, you must pre-register an independent physical justification for why $0.25$ is invariant across your environmental sweeps.

### 3. Avoiding the Construction Trap in "Tracking Overlap"
Your third hypothesis claims the system will maintain "target tracking overlap > 0.85" when transitioning to self-generated queries. 
*   **Skeptical Critique:** If "tracking overlap" is calculated using the L2 latent state itself (which is trained to predict the target), this is a constructional identity, not an empirical discovery. It merely states that the model's prediction correlates with its own prediction history.
*   **Directive:** Define "target tracking overlap" using an **empirically independent physical metric**. Specifically, you must map the attention locus (e.g., the spatial coordinate of the receptive field or attention token) directly to the ground-truth physical coordinate of the target object in the 1D sandbox. Tracking is only verified if the spatial gating tracks the physical object's trajectory better than the non-gated baseline.

---

### Pre-Registration Notice
The Orchestrator will automatically write and commit your finalized hypothesis and quantitative falsification criteria to `src/pre_registration.md` before execution. Ensure your mathematical formulations are precise, avoid hyperbolic language, and explicitly address the three points above in your final pre-registration and code design.

---

