# Research Manager Log - Iteration 011

## Iteration 011 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
In a dual-stream decoupled architecture, grounding the coordinate stream ($z^{coord}$) during an initial joint-training phase, followed by weight freezing (consolidation) and stop-gradient decoupling of $z^{coord}$ (Progressive Decoupling with Representational Consolidation, PDRC), will prevent semantic blindness and representation collapse while preserving high spatial localization and predictive capacity. Specifically, compared to the failed immediate-decoupling baseline (Arm D), PDRC (Arm E) will achieve a 0% collapse rate, average soft spatial variance $\le 100.0$, and a test prediction loss ratio vs the joint-training baseline (Arm A) of $\le 1.15$.

**Proposed Falsification Criterion:**
The hypothesis will be falsified if any of the following occur over a 5-seed comparative sweep:
1. The representation collapse rate of Arm E is $> 0\%$ (where collapse is defined as test prediction loss $> 1.0$ or soft spatial variance $> 500.0$).
2. The average soft spatial variance of Arm E is $> 100.0$.
3. The ratio of the average test prediction loss of Arm E to Arm A is $\ge 1.15$.
4. The absolute correlation ($r$) of the coordinate stream with the physical object centroids is $< 0.25$, or the centroid decoding MSE is $> 85.0$ (signaling semantic blindness).

**Proposed Method:**
1. Extend the existing codebase in `src/` to support Arm E: Progressive Decoupling with Representational Consolidation (PDRC).
2. Implement a two-stage training schedule:
   - Stage 1 (Grounding, $0 \le t < T_{ground}$): Jointly train both $z^{coord}$ (with spatial bottleneck) and $z^{dyn}$ streams with prediction gradients.
   - Stage 2 (Decoupled, $t \ge T_{ground}$): Inject a stop-gradient at the output of $z^{coord}$ before it feeds into the temporal predictor, and freeze the weights of the coordinate stream encoder. The dynamics stream and predictor continue to train.
3. Run a 5-seed sweep comparing Arm A (Gentle Bottleneck), Arm C (DSMC), Arm D (DSDT), and Arm E (PDRC) on the 1D physics sandbox.
4. Measure and log test prediction loss, soft spatial variance, centroid decoding MSE, and coordinate-centroid correlation across all arms.

---

## Iteration 011 -> Planner [Strategic Guidance]

# Manager's Note: Strategic Guidance for Iteration 11

## 1. The Plasticity-Adaptability Conflict in PDRC (Scientific Critique)
The proposed Progressive Decoupling with Representational Consolidation (PDRC - Arm E) introduces a critical architectural contradiction: **hard-freezing the coordinate encoder weights after $T_{ground}$ violates the core principles of the Thalamus architecture.** 
*   **The Adaptability Bottleneck:** Under the mandatory environmental variations (Section 3) and generalization metrics (Section 6), the model must adapt to novel objects (such as transition from $N=3$ to $N=4$ objects). If the coordinate encoder weights are completely frozen in Stage 2, the network will be physically incapable of recruiting new coordinate dimensions or adapting its spatial representations to novel entity dynamics.
*   **The Lifelong Learning Violation:** A step-function epoch trigger ($T_{ground}$) is a non-biological engineering patch rather than an emergent, dynamical system.

## 2. Strategic Redirection: Soft-Grounding vs. Non-Parametric Projection
To maintain scientific rigour and adhere to our architectural pillars, you must evaluate two distinct paradigms for resolving the "semantic blindness vs. interference" bottleneck:
*   **Arm E: Progressive Decoupling (PDRC):** Run this as proposed, but **you must explicitly test it under environmental variation/novel object introduction during Stage 2**. This will empirically expose whether freezing weights induces a fatal adaptability bottleneck.
*   **Arm F: Non-Parametric Soft-Argmax Projection:** Implement a comparison arm where the spatial coordinate $z^{coord}$ is derived directly as a differentiable, non-parametric projection (e.g., via a spatial soft-argmax operator) over the predictive dynamics channel $z^{dyn}$. Because there are no separate coordinate encoder weights, there is **zero gradient interference**, yet the spatial coordinates remain completely grounded in the predictive dynamics stream.

## 3. Pre-Registration Mandate & Falsification Update
The Orchestrator will automatically write your pre-registration to `src/pre_registration.md`. You must update your falsification criteria to reflect the adaptability test:
1.  **Generalization Penalty:** Arm E (PDRC) must be falsified if, upon introducing a 4th novel object in Stage 2, its coordinate-centroid correlation drops below $0.25$ or its centroid decoding MSE exceeds $85.0$ (proving that freezing weights breaks adaptation to novelty).
2.  **Comparative Rigour:** Quantify the performance of Arm F (Soft-Argmax Projection) against Arm E (PDRC) and Arm A (Gentle Bottleneck). If Arm F achieves comparable spatial localization without the need for a non-biological freezing schedule, it must be preferred.

---

