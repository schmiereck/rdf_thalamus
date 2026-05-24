# RDF Scientific Pre-Registration

*   **Iteration:** 011
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
In a dual-stream decoupled architecture, grounding the coordinate stream ($z^{coord}$) during an initial joint-training phase, followed by weight freezing (consolidation) and stop-gradient decoupling of $z^{coord}$ (Progressive Decoupling with Representational Consolidation, PDRC), will prevent semantic blindness and representation collapse while preserving high spatial localization and predictive capacity. Specifically, compared to the failed immediate-decoupling baseline (Arm D), PDRC (Arm E) will achieve a 0% collapse rate, average soft spatial variance $\le 100.0$, and a test prediction loss ratio vs the joint-training baseline (Arm A) of $\le 1.15$.

**Generalization Penalty for Arm E (PDRC)**:
While PDRC (Arm E) preserves spatial localization on known objects, freezing the coordinate head weights breaks plasticity. Therefore, upon introducing a 4th novel object in Stage 2 (after $T_{ground}=1500$), Arm E's coordinate head cannot adapt to locate the novel object, resulting in a severe generalization penalty.

**Arm F (Non-Parametric Soft-Argmax Projection) Evaluation**:
By deriving coordinates directly as a differentiable, non-parametric projection (soft-argmax) over the predictive dynamics channel, Arm F avoids gradient interference, remains fully grounded, and maintains full plasticity/adaptability to novel objects without needing a non-biological freezing schedule. Thus, Arm F will avoid both the representation collapse of Arm D and the generalization penalty of Arm E, achieving high coordinate-centroid correlation and low decoding MSE on the 4th novel object while maintaining strong prediction capabilities.

## 2. Falsification Criteria
The original PDRC hypothesis will be falsified if any of the following occur over a 5-seed comparative sweep:
1. The representation collapse rate of Arm E is $> 0\%$ (where collapse is defined as test prediction loss $> 1.0$ or soft spatial variance $> 500.0$).
2. The average soft spatial variance of Arm E is $> 100.0$.
3. The ratio of the average test prediction loss of Arm E to Arm A is $\ge 1.15$.
4. The absolute correlation ($r$) of the coordinate stream with the physical object centroids is $< 0.25$, or the centroid decoding MSE is $> 85.0$ (signaling semantic blindness).

**New/Updated Falsification Criteria**:
5. **Generalization Penalty for Arm E (PDRC)**: Arm E must be falsified if, upon introducing a 4th novel object in Stage 2 (after $T_{ground}=1500$), its coordinate-centroid correlation drops below $0.25$ or its centroid decoding MSE exceeds $85.0$ (proving that freezing weights breaks adaptation to novelty).
6. **Arm F (Non-Parametric Soft-Argmax Projection)**: Arm F's hypothesis will be falsified if, on Stage 2 (the 4th novel object), its coordinate-centroid correlation is $< 0.50$, its centroid decoding MSE is $> 50.0$, or its prediction loss ratio vs Arm A is $\ge 1.20$.

## 3. Proposed Method
1. Extend the existing codebase in `src/` to support Arm E and Arm F:
   - Arm E (PDRC): Jointly train in Stage 1 ($0 \le t < T_{ground}$), then freeze coordinate head weights (`encoder.conv_spatial_coord`) and inject stop-gradient before the predictor in Stage 2 ($t \ge T_{ground}$).
   - Arm F (Non-Parametric Soft-Argmax Projection): Use a single-backbone encoder (`NonParametricJEPASpatial`) that extracts spatial feature maps. Derive coordinates non-parametrically using a soft-argmax operator over the predictive dynamics channel, and extract dynamics from the same feature maps, avoiding weight-based gradient conflicts.
2. Run a 5-seed sweep comparing Arm A, Arm C, Arm D, Arm E, and Arm F on the 1D physics sandbox.
   - Stage 1: Train passively on $N=3$ objects for 1500 steps.
   - Stage 2: Train actively under probing on $N=4$ objects for steps 1501 to 3000 (PDController tracks the 4th object, and the bottleneck targets channel 3).
3. Evaluate all arms on a fresh test set of 200 passive steps with $N=4$ objects.
4. Measure and log prediction loss, soft spatial variance, centroid decoding MSE, and coordinate-centroid correlation across all arms. Generate comparisons, summary reports, and plots.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
