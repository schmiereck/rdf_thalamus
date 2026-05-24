# Research Manager Log - Iteration 010

## Iteration 010 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
In the Thalamus architecture, decoupling each recruited representation channel into a dual-stream latent space—consisting of a highly localized Spatial Coordinate Stream ($z^{coord}$, subject to a strong spatial variance minimization penalty $\lambda$) and a parallel Temporal Dynamics Stream ($z^{dyn}$, free of the spatial penalty and optimized for transition dynamics)—resolves the fundamental trade-off between spatial localization and predictive capacity. Specifically, this Dual-Stream Decoupled Thalamus (DSDT) architecture will achieve a soft spatial variance $\le 75.0$ (matching the spatial localization of the DSMC/Strong Bottlenecks) while simultaneously reducing the test simulation prediction loss by at least 15% relative to the single-stream DSMC (Arm C of Phase 9), thereby achieving a test simulation loss ratio vs. the Gentle Bottleneck (Arm A) of $< 1.10$.

**Proposed Falsification Criterion:**
The hypothesis will be falsified if, across a 5-seed comparative sweep, any of the following occur:
1. The mean soft spatial variance of the Spatial Coordinate Stream ($z^{coord}$) is $> 75.0$.
2. The overall test simulation loss of the DSDT model is not reduced by at least 15% compared to Arm C of Phase 9 (i.e., the ratio of DSDT test simulation loss to Arm A's test simulation loss is $\ge 1.10$).
3. The Dual-Stream representation suffers from structural instability or collapse (defined as a representation collapse rate $> 0.0\%$ across the 5 training seeds).

**Proposed Method:**
1. **Dual-Stream Architecture Implementation**: Modify the representation node structure in the model definition files (e.g., `src/models.py` or `src/thalamus.py`). For each recruited dimension, split the latent output into:
   - $z^{coord}_t \in \mathbb{R}^1$: computed via a highly constrained spatial softmax/centroid bottleneck and regularized by the adaptive spatial variance loss $\mathcal{L}_{var} = \lambda(t) \cdot \text{Var}(z^{coord}_t)$.
   - $z^{dyn}_t \in \mathbb{R}^d$: computed without spatial compression, preserving high-frequency visual and motion features.
2. **Predictive Coupling and Stop-Gradients**: Implement the temporal prediction network such that $z^{dyn}_t$ is used to predict $z^{coord}_{t+1}$ and $z^{dyn}_{t+1}$. Apply a stop-gradient operator on $z^{coord}_t$ during the temporal prediction backpropagation. This ensures that temporal prediction gradients do not bleed into and dilute the spatial localization of the coordinate stream, while allowing the dynamics stream to leverage spatial coordinate context.
3. **Control and Experimental Setup**:
   - **Arm A (Control 1)**: Gentle Spatial Bottleneck (re-run or compared with Phase 9 baseline).
   - **Arm C (Control 2)**: Single-stream DSMC (from Phase 9).
   - **Arm D (Experimental)**: Dual-Stream Decoupled Thalamus (DSDT) with adaptive spatial variance penalty $\lambda(t)$.
4. **Multi-Seed Evaluation Sweep**: Train all arms over a 5-seed sweep under the parameterized 1D physics environment (including the $N=3 \to N=4$ object transition event).
5. **Metrics Logging**: Log soft spatial variance, centroid decoding MSE (using a linear probe as a diagnostic check of coordinate quality), test simulation prediction loss, and representation collapse rates.

---

## Iteration 010 -> Planner [Strategic Guidance]

# Manager's Note: Strategic Guidance for Phase 10 (Dual-Stream Decoupled Thalamus)

The transition to a Dual-Channel Latent Space (Architectural Split) is highly logical. It directly addresses the fundamental Pareto trade-off identified in Phase 9. However, splitting the architecture introduces new failure modes. You must approach this implementation with high skepticism to ensure we are observing genuine emergent physics rather than structural artifacts.

Before proceeding, note that **the Orchestrator will automatically write and commit your finalized hypothesis and falsification criteria to `src/pre_registration.md`** before execution begins. Your sub-agents must read and strictly adhere to this file.

---

### 1. The "Semantic Blindness" Risk (Stop-Gradient Critique)
Your proposed plan applies a `stop-gradient` operator on the Spatial Coordinate Stream ($z^{coord}$) during temporal prediction backpropagation. While this protects the spatial bottleneck from being diluted by prediction gradients, **it risks decoupling the coordinate stream from the physics of the environment entirely.** 
*   **The Danger:** Without predictive feedback, $z^{coord}$ might become "semantically blind"—merely performing a spatial auto-encoding of the most prominent static pixel clumps or high-frequency noise, rather than tracking functionally significant physical entities.
*   **Requirement:** Your pre-registration must include a diagnostic threshold for the **linear probe coordinate MSE** on $z^{coord}$. If the decoupled $z^{coord}$ cannot be decoded into accurate object centroids (e.g., failing to match or exceed the coordinate MSE of Arm A), the architecture has failed to ground its spatial channel, regardless of what the test loss shows.

### 2. The Construction-vs-Empirical Test for Dual-Streams
To avoid a trivial "constructional" success (i.e., "we built two channels and they behave differently because we regularized them differently"), you must prove that **information actually flows constructively between them**.
*   **The Test:** Your temporal prediction network should use $z^{dyn}_t$ *and* $z^{coord}_t$ jointly to predict $z^{dyn}_{t+1}$. 
*   **Falsification Guard:** If a control run that zero-masks the $z^{coord}_t$ input to the predictor performs *just as well* as the full DSDT model, then the dynamics stream is ignoring the spatial stream, and the "dual-stream integration" is an illusion.

### 3. Strict Parameter-Tuning Hygiene
To ensure a fair comparison with Phase 9:
*   Use the **exact same evaluation harness**, the same 5 seeds, and identical 1D physics sandbox parameters (including the exact step count for the $N=3 \to N=4$ transition).
*   Do not post-hoc tune the learning rates, the capacity of the dynamics channel, or the VICReg covariance weights specifically to favor Arm D. The baseline parameters must remain frozen. Use the precise quantitative criteria you outlined in your draft.

---

