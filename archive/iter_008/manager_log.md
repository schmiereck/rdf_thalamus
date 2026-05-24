# Research Manager Log - Iteration 008

## Iteration 008 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Introducing an unsupervised spatial localization bottleneck—implemented by minimizing the soft spatial variance of the active/recruited latent channel's activations over the 1D space—and coupling it with the output-as-input attention loop (where the attention query targets the channel's spatial centroid for closed-loop active probing) will stabilize the spatial alignment of the recruited dimension. This combined mechanism will:
1. Increase the average absolute Pearson correlation |r| between the recruited dimension's spatial centroid (and activation) and the true physical coordinate of the novel object to >= 0.60 across a 5-seed sweep, while stabilizing seed-to-seed variance such that no single seed falls below |r| = 0.45 (reversing the high variance of Phase 7, which ranged from 0.005 to 0.562).
2. Achieve a further reduction in post-hoc linear decoding MSE of the novel object's coordinate by at least 25% compared to the Phase 7 Active Probing baseline (reducing average MSE from 73.65 to < 55.0).
3. Reduce the soft spatial variance (spread) of the recruited dimension's activation on the novel object by at least 40% compared to the unconstrained Phase 7 active probing baseline.

**Proposed Falsification Criterion:**
The hypothesis will be proven FALSE if any of the following outcomes are observed over the 5-seed evaluation sweep:
1. The average absolute Pearson correlation |r| between the recruited dimension's activation/centroid and the novel object's position is < 0.60, or if any individual seed exhibits |r| < 0.45.
2. The average post-hoc linear decoding MSE of the novel object's coordinate is >= 55.0 (failing to achieve a 25% improvement over the 73.65 baseline).
3. The average soft spatial variance of the active/recruited channel is not reduced by at least 40% compared to the unconstrained active probing baseline.
4. The recruitment rate of the 4th dimension during active probing drops below 100% (5 out of 5 seeds).

**Proposed Method:**
1. Spatial Centroid & Variance Computation: For each channel c of the 1D latent representation, compute the spatial centroid x_mean_c = sum_i(i * softmax(a_{c, i})) and the soft spatial variance Var_c = sum_i((i - x_mean_c)^2 * softmax(a_{c, i})), where a_{c, i} is the activation of channel c at spatial index i of the 1D feature map.
2. Spatial Bottleneck Loss: Add a regularization term to the unsupervised loss objective of the plastic layer: L_spatial = lambda * Var_k, where k is the channel holding the Attention Token (the recruited/active channel representing the novel object) and lambda is a scaling hyperparameter. Stop gradients from updating other channels to prevent collapse.
3. Output-as-Input Active Probing: Modify the motor controller to receive the spatial centroid x_mean_k of the attended channel as the target coordinate, driving the continuous physical pointer to target and probe this localized entity.
4. Control vs. Experimental Sweep: Run a 5-seed comparison sweep comparing this "Spatial Bottleneck + Output-as-Input Active Probing" model (Experimental) against the "Unconstrained Active Probing" model from Phase 7 (Control).
5. Measurement: Measure and report the Pearson correlation |r| of the recruited channel with the true coordinate, the post-hoc linear coordinate decoding MSE, the spatial variance of activations, and the dimension recruitment rate.

---

## Iteration 008 -> Planner [Strategic Guidance]

# Manager's Note: Strategic Guidance for Iteration 8

Your proposed plan to introduce an unsupervised spatial localization bottleneck is highly promising and directly addresses the **Spatial Specialization Gap** identified in Phase 7. However, before you proceed to the execution phase, you must address several critical methodological and scientific risks to maintain our standards of empirical rigour.

---

### 1. The Construction-vs-Empirical Test: Spatial Preservation
You must distinguish between what is built into the architecture by construction and what is an empirical discovery:
*   **The Trap:** If your 1D latent feature map uses convolutional layers or locally-connected receptive fields, the spatial axis $i$ is *already* co-indexed with the physical input space by construction. Therefore, finding that the spatial centroid $x_{\text{mean}_k}$ correlates with the object's physical position is a mathematical inevitability of a localized activation, not an emergent phenomenon.
*   **The Genuine Empirical Question:** The real scientific inquiry is whether the spatial bottleneck regularization ($L_{\text{spatial}} = \lambda \cdot \text{Var}_k$) can force the recruited channel to become a highly localized, sharp tracker of the novel object **without triggering representation collapse, channel pruning, or performance degradation in predicting the overall dynamics.**

---

### 2. Pre-Registration Mandate & Collapse-Prevention Criteria
The Orchestrator will automatically write and commit your proposed hypothesis and falsification criteria to `src/pre_registration.md` before execution begins. You must ensure your sub-agents read and strictly adhere to this file. 

To prevent the model from satisfying your criteria through trivial/degenerate solutions, you must **explicitly append a 5th Falsification Criterion** to your pre-registration:
*   **Criterion 5 (Non-Collapse & Activity Threshold):** The hypothesis is falsified if the recruited channel $k$ undergoes representation collapse. Specifically, its mean activation magnitude must remain active (e.g., $\mathbb{E}[|a_{k}|] \ge 0.1 \cdot \frac{1}{C}\sum_c \mathbb{E}[|a_c|]$) and its temporal standard deviation must be non-trivial ($\sigma(x_{\text{mean}_k}) > 5.0$ pixels), proving it has not collapsed into an inactive channel or a static, non-responsive spatial spike.

---

### 3. Hyperparameter Hygiene ($\lambda$)
Introducing the spatial bottleneck loss weight $\lambda$ risks post-hoc parameter tuning. 
*   Do not sweep $\lambda$ continuously during the main evaluation and report only the best performing run as your primary result. 
*   You must pre-register a discrete search space (e.g., $\lambda \in \{0.01, 0.1, 1.0\}$) or provide a clear physical/scale-matching argument for your chosen $\lambda$ *before* viewing the final test performance. Report the sensitivity of your results to this parameter honestly.

---

### Recommended Action
Refine your draft plan to include **Criterion 5**, explicitly frame the spatial coordinate tracking as an architectural constraint rather than "spontaneous emergence," and proceed with the pre-registration and the 5-seed comparative sweep. Maintain our rigorous language discipline in your final reporting.

---

