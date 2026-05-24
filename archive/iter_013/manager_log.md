# Research Manager Log - Iteration 013

## Iteration 013 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Integrating explicit pixel-position encodings (Linear Normalized or Sinusoidal) into the input of the convolutional backbone in the Thalamus architecture (increasing input channels from 3 to 4 or more) under Closed-Loop Thalamic Subsumption Motorics (CLTS) will resolve the marginal spatial representation drift observed under active control. Specifically, the centroid decoding MSE of the novel object under CLTS control will decrease from the Phase 12 baseline of 85.85 to below 75.0, while maintaining a soft spatial variance below 10.0 and a post-collision test simulation loss below 0.050.

**Proposed Falsification Criterion:**
The hypothesis will be falsified if:
1. The centroid decoding MSE of the novel object under active CLTS control with positional encoding is >= 75.0; OR
2. The soft spatial variance of the coordinate encoder exceeds 10.0; OR
3. The post-collision test simulation loss exceeds 0.050.

**Proposed Method:**
1. Modify `src/thalamus.py` to support explicit positional encodings at the input level:
   - Arm H (Linear Pos): 4 input channels, where the 4th channel is the linear normalized position `pos_i = i / 127.0`.
   - Arm I (Sinusoidal Pos): 7 input channels, with 3 RGB channels and 4 sinusoidal positional embeddings (using frequencies of 10 and 100).
2. Maintain the same CLTS active control scheme in `src/motor.py` as established in Phase 12.
3. Create `src/run_phase13_experiments.py` to run a 5-seed comparative sweep of Arm H (Linear Pos) and Arm I (Sinusoidal Pos) against the original RGB-only CLTS baseline.
4. Log and evaluate post-collision test simulation loss, centroid decoding MSE of the novel object (generalization test), soft spatial variance, and pointer spatial entropy.

---

## Iteration 013 -> Planner [Strategic Guidance]

### Manager's Note: Strategic Guidance for Phase 13

#### 1. The Position-Shortcut Risk (Construction-vs-Empirical Test)
By injecting explicit spatial coordinate channels (`pos_i`) directly into the input, we risk introducing a **constructional shortcut**. A linear probe or the soft-argmax stream can trivially exploit this channel to achieve low centroid decoding MSE without actually learning the underlying physics or object features. 
*   **The Danger:** The network might over-index on the positional channel and become "semantically blind" to non-spatial features (such as color, size, or mass), leading to a failure in temporal prediction when collisions alter velocities.
*   **Strategic Directive:** You must verify that the model does not achieve low centroid decoding MSE at the expense of predictive capacity. If the test simulation loss of the positional-encoded model degrades or fails to outperform the RGB-only baseline, we must treat this as a negative result (representation collapse into a trivial coordinate pass-through).

#### 2. Rigorous Falsification & Pre-Registration Mandate
The proposed plan is approved to proceed to execution **only** under a strengthened, highly quantitative pre-registration. The Orchestrator will write this to `src/pre_registration.md` before the execution phase begins, and all sub-agents must read and strictly adhere to it.

We are tightening your falsification criteria to protect against the position-shortcut:
*   **Criterion 1 (Coordinate Accuracy):** The centroid decoding MSE of the novel object under active CLTS control must be **$< 75.0$** (reflecting a real mitigation of the active-perception drift).
*   **Criterion 2 (Spatial Tightness):** The soft spatial variance of the coordinate encoder must remain **$< 10.0$**.
*   **Criterion 3 (Predictive Integrity):** The post-collision test simulation loss must remain **$< 0.050$** (proving the network still learns temporal dynamics and physical properties, rather than just copying coordinates).
*   **Criterion 4 (Generalization / Robustness):** The positional-encoded models (Arm H/I) must achieve a post-collision test simulation loss that is **statistically non-inferior** to the RGB-only baseline (Arm G) under a Levene or t-test ($p > 0.05$). If adding position information hurts prediction, the hypothesis is falsified.

#### 3. Execution & Architectural Hygiene
*   Implement both **Arm H (Linear Normalized)** and **Arm I (Sinusoidal Embeddings)** inside `src/thalamus.py` as clean, parameterized options. Do not break backward compatibility with the raw RGB input mode.
*   Ensure that the linear probe for centroid decoding is trained on the latents *after* the representation has been subjected to active CLTS physical perturbations, to accurately measure drift under continuous closed-loop control.
*   Document the results using restrained scientific language (e.g., "is compatible with", "provides evidence for mitigating representational drift"). Avoid declaring a "breakthrough" if the improvement is a trivial consequence of the input modification.

---

