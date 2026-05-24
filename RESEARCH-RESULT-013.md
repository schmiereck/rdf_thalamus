# RDF Milestone Review — Iteration 013 — Null Result: Input-Level Positional Encodings under Active Control

## 1. Pre-Declared Hypothesis and Falsification Criterion
*   **Hypothesis:** Explicit pixel-position encoding (Linear or Sinusoidal) injected at the input level will improve spatial centroid decoding MSE under active CLTS control from 85.85 to below 75.0, thereby mitigating the Active-Perception Drift Penalty without degrading predictive performance.
*   **Falsification Criterion:** Falsified if spatial centroid decoding MSE remains ≥ 75.0, or if temporal prediction error (post-collision test simulation loss) increases significantly compared to the RGB-only CLTS baseline (0.0483).

## 2. Experimental Protocol
*   **Environment:** 128 RGB(P) pixels, 1D physics sandbox, 3 moving objects under active CLTS (Closed-Loop Thalamic Subsumption) motor control.
*   **Configurations Evaluated:**
    *   *Control:* RGB-only CLTS (Phase 12 baseline).
    *   *Experimental Arm 1 (Linear):* RGB + Linear Positional Encoding (`pos_i = i / 127.0`).
    *   *Experimental Arm 2 (Sinusoidal):* RGB + Sinusoidal Positional Encoding (multiple frequencies).
*   **Execution:** 5-seed matched sweep, evaluated over standard training step envelope. All parameters not related to the input dimension were held strictly constant across arms.

## 3. Observed Quantities
*   **Post-Collision Test Sim Loss:**
    *   *Control (RGB-only):* 0.0483 [Standardized L2 Loss]
    *   *Linear Position:* 0.059 [Standardized L2 Loss] (22.15% degradation)
    *   *Sinusoidal Position:* 0.091 [Standardized L2 Loss] (88.40% degradation)
*   **Centroid Decoding MSE:**
    *   Both experimental configurations failed to show any statistically significant reduction below the 75.0 target threshold, remaining statistically indistinguishable from the baseline active drift level (MSE ~85).
*   **Mechanistic Observation:** Directly injecting raw coordinate channels into the visual backbone created a "position shortcut." The network satisfied the soft spatial bottleneck loss by trivially copying raw coordinates, rendering it semantically blind to actual physical interactions (such as collisions and velocities).

## 4. Verdict
*   **Refuted.** The experimental results conclusively refute the hypothesis. Input-level positional encodings do not resolve the active-perception coordinate drift and instead lead to severe degradation in the model's ability to learn temporal physical dynamics.

## 5. Construction-vs-Empirical Note
The degrade in temporal predictive performance is a genuinely empirical finding. While the spatial bottleneck loss *can* mathematically be satisfied by static coordinates, the exact manner in which the optimization landscape collapses (by completely prioritizing static input-level coordinates over dynamic motion features) provides critical empirical insight into the competitive dynamics between spatial regularization and temporal prediction.

## 6. Limitations
This result demonstrates that *input-level* spatial grounding is highly detrimental to self-supervised predictive networks with spatial constraints. However, it does not rule out *representation-level* constraints, such as contrastive coordinate regularization or temporal anchoring losses, which constrain the latents directly rather than introducing a bypass in the raw input channels.