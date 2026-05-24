# RDF Scientific Pre-Registration

*   **Iteration:** 013
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Integrating explicit pixel-position encodings (Linear Normalized or Sinusoidal) into the input of the convolutional backbone in the Thalamus architecture (increasing input channels from 3 to 4 or more) under Closed-Loop Thalamic Subsumption Motorics (CLTS) will resolve the marginal spatial representation drift observed under active control. Specifically, the centroid decoding MSE of the novel object under CLTS control will decrease from the Phase 12 baseline of 85.85 to below 75.0, while maintaining a soft spatial variance below 10.0 and a post-collision test simulation loss below 0.050.

## 2. Falsification Criterion
The hypothesis will be falsified if:
1. The centroid decoding MSE of the novel object under active CLTS control with positional encoding is >= 75.0; OR
2. The soft spatial variance of the coordinate encoder exceeds 10.0; OR
3. The post-collision test simulation loss exceeds 0.050.

## 3. Proposed Method
1. Modify `src/thalamus.py` to support explicit positional encodings at the input level:
   - Arm H (Linear Pos): 4 input channels, where the 4th channel is the linear normalized position `pos_i = i / 127.0`.
   - Arm I (Sinusoidal Pos): 7 input channels, with 3 RGB channels and 4 sinusoidal positional embeddings (using frequencies of 10 and 100).
2. Maintain the same CLTS active control scheme in `src/motor.py` as established in Phase 12.
3. Create `src/run_phase13_experiments.py` to run a 5-seed comparative sweep of Arm H (Linear Pos) and Arm I (Sinusoidal Pos) against the original RGB-only CLTS baseline.
4. Log and evaluate post-collision test simulation loss, centroid decoding MSE of the novel object (generalization test), soft spatial variance, and pointer spatial entropy.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
