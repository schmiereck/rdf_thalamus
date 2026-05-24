# RDF Scientific Pre-Registration

*   **Iteration:** 007
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Introducing downstream task-coupling (specifically, backpropagating a spatial target localization error from a linear readout probe to the newly recruited 4th latent dimension) during the N=3 to N=4 transition will force the recruited dimension to self-organize into a coordinate representation of the novel object. This task-coupled representation alignment will achieve:
1. A mean absolute Pearson correlation coefficient |r| >= 0.60 between the recruited 4th dimension and the physical position of the 4th object across 5 independent seeds.
2. A reduction of at least 25% in the target localization error (MSE) compared to the uncoupled (unsupervised surprise-only) baseline model, while preserving the overall temporal prediction loss within 10% of the baseline.

## 2. Falsification Criterion
The hypothesis will be falsified if any of the following occur:
1. The mean absolute Pearson correlation |r| between the recruited 4th dimension's activity and the physical position of the 4th object across the 5 evaluation seeds is less than 0.60.
2. The task-coupled model fails to reduce the target localization MSE on the 4th object by at least 25% compared to the uncoupled baseline.
3. The task-coupled model's overall temporal prediction loss (surprise) degrades by more than 10% compared to the uncoupled baseline, indicating representational drift or collapse.

## 3. Proposed Method
1. Define a Downstream Task-Coupled variant of the Thalamus architecture. This variant includes an auxiliary linear readout probe that predicts the 1D physical position of the attended object.
2. During the N=4 generalization transition, route the localization error gradients exclusively to the newly recruited 4th representation dimension (using the plasticity gating mechanism) while keeping the stable subspace frozen to prevent catastrophic interference.
3. Run a 5-seed comparative sweep comparing:
   - Control Group: Uncoupled baseline (unsupervised surprise minimization only, with a post-hoc linear probe trained frozen).
   - Experimental Group: Downstream Task-Coupled model (co-trained with the localization loss backpropagating to the recruited dimension).
4. Measure and log:
   - The Pearson correlation coefficient |r| between the 4th dimension's activity and the physical position of the 4th object.
   - The MSE of the localization prediction.
   - The overall system prediction loss (surprise).
5. The implementation will modify or extend the existing training/evaluation scripts (e.g., in `src/`) to incorporate the task-coupled training loop and the metric tracking.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
