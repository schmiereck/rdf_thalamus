# Research Manager Journal & Strategic State

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 10 (Dual-Stream Decoupling & Active Probing) Complete.
*   **Active Direction:** Resolving the "semantic blindness" of isolated representational streams. Transitioning from absolute stop-gradient isolation to a *Task-Grounded Dual-Stream* architecture to preserve spatial coordinate tracking without inducing optimization interference.
*   **Confidence Score:** 85% (Adjusted down from 95% to reflect the discovery of the semantic blindness barrier, though our trajectory remains highly structured).

## 2. Strategic Insights & Lessons Learned
*   **The "Semantic Blindness" of Absolute Decoupling:** Phase 10 results have rigorously established that completely shielding a highly bottlenecked coordinate stream from temporal prediction gradients (via stop-gradients) while applying spatial variance minimization and VICReg constraints leads to total representational collapse. Without downstream task gradients to anchor the representation to actual physical entities, the spatial optimization pressure trivially collapses the channel into arbitrary static spikes.
*   **The Grounding Requirement:** A representational channel cannot organize itself into physically meaningful spatial coordinates purely based on local statistics (entropy minimization/contrastive spread) if it lacks a functional link to physical dynamics. Prediction or reconstruction gradients are not just "tasks"—they are the essential grounding mechanisms that bind latent dimensions to physical reality.
*   **Failure of Simple Detachment:** While stop-gradients are useful for preventing representational drift in multi-scale networks, their naive application to separate "where" and "what" streams prevents "where" from learning any semantic relationships to the input.

## 3. Loop & Bottleneck Detection
*   **The "Grounding vs. Interference" Bottleneck:** We face a dual-sided failure envelope:
    1. *Joint Training (Phase 9):* Joint gradient flow causes spatial regularization to interfere with temporal dynamics prediction (22.8% accuracy penalty).
    2. *Strict Decoupling (Phase 10):* Strict stop-gradients eliminate interference but trigger 100% semantic blindness and representational collapse.
*   **Mitigation Strategy:** We must explore "Soft Grounding." Instead of absolute stop-gradients, we must implement either:
    - A dual-stage curriculum where a shared backbone is frozen after learning dynamics, and the coordinate head is trained on top of the frozen features.
    - A projection-based coordinate extractor that uses a soft-argmax operator over the predictive dynamics channel, thereby enforcing spatial organization by construction rather than by independent optimization.

## 4. Alternate Research Paths
*   **Auxiliary Action-Conditioned Grounding:** Ground the spatial channel by forcing it to predict ego-motion or relative pointer offsets under motor commands, bypassing the full temporal predictive model's gradients but still enforcing physical grounding.
*   **Post-Hoc Coordinate Extraction (Soft-Argmax Projection):** Derive spatial coordinates directly as a differentiable, non-parametric projection of the predictive dynamics stream, guaranteeing no gradient interference because there are no separate spatial weights to train.