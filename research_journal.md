# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 15 (Dual Control & Surprise-Adaptive Regularization) Complete.
*   **Active Direction:** Resolving structural learning bottlenecks in dual-control architectures. Phase 15 demonstrated that while smooth representation-level constraints (CCR-Covariance) are stable, dynamically scaling these constraints based on instantaneous surprise (SA-CCR) introduces destructive gradient fights during physical collisions. More critically, the implementation of a structural Dual Control system (Arm N) exposed a fundamental **"cold-start" pathological reject loop**: newly spawned, untrained dimensions generate high prediction errors, causing the Minimum Description Length (MDL) consistency gate to reject them before they can converge. Our active direction is to solve this initialization bias by designing an asymmetric "shadow-dimension" warm-up protocol or switching to entropic, prediction-independent MDL gating.
*   **Confidence Score:** 82% (Adjusted down from 89% due to the discovery of the cold-start structural bottleneck in prediction-based MDL gating).

## 2. Strategic Insights & Lessons Learned
*   **The Cold-Start Pathological Reject Loop:** A newly initialized representational node/dimension naturally lacks optimized predictive weights. Consequently, evaluating its utility using a ratio of temporal prediction errors ($L_{\text{consistency}} = \text{Var}[e_{\text{new}}] / \text{Var}[e_{\text{old}}]$) immediately after spawning guarantees rejection. The system enters a permanent structural stagnation where no new dimensions can ever clear the MDL gate.
*   **High-Frequency Kinematics Gradient Clash:** Instantaneously modulating regularization weights based on temporal surprise (SA-CCR) is highly unstable. During elastic collisions, surprise spikes naturally. Increasing covariance regularization at this exact frame forces the encoder to violently decorrelate features precisely when it should be absorbing the transient high-frequency kinematics of the collision, leading to representational divergence and exploding simulation loss (e.g., Seed 456).

## 3. Loop & Bottleneck Detection
*   **MDL Stagnation Bottleneck:** The Categorizer rejects all structural additions because it expects newly allocated pathways to immediately outperform stable, long-trained pathways in temporal prediction.
*   **Mitigation Strategy:** Decouple the structural validation from temporal prediction. Proposed dimensions must either:
    1. Train in a non-blocking "Shadow State" (inference-only to the rest of the network, but plastic locally) for a fixed warm-up window ($N_{\text{warm}} = 500$ steps) before the consistency audit.
    2. Be evaluated using spatial activation entropy or mutual information of the encoder, bypassing the temporal predictor's training lag entirely.

## 4. Alternate Research Paths
*   **Shadow-Dimension Recruitment (Phase 15.1):** Implement a structural staging area where recruited dimensions are stabilized via local gradients before they are permitted to influence the active latent representation or undergo the consistency audit.
*   **Spatial Entropic MDL Gates (Phase 15.2):** Formulate the consistency loss $L_{\text{consistency}}$ using the spatial activation profiles of the soft-argmax bottleneck rather than prediction error, evaluating coordinate compression directly.
*   **Aggressive Spatial Compression (Phase 13 / Dimension-Width Trade-off):** Integrate the validated fixed CCR-Covariance into a multi-scale spatial hierarchy (128 -> 32 -> 8 -> 2 nodes) with spatial micro-columns to force disentanglement.