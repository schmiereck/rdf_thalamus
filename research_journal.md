# Research Manager Journal & Strategic State

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 5 (Active Probing vs. Passive Observation) and Phase 8 (Unsupervised Spatial Bottlenecks) Complete.
*   **Active Direction:** Resolving the "Spatial Specialization Gap" by introducing unsupervised spatial bottlenecks and closed-loop "output-as-input" active probing.
*   **Confidence Score:** 94% (Reflecting the high empirical rigor of 5-seed systematic sweeps, explicit null controls, and the honest reporting of pre-registered target falsifications).

## 2. Strategic Insights & Lessons Learned
*   **The Spatial Decoupling Trade-off:** Introducing an unsupervised spatial bottleneck (spatial convolutional constraint + soft spatial variance minimization $L_{spatial} = \lambda \cdot \text{Var}_k$) successfully forces physical coordinate localization, but is bound by a strict trade-off. A gentle bottleneck ($\lambda = 0.01$) optimizes coordinate decodability (16.9% MSE reduction over control), whereas a strong bottleneck ($\lambda \ge 0.1$) over-regularizes the latent space and severely degrades temporal prediction accuracy.
*   **Bottleneck-Induced Collapse Prevention:** Intriguingly, enforcing even a minor spatial bottleneck completely eliminates representation collapse (reducing the collapse rate from 40% in the unconstrained control to 0% across all bottlenecked configurations). Forcing representational units to span spatial coordinates prevents the multi-dimensional manifold from co-varying into a trivial, single-point attractor state.
*   **Inherent Manifold Dispersion:** The "Spatial Specialization Gap" is highly resilient. Even with explicit spatial penalties, the network prefers to distribute coordinate information across a low-dimensional manifold rather than aligning it perfectly with a single recruited axis ($|r| = 0.42 \pm 0.15$ vs. the target of $\ge 0.60$). This indicates that distributed population codes are the natural thermodynamic default for temporal surprise minimization.

## 3. Loop & Bottleneck Detection
*   **The "Disentanglement-Accuracy" Loop:** We are observing a fundamental trade-off loop. Increasing the spatial regularization penalty to force cleaner, axis-aligned representations directly conflicts with the temporal prediction objective (Pillar B). The model cannot reconstruct or predict complex temporal dynamics if its latent space is forced to behave as a pure, low-complexity spatial tracker.
*   **Mitigation Strategy:** Move away from static spatial penalties. Future architectures should explore a *dual-channel latent space* (e.g., separating slowly-varying spatial/inertial coordinate representations from high-frequency dynamic physical properties) or employ *adaptive bottlenecking* where the spatial penalty is selectively relaxed once spatial tracking stability is achieved.

## 4. Alternate Research Paths
*   **Dual-Channel Latent Space (Architectural Split):** Partitioning the latent space into a low-capacity, highly bottlenecked spatial tracking channel (e.g. via a soft-argmax operator) and an unbottlenecked, high-capacity dynamic predictive channel.
*   **Closed-Loop Surprise-Gated Motor Control:** Dynamically adjusting the closed-loop PD controller's gain based on the gating attention token's surprise value, testing whether "active play" naturally subsides as local prediction error approaches zero.