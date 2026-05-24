# Research Manager Journal & Strategic State

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 4 (Generalization & Noise Robustness) and Phase 7 (Active Probing Analysis) Complete.
*   **Active Direction:** Integrating active closed-loop motor interaction with unsupervised representation learning to investigate spatial coordinate specialization and recruitment dynamics. 
*   **Confidence Score:** 92% (Reflecting the high empirical rigor of using a 5-seed comparative sweep, zero task-gradient leakage, and honest refutation of single-dimension spatial specialization).

## 2. Strategic Insights & Lessons Learned
*   **Active Probing as a Physical Catalyst:** Closing the motor loop via unsupervised PD-control (targeting high-surprise entities) acts as a temporal structuring mechanism. By generating consistent physical collisions, it stabilizes dynamic dimension recruitment (GDASR) from a highly erratic $60\%$ rate in passive observation to a highly deterministic $100\%$ rate (step variance reduced from $184.2$ to $1.2$).
*   **Unsupervised Feature Dispersion:** Purely unsupervised temporal surprise + VICReg covariance regularization ($25.0$) succeeds in expanding the latent manifold to capture novel variance, but *does not* force spatial specialization onto the newly recruited dimension ($|r| = 0.2184 \pm 0.231$, refuting the single-dimension tracking hypothesis). Instead, the coordinate information is distributed across the entire multi-dimensional latent manifold, improving overall offline decodability by $19.9\%$.
*   **The Spatial Specialization Gap (Confirmed):** Unsupervised residual-capacity expansion acts as a general dynamical variance absorber. To localize specific physical invariants (like coordinates or masses) to individual dimensions, the network requires structural inductive biases (e.g., spatial-pooling bottlenecks or attention maps) rather than relying purely on temporal prediction error.

## 3. Loop & Bottleneck Detection
*   **The "Distributed Coordinate" Loop:** We observe a recurring pattern where adding capacity (Pillar A) under a temporal surprise objective (Pillar B) reduces prediction error and captures environmental dynamics, but fails to isolate spatial variables. The network naturally favors distributed, highly correlated population-codes over clean, axis-aligned physical variables.
*   **Mitigation Strategy:** Moving forward, instead of expecting single-dimension alignment to emerge organically from temporal surprise, we must introduce spatial bottlenecks. For example, implementing a localized spatial pooling layer or gating dimension recruitment with spatial-attention coordinate readouts.

## 4. Alternate Research Paths
*   **Spatial Bottlenecks (Inductive Biases):** Testing the integration of 1D spatial attention grids or soft-argmax coordinate extraction layers directly into the JEPA latent space to force axis-aligned coordinate representations.
*   **Task-Gated Plasticity Gating:** Restricting plasticity updates and dimension recruitment not just to layers with high surprise, but to representations that directly impact motor-control stability (e.g. tracking error in the closed-loop controller).