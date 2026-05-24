# Research Manager Journal & Strategic State

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 4 (Generalization & Noise Robustness) Evaluated.
*   **Active Direction:** Synthesizing the complete performance of the "Thalamus" architecture across all phases (Pillars A through E) to construct the final strategic evaluation report. Incorporating the user's insight to investigate how background static/dynamic noise during training influences representation stability.
*   **Confidence Score:** 90% (Up from 85%, reflecting the high integrity of our evaluations, including the explicit documentation of structural constraints and refuted hypotheses).

## 2. Strategic Insights & Lessons Learned
*   **Mathematical Noise Filtering vs. Emergent Attention:** The Z-score normalization of surprise ($z_i = (S_i - \mu_i) / \sigma_i$) successfully isolates anomalous surprise from steady-state unpredictable noise (Noisy-TV). However, this is an algebraic identity of the construction: high background variance $\sigma_i$ mathematically dampens the normalized surprise of that layer, preventing attention trapping. This is a robust engineering tool, not a cognitive emergence.
*   **The Spatial Specialization Gap:** Dynamic dimension recruitment (GDASR) successfully triggers under novel complexity ($N=4$) and provides a $6.0\%$ prediction loss reduction. However, the recruited dimension's activity does not correlate with the spatial coordinate of the novel object ($|r| = 0.0456$). Unsupervised, prediction-error-driven recruitment acts as a general residual capacity expander rather than self-organizing into discrete spatial tracking coordinates.

## 3. Loop & Bottleneck Detection
*   **The "Capacity without Specialization" Loop:** Simply adding dimensions based on residual error variance (Pillar A) allows the network to capture more dynamical variance, but does not force those dimensions to represent semantic or spatial invariants (like the coordinate of the new object).
*   **Mitigation Strategy:** To achieve true object-localized representations, dimension recruitment must be coupled with spatial spatial-pooling constraints or downstream motor-task gradients (e.g. active tracking target coordinates) that force spatial specialization.

## 4. Alternate Research Paths
*   **Background Noise Injection as a Regularizer (User Hint):** Introducing structured static and dynamic background noise during early training phases to force the latent layers to build robust spatial pooling filters before executing active motor tasks.
*   **Task-Modulated Dimension Recruitment:** Exploring a mechanism where dimension recruitment is gated not only by passive prediction error, but by the agent's inability to achieve motor goals (e.g., failing to coordinate a "push" action on a high-surprise entity).