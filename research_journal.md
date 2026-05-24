# Research Manager Journal & Strategic State

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 2 (Thalamic Gating Evaluation).
*   **Active Direction:** Resolving the tracking-lag bottleneck in attention-token routing. Transitioning from fixed-step stabilization cooldowns to adaptive, surprise-modulated gating mechanisms.
*   **Confidence Score:** 80% (Slightly adjusted down from 85% to reflect the rigid-cooldown tracking failure, though the representation accuracy gains remain highly encouraging).

## 2. Strategic Insights & Lessons Learned
*   **The Tracking-Lag Bottleneck:** A rigid token-holding cooldown of 200 steps introduces a fundamental physical mismatch when representing highly dynamic environments. In a 128-pixel grid, objects moving at velocities of 1–2 pixels/step traverse a 32-pixel local receptive field in 16–32 steps. Enforcing a 200-step stabilization lock prevents the gating mechanism from updating its focus to match the object's spatial displacement, causing the attention trace to fall behind the physical entity.
*   **Local Gating Efficiency:** Despite the tracking lag, limiting plasticity to the gated layer yields a substantial performance improvement (18.1% lower predictive loss than the non-gated control, and 49.1% lower than the B1 baseline). This provides strong empirical support for the hypothesis that localized plasticity focus acts as a powerful regularizer, preventing global weight drift and catastrophic interference during multi-object transitions.

## 3. Loop & Bottleneck Detection
*   **Identified Bottleneck:** The structural stability of lower-level representations requires a minimal temporal window of static plasticity (to prevent input drift at higher levels). However, object kinematics require rapid, sub-30-step spatial tracking updates. A static, step-based cooldown cannot satisfy both constraints simultaneously.
*   **Mitigation Strategy for Next Iteration:** We must replace the step-counter cooldown with a dynamic, surprise-driven release. Plasticity gating should unlock when local surprise drops below a normalized threshold *or* when a sudden, high-amplitude surprise spike occurs in a neighboring spatial segment (indicating an object crossing a receptive field boundary).

## 4. Alternate Research Paths
*   **Surprise-Gradient Cooldown Release (Priority):** Transitioning token release to a derivative-based metric (releasing the token when the local rate of surprise reduction $dE/dt$ levels off).
*   **Velocity-Conditioned Gating:** Informing the gating duration directly with a fast, lower-level kinematic estimator so that the cooldown scales inversely with estimated object velocity.