# RDF Milestone Review — Iteration 004 — Refuted Hypothesis: Rigid-Cooldown Thalamic Gating

## 1. Pre-Declared Hypothesis and Falsification Criterion
*   **Hypothesis:** Implementing a localized thalamic attention token with a fixed 200-step stabilization cooldown improves spatial tracking and representation learning accuracy by preventing lower-layer input drift without incurring registration lag or tracking dropouts.
*   **Falsification Criterion:** The hypothesis is refuted if the gated network fails to maintain spatial overlap (saliency registration) with moving entities, or if the tracking lag exceeds the time taken for a physical entity to cross a local spatial segment's receptive field boundary.

## 2. Experimental Protocol
*   **Grid and Physics Parameters:** 128-pixel 1D RGB sandbox. Three distinct physical objects moving at velocities of 1.0 to 2.0 pixels/step under continuous elastic boundary and inter-object collisions.
*   **Network Architectures evaluated (5-seed sweep):**
    *   *Gated Model:* Multi-scale DynamicJEPA with Thalamic Gating and a fixed 200-step token-holding cooldown.
    *   *Non-Gated Control:* Identical architecture with globally active plasticity (no attention token).
    *   *Baseline B1:* Fixed-dimensionality standard JEPA.
*   **Variables Held Constant:** Initial object velocities, spatial boundaries, mass distributions, optimizer hyper-parameters, and random seeds across all baseline and experimental models.

## 3. Observed Quantities
*   **Predictive Latent Loss (Average over 5 seeds):**
    *   Gated Model: 0.0511 ± 0.0031 (49.1% reduction vs. B1, 18.1% reduction vs. non-gated control).
    *   Non-Gated Control: 0.0624 ± 0.0042.
    *   Baseline B1: 0.1004 ± 0.0055.
*   **Tracking Overlap & Receptive Field Crossing:** Saliency tracking lag was measured to be up to 170 steps. Objects traversed the 32-pixel spatial receptive field segments in 16 to 32 steps. Because the 200-step stabilization lock exceeded this crossing time, the attention token remained locked to old spatial locations long after the physical object had departed.
*   **Falsification Verdict:** Refuted under the pre-declared tracking lag threshold.

## 4. Verdict
**Refuted.** While the quantitative reduction in predictive latent loss (18.1% vs. non-gated control, 49.1% vs. B1) is highly statistically significant and indicates that localized plasticity provides a powerful regularizing effect, the rigid 200-step cooldown mechanism is physically incompatible with the kinematics of the sandbox environment, leading to a severe tracking-lag bottleneck.

## 5. Construction-vs-Empirical Note
The reduction in prediction error is a genuine empirical finding showing that gating prevents co-adaptation of weights across inactive regions. Conversely, the tracking lag is not a bug, but an empirical consequence of mismatching the environment's characteristic physical timescales (spatial transit time) with the model's architectural hyper-parameters (rigid step-based cooldown).

## 6. Limitations
*   This result only refutes *rigid, step-based* cooldowns. It does not refute the viability of Thalamic Gating as a whole.
*   The performance metrics were gathered under clean, elastic-collision dynamics; behavior under highly chaotic or frictional multi-body interactions has not yet been established.
*   An adaptive gating mechanism that couples the token release to surprise derivatives or velocity-scaled estimators is required to achieve continuous tracking alignment.