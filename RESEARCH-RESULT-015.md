# RDF Milestone Review — Iteration 015 — Null Result: Dual Control "Cold-Start" and SA-CCR Pathology

## 1. Pre-Declared Hypothesis and Falsification Criterion
The iteration aimed to evaluate if dynamically adjusting coordinate regularization strength as a function of instantaneous surprise (SA-CCR, Arms L and M) or decoupling the network into a fast Surprise Detector and slow deliberative Categorizer (Dual Control, Arm N) improves representation stability and dynamics modeling over fixed CCR-Covariance (Arm K).

**Falsification Criteria:**
1. **SA-CCR Instability:** If surprise-modulated covariance scaling increases post-collision test simulation loss above 0.100 or causes representational divergence.
2. **Categorizer Stagnation:** If the Dual Control network (Arm N) fails to recruit a 4th dimension (recruitment rate = 0.0%) when transitioning from 3 to 4 objects, or if centroid decoding MSE exceeds the 70.0 threshold due to structural rejection.

## 2. Experimental Protocol
- **Environment:** 1D physics sandbox, 128 RGB pixels, 3 objects transitioning to 4 objects at step 1500 under continuous active CLTS motor control.
- **Grid & Steps:** 3000 steps per run, evaluated across 5 deterministic seeds (including seed 456).
- **Experimental Arms:**
  - **Arm L (Positive SA-CCR):** Covariance weight $\lambda_{cov}$ scaled proportionally with instantaneous surprise.
  - **Arm M (Inverse SA-CCR):** Covariance weight $\lambda_{cov}$ scaled inversely with instantaneous surprise.
  - **Arm N (Dual Control):** Saliency-driven Attention Token routing combined with an MDL-based Categorizer using consistency ratio validation: $L_{\text{consistency}} = \text{Var}_{\text{scenarios}}[z_{\text{new}}] / \text{Var}_{\text{scenarios}}[z_{\text{old}}] < 1.0$.
- **Control:** Arm K (Fixed CCR-Covariance, $\lambda_{cov} = 25.0$).

## 3. Observed Quantities
- **Arm L (Positive SA-CCR):** Instability confirmed. Seed 456 experienced complete gradient explosion. Recruited dimensions ($d_t$) overshot to 5, and the post-collision test simulation loss exploded to 14.88.
- **Arm M (Inverse SA-CCR):** Stable but statistically indistinguishable from baseline. Post-collision test simulation loss was 0.0912 (vs Arm K: 0.0901) and centroid decoding MSE was 63.12 (vs Arm K: 62.63).
- **Arm N (Dual Control):** Stagnation confirmed. The 4th dimension recruitment rate was 0.0% across all 5 seeds. The $L_{\text{consistency}}$ metric remained consistently above 1.5 (ranging from 1.5 to 3.2), rejecting every single proposed dimension. Consequently, centroid decoding MSE on the novel 4th object remained at 130.39 (falsification threshold: 70.0).

## 4. Verdict
**REFUTED.** The hypothesis that instantaneous surprise feedback on regularization or prediction-based MDL gating improves structural learning is refuted. The dual control hypothesis is unresolved in its ideal form but refuted under the current prediction-error-based formulation.

## 5. Construction-vs-Empirical Note
The failure of the Categorizer in Arm N is an empirical validation of a mathematical constraint: a newly spawned neural projection layer is initialized with random weights and has had zero optimization steps. Thus, its initial temporal predictions are mathematically guaranteed to have higher variance than a fully converged lower-dimensional baseline. Comparing the raw initial state of a newly spawned node directly to the stable baseline via a prediction-error ratio guarantees rejection by construction. This represents a definitional identity of training dynamics, not a failure of the dual-control concept itself.

## 6. Limitations
This result does not prove that separating surprise detection from categorization is invalid. It demonstrates that:
1. Temporal prediction error cannot be used to evaluate newly spawned dimensions unless those dimensions are allowed an asymmetric, non-blocking warm-up period to minimize their initial prediction errors.
2. Immediate feedback loops between high-frequency physics surprise and representation-level regularization parameters are dynamically unstable and must be low-pass filtered or temporally decoupled.