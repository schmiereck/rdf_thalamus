# RDF Scientific Pre-Registration

*   **Iteration:** 002
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
A Dynamic Representation Network (Thalamus Phase 1) using Gradient-Driven Active Subspace Recruitment (GDASR) combined with VICReg-style anti-collapse regularization will:
1. Dynamically recruit new representational dimensions (incrementing active dimension count $d_t$) when transitioned from a low-complexity training environment (2 objects) to a high-complexity evaluation environment (3 objects) in a 1D physics sandbox.
2. Achieve a post-recruitment temporal prediction error on the 3-object environment that is at least 30% lower than a fixed-dimensional Joint Embedding Predictive Architecture (Baseline B1 with fixed $d_t = 2$) trained under identical conditions.
3. Avoid representation collapse, maintaining a latent standard deviation of $\sigma > 0.1$ for all active dimensions and an average absolute cross-dimension correlation of $r < 0.3$.

## 2. Falsification Criterion
The hypothesis will be proven false if any of the following conditions are met:
1. The dynamic representation network fails to recruit any new dimension (remains at $d_{\text{init}} = 2$) within 5000 steps of exposure to the 3-object environment.
2. The mean temporal prediction error of the dynamic network on the 3-object environment after adaptation is NOT at least 30% lower than that of the fixed-dimensional baseline (B1).
3. The representations collapse during training or evaluation, defined as any active dimension having a standard deviation $\sigma \le 0.1$ or the average absolute correlation between active dimensions being $r \ge 0.3$ across the evaluation batch.

## 3. Proposed Method
1. Implement a 1D physics sandbox (`src/environment.py`) generating a 1D array of 128 RGB pixels, simulating N elastic colliding objects with randomized sizes, colors, masses, and initial velocities, with parameterizable N.
2. Implement `src/models.py` containing:
   - An Encoder (CNN/MLP) outputting up to $D_{\max} = 8$ dimensions, with only the first $d_t$ dimensions active.
   - A Predictor (MLP or lightweight GRU) forecasting the active latent states $\hat{z}_{t+1}$ from history.
   - Baseline B1 (FixedJEPA) with fixed $d_t = 2$ and VICReg loss.
   - DynamicJEPA implementing GDASR: starts with $d_t = 2$, tracks the exponential moving average of prediction error, and increments $d_t$ when error exceeds $\theta_{\text{recruit}}$ after a cooldown $N_{\text{cooldown}}$.
3. Create `src/train.py` to train both models first on $N=2$ objects, then transition them to $N=3$ objects.
4. Perform 5 independent runs with different random seeds to calculate the mean and standard deviation of prediction errors, dimension count, latent variance, and cross-correlations.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
