# RDF Scientific Pre-Registration

*   **Iteration:** 003
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Increasing the VICReg covariance regularization weight to 25.0, in conjunction with a 1000-step representation-warmup phase across all models, will prevent representation collapse (reducing the mean absolute cross-dimension correlation $r$ from >0.99 to <0.30). Furthermore, replacing the cumulative error buffer with a rolling sliding-window error buffer of size 500 (cleared only post-warmup at step 1000, with no Oracle reset during environmental transitions) will enable sensitive and reliable recruitment of a new dimension (recruitment rate >80%) when transitioning from 2 to 3 objects in the 1D physics sandbox, without increasing the temporal prediction simulation loss compared to the baseline B1.

## 2. Falsification Criterion
The hypothesis will be falsified if any of the following occur:
1. The mean absolute correlation ($r$) between representation dimensions is >= 0.30 at the end of training.
2. The dynamic recruitment rate for the GDASR model upon the N=3 object transition is <= 80%.
3. The final temporal prediction simulation loss of the recruiting DynamicJEPA model is > 0.080 (or more than 10% worse than the non-recruiting B1 baseline of 0.06662).

## 3. Proposed Method
1. Modify the training and model configuration files (e.g., in `src/`) to increase `cov_weight` from 1.0 to 25.0 in the VICReg loss calculation for all models.
2. Implement a representation-warmup phase of 1000 steps during which gradient updates are performed normally but dimension recruitment is disabled across all models.
3. Replace the cumulative error buffer in the GDASR recruitment module with a rolling sliding-window buffer of size 500.
4. Programmatically reset/clear this error buffer immediately following the warmup phase (at step 1001), but do NOT programmatically reset or clear the error buffer or change any parameters during the N=2 to N=3 transition at step 1501.
5. Run the full evaluation suite of 15 experiments (DynamicJEPA, B1, and B1_large across 5 deterministic seeds) on the 1D physics sandbox.
6. Measure and log: mean absolute correlation between dimensions, recruitment rate upon N=3 transition, and final temporal prediction simulation loss.
