# RDF Scientific Pre-Registration

*   **Iteration:** 016
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Introducing a Probationary Warm-Up Period (WUP) of W steps for newly proposed
dimensions, combined with a Predictability-Variance-Uniqueness (PVU) gating
metric, will resolve the cold-start rejection bias in the Dual Control
Categorizer. Specifically, under CLTS active control during the N=3 to N=4
object transition, this mechanism (Arms O and O_big) will successfully
recruit the 4th dimension in at least 4 out of 5 seeds (recruitment rate >=
80%), with the recruited dimension being non-collapsed (variance > 1e-3),
highly predictable (relative prediction error U_new < 0.5), and
non-redundant (maximum absolute correlation with the existing 3 dimensions
< 0.8). This will enable the network to successfully track the novel 4th
object, reducing the post-transition centroid decoding MSE strictly below
70.0 (vs. 130.39 obtained by Arm N in Phase 15), without increasing the
test simulation loss above 0.15.

### Predictability Metric (Stable Ratio Formulation)
The predictability metric for a newly proposed latent dimension is defined as
a mathematically stable ratio with an explicit epsilon guard:

    U_new = MSE_new / (Var_new + epsilon),   with epsilon = 1e-5

where:
*   `MSE_new` is the mean-squared-error of the latent coordinate prediction
    for the newly proposed dimension, computed over a validation batch of
    `100` transitions sampled from the replay buffer.
*   `Var_new` is the variance of the target latent coordinate for that
    same proposed dimension across the same `100` validation transitions.
*   `epsilon = 1e-5` is the numerical-stability term that prevents division
    by zero when the proposed dimension is collapsed or nearly constant
    across the validation batch.

A small `U_new` indicates that the proposed dimension is predictable
relative to its own dynamic range, i.e. the latent prediction error is
small compared with the variation the dimension itself exhibits.

## 2. Falsification Criterion
The hypothesis will be falsified if any of the following occur:
1. The 4th dimension is recruited in fewer than 3 seeds (less than 60%
   recruitment rate) across the WUP-PVU arms (Arm O and Arm O_big).
2. The mean post-transition centroid decoding MSE of Arm O (and/or Arm
   O_big) is `>= 70.0`. (Success is strictly `mse_cent < 70.0`.)
3. The recruited 4th dimension in Arm O (and/or Arm O_big) is redundant
   with existing dimensions (mean max absolute correlation `>= 0.8`) or
   is collapsed (mean variance `<= 1e-3`).
4. The test simulation loss of Arm O (and/or Arm O_big) is significantly
   higher than the baseline Arm K (Welch's `p < 0.05` and mean loss
   `> 0.15`).

## 3. Proposed Method
1. Identify the dimension recruitment and MDL consistency gating logic
   within `src/thalamus.py` and `src/models_dual_stream.py`.
2. Implement Arm O / Arm O_big (WUP-PVU, W=100 / W=500):
   a. When the `3 -> 4` transition is proposed at step 1800, set
      `probationary = True`, `probation_end_step = 1800 + W`, and
      `branch_model.d_t = 4` so that the encoder/predictor heads for the
      4th dimension receive gradient updates during the probation window.
   b. Allow the probationary dimension's encoder and predictor weights to
      be updated via standard gradient descent during the probation.
   c. Protect active exploration: in the CLTS motor loop pass only the
      first 3 dimensions of `(zp_coord, zt_coord, zp_dyn, zt_dyn,
      centroids)` to `get_action`, i.e. keep `d_t_effective = 3` so the
      controller does not chase a noisy un-converged channel.
   d. Disable GDASR / further proposals during probation.
   e. At `step == probation_end_step` sample 100 validation transitions
      and evaluate PVU on the new dimension:
        - Non-collapse: `var_new > 1e-3`.
        - Predictability: `U_new = MSE_new / (Var_new + 1e-5) < 0.5`.
        - Uniqueness: `max_corr < 0.8` (max absolute Pearson correlation
          of the new dim with each of the first 3 dims).
      If all three pass, accept; otherwise revert `branch_model.d_t = 3`,
      schedule another probation attempt 50 steps later.
3. Implement Arm P / Arm P_big (WUP-MDL, W=100 / W=500): identical
   warm-up mechanics as Arm O / O_big, but the gating criterion at the
   end of the probation window is the MDL consistency ratio
   `ratio = sim_new / sim_old < 1.0`.
4. Implement matched controls Arm K (no probation, unconditional 3->4 at
   step 1800) and Arm N (original immediate MDL gate at step 1800+, no
   probation, identical to Phase 15).
5. Run a 5-seed matched comparative sweep across Arms K, N, O, O_big, P,
   P_big using cached passive pre-trained states (two cached states per
   seed: one for K/O/O_big/P/P_big, one for N) to avoid redundant
   training.
6. Log and evaluate: centroid decoding MSE (post-transition and overall),
   dimension count, test simulation loss, soft spatial variance,
   pointer-spatial coverage entropy, coordinate velocity standard
   deviation and mean absolute velocity, final `d_t`, transition
   accepted step, and final gating criteria. Conduct Welch's t-test and
   Levene's test comparisons of Arms O and O_big vs Arm K and Arm N.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
