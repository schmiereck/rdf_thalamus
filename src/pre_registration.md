# RDF Scientific Pre-Registration

*   **Iteration:** 014
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Applying Contrastive Coordinate Regularization (CCR)—comprising temporal smoothness (minimizing consecutive-frame coordinate distance) and spatial separation (either via a pairwise hinge-loss or a VICReg-style covariance regularization) directly on the non-parametric soft-argmax bottleneck—will constrain active-perception coordinate drift without introducing input-level optimization shortcuts. This self-supervised constraint will reduce the centroid decoding MSE of the novel object under active control to below 70.0, while maintaining a post-collision test simulation loss below 0.050.

## 2. Falsification Criterion
The hypothesis will be falsified if any of the following outcomes are observed:
1. The mean centroid decoding MSE of the novel object under active CLTS control for the best CCR arm (Arm J or K) is >= 70.0 (matching the hypothesis that CCR reduces it to < 70.0).
2. The mean post-collision test simulation loss at step 3000 for the best CCR arm exceeds 0.050 (indicating that coordinate regularization degrades physics prediction).
3. The pointer spatial entropy under active CLTS control drops below 3.5 (indicating that the regularization constrains the agent's exploratory behaviors).
4. The soft spatial variance of the coordinate encoder exceeds 10.0 (indicating a loss of spatial tightness of the bottleneck).
5. The 'lazy encoder' failure mode is detected, defined as a representational freezing where the coordinate velocity standard deviation of the novel object (std_vel_3) is < 1.5 while centroid decoding MSE remains high (>= 70.0).

## 3. Proposed Method
1. **Modify model code (`src/models_dual_stream.py`)**:
   Implement Contrastive Coordinate Regularization (CCR) inside `NonParametricJEPASpatial.forward`.
   - **Temporal Smoothness ($L_{smooth}$)**: L2 distance of consecutive-frame coordinates over the 4-frame sequence ($z_0, z_1, z_2, z_3$ normalized to $[0, 1]$ by dividing by 127.0). Specifically, `torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-8).mean()` on active coordinate channels.
   - **Spatial Separation**:
     - *Hinge-loss (for Arm J)*: Pairwise $relu(\epsilon - |z_i - z_j|)$ over all pairs of active coordinate channels up to $d_t$, averaged over all 4 frames of the transition sequence. Use $\epsilon = 0.15$.
       - *Physical/geometric grounding of $\epsilon = 0.15$*: Since coordinates are normalized to $[0, 1]$ on a 128-pixel canvas, $\epsilon = 0.15$ corresponds to $0.15 \times 128 = 19.2$ pixels. This aligns with the maximum contact distance of two contacting objects of maximum radius $r=8$ (i.e., $8 + 8 = 16$ pixels) plus a small spatial buffer of 3.2 pixels.
     - *Covariance-loss (for Arm K)*: Off-diagonal covariance penalty on the normalized active coordinates $z_0, z_1, z_2, z_3$ across the batch, averaged over all 4 frames of the transition sequence.
   - Return CCR loss components in the loss dictionary: `"ccr_smooth_loss"` and `"ccr_spatial_loss"`.
   - Allow toggling `ccr_mode` (`'none'`, `'hinge'`, `'covariance'`) and specifying `ccr_smooth_weight` and `ccr_spatial_weight` (default weights to 10.0).

2. **Track Coordinate Velocity Metrics**:
   To robustly detect the 'lazy encoder' failure mode, track the standard deviation (`std_vel_3`) and mean absolute velocity (`mean_abs_vel_3`) of the coordinate channel of the novel object (channel 3) over the evaluation sequence:
   - Velocity is computed as the first-order difference of the coordinate of the novel object across the 200 evaluation frames: $v_t = z_{3, t} - z_{3, t-1}$.
   - We define `std_vel_3` as the standard deviation of $\{v_t\}$, and `mean_abs_vel_3` as the mean of $\{|v_t|\}$.
   - If `std_vel_3 < 1.5` under active control and the MSE is high, it is classified as a 'lazy encoder' failure mode.

3. **Run a matched 5-seed comparative sweep**:
   Train Arm G (Original RGB CLTS baseline), Arm J (CCR-Hinge), and Arm K (CCR-Covariance) on matched environment sequences (N=3 passive pre-training, N=4 active CLTS training). Ensure that training steps 1..1500 (passive) and 1501..3000 (active) use the appropriate `ccr_mode` and weights for each arm.

4. **Evaluate and Analyze**:
   Save summary statistics to `archive/iter_14/results/summary_phase14.csv`, save audit logs to `archive/iter_14/results/audit_results_phase14.json`, and plot adaptation curves to `archive/iter_14/results/adaptation_curves_phase14.png`. Report Welch's t-test and Levene's test comparisons.
