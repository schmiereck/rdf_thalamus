# Research Manager Log - Iteration 016

## Iteration 016 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Introducing a Probationary Warm-Up Period (WUP) of W = 100 steps for newly proposed dimensions, combined with a Predictability-Variance-Uniqueness (PVU) gating metric, will resolve the cold-start rejection bias in the Dual Control Categorizer. Specifically, under CLTS active control during the N=3 to N=4 object transition, this mechanism (Arm O) will successfully recruit the 4th dimension in at least 4 out of 5 seeds (recruitment rate >= 80%), with the recruited dimension being non-collapsed (variance > 1e-3), highly predictable (relative prediction error U_new = MSE / Var < 0.5), and non-redundant (maximum absolute correlation with existing dimensions < 0.8). This will enable the network to successfully track the novel 4th object, reducing the post-transition centroid decoding MSE from 130.39 (Arm N) to below 70.0, without increasing the test simulation loss above 0.15.

**Proposed Falsification Criterion:**
The hypothesis will be falsified if any of the following occur:
1. The 4th dimension is recruited in fewer than 3 seeds (less than 60% recruitment rate) in Arm O.
2. The mean post-transition centroid decoding MSE of Arm O is >= 75.0.
3. The recruited 4th dimension in Arm O is redundant with existing dimensions (mean max absolute correlation >= 0.8) or is collapsed (mean variance <= 1e-3).
4. The test simulation loss of Arm O is significantly higher than the baseline Arm K (Welch's p < 0.05 and mean loss > 0.15).

**Proposed Method:**
1. Identify the dimension recruitment and MDL consistency gating logic within `src/thalamus.py` and `src/models_dual_stream.py`.
2. Implement Arm O: 
   a. When a new dimension is proposed, set its state to `probationary=True` and initialize a counter for W = 100 steps.
   b. Allow the probationary dimension's encoder and predictor weights to be updated via standard gradient descent during training.
   c. Prevent the probationary dimension from affecting attention routing or triggering new proposals during these 100 steps.
   d. At the end of the W steps, evaluate the PVU criteria on the consistency buffer:
      - Non-collapse: Variance of the new dimension > 1e-3.
      - Predictability: Relative prediction error U_new = MSE / Var < 0.5.
      - Uniqueness: Maximum absolute correlation with existing dimensions < 0.8.
   e. If all three criteria are met, accept and integrate the dimension; otherwise, prune it.
3. Implement Arm P (Control): Same probationary warm-up period, but evaluates acceptance using the original total simulation loss ratio `L_consistency = sim_new / sim_old < 1.0`.
4. Run a 5-seed comparative sweep across Arms K (Baseline), N (Original immediate MDL gate), O (WUP-PVU), and P (WUP-MDL).
5. Log and evaluate: Centroid decoding MSE, dimension count, test simulation loss, soft spatial variance, and correlation statistics. Conduct statistical significance tests (Welch's t-test and Levene's test) to validate the hypothesis.

---

## Iteration 016 -> Planner [Strategic Guidance]

### Manager's Note: Strategic Guidance & Scientific Review

Your proposed approach to resolving the "cold-start" pathological reject loop in the Dual Control Categorizer is promising, but the plan contains several critical design risks and logical discrepancies that must be resolved before proceeding to execution.

#### 1. Eliminate Threshold Discrepancies (Falsification Rigorousness)
There is a soft margin of error in your proposed metrics that violates our pre-registration discipline:
*   Your **hypothesis** states that Arm O will reduce post-transition centroid decoding MSE to **below 70.0**.
*   Your **falsification criterion** states the hypothesis is falsified if the MSE is **$\ge$ 75.0**.
This leaves a $5.0$ MSE "gray zone" of post-hoc justification. **You must resolve this discrepancy.** Tighten the falsification criterion: the hypothesis must be considered falsified if the mean post-transition centroid decoding MSE of Arm O is **$\ge$ 70.0**.

#### 2. The $W = 100$ Under-Training Risk (Parameter-Tuning Hygiene)
A probationary window of $W = 100$ steps is exceptionally short for neural network convergence, especially compared to the 1000-step representation-warmup established in Phase 3.
*   **Skeptical Question:** If $W = 100$ is too short, the prediction head will remain under-trained, and the PVU gate will reject the dimension, leading to a false negative result that merely reflects a poor choice of $W$.
*   **Direction:** You must either provide a clear justification for why 100 steps is sufficient for local head convergence, or restructure your sweep to evaluate $W$ at both a rapid scale ($W=100$) and a more conservative scale ($W=500$). Treat any scenario where a dimension is rejected at $W=100$ but accepted at $W=500$ not as a structural failure of Dual Control, but as a basic training-lag constraint.

#### 3. Mathematical Stability of the Predictability Metric ($U_{\text{new}}$)
Your proposed predictability metric is formulated as $U_{\text{new}} = \text{MSE} / \text{Var}$. 
*   If a newly recruited dimension is highly regularized or near-collapsed (e.g., its variance is close to your minimum threshold of $1e-3$), this ratio will explode due to a tiny denominator, leading to spurious rejections of low-variance but stable dimensions.
*   **Direction:** Formulate the denominator with a safety epsilon, i.e., $U_{\text{new}} = \text{MSE} / (\text{Var} + \epsilon)$, and explicitly define how both $\text{MSE}$ and $\text{Var}$ are aggregated over the consistency buffer (e.g., spatial vs. temporal dimensions) to ensure the metric is mathematically stable.

---

### Pre-Registration Mandate
The Orchestrator will automatically write and commit your finalized hypothesis and quantitative falsification criteria to `src/pre_registration.md`. Ensure that your updated plan:
1. Aligns the centroid MSE threshold strictly to **$< 70.0$** across both the hypothesis and the falsification criteria.
2. Incorporates the math for the stabilized $U_{\text{new}}$ ratio.
3. Adheres strictly to our language discipline (avoiding terms like "breakthrough" or "perfectly tracking" in your code comments and final report).

---

