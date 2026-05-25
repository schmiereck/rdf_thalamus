# RDF Milestone Review — Iteration 016 — Resolution of Cold-Start Pathology in Dual Control via Probationary Warm-Up

## 1. Pre-Declared Hypothesis and Falsification Criterion
- **Hypothesis 1 (WUP-MDL):** Introducing a Probationary Warm-Up Period (WUP) of 500 steps for newly recruited dimensions will bypass the cold-start predictor bias, resulting in a $>80\%$ recruitment rate during the $N=3 \rightarrow N=4$ transition and reducing post-transition centroid decoding MSE to $<65.0$ (outperforming the failed Arm N control of 130.39).
- **Hypothesis 2 (WUP-PVU):** Implementing a multi-criteria Projection Vector Utility (PVU) gate that explicitly penalizes redundant cross-dimension correlation will stabilize recruitment while maintaining cross-dimension correlation below $0.5$.
- **Falsification Criteria:** 
  - WUP-MDL fails to recruit in $>20\%$ of runs or centroid MSE remains $\ge 65.0$.
  - WUP-PVU fails to recruit or fails to restrict cross-dimension correlation below the pre-declared thresholds.

## 2. Experimental Protocol
- **Environment:** 1D continuous physics sandbox (128 RGB pixels), transitioning from 3 objects to 4 objects at step 1500.
- **Matched Sweep:** 5 independent seeds evaluated across four architectural configurations:
  - **Arm K:** Baseline (no dual control, fixed structure).
  - **Arm N:** Standard Dual Control (MDL gate evaluated immediately upon spawning; no warm-up).
  - **Arm O (and O_big):** Dual Control with WUP + PVU correlation gating.
  - **Arm P (and P_big):** Dual Control with WUP + MDL prediction gating.
- **WUP Parameters:** $N_{\text{warm}} = 500$ steps of local prediction-head and encoder training under a shadow status before consistency auditing.

## 3. Observed Quantities
- **Recruitment Rate:**
  - Arm N (Control): $0\%$ (0/5 seeds passive, 0/31 active retries).
  - Arm P (WUP-MDL): $100\%$ (5/5 seeds passive, successfully transitioning $d_t = 3 \rightarrow 4$).
  - Arm O (WUP-PVU): $0\%$ (0/5 seeds recruited; all proposed dimensions rejected due to correlation exceeding the $0.8$ threshold).
- **Centroid Decoding MSE:**
  - Arm N (Control): $130.39$ (unable to represent the 4th object).
  - Arm P (WUP-MDL): $52.68$ (significant representation accuracy improvement).
  - Arm K (Baseline): $62.64$.
- **Cross-Dimension Correlation:**
  - Post-recruitment dimension correlation in Arm P exceeded $0.80$ in all seeds during physical collisions, triggering rejection in Arm O.

## 4. Verdict
- **Hypothesis 1 (WUP-MDL): CONSISTENT.** WUP completely resolved the cold-start rejection bias, enabling $100\%$ recruitment and yielding a centroid decoding MSE of $52.68$ (surpassing the target threshold of $<65.0$ and outperforming the baseline Arm K's $62.64$).
- **Hypothesis 2 (WUP-PVU): REFUTED (Honest Null Result).** WUP-PVU failed to recruit any dimensions because the coordinate dimensions of objects in a shared 1D space are naturally highly correlated ($r > 0.8$). Enforcing low static correlation is physically incompatible with coordinate representation in this environment.

## 5. Construction-vs-Empirical Note
The rejection of new dimensions in Arm N was a mathematical consequence of construction: evaluating a randomly initialized ("cold") predictor against a mature, trained predictor guarantees a high error ratio ($L_{\text{consistency}} \gg 1.0$). The success of Arm P is an empirical validation of optimization timescales, proving that a local 500-step gradient warm-up is sufficient for the predictor to stabilize. The failure of Arm O reveals a fundamental physical constraint of the 1D environment: spatial coordinate representations cannot be mutually orthogonal when entities interact continuously along a single dimension.

## 6. Limitations
- This evaluation was conducted entirely within a 1D physics environment; the correlation constraints of coordinate dimensions may behave differently in 2D or 3D spaces where degrees of freedom are higher.
- The sensitivity of the system to the length of the warm-up window ($N_{\text{warm}}$) was not swept; it is unknown if shorter windows (e.g., 100 steps) would suffice or if longer windows are required as the number of active entities scales.