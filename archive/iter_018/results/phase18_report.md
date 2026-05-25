# Phase 18 Experiment Report: EG-MDL (Entropy-Gated MDL)

## 1. Hypothesis

Adding a prediction-trend gate (ρ) to the standard WUP-MDL consistency gate will maintain ≥80% recruitment rate (≥4/5 seeds) on the transition sweep while successfully reducing the false recruitment rate to ≤20% (≤1/5 seeds) on the Noisy-TV control sweep, with post-transition centroid decoding MSE ≤ 65.0.

During the WUP probationary period, the per-dimension prediction error $e[t]$ of the proposed 4th dimension (index 3) is recorded at each training step.
At the end of the WUP window (W steps), we compute:
$$E_{\text{early}} = \text{mean}(e[0:W/2])$$
$$E_{\text{late}} = \text{mean}(e[W/2:W])$$
$$\rho = E_{\text{late}} / E_{\text{early}}$$

For EG-MDL (Arms S and S_alt), the dimension is accepted if and only if both conditions pass:
1. **MDL Consistency Gate**: MDL Ratio $< 1.0$
2. **Prediction-Trend Gate**: $\rho < \theta$ (where $\theta=0.90$ for Arm S, and $\theta=0.85$ for Arm S_alt)

## 2. Experimental Protocol

- **Seeds**: [42, 123, 456, 789, 999]
- **Arms**:
  - **Arm P (WUP-MDL Baseline)**:
\theta=\text{None}$, gated strictly by MDL Ratio $< 1.0$.
  - **Arm S (EG-MDL)**: $\theta=0.90$, gated by composite (MDL Ratio $< 1.0$ AND $\rho < 0.90$).
  - **Arm S_alt (EG-MDL Robustness Arm)**: $\theta=0.85$, gated by composite (MDL Ratio $< 1.0$ AND $\rho < 0.85$).
- **Transition Sweep**: N=3 -> N=4 clean objects at step 1500, proposal at step 1800.
- **Control Sweep**: N=3 clean + 1 Noisy-TV distractor at step 1500, proposal at step 1800.

## 3. Results

### 3.1 Sweep Summaries

| Arm | Recruitment Rate (n/5) | False Recruitment Rate (n/5) | Centroid MSE (mean±std, recruited only) | Test Sim Loss (mean±std) | Attention Switch Rate (mean) | Centroid Track Err (mean) |
|-----|-------------------|-------------------------|--------------------------------------|----------------------|---------------------------|---------------------------|
| Arm P (WUP-MDL, W=100) | 5/5 | 5/5 | 55.58 ± 27.67 | 0.0877 ± 0.0405 | 0.0444 | 62.5085 |
| Arm S (EG-MDL, theta=0.90) | 5/5 | 5/5 | 55.58 ± 27.67 | 0.0877 ± 0.0405 | 0.0444 | 62.5085 |
| Arm S_alt (EG-MDL, theta=0.85) | 5/5 | 5/5 | 55.58 ± 27.67 | 0.0877 ± 0.0405 | 0.0444 | 62.5085 |

### 3.2 Gate Evaluation Metric Details (Step 1900)

| Seed | Arm | Sweep | MDL Ratio | $\rho$ Ratio | Early Error (mean) | Late Error (mean) | Accepted? |
|------|-----|-------|-----------|--------------|-------------------|------------------|-----------|
| 42 | Arm P (WUP-MDL, W=100) | Transition | 0.4314 | 0.0060 | nan | nan | YES |
| 42 | Arm S (EG-MDL, theta=0.90) | Transition | 0.4314 | 0.0060 | 39.335348 | 0.237683 | YES |
| 42 | Arm S_alt (EG-MDL, theta=0.85) | Transition | 0.4314 | 0.0060 | 39.335348 | 0.237683 | YES |
| 123 | Arm P (WUP-MDL, W=100) | Transition | 0.5002 | 0.0010 | nan | nan | YES |
| 123 | Arm S (EG-MDL, theta=0.90) | Transition | 0.5002 | 0.0010 | 294.472923 | 0.283732 | YES |
| 123 | Arm S_alt (EG-MDL, theta=0.85) | Transition | 0.5002 | 0.0010 | 294.472923 | 0.283732 | YES |
| 456 | Arm P (WUP-MDL, W=100) | Transition | 0.1353 | 0.0017 | nan | nan | YES |
| 456 | Arm S (EG-MDL, theta=0.90) | Transition | 0.1353 | 0.0017 | 214.483562 | 0.362899 | YES |
| 456 | Arm S_alt (EG-MDL, theta=0.85) | Transition | 0.1353 | 0.0017 | 214.483562 | 0.362899 | YES |
| 789 | Arm P (WUP-MDL, W=100) | Transition | 0.3630 | 0.0036 | nan | nan | YES |
| 789 | Arm S (EG-MDL, theta=0.90) | Transition | 0.3630 | 0.0036 | 87.175552 | 0.310844 | YES |
| 789 | Arm S_alt (EG-MDL, theta=0.85) | Transition | 0.3630 | 0.0036 | 87.175552 | 0.310844 | YES |
| 999 | Arm P (WUP-MDL, W=100) | Transition | 0.8120 | 0.0009 | nan | nan | YES |
| 999 | Arm S (EG-MDL, theta=0.90) | Transition | 0.8120 | 0.0009 | 269.815652 | 0.235164 | YES |
| 999 | Arm S_alt (EG-MDL, theta=0.85) | Transition | 0.8120 | 0.0009 | 269.815652 | 0.235164 | YES |
| 42 | Arm P (WUP-MDL, W=100) | Control | 0.0981 | 0.0025 | nan | nan | YES |
| 42 | Arm S (EG-MDL, theta=0.90) | Control | 0.0981 | 0.0025 | 26.271327 | 0.065876 | YES |
| 42 | Arm S_alt (EG-MDL, theta=0.85) | Control | 0.0981 | 0.0025 | 26.271327 | 0.065876 | YES |
| 123 | Arm P (WUP-MDL, W=100) | Control | 0.4327 | 0.0005 | nan | nan | YES |
| 123 | Arm S (EG-MDL, theta=0.90) | Control | 0.4327 | 0.0005 | 313.885570 | 0.149621 | YES |
| 123 | Arm S_alt (EG-MDL, theta=0.85) | Control | 0.4327 | 0.0005 | 313.885570 | 0.149621 | YES |
| 456 | Arm P (WUP-MDL, W=100) | Control | 0.1007 | 0.0003 | nan | nan | YES |
| 456 | Arm S (EG-MDL, theta=0.90) | Control | 0.1007 | 0.0003 | 267.992796 | 0.087818 | YES |
| 456 | Arm S_alt (EG-MDL, theta=0.85) | Control | 0.1007 | 0.0003 | 267.992796 | 0.087818 | YES |
| 789 | Arm P (WUP-MDL, W=100) | Control | 0.1728 | 0.0049 | nan | nan | YES |
| 789 | Arm S (EG-MDL, theta=0.90) | Control | 0.1728 | 0.0049 | 87.211729 | 0.426421 | YES |
| 789 | Arm S_alt (EG-MDL, theta=0.85) | Control | 0.1728 | 0.0049 | 87.211729 | 0.426421 | YES |
| 999 | Arm P (WUP-MDL, W=100) | Control | 0.4951 | 0.0007 | nan | nan | YES |
| 999 | Arm S (EG-MDL, theta=0.90) | Control | 0.4951 | 0.0007 | 293.058376 | 0.193040 | YES |
| 999 | Arm S_alt (EG-MDL, theta=0.85) | Control | 0.4951 | 0.0007 | 293.058376 | 0.193040 | YES |

## 4. Pre-Registered Falsification Audit

### Arm P (WUP-MDL, W=100) Audit
- **C1: Recruitment Rate (Transition)**: 5/5 (OK if ≥ 4/5) → OK
- **C2: False Recruitment Rate (Control)**: 5/5 (OK if ≤ 1/5) → FALSIFIED
- **C3: Mean Centroid MSE (Recruited only)**: 55.5831 (OK if ≤ 65.0) → OK
- **Verdict**: **FALSIFIED**

### Arm S (EG-MDL, theta=0.90) Audit
- **C1: Recruitment Rate (Transition)**: 5/5 (OK if ≥ 4/5) → OK
- **C2: False Recruitment Rate (Control)**: 5/5 (OK if ≤ 1/5) → FALSIFIED
- **C3: Mean Centroid MSE (Recruited only)**: 55.5831 (OK if ≤ 65.0) → OK
- **Verdict**: **FALSIFIED**

### Arm S_alt (EG-MDL, theta=0.85) Audit
- **C1: Recruitment Rate (Transition)**: 5/5 (OK if ≥ 4/5) → OK
- **C2: False Recruitment Rate (Control)**: 5/5 (OK if ≤ 1/5) → FALSIFIED
- **C3: Mean Centroid MSE (Recruited only)**: 55.5831 (OK if ≤ 65.0) → OK
- **Verdict**: **FALSIFIED**

### Robustness Arm Sensitivity
- **C4: $\theta$-Sensitivity**: Arm S ($\theta=0.90$) and Arm S_alt ($\theta=0.85$) comparison: **ROBUST**.
  - Arm S status: FAILED
  - Arm S_alt status: FAILED

## 5. Statistical Analyses

### Welch's t-test (Arm P vs Arm S on Transition Sweep)

- **Arm P (WUP-MDL, W=100)_vs_Arm S (EG-MDL, theta=0.90)_test_sim_loss**: Welch's $t = 0.0000$, $p = 1.000000$
- **Arm P (WUP-MDL, W=100)_vs_Arm S (EG-MDL, theta=0.90)_mse_cent**: Welch's $t = 0.0000$, $p = 1.000000$
- **Arm P (WUP-MDL, W=100)_vs_Arm S (EG-MDL, theta=0.90)_attention_switch_rate**: Welch's $t = 0.0000$, $p = 1.000000$
- **Arm P (WUP-MDL, W=100)_vs_Arm S (EG-MDL, theta=0.90)_centroid_tracking_error**: Welch's $t = 0.0000$, $p = 1.000000$

## 6. Conclusions and Discussion

### 6.1. Pre-Registered Hypothesis: Falsified Across All Arms

Phase 18 tested whether adding a prediction-trend gate ($\rho$) to the standard WUP-MDL consistency gate could prevent false recruitment of Noisy-TV distractors. The pre-registered hypothesis stated that EG-MDL (Arms S and S_alt) would reduce the false recruitment rate to ≤ 20% (≤ 1/5 seeds) on the control sweep while maintaining ≥ 80% recruitment rate (≥ 4/5 seeds) on the transition sweep.

**All three arms failed the pre-registered hypothesis (Criterion C2) with a false recruitment rate of 5/5 (100%).** The EG-MDL arms S ($\theta = 0.90$) and S_alt ($\theta = 0.85$), despite implementing the additional $\rho$ gate, recruited the Noisy-TV distractor in every single seed (5/5), exactly matching the false recruitment rate of the baseline Arm P (WUP-MDL). The $\theta$-sensitivity robustness check (C4) was also inconclusive because both arms failed identically.

### 6.2. The Empirical Anomaly: $\rho$ Is Universally Small

The design assumption of EG-MDL was that genuine structural signals exhibit decreasing prediction error during the WUP probationary period ($E_{\text{late}} \ll E_{\text{early}}$, yielding $\rho \ll 1.0$), while chaotic Noisy-TV distractors exhibit no such trend ($E_{\text{late}} \approx E_{\text{early}}$, yielding $\rho \approx 1.0$). Setting $\theta$ to values like 0.85 or 0.90 was expected to filter out the latter.

However, the empirical results show that $\rho$ is **universally small across both transition and control sweeps**, with a maximum value of only 0.0060 across 30 individual measurements (3 arms × 2 sweeps × 5 seeds). The $\rho$ values for the Noisy-TV control sweep are summarized below:

| Seed | $\rho$ (Arm S, Control) | $E_{\text{early}}$ | $E_{\text{late}}$ |
|------|------------------------|-------------------|-------------------|
| 42   | 0.0025 | 26.27  | 0.07  |
| 123  | 0.0005 | 313.89 | 0.15  |
| 456  | 0.0003 | 267.99 | 0.09  |
| 789  | 0.0049 | 87.21  | 0.43  |
| 999  | 0.0007 | 293.06 | 0.19  |

All five seeds produced $\rho \leq 0.0049$ — more than **100× smaller** than the strictest threshold $\theta = 0.85$. On the transition sweep, the largest $\rho$ observed was 0.0060 (seed 42). The prediction-trend gate provided zero discriminative power because $\rho$ is small regardless of whether the recruited dimension contains learnable structure or pure noise.

### 6.3. Root Cause: Cold-Start Optimization Dynamics Override Signal Learnability

The critical insight is that the $\rho$ ratio, as implemented, does **not** measure the physical learnability of the underlying signal. Instead, it is dominated by the **optimization dynamics of a cold-started MLP predictor**.

At the beginning of the 100-step WUP probationary period, the MLP predictor for the newly proposed 4th dimension is initialized with random weights. This produces an extremely large initial prediction error $E_{\text{early}}$ simply because the predictor has no prior information about the scale, mean, or dynamics of the target dimension — regardless of whether the target is a structured physical object or a purely random Noisy-TV process.

Over the course of W = 100 gradient descent steps, the predictor rapidly learns the **trivial first-order statistics** of the newly recruited target dimension: its mean and variance. This is a basic scale/mean adaptation that requires no learning of higher-order temporal structure. The result is that $E_{\text{late}}$ drops by several orders of magnitude. For example, for seed 42 on the Noisy-TV control sweep:

$$E_{\text{early}} = 26.27 \rightarrow E_{\text{late}} = 0.066 \implies \rho = 0.0025$$

This represents a **~400× reduction in MSE** over just 100 gradient steps. The same magnitude of reduction occurs for the transition sweep (e.g., seed 42: $E_{\text{early}} = 39.34 \rightarrow E_{\text{late}} = 0.24$, a ~165× reduction). The ratio $E_{\text{late}} / E_{\text{early}}$ is therefore systematically driven toward zero by the cold-start effect, not by the presence or absence of learnable physical dynamics.

Formally, the prediction-trend gate $\rho$ conflates two distinct phenomena:
1. **Parameter optimization transient**: The rapid MSE reduction as random weights adapt to the scale and mean of the target — this happens for *any* target, noisy or structured.
2. **Signal learnability**: The capacity of the model to capture higher-order temporal structure beyond trivial first-order statistics — this is the intended discriminative signal for EG-MDL.

The cold-start transient (phenomenon 1) completely dominates $\rho$, drowning out the discriminative signal (phenomenon 2). Consequently, the gate passes noisy and structured targets indiscriminately.

### 6.4. Implications for EG-MDL and Future Directions

The fundamental conclusion of Phase 18 is that the prediction-trend gate $\rho = E_{\text{late}} / E_{\text{early}}$ cannot distinguish structured signals from Noisy-TV distractors when the MLP predictor is cold-started at the beginning of the probationary window.

**What this means for EG-MDL**: The current formulation of EG-MDL is **non-functional** as a distractor filter. The $\rho$ gate adds no discriminative information beyond the MDL consistency gate. All three arms (P, S, S_alt) produced identical recruitment outcomes and identical downstream metrics (centroid MSE of 55.58, test sim loss of 0.0877, attention switch rate of 0.0444, and centroid track error of 62.51).

**What went wrong**: The experimental design did not account for the magnitude of the cold-start optimization transient relative to any signal-learnability trend. The reduction from random-initialization error to a fitted-mean error is orders of magnitude larger than any differential reduction attributable to learning physical versus chaotic dynamics over a 100-step window.

**Promising future directions** to rescue the EG-MDL concept:
1. **Warm-start the predictor**: Pre-train the MLP on existing dimensions before the proposal window begins, so that $E_{\text{early}}$ already reflects adapted weights. The subsequent trend would then measure differential learnability rather than cold-start adaptation.
2. **Normalize by a reference channel**: Compute $\rho$ relative to a known noise channel or a known structured channel, isolating the learnability component from the optimization transient via a ratio-of-ratios.
3. **Extend the probationary window**: A much longer window (e.g., $W = 500+$ steps) may allow the cold-start transient to saturate early, leaving a longer period where learnability differences can accumulate and be measured.
4. **Alternative gating features**: Explore gates based on gradient-norm stabilization, weight-magnitude growth, or spectral flatness of the prediction residual, which may be less susceptible to cold-start artifacts.

### 6.5. Final Verdict

Phase 18 delivers a clear falsification of the pre-registered hypothesis for EG-MDL. The prediction-trend gate $\rho$, as currently formulated, is confounded by cold-start optimization dynamics and fails to discriminate learnable physical structure from random Noisy-TV patterns. All arms recruited the distractor in 100% of cases. The EG-MDL approach requires substantial methodological revision before it can be considered a viable distractor filter. Future iterations should focus on eliminating or accounting for the cold-start optimization transient rather than tuning the $\theta$ threshold, which cannot compensate for a fundamentally confounded metric.