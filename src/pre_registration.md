---
# RDF Scientific Pre-Registration

*   **Iteration:** 009
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
An adaptive, surprise-modulated curriculum for the soft spatial variance regularization weight $\lambda(t)$—where $\lambda(t)$ starts near $0$ and scales up to $\lambda_{max} = 0.10$ as the running mean of local surprise $\bar{S}(t)$ decays (e.g., $\lambda(t) = \lambda_{max} \cdot [1 - \min(1, \bar{S}(t) / S_0)]$ where $S_0$ is a normalization constant)—will resolve the localization-prediction trade-off.
Specifically, during the $N=3 \rightarrow N=4$ object generalization transition under a 5-seed sweep:
1. It will achieve tight spatial localization of the recruited channel, yielding a final mean soft spatial variance of $< 100.0$ (comparable to the static $\lambda=0.1$ bottleneck).
2. It will simultaneously preserve representation capacity, achieving a post-hoc linear centroid decoding MSE of $< 65.0$ (outperforming both the static $\lambda=0.01$ bottleneck's MSE of 69.11 and the static $\lambda=0.1$ bottleneck's MSE of 106.87).
3. It will increase the mean absolute Pearson correlation $|r|$ of physical coordinates on the recruited dimension to $\ge 0.40$ (compared to $0.2907$ achieved by static $\lambda=0.1$).
4. The adaptive curriculum must not statistically degrade the final temporal prediction loss, achieving a final test L2/surprise loss ratio of < 1.15 compared to the static lambda = 0.01 baseline (Arm A).

## 2. Falsification Criterion
The hypothesis will be falsified if ANY of the following outcomes are observed over the 5-seed comparative evaluation:
1. The surprise-modulated curriculum fails to localize the channel, resulting in a mean soft spatial variance of $\ge 150.0$.
2. The final centroid decoding MSE is $\ge 69.11$ (failing to outperform the static $\lambda=0.01$ baseline), or $\ge 83.12$ (failing to outperform the control/no-bottleneck baseline).
3. The average absolute Pearson correlation $|r|$ of the physical centroid coordinate against the recruited latent dimension remains $< 0.35$.
4. The representation collapse rate under the adaptive curriculum is $> 0.0\%$ (i.e., at least one of the 5 seeds experiences collapse).
5. The final mean test temporal prediction loss (test L2/surprise loss) of the adaptive curriculum (Arm C) is statistically degraded (defined as a >15% increase, i.e., ratio >= 1.15) compared to the static lambda = 0.01 baseline (Arm A).

## 3. Proposed Method
1. **Surprise-Modulated Controller**: Implement an adaptive controller for the spatial bottleneck weight $\lambda(t)$. $\bar{S}(t)$ will track the exponential moving average of local temporal-prediction surprise of the recruited dimension. The weight will be defined as $\lambda(t) = \lambda_{max} \cdot \max(0, 1 - \bar{S}(t) / S_0)$, where $\lambda_{max} = 0.10$ and $S_0 = 0.15$.
   - To guard against rapid oscillations, apply a step-to-step rate limit (clipping change to maximum +/-0.002 per step).
2. **Experimental Run & Environment**:
   - Use the 1D physics sandbox with parameterizable environmental variation.
   - Start training with $N=3$ objects for 1000 steps (passive), then transition to $N=4$ objects for 1000 steps with active closed-loop probing.
   - Trigger dimension recruitment and activate the adaptive bottleneck regulator on the newly recruited channel.
3. **Comparative Sweeps**:
   - Run a 5-seed sweep across 3 experimental arms: Arm A (static $\lambda = 0.01$), Arm B (static $\lambda = 0.10$), and Arm C (Experimental DSMC).
4. **Code and Scripts**:
   - Continue or create the script `run_phase9_experiments.py` to run this 5-seed comparative evaluation.
   - Calculate, log, and plot: soft spatial variance, centroid decoding MSE, absolute Pearson correlation $|r|$, latent dimension variance, and test temporal prediction loss (test L2/surprise loss).

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*