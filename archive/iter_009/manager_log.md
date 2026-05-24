# Research Manager Log - Iteration 009

## Iteration 009 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
A Dynamic Surprise-Modulated Spatial Bottleneck Curriculum (DSMC)—which dynamically scales the spatial variance penalty weight $\lambda_t$ as an inverse function of the running average of local prediction surprise $\bar{S}_t$ via $\lambda_t = \lambda_{max} \cdot \exp(-\gamma \cdot \bar{S}_t)$ (where $\lambda_{max} = 0.10$, $\gamma = 10.0$, and $\bar{S}_t$ is the exponentially-weighted moving average of local temporal prediction error)—resolves the trade-off between spatial localization and cognitive predictive capacity. Specifically, across a 5-seed comparative sweep under the 1D physics sandbox with the N=3 to N=4 generalization transition, DSMC will simultaneously achieve:
1. Tight spatial localization: mean soft spatial variance of the recruited dimension $\le 120.0$ (comparable to the static strong bottleneck of $\lambda = 0.10$ which achieved $60.29$ but suffered from severe prediction degradation).
2. Superior predictive capacity: mean post-hoc centroid decoding MSE $\le 69.11$ (matching or exceeding the gentle bottleneck of $\lambda = 0.01$, which achieved $69.11$ but had large spatial spread).
3. Complete structural stability: 0.0% representation collapse rate across all 5 seeds (due to low-regularization exploration during early training phases and post-transition shock).

**Proposed Falsification Criterion:**
The hypothesis will be falsified if the DSMC configuration:
1. Fails to achieve a mean soft spatial variance of the recruited channel $\le 120.0$ across the 5 evaluation seeds, OR
2. Fails to achieve a mean centroid decoding MSE $\le 72.0$ across the 5 evaluation seeds, OR
3. Experiences a representation collapse rate $> 0.0\%$ (any of the 5 seeds collapsing).

**Proposed Method:**
1. Implement a local surprise tracker in the representation network module (e.g., in `src/models.py`) that computes the exponentially-weighted moving average (EWMA) of the local temporal-prediction surprise: $\bar{S}_t = \alpha \cdot \bar{S}_{t-1} + (1 - \alpha) \cdot S_t$, with smoothing factor $\alpha = 0.95$ and initialized at $1.0$.
2. Implement the DSMC controller: at each training step, calculate the dynamic spatial bottleneck penalty weight $\lambda_t = \lambda_{max} \cdot \exp(-\gamma \cdot \bar{S}_t)$, where $\lambda_{max} = 0.10$ and scaling factor $\gamma = 10.0$. This ensures that when surprise is high (early in training or immediately post-transition), $\lambda_t \to 0$ to maximize representational capacity, and as surprise decays, $\lambda_t \to 0.10$ to enforce localization.
3. Modify the training loop (e.g., in `src/train.py` or the corresponding training runner) to update, apply, and log $\lambda_t$ and $\bar{S}_t$ at each timestep.
4. Run a 5-seed comparative sweep comparing three experimental conditions:
   - Arm A (Gentle): Static bottleneck with fixed $\lambda = 0.01$.
   - Arm B (Strong): Static bottleneck with fixed $\lambda = 0.10$.
   - Arm C (Experimental): Dynamic bottleneck with DSMC ($\lambda_{max}=0.10, \gamma=10.0$).
5. Evaluate each arm on: (a) average centroid decoding MSE, (b) average soft spatial variance of the recruited channel, (c) representation collapse rate, and (d) the dynamic trajectory of $\lambda_t$ and surprise across the transition.

---

## Iteration 009 -> Planner [Strategic Guidance]

# Manager's Note: Strategic Guidance for Iteration 9

Your proposed plan to address the **Spatial Specialization Gap** using a **Dynamic Surprise-Modulated Spatial Bottleneck Curriculum (DSMC)** is conceptually sound, but it contains a few scientific loopholes that must be tightened before pre-registration and execution. 

Please adjust your plan according to the following directives:

### 1. Tighten the Hypothesis-to-Criterion Alignment
There is a discrepancy between your stated hypothesis and your falsification criteria. Your hypothesis claims that DSMC will achieve a mean post-hoc centroid decoding MSE of $\le 69.11$ (matching or exceeding the gentle static bottleneck), yet your falsification criterion allows a looser threshold of $\le 72.0$. 
*   **Correction:** Align these two. Set the falsification threshold for the mean centroid decoding MSE to **$\le 70.0$** to ensure that any "success" is genuinely comparable to the gentle bottleneck baseline ($69.11$) rather than a statistical regression toward the unconstrained control.

### 2. Close the "Inactive Curriculum" Loophole
A key risk of surprise-dependent scheduling is that if temporal prediction surprise $\bar{S}_t$ never sufficiently decays, the dynamic penalty $\lambda_t$ will remain permanently small (hovering near the gentle regime of $\approx 0.01$). Under this failure mode, the model would trivially achieve low prediction error because it never actually localized.
*   **Mandate:** You must introduce a **curriculum activity sanity check** in your evaluation. Log and report the average final penalty weight $\lambda_T$ at the end of training across all 5 seeds. To assert that the curriculum successfully executed, the final penalty must satisfy **$\text{Mean}(\lambda_T) \ge 0.05$** (at least half of $\lambda_{max}$). If the curriculum fails to ramp up to this level of regularization but passes the other metrics, it must be reported as a failure of the curriculum to activate, not a successful resolution of the trade-off.

### 3. Pre-Registration & Scientific Language Discipline
*   The Orchestrator will automatically write and commit your exact hypothesis, parameters, and falsification criteria to `src/pre_registration.md` at the start of the execution phase. Your sub-agents must read this file and strictly adhere to it.
*   Avoid any predictive claims of "resolving" or "solving" the trade-off in the final reporting unless both the spatial variance ($\le 120.0$) and decoding MSE ($\le 70.0$) targets are simultaneously achieved under the active-curriculum condition ($\lambda_T \ge 0.05$).
*   If the experiment shows that the trade-off cannot be bypassed (e.g., if any increase in $\lambda_t$ immediately degrades temporal prediction regardless of training stage), report this null result clearly. Mapping a fundamental Pareto frontier in representation learning is a highly valuable scientific contribution.

---

