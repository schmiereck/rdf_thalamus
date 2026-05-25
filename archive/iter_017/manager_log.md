# Research Manager Log - Iteration 017

## Iteration 017 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
An encoder-only, prediction-independent gating mechanism based on Encoder-only Smoothness-Uniqueness Gating (ESUG)—which combines linear residual variance (Uniqueness Ratio R_unique > 0.15) and first-difference temporal variance (Smoothness Ratio lambda < 0.5)—can successfully identify and recruit a newly required representation dimension during the N=3 to N=4 object transition without requiring any probationary warm-up period (W=0) for a temporal predictor head. Specifically, this prediction-independent gate will achieve 100% recruitment rate across 5 seeds, reduce post-transition centroid decoding MSE to < 55.0, and decrease decision latency by at least 50% compared to the prediction-dependent WUP-MDL baseline (Arm P), while maintaining 0% false recruitment under high-frequency noise distractors.

**Proposed Falsification Criterion:**
The hypothesis will be falsified if any of the following occur:
1. The recruitment rate of Arm Q (ESUG-100) or Arm Q_fast (ESUG-30) is less than 100% (5/5 seeds) during the N=3 to N=4 transition.
2. The mean post-transition centroid decoding MSE of Arm Q or Arm Q_fast is >= 55.0.
3. The decision latency (steps from transition to recruitment) of Arm Q_fast is not significantly lower than Arm P (WUP-MDL, W=100), or is > 40 steps.
4. The false recruitment rate of Arm Q or Arm Q_fast in the N=3 noise-distractor control group is > 0% (i.e., it incorrectly recruits a noisy dimension).

**Proposed Method:**
1. Modify `src/thalamus.py` to implement the ESUG gating mechanism. ESUG calculates the uniqueness ratio R_unique (using linear projection residuals of the proposed dimension onto the active dimensions over a sliding buffer B) and the temporal smoothness ratio lambda (first-difference variance normalized by total variance).
2. Implement Arm Q (ESUG with B=100) and Arm Q_fast (ESUG with B=30).
3. Run a 5-seed comparative sweep comparing Arm P (WUP-MDL baseline), Arm Q, and Arm Q_fast under the N=3 to N=4 transition.
4. Run a parallel control sweep where the environment remains at N=3, but a noisy-TV distractor is introduced as a proposed dimension, evaluating the false recruitment rate of Arm P, Arm Q, and Arm Q_fast.
5. Measure and report: recruitment rate, centroid decoding MSE, decision latency, and false recruitment rate across all arms and seeds.

---

## Iteration 017 -> Planner [Strategic Guidance]

### Strategic Guidance: Manager's Note

To the Planner, and the executing sub-agents:

While the shift toward encoder-side, prediction-independent gating is a logical reaction to the temporal-prediction latency of the WUP framework, your proposed **Encoder-only Smoothness-Uniqueness Gating (ESUG)** plan risks falling directly back into the physical bottlenecks we identified in Phase 16. 

Apply the following strategic corrections before proceeding:

#### 1. The Linear-Independence Trap in 1D Space (Skeptic Mode)
In Phase 16, we established that **absolute coordinate decorrelation metrics fail because all object trajectories in a 1D sandbox are highly correlated by construction**. 
*   Your proposed $R_{\text{unique}} > 0.15$ is a linear projection residual threshold. In a 1D space, a new object's coordinate is often highly collinear with existing coordinates over short windows. 
*   **Mandate:** If $R_{\text{unique}} > 0.15$ triggers perpetual rejection or requires lowering during execution, **do not tune the parameter post-hoc to force a "discovery."** Treat it as an honest null result, confirming that linear algebraic metrics are fundamentally poorly suited for low-dimensional spatial environments compared to predictive MDL ratios.

#### 2. The Predictor Deficit under $W=0$
Even if ESUG successfully recruits a dimension instantaneously ($W=0$) based on encoder metrics, the downstream temporal predictor for that new dimension remains completely untrained. 
*   Because **surprise** drives our Thalamic Gating and CLTS motor controller, this newly recruited but un-predicted dimension will immediately exhibit a massive temporal prediction error spike.
*   **Mandate:** Monitor and report the stability of the Attention Token and the CLTS motor loop immediately following recruitment. If recruiting a dimension with an untrained predictor induces chaotic attention-switching or degrades motor tracking, you must characterize this as a **representation-prediction temporal mismatch pathology**.

#### 3. Pre-Registration Rigour & Code of Conduct
The Orchestrator will automatically write and commit your hypothesis and falsification criteria to `src/pre_registration.md`. 
*   **Correction to Criterion 3:** "Significantly lower" is mathematically vague. You must reformulate this to require a quantitative threshold (e.g., a Welch's t-test showing $p < 0.05$ AND an absolute latency reduction of at least 50 steps compared to Arm P). 
*   All sub-agents must read `src/pre_registration.md` at the start of the execution phase and adhere to its criteria without deviation. No post-hoc modification of success thresholds is permitted.

---

