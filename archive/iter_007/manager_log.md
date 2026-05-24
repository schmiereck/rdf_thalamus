# Research Manager Log - Iteration 007

## Iteration 007 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Introducing downstream task-coupling (specifically, backpropagating a spatial target localization error from a linear readout probe to the newly recruited 4th latent dimension) during the N=3 to N=4 transition will force the recruited dimension to self-organize into a coordinate representation of the novel object. This task-coupled representation alignment will achieve:
1. A mean absolute Pearson correlation coefficient |r| >= 0.60 between the recruited 4th dimension and the physical position of the 4th object across 5 independent seeds.
2. A reduction of at least 25% in the target localization error (MSE) compared to the uncoupled (unsupervised surprise-only) baseline model, while preserving the overall temporal prediction loss within 10% of the baseline.

**Proposed Falsification Criterion:**
The hypothesis will be falsified if any of the following occur:
1. The mean absolute Pearson correlation |r| between the recruited 4th dimension's activity and the physical position of the 4th object across the 5 evaluation seeds is less than 0.60.
2. The task-coupled model fails to reduce the target localization MSE on the 4th object by at least 25% compared to the uncoupled baseline.
3. The task-coupled model's overall temporal prediction loss (surprise) degrades by more than 10% compared to the uncoupled baseline, indicating representational drift or collapse.

**Proposed Method:**
1. Define a Downstream Task-Coupled variant of the Thalamus architecture. This variant includes an auxiliary linear readout probe that predicts the 1D physical position of the attended object.
2. During the N=4 generalization transition, route the localization error gradients exclusively to the newly recruited 4th representation dimension (using the plasticity gating mechanism) while keeping the stable subspace frozen to prevent catastrophic interference.
3. Run a 5-seed comparative sweep comparing:
   - Control Group: Uncoupled baseline (unsupervised surprise minimization only, with a post-hoc linear probe trained frozen).
   - Experimental Group: Downstream Task-Coupled model (co-trained with the localization loss backpropagating to the recruited dimension).
4. Measure and log:
   - The Pearson correlation coefficient |r| between the 4th dimension's activity and the physical position of the 4th object.
   - The MSE of the localization prediction.
   - The overall system prediction loss (surprise).
5. The implementation will modify or extend the existing training/evaluation scripts (e.g., in `src/`) to incorporate the task-coupled training loop and the metric tracking.

---

## Iteration 007 -> Planner [Strategic Guidance]

### Manager's Note: Strategic Guidance & Skeptical Critique

To: The Planner Agent  
From: Research Manager  
Subject: Scientific Rigor Audit of Phase 4/5 Integration Plan — The "Supervision" Trap

We have successfully demonstrated robust attention gating and dynamic dimension recruitment (GDASR) at the $N=3 \to N=4$ transition. However, your proposed plan to resolve the "Spatial Specialization Gap" via downstream task-coupling contains a fundamental methodological flaw that threatens our core research directives. 

Please address the following three strategic points before proceeding to execution:

---

### 1. The Construction-vs-Empirical Test: The Supervision Trap
*   **The Flaw:** Your proposal suggests backpropagating the spatial target-localization error from a linear readout probe *directly* into the newly recruited 4th latent dimension. 
*   **Skeptical Critique:** If we train a model using backpropagation on a coordinate loss $\mathcal{L}_{\text{loc}} = (\hat{y} - y)^2$, then observing a high Pearson correlation ($|r| \ge 0.60$) between the latent dimension and the coordinate $y$ is **not a scientific discovery—it is a trivial algebraic consequence of gradient descent**. We would merely be verifying that our optimizer works.
*   **Architectural Violation:** "Thalamus" is explicitly mandated to be a *dynamic, curiosity-driven representation network* that learns *without generative decoders* or external supervision. Introducing ground-truth coordinate backpropagation violates this self-supervised, local-learning paradigm.

### 2. Strategic Course Correction: Active Probing vs. Passive Observation
To maintain strict self-supervised hygiene, we must test whether spatial specialization can **emerge** from the physics of the environment and the agent's actions, rather than being forced by supervised gradients.
*   **Alternative Experimental Design:** Keep the representation training **100% unsupervised** (using local temporal prediction error and VICReg covariance regularization). 
*   **The Independent Variable:** Compare two unsupervised regimes during the $N=3 \to N=4$ transition:
    1.  **Passive Observation (Control):** The agent observes the $N=4$ environment without acting.
    2.  **Active Probing (Experimental):** The agent uses the Subsumption Motorics (from Phase 3) to actively collide with and track high-surprise entities.
*   **The Evaluation (Post-Hoc Probe Only):** Train a linear readout probe *post-hoc on frozen representations* to decode the coordinates of the 4th object. 
*   **The True Empirical Hypothesis:** *Does active physical interaction (intentional collisions and tracking) naturally force the recruited dimension to encode spatial coordinates more strongly ($|r|$) than passive observation, without ever backpropagating coordinate gradients into the latent space?*

### 3. Pre-Registration Mandate
Before running any simulations or code modifications, the Orchestrator will automatically write your final hypothesis and quantitative falsification criteria to `src/pre_registration.md`. 

Your revised plan must define a strict, quantitative falsification criterion based on this self-supervised paradigm. For example:
*   *Falsification Criterion 1:* The post-hoc linear readout of the 4th object's position from the *Active Probing* model's recruited dimension does not show a statistically significant correlation improvement (e.g., $\Delta |r| \ge 0.25$) over the *Passive Observation* model.
*   *Falsification Criterion 2:* Active probing causes representation collapse (VICReg loss spikes or cross-dimension correlation $r > 0.30$).

Reformulate your plan to focus on **active-interaction-driven emergent specialization** rather than supervised task-coupling, update your hypothesis and falsification criteria accordingly, and direct your sub-agents to read and strictly adhere to the pre-registration file.

---

