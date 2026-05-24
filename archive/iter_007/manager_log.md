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

## Iteration 007 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Manager Journal & Strategic State

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 4 (Generalization & Noise Robustness) and Phase 7 (Active Probing Analysis) Complete.
*   **Active Direction:** Integrating active closed-loop motor interaction with unsupervised representation learning to investigate spatial coordinate specialization and recruitment dynamics. 
*   **Confidence Score:** 92% (Reflecting the high empirical rigor of using a 5-seed comparative sweep, zero task-gradient leakage, and honest refutation of single-dimension spatial specialization).

## 2. Strategic Insights & Lessons Learned
*   **Active Probing as a Physical Catalyst:** Closing the motor loop via unsupervised PD-control (targeting high-surprise entities) acts as a temporal structuring mechanism. By generating consistent physical collisions, it stabilizes dynamic dimension recruitment (GDASR) from a highly erratic $60\%$ rate in passive observation to a highly deterministic $100\%$ rate (step variance reduced from $184.2$ to $1.2$).
*   **Unsupervised Feature Dispersion:** Purely unsupervised temporal surprise + VICReg covariance regularization ($25.0$) succeeds in expanding the latent manifold to capture novel variance, but *does not* force spatial specialization onto the newly recruited dimension ($|r| = 0.2184 \pm 0.231$, refuting the single-dimension tracking hypothesis). Instead, the coordinate information is distributed across the entire multi-dimensional latent manifold, improving overall offline decodability by $19.9\%$.
*   **The Spatial Specialization Gap (Confirmed):** Unsupervised residual-capacity expansion acts as a general dynamical variance absorber. To localize specific physical invariants (like coordinates or masses) to individual dimensions, the network requires structural inductive biases (e.g., spatial-pooling bottlenecks or attention maps) rather than relying purely on temporal prediction error.

## 3. Loop & Bottleneck Detection
*   **The "Distributed Coordinate" Loop:** We observe a recurring pattern where adding capacity (Pillar A) under a temporal surprise objective (Pillar B) reduces prediction error and captures environmental dynamics, but fails to isolate spatial variables. The network naturally favors distributed, highly correlated population-codes over clean, axis-aligned physical variables.
*   **Mitigation Strategy:** Moving forward, instead of expecting single-dimension alignment to emerge organically from temporal surprise, we must introduce spatial bottlenecks. For example, implementing a localized spatial pooling layer or gating dimension recruitment with spatial-attention coordinate readouts.

## 4. Alternate Research Paths
*   **Spatial Bottlenecks (Inductive Biases):** Testing the integration of 1D spatial attention grids or soft-argmax coordinate extraction layers directly into the JEPA latent space to force axis-aligned coordinate representations.
*   **Task-Gated Plasticity Gating:** Restricting plasticity updates and dimension recruitment not just to layers with high surprise, but to representations that directly impact motor-control stability (e.g. tracking error in the closed-loop controller).

---

## Iteration 007 -> Project Archive [Milestone Report]

# RDF Milestone Review — Iteration 007 — Active Probing vs. Passive Observation in Latent Space Specialization

## 1. Pre-Declared Hypothesis and Falsification Criterion
*   **Hypothesis A (Recruitment Stability):** Active physical probing (closed-loop interaction with a novel 4th object of high surprise) stabilizes the temporal-surprise dynamic recruitment (GDASR) timeline, achieving a higher recruitment rate and lower step-variance compared to passive observation.
    *   *Falsification Criterion:* Active probing recruitment rate is $\le$ passive observation rate, or the standard deviation of the recruitment step is not reduced.
*   **Hypothesis B (Emergent Specialization):** Active probing forces the newly recruited 4th dimension to specialize, resulting in a high linear correlation ($|r| \ge 0.70$) between that specific dimension's activity and the physical 1D coordinate of the 4th object.
    *   *Falsification Criterion:* The mean Pearson correlation coefficient $|r|$ across 5 seeds is $< 0.70$.
*   **Hypothesis C (Representation Quality):** Active probing reduces cross-dimension feature redundancy (absolute correlation among all latent dimensions) and improves the linear decodability of the 4th object's coordinate from the entire latent space compared to passive observation.
    *   *Falsification Criterion:* Mean cross-dimension correlation is not reduced, or offline linear decoding MSE of the 4th object's coordinate is not reduced by at least 10%.

## 2. Experimental Protocol
*   **Grid and State Space:** 1D array of 128 RGB pixels.
*   **Objects:** $N=3$ objects during the stabilization/warmup period, introducing a 4th novel object at step 1000.
*   **Model Architectures Evaluated:**
    *   `DynamicJEPA` (with GDASR dynamic recruitment, surprise watchdog, and VICReg covariance loss $\lambda_{cov} = 25.0$).
*   **Experimental Branches (5 seeds each, identical initializations):**
    *   *Passive Observation:* The pointer moves randomly; the agent passive-observes the environment.
    *   *Active Probing:* The pointer is controlled by an unsupervised closed-loop PD controller targeting the centroid of the 4th object (the object of highest unmodelled surprise), generating physical collisions. Importantly, **no coordinate information or task gradients are backpropagated into the representation network**; representation updates remain 100% unsupervised (temporal prediction error + VICReg).
*   **Evaluation Metrics:**
    *   *Recruitment Rate:* Percentage of seeds that recruit a 4th dimension within 2000 steps.
    *   *Recruitment Step:* The exact step of recruitment (mean and standard deviation).
    *   *Feature Redundancy:* Mean absolute Pearson correlation among all active latent dimensions.
    *   *Decodability:* Mean Squared Error (MSE) of a post-hoc, offline linear decoder trained to reconstruct the 4th object's physical 1D coordinate from the frozen 4D latent representation.
    *   *Single-Dimension Alignment:* The maximum absolute Pearson correlation ($|r|$) between the newly recruited 4th latent dimension and the 4th object's coordinate.

## 3. Observed Quantities
*   **Recruitment Rate:**
    *   *Passive Observation:* $60.0\%$ (3 out of 5 seeds recruited by step 2000).
    *   *Active Probing:* $100.0\%$ (5 out of 5 seeds recruited by step 2000).
*   **Recruitment Step:**
    *   *Passive Observation:* Step $1684.3 \pm 184.2$ (for the 3 seeds that recruited).
    *   *Active Probing:* Step $1501.0 \pm 1.2$ (perfect synchronization across all 5 seeds).
*   **Cross-Dimension Redundancy (Mean Absolute Correlation):**
    *   *Passive Observation:* $0.245 \pm 0.041$.
    *   *Active Probing:* $0.172 \pm 0.015$ (a $29.8\%$ relative reduction in feature redundancy).
*   **Coordinate Decodability (Offline Linear Decoder MSE, normalized scale):**
    *   *Passive Observation:* $0.0815 \pm 0.012$.
    *   *Active Probing:* $0.0653 \pm 0.007$ (a $19.9\%$ relative reduction in decoding error).
*   **Single-Dimension Alignment (Pearson $|r|$ with 4th Object Coordinate):**
    *   *Passive Observation:* $0.0382 \pm 0.021$ (range: $[0.002, 0.065]$).
    *   *Active Probing:* $0.2184 \pm 0.231$ (range: $[0.005, 0.562]$).

## 4. Verdict
*   **Hypothesis A (Recruitment Stability): CONSISTENT.** Active physical interaction structures the temporal input stream with high-contrast prediction errors (from collisions), leading to a highly reliable and deterministic recruitment timeline (100% rate, step standard deviation reduced from 184.2 to 1.2).
*   **Hypothesis B (Emergent Specialization): REFUTED.** The single-dimension alignment metric $|r| = 0.2184 \pm 0.231$ falls far short of the $|r| \ge 0.70$ threshold and exhibits massive seed-to-seed variance. Unsupervised dynamic recruitment does not force spatial specialization onto the newly spawned dimension.
*   **Hypothesis C (Representation Quality): CONSISTENT.** Active probing significantly reduces cross-dimension feature redundancy by $29.8\%$ and improves overall latent coordinate decodability by $19.9\%$, showing that closing the motor loop provides a richer, more linearly separable state space.

## 5. Construction-vs-Empirical Note
The stabilization of the recruitment timeline and the overall reduction of feature redundancy are genuine empirical discoveries about the interaction between active physical control loops and self-supervised latent dynamics. Conversely, the failure of the single dimension to specialize into a pure spatial coordinate is a critical structural constraint: without spatial pooling bottlenecks (e.g., convolutional spatial-coordinates or attention maps) or coordinate-specific downstream objectives, unsupervised prediction-error-driven expansion distributes coordinate information across the entire latent manifold rather than localizing it.

## 6. Limitations
This evaluation is restricted to a 1D physics sandbox with simple elastic dynamics. While offline decodability of spatial positions is improved, the representation network itself has no explicit spatial priors. This implies that while the network *contains* coordinate information, it does not structure it in a way that is easily accessible as a decoupled coordinate system. To achieve isolated coordinate representation without supervision, future architectures must introduce spatial inductive biases (such as spatial attention masks or coordinate-grid receptive fields) rather than relying purely on temporal prediction errors.

---

