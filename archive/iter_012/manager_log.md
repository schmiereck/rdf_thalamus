# Research Manager Log - Iteration 012

## Iteration 012 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Coupling the stable non-parametric soft-argmax projection of Arm F with a Closed-Loop Thalamic Subsumption (CLTS) motor policy (Arm G)—where a high-level epistemic collision controller is dynamically activated (subsumed) proportional to the local predictive surprise of the attended object—will significantly accelerate the adaptation and predictive accuracy of physical dynamics under parameter shifts (such as a 2x shift in object mass) compared to both passive observation and random exploration, without destabilizing the representational grounding of the coordinate encoder.

Specifically, CLTS will:
1. Accelerate the convergence of temporal prediction models on post-collision dynamics.
2. Maintain representational stability (soft spatial variance < 20.0, centroid decoding MSE < 85.0) despite active, self-generated physical manipulation of the environment.

**Proposed Falsification Criterion:**
The hypothesis will be proven false if any of the following quantitative conditions are met:
1. Predictive Performance: Arm G (CLTS) does not achieve at least a 20% lower post-collision latent prediction MSE compared to the random exploration baseline (Arm F-Random) within 1500 steps of evaluation after a 2x mass perturbation is introduced.
2. Representational Instability: The soft spatial variance of Arm G's coordinate encoder increases above 20.0, or the centroid decoding MSE exceeds 85.0 during active closed-loop interaction, showing that the closed-loop feedback loop destabilizes the non-parametric projection.
3. Adaptation Efficiency: The number of interaction steps required for the model's prediction error to return to pre-perturbation baseline is not at least 15% shorter than that of the random exploration baseline.

**Proposed Method:**
1. Implement the Closed-Loop Thalamic Subsumption (CLTS) Controller:
   - Design a 3-layer subsumption controller:
     * Layer 1 (Reflexive): Align the agent's 1D pointer with the spatial coordinate of the object currently holding the Thalamic Attention Token (based on highest normalized local surprise).
     * Layer 2 (Kinematic Tracking): Match the velocity of the attended object.
     * Layer 3 (Epistemic Probing): When local prediction error (surprise) of the attended object exceeds a dynamic threshold, override lower layers to accelerate directly towards the object and trigger a "push" action to induce an active collision.
2. Run a 5-seed comparative sweep across three experimental arms:
   - Arm F-Passive: Passive observation (no motor actions, objects move naturally).
   - Arm F-Random: Random motor actions (uniform random acceleration and push).
   - Arm G (CLTS): The proposed closed-loop surprise-modulated subsumption motorics.
3. Introduce Perturbation Phase:
   - Train normally for 1500 steps.
   - At step 1501, abruptly change the mass and elasticity of one of the objects by a factor of 2x.
   - Track the post-collision prediction error recovery curves, coordinate soft spatial variance, and centroid decoding MSE for all arms.
4. Modified/Created Files:
   - Create/modify `src/motor.py` to define the CLTS controller.
   - Modify `src/train.py` or the corresponding experiment execution script to support closed-loop motor coupling, online mass perturbation, and metric logging.

---

## Iteration 012 -> Planner [Strategic Guidance]

# Manager's Note: Strategic Guidance for the Closed-Loop Subsumption Motorics Integration (Phase 12)

The transition from a passive, non-parametric spatial representation (Arm F) to a closed-loop active-probing motor system (Arm G / CLTS) represents the final architectural integration of the Thalamus project. While the proposed plan is structurally sound, it is highly vulnerable to several classic methodological pitfalls in active learning and closed-loop control. You must address the following strategic and scientific constraints before executing this phase:

### 1. The Active Exploration Evaluation Paradox (Methodological Warning)
Evaluating an active probing agent (Arm G) using cumulative or raw online prediction loss is a dangerous methodological trap. 
*   **The Problem:** Because Arm G actively seeks out and forces high-surprise collision events, its *online* prediction error may appear significantly **higher** than the Passive baseline (which allows objects to glide smoothly without disruptive collisions). 
*   **The Directive:** You must not rely solely on online average loss. To ensure a scientifically rigorous and fair comparison, you must evaluate predictive accuracy and adaptation speed on a **standardized, passive evaluation sequence** (where motor actions are disabled) or specifically isolate a fixed window of steps immediately following a collision (e.g., $t+1$ to $t+5$). Specify this distinction clearly in your pre-registration.

### 2. Guarding Against Closed-Loop "Delusional" Feedback Loops
By coupling the non-parametric soft-argmax projection with the motor controller, you are closing a feedback loop: the motor actions are guided by the extracted coordinates, and the representation is updated based on the sensory consequences of those actions.
*   **The Problem:** This mutual dependency can easily trigger a runaway positive feedback loop where the agent becomes "delusional"—locking onto a static or oscillating point, ignoring the actual object boundaries, and restricting its spatial coverage to a small, self-consistent pocket of the state space.
*   **The Directive:** You must track **Spatial Coverage Entropy** (the Shannon entropy of the pointer and object coordinate distributions across the 1D space) as a diagnostic metric. If Arm G's spatial coverage drops significantly below that of Arm F-Random, or if the centroid decoding MSE drifts during active control, you must treat this as a representation collapse and trigger your falsification criteria.

### 3. Refinement of Pre-Registration & Quantitative Falsification
The proposed pre-registration plan is promising, but the falsification criteria must be sharpened to prevent post-hoc rationalizations:
*   **Criterion 1 (Predictive Recovery):** Instead of a vague "post-collision latent prediction MSE", define the exact evaluation protocol (e.g., "prediction error evaluated on a standardized test set of 100 passive collision steps").
*   **Criterion 3 (Adaptation Velocity):** Clarify how "return to pre-perturbation baseline" is calculated. We highly recommend measuring the **Area Under the Curve (AUC)** of the prediction error post-perturbation ($t \in [1501, 2000]$) to capture both the speed and stability of adaptation in a single robust metric.
*   **Pre-Registration Step:** Ensure that your finalized hypothesis, metrics, and quantitative falsification boundaries are written to `src/pre_registration.md` at the very beginning of your run. All subsequent sub-agents must read and strictly adhere to this file. Do not alter the falsification criteria post-hoc once the simulation has begun. Use restrained, objective language throughout.

---

