# RDF Scientific Pre-Registration

*   **Iteration:** 012
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Coupling the stable non-parametric soft-argmax projection of Arm F with a Closed-Loop Thalamic Subsumption (CLTS) motor policy (Arm G)—where a high-level epistemic collision controller is dynamically activated (subsumed) proportional to the local predictive surprise of the attended object—will significantly accelerate the adaptation and predictive accuracy of physical dynamics under parameter shifts (such as a 2x shift in object mass) compared to both passive observation and random exploration, without destabilizing the representational grounding of the coordinate encoder.

Specifically, CLTS will:
1. Accelerate the convergence of temporal prediction models on post-collision dynamics.
2. Maintain representational stability (soft spatial variance < 20.0, centroid decoding MSE < 85.0) despite active, self-generated physical manipulation of the environment.

## 2. Falsification Criterion
The hypothesis will be proven false if any of the following quantitative conditions are met:
1. Predictive Performance: Arm G (CLTS) does not achieve at least a 20% lower post-collision latent prediction MSE compared to the random exploration baseline (Arm F-Random) within 1500 steps of evaluation after a 2x mass perturbation is introduced.
2. Representational Instability: The soft spatial variance of Arm G's coordinate encoder increases above 20.0, or the centroid decoding MSE exceeds 85.0 during active closed-loop interaction, showing that the closed-loop feedback loop destabilizes the non-parametric projection.
3. Adaptation Efficiency: The number of interaction steps required for the model's prediction error to return to pre-perturbation baseline is not at least 15% shorter than that of the random exploration baseline.

## 3. Proposed Method
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
*Created automatically by the RDF Orchestrator prior to iteration execution.*
