# Research Manager Log - Iteration 005

## Iteration 005 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
1. Replacing the rigid 200-step attention cooldown with an adaptive, surprise-modulated cooldown ($C_t \in [10, 30]$ steps) combined with a Proportional-Derivative (PD) reflexive motor tracking controller in $M_{active}$ will resolve the physical tracking lag from Phase 2, increasing the physical pointer-to-object tracking overlap $\mathcal{O}_{track}$ from $11.20\%$ to $\ge 70.0\%$ in the test environment.
2. Progressive training of the Subsumption Motorics hierarchy (Lower-layer tracking -> Upper-layer perturbation) will allow the agent to learn the physical dynamics of hidden mass via intentional collisions. This causal sensitivity advantage will manifest such that, when evaluated on post-collision velocity predictions with altered hidden masses, $M_{active}$ will achieve an L2 prediction loss $\mathcal{L}_{collision}$ that is at least $35\%$ lower than both the passive control ($M_{no\_motor}$) and the random control ($M_{random}$).
3. Transitioning from externally primed attention queries (guided by color/size) to self-generated attention queries (output-as-input loop) will remain stable, with the test prediction loss on the attended locus increasing by no more than $15\%$ relative to the primed attention baseline.

**Proposed Falsification Criterion:**
The hypothesis will be proven false if any of the following occur:
1. The physical tracking overlap $\mathcal{O}_{track}$ of $M_{active}$ across the 5-seed test suite is $< 70.0\%$.
2. The post-collision L2 prediction loss ratio $\frac{\mathcal{L}_{collision}(M_{active})}{\mathcal{L}_{control}} \ge 0.65$ for either control $M_{control} \in \{M_{random}, M_{no\_motor}\}$ (failing to demonstrate a $35\%$ reduction in prediction error).
3. The prediction loss on the attended locus under self-generated attention is $> 1.15$ times the loss under externally primed attention.
4. Active closed-loop motor coupling causes drift collapse or numeric instability, resulting in an overall test L2 prediction loss higher than the baseline B1 model (0.0452).

**Proposed Method:**
1. **Adaptive Cooldown & Attention (`src/thalamus.py`)**: Replace the rigid 200-step cooldown with a dynamic, surprise-modulated cooldown $C_t = \text{clip}(\frac{\alpha}{|\Delta E_t| + \epsilon}, 10, 30)$ where $\Delta E_t = E_t - E_{t-1}$ is the temporal change in local surprise.
2. **Subsumption Motor Controller (`src/motor.py`)**: Create/modify the motor module to implement three hierarchical layers:
   - *Lower Layer (Reflexive Tracking)*: A PD controller that maps the attended spatial centroid of the attention token to continuous pointer acceleration to match the object's position.
   - *Middle Layer (Predictive Kinematics)*: Uses the temporal latent predictions to preemptively adjust the pointer's velocity and anticipate the object's path.
   - *Upper Layer (Deliberate Perturbation)*: Detects when tracking error is low and surprise is stabilized ("boredom"), and overrides lower layers to trigger a "push" command to collide with the object.
3. **Environment Alignment (`src/environment.py`)**: Verify that the 1D physics sandbox properly integrates continuous pointer actions (acceleration, push commands) and exposes correct bounding box/centroid targets.
4. **Progressive Coupling & Sweep (`src/train_eval.py`)**:
   - Implement the staged training schedule: Steps 0-1000: Decoupled random motor actions; Steps 1000-3000: Lower-layer visual tracking active; Steps 3000-5000: Full subsumption motorics (predictive tracking + deliberate push perturbations) active.
   - Execute a systematic 5-seed sweep across: $M_{no\_motor}$ (passive), $M_{random}$ (random exploration), and $M_{active}$ (proposed model).
   - Evaluate causal sensitivity by randomizing object masses in the test environments and measuring post-collision velocity prediction error. Log attention token traces to measure the self-generated vs. primed attention loss.

---

## Iteration 005 -> Planner [Strategic Guidance]

### Manager's Note: Strategic Guidance & Scientific Audit

Your transition into Phase 3 (Motor & Closed Loop) is highly promising, particularly your strategy to resolve the tracking lag via an adaptive, surprise-modulated cooldown. However, before proceeding to execution, we must apply strict scientific skepticism to your proposed plan to ensure your findings are truly empirical rather than constructional.

Address the following three directives in your final pre-registration and experimental setup:

---

### 1. The Tracking Control Trap: Demanding an Ablation Baseline
**The Construction-vs-Empirical Test:** 
If your lower-layer tracking is a Proportional-Derivative (PD) controller mapping the spatial centroid of the attention token directly to motor acceleration, a high tracking overlap ($\mathcal{O}_{track} \ge 70\%$) might simply verify that your classical control loop works, rather than proving that the *Thalamus representation network* has learned meaningful latent dynamics.
*   **Mandate:** You must prove that tracking is causally dependent on the *quality of the learned representations*, not just a trivial spatial-argmax on raw inputs. 
*   **Action:** Alongside your main sweep, run a **Representation Ablation Control**. Evaluate the tracking performance of $M_{active}$ when its attention token spatial mapping is computed from (a) a randomly initialized (untrained) Thalamus network, and (b) a network where latent space activations are spatially shuffled. If these ablated models also achieve high tracking overlap, your metric is constructional, and your hypothesis is refuted.

### 2. Standardization of Causal Sensitivity Evaluation
**Avoiding Evaluation Bias:**
Evaluating $M_{active}$ (which actively collides) against $M_{no\_motor}$ (which observes passively) on their respective self-generated trajectories introduces severe evaluation bias. $M_{active}$ will experience a vastly different distribution of states (frequent collisions) than $M_{no\_motor}$ (rare, natural collisions).
*   **Mandate:** Predictive loss post-collision ($\mathcal{L}_{collision}$) must be evaluated on a **strictly identical, standardized test set of collision events** (with varying hidden masses and initial velocities) that is completely independent of the training trajectory.
*   **Action:** Design a static, deterministic test benchmark containing 100 standardized collision episodes. Run $M_{active}$, $M_{random}$, and $M_{no\_motor}$ through this identical test set *without weight updates* to compare their predictive performance under equal physical conditions.

### 3. Rigorous Definition of "Attended Spatial Centroid"
Your proposed method relies on extracting the "attended spatial centroid of the attention token" to feed the PD controller.
*   **Mandate:** Explicitly define *how* this centroid is computed in `src/pre_registration.md`. It must be derived solely from the local receptive field or the latent attention weights of the gated layer. Under no circumstances may the controller access ground-truth physics engine coordinates (e.g., direct object positions or bounding boxes) during inference. This boundary must be mathematically clean and documented.

---

### Pre-Registration Notice
Your finalized hypotheses, mathematical formulations of the adaptive cooldown $C_t$, and the above falsification criteria will be automatically committed to `src/pre_registration.md` by the Orchestrator. Ensure your sub-agents read this file before commencing any training and strictly report deviations as failures. Proceed with updating the plan and executing Phase 3 under these constraints.

---

