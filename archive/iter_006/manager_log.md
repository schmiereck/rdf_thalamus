# Research Manager Log - Iteration 006

## Iteration 006 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
In the Phase 4 Generalization and Noise Robustness evaluation:
1. Under clean conditions, introducing a 4th unseen object to a network trained on N=3 objects will trigger dynamic dimension recruitment in ThalamusNet within 500 timesteps, reducing few-shot prediction loss on the novel object by at least 30% relative to a fixed-dimension JEPA (B1) baseline.
2. Under dynamic background noise (i.i.d. pixel-level Gaussian noise, σ = 0.15), ThalamusNet's hierarchical latent representations and VICReg-based covariance regularization will project out high-frequency spatial-temporal noise, resulting in a prediction loss ratio (L_noise / L_clean) that is at least 2.0x lower than that of B1 (JEPA) and B2 (NGC) baselines.
3. The Thalamic Attention watch-dog is resilient to the Noisy-TV trap; the attention token will spend at least 70% of its active timesteps tracking the structured physical entities rather than the unpredictable background noise.

**Proposed Falsification Criterion:**
The hypothesis will be proven false if:
1. ThalamusNet fails to recruit a new dimension within 500 steps of the 4th object's introduction, or its prediction loss on the novel object is not at least 30% lower than B1.
2. The prediction loss ratio (L_noise / L_clean) for ThalamusNet under dynamic noise is less than 1.5x lower than the ratio of B1 or B2.
3. The attention token's spatial tracking overlap with the physical objects falls below 60% in the presence of dynamic noise, indicating attentional trapping by the unpredictable background.

**Proposed Method:**
1. Modify the 1D physics sandbox (in src/) to support:
   a. Static background noise (constant random pixel offset pattern).
   b. Dynamic background noise (i.i.d. Gaussian noise σ = 0.15 added to background pixels at each timestep).
   c. N=4 object initialization with a novel, unseen physical parameter range (size, mass, color, velocity).
2. Run a 5-seed comparison sweep evaluating ThalamusNet, B1 (JEPA), and B2 (NGC) under:
   - Clean test environments.
   - Static noise environments.
   - Dynamic noise (Noisy-TV) environments.
3. Perform the 4-object generalization test: train on N=3, inject the 4th object, and log prediction error curves, dimension recruitment timesteps, and post-recruitment specialization.
4. Perform the causal-sensitivity sweep on the 4th object by altering its mass/velocity post-collision and measuring prediction adaptation times.
5. Log the complete Phase 4 metrics battery: compute/memory profiles (FLOPs/timestep), plasticity-token traces, and the transition from externally primed to self-generated attention under noisy conditions.

---

## Iteration 006 -> Planner [Strategic Guidance]

### Manager's Note: Strategic Guidance for Phase 4 (Generalization & Noise Robustness)

While your integration of static/dynamic noise and multi-object generalization is highly aligned with the transition to Phase 4, the proposed plan contains severe logical inconsistencies with our established physical baselines and risks producing trivial constructional victories. You must revise your plan and pre-register your hypotheses under the following strict scientific guidelines:

---

### 1. The Overlap Reality Check (Metric Hygiene & Historical Alignment)
* **The Flaw:** Your proposed hypothesis (3) and falsification criterion (3) require the attention token to maintain a **60% to 70% spatial tracking overlap** under noise. However, in Phase 3 (Iter 5), we explicitly falsified this assumption: even under *clean* conditions, the active tracking overlap was established at **22.8%** due to the "active-perception entropy trade-off." Setting an absolute threshold of 60% under noise is mathematically impossible and ignores our own empirical record.
* **The Correction:** You must redefine your attention tracking metrics **relativistically**. Instead of an absolute 60% threshold, the falsification criterion must state that *the attention token's tracking overlap under noise does not degrade by more than a specified relative fraction of its clean baseline performance* (e.g., $Overlap_{noise} \ge 0.8 \times Overlap_{clean}$). 

---

### 2. The "Noisy-TV" vs. "Simple Gaussian Blur" Test (Construction vs. Empirical)
* **The Flaw:** Testing resilience purely against i.i.d. pixel-level Gaussian noise is a weak, constructional test. Any standard CNN encoder with spatial pooling or VICReg covariance constraints will naturally smooth out pixel-level i.i.d. noise as a mathematical consequence of spatial averaging. Overcoming this is not an emergent property of the Thalamic Attention Watchdog.
* **The Correction:** To genuinely test the Thalamic watchdog against the classic **Noisy-TV trap**, you must distinguish between:
  1. *Global high-frequency noise* (pixel-level Gaussian noise).
  2. *Localized, structured entropic noise* (e.g., introducing a "Noisy-TV entity" in the 1D physics sandbox—such as a 4th block that behaves as a random-walk generator or flickers with maximum entropy, having high local surprise but zero temporal predictability).
* Your pre-registration must clearly hypothesize how the attention mechanism avoids being permanently trapped by a localized, high-surprise, unmodelable entity (the true Noisy-TV), rather than just filtering out global pixel-level blur.

---

### 3. Generalization Rigour & Capacity Baselines
* **The Flaw:** Simply showing that ThalamusNet recruits a 4th dimension and beats a fixed-capacity $d=2$ JEPA (B1) on a 4-object task is a trivial consequence of capacity.
* **The Correction:** You must evaluate the generalization of the recruited dimension against both $B1$ (low capacity) and $B1\_large$ (static high capacity). You must prove that:
  * Dynamic recruitment yields better or comparable few-shot adaptation than a model that was statically over-parameterized from the start ($B1\_large$).
  * The recruited dimension specifically correlates with the novel object's state trajectory (demonstrating representation specialization) rather than merely drifting as a general error-sponge.

---

### Pre-Registration Mandate
Before running the simulations, ensure that your revised, mathematically rigorous hypotheses and relativistic falsification criteria are fully detailed and written to `src/pre_registration.md`. The execution sub-agents must read this file and adhere strictly to these exact definitions.

---

