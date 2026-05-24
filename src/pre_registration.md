# RDF Scientific Pre-Registration

*   **Iteration:** 007
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Under the dynamic, unsupervised JEPA representation-learning regime (GDASR), active physical interaction (Active Probing via Subsumption Motorics) with a newly introduced 4th object during the N=3 -> N=4 generalization transition will naturally force the newly recruited 4th latent dimension to represent the object's spatial coordinates more strongly than passive observation. Specifically:
- A post-hoc linear readout probe trained on frozen latent representations of the Active Probing model to decode the 1D physical position of the 4th object will show a statistically significant correlation improvement (\Delta |r| >= 0.25) over the post-hoc linear readout probe trained on the Passive Observation model.
- The Active Probing model will achieve a Pearson correlation |r| >= 0.40 between the activity of its recruited 4th dimension and the physical position of the 4th object, compared to |r| < 0.15 for the Passive Observation model.
- This active-probing-driven spatial specialization will emerge without any backpropagation of coordinate/supervised loss gradients into the representation network (which remains 100% unsupervised via local temporal prediction and VICReg), and without causing representation collapse (cross-dimension correlation r_cross <= 0.30).

## 2. Falsification Criteria
- Falsification Criterion 1: The post-hoc linear readout of the 4th object's position from the Active Probing model's recruited dimension does NOT show a correlation improvement of at least \Delta |r| >= 0.25 over the Passive Observation model.
- Falsification Criterion 2: The absolute Pearson correlation |r| between the recruited 4th dimension's activity and the physical position of the 4th object across the 5 evaluation seeds for the Active Probing model is less than 0.40.
- Falsification Criterion 3: Active probing causes representation collapse (VICReg covariance/variance loss spikes, or cross-dimension correlation r_cross > 0.30).

## 3. Proposed Method
1. Enforce 100% unsupervised representation learning under VICReg loss constraint. No spatial coordinates are ever backpropagated into the JEPA representation networks.
2. Formulate 5 independent seeds: [42, 123, 456, 789, 999].
3. For each seed:
   - Phase 1: Train a `DynamicJEPA` model passively in an N=3 object environment from step 1 to step 1500. Ensure warmup of 1000 steps during which recruitment logic is not updated, cooldown parameter `cooldown=300`, `stabilization_period=100`, `k=4` are used. Only update recruitment logic when step > 1000. At step 1500, clone/checkpoint the model into two branches.
   - Phase 2 (Branch A - Passive Observation Control): Step 1501 to step 3000 in an N=4 environment with passive observation. Actions are passive: `action = {"acc": 0.0, "push": False}`.
   - Phase 2 (Branch B - Active Probing Experimental): Step 1501 to step 3000 in an N=4 environment with active probing. Actions are continuous continuous acceleration calculated via PD-controller targeting the 4th object's position (info['positions'][3]), and trigger a push when the pointer is within 5 pixels of the object, with a 15-step cooldown.
4. Evaluation: Generate a separate test environment of N=4 objects (seed + 5000) for 200 steps to collect the frozen 4th dimension activity $z_4$ and 4th object position $y_4$. Use analytical OLS linear regression to fit a 1D post-hoc linear probe mapping $z_4$ to $y_4$ on the first 100 test transitions, then evaluate on the remaining 100 transitions.
5. Report Pearson correlation coefficient $|r|$, prediction MSE, and check representation collapse metrics.
