# RDF Milestone Review — Iteration 012 — Closed-Loop Thalamic Subsumption (CLTS) Motorics

## 1. Pre-Declared Hypothesis and Falsification Criterion
- **Pre-registered Hypothesis:** Coupling the non-parametric soft-argmax projection stream (Arm F) with the 3-layer Closed-Loop Thalamic Subsumption (CLTS) motor policy (Arm G) will enable high-efficiency curiosity-driven physical exploration, reducing post-collision prediction error while maintaining a centroid decoding Mean Squared Error (MSE) below the threshold of 85.0.
- **Falsification Criterion:** The hypothesis is formally falsified if:
  1. The average centroid decoding MSE across the 5 evaluation seeds exceeds 85.0 under the active CLTS motor policy.
  2. CLTS fails to demonstrate a statistically significant reduction in post-collision prediction error compared to passive observation and random babbling baselines.

## 2. Experimental Protocol
- **Environment:** 1D Physics Sandbox, 128 RGB pixels, containing 3 distinct moving objects of varying sizes, colors, and masses. Generalization phases introduced a 4th novel object.
- **Step Count:** 2000 steps per seed.
- **Random Seeds:** Sweep executed across 5 deterministic seeds (seeds 42, 43, 44, 45, 46).
- **Baselines & Controls:** 
  - *Control A:* Passive Observation (zero motor input).
  - *Control B:* Random Motor Babbling (random acceleration and push actions).
  - *Experimental (Arm G):* CLTS Motor Policy (3-layer subsumption architecture mapping attention-token spatial coordinates to pointer acceleration and push commands).
- **Measurements:** Post-collision temporal prediction error (L2 loss), spatial coverage entropy (grid-cell occupancy), and centroid decoding MSE (against ground-truth physical coordinates of the target object).

## 3. Observed Quantities
- **Centroid Decoding MSE:** 
  - Passive Control: 75.36 (with non-parametric projection).
  - CLTS Active Policy: 85.85 (averaged over 5 seeds).
- **Post-Collision Prediction Error:** 
  - Passive Control: 0.0948 L2 loss.
  - Random Control: 0.0557 L2 loss.
  - CLTS Active Policy: 0.0236 L2 loss (a 75.1% reduction vs. passive, and a 57.6% reduction vs. random).
- **Spatial Coverage Entropy:**
  - CLTS showed a measured increase in spatial coverage (pointer-to-object distance tracked closely around target boundaries, with exploration spread across the entire 128-pixel space).

## 4. Verdict
- **Verdict:** **REFUTED** on representational stability; **CONSISTENT** on predictive and exploratory efficiency.
- **Justification:** The active physical interaction of CLTS met all operational goals for active learning, outperforming random controls by 57.6% in post-collision error reduction and showing superior spatial exploration. However, it formally triggered the pre-declared falsification criterion because the average centroid decoding MSE rose to 85.85, exceeding the strict 85.0 limit. This indicates that active physical contact introduces an unmodeled representation drift.

## 5. Construction-vs-Empirical Note
The spatial coordinates in this architecture are extracted via a non-parametric soft-argmax projection over the latent feature maps. Because there are no parametric heads explicitly trained on ground-truth coordinates, the localization is purely empirical—emerging from the spatial consistency of the temporal prediction dynamics. The observed drift under active control is a genuinely empirical phenomenon: it demonstrates that changing the environmental state transition matrix through physical manipulation feedback-loops directly alters the internal representations of the system.

## 6. Limitations
- **No Active Calibration:** The system lacks an active calibration loop to correct for representation drift during physical contact. Once a collision perturbates the visual backbone, the coordinate tracking error accumulates.
- **1D Space Constraint:** This evaluation was limited to a 1D physics sandbox. The drift penalty is expected to compound in multi-dimensional space (2D/3D), where physical interaction can cause rotational or depth-based occlusion.
- **Unbounded Transient Perturbations:** The study does not isolate the exact frames during which the drift occurs (e.g., whether it is a continuous decay or a step-function jump during the exact frame of elastic collision).