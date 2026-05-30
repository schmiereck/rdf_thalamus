# RDF Scientific Pre-Registration

*   **Iteration:** 030
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
ARM 1 (Integration Smoke-Test): A representation with ΔR²_color ≈ 0.27 (SFA+VICReg
on separate backbone, 0% collapse, iter_029 Arm B) produces CLTS motor behavior
that is functionally adequate for Pillar D (thalamic gating) and Pillar E (motor
probing). Specifically, three pre-registered gates will be tested:
  G1 (Tracking Functionality): CLTS mean |pointer_pos - attended_centroid| < 20 pixels
    over a 2000-step closed-loop evaluation, demonstrating that the z_coord centroid
    readout supports functional PD tracking.
  G2 (Attention Validity): CLTS surprise-driven attention switches to the collision-
    involved channel within 15 steps after a detected collision in >50% of events,
    while CLTS-Random (uniform random attention selection) switches to the collision
    channel in <35% of events (chance baseline for 3 objects). A difference >15pp
    confirms the surprise signal is informative.
  G3 (Causal Sensitivity): After a forced-probe mass perturbation (mass×3 + push on
    object 0 at step 1000), CLTS attention switches to object 0's channel within
    20 steps in >60% of perturbation events, while CLTS-Random does so in <33%.
    A difference >15pp confirms the representation supports causal probing.
If ≥2 of 3 gates pass, the representation is sufficient for downstream tasks and the
project advances to Phase 2/3 integration without further objective optimization.
If 0-1 gates pass, the representation is insufficient and the objective hunt is
justified.

ARM 2 (M2 Decisive Test): Temporal identity contrastive binding (D1) or variance-
ramped SFA (D2) achieves mean ΔR²_color ≥ 0.30 on a 30-seed union bank with
variance-stability (std/mean < 2.0). D1 uses same-object-across-time as positive
pairs and other-objects-as-negatives in an NT-Xent loss; D2 anneals sfa_weight
from 0 to 5.0 over the first 4000 training steps. If D1 passes, M2 is revised to
"contrastive identity binding as primary" (slowness underspecifies identity). If D2
passes, M2 is revised to "variance-ramped SFA as primary." If neither passes, the
ΔR²_color proxy is acknowledged as unachievable by any tested decoder-free objective
and ARM 1's verdict governs the project direction.

## 2. Falsification Criterion
ARM 1 falsified if ≤1 of 3 gates pass, meaning the current representation does not
support meaningful downstream behavior and further objective optimization is justified.
Specifically:
  G1 falsified: CLTS mean tracking error ≥ 20 pixels (centroid readout is too noisy
    for functional tracking)
  G2 falsified: CLTS collision-switching rate ≤ CLTS-Random + 15pp (surprise signal
    is not informative for attention routing)
  G3 falsified: CLTS post-perturbation attention switch ≤ CLTS-Random + 15pp
    (representation does not support causal probing after mass change)

ARM 2 D1 falsified: mean ΔR²_color < 0.30 OR std/mean ≥ 2.0 on the 30-seed bank
ARM 2 D2 falsified: mean ΔR²_color < 0.30 OR std/mean ≥ 2.0 on the 30-seed bank
Both D1 and D2 falsified: the ΔR²_color ≥ 0.30 target is unachievable by any
decoder-free objective tested; the project accepts the current representation quality
per ARM 1's verdict.

## 3. Proposed Method
ARM 1 (Integration Smoke-Test — highest priority, runs first):
1. Load pre-trained SFA+VICReg checkpoints from iter_029 Arm B (10 seeds: 7, 17, 31,
   83, 97, 113, 127, 149 + hard seeds 53, 71 reported separately).
2. For each seed, run two 2000-step closed-loop evaluations:
   a. CLTS: CLTSMotorController with surprise-driven attention + PD tracking +
      push-when-surprised-and-near (existing probe-when-surprised logic)
   b. CLTS-Rand: Same controller but token_locus selected uniformly at random
      from d_t channels (replacing argmax of normalized surprise), keeping PD
      tracking and push logic identical — controls for reflexive PD tracking
3. Warm-up: first 200 steps with CLTS but no attention switching (fixed locus 0)
   to let EMA surprise statistics converge; metrics start at step 200.
4. Collision detection: at each step, compare object velocities before/after; flag
   collision if |delta_v| > 2.0 for any pair of adjacent objects. Record whether
   CLTS attention switches to a collision-involved channel within 15 steps.
5. Mass perturbation: at step 1000, change object-0 mass to 3× current; immediately
   force pointer near object 0 (set pointer_pos = object_0_pos ± 5) and push.
   Measure whether attention switches to object 0's channel within 20 steps.
6. Compute gate metrics and paired comparison (CLTS vs CLTS-Rand per seed).
7. Decision: ≥2/3 gates pass → representation sufficient → project advances.

Files: src/run_phase0_integration.py (NEW)

ARM 2 (M2 Decisive Test — contingent on ARM 1):
1. Expand union seed bank to 30 seeds (original 10 + fresh 10 + new: 173, 179,
   181, 191, 193, 197, 199, 211, 223, 227).
2. D1 (Temporal Identity Contrastive):
   - New loss: NT-Xent where same-object-at-consecutive-timesteps = positive pair,
     different-objects = negatives.
   - Implementation: encode x_target and x_hist[:,-1] to get z_coord_t, z_dyn_t,
     z_coord_{t-1}, z_dyn_{t-1}. Sort z_coord and positions to match channels to
     objects (sorted matching, O(d_t log d_t) on GPU). For each object, z_dyn at
     the matched channel at time t and t-1 form a positive pair; z_dyn from other
     objects at t-1 are negatives. Uses existing id_contrastive_proj projection head.
   - Loss computed in training loop (not in model forward), added to total loss with
     weight temporal_contrastive_weight=5.0.
   - Architecture: NonParametricJEPASpatialSeparateDyn, primary_objective="sfa",
     mask_dyn_sim=True, coord_vicreg=True, sfa_weight=0 (SFA disabled; replaced
     by temporal contrastive). d_t=3, d_max=8.
3. D2 (Variance-Ramped SFA):
   - Same architecture as D1 but primary_objective="sfa" with sfa_weight
     linearly ramped from 0 to 5.0 over steps 0-4000, then constant at 5.0.
   - Ramping prevents early-training SFA instability (gradient conflict with
     VICReg variance term during representation bootstrapping).
4. Train 2 arms × 30 seeds for 8000 steps each (separate backbone, mask_dyn_sim=True,
   coord_vicreg=True, d_t=3, d_max=8, buffer=4000, batch_size=32, lr=3e-4).
5. Evaluate: ΔR²_color, collapse rate, centroid MSE (same protocol as iter_029).
6. Gate: mean ΔR²_color ≥ 0.30 AND variance-stability std/mean < 2.0.
7. Decision: D1 passes → M2 revised to contrastive identity binding; D2 passes →
   M2 revised to variance-ramped SFA; neither passes → proxy acknowledged as
   unachievable, ARM 1's verdict governs.

Files: src/run_phase0_m2_decisive.py (NEW)

PRESERVE across both arms:
- Separate backbone + collapse-avoiding config (mask_dyn_sim=True, coord_vicreg=True)
- d_t=3 frozen, GDASR log-only (M3)
- Pooled batch VICReg (M1)
- No positional encoding (cross-objective regularity from iter_013 and iter_021)
- Hard seeds 53/71 reported separately, not averaged away
- Report σ alongside mean for all metrics (≥5 seeds minimum per condition)

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
