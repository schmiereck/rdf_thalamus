CORRECTED ARCHITECTURE CEILING PROBE — iter_025 re-run addressing Research Manager criticisms.

You MUST first read src/pre_registration.md to understand the pre-registered hypotheses and criteria.

## Context

The original iter_025 experiment had five critical methodological flaws identified by the Research Manager:
1. 60% collapse rate across ALL arms including control (setup failure, not finding)
2. Invalid noise floor leading to unsupported threshold
3. 47-67% matching mismatch rate (pass/fail depends on matching choice)
4. Only 2 non-collapsed seeds per arm (below Section 9's ≥5 requirement)
5. iter_023's d_max=16 / 0.137 result never tested against SFA-off control

## YOUR TASK

Create src/run_phase0_id_probe_v2.py — a corrected experiment runner that addresses ALL five criticisms. The existing code is in src/run_phase0_id_probe.py and src/models_dual_stream.py.

### CRITICAL CHANGES FROM iter_025:

**1. Fix collapse (Criticism 2):**
- Change learning rate from 1e-3 to 3e-4 (more conservative, standard for VICReg)
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`
- Increase training steps from 5000 to 8000 (more time for convergence at lower LR)
- Keep all other hyperparameters the same (var_weight=25, cov_weight=25, sim_weight=25, batch_size=32, replay_buffer=2000)

**2. Defensible threshold (Criticism 1):**
- Pre-declare threshold = 0.10 as an EFFECT SIZE threshold (z_dyn explains ≥10% more color variance than z_coord)
- DO NOT use noise floor runs (they were invalid in iter_025 because R² on random encoders can exceed 1)
- The 0.10 threshold means: "the improvement in color predictability from z_dyn over z_coord is at least 10% of variance" — a meaningful effect size, not derived from a flawed noise floor

**3. Resolve matching ambiguity (Criticism 3):**
- Pre-declare Hungarian matching as the SOLE matching scheme for evaluation
- During TRAINING, also use Hungarian matching (not sorted) for the supervised/contrastive loss
- Report the sorted-matching result as a secondary check, but the PRIMARY verdict uses Hungarian only
- If sorted and Hungarian disagree on pass/fail for >25% of seeds, REFUSE to claim falsification and report "matching-dependent outcome"

**4. More seeds (Criticism 4):**
- Use 10 seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
- Goal: ≥5 non-collapsed seeds per arm (if collapse rate drops to ~20% with lower LR, 10 seeds gives ~8 non-collapsed)
- All 10 seeds must complete the full 8000 training steps

**5. Audit iter_023 d_max=16 claim (Criticism 5):**
- Add Arm E: JEPA+VICReg control with d_max=16, NO supervised/contrastive objective
- If Arm E achieves delta_R2_color ≥ 0.10, the d_max=16 improvement is confirmed as a capacity effect (since it occurs without identity objective)
- If Arm E does NOT achieve delta_R2_color ≥ 0.10, the iter_023 claim needs revision

### ARM CONFIGURATIONS (5 arms × 10 seeds × 8000 steps = 50 runs):

Arm A: JEPA+VICReg Control (d_max=8)
  - primary_objective="jepa", var_weight=25, cov_weight=25, sim_weight=25
  - d_max=8, d_t=3, dyn_readout="centroid_gated", pos_encoding="none"
  - CCR covariance mode (ccr_smooth_weight=10, ccr_spatial_weight=10)
  - gdasr_log_only=True
  - NO supervised/contrastive loss
  - Learning rate: 3e-4, gradient clipping: max_norm=1.0

Arm B: Supervised Color Probe + VICReg (d_max=8) [CRITICAL DIAGNOSTIC]
  - Same as Arm A PLUS supervised_color_loss with supervised_weight=25.0
  - Training matching: Hungarian
  - Ramp supervised weight 0.1 → 25.0 over first 500 steps
  - Learning rate: 3e-4, gradient clipping: max_norm=1.0

Arm C: ID-Contrastive + VICReg (d_max=8)
  - Same as Arm A PLUS id_contrastive_loss with contrastive_weight=25.0
  - Training matching: Hungarian
  - Learning rate: 3e-4, gradient clipping: max_norm=1.0

Arm D: Supervised Color Probe + VICReg (d_max=16)
  - Same as Arm B but d_max=16
  - Tests whether more channels help supervised encoding

Arm E: JEPA+VICReg Control (d_max=16) [NEW — capacity audit]
  - Same as Arm A but d_max=16
  - NO supervised/contrastive loss
  - If this achieves delta_R2_color ≥ 0.10, the d_max=16 improvement is confirmed as a capacity effect

### IMPLEMENTATION DETAILS:

1. Base the script on src/run_phase0_id_probe.py — copy it and modify
2. Change the training loop:
   - lr = 3e-4 instead of 1e-3
   - Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before optimizer.step()
   - 8000 steps instead of 5000
   - For Arms B, D: use matching_mode="hungarian" during training (not "sorted")
   - For Arm C: use matching_mode="hungarian" during training
3. Evaluation:
   - Primary: Hungarian matching
   - Secondary: also compute sorted matching (but verdict uses Hungarian only)
   - Report mismatch rate between Hungarian and sorted
4. Seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
5. Checkpoint evaluations at steps 2000, 4000, 6000, 8000
6. Same evaluation metrics as iter_025: semantic probes, collapse check, centroid MSE, tracking quality, temporal variance, within/between trajectory variance, shuffled-frame control
7. Add per-dimension std logging during training (every 500 steps) to track when collapse happens
8. Save results to archive/iter_025/results_v2/

### FALSIFICATION CRITERIA (pre-declared):

H1 (Architecture Capacity): Arm B achieves delta_R2_color ≥ 0.10 (mean over non-collapsed seeds, Hungarian matching) with collapse rate ≤ 2/10.

H2 (ID-Contrastive Viability): Arm C achieves delta_R2_color ≥ 0.10 (mean over non-collapsed seeds, Hungarian matching) with collapse rate ≤ 2/10.

Capacity Audit: If Arm E achieves delta_R2_color ≥ 0.10, the d_max=16 improvement is confirmed as capacity effect (not objective effect).

FOUR OUTCOME QUADRANTS (same as iter_025, but with Hungarian matching):
B succeeds, C succeeds → Architecture CAN encode identity; ID-contrastive viable
B succeeds, C fails → Architecture CAN encode; contrastive formulation insufficient
B fails, C succeeds → Check implementation (supervised should ≥ contrastive)
B fails, C fails → Result consistent with architecture-level bottleneck (CONDITIONAL: if collapse rate ≤ 2/10 in control Arm A. If Arm A also has >2/10 collapse, the experiment is still underpowered and no architecture claim is earned.)

Language requirements:
- Positive Arm B: "compatible with sufficient architectural capacity under direct supervision" — NOT "demonstrates architecture can encode identity"
- Positive Arm C: "supervised (slot IDs are privileged)" — NOT "decoder-free self-supervision solved"
- Negative result: "consistent with an architecture-level bottleneck on identity encoding, conditional on [mismatch rate]%" — NOT "architecture cannot encode identity"

### ADDITIONAL REQUIREMENTS:
- If Arm A collapse rate > 2/10, the experiment is UNDERPOWERED and NO architecture claim is valid — report this honestly
- If sorted and Hungarian matching disagree on pass/fail for >25% of non-collapsed seeds in Arm B or C, report "matching-dependent outcome; falsification claim not earned"
- Monitor per-dimension std during training to diagnose WHEN collapse happens

### IMPORTANT:
- Read src/models_dual_stream.py to understand the NonParametricJEPASpatial model
- The model's compute_supervised_color_loss and compute_id_contrastive_loss already support matching_mode="hungarian"
- The model's match_channels_hungarian method already exists
- Do NOT modify src/models_dual_stream.py — only create the new runner script

After creating the script, update src/pre_registration.md with the corrected plan (v2).

Then RUN the experiment (all 50 training runs). Save all results to archive/iter_025/results_v2/.

Finally, write a comprehensive analysis to archive/iter_025/results_v2/final_analysis.md including:
- Per-arm results (collapse rate, mean delta_R2_color, per-seed details)
- H1 and H2 falsification verdicts
- Capacity audit (Arm E vs iter_023)
- Matching dependency report (sorted vs Hungarian disagreement rate)
- Whether the experiment is adequately powered (Arm A collapse rate ≤ 2/10)
- Quadrant assignment and next-step recommendation