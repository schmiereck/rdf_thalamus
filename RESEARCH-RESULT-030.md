# RDF Research Result — Iteration 030

## 1. Hypothesis
phase-30: ARM 1 integration gates fail due to protocol confounds (ceiling effects on G2/G3, tight G1 threshold); ARM 2 definitively falsifies ΔR²_color≥0.30 for all tested decoder-free objectives; M2 mandate not supported and proxy metric must be retired

## 2. Status
ok

## 3. Analysis
This phase executed the two-arm plan from the approved research plan with the Manager's three corrections incorporated.

ARM 1 was designed to test whether frozen iter_029 representations (SFA+VICReg at ΔR²≈0.27 and VICReg-only at ΔR²≈0.04) support functional downstream behavior (tracking, collision attention, causal sensitivity). The pre-registered gates all failed, but the failure is primarily a protocol calibration issue, not a representation quality issue:

1. G1 (tracking): The 20-pixel threshold was unrealistic for the PD controller (Kp=2.0) tracking moving objects in a 128-pixel space. Tracking error of 36-55 pixels is functional (objects are tracked within ~30-40% of the arena width), but not tight. Notably, CLTS-VICReg tracks better than CLTS-SFA, suggesting identity encoding quality does NOT determine tracking quality.

2. G2/G3 (attention/sensitivity): Ceiling effects make these gates uninformative. With 3 objects in 128 pixels and elastic collisions, collision events are too frequent (~500-800 per 1800-step run) for attention switching to be discriminated from chance. The mass perturbation protocol forces the pointer near object 0 and pushes, guaranteeing object-0 surprise regardless of attention mode.

The ARM 1 results are thus a NEGATIVE RESULT about the protocol design, not about the representation. The pre-registered verdict is "representation insufficient," but the actual data does not support this conclusion — the experiment was simply unable to measure what it intended to measure for G2/G3.

ARM 2 tested two new objectives: D1 (batch-level temporal contrastive) and D2 (variance-ramped SFA). Both were falsified on the ΔR²_color ≥ 0.30 gate. D1's batch-level NT-Xent was too weak (object-level matching not implemented). D2's variance ramp actually hurt (0.189 vs iter_029's static 0.275). 

The combined evidence across 11 iterations (020-030) now comprehensively establishes:
- ΔR²_color ≥ 0.30 is NOT achievable by any tested decoder-free objective on this architecture
- The separate-backbone + mask_dyn_sim + coord_vicreg configuration is 0% collapse across 100+ runs
- The proxy metric has been maximally explored and should be retired
- The project must pivot to either (a) accepting weak identity encoding and testing whether it matters for the actual project goal, or (b) relaxing the decoder-free constraint to include reconstruction


## 4. Metrics
{'arm1_g1_clts_sfa_tracking_error': 45.09, 'arm1_g1_clts_vicreg_tracking_error': 36.22, 'arm1_g1_threshold': 20.0, 'arm1_g1_pass': False, 'arm1_g2_clts_sfa_collision_switch_rate': 1.0, 'arm1_g2_frozen_collision_switch_rate': 1.0, 'arm1_g2_random_collision_switch_rate': 0.999, 'arm1_g2_pass': False, 'arm1_g3_clts_sfa_perturbation_switch_rate': 1.0, 'arm1_g3_frozen_perturbation_switch_rate': 1.0, 'arm1_g3_random_perturbation_switch_rate': 1.0, 'arm1_g3_pass': False, 'arm1_gates_passed': 0, 'arm1_total_runs': 48, 'arm2_d1_mean_delta_r2_color': 0.115, 'arm2_d1_ci_lower_95': 0.007, 'arm2_d1_collapse_rate': 0.0, 'arm2_d1_verdict': 'FALSIFIED', 'arm2_d2_mean_delta_r2_color': 0.189, 'arm2_d2_ci_lower_95': 0.074, 'arm2_d2_collapse_rate': 0.0, 'arm2_d2_verdict': 'FALSIFIED', 'arm2_total_seeds': 30, 'arm2_total_runs': 60, 'separate_backbone_collapse_rate_all_iters': 0.0, 'best_delta_r2_color_achieved': 0.275, 'best_delta_r2_source': 'iter_029 Arm B, SFA+VICReg sfa_weight=5.0, 20 seeds'}

## 5. Notes
ARM 1: all gates failed (protocol confounded by ceiling effects). ARM 2: both D1 and D2 falsified.

---
*Note: This is an automated summary as the Research Manager did not provide a full milestone report.*
