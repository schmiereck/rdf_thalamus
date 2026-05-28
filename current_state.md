# Current Research State
Phase: Phase 22 Complete — Architecture Ceiling Falsified, SFA Non-Functional

## Goal
Design and evaluate a neural architecture achieving hierarchical abstraction
without generative decoders, with SFA+VICReg shaping z_dyn and soft-argmax
tracking position (z_coord). Current focus: making SFA actually function as
the primary representation objective.

## Confirmed
- **ARCHITECTURE CEILING FALSIFIED (iter_022, 21.3)**: Single-scalar z_dyn
  bottleneck is NOT the primary cause of disentanglement failure. Expanding
  per-channel capacity to K=4 (12 scalars for 12 identity DOF) collapsed
  100% and performed WORSE on identity encoding (delta_R2_identity = -0.055
  vs Ctrl -0.035, a -0.021 regression).
- **SFA NON-FUNCTIONAL (iter_022, all arms)**: Normalized temporal variance
  confirms z_dyn changes 300-1000× more (relative to magnitude) than z_coord
  across ALL arms. SFA is NOT making z_dyn slow. Root cause: sfa_weight=0.1
  vs var_weight=25.0, a 250× gradient imbalance.
- **K=4 VICReg COLLAPSE (iter_022, Arm C)**: 12 active features at
  batch_size=32 overwhelms VICReg variance enforcement. Per-dim std
  consistently 0.2-0.5 (below 0.5 threshold). Per-sub-feature probes
  uninformative due to collapse.
- **CONV4 SOURCE UNSTABLE (iter_022, Arm A)**: dyn_source="conv4" collapsed
  3/5 seeds. Linear(128→1) projection produces highly correlated channels.
- **MORE CHANNELS HELP COLOR (iter_022, Arm B)**: d_max=16 achieved
  delta_R2_color = +0.130 (+0.080 over Ctrl) with stable training (1/5
  collapse). But delta_R2_identity remained negative (-0.027).
- **CGIR REPLICATED (iter_022, Ctrl)**: delta_R2_color = 0.050, collapse 1/5,
  consistent with iter_021 Arm A results.
- **VICReg BATCH-LEVEL WORKS (iter_020-022)**: For K=1 configurations with
  reasonable collapse rates, confirming M1.
- **JEPA BASELINE HEAVILY COLLAPSED (iter_020)**: JEPA+CCR 4/5 collapse,
  confirming M2's demotion.

## Refuted / Falsified
- **SINGLE-SCALAR BOTTLENECK (iter_022 C4)**: FALSIFIED. K=4 sub-features
  did not improve identity encoding; collapsed 100%.
- **CGIR AS PRIMARY CAUSE (iter_021 C3)**: FALSIFIED. CGIR produces +0.124
  shift but insufficient for threshold.
- **IDENTITY-VS-POSITION SEPARATION (iter_021-022)**: No arm achieves
  delta_R2_identity ≥ 0.10. The "identity" label for z_dyn is misleading.
- **SFA EFFECTIVENESS (iter_022 C5)**: FALSIFIED across all arms.

## Best Result
- Arm B (d_max=16, CGIR+SFA+CCR): delta_R2_color = +0.130, delta_R2_identity
  = -0.027, 1/5 collapsed. Best color disentanglement to date, but compound
  identity still fails.

## In Progress
- None active. Next: SFA effectiveness sweep (sfa_weight 0.1 → 25.0).

## Open Questions
1. Can SFA be made effective by increasing sfa_weight (0.1→1.0→5.0→25.0)?
   This is the most impactful lever — if SFA doesn't shape z_dyn, no
   architectural change can help.
2. Is SFA fundamentally flawed for this CNN? Higher weight may conflict with
   VICReg's variance requirement, producing noisy-but-uninformative z_dyn.
3. Would per-channel VICReg fix K=4 collapse? Enables valid sub-feature test.
4. Does d_max=24/32 further improve color disentanglement?
5. Can compound identity (color+size) ever be encoded in z_dyn given z_coord's
   spatial-identity correlation?
6. Does increasing training steps beyond 5000 help?
7. Is JEPA readout (sim_weight=25) interfering with SFA through encoder gradients?
