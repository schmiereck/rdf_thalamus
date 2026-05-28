# RDF Scientific Pre-Registration

*   **Iteration:** 020
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
The spatial-mean z_dyn computation (z_dyn = a_spatial.mean(dim=-1)) in
NonParametricEncoder is the primary structural cause of the semantic
disentanglement failure (delta_R2_color = -0.074, target ≥ 0.10) observed
across all arms in iter_020. Replacing it with a Centroid-Gated Identity
Readout (CGIR) — a separate 1x1 convolution (conv_identity) producing an
identity feature map, pooled at the soft-argmax-attended spatial positions
with stop-gradient on the spatial attention — will enable SFA+VICReg to
produce z_dyn representations that encode object identity (color) while
z_coord encodes object position. Specifically:

(1) CGIR+SFA will train without collapse (per-dim std ≥ 0.5 in ≥ 4/5 seeds).
(2) CGIR+SFA centroid MSE will be within 10% of the mean-pooling SFA baseline
    (MSE_CGIR ≤ 1.10 × MSE_mean, where MSE_mean ≈ 121.9 from iter_020 Arm A1).
(3) CGIR+SFA will achieve semantic disentanglement: delta_R2_color ≥ 0.10,
    where delta_R2_color = R²_dyn_color - R²_coord_color. This is the metric
    that persistently failed in iter_020 (all arms had negative delta_R2_color).

The CGIR mechanism works because: (a) the soft-argmax attention naturally
localizes each channel d to a specific object, (b) pooling identity features
at the attended position reads out the object's local appearance (color),
(c) SFA on z_dyn then correctly encourages this readout to be slow (color
IS slow across frames), and (d) VICReg prevents the trivial constant solution.
The stop-gradient on the spatial attention prevents SFA from distorting
position tracking. This is structurally different from the mean-pooling
z_dyn, which averages over ALL spatial positions and cannot isolate per-object
identity information.

## 2. Falsification Criterion
The hypothesis is falsified if ANY of the following hold across the 5-seed
sweep for the CGIR+SFA arm:

C1 (Collapse): Per-dimension std < 0.5 in ≥ 2 out of 5 seeds (i.e., the
    CGIR architecture causes representation collapse no better than the
    mean-pooling baseline).

C2 (Centroid MSE degradation): Mean centroid-decoding MSE of the CGIR+SFA
    arm exceeds 1.10 × mean MSE of the mean-pooling SFA baseline (i.e.,
    CGIR degrades spatial readout by more than 10% relative to iter_020
    Arm A1 result of MSE ≈ 121.9).

C3 (Semantic disentanglement failure): delta_R2_color < 0.10 for the
    CGIR+SFA arm, where delta_R2_color = R²_dyn_color - R²_coord_color.
    This is the same criterion that failed in iter_020. If CGIR does not
    fix it, the root cause is NOT the spatial-mean computation and the
    hypothesis is falsified. A delta_R2_color ≥ 0.10 would mean z_dyn
    predicts object color meaningfully better than z_coord, confirming
    identity-position separation.

## 3. Proposed Method
Step 1: Modify NonParametricEncoder (src/models_dual_stream.py).
- Add conv_identity = nn.Conv1d(128, d_max, kernel_size=1) as a separate
  identity head alongside the existing conv_spatial.
- Add a dyn_readout parameter ("mean" or "centroid_gated") to control
  z_dyn computation (default "mean" for backward compatibility).
- In "centroid_gated" mode:
  a) Compute backbone features via existing conv1-4 → (B, 128, 8).
  b) Compute position spatial map: conv_spatial(features) → (B, d_max, 8)
     → interpolate → (B, d_max, 128) → soft-argmax → z_coord.
  c) Compute identity spatial map: conv_identity(features) → (B, d_max, 8)
     → interpolate → (B, d_max, 128) → a_identity.
  d) Compute z_dyn via centroid-gated readout:
     p_c = softmax(a_spatial, dim=-1)  # spatial attention, shape (B, d_max, 128)
     z_dyn = sum(a_identity * p_c.detach(), dim=-1)  # (B, d_max)
     The p_c.detach() prevents SFA gradient from distorting position tracking.
- Keep forward_spatial() unchanged for backward compatibility.
- Parameter increase: conv_identity adds 128×8 + 8 = 1,032 params (< 0.5% of total).

Step 2: Update NonParametricJEPASpatial.forward() SFA mode.
- The existing SFA mode computes sfa_loss on z_dyn_target vs z_dyn_prev.
  With CGIR, z_dyn is now computed differently but has the same shape (B, d_max).
  No changes to the loss computation are needed — only the z_dyn values change.
- The VICReg var/cov losses on z_dyn remain unchanged.
- The predictor (surprise readout) remains unchanged (takes detached z_dyn and z_coord).
- The CCR losses on z_coord remain unchanged.

Step 3: Create src/run_phase0_sfa_cgir.py.
Three arms × 5 seeds:

Arm A (CGIR+SFA w=0.1): Centroid-gated identity readout, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="none", CCR=none,
  d_t=3, gdasr_log_only=True. This is the primary test of the hypothesis.

Arm B (Mean+SFA w=0.1+CCR): Original mean-pooling z_dyn, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="none", CCR=covariance
  (ccr_smooth=10, ccr_spatial=10), d_t=3, gdasr_log_only=True.
  Direct replication of iter_020 Arm A1 for comparison — confirms that
  the only difference is the CGIR architecture change.

Arm C (CGIR+SFA w=0.1+pos): Centroid-gated identity readout, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="sinusoidal", CCR=none,
  d_t=3, gdasr_log_only=True. Tests whether positional encoding further
  improves CGIR (it helped collapse in iter_020 Arm C).

Seeds: [42, 123, 456, 789, 999].
Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000.
Evaluation: same protocol as iter_020 (collapse check, centroid MSE via
  linear probe, slowness metrics, VICReg health, semantic disentanglement
  probes with delta_R2_color, GDASR growth-point logging).
Checkpoint evaluation at step 2500 and final evaluation at step 5000.

Step 4: Falsification audit.
Compare CGIR+SFA arm vs Mean+SFA baseline on all three criteria.
The primary claim is that CGIR enables delta_R2_color ≥ 0.10 (C3).
If CGIR achieves C3 but degrades C1 or C2, report the trade-off honestly.
If CGIR fails C3, the root cause is NOT the spatial-mean computation and
the hypothesis is falsified — future work would need to explore other
architectural changes (slot attention, explicit color bottlenecks, etc.).

Files to create/modify:
- MODIFY: src/models_dual_stream.py (add CGIR to NonParametricEncoder,
  add dyn_readout parameter, implement centroid-gated readout)
- CREATE: src/run_phase0_sfa_cgir.py (experiment runner, 3 arms × 5 seeds)
- UPDATE: src/pre_registration.md (with Phase 0 CGIR pre-registration)

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
