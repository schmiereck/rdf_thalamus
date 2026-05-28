# RDF Scientific Pre-Registration

*   **Iteration:** 021
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis

The spatial-mean z_dyn computation (z_dyn = a_spatial.mean(dim=-1)) in
NonParametricEncoder is the primary structural cause of the semantic
disentanglement failure (delta_R2_color = -0.074, target ≥ 0.10) observed
across all arms in iter_020. This was a measured null result: all arms had
negative delta_R2_color, meaning z_coord predicted color at least as well as
z_dyn. This iteration tests whether a specific architectural change (CGIR)
changes that outcome — the prior outcome was not "wrong" but motivates this
architectural intervention.

Replacing the spatial-mean with a Centroid-Gated Identity Readout (CGIR) — a
separate 1x1 convolution (conv_identity) producing an identity feature map,
pooled at the soft-argmax-attended spatial positions with stop-gradient on the
spatial attention — will structurally route per-object appearance information
into z_dyn. This is a structural routing change, NOT an emergent disentanglement
claim: the architecture wires z_dyn to read from object positions (via
soft-argmax attention) rather than averaging over all positions.

Specifically:

(1) CGIR+SFA+CCR will train without collapse (per-dim std ≥ 0.5 in ≥ 4/5 seeds).
(2) CGIR+SFA+CCR centroid MSE will be within 10% of the mean-pooling SFA baseline
    (MSE_CGIR ≤ 1.10 × MSE_mean, where MSE_mean ≈ 121.9 from iter_020 Arm A1).
(3) CGIR+SFA+CCR will achieve semantic disentanglement: delta_R2_color ≥ 0.10,
    where delta_R2_color = R²_dyn_color - R²_coord_color.
(4) [Optional but recommended] CGIR+SFA+CCR will predict object identity
    (color+size compound) better than position: delta_R2_identity ≥ 0.10,
    where delta_R2_identity = R²_dyn_identity - R²_coord_identity. If C3 passes
    but C4 fails, the "identity" label is misleading and should be "color readout."

The CGIR mechanism works because: (a) the soft-argmax attention naturally
localizes each channel d to a specific object, (b) pooling identity features
at the attended position reads out the object's local appearance, (c) SFA on
z_dyn then correctly encourages this readout to be slow (appearance IS slow
across frames), and (d) VICReg prevents the trivial constant solution.
The stop-gradient on the spatial attention prevents SFA from distorting
position tracking.

## 2. Falsification Criteria

The hypothesis is falsified if ANY of the following hold across the 5-seed
sweep for the primary CGIR+SFA+CCR arm (Arm A):

C1 (Collapse): Per-dimension std < 0.5 in ≥ 2 out of 5 seeds.

C2 (Centroid MSE degradation): Mean centroid-decoding MSE of Arm A exceeds
    1.10 × mean MSE of the mean-pooling SFA baseline (Arm B, MSE ≈ 121.9
    from iter_020). If CCR is needed for z_coord tracking, Arm D (CGIR+SFA
    no CCR) may fail C2 while Arm A passes — this would confirm CCR's role.

C3 (Semantic disentanglement failure): delta_R2_color < 0.10 for Arm A.
    This is the same criterion that failed in iter_020. If CGIR does not fix
    it, the root cause is NOT the spatial-mean computation and the hypothesis
    is falsified.

C4 (Identity vs color-only): delta_R2_identity < 0.10 for Arm A, where
    delta_R2_identity = R²_dyn_identity - R²_coord_identity and "identity"
    is a compound label encoding both color and size. This is advisory
    (optional) — failure here would narrow the interpretation of C3 from
    "identity-position separation" to "color-position separation."

## 3. Proposed Method

Step 1: Modify NonParametricEncoder (src/models_dual_stream.py).
  - Added conv_identity, dyn_readout parameter, CGIR readout with stop-gradient.
  - Parameter increase: conv_identity adds 128×8 + 8 = 1,032 params (< 0.5% of total).

Step 2: Create src/run_phase0_sfa_cgir.py.
Four arms × 5 seeds:

Arm A (CGIR+SFA+CCR): Centroid-gated identity readout, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="none", CCR=covariance
  (ccr_smooth=10, ccr_spatial=10), d_t=3, gdasr_log_only=True.
  This is the primary test of the hypothesis. CCR is included because without
  it, nothing directly shapes z_coord (M2 demotes JEPA to readout with
  stop-gradient), and tracking degradation would confound the CGIR evaluation.

Arm B (Mean+SFA+CCR): Original mean-pooling z_dyn, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="none", CCR=covariance
  (ccr_smooth=10, ccr_spatial=10), d_t=3, gdasr_log_only=True.
  Direct replication of iter_020 Arm A1 for comparison — the only difference
  from Arm A is dyn_readout (mean vs centroid_gated). This isolates the CGIR
  architectural change.

Arm C (CGIR+SFA+CCR+pos): Centroid-gated identity readout, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="sinusoidal", CCR=covariance
  (ccr_smooth=10, ccr_spatial=10), d_t=3, gdasr_log_only=True.
  Tests whether positional encoding further improves CGIR (it helped collapse
  in iter_020 Arm C).

Arm D (CGIR+SFA no CCR): Centroid-gated identity readout, sfa_weight=0.1,
  var_weight=25.0, cov_weight=25.0, pos_encoding="none", CCR=none, d_t=3,
  gdasr_log_only=True. Tests whether CGIR alone is sufficient or CCR is
  needed for z_coord tracking. If Arm D fails C2 but Arm A passes, this
  confirms CCR's role in maintaining centroid tracking.

Seeds: [42, 123, 456, 789, 999].
Training: 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000.
Evaluation: same protocol as iter_020 (collapse check, centroid MSE via
  linear probe, slowness metrics, VICReg health, semantic disentanglement
  probes with delta_R2_color, GDASR growth-point logging).
Additional: C4 identity probe — linear probe on compound identity (color+size).
Checkpoint evaluation at step 2500 and final evaluation at step 5000.

Step 3: Falsification audit.
Compare all arms against all criteria. The primary comparison is Arm A vs B
(CGIR vs mean-pooling, all else equal). If Arm A achieves C3 but Arm B
does not, the CGIR architectural change is confirmed as causally responsible.

Files modified:
- src/models_dual_stream.py (CGIR implementation)
- src/pre_registration.md (this file)
- src/run_phase0_sfa_cgir.py (to be created)

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
