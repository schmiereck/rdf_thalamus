# RDF Scientific Pre-Registration

*   **Iteration:** 022
*   **Phase:** 21
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
The persistent semantic disentanglement failure (delta_R2_identity = -0.035 in iter_021)
is caused by an architectural information bottleneck: z_dyn[c] is a single scalar
per channel, and with d_t=3 active channels, only 3 scalars are available to encode
3 objects × 4 identity features (R, G, B, radius) = 12 values. This makes it
information-theoretically impossible for z_dyn to fully encode identity.

Expanding z_dyn[c] from 1 scalar to K=4 sub-features per channel (total capacity:
3×4=12, matching the identity DOF) via a conv4-backed centroid-gated per-channel
feature head will raise mean delta_R2_identity from -0.035 to ≥ +0.10, and
mean delta_R2_color from +0.050 to ≥ +0.15, across 5 seeds at 5000 steps.

The three interventions are isolatable and ordered by cost:
(a) Conv4 feature readout: richer source (128-dim conv4 vs 8-dim conv_identity),
    but z_dyn[c] remains 1 scalar. Expected: small improvement if d_max=8
    projection loses identity information.
(b) Expanded d_max=16: more spatial channels, but z_dyn[c] remains 1 scalar.
    Expected: minimal improvement since the bottleneck is per-channel capacity,
    not channel count.
(c) Sub-features K=4: expands per-channel capacity from 1 to 4 scalars.
    Expected: largest improvement, as this directly addresses the bottleneck.

## 2. Falsification Criterion

### Directive 1: Unambiguous Falsification Formula

All criteria use IMPROVEMENT-based thresholds: metric(arm) - metric(control).

**PRIMARY (C4 — Identity):**
```
If mean_over_seeds(delta_R2_identity[Arm_C] - delta_R2_identity[Ctrl]) < 0.10,
then the single-scalar bottleneck is NOT the primary cause,
and the hypothesis is falsified.
```

**SECONDARY (C3 — Color):**
```
If mean_over_seeds(delta_R2_color[Arm_C] - delta_R2_color[Ctrl]) < 0.10,
then sub-features do not selectively improve color encoding
beyond what the control achieves.
```

**ANCILLARY (C1 — Collapse):**
```
If collapsed_seeds(Ctrl) >= 2 OR collapsed_seeds(Arm_C) >= 2,
then training instability prevents interpretation.
```

**ANCILLARY (C2 — Tracking):**
```
If mean_over_seeds(centroid_MSE[Arm_C]) > 1.10 * mean_over_seeds(centroid_MSE[Ctrl]),
then the architectural change degrades coordinate tracking.
```

### Directive 2: Normalized Temporal Variance

Add to metrics:
```
normalized_temporal_var(z) = mean(frame-to-frame Δz²) / mean(z²)
```

SFA effectiveness criterion:
```
SFA_effective = normalized_temporal_var(z_dyn) < normalized_temporal_var(z_coord)
```

If Arm C passes the identity threshold (C4) BUT
normalized_dyn_var[Arm_C] ≥ normalized_coord_var[Arm_C],
the result is interpreted as:
"capacity enables encoding; SFA is along for the ride"
— a WEAKER claim than "SFA shapes disentanglement."

### Directive 3: Capacity vs SFA Distinction

Pre-commit:
If Arm C passes delta_R2_identity ≥ 0.10 improvement over Ctrl
but per-sub-feature probes show no selective encoding
(each sub-feature has similar R² across all 4 identity dimensions),
then report as:
"capacity enables identity encoding, but SFA does not produce
disentangled sub-feature specialization"
— NOT as "SFA shapes disentanglement."

## 3. Proposed Method
== EXPERIMENT DESIGN ==

4 arms × 5 seeds × 5000 steps. Seeds: [42, 123, 456, 789, 999].

Arm Ctrl (CGIR+SFA+CCR, d_max=8, K=1, dyn_source="spatial"):
  Replication of iter_021 Arm A as within-run control. Same architecture:
  conv_identity: 128→8, centroid-gated readout, z_dyn shape (B, 8).
  Purpose: Establish baseline for improvement-based criteria.

Arm A (Conv4 CGIR, d_max=8, K=1, dyn_source="conv4"):
  Intervention (a): Richer feature source. Instead of reading z_dyn from the
  d_max=8 conv_identity projection, read from conv4 features (B, 128, 8)
  at the centroid-gated spatial position. Implementation: interpolate conv4
  to (B, 128, 128), use p_c from spatial stream to soft-attend, producing
  (B, d_max, 128) attended features, then linear(128→1) per channel →
  z_dyn (B, d_max). z_dyn[c] is still 1 scalar. Tests whether the d_max=8
  projection bottleneck is the cause.

Arm B (Expanded d_max=16, K=1, dyn_source="spatial", CGIR):
  Intervention (b): More latent spatial channels. d_max=16 instead of 8.
  conv_spatial and conv_identity project 128→16. More channels means each
  channel can specialize more, but z_dyn[c] is still 1 scalar.
  Tests whether channel count vs per-channel capacity matters.

Arm C (Sub-Features K=4, d_max=8, dyn_source="spatial", CGIR):
  Intervention (c): Expanded per-channel capacity. conv_identity projects
  128→d_max*K=32. Reshape to (B, d_max, K, 128) after interpolation.
  Use shared centroid attention p_c for all K sub-features of each channel:
  z_dyn[c,k] = sum_s p_c[c,s] * a_identity[c*K+k, s] → (B, d_max, K).
  Flatten to (B, d_max*K=32) for SFA/VICReg/predictor. SFA slowness on
  all d_t*K=12 active features. VICReg on (B, 12) batch-wise. Predictor
  input: H*(d_max + d_max*K)=3*(8+32)=120, output: d_max+d_max*K=40.
  Tests whether 1 scalar per channel is the fundamental bottleneck.

== ALL ARMS PRESERVE ==
- SFA on z_dyn (all K sub-features per channel), sfa_weight=0.1
- CCR covariance mode on z_coord, ccr_smooth=10, ccr_spatial=10
- d_t=3 frozen, GDASR log-only
- pos_encoding="none" (pos-encoding-hurts is a cross-objective regularity)
- Adam lr=1e-3, batch=32, replay_buffer=2000
- JEPA predictor as readout with stop-gradient (M2)

== METRICS ==

1. NORMALIZED TEMPORAL VARIANCE:
   - temporal_var(z) = mean((z[t+1] - z[t])²) over consecutive frames
   - spatial_var(z) = mean(z²) over all frames (proxy for magnitude)
   - normalized_var = temporal_var / (spatial_var + 1e-8)
   - Report separately for z_dyn (active dims only: d_t * K) and z_coord (d_t)
   - SFA effective iff normalized_dyn_var < normalized_coord_var

2. CENTROID TRACKING QUALITY:
   - For each channel c, compute corr(Δz_coord[c], Δtrue_pos[matched_obj[c]])
   - Also compute corr(z_coord[c], true_pos[matched_obj[c]]) (level tracking)
   - High correlation = good tracking. Low = tracking failure.
   - Match channels to objects using the same dim_to_obj mapping as semantic probes.

3. PER-SUB-FEATURE IDENTITY PROBES (Arm C with K=4 only):
   - For each channel c in [0, d_t) and sub-feature k in [0, K):
     compute R² of z_dyn[c*K+k] against [R, G, B, radius_normalized] individually
   - Also compute per-sub-feature multivariate R² against the full identity vector
   - Test: if disentangled, some (c,k) pairs should have high R² for specific identity
     dimensions. If distributed, all (c,k) have similar R² across all dims.

4. SEMANTIC PROBES (standard, all arms):
   - For fair comparison across arms, pool K sub-features per channel:
     z_dyn_pooled[c] = mean(z_dyn[c*K:(c+1)*K])
   - Then use z_dyn_pooled[c] for the standard probe, just like the K=1 case
   - This ensures each arm produces the same number of probed dimensions

5. COLLAPSE CHECK: per-dim std < 0.5 in < 2/5 seeds (same as iter_021).
6. CENTROID MSE: arm MSE ≤ 1.10 × ctrl MSE (no tracking degradation).
7. GDASR growth-point logging (log-only, per M3).

== FALSIFICATION AUDIT ==

```
C1 (Collapse): Ctrl collapsed seeds < 2 AND Arm C collapsed seeds < 2
C2 (Tracking): Arm C centroid MSE ≤ 1.10 × Ctrl centroid MSE
C3 (Color):   mean_over_seeds(delta_R2_color[Arm C] - delta_R2_color[Ctrl]) ≥ 0.10
C4 (Identity): mean_over_seeds(delta_R2_identity[Arm C] - delta_R2_identity[Ctrl]) ≥ 0.10
C5 (SFA effective, advisory): normalized_dyn_var[Arm C] < normalized_coord_var[Arm C]

OVERALL: C1 AND C2 AND C4 → hypothesis validated
```

If C4 passes but C5 fails: "capacity enables encoding; SFA is along for the ride."
If C4 passes but per-sub-feature probes show no selective encoding:
"capacity enables identity encoding, but SFA does not produce disentangled
sub-feature specialization."

== CODE CHANGES ==

1. src/models_dual_stream.py:
   - Add `sub_features` (K, default=1) and `dyn_source` ("spatial"|"conv4",
     default="spatial") to NonParametricEncoder.
   - For dyn_source="conv4", K=1: add dyn_proj Linear(128,1), compute attended
     conv4 features via centroid attention, then project 128→1 per channel.
   - For sub_features=K: modify conv_identity to output d_max*K channels,
     reshape after interpolation, apply shared centroid attention, produce
     z_dyn of shape (B, d_max*K) when flattened.
   - Add `d_dyn` property to NonParametricEncoder.
   - Modify DualStreamPredictor to accept d_dyn=d_max*K instead of d_max,
     with input H*(d_max + d_dyn) and output d_max + d_dyn.
   - Modify NonParametricJEPASpatial to pass K and dyn_source through.
   - In SFA mode: handle z_dyn shape (B, d_max*K) for SFA, VICReg, predictor.
   - In JEPA mode: handle z_dyn shape (B, d_max*K) for predictor and losses.

2. src/run_phase0_sfa_archceiling.py (NEW):
   - 4 arms × 5 seeds × 5000 steps
   - Same training loop as run_phase0_sfa_cgir.py
   - Updated evaluation: normalized temporal variance, centroid tracking
     quality, per-sub-feature identity probes for Arm C
   - Checkpoint at step 2500 and final at 5000
   - Falsification audit using improvement-based criteria
   - Output directory: archive/iter_022/results/

3. src/pre_registration.md: Update with this plan.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
