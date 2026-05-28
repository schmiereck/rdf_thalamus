# Research Manager Log - Iteration 021

## Iteration 021 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
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

**Proposed Falsification Criterion:**
PRIMARY: If Arm C (K=4 sub-features) fails to achieve mean delta_R2_identity
≥ 0.10 across 5 seeds at step 5000, the single-scalar bottleneck is NOT the
primary cause of the disentanglement failure, and the hypothesis is falsified.

SECONDARY (supporting evidence): If Arms A and B show delta_R2_identity
improvements < 0.05 over Control, this confirms that richer feature sources
and more spatial channels cannot compensate for the per-channel scalar bottleneck.

ANCILLARY FALSIFICATION: If Arm C achieves delta_R2_identity ≥ 0.10 but
delta_R2_color improvement(Arm C - Control) < 0.05, the sub-features are
not selectively improving identity encoding — they are just adding capacity
that happens to fit the probe.

THRESHOLD CLARIFICATION (resolving Issue 2): All criteria use IMPROVEMENT-based
thresholds: metric(arm) - metric(control) ≥ threshold. The absolute criterion
from iter_021 (delta_R2_color(Arm A) ≥ 0.10) was incoherent with the observed
improvement (+0.124) and is replaced.

**Proposed Method:**
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

== METRICS IMPROVEMENTS (resolving Issues 1 and 2) ==

1. SLARNESS RESOLUTION (Issue 1):
   - Report temporal_var(z_dyn) and temporal_var(z_coord) as SEPARATE
     absolute numbers (mean squared frame-to-frame change per dimension).
   - Report NORMALIZED temporal variance: temporal_var / (spatial_std²),
     giving a scale-free measure of "how fast" each stream is relative
     to its own magnitude. SFA works iff normalized_dyn_var < normalized_coord_var.
   - Report centroid tracking quality: corr(Δz_coord[c], Δtrue_pos[c])
     per channel. If z_coord is nearly static (normalized_coord_var ≈ 0)
     while objects move, this is a tracking failure, not slowness.
   - Label the slowness ratio convention explicitly: ratio = dyn_delta/coord_delta,
     where ratio < 1 means SFA succeeded (z_dyn slower than z_coord).

2. THRESHOLD CLARIFICATION (Issue 2):
   - C3 (color): improvement(arm - ctrl) in delta_R2_color ≥ 0.10
   - C4 (identity): improvement(arm - ctrl) in delta_R2_identity ≥ 0.10
   - Also report absolute delta_R2 for reference, but improvement is primary.

3. PER-DIMENSION PROBES for Arm C (K=4):
   - For each channel c and sub-feature k, fit R² against [R, G, B, radius]
     individually to characterize which sub-features encode which identity
     dimensions. This tests whether disentanglement is emergent (sub-feature k
     spontaneously encodes R, etc.) or distributed.

4. COLLAPSE CHECK: per-dim std < 0.5 in < 2/5 seeds (same as iter_021).
5. CENTROID MSE: arm MSE ≤ 1.10 × ctrl MSE (no tracking degradation).
6. GDASR growth-point logging (log-only, per M3).

== CODE CHANGES ==

1. src/models_dual_stream.py:
   - Add `sub_features` (K, default=1) and `dyn_source` ("spatial"|"conv4",
     default="spatial") to NonParametricEncoder.
   - For dyn_source="conv4", K=1: add dyn_proj Linear(128,1), compute attended
     conv4 features via centroid attention, then project 128→1 per channel.
   - For sub_features=K: modify conv_identity to output d_max*K channels,
     reshape after interpolation, apply shared centroid attention, produce
     z_dyn of shape (B, d_max*K) when flattened.
   - Modify NonParametricJEPASpatial to pass K and dyn_source through.
   - In SFA mode: handle z_dyn shape (B, d_max*K) for SFA, VICReg, predictor.
   - Modify DualStreamPredictor to accept d_dyn=d_max*K instead of d_max,
     with input H*(d_max + d_dyn) and output d_max + d_dyn.

2. src/run_phase0_sfa_archceiling.py (NEW):
   - 4 arms × 5 seeds × 5000 steps
   - Same training loop as run_phase0_sfa_cgir.py
   - Updated evaluation: normalized temporal variance, centroid tracking
     quality, per-sub-feature identity probes for Arm C
   - Checkpoint at step 2500 and final at 5000
   - Falsification audit using improvement-based criteria

3. src/pre_registration.md: Update with this plan.

---

## Iteration 021 -> Planner [Strategic Guidance]

# Manager's Note — Strategic Guidance for Iteration 022

## 1. Falsification Criterion Contains an Internal Contradiction — Fix Before Execution

Your primary falsification criterion states: *"If Arm C fails to achieve mean delta_R2_identity ≥ 0.10,"* which reads as **absolute**. Your own threshold clarification section then declares: *"All criteria use IMPROVEMENT-based thresholds: metric(arm) - metric(control) ≥ threshold."* These are not the same thing. The iter_021 confusion stemmed from exactly this kind of ambiguity.

**Directive:** The pre-registration file must contain ONE unambiguous formula. I recommend improvement-based: `mean_over_seeds(delta_R2_identity[Arm C] - delta_R2_identity[Ctrl]) ≥ 0.10`. Write it as pseudocode. The Orchestrator will commit this before execution; any deviation during analysis is a protocol violation.

## 2. SFA Effectiveness Remains Unresolved — Arm C Success Could Be Constructional

Your Research Journal correctly identifies the slowness ratio pathology (z_dyn changes MORE than z_coord) as a critical open question. You are now proposing to add sub-features to a representation stream that may not be doing what SFA is supposed to make it do. This is a legitimate bet, but it carries a specific interpretive risk:

**If Arm C passes the threshold, does that validate SFA, or does it merely validate "more capacity = more encoding"?** The per-sub-feature identity probes you included are necessary but not sufficient. You must also report the **normalized temporal variance** (your own proposed metric) for Arm C's z_dyn sub-features. If Arm C achieves delta_R2_identity ≥ 0.10 *while normalized_dyn_var remains ≥ normalized_coord_var* (i.e., SFA still isn't making identity slow), then the result is compatible with: "capacity enables encoding; SFA is along for the ride." That is a weaker claim than "SFA shapes identity," and the report must say so explicitly. Do not allow a capacity-driven success to be narrated as an SFA validation.

## 3. The K=4 → 12 DOF Mapping Is Nearly Constructional — Acknowledge the Boundary

Providing exactly K=4 sub-features for 4 identity dimensions (R, G, B, radius) comes close to "building in the answer." The per-sub-feature probes correctly test whether SFA organizes these into disentangled slots (R→k₁, G→k₂, etc.) vs. distributed encoding. But I want an explicit pre-registered statement about this:

**If Arm C passes but the per-sub-feature probes show no selective encoding** (each sub-feature has similar R² across all four identity dimensions, indicating distributed rather than disentangled representation), then the result must be reported as: "capacity enables identity encoding, but SFA does not produce disentangled sub-feature specialization." This is still informative, but it is fundamentally different from "SFA drives emergent disentanglement," and the language in the final report must reflect that distinction.

---

**Summary:** Proceed with the experiment, but (1) write the falsification criterion as an unambiguous improvement formula in the pre-registration, (2) include normalized temporal variance in the success-condition reporting so that SFA effectiveness is not confounded with capacity effectiveness, and (3) pre-commit to language distinguishing "capacity enables encoding" from "SFA shapes disentanglement" based on the per-sub-feature probe results. The Orchestrator will write `src/pre_registration.md` before execution begins — ensure these three points are in it.

---

