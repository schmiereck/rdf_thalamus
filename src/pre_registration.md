# RDF Scientific Pre-Registration

*   **Iteration:** 024
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Two-part hypothesis tested simultaneously:

PART A (multi-step slowness): We test whether multi-step SFA with temporal
horizon k∈{20,50,100} accumulates gradient over longer windows than single-step
SFA (k=1), enabling extraction of features that are slow at longer timescales.
Specifically, multi-step SFA computes L_SFA_k = ||z_dyn(t) - z_dyn(t-k)||² / k
using a z_dyn trajectory buffer maintained during training. We predict that if
identity features require longer temporal integration to separate from
position-related z_dyn variation, then k>>1 should produce delta_R2_color
improvement where k=1 failed. This is refuted if delta_R2_color does not exceed
the single-step baseline across all k values.

PART B (temporal contrastive probe): We test whether temporal contrastive
learning (NT-Xent) on z_dyn — where positive pairs are same-trajectory z_dyn at
different timesteps and negative pairs are z_dyn from different trajectories in
the same batch — can produce identity encoding. The hypothesis is that:
(1) temporal invariance (positives) makes z_dyn stable like SFA, and
(2) cross-scene discrimination (negatives) forces z_dyn to encode scene-specific
information that is NOT position (handled by z_coord), leaving identity as the
primary discriminable attribute. This is refuted if delta_R2_color does not
exceed established baselines. Note: NT-Xent at τ=0.1 with VICReg simultaneously
is a known fight, and a silently-collapsed Arm D would be misread as a null.
The NT-Xent loss is:
L_contra = -log(exp(sim(z_target_dyn[i], z_hist_dyn[i,-1])/τ) / Σ_j exp(sim(z_target_dyn[i], z_hist_dyn[j,-1])/τ))
with cosine similarity and temperature τ=0.1.

## 2. Falsification Criterion
PRE-DECLARED FALSIFICATION CRITERIA:

1. M2 (slowness as representation-shaping mechanism) is REFUTED iff
   delta_R2_color < 0.10 across ALL k ∈ {20, 50, 100} for d_max=8 (Arms A-C)
   AND delta_R2_color ≤ 0.137 (iter_023 d_max=16 baseline) for d_max=16 (Arm E).
   No post-hoc reframing of "SFA works" if only z_dyn temporal variance moves —
   variance reduction is mechanical, not the claim.

2. Arm D is consistent with a genuine objective-driven effect iff
   delta_R2_color ≥ 0.10 at d_max=8 AND exceeds the best d_max=8 multi-step SFA
   arm by ≥ 0.05 with non-overlapping seed CIs. Also, Arm D must pass a collapse
   gate: ≤ 2/5 collapsed seeds (matching the SFA arms). NT-Xent at τ=0.1
   with VICReg simultaneously is a known fight, and a silently-collapsed Arm D
   would be misread as a null.

3. C5 is DROPPED ENTIRELY from this iteration. It is a structurally impossible
   metric artifact (iter_023: 0/35 seeds). Re-running it would be construction,
   not evidence.

4. All 5000 steps MUST complete before any falsification judgment. The early
   step-2000 checkpoint is diagnostic/monitoring only — it must not be used to
   terminate runs or declare the narrative "killed." Killing runs early destroys
   the dataset needed for the post-mortem analysis. A clean double null at step
   5000 is a successful iteration outcome that justifies pivoting to
   object-tracking-ID contrastive in iter_025.

## 3. Proposed Method
EXPERIMENT DESIGN: 6 arms × variable seeds × 5000 steps.
Seeds: [42, 123, 456, 789, 999] for main arms; [42] for diagnostic arm.
Total runs: 26 (within the ~35-run budget of iter_023).

ARM CONFIGURATIONS:

Arm A (Multi-step SFA k=20, d_max=8): 5 seeds
  - primary_objective="sfa", sfa_weight=10.0 (ramp 0.1→10.0 over 500 steps)
  - Multi-step SFA with k=20 using z_dyn trajectory buffer
  - CGIR, CCR covariance, d_t=3, sim_weight=25.0, var_weight=25.0, cov_weight=25.0

Arm B (Multi-step SFA k=50, d_max=8): 5 seeds
  - Same as Arm A but sfa_k=50

Arm C (Multi-step SFA k=100, d_max=8): 5 seeds
  - Same as Arm A but sfa_k=100

Arm D (Temporal Contrastive, d_max=8): 5 seeds
  - primary_objective="contrastive", contrastive_weight=25.0, temperature=0.1
  - NO SFA loss (sfa_weight=0). NT-Xent replaces SFA as the temporal objective.
  - CGIR, CCR covariance, d_t=3, sim_weight=25.0, var_weight=25.0, cov_weight=25.0

Arm E (d_max=16 + Multi-step SFA k=50): 5 seeds
  - Same as Arm B but d_max=16 (carrying forward the best capacity from iter_023)

Arm F (JEPA stop-gradient diagnostic): 1 seed [n=1, indicative only, not evidence on its own]
  - Same as Arm B (k=50, d_max=8) but sim_weight=0
  - Tests whether removing the JEPA readout entirely changes the picture

MULTI-STEP SFA IMPLEMENTATION:
- Maintain a z_dyn trajectory buffer (collections.deque, maxlen=110) alongside
  the replay buffer during training
- At each step, encode the current environment frame (x_current) through the
  encoder WITH gradients and store z_dyn.detach() in the trajectory buffer
- For the multi-step SFA loss: sfa_loss_k = MSE(z_dyn_current, z_dyn_trajectory[-k-1]) / k
- NOTE on k=100: with k=100, objects may exit and re-enter over that horizon.
  A representation encoding a batch-statistic (e.g., global color histogram,
  slowly drifting scene mean) would satisfy ||z_dyn(t) - z_dyn(t-k)||² at
  near-zero cost without encoding per-object identity. The invariance-vs-
  discrimination diagnostic (Metric 1b above) must accompany any k=100 claim.
- The gradient flows through z_dyn_current back to the encoder; z_past is a
  fixed (detached) target from the buffer
- This requires one additional encoder forward pass per step (batch_size=1),
  which is ~3% overhead over the main batch forward pass
- The SFA loss is ADDED to the main loss (VICReg + JEPA readout), replacing
  the single-step SFA in the model's forward pass (set sfa_weight=0 in model,
  compute multi-step SFA externally)

TEMPORAL CONTRASTIVE (NT-Xent) IMPLEMENTATION:
- Computed inside NonParametricJEPASpatial.forward() in the SFA branch
- Uses z_target_dyn (anchor, WITH gradients) and z_hist_dyn[:,-1] (positive,
  WITH gradients — standard SimCLR, no stop-gradient needed since VICReg
  prevents collapse)
- NT-Xent loss:
  z_anchor = F.normalize(z_target_dyn[:, :d_t_dyn], dim=-1)  # (B, d_t_dyn)
  z_positive = F.normalize(z_hist_dyn[:, -1, :d_t_dyn], dim=-1)  # (B, d_t_dyn)
  sim_matrix = mm(z_anchor, z_positive.T) / τ  # (B, B)
  labels = arange(B)  # diagonal = positive pairs
  contrastive_loss = cross_entropy(sim_matrix, labels)
- Uses full d_t-dimensional z_dyn vector (not per-dimension) to allow the
  model to allocate different dimensions to different identity aspects

CODE CHANGES:
1. src/models_dual_stream.py: Add contrastive_weight and temperature parameters
   to NonParametricJEPASpatial.__init__(). In the SFA forward branch, add
   NT-Xent contrastive loss computation before the detach operations. Add
   "contrastive" as a new primary_objective option that uses NT-Xent instead
   of SFA. Add contrastive_loss to the returned loss dict.

2. src/run_phase0_sfa_multistep.py (NEW): Main experiment runner.
   - Based on run_phase0_sfa_sweep.py structure
   - 6 arms × variable seeds × 5000 steps
   - For Arms A-C, E: multi-step SFA via z_dyn trajectory buffer
   - For Arm D: temporal contrastive (model-level)
   - For Arm F: sim_weight=0 diagnostic
   - Same evaluation suite: semantic probes, collapse checks, centroid MSE,
     tracking quality, normalized temporal variance
   - Results saved to archive/iter_024/results/

3. src/pre_registration.md: Updated with this plan.

METRICS (same as iter_023, directly comparable):
1. delta_R2_color (PRIMARY criterion — improvement over iter_023 baseline of 0.05)
1b. INVAR-VS-DISCRIMINATIVE DIAGNOSTIC (required before any 'k=N works' claim):
    (a) within-trajectory z_dyn variance vs. between-trajectory z_dyn variance,
    (b) whether the same z_dyn would pass delta_R2_color on a shuffled-frame
        control where the temporal label is destroyed.
    If shuffling does not collapse the probe, the signal was not in
    z_dyn-via-SFA — it was in the encoder geometry, and the result is
    constructional. This guards against a representation encoding a batch
    statistic (e.g., global color histogram, slowly drifting scene mean) that
    satisfies ||z_dyn(t) - z_dyn(t-k)||² at near-zero cost without encoding
    per-object identity, which is a constructional pass, not evidence for M2.
2. delta_R2_identity
3. C1 (Collapse): per_dim_std < 0.5 in < 2/5 seeds per arm
4. Centroid MSE
5. Normalized temporal variance (dyn and coord)
6. Slowness ratio
7. Per-dim std, collapse counts
8. Tracking quality
9. GDASR growth-point logs (log-only mode)
10. Step-2000 checkpoint evaluation (diagnostic/monitoring only; ALL 5000 steps must complete)

TRAINING PROTOCOL:
- 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000
- d_t=3 frozen, gdasr_log_only=True
- All arms: CGIR dyn_readout, CCR covariance mode
- SFA ramp: 0.1 → target_weight over 500 steps (stability from iter_023 A6)
- For multi-step SFA arms: trajectory buffer pre-filled during the first 110
  steps of environment interaction before SFA loss computation begins

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
