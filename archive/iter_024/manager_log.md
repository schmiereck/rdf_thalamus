# Research Manager Log - Iteration 024

## Iteration 024 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (for iteration 024 planning)

1. Directional: the next iteration must run the multi-step SFA test (k>>1, e.g. k∈{20, 50, 100}) on the existing d_max=8 CNN as the primary arm, with the d_max=16 ramp configuration carried forward as a second arm. This is the last slowness-based shot at M2 — if delta_R2_color stays flat across k, slowness on z_dyn is functionally dead and the planner should be prepared to pivot, not iterate further within the slowness framing.

2. Discipline: pre-declare the falsification criterion in the plan itself, before runs begin. State: "M2 is refuted as a representation-shaping mechanism iff delta_R2_color < 0.10 across all k ∈ {20, 50, 100} for d_max=8 and ≤ 0.137 (iter_023 d_max=16 baseline) for d_max=16." No post-hoc reframing of "SFA works" if only z_dyn temporal variance moves — variance reduction is mechanical, not the claim. Also drop C5 from the plan entirely; it is a derived identity and re-running it would be construction, not evidence.

3. Discipline / scope: the plan should include a contingency arm — at least one non-slowness probe (preferably the temporal-augmentation contrastive variant from the journal, since object-tracking IDs need infrastructure work) staged for the same iteration if multi-step SFA underperforms by an early checkpoint (e.g. step 2000). This avoids burning a full iteration on a third consecutive null result and gives the iteration two independent chances at clearing the identity-encoding bottleneck. Also retain the JEPA-readout stop-gradient diagnostic (Open Question 7) as a low-cost ablation arm — a single seed checking whether sim_weight=0 changes the picture is cheap insurance.

---

## Iteration 024 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Two-part hypothesis tested simultaneously:

PART A (last slowness shot): Multi-step SFA with temporal horizon k∈{20,50,100}
accumulates gradient over longer windows than single-step SFA (k=1), enabling
extraction of features that are slow at longer timescales. Specifically, multi-step
SFA computes L_SFA_k = ||z_dyn(t) - z_dyn(t-k)||² / k using a z_dyn trajectory
buffer maintained during training. If identity features require longer temporal
integration to separate from position-related z_dyn variation, then k>>1 should
produce delta_R2_color improvement where k=1 failed.

PART B (non-slowness probe): Temporal contrastive learning (NT-Xent) on z_dyn,
where positive pairs are same-trajectory z_dyn at different timesteps and negative
pairs are z_dyn from different trajectories in the same batch, will produce
identity encoding because: (1) temporal invariance (positives) makes z_dyn stable
like SFA, and (2) cross-scene discrimination (negatives) forces z_dyn to encode
scene-specific information that is NOT position (handled by z_coord), leaving
identity as the primary discriminable attribute. The NT-Xent loss is:
L_contra = -log(exp(sim(z_target_dyn[i], z_hist_dyn[i,-1])/τ) / Σ_j exp(sim(z_target_dyn[i], z_hist_dyn[j,-1])/τ))
with cosine similarity and temperature τ=0.1.

**Proposed Falsification Criterion:**
PRE-DECLARED FALSIFICATION CRITERIA:

1. M2 (slowness as representation-shaping mechanism) is REFUTED iff
   delta_R2_color < 0.10 across ALL k ∈ {20, 50, 100} for d_max=8 (Arms A-C)
   AND delta_R2_color ≤ 0.137 (iter_023 d_max=16 baseline) for d_max=16 (Arm E).
   No post-hoc reframing of "SFA works" if only z_dyn temporal variance moves —
   variance reduction is mechanical, not the claim.

2. Temporal contrastive (Arm D) is deemed PROMISING iff delta_R2_color ≥ 0.15,
   exceeding the iter_023 best of 0.137 (which was a capacity effect, not an
   objective effect). If delta_R2_color < 0.10 for Arm D, the temporal
   contrastive approach does not meaningfully improve identity encoding.

3. C5 is DROPPED ENTIRELY from this iteration. It is a structurally impossible
   metric artifact (iter_023: 0/35 seeds). Re-running it would be construction,
   not evidence.

4. Early checkpoint at step 2000: if ALL multi-step SFA arms (A-C) show
   delta_R2_color < 0.08 at the checkpoint, the slowness hypothesis is
   effectively dead regardless of final-step results, and the contrastive
   finding becomes the primary outcome.

**Proposed Method:**
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

Arm F (JEPA stop-gradient diagnostic): 1 seed
  - Same as Arm B (k=50, d_max=8) but sim_weight=0
  - Tests whether removing the JEPA readout entirely changes the picture

MULTI-STEP SFA IMPLEMENTATION:
- Maintain a z_dyn trajectory buffer (collections.deque, maxlen=110) alongside
  the replay buffer during training
- At each step, encode the current environment frame (x_current) through the
  encoder WITH gradients and store z_dyn.detach() in the trajectory buffer
- For the multi-step SFA loss: sfa_loss_k = MSE(z_dyn_current, z_dyn_trajectory[-k-1]) / k
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
2. delta_R2_identity
3. C1 (Collapse): per_dim_std < 0.5 in < 2/5 seeds per arm
4. Centroid MSE
5. Normalized temporal variance (dyn and coord)
6. Slowness ratio
7. Per-dim std, collapse counts
8. Tracking quality
9. GDASR growth-point logs (log-only mode)
10. Step-2000 checkpoint evaluation for early falsification

TRAINING PROTOCOL:
- 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000
- d_t=3 frozen, gdasr_log_only=True
- All arms: CGIR dyn_readout, CCR covariance mode
- SFA ramp: 0.1 → target_weight over 500 steps (stability from iter_023 A6)
- For multi-step SFA arms: trajectory buffer pre-filled during the first 110
  steps of environment interaction before SFA loss computation begins

---

## Iteration 024 -> Planner [Strategic Guidance]

Manager's Note — Iteration 024

The plan is broadly aligned with the pre-planning hints (multi-step SFA sweep, contrastive contingency, JEPA stop-gradient diagnostic, C5 dropped, early checkpoint). Three things must change before execution.

**1. Falsification criterion for Arm D is mis-calibrated and asymmetric.** The plan declares Arm D "PROMISING iff delta_R2_color ≥ 0.15," but the d_max=16 baseline (0.137) was a *channel-capacity* effect at d_max=16, while Arm D runs at **d_max=8**. The fair comparison for an objective-driven gain is the d_max=8 baseline (delta_R2_color ≈ 0.05). State the contrastive criterion as: "Arm D is consistent with a genuine objective-driven effect iff delta_R2_color ≥ 0.10 at d_max=8 **and** exceeds the best d_max=8 multi-step SFA arm by ≥ 0.05 with non-overlapping seed CIs." Without the head-to-head clause, you cannot separate "contrastive works" from "any non-collapsing temporal objective gives ~0.10." Also add a pre-declared **collapse gate** for Arm D matching the SFA arms (≤ 2/5 collapsed seeds); NT-Xent at τ=0.1 with VICReg simultaneously is a known fight, and a silently-collapsed Arm D would be misread as a null.

**2. Construction-vs-empirical discipline on multi-step SFA — protect against the obvious trivial solution.** With k=100 and an environment where many objects exit/re-enter the scene over that horizon, a representation that encodes a *batch-statistic* (e.g. the global colour histogram, slowly drifting scene mean) will satisfy `||z_dyn(t) - z_dyn(t-k)||²` at near-zero cost without encoding per-object identity. That is a constructional pass, not evidence for M2. Before any "k=N works" claim, require an **invariance-vs-discrimination diagnostic** alongside delta_R2_color: report (a) within-trajectory z_dyn variance vs. between-trajectory z_dyn variance, and (b) whether the same z_dyn would pass delta_R2_color on a *shuffled-frame* control where the temporal label is destroyed. If shuffling does not collapse the probe, the signal was not in z_dyn-via-SFA, it was in the encoder geometry, and the result is constructional.

**3. Language and pre-registration.** The proposed `pre_registration.md` is good in structure but the Orchestrator will commit it before execution — so tighten the hypothesis text: drop "will produce identity encoding because…" (assertive) and reframe as "we test whether… we predict that…, refuted if…." Also drop the phrase "last slowness shot" from any committed artefact; that framing pressures a positive-spin report on a marginal result. The honest framing is: this iteration tests multi-step SFA and temporal contrastive as two independent candidates; **both may fail, and a clean double null is a successful iteration outcome that justifies pivoting to object-tracking-ID contrastive (the journal's HIGH-PRIORITY path) in iter_025.**

Minor: Arm F at a single seed is fine as a diagnostic but its result is not evidence on its own — instruct the executor to report it explicitly as "n=1, indicative only." And confirm that the early-step-2000 checkpoint kills only the *narrative*, not the runs: keep all 5000 steps so the dataset for the post-mortem stays intact.

Proceed once the criterion in (1) is rewritten, the diagnostic in (2) is added to the metrics list, and the pre-registration file is committed verbatim with the language fixes in (3).

---

