You are executing iter_024 of the Thalamus project. Read src/pre_registration.md FIRST to get the full experiment plan and falsification criteria.

CRITICAL: Before running any code, read the pre-registered hypothesis and falsification criteria in src/pre_registration.md and strictly adhere to them.

The experiment tests two independent hypotheses:
PART A: Multi-step SFA (k=20,50,100) accumulates gradient over longer windows, potentially producing identity-position separation where single-step SFA (k=1) failed.
PART B: Temporal contrastive (NT-Xent) on z_dyn may produce identity encoding through temporal invariance + cross-scene discrimination.

EXPERIMENT DESIGN: 6 arms × variable seeds × 5000 steps (26 total runs).
Seeds: [42, 123, 456, 789, 999] for main arms; [42] for diagnostic arm.

ARM CONFIGURATIONS:
Arm A (Multi-step SFA k=20, d_max=8): 5 seeds
Arm B (Multi-step SFA k=50, d_max=8): 5 seeds
Arm C (Multi-step SFA k=100, d_max=8): 5 seeds
Arm D (Temporal Contrastive NT-Xent, d_max=8): 5 seeds
Arm E (d_max=16 + Multi-step SFA k=50): 5 seeds
Arm F (JEPA stop-gradient diagnostic, sim_weight=0, k=50, d_max=8): 1 seed [n=1, indicative only]

CODE CHANGES NEEDED:

1. src/models_dual_stream.py — Add to NonParametricJEPASpatial:
   a. Add contrastive_weight and temperature parameters to __init__()
   b. Add "contrastive" as a new primary_objective option
   c. In the forward method, when primary_objective="contrastive", compute NT-Xent loss:
      - z_anchor = F.normalize(z_target_dyn[:, :d_t_dyn], dim=-1)
      - z_positive = F.normalize(z_hist_dyn[:, -1, :d_t_dyn], dim=-1) (last history step)
      - sim_matrix = mm(z_anchor, z_positive.T) / τ
      - labels = arange(B) (diagonal = positive pairs)
      - contrastive_loss = cross_entropy(sim_matrix, labels)
   d. In the SFA branch, keep single-step SFA (k=1) as model-level computation; multi-step SFA will be computed externally via trajectory buffer
   e. Add contrastive_loss to the returned loss dict

2. src/run_phase0_sfa_multistep.py (NEW) — Main experiment runner:
   - Based on src/run_phase0_sfa_sweep.py structure
   - 6 arms × variable seeds × 5000 steps
   - For Arms A-C, E: multi-step SFA via z_dyn trajectory buffer
     * Maintain a z_dyn trajectory buffer (collections.deque, maxlen=110)
     * At each step, encode current frame and store z_dyn.detach() in buffer
     * Multi-step SFA loss: MSE(z_dyn_current, z_dyn_trajectory[-k-1]) / k
     * Gradient flows through z_dyn_current; z_past is detached from buffer
     * Pre-fill buffer during first 110 steps before SFA loss computation
     * Set sfa_weight=0 in model forward, add multi-step SFA loss externally
   - For Arm D: temporal contrastive (model-level NT-Xent)
   - For Arm F: sim_weight=0 diagnostic
   - All arms: CGIR dyn_readout, CCR covariance mode, sfa ramp 0.1→target over 500 steps

3. INVARIANCE-VS-DISCRIMINATION DIAGNOSTIC (Manager correction #2):
   For multi-step SFA arms, compute alongside delta_R2_color:
   a. Within-trajectory z_dyn variance: for each trajectory, compute var(z_dyn) across timesteps
   b. Between-trajectory z_dyn variance: compute var(mean(z_dyn per trajectory)) across trajectories
   c. Shuffled-frame control: randomly shuffle the temporal order of frames within each trajectory,
      then re-run the semantic probe. If delta_R2_color on shuffled data is comparable to the
      unshuffled version, the signal was in the encoder geometry, not in z_dyn-via-SFA.

4. Step-2000 checkpoint: diagnostic/monitoring only. ALL 5000 steps must complete.
   Step-2000 results inform the narrative but do NOT terminate runs.

5. Metrics (same as iter_023 plus new diagnostics):
   - delta_R2_color (PRIMARY)
   - delta_R2_identity
   - Collapse rate (≤ 2/5 seeds per arm)
   - Centroid MSE
   - Normalized temporal variance (dyn and coord)
   - Per-dim std, slowness ratio
   - Within-vs-between trajectory z_dyn variance
   - Shuffled-frame delta_R2_color
   - GDASR growth-point logs

6. FALSIFICATION CRITERIA (from pre-registration):
   a. M2 (slowness as representation-shaping mechanism) is REFUTED iff delta_R2_color < 0.10 across ALL k ∈ {20, 50, 100} for d_max=8 (Arms A-C) AND delta_R2_color ≤ 0.137 for d_max=16 (Arm E).
   b. Arm D is consistent with genuine objective-driven effect iff delta_R2_color ≥ 0.10 at d_max=8 AND exceeds the best d_max=8 multi-step SFA arm by ≥ 0.05 with non-overlapping seed CIs. Collapse gate: ≤ 2/5 collapsed seeds.
   c. C5 is DROPPED entirely.
   d. Before any "k=N works" claim, the invariance-vs-discrimination diagnostic must be checked.

TRAINING PROTOCOL:
- 5000 steps, Adam lr=1e-3, batch_size=32, replay_buffer=2000
- d_t=3 frozen, gdasr_log_only=True
- All arms: CGIR dyn_readout, CCR covariance mode
- SFA ramp: 0.1 → target_weight over 500 steps (stability from iter_023 A6)

Results saved to archive/iter_024/results/

IMPORTANT: The existing code in src/models_dual_stream.py already has a NonParametricJEPASpatial class with primary_objective="sfa" mode. You need to ADD the "contrastive" mode to it. Do NOT rewrite the whole file — just add the new functionality. Similarly, src/run_phase0_sfa_sweep.py is a reference for the runner structure but the new runner should be src/run_phase0_sfa_multistep.py.

After all experiments complete, compile the falsification audit exactly as specified in the pre-registration document.