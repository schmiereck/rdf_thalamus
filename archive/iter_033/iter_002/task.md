OVERWRITE the file `src/pre_registration.md` with the following EXACT content. This is the corrected pre-registration incorporating the Research Manager's binding corrections. The file must contain this content VERBATIM — do not modify, add to, or abbreviate it.

```
# Pre-Registration: Iter_033 — Three-Condition Oracle Bracket (Corrected per Manager Review)

## Hypothesis
The best available mean-pool representations (SFA+VICReg sfa_weight=5.0, separate backbone, ΔR²≈0.275, 0% collapse, iter_029 Arm B; AND VICReg-only, separate backbone, ΔR²≈0.045, 0% collapse, iter_029 Arm A) support functional surprise-driven behavior that accounts for at least 20% of the gap between random (lower bound) and oracle (perfect-perception upper bound) baselines, as measured by post-collision attention selectivity (version B: attended object matches max-velocity-change object) on N=2 collision-sparse environments with d_t=3 frozen.

## Conditions (4 total, all sharing identical environment, seed bank, and CLTSMotorController logic)

1. **RANDOM (lower bound):** CLTSMotorController with random token_locus selection each step. All EMA statistics (mu, sigma) are zeroed/reset each step so no learned structure influences behavior. Attention cooldown is set to 0 so locus is re-randomized every step.

2. **LEARNED-VICReg (test arm A):** Load iter_029 Arm A checkpoints (VICReg-only, separate backbone, mean-pool readout, d_max=8, d_t=3). Run encoder + predictor forward pass, feed z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn to the standard CLTSMotorController.get_action(). Checkpoint naming: a_vicreg-only_control_seed{N}.pt

3. **LEARNED-SFA (test arm B):** Load iter_029 Arm B checkpoints (SFA+VICReg sfa_weight=5.0, separate backbone, mean-pool readout, d_max=8, d_t=3). Run encoder + predictor forward pass, feed to the same CLTSMotorController.get_action(). Checkpoint naming: b_sfavicreg,_sfa_5.0_seed{N}.pt

4. **ORACLE (upper bound):** Feed ground-truth positions (info['positions'][:d_t]) as z_coord, ground-truit mean-color per object (np.mean(info['colors'][:d_t], axis=1)) as z_dyn, linear-extrapolation predicted positions (prev_pos + prev_vel * dt) as z_pred_coord, and z_dyn as z_pred_dyn (identity is constant). Feed these tensors to the SAME CLTSMotorController.get_action() method — identical surprise computation (per-channel MSE + EMA normalization + attention cooldown), identical PD tracking, identical push logic.

**IMPORTANT:** The ORACLE condition represents the behavioral ceiling UNDER THE EXISTING MOTOR CODE (with its EMA statistics and push thresholds calibrated implicitly against noisy learned surprise). The ORACLE will produce qualitatively different surprise distributions (near-zero between collisions, sharp spikes at collisions). This bracket quantifies how much of the ORACLE-RANDOM gap is captured by the learned representations, not the absolute behavioral ceiling.

### Channel-to-Object Mapping
All conditions use the same closest-centroid mapping: for each channel c, find the object whose position is closest to centroids[0, c]. For ORACLE, centroids are the ground-truth positions, so the mapping is trivially correct. For LEARNED conditions, the mapping uses the learned soft-argmax centroids. For RANDOM, the mapping uses whatever centroids are available (which will be meaningless, providing the random baseline).

### Surprise Decomposition
Under ORACLE, z_pred_dyn = z_dyn (identity is constant), so err_dyn = 0 and surprise is purely position-driven (err_coord). Under LEARNED conditions, surprise is the sum err_coord + err_dyn. This difference is reported per-condition as part of the surprise distribution characterization.

## Environment
PhysicsSandbox(N=2) — collision-sparse by design (2 objects in 128 pixels). Mass perturbation at step 1000: multiply object 0's mass by 1.5x. 2000 evaluation steps per seed.

## Seed Bank
12 seeds: [7, 17, 31, 53, 71, 83, 97, 101, 107, 113, 137, 163]
Includes hard seeds 53 and 71 (mandated by user/manager).

## d_t = 3 (FROZEN)
All conditions use d_t=3. For N=2, one channel is unused. This matches the architecture under test (trained at d_t=3) and tests whether the unused channel stays VICReg-clean or collapses. This is a deliberate deviation from the earlier plan (which used d_t=2) to keep the architecture identical to the named config.

## Primary Behavioral Metric
Post-collision attention selectivity (version B): for each collision event, within POST_COLLISION_WINDOW=15 steps after the collision, the fraction of steps where the attended object (token_locus mapped to object index via closest-centroid) matches the max-velocity-change object. This is the same metric as iter_031 Part B, directly calibrating the 0.59-vs-0.44 signal.

## Secondary Metrics (reported but do NOT drive the gate)
- Mean tracking error (pointer position vs attended centroid, in pixels)
- Perturbation selectivity (fraction of steps 1000-1099 where attended object = object 0)

## Ordering Sanity Check (PRE-COMMITTED)
Before computing g, require: ORACLE_primary >= RANDOM_primary (at minimum). If LEARNED beats ORACLE, or RANDOM beats ORACLE, the metric or oracle construction is broken and g is meaningless — report this outcome plainly rather than computing a ratio.

## Branch (c) Threshold (PRE-COMMITTED)
If |ORACLE_primary - RANDOM_primary| < 0.10, the task or motor protocol is the bottleneck, NOT perception. This invalidates the behavioral-pivot strategy for this specific protocol. The report must state this plainly, not reinterpret partial signals.

## Decision Rule (PRE-COMMITTED, VERBATIM)
g = (LEARNED_primary - RANDOM_primary) / (ORACLE_primary - RANDOM_primary)

Computed separately for LEARNED-VICReg (g_vr) and LEARNED-SFA (g_sfa).

(a) g >= 0.70 AND lower bootstrapped 95% CI >= 0.50: representation is consistent with sufficiency for the behavior. Project ADVANCES to Phase 2/3 integration on the mean-pool representation.

(b) g <= 0.20: representation PROVABLY limits behavior. ONLY THEN is constraint-relaxation justified (iter_034+: decoder, higher d_t, or the deferred VICReg-upstream-of-gate fix for iter_032 cross-backbone collapse), with the concrete target set by the measured (ORACLE - RANDOM) gap.

(c) |ORACLE_primary - RANDOM_primary| < 0.10: the TASK or MOTOR PROTOCOL is the bottleneck, NOT perception. Fix the protocol/environment. Do NOT touch the representation. This branch invalidates the behavioral-pivot strategy for this specific protocol.

(d) 0.20 < g < 0.70: partial sufficiency. Advance to Phase 2/3 but flag perception as a known secondary limiter to revisit.

## Reporting Requirements
- Report the raw triple (RANDOM_primary, LEARNED_primary, ORACLE_primary) with CIs (mean plus/minus std across seeds), not only g.
- Report g with bootstrapped 95% CI (10000 resamples of seeds).
- If the lower CI of g does not exclude the boundary between adjacent branches, report BOTH branches as possible.
- Report per-condition surprise distributions (mean, std, histogram of surprise values).
- Report err_coord vs err_dyn decomposition for LEARNED conditions.
- Use restrained language: consistent with sufficiency under this protocol / does not refute the representation's adequacy for the measured behavior — never the representation is sufficient.
- A passed gate at n=12 with one primary metric on one environment is evidence, not proof.

## Preserved Constraints
- Separate backbone + iter_029 configs (Arm A: VICReg-only, Arm B: SFA+VICReg sfa=5.0)
- d_t=3 frozen (GDASR log-only, M3)
- Decoder-free (no reconstruction)
- No positional encoding
- M2 mandate stays untestable until constraint-relaxation phase
- No new representation-side delta_R2 gate alongside the behavioral gate
```

Write this EXACT content to `src/pre_registration.md`, replacing whatever is currently there. Do not modify or add anything.