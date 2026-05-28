# Research Manager Log - Iteration 020

## Iteration 020 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
On the existing NonParametricJEPASpatial CNN encoder with frozen d_t=3 and no dynamic
recruitment, replacing the JEPA prediction objective with SFA (slowness) on z_dyn as the
primary representation objective, combined with batch VICReg on z_dyn, will:
(1) Train without representation collapse (all per-dimension std >= 0.5, no has_collapsed trigger),
(2) Achieve centroid-decoding MSE via Arm F (soft-argmax on z_coord) at most 10% higher
    than the JEPA+VICReg baseline (i.e., MSE_SFA <= 1.10 * MSE_JEPA, where MSE_JEPA ~ 55.6),
(3) Produce z_dyn representations that are significantly slower (mean ||z_dyn(t) - z_dyn(t-1)||^2
    at least 40% lower) than z_coord temporal variation, indicating that SFA separates slow
    identity from fast position by construction.

**Proposed Falsification Criterion:**
The hypothesis is falsified if ANY of the following hold across the 5-seed sweep:
C1 (Collapse): SFA+VICReg arm collapses (has_collapsed=True OR any active dimension std < 0.5)
    in >= 2 out of 5 seeds.
C2 (Centroid MSE): Mean centroid-decoding MSE of the SFA arm exceeds 1.10 * mean MSE of the
    JEPA baseline arm (i.e., SFA degrades spatial readout by more than 10%).
C3 (Slowness separation): The ratio mean(||z_dyn(t)-z_dyn(t-1)||^2) / mean(||z_coord(t)-z_coord(t-1)||^2)
    is >= 0.6 for the SFA arm (i.e., z_dyn is not substantially slower than z_coord,
    failing the identity-position separation hypothesis).

**Proposed Method:**
Step 1: Create SFA training infrastructure.
- MODIFY src/models_dual_stream.py: Add an SFA loss computation to NonParametricJEPASpatial.forward()
  that computes L_sfa = ||z_dyn_target - z_dyn_prev||^2 over the batch, where z_dyn_prev is
  z_dyn from the previous frame in the history. Also add batch VICReg on z_dyn_target (the 
  target frame's dynamics representation). The total representation loss becomes:
  L_repr = sfa_weight * L_sfa + var_weight * VICReg_var(z_dyn) + cov_weight * VICReg_cov(z_dyn)
  The JEPA sim_loss is retained ONLY as a readout/surprise signal (stop-gradient detached from
  the representation), not as a training objective for the encoder.
- Keep the predictor's forward pass for surprise computation, but detach its output from the
  encoder's gradient path when SFA is the primary objective.

Step 2: Create the Phase 0 experiment runner.
- CREATE src/run_phase0_sfa.py: A 5-seed sweep comparing three arms:
  Arm A (SFA+VICReg): Primary objective = SFA on z_dyn + batch VICReg on z_dyn.
    - sfa_weight=1.0, var_weight=25.0, cov_weight=25.0
    - Predictor is retained for surprise readout but gradients do NOT flow to encoder from sim_loss.
    - z_coord is NOT slowed (M2 mandate).
    - d_t frozen at 3, no recruitment.
  Arm B (JEPA+VICReg Baseline, B1): Same as current NonParametricJEPASpatial with
    sim_weight=25.0, var_weight=25.0, cov_weight=25.0. d_t frozen at 3, no recruitment.
    This is the direct comparison baseline.
  Arm C (SFA+VICReg with pos_encoding="sinusoidal"): Same as Arm A but with sinusoidal
    position encoding added to the input. Tests whether explicit position channels
    narrow the Spatial Specialization Gap under SFA.

Step 3: Training protocol (shared across arms).
- Environment: PhysicsSandbox(N=3, seed=seed) for 3000 training steps, batch_size=32.
- History buffer: deque(maxlen=4) for H=3 history + 1 target.
- Replay buffer: capacity 2000, prefill 100 transitions.
- Optimizer: Adam, lr=1e-3.
- d_t = 3 frozen from start (no recruitment logic).
- GDASR runs in log-only mode (computes and logs growth-point signals without modifying d_t).
- Seeds: [42, 123, 456, 789, 999].

Step 4: Evaluation protocol (at step 3000 and checkpoint at 1500).
- Non-collapse check: has_collapsed criterion + per-dimension std >= 0.5.
- Centroid decoding MSE: linear probe on soft-argmax centroids vs. true object positions
  (all 3 objects, report mean and per-object).
- Slowness metrics: mean ||z_dyn(t) - z_dyn(t-1)||^2 and mean ||z_coord(t) - z_coord(t-1)||^2
  over 200 test frames, and their ratio.
- VICReg health: per-dimension std, mean absolute correlation between active dimensions.
- GDASR growth-point log: count and timing of would-have-recruited events.

Step 5: Statistical analysis.
- Welch's t-test comparing SFA arm vs JEPA arm on centroid MSE.
- Welch's t-test comparing SFA arm z_dyn slowness ratio vs 0.6 threshold.
- Cohen's d for effect sizes.
- Report with confidence intervals per reporting standards (Section 9).

Step 6: Falsification audit against C1, C2, C3.
- C1: Count collapsed seeds per arm.
- C2: Compare mean MSE_SFA vs 1.10 * mean MSE_JEPA.
- C3: Compute slowness ratio for SFA arm; check >= 0.6.

Files to create/modify:
- MODIFY: src/models_dual_stream.py (add SFA objective option to NonParametricJEPASpatial.forward)
- CREATE: src/run_phase0_sfa.py (experiment runner with 3 arms × 5 seeds)
- UPDATE: src/pre_registration.md (with Phase 0 pre-registration)

---

## Iteration 020 -> Planner [Strategic Guidance]

# Manager's Note – Phase 0 Objective Migration

## 1. Construction-vs-Empirical Red Flag on Claim (3) [CRITICAL]

The hypothesis claim (3) and falsification criterion C3 suffer a **constructional confound**. You imposed an explicit slowness penalty on `z_dyn` and none on `z_coord`. The optimizer minimizing what you asked it to minimize — yielding `||Δz_dyn||² / ||Δz_coord||² < 0.6` — is a **verification of the gradient path**, not empirical evidence that identity and position are semantically disentangled. The ratio tells us "the loss term worked," not "the representation factorized meaningfully."

**Required action:**
- **Reframe claim (3) as a sanity check** (verify the SFA objective is active and VICReg prevents collapse to a constant). State this explicitly in the pre-registration.
- **Add a genuinely empirical disentanglement metric:** linear probes predicting object identity (color/size class) from `z_dyn` vs. `z_coord`, and object centroid position from `z_coord` vs. `z_dyn`. If SFA creates semantic separation, `z_dyn` should predict identity well and position poorly, and vice versa for `z_coord`. *This* is the test of the Spatial-Gap hypothesis — not the slowness ratio.
- **Replace or augment C3** with a criterion on the semantic probes, e.g.: "C3 is falsified if the linear-probe accuracy for object identity from `z_dyn` does not exceed that from `z_coord` by at least 10 percentage points."

## 2. Pre-Registration and Parameter Hygiene

The Orchestrator will automatically write `src/pre_registration.md` from this plan. Sub-agents **must read and strictly adhere** to it during execution.

The initial weight `sfa_weight=1.0` is a declared starting point. If the first run shows SFA is too weak relative to VICReg (slowness ratio near 1.0) or too strong (collapse toward constant despite VICReg), adjusting it is legitimate — but **document the adjustment and its rationale** in the run log. Do not silently tune after viewing results and report as if the final value was the plan all along.

## 3. The Centroid MSE Bar Is Asymmetric

C2 tests that SFA doesn't *degrade* by more than 10% vs. JEPA. This is the right gate for a bridge phase. But per the sml findings (SFA+VICReg 82% vs. JEPA+VICReg 61%), SFA may *improve* the representation. Report both directions honestly: if MSE_SFA < MSE_JEPA, that is compatible with the sml transfer and should be noted; if MSE_SFA > MSE_JEPA but within 10%, that is a successful bridge but a signal that RGB physics may differ from binary sml in ways worth characterizing. Do not frame a non-improvement as a failure if it passes C2.

---

