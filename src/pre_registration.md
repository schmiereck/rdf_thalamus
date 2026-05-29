# RDF Scientific Pre-Registration

*   **Iteration:** 025 v2 (corrected re-run)
*   **Pre-Registration File:** src/pre_registration.md
*   **Revisions:** Corrected for five critical methodological flaws identified by the Research Manager after iter_025:
    1. Collapse fix: lr 3e-4 + gradient clipping + 8000 steps
    2. Defensible threshold: pre-declared effect size 0.10 (no noise floor)
    3. Matching resolved: Hungarian is sole primary scheme; sorted is secondary check
    4. Adequate power: 10 seeds (target ≥5 non-collapsed per arm)
    5. Capacity audit: Arm E (d_max=16 control) tests iter_023 claim without identity objective

## 1. Hypothesis
The failure of identity encoding in z_dyn (delta_R2_color < 0.10 across iter_021-024)
is attributable to the objective being insufficiently discriminative, NOT to the
architecture being incapable. Specifically:

**H1 (Architecture Capacity):** The shared-CNN dual-stream NonParametricJEPASpatial
encoder CAN encode object identity in z_dyn when provided with a direct supervised
color regression loss that backpropagates through the encoder. Under this condition,
delta_R2_color ≥ 0.10 (mean over non-collapsed seeds, Hungarian matching) with
collapse rate ≤ 2/10.

**H2 (ID-Contrastive Viability):** A color-similarity-based contrastive objective
(using privileged environment slot IDs to define positive/negative pairs) is
sufficient as a self-supervised proxy for identity encoding, achieving delta_R2_color
≥ 0.10 (mean over non-collapsed seeds, Hungarian matching) with collapse rate ≤ 2/10.

The full hypothesis (H1 AND H2) is falsified if H1 fails (architecture ceiling
reached). Partial falsification (H1 holds, H2 fails) means the architecture CAN
encode but the contrastive formulation is insufficient — a less severe outcome.

## 2. Falsification Criterion

### PRIMARY FALSIFICATION (H1 — Architecture Ceiling):
Arm B (Supervised Color Probe + VICReg, d_max=8) fails to achieve
delta_R2_color ≥ 0.10 (mean over non-collapsed seeds, Hungarian matching)
OR collapse rate > 2/10, on the seed set [7, 17, 31, 53, 71, 83, 97, 113, 127, 149].

If sorted and Hungarian matching disagree on pass/fail for >25% of non-collapsed
seeds in Arm B, the outcome is **matching-dependent** and the falsification claim
is NOT earned.

If H1 is falsified: the result is **consistent with an architecture-level bottleneck
on identity encoding, conditional on the matching disagreement rate** — **NOT**
"the architecture cannot encode identity."

Language: "consistent with an architecture-level bottleneck on identity encoding,
conditional on [mismatch rate]%" — **NOT** "architecture cannot encode identity."

### SECONDARY FALSIFICATION (H2 — ID-Contrastive):
Arm C (ID-Contrastive + VICReg) fails to achieve delta_R2_color ≥ 0.10 (mean over
non-collapsed seeds, Hungarian matching) with collapse rate ≤ 2/10, while Arm B
succeeds.

If sorted and Hungarian matching disagree on pass/fail for >25% of non-collapsed
seeds in Arm C, the outcome is **matching-dependent** and the falsification claim
is NOT earned.

If H2 is falsified but H1 holds: the architecture CAN encode, but the contrastive
formulation is insufficient. Try direct supervised as training objective or stronger
contrastive variants.

Language: "contrastive formulation insufficient; architecture not the bottleneck."

### EFFECT-SIZE THRESHOLD:
**Threshold = 0.10** (pre-declared effect size). This means "the improvement in
color predictability from z_dyn over z_coord is at least 10% of variance" — a
meaningful effect size, NOT derived from a flawed noise floor.

### ARM A POWER CHECK:
If Arm A (JEPA control) collapse rate > 2/10, the experiment is **UNDERPOWERED**
and NO architecture claim is valid — report this honestly and suspend comparative
analysis until collapse is fixed.

### CAPACITY AUDIT (Arm E):
Arm E (JEPA+VICReg Control, d_max=16, NO supervised/contrastive objective) tests
whether the d_max=16 improvement observed in iter_023 is a capacity effect.
- If Arm E achieves delta_R2_color ≥ 0.10, the d_max=16 improvement is confirmed
  as a capacity effect (occurs without identity objective).
- If Arm E does NOT achieve delta_R2_color ≥ 0.10, the iter_023 claim needs revision.

### FOUR OUTCOME QUADRANTS (pre-declared next moves):

| Arm B | Arm C | Interpretation | Next Move |
|-------|-------|----------------|-----------|
| ✓ | ✓ | Architecture CAN encode identity; ID-contrastive viable | Continue developing ID-contrastive |
| ✓ | ✗ | Architecture CAN encode; contrastive formulation insufficient | Try direct supervised objective |
| ✗ | ✓ | Check implementation (supervised ≥ contrastive expected) | Debug matching / contrastive impl |
| ✗ | ✗ | Consistent with architecture-level bottleneck (conditional on mismatch rate%) | Next iter: separate z_dyn encoder |

Any positive Arm B result is stated as: **"is compatible with sufficient
architectural capacity under direct supervision"** — **NOT** "demonstrates the
architecture can encode identity."

Any positive Arm C result is qualified as: **"supervised (slot IDs are privileged
information), not evidence that the decoder-free self-supervised problem is solved."**

## 3. Proposed Method

### EXPERIMENT DESIGN:
- **Main experiment:** 5 arms × 10 seeds × 8000 steps = 50 runs
- No noise floor runs (the previous floor was invalid because R² on random
  encoders can exceed 1, making floor-derived thresholds unsupportable).

Seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149] — disjoint from iter_021-024.

### ARM CONFIGURATIONS:

**Arm A: JEPA+VICReg Control (d_max=8)** — 10 seeds
  - primary_objective="jepa", var_weight=25, cov_weight=25, sim_weight=25
  - d_max=8, d_t=3, dyn_readout="centroid_gated", pos_encoding="none"
  - CCR covariance mode (ccr_smooth_weight=10, ccr_spatial_weight=10)
  - gdasr_log_only=True
  - NO supervised/contrastive loss
  - Learning rate: 3e-4, gradient clipping: max_norm=1.0
  - Provides baseline incidental identity encoding and **power check**.

**Arm B: Supervised Color Probe + VICReg (d_max=8)** — 10 seeds [CRITICAL DIAGNOSTIC]
  - Same as Arm A PLUS supervised_color_loss with supervised_weight=25.0
  - Training matching: **Hungarian** (not sorted)
  - Ramp supervised weight 0.1 → 25.0 over first 500 steps
  - Learning rate: 3e-4, gradient clipping: max_norm=1.0

**Arm C: ID-Contrastive + VICReg (d_max=8)** — 10 seeds
  - Same as Arm A PLUS id_contrastive_loss with contrastive_weight=25.0
  - Training matching: **Hungarian** (not sorted)
  - Learning rate: 3e-4, gradient clipping: max_norm=1.0

**Arm D: Supervised Color Probe + VICReg (d_max=16)** — 10 seeds
  - Same as Arm B but d_max=16
  - Tests whether more channels help supervised encoding

**Arm E: JEPA+VICReg Control (d_max=16)** — 10 seeds [NEW — capacity audit]
  - Same as Arm A but d_max=16
  - NO supervised/contrastive loss
  - If this achieves delta_R2_color ≥ 0.10, the d_max=16 improvement is confirmed
    as a capacity effect (not objective effect).

### FILES TO CREATE/MODIFY:

1. src/run_phase0_id_probe_v2.py (NEW):
   - Main experiment runner, based on run_phase0_id_probe.py structure
   - Extended ReplayBuffer: stores (x_hist, x_target, positions, colors, radii)
   - 5 arms × 10 seeds × 8000 steps
   - For Arms B, D: compute supervised_color_loss with **Hungarian matching** during
     training; sorted is computed only for logging mismatch rate.
   - For Arm C: compute id_contrastive_loss with **Hungarian matching** during
     training; sorted computed for logging mismatch rate.
   - Gradient clipping before optimizer.step()
   - Per-dimension std logging every 500 steps to diagnose WHEN collapse happens
   - Checkpoint evaluations at steps 2000, 4000, 6000, 8000
   - Same evaluation suite as iter_025: semantic probes, collapse checks,
     centroid MSE, tracking quality, normalized temporal variance,
     within/between trajectory variance, shuffled-frame control
   - PRIMARY evaluation: Hungarian matching. SECONDARY: sorted matching.
   - Report mismatch rate and scheme agreement on pass/fail.
   - Results saved to archive/iter_025/results_v2/

2. src/pre_registration.md: Updated with this corrected v2 plan.

### TRAINING PROTOCOL:
- Main runs: 8000 steps, Adam lr=3e-4, batch_size=32, replay_buffer=2000
- Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
- d_t=3 frozen, gdasr_log_only=True (M3 preserved)
- VICReg: var_weight=25, cov_weight=25, sim_weight=25 (batch-level, M1 preserved)
- All arms: centroid_gated dyn_readout, CCR covariance mode
- supervised_weight=25.0 (ramped 0.1→25.0 over first 500 steps)
- contrastive_weight=25.0 for Arm C
- Color probe head initialized with small random weights (std=0.01)

### EVALUATION PROTOCOL:
- Same as iter_025: semantic probes, collapse check (per_dim_std < 0.5),
  centroid MSE, tracking quality, normalized temporal variance,
  within/between trajectory variance, shuffled-frame control
- PRIMARY METRIC: delta_R2_color (frozen-encoder linear probe, Hungarian matching)
- ADDITIONAL: report sorted-matching delta_R2_color as secondary check
- Checkpoint evaluations at steps 2000, 4000, 6000, 8000
- Compute and report:
  - delta_R2_color under Hungarian (primary) and sorted (secondary)
  - Eval mismatch rate (fraction of dims with different assignments)
  - Whether pass/fail outcome differs between schemes per seed

### CONSTRAINTS (corrected from iter_025):
- Seed set [7, 17, 31, 53, 71, 83, 97, 113, 127, 149], disjoint from iter_021-024
- No separate encoder for z_dyn in this iteration (resist scope creep)
- **Matching scheme:** Hungarian is the SOLE primary matching scheme for both
  training and evaluation. Sorted is reported as a secondary stability check.
  If sorted and Hungarian disagree on pass/fail for >25% of non-collapsed seeds
  in Arm B or C, report "matching-dependent outcome; falsification claim not earned."
- **No noise floor:** The previous noise floor was methodologically invalid
  (R² on random encoders can exceed 1). Threshold is pre-declared at 0.10 as an
  effect-size criterion.
- **Arm A power check:** If collapse rate > 2/10, the experiment is underpowered.
  Report this honestly and do NOT claim architecture-level bottlenecks.
- Language hygiene:
  - Positive Arm B: "compatible with sufficient architectural capacity under direct supervision"
  - Positive Arm C: "supervised (slot IDs are privileged), not evidence decoder-free self-supervision solved"
  - Negative result: "consistent with an architecture-level bottleneck on identity encoding, conditional on [mismatch rate]%"

---
*Corrected v2 pre-registration addressing all five Research Manager criticisms.*
*Date: Auto-generated.*
