# RDF Scientific Pre-Registration

*   **Iteration:** 031
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis

This iteration asks a two-sided question with a confirmatory gate structure:

**(a) Ceiling question:** Does Reconstruction+VICReg on the separate-backbone architecture achieve mean ΔR²_color ≥ 0.30 (lower 95% CI ≥ 0.18) on non-collapsed seeds?

**(b) Non-trivial-margin question:** Does it do so with a non-trivial margin over controls — specifically, over both (i) the under-capacity control (d_max=2, trained encoder) and (ii) the random-encoder control (d_max=8, frozen random encoder)?

Without control (b), a "pass" on (a) alone verifies only that the bottleneck has sufficient capacity to preserve color information under a supervised pixel-MSE target — it does **not** establish that training the encoder contributed anything meaningful. The finding would be trivially explained by decoder-only reconstruction from a random or severely capacity-limited encoding.

Therefore, the hypothesis is supported **only if** all four falsification gates (F1–F4, see §2) are passed. If all pass, the result is consistent with the architecture having sufficient capacity to preserve color information under a supervised target (rejecting the null that the M2 failure was objective-specific, not architectural). The language of confirmation is: *"consistent with the architecture having sufficient capacity to preserve color information under a supervised target"* — not *"establishing that the architecture supports identity encoding."*

## 2. Falsification Criteria (Four Gates)

All four gates must be passed for the hypothesis to be supported.

| Gate | Criterion | Rationale |
|------|-----------|-----------|
| **F1** | mean ΔR²_color (Arm A: d_max=8, trained encoder) ≥ **0.30** on non-collapsed seeds | Reconstruction must clear the decoder-free ceiling (best: SFA+VICReg 0.275). If FAILED: reconstruction does not clear even the decoder-free ceiling — the architecture constrains identity encoding regardless of objective class. |
| **F2** | Lower 95% CI of mean ΔR²_color (Arm A) ≥ **0.18** | Variance-stability gate (same as iter_029/030). If FAILED but F1 passed: result is "directionally positive, not robust" — triggers variance investigation, NOT mandate revision. |
| **F3** | mean ΔR²_color (Arm A: d_max=8, trained) — mean ΔR²_color (Arm B: d_max=2, trained) ≥ **0.10** | Capacity-matters gate. If FAILED: even severe under-capacity preserves color via reconstruction gradient — the finding is trivially explained by bottleneck capacity, not meaningful encoder training. |
| **F4** | mean ΔR²_color (Arm A: d_max=8, trained) — mean ΔR²_color (Arm C: d_max=8, random-encoder) ≥ **0.10** | Training-matters gate. If FAILED: random features with a trained decoder suffice to preserve color information — the finding is constructional, not a meaningful training effect on the encoder. |

**Decision rules:**
- **ALL of F1–F4 pass** → Hypothesis supported (see §6.1 for mandate revision).
- **F1 fails** → Hypothesis not supported. Mandate revision per §6.2.
- **F1 passes, F2 fails** → "Directionally positive, not robust." Variance investigation triggered. Mandate revision per §6.3 (not automatic; see §6.2–6.5 for specific failure cases).
- **F3 fails** → Hypothesis not supported. Finding trivially explained by capacity (§6.4).
- **F4 fails** → Hypothesis not supported. Finding constructional (§6.5).

## 3. Proposed Method

## Part A: Reconstruction+VICReg Three-Arm Ceiling Probe (PRIMARY)

### A1: Model Implementation

Create `src/models_recon.py` containing `ReconVICRegSeparateDyn`:
- **Encoder:** `SeparateDynEncoder` (existing separate coord + dyn backbones, unchanged)
- **Decoder:** Deconv head on dyn spatial features a_dyn (B, d_max, 8) → (B, 3, 128)

  Architecture:
  ```
  ConvTranspose1d(d_max, 128, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(128, 64, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(64, 32, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(32, 3, k=5, s=2, p=2, op=1)
  ```
- **Loss:** `recon_weight × MSE(x_recon, x_target) + var_weight × [VICReg_var(z_dyn) + VICReg_var(z_coord)] + cov_weight × [VICReg_cov(z_dyn) + VICReg_cov(z_coord)] + sim_weight × predictor_loss`
- **Predictor:** DualStreamPredictor with stop-gradient on encoder output (surprise readout only)
- All attributes needed for evaluation pipeline: encoder, d_t, d_max, sub_features, color_probe_weight, color_probe_bias, id_contrastive_proj, gdasr_growth_points

### A2: Three Arms

All arms use `ReconVICRegSeparateDyn` with the training configuration in §A3. They differ only in `d_max` and whether the encoder is trained or frozen.

| Arm | d_max | Encoder | recon_weight | Seeds |
|-----|-------|---------|-------------|-------|
| **A (PRIMARY)** | 8 | Trained | 25.0 | 20 |
| **B (under-capacity control)** | 2 | Trained | 25.0 | 20 |
| **C (random-encoder control)** | 8 | **Random (frozen at init, never trained, only decoder trained)** | 25.0 | 20 |

**recon_weight is locked at 25.0** (natural midpoint of exploratory range {10, 25, 50}). No hyperparameter scan is run.

**Arm C details:** The encoder weights are initialized randomly and then **frozen** for the entire training run. Only the decoder, predictor, and (optionally) VICReg projection heads receive gradient updates. This simulates a constructional baseline where random features plus a trained decoder must preserve color information.

### A3: Training Configuration

- **Seeds (union bank, identical to iter_029/030):**
  - 10 original: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
  - 10 fresh: [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]
  - Total: 20 seeds per arm
- **Steps:** 8000
- **Batch size:** 32
- **Learning rate:** 3e-4
- **d_t:** 3 (frozen)
- **pos_encoding:** "none"
- **coord_vicreg:** True
- **var_weight:** 25.0
- **cov_weight:** 25.0
- **sim_weight:** 1.0
- **recon_weight:** 25.0 (locked, see §A2)
- **GDASR:** log-only mode (no recruitment)

### A4: Evaluation (identical pipeline to iter_029/030)

All metrics computed with **95% CI across 20 seeds** per arm:

- **ΔR²_color (primary):** linear probe from z_dyn to object color, channel-object matched
- **Centroid MSE:** soft-argmax centroid decoding
- **Collapse rate:** per-dim std < 0.5 threshold
- **VICReg health:** per-dim std, mean absolute cross-correlation
- **Reconstruction MSE**
- **Non-collapsed seeds:** seeds where per-dim std ≥ 0.5 for all d_max dimensions are included in primary analysis; collapsed seeds are reported separately

### Comparison Baselines (from iter_029, no re-run)

- VICReg-only: ΔR²_color ≈ 0.045 (20 seeds)
- SFA+VICReg sfa=5.0: ΔR²_color ≈ 0.275 (20 seeds)
- Random-encoder baseline (Arm C itself serves as the within-experiment control)

## Part B: Protocol Calibration (PRE-REGISTERED — required preamble before any protocol experiments)

### B1: N=2 Collision-Sparse CLTS Evaluation

**Purpose:** Calibrate the surprise-driven CLTS readout in a simpler environment before deploying in the full protocol. Pre-register gate formulations before running.

**Checkpoint:** Use existing VICReg-only checkpoint from iter_029 (best tracking: ~36.22 px).

**Environment:** PhysicsSandbox(N=2) — fewer collisions than N=3, enabling cleaner attribution.

**Procedure per seed (5 seeds):**
Three conditions run per seed, each for 2000 evaluation steps:
1. **Surprise-driven:** Standard surrogate-readout attention (M3 mechanism)
2. **Frozen (locus=0):** Attention locus clamped to channel 0 throughout
3. **Random:** Attention drawn uniformly at random at each step

**Metrics measured per condition:**
- CLTS tracking error (px between attended channel centroid and ground-truth target object)
- Collision count per 100 steps
- Post-collision attention selectivity: fraction of steps in [t, t+10] after a collision (within a ±20-step window) where the colliding channel is attended
- Post-perturbation attention selectivity: fraction of steps in [t, t+10] after a perturbation (within a ±20-step window) where the changed object's channel is attended

**Pre-registered gate formulation (uses MEASURED random baselines, not assumed values):**

| Gate | Criterion | Rationale |
|------|-----------|-----------|
| **G1_tracking** | CLTS tracking error (surprise-driven) ≤ random_tracking_mean − **1 × random_tracking_std** | Surprise-driven tracking must exceed random baseline by at least 1σ. Uses measured mean and std of the random condition across steps and seeds. |
| **G2_collision_selectivity** | Fraction of post-collision steps where colliding channel is attended (surprise-driven) ≥ **random_selectivity × 1.5** | Surprise-driven collision response must be at least 1.5× the random baseline selectivity. |
| **G3_perturbation_selectivity** | Fraction of post-perturbation steps where changed object's channel is attended (surprise-driven) ≥ **random_selectivity × 1.5** | Surprise-driven perturbation response must be at least 1.5× the random baseline selectivity. |

**Reporting:** Measured random baselines (mean ± std across seeds and steps) must be reported explicitly alongside surprise-driven values.

### B2: Subtle Mass Perturbation Test

**Environment:** Same N=2 environment as B1.

**Procedure:** At step 1000, the mass of object 0 changes by **1.5×** (not 10× as in prior perturbation tests — this is a subtle, ecologically plausible change).

**Metric:** Perturbation attention selectivity — fraction of steps in [t, t+50] after perturbation where the changed object's channel is attended, for surprise-driven vs. random baselines.

**Gate:** Same as G3 above: surprise-driven selectivity ≥ random_selectivity × 1.5.

## 4. Mandate Revision (PRE-COMMITTED)

The following mandate revisions are pre-registered and will be executed based on the falsification gate outcomes, without additional deliberation.

### 6.1 If ALL of F1–F4 pass (hypothesis supported)

> "Reconstruction+VICReg on the separate-backbone architecture achieves mean ΔR²_color ≥ 0.30 with robust variance. The result is consistent with the architecture having sufficient capacity to preserve color information under a supervised pixel-MSE target. The d_max=2 control (F3) and random-encoder control (F4) confirm that the finding is non-trivial: both capacity and training are required.
>
> **M2 is revised** from 'SFA+VICReg as primary' to 'Reconstruction+VICReg as primary representation objective, decoder-free constraint relaxed as pragmatic compromise. SFA demoted to comparison baseline B1. Surprise readout retained via stop-gradient predictor. Future work may explore BYOL-style decoder-free alternatives approaching the reconstruction ceiling.'"

### 6.2 If F1 or F2 fails (ceiling not cleared or variance-unstable)

> "Reconstruction+VICReg fails to achieve ΔR²_color ≥ 0.30 with variance-stability. Even a supervised pixel-reconstruction target cannot make the mean-readout z_dyn stream encode identity above the 0.30 threshold. The z_dyn readout architecture itself constrains identity encoding regardless of objective class.
>
> **M2 revision pending architectural redesign:** priority is centroid-gated z_dyn readout or increased d_max."

### 6.3 If F3 fails (capacity does not matter — d_max=2 within 0.10 of d_max=8)

> "d_max=2 under-capacity control achieves ΔR²_color within 0.10 of d_max=8. The finding is trivially explained by bottleneck capacity — even severe under-capacity preserves color via reconstruction gradient. The result does not support a meaningful training effect."

### 6.4 If F4 fails (training does not matter — random encoder within 0.10 of trained)

> "Random-encoder control achieves ΔR²_color within 0.10 of trained-encoder. The finding is constructional — random features with a trained decoder are sufficient to preserve color information. The result does not support a meaningful training effect on the encoder."

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*