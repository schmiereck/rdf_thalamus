# Phase 0 Recovery: Comparison Report — Original vs. Recovery

## Executive Summary

| Metric | Original Arm A | Recovery A1 | Recovery A2 | Recovery B | Recovery C |
|--------|---------------|-------------|-------------|------------|------------|
| Collapsed seeds | **2/5** | 1/5 | 1/5 | 4/5 | **0/5** |
| Centroid MSE | 121.37 ± 59.35 | 121.86 ± 33.82 | 119.66 ± 43.80 | 113.70 ± 35.13 | 115.87 ± 43.62 |
| Slowness ratio | **805.2** ± 1296.8 | 338.0 ± 345.9 | 485.0 ± 658.0 | 0.63 ± 0.77 | 1643.6 ± 2866.0 |
| Delta R² (color) | **−0.087** ± 0.18 | −0.074 ± 0.26 | −0.089 ± 0.29 | −0.154 ± 0.20 | −0.004 ± 0.20 |
| Avg per-dim std | 0.68 ± 0.48 | 0.81 ± 0.39 | 0.83 ± 0.37 | 0.58 ± 0.36 | 0.90 ± 0.09 |

**Overall Recovery Verdict: FALSIFIED** (C3 semantic disentanglement criterion still fails)

---

## 1. What Was Changed (Recovery Protocol)

| Change | Rationale |
|--------|-----------|
| **Bug fix**: `.detach()` on `z_prev_dyn_active` in SFA loss | Prevents doubled gradient signal through both `z_target` and `z_prev` |
| **5000 steps** (was 3000) | More training time for representation to stabilize |
| **CCR covariance mode** (was `none`) | Imposes spatial decorrelation + smoothness on z_coord to prevent collapse |
| **sfa_weight = 0.1** (A1, C) | Weaker slowness regularization to avoid over-constraining z_dyn |
| **Seeds**: [42, 123, 456, 789, 999] | Same as original for direct comparison |

---

## 2. Collapse Rate Analysis (C1)

**Original Phase 0:**
- Arm A (SFA w=1.0): 2/5 seeds collapsed (seeds 42, 789)
- Arm B (JEPA): 4/5 seeds collapsed (seeds 42, 123, 789, 999)
- Arm C (SFA+pos_enc): 1/5 seeds collapsed (seed 789)

**Recovery Phase 0 @ step 5000:**
- Arm A1 (SFA w=0.1): 1/5 seeds collapsed (seed 42) — **PASS C1** (< 2 collapsed)
- Arm A2 (SFA w=1.0): 1/5 seeds collapsed (seed 42) — **PASS C1** (< 2 collapsed)
- Arm B (JEPA+CCR): 4/5 seeds collapsed (seeds 42, 123, 789, 999) — **unchanged from original**
- Arm C (SFA w=0.1+pos): 0/5 seeds collapsed — **perfect**

**Interpretation:**
- The `.detach()` fix alone appears to help: A2 (with weight still at 1.0) drops from 2/5 to 1/5 collapsed.
- The combination with lower sfa_weight (A1) yields the same improvement.
- Positional encoding + low sfa_weight (Arm C) yields **zero collapse**, suggesting the architecture benefits from positional anchors when slowness is weak.
- The JEPA baseline remains heavily collapsed even with CCR and extended training, indicating this is a deeper issue with the JEPA baseline under the current training protocol, not specific to SFA.

---

## 3. Centroid Decoding MSE (C2)

| Arm | Mean | Std | vs. JEPA threshold (1.10×B) |
|-----|------|-----|---------------------------|
| A1 (SFA w=0.1) | 121.86 | 33.82 | **PASS** (threshold = 125.07) |
| A2 (SFA w=1.0) | 119.66 | 43.80 | **PASS** |
| C (SFA+pos) | 115.87 | 43.62 | **PASS** |
| B (JEPA+CCR) | 113.70 | 35.13 | baseline |

All SFA recovery arms meet the C2 criterion (MSE within 10% of JEPA baseline).

---

## 4. Semantic Disentanglement (C3) — The Persistent Failure

The key metric is **`delta_R2_color = R²_dyn_color − R²_coord_color`**.
A positive value means z_dyn predicts color better than z_coord, indicating the architecture separates identity from position.

| Arm | Delta R² Color | Result |
|-----|----------------|--------|
| Original A (SFA w=1.0) | **−0.087** | FAIL |
| Recovery A1 (SFA w=0.1) | **−0.074** | **FAIL** |
| Recovery A2 (SFA w=1.0) | **−0.089** | **FAIL** |
| Recovery C (SFA+pos) | **−0.004** | **FAIL** (closest to 0) |
| Recovery B (JEPA+CCR) | −0.155 | FAIL |

**None of the recovery arms achieve delta_R2_color ≥ 0.10. In fact, all values remain negative**, meaning z_coord still predicts color at least as well as z_dyn in every arm.

### Per-dimension probe details (Arm A1, step 5000):

| Seed | R²_dyn_color | R²_coord_color | Delta |
|------|--------------|----------------|-------|
| 42 | −0.012 | −0.188 | +0.176 |
| 123 | −0.060 | −0.030 | −0.030 |
| 456 | −0.332 | −0.424 | +0.092 |
| 789 | −0.729 | −0.227 | **−0.502** |
| 999 | −0.173 | −0.069 | −0.104 |

Seed 789 is an outlier where z_dyn color prediction is very poor, dragging the mean down. Even the best seed does not exceed +0.20.

---

## 5. Slowness Sanity Check

The slowness ratio `mean_dyn_delta / mean_coord_delta` was originally used as a sanity check (expect ratio < 0.6 for SFA).

**Critical insight:** In uncollapsed SFA seeds, the ratio is >> 1 (e.g., 780, 625, 212). This happens because z_dyn is changing MORE than z_coord — the **opposite** of what slowness should achieve. In collapsed seeds, z_dyn is constant (ratio ≈ 0), but that's trivial and uninformative.

| Arm | Ratio | Interpretation |
|-----|-------|----------------|
| Original A | 805 | z_dyn collapsed or non-functional; ratio meaningless |
| Recovery A1 | 338 | Same pathology: z_dyn changes MORE than z_coord |
| Recovery A2 | 485 | Same |
| Recovery C | 1644 | Same, but with 0 collapse |
| Recovery B | 0.63 | JEPA: z_dyn is relatively stable; but 80% are collapsed |

**Conclusion:** The slowness ratio is not a meaningful sanity check here. High ratios indicate z_dyn is NOT slow — the SFA objective is failing to stabilize z_dyn relative to z_coord. However, note that Arm C (0 collapse, ratio 1644) shows that high ratio and non-collapse CAN co-occur: z_dyn is actively varying but not collapsed.

---

## 6. VICReg Health (per-dim std)

| Arm | Avg per-dim std | Interpretation |
|-----|-----------------|----------------|
| Original A | 0.68 | Mixed: collapsed seeds have std < 0.02 |
| Recovery A1 | 0.81 | Improved: fewer collapsed dimensions |
| Recovery A2 | 0.83 | Improved similarly |
| Recovery B | 0.58 | Still collapsed heavily |
| Recovery C | **0.90** | Best representation spread; no collapse |

Arm C (SFA w=0.1 + sinusoidal position encoding) has the healthiest per-dim std and zero collapse. The positional encoding provides an architectural anchor that prevents z_coord collapse even under weak SFA regularization.

---

## 7. Checkpoint Comparison (step 2500 vs 5000)

For Arm A1, comparing checkpoints:

| Metric | @ 2500 | @ 5000 | Trend |
|--------|--------|--------|-------|
| Collapsed seeds | 2/5 | 1/5 | Improved |
| Avg per-dim std | 0.72 | 0.81 | Improved |
| Centroid MSE | 100.12 | 121.86 | Slightly worse (trade-off) |
| Delta R² color | −0.096 | −0.074 | Improved toward 0 |

Extended training helps representations stabilize without additional collapse, and semantic separation slightly improves.

---

## 8. Falsification Audit Summary

### Arm A1 (SFA w=0.1) — Primary Recovery Arm
| Criterion | Threshold | Value | Result |
|-------------|-----------|-------|--------|
| C1 (Collapse) | < 2/5 | 1/5 | **PASS** |
| C2 (MSE) | ≤ 1.10 × JEPA | 121.86 ≤ 125.07 | **PASS** |
| C3 (Semantic ΔR²) | ≥ 0.10 | −0.074 | **FAIL** |

### Arm A2 (SFA w=1.0 fixed) — Detach-fix-only control
| Criterion | Threshold | Value | Result |
|-------------|-----------|-------|--------|
| C1 (Collapse) | < 2/5 | 1/5 | **PASS** |
| C2 (MSE) | ≤ 1.10 × JEPA | 119.66 ≤ 125.07 | **PASS** |
| C3 (Semantic ΔR²) | ≥ 0.10 | −0.089 | **FAIL** |

### Arm C (SFA w=0.1+pos) — Architecture variant
| Criterion | Threshold | Value | Result |
|-------------|-----------|-------|--------|
| C1 (Collapse) | < 2/5 | 0/5 | **PASS** |
| C2 (MSE) | ≤ 1.10 × JEPA | 115.87 ≤ 125.07 | **PASS** |
| C3 (Semantic ΔR²) | ≥ 0.10 | −0.004 | **FAIL** (closest to target) |

---

## 9. Root-Cause Assessment: Why C3 Still Fails

The recovery fixes **did improve collapse rates** (C1) and preserved centroid decoding (C2), but **semantic disentanglement remains elusive**.

### Hypothesis 1: Architecture bottleneck
`z_dyn` in `NonParametricEncoder` is computed as `a_spatial.mean(dim=-1)`. This is a **spatial mean** over 128 positions. It does NOT naturally encode object identity/color — it encodes the total "activation mass" of each channel. Channels may localize to objects, but their mean is functionally a proxy for spatial extent, not color identity.

### Hypothesis 2: SFA targets the wrong signal
The SFA loss minimizes `||z_dyn(t) − z_dyn(t−1)||²`. But in the PhysicsSandbox environment:
- Colors are static object properties that change only on reset (every ~200 steps), NOT between consecutive frames.
- z_prev and z_target are from consecutive frames within a trajectory.

Since colors don't change between consecutive frames, a trivial constant z_dyn would have zero SFA loss and satisfy the objective — which is exactly what collapse achieves. When SFA doesn't collapse, z_dyn may encode something else (noise, residual variance, position) rather than color.

### Hypothesis 3: VICReg variance/covariance dominate
The VICReg variance and covariance losses (both weighted at 25.0) force variance expansion and decorrelation. This opposes collapse but also distracts z_dyn from learning color-specific features, since color is a discrete/static signal and VICReg optimizes for batch-wise variance.

---

## 10. Recommendations for Next Iteration

1. **Implement multi-frame SFA**: Instead of comparing t vs t−1, compare z_dyn at the SAME object across trajectories with DIFFERENT colors. This creates a cross-trajectory slowness signal that color must represent.

2. **Add explicit color bottleneck**: Introduce a small projection layer in z_dyn path that is trained with a cross-entropy or contrastive objective on color labels, either supervised or via a temporal contrast (same object, different colors across resets).

3. **Reduce VICReg weight on z_dyn stream**: When primary_objective='sfa', consider lowering var_weight/cov_weight on z_dyn so SFA isn't fighting an expansion objective.

4. **Try SwAV or contrastive approach**: Rather than pure SFA, use a contrastive loss where augmentations (color jitters if available, or synthetic color swaps) create positive/negative pairs for z_dyn.

5. **Consider architecture change**: Replace the spatial mean with a slot-attention or object-binding mechanism so that individual z_dyn dimensions can bind to individual object identities rather than being a spatial average.

---

## Data Files

All recovery results are saved under `archive/iter_020/results/`:
- `summary_phase0_recovery.csv` — full per-run metrics @ step 5000
- `summary_phase0_recovery_cp2500.csv` — checkpoint metrics @ step 2500
- `aggregated_phase0_recovery.csv` — per-arm mean ± std
- `audit_phase0_recovery.json` — machine-readable audit object
- `runs/` — per-run CSVs, JSONs, and training logs
- `checkpoints/` — model state dicts (.pt)

## Fixes Applied to Codebase

- **`src/models_dual_stream.py`**: Added `.detach()` to `z_prev_dyn_active` in SFA loss (line ~741):
  ```python
  sfa_loss = F.mse_loss(z_target_dyn_active, z_prev_dyn_active.detach())
  ```
- **`src/run_phase0_sfa_recovery.py`**: New recovery runner implementing extended training (5000 steps), CCR covariance mode, 4 recovery arms, and checkpoint evaluation.

---

*Report generated after Phase 0 Recovery run (4 arms × 5 seeds, 5000 steps each, with checkpoint @ 2500).*
