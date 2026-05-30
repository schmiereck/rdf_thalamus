Write the pre-registration file at src/pre_registration.md for iter_032. This must incorporate three Manager corrections (C1, C2, C3) to the previously approved plan. Overwrite the existing file with the following content:

```markdown
# RDF Scientific Pre-Registration — Iter 032

*   **Iteration:** 032
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Replacing mean-pool z_dyn with attention-pooled multi-dimensional feature vectors (sub_features=K=4) from the separate dyn backbone, gated by the coord backbone's soft-argmax attention, will achieve mean ΔR²_color ≥ 0.30 on non-collapsed seeds (N=20 union seed bank) on ARM E2 (VICReg-only + rich readout). The mean-pool spatial averaging was the demonstrated dominant bottleneck (iter_031: Reconstruction+VICReg reached MSE=0.018 but ΔR²=0.063); the attention-pooled readout fixes both the spatial low-pass problem (attend-at-centroid instead of average-everywhere) and the single-scalar capacity problem (K features per channel instead of 1). The separate-backbone architecture with mask_dyn_sim=True eliminates the known collapse driver (iter_027-028).

Pre-registered prediction for E1.5 (scalar centroid-sampling): E1.5 should yield a partial gain (~+0.10, similar to iter_021 CGIR's +0.124) but miss the 0.30 threshold. Only the rich K=4 readout (E2) should clear 0.30.

## 2. Falsification Criterion
F1 FAIL (E2 mean ΔR²_color < 0.30): the rich readout does not break through the identity-encoding threshold. If F1 fails, this is the third convergent signal (after iter_021 CGIR +0.124 partial gain and the 5-objective convergent null) that ΔR²_color ≥ 0.30 is the wrong target on this architecture, and the project hard-pivots to behavioral evaluation.

IMPORTANT: F1 applies to E2 specifically. If only E3 passes F1 but E2 does not, that pattern does NOT clear F1 — it would mean SFA is doing the work, not the rich readout, which is a different claim than the one being tested.

## 3. Proposed Method
Four-arm experiment on 20-seed union bank (seeds 7,17,31,53,71,83,97,101,103,107,109,113,127,131,137,139,149,151,157,163).

All arms: SeparateDynEncoder-based model, d_max=8, d_t=3, pos_encoding="none", gdasr_log_only=True, coord_vicreg=True, mask_dyn_sim=True, 8000 training steps, batch_size=32, lr=3e-4, replay_buffer_capacity=4000.

E1 — VICReg-only + MEAN-POOL CONTROL: dyn_readout="mean", sub_features=1, primary_objective="sfa" with sfa_weight=0 (VICReg-only). Re-runs the iter_029 Arm A baseline for paired-seed comparison.

E1.5 — VICReg-only + CENTROID-GATED SCALAR (sub_features=1): dyn_readout="centroid_gated", sub_features=1, primary_objective="sfa" with sfa_weight=0 (VICReg-only). Isolates the spatial low-pass fix (attend-at-centroid vs average-everywhere) WITHOUT the capacity increase. Pre-registered prediction: partial gain ~+0.10, insufficient to clear 0.30.

E2 — VICReg-only + RICH READOUT (sub_features=4): dyn_readout="centroid_gated", sub_features=4, primary_objective="sfa" with sfa_weight=0 (VICReg-only). PRIMARY F1 arm. Tests whether the readout change (both spatial fix AND capacity increase) rescues identity encoding.

E3 — SFA+VICReg + RICH READOUT (sub_features=4): dyn_readout="centroid_gated", sub_features=4, primary_objective="sfa", sfa_weight=5.0 (the best from iter_029). Informational objective comparison only — does NOT rescue F1 if E2 fails.

Key architectural change: Modify SeparateDynEncoder to support centroid_gated readout with sub_features>1:
1. In coord backbone forward, expose a_spatial (B, d_max, 128) and p_c (B, d_max, 128)
2. In dyn backbone forward, replace conv_identity_dyn(128->d_max) with conv_identity_dyn(128->d_max*K) for K=sub_features
3. After interpolating a_dyn to (B, d_max*K, 128), reshape to (B, d_max, K, 128)
4. Attend: z_dyn = einsum('bcs,bcks->bck', p_c.detach(), a_dyn_reshaped) → (B, d_max, K) → (B, d_max*K)
5. Stop-gradient on p_c (consistent with existing centroid_gated convention)
6. For sub_features=1 centroid_gated: same pattern with K=1

Modified model class: SeparateDynEncoder in src/models_separate_dyn.py
Modified model class: NonParametricJEPASpatialSeparateDyn in src/models_separate_dyn.py
New runner script: src/run_iter032.py

## 4. Gates (PRE-REGISTERED, BINDING)

F1: mean ΔR²_color (E2, non-collapsed seeds) ≥ 0.30 — PRIMARY readout-fix claim
F2: lower 95% CI of E2 mean ΔR²_color ≥ 0.18 — variance stability
F3: collapse rate ≤ 0.10 across ALL arms — safety check
F4: E2 − E1 paired-seed mean ΔR² improvement ≥ 0.10 — readout-matters (spatial + capacity)
F5: E2 − E1.5 paired-seed mean ΔR² improvement ≥ 0.10 — rich-matters (K=4, not just spatial fix)
F6: E3 − E2 paired-seed mean ΔR² improvement — informational only, no gate (too underpowered for decisive SFA-vs-VICReg claim)

## 5. Binding Decision Rule

(a) If E2 clears F1 (≥0.30) AND F2 (lower CI ≥ 0.18) AND F4 (E2−E1≥0.10) AND F5 (E2−E1.5≥0.10): representation foundation clears the declared gate → advance to Phase 2/3 motor/CLTS integration.

(b) If E2 fails F1 (< 0.30): this is the THIRD convergent signal → hard-pivot to behavioral evaluation (centroid tracking, collision selectivity, causal sensitivity) with the best available representation. No further representation-only iterations.

SPECIAL: If E3 alone passes F1 but E2 does not → does NOT clear F1. That pattern means SFA is doing the work, not the rich readout.

## 6. Pre-Registered Behavioral-Pivot Protocol (for iter_033, if triggered by branch b)

If F1 fails and the project hard-pivots, iter_033 will evaluate behavioral metrics using the best available representation (likely E3 or E2 from iter_032, whichever has highest ΔR²_color). The following gates are pre-registered NOW to prevent retroactive threshold choice:

G1 — Tracking: surprise-driven mean tracking error ≤ 0.75 × random baseline (random baseline = 38.75 px from iter_031; threshold = 29.06 px). Calibrated against measured random baseline, not arbitrary.

G2 — Collision selectivity: surprise-driven collision selectivity ≥ random × 1.5 (ratio) OR surprise-driven − random ≥ 0.20 (absolute Δ). The existing 0.59-vs-0.44 signal from iter_031 is a real directional signal and must not be dismissed.

G3 — Perturbation selectivity: surprise-driven perturbation selectivity (post mass-perturbation) ≥ random baseline + 0.10 (absolute). Tests whether the agent detects the hidden parameter change.

Note on M2 interaction: Under M2, the surprise signal becomes z_coord-dominated post-SFA (since SFA makes z_dyn slow and easy to predict). Collision-selectivity is therefore on the correct axis to detect a real effect — the mechanism naturally attends to positional unpredictability (post-collision motion).

Environment: N=2 collision-sparse (as calibrated in iter_031 Part B) to eliminate ceiling effects. Seeds: same union bank. Conditions: surprise-driven, frozen, random.

---
*Created by the RDF Orchestrator prior to iter_032 execution, incorporating Manager corrections C1, C2, C3.*
```