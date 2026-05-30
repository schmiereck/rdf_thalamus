# RDF Scientific Pre-Registration

*   **Iteration:** 032
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
Replacing mean-pool z_dyn with attention-pooled multi-dimensional feature vectors
(sub_features=K=4) from the separate dyn backbone, gated by the coord backbone's
soft-argmax attention, will achieve mean ΔR²_color ≥ 0.30 on non-collapsed seeds
(N=20 union seed bank). The mean-pool spatial averaging was the demonstrated
structural bottleneck (iter_031: Reconstruction+VICReg reached MSE=0.018 but
ΔR²=0.063); the attention-pooled readout fixes both the spatial low-pass problem
(attend-at-centroid instead of average-everywhere) and the single-scalar capacity
problem (K features per channel instead of 1). The separate-backbone architecture
with mask_dyn_sim=True eliminates the known collapse driver (iter_027-028).

## 2. Falsification Criterion
F1: Mean ΔR²_color (Arm E2 or E3, non-collapsed seeds) < 0.30 — the rich readout
does not break through the identity-encoding threshold. If F1 fails, this is the
third convergent signal (after iter_021 CGIR +0.124 partial gain and the 5-objective
convergent null) that ΔR²_color ≥ 0.30 is the wrong target on this architecture,
and the project hard-pivots to behavioral evaluation.

## 3. Proposed Method
Three-arm experiment on 20-seed union bank (seeds 7,17,31,53,71,83,97,101,103,107,109,113,127,131,137,139,149,151,157,163).

All arms: SeparateDynEncoder with RICH READOUT, d_max=8, d_t=3, pos_encoding="none",
gdasr_log_only=True, coord_vicreg=True, mask_dyn_sim=True, 8000 training steps,
batch_size=32, lr=3e-4, replay_buffer_capacity=4000.

E1 — VICReg-only + mean-pool CONTROL (existing iter_029 Arm A baseline, re-run
     for paired-seed comparison): SeparateDynEncoder with dyn_readout="mean",
     sub_features=1, primary_objective="sfa" with sfa_weight=0 (VICReg-only).

E2 — VICReg-only + RICH READOUT (sub_features=4): Same objective as E1 but with
     dyn_readout="centroid_gated", sub_features=4, dyn_source="spatial".
     This tests whether the readout change alone rescues identity encoding.

E3 — SFA+VICReg + RICH READOUT (sub_features=4): Same as E2 but with
     primary_objective="sfa", sfa_weight=5.0 (the best from iter_029).
     This tests whether SFA adds value on top of the fixed readout.

Key architectural change: Modify SeparateDynEncoder to support centroid_gated readout
with sub_features>1 by:
1. In coord backbone forward, expose a_spatial (B, d_max, 128) and p_c (B, d_max, 128)
2. In dyn backbone forward, replace conv_identity_dyn(128->d_max) with
   conv_identity_dyn(128->d_max*K) for K=sub_features
3. After interpolating a_dyn to (B, d_max*K, 128), reshape to (B, d_max, K, 128)
4. Attend: z_dyn = einsum('bcs,bcks->bck', p_c.detach(), a_dyn_reshaped) → (B, d_max, K) → (B, d_max*K)
5. Stop-gradient on p_c (consistent with existing centroid_gated convention)

New model class: RichDynSeparateEncoder in src/models_separate_dyn.py
New runner script: src/run_iter032.py

Pre-registered decision rule (BINDING):
(a) If E2 or E3 clears ΔR²_color ≥ 0.30 with lower-95%-CI ≥ 0.18 AND 
    E2−E1 paired-seed ΔR² improvement ≥ 0.10 (F4 gate): representation foundation 
    is SOLVED → advance to Phase 2/3 motor/CLTS integration.
(b) If only a partial gain (< 0.30): this is the THIRD convergent signal → 
    hard-pivot to behavioral evaluation (centroid tracking, collision selectivity, 
    causal sensitivity) with the best available representation. No further 
    representation-only iterations.

Gates:
F1: mean ΔR²_color (best of E2/E3, non-collapsed) ≥ 0.30
F2: lower 95% CI of that mean ≥ 0.18
F3: collapse rate ≤ 0.10 across all arms
F4: E2−E1 paired-seed mean ΔR² improvement ≥ 0.10 (readout-matters)
F5: E3−E2 paired-seed mean ΔR² improvement (objective-matters, informational only;
    no gate — too underpowered for a decisive SFA-vs-VICReg claim)

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
