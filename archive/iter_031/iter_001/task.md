Write the corrected pre-registration file for iter_031 to src/pre_registration.md. This file MUST incorporate ALL of the Research Manager's corrections from the planning phase. Read the existing src/pre_registration.md first, then OVERWRITE it with the corrected version.

CRITICAL CORRECTIONS TO INCORPORATE:

1. HYPOTHESIS — Rewrite as a two-sided question:
   - (a) Does Reconstruction+VICReg achieve mean ΔR²_color ≥ 0.30 with lower 95% CI ≥ 0.18?
   - (b) Does it do so with a NON-TRIVIAL MARGIN over controls (d_max=2 under-capacity and random-encoder)?
   - WITHOUT control (b), a "pass" only verifies that the bottleneck has capacity to preserve color, not that training did anything. Language must say "is consistent with the architecture having sufficient capacity to preserve color information under a supervised target" NOT "establishing that the architecture supports identity encoding."

2. FALSIFICATION CRITERIA — Corrected gates:
   - F1: mean ΔR²_color ≥ 0.30 on non-collapsed seeds (d_max=8, trained encoder) — if FAILED, reconstruction does not clear even the decoder-free ceiling
   - F2: lower 95% CI ≥ 0.18 (variance-stability gate, same as iter_029/030) — if FAILED but F1 passed, result is "directionally positive, not robust" and triggers variance investigation, NOT mandate revision
   - F3: mean ΔR²_color (d_max=8, trained) must EXCEED mean ΔR²_color (d_max=2, trained) by ≥ 0.10 — confirms capacity matters (not a trivial encoding)
   - F4: mean ΔR²_color (d_max=8, trained) must EXCEED mean ΔR²_color (d_max=8, random-encoder) by ≥ 0.10 — confirms training matters (not constructional)
   - ALL of F1-F4 must pass for the hypothesis to be supported

3. DROP A2 HYPERPARAMETER SCAN entirely. Lock recon_weight=25.0 (natural midpoint of 10,25,50). No in-flight sweep.

4. THREE ARMS (not two):
   - Arm A: Reconstruction+VICReg, d_max=8, trained encoder, recon_weight=25.0 (PRIMARY, 20 seeds)
   - Arm B: Reconstruction+VICReg, d_max=2, trained encoder, recon_weight=25.0 (under-capacity control, 20 seeds)
   - Arm C: Reconstruction+VICReg, d_max=8, RANDOM-ENCODER (frozen at init, never trained, only decoder trained), recon_weight=25.0 (random encoder control, 20 seeds)

5. SEED BANK: Same union bank as iter_029/030: 10 original [7, 17, 31, 53, 71, 83, 97, 113, 127, 149] + 10 fresh [101, 103, 107, 109, 131, 137, 139, 151, 157, 163] = 20 seeds

6. TRAINING: 8000 steps, batch_size=32, lr=3e-4, d_t=3 (frozen), pos_encoding="none", coord_vicreg=True, var_weight=25.0, cov_weight=25.0, sim_weight=1.0, GDASR log-only

7. EVALUATION: Same pipeline as iter_029/030 (ΔR²_color linear probe, centroid MSE, collapse rate, VICReg health, reconstruction MSE). All metrics with 95% CI across 20 seeds.

8. PART B PROTOCOL CALIBRATION — Pre-register the margin formulas BEFORE running:
   - B1: N=2 collision-sparse CLTS evaluation using existing VICReg-only checkpoint
     - 3 conditions per seed (5 seeds): surprise-driven, frozen (locus=0), random
     - 2000 evaluation steps per condition
     - Gate formulation (pre-registered): 
       * G1_tracking: CLTS tracking error must be ≤ random_tracking_mean − 1*random_tracking_std
       * G2_collision_selectivity: fraction of post-collision steps where colliding channel is attended must be ≥ random_selectivity × 1.5
       * G3_perturbation_selectivity: fraction of post-perturbation steps where changed object's channel is attended must be ≥ random_selectivity × 1.5
     - These gates use MEASURED random baselines, not assumed values
   - B2: Subtle mass perturbation test (1.5× mass change at step 1000)

9. MANDATE REVISION LANGUAGE (pre-committed for both outcomes):
   - If ALL of F1-F4 pass: "Reconstruction+VICReg on the separate-backbone architecture achieves mean ΔR²_color ≥ 0.30 with robust variance. The result is consistent with the architecture having sufficient capacity to preserve color information under a supervised pixel-MSE target. The d_max=2 control (F3) and random-encoder control (F4) confirm that the finding is non-trivial: both capacity and training are required. M2 is revised from 'SFA+VICReg as primary' to 'Reconstruction+VICReg as primary representation objective, decoder-free constraint relaxed as pragmatic compromise. SFA demoted to comparison baseline B1. Surprise readout retained via stop-gradient predictor. Future work may explore BYOL-style decoder-free alternatives approaching the reconstruction ceiling.'"
   - If F1 or F2 fails: "Reconstruction+VICReg fails to achieve ΔR²_color ≥ 0.30 with variance-stability. Even a supervised pixel-reconstruction target cannot make the mean-readout z_dyn stream encode identity above the 0.30 threshold. The z_dyn readout architecture itself constrains identity encoding regardless of objective class. M2 revision pending architectural redesign: priority is centroid-gated z_dyn readout or increased d_max."
   - If F3 fails: "d_max=2 under-capacity control achieves ΔR²_color within 0.10 of d_max=8. The finding is trivially explained by bottleneck capacity — even severe under-capacity preserves color via reconstruction gradient. The result does not support a meaningful training effect."
   - If F4 fails: "Random-encoder control achieves ΔR²_color within 0.10 of trained-encoder. The finding is constructional — random features with a trained decoder are sufficient to preserve color information. The result does not support a meaningful training effect on the encoder."

Write the complete pre-registration file. Use clear section headers. Do NOT include the old A2 hyperparameter scan.