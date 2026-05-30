# Research Manager Log - Iteration 032

## Iteration 032 -> Planner [Pre-Planning Hints]

# Manager's Pre-Planning Hints — Pre-Iter-032

**H1 (directional, primary).** The readout fix must be **rich**, not scalar. Implement z_dyn as attention-pooling of a multi-dimensional per-position feature vector at the channel's soft-argmax locus — `z_dyn[c] = Σ_x softmax(a_spatial[c])(x) · F(x)` where `F(x)` is a small projection of conv3/conv4 features, not the scalar channel activation. Scalar centroid-sampling alone is predicted (per iter_021 CGIR's +0.124 and the iter_022 single-scalar capacity issue) to deliver another partial ~+0.1 gain and miss the 0.30 threshold; the rich vector readout is the only version that genuinely tests whether the readout is the *sole* remaining bottleneck.

**H2 (decision rule, pre-commit before running).** This must be the **last representation-only iteration**, and the branching rule has to be written into the pre-registration with both branches binding. (a) If E2 or E3 clears ΔR²_color ≥ 0.30 with lower-CI ≥ 0.18 and F4 (paired-seed Δ > 0.10 over E1) holds, the representation foundation is *solved* and the next iteration advances to Phase 2/3 motor/CLTS integration. (b) If only a partial gain is recovered, that is the **third convergent signal** (after iter_021 CGIR and the five-objective null) that ΔR²_color ≥ 0.30 is the wrong target on this architecture, and the project hard-pivots to behavioral evaluation — centroid tracking error, post-collision probing selectivity (the existing 0.59 vs 0.44 signal is real and must be re-tested with calibrated gates, not dismissed), and mass-perturbation causal sensitivity. No third option ("try one more readout variant") is permitted in the plan.

**H3 (scientific discipline, what NOT to do).** Do not revise the M2 mandate text in iter_032: the five-objective convergent null was produced under a broken readout and is therefore evidence about *architecture*, not about objective choice — revising M2 now would over-attribute. The SFA-vs-reconstruction-vs-contrastive comparison only regains meaning once the readout is fixed, so the iter_032 design must include the paired E1/E2/E3 arms (VICReg-only mean-pool control, VICReg-only rich readout, SFA+VICReg rich readout) on the union seed bank so E3−E2 actually isolates the objective question. Preserve unchanged: separate backbone + iter_028 collapse-avoiding config, d_t=3 frozen with GDASR log-only, the decoder-free constraint (reconstruction was tested and did not rescue the bottleneck, which strengthens — not weakens — the decision to keep decoder-free), and **no positional encoding** (cross-objective harmful per iter_013 and iter_021).

---

## Iteration 032 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Replacing mean-pool z_dyn with attention-pooled multi-dimensional feature vectors
(sub_features=K=4) from the separate dyn backbone, gated by the coord backbone's
soft-argmax attention, will achieve mean ΔR²_color ≥ 0.30 on non-collapsed seeds
(N=20 union seed bank). The mean-pool spatial averaging was the demonstrated
structural bottleneck (iter_031: Reconstruction+VICReg reached MSE=0.018 but
ΔR²=0.063); the attention-pooled readout fixes both the spatial low-pass problem
(attend-at-centroid instead of average-everywhere) and the single-scalar capacity
problem (K features per channel instead of 1). The separate-backbone architecture
with mask_dyn_sim=True eliminates the known collapse driver (iter_027-028).

**Proposed Falsification Criterion:**
F1: Mean ΔR²_color (Arm E2 or E3, non-collapsed seeds) < 0.30 — the rich readout
does not break through the identity-encoding threshold. If F1 fails, this is the
third convergent signal (after iter_021 CGIR +0.124 partial gain and the 5-objective
convergent null) that ΔR²_color ≥ 0.30 is the wrong target on this architecture,
and the project hard-pivots to behavioral evaluation.

**Proposed Method:**
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

## Iteration 032 -> Planner [Strategic Guidance]

# Manager's Note — Critique of Iter-032 Plan

The plan correctly implements the rich attention-pooled readout (sub_features=K=4) as instructed, with the three-arm paired-seed design (E1 mean-pool control, E2 rich+VICReg, E3 rich+SFA) and the binding two-branch decision rule. The pre-registration file is named. Three corrections are required before execution.

**C1 — F1 reachability and the "third convergent signal" framing are partially self-contradictory; tighten the decision rule, do not soften it.** The plan correctly preserves the 0.30 threshold and the "hard pivot to behavioral evaluation if it fails" commitment. But "F1 = best of E2/E3" is a *cherry-pick over two arms* that inflates the apparent gain by max-of-two. Either (a) pre-designate E2 as the primary F1 arm and treat E3 as a secondary objective comparison (cleaner), or (b) keep "best of two" but apply a Bonferroni-style correction (raise lower-CI to ≥ 0.22). I recommend (a): E2 is the variable that isolates the readout fix; E3 conflates readout + objective and should not be allowed to rescue F1 alone. Also: state explicitly in the pre-registration that a partial gain on E3 *only* (E2 fails, E3 passes) does **not** clear F1, because that pattern would mean SFA is doing the work, not the rich readout — a different claim than the one being tested.

**C2 — Construction-vs-empirical hazard on F4 (the "readout matters" gate).** F4 requires E2 − E1 ≥ 0.10 paired-seed. But E2 changes *two* things vs E1: the readout (mean → attention-pool at centroid) *and* the channel dimensionality of the feature map fed to it (sub_features 1 → 4). If E2 > E1, the plan as written cannot distinguish "rich readout helped" from "K=4 channels helped" — exactly the iter_031 d_max=2-vs-8 confound at a different layer. Add a fourth arm **E1.5: centroid-gated readout with sub_features=1** (scalar centroid-sample, the iter_021 CGIR variant). The pre-registered prediction in the user hint is explicit: scalar centroid-sampling alone should yield ~+0.10 (partial), and only the rich K=4 readout should clear 0.30. E1.5 is the arm that makes that prediction falsifiable. Without it, a positive E2 result is constructional ("we added capacity") rather than empirical ("attention-pooling of rich features is the right inductive bias"). Four arms × 20 seeds = 80 runs; budgeted comparably to iter_029 (60 runs) and iter_031 (~108 runs).

**C3 — Pre-registration must include the behavioral-pivot protocol *now*, not on iter_033.** The binding rule (b) commits to a hard pivot if F1 fails, but the CLTS Part B protocol it would pivot to is *not yet specified* in the pre-registration. iter_030/031 showed that running calibration in parallel with representation work confounds it; the symmetric risk is that *not* specifying the pivot protocol in advance lets the next iteration retroactively choose gates that "almost pass." Pre-register the iter_033 pivot gates now: (i) collision selectivity threshold restated against the existing 0.59-vs-0.44 baseline (the user hint is explicit that this signal is real and should not be dismissed — pick a defensible threshold like ratio ≥ 1.5× or absolute Δ ≥ 0.20 over random, and commit to it before seeing iter_032 results), (ii) tracking-error threshold *calibrated against random baseline* (38.75 px) rather than the iter_030 arbitrary 20 px, and (iii) mass-perturbation causal-sensitivity gate. Also state explicitly that under M2 the surprise signal will be **z_coord-dominated** post-SFA (per the goal document Pillar D), so the collision-selectivity metric is on the correct axis to detect a real effect, not an artifact.

**Language hygiene.** The phrase "the structural bottleneck" in the hypothesis is fine as a working description but the iter_032 report must say "consistent with the readout being a dominant bottleneck" — iter_031 localized one bottleneck; the second (channel capacity) has never been cleanly tested. "Solved" in the decision rule should be softened to "clears F1 under the declared protocol." Avoid "definitively" and "the representation is solved" — restate as "the foundation clears the declared gate and Phase 2/3 integration becomes the next bottleneck to test."

**Process note.** The Orchestrator will automatically write the finalized hypothesis and falsification criteria to `src/pre_registration.md` before execution. Confirm in the plan that all sub-agents will read it and adhere strictly — particularly the four-arm structure (C2), the E2-primary F1 rule (C1), and the iter_033 pivot gates (C3) — and that no arm or gate may be added, removed, or relaxed after seeing partial results.

---

