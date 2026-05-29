# Research Manager Log - Iteration 028

## Iteration 028 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (for the iter_028 Planner)

1. Directional — run the pre-registered C1/C2/C3 control matrix and nothing else. The single most important arm is C1 (`mask_dyn_sim=True` on the **shared** backbone): without it we cannot distinguish "separate backbone was load-bearing" from "removing `sim_loss_dyn` is what matters." Treat C1 as the iteration's primary arm; C2 (fresh seed bank, n=10) and C3 (±10% var/cov perturbation) are the robustness gates. Hold buffer=4000, Hungarian-primary, d_max=8, d_t=3, sim_weight/var_weight/cov_weight=25/25/1 constant. Do not introduce SFA, slowness, or new objectives in this iteration — that is iter_029 territory and would confound the diagnostic.

2. Scientific discipline — pre-register the verdict language in the plan itself, not after the fact. Required falsification thresholds (carry forward from the journal): C1 collapses ≥20% → separate-backbone *was* load-bearing and the Arm-C signal collapses to that confound; C2 collapses ≥20% → seed-bank artefact; C3 collapses ≥20% → not robust. The plan must also explicitly state, before running, that Arm C's relationship to the std-based collapse criterion is partially constructional (VICReg's variance hinge enforces the same quantity the gate measures), and that even a fully-confirmed C1+C2+C3 result will be reported as "removing `sim_loss_dyn` from the z_dyn gradient path does not destabilize VICReg-maintained variance" — *not* as "sim_loss_dyn causes collapse." Ban the words "breakthrough", "causal driver", "eliminated", and "BEST" from the result write-up; the iter_027 overclaim loop is now a tracked failure mode and the plan must show it has been internalized.

3. Discipline-on-metrics — beyond the std-based collapse gate, the plan must pre-declare at least one *independent* readout that is not directly enforced by VICReg: e.g. delta_R2_color (already used in iter_027), per-channel covariance off-diagonals, or a held-out identity-decoding probe. If C1 passes the std gate but the independent readout shows no improvement over the collapsed baseline, the Arm-C mechanism is downgraded regardless of the std-collapse numbers. This is the guard against the constructional concern. Only if C1 passes *both* the std gate and the independent readout does the iter_029 promotion to M2-style SFA-on-`z_dyn` become justified.

---

## Iteration 028 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Removing the JEPA similarity loss gradient on z_dyn (mask_dyn_sim=True) from the
shared-backbone NonParametricJEPASpatial encoder does not destabilize the
VICReg-maintained per-dimension variance of z_dyn, AND z_dyn retains meaningful
identity encoding as measured by an independent (non-VICReg-enforced) readout.

Formally: Let C_C1 be the dual-criterion collapse rate of arm C1 (shared backbone,
mask_dyn_sim=True, weights 25/25/1) over 10 seeds. Let ΔR²_C1 be the mean
delta_R2_color of C1 across non-collapsed seeds. The hypothesis is:
(H1) C_C1 ≤ 0.10 (std-based collapse gate)
(H2) ΔR²_C1 ≥ 0.10 (independent semantic readout gate)

Both conditions must hold. This is the critical isolate that iter_027's Arm C
could not provide (it confounded separate-backbone with mask_dyn_sim).

Constructional acknowledgment: VICReg's variance hinge (γ - std, hinge at γ=1)
directly enforces per-dimension std ≥ 1, which trivially satisfies the collapse
gate (std < 0.5). Even a fully-confirmed C1+C2+C3 result will be reported as
"removing sim_loss_dyn from the z_dyn gradient path does not destabilize
VICReg-maintained variance" — NOT as "sim_loss_dyn causes collapse."

**Proposed Falsification Criterion:**
Four independent falsification conditions, any one of which refutes the hypothesis
or downgrades its interpretation:

F1: C1 collapse rate ≥ 0.20 (dual criterion) → mask_dyn_sim alone is insufficient
    on the shared backbone; the separate-backbone architecture was load-bearing.
    The Arm C signal from iter_027 collapses to that confound.

F2: C2 collapse rate ≥ 0.20 (fresh seed bank) → the C1 result is seed-dependent,
    not general.

F3: C3 collapse rate ≥ 0.20 (±10% weight perturbation) → the C1 result is not
    robust to reasonable hyperparameter variation.

F4: C1 passes the std gate (C_C1 ≤ 0.10) BUT ΔR²_C1 < 0.05 → the VICReg-maintained
    variance is constructional; z_dyn has variance but no semantic content. The
    mechanism is downgraded regardless of the std-collapse numbers. Only if C1
    passes BOTH the std gate AND the independent readout does the iter_029
    promotion to M2-style SFA-on-z_dyn become justified.

Additional guard: D0 (shared backbone, mask_dyn_sim=False, weights 25/25/1).
If D0 collapse rate < 0.20, then the cov_weight change (25→1) alone reduces
collapse, confounding the C1 attribution. The report must state this.

**Proposed Method:**
Step-by-step experimental protocol:

1. CREATE src/run_phase0_mask_dyn_sim_shared.py — the experiment runner.
   Based on run_phase0_separate_dyn.py, simplified to shared-backbone only.
   Implements mask_dyn_sim via loss adjustment after forward():
     adjusted_loss = loss_dict["loss"] - sim_weight * loss_dict["sim_loss_dyn"]
   This is identical to NonParametricJEPASpatialSeparateDyn's approach.
   No modification to models_dual_stream.py.

2. FOUR ARMS (10 seeds each, 40 total runs):

   D0 — Weight-change anchor (shared backbone, mask_dyn_sim=False, weights 25/25/1):
     NonParametricJEPASpatial, dyn_readout="mean", d_max=8, d_t=3,
     pos_encoding="none", primary_objective="jepa", lr=3e-4, batch_size=64,
     buffer=4000, 8000 steps, gradient clipping max_norm=1.0,
     ccr_mode="covariance", ccr_smooth_weight=10, ccr_spatial_weight=10,
     gdasr_log_only=True, sim_weight=25, var_weight=25, cov_weight=1,
     seeds=[7, 17, 31, 53, 71, 83, 97, 113, 127, 149].
     Purpose: Establish shared-backbone collapse rate with cov_weight=1.
     If D0 ≈ 30-40% (like iter_027), the weight change alone doesn't help.

   C1 — Primary arm (shared backbone, mask_dyn_sim=True, weights 25/25/1):
     Same as D0 but mask_dyn_sim=True. Same seeds.
     The critical isolate: does removing sim_loss_dyn prevent collapse
     without the separate-backbone confound?

   C2 — Seed robustness (shared backbone, mask_dyn_sim=True, weights 25/25/1):
     Same as C1 but fresh seed bank: [101, 103, 107, 109, 131, 137, 139,
     151, 157, 163]. Tests whether C1's result is seed-dependent.

   C3 — Weight robustness (shared backbone, mask_dyn_sim=True, weights 27.5/27.5/1.1):
     Same as C1 but var_weight=27.5, cov_weight=1.1 (+10% perturbation).
     Original seeds. Tests sensitivity to weight variation.

3. EVALUATION at step 8000 (same protocol as iter_027):
   - Dual collapse criterion: collapsed_eval OR collapsed_train (per-dim std < 0.5)
   - Train-vs-eval std gap: report per-seed, co-equally with collapse rates
   - Hungarian-primary matching for semantic probes
   - Semantic probes: delta_R2_color (INDEPENDENT READOUT, pre-declared gate ≥0.10),
     r2_dyn_color, r2_coord_color, r2_dyn_pos, r2_coord_pos,
     r2_dyn_identity, delta_r2_identity
   - VICReg health: per_dim_std, mean_abs_corr
   - Centroid MSE (reference only, NOT used for arm selection)
   - Training loss sanity: mean total loss at step 8000, loss > 50 → disqualified
   - Parameter count per arm (logged before runs start)

4. STOP RULE: All 40 runs complete. No early termination.

5. PRE-REGISTERED OUTCOME CLASSIFICATION:
   - If C1 ≤ 10% AND ΔR²_C1 ≥ 0.10: CONFIRMED — mask_dyn_sim on shared
     backbone does not destabilize VICReg-maintained variance AND preserves
     semantic encoding. Promotion to M2-style SFA (iter_029) is justified.
   - If C1 ≥ 20%: FALSIFIED — mask_dyn_sim alone insufficient; separate
     backbone was load-bearing.
   - If C1 ≤ 10% BUT ΔR²_C1 < 0.05: DOWNGRADED — VICReg variance is
     constructional; no semantic content despite maintained variance.
   - If C2 ≥ 20%: SEED-DEPENDENT — C1 result does not generalize.
   - If C3 ≥ 20%: NOT ROBUST — C1 result sensitive to weight perturbation.

6. LANGUAGE CONSTRAINTS (tracked failure mode from iter_027 overclaim):
   - Use "does not destabilize VICReg-maintained variance" or "is consistent
     with"; do NOT use "breakthrough", "causal driver", "eliminated", "BEST",
     "proves", "demonstrates", or "resolves."
   - Even a fully-confirmed C1+C2+C3 result is reported per the constructional
     framing in the hypothesis, not as "sim_loss_dyn causes collapse."

FILES TO CREATE:
- src/run_phase0_mask_dyn_sim_shared.py (NEW): experiment runner

FILES TO MODIFY:
- src/pre_registration.md (UPDATE with iter_028 plan)

FILES NOT TO MODIFY:
- src/models_dual_stream.py (mask_dyn_sim handled in runner, not model)
- src/models_separate_dyn.py (not needed for this experiment)
- src/environment.py

Total runs: 4 arms × 10 seeds = 40 runs × 8000 steps each.
Expected wall time: ~30-40 minutes with parallel workers (CPU).

---

## Iteration 028 -> Planner [Strategic Guidance]

Manager's Note (iter_028 — Plan Critique)

The proposed plan is structurally sound and internalizes the iter_027 discipline (pre-registered thresholds, banned-language list, dual-gate H1+H2, D0 weight-change anchor). Approve in principle, with **three required corrections before execution**.

1. **The D0 anchor as drafted does not isolate what you claim.** D0 is described as "shared backbone, mask_dyn_sim=False, weights 25/25/1," and its stated purpose is to test whether the cov_weight change (25→1) alone reduces collapse. But iter_026/027 baselines were *already* run at cov_weight=1 (that is the canonical setting), so D0 reproduces the existing 30–40% baseline rather than isolating a weight delta. If you actually want to rule out a cov_weight confound, D0 must vary the weight that *changed* between iter_027's reference and C1 — and since C1 uses the same 25/25/1 as iter_027's baselines, **there is no weight delta to confound, and D0 collapses to a baseline replication.** Either (a) re-label D0 honestly as "shared-backbone JEPA+VICReg baseline replication" and drop the cov_weight-confound rationale from the falsification text, or (b) re-specify D0 to actually vary cov_weight if you genuinely suspect that confound. Pick one; do not ship the current mismatch between rationale and arm definition.

2. **The independent readout (H2 / F4) needs a pre-declared null reference, not just a threshold.** ΔR²_color ≥ 0.10 is reasonable but the gate is only meaningful relative to what a *collapsed* or *random-projection* z_dyn yields on the same probe. iter_025 showed ΔR² values in the −0.10 to +0.14 range across configurations; +0.10 is near the noise floor of that distribution. Required additions to the pre-registration before execution: (i) report ΔR²_color for D0 alongside C1/C2/C3 as the in-iteration null reference; (ii) state F4 as "C1 passes std gate AND ΔR²_C1 < D0_ΔR²_color + 0.05" or similar relative form, not an absolute 0.05/0.10 number floating free; (iii) include `mean_abs_corr` in the H2 gate — iter_027 Arm C's 0.21 vs Arm B's 0.41 was at least as informative as ΔR². A representation that passes the std gate but matches a collapsed baseline on independent readouts is exactly the construction-vs-empirical failure mode you flagged, and the threshold must be calibrated to detect it.

3. **Scientific-discipline reminders (mandatory).** (a) **Pre-registration:** before running any code, write the final hypothesis (H1 ∧ H2), the four falsification conditions (F1–F4) with the relative-threshold correction from point 2, the seed banks, the arm specifications, and the banned-language list to `src/pre_registration.md` and commit it. The Orchestrator will commit this file automatically — do not run experiments before that file exists on disk. (b) **Language hygiene under success:** even if all four conditions pass, the result write-up uses "is consistent with" / "does not destabilize" / "does not refute"; the words "breakthrough," "causal driver," "eliminated," "BEST," "proves," "demonstrates," "resolves" remain banned, exactly as you listed. (c) **n=10 power caveat must appear in the report:** Fisher's exact for 0/10 vs 3/10 gives p ≈ 0.21; the design cannot formally distinguish 0% from 10–20% at this sample size. State this limit in the pre-registration so the result write-up cannot quietly elide it. (d) **Do not introduce SFA, slowness, or any new objective in this iteration** — that is iter_029 territory and would confound the diagnostic; the plan correctly stays within JEPA+VICReg ± mask, keep it that way.

A clean null on C1 (≥20% collapse) is a fully acceptable outcome that closes the Arm-C mechanism cleanly and is a success of the protocol — propagate it as such, do not re-frame it as partial progress.

---

