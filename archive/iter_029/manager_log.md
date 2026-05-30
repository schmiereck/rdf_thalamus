# Research Manager Log - Iteration 029

## Iteration 029 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints for iter_029

1. Directional: The iter_028 result lands cleanly — both the loss-masking AND the separate-backbone intervention are needed to reach 0%, and even then the C2-vs-C1 seed-bank dependence shows that VICReg-only stability is fragile. Do **not** spend iter_029 on another sweep of mask/backbone permutations; that 2×2 is now saturated. Instead, take the goal-document's intended next step (Journal §4, iter_029 conditional path): promote z_dyn shaping from VICReg-only to the actual **M2 configuration — explicit SFA slowness term `||z_dyn(t) − z_dyn(t−1)||² + pooled VICReg`** on the separate-backbone architecture, with `sim_loss_dyn` masked and the prediction-error pathway demoted to a stop-gradient readout. This converts the iter_027/028 "remove the competing pressure" finding into the constructive direction the project actually needs.

2. Scientific discipline: The construction-vs-empirical caveat from iter_027 gets **worse**, not better, once an explicit slowness term is added — slowness on z_dyn plus a VICReg variance hinge can together overfit the very std/temporal-stability quantities the collapse gate measures. The Planner must pre-register (a) a falsification target that is *independent* of the training objective — recommended: centroid-decoding MSE via Arm F and ΔR²_color/identity recovered by a held-out linear probe, with a numerical threshold relative to the iter_028 C2 numbers (ΔR²_color 0.51, mean_abs_corr 0.44) before the run starts; (b) a no-slowness control arm (VICReg-only, i.e. the iter_028 C2 configuration) on the same seed bank so the *marginal* contribution of the SFA term is measurable rather than confounded with the architecture+masking change; and (c) honest reporting that "0% collapse" under an objective that directly optimizes the collapse quantity is expected and is not the headline metric.

3. Seed-bank handling: The hard-seed signal (53, 71 collapsing under C1/C3 but seed 71 surviving D0, with a qualitatively different collapse mode at std ~0.01) is real information and must not be paved over. Require the Planner to (i) run iter_029 on a **superset seed bank** that explicitly includes 53 and 71 alongside the C2 fresh seeds, and (ii) report per-seed outcomes and the std-magnitude of any collapses, so we can tell whether SFA fixes the severe-mode failure or merely shifts it. If SFA eliminates the hard-seed collapse, that is the strongest empirical content available; if it does not, that is the iter_030 lead.

---

## Iteration 029 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
On the separate-backbone architecture (SeparateDynEncoder via NonParametricJEPASpatialSeparateDyn),
adding an explicit SFA slowness term ||z_dyn(t) - z_dyn(t-1)||² to the VICReg-only z_dyn objective
improves identity encoding as measured by ΔR²_color from a held-out linear probe, without introducing
collapse on the hard-seed seed bank [7, 17, 31, 53, 71, 83, 97, 113, 127, 149].
Specifically: Arm B (SFA+VICReg, sfa_weight=5.0, mask_dyn_sim=True) will show ΔR²_color > 0.1812
(the VICReg-only baseline from iter_027 Arm C) when computed over non-collapsed seeds, AND will
show collapse rate ≤ 10% on the same seed bank.

**Proposed Falsification Criterion:**
The hypothesis is falsified if EITHER:
(F1) Arm B (SFA+VICReg) shows ΔR²_color ≤ 0.1812 (the VICReg-only baseline) when computed over
     non-collapsed seeds — meaning SFA adds no marginal identity-encoding benefit even on the
     gradient-isolated separate-backbone architecture; OR
(F2) Arm B shows collapse rate > 10% (≥2 of 10 seeds) on the original seed bank — meaning SFA
     destabilizes the previously stable VICReg-only regime.
Either outcome refutes the claim that SFA constructively shapes z_dyn on this architecture.

**Proposed Method:**
Three-arm experiment on the original 10-seed bank [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]:

Arm A (Control — reuse iter_027 Arm C): Separate backbone, primary_objective="jepa",
  mask_dyn_sim=True, var_weight=25, cov_weight=25, sim_weight=1. This is VICReg-only on z_dyn.
  Known result: 0% collapse, ΔR²_color=0.1812. Data reused from iter_027.

Arm B (SFA+VICReg, primary): Separate backbone, primary_objective="sfa", mask_dyn_sim=True,
  sfa_weight=5.0, var_weight=25, cov_weight=25, sim_weight=1. This adds the SFA slowness term
  to the VICReg-only z_dyn objective, with JEPA prediction as a stop-gradient readout.
  10 new runs on the original seed bank.

Arm C (SFA+VICReg, conservative): Same as Arm B but sfa_weight=1.0.
  10 new runs on the original seed bank.
  
Total new runs: 20 (Arms B and C). Arm A data reused from iter_027.

Training: 8000 steps, batch_size=32, lr=3e-4, d_t=3 (frozen), buffer=4000.
Architecture: NonParametricJEPASpatialSeparateDyn (SeparateDynEncoder + DualStreamPredictor).

Evaluation metrics per run:
- Collapse check: per-dim std of z_dyn (threshold 0.5) on both eval and train batches
- ΔR²_color: held-out linear probe predicting RGB color from z_dyn (identity encoding quality)
- Centroid MSE: Arm F soft-argmax position decoding (spatial encoding quality)
- mean_abs_corr: VICReg health metric on z_dyn
- sfa_loss trajectory: verify slowness term is active and decreasing
- Per-seed outcomes with per-dim std magnitudes for any collapses

Key confounds to report:
- SFA mode sets var_loss_coord=0 and cov_loss_coord=0 (no VICReg on coord stream),
  while the JEPA-mode control gives coord stream full VICReg. This could affect z_coord
  quality but NOT z_dyn quality, so ΔR²_color comparison remains fair.
- "0% collapse" under SFA+VICReg is expected (SFA + VICReg both encourage high std)
  and is NOT the headline metric. The headline is ΔR²_color improvement.

Files to create:
- src/run_phase0_sfa_separate_backbone.py: experiment runner for Arms B and C
  (modeled after src/run_phase0_separate_dyn.py)

Files to reuse (unchanged):
- src/models_separate_dyn.py: NonParametricJEPASpatialSeparateDyn with mask_dyn_sim support
- src/models_dual_stream.py: NonParametricJEPASpatial with SFA mode
- src/environment.py: PhysicsSandbox

---

## Iteration 029 -> Planner [Strategic Guidance]

## Manager's Note — Critique of iter_029 Plan

**Endorsed in principle, but three structural fixes are required before pre-registration.** This is the right next step per the journal's iter_029 conditional path. The plan correctly identifies the M2 promotion, reuses iter_027 Arm C as a free control, and pre-declares that "0% collapse" is not the headline. Hold those properties. Fix the following:

**1. Seed-bank confound is unaddressed and will eat the result.**
Running Arm B/C only on the **original** seed bank (with hard seeds 53, 71) means a positive ΔR²_color result is ambiguous between (a) "SFA constructively improves identity encoding" and (b) "SFA happens to stabilize the hard seeds that VICReg-only sometimes can't." iter_028 already proved this matters — C1 and C2 differed by 20 vs 0 % collapse on *identical configs*. **Mandate:** run Arms B and C on a **union seed bank**: original {7,17,31,53,71,83,97,113,127,149} ∪ the iter_028 C2 fresh seeds (n = 20 each arm). Report per-seed outcomes, and explicitly tabulate the four hard-seed cases (D0/C1/B/C × {53,71}) so the question "does SFA fix hard seeds?" gets a clean answer.

**2. Construction-vs-empirical discipline and effect-size threshold are weak.**
SFA on z_dyn *by definition* makes z_dyn temporally stable, and identity *is* the temporally stable signal — so the M2 mechanism *predicts* ΔR²_color will rise. A positive result therefore is "consistent with M2" rather than a discovery. The falsification threshold "ΔR²_color > 0.1812" has no margin and will be triggered by noise. Pre-register a **practical-significance threshold**: ΔR²_color ≥ 0.1812 + 2σ (compute σ from the per-seed variance in iter_027 Arm C), or in absolute terms ≥ 0.30 — whichever is larger. Anything between 0.18 and that threshold is reported as "no detectable marginal benefit." Also pre-register language: "is consistent with M2's predicted mechanism," not "demonstrates SFA improves identity."

**3. Two confounds in the arm definitions need explicit handling.**
(a) The plan notes SFA-mode zeros `var_loss_coord` and `cov_loss_coord` while the JEPA-mode control keeps them — this is *not* a fair comparison even for z_dyn, because gradient interactions through the shared optimizer step are not orthogonal across streams. Either patch SFA mode to keep coord-stream VICReg on, or add a third arm (SFA + coord-VICReg-on) and treat the existing Arm B as a confound-check rather than the primary. (b) Add a **centroid-MSE no-degradation floor** to the falsification criterion (e.g., centroid_mse must not exceed iter_027 Arm C by more than 1σ). Otherwise a "win" on ΔR²_color that silently destroys the spatial readout would still pass the gate.

**Pre-registration reminder:** The Orchestrator will automatically commit `src/pre_registration.md` based on this plan and instruct sub-agents to read and adhere to it during execution. Before that write-out, the Planner must update the hypothesis and falsification block to incorporate the union seed bank, the σ-based ΔR² threshold, the per-seed hard-seed table requirement, the coord-stream VICReg fix, and the centroid-MSE floor. Without these, the plan does not have a quantitative falsification criterion sufficient to proceed to execution.

---

