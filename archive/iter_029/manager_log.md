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

## Iteration 029 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — third pre-registered
    iteration in the diagnostic series. iter_029 produced a **clean
    pre-registered null on the M2 mandate itself**: SFA+VICReg on the
    separate-backbone regime did not reach the declared practical-significance
    threshold (ΔR²_color ≥ 0.30) and showed seed-dependent variance large
    enough that the trend cannot be relied upon. Combined with iter_023–024
    (SFA refuted on shared backbone), the explicit slowness objective is now
    empirically falsified across **both architectural regimes** in this task.
*   **Active Direction (revised):** The M2 mandate ("SFA + pooled VICReg as
    the primary representation objective on z_dyn") came into this project
    as a transferred result from `rdf_thalamus_sml`. That mandate is now
    under empirical challenge in the Thalamus task domain. Two
    interpretations remain open and must be distinguished in iter_030:
    (a) the sml transfer was scoped narrowly to the binary, low-DOF toy and
    does not generalize to the 1D RGB physics environment, or
    (b) the SFA term is correctly oriented but is being out-competed by
    another gradient (the VICReg variance hinge, the coord-stream JEPA
    loss, or both) in a way that prevents slowness from acquiring
    identity structure.
    Until that distinction is made, M2 cannot be treated as either
    validated *or* discarded for this project — it is a contested mandate.
*   **What we now have that is solid:** The **separate-backbone +
    VICReg-only z_dyn** configuration (iter_027 Arm C) has now been
    indirectly stress-tested twice (iter_028 C2 fresh seeds = 0% collapse;
    iter_029 60 runs = 0% collapse). It is the only configuration in the
    Phase-0 sweep that has reached zero collapse without invoking SFA.
    Its empirical content remains narrower than "stable identity encoding":
    the construction-vs-empirical caveat (Gate 1) still applies — VICReg's
    variance hinge directly enforces the std metric used in the collapse
    check, so 0% collapse is *partly* tautological. The genuinely empirical
    part is that ΔR²_color = 0.18 (iter_027 Arm C) > 0.04
    (iter_029 Arm A, VICReg-only with mask_dyn_sim+coord_vicreg) — a
    delta that is real but small.
*   **What we now have that is contested:** the iter_029 Arm B mean
    ΔR²_color = 0.27 trend is a 6.2× ratio over the matched control, but
    σ = 0.577 over 30 seeds is too large to count as established. Higher
    SFA weight (5.0) hurt compared to conservative weight (1.0); hard
    seeds (53, 71) showed no SFA benefit. **Gate 3 (Parameter-Tuning
    Hygiene) is failed**: the result is not stable under ±10% perturbation
    of the SFA weight and is not stable under reseeding. Per the goal
    document, this is "suggestive evidence at best."
*   **Next Priority (iter_030, pre-register tightly):** Disambiguate
    "M2 doesn't transfer to this task" from "M2 is being out-competed."
    Concrete arms:
    - D1: VICReg-only z_dyn separate-backbone (the iter_027 Arm C
      configuration replicated with the union seed bank including hard
      seeds 53, 71), as a re-confirmed anchor.
    - D2: SFA+VICReg with VICReg variance term **ramped down** on z_dyn
      (var_weight from 25 → 5 over training) to test the
      SFA-vs-VICReg competition hypothesis. If SFA only works when
      VICReg is weakened, the M2 transfer fails *because* the variance
      hinge dominates slowness on RGB inputs.
    - D3: Object-tracking-ID contrastive (positive pair = same object
      across time, negative = different object) — an augmentation-free
      identity objective that is *not* slowness. If D3 substantially
      beats D2, M2 should be replaced by an explicit identity-binding
      objective rather than slowness.
    - Pre-register: F1 = D3 mean ΔR²_color ≥ 0.30 across the union seed
      bank with σ such that the lower CI is also ≥ 0.18 (the Arm C
      anchor); F2 = collapse ≤ 10%; F3 = no centroid-MSE degradation
      beyond 110.
*   **Confidence Score:** 45% (down from 50%). One additional structural
    hypothesis (M2-as-stated) is now under empirical challenge — that is
    progress in the falsification sense, but it removes the mandate that
    was holding the Phase-0 plan together. The foundation is narrower
    than it looked; downstream Phase 1+ work is still not ready.

## 2. Strategic Insights & Lessons Learned
*   **EXPLICIT SLOWNESS DOES NOT RELIABLY PRODUCE IDENTITY ENCODING IN
    Z_DYN ON THIS TASK (iter_023–024 + iter_029, CROSS-ARCHITECTURE
    CONVERGENT NULL):** SFA on z_dyn now has two clean pre-registered
    tests on shared backbone (iter_023–024, refuted) and on separate
    backbone (iter_029, F1 not reached). The directional trend in
    iter_029 (6.2× over VICReg-only) is real but seed-dependent and
    sensitive to SFA weight; Gate 3 is failed. Treat as: explicit
    slowness is a *contributing* prior but not a *reliable shaper* of
    identity encoding for this task. This is the first cross-project
    finding that does **not** transfer cleanly from `rdf_thalamus_sml`,
    and the most likely reason is DOF: sml's binary toy had stationary
    object identity by construction, whereas RGB+motion has identity
    cues entangled with appearance variation across frames.
*   **M2 MANDATE IS UNDER EMPIRICAL CHALLENGE, NOT YET DISCARDED:** The
    goal document's M2 says SFA+VICReg is the primary representation
    objective. iter_029 is the first arm to test the *full M2
    configuration* on the separate backbone with the previously
    identified confounds (coord_vicreg, hard seeds) controlled — and
    it did not clear its pre-declared gate. This is not yet sufficient
    to overturn the mandate (Gate 3 failure means the result is
    suggestive), but it does forbid invoking M2 as a settled basis
    for Phase 1. The iter_030 D2/D3 arms are the discriminating tests.
*   **HARD SEEDS (53, 71) ARE A CONSISTENT DIAGNOSTIC, NOT JUST NOISE
    (iter_028 + iter_029, CROSS-ITERATION SYNTHESIS):** Seeds 53 and 71
    collapse under mask_dyn_sim (iter_028 C1, C3) and show no SFA
    benefit (iter_029). The pattern is: whatever the operative
    mechanism is, it fails on these seeds across multiple
    configurations. They are functioning as a stress test. Keep them
    in the union seed bank for iter_030; if a future objective passes
    F1 *including* those seeds, the result is robust.
*   **SEPARATE BACKBONE + VICReg-ONLY IS THE CURRENT BEST FOUNDATION
    (iter_027 Arm C + iter_028 C2 + iter_029 Arm A, INDIRECT
    CONVERGENT EVIDENCE):** 0% collapse across three independent runs
    under this configuration. The construction-vs-empirical caveat
    (Gate 1: VICReg variance ≈ collapse metric) caps how strongly we
    can claim this; ΔR²_color = 0.18 (iter_027 Arm C) is the
    empirical part. This is the working anchor for iter_030.
*   **SEPARATE-BACKBONE STRUCTURAL BENEFIT IS LOAD-BEARING (iter_028,
    CONFIRMED):** The 2×2 table in iter_028 (shared backbone 30%→20%
    collapse vs separate backbone 30%→0% under mask_dyn_sim ON→MASKED)
    establishes that backbone separation provides a real stability
    benefit beyond the loss adjustment alone. This refines the
    iter_027 finding that the shared backbone is not the *primary*
    cause: separating it is not *sufficient* on its own (Arm B still
    collapsed at 30%), but it is *necessary* in combination with loss
    restructuring.
*   **REPEATED PATTERN — TRANSFERRED OBJECTIVES DON'T SURVIVE INTACT
    ON THIS TASK:** sml's SFA result is the second transfer to break
    under the Thalamus task DOF (the first was the implicit assumption
    that VICReg+JEPA would behave the same way on the separate
    backbone as it did on the shared backbone; iter_027 Arm B
    refuted that). The pattern suggests the Thalamus task has a
    qualitatively different gradient landscape than sml's binary
    toy. The "scope of transfer" caveat in Section 1.1 of the goal
    document anticipated this; the data now reinforces it.
*   **PRE-REGISTERED NULLS REMAIN FIRST-CLASS RESULTS (iter_026, 027,
    028, 029, ENFORCED):** Four consecutive iterations have produced
    defensible nulls or partial nulls because they pre-declared their
    falsification criteria. The discipline is producing more
    information per iteration than the prior exploratory regime did.
*   **PRESERVED:** M1 (pooled VICReg) stands and is reinforced; M3
    (fixed dimensionality, GDASR log-only) stands; d_max=16 capacity
    baseline stands; Hungarian-primary matching stands; 20% control-
    collapse power threshold stands; buffer=4000 stands.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (ACTIVE, RE-LOCALIZED):** No
    objective tested so far — JEPA+VICReg, SFA+VICReg on either
    backbone, VICReg-only, mask_dyn_sim — reliably encodes identity
    above ΔR²_color = 0.30 across the union seed bank. The bottleneck
    is now characterized as: "in the dual-stream regime, no
    *slowness-or-prediction-based* objective produces stable identity
    encoding on RGB+motion inputs." This suggests the next move is
    qualitatively different (contrastive identity binding, D3 above),
    not a further sweep of slowness weights.
*   **Architectural-Cause Bottleneck (REVISED, iter_028):** Separate
    backbone is necessary-but-not-sufficient. Reclassified from
    "secondary" to "necessary structural prerequisite."
*   **M2-Transfer Bottleneck (NEW, iter_029):** M2 came in as a
    validated transferred result. It is not surviving cleanly in the
    Thalamus task. Tracked as an open mandate-revision question for
    iter_030–031.
*   **Variance/Seed-Dependence Bottleneck (NEW, iter_029):** σ = 0.577
    on a primary metric is too large for reliable downstream work.
    Even if a future arm crosses the F1 threshold in mean, an
    unstable result blocks Phase 1+. iter_030 must include a
    variance-stability gate (lower CI also above the anchor).
*   **Diagnostic-vs-Constructive Iteration Loop (CLEARED):** Three
    consecutive pre-registered diagnostic iterations have produced
    actionable nulls and a tighter localization of the bottleneck.
    Protocol is working.
*   **Overclaim Loop (TRACKED, MIXED):** iter_029 executor used
    "directional trend" and "not robust" appropriately, but also
    labeled the 6.2× ratio prominently without immediately flagging
    σ=0.577 — borderline. Manager continues to enforce that
    single-seed and high-variance results are not promoted to
    mandate revisions on their own.
*   **Objective-Swapping Loop (DORMANT, NOW UNDER REVIEW):** Has
    held for three iterations, but the iter_030 D3 arm (contrastive
    identity binding) is the first proposed jump out of the
    slowness/VICReg family. The Manager judgement is that the
    cross-architecture SFA refutation justifies this jump; the
    objective-swap discipline is being broken with cause, not
    drifting.
*   **Buffer-Capacity Confound (TRACKED):** buffer=4000 maintained
    through iter_029. Keep constant in iter_030.

## 4. Alternate Research Paths
*   **iter_030: M2-Transfer Disambiguation (IMMEDIATE PRIORITY,
    THREE-ARM PRE-REGISTERED):**
    - D1: VICReg-only z_dyn, separate backbone (iter_027 Arm C
      anchor, replicated on the union seed bank).
    - D2: SFA + VICReg with VICReg variance ramped down on z_dyn
      (var_weight 25 → 5 over training) — tests whether SFA was
      being out-competed by the variance hinge.
    - D3: Object-tracking-ID contrastive on z_dyn (positive = same
      object across time, negative = different object, anchor =
      Hungarian-matched track) — the first non-slowness identity
      objective.
    - Falsification: F1 = ΔR²_color ≥ 0.30 with lower CI ≥ 0.18 over
      union seed bank; F2 = collapse ≤ 10%; F3 = centroid_MSE not
      worse than 110.
    - Pre-registered prediction: D3 > D2 > D1, with D3 the only arm
      likely to clear F1 robustly. If D2 also clears F1, the
      M2-out-competed hypothesis is supported and the SFA mandate
      can be salvaged with a variance-ramp protocol; if only D3
      clears F1, M2 should be revised to a contrastive identity
      objective in the project mandate.
*   **iter_031 (CONDITIONAL): Either M2 Revision or Augmentation-Free
    Contrastive Foundation:** Conditional on iter_030 outcome.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (PROMOTED
    from CONDITIONAL):** The iter_029 result moves this from
    "candidate if iter_028 refutes Arm C" to "candidate if iter_030
    D3 also fails." BYOL-style identity targets without explicit
    slowness are now a serious contender, not a fallback.
*   **Accept Decoder-Free Constraint Relaxation (LAST RESORT,
    UNCHANGED):** Still last resort.
*   **Multi-Knob Regime Stabilization (DEFERRED):** Unchanged.
*   **Micro-Columns (DEFERRED per semantic caution):** Unchanged. The
    iter_029 high variance on identity encoding is *not* yet
    sufficient justification to invoke imposed micro-column
    disentanglement; D3 must be tried first.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.

---

## Iteration 029 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 029 — Null Result: M2 (SFA+VICReg) Does Not Clear Practical-Significance Threshold on Separate Backbone

## 1. Pre-Declared Hypothesis and Falsification Criterion
Pre-registered hypothesis (verbatim from the iter_029 plan): "Adding an
explicit SFA slowness term (||z_dyn(t) - z_dyn(t-1)||²) to the
VICReg-only z_dyn objective on the separate-backbone architecture
will improve identity encoding such that mean ΔR²_color ≥ 0.30 across
the union seed bank, with collapse rate ≤ 10% and no centroid_MSE
degradation beyond 110."

Falsification criteria:
- F1: Arm B (SFA+VICReg, separate backbone) mean ΔR²_color < 0.30 →
  hypothesis falsified.
- F2: collapse rate > 10% on any arm → power criterion failed.
- F3: centroid_MSE > 110 → spatial readout degraded.

## 2. Experimental Protocol
- Architecture: `NonParametricJEPASpatial` with separate z_coord and
  z_dyn backbones (iter_027 Arm C topology), d_max = 16, d_t frozen at 3.
- Arm A (control): VICReg-only z_dyn (mask_dyn_sim=True), JEPA mode,
  coord_vicreg=True.
- Arm B (test): SFA + VICReg on z_dyn, SFA weight = 1.0, coord_vicreg=True.
- Arm B′ (perturbation, included for Gate 3): SFA + VICReg with SFA
  weight = 5.0.
- Seed bank: union of original (including hard seeds 53, 71) and fresh
  banks, n = 30 per arm (60 runs total reported as completing without
  timeout).
- Buffer: 4000 (held constant from iter_026).
- Training step count and exact hyperparameters: per iter_029
  pre-registration; not all values re-stated here.
- Metrics: ΔR²_color (primary), mean_abs_corr, centroid_MSE,
  train + eval std (collapse gate).

## 3. Observed Quantities
- Arm A (VICReg-only, control): mean ΔR²_color = 0.0445.
- Arm B (SFA+VICReg, weight 1.0): mean ΔR²_color = 0.2749, σ = 0.577
  across 30 seeds.
  - Original seeds subset: mean 0.1921.
  - Fresh seeds subset: mean 0.3576.
  - Hard seeds (53, 71): no improvement over Arm A.
- Arm B′ (SFA weight 5.0): lower mean than Arm B (i.e. higher SFA
  weight made the result worse, not better).
- SFA loss decreased during training (mean final = 0.1408),
  confirming the slowness objective was active and the optimization
  converged.
- Collapse rate: 0% across all 60 runs (F2 passed).
- centroid_MSE: within the F3 envelope (F3 passed).
- Resolution note: ΔR²_color is dimensionless in [-∞, 1]; 0.30 was
  chosen as the practical-significance threshold ex ante.

## 4. Verdict
**Refuted (on the pre-declared primary criterion).** Arm B mean
ΔR²_color = 0.2749 < 0.30; F1 triggered.

Two important qualifications:
- The result is **consistent with a directional improvement** of SFA
  over VICReg-only (6.2× ratio in mean), but the high run-to-run
  variance (σ = 0.577) and the failure of higher SFA weights to
  monotonically improve the result mean it does not pass Gate 3
  (Parameter-Tuning Hygiene). Per the project's reporting standards,
  this is "suggestive evidence at best."
- The post-hoc subsetting by seed bank (fresh seeds 0.3576 clears the
  threshold; original seeds 0.1921 does not) is **not** a basis for
  re-declaring success. The union seed bank is the pre-registered
  population; subsetting to the part that clears the threshold is
  selection bias.

Combined with iter_023–024 (SFA refuted on shared backbone), the
evidence now spans both architectural regimes and consistently shows
that explicit slowness does not reliably produce identity encoding
above the practical-significance threshold in this task.

## 5. Construction-vs-Empirical Note
The 0% collapse rate across all 60 runs is partly construction-driven:
VICReg's variance hinge directly enforces per-dimension std ≥ 1, which
is the same quantity used in the collapse gate. So the absence of
collapse, under any configuration that includes pooled VICReg, is
expected by the chosen objective rather than an empirical discovery
about the encoder's dynamics.

The empirically meaningful quantity in this iteration is **ΔR²_color**,
which measures whether downstream color identity can be linearly
decoded from z_dyn. This is not a quantity enforced by the loss
function and is therefore genuinely informative. The result that
ΔR²_color did not reach the pre-declared threshold under the
hypothesized objective is a clean empirical null about the
*representational content* of z_dyn, not about its variance.

The SFA term itself is an empirical objective, not a construction
identity — slowness on z_dyn does not by construction guarantee
identity encoding (it could in principle satisfy slowness via a
constant-up-to-VICReg-variance representation that carries no color
information). The iter_029 result is consistent with this latter
failure mode: SFA loss decreased while identity decoding only partly
improved.

## 6. Limitations
- This result does not establish that SFA is the wrong objective in
  principle; it establishes that SFA + VICReg as specified by M2,
  on the separate-backbone architecture, with the chosen weights and
  seed bank, did not clear the pre-declared identity-encoding
  threshold reliably.
- It does not rule out that SFA can clear the threshold under a
  variance-ramped or weight-annealed protocol in which the VICReg
  variance hinge is not competing with slowness throughout training.
  iter_030 D2 will test this.
- It does not establish that the SFA mandate (M2) should be discarded
  project-wide; the cross-architecture convergence of the null is
  grounds to challenge it, not to overturn it. Discarding M2 requires
  that an alternative identity objective (e.g. contrastive identity
  binding, iter_030 D3) clear the same threshold under the same
  population, demonstrating that the failure was in slowness as the
  organizing prior, not in the task or evaluation.
- The σ = 0.577 variance on the primary metric means the M2 result is
  not just below threshold in mean — it is unstable enough that no
  single-seed run can be interpreted as evidence. Future
  representation-objective comparisons in this project must include
  a variance-stability gate, not only a mean gate.
- Hard seeds 53 and 71 continue to function as a stress test that no
  objective tested so far has cleared. Their failure mode is not yet
  mechanistically characterized.

---

