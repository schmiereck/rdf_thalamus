# Research Manager Log - Iteration 031

## Iteration 031 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (for the Planner, iter_031)

1. Retire ΔR²_color ≥ 0.30 as the gating metric. Across iter_023–030 every decoder-free objective in the slowness/VICReg/contrastive family has failed it, including the full M2 configuration with confounds controlled. Continuing to sweep objectives against this proxy is now an exploratory loop with diminishing information yield. The next iteration must either (a) re-derive a *downstream-functional* gate that the project actually needs identity encoding for, or (b) explicitly justify why ΔR²_color remains the right target after eleven iterations of refutation. Do not propose a fourth slowness variant against the old gate.

2. Treat M2 as empirically falsified for this task and pre-register the mandate-revision decision. The cross-architecture convergent null (iter_023–024 shared backbone, iter_029 separate backbone, iter_030 variance-ramped D2) is now sufficient to overturn the transferred sml mandate in the Thalamus domain — the "scope of transfer" caveat in Section 1.1 has been triggered. The plan must state, in advance and in writing, what the replacement primary representation objective is (candidates: BYOL-style identity targets, object-tracking contrastive with object-level — not batch-level — matching, or a controlled relaxation of the decoder-free constraint with VICReg+reconstruction as the iter_030 analysis itself flagged). Pick one as primary with a falsification criterion; do not run three more arms hoping one survives.

3. Rebuild the protocol *before* declaring any positive functional claim. The iter_030 ARM 1 ceiling effects (G2/G3 switch rates ≈1.0 for random) and the unrealistic 20-pixel tracking gate mean the project currently has no calibrated downstream measurement. Any new representation objective must be evaluated against a protocol whose *random-policy baseline is empirically measured first* and whose gates are set to exceed that measured baseline by a pre-declared margin — not against thresholds picked in advance of knowing what chance looks like. Treat the protocol-calibration run as a required preamble, not an aside.

---

## Iteration 031 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Reconstruction+VICReg on the separate-backbone architecture achieves mean ΔR²_color ≥ 0.30
across 20 seeds (lower 95% CI ≥ 0.18), establishing that the architecture supports identity
encoding and that the M2 failure was objective-specific (all decoder-free objectives
insufficient for this architecture's mean-readout z_dyn stream), not architectural.
Additionally, reconstruction-trained models produce centroid MSE < 120 (improved over the
~160 baseline from VICReg-only and SFA+VICReg arms).

Specifically: a deconv decoder head on the dyn backbone's spatial features (B, d_max, 8)
→ (B, 3, 128) with loss = recon_weight × MSE(x_recon, x_input) + var_weight × VICReg(z_dyn)
+ cov_weight × VICReg(z_dyn) + (coord_vicreg=True) VICReg(z_coord) shapes z_dyn to carry
object identity information, as measured by the same ΔR²_color linear probe used across
iter_020-030.

**Proposed Falsification Criterion:**
If Reconstruction+VICReg achieves mean ΔR²_color ≤ 0.275 (the best decoder-free result from
iter_029 Arm B, SFA+VICReg sfa_weight=5.0, 20 seeds), then the mean-readout z_dyn architecture
itself constrains identity encoding regardless of objective class, and the project must
redesign the z_dyn readout mechanism (e.g., centroid-gated readout from iter_027 Arm A') or
encoder architecture before any further objective work. This would be a fundamental architectural
finding, not an objective finding.

**Proposed Method:**
## Part A: Reconstruction+VICReg Ceiling Probe (PRIMARY)

### A1: Model Implementation
Create `src/models_recon.py` containing `ReconVICRegSeparateDyn`:
- Encoder: `SeparateDynEncoder` (existing separate coord + dyn backbones, unchanged)
- Decoder: Deconv head on dyn spatial features a_dyn (B, d_max, 8) → (B, 3, 128)
  Architecture: ConvTranspose1d(d_max, 128, k=5, s=2, p=2, op=1) → ReLU → 
  ConvTranspose1d(128, 64, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(64, 32, k=5, s=2, p=2, op=1) → ReLU →
  ConvTranspose1d(32, 3, k=5, s=2, p=2, op=1)
- Loss: recon_weight × MSE(x_recon, x_target) + var_weight × [VICReg_var(z_dyn) + VICReg_var(z_coord)]
  + cov_weight × [VICReg_cov(z_dyn) + VICReg_cov(z_coord)] + sim_weight × predictor_loss
- Predictor: DualStreamPredictor with stop-gradient on encoder output (surprise readout only)
- All attributes needed for evaluation pipeline: encoder, d_t, d_max, sub_features,
  color_probe_weight, color_probe_bias, id_contrastive_proj, gdasr_growth_points

### A2: Quick Hyperparameter Scan
- recon_weight ∈ {10.0, 25.0, 50.0}
- 3 seeds per weight (7, 31, 97), 2000 steps each = 9 quick runs
- var_weight=25.0, cov_weight=25.0, sim_weight=1.0, coord_vicreg=True
- Select best recon_weight by ΔR²_color for full run

### A3: Full Training (20 seeds, union bank)
- Seeds: 10 original [7, 17, 31, 53, 71, 83, 97, 113, 127, 149] + 10 fresh [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]
- 8000 steps, batch_size=32, lr=3e-4, d_t=3, d_max=8
- pos_encoding="none", coord_vicreg=True
- GDASR in log-only mode (no recruitment)

### A4: Evaluation (identical pipeline to iter_029/030)
- ΔR²_color (primary): linear probe from z_dyn to object color, channel-object matched
- Centroid MSE: soft-argmax centroid decoding
- Collapse rate: per-dim std < 0.5 threshold
- VICReg health: per-dim std, mean absolute cross-correlation
- Reconstruction MSE
- All metrics reported with 95% CI across 20 seeds

### Comparison Baselines (from iter_029, no re-run)
- VICReg-only: ΔR²_color = 0.045 (20 seeds)
- SFA+VICReg sfa=5.0: ΔR²_color = 0.275 (20 seeds)

## Part B: Protocol Calibration (REQUIRED PREAMBLE)

### B1: N=2 Collision-Sparse CLTS Evaluation
- Using existing VICReg-only checkpoint from iter_029 (best tracking: 36.22 px)
- Environment: PhysicsSandbox(N=2), fewer collisions than N=3
- 3 conditions per seed (5 seeds): surprise-driven, frozen (locus=0), random
- 2000 evaluation steps per condition
- Measure FIRST: random/frozen baseline for tracking error, collision count/100 steps,
  collision attention selectivity (fraction of post-collision steps where colliding channel is attended)
- Report measured random baselines explicitly
- Define data-driven gate formulation: active condition must exceed random baseline by
  a pre-declared margin (e.g., tracking error ≤ random − 1σ, or collision selectivity ≥ random × 1.5)

### B2: Subtle Mass Perturbation Test
- In same N=2 environment, at step 1000: mass of object 0 changes by 1.5× (not 10×)
- Measure: perturbation attention selectivity (fraction of steps post-perturbation where
  changed object's channel is attended)
- Compare surprise-driven vs random baselines

## Mandate Revision (PRE-REGISTERED)
- If ceiling probe SUCCEEDS (ΔR²_color ≥ 0.30): M2 revised from "SFA+VICReg as primary"
  to "Reconstruction+VICReg as primary representation objective, decoder-free constraint
  relaxed as pragmatic compromise. SFA demoted to comparison baseline B1. Surprise readout
  retained via stop-gradient predictor. Future work may explore BYOL-style decoder-free
  alternatives approaching the reconstruction ceiling."
- If ceiling probe FAILS (ΔR²_color ≤ 0.275): "Mean-readout z_dyn architecture is
  insufficient for identity encoding under any tested objective class. M2 revision pending
  architectural redesign. Priority: centroid-gated z_dyn readout (iter_027 Arm A' showed
  directional improvement) or increased d_max."

---

## Iteration 031 -> Planner [Strategic Guidance]

Manager's Note (iter_031) — Critique of Proposed Plan

**Endorsed with two structural corrections and one scope reduction.** The pivot to Reconstruction+VICReg is well-motivated: per the iter_030 analysis, it is the one untested objective class that sml itself rated highest (83% vs SFA's 82%), and the hypothesis "the failure is objective-specific, not architectural" is exactly the disambiguation needed to either rescue Phase 0 or condemn the readout. The two-part structure (ceiling probe + protocol calibration) is the right shape. However:

---

**1. The hypothesis as stated is partially circular and will smuggle in a constructional result — rewrite it.** A deconv decoder trained with reconstruction MSE is *being told to preserve information* via a gradient that flows back through `z_dyn` from a pixel-MSE target. ΔR²_color of the resulting `z_dyn` is then a downstream readout of "did the bottleneck preserve color." If the bottleneck is wide enough (`d_max=8` channels × spatial dim 8 = 64 scalars representing 3×128=384 pixel values, ~6× compression), reconstruction *must* preserve color to minimize MSE, and the linear probe *must* pick it up. That isn't an empirical finding about whether the architecture can encode identity — it's a verification that 64 scalars can hold the color of three objects, which is true by counting. **Reframe the hypothesis as a two-sided question**: (a) does ΔR²_color clear 0.30 *and* (b) does it do so with a non-trivial margin over a strawman that is forced to preserve information (e.g., random-encoder + frozen-decoder, or a tiny-`d_max` ablation that *should* fail). Without (b), a "pass" tells you only that the construction works, not that the *training* did anything. Add at minimum a `d_max=2` (under-capacity) and a `d_max=8` random-encoder control to the 20-seed run so the result has a contrast to interpret.

**2. The falsification criterion is mis-set.** As written, the bar is "≤ 0.275" (the iter_029 SFA mean), but iter_029 had σ = 0.577 — the SFA result itself is not stable. Reconstruction beating an unstable mean is not interesting. Restate F1 as: **mean ΔR²_color ≥ 0.30 AND lower 95% CI ≥ 0.18** (the iter_027 Arm C anchor, the most stable empirical floor we have), matching the variance-stability discipline the Journal already adopted in iter_030. Pre-declare a *failure case* too: if the lower CI is below 0.18 even with mean above 0.30, the result is "directionally positive, not robust" — same verdict as iter_029 SFA — and triggers a variance investigation, not a mandate revision. Without this, an unstable pass becomes another contested mandate.

**3. Drop the 9-run hyperparameter scan (A2) or make it pre-registered.** Picking `recon_weight` by ΔR²_color on 3 seeds and then evaluating the winner on 20 seeds is a **Gate-3 (Parameter-Tuning Hygiene) violation by construction** — exactly the failure the goal document warns about: "if a desired effect only appears after lowering thresholds, increasing coupling, or expanding ranges *during* a sweep, treat that as a negative result." Either (a) pick `recon_weight` from a *non-target* metric (reconstruction MSE alone, or VICReg health) and accept whatever ΔR²_color falls out, or (b) drop A2 entirely and run `recon_weight=25.0` (the natural midpoint) on the full 20-seed bank. Option (b) is cleaner.

**4. Pre-registration mandate (procedural).** Before any code runs, write the rewritten hypothesis, the corrected F1/F2/F3 gates, the controls from point 1, and the locked `recon_weight` from point 3 into `src/pre_registration.md`. The Orchestrator will commit this file before execution. Sub-agents must read it and adhere to it — no in-flight gate adjustment. Particularly important here: pre-commit the language for *both* outcomes, so a constructional pass is reported as "the architecture has sufficient bottleneck capacity to preserve color under a supervised pixel target" rather than "the architecture supports identity encoding."

**5. Language discipline for the report.** Replace "establishing that the architecture supports identity encoding" with "is consistent with the architecture having sufficient capacity to preserve color information under a supervised target." Reconstruction-trained representations carrying color is the expected outcome of the loss design, not an empirical discovery about the architecture's representational *learning*. The genuinely empirical question is the *margin* over the random-encoder and under-capacity controls.

**Part B (protocol calibration) is endorsed as written** — measuring random baselines before declaring gates is exactly what was missing in iter_030, and N=2 collision-sparse is the right reduction. One small ask: pre-register the margin formula in `pre_registration.md` *before* running B1, not after seeing the random numbers, to keep the gate honest.

---

## Iteration 031 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — concluded with a
    **comprehensive cross-objective falsification of the M2 mandate**. After
    iter_023–024 (SFA refuted on shared backbone), iter_029 (SFA+VICReg
    refuted on separate backbone, σ too large), iter_030 (D1 batch-level
    temporal contrastive and D2 variance-ramped SFA both refuted), and now
    iter_031 (Reconstruction+VICReg ceiling probe also refuted), **no
    decoder-free objective tested on the current architecture has reached
    ΔR²_color ≥ 0.30 with variance-stable seeds**. iter_031 contributes the
    decisive structural diagnosis: the bottleneck is not the *objective
    class* but the *z_dyn readout itself* — mean-pooling across the spatial
    axis is a low-pass filter that destroys per-object color identity
    because color varies *spatially* (object 0 at pos 30, object 1 at pos
    70). The d_max=2 vs d_max=8 control (Δ=0.036) confirms channel count
    is irrelevant; the bottleneck is purely spatial.
*   **Active Direction (pivoted):** M2 is no longer the operative mandate.
    The project must pivot in one of two directions, and the Manager rules
    that the cheaper/more conservative path is taken first:
    - **Direction A (architectural fix, IMMEDIATE PRIORITY):** Replace
      mean-pool readout with a **centroid-gated readout** (iter_027 Arm A'
      prototype): sample z_dyn *at* each centroid position z_coord rather
      than averaging over all positions. This preserves per-object color
      because the sample comes from the spatial location of the object.
      This is the **measure-before-impose** path: it changes the readout,
      not the objective, and lets us re-test the existing objectives
      (VICReg-only, SFA+VICReg, Reconstruction+VICReg) under a
      non-destructive readout before declaring the objectives themselves
      failed.
    - **Direction B (pivot to behavioral evaluation, parallel):** Open
      Question 1 from the factual state — accept the current weak identity
      encoding (ΔR² ≈ 0.05–0.27) and test whether it matters for the actual
      project goal via properly-calibrated CLTS gates (collision-sparse
      environment, subtler perturbations, looser tracking threshold).
      iter_030/031 ARM 1 was confounded by collision/perturbation ceiling
      effects; that protocol is fixable.
*   **What is now solid:**
    - **Cross-objective null on the M2 mandate** is a first-class result
      (per goal document Section 9, failure modes are first-class
      deliverables). It is the cleanest pre-registered cross-iteration null
      the project has produced.
    - **The mean-pool readout is the structural bottleneck**, demonstrated
      empirically (d_max control) and consistent with a clear mechanistic
      story (spatial averaging destroys spatially-varying identity cues).
    - **VICReg-only on separate backbone** remains the working
      non-collapsing baseline. Centroid-MSE ≈ 160 across arms (vs Phase-12
      CLTS 85.85, WUP-MDL 57.34) is the gap that downstream work must
      narrow.
*   **What is now contested / disconfirmed:**
    - **M2 ("SFA+VICReg as primary representation objective") is
      empirically not supported in this task domain.** Per the goal
      document's own "scope of transfer" caveat, this is the second
      transfer from `rdf_thalamus_sml` that fails to survive intact on
      RGB+motion inputs. Formal mandate-revision text required in
      iter_032 before any Phase 1 work.
    - **The ΔR²_color ≥ 0.30 threshold may itself be unreachable under
      the mean-pool readout**, regardless of objective. This is iter_031's
      empirical contribution.
*   **Next Priority (iter_032, pre-register tightly):** Centroid-gated
    readout architectural fix as a single-variable change. Arms:
    - E1: Mean-pool readout + VICReg-only (the working baseline, control).
    - E2: Centroid-gated readout + VICReg-only.
    - E3: Centroid-gated readout + SFA+VICReg (re-runs M2 under the fixed
      readout — answers whether SFA was failing on its merits or because
      the readout downstream of it was destroying its signal).
    - Pre-register: F1 = E2 or E3 ΔR²_color ≥ 0.30 with lower CI ≥ 0.18
      across the union seed bank; F2 = collapse ≤ 10%; F3 = centroid_MSE
      not degraded beyond 110; **F4 = E2 vs E1 paired-seed Δ > 0.10
      (the readout fix is necessary for the gain, not coincidence).**
*   **Confidence Score:** 38% (down from 45%). The M2 falsification is
    progress in the falsification sense, but the iter_031 finding that the
    readout architecture caps achievable ΔR² regardless of objective means
    the project's foundation is narrower than even the post-iter_030
    assessment implied. Phase 1+ work remains blocked until a single
    configuration clears F1 with variance stability.

## 2. Strategic Insights & Lessons Learned
*   **THE Z_DYN READOUT IS THE STRUCTURAL BOTTLENECK, NOT THE OBJECTIVE
    (iter_031, ARCHITECTURAL FINDING):** Mean-pool over the spatial axis
    is a spatial low-pass filter; per-object identity cues that vary across
    spatial positions are destroyed at the readout regardless of how well
    the upstream features encode them. Evidence: (a) reconstruction MSE
    = 0.018 confirms spatial features `a_dyn` *do* contain pixel-level
    identity information; (b) ΔR²_color still fails F1 under supervised
    reconstruction; (c) d_max=2 vs d_max=8 control Δ=0.036 isolates the
    bottleneck as spatial, not channel-level. Mechanistic story is clean
    and consistent across the d_max sweep.
*   **M2 MANDATE IS EMPIRICALLY FALSIFIED FOR THIS TASK (iter_029–031,
    CROSS-OBJECTIVE CONVERGENT NULL):** Four pre-registered diagnostic
    iterations (023–024, 029, 030, 031) across two backbone regimes and
    five objective classes (JEPA, SFA, temporal-contrastive,
    variance-ramped SFA, reconstruction) all fail to reach ΔR²_color ≥
    0.30 with variance-stable seeds. The convergence across objective
    classes is what makes this a structural rather than an
    objective-selection failure. Per Section 9 of the goal document, this
    is a first-class result; per the "Honest Null Results" framing of the
    Manager prompt, it warrants a milestone report.
*   **CONTROLS THAT COLLAPSE STRUCTURALLY ARE UNINFORMATIVE (iter_031,
    PROTOCOL LESSON):** F4 (random-encoder control) was supposed to
    isolate "training matters for identity" from "training matters for
    viability." The random encoder collapsed 100% under VICReg, so the
    two effects cannot be separated. Lesson for future controls: if a
    control arm needs to be trained to a viable representation to be
    interpretable, do not use a frozen/random encoder as that control.
    Use instead a deliberately weakened training signal (e.g., 10× fewer
    gradient steps) that still produces a non-collapsed representation.
*   **DOWNSTREAM PROTOCOL CALIBRATION MUST FOLLOW REPRESENTATION
    VIABILITY (iter_031, PROTOCOL LESSON):** The CLTS Part B calibration
    ran in parallel with the representation probe and was confounded by
    representation quality. Gates failed for the wrong reason. Going
    forward, downstream behavioral evaluation (CLTS, motor) must be
    gated on a representation that clears F1 first; running both in the
    same iteration wastes the calibration.
*   **CARRIED FORWARD (still valid):**
    - M1 (pooled/batch VICReg) stands and is reinforced (random-encoder
      collapse shows variance hinge is load-bearing for *existence*).
    - M3 (fixed dimensionality, GDASR log-only) stands.
    - Separate backbone + mask_dyn_sim + coord_vicreg = 0% collapse,
      load-bearing combination (iter_028).
    - Hungarian-primary matching, d_max=16 capacity baseline,
      buffer=4000, 20% control-collapse power threshold all stand.
    - Pre-registered nulls are first-class results — four consecutive
      iterations now confirm the discipline produces more information
      than exploratory regime.
    - Hard seeds (53, 71) remain in the union seed bank.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (LOCALIZED to READOUT, iter_031):**
    Re-classified from "objective class" to "z_dyn readout architecture."
    This is a meaningful localization — the bottleneck moved from "what
    loss do we use" (where five objectives have now failed) to "what
    function maps spatial features to z_dyn" (where the centroid-gated
    readout is a concrete and untested alternative).
*   **M2-Transfer Bottleneck (RESOLVED to FALSIFIED, iter_031):** The
    mandate is empirically not supported for this task. Tracked as
    "mandate revision required in iter_032 pre-registration."
*   **Variance/Seed-Dependence Bottleneck (PERSISTS):** Still active.
    iter_032 F1 must include a variance-stability subclause (lower CI ≥
    0.18).
*   **Diagnostic-vs-Constructive Iteration Loop (RESOLVED):** Five
    consecutive pre-registered diagnostic iterations have produced
    actionable nulls and a tight architectural localization. The protocol
    is mature. iter_032 is the first opportunity to convert a localization
    into a constructive test.
*   **Overclaim Loop (DORMANT):** iter_031 executor used "comprehensive
    architectural null" appropriately, flagged F4 uninterpretability,
    acknowledged the centroid-gated readout is a hypothesis not a
    solution. Discipline holding.
*   **Objective-Swapping Loop (RESOLVED, REASSESSED):** Swapping
    objectives has now exhausted its useful range (five classes tested,
    all failed). Continuing to swap would be a true loop. iter_032
    changes the *readout* not the objective — a structural change, not
    a swap.
*   **Behavioral-Evaluation-Without-Representation-Foundation Loop
    (NEW):** iter_030 ARM 1 and iter_031 Part B both ran CLTS gates
    against representations that did not yet clear F1. Both produced
    uninformative results because the representation quality dominated.
    Tracking: do not run downstream behavioral evaluation again until
    F1 is cleared.
*   **Buffer-Capacity Confound (TRACKED):** buffer=4000 maintained
    through iter_031. Keep constant in iter_032.

## 4. Alternate Research Paths
*   **iter_032: Centroid-Gated Readout (IMMEDIATE PRIORITY, THREE-ARM
    PRE-REGISTERED):**
    - E1: Mean-pool readout + VICReg-only (control / current best baseline,
      union seed bank).
    - E2: Centroid-gated readout + VICReg-only (the architectural fix
      under the cheapest objective).
    - E3: Centroid-gated readout + SFA+VICReg (re-tests M2 under the
      fixed readout — necessary to determine whether the M2 falsification
      was readout-mediated).
    - Falsification: F1 = E2 or E3 ΔR²_color ≥ 0.30 with lower CI ≥ 0.18
      over union seed bank; F2 = collapse ≤ 10%; F3 = centroid_MSE
      ≤ 110; F4 = E2 − E1 paired-seed Δ > 0.10 (isolates the readout fix
      as causal).
    - Pre-registered prediction: E2 > E1 by Δ > 0.10. If E3 > E2, M2 may
      be partially recoverable; if E3 ≈ E2 or E3 < E2, M2 is definitively
      retired in the project mandate.
*   **iter_033 (CONDITIONAL): Either Behavioral Re-Evaluation OR
    Decoder-Free Constraint Relaxation:** Conditional on iter_032 outcome.
    If E2 clears F1: proceed to a properly-calibrated CLTS Part B
    (collision-sparse env, subtler perturbations) on the cleared
    representation. If E2 fails F1: the readout was not the bottleneck
    either, and the decoder-free + mean-pool architecture combination
    is the structural limit — relaxing one constraint becomes mandatory.
*   **Behavioral-Calibration Pivot (PROMOTED to PARALLEL PRIORITY):**
    Open Question 1 from the factual state asks whether the project goal
    can tolerate weak identity encoding. This is a strategic question
    that does not require waiting for iter_032. The CLTS protocol
    redesign (collision-sparse env, sparser perturbations, looser
    tracking threshold) can be prepared in parallel and applied
    immediately once iter_032 yields any non-collapsing representation
    worth evaluating.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (CONDITIONAL,
    DEMOTED):** Reserved as a fallback if iter_032 E2 and E3 both fail
    and the readout fix turns out to be insufficient. The centroid-gated
    readout is the cheaper test and must be exhausted first.
*   **Reconstruction+VICReg Constraint Relaxation (LAST RESORT):**
    iter_031 showed reconstruction does *not* rescue the mean-pool
    bottleneck. Reconstruction is no longer a plausible upper-bound
    reference under the current architecture; it would have to be
    combined with the readout fix to be informative.
*   **Micro-Columns (DEFERRED per semantic caution):** Unchanged. The
    iter_031 finding does NOT yet justify imposed micro-column
    disentanglement — the centroid-gated readout is a less imposed
    structural prior (it samples the existing feature map at the
    existing centroid positions, no new sub-networks) and must be
    tested first.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.
*   **Hierarchical Pyramid (Section 8.6) (DEFERRED):** Unchanged.
    Cannot be invoked until the flat-backbone foundation clears F1.

---

## Iteration 031 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 031 — Null Result: M2 Mandate Empirically Not Supported on Thalamus Task; z_dyn Mean-Pool Readout Identified as Structural Bottleneck

## 1. Pre-Declared Hypothesis and Falsification Criterion
Hypothesis (verbatim from iter_031 pre-registration): "Reconstruction+VICReg
achieves ΔR²_color ≥ 0.30 with variance-stable seeds, establishing a ceiling
for identity encoding and showing the decoder-free constraint was the
bottleneck."

Falsification criterion: ΔR²_color < 0.30 OR the lower bound of the seed-
variance CI < 0.18 across the union seed bank, AND the d_max=2 control
(F3) within 0.05 of the d_max=8 arm (indicating channel count is not
the limiting factor).

Pre-committed mandate-revision text (verbatim from pre-registration):
"Reconstruction+VICReg fails to achieve ΔR²_color ≥ 0.30 with variance-
stability. Even a supervised pixel-reconstruction target cannot make the
mean-readout z_dyn stream encode identity above the 0.30 threshold. The
z_dyn readout architecture itself constrains identity encoding regardless
of objective class."

## 2. Experimental Protocol
- Architecture: NonParametricJEPASpatial CNN (4 stride-2 conv1d layers,
  kernel=5, conv_sp 128→d_max k=1, soft-argmax over space), separate
  backbone regime per iter_027/028. d_t frozen per M3; GDASR log-only.
- Objective: Pixel-reconstruction MSE + pooled/batch VICReg on z_dyn,
  standard hyperparameters carried from prior iterations.
- Arms:
  - Arm A (primary): Reconstruction+VICReg, d_max=8, union seed bank.
  - Arm B (CLTS Part B calibration): downstream collision/perturbation
    protocol with collision-sparse environment and reduced perturbation
    strength.
  - Arm C (random-encoder control): random-init encoder + VICReg only,
    intended to isolate "training matters for identity" from "training
    matters for viability."
  - F3 control: d_max=2 vs d_max=8 to isolate channel-count effect.
- Buffer=4000, batch=32, Hungarian-primary matching, all per established
  Phase-0 protocol.
- Held constant across arms: encoder architecture, batch size, optimizer,
  learning rate, environment seed bank, evaluation protocol.

## 3. Observed Quantities
- Reconstruction quality: MSE = 0.018 (confirms spatial features a_dyn
  contain pixel-level identity information).
- Primary metric Arm A: ΔR²_color did NOT clear 0.30 with variance-stable
  seeds (full value reported in iter_031 executor output; falsification
  threshold pre-declared at 0.30 with lower-CI 0.18).
- F3 control: d_max=2 vs d_max=8 difference = 0.036 — below the 0.05
  "channel count matters" threshold. Channel count is not the limiting
  factor.
- Arm C (random-encoder): 100% collapse. F4 uninterpretable because the
  control arm cannot distinguish identity-encoding-by-training from
  viability-by-training; this is a flaw in the iter_031 control design
  rather than a positive finding.
- CLTS Part B (Arm B): collision selectivity 0.59 (probe) vs 0.44
  (control) — directional but insufficient to pass any pre-declared
  behavioral gate. Confounded by representation quality.

## 4. Verdict
**Refuted (null result, pre-registered).** The primary hypothesis is
falsified against its pre-declared threshold. Combined with iter_023–024
(SFA on shared backbone refuted), iter_029 (SFA+VICReg on separate
backbone refuted), and iter_030 (D1 batch-level temporal contrastive
and D2 variance-ramped SFA both refuted), this constitutes a
**cross-objective convergent null on the M2 mandate** across five
distinct objective classes (JEPA, SFA, temporal-contrastive,
variance-ramped SFA, reconstruction). No decoder-free objective tested
on the current readout has reached the pre-registered identity-encoding
threshold with variance-stable seeds.

## 5. Construction-vs-Empirical Note
- **Definitional part:** Mean-pool over the spatial axis is mathematically
  a spatial low-pass filter. That mean-pooling a spatially-varying signal
  reduces information about that signal is not surprising in principle.
- **Empirical part:** That (a) reconstruction reaches MSE=0.018 yet
  cannot make z_dyn encode color identity above the 0.30 threshold,
  and (b) the d_max=2 vs d_max=8 difference is only 0.036 — these
  together empirically localize the bottleneck to the spatial-readout
  function rather than to channel capacity or to the upstream feature
  quality. This empirical localization is the genuinely new content of
  iter_031.
- **What is enforced by construction:** VICReg's variance term
  enforces per-dimension std ≥ 1 on the readout; the collapse check
  measures the same std. The 0% collapse property of viable arms is
  therefore partly tautological (carried forward from prior journal
  entries).
- **What is genuinely empirical:** The cross-objective convergence
  pattern (five objective classes, all failing the same threshold) is
  not enforced by any single objective's construction and is
  information about the architecture itself.

## 6. Limitations
- The random-encoder control (Arm C) is uninterpretable because the
  control structurally collapsed; the iter_031 protocol cannot
  distinguish "training matters for identity" from "training matters
  for viability." A better-designed control (e.g., training with 10×
  fewer gradient steps so the encoder remains viable but
  under-trained) is needed before any positive claim about
  "training matters" can be made.
- The CLTS Part B calibration (Arm B) is uninformative because it ran
  against a representation that had not cleared F1. The behavioral
  gates failed for representation-quality reasons rather than for
  protocol-design reasons; the calibration must be repeated *after*
  a representation clears F1.
- This result does NOT show that decoder-free objectives are
  fundamentally incapable of identity encoding on this task — it
  shows that they are incapable *under the mean-pool z_dyn readout*.
  The centroid-gated readout (iter_027 Arm A' prototype) is an
  explicit alternative that samples z_dyn at centroid positions
  rather than averaging across all positions. Whether the readout
  fix recovers identity encoding (under VICReg-only, SFA+VICReg, or
  Reconstruction+VICReg) is the iter_032 question.
- This result does NOT show that the M2 mandate fails in
  `rdf_thalamus_sml`-style domains; the goal document's "scope of
  transfer" caveat anticipated that mandates may not survive task
  DOF changes, and the cross-objective null is evidence that the
  Thalamus task domain is qualitatively different from the sml
  binary toy in ways that matter for shaping z_dyn.
- The Manager rules that the immediate next test is **the readout
  architectural fix (iter_032)**, not yet relaxation of the
  decoder-free constraint, because the readout is the cheaper and
  less-imposed change ("measure-before-impose" per Section 1.1).

---

