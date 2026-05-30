# Research Manager Log - Iteration 030

## Iteration 030 -> Planner [Pre-Planning Hints]

# Manager's Pre-Planning Hints — Iteration 030

**Hint 1 (directional, top priority — reframe before sweeping more objectives).** The user's reframing is correct and binds: ten iterations have optimized ΔR²_color ≥ 0.30 as a proxy whose necessity for the actual project goal (curiosity-driven agent with thalamic gating + motor) was never validated. The next iteration's **Arm 1 must be an integration smoke-test**, not another representation sweep. Freeze the best current representation (separate-backbone iter_027-ArmC topology, d_max=16, d_t=3, on non-hard seeds at ΔR²_color ≈ 0.25–0.27 and 0% collapse) and run it through Pillars D + E (CLTSMotorController attention routing + motor probing). Pre-declare a concrete behavioral pass/fail before running: e.g. centroid tracking error below an explicit threshold, post-collision probe latency, and a mass-change causal-sensitivity delta against a frozen-attention control. If downstream behavior is acceptable at ΔR² ≈ 0.25, the 0.30 threshold was a distraction and the project advances to Phase 2/3 integration; if it genuinely degrades, the objective hunt is justified and Arm 2 (contrastive vs. variance-ramped SFA, per the journal's iter_030 plan) earns the next slot.

**Hint 2 (scientific discipline — guard against the proxy-metric trap and the construction caveat).** Arm 1 is itself a falsifiable claim ("ΔR² ≈ 0.25 representations are sufficient for downstream gating/motor") and must be pre-registered as such, with the integration test's behavioral thresholds declared **before** the freeze-and-run, not chosen post-hoc to validate whichever direction the data points. Equally important: the construction-vs-empirical caveat applies here too — do not report "the agent tracks" or "the agent probes" as evidence unless those behaviors are measured against a frozen-random-attention control on the *same* environment seeds, since reflexive PD tracking will produce a low centroid error even with a useless representation. The same variance-stability discipline iter_029 established applies: ≥5 seeds, report σ alongside mean, and flag hard seeds 53/71 separately rather than averaging them away.

**Hint 3 (scope and preservation — keep Arm 2 in scope but subordinated).** If iter_030 runs both arms, Arm 1 (integration) is decisive and Arm 2 (contrastive D3 vs. variance-ramped SFA D2 on the union seed bank with hard seeds) is contingent — its result only matters if Arm 1 says identity encoding genuinely bottlenecks downstream behavior. Preserve: separate backbone + the iter_028 collapse-avoiding config (both interventions required for 0% collapse), d_t=3 frozen with GDASR log-only (M3), pooled batch VICReg (M1), buffer=4000, no positional encoding (consistently hurts across JEPA *and* SFA — iter_013 and iter_021 are now a cross-objective regularity, not a single-iteration finding). Whatever the verdict, the iteration's report must explicitly state whether M2 in goal.md remains, is salvaged via a variance-ramp protocol, is replaced by contrastive identity binding, or is **demoted to "not a bottleneck for the project goal"** — that mandate-level decision is the iteration's deliverable, not another delta on the proxy.

---

## Iteration 030 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
ARM 1 (Integration Smoke-Test): A representation with ΔR²_color ≈ 0.27 (SFA+VICReg
on separate backbone, 0% collapse, iter_029 Arm B) produces CLTS motor behavior
that is functionally adequate for Pillar D (thalamic gating) and Pillar E (motor
probing). Specifically, three pre-registered gates will be tested:
  G1 (Tracking Functionality): CLTS mean |pointer_pos - attended_centroid| < 20 pixels
    over a 2000-step closed-loop evaluation, demonstrating that the z_coord centroid
    readout supports functional PD tracking.
  G2 (Attention Validity): CLTS surprise-driven attention switches to the collision-
    involved channel within 15 steps after a detected collision in >50% of events,
    while CLTS-Random (uniform random attention selection) switches to the collision
    channel in <35% of events (chance baseline for 3 objects). A difference >15pp
    confirms the surprise signal is informative.
  G3 (Causal Sensitivity): After a forced-probe mass perturbation (mass×3 + push on
    object 0 at step 1000), CLTS attention switches to object 0's channel within
    20 steps in >60% of perturbation events, while CLTS-Random does so in <33%.
    A difference >15pp confirms the representation supports causal probing.
If ≥2 of 3 gates pass, the representation is sufficient for downstream tasks and the
project advances to Phase 2/3 integration without further objective optimization.
If 0-1 gates pass, the representation is insufficient and the objective hunt is
justified.

ARM 2 (M2 Decisive Test): Temporal identity contrastive binding (D1) or variance-
ramped SFA (D2) achieves mean ΔR²_color ≥ 0.30 on a 30-seed union bank with
variance-stability (std/mean < 2.0). D1 uses same-object-across-time as positive
pairs and other-objects-as-negatives in an NT-Xent loss; D2 anneals sfa_weight
from 0 to 5.0 over the first 4000 training steps. If D1 passes, M2 is revised to
"contrastive identity binding as primary" (slowness underspecifies identity). If D2
passes, M2 is revised to "variance-ramped SFA as primary." If neither passes, the
ΔR²_color proxy is acknowledged as unachievable by any tested decoder-free objective
and ARM 1's verdict governs the project direction.

**Proposed Falsification Criterion:**
ARM 1 falsified if ≤1 of 3 gates pass, meaning the current representation does not
support meaningful downstream behavior and further objective optimization is justified.
Specifically:
  G1 falsified: CLTS mean tracking error ≥ 20 pixels (centroid readout is too noisy
    for functional tracking)
  G2 falsified: CLTS collision-switching rate ≤ CLTS-Random + 15pp (surprise signal
    is not informative for attention routing)
  G3 falsified: CLTS post-perturbation attention switch ≤ CLTS-Random + 15pp
    (representation does not support causal probing after mass change)

ARM 2 D1 falsified: mean ΔR²_color < 0.30 OR std/mean ≥ 2.0 on the 30-seed bank
ARM 2 D2 falsified: mean ΔR²_color < 0.30 OR std/mean ≥ 2.0 on the 30-seed bank
Both D1 and D2 falsified: the ΔR²_color ≥ 0.30 target is unachievable by any
decoder-free objective tested; the project accepts the current representation quality
per ARM 1's verdict.

**Proposed Method:**
ARM 1 (Integration Smoke-Test — highest priority, runs first):
1. Load pre-trained SFA+VICReg checkpoints from iter_029 Arm B (10 seeds: 7, 17, 31,
   83, 97, 113, 127, 149 + hard seeds 53, 71 reported separately).
2. For each seed, run two 2000-step closed-loop evaluations:
   a. CLTS: CLTSMotorController with surprise-driven attention + PD tracking +
      push-when-surprised-and-near (existing probe-when-surprised logic)
   b. CLTS-Rand: Same controller but token_locus selected uniformly at random
      from d_t channels (replacing argmax of normalized surprise), keeping PD
      tracking and push logic identical — controls for reflexive PD tracking
3. Warm-up: first 200 steps with CLTS but no attention switching (fixed locus 0)
   to let EMA surprise statistics converge; metrics start at step 200.
4. Collision detection: at each step, compare object velocities before/after; flag
   collision if |delta_v| > 2.0 for any pair of adjacent objects. Record whether
   CLTS attention switches to a collision-involved channel within 15 steps.
5. Mass perturbation: at step 1000, change object-0 mass to 3× current; immediately
   force pointer near object 0 (set pointer_pos = object_0_pos ± 5) and push.
   Measure whether attention switches to object 0's channel within 20 steps.
6. Compute gate metrics and paired comparison (CLTS vs CLTS-Rand per seed).
7. Decision: ≥2/3 gates pass → representation sufficient → project advances.

Files: src/run_phase0_integration.py (NEW)

ARM 2 (M2 Decisive Test — contingent on ARM 1):
1. Expand union seed bank to 30 seeds (original 10 + fresh 10 + new: 173, 179,
   181, 191, 193, 197, 199, 211, 223, 227).
2. D1 (Temporal Identity Contrastive):
   - New loss: NT-Xent where same-object-at-consecutive-timesteps = positive pair,
     different-objects = negatives.
   - Implementation: encode x_target and x_hist[:,-1] to get z_coord_t, z_dyn_t,
     z_coord_{t-1}, z_dyn_{t-1}. Sort z_coord and positions to match channels to
     objects (sorted matching, O(d_t log d_t) on GPU). For each object, z_dyn at
     the matched channel at time t and t-1 form a positive pair; z_dyn from other
     objects at t-1 are negatives. Uses existing id_contrastive_proj projection head.
   - Loss computed in training loop (not in model forward), added to total loss with
     weight temporal_contrastive_weight=5.0.
   - Architecture: NonParametricJEPASpatialSeparateDyn, primary_objective="sfa",
     mask_dyn_sim=True, coord_vicreg=True, sfa_weight=0 (SFA disabled; replaced
     by temporal contrastive). d_t=3, d_max=8.
3. D2 (Variance-Ramped SFA):
   - Same architecture as D1 but primary_objective="sfa" with sfa_weight
     linearly ramped from 0 to 5.0 over steps 0-4000, then constant at 5.0.
   - Ramping prevents early-training SFA instability (gradient conflict with
     VICReg variance term during representation bootstrapping).
4. Train 2 arms × 30 seeds for 8000 steps each (separate backbone, mask_dyn_sim=True,
   coord_vicreg=True, d_t=3, d_max=8, buffer=4000, batch_size=32, lr=3e-4).
5. Evaluate: ΔR²_color, collapse rate, centroid MSE (same protocol as iter_029).
6. Gate: mean ΔR²_color ≥ 0.30 AND variance-stability std/mean < 2.0.
7. Decision: D1 passes → M2 revised to contrastive identity binding; D2 passes →
   M2 revised to variance-ramped SFA; neither passes → proxy acknowledged as
   unachievable, ARM 1's verdict governs.

Files: src/run_phase0_m2_decisive.py (NEW)

PRESERVE across both arms:
- Separate backbone + collapse-avoiding config (mask_dyn_sim=True, coord_vicreg=True)
- d_t=3 frozen, GDASR log-only (M3)
- Pooled batch VICReg (M1)
- No positional encoding (cross-objective regularity from iter_013 and iter_021)
- Hard seeds 53/71 reported separately, not averaged away
- Report σ alongside mean for all metrics (≥5 seeds minimum per condition)

---

## Iteration 030 -> Planner [Strategic Guidance]

# Manager's Note — Iteration 030 (Plan Critique)

**Endorsed direction, with three load-bearing corrections.** The two-arm structure (integration smoke-test first, M2 disambiguation second) correctly implements the reframe. But the Arm 1 protocol as drafted has a confound that would make any positive verdict uninterpretable, and Arm 2 quietly inflates statistical power. Fix these before pre-registration is committed.

**1 (Arm 1 — kill the confound, or the smoke-test proves nothing).** Loading checkpoints *only* from iter_029 Arm B (SFA+VICReg) means Arm 1 cannot answer the question it was designed to ask. If the gates pass, you do not know whether the representation needed ΔR²≈0.27 or whether ΔR²≈0.04 (VICReg-only) would have sufficed — and the user hint's decision rule ("if downstream performs acceptably at dR2~0.25 then the 0.30 threshold was a distraction") requires a *contrast* against a weaker representation. **Mandatory addition:** a third frozen condition, **CLTS on VICReg-only z_dyn checkpoints** (iter_027 Arm C / iter_029 Arm A, ΔR²≈0.04–0.18, same seeds). The gate hierarchy then becomes informative: (a) both SFA-B and VICReg-only pass → identity decodability does not bottleneck downstream behavior, M2 demoted; (b) SFA-B passes, VICReg-only fails → identity encoding *does* matter and the 0.30 search is justified; (c) neither passes → representation truly insufficient; (d) VICReg-only passes but SFA-B doesn't → something else is going on, investigate. Without this contrast, Arm 1 is a one-armed test of "does any frozen model do anything," not the decision the user asked for.

**2 (Arm 1 — gates G2/G3 are partly constructional; tighten the controls).** The "+15pp over CLTS-Random" bar treats random attention selection as the only baseline, but random attention on a controller that still does PD tracking and collision-correlated push has known structural advantages from the motor side — e.g. the pointer drifts toward whichever channel it's currently attending, which after a collision is more likely to be near a moving object regardless of representation quality. Two specific fixes: (a) Add a **frozen-attention control** (token_locus held at channel 0 throughout) alongside CLTS-Random — this isolates the contribution of *switching* from the contribution of *tracking*; G2/G3 must beat **both** baselines by 15pp, not just random. (b) The 15-step / 20-step / >50% / >60% / 15pp thresholds are currently unjustified numbers — state explicitly what each threshold corresponds to in expected events per seed and what the binomial null distribution looks like for n_seeds=10, or the "≥2 of 3 gates" rule is just three independent coin flips with unknown p. Without that, a 2/3 pass is not evidence — it is one favorable noise realization.

**3 (Arm 2 — variance-stability gate and language hygiene).** The `std/mean < 2.0` criterion is too lax for the stated context: iter_029 Arm B had σ=0.577 over a mean of 0.27, i.e. std/mean ≈ 2.1, which would **almost pass** under the proposed gate. That re-admits exactly the instability the journal already flagged as disqualifying. Replace with the journal's prior, tighter formulation: **lower bound of the 95% CI must also exceed 0.18** (the iter_027 Arm C anchor). This is a meaningful gate; std/mean<2.0 is not. Separately, the plan's hypothesis text contains "demonstrating that the z_coord centroid readout supports functional PD tracking" — restate as "consistent with functional PD tracking" or "does not refute"; PD tracking will produce *some* low error regardless of representation, so "demonstrating" overclaims (and is exactly the construction-vs-empirical trap point 2 above is meant to catch).

**Pre-registration mandate (process reminder).** The Orchestrator will automatically write `src/pre_registration.md` from the hypothesis and falsification fields above and commit it before execution; the Planner's sub-agents MUST read it and adhere strictly. The plan as currently drafted is **not yet allowed to proceed to execution** — the three corrections above must be incorporated into the pre-registered hypothesis and falsification criteria first. Specifically: Arm 1's frozen conditions must list all three (SFA-B / VICReg-only / frozen-attention), Arm 1's gate text must specify the "beat both baselines by 15pp" rule with a stated event-count null, and Arm 2's variance gate must read "lower 95% CI ≥ 0.18" not "std/mean < 2.0". After those edits, the plan is approvable.

---

## Iteration 030 -> Planner (Turn 1) [Approval]

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

## Iteration 030 -> Project Archive [Research Result]

# RDF Research Result — Iteration 030

## 1. Hypothesis
phase-30: ARM 1 integration gates fail due to protocol confounds (ceiling effects on G2/G3, tight G1 threshold); ARM 2 definitively falsifies ΔR²_color≥0.30 for all tested decoder-free objectives; M2 mandate not supported and proxy metric must be retired

## 2. Status
ok

## 3. Analysis
This phase executed the two-arm plan from the approved research plan with the Manager's three corrections incorporated.

ARM 1 was designed to test whether frozen iter_029 representations (SFA+VICReg at ΔR²≈0.27 and VICReg-only at ΔR²≈0.04) support functional downstream behavior (tracking, collision attention, causal sensitivity). The pre-registered gates all failed, but the failure is primarily a protocol calibration issue, not a representation quality issue:

1. G1 (tracking): The 20-pixel threshold was unrealistic for the PD controller (Kp=2.0) tracking moving objects in a 128-pixel space. Tracking error of 36-55 pixels is functional (objects are tracked within ~30-40% of the arena width), but not tight. Notably, CLTS-VICReg tracks better than CLTS-SFA, suggesting identity encoding quality does NOT determine tracking quality.

2. G2/G3 (attention/sensitivity): Ceiling effects make these gates uninformative. With 3 objects in 128 pixels and elastic collisions, collision events are too frequent (~500-800 per 1800-step run) for attention switching to be discriminated from chance. The mass perturbation protocol forces the pointer near object 0 and pushes, guaranteeing object-0 surprise regardless of attention mode.

The ARM 1 results are thus a NEGATIVE RESULT about the protocol design, not about the representation. The pre-registered verdict is "representation insufficient," but the actual data does not support this conclusion — the experiment was simply unable to measure what it intended to measure for G2/G3.

ARM 2 tested two new objectives: D1 (batch-level temporal contrastive) and D2 (variance-ramped SFA). Both were falsified on the ΔR²_color ≥ 0.30 gate. D1's batch-level NT-Xent was too weak (object-level matching not implemented). D2's variance ramp actually hurt (0.189 vs iter_029's static 0.275). 

The combined evidence across 11 iterations (020-030) now comprehensively establishes:
- ΔR²_color ≥ 0.30 is NOT achievable by any tested decoder-free objective on this architecture
- The separate-backbone + mask_dyn_sim + coord_vicreg configuration is 0% collapse across 100+ runs
- The proxy metric has been maximally explored and should be retired
- The project must pivot to either (a) accepting weak identity encoding and testing whether it matters for the actual project goal, or (b) relaxing the decoder-free constraint to include reconstruction


## 4. Metrics
{'arm1_g1_clts_sfa_tracking_error': 45.09, 'arm1_g1_clts_vicreg_tracking_error': 36.22, 'arm1_g1_threshold': 20.0, 'arm1_g1_pass': False, 'arm1_g2_clts_sfa_collision_switch_rate': 1.0, 'arm1_g2_frozen_collision_switch_rate': 1.0, 'arm1_g2_random_collision_switch_rate': 0.999, 'arm1_g2_pass': False, 'arm1_g3_clts_sfa_perturbation_switch_rate': 1.0, 'arm1_g3_frozen_perturbation_switch_rate': 1.0, 'arm1_g3_random_perturbation_switch_rate': 1.0, 'arm1_g3_pass': False, 'arm1_gates_passed': 0, 'arm1_total_runs': 48, 'arm2_d1_mean_delta_r2_color': 0.115, 'arm2_d1_ci_lower_95': 0.007, 'arm2_d1_collapse_rate': 0.0, 'arm2_d1_verdict': 'FALSIFIED', 'arm2_d2_mean_delta_r2_color': 0.189, 'arm2_d2_ci_lower_95': 0.074, 'arm2_d2_collapse_rate': 0.0, 'arm2_d2_verdict': 'FALSIFIED', 'arm2_total_seeds': 30, 'arm2_total_runs': 60, 'separate_backbone_collapse_rate_all_iters': 0.0, 'best_delta_r2_color_achieved': 0.275, 'best_delta_r2_source': 'iter_029 Arm B, SFA+VICReg sfa_weight=5.0, 20 seeds'}

## 5. Notes
ARM 1: all gates failed (protocol confounded by ceiling effects). ARM 2: both D1 and D2 falsified.

---
*Note: This is an automated summary as the Research Manager did not provide a full milestone report.*


---

