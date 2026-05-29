# Research Journal – Thalamus Project

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** Phase 0 (Objective Migration) — collapse-elimination
    sub-phase has produced a **second pre-registered null** (iter_027) on the
    structural-cause hypothesis, plus a **suggestive within-architecture
    ablation** (Arm C) that re-frames the active question from "where does
    the gradient competition happen?" to "which loss term drives z_dyn
    collapse?".
*   **Active Direction:** iter_027 tested the hypothesis that the **shared
    CNN backbone** is the primary cause of z_dyn collapse. Arm B (separate
    backbone, full JEPA+VICReg, same hyperparameters) was the falsification
    vehicle. Arm B collapsed at **30%** — indistinguishable from the
    shared-backbone baseline (30–40%). The shared-backbone hypothesis is
    **refuted**. This is the second consecutive iteration in which a
    pre-registered structural hypothesis was cleanly killed; this is
    methodologically healthy.
*   **The Arm C signal (treat with discipline):** Arm C, identical to Arm B
    except `mask_dyn_sim=True` (i.e. `sim_loss_dyn` removed; z_dyn shaped
    only by VICReg variance + covariance), showed 0% collapse over 10 seeds
    AND the highest measured delta_R2_color (0.18). The agent labelled this
    a "breakthrough" and stated "sim_loss_dyn is the causal driver of z_dyn
    collapse" — **this language is rejected by the Manager as overclaim**.
    What we actually have:
    - **Construction-versus-empirical caveat:** VICReg's variance hinge
      directly penalizes std < 1, which is the same quantity used in the
      eval-std collapse criterion. When z_dyn is shaped *only* by VICReg,
      the optimizer is being told almost exactly what the collapse metric
      is measuring. 0% collapse under VICReg-only is therefore *partly* a
      tautology of the chosen objective, not a clean empirical discovery.
      The empirical content of Arm C is narrower: "removing `sim_loss_dyn`
      does not destabilize VICReg's variance preservation under the
      separate-backbone regime" — which is informative but is not the same
      as "sim_loss_dyn causes collapse."
    - **Not pre-registered:** the iter_027 pre-registration covered the
      B-vs-baseline comparison. Arm C is an exploratory addition. A 30%
      vs 0% delta with n=10 is suggestive (Fisher's exact p ≈ 0.21
      approx; the difference is not formally significant at n=10).
    - **Missing critical control:** the same `mask_dyn_sim=True` ablation
      has not yet been run on the **shared backbone**. Without that arm,
      we cannot distinguish "separate backbone + no sim_dyn" from "no
      sim_dyn anywhere" as the operative intervention.
    - **No robustness check:** Arm C has not been tested under perturbation
      (±10% on var_weight, alternate seeds, ramped sim_weight to z_coord
      but masked from z_dyn).
*   **Updated mechanism hypothesis (TENTATIVE, requires iter_028
    confirmation):** When `z_target_dyn` is *not* stop-gradiented in the
    current JEPA implementation, gradient flow from `sim_loss_dyn` may push
    the encoder toward predictable-but-degenerate z_dyn representations,
    and this pressure overrides VICReg's variance hinge in ~30% of seeds.
    This is consistent with the iter_026 observation that *increasing*
    var_weight (25→50) worsened collapse (the JEPA pressure was already
    dominant; pushing variance harder destabilized the joint optimization
    further). If true, this hypothesis also aligns with the M2 mandate:
    identity (z_dyn) should be shaped by a slowness/identity objective,
    with prediction error treated as a *readout* signal, not as gradient
    input to z_dyn.
*   **Next Priority (iter_028):** Pre-registered control matrix to convert
    the Arm C signal from suggestive to confirmed (or to refute it):
    - C1: `mask_dyn_sim=True` on **shared backbone** (the missing arm).
    - C2: Arm C replication with a different random seed bank (n=10).
    - C3: Robustness perturbation of Arm C (±10% var_weight, ±10%
      cov_weight; one ramp variant).
    - Pre-register: collapse gate ≤10%; report train AND eval std;
      Hungarian-primary matching; buffer=4000 (carry forward iter_026
      confound control). Falsification: if C1 collapses ≥20%, the
      "separate backbone" was load-bearing after all; if C2 collapses
      ≥20%, Arm C was a seed-bank artefact; if C3 collapses ≥20%, the
      result is not robust.
*   **Confidence Score:** 50% (recovered slightly from 40%). One additional
    structural hypothesis ruled out (good), one promising ablation arm
    identified (good but unverified), the mechanism story is more
    narrowly constrained. But: two consecutive iterations have failed
    their primary gate; Arm C is unconfirmed; the foundation for downstream
    Phase 1+ work is still not in place.

## 2. Strategic Insights & Lessons Learned
*   **SHARED CNN BACKBONE IS NOT THE PRIMARY CAUSE OF Z_DYN COLLAPSE
    (iter_027, CONFIRMED via pre-registered null):** Separate backbones
    collapse at the same rate as shared backbones (30%) under
    identical JEPA+VICReg objectives. Architectural decoupling at the
    encoder level alone does not buy stability. This refutes the iter_026
    hypothesis that drove iter_027.
*   **THE SIM_LOSS-vs-VICReg COMPETITION IS THE NEW LEAD HYPOTHESIS
    (iter_027, SUGGESTIVE NOT CONFIRMED):** Within the separate-backbone
    regime, removing `sim_loss_dyn` (Arm C) eliminated collapse and
    coincided with the highest delta_R2_color (0.18). Three caveats
    gate any stronger claim:
    (a) VICReg-only naturally maintains the very std metric used for the
        collapse check — partial construction-versus-empirical concern;
    (b) the within-architecture comparison was not pre-registered;
    (c) the matching shared-backbone arm has not been run, so we cannot
        yet say whether separate backbones were necessary.
    Treat as a high-priority hypothesis to confirm in iter_028, not as
    established fact.
*   **REPEATED PATTERN — JEPA OBJECTIVE PRESSURE COMPETES WITH VICReg
    (iter_026 + iter_027 cross-iteration synthesis):** iter_026 found
    that strengthening VICReg variance worsened collapse; iter_027
    found that weakening JEPA pressure on z_dyn (by masking
    `sim_loss_dyn`) eliminated it. Both data points push in the same
    direction: under the current implementation, `sim_loss_dyn` and
    VICReg compete, and `sim_loss_dyn` wins often enough to collapse
    z_dyn. This is mechanism-level convergent evidence (across two
    iterations and four arms) — stronger than either iteration alone.
*   **CONNECTION TO M2 MANDATE (RECONNECTING TO GOAL):** The iter_027
    Arm C finding, if confirmed, is structurally aligned with the M2
    mandate from the goal document: M2 says z_dyn should be shaped by an
    identity/slowness objective, with JEPA-style prediction error
    demoted to a readout. Masking `sim_loss_dyn` from the z_dyn gradient
    path is the minimal version of that demotion. iter_028 should
    include this framing explicitly in its pre-registration.
*   **PRE-REGISTERED NULLS REMAIN FIRST-CLASS RESULTS (iter_026,
    iter_027, ENFORCED):** Two consecutive iterations have produced
    defensible nulls because they pre-declared their falsification
    criterion. The discipline holds.
*   **PRESERVED:** M2 stream-assignment guidance stands; M1 (pooled
    VICReg) stands; d_max=16 capacity baseline stands; Hungarian-primary
    matching stands; 20% control-collapse power threshold stands.

## 3. Loop & Bottleneck Detection
*   **Identity Encoding Bottleneck (ACTIVE, MORE NARROWLY LOCALIZED):**
    Now traced to the gradient interaction between `sim_loss_dyn` and
    the VICReg variance term on z_dyn, regardless of backbone
    architecture. Awaiting iter_028 confirmation.
*   **Architectural-Cause Bottleneck (PROVISIONALLY DOWNGRADED):**
    iter_027 found that separating the backbone alone does not resolve
    collapse. This bottleneck is reclassified from "primary" to
    "secondary" — it may still matter, but it is not load-bearing.
*   **Capacity-vs-Objective Confound (RESOLVED, iter_025 v2):** Still
    closed.
*   **Matching-Procedure Confound (RESOLVED, iter_025 v2):** Still
    closed.
*   **Diagnostic-vs-Constructive Iteration Loop (CLEARED):** Two
    consecutive pre-registered diagnostic iterations produced
    actionable nulls and a candidate mechanism. The protocol is paying
    off; keep going.
*   **Buffer-Capacity Confound (TRACKED):** iter_027 used buffer=4000
    throughout (per the iter_026 instruction). Keep buffer=4000
    constant in iter_028.
*   **Overclaim Loop (NEW, NOW TRACKED):** iter_027 executor used
    "breakthrough", "completely eliminated", "BEST semantic encoding",
    and "causal driver" for an unconfirmed within-architecture
    ablation. Manager has flagged this. iter_028 pre-registration MUST
    explicitly hold Arm-C claims at "suggestive" until C1+C2+C3 are run.
*   **Objective-Swapping Loop (DORMANT, ENFORCED):** Holds. iter_028
    keeps the SFA/JEPA+VICReg objective family and varies only loss
    masking and backbone topology.
*   **Logistics:** Executor token limits persist. Tracked, not blocking.

## 4. Alternate Research Paths
*   **iter_028: Sim-Loss Causal Confirmation (IMMEDIATE PRIORITY,
    ABLATION):** Three-arm pre-registered design (C1, C2, C3 above)
    to convert the iter_027 Arm C signal from suggestive to confirmed
    or refuted. Buffer held at 4000; train + eval std reported;
    Hungarian-primary matching; n=10 minimum per arm. Falsification
    criteria as listed.
*   **iter_029 (CONDITIONAL on iter_028 success): Promote to
    SFA-on-z_dyn architecture (per M2 mandate):** If Arm C
    mechanism survives iter_028, the next move is the actual M2
    configuration: SFA slowness term (`||z_dyn(t) - z_dyn(t-1)||^2`)
    on z_dyn instead of just VICReg-only, paired with stop-gradient
    readout of prediction error for the surprise signal. This is the
    goal-document's intended primary representation objective and is
    where Phase 0 should land.
*   **Multi-Knob Regime Stabilization (DEFERRED):** Demoted further.
    Two structural hypotheses have produced nulls; the action now is to
    confirm the loss-masking mechanism, not to revisit hyperparameter
    sweeps.
*   **Object-Tracking-ID Contrastive (DEFERRED to iter_030+):**
    Conditional on iter_028/029 outcome.
*   **Augmentation-Based Self-Supervision (BYOL/SimCLR) (CONDITIONAL):**
    Becomes lead candidate IF iter_028 refutes the Arm C mechanism.
*   **Accept Decoder-Free Constraint Relaxation (LAST RESORT):**
    Unchanged.
*   **Micro-Columns (DEFERRED per semantic caution):** Unchanged.
*   **Fixed-Dimensionality + M3 Regime (ACTIVE):** Preserved.