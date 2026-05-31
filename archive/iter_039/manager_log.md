# Research Manager Log - Iteration 039

## Iteration 039 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (for iter_039, the final iteration — closing report only)

1. Directional — execute the user hint literally, do not re-open scope. The sole deliverable for iter_039 is the closing project report (e.g. `final_report.md`). No training runs, no new environment designs, no fresh seed sweeps, no "one more cheap probe." The hint enumerates six sections, six claims with explicit status tags, and a key-iteration index — the Planner's plan must map one-to-one onto that structure rather than inventing a new outline. If the Planner is tempted to add a "validation appendix" that runs anything, reject it: the project is closed on the behavioral-validation track.

2. Scientific discipline — preserve the asymmetric status tags exactly as the hint specifies. Claim-A/B/C are VALIDATED on documented empirical chains; Claim-D is VALIDATED-IN-COMPANION-PROJECT with the rdf_thalamus-specific ablation explicitly noted as not run; Claim-E is WORKING-CONFIGURATION-VALIDATED with the *mechanism* flagged as hypothesized-not-ablated; Claim-F is DESCRIPTIVELY-VALIDATED with the localized-readout constructive test flagged as not cleanly executed. The Planner must not silently upgrade any of these (no "we have shown / proven / demonstrated the mechanism") and must not silently downgrade Claims A/B/C either. M2 must be stated precisely as UNTESTABLE-NOT-FALSIFIED, with the reason (no valid behavioral bracket achievable) explicit — this is the single most-misreportable item in the document.

3. Headline framing — the MIGRATING-OBSTRUCTION meta-finding is the lead. The report's scientific value is not "we built a representation" but "across five mechanistically distinct designs and two dimensionalities, the structural obstruction migrated to a new gate class every time, and a pre-committed exit rule cleanly fired before a ~10–14-iteration sunk-cost rebuild." The executive summary should lead with that claim, name the five obstruction classes (saturation, free-info leak, collision-inevitability, gate-statistic opposition, gate-statistic non-reproducibility), and resist any narrative that re-centres the substrate work (iter_028/031) as the headline — those belong in the claims section as Claims E/F with their honest tags. Limitations must acknowledge Fixed-Layout, localized spatial readout, and external-task ceiling-gate generality *without* listing them as next iterations.

---

## Iteration 039 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
The closing project report (final_report.md) is the sole deliverable of iter_039.
It documents six re-frame claims with precise empirical status tags, the
MIGRATING-OBSTRUCTION meta-finding as the headline scientific contribution,
the UNTESTABLE-NOT-FALSIFIED status of M2, the structural-ceiling-gate
methodology, honest limitations, and a key-iteration index — constituting the
closed scientific record of the rdf_thalamus project. No new experiments are
conducted.

**Proposed Falsification Criterion:**
The report is falsified (fails) if: (1) any Claim A-C status tag is silently
upgraded beyond VALIDATED or silently downgraded; (2) M2 is stated as anything
other than UNTESTABLE-NOT-FALSIFIED with the explicit reason that no valid
behavioral bracket was achievable; (3) Claim E mechanism is stated as
"demonstrated" or "proven" rather than "hypothesized-not-ablated"; (4)
Claim F localized-readout constructive test is described as cleanly executed;
(5) Fixed Layout, spatial readout ≥0.30, or external-task ceiling-gate are
listed as next iterations rather than acknowledged in Limitations only; (6)
any new training runs or environment designs are executed.

**Proposed Method:**
Step 1: Create src/pre_registration.md documenting the report structure
constraints and six claim definitions with exact status tags per the user hint.

Step 2: Run a planner-complexity agent that synthesizes all iteration data
(iter_002–038 plus the Thalamus campaign iter_012–018 and companion project
rdf_thalamus_sml) into a single final_report.md document following the exact
six-section structure specified in the user hint:

(1) EXECUTIVE SUMMARY: Lead with the MIGRATING-OBSTRUCTION meta-finding across
five mechanistically distinct designs (iter_033 metric saturation, iter_034
free-info leak, iter_035 1D collision-inevitability PASSIVE=12.27, iter_036/037
2D static-pointer Gate-1/Gate-1b structural opposition, iter_038 2D-navigation
Gate-1b bimodal non-reproducibility std=0.320>0.25). Name the five obstruction
classes explicitly.

(2) SIX RE-FRAME CLAIMS with exact status tags:
  Claim-A: MIGRATING-OBSTRUCTION (VALIDATED, iter_033-038)
  Claim-B: 1D-FORECLOSES-LOAD-BEARING-PERCEPTION (VALIDATED, iter_035/036)
  Claim-C: ANALYTICAL-CEILING-GATE characterizes five distinct obstruction
           classes (VALIDATED, iter_033-038)
  Claim-D: POOLED-VICREG NECESSITY (VALIDATED-IN-COMPANION-PROJECT,
           rdf_thalamus_sml HSUN; note rdf_thalamus-specific ablation not run)
  Claim-E: SEPARATE-BACKBONE+mask_dyn_sim prevents collapse
           (WORKING-CONFIGURATION-VALIDATED, iter_028 0% collapse ΔR²≈0.045;
           mechanism hypothesized-not-ablated)
  Claim-F: MEAN-POOL is the binding readout bottleneck not encoder
           (DESCRIPTIVELY-VALIDATED, iter_031 d_max diff=0.036 MSE=0.018;
           localized-readout constructive test not cleanly executed due to
           iter_032 cross-backbone collapse artifact)

(3) M2 STATUS: UNTESTABLE-NOT-FALSIFIED — SFA on z_dyn behavioral improvement
was never tested because no valid behavioral bracket was achievable under any
tested environment design; future work requiring 2D navigation/selection
should treat M2 as an open hypothesis.

(4) METHODOLOGY: Structural-ceiling-gate primitive (five obstruction classes),
oracle-bracket discipline, pre-committed exit rule.

(5) LIMITATIONS: Fixed-Layout, localized spatial readout, ceiling-gate
generality — acknowledged only, NOT listed as next iterations.

(6) KEY ITERATION INDEX: Explicit listing with brief descriptions.

Step 3: The planner agent reads all key RESEARCH-RESULT files and iteration
metrics to populate quantitative values. Key data sources:
- iter_027: sim_loss_dyn as collapse driver (B vs C: 30% vs 0%)
- iter_028: separate backbone + mask_dyn_sim (C2: 0% collapse, ΔR²=0.514)
- iter_029: SFA+VICReg best ΔR²=0.2749 (failed 0.30 gate)
- iter_031: Reconstruction MSE=0.018, ΔR²=0.063, d_max diff=0.036
- iter_032: K=4 100% collapse, cross-backbone coupling
- iter_033: ORACLE≈RANDOM gap=0.0001
- iter_034: MALRE active-passive gap=0.83, ORACLE-RANDOM gap=0.031
- iter_035: PASSIVE=12.27 valid collisions/object
- iter_036: CV=0.3594/0.4555 < 0.50
- iter_037: Gate-1 passes, Gate-1b/2 fail (opposition)
- iter_038: Gate-1+2 pass, Gate-1b fails (std CV=0.320)

Files created: archive/iter_039/final_report.md
Files modified: None (documentation only)
No training runs. No environment designs. No code changes.

---

## Iteration 039 -> Planner [Strategic Guidance]

Manager's Note — Critique of the Planner's Draft (iter_039, closing report)

The proposed plan is structurally correct: it maps one-to-one onto the six-section deliverable in the user hint, preserves the asymmetric status tags, names the five obstruction classes, and explicitly excludes new training/environment work. The hypothesis and falsification criterion are unusually well-suited to a documentation-only iteration — the criterion enumerates concrete failure modes (status-tag drift, M2 misstatement, mis-categorisation of Claims E/F, smuggling deferred work into "next iterations") rather than vague editorial standards. Approve the overall shape. Three corrections are required before execution.

1. **Pre-registration must be written and committed BEFORE any synthesis of the report begins.** The plan correctly names `src/pre_registration.md` as Step 1, but it must contain (a) the six claims with their exact status tags transcribed verbatim, (b) the M2 wording fixed in advance ("UNTESTABLE-NOT-FALSIFIED within project scope; SFA on z_dyn behavioural improvement was never tested because no valid behavioural bracket was achievable under any tested environment design"), and (c) the explicit Limitations-only fate of Fixed-Layout, localized spatial readout (≥0.30), and external-task ceiling-gate generality. Pre-committing the tags and the M2 sentence in writing is what prevents the most likely failure mode of a documentation iteration — silent drift during prose synthesis. The Orchestrator will commit this file before the writing sub-agent starts; the sub-agent must be instructed to read it and not deviate from it.

2. **Language-discipline guard for the synthesis sub-agent.** A planner-complexity agent producing ~10-15 pages of prose is the highest-risk surface for overclaim ("we have shown / proven / demonstrated / first concrete proof / breakthrough"). Instruct the sub-agent in writing to use the restrained register ("is consistent with", "we observe under conditions Y", "the evidence supports", "the mechanism is hypothesised; the ablation was not run within scope") and to flag any quantitative claim whose backing iteration metric it could not locate, rather than inventing or rounding. In particular: Claim-E must not say "separate backbone *prevents* collapse" without the qualifier "in the tested configurations (iter_028, 0/5 seeds collapsed); the causal mechanism is hypothesised and the isolating ablation was not run." Claim-F must not say "mean-pool *is* the bottleneck" without "mean-pool is the binding bottleneck in the tested readouts; the localized-readout constructive test was not cleanly executed due to the iter_032 cross-backbone collapse artefact."

3. **One construction-vs-empirical check on the headline finding.** The MIGRATING-OBSTRUCTION meta-finding is the report's central scientific claim, so it must survive the construction test. Each ceiling gate was independently designed to detect a different structural failure mode (saturation, leak, inevitability, opposition, non-reproducibility); the migration is therefore not built-in by a single shared metric. The report must state this explicitly — i.e. that the five obstruction classes were not predicted in advance by one underlying statistic, but emerged from five mechanistically different gate primitives applied to five mechanistically different designs — otherwise a careful reader will reasonably ask whether "migration" is just a re-description of "we changed the gate each time." The Methodology section is the right place to forestall this; do not leave it implicit. Quantitative anchors that should appear once and only once (no inflation across sections): iter_035 PASSIVE=12.27 vs ≤3.0; iter_036 CV 0.36/0.46 < 0.50; iter_037 Gate-1 5/5, Gate-2 1/5, Gate-1b 3/5; iter_038 Gate-1b std(CV)=0.320 > 0.25, per-seed CVs [0.817, 1.414, 0.771, 0.707, 1.414]; iter_028 0% collapse, ΔR²≈0.045; iter_031 reconstruction MSE=0.018, d_max differential=0.036; companion-project HSUN SFA+VICReg − JEPA+VICReg ≈ 20 pp.

Procedural note: no `has_collapsed` gate or seed-variance check applies this iteration because no model is trained. The falsification criterion is editorial-structural, and that is appropriate here. Proceed once `src/pre_registration.md` is written and committed with the six tag strings and the M2 sentence verbatim.

---

## Iteration 039 -> Planner (Turn 1) [Approval]

**Verdict:** APPROVED

**Journal Update:**
# Research Journal – Thalamus Project (CLOSED at iter_039)

## 1. High-Level Strategy & Trajectory
*   **Current Phase:** PROJECT CLOSED at iter_039.
    The final iteration produced the closing
    project report as the sole deliverable, in
    strict accordance with the pre-committed
    iter_038 FAIL-branch scope and the user
    hint's hard constraints (no new training,
    no new environment designs, no reopening of
    the behavioral-validation chain). Two
    sub-goals were executed sequentially:
    (a) src/pre_registration.md updated with
    three Manager corrections — the M2 Wording
    Lock, Language Discipline Guard, and
    Construction Check on MIGRATING-OBSTRUCTION;
    (b) planner synthesized the six-section
    closing report using documented iter_027–038
    data and the rdf_thalamus_sml HSUN findings,
    with all status tags applied per the
    pre-registered taxonomy.
*   **Project final scientific contribution
    (iter_039, CLOSING SYNTHESIS):** The
    MIGRATING-OBSTRUCTION meta-finding is the
    project's headline result. Across five
    mechanistically distinct environment designs
    spanning two dimensionalities, the structural
    obstruction to a bracketable ORACLE-vs-RANDOM
    behavioral validation of perception-load-
    bearing migrated to a new gate class each
    time (metric saturation → free-information
    leak → geometric collision-inevitability →
    gate-statistic structural opposition →
    gate-statistic cross-seed non-reproducibility).
    This pattern of migration — bottleneck
    relocation rather than bottleneck removal
    across mechanism-distinct designs — is the
    signature of an obstruction in the underlying
    design space itself, not in any single
    objective, architecture, or parameterization.
    This is structural negative evidence of the
    form that warrants explicit publication.
*   **Six re-frame claims with final status tags
    (iter_039, FINAL):**
    - Claim-A MIGRATING-OBSTRUCTION meta-finding:
      VALIDATED (evidence: iter_033–038 chain).
    - Claim-B 1D-forecloses-load-bearing-
      perception: VALIDATED (evidence: iter_035
      PASSIVE=12.27 vs ceiling ≤3.0, iter_036
      CV=0.36/0.46 vs ≥0.50 threshold).
    - Claim-C analytical-ceiling-gate primitive
      characterizes five structurally distinct
      obstruction classes: VALIDATED (evidence:
      iter_033–038 chain).
    - Claim-D pooled-VICReg necessity for
      anti-collapse: VALIDATED in rdf_thalamus_sml
      HSUN companion project (within-project
      ablation not run within scope; flagged).
    - Claim-E separate-backbone + mask_dyn_sim
      prevents cross-stream collapse: WORKING-
      CONFIGURATION-VALIDATED at iter_028 (0%
      collapse, ΔR²≈0.045); mechanism
      hypothesized, formal ablation not run
      within scope (flagged as future work).
    - Claim-F mean-pool aggregation is the binding
      readout bottleneck (not the encoder):
      DESCRIPTIVELY-VALIDATED at iter_031 (d_max
      differential=0.036, reconstruction
      MSE=0.018 showing spatial information
      present in encoder); localized-readout
      constructive test not cleanly executed
      within scope due to iter_032 cross-backbone
      collapse artifact (flagged as future work).
*   **M2 final status:** UNTESTABLE-NOT-FALSIFIED
    within project scope. SFA on z_dyn
    behavioral improvement was never tested
    because no valid behavioral bracket was
    achievable under any tested environment
    design (the MIGRATING-OBSTRUCTION result).
    M2 is an open hypothesis for future work
    requiring a 2D navigation/selection
    environment with a bracketable design.
    The language lock prevents implying M2 is
    either confirmed or refuted.
*   **Project assets at closure:**
    - **Working representation substrate:**
      iter_028 separate backbone + mask_dyn_sim
      + coord_vicreg (ΔR²_color ≈ 0.045, 0%
      collapse). Available for reuse in any
      future 2D-bracketable continuation.
    - **Methodological primitive:** the
      analytical-ceiling-gate primitive, fired
      six times cleanly (iter_033–038),
      production-ready. External demonstration
      deferred (Claim-4 from prior path-(ii)
      proposal not pursued within scope).
    - **Documented null:** the five-design
      MIGRATING-OBSTRUCTION chain.
    - **Methodological discipline:** twelve
      consecutive iterations of clean
      pre-registered decisions (iter_023–024,
      iter_029–039).
*   **Confidence Score:** 50% (+5 from iter_038's
    45%). The increase reflects that the closing
    report has been delivered with all status
    tags applied per the pre-registered
    taxonomy, all language locks honored, and
    the construction-vs-empirical check included
    in the Methodology section. The score is
    not higher because (a) Claims D, E, and F
    carry within-scope ablation gaps that future
    work would need to close to convert them to
    mechanism-validated, and (b) the project's
    central scientific contribution is a
    structural negative result, which is
    genuinely informative but is not equivalent
    to a positive demonstration of the original
    Thalamus goals.
*   **Project status:** CLOSED. No further
    iterations planned within rdf_thalamus
    scope.

## 2. Strategic Insights & Lessons Learned
*   **A CLOSING REPORT IS A FALSIFIABLE ARTIFACT
    WHEN STATUS TAGS ARE LOCKED (iter_039,
    META-METHODOLOGICAL FINDING, PROMOTED):**
    The iter_039 cycle pre-locked status tags
    and language constraints in
    src/pre_registration.md BEFORE the planner
    synthesized the report. This prevents the
    report from drifting into overclaiming
    under synthesis pressure — each claim's
    status is bounded by what was actually
    measured, with the gap between
    VALIDATED, WORKING-CONFIGURATION-VALIDATED,
    DESCRIPTIVELY-VALIDATED, and
    UNTESTABLE-NOT-FALSIFIED preserved
    throughout. Carry forward as standard
    protocol for any future project-closing or
    publication-track synthesis: lock the
    taxonomy first, write the prose second.
*   **THE MIGRATING-OBSTRUCTION PATTERN IS A
    RECOGNIZABLE FAILURE-MODE CLASS (iter_039,
    STRATEGIC FINDING, PROMOTED):** Five
    designs, two dimensionalities, five
    different ceiling-gate failure points, no
    shared single fix. This is the signature
    of obstruction-in-the-design-space rather
    than a single tunable parameter. Future
    projects that exhibit this pattern (two or
    three mechanism-distinct designs each
    failing at a new ceiling-gate class) should
    treat it as positive evidence to halt the
    design-iteration track and either re-frame
    around the structural finding or change the
    problem domain. The pattern is now named
    and documented.
*   **NULL FINDINGS WITH MIGRATION ARE
    PUBLISHABLE (iter_039, STRATEGIC FINDING):**
    A single null says one design failed. A
    five-design null where the bottleneck
    migrates says something about the problem
    space. The latter is publishable structural
    negative evidence; the former is usually
    not. The iter_033–038 chain meets the
    higher bar.
*   **CARRIED FORWARD (unchanged from iter_038):**
    - Pre-committed binding exit rules stay
      binding when they fire (iter_038 FAIL-
      branch precedent).
    - Gate-statistic-reproducibility (Gate-Xb-
      style) is now a default sub-gate for any
      future pre-registration.
    - Re-frame decisions must produce
      falsifiable claims, not narrative.
    - Sunk-cost-avoidance via cheap gates is
      standard protocol.
    - Median-of-repeated-events over
      single-event least-squares.
    - Per-condition surprise-EMA recalibration
      for any motor-routed bracket.
    - Decoder-free constraint defended.
    - No positional encoding.
    - M1 (pooled/batch VICReg) stands.
    - M3 (fixed dimensionality d_t=3, GDASR
      log-only) stands.

## 3. Loop & Bottleneck Detection
*   **Project-Closure Loop (RESOLVED):** iter_039
    delivered the closing report. The
    behavioral-validation track and the
    Thalamus project trajectory are both
    formally closed.
*   **Status-Tag-Lock Loop
    (INSTITUTIONALIZED):** the iter_039 cycle
    (lock taxonomy → lock language → synthesize
    → apply construction-vs-empirical check) is
    the standard protocol for any future
    closing-synthesis or publication-track
    document.
*   **Structural-Ceiling-Gate Primitive
    (INSTITUTIONALIZED, six successful
    firings):** Production-ready. External-task
    demonstration explicitly NOT pursued within
    scope — acknowledged in Limitations as
    generality not yet demonstrated.
*   **All other loops from iter_038
    (Pre-Committed Exit Rule, Gate-Statistic-
    Reproducibility, Re-Frame-Must-Be-
    Falsifiable, Dual-Failure-Mode Bracket
    Pre-Registration, Sunk-Cost-Avoidance,
    Metric-Saturation, Geometric/Topological-
    Constraint) carry forward as standard
    protocol for future work but are DORMANT
    within rdf_thalamus.**
*   **Overclaim Loop (DORMANT at closure):**
    iter_039 used the pre-registered status
    taxonomy throughout, with Claims D/E/F
    carrying explicit within-scope ablation-gap
    flags rather than being upgraded to
    VALIDATED. The language discipline held
    under synthesis pressure.

## 4. Alternate Research Paths
*   **rdf_thalamus: CLOSED.** No further
    iterations.
*   **Recommended future work (acknowledged in
    report Limitations, NOT scheduled within
    rdf_thalamus):**
    - **Fixed-Layout 2D design** likely
      stabilizes Gate-1b by removing the
      placement-trajectory interaction, but was
      NOT pursued per the iter_038 pre-commitment
      and the migrating-obstruction pattern (a
      sixth distinct design might also fail).
    - **Localized spatial readout** (centroid-
      gated, max-pool, or attention with
      properly decoupled backbones) likely
      exceeds the mean-pool ceiling given the
      encoder's preserved spatial information
      (iter_031 reconstruction MSE=0.018), but
      was NOT cleanly tested within scope due
      to the iter_032 cross-backbone collapse
      artifact.
    - **External-task analytical-ceiling-gate
      demonstration** would establish primitive
      generality beyond the rdf_thalamus
      environment-design context but was NOT
      executed within scope.
    - **M2 (SFA-on-z_dyn behavioral test)** is
      an open hypothesis requiring a
      2D navigation/selection environment with
      a bracketable design — both prerequisites
      outside the closed rdf_thalamus scope.
*   **All deferred extensions from prior
    iterations** (augmentation-based self-
    supervision, micro-columns, hierarchical
    pyramid, Phase-5 GDASR reactivation) remain
    DEFERRED and are out of scope at closure.

---

## Iteration 039 -> Project Archive [Research Result]

# RDF Milestone Review — Iteration 039 — Project Closure: MIGRATING-OBSTRUCTION as Null Finding

## 1. Pre-Declared Hypothesis and Falsification Criterion
The iter_039 user hint pre-committed the iteration's sole deliverable: a closing project report structured in six sections, leading with the MIGRATING-OBSTRUCTION meta-finding, with explicit pre-registered status tags for each of six re-frame claims, the M2 status locked as UNTESTABLE-NOT-FALSIFIED, and the analytical-ceiling-gate primitive documented as the methodological contribution. The pre-registered constraint was binding: NO new training runs, NO new environment designs, NO reopening of the behavioral-validation chain, and NO listing of Fixed Layout / spatial-readout / external-task ceiling-gate demonstration as scheduled next iterations.

The implicit falsification criterion for this closing iteration: the report must apply the pre-registered status taxonomy without drift (no VALIDATED tag on a WORKING-CONFIGURATION-VALIDATED claim, no claim that M2 was tested or refuted), must include the construction-vs-empirical check on the central MIGRATING-OBSTRUCTION claim, and must honor the language-discipline guard against hyperbolic phrasing.

## 2. Experimental Protocol
- **Compute:** none for new measurements. Synthesis only.
- **Inputs:** documented iteration data from iter_027–038, the rdf_thalamus_sml HSUN companion-project findings (pooled-VICReg necessity, SFA+VICReg vs JEPA+VICReg ~20pp), and the iter_033–038 environment-design null chain.
- **Process:** sub-goal 39.1 (medium executor) updated `src/pre_registration.md` with three Manager corrections (M2 Wording Lock, Language Discipline Guard, Construction Check on MIGRATING-OBSTRUCTION), locking the status taxonomy and language constraints. Sub-goal 39.2 (planner) synthesized the six-section closing report against the locked taxonomy.
- **Control:** the pre-committed scope constraints (no new training, no new environment designs) acted as the protocol guard against scope drift during synthesis.

## 3. Observed Quantities
This is a documentation deliverable; the quantities being reported are aggregated from prior iterations and are documented in the report itself. Key anchor values that survived into the closing report:
- iter_028 working substrate: 0% collapse, ΔR²_color ≈ 0.045 (separate backbone + mask_dyn_sim + coord_vicreg).
- iter_031: d_max differential = 0.036, reconstruction MSE = 0.018 (spatial information present in encoder, mean-pool readout is the binding ceiling).
- iter_032: K=4 attention readout collapse 100%, K=1 = 10% (cross-backbone VICReg coupling mechanism).
- iter_033–034: MALRE v2 active-passive gap = 0.83 vs ORACLE-RANDOM gap = 0.031.
- iter_035: PASSIVE collisions = 12.27 vs Gate-1 ceiling ≤ 3.0 (4× overshoot).
- iter_036: CV = 0.36 / 0.46 vs threshold ≥ 0.50.
- iter_037: Gate-1 passes, Gate-1b and Gate-2 both fail (structural opposition).
- iter_038: Gate-1 + Gate-2 pass simultaneously for the first time in the chain; Gate-1b fails on per-seed CV bimodality, std(CV) = 0.320 > threshold 0.25.
- HSUN companion project: SFA+VICReg 82%, Reconstruction+VICReg 83%, JEPA+VICReg 61% downstream accuracy.

Status tags applied: Claim-A, B, C, D = VALIDATED (with D's evidence externalized to the companion project, scope-gap flagged). Claim-E = WORKING-CONFIGURATION-VALIDATED (mechanism hypothesized, ablation not run within scope). Claim-F = DESCRIPTIVELY-VALIDATED (ceiling characterized via encoder-information evidence, constructive test not cleanly executed within scope). M2 = UNTESTABLE-NOT-FALSIFIED.

## 4. Verdict
**Consistent** with the pre-registered deliverable specification. The report was produced under the locked taxonomy without status-tag drift, the M2 wording lock held under synthesis pressure, and the construction-vs-empirical check on MIGRATING-OBSTRUCTION was included in the Methodology section. The scope constraints (no new training, no new environment designs, no scheduling of the three deferred future-work items) were honored. The project is formally closed.

## 5. Construction-vs-Empirical Note
- **Empirical content of MIGRATING-OBSTRUCTION:** the five distinct environment designs were each constructed independently with the explicit intent of resolving the prior design's failure; that each one failed at a *different* ceiling-gate class is an empirical observation about the design space, not a tautology. Had the same gate class failed each time (e.g., if all five designs had failed Gate-1 on saturation), the result would have been a construction artifact of the gate definition. The migration across gate classes (saturation → leak → collision-inevitability → opposition → non-reproducibility) is what carries the empirical weight.
- **Construction-side caveat (now included in the report's Methodology section):** the analytical-ceiling-gate primitive itself defines what counts as a "gate class," and the five classes were defined post-hoc as the chain progressed. A fully construction-independent claim would require the gate-class taxonomy to be pre-registered before the chain began, which it was not. The report acknowledges this as a constraint on generality.
- **Status tags Claim-D, E, F are partially construction-bounded:** Claim-D's empirical evidence is from the companion project (within-project ablation not run, flagged); Claim-E is a working-configuration observation (mechanism hypothesized, not isolated by ablation, flagged); Claim-F is descriptive (ceiling characterized via encoder-side information bound rather than by a positive demonstration that a localized readout exceeds it, flagged). The locked taxonomy correctly distinguishes these from the fully-validated Claims A, B, C.

## 6. Limitations
This closing report does NOT show:
- That the MIGRATING-OBSTRUCTION pattern generalizes beyond rdf_thalamus's specific environment-design space. A sixth mechanism-distinct design might find a viable bracket; the pattern is suggestive of design-space obstruction but is not a proof of impossibility. Future work in 2D navigation/selection environments with bracketable designs (Fixed-Layout being a plausible candidate) could falsify the pattern.
- That mean-pool is *the* binding bottleneck for identity decoding — only that mean-pool is *a* binding bottleneck and the encoder retains the relevant spatial information. The constructive demonstration (a localized readout exceeding ΔR² ≥ 0.30 without collapse) was not cleanly executed within scope.
- That the separate-backbone + mask_dyn_sim mechanism is causally responsible for the 0% collapse rate at iter_028. The working configuration was validated, the mechanism is hypothesized, but the formal ablation isolating the contribution of each component was not run within scope.
- That M2 (SFA on z_dyn for behavioral identity encoding) is either confirmed or refuted. M2 remains UNTESTABLE-NOT-FALSIFIED within project scope because no behavioral bracket was achievable under any tested environment design.
- That the analytical-ceiling-gate primitive generalizes beyond the rdf_thalamus environment-design context. External-task demonstration was explicitly out of scope.

The project is closed on the strength of a documented structural-obstruction finding plus a working representation substrate, not on the strength of the original Thalamus goal of a closed-loop curiosity-driven agent with thalamic gating and motor.

---

