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

