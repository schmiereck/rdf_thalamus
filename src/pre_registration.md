# RDF Scientific Pre-Registration

*   **Iteration:** 039
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
The closing project report (final_report.md) is the sole deliverable of iter_039.
It documents six re-frame claims with precise empirical status tags, the
MIGRATING-OBSTRUCTION meta-finding as the headline scientific contribution,
the UNTESTABLE-NOT-FALSIFIED status of M2, the structural-ceiling-gate
methodology, honest limitations, and a key-iteration index — constituting the
closed scientific record of the rdf_thalamus project. No new experiments are
conducted.

## 2. Falsification Criterion
The report is falsified (fails) if: (1) any Claim A-C status tag is silently
upgraded beyond VALIDATED or silently downgraded; (2) M2 is stated as anything
other than UNTESTABLE-NOT-FALSIFIED with the explicit reason that no valid
behavioral bracket was achievable; (3) Claim E mechanism is stated as
"demonstrated" or "proven" rather than "hypothesized-not-ablated"; (4)
Claim F localized-readout constructive test is described as cleanly executed;
(5) Fixed Layout, spatial readout ≥0.30, or external-task ceiling-gate are
listed as next iterations rather than acknowledged in Limitations only; (6)
any new training runs or environment designs are executed.

## 3. Proposed Method
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
*Created automatically by the RDF Orchestrator prior to iteration execution.*
