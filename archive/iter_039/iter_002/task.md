You are writing the CLOSING PROJECT REPORT for the rdf_thalamus project. This is the FINAL and ONLY deliverable of iteration 039. NO new training runs, NO new environment designs, NO code changes. The sole output is archive/iter_039/final_report.md.

**MANDATORY: Read src/pre_registration.md FIRST and follow ALL constraints therein, including the M2 Wording Lock (Section 4), Language Discipline Guard (Section 5), and Construction Check on MIGRATING-OBSTRUCTION (Section 6).**

## Report Structure (exactly six sections)

### Section 1: EXECUTIVE SUMMARY
Lead with the MIGRATING-OBSTRUCTION meta-finding as the headline scientific contribution. Across five mechanistically distinct designs spanning two dimensionalities (1D and 2D), the structural obstruction to making perception behaviorally load-bearing migrated to a new gate class every time. This is the signature of an obstruction in the design space itself. Name the five obstruction classes explicitly:
1. Metric saturation (iter_033: ORACLE ≈ RANDOM on behavioral-pivot selectivity, gap = 0.0001)
2. Free-information leak (iter_034: v2 MALRE active-passive gap = 0.83 but ORACLE-RANDOM gap = 0.031)
3. Geometric collision-inevitability (iter_035: 1D passive pointer gets PASSIVE = 12.27 valid collisions/object vs threshold ≤ 3.0)
4. Gate-statistic structural opposition (iter_036/037: 2D static pointer — Gate-1 passes 5/5, but Gate-1b and Gate-2 mutually oppose)
5. Gate-statistic non-reproducibility (iter_038: 2D navigation — Gate-1 and Gate-2 both pass simultaneously for the first time, but Gate-1b per-seed CV is bimodal with std = 0.320 > 0.25)

Include the pre-committed exit rule that fired cleanly at iter_038, preventing a ~10-14-iteration sunk-cost 2D rebuild.

### Section 2: SIX RE-FRAME CLAIMS
Each stated as a falsifiable proposition with its exact status tag and key iteration reference. Use the EXACT status tags below — do NOT upgrade or downgrade:

**Claim-A: MIGRATING-OBSTRUCTION**
- Status: VALIDATED
- Evidence: iter_033-038 documented chain across five mechanistically distinct designs and two dimensionalities
- Quantitative anchors: iter_033 ORACLE-RANDOM gap = 0.0001; iter_034 ORACLE-RANDOM gap = 0.031; iter_035 PASSIVE = 12.27 vs ≤ 3.0; iter_036 CV = 0.3594/0.4555 < 0.50; iter_037 Gate-1 5/5, Gate-2 1/5, Gate-1b 3/5; iter_038 Gate-1b std(CV) = 0.320 > 0.25, per-seed CVs [0.817, 1.414, 0.771, 0.707, 1.414]

**Claim-B: 1D-FORECLOSES-LOAD-BEARING-PERCEPTION**
- Status: VALIDATED
- Evidence: iter_035 PASSIVE = 12.27 (4× ceiling overshoot); iter_036 CV = 0.36/0.46 < 0.50 under foveated gaze
- The 1D × N=3 × 128px sandbox cannot make perception behaviorally load-bearing under an ORACLE-vs-RANDOM bracket

**Claim-C: ANALYTICAL-CEILING-GATE characterizes five structurally distinct obstruction classes**
- Status: VALIDATED
- Evidence: iter_033-038 chain; the ceiling-gate primitive detected five different failure modes at five different structural levels

**Claim-D: POOLED-VICREG NECESSITY for anti-collapse**
- Status: VALIDATED-IN-COMPANION-PROJECT
- Evidence: rdf_thalamus_sml HSUN companion project — without batch-level VICReg, every objective collapsed to ~44-48% downstream accuracy; with pooled VICReg, 61-83%. SFA+VICReg ≈ 82%, Reconstruction+VICReg ≈ 83%, JEPA+VICReg ≈ 61%, a ~20pp gap favoring non-predictive objectives.
- **IMPORTANT NOTE:** The rdf_thalamus-specific ablation (remove VICReg from the current architecture and check collapse) was NOT run within scope. The existing code already computes VICReg at batch level (calc_var_loss / calc_cov_loss on single-frame latents (B, d_t), batch=32), consistent with the companion project finding.

**Claim-E: SEPARATE-BACKBONE + mask_dyn_sim prevents cross-stream collapse**
- Status: WORKING-CONFIGURATION-VALIDATED
- Evidence: iter_028 C2 arm: 0% collapse (0/10 seeds) with separate backbone + VICReg-only z_dyn (mask_dyn_sim=True); ΔR²_color ≈ 0.045 (working but weak identity encoding). iter_027 identified sim_loss_dyn as the causal driver of z_dyn collapse (Arm B: 30% collapse with JEPA+VICReg on separate backbone; Arm C: 0% collapse with VICReg-only on separate backbone)
- **CRITICAL LANGUAGE:** "In the tested configurations (iter_028, 0/5 seeds collapsed in C2 seed-robustness arm); the causal mechanism is hypothesised and the isolating ablation was not run within scope." Do NOT say "separate backbone prevents collapse" without this qualifier. The mechanism by which separate backbone + mask_dyn_sim eliminates collapse is hypothesised — the ablation isolating architecture from loss configuration was not run.

**Claim-F: MEAN-POOL is the binding readout bottleneck not the encoder**
- Status: DESCRIPTIVELY-VALIDATED
- Evidence: iter_031: Reconstruction+VICReg ceiling probe — reconstruction MSE = 0.018 (spatial information present in encoder), but ΔR²_color = 0.063 for mean-readout z_dyn (d_max=8) vs 0.027 for d_max=2; d_max differential = 0.036. The encoder carries spatial information (MSE=0.018) that mean-pooling destroys.
- **CRITICAL LANGUAGE:** "Mean-pool is the binding bottleneck in the tested readouts; the localized-readout constructive test was not cleanly executed due to the iter_032 cross-backbone collapse artefact." Do NOT say "mean-pool is the bottleneck" without this qualifier. iter_032 showed centroid-gated readout (K=4) collapsed 100% (20/20 seeds) due to cross-backbone VICReg coupling — this is an artefact of the particular readout mechanism, not a clean test of whether localized spatial readout can exceed mean-pool performance.

### Section 3: M2 STATUS
Use EXACTLY this sentence (per pre-registration M2 Wording Lock):
"UNTESTABLE-NOT-FALSIFIED within project scope; SFA on z_dyn behavioural improvement was never tested because no valid behavioural bracket was achievable under any tested environment design."

Then elaborate:
- M2 was the project's central representational mandate (SFA on z_dyn as primary objective, demoting JEPA)
- In proxy-metric space, SFA+VICReg showed directional improvement (iter_029: ΔR² = 0.27 vs VICReg-only 0.04) but failed the pre-registered 0.30 gate
- The behavioral test that would discriminate SFA from JEPA was never achievable because no environment design produced a valid ORACLE-vs-RANDOM bracket (iter_033-038 null chain)
- Future work requiring a 2D navigation/selection environment should treat M2 as an open hypothesis
- Note: iter_023-024 comprehensively falsified SFA's ability to produce identity encoding via slowness alone (multi-step SFA and NT-Xent both failed); iter_029 showed directional trend on separate backbone. The proxy-metric evidence is mixed; the behavioral evidence is absent.

### Section 4: METHODOLOGY
Summarize three methodological contributions:
1. **Structural-ceiling-gate primitive:** An analytical gate that measures whether PASSIVE information access (the information available to an agent that takes no action) saturates the performance metric before active perception can contribute. Five obstruction classes characterized: (a) metric saturation, (b) free-information leak, (c) geometric collision-inevitability, (d) gate-statistic structural opposition, (e) gate-statistic non-reproducibility. **MUST INCLUDE the construction check:** each ceiling gate was independently designed to detect a different structural failure mode. The five obstruction classes were not predicted in advance by one underlying statistic, but emerged from five mechanistically different gate primitives applied to five mechanistically different designs. The migration is therefore not built-in by a single shared metric.
2. **Oracle-bracket discipline:** ORACLE (perfect predictor) vs RANDOM (untrained encoder) bracket to determine whether the task/protocol itself is the bottleneck (branch c) rather than representation quality (branch a) or representation insufficiency (branch b). iter_033 confirmed branch c: ORACLE ≈ RANDOM (gap = 0.0001).
3. **Pre-committed exit rule:** A pre-registered FAIL branch that terminates the environment-design track cleanly, preventing sunk-cost escalation. Fired at iter_038 after Gate-1b failed with std(CV) = 0.320 > 0.25, preventing an estimated ~10-14 iteration 2D rebuild.

### Section 5: LIMITATIONS
Acknowledge WITHOUT reopening (do NOT list as next iterations):
1. **Fixed-Layout design:** Would likely stabilize Gate-1b but was not pursued per pre-commitment and the migrating-obstruction pattern.
2. **Localized spatial readout:** Likely exceeds the mean-pool ceiling given encoder spatial information (iter_031 reconstruction MSE = 0.018), but was not cleanly tested within scope due to iter_032 cross-backbone collapse artefact.
3. **Ceiling-gate generality:** The structural-ceiling-gate methodology was validated within rdf_thalamus only; its generality to tasks outside this project was not demonstrated.

### Section 6: KEY ITERATION INDEX
List with brief descriptions:
- **rdf_thalamus_sml HSUN** (companion project): Pooled VICReg anti-collapse validated; SFA+VICReg beats JEPA+VICReg by ~20pp on temporally-stable binary classification
- **iter_027**: sim_loss_dyn identified as causal driver of z_dyn collapse (B-vs-C: 30% vs 0%); separate backbone alone does NOT eliminate collapse (Second Null)
- **iter_028**: Working representation substrate established — separate-backbone + mask_dyn_sim (C2: 0% collapse, ΔR²=0.514 on fresh seeds); hard-seed pattern (seeds 53, 71) identified; F1 falsified (mask_dyn_sim on shared backbone still collapses 20%)
- **iter_029**: SFA+VICReg on separate backbone — directional trend ΔR²=0.2749 but fails pre-registered 0.30 gate; zero collapse across 60 runs
- **iter_031**: Mean-pool bottleneck localized — Reconstruction+VICReg ceiling probe: reconstruction MSE=0.018 (spatial info in encoder), ΔR²_color=0.063 (mean-pool destroys it); d_max invariance (diff=0.036)
- **iter_032**: Centroid-gated readout causes catastrophic collapse (K=4: 100%, K=1: 10%) — cross-backbone VICReg coupling identified as novel failure mode; branch (b) hard-pivot to behavioral evaluation
- **iter_033-034**: Behavioral pivot protocol — metric saturation discovered (ORACLE≈RANDOM gap=0.0001); free-info leak (MALRE v2 active-passive gap=0.83 vs ORACLE-RANDOM gap=0.031); branch (c) confirmed
- **iter_035**: 1D collision-inevitability — analytical ceiling gate FAILS (PASSIVE=12.27 valid collisions/object, 4× overshoot of ≤3.0 threshold); pass-through physics insufficient
- **iter_036-037**: 2D static-pointer — Gate-1 passes (5/5), Gate-1b/Gate-2 fail; structural opposition between Gate-1 and Gate-1b discovered
- **iter_038**: 2D navigation probe — Gate-1+Gate-2 both pass simultaneously for first time; Gate-1b FAILS (per-seed CV bimodal, std=0.320>0.25); pre-committed FAIL branch fires clean

## Additional quantitative data for the report

iter_027 detailed:
- Arm A (shared, mean readout): collapse 40%, ΔR²=0.1469
- Arm B (separate, JEPA+VICReg): collapse 30%, ΔR² not reported separately
- Arm C (separate, VICReg-only z_dyn): collapse 0%, ΔR²=0.1812
- Arm A' (shared, centroid-gated): collapse 30%

iter_028 detailed:
- D0 (shared, JEPA+VICReg baseline): collapse 30%, ΔR²=0.054, mean_abs_corr=0.999
- C1 (shared, mask_dyn_sim): collapse 20%, ΔR²=0.231
- C2 (separate, VICReg-only z_dyn, seed robustness): collapse 0%, ΔR²=0.514
- C3 (separate, weight-robustness): collapse 20%, ΔR²=0.168
- Hard seeds: 53 and 71 collapsed in C1, C3, and D0
- Parameter count: 80,336 (shared), 135,608 (separate)

iter_031 detailed:
- Arm A (d_max=8, trained): collapse 0%, ΔR²=0.0631, reconstruction MSE=0.0185, centroid MSE=160.32
- Arm B (d_max=2, trained): collapse 0%, ΔR²=0.0271
- Arm C (d_max=8, random encoder): collapse 100%
- d_max differential: ΔR²(A) - ΔR²(B) = 0.036
- CLTS protocol: tracking error 37.61-38.75; collision selectivity 0.44-0.59; perturbation selectivity 0.482-0.606

iter_032 detailed:
- E1 (mean-pool, VICReg-only): collapse 0%, ΔR²=0.131
- E1.5 (centroid-gated scalar, K=1): collapse 10%, ΔR²=0.078
- E2 (centroid-gated rich, K=4, VICReg-only): collapse 100%, ΔR²=0.116
- E3 (centroid-gated rich, K=4, SFA+VICReg): collapse 100%, ΔR²=0.138
- Parameter counts: 135,608 (E1), 151,016 (E2)

iter_033 detailed:
- 4 conditions × 12 seeds = 48 runs
- v3 ORACLE selectivity V-B = 0.5044; v3 RANDOM selectivity V-B = 0.5043; gap = 0.0001

iter_034 detailed:
- v1 (MAPE): FALSIFIED — pointer-object noise sensitivity
- v2 (MALRE): ORACLE = 0.503, RANDOM = 0.534, PASSIVE = 1.333
- Active-passive gap = 0.83; ORACLE-RANDOM gap = 0.031

iter_035 detailed:
- PASSIVE mean valid collisions per object = 12.27
- Analytical ceiling gate threshold = 3.0
- 4× ceiling overshoot

iter_036 detailed:
- Arm A (normal obj-obj): CV=0.3594, mean per-obj count=1.9333
- Arm B (pass-through obj-obj): CV=0.4555, mean per-obj count=2.5333
- Both below CV ≥ 0.50 threshold

iter_037 detailed:
- Gate-1: 5/5 pass, mean valid collisions/object = 0.33
- Gate-1b: 3/5 pass, mean CV = 0.91
- Gate-2: 1/5 pass, mean CV = 0.45
- 1D baseline: 12.27 passive collisions

iter_038 detailed:
- Gate-1: PASS (mean probe count 1.4 ≤ 3.0)
- Gate-2: PASS (5/5 seeds, CV≥0.50)
- Gate-1b: FAIL (mean CV=1.025, std CV=0.320 > 0.25)
- Per-seed CVs: [0.817, 1.414, 0.771, 0.707, 1.414]
- Trajectory coverage mean = 0.3602
- Probe budget utilization = 28%
- Total probes fired = 21

## LANGUAGE DISCIPLINE (from pre-registration)
- Use "is consistent with" not "proves" or "demonstrates"
- Use "we observe under conditions Y" not "we have shown"
- Use "the evidence supports" not "this confirms"
- Use "the mechanism is hypothesised; the ablation was not run within scope" not "the mechanism drives/prevents"
- Flag any quantitative claim whose backing iteration metric cannot be located
- NEVER use "breakthrough", "first concrete proof", "proven", "demonstrated" for any claim

## OUTPUT
Write the complete report to archive/iter_039/final_report.md. The report should be approximately 3000-5000 words, comprehensive, and precisely follow the six-section structure above.