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