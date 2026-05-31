Write the following complete, pre-drafted scientific project report to `archive/iter_039/final_report.md`. Create the directory if it does not exist. Do not modify or truncate any of the text.

```markdown
# Closing Project Report: Representation Dynamics under Gated Thalamic Subsumption (rdf_thalamus)

## SECTION 1: EXECUTIVE SUMMARY

This report marks the formal closure of the `rdf_thalamus` project, summarizing the scientific contributions, empirical validations, and structural limits observed across 39 iterations. 

The primary scientific contribution of this project is the discovery and characterization of the **MIGRATING-OBSTRUCTION** meta-finding. Across five mechanistically distinct designs spanning two dimensionalities (1D and 2D), the structural obstruction to making unsupervised perception behaviorally load-bearing migrated to a new gate class every time the environment was modified to bypass the previous obstruction. This is consistent with a structural constraint in the underlying design space itself, rather than a failure of individual representation objectives or engineering parameters. 

The five successive obstruction classes emerged as follows:
1. **Metric Saturation** (iter_033): On the initial behavioral-pivot protocol (N=2 objects under elastic dynamics), the performance of an ORACLE (perfect perception) was indistinguishable from RANDOM (untrained encoder), with a post-collision selectivity gap of exactly 0.0001. The metric was saturated by passive combinatorics.
2. **Free-Information Leak** (iter_034): Under the v2 MALRE active-passive protocol, we observed a massive active-passive coverage gap of 0.83, but the ORACLE-RANDOM gap remained at 0.031. Passive object-object collisions leaked sufficient mass-ratio information to the random baseline, rendering targeting policy irrelevant.
3. **Geometric Collision-Inevitability** (iter_035): Restructuring the environment to pass-through physics (removing object-object collisions) in 1D resulted in a PASSIVE (zero-action) pointer experiencing 12.27 valid collisions per object, severely overshooting the pre-registered ceiling threshold of ≤ 3.0 by over 4×. Geometric necessity on a 1D axis guarantees abundant interaction, bypassing any behavioral requirement.
4. **Gate-Statistic Structural Opposition** (iter_036/037): Moving to a 2D static-pointer design successfully passed the collision-inevitability gate (Gate-1 passes 5/5), but Gate-1b and Gate-2 mutually opposed one another. At any single parameterization, we observe under these conditions that rare collisions (Gate-1) and stable coefficient of variation (CV) across seeds cannot be simultaneously satisfied.
5. **Gate-Statistic Non-Reproducibility** (iter_038): Escalate to a 2D navigation probe where the pointer actively navigates under a random-walk policy. While Gate-1 and Gate-2 both passed simultaneously for the first time, Gate-1b failed due to bimodal per-seed CV with std = 0.320, overshooting the pre-registered stability threshold of ≤ 0.25. Bimodality was driven by the random initial placement of objects relative to the navigator's start state.

To prevent sunk-cost escalation, a pre-committed exit rule was pre-registered prior to iteration 038. This rule stated that if any gate in the 2D navigation de-risking probe failed, the behavioral-validation track would be declared intractable within project scope, halting a scheduled ~10-14-iteration 2D engine rebuild. The Gate-1b failure cleanly triggered this exit rule, prompting the immediate closure of the project and the re-framing of its deliverables into six falsifiable claims.

---

## SECTION 2: SIX RE-FRAME CLAIMS

### Claim-A: MIGRATING-OBSTRUCTION
*   **Status:** VALIDATED
*   **Evidence:** The documented iteration chain from iter_033 to iter_038 across five distinct designs and two dimensionalities.
*   **Quantitative Anchors:** 
    *   iter_033: ORACLE-RANDOM gap = 0.0001 on post-collision selectivity.
    *   iter_034: MALRE active-passive gap = 0.83, ORACLE-RANDOM gap = 0.031.
    *   iter_035: PASSIVE = 12.27 collisions/object vs threshold ≤ 3.0.
    *   iter_036: Random gaze event CV = 0.3594 (Arm A) and 0.4555 (Arm B) vs threshold ≥ 0.50.
    *   iter_037: Gate-1 passed 5/5, Gate-2 passed 1/5, Gate-1b passed 3/5.
    *   iter_038: Gate-1b std(CV) = 0.320 vs threshold ≤ 0.25, with bimodal per-seed CVs of [0.817, 1.414, 0.771, 0.707, 1.414].
*   **Proposition:** Unsupervised representation learning cannot be behaviorally validated using an ORACLE-vs-RANDOM bracket in a low-dimensional physical sandbox because environmental modifications designed to make perception load-bearing cause the bottleneck to migrate to a new structural gate class.

### Claim-B: 1D-FORECLOSES-LOAD-BEARING-PERCEPTION
*   **Status:** VALIDATED
*   **Evidence:** The analytical ceiling-gate measurements from iter_035 and iter_036.
*   **Quantitative Anchors:** PASSIVE = 12.27 (4× threshold of ≤ 3.0) in iter_035; CV = 0.3594/0.4555 under foveated gaze (radius = 8) in iter_036.
*   **Proposition:** A 1D arena (128px linear extent, N=3 objects) is structurally incapable of making perception behaviorally load-bearing. On a single spatial axis, collisions are either geometrically inevitable (if the pointer has physical extent) or spatial coverage is too uniform under random motion (if foveated gaze is utilized), foreclosing the possibility of an open perceptual bracket.

### Claim-C: ANALYTICAL-CEILING-GATE characterizes five structurally distinct obstruction classes
*   **Status:** VALIDATED
*   **Evidence:** The successful detection of five distinct failure modes across iter_033–038 without running full training runs, isolating architectural, behavioral, and statistical limits at zero training cost.
*   **Quantitative Anchors:** Managed through the unified chain of gate metrics across iter_033 to iter_038.
*   **Proposition:** Pre-training analytical ceiling gates can identify structural and statistical limitations of behavioral tasks before executing costly neural network optimization loops.

### Claim-D: POOLED-VICREG NECESSITY for anti-collapse
*   **Status:** VALIDATED-IN-COMPANION-PROJECT
*   **Evidence:** The `rdf_thalamus_sml` HSUN companion project. Without batch-level VICReg covariance and variance constraints, every tested unsupervised objective collapsed to downstream semantic accuracy equivalent to random chance (~44-48%). Incorporating pooled VICReg restored representation stability, yielding downstream classification accuracy of 61-83% (SFA+VICReg ≈ 82%, Reconstruction+VICReg ≈ 83%, JEPA+VICReg ≈ 61%), displaying a ~20 percentage point gap favoring non-predictive spatial objectives.
*   **Mandatory Qualifier:** The `rdf_thalamus`-specific ablation (removing VICReg from the dual-stream architecture and measuring collapse rate) was not run within scope. The existing architecture computes VICReg at the batch level (`calc_var_loss` and `calc_cov_loss` on single-frame latents $z \in \mathbb{R}^{B \times d_t}$, batch_size = 32), which is consistent with the stabilization mechanism validated in the companion project.

### Claim-E: SEPARATE-BACKBONE + mask_dyn_sim prevents cross-stream collapse
*   **Status:** WORKING-CONFIGURATION-VALIDATED
*   **Evidence:** iter_028 C2 arm (0% collapse, 0/10 seeds collapsed under dual criteria, mean $\Delta R^2_{\text{color}} \approx 0.514$ on fresh seeds; iter_028 separate backbone baseline color regression $\Delta R^2 \approx 0.045$). In iter_027, the causal driver of $z_{\text{dyn}}$ collapse was localized to the JEPA dynamic prediction loss $\mathcal{L}_{\text{sim\_dyn}}$ (Arm B separate backbone with JEPA+VICReg collapsed 30%, whereas Arm C separate backbone with VICReg-only collapsed 0%, $\Delta R^2_{\text{color}} = 0.1812$).
*   **Mandatory Qualifier:** In the tested configurations (iter_028, 0/5 seeds collapsed in the C2 seed-robustness arm); the causal mechanism is hypothesised and the isolating ablation was not run within scope.

### Claim-F: MEAN-POOL is the binding readout bottleneck not the encoder
*   **Status:** DESCRIPTIVELY-VALIDATED
*   **Evidence:** iter_031 Reconstruction+VICReg ceiling probe. The spatial encoder successfully compressed high-fidelity pixel information, achieving a reconstruction MSE of 0.018. However, when using the mean-pool readout to extract $z_{\text{dyn}}$ features, color decoding performance remained low (mean $\Delta R^2_{\text{color}} = 0.063$ for $d_{\text{max}} = 8$ and $0.027$ for $d_{\text{max}} = 2$). The difference of $0.036$ between $d_{\text{max}}=8$ and $d_{\text{max}}=2$ is below the $0.05$ threshold, showing that spatial averaging acts as an information filter that discards color-spatial association.
*   **Mandatory Qualifier:** Mean-pool is the binding bottleneck in the tested readouts; the localized-readout constructive test was not cleanly executed due to the iter_032 cross-backbone collapse artefact.

---

## SECTION 3: M2 STATUS

"UNTESTABLE-NOT-FALSIFIED within project scope; SFA on z_dyn behavioural improvement was never tested because no valid behavioural bracket was achievable under any tested environment design."

M2 was designated as the project's central representational mandate, positing that Slowness Feature Analysis (SFA) acting on the separate $z_{\text{dyn}}$ dynamic stream would serve as the primary objective for temporal identity binding, demoting the Predictive JEPA objective. 

While proxy-metric evidence showed SFA+VICReg provided a directional improvement in downstream linear color decoding (iter_029 SFA+VICReg achieved mean $\Delta R^2_{\text{color}} = 0.2749$ across 30 seeds, a 6.2× improvement over the VICReg-only baseline of $0.0445$), it failed to pass the pre-registered practical significance threshold of $\ge 0.30$. Furthermore, earlier iterations (iter_023–024) comprehensively falsified the capacity of SFA to produce identity encoding via slowness alone under a shared backbone architecture, where both multi-step SFA and temporal contrastive (NT-Xent) formulations failed to prevent collapse or bind identity features without structural separation. 

The ultimate behavioral validation of M2 — specifically, whether SFA-derived representations outperform predictive (JEPA) or static (VICReg-only) representations in closed-loop control tasks — was never executed. This omission occurred because no environment design across 1D and 2D variations succeeded in producing a valid ORACLE-vs-RANDOM behavioral bracket. Because the behavioral evaluation protocol remained degenerate (ORACLE ≈ RANDOM), SFA's behavioral utility was never tested. Future research requiring a 2D navigation/selection environment should treat M2 as an open, untransferred hypothesis.

---

## SECTION 4: METHODOLOGY

The project has yielded three distinct methodological contributions that formalize research loops in closed-loop unsupervised representation learning:

### 1. Structural-Ceiling-Gate Primitive
Unsupervised representation evaluations typically rely on downstream task performance, confounding representation quality with task and policy design. We developed an analytical gating mechanism that evaluates the environment's information structure *prior* to representation training. By measuring passive information rates (e.g., collisions under passive or random policies), these gates determine if task metrics saturate before active perception can contribute. 

Each ceiling gate was independently designed to detect a different structural failure mode:
*   **Metric Saturation:** Checked if the metric ceiling is mathematically bounded near the random baseline.
*   **Free-Information Leak:** Measured if the passive policy acquires sufficient state data to solve the task.
*   **Geometric Collision-Inevitability:** Verified if the physical boundaries of the arena force interactions.
*   **Gate-Statistic Structural Opposition:** Analyzed if reducing collision rates and maintaining statistical stability are in mathematical conflict.
*   **Gate-Statistic Non-Reproducibility:** Measured if the variance of the evaluation metric across seeds is stable under random initial conditions.

The migration of the structural obstruction was therefore not built-in by a single shared metric; the five obstruction classes were not predicted in advance by one underlying statistic, but emerged from five mechanistically different gate primitives applied to five mechanistically different designs.

### 2. Oracle-Bracket Discipline
To isolate representation failures from task/protocol failures, we established a strict three-condition bracket: RANDOM (untrained encoder baseline), LEARNED (trained representation), and ORACLE (ground-truth perceptual state). Under this discipline:
*   An open bracket requires $\text{ORACLE} - \text{RANDOM} \ge 0.15$, indicating that the task is sensitive to perception.
*   If the bracket is closed ($\text{ORACLE} \approx \text{RANDOM}$), the task or motor controller is identified as the bottleneck (Branch C), preventing false-negative attributions to the representation encoder.

### 3. Pre-Committed Exit Rules
To prevent sunk-cost escalation, we incorporated pre-registered, binding decision rules at key pivot points. In iter_038, the pre-registered exit rule declared that if any gate in the 2D navigation probe failed, the behavioral-validation strategy would be abandoned, preventing an estimated 10-14 iteration engineering rebuild of the 2D physics engine. The clean firing of the FAIL branch on Gate-1b ($\text{std}(CV) = 0.320 > 0.25$) successfully terminated the track.

---

## SECTION 5: LIMITATIONS

We acknowledge several structural limitations of the completed project, preserved here without reopening:

1.  **Fixed-Layout Design Constraints:** In the 2D navigation probe (iter_038), Gate-1b failed because the random walk interacted with uniform-random object placement, creating bimodal coverage statistics depending on the proximity of objects to the starting coordinate. A fixed-layout environment (holding object coordinates constant across seeds) would likely stabilize Gate-1b. This was not pursued due to pre-committed exit criteria and the broader migrating-obstruction pattern.
2.  **Unresolved Localized Spatial Readout:** Pixel reconstruction in iter_031 (MSE = 0.018) indicates that the spatial encoder retains identity features, but mean-pooling destroys this information. While localized spatial readouts (such as centroid-gating, iter_032) are mathematically positioned to bypass this bottleneck, they were not cleanly tested within scope due to the cross-backbone attention coupling collapse artefact discovered in iter_032.
3.  **Ceiling-Gate Generality:** The structural-ceiling-gate methodology was validated exclusively within the physical sandbox and CLTS environments of the `rdf_thalamus` project. Its mathematical generality to tasks outside this specific closed-loop control family remains untested.

---

## SECTION 6: KEY ITERATION INDEX

### Companion Project: `rdf_thalamus_sml` (HSUN)
*   **Key Results:** Validated the necessity of pooled batch-level VICReg for preventing representation collapse. Demonstrated that SFA+VICReg outscores JEPA+VICReg by ~20 percentage points on temporally-stable downstream binary classification tasks.

### Iteration 027: sim_loss_dyn localized as collapse driver
*   **Key Results:** Falsified the hypothesis that the shared CNN backbone is the primary driver of collapse (Second Null: separate backbone Arm B collapsed 30%). Localized the causal driver of $z_{\text{dyn}}$ collapse to the predictive similarity loss $\mathcal{L}_{\text{sim\_dyn}}$ (Arm C separate backbone with VICReg-only collapsed 0%, $\Delta R^2_{\text{color}} = 0.1812$, Arm A' shared centroid collapsed 30%).

### Iteration 028: Working representation substrate established
*   **Key Results:** Established a robust non-collapsing substrate using a separate backbone architecture with dynamic similarity masked ($\text{mask\_dyn\_sim} = \text{True}$).
*   **Metrics:** Arm C2 achieved 0% collapse, mean $\Delta R^2_{\text{color}} = 0.514$ on fresh seeds. Under shared backbone (Arm C1), collapse was 20%. Hard-seed patterns identified on seeds 53 and 71. Parameter count: 135,608 (separate) vs 80,336 (shared).

### Iteration 029: SFA+VICReg on separate backbone
*   **Key Results:** Tested M2 on the separate backbone. SFA+VICReg converged (final SFA loss = 0.1408) and showed directional improvement (mean $\Delta R^2_{\text{color}} = 0.2749$ across 30 seeds), but failed the pre-registered significance gate ($\ge 0.30$) with high run-to-run instability ($\sigma = 0.577$). Collapse rate was 0%.

### Iteration 031: Mean-pool bottleneck localized
*   **Key Results:** Ran Reconstruction+VICReg ceiling probe. Reached reconstruction MSE of 0.018, but downstream linear decoding remained low (mean $\Delta R^2_{\text{color}} = 0.0631$ for $d_{\text{max}} = 8$ and $0.0271$ for $d_{\text{max}} = 2$). Localized the readout bottleneck to mean-pool aggregation ($d_{\text{max}}$ differential = 0.036). Established the CLTS tracking calibration (tracking error 37.61-38.75 px).

### Iteration 032: Centroid-gated readout collapse
*   **Key Results:** Attempted centroid-gated readout to resolve the mean-pool bottleneck. Discovered a new failure mode: cross-backbone VICReg coupling. 
*   **Metrics:** Arm E2 (K=4) collapsed 100% (mean $\Delta R^2 = 0.116$); Arm E1.5 (K=1) collapsed 10% (mean $\Delta R^2 = 0.078$); Arm E1 control (mean-pool) collapsed 0% ($\Delta R^2 = 0.131$). Triggered pre-committed pivot to behavioral evaluation.

### Iteration 033-034: Behavioral pivot and metric saturation
*   **Key Results:** Swapped to a surprise-driven attention task under CLTSMotorController. Discovered metric saturation under N=2 combinatorics: ORACLE post-collision selectivity was 0.5044 and RANDOM was 0.5043 (gap = 0.0001). Iteration 034 MALRE v2 opened the active-passive gap to 0.83, but the ORACLE-RANDOM gap was 0.031, revealing a free-information leak from passive collisions.

### Iteration 035: 1D collision-inevitability
*   **Key Results:** Removed object-object collisions (pass-through physics) in 1D to eliminate the free-information leak. The analytical ceiling gate failed: a PASSIVE pointer registered 12.27 valid collisions per object (over 4× the pre-registered limit of ≤ 3.0), showing that collisions are geometrically inevitable on a 1D shared axis.

### Iteration 036-037: 2D static-pointer foveated gaze
*   **Key Results:** Replaced physical pointer with foveated gaze (radius = 8) to bypass collision inevitability. In 1D (iter_036), RANDOM gaze event CV was 0.36 (Arm A) and 0.46 (Arm B), failing the coverage heterogeneity gate (CV ≥ 0.50). Moving to a 2D static-pointer (iter_037) resolved collision inevitability (Gate-1 passed 5/5, passive rate = 0.33), but Gate-1b and Gate-2 failed due to structural opposition under static observation.

### Iteration 038: 2D navigation probe and exit rule firing
*   **Key Results:** Implemented a moving 2D pointer that navigates under a random-walk policy to decouple Gate-1 and Gate-2. 
*   **Metrics:** Gate-1 passed (non-saturating passive collisions); Gate-2 passed (CV ≥ 0.50). However, Gate-1b failed due to non-reproducibility across seeds (std = 0.320 > 0.25, CVs = [0.817, 1.414, 0.771, 0.707, 1.414]). Trajectory coverage mean was 0.3602, probe budget utilization was 28%, and 21 probes were fired. The pre-committed FAIL exit rule fired cleanly, terminating the behavioral validation track and closing the project.
```

---

Upon successful file write, verify that `archive/iter_039/final_report.md` exists and matches the provided text. Let me know when you are done.
