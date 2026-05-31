# Decision-Support Package: iter_037 — 2D Environment Gates

*   **Iteration:** 037
*   **File:** `archive/iter_037/results/decision_support.md`
*   **Purpose:** Measured evidence and cost analysis for a human go/no-go decision on path (i) — 2D environment redesign
*   **Note:** This document does NOT make the decision. It provides structured evidence for the human researcher to evaluate.

---

## 1. Gate Results Summary

### Pre-Registered Parameters

| Parameter | Value |
|-----------|-------|
| Arena | 64×64 pixels |
| N (objects) | 3 |
| Object radius | 3.0 – 8.0 (mass = radius) |
| Object velocity per component | [−2.0, −0.5] ∪ [0.5, 2.0] |
| Pointer | radius=4.0, mass=10.0, start=(32,32) |
| Gaze radius (Gate-2) | 8 pixels |
| Probe budget (Gate-2) | 20 |
| Probe probability (Gate-2) | 0.01 |
| Steps / Substeps | 2000 / 10 |
| Seeds | [7, 31, 53, 71, 83] |
| Decision rule | ≥4/5 seeds must individually pass each gate |

### Gate-1 (PASSIVE Boundedness): **PASS** (5/5 seeds)

**Threshold:** Mean per-object valid collisions ≤ 3.0

| Seed | Valid collisions (obj 0, 1, 2) | Mean valid | Decision |
|------|-------------------------------|------------|----------|
| 7    | (0, 0, 0)                     | 0.00       | PASS     |
| 31   | (0, 1, 0)                     | 0.33       | PASS     |
| 53   | (0, 1, 2)                     | 1.00       | PASS     |
| 71   | (0, 0, 0)                     | 0.00       | PASS     |
| 83   | (0, 1, 0)                     | 0.33       | PASS     |

**Per-seed rule:** 5/5 seeds ≤ 3.0 → **PASS**

### Gate-1b (PASSIVE Collision Heterogeneity): **FAIL** (3/5 seeds)

**Threshold:** CV of per-object valid collision counts ≥ 0.30

| Seed | Valid collisions (obj 0, 1, 2) | Mean | CV valid | Decision |
|------|-------------------------------|------|----------|----------|
| 7    | (0, 0, 0)                     | 0.00 | 0.000    | FAIL     |
| 31   | (0, 1, 0)                     | 0.33 | 1.414    | PASS     |
| 53   | (0, 1, 2)                     | 1.00 | 0.816    | PASS     |
| 71   | (0, 0, 0)                     | 0.00 | 0.000    | FAIL     |
| 83   | (0, 1, 0)                     | 0.33 | 1.414    | PASS     |

**Per-seed rule:** 3/5 seeds ≥ 0.30 → **FAIL** (requires ≥4/5)

### Gate-2 (RANDOM Gaze Heterogeneity): **FAIL** (1/5 seeds)

**Threshold:** CV of per-object probe-event counts ≥ 0.50

| Seed | Probes (obj 0, 1, 2) | Mean | CV | Total fired | Decision |
|------|---------------------|------|-----|-------------|----------|
| 7    | (0, 2, 7)           | 3.00 | 0.981 | 20        | PASS     |
| 31   | (2, 2, 3)           | 2.33 | 0.202 | 18        | FAIL     |
| 53   | (3, 2, 5)           | 3.33 | 0.374 | 20        | FAIL     |
| 71   | (1, 2, 3)           | 2.00 | 0.408 | 20        | FAIL     |
| 83   | (3, 6, 4)           | 4.33 | 0.288 | 20        | FAIL     |

**Per-seed rule:** 1/5 seeds ≥ 0.50 → **FAIL** (requires ≥4/5)

### Overall Gate Assessment

| Gate | Result | Seeds Passing |
|------|--------|--------------|
| Gate-1 (PASSIVE Boundedness) | **PASS** | 5/5 |
| Gate-1b (PASSIVE Heterogeneity) | **FAIL** | 3/5 |
| Gate-2 (RANDOM Gaze Heterogeneity) | **FAIL** | 1/5 |

**Per pre-registered criteria:** Two gates failed. At the tested parameterization (64×64, N=3, gaze_radius=8), the 2D environment does not sufficiently relax all three 1D structural constraints simultaneously. Path (i) is **blocked at the cheap-gate level** for this parameterization.

---

## 2. Scientific Analysis of Each Gate

### Gate-1 (PASSIVE Boundedness): PASS

**The result is consistent with the pre-registered hypothesis.** In 1D, the PASSIVE pointer accumulated 12.27 valid collisions per object because all entities share the axis — the collision cross-section equals the full arena width, making collisions geometrically inevitable. In 2D, the mean valid collisions across all 5 seeds range from 0.00 to 1.00 per object, which is well below the 3.0 threshold and represents a ≥12× reduction relative to the 1D baseline.

This confirms the near-tautological expectation: 2D geometry removes collision inevitability. A passively centered pointer at (32, 32) in a 64×64 arena with N=3 objects receives very few collisions because objects can pass the pointer off-axis in the y-dimension. The effective 2D collision probability per object trajectory is approximately O(r_ptr / W)² ≈ (8/64)² ≈ 1.6% per axis crossing, meaning the vast majority of object trajectories simply bypass the pointer.

**Scientific significance:** Gate-1 passing demonstrates that 2D removes one of the two structural constraints that defeated the 1D testbed (collision inevitability). The quantitative measurement is unambiguous: no seed exceeds even one-third of the 1D collision rate.

**Caveat:** Gate-1 passing alone is not sufficient for 2D viability. The pre-registration correctly noted that Gate-1 is near-tautological ("2D has more room for things to not hit each other than 1D") — which is precisely why Gate-1b was added as a complement. The meaningful question is not whether 2D reduces collisions, but whether it reduces them in a way that preserves exploitable structure.

### Gate-1b (PASSIVE Collision Heterogeneity): FAIL (3/5 seeds)

**IMPORTANT NUANCE: The failure mode is qualitatively different from what was anticipated in the pre-registration.**

Gate-1b was designed to test whether passive collisions are heterogeneous enough across objects for ORACLE to exploit differentially. The pre-registration reasoned that in 2D, objects at different positions and velocities would have geometrically different encounter probabilities with the static pointer, yielding a CV ≥ 0.30.

The actual finding is that **passive collisions are so rare that the CV metric loses discriminative power**:

* **Seeds 7, 71:** 0 total collisions across all objects → CV = 0.000. This is not "low heterogeneity" — it is the complete absence of events. With no collisions, there is no data from which to compute heterogeneity.
* **Seeds 31, 83:** 1 collision total (on object 1 only) → CV = 1.414. Mathematically, this exceeds the 0.30 threshold and would pass. But it is meaningless: a single-event occurrence produces a mechanically high CV that reflects Poisson noise, not structural heterogeneity.
* **Seed 53:** 3 collisions total (0, 1, 2 across objects) → CV = 0.816. This is the only seed with enough events to make the CV metric genuinely informative, and it passes.

**This is actually a consequence of Gate-1 passing "too well."** The reduction in passive collisions is so effective (from 12.27/object in 1D to 0.0–1.0/object in 2D) that there are too few events left to measure heterogeneity meaningfully. The CV metric requires a minimum event rate to be stable; below that rate, CV becomes dominated by sampling noise rather than structural signal.

**Implication for the behavioral bracket:** In 2D at this parameterization, ORACLE's advantage would not come from exploiting heterogeneous collision rates across different objects (the mechanism the 1D bracket assumed). Instead, it would need to come from the binary question of **whether to collide at all** — navigating toward or away from objects as a categorical choice. This is a fundamentally different behavioral structure than the 1D bracket was designed to test. The 1D bracket assumed that perception determines *which* objects to interact with more; in 2D, perception would determine *whether* to interact with any object. The bracket design would need to be restructured to test this different behavioral question.

### Gate-2 (RANDOM Gaze Heterogeneity): FAIL (1/5 seeds)

**The result is consistent with the pre-registration's calibration analysis.** The pre-registration correctly estimated that under pure Poisson noise, with probe_budget=20 distributed across 3 objects (mean ≈ 6.67/object), the expected CV ≈ √(6.67) / 6.67 ≈ 0.39 — already below the 0.50 threshold.

The actual CV values are: 0.981 (seed 7), 0.202 (seed 31), 0.374 (seed 53), 0.408 (seed 71), 0.288 (seed 83). Four of five seeds cluster tightly around the Poisson baseline of ~0.39:

* Seeds 31 (0.202), 53 (0.374), and 71 (0.408) are all in the Poisson-noise band, indicating the 2D random walk adds little spatial clustering beyond what you would expect from random allocation of 20 probes among 3 objects.
* Seed 83 (0.288) is even *below* the Poisson expectation, suggesting the gaze trajectory was unusually well-distributed.
* Only seed 7 (CV = 0.981) shows genuine heterogeneity: the gaze happened to fire 0, 2, 7 probes across the three objects, producing a heavily skewed distribution. This is likely because the random walk trajectory got "stuck" in one region near object 2, rather than reflecting a structural property of the arena design.

**Interpretation:** A 2D random walk with gaze_radius=8 in a 64×64 arena with 3 objects still produces fairly uniform coverage across objects over 2000 steps. The random walk mixes well enough that its spatial coverage — while less perfectly uniform than 1D — is still not clustered enough to produce CV ≥ 0.50 consistently. The 2D arena at this size is small enough relative to the gaze radius that the gaze can reach all objects within a reasonable number of steps.

**This is a new structural issue, distinct from 1D.** In 1D (iter_036), coverage uniformity was driven by the fact that all objects share the axis — there is no "far" object. In 2D, objects *can* be far apart spatially, but the random walk with gaze_radius=8 is still effective enough at sweeping the arena that it produces near-uniform coverage on the timescale of 2000 steps.

---

## 3. Option (iii) Decoder-Free Relaxation: Explicit Rejection

**This option is mis-targeted and should not be pursued as a response to the current findings.**

The binding constraint identified across four iterations is **environmental**, not representational. The four-iteration null finding documents that perception is not behaviorally load-bearing in the 1D testbed under any of four distinct mechanism redesigns. The environment itself — not the quality of the agent's representations — is what prevents the behavioral bracket from discriminating.

Adding a decoder (allowing reconstruction loss in addition to VICReg) addresses **representation quality**, not behavioral load-bearing. A better representation of a non-discriminative environment remains a non-discriminative environment. A more precise map of a maze where all paths lead to the same destination is not a solution to the maze's structural problem.

This is not a hypothetical concern. **Reconstruction + VICReg was already tested in iter_031** (the last iteration before the 1D behavioral bracket work began): it produced ΔR²_color = 0.063 with mean-pool readout. The decoder reconstructs pixels adequately, but the mean-pool spatial bottleneck prevents identity encoding regardless. More critically, even if a decoder improved ΔR²_color, the environmental constraints documented in iter_033–036 would remain unchanged.

**The decoder-free constraint is not the current blocker; the environment is.** Relaxing decoder-free would be solving a different problem than the one that has been identified. It would divert development effort toward representation quality improvements that have no evidence of being behaviorally load-bearing — a pattern the project has already demonstrated does not produce discrimination.

---

## 4. What the 2D Gate Results Mean for Path (i)

### The result is AGAINST 2D viability at the pre-registered parameterization

Two of three gates fail at the tested parameterization (64×64, N=3, gaze_radius=8). Per the pre-registration, this blocks path (i) at the cheap-gate level. The evidence does not support proceeding to a full 2D behavioral bracket with these parameters.

### But: the failure modes are specific and informative

The gates that fail did not fail for the same reasons the 1D gates failed:

* **Gate-1b fails because collisions are too rare** — a different kind of "too few" problem than 1D's "too many." In 1D, the problem was that there were too many passive collisions (12.27/object), leaving no room for ORACLE to show improvement. In 2D, there are too few (0–1/object), leaving no data from which to measure heterogeneity. This is not a contradiction of Gate-1; it is a consequence of Gate-1 working more effectively than expected.
* **Gate-2 fails because 2D random walk coverage is still fairly uniform** at this arena size relative to gaze radius. This is a genuinely new structural issue arising from the interaction of arena dimensions and gaze parameters, not a recapitulation of the 1D coverage problem.

These are NOT the same failure modes as 1D. They provide measured evidence that 2D changes the problem structure, even if it does not fully solve it at the tested parameterization.

### Potential parameterizations that might change gate outcomes

The following parameterizations are identified as plausible alternatives. **These are NOT tested, NOT endorsed, and NOT evidence that any of them would pass. They are noted only to be explicit about what the cheap-gate result does and does not rule out.**

| Modification | Expected effect on Gate-1 | Expected effect on Gate-1b | Expected effect on Gate-2 | Trade-off |
|---|---|---|---|---|
| Larger arena (e.g., 128×128) | Likely still passes (even fewer collisions) | Likely worsens (even fewer events to measure heterogeneity) | May improve (random walk covers less of a larger arena) | Improves Gate-2 at cost of Gate-1b |
| Smaller gaze radius (e.g., 4) | No direct effect | No direct effect | May improve (more spatially clustered coverage) | Reduces probe effectiveness |
| Fewer objects (N=2) | Likely still passes | Effects unclear (fewer objects means less room for heterogeneity) | May improve (fewer objects → easier to get uneven coverage) | Changes the problem being tested |

Each of these would require its own pre-registered gate experiment. The cheap-gate design exists precisely to test one parameterization cheaply before committing; a second parameterization would consume another iteration of gate-experiment work with no guarantee of passing.

### The fundamental tension in 2D

There is a structural tension between Gate-1 and Gate-1b that may be fundamental to 2D parameterizations with a static central pointer:

* **Gate-1 requires LOW passive collision rates.** The pointer should not hit objects by default, so that ORACLE's targeted collisions represent a meaningful deviation from the baseline.
* **Gate-1b requires DIFFERENT passive collision rates across objects.** Some objects should be hit more than others, so that heterogeneity exists for ORACLE to exploit.

In 2D with a static central pointer, if collisions are rare (Gate-1 passes well), they tend to be **uniformly rare** (Gate-1b fails). This is because at low collision rates, the difference in encounter probability between objects becomes dominated by geometric luck rather than structural design. If collisions are heterogeneous enough to achieve Gate-1b (as in seed 53 with CV=0.816), it is usually because the total collision count is higher (3 events), which risks pushing toward the Gate-1 ceiling.

The two gates are not logically mutually exclusive — seed 53 shows they can both be satisfied with sufficient events. But at the 64×64, N=3 parameterization, the collision rate is so low that the conditions for both gates to pass simultaneously are fragile. This tension may be inherent to any 2D setup with a static pointer; resolving it might require a fundamentally different behavioral test design — one based on **active navigation** (does the agent choose where to go?) rather than **selective attention** (does the agent choose which object to perceive?).

---

## 5. What a Full 2D Commitment Would Require (if pursued)

### A. Components to build

| # | Component | Description | Estimated effort |
|---|-----------|-------------|-----------------|
| 1 | **2D encoder** | Replace 1D-conv layers with 2D-conv layers (all 4 conv layers in the separate-backbone architecture). Spatial and dynamic heads both need redesign. Soft-argmax must operate over a 2D spatial map rather than a 1D line. | 1–2 iterations |
| 2 | **2D soft-argmax centroid** | Output must be (B, d_max, 2) coordinates instead of (B, d_max, 1). Soft-argmax over 2D requires a different parameterization (temperature, grid handling, boundary treatment). | Included in #1 |
| 3 | **2D PhysicsSandbox2D (production)** | Production version with rendering to (3, 64, 64) RGB image input for the encoder. The gate-experiment version is a minimal test harness; the production version needs proper image rendering, batched stepping, and gradient-compatible interfaces. | 0.5–1 iteration |
| 4 | **2D CLTSMotorController** | Extends CLTSMotorController to 2D: 2D acceleration commands, 2D PD tracking, 2D velocity matching. The surprise-triggered push mechanism needs 2D direction handling. | 0.5–1 iteration |
| 5 | **Re-validate non-collapse and semantic encoding** | Repeat iter_027–030 work (VICReg collapse prevention, ΔR²_color measurement, SFA integration, separate-backbone validation) in the 2D setting. Architectural changes may introduce new failure modes (e.g., 2D convs may have different collapse dynamics). | 2–3 iterations |
| 6 | **Re-validate behavioral bracket** | Repeat iter_033–036 work (POMLRE, MALRE, ORACLE-vs-RANDOM bracket) in the 2D setting. The behavioral test itself may need redesign (see Gate-1/Gate-1b tension above). | 2–3 iterations |
| | | **Total estimated effort: ~7–10 additional iterations** | |

**Compute cost:** A 2D convolution has approximately 4× the FLOPs of a 1D convolution at the same spatial dimension (because the kernel operates in two dimensions). For a 64×64 input, training time per iteration would increase significantly, potentially reducing the number of seeds/runs affordable within the same compute budget.

### B. What carries over unchanged

The following are architecture-independent or design-level components that would not need modification:

1. **M1 batch-VICReg** — The VICReg objective (invariance, variance, covariance regularization) operates at the representation level and is architecture-independent.
2. **iter_028 separate-backbone + mask_dyn_sim** (0% collapse fix) — The separate-backbone technique and the dynamic-similarity masking for preventing dynamic-branch collapse are architectural patterns that carry over to 2D convs.
3. **Decoder-free constraint** — The standing commitment to not use a pixel-reconstruction decoder remains in force.
4. **M3 frozen-dim regime, GDASR log-only** — The analysis-level protocol for frozen dimensions and log-only GDASR work is methodology-independent.
5. **Analytical-ceiling-gate + oracle-bracket methodology** — The scientific method (pre-register gates, run cheap experiments, measure headroom, commit or reject) transfers directly to 2D.
6. **Pre-committed-rule discipline** — The rule-based decision framework carries over unchanged.
7. **All metric designs** — POMLRE, ΔR²_color, centroid MSE, MALRE, and other metric designs are definition-level, not architecture-level.

### C. Risks

* **Gates failing does not guarantee the full bracket would discriminate even if gates were to pass under a different parameterization.** The gates are necessary but not sufficient preconditions. Passing all three gates with different parameters would be measured evidence for 2D viability, not proof.
* **New failure modes may emerge in 2D** that have no analogy in the 1D null chain. For example, 2D spatial encoding may have its own collapse modes, or the 2D motor controller may have stability issues not present in 1D.
* **The Gate-1/Gate-1b tension may be fundamental**, meaning that no single 2D parameterization with a static pointer can simultaneously satisfy both gates to a robust degree. This would require a fundamentally different behavioral test design — not a parameterization change but a redesign of what behavior is being tested (navigation vs. selection).

---

## 6. Path (ii) Scope (Re-Frame Deliverable)

If the decision is made not to pursue 2D, the project can produce a coherent deliverable framed around the **representation and mechanism findings**, explicitly reporting the environmental null finding rather than treating it as an unresolved problem.

The deliverable would include:

* **Representation findings:** VICReg-only z_dyn gives 0% collapse. ΔR²_color ≈ 0.045 (best decoder-free result) to ≈ 0.275 (SFA+VICReg at sfa=5.0). These document what the separate-backbone VICReg architecture achieves for object-identity encoding in the constrained setting, independent of behavioral validation.
* **Thalamic gating mechanism:** The surprise-detector + categorizer dual-control system with per-channel EMA normalization, as developed in the M2 lineage. This is a novel mechanism for dynamic routing between spatial and dynamic representations based on prediction error.
* **Motor controller:** CLTSMotorController with PD tracking, velocity matching, and surprise-triggered push. A parameterized motor control architecture that operates downstream of perception.
* **The four-iteration null as a negative finding:** "The 1D × N=3 × 128px sandbox cannot make perception behaviorally load-bearing under an ORACLE-vs-RANDOM bracket, across four mechanism-distinct redesigns." This is a clean, citable negative result that documents where perception is not the bottleneck.
* **2D gate measurements:** Gate-1 passes (collision inevitability IS removed in 2D). Gate-1b and Gate-2 fail (heterogeneity not achieved at the tested parameterization). These provide measured evidence for the partial promise and current limitations of 2D.
* **NOT included:** A claim of behavioral validation of the curiosity-driven perception-action thesis. The project would not claim to have demonstrated that better perception produces better behavior, because the environment structure prevented this from being testable in 1D and the cheap gates did not support it in 2D.

---

## 7. HEADLINE: Iter_037 Does NOT Make the Decision

### Path selection among (i) 2D rebuild and (ii) deliverable re-frame is a human-scale strategic decision about project scope, cost, and goals.

Iter_037 provides the following to inform that decision:

* **A measured null finding on the 1D testbed:** Four iterations, four distinct mechanisms, all converging on the same structural limitation of 1D geometry with a central interaction target. This is a well-documented negative result.
* **Measured 2D gate results at one parameterization (64×64, N=3, gaze_radius=8):** Gate-1 passes (collision inevitability is removed). Gate-1b fails (collisions too rare for heterogeneity measurement). Gate-2 fails (2D random walk coverage is still too uniform).
* **Cost/scope analysis for a full 2D commitment:** ~7–10 additional iterations, with ~4× FLOP increase per training step, with no guarantee of behavioral discrimination even if gate parameters are adjusted.
* **An explicit rejection of option (iii) as mis-targeted:** Decoder relaxation does not address the environmental constraint that has been identified as the binding bottleneck.

The human researcher must decide whether to:

* **(a)** Invest in exploring different 2D parameterizations (larger arena, different gaze radius, different object count) that might pass the gates — with the understanding that each new parameterization requires its own pre-registered gate experiment and may encounter the same or new failure modes.
* **(b)** Commit to a full 2D rebuild based on the Gate-1 result alone — accepting that collision inevitability IS removed in 2D, and that the full bracket may still discriminate even if Gate-1b and Gate-2 fail at the current parameterization, while taking on the ~7–10 iteration cost with associated uncertainty.
* **(c)** Re-frame the deliverable around representation + mechanism findings without behavioral validation — producing a coherent contribution on VICReg-based spatial encoding, thalamic gating, and motor control, while explicitly reporting the 1D null and 2D gate results as boundary conditions rather than failures.
* **(d)** Pursue another path not yet identified.

Each option has different implications for project scope, time investment, compute cost, and the nature of the final contribution. None of these implications can be automatically derived from the gate results alone. **The gates provide measured evidence; the human researcher provides the strategic context.**

---

*Document generated from iter_037 results. All numerical values sourced directly from gate experiment output files. Interpretations are constrained to what the measurements support.*
