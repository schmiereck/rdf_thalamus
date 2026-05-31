You are creating the decision-support package for iter_037 of the Thalamus project. This is a structured analysis for a human go/no-go decision on whether to pursue a 2D environment redesign. You do NOT make the decision — you provide measured evidence and cost analysis.

Read the following files first:
1. `src/pre_registration.md` — the pre-registered gates and criteria
2. `archive/iter_037/results/gate_summary.txt` — the actual gate results
3. `archive/iter_037/results/gate1_results.csv` — per-seed Gate-1/1b data
4. `archive/iter_037/results/gate2_results.csv` — per-seed Gate-2 data
5. `archive/iter_037/results/null_finding_1d.md` — the four-iteration null finding

Then write the decision-support document to `archive/iter_037/results/decision_support.md`.

The document MUST contain ALL of the following sections, with honest reporting:

## 1. Gate Results Summary

Present each gate's result with:
- The pre-registered threshold
- Per-seed values (all 5 seeds individually)
- The per-seed decision rule result (≥4/5?)
- Overall gate pass/fail

## 2. Scientific Analysis of Each Gate

### Gate-1 (PASSIVE Boundedness): PASS
- 2D reduces mean valid collisions from 12.27/object (1D) to 0.0-1.0/object
- This confirms the near-tautological expectation: 2D geometry removes collision inevitability
- Quantitative measurement: PASSIVE pointer at (32,32) in 64×64 with N=3 objects gets very few collisions

### Gate-1b (PASSIVE Collision Heterogeneity): FAIL (3/5 seeds)
- IMPORTANT NUANCE: The failure mode is qualitatively different from anticipated
- Gate-1b was designed to test "are passive collisions heterogeneous enough for ORACLE to exploit differentially?"
- The actual finding: passive collisions are SO RARE that CV is undefined/degenerate
  - Seeds 7, 71: 0 collisions total → CV=0
  - Seeds 31, 83: 1 collision on 1 object → CV=1.414 (mathematically high but meaningless — single-event noise)
  - Seed 53: 3 collisions → CV=0.816 (only meaningful case, and it passes)
- This is actually a consequence of Gate-1 passing "too well" — collisions are so rare that the CV metric loses discriminative power
- The implication: in 2D, ORACLE's advantage is NOT about selecting WHICH objects to collide with more (heterogeneity exploitation), but about WHETHER to collide at all (binary navigation). This is a fundamentally different behavioral structure than the 1D bracket was designed to test.

### Gate-2 (RANDOM Gaze Heterogeneity): FAIL (1/5 seeds)
- CV values: 0.981, 0.202, 0.374, 0.408, 0.288
- Mean CV = 0.45 (below 0.50)
- 2D random walk with gaze_radius=8 and probe_budget=20 still produces fairly uniform coverage across 3 objects in 64×64
- Only seed 7 passes (0,2,7 probes → highly uneven, likely because the gaze happened to get stuck near one object)
- The pre-registration correctly identified that Poisson noise alone gives CV≈0.39; the actual CVs cluster around this value, meaning the 2D random walk adds little clustering beyond Poisson

## 3. Option (iii) Decoder-Free Relaxation: Explicit Rejection

Document that this option is mis-targeted:
- The binding constraint is environmental (4 iterations demonstrate perception is not load-bearing in 1D under any tested configuration)
- Adding a decoder addresses representation quality, NOT behavioral load-bearing
- Reconstruction+VICReg was already tested (iter_031): ΔR²_color=0.063 with mean-pool readout
- The decoder-free constraint is not the current blocker; the environment is
- Relaxing decoder-free would be solving a different problem than the one identified

## 4. What the 2D Gate Results Mean for Path (i)

### The result is AGAINST 2D viability at the pre-registered parameterization
Two of three gates fail. Per the pre-registered criteria, this blocks path (i) at the cheap-gate level for 64×64, N=3, gaze_radius=8.

### But: the failure modes are specific and informative
- Gate-1b fails because collisions are too rare (a different kind of "too few" problem than 1D's "too many")
- Gate-2 fails because 2D random walk coverage is still fairly uniform at this arena size/gaze radius
- These are NOT the same failure modes as 1D; they are genuinely new structural issues

### Potential parameterizations that might change gate outcomes (NOT tested, NOT endorsed):
- Larger arena (e.g., 128×128): might help Gate-2 (random walk covers less) but worsens Gate-1b (even fewer passive collisions)
- Smaller gaze radius (e.g., 4): might help Gate-2 (more clustered coverage) but reduces probe effectiveness
- Fewer objects (N=2): might help both gates (easier to get heterogeneous coverage)
- BUT: these are additional investments with no guarantee, and each would need its own pre-registered gates

### The fundamental tension in 2D:
Gate-1 and Gate-1b are in tension. Gate-1 requires LOW passive collision rates (pointer doesn't hit objects by default). Gate-1b requires DIFFERENT passive collision rates across objects (some hit more than others). In 2D with a static central pointer, if collisions are rare (Gate-1 passes), they tend to be uniformly rare (Gate-1b fails). If they're heterogeneous, it's because of geometric luck, not structural design. This tension may be fundamental to the 2D parameterization.

## 5. What a Full 2D Commitment Would Require (if pursued)

A. Components to build:
1. 2D encoder: 1D-conv → 2D-conv (all 4 conv layers, spatial/dyn heads, soft-argmax over 2D spatial map) — estimated 1-2 iters
2. 2D soft-argmax centroid: must output (B, d_max, 2) coordinates — requires new head design
3. 2D PhysicsSandbox2D: production version with rendering to (3, 64, 64) RGB image — estimated 0.5-1 iter
4. 2D CLTSMotorController: 2D pointer with 2D acceleration — estimated 0.5-1 iter
5. Re-validate non-collapse and semantic encoding in 2D: repeat iter_027-030 work — estimated 2-3 iters
6. Re-validate behavioral bracket in 2D: repeat iter_033-036 work — estimated 2-3 iters
7. Total: ~7-10 additional iterations
8. Compute cost: 2D conv ≈ 4× FLOPs of 1D conv at same resolution

B. What carries over unchanged:
1. M1 batch-VICReg (objective-level, architecture-independent)
2. iter_028 separate-backbone + mask_dyn_sim (0% collapse fix)
3. Decoder-free constraint
4. M3 frozen-dim regime, GDASR log-only
5. Analytical-ceiling-gate + oracle-bracket methodology
6. Pre-committed-rule discipline
7. All metric designs (POMLRE, ΔR²_color, centroid MSE, etc.)

C. Risks:
- Gates failing does NOT guarantee the full bracket would discriminate even if gates were to pass under a different parameterization
- New failure modes may emerge in 2D
- The Gate-1/Gate-1b tension may be fundamental, requiring a fundamentally different behavioral test design (navigation vs. selection)

## 6. Path (ii) Scope (Re-frame Deliverable)

If 2D is not pursued:
- Report representation findings: VICReg-only z_dyn gives 0% collapse, ΔR²_color ≈ 0.045 (best decoder-free) to 0.275 (SFA+VICReg sfa=5.0)
- Report thalamic gating mechanism: surprise-detector + categorizer dual-control, per-channel EMA normalization
- Report motor controller: CLTSMotorController with PD tracking, velocity matching, surprise-triggered push
- Report the four-iteration null as a negative finding: the 1D testbed cannot validate that better perception produces better behavior, which is itself a clean result
- Report the 2D gate measurements: Gate-1 passes (collision inevitability removed), Gate-1b and Gate-2 fail (heterogeneity not achieved at tested parameterization)
- Do NOT claim behavioral validation of the curiosity-driven perception-action thesis

## 7. HEADLINE: Iter_037 Does NOT Make the Decision

**Path selection among (i) 2D rebuild, (ii) deliverable re-frame is a human-scale strategic decision about project scope, cost, and goals.** Iter_037 provides:
- A measured null finding on the 1D testbed (four iterations, four distinct mechanisms)
- Measured 2D gate results at one parameterization (64×64, N=3, gaze_radius=8)
- Cost/scope analysis for a full 2D commitment
- An explicit rejection of option (iii) as mis-targeted

The human researcher must decide whether to:
(a) Invest in exploring different 2D parameterizations that might pass the gates
(b) Commit to a full 2D rebuild based on the Gate-1 result alone (collision inevitability IS removed)
(c) Re-frame the deliverable around representation + mechanism findings without behavioral validation
(d) Another path not yet identified

Write the complete document. Be precise, honest, and restrained in language. Use "is consistent with" / "does not refute" / "provides measured evidence for" — avoid "2D works", "2D validates", "2D solves the 1D problem".