# Research Manager Log - Iteration 037

## Iteration 037 -> Planner [Pre-Planning Hints]

## Manager's Pre-Planning Hints (for iter_037)

**1 — Iter_037 is a decision-support and de-risking iteration, not a meta-decision execution. Do not autonomously pick among paths (i)/(ii)/(iii).** The choice between a 2D rebuild, deliverable re-frame, or relaxing decoder-free is a human-scale strategic call about scope, cost, and project identity. Iter_037's deliverables are: (a) the four-iteration environment-design null crystallized as a standalone, citable finding (iter_033 saturation, iter_034 free information, iter_035 collision-inevitability 12.27 vs 3.0, iter_036 coverage-uniformity CV 0.36/0.46 vs 0.50), and (b) a structured decision package for the human with explicit costs, what carries over, and what must be rebuilt. The planner must explicitly state that the path selection is left to the human.

**2 — Apply the project's own validated discipline to the most expensive option before committing to it. Pre-register and run ONLY the cheap analytical gates for a minimal 2D arena — no training, no learned encoder, no ORACLE bracket, no representation re-architecture.** The two gates must mirror the 1D failures and address both dual-failure-mode conditions: Gate-1 (boundedness) — under a non-perceptual 2D pointer in a 2D arena, PASSIVE per-object collision count ≤ 3.0, since off-axis trajectories can no longer be collision-inevitable; Gate-2 (heterogeneity) — under RANDOM 2D gaze, per-object probe-event CV ≥ 0.50, since a finite random 2D walk cannot cover area as uniformly as a 1D line. Both gates must be pre-registered with arena size, N, gaze radius, rollout length, and seeds fixed in advance. Failing either gate kills path (i) before any 2D engineering investment; passing both produces measured evidence for an informed human go/no-go, not hope.

**3 — Hold scientific discipline tight on three specific risks.** (a) Path (iii) decoder-relaxation should be explicitly rejected as mis-targeted in iter_037's analysis: the binding constraint is environmental (perception not load-bearing under any tested 1D regime), not representational, and a mean-pool reconstruction readout was already tested and failed in iter_031 — adding a decoder does not explain why behavior does not need perception, and the planner must say so. (b) The four-iteration null permits the statement "the 1D × N=3 × 128px sandbox cannot make perception behaviorally load-bearing under an ORACLE-vs-RANDOM bracket across four mechanism-distinct redesigns" — it does **not** permit "no 1D environment could ever work" nor any claim about M2; phrase the finding with that restraint. (c) The 2D cheap-gate result, whether pass or fail, must be reported as a single-condition geometric-statistical measurement at one pre-registered radius/arena/N — not as validation that 2D "works"; passing the gate only unblocks a human decision, it does not establish that the full 2D bracket would discriminate.

---

## Iteration 037 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
In a minimal 2D physics arena (64×64 pixels, N=3 circular objects with radius
∈ [3,8], velocity components ∈ [-2.0,-0.5]∪[0.5,2.0], mass ∝ radius, substeps=10),
the two structural constraints that defeated the 1D testbed are removed:

Gate-1 (Boundedness): A PASSIVE physical pointer (radius=4, mass=10, starting
at arena center (32,32) with zero velocity) accumulates ≤ 3.0 valid elastic
collisions per object over 2000 steps, because objects can pass the pointer
off-axis in the y-dimension (unlike 1D where all entities share the axis).
A "valid collision" is defined identically to iter_035: proximity
(2D distance < r_ptr + r_obj + 4.0) AND |Δv| > 0.5 per component.

Gate-2 (Heterogeneity): Under a RANDOM ghostly gaze policy (2D acceleration
∈ [-10,10]², probe probability p=0.01, gaze_radius=8 pixels, probe budget=20)
the coefficient of variation (std/mean) of per-object probe-event counts is
≥ 0.50, because a finite 2D random walk with bounded step size cannot cover
a 2D arena as uniformly as a 1D random walk covers a line segment.

Both gates are necessary preconditions for a 2D ORACLE-vs-RANDOM bracket to
have headroom. Neither gate is sufficient to establish that the full 2D
bracket would discriminate — passing only unblocks a human go/no-go decision.

**Proposed Falsification Criterion:**
The hypothesis is falsified if EITHER gate fails:

Gate-1 FAIL: Mean per-object valid collision count under PASSIVE > 3.0
(averaged across 5 seeds: [7, 31, 53, 71, 83]). This means 2D does not
sufficiently reduce the free-information ceiling — the pointer still gets
too many collisions without targeted perception, and path (i) is blocked
at the cheap-gate level.

Gate-2 FAIL: Mean per-object probe-event CV under RANDOM gaze < 0.50
(averaged across 5 seeds). This means 2D random gaze coverage is still
too even for ORACLE targeting to have headroom, and path (i) is blocked.

If BOTH gates pass, this is NOT validation that 2D "works" — it is only
measured evidence that the two 1D structural constraints (collision
inevitability, coverage uniformity) are relaxed in 2D at the tested
parameterization (64×64, N=3, gaze_radius=8). The full bracket result
remains unknown.

**Proposed Method:**
Step 1 — Four-Iteration Null Finding Document:
Write a standalone, citable markdown document crystallizing the
environment-design null chain across iter_033–036:
  - iter_033: behavioral-pivot metric saturation (ORACLE≈RANDOM, gap 0.0001)
  - iter_034: free autonomous information (MALRE active-passive gap=0.83,
    but ORACLE-RANDOM gap=0.031, 3/8 seeds)
  - iter_035: 1D collision inevitability (PASSIVE 12.27 per-object vs 3.0
    ceiling gate — 4× overshoot)
  - iter_036: coverage uniformity (RANDOM CV 0.36/0.46 vs 0.50 threshold —
    both arms fail)
Phrased precisely: "the 1D × N=3 × 128px sandbox cannot make perception
behaviorally load-bearing under an ORACLE-vs-RANDOM bracket across four
mechanism-distinct redesigns." Does NOT claim "no 1D environment could
ever work" or make any claim about M2.

Step 2 — 2D Cheap Gate Experiment (src/run_iter037_2d_gates.py):
A. Implement PhysicsSandbox2D — minimal 2D physics for gate tests:
   - 2D positions (N×2), 2D velocities (N×2), radii, masses, colors
   - 2D elastic circle-circle collision (velocity exchange along the
     line connecting centers using standard 2D elastic formula)
   - Wall bounces at [0,64]² boundaries
   - 10 substeps per step, same as 1D
   - Collision event recording: (step, obj_idx, pre/post pointer & obj
     velocities) — same structure as iter_035 for direct comparison
   - Pointer: radius=4, mass=10, color=white
   - Reset: randomized positions (non-overlapping, same segment logic
     adapted to 2D), randomized velocities per component ∈
     [-2.0,-0.5]∪[0.5,2.0]

B. Gate-1 (PASSIVE boundedness):
   - 5 seeds × 2000 steps
   - Pointer starts at (32, 32) with zero velocity, zero acceleration
   - Count per-object valid collisions (2D distance < r_ptr+r_obj+4.0
     AND |Δv_x|>0.5 OR |Δv_y|>0.5 — adapted from 1D thresholds)
   - Report mean per-object count, compare to 3.0 threshold
   - Also compute per-object collision counts without the |Δv| filter
     (raw proximity collisions) for diagnostic comparison with 1D

C. Gate-2 (RANDOM gaze heterogeneity):
   - 5 seeds × 2000 steps
   - Ghostly gaze pointer: no physical collisions during substep loop
     (same as iter_036 FoveatedGazeSandbox)
   - 2D random acceleration: each component ∈ [-10, 10] per step
   - Probe: probability p=0.01 per step, gaze_radius=8 (2D Euclidean
     distance), probe_budget=20
   - On probe: find nearest object with center within gaze_radius;
     if found, apply 2D elastic collision (along center-connecting
     line) between gaze and object; record event
   - Count per-object probe events, compute CV = std(counts)/mean(counts)
   - Also compute: mean per-object count, and compare coverage
     distribution to 1D baseline

D. Sanity checks (adapted from iter_035/036):
   - S1: PhysicsSandbox2D produces physically correct 2D elastic
     collisions (verify momentum and energy conservation on a
     controlled test)
   - S2: Pointer stays within bounds
   - S3: Objects stay within bounds
   - S4: Gate-1 PASSIVE pointer actually has zero velocity throughout
   - S5: Gate-2 RANDOM gaze actually fires ~20 probes over 2000 steps

E. Pre-registered parameters (fixed before execution):
   - Arena: 64×64 pixels
   - N=3 objects
   - Object radius: [3.0, 8.0]
   - Object mass: = radius
   - Object velocity per component: [-2.0,-0.5]∪[0.5,2.0]
   - Pointer radius: 4.0, mass: 10.0
   - Pointer start (Gate-1): (32, 32), velocity (0, 0)
   - Gaze radius (Gate-2): 8 pixels
   - Probe budget (Gate-2): 20
   - Probe probability (Gate-2): 0.01
   - Substeps: 10
   - Steps: 2000
   - Seeds: [7, 31, 53, 71, 83]
   - Gate-1 threshold: ≤ 3.0 per-object valid collisions
   - Gate-2 threshold: CV ≥ 0.50 per-object probe counts

Step 3 — Option (iii) Explicit Rejection:
Document that decoder-relaxation (option iii) is mis-targeted:
  - The binding constraint is environmental: four iterations demonstrate
    that perception is not load-bearing in the 1D testbed under any
    tested configuration
  - Adding a decoder does not address why behavior does not need
    perception — it addresses representation quality, which is a
    different constraint
  - Reconstruction+VICReg was already tested (iter_031): ΔR²_color=0.063
    with mean-pool readout — the decoder reconstructs pixels but the
    mean-pool spatial bottleneck prevents identity encoding regardless
  - The decoder-free constraint is not the current blocker; the
    environment is

Step 4 — Decision-Support Package (NOT a decision):
Produce structured analysis for human go/no-go on path (i):

A. If both gates pass — evidence FOR 2D viability:
   - What a full 2D commitment requires:
     1. 2D encoder: 1D-conv → 2D-conv (all 4 conv layers, spatial/dyn
        heads, soft-argmax over 2D spatial map) — estimated 1-2 iters
     2. 2D soft-argmax centroid: must output (B, d_max, 2) coordinates
        — requires new head design
     3. 2D PhysicsSandbox2D: production version with rendering to
        (3, 64, 64) RGB image — estimated 0.5-1 iter
     4. 2D CLTSMotorController: 2D pointer with 2D acceleration —
        estimated 0.5-1 iter
     5. Re-validate non-collapse and semantic encoding in 2D:
        repeat iter_027-030 work — estimated 2-3 iters
     6. Re-validate behavioral bracket in 2D: repeat iter_033-036 work
        — estimated 2-3 iters
     7. Total: ~7-10 additional iterations
     8. Compute cost: 2D conv ≈ 4× FLOPs of 1D conv at same resolution
   - What carries over unchanged:
     1. M1 batch-VICReg (objective-level, architecture-independent)
     2. iter_028 separate-backbone + mask_dyn_sim (0% collapse fix)
     3. Decoder-free constraint
     4. M3 frozen-dim regime, GDASR log-only
     5. Analytical-ceiling-gate + oracle-bracket methodology
     6. Pre-committed-rule discipline
     7. All metric designs (POMLRE, ΔR²_color, centroid MSE, etc.)
   - Risk: gates passing does NOT guarantee the full bracket
     discriminates; new failure modes may emerge in 2D

B. If either gate fails — evidence AGAINST 2D viability:
   - 2D does not resolve the 1D structural constraints at the tested
     parameterization
   - Remaining options: (ii) re-frame deliverable, or explore different
     2D parameterizations (larger arena, fewer objects, different gaze
     radius) — but these are additional investments with no guarantee

C. Path (ii) scope (re-frame deliverable):
   - Report representation findings: VICReg-only z_dyn gives 0% collapse,
     ΔR²_color ≈ 0.045 (best decoder-free) to 0.275 (SFA+VICReg sfa=5.0)
   - Report thalamic gating mechanism: surprise-detector + categorizer
     dual-control, per-channel EMA normalization
   - Report motor controller: CLTSMotorController with PD tracking,
     velocity matching, surprise-triggered push
   - Report the four-iteration null as a negative finding: the 1D
     testbed cannot validate that better perception produces better
     behavior, which is itself a clean result
   - Do NOT claim behavioral validation of the curiosity-driven
     perception-action thesis

D. Explicit statement: Path selection among (i)/(ii) is a human-scale
   strategic decision about project scope, cost, and goals. Iter_037
   provides measured evidence and cost analysis; it does not make the
   decision.

FILES CREATED:
- src/run_iter037_2d_gates.py (new — 2D gate experiment)
- archive/iter_037/results/ (output directory)

FILES PRESERVED (no modification):
- src/environment.py (1D environment unchanged)
- src/models_dual_stream.py (1D encoder unchanged)
- src/models_separate_dyn.py (separate backbone unchanged)
- src/motor.py (CLTSMotorController unchanged)
- All standing mandates (M1, M3, GDASR log-only, decoder-free)
- M2 remains untestable, not falsified

---

## Iteration 037 -> Planner [Strategic Guidance]

## Manager's Note (Strategic Critique of Proposed Plan, iter_037)

**1 — The plan is broadly well-scoped and faithful to the de-risking mandate, but the two gate parameterizations are arbitrarily ported from 1D without justifying that the 2D thresholds remain calibrated. This must be fixed before pre-registration is frozen.** Gate-1 reuses the "≤ 3.0 valid collisions / object" threshold from iter_035 verbatim, and Gate-2 reuses "CV ≥ 0.50" from iter_036 — but neither threshold was originally derived from first principles. They were operational cutoffs chosen for the 1D setting. The planner must add one short paragraph each justifying why those numbers remain the right cutoffs in a 64×64 / 2000-step / N=3 / r_gaze=8 / probe_budget=20 regime — or pre-register revised cutoffs derived from the new geometry. Otherwise an "obvious" pass can be a calibration artifact (e.g. Gate-2 trivially passes simply because probe_budget=20 with p=0.01 yields ~20 events spread over 3 objects, where Poisson noise alone can push CV ≥ 0.50). Also: report each gate's per-seed values and the across-seed standard deviation, not just the mean — a mean-passing gate that fails on 2/5 seeds is not a pass. Pre-register the per-seed decision rule (e.g. "≥4/5 seeds must individually meet threshold") in addition to the mean.

**2 — The construction-vs-empirical test is partially failed by the framing of Gate-1. "2D pointer collides less than 1D pointer" is geometrically near-tautological** — moving from a 1-axis-share regime to a 2-axis regime cannot fail to reduce collision frequency at fixed pointer/object size. So a Gate-1 pass would be a verification of the chosen geometry, not an empirical discovery. The planner should either (a) reframe Gate-1 honestly as a *quantitative calibration measurement* ("at this arena/object size, the reduction is sufficient to fall under 3.0") rather than a hypothesis test, or (b) add an additional empirical-content gate whose outcome is *not* predictable from the dimension change alone — e.g. measure whether **per-object collision counts under PASSIVE are themselves heterogeneous** (a 2D dual of Gate-2 applied to PASSIVE), since uniform-but-low collision rates would still leave the dual-failure-mode (b) condition unmet on the bound side. Gate-2 has more empirical content (a 2D random walk's coverage uniformity over 2000 steps is genuinely uncertain), but Gate-1 as currently written largely restates the construction.

**3 — Pre-registration mechanics and language hygiene.** (a) The Orchestrator will automatically write the pre-registration to `src/pre_registration.md` from the `hypothesis` and `falsification_criterion` fields and commit it before execution begins; sub-agents must read and strictly adhere to it. Therefore everything that needs to bind execution — including the calibration justification from point 1, the per-seed decision rule, and the additional empirical-content gate from point 2 — must be written into those YAML fields *now*, not added as commentary later. (b) Language: the plan already restrains itself well ("passing only unblocks a human go/no-go decision"); preserve that discipline in the deliverable. The decision-support write-up must use "is consistent with" / "does not refute" / "provides measured evidence for" — avoid "2D works", "2D validates", "2D solves the 1D problem" in any deliverable phrasing. (c) Step 1 (four-iteration null finding document) and Step 3 (decoder-free rejection rationale) are good and should be retained as written; they execute the human-hint's first and third asks cleanly. (d) Step 4D's explicit "iter_037 does not make the decision" statement is exactly right — keep it prominent and ensure the output handoff to the human surfaces it as the headline, not a footnote.

---

