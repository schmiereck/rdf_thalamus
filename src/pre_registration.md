# RDF Scientific Pre-Registration

*   **Iteration:** 037
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
In a minimal 2D physics arena (64×64 pixels, N=3 circular objects with radius
∈ [3,8], velocity components ∈ [-2.0,-0.5]∪[0.5,2.0], mass ∝ radius, substeps=10),
the two structural constraints that defeated the 1D testbed are removed:

Gate-1 (Boundedness): A PASSIVE physical pointer (radius=4, mass=10, starting
at arena center (32,32) with zero velocity) accumulates ≤ 3.0 valid elastic
collisions per object over 2000 steps, because objects can pass the pointer
off-axis in the y-dimension (unlike 1D where all entities share the axis).
A "valid collision" is defined identically to iter_035: proximity
(2D distance < r_ptr + r_obj + 4.0) AND |Δv| > 0.5 per component.

Gate-1b (PASSIVE collision heterogeneity): Under the same PASSIVE conditions,
the coefficient of variation (CV = std/mean) of per-object collision counts
is ≥ 0.30, indicating that different objects naturally receive different
collision rates based on their trajectories relative to the static pointer.

Gate-2 (RANDOM gaze heterogeneity): Under a RANDOM ghostly gaze policy (2D
acceleration ∈ [-10,10]², probe probability p=0.01, gaze_radius=8 pixels,
probe budget=20) the coefficient of variation (std/mean) of per-object
probe-event counts is ≥ 0.50, because a finite 2D random walk with bounded
step size cannot cover a 2D arena as uniformly as a 1D random walk covers
a line segment.

All three gates are necessary preconditions for a 2D ORACLE-vs-RANDOM bracket
to have headroom. No gate is sufficient to establish that the full 2D bracket
would discriminate — passing only unblocks a human go/no-go decision.

### Gate Threshold Calibration Justification

**Gate-1 (≤3.0): Why this cutoff remains appropriate in 2D.** In 1D, PASSIVE
accumulates 12.27 valid collisions per object because all entities share the
axis — the collision cross-section equals the full arena width, making collisions
inevitable. In 2D (64×64, N=3, pointer radius=4), the effective collision
probability per object trajectory is ~O(r_ptr/W)² ≈ (8/64)² ≈ 1.6% per axis
crossing, meaning objects can bypass the pointer off-axis in the y-dimension.
The 3.0 threshold represents a collision rate where ORACLE targeting could
produce ~10–20× more collisions than PASSIVE, providing substantial headroom
for behavioral discrimination. This is an operational threshold: it is sufficient
(not necessary) for the PASSIVE ceiling to be low enough that targeted perception
has meaningful headroom. The value is not chosen because it is the maximal
possible value; it is chosen because at 12.27 (the 1D result), there was zero
headroom — any threshold well below that in 2D would suffice, and 3.0 provides
a conservative margin.

**Gate-1b (CV ≥ 0.30): Why heterogeneity matters.** In 1D, CV_passive ≈ 0
because all objects share the axis and therefore experience identical collision
rates regardless of their individual trajectories. In 2D, objects at different
positions/velocities have geometrically different encounter probabilities with
the static pointer at center. A CV ≥ 0.30 indicates moderate heterogeneity in
per-object collision rates — enough variation for ORACLE's targeted collisions
to create meaningful differentiation beyond what PASSIVE produces passively.
The 0.30 threshold is a moderate bar: below 0.30, objects are too similar in
passive collision rates, and targeted action cannot exploit geometric differences
because they are too small. Gate-1b addresses the critique that Gate-1 alone
is near-tautological — Gate-1 measures whether the PASSIVE ceiling is low
enough, while Gate-1b measures whether there is heterogeneity that ORACLE can
exploit. Both are needed.

**Gate-2 (CV ≥ 0.50): Why random coverage alone does not trivially pass.**
With probe_budget=20 and p=0.01 over 2000 steps, the expected number of probes
is ~20, distributed across 3 objects. Under pure Poisson allocation (mean ≈
6.67/object), CV ≈ √(6.67)/6.67 ≈ 0.39 — below the 0.50 threshold. This means
random Poisson noise alone would NOT trivially satisfy the gate. CV ≥ 0.50
requires genuine spatial clustering in the gaze trajectory — some objects visited
more than others due to 2D random walk coverage geometry. Whether this occurs
is an empirical question, not a mathematical certainty of the budget.

### Per-Seed Decision Rule
For ALL gates (1, 1b, 2), the binding decision criterion is:
**≥4 out of 5 seeds must individually meet the threshold.** The mean across
seeds is also reported, but the per-seed rule takes precedence. This ensures
robustness to initial conditions — a passing mean driven by a single lucky seed
does not constitute evidence. All 5 seeds ([7, 31, 53, 71, 83]) are evaluated
independently; at most 1 seed may fail per gate.

## 2. Falsification Criterion
The hypothesis is falsified if ANY gate fails under the per-seed decision rule:

Gate-1 FAIL: Fewer than 4 of 5 seeds achieve mean per-object valid collision
count ≤ 3.0 under PASSIVE. This means 2D does not sufficiently reduce the
free-information ceiling — the pointer still gets too many collisions without
targeted perception, and path (i) is blocked at the cheap-gate level.

Gate-1b FAIL: Fewer than 4 of 5 seeds achieve per-object collision CV ≥ 0.30
under PASSIVE. This means there is insufficient heterogeneity in passive
collision rates for targeted action to exploit — objects are too similar
geometrically, and ORACLE cannot differentiate its collision pattern from
PASSIVE in a meaningful way.

Gate-2 FAIL: Fewer than 4 of 5 seeds achieve per-object probe-event CV ≥ 0.50
under RANDOM gaze. This means 2D random gaze coverage is still too even for
ORACLE targeting to have headroom, and path (i) is blocked.

If ALL three gates pass, this is NOT validation that 2D "works" — it is only
measured evidence that the three 1D structural constraints (collision
inevitability, collision homogeneity, coverage uniformity) are relaxed in 2D
at the tested parameterization (64×64, N=3, gaze_radius=8). The full bracket
result remains unknown.

## 3. Proposed Method
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
   - Per-seed decision: ≥4/5 seeds must individually have count ≤ 3.0

C. Gate-1b (PASSIVE collision heterogeneity):
   - Uses the same collision data from Gate-1
   - Compute CV = std(per-object collision counts) / mean(per-object collision counts)
   - Threshold: CV ≥ 0.30
   - Per-seed decision: ≥4/5 seeds must individually have CV ≥ 0.30

D. Gate-2 (RANDOM gaze heterogeneity):
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
   - Threshold: CV ≥ 0.50
   - Per-seed decision: ≥4/5 seeds must individually have CV ≥ 0.50

E. Sanity checks (adapted from iter_035/036):
   - S1: PhysicsSandbox2D produces physically correct 2D elastic
     collisions (verify momentum and energy conservation on a
     controlled test)
   - S2: Pointer stays within bounds
   - S3: Objects stay within bounds
   - S4: Gate-1 PASSIVE pointer actually has zero velocity throughout
   - S5: Gate-2 RANDOM gaze actually fires ~20 probes over 2000 steps

F. Pre-registered parameters (fixed before execution):
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
   - Gate-1b threshold: CV ≥ 0.30 per-object collision counts
   - Gate-2 threshold: CV ≥ 0.50 per-object probe counts
   - Decision rule: ≥4/5 seeds must individually pass each gate

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

A. If all three gates pass — evidence FOR 2D viability:
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

B. If any gate fails — evidence AGAINST 2D viability:
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
- archive/iter_037/results/null_finding_1d.md (four-iteration null finding)

FILES PRESERVED (no modification):
- src/environment.py (1D environment unchanged)
- src/models_dual_stream.py (1D encoder unchanged)
- src/models_separate_dyn.py (separate backbone unchanged)
- src/motor.py (CLTSMotorController unchanged)
- All standing mandates (M1, M3, GDASR log-only, decoder-free)
- M2 remains untestable, not falsified

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
