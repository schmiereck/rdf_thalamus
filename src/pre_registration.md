# RDF Scientific Pre-Registration

*   **Iteration:** 038
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
In a 2D arena (64×64, N=3 objects) with a PHYSICALLY NAVIGATING pointer (random
acceleration ∈ [-1.5, 1.5]², velocity capped at 3.5 px/step per component) that
must navigate to objects and actively probe them (probe probability p=0.015 per
step when within PROBE_RADIUS=10 of any object, budget=15 probes), the three
structural constraints that blocked the behavioral bracket across iter_033-037
are simultaneously relaxed:

Gate-1 (Non-saturation): Mean per-object probe-event count under RANDOM
navigation ≤ 3.0. Random navigation produces bounded, non-saturating
information access because 2D geometry allows off-axis passage and the probe
budget + probability cap total events.

Gate-1b (CV stability): Per-object probe-event count CV is structurally
meaningful and reproducible across seeds (mean per-seed CV ≥ 0.30 AND
std(per-seed CVs) ≤ 0.25). The navigation pointer creates enough events
(estimated ~7-10 per seed) for CV to reflect structural spatial clustering
rather than Poisson noise, unlike iter_037 where 0-1 events per seed made
CV undefined or noise-dominated.

Gate-2 (Heterogeneity): Mean per-seed per-object probe-event CV ≥ 0.50
(with ≥4/5 seeds individually ≥ 0.50). A 2D random walk with moderate speed
(~3 px/step) in a 64×64 arena has mixing time ~455 steps; over 2000 steps
(~4.4 mixing times), the walk retains partial spatial structure, creating
naturally uneven per-object visit rates. Objects in the center sector (where
the walk starts and to which wall-bounces frequently redirect) receive more
visits than objects in peripheral sectors.

## 2. Falsification Criterion
The hypothesis is falsified if ANY gate fails under the per-seed decision rule:

Gate-1 FAIL: Fewer than 4 of 5 seeds achieve mean per-object probe-event
count ≤ 3.0 under RANDOM navigation. This means the moving pointer creates
too many information events, leaving insufficient headroom for ORACLE targeting.

Gate-1b FAIL: Either (a) mean per-seed CV < 0.30, meaning the heterogeneity
is too weak to be structurally meaningful even after averaging across seeds; or
(b) std(per-seed CVs) > 0.25, meaning the heterogeneity is not reproducible —
it varies wildly from seed to seed, dominated by trajectory luck rather than
structural spatial effects.

Gate-2 FAIL: Fewer than 4 of 5 seeds achieve per-object probe-event CV ≥ 0.50.
This means random navigation does not create sufficient per-object heterogeneity
for ORACLE's targeted perception to exploit.

If any gate fails, the behavioral-validation strategy is declared not tractable
within project scope, and the project pivots to path (ii) re-frame.

## 3. Proposed Method
## Experiment: 2D Navigation Gate Probe

### Step 1 — Pre-Registration and Null Finding Preservation
Update src/pre_registration.md with iter_038 parameters and gate definitions.
Preserve the iter_037 null finding document (archive/iter_037/results/null_finding_1d.md)
as a standing reference — no modification.

### Step 2 — Implement Navigation Gate Experiment
Create src/run_iter038_nav_gates.py extending PhysicsSandbox2D from iter_037.

**Environment parameters (pre-registered, FROZEN):**
- Arena: 64×64 (same as iter_037)
- N=3 objects (same as iter_037: radius ∈ [3.0, 8.0], mass=radius,
  velocity per component ∈ [-2.0, -0.5] ∪ [0.5, 2.0])
- Pointer: radius=4, mass=10, starts at (32, 32) with velocity (0, 0)
- Substeps: 10, dt = 1/substeps
- Seeds: [7, 31, 53, 71, 83] (same as iter_037)

**Navigation (RANDOM policy):**
- Each step: sample 2D acceleration from Uniform[-1.5, 1.5]²
- Apply acceleration to pointer velocity via substep integration
- Velocity cap: 3.5 px/step per component (applied after each step)
- Physical elastic collisions between pointer and objects (same as iter_037)
- Wall bounces for pointer and objects (same as iter_037)

**Probe mechanism:**
- PROBE_RADIUS = 10.0 (pointer center within 10 pixels of object center)
- At each step, if any object is within PROBE_RADIUS, a probe opportunity exists
- Probe decision: probability p = 0.015 per step (when opportunity exists)
- Probe budget: 15 total probes per rollout
- On successful probe: record event for the NEAREST object within PROBE_RADIUS
- After budget exhausted: pointer still navigates but no more probe events
- Physical collisions are SEPARATE from probe events (collision dynamics still
  occur, but don't count as information events for the gates)

**Measurements (per seed):**
- Per-object probe-event counts (count_0, count_1, count_2)
- Per-object probe-event CV = std(counts) / mean(counts)
- Per-object valid collision counts (diagnostic, for comparison with iter_037)
- Per-object proximity-event counts (diagnostic: within PROBE_RADIUS, no probe)
- Total probes fired, budget remaining
- Pointer trajectory statistics (mean speed, arena coverage)

**Gate definitions:**
Gate-1 (Non-saturation):
- Metric: mean per-object probe-event count
- Threshold: ≤ 3.0
- Per-seed rule: ≥4/5 seeds must individually pass
- Justification: same as iter_037; 3.0 provides headroom for ORACLE to
  produce 5-10× more events via targeted navigation

Gate-1b (CV stability):
- Condition (a): mean of per-seed CVs across 5 seeds ≥ 0.30
- Condition (b): std of per-seed CVs across 5 seeds ≤ 0.25
- Both conditions must hold
- Justification: (a) ensures the heterogeneity is not trivially zero;
  (b) ensures it's reproducible (not driven by lucky/unlucky trajectories).
  In iter_037, per-seed CVs were [0.00, 1.41, 0.82, 0.00, 1.41] (std=0.62)
  — wildly unstable. With active navigation producing more events,
  CV should be more stable.

Gate-2 (Heterogeneity):
- Metric: per-seed per-object probe-event CV
- Threshold: ≥ 0.50
- Per-seed rule: ≥4/5 seeds must individually pass
- Justification: CV ≥ 0.50 indicates meaningful per-object heterogeneity
  that ORACLE's targeted navigation could exploit differently than RANDOM.
  Below 0.50, the random navigator already accesses all objects fairly
  equally, leaving no behavioral leverage for perception.

**Sanity checks:**
- S1: Physics conservation (2D elastic collision, same as iter_037)
- S2: Pointer stays within [0, 64]² throughout
- S3: Objects stay within [0, 64]² throughout
- S4: RANDOM navigation actually applies random acceleration (no targeting)
- S5: Velocity cap enforced (max 3.5 per component per step)
- S6: Probe mechanism fires correctly (within PROBE_RADIUS, with probability)
- S7: PASSIVE sanity check: rerun iter_037 Gate-1 on 1 seed to confirm
  static pointer still gives mean ≤ 3.0 (re-verification, not new measurement)

### Step 3 — Geometric Argument for Joint Gate Satisfaction

**Why this design escapes the Gate-1/Gate-1b opposition identified in iter_037:**

In iter_037 (static pointer at center), the opposition was:
- Gate-1 passes "too well" → collisions are very rare (0-1 per object)
- Gate-1b fails → too few events for CV to be structurally meaningful
- These are coupled: the same geometric fact (2D allows off-axis passage)
that keeps collisions low also makes them too rare for CV measurement.

The navigation task DECOUPLES these concerns through two mechanisms:

(1) **Active event generation:** The moving pointer traverses the arena,
encountering objects more frequently than the static pointer. The probe
mechanism (PROBE_RADIUS=10, p=0.015) produces ~7-10 probe events per seed
(vs 0-2 for the static pointer), providing enough data for CV to be
structurally meaningful. This addresses Gate-1b directly.

(2) **Budget-capped event rate:** The probe budget (15) and probability
(0.015) ensure the total event rate is bounded, even though the pointer
moves. The estimated mean per-object count (~2.3) is well below the Gate-1
threshold of 3.0. This preserves Gate-1.

The coupling is broken because event generation is now determined by
navigation (which the agent controls) rather than passive collision
probability (which is a fixed property of the geometry).

**Why spatial clustering supports Gate-2:**

A 2D random walk with speed ~3 px/step in a 64×64 arena has diffusion
coefficient D ≈ 4.5 and mixing time τ_mix ≈ L²/(2D) ≈ 455 steps. Over
2000 steps (~4.4 τ_mix), the walk retains partial spatial structure:
- The walk's position is autocorrelated (it tends to continue in the same
  direction for 50-100 steps before changing course)
- It spends extended periods near one object before diffusing to another
- Objects in the center sector (where the walk starts and to which wall
  bounces frequently redirect) receive more visits than peripheral objects

With segment-based object initialization (Object 1 in the center sector),
the pointer starts near Object 1 and visits it more frequently in the
first ~500 steps. This initial bias persists even after mixing because
the walk returns to the center more often (wall bounces redirect toward
center). The expected per-object visit probability distribution is
approximately [0.25, 0.40, 0.35], producing expected CV ≈ 0.50 from
structural asymmetry alone. Random walk clustering amplifies this above
the Poisson baseline (~0.39 for uniform allocation of 9 probes).

### Step 4 — Run Experiment and Record Results
Execute src/run_iter038_nav_gates.py, save all results to
archive/iter_038/results/.

### Step 5 — Apply Pre-Committed Exit Rule

**PASS branch (all three gates clear):**
Write up the result. Explicitly hand off to HUMAN go/no-go on full 2D rebuild.
The agent does NOT auto-start the rebuild. The human decides whether to commit
~10-14 iterations to the 2D migration. Document:
- The specific gate measurements that passed
- The estimated cost/scope of a full 2D rebuild (from iter_037 decision support)
- What carries over unchanged (M1, M3, iter_028 substrate, decoder-free)
- What must be rebuilt (2D encoder, 2D soft-argmax, production PhysicsSandbox2D,
  2D CLTSMotorController, re-validate non-collapse, re-validate bracket)
- Caveat: gates passing does NOT guarantee the full bracket discriminates

**FAIL branch (any gate fails):**
Declare the behavioral-validation strategy not tractable within project scope.
Iter_039 becomes path (ii) re-frame, with the following pre-registered
falsifiable claims (each with its own gate):

1. **M1 pooled-VICReg necessity:** Without batch-level VICReg, the
   separate-backbone architecture collapses in ≥4/5 seeds. Gate: train
   without VICReg; check has_collapsed or per-dim std < 5.0.

2. **iter_028 collapse mechanism:** sim_loss_dyn is the causal driver of
   z_dyn collapse on the shared backbone (removing it reduces collapse from
   ≥30% to ≤10%); the separate backbone eliminates collapse regardless.
   Gate: (a) shared backbone without sim_loss_dyn collapses ≤1/5 seeds;
   (b) separate backbone collapses 0/5 seeds.

3. **iter_031 mean-pool bottleneck:** Mean-pool spatial readout limits
   ΔR²_color < 0.10. No alternative readout achieves ΔR² ≥ 0.30 without
   collapse. Gate: test centroid-gated or max-pool; falsified if any
   achieves ΔR² ≥ 0.30 in ≥4/5 seeds.

4. **Structural-ceiling-gate primitive:** The analytical ceiling gate
   (measuring PASSIVE information access to predict bracket viability)
   is a valid research primitive. Gate: apply to a fresh task with known
   ground-truth discriminability; falsified if prediction is wrong.

5. **1D environmental null (documented):** The four-iteration null chain
   (iter_033-036) is a standing negative finding — not re-tested but
   preserved as a citable result.

6. **2D navigation null (from iter_038):** The specific gate failure from
   iter_038 is documented with the failure mode, parameterization, and
   measured values — a data point, not a narrative.

Each claim is falsifiable with its own gate. The re-frame is NOT a
deliverable-by-narration; it is a set of specific, testable, quantitative
claims about the representation, methodology, and environmental findings.

### Files Created
- src/run_iter038_nav_gates.py (new — 2D navigation gate experiment)
- src/pre_registration.md (updated with iter_038 parameters)
- archive/iter_038/results/ (output directory with gate results)

### Files Preserved (NO modification)
- src/environment.py (1D environment unchanged)
- src/models_dual_stream.py (1D encoder unchanged)
- src/models_separate_dyn.py (separate backbone unchanged)
- src/motor.py (CLTSMotorController unchanged)
- src/run_iter037_2d_gates.py (preserved for reference)
- All standing mandates (M1, M3, GDASR log-only, decoder-free, iter_028 substrate)
- M2 remains untestable, not falsified

## 4. Object Placement Protocol

- Object initial positions are sampled UNIFORM-RANDOM in the arena interior.
  Both x and y coordinates are drawn independently from
  Uniform([margin, ARENA_SIZE - margin]) where margin = radius_i + 0.5.
  This is **NOT** segment-based placement. Objects may cluster near each
  other or near the geometric center — this is expected and not controlled.
- The pointer spawns at the geometric center (32, 32) **independent of
  object positions**. It does not receive any knowledge of where objects are.
- The "expected CV ≈ 0.50" claim in the hypothesis is reframed: it is a
  prediction about **EMERGENT** random-walk clustering from autocorrelated
  trajectories + wall bounces, **NOT** from chosen object placement. If
  CV ≥ 0.50 cannot survive uniform-random object placement, the gate MUST
  fail. The heterogeneity must come from the dynamics, not from the setup.
- **Explicit statement:** No object placement bias is permitted. The CV must
  emerge from the dynamics of the random walk and the resulting spatial
  clustering, not from any constructional advantage.

## 5. Statistical Discipline

- The std(per-seed CVs) in Gate-1b is a sample-of-5 statistic interpreted at
  face value, not bootstrapped. The failure mode is honest, not statistical-magic.
  If the five seeds produce wildly different CVs, that is a real finding, not
  a sampling artifact.
- The seed list **[7, 31, 53, 71, 83] is FROZEN**. No post-hoc seed
  additions to chase a gate. Adding more seeds is a form of p-hacking.
- **Single declared parameterization** — no parameter sweep, no fallback
  parameterization. This design either passes or fails. If one parameter
  combination passes and another fails, that is not evidence in favor of
  the hypothesis; it is evidence that the result depends on tuning.
- **Mid-run tuning of any parameter** (p, PROBE_RADIUS, budget, velocity cap,
  arena size) is **FORBIDDEN**. The parameters declared here are the only
  parameters that will be tested.

## 6. Reporting Language

- Results must be phrased using the following approved formulations:
  - "is consistent with bracket-admission"
  - "does not refute the null"
  - "the tested parameterization fails Gate-X at X.XX vs threshold Y.YY"
- **NEVER** use the following language:
  - "demonstrates"
  - "proves"
  - "shows that 2D navigation works"
  - "validates"
- **On PASS:** hand off to HUMAN go/no-go on full 2D rebuild. The agent
  does **NOT** begin 2D rebuild work. The scope of a full rebuild (estimated
  ~10-14 iterations) must be acknowledged as a human decision, not an
  automated continuation.
- **On FAIL of any gate:** the six re-frame claims in the existing exit rule
  (Section 3, Step 5 FAIL branch) become the iter_039 scope. The failure is
  recorded as a data point, not as a narrative failure.

---
*Created automatically by the RDF Orchestrator prior to iteration execution.*
