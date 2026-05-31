# Research Manager Log - Iteration 038

## Iteration 038 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (iter_038, pre-planning):

1. **Directional scope lock — execute exactly one cheap 2D-navigation gate probe, nothing more.** The next iteration is a rollout-only structural measurement of a MOVING pointer in a 2D arena that must navigate to an object's location to probe/excite it, with object-dynamics information emitted only on successful arrival under a finite action budget, measured under a RANDOM navigation policy with ≥5 seeds. No training, no learned model, no representation work, no ORACLE bracket, no parameter sweep beyond a single pre-declared parameterization. This is explicitly the user-mandated path (a) in disciplined form, and is explicitly NOT path (b) (do not start the 10–14-iteration rebuild on the strength of any partial result). M1, M3, decoder-free, and the iter_028 substrate are preserved verbatim and must not be touched.

2. **Scientific discipline — pre-register the binding three-gate exit rule and the navigation-task specification in writing BEFORE any rollout, and treat the discovered Gate-1/Gate-1b opposition as a falsifier, not a tuning knob.** All three gates must clear at the single declared parameterization for PASS: Gate-1 (PASSIVE per-object collisions non-saturating, ≤3.0), Gate-1b (per-object collision-count CV stable across ≥5 seeds, with the stability criterion numerically defined in the plan — e.g. CV's own across-seed dispersion below a stated bound, not eyeballed), and Gate-2 (RANDOM per-object event CV ≥0.50). Per-seed decision rules and the exact arena/N/gaze_radius/budget/rollout-length must be posted before measurement, justified by a brief geometric argument for why this parameterization can in principle satisfy all three jointly (since iter_037 just showed they can be in structural opposition — if the planner cannot articulate why this design escapes that opposition, the iteration is mis-specified). Result language must stay restrained: "consistent with bracket-admission" or "does not refute," never "shows" or "proves."

3. **Pre-commit both branches of the exit rule, and forbid the agent from auto-executing either escalation.** On PASS: write up the result and explicitly hand off to a HUMAN go/no-go on the full 2D rebuild — the agent does not start the rebuild. On FAIL on any gate: the behavioral-validation strategy is declared not tractable within project scope, and iter_039 becomes path (ii) re-frame, which must be scoped now (not later) as a list of pre-registered falsifiable claims each with its own gate (representation: M1 pooled-VICReg necessity, the iter_028 separate-backbone+mask_dyn_sim collapse mechanism, the iter_031 mean-pool-readout bottleneck localization; methodology: structural-ceiling-gate primitive demonstrated on a fresh task; environment: the 1D-forecloses-load-bearing-perception null plus the 2D static-pointer opposition finding) — not a deliverable-by-narration. The planner's draft must contain both branch specifications so neither outcome leaves the project in an undefined state.

---

## Iteration 038 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
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

**Proposed Falsification Criterion:**
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

**Proposed Method:**
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

---

## Iteration 038 -> Planner [Strategic Guidance]

# Manager's Note (iter_038, plan critique)

The plan is well-structured and respects the dual-branch pre-commitment discipline. Three concerns require resolution before the Orchestrator writes `src/pre_registration.md` and the experiment is run.

**1. Construction-vs-empirical risk on Gate-2 — object placement must NOT be engineered for the predicted CV.** The geometric argument in Step 3 says "Object 1 in the center sector (where the walk starts and to which wall-bounces frequently redirect) receive more visits than peripheral objects," and projects "expected per-object visit probability distribution is approximately [0.25, 0.40, 0.35], producing expected CV ≈ 0.50 from structural asymmetry alone." If you achieve Gate-2 by placing one object near the pointer start, the CV is *constructional* — it merely restates "you put an object near the spawn point" — and that bias will not survive ORACLE comparison (an ORACLE policy would exploit the same bias, collapsing the differential). The pre-registration MUST specify (a) object initial positions are sampled uniform-random in the arena interior, matching iter_037's protocol; (b) the pointer spawn is at the geometric center independent of object positions; and (c) the "expected CV ≈ 0.50" claim is reframed as a prediction about emergent random-walk clustering from autocorrelated trajectories + wall bounces, NOT from chosen placement. If CV ≥ 0.50 cannot survive random object placement, the gate must fail and the project pivots to (ii); do not patch by introducing placement bias.

**2. Gate-1 headroom is suspiciously thin and Gate-1b std criterion is fragile.** Estimated mean per-object probe-event count is ~2.3 (from ~7 events / 3 objects) against a threshold of 3.0 — a ~23% margin where any miscalibration of `p=0.015` or PROBE_RADIUS pushes through. Worse, your earlier worked example shows ~7–10 events / 3 objects can yield individual seeds with 4+ events on one object. Lock the parameters now and explicitly forbid mid-run tuning. Separately, Gate-1b's `std(per-seed CVs) ≤ 0.25` over only N=5 seeds is itself a noisy estimator (the std of 5 numbers has ~50% sampling-error CI); state in the pre-registration that this is a sample-of-5 statistic interpreted at face value and not bootstrapped, so the failure mode is honest, not statistical-magic.

**3. Pre-registration write order, language discipline, and the binding exit rule.** Per the Orchestrator's protocol, the `hypothesis` and `falsification_criterion` fields above will be written automatically to `src/pre_registration.md` and committed BEFORE the rollout runs — the sub-agents MUST read that file and adhere to it without re-tuning. Add three items to the file: (i) the fixed seed list `[7, 31, 53, 71, 83]` and the explicit ban on adding seeds post-hoc to chase a gate; (ii) the statement "no parameter sweep, no fallback parameterization — a single declared design either passes or fails"; (iii) restrained reporting language — results must be phrased as "is consistent with bracket-admission" / "does not refute the null" / "the tested parameterization fails Gate-X at X.XX vs threshold Y.YY," never "demonstrates," "proves," or "shows that 2D navigation works." On PASS: hand off to human go/no-go, do NOT begin 2D rebuild work. On FAIL of any gate: the six re-frame claims in Step 5 are the iter_039 scope — that scope must be referenced (not re-derived) in iter_039 so the FAIL branch is institutional, not improvised.

---

