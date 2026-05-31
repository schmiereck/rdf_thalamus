## Task: Update pre-registration + create iter_038 navigation gate experiment script

### Context
This is iter_038 of the Thalamus project. We are running a SINGLE bounded 2D navigation gate probe to test whether a PHYSICALLY NAVIGATING pointer in a 2D arena resolves the structural constraints that blocked behavioral validation across iter_033-037. No training, no learned model, no representation work.

The existing pre-registration is at `src/pre_registration.md` (already contains the hypothesis, falsification criteria, and method). The existing iter_037 code is at `src/run_iter037_2d_gates.py` (contains PhysicsSandbox2D class).

### Step 1: Update src/pre_registration.md

Add the following THREE mandatory additions from the Research Manager (append to the existing file, do NOT remove any existing content):

**Addition 1 — Object placement protocol:**
Add a section "## 4. Object Placement Protocol" stating:
- Object initial positions are sampled UNIFORM-RANDOM in the arena interior (both x and y coordinates uniform in [margin, ARENA_SIZE - margin] where margin = radius_i + 0.5). This is NOT segment-based placement.
- The pointer spawns at the geometric center (32, 32) independent of object positions.
- The "expected CV ≈ 0.50" claim in the hypothesis is reframed: it is a prediction about EMERGENT random-walk clustering from autocorrelated trajectories + wall bounces, NOT from chosen object placement. If CV ≥ 0.50 cannot survive uniform-random object placement, the gate MUST fail.
- Explicit statement: "No object placement bias is permitted. The CV must emerge from the dynamics of the random walk and the resulting spatial clustering, not from any constructional advantage."

**Addition 2 — Statistical discipline:**
Add a section "## 5. Statistical Discipline" stating:
- The std(per-seed CVs) in Gate-1b is a sample-of-5 statistic interpreted at face value, not bootstrapped. The failure mode is honest, not statistical-magic.
- The seed list [7, 31, 53, 71, 83] is FROZEN. No post-hoc seed additions to chase a gate.
- Single declared parameterization — no parameter sweep, no fallback parameterization. This design either passes or fails.
- Mid-run tuning of any parameter (p, PROBE_RADIUS, budget, velocity cap, arena size) is FORBIDDEN.

**Addition 3 — Reporting language:**
Add a section "## 6. Reporting Language" stating:
- Results must be phrased as: "is consistent with bracket-admission" / "does not refute the null" / "the tested parameterization fails Gate-X at X.XX vs threshold Y.YY"
- NEVER use: "demonstrates", "proves", "shows that 2D navigation works", "validates"
- On PASS: hand off to HUMAN go/no-go on full 2D rebuild. The agent does NOT begin 2D rebuild work.
- On FAIL of any gate: the six re-frame claims in the existing exit rule (Section 3, Step 5 FAIL branch) become the iter_039 scope.

### Step 2: Create src/run_iter038_nav_gates.py

Build on the PhysicsSandbox2D from `src/run_iter037_2d_gates.py`. The key differences from iter_037:

**CRITICAL CHANGE 1 — Object placement:** UNIFORM RANDOM in arena interior, NOT segment-based. Both x and y coordinates uniform in [margin, ARENA_SIZE - margin] where margin = radius_i + 0.5. This means objects can cluster near each other or near the center — the CV must emerge from the random walk dynamics, not from placement bias.

**CRITICAL CHANGE 2 — Navigating pointer:** The pointer RECEIVES random acceleration each step: acc sampled from Uniform[-1.5, 1.5]². Velocity cap: 3.5 px/step per component (applied after each step's substep integration). The pointer physically moves around the arena.

**CRITICAL CHANGE 3 — Probe mechanism replaces collision-based information events:** The information events for the three gates are PROBE EVENTS, not collision events. The probe mechanism:
- PROBE_RADIUS = 10.0 (pointer center within 10 pixels of object center)
- At each step, if any object is within PROBE_RADIUS, a probe opportunity exists
- Probe decision: probability p = 0.015 per step (when opportunity exists)  
- Probe budget: 15 total probes per rollout (once exhausted, no more probes)
- On successful probe: record event for the NEAREST object within PROBE_RADIUS
- Physical collisions STILL OCCUR (elastic collision physics unchanged) but are DIAGNOSTIC only — they do NOT count as information events for the gates

**Frozen parameters (matching pre-registration):**
- ARENA_SIZE = 64.0
- N_OBJ = 3
- N_STEPS = 2000
- SUBSTEPS = 10
- SEEDS = [7, 31, 53, 71, 83]
- PTR_RADIUS = 4.0, PTR_MASS = 10.0, PTR_START = (32, 32), PTR_VEL_INIT = (0, 0)
- OBJ_RADIUS_RANGE = [3.0, 8.0], OBJ_MASS = radius
- OBJ_VEL ranges: per component in [-2.0, -0.5] ∪ [0.5, 2.0]
- NAV_ACC_RANGE = [-1.5, 1.5] (per component)
- VEL_CAP = 3.5 (per component)
- PROBE_RADIUS = 10.0
- PROBE_P = 0.015
- PROBE_BUDGET = 15

**Gate definitions (EXACTLY as pre-registered):**
- Gate-1: per-seed mean per-object probe-event count ≤ 3.0; ≥4/5 seeds pass
- Gate-1b: mean of per-seed CVs ≥ 0.30 AND std of per-seed CVs ≤ 0.25
- Gate-2: per-seed per-object probe-event CV ≥ 0.50; ≥4/5 seeds pass

**Structure of the experiment:**
1. Run sanity checks (S1: physics conservation, S2/S3: bounds, S4: random acceleration is truly random, S5: velocity cap, S6: probe mechanism, S7: re-verify iter_037 Gate-1 on 1 seed with static pointer)
2. For each seed, run the FULL NAVIGATION experiment (navigating pointer + probe mechanism), record per-object probe-event counts
3. For each seed, compute per-seed metrics (mean, CV)
4. Apply all three gates
5. Apply the exit rule (PASS → hand off to human; FAIL → pivot to re-frame)

**Output files** (write to `archive/iter_038/results/`):
- `probe_events.csv`: per-seed per-object probe-event counts + per-seed mean + CV
- `diagnostic_collisions.csv`: per-seed per-object collision counts (diagnostic, for comparison with iter_037)
- `sanity_checks.txt`: all sanity check results
- `gate_summary.txt`: full gate results with per-seed breakdowns
- `trajectory_stats.csv`: per-seed pointer trajectory statistics (mean speed, arena coverage measured as fraction of 64×64 grid cells visited)

**Important implementation details:**
- The random acceleration for the pointer uses a SEPARATE RNG from the environment RNG (use seed * 1000 + 7 for the navigation RNG) so that the environment (object initialization, object dynamics) and the navigation policy are independently reproducible
- Probe events are counted at STEP granularity (max one probe per step even if multiple objects are within PROBE_RADIUS — recorded for the NEAREST object only)
- The velocity cap is applied AFTER the full step (all substeps), capping each component to [-3.5, 3.5]
- After the probe budget is exhausted, the pointer still navigates (random acceleration continues) but no more probe events are counted
- Physical elastic collisions between pointer and objects still occur (same collision resolution as iter_037)

DO NOT run the experiment yet. Only create the script and update the pre-registration. The experiment will be run in the next sub-goal.

Write the complete script as src/run_iter038_nav_gates.py and update src/pre_registration.md.