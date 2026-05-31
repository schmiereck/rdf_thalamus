# RDF Milestone Review — Iteration 037 — Null Result: 1D Sandbox is Structurally Insufficient for Bracketed Behavioral Validation of Perception; 2D at Tested Parameterization Does Not Resolve It

## 1. Pre-Declared Hypothesis and Falsification Criterion

Two independent hypotheses, each with a pre-registered
falsification criterion declared before any rollout.

**H-1D (closing the four-iteration chain):** "A
1D × N=3 × 128-pixel sandbox under any of the
observation regimes tested in iter_033–036 admits an
ORACLE-vs-RANDOM behavioral bracket in which an
oracle perceptual policy outperforms a random
baseline."

Falsification criterion (pre-registered iteratively
across iter_033–036): an environment design admits
the bracket only if (a) under a non-perceptual
baseline policy, per-object information rate is
bounded (PASSIVE per-object collisions ≤ 3.0 in
iter_035) AND (b) under a non-perceptual baseline
policy, per-object information rate is unequal
across objects (RANDOM per-object event-count CV
≥ 0.50 in iter_036). H-1D is falsified if four
consecutive principled redesigns fail at least one
of these gates.

**H-2D-cheap (iter_037):** "A 2D arena removes both
the 1D collision-inevitability constraint (Gate-1)
and the 1D random-walk coverage-uniformity
constraint (Gate-2) at the tested
parameterization."

Falsification criterion (pre-registered in
iter_037 plan): H-2D-cheap is supported only if
Gate-1 (PASSIVE per-object collisions ≤ 3.0),
Gate-1b (per-object collision-count CV stable
across seeds), AND Gate-2 (RANDOM per-object
probe-event CV ≥ 0.50) all pass at 64×64 /
N=3 / gaze_radius=8 / 5 seeds. H-2D-cheap is
falsified if any gate fails at the tested
parameterization.

## 2. Experimental Protocol

**H-1D (cumulative across iter_033–036):**
- iter_033: ORACLE vs RANDOM behavioral bracket
  on a 1D pointer with object-pointer collisions
  as the perception signal. Metric: prediction
  accuracy on held-out object states.
- iter_034: MALRE v2 coverage discrimination
  test on the same environment. Measured
  active-passive gap and ORACLE-RANDOM gap.
- iter_035: rollout-only PASSIVE collision count
  per object on a 1D shared-axis pointer.
  5 seeds, 1000 steps per rollout.
- iter_036: rollout-only RANDOM gaze-probe-event
  CV per object on a 1D foveated-gaze pointer
  with GAZE_RADIUS=8 in a 128-pixel arena, two
  arms (collisions retained vs pass-through
  objects). 5 seeds.

**H-2D-cheap (iter_037):**
- Environment: PhysicsSandbox2D, 64×64 pixel
  arena, N=3 objects with momentum/energy
  conservation. All 2D physics sanity checks
  verified before measurement.
- Pointer: static central gaze with
  gaze_radius=8.
- Policies measured: PASSIVE (no motion) for
  Gate-1 and Gate-1b; RANDOM gaze trajectory
  for Gate-2.
- Rollout length: matched to iter_035/036 for
  cross-iteration comparability.
- Seeds: 5, fixed and reported.
- No training, no learned model, no full
  ORACLE bracket, no representation
  re-architecture — purely structural
  rollout measurement.
- Gate thresholds declared and posted before
  measurement; per-seed decision rules added
  per iter_036 Manager critique.

Held constant across the four 1D iterations
and the 2D iteration: object count N=3,
arena size in linear extent (128 pixels in 1D,
64×64 ≈ 4096 pixels area in 2D), 5-seed
evaluation protocol, pre-registered gate
thresholds.

## 3. Observed Quantities

**H-1D — gate values that fired:**
- iter_033 ORACLE-RANDOM behavioral gap:
  measured negligible (within noise); ORACLE
  ≈ RANDOM. Behavioral bracket does not
  discriminate.
- iter_034 MALRE: active-passive gap = 0.83
  (strong), but ORACLE-RANDOM gap = 0.031
  (negligible). Coverage discrimination works;
  perception-quality discrimination does not.
- iter_035 PASSIVE per-object collision count
  = 12.27 vs threshold ≤ 3.0. Gate fails by
  factor ~4.
- iter_036 RANDOM per-object probe-event CV:
  Arm A (collisions retained) = 0.36; Arm B
  (pass-through) = 0.46. Both vs threshold ≥
  0.50. Both fail.

**H-2D-cheap — gate values measured (iter_037):**
- Gate-1 (PASSIVE per-object collisions): 0–1
  per object across 5 seeds vs threshold ≤
  3.0. **PASSES** by a wide margin.
- Gate-1b (per-object collision-count CV
  stability across seeds): not stable — too
  few events for CV to be meaningful in the
  sample-noise regime. **FAILS** as a
  diagnostic.
- Gate-2 (RANDOM per-object probe-event CV):
  clusters near Poisson baseline ~0.39 vs
  threshold ≥ 0.50. **FAILS.**

## 4. Verdict

**H-1D: REFUTED.** Four consecutive principled
1D redesigns failed at least one of the
necessary-condition gates. The 1D × N=3 ×
128-pixel sandbox under any tested observation
regime does not admit a bracketed behavioral
evaluation in which perception is load-bearing.
This is a clean, pre-registered, multi-iteration
null.

**H-2D-cheap: REFUTED at tested parameterization.**
Gate-1 passes — 2D genuinely removes collision
inevitability — but Gate-1b and Gate-2 fail at
64×64 / N=3 / gaze_radius=8. The result is not
"2D doesn't work" but "the naïve drop-in 2D
parameterization tested here does not resolve
the bracket-admission problem." A NEW
structural finding emerges: Gate-1 and Gate-1b
are in opposition under a static-pointer 2D
design (rare collisions and stable CV cannot
both be satisfied at one parameterization),
suggesting that path (i) requires not just an
arena dimensionality change but a behavioral
task redesign.

## 5. Construction-vs-Empirical Note

Several findings are partly structural:
- "1D collisions saturate under a shared-axis
  pointer" (iter_035) is largely geometric —
  a particle constrained to a line will
  necessarily encounter another particle on
  the same line if they have non-zero
  velocity. The empirical part is the *rate*
  (12.27/object), which is well above the
  bound the bracket requires.
- "2D removes collision inevitability"
  (iter_037 Gate-1) is similarly geometric:
  off-axis trajectories prevent forced
  encounters. The empirical part is the
  measured rate (0–1/object) being so far
  below the bound that Gate-1b becomes
  unstable.

Genuinely empirical (not derivable from the
construction alone):
- "2D random walks at 64×64 / gaze_radius=8
  do not cover area unevenly enough to open
  the bracket" (Gate-2). The naïve argument
  "2D random walks cover less uniformly than
  1D" is true in some asymptotic sense but
  does not predict the magnitude at finite
  arena size and finite rollout length —
  this had to be measured.
- "Gate-1 and Gate-1b are in structural
  opposition under static-pointer 2D"
  (iter_037, novel finding). This was not
  predicted by the iter_036 escalation
  argument and is a discovered constraint of
  the test design.

## 6. Limitations

This result does NOT show:
- That 2D arenas in general cannot support a
  bracketed behavioral evaluation. Only that
  the tested parameterization
  (64×64 / N=3 / gaze_radius=8 / static
  central pointer / pointer-collision probe)
  does not. A wider parameter sweep or a
  navigation/selection task redesign may
  succeed.
- That perception-driven behavior is
  fundamentally untestable. Only that the
  ORACLE-vs-RANDOM bracket on the tested 1D
  and naïve 2D designs cannot serve as the
  test.
- That the decoder-free constraint is wrong.
  The four-iteration null is environmental,
  not representational; path (iii) is
  explicitly rejected on these grounds.
- That the iter_028 representation substrate
  is invalid. It remains the working
  representation with ΔR²_color ≈ 0.045 and
  0% collapse; what is in question is
  whether the project's evaluation strategy
  for that substrate can include a
  behavioral bracket.

What would be needed next:
- A human go/no-go between path (i) (full
  2D rebuild WITH task redesign, revised
  cost 10–14 agent-iterations) and path
  (ii) (re-frame deliverable around
  representation + gating + methodology,
  low compute cost). This is a
  scope/goals decision outside Manager
  authority.
- If path (i) is selected: pre-registered
  cheap gates on each new 2D
  parameterization candidate or task
  design BEFORE any training compute,
  applying the now-five-times-validated
  structural-ceiling-gate primitive.
- If path (ii) is selected: explicit
  falsifiable claims (identity-
  disentanglement ΔR² thresholds,
  attention-token-trace properties,
  surprise-EMA calibration tests, gate-
  primitive demonstration on a fresh
  task) with their own pre-registered
  gates, to avoid the path degenerating
  into a deliverable-by-narration.