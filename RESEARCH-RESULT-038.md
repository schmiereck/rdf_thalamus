# RDF Milestone Review — Iteration 038 — Null Result: Behavioral-Validation Strategy Declared Intractable Within Project Scope

## 1. Pre-Declared Hypothesis and Falsification Criterion

**Hypothesis (verbatim from the iter_038 pre-registration in the user hint):**
"A 2D NAVIGATION/SELECTION task — a MOVING pointer that must navigate to an
object's location to probe/excite it, where object dynamics information is
emitted ONLY when the moving pointer reaches and probes that specific object,
under a finite action budget, measured under a RANDOM navigation policy —
directly attacks the iter_037 Gate-1/Gate-1b opposition: a random navigator
reaches different objects unevenly (raising Gate-2 per-object event CV
toward ≥ 0.50) and causes collisions by arrival rather than by geometric
inevitability (keeping Gate-1 non-saturating while making the per-object rate
tunable so Gate-1b stabilizes)."

**Binding pre-committed exit rule:**
- PASS branch: all three gates clear → human go/no-go on full
  10-14-iteration 2D rebuild. Agent does NOT auto-start the rebuild.
- FAIL branch: any gate fails → behavioral-validation strategy is declared
  not tractable within project scope. Project pivots to path (ii) re-frame
  constructed as explicit pre-registered falsifiable claims with their own
  gates, not deliverable-by-narration.

**Pre-registered gates:**
- Gate-1: PASSIVE per-object collision count non-saturating.
- Gate-1b: per-object collision-count CV stable across ≥ 5 seeds
  (std of per-seed CV ≤ 0.25).
- Gate-2: per-object event CV ≥ 0.50 under random navigation.

## 2. Experimental Protocol

- Arena: 2D, 64 × 64.
- Objects: N = 3, uniform-random placement per seed.
- Pointer: physically navigating (NOT static), random-walk navigation policy.
- Probe mechanism: object dynamics information emitted ONLY when the moving
  pointer reaches and probes that specific object (finite action budget).
- Seeds: 5.
- Mode: rollouts only — no training, no learned model, no ORACLE bracket,
  no representation re-architecture.
- Pre-registered gates as listed above; exit rule pre-committed before run.

## 3. Observed Quantities

- **Gate-1:** PASSED. Per-object PASSIVE collision count remains
  non-saturating, consistent with the iter_037 2D-geometry finding that
  collision inevitability is removed.
- **Gate-2:** PASSED. Per-object event CV ≥ 0.50 under random navigation.
  The user-hint mechanism (uneven object reach by a random navigator) is
  confirmed.
- **Gate-1b:** FAILED. Per-seed CV values are bimodal at approximately
  [0.75, 1.41, 0.77, 0.71, 1.41], std = 0.320 vs. pre-registered
  threshold 0.25. The CV statistic is structurally meaningful (it
  measures uneven coverage) but is NOT structurally reproducible across
  seeds at the chosen parameterization.

**Diagnostic mechanism for the Gate-1b failure:** the bimodality reflects
two distinct regimes that uniform-random object placement + random walk
sample with nonzero probability — objects near the pointer's starting
region (heavy clustering, high CV) vs. distant objects (different pattern).
This is a property of the placement-trajectory interaction, not measurement
noise.

**Comparison to falsification threshold:** Gate-1b std CV = 0.320 exceeds
the pre-registered threshold of 0.25 by 28%. The FAIL branch of the
pre-committed exit rule was applied without modification.

## 4. Verdict

**Null result — Refuted (with respect to the project-internal claim that
some 2D environment-design parameterization within scope could make
perception behaviorally load-bearing under an ORACLE-vs-RANDOM bracket).**

The 2D navigation probe was specifically engineered (per the user hint) to
attack the iter_037 Gate-1/Gate-1b opposition by replacing the static
pointer with an active navigator. It succeeded at decoupling the two
failure modes — Gate-1 and Gate-2 both pass simultaneously, which neither
the static-pointer 2D design nor any 1D design achieved — but a new
failure mode emerged: the gate statistic itself is not reproducible
across seeds.

Combined with the four prior environment-design results (iter_033 metric
saturation, iter_034 v2 MALRE free-information leak, iter_035 1D
collision-inevitable shared-axis pointer, iter_036/037 2D static-pointer
foveated gaze), this is a five-design, two-dimensionality null chain in
which the analytical-ceiling-gate primitive identifies a different
structural obstruction in each design. The bottleneck migrates as designs
change. This is the signature of an obstruction in the underlying
design space, not a lack of design effort.

The pre-committed exit rule fires its FAIL branch: the behavioral-
validation strategy (making perception load-bearing through environmental
design under an ORACLE-vs-RANDOM bracket) is declared not tractable
within project scope. The project pivots to path (ii) re-frame, hardened
per the user hint into six pre-registered falsifiable claims.

## 5. Construction-vs-Empirical Note

Construction-side: the bounds of the per-seed CV values are not fixed by
the construction. They reflect actual sampled outcomes of the
placement-trajectory interaction under the random navigation policy. The
bimodal distribution across seeds is an empirical observation, not an
algebraic identity of the design.

Empirical-side: the genuine new information from this iteration is the
identification of a new ceiling-gate failure class — "the metric is
structurally meaningful but not structurally reproducible." This is
distinct from the prior four failure classes (saturation, coverage
uniformity, collision inevitability, Gate-1/Gate-1b opposition). It
expands the inventory of obstructions the analytical-ceiling-gate
primitive has now characterized.

The five-iteration unified null is also empirical: that five distinct
designs across two dimensionalities each fail at a different ceiling-gate
point is not derivable from any individual design's construction. It is
a property measured across the design space.

## 6. Limitations

- This result does NOT show that perception can never be made
  load-bearing — only that it has not been made load-bearing under the
  five designs tested within an ORACLE-vs-RANDOM bracket and within
  project resource scope.
- This result does NOT falsify M2. M2 remains untestable under any
  tested environmental regime, not falsified. Path (ii) does not require
  M2 to be tested.
- The Gate-1b non-reproducibility could in principle be stabilized by
  constraining initial conditions or by inflating the seed budget. The
  project chose not to pursue either because (a) both moves convert the
  cheap de-risking pass into a committed expensive arm, and (b) the
  five-design pattern already provides strong evidence of a structural
  obstruction. A different project with different resource constraints
  could choose differently.
- The decision to pivot to path (ii) rests on honoring the pre-committed
  exit rule. If the exit rule had been written differently
  (e.g. weighted-pass), the project decision could be different. The
  strength of this result is in the pre-commitment + clean firing
  pattern, not in the standalone Gate-1b number.
- Path (ii) re-frame is now committed but not yet validated — the six
  claims must actually pass their own pre-registered gates in iter_039+.
  The success of the re-frame as a deliverable is contingent on those
  future validations, not on the present iteration.