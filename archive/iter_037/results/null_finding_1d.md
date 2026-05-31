# Four-Iteration Null Finding: The 1D × N=3 × 128px Sandbox

*   **Iteration:** 037
*   **Covers:** iter_033, iter_034, iter_035, iter_036
*   **File:** archive/iter_037/results/null_finding_1d.md

---

## Statement of Finding

Across four mechanism-distinct redesigns, the 1D × N=3 × 128px sandbox
cannot make perception behaviorally load-bearing under an ORACLE-vs-RANDOM
bracket. This is a negative result about this specific environment
parameterization; it does **not** claim that no 1D environment could
ever work, and it makes no claim about M2 (the thalamic gating module).

The following sections document each iteration's mechanism and failure
mode, forming a complete null chain: each redesign directly addressed
the specific failure of its predecessor, and the chain terminates at
the most radical redesign possible within the 1D constraint.

---

## iter_033 — Behavioral-Pivot Metric Saturation

**Mechanism tested:** The original ORACLE-vs-RANDOM behavioral bracket
in the 1D arena, using the POMLRE metric to measure whether better
perception (ORACLE's direct object targeting) produces measurably
different behavior than RANDOM's indiscriminate probing.

**Failure mode:** ORACLE and RANDOM are indistinguishable (gap = 0.0001).
The metric saturates at ceiling for both arms, meaning the environment
does not create any situation where the quality of perception
determines behavioral outcome. There is no behavioral leverage for
perception to act on — perception is already "good enough" even when
it is random. The 1D arena with N=3 objects provides such a constrained
state space that any reasonable action sequence produces similar
behavioral outcomes regardless of what the agent perceives.

**What this ruled out:** The basic bracket design. If the environment
itself does not make perception load-bearing, no metric refinement
within the same environment can fix this.

---

## iter_034 — Free Autonomous Information

**Mechanism tested:** The MALRE (Motor-Analytic Lower-Relative-Entropy)
metric to measure how much information the agent can extract from the
environment *without* targeted perception — i.e., the "free" information
available from the physics dynamics alone. This addressed the iter_033
critique by asking: maybe the environment is too rich in free information,
making targeted perception unnecessary.

**Failure mode:** MALRE showed a substantial active-passive gap of 0.83
(the active policy does extract more information than passive observation),
but the ORACLE-vs-RANDOM gap was only 0.031 and significant in merely
3 of 8 seeds. The environment does contain extractable information, but
the gap between *best possible* targeting (ORACLE) and *random* targeting
is negligible for most seeds. The issue is not a lack of information in
the environment; it is that the information is equally reachable without
targeted perception. The 1D geometry makes everything equally accessible.

**What this ruled out:** The hypothesis that "free information saturation"
was the sole problem. The environment does have extractable information,
but the structural geometry of 1D makes targeted vs. random targeting
functionally equivalent.

---

## iter_035 — 1D Collision Inevitability

**Mechanism tested:** Direct measurement of the passive collision rate
as a hard ceiling on how much information the pointer can receive without
targeted action. Introduced a pre-registered analytical ceiling gate:
PASSIVE must produce ≤ 3.0 valid elastic collisions per object over
2000 steps for ORACLE to have meaningful headroom.

**Failure mode:** PASSIVE accumulated 12.27 valid collisions per object
on average — 4.1× above the 3.0 ceiling gate. In 1D, all objects share
the same axis as the pointer, so the collision cross-section equals the
full arena width. Every object trajectory must cross the pointer's
position (or come within the extended collision radius). Collision is
not rare; it is geometrically inevitable. ORACLE can at best multiply
this rate; it cannot create differential targeting when everything
already collides by default. The PASSIVE ceiling is too high for any
targeted strategy to show meaningful improvement.

**What this ruled out:** The 1D geometry itself as a viable arena for
behavioral discrimination. Collision inevitability is a structural
property of 1D with a central pointer: it cannot be fixed by metric
redesign, parameter tuning, or policy changes within the same geometry.

---

## iter_036 — Coverage Uniformity

**Mechanism tested:** The most radical 1D redesign possible — replacing
the physical pointer with a ghostly foveated gaze that moves via random
walk and fires probes probabilistically. This removed collision inevitability
entirely (the gaze is non-physical) and tested whether coverage
heterogeneity (some objects probed more than others) could create
behavioral headroom even in 1D.

**Failure mode:** Both arms failed the coverage-uniformity gate. RANDOM
produced per-probe CVs of 0.36 and 0.46 (across two measurement methods),
both below the 0.50 threshold. ORACLE similarly failed to achieve
meaningful heterogeneity. The 1D random walk over a line segment with
N=3 objects converges toward near-uniform coverage because: (a) the
arena is small relative to gaze_radius=8 (on a 128px line, objects are
never far apart), (b) the random walk mixes rapidly in 1D, and (c) all
objects share the same axis so they cannot be "hidden" or "unreachable."
Even removing physical collisions entirely, the 1D geometry enforces
coverage convergence that eliminates the heterogeneity needed for
targeted perception to matter.

**What this ruled out:** The last possible escape route within 1D.
Foveated gaze was designed to be the most permissive setup — no physical
constraints, no collision inevitability, purely probabilistic sampling.
It still failed. There is no remaining mechanism to redesign within 1D
that has not been tried.

---

## Why This Chain Is a Complete Null

Each iteration addressed the specific failure mode of its predecessor:

| Iteration | Addressed failure of | New mechanism tested | Result |
|-----------|--------------------|--------------------:|--------|
| iter_033 | — (baseline) | POMLRE behavioral bracket | ORACLE ≈ RANDOM (gap 0.0001) |
| iter_034 | Metric saturation | MALRE free-information analysis | Active-passive gap OK, but ORACLE-RANDOM still negligible |
| iter_035 | Free info ≠ targeted info | Collision ceiling gate | PASSIVE 4.1× above ceiling — collision inevitable in 1D |
| iter_036 | Collision inevitability | Foveated gaze (no physical collisions) | Coverage too uniform — both arms fail CV gate |

The chain terminates at iter_036 because foveated gaze is the most
radical departure from the original design that remains within the 1D
constraint. It removed the physical pointer, the collision dynamics,
and the deterministic physics — leaving only a random walker with a
probabilistic probe. If the 1D geometry cannot support behavioral
discrimination under these maximally permissive conditions, no further
redesign within 1D can reasonably be expected to succeed.

---

## Scope and Boundaries

**This document covers only the 1D × N=3 × 128px sandbox.**
The finding is scoped to this specific parameterization. Different object
counts, arena sizes, or interaction mechanics may yield different results
within 1D — that is an open empirical question not addressed here.

**This document makes no claim about M2 (thalamic gating).**
M2 was designed as a perception module that gates between spatial and
dynamic representations based on surprise. Its performance was never
the bottleneck in these iterations; the environment structure was.
M2 remains untestable under these conditions, not falsified.

**This document makes no claim about higher dimensions.**
The null result here motivates (but does not guarantee) testing in 2D,
where off-axis passage removes collision inevitability and 2D random
walks mix more slowly than 1D walks. Whether 2D succeeds is a separate
empirical question.

**This is a clean negative result regardless of what comes next.**
Four iterations, four mechanism-distinct approaches, all converging on
the same structural limitation of 1D geometry with a central interaction
target. The result is robust, well-documented, and does not depend on
any particular metric, policy, or hyperparameter choice. Finding where
perception is *not* behaviorally load-bearing is a valid scientific
contribution — it defines the boundary conditions of the problem.
