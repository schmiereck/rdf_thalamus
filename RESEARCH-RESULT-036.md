# RDF Milestone Review — Iteration 036 — Null Result: Foveated Gaze in 1D Cannot Make Perception Load-Bearing; Four-Iteration Environment-Design Null Chain Complete

## 1. Pre-Declared Hypothesis and Falsification Criterion
Verbatim from the iter_035/036 pre-registration recorded in
the prior journal:

- Hypothesis: A 1D environment in which the pointer is a
  non-physical gaze locus, object state is observable only
  within a foveated window of radius r, and excitation
  requires an agent-issued localized probe action, will
  cause coverage of objects to depend on gaze policy,
  enabling ORACLE − RANDOM ≥ 0.15 on a per-object
  mass-estimation metric.
- Analytical ceiling gate (pre-registered, computed first):
  under RANDOM policy, the coefficient of variation of
  per-object probe-event counts must be ≥ 0.50, computed
  analytically or by a short rollout. Below this threshold,
  RANDOM already distributes coverage too evenly for ORACLE
  to have headroom, and the bracket experiment is blocked.
- Two arms factor the two levers: Arm A retains obj-obj
  collisions (foveation alone); Arm B removes them
  (foveation + pass-through).
- Escalation rule: if the CV gate fails in both arms, the
  foveation lever is also insufficient and the
  pre-committed meta-escalation (path i/ii/iii in the
  journal) triggers without further 1D iteration.

## 2. Experimental Protocol
- Arena: 1D, 128 pixels, N=3 objects.
- Pointer: non-physical gaze locus, no collisions with objects.
- Observation: object state available iff object centroid
  within gaze window of radius GAZE_RADIUS=8 pixels.
- Excitation: localized probe-force impulse at the gaze
  locus is the only mechanism that produces object dynamics
  changes.
- Arms:
  - Arm A: foveation + obj-obj collisions retained.
  - Arm B: foveation + pass-through obj-obj physics.
- Policy under measurement: RANDOM gaze trajectory.
- Measurement: per-object probe-event counts under a fixed
  rollout; coefficient of variation across the 3 objects.
- Gate: CV ≥ 0.50 required for full-bracket execution.
- Compute scope: only the analytical ceiling gate was
  executed; the full ORACLE-vs-RANDOM bracket and any
  learned model training were blocked by the gate failure.

## 3. Observed Quantities
- Arm A RANDOM per-object event-count CV: **0.36**.
- Arm B RANDOM per-object event-count CV: **0.46**.
- Pre-registered threshold: **0.50**.
- Result: gate fails in both arms (Arm A by 0.14, Arm B by
  0.04). Arm B is closer to threshold but still below.
- Cumulative null chain across four iterations of
  environment redesign:
  - iter_033: ORACLE − RANDOM ≈ 0 on behavioral pivot.
  - iter_034: v2 MALRE ORACLE − RANDOM = 0.031 within
    active regime.
  - iter_035: PASSIVE pointer 12.27 valid collisions per
    object vs threshold 3.0 (saturation gate, 4× overshoot).
  - iter_036: RANDOM CV 0.36 / 0.46 vs threshold 0.50
    (heterogeneity gate, both arms below).

## 4. Verdict
**Refuted.** The pre-declared hypothesis that foveated gaze
in a 1D × N=3 × 128-pixel arena makes perception
behaviorally load-bearing under an ORACLE-vs-RANDOM bracket
is refuted: the necessary precondition (RANDOM coverage
heterogeneity) is not met under either tested arm. In
combination with the prior three iterations, the broader
hypothesis that *any* tested 1D-sandbox configuration can
produce ORACLE − RANDOM discrimination on a behavioral
metric is refuted across four pre-registered, mechanistically
distinct redesigns. The pre-committed meta-escalation to a
choice among (i) 2D environment, (ii) re-frame the
deliverable, or (iii) relax the decoder-free constraint is
triggered. Iter_037 will execute that meta-decision; no
further 1D environment redesign is sanctioned.

## 5. Construction-vs-Empirical Note
This result is empirical, not definitional. The CV
threshold of 0.50 is a pre-registered design choice
motivated by the need for ORACLE to have measurable
headroom over RANDOM; it was not set to guarantee any
particular outcome and could in principle have been
exceeded by a redesign that concentrated objects spatially
or used a more restrictive gaze radius. The measured CV
values (0.36, 0.46) are properties of how random gaze
trajectories of finite length cover a 1D arena of 128
pixels containing 3 objects — a geometric-statistical
property of the test environment under the chosen radius.
The four-iteration null chain itself is a stronger
empirical statement: each iteration tested a distinct
mechanism (collision pattern, environment density,
geometry, observation gating) and each failed at a
structurally different gate. The cumulative pattern is
empirical evidence for the constraint, not a derivation
from it.

## 6. Limitations
- The CV gate failure was tested only at GAZE_RADIUS=8
  pixels. A substantially smaller radius (e.g. 2 pixels)
  might push CV above 0.50, but would create a separate
  exploration problem and would not have been within the
  pre-registered parameter envelope. Whether *any* radius
  in (0, 8] could pass the gate has not been measured.
  This is acknowledged but not chased, because moving
  outside the pre-registered envelope to recover the
  hypothesis would violate Gate 3 hygiene.
- The result does not show that no 1D environment could
  ever produce an open bracket; it shows that the four
  principled, mechanism-distinct redesigns tried so far
  have not, and the prior on further 1D tweaks succeeding
  is now low.
- The result does not invalidate the iter_028 representation
  substrate or the MALRE v2 coverage test; both remain
  valid within their established scopes.
- The result does not pre-judge the iter_037 meta-decision.
  It only forces that decision to be made.
- The four-iteration sequence does not establish
  representation-quality conclusions of any kind; it
  establishes only environment-design conclusions. The M2
  mandate's "untestable" status is unchanged.
- The analytical ceiling gate primitive has now saved
  compute four times, but no positive validation of any
  behavioral claim has been produced; the project's
  methodological maturity is currently outpacing its
  scientific progress, which is itself the signal driving
  the meta-escalation.