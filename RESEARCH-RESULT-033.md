# RDF Milestone Review — Iteration 033 — Null Result: Behavioral Pivot Protocol Degenerate Under N=2 Post-Collision Selectivity

## 1. Pre-Declared Hypothesis and Falsification Criterion
Pre-registered hypothesis (verbatim from iter_033 pre-registration,
per the user hint binding rule): the iter_028 separate-backbone +
mask_dyn_sim + coord_vicreg + VICReg-only representation (ΔR²_color
= 0.045) drives surprise-based behavior on CLTSMotorController. The
test was constructed as a three-condition bracket — RANDOM (ablation
policy), LEARNED (iter_028 substrate), ORACLE (ground-truth
positions + ground-truth identity + physics-based surprise
predictor) — on identical environment (N=2 collision-sparse,
calibrated in iter_031 Part B), seed bank (≥10 seeds including
hard seeds 53/71), and motor logic.

Pre-committed decision rule on the normalized gap
g = (LEARNED − RANDOM) / (ORACLE − RANDOM) on the primary
behavioral metric (post-collision selectivity):
- (a) g ≥ 0.70 with lower CI ≥ 0.50 → representation sufficient,
  advance to Phase 2/3.
- (b) g ≤ 0.20 → representation provably limits behavior, license
  constraint-relaxation.
- (c) (ORACLE − RANDOM) itself small (degenerate bracket) → task or
  motor protocol is the bottleneck, fix the protocol; do NOT touch
  the representation.
- (d) intermediate g (0.20–0.70) → partial sufficiency.

## 2. Experimental Protocol
- Environment: 1D RGB physics sandbox, 128 pixels, N=2 objects
  under elastic collision dynamics (iter_031 calibration).
- Conditions: 3 (RANDOM, LEARNED, ORACLE), seeds ≥10 each,
  including hard seeds 53 and 71.
- Motor: CLTSMotorController, default config (Kp=2.0, Kd=0.5,
  Kv=0.5, push trigger at mu+sigma + |err|≤6.0, 15-step attention
  cooldown).
- LEARNED representation: iter_028 separate-backbone CNN
  (d_t=3 frozen, GDASR log-only, decoder-free, no positional
  encoding), mean-pool z_dyn readout.
- ORACLE: ground-truth `info['positions']` as z_coord and
  color/size as z_dyn, with surprise computed from a known-dynamics
  physics predictor. Implemented in three successive versions
  (v1: timing bug + linear extrapolation; v2: timing fix + linear
  extrapolation; v3: full physics simulator).
- Primary metric: post-collision attention selectivity (binary
  match on the max-velocity-change object within an N-step window).
- Secondary metrics: centroid tracking error; causal sensitivity to
  mass change (planned, not reached because bracket degenerated).

## 3. Observed Quantities
- ORACLE-v1 (buggy timing): surprise scale ~146 000; ORACLE < RANDOM
  (flagged as buggy implementation, not a real branch (b) signal).
- ORACLE-v2 (timing fixed, linear extrapolation): surprise scale
  ~164 000; ORACLE − RANDOM ≈ 0.007 on primary metric.
- ORACLE-v3 (full physics predictor, definitive): surprise scale
  ~310; ORACLE − RANDOM = 0.0001 on primary metric.
- LEARNED tracking error: 33 px; ORACLE tracking error: 58 px
  (paradoxically worse under perfect perception; attributed to
  surprise-EMA calibrating differently under the qualitatively
  different ORACLE surprise distribution).
- Random baseline ceiling under the metric's structure: ~0.50,
  matching the empirical ORACLE rate (both objects participate in
  every collision; "correct" choice is one of two).
- Falsification threshold for branch (c): |ORACLE − RANDOM| < 0.10.
  Observed gap 0.0001 ≪ 0.10. Branch (c) fires unambiguously.

## 4. Verdict
**Refuted (the protocol-discriminates-perception assumption was
refuted; representation sufficiency is not adjudicated by this
experiment).** The bracket degenerated: ORACLE and RANDOM are
empirically indistinguishable on the primary metric. Per the
pre-committed branch (c) rule, the conclusion is that the task /
motor-protocol combination does not discriminate perception
quality, and the representation is therefore not under test. The
representation's behavioral sufficiency (branch a) and behavioral
insufficiency (branch b) are both UNTESTED, not refuted, by this
experiment. No license for constraint relaxation is created.

## 5. Construction-vs-Empirical Note
Part of the result follows from construction: with N=2 objects and
both participating in every collision, the random match rate on a
"pick the right object" metric is exactly 1/2 by combinatorics,
and this is also the metric's saturation point. The structural
ceiling could have been computed analytically before the run —
this is a *methodological failure* of the iter_031/033 protocol
design that the iter_033 bracket caught empirically.

What is genuinely empirical: (1) that the LEARNED representation
produces non-collapsing surprise statistics at all on the
iter_028 substrate, confirming the substrate is at least
behaviorally viable; (2) that ORACLE under sharp clean surprise
produces *worse* tracking than LEARNED under noisy surprise, which
is a non-obvious mechanistic finding about
CLTSMotorController's surprise-EMA coupling and is independent of
the metric saturation issue; (3) that three ORACLE implementations
were needed to get the bracket right, validating the practice of
ORACLE sanity-checking before interpretation.

## 6. Limitations
- This result does NOT show that the iter_028 representation is
  insufficient for behavior. It shows only that the iter_033
  protocol cannot tell. The representation may be sufficient,
  insufficient, or partial; the bracket degeneracy makes all
  three indistinguishable.
- This result does NOT show that surprise-driven attention is a
  bad mechanism. It shows that the chosen metric does not
  discriminate good from bad attention policies in the N=2 regime
  with the current cooldown.
- This result does NOT generalize beyond the tested protocol.
  N≥3 environments, continuous metrics, or shorter-cooldown
  motors may all open the bracket; iter_034 will test these.
- The "ORACLE under perfect perception produces worse tracking"
  finding is an observation about CLTSMotorController, not about
  perception, and requires its own mechanistic follow-up before
  it can be promoted to a strategic constraint.
- The iter_031 0.59-vs-0.44 directional signal is now reinterpreted
  as in-distribution noise around a saturated metric, not as
  evidence of representation-driven behavior. Earlier journal
  entries treating that signal as a candidate finding should be
  read with this correction in mind.
- What would be needed next: an iter_034 calibration run producing
  a non-degenerate bracket (ORACLE − RANDOM ≥ 0.15) on at least
  one redesigned protocol axis (N≥3, continuous metric, or
  motor-cooldown sweep), with ORACLE sanity-checks
  (surprise scale within expected range, surprise spikes aligned
  to ground-truth events). Only then can iter_035 attach LEARNED
  and run the branch (a)/(b)/(c)/(d) rule meaningfully.