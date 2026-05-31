# Research Manager Log - Iteration 034

## Iteration 034 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (iter_034)

1. Directional — this iteration is benchmark-validation only, not a sufficiency test. Build a non-degenerate behavioral benchmark as a stand-alone deliverable: an active-vs-passive *dynamics-learning* metric on N≥3 (so collisions are pairwise-specific and the 1/2-saturation regime is avoided), continuous-valued (held-out prediction error or information gain about a hidden physical parameter such as mass), and pre-registered with two pass/fail gates — non-degeneracy (ORACLE − RANDOM ≥ 0.15 with lower CI clear of zero over ≥5 seeds including 53/71) AND end-goal validity (the metric measures the actual Pillar-E curiosity capability, not a proxy for it). Do NOT attach the LEARNED iter_028 representation and do NOT run the branch (a)/(b)/(c)/(d) rule in this iteration — that is iter_035.

2. Directional / discipline — neutralize the CLTSMotorController surprise-EMA confound *before* the bracket is interpreted, otherwise iter_034 measures controller calibration rather than perception. Either (preferred) choose a metric that does not route through the EMA-coupled motor at all (e.g. probe-prediction error on a held-out window, scored from raw rollouts), or per-condition recalibrate the surprise-EMA so RANDOM/LEARNED/ORACLE each receive a fairly tuned controller. State the chosen path explicitly in the pre-registration and justify why it removes the 33-vs-58 px artifact observed in iter_033.

3. Discipline — ORACLE correctness is a PRECONDITION, not a finding. Pre-register the ORACLE sanity checks as gates that must pass before any RANDOM/ORACLE comparison is read: surprise scale within an explicitly-stated physical range, and ≥80% of surprise spikes time-aligned (±2 steps) with ground-truth collision events. Compute and report the metric's analytic random-baseline ceiling in the pre-registration; if useful range (1 − ceiling) < 0.3, reject the metric before running. Use restrained language throughout — a passed benchmark is "a validated discriminating protocol," not "evidence the agent learns dynamics." Preserve M1/M3, the iter_028 substrate, d_t=3 frozen, decoder-free, no positional encoding; do not reopen M2.

---

## Iteration 034 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
A dynamics-learning benchmark with N=3 objects, where the primary metric is
mass-estimation MAPE (Mean Absolute Percentage Error) computed from ground-truth
collision logs, is non-degenerate: an ORACLE-TARGETED policy that actively probes
under-observed objects achieves mass-estimation MAPE at least 0.15 lower than a
RANDOM policy (RANDOM_MAPE - ORACLE_MAPE ≥ 0.15), with the lower 95% bootstrap
CI of this gap ≥ 0.05 over ≥8 seeds including hard seeds 53 and 71. The metric
directly measures the Pillar-E end-goal capability (hidden-parameter inference
through active interaction) without routing through the EMA-coupled motor
controller, thereby neutralizing the 33-vs-58 px tracking artifact from iter_033.

**Proposed Falsification Criterion:**
The benchmark is falsified if EITHER:
(F1) RANDOM_MAPE - ORACLE_MAPE < 0.15 (the ORACLE-TARGETED policy does not
     substantially outperform random action on mass estimation), OR
(F2) The lower 95% bootstrap CI of (RANDOM_MAPE - ORACLE_MAPE) includes zero
     (the gap is not statistically reliable), OR
(F3) Any ORACLE sanity-check precondition fails:
     S1: ORACLE achieves ≥3 collisions per object (targeting works),
     S2: ORACLE total collision count ≥ PASSIVE total collision count,
     S3: ≥90% of logged collision events show |Δv| > 0.5 px/step,
     S4: ORACLE mean surprise per step ∈ [0.01, 100] (physical range,
         not 146k as in the buggy iter_033 v1/v2),
     S5: ≥80% of ORACLE surprise spikes align (±2 steps) with ground-truth
         collision events.
If F3 fires, the ORACLE implementation is buggy and no comparison is interpreted.

**Proposed Method:**
Step 1: Create src/run_iter034_benchmark.py implementing the full experiment.

ENVIRONMENT: PhysicsSandbox(N=3), 2000 interaction steps, 8 seeds
[7, 31, 53, 71, 83, 97, 113, 163].

THREE CONDITIONS (no LEARNED representation — that is iter_035):

(A) ORACLE-TARGETED: Custom policy that (i) maintains per-object collision
    count, (ii) moves pointer toward the object with fewest observed collisions
    using PD control (Kp=2.0, Kd=0.5), (iii) pushes when within |error| ≤ 6.0
    of target, (iv) uses the full-physics simulate_physics() from iter_033 v3
    for surprise computation. Does NOT use CLTSMotorController — avoids the
    EMA confound entirely by implementing a clean information-gain-maximizing
    policy directly.

(B) RANDOM: Uniform random acceleration ∈ [-10, 10], random push with p=0.1,
    no motor controller.

(C) PASSIVE: No pointer action (acc=0, push=False). Pointer acts as a
    passive object. Only natural object-object collisions provide mass info.

MASS ESTIMATION PROCEDURE:
- During interaction, log all collision events: (step, obj_i, obj_j, v_i_pre,
  v_j_pre, v_i_post, v_j_post). Collision detection: |pos_i - pos_j| <
  radii_i + radii_j + threshold AND |Δv| > 0.5 for either object.
- From elastic collision physics, each collision (i,j) gives:
  m_i * (v_i - v_i') = m_j * (v_j' - v_j)
  → linear constraint on mass vector [m_0, m_1, m_2].
- Pointer-object collisions give absolute mass (pointer mass = 10).
- Solve overdetermined system via least-squares (np.linalg.lstsq).
- Objects with 0 observed collisions: m_hat = 5.5 (prior mean).
- MAPE = mean(|m_hat_i - m_true_i| / m_true_i) across 3 objects.

PRIMARY METRIC: MAPE (lower is better).
Gate 1 (non-degeneracy): RANDOM_MAPE - ORACLE_MAPE ≥ 0.15, lower 95% CI ≥ 0.05.
Gate 2 (end-goal validity): The metric measures hidden-parameter inference
through active interaction, which IS the Pillar-E artificial-curiosity
capability, not a proxy for it. Justified because mass is only observable
through collisions, and active probing systematically increases collision
coverage.

SECONDARY METRIC: Held-out velocity prediction MSE.
Split collision data 80/20 by time. Fit masses from training collisions,
predict post-collision velocities on test collisions, compute MSE.

ORACLE SANITY CHECKS (pre-conditions, must ALL pass before interpreting F1/F2):
S1: ORACLE achieves ≥3 collisions per object (mean across seeds).
S2: ORACLE total collision count ≥ PASSIVE total collision count (per seed).
S3: ≥90% of logged collision events show |Δv| > 0.5 px/step.
S4: ORACLE mean surprise per step ∈ [0.01, 100].
S5: ≥80% of ORACLE surprise spikes align (±2 steps) with collisions.

EMA CONFOUND NEUTRALIZATION (stated explicitly per Manager hint):
The mass-estimation metric is computed entirely from ground-truth collision
logs (positions, velocities from env.info), NOT from the agent's surprise
signal or motor controller state. The ORACLE-TARGETED policy uses a custom
information-gain-maximizing controller (not CLTSMotorController), so the
different surprise distributions between conditions cannot affect the metric.
The iter_033 tracking artifact (ORACLE 58 px vs LEARNED 33 px) arose because
the CLTSMotorController's EMA calibrated differently under qualitatively
different surprise distributions, causing different attention switching rates.
By not routing the metric through the EMA-coupled motor, this artifact is
structurally impossible in the new benchmark.

Step 2: Run the experiment (8 seeds × 3 conditions = 24 runs).

Step 3: Analyze results. Report per-seed MAPE for each condition, compute
gap and CI, check all sanity preconditions, apply gates.

FILES CREATED/MODIFIED:
- src/run_iter034_benchmark.py (new, main experiment)
- src/pre_registration.md (auto-generated from plan)

PRESERVED (per directive): separate backbone architecture, d_t=3 frozen,
GDASR log-only (M3), decoder-free, no positional encoding, M2 not reopened.
No LEARNED representation is used in this iteration.

---

## Iteration 034 -> Planner [Strategic Guidance]

# Manager's Note — iter_034 Plan Critique

**1. Discipline — the proposed gate is a construction-vs-empirical risk; reframe it as a benchmark sanity check, not a finding.**
ORACLE-TARGETED is defined as "move toward the object with fewest observed collisions" while RANDOM is uniform action. RANDOM_MAPE − ORACLE_MAPE ≥ 0.15 is therefore asking "does a hand-coded coverage-maximizing controller produce more diverse collisions than uniform random action on a metric whose error depends on collision coverage?" — the answer is yes by construction. That is fine for *benchmark validation* (the whole point of this iteration), but the pre-registration must state this explicitly: a passing gate establishes *that the metric responds to targeted vs untargeted action*, **not** that the benchmark has measured perception or curiosity. Avoid "the benchmark measures hidden-parameter inference" in the report; use "the benchmark discriminates targeted from random coverage of the collision manifold, which is a necessary precondition for measuring perception in iter_035." Add PASSIVE to the gates (e.g. require ORACLE < RANDOM < PASSIVE on MAPE, with both gaps clear of zero) — without it, the run cannot detect the failure mode where natural inter-object collisions saturate mass inference and action contributes nothing.

**2. Construction-vs-empirical — 1D elastic collision physics may make MAPE trivially solvable, collapsing the benchmark to a pure coverage test.** In 1D under elastic collision, momentum + kinetic-energy conservation determines the mass ratio of a colliding pair *exactly from a single clean collision* (the plan only writes the momentum constraint, but KE conservation is also available and the system becomes overdetermined per collision). Pointer-object collisions (anchor mass = 10) then give absolute mass exactly. So with noiseless `env.info` velocities, MAPE for objects that have collided at least once with the pointer chain is ≈ 0, and for unobserved objects defaults to the prior. The MAPE gap then collapses to a function of *which objects got touched* — a coverage indicator dressed as inference. Either (a) inject realistic observation noise on velocities before fitting masses, so the inference itself has non-trivial estimator variance, or (b) acknowledge in the pre-registration that this iter_034 metric is a coverage-discrimination test and that iter_035's perception comparison will need the noise injected so LEARNED-with-imperfect-velocities is not artificially handicapped against ORACLE-with-ground-truth-velocities. Pick one in writing before running.

**3. Discipline — pre-registration and the iter_035 attachment plan.** Reminder: the Orchestrator will auto-write and commit `src/pre_registration.md` from the `hypothesis` and `falsification_criterion` fields above; the sub-agents must read it and adhere. Three corrections before that file is frozen: (i) clarify S4 — for a ground-truth physics predictor, surprise should be ≈ 0 except at numerical noise; a [0.01, 100] band is suspiciously wide and may mask a buggy predictor (state the *expected* mean explicitly, not just an envelope); (ii) tighten language throughout — "validates that targeted probing beats random on MAPE" rather than "measures the Pillar-E end-goal capability" (the latter is a claim about the agent, not about the benchmark); (iii) sketch the iter_035 attachment now: if LEARNED is plugged into the same custom information-gain controller, the policy is hard-wired and perception barely enters; if LEARNED routes through CLTSMotor with surprise-driven targeting, the iter_033 EMA confound returns. State which path iter_035 will take, and what *that* iteration's bracket will look like, so iter_034's benchmark is shaped against a known downstream use. A benchmark with no committed downstream is just bookkeeping.

PRESERVE: M1/M3, iter_028 substrate, d_t=3 frozen, decoder-free, no positional encoding, M2 not reopened.

---

