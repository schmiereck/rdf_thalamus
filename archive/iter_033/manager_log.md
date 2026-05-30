# Research Manager Log - Iteration 033

## Iteration 033 -> Planner [Pre-Planning Hints]

## Manager's Pre-Planning Hints — iter_033

**Hint 1 (directional, binding).** Execute the user-mandated three-condition oracle bracket exactly as specified: RANDOM (lower bound), LEARNED (iter_028 mean-pool E1 substrate — *not* the collapsed centroid variants), and ORACLE (ground-truth positions/identity fed as `z_coord`/`z_dyn`, with surprise from a known-dynamics physics predictor). All three conditions must share environment, seed bank (n≥10, including hard seeds 53 and 71), and `CLTSMotorController` logic — any deviation in motor code or environment between conditions invalidates the bracket. The normalized gap `g = (LEARNED - RANDOM) / (ORACLE - RANDOM)` with its four-branch decision rule (a/b/c/d) must be written into the pre-registration verbatim, before the run, and committed to before any seed executes.

**Hint 2 (directional, scope discipline).** Resist three predictable scope drifts: (i) do not re-open the M2 objective comparison — it stays "untestable" until constraint-relaxation provides a working non-mean-pool readout; (ii) do not introduce a new representation-side ΔR² gate alongside the behavioral gate, since reintroducing one revives the dissolved representation-quality-gate loop; (iii) do not let "ORACLE was hard to build" become a reason to weaken its construction — if the oracle predictor is approximate, that approximation must be quantified and reported as a bracket-ceiling caveat, not absorbed silently. The primary behavioral metric for the `g` computation must itself be pre-declared (recommend post-collision probing selectivity, since it directly calibrates the iter_031 1.34× signal); secondary metrics (tracking error, mass-change causal sensitivity) get reported but do not drive the gate.

**Hint 3 (scientific discipline).** The oracle bracket is precisely the right instrument because it makes the `g` ratio *relative to a measured ceiling*, not a guessed absolute — preserve that relativization in the language of the report. Use restrained phrasing throughout: "the representation is consistent with sufficiency for this behavior," not "the representation is sufficient"; "perception does not appear to limit behavior at the measured n," not "perception is solved." Treat branch (c) — small `(ORACLE - RANDOM)` — as a first-class possible outcome that would *invalidate* the whole behavioral-pivot strategy by showing the task itself does not discriminate perception quality; if (c) fires, the report must say so plainly rather than reinterpreting partial signals as evidence for branches (a) or (d).

---

## Iteration 033 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
The best available mean-pool representation (SFA+VICReg sfa_weight=5.0, separate backbone,
ΔR²≈0.275, 0% collapse, iter_029 Arm B) supports functional surprise-driven behavior
that accounts for at least 20% of the gap between random (lower bound) and oracle
(perfect-perception upper bound) baselines, as measured by post-collision attention
selectivity on N=2 collision-sparse environments. Specifically, the normalized gap
g = (LEARNED - RANDOM) / (ORACLE - RANDOM) on post-collision attention selectivity
(version B: attended object matches max-velocity-change object) will be > 0.20,
with the four-branch decision rule determining project direction.

**Proposed Falsification Criterion:**
F1 FAIL: g <= 0.20 — the learned representation accounts for less than 20% of the
oracle-random gap, meaning the representation PROVABLY limits behavior and
constraint-relaxation (decoder, higher d_t, or the VICReg-upstream-of-gate fix
for the iter_032 cross-backbone collapse) is justified.

Additionally, if branch (c) fires — (ORACLE - RANDOM) is itself small (|difference| < 0.10
on primary metric) — the task or motor protocol is the bottleneck, NOT perception,
and the behavioral-pivot strategy itself is invalidated for this protocol/environment.

The lower 95% CI of g (bootstrapped across seeds) must be reported; if it excludes
the boundary between adjacent branches, the branch assignment is confident; if not,
both branches are reported as possible.

**Proposed Method:**
## Three-Condition Oracle Bracket Experiment

### Conditions (all sharing identical environment, seed bank, and CLTSMotorController logic)

1. **RANDOM (lower bound):** CLTSMotorController with random token_locus selection
   each step (no encoder, no prediction). The controller's attention_cooldown is
   set to 0 so locus is re-randomized every step, and all surprise EMA values are
   zeroed to prevent any learned structure from influencing behavior.

2. **LEARNED (test condition):** Load iter_029 Arm B checkpoints (SFA+VICReg,
   sfa_weight=5.0, separate backbone, mean-pool readout, d_max=8, d_t=3 trained).
   Evaluate with d_t=2 (first 2 channels active for N=2 objects). Run encoder +
   predictor forward pass, feed z_pred_coord, z_target_coord, z_pred_dyn,
   z_target_dyn to the standard CLTSMotorController.get_action().

3. **ORACLE (upper bound):** Feed ground-truth positions (info['positions'][:d_t])
   as z_coord, ground-truth mean-color (np.mean(info['colors'][:d_t], axis=1))
   as z_dyn, linear-extrapolation predicted positions (prev_pos + prev_vel * dt)
   as z_pred_coord, and z_dyn as z_pred_dyn (identity is constant). Feed these
   tensors to the SAME CLTSMotorController.get_action() method — identical
   surprise computation (per-channel MSE + EMA normalization + attention cooldown),
   identical PD tracking, identical push logic.

### Environment
PhysicsSandbox(N=2) — collision-sparse by design. Mass perturbation at step 1000:
multiply object 0's mass by 1.5× (same as iter_031 Part B). 2000 evaluation steps
per seed.

### Seed Bank
12 seeds: [7, 17, 31, 53, 71, 83, 97, 101, 107, 113, 137, 163]
Includes hard seeds 53 and 71 (mandated by user/manager).

### Primary Behavioral Metric
Post-collision attention selectivity (version B): for each collision event,
within POST_COLLISION_WINDOW=15 steps after the collision, the fraction of steps
where the attended object (token_locus mapped to object index via closest-centroid)
matches the max-velocity-change object. This is the same metric as iter_031 Part B,
directly calibrating the 0.59-vs-0.44 signal.

### Secondary Metrics (reported but do NOT drive the gate)
- Mean tracking error (pointer position vs attended centroid, in pixels)
- Perturbation selectivity (fraction of steps 1000-1099 where attended object = object 0)

### Decision Rule (PRE-COMMITTED, VERBATIM from manager/user)
g = (LEARNED_primary - RANDOM_primary) / (ORACLE_primary - RANDOM_primary)

(a) g >= 0.70 AND lower bootstrapped 95% CI >= 0.50: representation is SUFFICIENT
    for the behavior. ΔR²_color >= 0.30 was a distraction. Project ADVANCES to
    Phase 2/3 integration on the mean-pool representation.

(b) g <= 0.20: representation PROVABLY limits behavior. ONLY THEN is
    constraint-relaxation justified (iter_034+: decoder, higher d_t, or the
    deferred VICReg-upstream-of-gate fix for iter_032 cross-backbone collapse),
    with the concrete target set by the measured (ORACLE - RANDOM) gap.

(c) |ORACLE_primary - RANDOM_primary| < 0.10: the TASK or MOTOR PROTOCOL is the
    bottleneck, NOT perception. Fix the protocol/environment. Do NOT touch the
    representation. This branch invalidates the behavioral-pivot strategy for
    this specific protocol.

(d) 0.20 < g < 0.70: partial sufficiency. Advance to Phase 2/3 but flag perception
    as a known secondary limiter to revisit.

### Implementation Details

**File: src/run_iter033.py** (new script)

1. OraclePhysicsPredictor class: predicts next-step positions via linear
   extrapolation (pos + vel * dt). Surprised at collisions (velocity changes)
   but correct for constant-velocity motion. This is "known dynamics" — the
   predictor knows about constant-velocity physics but is genuinely surprised
   by collision events (which change velocities unpredictably).

2. OracleController class: wraps ground-truth perception into the CLTSMotorController
   interface. Constructs z_coord (1, d_max) tensor from info['positions'],
   z_dyn (1, d_max) from mean color values, z_pred_coord from physics predictor,
   z_pred_dyn = z_dyn. Calls CLTSMotorController.get_action() with these tensors.
   This ensures IDENTICAL motor code across all three conditions.

3. For the RANDOM condition: uses CLTSMotorController but overrides token_locus
   to random each step and zeroes all EMA statistics, so no learned surprise
   structure influences behavior. Equivalent to the "random" condition from
   iter_031 Part B.

4. For the LEARNED condition: loads iter_029 Arm B checkpoints
   (archive/iter_029/results/checkpoints/b_sfavicreg,_sfa_5.0_seed{N}.pt),
   sets d_t=2 for evaluation, runs standard encoder+predictor forward pass,
   feeds outputs to CLTSMotorController.

5. Collision detection: same as iter_031 Part B (COLLISION_DIST_THRESHOLD=4.0,
   COLLISION_VELOCITY_CHANGE_THRESHOLD=1.0, POST_COLLISION_WINDOW=15).

6. Channel-to-object mapping: use closest-centroid matching (same as iter_031).

7. Bootstrap CI for g: resample seeds 10000 times, compute g for each resample,
   report 95% percentile CI.

### Preserved Constraints
- Separate backbone + iter_029 config (SFA+VICReg, sfa_weight=5.0)
- d_t=2 frozen for N=2 (GDASR log-only, M3)
- Decoder-free (no reconstruction)
- No positional encoding
- M2 mandate stays "untestable" until constraint-relaxation phase
- No new representation-side ΔR² gate alongside the behavioral gate

---

## Iteration 033 -> Planner [Strategic Guidance]

## Manager's Note — iter_033 Plan Critique

**Point 1 (binding — substrate deviates from user hint, must be reconciled).**
The user hint *explicitly names the substrate*: "LEARNED — the best VIABLE representation, i.e. mean-pool E1 (separate backbone, **iter_028 config**, **d_t=3** frozen)". The plan instead loads **iter_029 Arm B** (SFA+VICReg sfa=5.0) at **d_t=2**. Both deviations need explicit handling before execution:

- *Substrate choice:* iter_029 Arm B is also mean-pool / 0% collapse and has higher ΔR² (0.275 vs 0.045), so a "best viable" interpretation includes it — but silently substituting it changes what a failure means. If g<0.20 on Arm B, it does *not* rule out that E1-VICReg-only would have done the same (or better, since SFA's slowness on z_dyn shrinks err_dyn and could systematically degrade the surprise readout — the very M2-interaction warning in the goal doc, §4.D). Either (a) include both as parallel LEARNED arms (preferred — and far more diagnostic), or (b) drop Arm B and use the user-named iter_028 substrate. Do **not** silently swap.
- *d_t = 2 vs 3:* the user hint says "d_t=3 frozen". Using d_t=2 because the env is N=2 is defensible, but it must be flagged in the pre-registration as a *deliberate* deviation, not a quiet matching-to-task. Better: keep d_t=3 (one unused channel) so the architecture under test is identical to the named config; this also tests whether the unused channel collapses or stays VICReg-clean.

**Point 2 (binding — ORACLE-bracket integrity).** Three issues that, if unaddressed, will produce an uninterpretable bracket:

- *Same-code-different-input is not the same as same-conditions.* `CLTSMotorController`'s EMA statistics, attention-cooldown, and push threshold (`μ+σ`) were calibrated implicitly against noisy learned surprise. ORACLE will produce qualitatively different surprise (clean zero between collisions, sharp spikes at collisions); the SAME `get_action()` will behave differently because the surprise distribution shifts shape. Pre-commit that the ORACLE represents *"ceiling under the existing motor code,"* not the absolute behavioral ceiling — and report the per-condition surprise distributions so the reader can see the bracket is not deformed by EMA mismatch.
- *Bracket ordering is not guaranteed.* Pre-commit a sanity check **before** computing g: require `ORACLE > LEARNED > RANDOM` on the primary metric (or at minimum `ORACLE > RANDOM`). If LEARNED beats ORACLE, or RANDOM beats ORACLE, the metric or the oracle construction is broken and g is meaningless — report this outcome plainly rather than computing a ratio.
- *Channel-to-object mapping is asymmetric.* "Closest-centroid" mapping is trivially correct for ORACLE (its centroid *is* the position) but biased for LEARNED (its centroid is a learned soft-argmax). Either use the same imperfect mapping for ORACLE too (e.g. forward through the encoder of a random untrained net, or accept the asymmetry and quantify its size).
- z_pred_dyn = z_dyn under ORACLE means err_dyn ≡ 0; surprise is purely position-driven. Confirm in the plan that LEARNED's surprise is summed the same way and report the position/identity decomposition.

**Point 3 (scientific discipline).** The pre-registration file must be written and committed **before any seed executes** — the Orchestrator will write `src/pre_registration.md` from the plan's `hypothesis` and `falsification_criterion` fields, so both must be sharpened now. Specifically:

- Report the **raw triple** `(RANDOM, LEARNED, ORACLE)` with CIs, not only `g`. A ratio hides denominator instability — with n=12 and a possibly small `(ORACLE − RANDOM)` gap, `g` can swing wildly while the underlying values are stable.
- Promote **branch (c)** to a first-class hypothesis in the falsification text, not a footnote — it would invalidate the entire behavioral-pivot strategy and that is exactly the kind of outcome the pre-registration must explicitly own *before* seeing data. Pre-commit the (c)-threshold (currently `|ORACLE − RANDOM| < 0.10`) and its construction.
- Restrained language in the report: "is consistent with sufficiency under this protocol" / "does not refute the representation's adequacy for the measured behavior" — never "the representation is sufficient." A passed gate at n=12 with one primary metric on one environment is evidence, not proof.

---

