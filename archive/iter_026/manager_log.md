# Research Manager Log - Iteration 026

## Iteration 026 -> Planner [Pre-Planning Hints]

Manager's Pre-Planning Hints (iter_026)

1. **Hard scope lock — collapse-elimination only, no new objectives.** iter_026's sole deliverable is a training regime in which Arm A (control: training pipeline with no identity objective) collapses on ≤10% of seeds over ≥10 seeds. Reject any plan that introduces, tests, or "lightly probes" a new representation objective (ID-contrastive, separate encoder, BYOL, supervised variant, etc.) — the journal's Objective-Swapping Loop and Diagnostic-vs-Constructive warning both apply. If the Planner proposes mixing in a second arm "for free," that arm must be the same control under a *different regime knob*, not a different objective.

2. **Sweep regime knobs one-at-a-time against a pre-declared collapse-rate gate, not against downstream metrics.** The dependent variable for iter_026 is collapse rate alone; delta_R2_color and other downstream metrics are not the falsification target this iteration and must not be used to pick winners (doing so would re-introduce survivor-bias reasoning). The plan must pre-declare: which knobs are swept (LR, VICReg variance floor, VICReg-coefficient warm-up, batch size), the sweep grid, the order, and the stop rule (e.g. "first configuration to achieve ≤10% collapse over ≥10 seeds is the new canonical regime; if none clears, report the best achieved rate as a null and do not declare a winner"). Hungarian-primary matching remains the standing rule when matching enters any auxiliary reporting.

3. **Pre-declare what counts as collapse and what disqualifies a configuration, before any seeds are run.** Use the existing `has_collapsed` criterion in `evaluate_branch` augmented with the per-dimension batch-std floor, exactly as the goal specifies — do not redefine the collapse criterion mid-experiment to make a configuration pass. Also pre-declare a minimum training-loss sanity check (e.g. VICReg variance term actually engaged, not silently saturated) so that "no collapse" cannot be earned by a degenerate regime that simply fails to train. The honest acceptable outcome of iter_026 is a measured null ("no swept configuration cleared the ≤10% gate") — the Planner must treat that as a first-class deliverable, not a failure to be avoided by gate-relaxation.

---

## Iteration 026 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
There exists a training-regime configuration within the swept parameter space
{batch_size ∈ {32, 64}, var_weight ∈ {25, 50}, sim_weight ∈ {25 constant, 0→25 ramp},
learning_rate ∈ {3e-4, 1e-4}} that reduces the z_dyn collapse rate of the
NonParametricJEPASpatial encoder (JEPA+VICReg, d_max=8, d_t=3, centroid_gated
readout) from the current 30% (iter_025 v2) to ≤10% over 10 seeds with 8000
training steps. The most likely candidate is batch_size=64, because VICReg's
variance and covariance estimation reliability scales inversely with
1/sqrt(B), and B=32 provides marginal statistical power for per-dimension
std estimation across d_t=3 dimensions.

**Proposed Falsification Criterion:**
No single-knob variation in the sweep achieves ≤10% collapse rate (i.e., ≤1
collapsed seed out of 10) over the seed set [7, 17, 31, 53, 71, 83, 97, 113,
127, 149], where collapse is defined as: at the final evaluation (step 8000),
any of the first d_t=3 z_dyn dimensions has batch-std < 0.5 computed over 200
evaluation samples from a fresh PhysicsSandbox(N=3). Additionally, any
configuration where the mean training loss at step 8000 exceeds 100 (diverged)
or where the mean per-dimension z_dyn std at the final training log is < 0.1
(VICReg trivially satisfied / representation collapsed at training time) is
disqualified regardless of the evaluation collapse rate.

**Proposed Method:**
Step-by-step experimental protocol:

1. Create src/run_phase0_collapse_sweep.py based on run_phase0_id_probe_v2.py,
   stripped to only the JEPA+VICReg control arm (no supervised/contrastive).
   
2. Sweep four regime knobs one-at-a-time against the canonical baseline:

   Arm A0 (canonical repeat): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=32. Identical to iter_025 v2 Arm A.
     10 seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149].
   
   Arm A1 (batch_size=64): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=64, replay_buffer_capacity=4000.
     Same 10 seeds.
   
   Arm A2 (var_weight=50): lr=3e-4, var_weight=50, cov_weight=25,
     sim_weight=25, batch_size=32. Same 10 seeds.
   
   Arm A3 (sim_weight warm-up): lr=3e-4, var_weight=25, cov_weight=25,
     sim_weight ramped 0→25 over 1000 steps, batch_size=32. Same 10 seeds.
   
   Arm A4 (lr=1e-4): lr=1e-4, var_weight=25, cov_weight=25,
     sim_weight=25, batch_size=32. Same 10 seeds.

3. All arms share: gradient clipping max_norm=1.0, 8000 training steps,
   d_max=8, d_t=3, dyn_readout="centroid_gated", pos_encoding="none",
   CCR covariance mode (ccr_smooth_weight=10, ccr_spatial_weight=10),
   gdasr_log_only=True, Adam optimizer, replay_buffer pre-fill 200 transitions
   (doubled from 100 to ensure batch_size=64 is always achievable from step 1).

4. Evaluation at step 8000 only (no intermediate checkpoints to save compute):
   - Collapse check: per_dim_std < 0.5 on 200 eval samples from fresh env
   - VICReg health: per_dim_std, mean_abs_corr on eval samples
   - Training loss sanity: mean total loss, var_loss, sim_loss at final step
   - Centroid MSE (for reference only, NOT used for regime selection)
   - Semantic probes (for reference only, NOT used for regime selection)

5. Stop rule: Arms are evaluated in order A0, A1, A2, A3, A4. The first arm
   achieving ≤10% collapse rate (≤1/10 seeds collapsed) passes the gate and
   becomes the new canonical regime. If no arm clears, report the best achieved
   collapse rate as a measured null and do NOT declare a winner.

6. The ONLY dependent variable is collapse rate. delta_R2_color, centroid MSE,
   and other downstream metrics are recorded for reference but MUST NOT be used
   to pick a winning regime.

7. Hungarian matching remains the standing rule for any auxiliary semantic
   probe reporting (not relevant for collapse measurement but included for
   consistency).

Files to create/modify:
- src/run_phase0_collapse_sweep.py (NEW): collapse-rate sweep runner
- src/pre_registration.md (UPDATE): this plan

---

## Iteration 026 -> Planner [Strategic Guidance]

Manager's Note (iter_026 plan critique)

The plan is well-aligned with the prior pre-planning lock (collapse-elimination only, no new objectives, one-knob-at-a-time, single dependent variable). Three corrections are required before execution.

1. **Falsification-criterion mismatch with the standing collapse rule.** The pre-registration uses a single ad-hoc rule — `per_dim_std < 0.5` on 200 eval samples — and silently drops the project's existing `has_collapsed` criterion (`e_a_dim >= 0.1*e_a_all` AND `std_x_mean > 5.0`, augmented with the per-dim batch-std floor) mandated in Section 6 of the goal and re-affirmed in the prior hints. Redefining the collapse metric for the one iteration whose entire purpose is to measure collapse rate is precisely the gate-relaxation failure mode flagged in the previous pre-planning hint. Mandatory change: collapse must be declared if **either** the existing `has_collapsed` criterion fires **or** any of the active d_t z_dyn dimensions has batch-std below the pre-declared floor, and that floor must be justified (anchored to iter_025 v2 measurements) rather than introduced as a new number. Report both components per seed so the rate is reproducible under the original rule as well.

2. **The "VICReg trivially satisfied" guard at std < 0.1 is too lax and the divergence guard at loss > 100 is unmotivated — both need pre-declared values traceable to existing data.** The VICReg variance term targets std ≥ 1, so a "trivially satisfied" disqualifier set at std < 0.1 still admits configurations that have effectively given up on the variance constraint (anything in [0.1, ~0.7] is also a failed-to-train regime). Pre-register the sanity floor closer to the VICReg target itself (e.g. mean per-dim std at final step ≥ 0.5, with rationale) and pre-register the divergence threshold from observed training-loss scales in iter_023–025, not a round number. Otherwise iter_026 risks "passing" a configuration that didn't collapse only because it didn't learn.

3. **Two design issues that will compromise the one-knob-at-a-time claim if uncorrected.** (a) Arm A1 changes *two* things — batch_size 32→64 **and** replay_buffer_capacity 100→4000 — so any A1 effect cannot be attributed to batch size alone. Either hold replay capacity constant across all arms at a value that supports B=64 from step 1 (preferred), or add the matched A0' run at B=32 with the larger buffer to isolate. (b) The hypothesis pre-commits to B=64 as "most likely" via a 1/√B argument, but the sweep is sequential with an early-stop on the first pass; if A0 happens to pass at 1/10 by sampling luck, the substantive B-vs-VICReg-floor question is never tested. Pre-register the sequential stop rule with a robustness check: if any arm passes, also run the canonical A0 to completion under the new regime's seed set to confirm the pass replicates, and report the full collapse rate of every arm that was started rather than truncating outputs.

Procedural reminders: write the exact hypothesis, the corrected collapse definition (point 1), the corrected sanity floors (point 2), the corrected arm specifications (point 3), the seed list, the stop rule, and the explicit prohibition on using delta_R2_color or any downstream metric to pick a winner into `src/pre_registration.md` **before** any seeds run; the Orchestrator will commit that file. Language discipline: a measured null ("no swept configuration cleared the ≤10% gate under the corrected rule") is a first-class deliverable for this iteration — do not soften it in the report if it occurs. Avoid "stable", "solved", "fixed" in any write-up; prefer "is consistent with ≤10% collapse under the pre-registered protocol" only if both the collapse rule and the sanity floors are met.

---

