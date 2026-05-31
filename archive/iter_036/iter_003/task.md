## Task: Fix Critical Bugs in src/run_iter036_benchmark.py

Read `src/run_iter036_benchmark.py` and fix the following bugs. The file is 945 lines. Write the corrected version back to the same path.

### Bug 1 (CRITICAL): Δv is across-collision-only, not across-substep

The probe events currently record the immediate collision velocities (v_gaze before collision → v_gaze after collision). But per the Manager's directive, the mass estimator must use Δv = (post-step velocity) - (pre-step velocity), where:
- Pre-step velocities are recorded BEFORE the probe collision is applied AND before the substep loop
- Post-step velocities are recorded AFTER the full physics substep loop completes

This means the Δv includes confounding physics (wall bounces, obj-obj collisions in Arm A). The mass estimator is NOT a definitional identity because of this noise.

**Fix approach:**
- In FoveatedGazeSandbox.step(), the `probe_events` should record pre-step velocities (saved at the top of step(), before probe) and post-step velocities (saved after the full substep loop).
- Each probe event should be: `(step, obj_idx, v_gaze_pre_step, v_obj_pre_step, v_gaze_post_step, v_obj_post_step)`
  where v_gaze_pre_step = self.pointer_vel saved BEFORE probe application, v_obj_pre_step = self.velocities[obj_idx] saved BEFORE probe application, and v_gaze_post_step / v_obj_post_step are the values AFTER the full substep loop.
- The problem: at the time we apply the probe, we know pre-step velocities but don't yet know post-step velocities. So we need to record pre-step info now and fill in post-step later.

**Implementation:** Instead of recording the full event in the probe application, record a partial event with the pre-step info and the obj_idx. After the substep loop completes, fill in the post-step velocities.

Replace the probe event recording in the probe application with:
```python
# Record partial probe event (pre-step info)
self._pending_probe_events.append((step_num, best_idx, float(v_gaze), float(v_obj)))
```

Then after the substep loop, before returning:
```python
# Finalize probe events with post-step velocities
for (step_num, obj_idx, v_gaze_pre, v_obj_pre) in self._pending_probe_events:
    self.probe_events.append((
        step_num, obj_idx,
        v_gaze_pre, v_obj_pre,
        float(self.pointer_vel), float(self.velocities[obj_idx])
    ))
self._pending_probe_events = []
```

And in reset(), add: `self._pending_probe_events = []`

### Bug 2: CV gate uses wrong metric

The CV gate currently computes CV of POMLRE values across seeds. The pre-registration says: compute per-object probe-event counts under RANDOM, then CV = std(counts)/mean(counts) across the 3 objects, averaged over seeds.

**Fix:** Change the CV gate to:
1. For each seed, get the per-object valid probe event counts (per_obj_n_valid) from the RANDOM run.
2. For each seed, compute CV = std(per_obj_n_valid) / mean(per_obj_n_valid) across the 3 objects.
3. Average CV across seeds.
4. Gate: CV ≥ 0.5 (coverage is sufficiently uneven) AND mean per-object count ≥ 0.5 (RANDOM gets at least some events).

Also change `gate_pass[arm] = cv >= 0.5` (note: the current code has `cv < 0.5` which is inverted — CV should be HIGH, meaning RANDOM coverage is uneven, for ORACLE to improve).

### Bug 3: Sanity checks don't match pre-registration

Replace the current sanity checks with the pre-registered S1-S6:

S1: ORACLE achieves ≥3 probe-induced collision events per object (mean across seeds)
S2: ORACLE probe success rate ≥ 60% (≥12 of 20 probe attempts result in a collision event — an object was found within the gaze window)
S3: ≥80% of ORACLE's probe-induced collision events have |Δv_obj| > 1.0 (informative velocity change for mass estimation)
S4: No single object receives >80% of ORACLE's total probe events
S5: ORACLE gaze stays in bounds ≥95% of steps
S6: Each of the 3 objects receives ≥10% of ORACLE's total probe events

For S2: need to track probe_attempts (times do_probe=True) vs probe_hits (times a probe actually found an object within gaze_radius). These are different from what's currently tracked.

For S3: compute |Δv_obj| = |v_obj_post_step - v_obj_pre_step| for each probe event. Check that ≥80% of these have |Δv_obj| > 1.0.

For S4 and S6: use per_obj_probe_counts from the ORACLE runs.

### Bug 4: ORACLE condition doesn't check gaze radius

The ORACLE condition currently probes when |error| ≤ 6.0. The pre-registration says: probe when the target's center is within GAZE_RADIUS=8 of the gaze center AND |error|≤6.0. Since |error| ≤ 6.0 already implies the target is within gaze_radius=8, this is technically fine (6 < 8), but to be explicit and match the pre-registration, add the gaze radius check too. Keep both checks.

### Bug 5: ORACLE probe_attempts tracking

In the current code, `probe_attempts` is incremented whenever |error| ≤ PROBE_ERROR_THRESH and budget > 0. But for S2, we need to track the number of times the probe action was actually sent (action['probe']=True), regardless of whether the gaze was near an object. Then `probe_hits` = number of times the probe found an object. Currently, the code tracks `probe_hits` as the number of new probe events from env, which counts successful probes. This is correct for hits. But `probe_attempts` should be the number of times action['probe']=True was sent.

Actually looking more carefully, the current code has:
```python
if abs(error) <= PROBE_ERROR_THRESH and env.probe_budget > 0:
    do_probe = True
    probe_attempts += 1
```

This counts attempts only when |error| ≤ threshold. But the probe might still miss (no object within gaze_radius). A "probe attempt" should be counted every time do_probe=True. And a "hit" is when the probe actually collides with an object. The current probe_hits tracking via `per_obj_probe_counts[e[1]] += 1` counts successful probes. So S2 = (sum of hits across seeds) / (sum of attempts across seeds) ≥ 0.60.

Actually wait - re-reading the pre-registration: "S2: ORACLE probe success rate ≥ 60% (≥12 of 20 probe attempts result in a collision event — an object was found within the gaze window)." This means we need to count how many of the 20 probe budget slots were actually used (attempts), and how many found an object (hits). The current code seems to count attempts correctly. But we need to make sure the hit counting is based on whether an object was within gaze_radius when the probe was fired, which is exactly what the env.probe_events records.

### Summary of changes needed:

1. Change probe event recording to use pre-step (before probe+substep) and post-step (after full physics) velocities
2. Fix CV gate: compute CV of per-object probe-event counts, require CV ≥ 0.5 and mean ≥ 0.5
3. Replace sanity checks with pre-registered S1-S6
4. Minor: add gaze_radius check to ORACLE probe trigger (already implied by |error|≤6.0 < 8)
5. Ensure probe_attempts/hits tracking matches S2 definition

Make these changes and write the corrected file. Do NOT run it.