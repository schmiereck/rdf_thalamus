# Research Manager Log - Iteration 030

## Iteration 030 -> Planner [Pre-Planning Hints]

# Manager's Pre-Planning Hints — Iteration 030

**Hint 1 (directional, top priority — reframe before sweeping more objectives).** The user's reframing is correct and binds: ten iterations have optimized ΔR²_color ≥ 0.30 as a proxy whose necessity for the actual project goal (curiosity-driven agent with thalamic gating + motor) was never validated. The next iteration's **Arm 1 must be an integration smoke-test**, not another representation sweep. Freeze the best current representation (separate-backbone iter_027-ArmC topology, d_max=16, d_t=3, on non-hard seeds at ΔR²_color ≈ 0.25–0.27 and 0% collapse) and run it through Pillars D + E (CLTSMotorController attention routing + motor probing). Pre-declare a concrete behavioral pass/fail before running: e.g. centroid tracking error below an explicit threshold, post-collision probe latency, and a mass-change causal-sensitivity delta against a frozen-attention control. If downstream behavior is acceptable at ΔR² ≈ 0.25, the 0.30 threshold was a distraction and the project advances to Phase 2/3 integration; if it genuinely degrades, the objective hunt is justified and Arm 2 (contrastive vs. variance-ramped SFA, per the journal's iter_030 plan) earns the next slot.

**Hint 2 (scientific discipline — guard against the proxy-metric trap and the construction caveat).** Arm 1 is itself a falsifiable claim ("ΔR² ≈ 0.25 representations are sufficient for downstream gating/motor") and must be pre-registered as such, with the integration test's behavioral thresholds declared **before** the freeze-and-run, not chosen post-hoc to validate whichever direction the data points. Equally important: the construction-vs-empirical caveat applies here too — do not report "the agent tracks" or "the agent probes" as evidence unless those behaviors are measured against a frozen-random-attention control on the *same* environment seeds, since reflexive PD tracking will produce a low centroid error even with a useless representation. The same variance-stability discipline iter_029 established applies: ≥5 seeds, report σ alongside mean, and flag hard seeds 53/71 separately rather than averaging them away.

**Hint 3 (scope and preservation — keep Arm 2 in scope but subordinated).** If iter_030 runs both arms, Arm 1 (integration) is decisive and Arm 2 (contrastive D3 vs. variance-ramped SFA D2 on the union seed bank with hard seeds) is contingent — its result only matters if Arm 1 says identity encoding genuinely bottlenecks downstream behavior. Preserve: separate backbone + the iter_028 collapse-avoiding config (both interventions required for 0% collapse), d_t=3 frozen with GDASR log-only (M3), pooled batch VICReg (M1), buffer=4000, no positional encoding (consistently hurts across JEPA *and* SFA — iter_013 and iter_021 are now a cross-objective regularity, not a single-iteration finding). Whatever the verdict, the iteration's report must explicitly state whether M2 in goal.md remains, is salvaged via a variance-ramp protocol, is replaced by contrastive identity binding, or is **demoted to "not a bottleneck for the project goal"** — that mandate-level decision is the iteration's deliverable, not another delta on the proxy.

---

## Iteration 030 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
ARM 1 (Integration Smoke-Test): A representation with ΔR²_color ≈ 0.27 (SFA+VICReg
on separate backbone, 0% collapse, iter_029 Arm B) produces CLTS motor behavior
that is functionally adequate for Pillar D (thalamic gating) and Pillar E (motor
probing). Specifically, three pre-registered gates will be tested:
  G1 (Tracking Functionality): CLTS mean |pointer_pos - attended_centroid| < 20 pixels
    over a 2000-step closed-loop evaluation, demonstrating that the z_coord centroid
    readout supports functional PD tracking.
  G2 (Attention Validity): CLTS surprise-driven attention switches to the collision-
    involved channel within 15 steps after a detected collision in >50% of events,
    while CLTS-Random (uniform random attention selection) switches to the collision
    channel in <35% of events (chance baseline for 3 objects). A difference >15pp
    confirms the surprise signal is informative.
  G3 (Causal Sensitivity): After a forced-probe mass perturbation (mass×3 + push on
    object 0 at step 1000), CLTS attention switches to object 0's channel within
    20 steps in >60% of perturbation events, while CLTS-Random does so in <33%.
    A difference >15pp confirms the representation supports causal probing.
If ≥2 of 3 gates pass, the representation is sufficient for downstream tasks and the
project advances to Phase 2/3 integration without further objective optimization.
If 0-1 gates pass, the representation is insufficient and the objective hunt is
justified.

ARM 2 (M2 Decisive Test): Temporal identity contrastive binding (D1) or variance-
ramped SFA (D2) achieves mean ΔR²_color ≥ 0.30 on a 30-seed union bank with
variance-stability (std/mean < 2.0). D1 uses same-object-across-time as positive
pairs and other-objects-as-negatives in an NT-Xent loss; D2 anneals sfa_weight
from 0 to 5.0 over the first 4000 training steps. If D1 passes, M2 is revised to
"contrastive identity binding as primary" (slowness underspecifies identity). If D2
passes, M2 is revised to "variance-ramped SFA as primary." If neither passes, the
ΔR²_color proxy is acknowledged as unachievable by any tested decoder-free objective
and ARM 1's verdict governs the project direction.

**Proposed Falsification Criterion:**
ARM 1 falsified if ≤1 of 3 gates pass, meaning the current representation does not
support meaningful downstream behavior and further objective optimization is justified.
Specifically:
  G1 falsified: CLTS mean tracking error ≥ 20 pixels (centroid readout is too noisy
    for functional tracking)
  G2 falsified: CLTS collision-switching rate ≤ CLTS-Random + 15pp (surprise signal
    is not informative for attention routing)
  G3 falsified: CLTS post-perturbation attention switch ≤ CLTS-Random + 15pp
    (representation does not support causal probing after mass change)

ARM 2 D1 falsified: mean ΔR²_color < 0.30 OR std/mean ≥ 2.0 on the 30-seed bank
ARM 2 D2 falsified: mean ΔR²_color < 0.30 OR std/mean ≥ 2.0 on the 30-seed bank
Both D1 and D2 falsified: the ΔR²_color ≥ 0.30 target is unachievable by any
decoder-free objective tested; the project accepts the current representation quality
per ARM 1's verdict.

**Proposed Method:**
ARM 1 (Integration Smoke-Test — highest priority, runs first):
1. Load pre-trained SFA+VICReg checkpoints from iter_029 Arm B (10 seeds: 7, 17, 31,
   83, 97, 113, 127, 149 + hard seeds 53, 71 reported separately).
2. For each seed, run two 2000-step closed-loop evaluations:
   a. CLTS: CLTSMotorController with surprise-driven attention + PD tracking +
      push-when-surprised-and-near (existing probe-when-surprised logic)
   b. CLTS-Rand: Same controller but token_locus selected uniformly at random
      from d_t channels (replacing argmax of normalized surprise), keeping PD
      tracking and push logic identical — controls for reflexive PD tracking
3. Warm-up: first 200 steps with CLTS but no attention switching (fixed locus 0)
   to let EMA surprise statistics converge; metrics start at step 200.
4. Collision detection: at each step, compare object velocities before/after; flag
   collision if |delta_v| > 2.0 for any pair of adjacent objects. Record whether
   CLTS attention switches to a collision-involved channel within 15 steps.
5. Mass perturbation: at step 1000, change object-0 mass to 3× current; immediately
   force pointer near object 0 (set pointer_pos = object_0_pos ± 5) and push.
   Measure whether attention switches to object 0's channel within 20 steps.
6. Compute gate metrics and paired comparison (CLTS vs CLTS-Rand per seed).
7. Decision: ≥2/3 gates pass → representation sufficient → project advances.

Files: src/run_phase0_integration.py (NEW)

ARM 2 (M2 Decisive Test — contingent on ARM 1):
1. Expand union seed bank to 30 seeds (original 10 + fresh 10 + new: 173, 179,
   181, 191, 193, 197, 199, 211, 223, 227).
2. D1 (Temporal Identity Contrastive):
   - New loss: NT-Xent where same-object-at-consecutive-timesteps = positive pair,
     different-objects = negatives.
   - Implementation: encode x_target and x_hist[:,-1] to get z_coord_t, z_dyn_t,
     z_coord_{t-1}, z_dyn_{t-1}. Sort z_coord and positions to match channels to
     objects (sorted matching, O(d_t log d_t) on GPU). For each object, z_dyn at
     the matched channel at time t and t-1 form a positive pair; z_dyn from other
     objects at t-1 are negatives. Uses existing id_contrastive_proj projection head.
   - Loss computed in training loop (not in model forward), added to total loss with
     weight temporal_contrastive_weight=5.0.
   - Architecture: NonParametricJEPASpatialSeparateDyn, primary_objective="sfa",
     mask_dyn_sim=True, coord_vicreg=True, sfa_weight=0 (SFA disabled; replaced
     by temporal contrastive). d_t=3, d_max=8.
3. D2 (Variance-Ramped SFA):
   - Same architecture as D1 but primary_objective="sfa" with sfa_weight
     linearly ramped from 0 to 5.0 over steps 0-4000, then constant at 5.0.
   - Ramping prevents early-training SFA instability (gradient conflict with
     VICReg variance term during representation bootstrapping).
4. Train 2 arms × 30 seeds for 8000 steps each (separate backbone, mask_dyn_sim=True,
   coord_vicreg=True, d_t=3, d_max=8, buffer=4000, batch_size=32, lr=3e-4).
5. Evaluate: ΔR²_color, collapse rate, centroid MSE (same protocol as iter_029).
6. Gate: mean ΔR²_color ≥ 0.30 AND variance-stability std/mean < 2.0.
7. Decision: D1 passes → M2 revised to contrastive identity binding; D2 passes →
   M2 revised to variance-ramped SFA; neither passes → proxy acknowledged as
   unachievable, ARM 1's verdict governs.

Files: src/run_phase0_m2_decisive.py (NEW)

PRESERVE across both arms:
- Separate backbone + collapse-avoiding config (mask_dyn_sim=True, coord_vicreg=True)
- d_t=3 frozen, GDASR log-only (M3)
- Pooled batch VICReg (M1)
- No positional encoding (cross-objective regularity from iter_013 and iter_021)
- Hard seeds 53/71 reported separately, not averaged away
- Report σ alongside mean for all metrics (≥5 seeds minimum per condition)

---

## Iteration 030 -> Planner [Strategic Guidance]

# Manager's Note — Iteration 030 (Plan Critique)

**Endorsed direction, with three load-bearing corrections.** The two-arm structure (integration smoke-test first, M2 disambiguation second) correctly implements the reframe. But the Arm 1 protocol as drafted has a confound that would make any positive verdict uninterpretable, and Arm 2 quietly inflates statistical power. Fix these before pre-registration is committed.

**1 (Arm 1 — kill the confound, or the smoke-test proves nothing).** Loading checkpoints *only* from iter_029 Arm B (SFA+VICReg) means Arm 1 cannot answer the question it was designed to ask. If the gates pass, you do not know whether the representation needed ΔR²≈0.27 or whether ΔR²≈0.04 (VICReg-only) would have sufficed — and the user hint's decision rule ("if downstream performs acceptably at dR2~0.25 then the 0.30 threshold was a distraction") requires a *contrast* against a weaker representation. **Mandatory addition:** a third frozen condition, **CLTS on VICReg-only z_dyn checkpoints** (iter_027 Arm C / iter_029 Arm A, ΔR²≈0.04–0.18, same seeds). The gate hierarchy then becomes informative: (a) both SFA-B and VICReg-only pass → identity decodability does not bottleneck downstream behavior, M2 demoted; (b) SFA-B passes, VICReg-only fails → identity encoding *does* matter and the 0.30 search is justified; (c) neither passes → representation truly insufficient; (d) VICReg-only passes but SFA-B doesn't → something else is going on, investigate. Without this contrast, Arm 1 is a one-armed test of "does any frozen model do anything," not the decision the user asked for.

**2 (Arm 1 — gates G2/G3 are partly constructional; tighten the controls).** The "+15pp over CLTS-Random" bar treats random attention selection as the only baseline, but random attention on a controller that still does PD tracking and collision-correlated push has known structural advantages from the motor side — e.g. the pointer drifts toward whichever channel it's currently attending, which after a collision is more likely to be near a moving object regardless of representation quality. Two specific fixes: (a) Add a **frozen-attention control** (token_locus held at channel 0 throughout) alongside CLTS-Random — this isolates the contribution of *switching* from the contribution of *tracking*; G2/G3 must beat **both** baselines by 15pp, not just random. (b) The 15-step / 20-step / >50% / >60% / 15pp thresholds are currently unjustified numbers — state explicitly what each threshold corresponds to in expected events per seed and what the binomial null distribution looks like for n_seeds=10, or the "≥2 of 3 gates" rule is just three independent coin flips with unknown p. Without that, a 2/3 pass is not evidence — it is one favorable noise realization.

**3 (Arm 2 — variance-stability gate and language hygiene).** The `std/mean < 2.0` criterion is too lax for the stated context: iter_029 Arm B had σ=0.577 over a mean of 0.27, i.e. std/mean ≈ 2.1, which would **almost pass** under the proposed gate. That re-admits exactly the instability the journal already flagged as disqualifying. Replace with the journal's prior, tighter formulation: **lower bound of the 95% CI must also exceed 0.18** (the iter_027 Arm C anchor). This is a meaningful gate; std/mean<2.0 is not. Separately, the plan's hypothesis text contains "demonstrating that the z_coord centroid readout supports functional PD tracking" — restate as "consistent with functional PD tracking" or "does not refute"; PD tracking will produce *some* low error regardless of representation, so "demonstrating" overclaims (and is exactly the construction-vs-empirical trap point 2 above is meant to catch).

**Pre-registration mandate (process reminder).** The Orchestrator will automatically write `src/pre_registration.md` from the hypothesis and falsification fields above and commit it before execution; the Planner's sub-agents MUST read it and adhere strictly. The plan as currently drafted is **not yet allowed to proceed to execution** — the three corrections above must be incorporated into the pre-registered hypothesis and falsification criteria first. Specifically: Arm 1's frozen conditions must list all three (SFA-B / VICReg-only / frozen-attention), Arm 1's gate text must specify the "beat both baselines by 15pp" rule with a stated event-count null, and Arm 2's variance gate must read "lower 95% CI ≥ 0.18" not "std/mean < 2.0". After those edits, the plan is approvable.

---

