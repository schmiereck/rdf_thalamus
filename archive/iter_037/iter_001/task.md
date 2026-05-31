You are working on the Thalamus project (curiosity-driven, decoder-free representation agent). Your task is TWO-PART:

**PART A: Update the pre-registration file at `src/pre_registration.md`**

The existing pre-registration has two gates for a 2D cheap-gate experiment. A research manager reviewed it and identified three issues that MUST be fixed before execution. Read the current `src/pre_registration.md`, then rewrite it incorporating these changes:

1. **Gate threshold calibration justification**: Add a short paragraph for EACH gate threshold explaining why the number remains the right cutoff in the 2D regime:
   - Gate-1 (≤3.0): In 1D, PASSIVE=12.27/object because all entities share the axis (collision cross-section = full arena width). In 2D (64×64, N=3, pointer radius=4), the effective collision probability per object trajectory is ~O(r_ptr/W)² ≈ (8/64)² ≈ 1.6% per axis crossing. The 3.0 threshold represents a collision rate where ORACLE targeting could produce ~10-20× more collisions than PASSIVE, providing headroom for behavioral discrimination. This is an operational threshold: sufficient (not necessary) for the PASSIVE ceiling to be low enough that targeted perception has headroom.
   - Gate-2 (CV ≥ 0.50): With probe_budget=20 and p=0.01 over 2000 steps, expected ~20 probes across 3 objects. Under pure Poisson allocation (mean ≈ 6.67/object), CV ≈ √(6.67)/6.67 ≈ 0.39 — BELOW the 0.50 threshold. So random Poisson noise alone would NOT trivially pass this gate. CV ≥ 0.50 requires genuine spatial clustering in the gaze trajectory (some objects visited more than others due to 2D random walk coverage geometry), which is an empirical question.

2. **Add Gate-1b (PASSIVE collision heterogeneity)**: Under PASSIVE, per-object collision counts should show CV ≥ 0.30, indicating that different objects naturally receive different collision rates based on their trajectories relative to the static pointer at center. In 1D, CV_passive ≈ 0 because all objects share the axis. In 2D, CV_passive > 0 is expected because objects at different positions/velocities have geometrically different encounter probabilities. The 0.30 threshold is a moderate bar: below 0.30, objects are too similar in passive collision rates for targeted action to create meaningful differentiation. This gate addresses the critique that Gate-1 is near-tautological — Gate-1 measures whether the PASSIVE ceiling is low enough, while Gate-1b measures whether there's heterogeneity that ORACLE can exploit.

3. **Per-seed decision rule**: For ALL gates (1, 1b, 2), the decision rule is: ≥4 out of 5 seeds must individually meet the threshold. The mean across seeds is also reported, but the per-seed rule is the binding criterion. This ensures robustness to initial conditions.

Also update the falsification criterion to include Gate-1b.

**PART B: Write the four-iteration null finding document at `archive/iter_037/results/null_finding_1d.md`**

Create the directory `archive/iter_037/results/` if it doesn't exist. Write a standalone, citable markdown document that crystallizes the environment-design null chain across iter_033–036:

- iter_033: behavioral-pivot metric saturation (ORACLE≈RANDOM, gap 0.0001)
- iter_034: free autonomous information (MALRE active-passive gap=0.83, but ORACLE-RANDOM gap=0.031, 3/8 seeds)
- iter_035: 1D collision inevitability (PASSIVE 12.27 per-object vs 3.0 ceiling gate — 4× overshoot)
- iter_036: coverage uniformity (RANDOM CV 0.36/0.46 vs 0.50 threshold — both arms fail)

The document must:
1. State the finding precisely: "the 1D × N=3 × 128px sandbox cannot make perception behaviorally load-bearing under an ORACLE-vs-RANDOM bracket across four mechanism-distinct redesigns."
2. NOT claim "no 1D environment could ever work" or make any claim about M2
3. Describe each iteration's specific mechanism and failure mode
4. Explain why the chain is a complete null: each iteration addressed the specific failure of the previous one, and the last (foveated gaze) was the most radical redesign, which still failed
5. Note that this is a clean negative result regardless of what comes next

Write both files. Do NOT modify any other source files.