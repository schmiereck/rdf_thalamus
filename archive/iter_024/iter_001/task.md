Update the file src/pre_registration.md to incorporate three critical corrections from the Research Manager. Read the current file first, then rewrite it with these changes:

**CORRECTION 1 — Falsification criterion for Arm D must be recalibrated:**
The plan declared Arm D "PROMISING iff delta_R2_color ≥ 0.15" but the d_max=16 baseline (0.137) was a channel-capacity effect at d_max=16, while Arm D runs at d_max=8. The fair comparison is the d_max=8 baseline (delta_R2_color ≈ 0.05). REPLACE the Arm D criterion with:
"Arm D is consistent with a genuine objective-driven effect iff delta_R2_color ≥ 0.10 at d_max=8 AND exceeds the best d_max=8 multi-step SFA arm by ≥ 0.05 with non-overlapping seed CIs. Also, Arm D must pass a collapse gate: ≤ 2/5 collapsed seeds (matching the SFA arms). NT-Xent at τ=0.1 with VICReg simultaneously is a known fight, and a silently-collapsed Arm D would be misread as a null."

**CORRECTION 2 — Add invariance-vs-discrimination diagnostic for multi-step SFA:**
With k=100 and environments where objects exit/re-enter over that horizon, a representation encoding a batch-statistic (e.g., global color histogram, slowly drifting scene mean) would satisfy ||z_dyn(t) - z_dyn(t-k)||² at near-zero cost without encoding per-object identity. This is a constructional pass, not evidence for M2. ADD the following diagnostic requirement:
"Before any 'k=N works' claim, require an invariance-vs-discrimination diagnostic alongside delta_R2_color: report (a) within-trajectory z_dyn variance vs. between-trajectory z_dyn variance, and (b) whether the same z_dyn would pass delta_R2_color on a shuffled-frame control where the temporal label is destroyed. If shuffling does not collapse the probe, the signal was not in z_dyn-via-SFA, it was in the encoder geometry, and the result is constructional."

**CORRECTION 3 — Language and framing fixes:**
- Drop assertive language like "will produce identity encoding because..." — reframe as "we test whether... we predict that..., refuted if..."
- Drop the phrase "last slowness shot" from any committed artifact
- The honest framing is: this iteration tests multi-step SFA and temporal contrastive as two independent candidates; both may fail, and a clean double null is a successful iteration outcome that justifies pivoting to object-tracking-ID contrastive in iter_025
- Arm F at a single seed: explicitly state "n=1, indicative only, not evidence on its own"
- Early-step-2000 checkpoint kills the narrative, not the runs: all 5000 steps must complete so the dataset for the post-mortem stays intact

Make all these changes while preserving the rest of the document structure. The resulting file should be complete and self-consistent.