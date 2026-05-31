Update the file src/pre_registration.md to incorporate three critical additions required by the Research Manager. The existing content is correct in structure; these are additive corrections only.

**Addition 1 — Exact M2 sentence (insert after the existing Section 3 "Proposed Method", as a new section "## 4. M2 Wording Lock")**

The report must use the following sentence verbatim for M2 status:
"UNTESTABLE-NOT-FALSIFIED within project scope; SFA on z_dyn behavioural improvement was never tested because no valid behavioural bracket was achievable under any tested environment design."
This exact wording is locked and must not be paraphrased, upgraded, or downgraded in the final report.

**Addition 2 — Language discipline guard (insert as a new section "## 5. Language Discipline Guard")**

The final report must use the following register exclusively:
- "is consistent with" (not "proves" or "demonstrates")
- "we observe under conditions Y" (not "we have shown")
- "the evidence supports" (not "this confirms")
- "the mechanism is hypothesised; the ablation was not run within scope" (not "the mechanism drives/prevents")
- Flag any quantitative claim whose backing iteration metric cannot be located rather than inventing or rounding.

Specific mandates:
- Claim-E must NOT say "separate backbone prevents collapse" without the qualifier "in the tested configurations (iter_028, 0/5 seeds collapsed); the causal mechanism is hypothesised and the isolating ablation was not run."
- Claim-F must NOT say "mean-pool is the bottleneck" without "mean-pool is the binding bottleneck in the tested readouts; the localized-readout constructive test was not cleanly executed due to the iter_032 cross-backbone collapse artefact."

**Addition 3 — Construction-vs-empirical check on headline (insert as a new section "## 6. Construction Check on MIGRATING-OBSTRUCTION")**

The report must state explicitly that each ceiling gate was independently designed to detect a different structural failure mode (saturation, leak, inevitability, opposition, non-reproducibility). The migration is therefore NOT built-in by a single shared metric — the five obstruction classes were not predicted in advance by one underlying statistic, but emerged from five mechanistically different gate primitives applied to five mechanistically different designs. This must appear in the Methodology section of the final report to forestall the objection that "migration" is just a re-description of "we changed the gate each time."

Renumber the existing "Falsification Criterion" section as Section 2 (it is already Section 2). Renumber subsequent sections accordingly so Sections 4, 5, 6 follow Section 3.

Do NOT modify the existing content in Sections 1-3. Only ADD the three new sections.