Update src/pre_registration.md with the three revisions required by the Research Manager for iter_025. Read the current file first, then rewrite it completely.

THREE REQUIRED REVISIONS:

1. MATCHING CONFOUND PROTOCOL:
The channel-to-object matching via sorted positions can fake or mask the ceiling result. When sort orders disagree (collisions, near-equal positions, swaps), the supervised target flips between objects, teaching z_dyn to encode position-rank rather than identity.

Add to pre-registration:
- All supervised arms (B, D) will run with TWO matching schemes:
  (a) Sorted-position matching (as planned): sort z_coord[:, :d_t] and info["positions"][:, :N] ascending → monotonic assignment
  (b) Oracle/Hungarian matching: use scipy.optimize.linear_sum_assignment with cost matrix C[d,o] = mean(|z_coord[:, d] - positions[:, o]|) per batch, which finds the optimal assignment minimizing total distance.
- Report the empirical mismatch rate between the two schemes on the evaluation set.
- If the two assignments disagree by more than 5% of samples on delta_R2_color outcome, the ceiling claim is conditional on the matching scheme and must be reported as such.
- For Arm C (ID-contrastive), also report mismatch rate and run with both schemes.

2. NOISE FLOOR AND SEPARATION CRITERIA:
The 0.10 delta_R2_color threshold was chosen against self-supervised objectives. Under supervised color probe (Arm B), the natural ceiling is likely well above 0.10, so passing 0.10 is weak evidence.

Add to pre-registration:
- Noise floor: Run 3 frozen-random-encoder baselines (fresh seeds [7, 17, 31]) with d_max=8, measuring delta_R2_color. This establishes the empirical floor. Expected: near 0 or slightly negative.
- Pre-declare:
  (i) Arm B must achieve delta_R2_color ≥ max(0.10, floor_mean + 0.08) to confirm H1 (architecture capacity). The 0.10 absolute threshold ensures the signal is non-trivial; the floor+0.08 ensures it clears the noise.
  (ii) Arm C must achieve delta_R2_color ≥ max(0.10, floor_mean + 0.08) with collapse rate ≤ 1/5 to confirm H2 (ID-contrastive viability).
  (iii) Arm D interpretation: If Arm D succeeds but Arm B fails, the result is attributed to channel capacity, not objective — consistent with iter_022-023 findings. If both B and D succeed, the supervised signal works at both capacities. Arm D alone succeeding is NOT evidence for H1.

3. LANGUAGE AND REPORTING HYGIENE:
- In the failure quadrant (B fails, C fails), the claim is: "consistent with an architecture-level bottleneck on identity encoding, conditional on the sorted-position matching scheme (mismatch rate: X%)" — NOT "the architecture cannot encode identity."
- A positive Arm B result is stated as: "is compatible with sufficient architectural capacity under direct supervision" — NOT "demonstrates the architecture can encode identity."
- A positive Arm C result is qualified as: "supervised (slot IDs are privileged information), not evidence that the decoder-free self-supervised problem is solved."
- If Arm A (JEPA control) on fresh seeds drifts materially from the iter_022-024 reference (e.g. > 0.03 absolute on delta_R2_color), the seed-batch is itself a confound and the cross-iteration comparison is suspended pending investigation — do not tune.

Also update the experimental design to include the noise floor runs (3 additional short runs before the main experiment).

Write the complete updated pre-registration to src/pre_registration.md.