Run the full iter_025 Architecture Ceiling Probe experiment and compile the analysis.

The code is already fixed and verified via dry run. The bugs (in-place operation, missing weight ramp) have been resolved. Now execute the full experiment.

## COMMAND TO RUN
```bash
cd /home/user && python src/run_phase0_id_probe.py --sequential
```

This will run:
- 3 noise floor runs (frozen encoder, 1000 steps each)  
- 20 main runs (4 arms × 5 seeds × 5000 steps)
Total: 23 runs, estimated ~45 minutes

If sequential is too slow, try:
```bash
cd /home/user && python src/run_phase0_id_probe.py --workers 4
```

## AFTER THE EXPERIMENT COMPLETES

1. Read the aggregated results from archive/iter_025/results/aggregated_phase0_id_probe.csv
2. Read individual run data from archive/iter_025/results/summary_phase0_id_probe.csv
3. Read the auto-generated analysis from archive/iter_025/results/analysis.md

4. Compile a comprehensive analysis including:
   a. Noise floor: mean delta_R2_color from the 3 frozen encoder runs → compute floor_mean
   b. Effective threshold: max(0.10, floor_mean + 0.08)
   c. Per-arm results (non-collapsed seeds only):
      - mean delta_R2_color under sorted matching
      - mean delta_R2_color under Hungarian matching (if available)
      - collapse rate (X/5)
      - mismatch rate between matching schemes
      - mean centroid MSE
      - mean tracking correlation
   d. Arm A drift check: compare against iter_022-024 reference (~0.05 delta_R2_color)
   e. Assign to four outcome quadrants:
      - B succeeds, C succeeds → H1+H2 confirmed
      - B succeeds, C fails → H1 confirmed, H2 refuted
      - B fails, C succeeds → check implementation bugs
      - B fails, C fails → architecture-level bottleneck (conditional on matching)
   f. Use the correct language per pre_registration.md:
      - Positive B: "compatible with sufficient architectural capacity under direct supervision"
      - Positive C: "supervised (slot IDs are privileged information)"
      - Negative B+C: "consistent with an architecture-level bottleneck on identity encoding, conditional on the sorted-position matching scheme (mismatch rate: X%)"

5. Write the analysis to archive/iter_025/results/final_analysis.md

## KEY REMINDERS
- The primary metric is delta_R2_color (frozen-encoder linear probe)
- Seeds: [7, 17, 31, 53, 71] for main, [7, 17, 31] for noise floor
- Do NOT modify the code unless there's a crash — the dry run passed
- If a run crashes, record it as failed and continue
- Report all raw data, not just aggregates