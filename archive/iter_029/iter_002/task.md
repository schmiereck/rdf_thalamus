Run the iter_029 SFA+VICReg separate-backbone experiment. This experiment has 3 arms × 20 seeds = 60 total runs.

Before running, READ the pre-registration file at `src/pre_registration.md` to understand the experimental design and adhere to the pre-registered criteria.

Execute the experiment:
```bash
cd /home/user && python -m src.run_phase0_sfa_separate_backbone --sequential 2>&1 | tail -100
```

Use `--sequential` to run one seed at a time (avoids GPU memory issues). Each run takes ~30-60 seconds at 8000 steps. Total should be ~30-60 minutes.

After all runs complete, the results will be saved to `archive/iter_029/results/`. Verify the summary CSV and analysis markdown were generated. Read the analysis file and report:
1. Per-arm collapse rate and ΔR²_color
2. Hard-seed outcomes (seeds 53, 71)
3. Original vs fresh seed bank comparison
4. F1/F2/F3 gate check outcomes
5. Pre-registered outcome classification

If any runs fail, report the failure count and which arm/seed combinations failed.

IMPORTANT: Do NOT modify any code. Just run the experiment and report results.