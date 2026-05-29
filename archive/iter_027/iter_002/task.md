
# Iteration 027 — Create Experiment Runner and Run Full Experiment

The pre-registration (`src/pre_registration.md`) and model code (`src/models_separate_dyn.py`) are already created. Your task is to create the experiment runner, validate it, and run the full 40-seed experiment.

## CRITICAL: Read Pre-Registration First

Read `src/pre_registration.md` before doing anything else. You MUST adhere to the pre-registered protocol. Do NOT retune the gate threshold, seed list, buffer size, or dual-collapse criterion mid-run.

## Step 1: Create `src/run_phase0_separate_dyn.py`

Create the experiment runner based on `src/run_phase0_collapse_sweep.py`. Adapt it with 4 arms and the separate-dyn model.

### Key Changes from run_phase0_collapse_sweep.py

1. Import both model classes:
```python
from src.models_dual_stream import NonParametricJEPASpatial
from src.models_separate_dyn import NonParametricJEPASpatialSeparateDyn
```

2. Arms configuration — exactly these 4 arms:

```python
ARMS = [
    {
        "name": "Aprime (shared, centroid_gated)",
        "model_type": "shared",
        "lr": 3e-4, "var_weight": 25.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 64,
        "d_max": 8, "d_t": 3, "dyn_readout": "centroid_gated",
        "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
        "mask_dyn_sim": False,
    },
    {
        "name": "A (shared, mean readout)",
        "model_type": "shared",
        "lr": 3e-4, "var_weight": 25.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 64,
        "d_max": 8, "d_t": 3, "dyn_readout": "mean",
        "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
        "mask_dyn_sim": False,
    },
    {
        "name": "B (separate, JEPA+VICReg)",
        "model_type": "separate",
        "lr": 3e-4, "var_weight": 25.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 64,
        "d_max": 8, "d_t": 3, "dyn_readout": "mean",
        "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
        "mask_dyn_sim": False,
    },
    {
        "name": "C (separate, VICReg-only on z_dyn)",
        "model_type": "separate",
        "lr": 3e-4, "var_weight": 25.0, "cov_weight": 25.0,
        "sim_weight": 25.0, "batch_size": 64,
        "d_max": 8, "d_t": 3, "dyn_readout": "mean",
        "pos_encoding": "none",
        "ccr_mode": "covariance", "ccr_smooth_weight": 10.0, "ccr_spatial_weight": 10.0,
        "replay_buffer_capacity": 4000,
        "mask_dyn_sim": True,
    },
]
```

3. Model creation in `run_single()`:
```python
if arm_config["model_type"] == "shared":
    model = NonParametricJEPASpatial(
        d_max=d_max, h=3, k=4, cooldown=300, stabilization_period=100,
        pos_encoding=pos_encoding, primary_objective="jepa",
        gdasr_log_only=True, dyn_readout=dyn_readout,
        sub_features=1, dyn_source="spatial",
    )
elif arm_config["model_type"] == "separate":
    model = NonParametricJEPASpatialSeparateDyn(
        d_max=d_max, h=3, k=4, cooldown=300, stabilization_period=100,
        pos_encoding=pos_encoding, primary_objective="jepa",
        gdasr_log_only=True, dyn_readout=dyn_readout,
        sub_features=1, dyn_source="spatial",
        mask_dyn_sim=arm_config.get("mask_dyn_sim", False),
    )
```

4. Parameter count logging: Before any training, compute and print parameter counts:
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```
Log parameter counts per arm in the results.

5. Save results to `archive/iter_027/results/` (not iter_026).

6. Keep ALL the evaluation, collapse check, semantic probe, and analysis functions from run_phase0_collapse_sweep.py — they work the same way.

7. The analysis generation (`_generate_analysis()`) must be updated to:
   - Use the 4 arm names: Aprime, A, B, C
   - Include parameter count comparison section
   - Include readout effect comparison (Aprime vs A)
   - Include pre-registered outcome classification (see below)
   - Include per-seed train-vs-eval std gap table (co-equal with collapse rates)
   - Use careful language ("consistent with", "does not refute")

### Pre-Registered Outcome Classification in Analysis

After the gate check, add a section "Pre-Registered Outcome Classification" with:

```
1. Arm B collapse rate: [X] → [classification]
2. Arm C collapse rate: [X] → [classification]

Outcome: [one of: POSITIVE CONSTRUCTIVE / SECOND NULL / SOFT NULL / ARM C FALSIFICATION]

Interpretation: [apply the pre-registered rules]
- If Arm B ≤10% and Arm C ≤10%: "Result is consistent with gradient decoupling but also consistent with added capacity. Not refuted, pending capacity control (iter_028)."
- If Arm B ≤10% and Arm C ≥20%: "Separate backbone with task objective reduces collapse; VICReg alone insufficient."
- If Arm B ≥20%: "Shared backbone is not the primary structural cause. Project pivots."
- If Arm B in (10%, 20%): "Soft null — same pivot as ≥20% (pre-committed)."
- If Arm C ≥20%: "ARM C FALSIFICATION — architectural change alone insufficient; project pivots."
```

### Train-vs-Eval Std Gap Table

Add a per-seed table (co-equal with collapse rates, not footnote):

```
| seed | arm | collapsed_eval | collapsed_train | collapsed | per_dim_std_eval | per_dim_std_train |
|------|-----|----------------|-----------------|-----------|------------------|-------------------|
| 7    | Aprime | ...          | ...             | ...       | ...              | ...               |
...
```

Include ALL 40 rows (4 arms × 10 seeds).

## Step 2: Dry-Run Validation

Run with `--dry-run` first to ensure:
- Both model classes instantiate correctly
- Forward pass works for all 4 arms
- Evaluation functions work
- No shape mismatches

Fix any errors before proceeding.

## Step 3: Full Experiment Run

Run the full experiment:
```bash
python src/run_phase0_separate_dyn.py --workers 8
```

If parallel workers cause issues, use `--sequential`.

Total: 4 arms × 10 seeds = 40 runs × 8000 steps.

## Step 4: Verify Results

After the experiment completes, verify:
1. All 40 runs completed (check the summary CSV has 40 rows)
2. Per-seed per_dim_std values are recorded for both train and eval
3. Parameter counts are logged
4. Analysis markdown is generated with all required sections
5. No sanity disqualifications (or if any, they're properly counted as collapsed)

## FILES TO CREATE
- src/run_phase0_separate_dyn.py (NEW)
- archive/iter_027/results/ (output directory, created by script)

## FILES TO READ
- src/pre_registration.md (MUST read first)
- src/models_separate_dyn.py (existing model code)
- src/run_phase0_collapse_sweep.py (reference for adaptation)
- src/models_dual_stream.py (for understanding model interface)

## IMPORTANT CONSTRAINTS
1. Do NOT retune the gate threshold, seed list, buffer size, or dual-collapse criterion mid-run
2. All 40 runs must complete — no early termination
3. The train-vs-eval std gap is a CO-EQUAL reporting requirement
4. Use careful language: "consistent with", "does not refute", "provides evidence for"
5. Results go to archive/iter_027/results/
