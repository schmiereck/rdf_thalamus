
# Iteration 027 — Separate z_dyn Encoder Architectural Probe

You are implementing a pre-registered scientific experiment to test whether the shared CNN backbone is the primary structural cause of z_dyn collapse in the NonParametricJEPASpatial encoder.

## CRITICAL: Pre-Registration Discipline

You MUST read `src/pre_registration.md` FIRST, then UPDATE it with the three mandatory Manager amendments described below. After updating, ALL experiment code and execution must follow the pre-registered protocol exactly. You must NOT retune the gate threshold, seed list, buffer size, or dual-collapse criterion mid-run.

## Step 0: Update Pre-Registration with Manager Amendments

Update `src/pre_registration.md` to incorporate these three amendments:

### Amendment 1 — Readout Swap Anchor Fix
The plan shifts `dyn_readout` from `centroid_gated` (iter_026 A1, the ~30% anchor) to `mean` (Arm A). Add a FOURTH arm — Arm A′ (shared backbone, `dyn_readout="centroid_gated"`, otherwise identical to A) — as the true iter_026 anchor. This creates the controlled chain: A′ → A → B. Update the hypothesis to state that C_sep ≤ 0.10 is measured **against Arm A in this iteration**, not against the iter_026 anchor, and the iter_026 number is informal context only.

### Amendment 2 — Parameter Count as Alternative Explanation
Pre-register the exact parameter count per arm (computed and logged BEFORE runs start). Pre-declare this interpretive rule: if Arm B ≤10% AND Arm C ≤10%, the result is *consistent with* gradient decoupling but *also consistent with* added capacity. The report must NOT claim "the shared-backbone hypothesis is confirmed" in that case — only "is not refuted, pending capacity control." A capacity-matched shared-backbone control (e.g. widened conv channels) would be the mandatory iter_028 follow-up.

### Amendment 3 — Language and Falsification Discipline
- Reword the hypothesis paragraph: "We conjecture that shared-parameter gradient competition contributes to collapse; the experiment tests the observable consequence C_sep ≤ 0.10, not the mechanism directly. A successful collapse-rate change is **consistent with** the mechanism, not a demonstration of it."
- The AMBIGUOUS MIDDLE band (10%, 20%) pre-committed default action: treat as a **soft null** that triggers the same pivot as ≥20%. The project does not iterate on this variable in the middle band.
- Sub-agents must read the pre-registration and refuse to retune the gate, seed list, buffer size, or dual-collapse threshold mid-run. State this explicitly.
- Throughout the report, use "is consistent with / does not refute / provides evidence for"; do NOT use "proves," "demonstrates," "stabilizes," or "resolves" without the capacity control.

## Step 1: Create `src/models_separate_dyn.py`

Create a new file `src/models_separate_dyn.py` with two classes:

### SeparateDynEncoder(nn.Module)
An encoder with TWO independent CNN backbones:

(a) **Coord backbone**: conv1→conv2→conv3→conv4→conv_spatial (IDENTICAL architecture to NonParametricEncoder in models_dual_stream.py). Produces z_coord via soft-argmax centroid from the spatial feature map.

(b) **Dyn backbone**: conv1_dyn→conv2_dyn→conv3_dyn→conv4_dyn→conv_identity_dyn (SAME architecture as coord backbone, SEPARATE parameters). Produces z_dyn via mean pooling over the spatial dimension.

Both backbones process the SAME RGB input independently. Each backbone adds positional encoding independently (using the same `add_positional_encoding` function imported from `models_dual_stream`).

Interface requirements (must match NonParametricEncoder):
- `forward(x)` → `(z_coord, z_dyn)` where z_coord is (B, d_max) and z_dyn is (B, d_max)
- `forward_spatial(x)` → spatial feature map (B, d_max, 128) from coord backbone only
- `d_dyn` property → returns `d_max * sub_features`
- Constructor: `__init__(self, d_max=8, pos_encoding="none", dyn_readout="mean", sub_features=1, dyn_source="spatial")`
  - Note: dyn_readout and sub_features are accepted for interface compatibility but the SeparateDynEncoder always uses mean readout on the dyn backbone.

Implementation detail for the dyn stream:
```python
dyn_features = self._forward_dyn_backbone(x)  # (B, 128, 8)
a_dyn = self.conv_identity_dyn(dyn_features)   # (B, d_max, 8)
z_dyn = a_dyn.mean(dim=-1)                     # (B, d_max)
```

The conv layers in each backbone are:
- conv1/conv1_dyn: Conv1d(in_channels, 16, k=5, s=2, p=2)
- conv2/conv2_dyn: Conv1d(16, 32, k=5, s=2, p=2)
- conv3/conv3_dyn: Conv1d(32, 64, k=5, s=2, p=2)
- conv4/conv4_dyn: Conv1d(64, 128, k=5, s=2, p=2)
- conv_spatial: Conv1d(128, d_max, k=1)
- conv_identity_dyn: Conv1d(128, d_max, k=1)

Where in_channels depends on pos_encoding: 3 for "none", 4 for "linear", 7 for "sinusoidal".

Use `calculate_centroid_and_variance` and `add_positional_encoding` imported from `src.models_dual_stream`.

### NonParametricJEPASpatialSeparateDyn(NonParametricJEPASpatial)
Inherits from NonParametricJEPASpatial. Modifications:
- Constructor adds `mask_dyn_sim=False` parameter
- In `__init__`, after calling `super().__init__(...)`, replace `self.encoder` with `SeparateDynEncoder(...)` and recreate `self.predictor = DualStreamPredictor(d_max=d_max, d_dyn=self.encoder.d_dyn, h=h)`
- Also recreate the probe heads (color_probe_weight, color_probe_bias, id_contrastive_proj) since they need to be on the same device
- Override `forward()` to handle `mask_dyn_sim`: after calling `super().forward(...)`, if `self.mask_dyn_sim` is True, subtract `sim_weight * sim_loss_dyn` from the loss (keeping sim_loss_dyn in the dict for logging)
- Override `clone()` to handle the new class

The mask_dyn_sim implementation in forward():
```python
def forward(self, x_hist, x_target, sim_weight=25.0, ...):
    result = super().forward(x_hist, x_target, sim_weight=sim_weight, ...)
    loss_dict = result[0]
    if self.mask_dyn_sim:
        # Remove sim_loss_dyn from total loss (keep for logging)
        loss_dict["loss"] = loss_dict["loss"] - sim_weight * loss_dict["sim_loss_dyn"]
    return result
```

NOTE: Be careful with the `sim_weight` parameter — it's passed to the forward method, so use it in the subtraction.

## Step 2: Create `src/run_phase0_separate_dyn.py`

Based on `src/run_phase0_collapse_sweep.py`, but with 4 arms and the separate-dyn model. Key changes:

### Arms Configuration

COMMON TO ALL ARMS:
- d_max=8, d_t=3, N=3
- pos_encoding="none"
- primary_objective="jepa", ccr_mode="covariance"
- ccr_smooth_weight=10, ccr_spatial_weight=10
- gdasr_log_only=True
- lr=3e-4, gradient clipping max_norm=1.0
- batch_size=64 (best from iter_026)
- replay_buffer_capacity=4000
- 8000 training steps, Adam optimizer
- 10 seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
- var_weight=25, cov_weight=25, sim_weight=25

ARM A′ (iter_026 anchor, shared backbone, centroid_gated):
  - Model: NonParametricJEPASpatial(dyn_readout="centroid_gated")
  - This exactly matches iter_026 A1 configuration
  - mask_dyn_sim is not applicable (shared backbone model)

ARM A (reference, shared backbone, mean readout):
  - Model: NonParametricJEPASpatial(dyn_readout="mean")
  - This re-anchors the shared-backbone collapse rate under mean readout

ARM B (experimental, separate backbone + JEPA+VICReg):
  - Model: NonParametricJEPASpatialSeparateDyn(dyn_readout="mean", mask_dyn_sim=False)
  - The independent dyn_backbone receives z_dyn's VICReg gradient without competition from z_coord's JEPA gradient

ARM C (separate-encoder control, no z_dyn task objective):
  - Model: NonParametricJEPASpatialSeparateDyn(dyn_readout="mean", mask_dyn_sim=True)
  - VICReg variance/covariance on both z_coord and z_dyn
  - JEPA sim_loss_coord active, sim_loss_dyn ZEROED OUT (does not shape representation)
  - Tests whether separate backbone alone suffices without a task objective for z_dyn

### Parameter Count Logging
Before any training, compute and log the parameter count for each arm. Use:
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Print parameter counts at the start and include them in the final analysis.

### Training Loop
Same as run_phase0_collapse_sweep.py but:
- The model creation depends on the arm:
  - For A-prime and A: use NonParametricJEPASpatial with the appropriate dyn_readout
  - For B and C: use NonParametricJEPASpatialSeparateDyn with mask_dyn_sim parameter
- Import both model classes

### Evaluation
Same protocol as iter_026:
- Collapse check (eval): per_dim_std < 0.5 on 200 eval samples from fresh env
- Collapse check (train): per_dim_std at step 8000 from training log
- Dual criterion: collapsed = collapsed_eval OR collapsed_train
- Train-vs-eval std gap: report per_seed_train_std and per_seed_eval_std side-by-side for EVERY seed in EVERY arm (co-equal reporting, not footnote)
- VICReg health: per_dim_std, mean_abs_corr on eval samples
- Training loss sanity: mean total loss, var_loss, sim_loss at final step
- Centroid MSE (reference only, NOT used for arm selection)
- Semantic probes: delta_R2_color, delta_R2_identity (reference only)
- Hungarian-primary matching for semantic probes
- Sanity disqualification: final_train_loss > 50 → counted as collapsed

### STOP RULE
ALL arms complete their full 10-seed runs. No early termination even if one arm passes the ≤10% gate.

### DEPENDENT VARIABLE
The ONLY dependent variable for the pre-registered gate is collapse rate (dual criterion). Centroid MSE, delta_R2_color, and other downstream metrics are recorded for diagnostic reference but MUST NOT be used to select a winning arm.

### Output
Save all results to `archive/iter_027/results/`:
- Per-seed CSV/JSON with all metrics including train and eval per_dim_std
- Training logs per seed
- Checkpoints per seed
- Final analysis markdown with:
  (a) Per-arm collapse rates under dual, eval-only, train-only criteria
  (b) Per-seed train-vs-eval std gap table (co-equal with collapse rates)
  (c) Gate status per arm
  (d) Parameter count comparison
  (e) Pre-registered outcome classification

## Step 3: Dry-Run Validation

Run the experiment script with `--dry-run` flag first to validate that:
- Both model classes can be instantiated
- The forward pass works for all arms
- The evaluation functions work
- No import errors or shape mismatches

Fix any errors before proceeding.

## Step 4: Full Experiment Run

Run the full experiment:
```
cd /path/to/project
python src/run_phase0_separate_dyn.py --workers 8
```

If `--workers` causes issues, use `--sequential`.

Total runs: 4 arms × 10 seeds = 40 runs × 8000 steps each.
Expected wall time: ~30-40 minutes with parallel workers (CPU).

## Step 5: Generate Analysis

After all runs complete, generate `archive/iter_027/results/final_analysis.md` with:

### Per-Arm Summary
For each arm (A′, A, B, C):
- N seeds
- Collapse rate (dual criterion)
- Collapse rate (eval-only)
- Collapse rate (train-only)
- Mean final train loss ± std
- Mean centroid MSE (REF ONLY)
- Mean delta_R2_color (REF ONLY)
- Parameter count

### Per-Seed Train-vs-Eval Std Gap Table (co-equal with collapse rates)
For every seed in every arm:
| seed | arm | collapsed_eval | collapsed_train | collapsed | per_dim_std_eval | per_dim_std_train | train_eval_gap |

### Gate Check
For each arm: PASS (≤10%) or FAIL (>10%)

### Pre-Registered Outcome Classification
Apply the pre-registered rules:
1. If Arm B ≤10%: POSITIVE CONSTRUCTIVE (report Arm C status)
2. If Arm B ≥20%: SECOND NULL — project pivots
3. If Arm B in (10%, 20%): SOFT NULL — same pivot as ≥20% (pre-committed)
4. If Arm C ≥20%: ARM C FALSIFICATION — project pivots

IMPORTANT: When reporting, use "is consistent with / does not refute / provides evidence for" language. Do NOT use "proves," "demonstrates," "stabilizes," or "resolves" without the capacity control.

If Arm B ≤10% AND Arm C ≤10%, note that the result is consistent with both gradient decoupling AND added capacity; report as "not refuted, pending capacity control."

### Readout Effect Report
Compare Arm A′ (centroid_gated, shared) vs Arm A (mean, shared) to isolate the readout effect on collapse rate.

### Parameter Count Documentation
Report exact parameter counts per arm and note the capacity confound interpretation.

## FILES TO CREATE
- src/pre_registration.md (UPDATE with amendments)
- src/models_separate_dyn.py (NEW)
- src/run_phase0_separate_dyn.py (NEW)
- archive/iter_027/results/ (output directory, created by script)

## FILES NOT TO MODIFY
- src/models_dual_stream.py (leave untouched for backward compatibility)
- src/environment.py
- All other existing files

## IMPORTANT CONSTRAINTS
1. Read the pre-registration FIRST and adhere to it strictly
2. Do NOT retune the gate threshold, seed list, buffer size, or dual-collapse criterion mid-run
3. All 40 runs must complete — no early termination
4. The train-vs-eval std gap is a CO-EQUAL reporting requirement, not a footnote
5. Use careful language: "consistent with", "does not refute", "provides evidence for" — not "proves" or "resolves"
