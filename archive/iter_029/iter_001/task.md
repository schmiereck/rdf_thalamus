You are preparing the codebase for iter_029 experiment. You must make three changes:

## 1. Update Pre-Registration File

Write `src/pre_registration.md` with the following content (replacing the entire file):

```markdown
# RDF Scientific Pre-Registration

*   **Iteration:** 029
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
On the separate-backbone architecture (SeparateDynEncoder via NonParametricJEPASpatialSeparateDyn),
adding an explicit SFA slowness term ||z_dyn(t) - z_dyn(t-1)||² to the VICReg-only z_dyn objective
is consistent with M2's predicted mechanism for improving identity encoding as measured by ΔR²_color
from a held-out linear probe, without introducing collapse on the union seed bank.

Specifically: Arm B (SFA+VICReg, sfa_weight=5.0, mask_dyn_sim=True, coord_vicreg=True) will show
ΔR²_color ≥ 0.30 (practical-significance threshold) when computed over non-collapsed seeds, AND will
show collapse rate ≤ 10% on the union seed bank.

NOTE: "0% collapse" under SFA+VICReg is expected (SFA + VICReg both encourage high std) and is NOT
the headline metric. The headline is ΔR²_color improvement above the practical-significance threshold.

## 2. Falsification Criterion
The hypothesis is falsified if ANY of the following hold:
(F1) Arm B shows mean ΔR²_color < 0.30 when computed over non-collapsed seeds — meaning SFA does not
     provide a practically significant identity-encoding benefit above the VICReg-only baseline;
(F2) Arm B shows collapse rate > 10% (>2 of 20 seeds) on the union seed bank — meaning SFA
     destabilizes the previously stable VICReg-only regime;
(F3) Arm B shows centroid_mse_mean exceeding the VICReg-only baseline (Arm A) by more than 1σ — meaning
     SFA degrades the spatial readout while potentially improving identity encoding.

The 0.30 threshold is derived from the Manager's instruction: max(0.1812 + 2σ, 0.30). The per-seed
σ from iter_027 Arm C is ≈0.35 (dominated by one outlier), making 0.1812 + 2σ ≈ 0.88 impractical.
The absolute floor of 0.30 is used as the practical-significance threshold. Results between 0.18 and
0.30 are reported as "no detectable marginal benefit."

Language constraint: positive results are reported as "consistent with M2's predicted mechanism,"
not as "demonstrates SFA improves identity."

## 3. Proposed Method
Three-arm experiment on a UNION seed bank of 20 seeds:
  Original seeds: [7, 17, 31, 53, 71, 83, 97, 113, 127, 149]
  Fresh seeds:   [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]

Arm A (VICReg-only Control): Separate backbone, primary_objective="jepa",
  mask_dyn_sim=True, var_weight=25, cov_weight=25, sim_weight=1.
  This is VICReg-only on z_dyn (the iter_027 Arm C configuration).
  20 NEW runs on the union seed bank.

Arm B (SFA+VICReg, primary): Separate backbone, primary_objective="sfa", mask_dyn_sim=True,
  sfa_weight=5.0, var_weight=25, cov_weight=25, sim_weight=1, coord_vicreg=True.
  This adds the SFA slowness term to the VICReg-only z_dyn objective, with JEPA prediction
  as a stop-gradient readout. Coord-stream VICReg is kept ON.
  20 runs on the union seed bank.

Arm C (SFA+VICReg, conservative): Same as Arm B but sfa_weight=1.0, coord_vicreg=True.
  20 runs on the union seed bank.

Total new runs: 60 (3 arms × 20 seeds).

Training: 8000 steps, batch_size=32, lr=3e-4, d_t=3 (frozen), buffer=4000.
Architecture: NonParametricJEPASpatialSeparateDyn (SeparateDynEncoder + DualStreamPredictor).

Evaluation metrics per run:
- Collapse check: per-dim std of z_dyn (threshold 0.5) on both eval and train batches
- ΔR²_color: held-out linear probe predicting RGB color from z_dyn (identity encoding quality)
- Centroid MSE: Arm F soft-argmax position decoding (spatial encoding quality)
- mean_abs_corr: VICReg health metric on z_dyn
- sfa_loss trajectory: verify slowness term is active and decreasing
- Per-seed outcomes with per-dim std magnitudes for any collapses
- Hard-seed table: explicit tabulation of seeds 53 and 71 outcomes across arms

Key design decisions (Manager's structural fixes):
1. Union seed bank includes both original hard seeds (53, 71) and fresh seeds to disambiguate
   SFA's identity-encoding benefit from hard-seed stabilization.
2. coord_vicreg=True: SFA mode now keeps coord-stream VICReg (var_loss_coord, cov_loss_coord)
   active, fixing the confound where SFA-mode previously zeroed coord VICReg while JEPA-mode
   kept it. This ensures the z_coord backbone receives consistent gradient signal.
3. centroid_mse degradation floor (F3): prevents a "win" on ΔR²_color that silently destroys
   the spatial readout.
```

## 2. Patch SFA Mode to Support coord_vicreg Flag

In `src/models_dual_stream.py`, the `NonParametricJEPASpatial` class needs a new parameter `coord_vicreg=True` in its `__init__`. Then in the `forward` method's SFA branch (the `if self.primary_objective in ["sfa", "contrastive"]:` block, around line 780-800), modify the code so that when `coord_vicreg=True`, it computes var_loss_coord and cov_loss_coord from z_target_coord, instead of setting them to zero.

Specifically, in the SFA branch of `NonParametricJEPASpatial.forward()`, find these lines:
```python
var_loss_coord = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
var_loss = var_loss_dyn

cov_loss_coord = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
cov_loss = cov_loss_dyn
```

Replace with:
```python
if self.coord_vicreg:
    z_target_coord_active = z_target_coord[:, :self.d_t]
    var_loss_coord = calc_var_loss(z_target_coord_active)
    cov_loss_coord = calc_cov_loss(z_target_coord_active)
else:
    var_loss_coord = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
    cov_loss_coord = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
var_loss = var_loss_dyn + var_loss_coord
cov_loss = cov_loss_dyn + cov_loss_coord
```

Also add `self.coord_vicreg = coord_vicreg` in `__init__`, add `coord_vicreg=True` to the constructor signature.

Also update the `clone` method to include `coord_vicreg=self.coord_vicreg`.

In `src/models_separate_dyn.py`, update `NonParametricJEPASpatialSeparateDyn.__init__` to:
1. Accept `coord_vicreg=True` parameter
2. Store `self.coord_vicreg = coord_vicreg`
3. Pass `coord_vicreg` when calling parent (but it uses `nn.Module.__init__()` so just set the attribute)
4. Update `clone` method to include `coord_vicreg=self.coord_vicreg`

## 3. Create Experiment Runner

Create `src/run_phase0_sfa_separate_backbone.py` modeled after `src/run_phase0_separate_dyn.py`. Key differences:

- ARMS definition:
  ```python
  ARMS = [
      {"name": "A (VICReg-only control)", "model_type": "separate", "primary_objective": "jepa",
       "mask_dyn_sim": True, "coord_vicreg": True, "sfa_weight": 25.0,
       "var_weight": 25.0, "cov_weight": 25.0, "sim_weight": 1.0,
       "lr": 3e-4, "batch_size": 32, "d_max": 8, "d_t": 3,
       "pos_encoding": "none", "replay_buffer_capacity": 4000},
      {"name": "B (SFA+VICReg, sfa=5.0)", "model_type": "separate", "primary_objective": "sfa",
       "mask_dyn_sim": True, "coord_vicreg": True, "sfa_weight": 5.0,
       "var_weight": 25.0, "cov_weight": 25.0, "sim_weight": 1.0,
       "lr": 3e-4, "batch_size": 32, "d_max": 8, "d_t": 3,
       "pos_encoding": "none", "replay_buffer_capacity": 4000},
      {"name": "C (SFA+VICReg, sfa=1.0)", "model_type": "separate", "primary_objective": "sfa",
       "mask_dyn_sim": True, "coord_vicreg": True, "sfa_weight": 1.0,
       "var_weight": 25.0, "cov_weight": 25.0, "sim_weight": 1.0,
       "lr": 3e-4, "batch_size": 32, "d_max": 8, "d_t": 3,
       "pos_encoding": "none", "replay_buffer_capacity": 4000},
  ]
  ```

- SEEDS = union of original [7, 17, 31, 53, 71, 83, 97, 113, 127, 149] + fresh [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]

- When creating the model for NonParametricJEPASpatialSeparateDyn, pass primary_objective and coord_vicreg from arm_config:
  ```python
  model = NonParametricJEPASpatialSeparateDyn(
      d_max=d_max, h=3, k=4, cooldown=300, stabilization_period=100,
      pos_encoding=pos_encoding, primary_objective=arm_config.get("primary_objective", "jepa"),
      sfa_weight=arm_config.get("sfa_weight", 25.0), gdasr_log_only=True,
      dyn_readout=dyn_readout, sub_features=1, dyn_source="spatial",
      mask_dyn_sim=mask_dyn_sim, coord_vicreg=arm_config.get("coord_vicreg", True))
  ```

- Pass sfa_weight from arm_config to model.forward() call
- Include sfa_loss in the logging (it's returned in the loss_dict when primary_objective="sfa")
- Results directory should be `archive/iter_029/results/`
- The run_single function should be mostly copied from run_phase0_separate_dyn.py but with these modifications
- The analysis function should include:
  1. Per-arm summary (collapse rate, ΔR²_color, centroid_mse, mean_abs_corr)
  2. Hard-seed table for seeds 53 and 71 across all arms
  3. Original vs fresh seed bank comparison
  4. Gate check against F1 (ΔR²_color ≥ 0.30), F2 (collapse ≤ 10%), F3 (centroid MSE floor)
  5. Pre-registered outcome classification

Please make all three changes. Verify the code is syntactically correct by running:
```bash
cd /home/user && python -c "from src.models_dual_stream import NonParametricJEPASpatial; from src.models_separate_dyn import NonParametricJEPASpatialSeparateDyn; print('Import OK')"
```

And verify the runner can be imported:
```bash
cd /home/user && python -c "import src.run_phase0_sfa_separate_backbone; print('Runner import OK')"
```

Also run a quick dry-run to verify it works end-to-end:
```bash
cd /home/user && python -m src.run_phase0_sfa_separate_backbone --dry-run --seeds 7 --sequential 2>&1 | tail -20
```
