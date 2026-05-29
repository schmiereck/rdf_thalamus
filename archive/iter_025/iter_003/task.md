Fix the remaining bug in src/run_phase0_id_probe.py and src/models_dual_stream.py, then run the full experiment.

## THE BUG
The experiment runner fails with:
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

This is in the `compute_id_contrastive_loss` method in src/models_dual_stream.py. The issue is in these lines:
```python
sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
```
This modifies sim_matrix in-place when autograd needs the original value. Fix it by using a new variable:
```python
sim_max = sim_matrix.max(dim=1, keepdim=True)[0].detach()
sim_shifted = sim_matrix - sim_max
```
And then use `sim_shifted` instead of `sim_matrix` for `exp_sim`.

Also check for any other in-place operations (like `masked_fill_` should be `masked_fill`, `+=` on tensors that need gradients, etc.).

## AFTER FIXING: RUN THE EXPERIMENT

1. First do a quick dry run to verify the fix:
```bash
cd /home/user && python src/run_phase0_id_probe.py --dry-run --sequential --seeds 7
```

2. If dry run passes, run the FULL experiment:
```bash
cd /home/user && python src/run_phase0_id_probe.py --sequential
```
Use --sequential for reliability. This will run:
- 3 noise floor runs (frozen encoder, 1000 steps each)
- 20 main runs (4 arms × 5 seeds × 5000 steps)
Total: 23 runs

3. After the experiment completes, read the results from archive/iter_025/results/ and compile an analysis.

## VERIFICATION CHECKLIST
- Fresh seeds [7, 17, 31, 53, 71] used for main experiment
- Seeds [7, 17, 31] used for noise floor
- Arm A: JEPA+VICReg control
- Arm B: Supervised Color Probe + VICReg, d_max=8
- Arm C: ID-Contrastive + VICReg, d_max=8  
- Arm D: Supervised Color Probe + VICReg, d_max=16
- All supervised arms report both sorted and Hungarian matching results
- Mismatch rate reported
- Collapse check (per_dim_std < 0.5)
- Full semantic probe suite (delta_R2_color primary metric)

## ANALYSIS TEMPLATE
After reading results, compute:
1. Noise floor: mean delta_R2_color from frozen encoder runs → floor_mean
2. Effective threshold: max(0.10, floor_mean + 0.08)
3. Per-arm: mean delta_R2_color (non-collapsed seeds only), collapse rate, mismatch rate
4. Compare each arm against threshold
5. Assign to four outcome quadrants
6. Write analysis to archive/iter_025/results/analysis.md

## KEY REMINDERS
- Do NOT change the model architecture (no separate encoder)
- The supervised weight ramp from 0.1→25.0 over 500 steps should be used for Arms B/D as default (to prevent collapse)
- Read src/pre_registration.md for full specification
- All results must be saved to archive/iter_025/results/