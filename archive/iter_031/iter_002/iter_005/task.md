Create the training and evaluation script `src/run_iter031_partA.py` in the current project workspace directly, without any delay or excessive analysis.

Here are the precise specifications and template-derived guidelines for `src/run_iter031_partA.py`:
1. Use `src/run_iter030_arm2.py` as your template (copy/adapt it directly).
2. It must import:
   `from src.models_recon import ReconVICRegSeparateDyn`
3. In `run_single(arm_config, seed, device, dry_run=False)`:
   - Create the model using `ReconVICRegSeparateDyn`:
     ```python
     model = ReconVICRegSeparateDyn(
         d_max=d_max, h=3, k=4, cooldown=300, stabilization_period=100,
         pos_encoding=pos_encoding, dyn_readout="mean", sub_features=1,
         dyn_source="spatial", coord_vicreg=coord_vicreg,
         recon_weight=recon_weight_val, var_weight=var_weight,
         cov_weight=cov_weight, sim_weight=sim_weight
     )
     ```
   - Freeze the encoder if `freeze_encoder=True` (Arm C):
     ```python
     if arm_config.get("freeze_encoder", False):
         for param in model.encoder.parameters():
             param.requires_grad = False
     ```
   - Make sure optimizer only optimizes trainable parameters:
     ```python
     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
     ```
   - Set `model.d_t = d_t` (and freeze it at 3, no recruitment).
   - The model forward pass returns `loss_dict, x_recon, (z_target_coord, z_target_dyn)`. It takes `x_hist_t, x_target_t`.
   - Update recruitment logic using: `model.update_recruitment_logic(loss_dict["sim_loss"].item(), target_dim=d_t, step=step)`.
4. The evaluation code `evaluate_run` must run the forward pass:
   ```python
   with torch.no_grad():
       loss_dict, _, (z_coord, z_dyn) = model(x_hist_t, x_target_t)
   ```
   and collect `recon_losses.append(loss_dict["recon_loss"].cpu().item())`, and return `"recon_mse_mean": float(np.mean(recon_losses))` along with the other standard metrics (collapsed_eval, per_dim_std, centroid_mse_mean, delta_r2_color, etc.).
5. Seeds bank: `[7, 17, 31, 53, 71, 83, 97, 113, 127, 149, 101, 103, 107, 109, 131, 137, 139, 151, 157, 163]` (20 seeds).
6. Arms configuration list:
   ```python
   ARMS = [
       {
           "name": "Arm A (d_max=8, trained)",
           "d_max": 8,
           "freeze_encoder": False,
           "recon_weight": 25.0,
           "var_weight": 25.0,
           "cov_weight": 25.0,
           "sim_weight": 1.0,
           "lr": 3e-4,
           "batch_size": 32,
           "d_t": 3,
           "pos_encoding": "none",
           "replay_buffer_capacity": 4000,
           "coord_vicreg": True,
       },
       {
           "name": "Arm B (d_max=2, trained)",
           "d_max": 2,
           "freeze_encoder": False,
           "recon_weight": 25.0,
           "var_weight": 25.0,
           "cov_weight": 25.0,
           "sim_weight": 1.0,
           "lr": 3e-4,
           "batch_size": 32,
           "d_t": 3,
           "pos_encoding": "none",
           "replay_buffer_capacity": 4000,
           "coord_vicreg": True,
       },
       {
           "name": "Arm C (d_max=8, random-encoder)",
           "d_max": 8,
           "freeze_encoder": True,
           "recon_weight": 25.0,
           "var_weight": 25.0,
           "cov_weight": 25.0,
           "sim_weight": 1.0,
           "lr": 3e-4,
           "batch_size": 32,
           "d_t": 3,
           "pos_encoding": "none",
           "replay_buffer_capacity": 4000,
           "coord_vicreg": True,
       }
   ]
   ```
7. Output folders inside `run_single_worker`:
   `runs_dir = "archive/iter_031/results/runs"`
   `checkpoints_dir = "archive/iter_031/results/checkpoints"`
   Make sure to create these directories recursively if they do not exist.
8. Adapt `_generate_analysis` for these 3 arms and print summaries and gates (F1-F4) with 95% CI.
9. Verify that running `python src/run_iter031_partA.py --dry-run` executes quickly and successfully across all 3 arms on a couple of seeds, and prints its output. Clean up any dry run test files and print confirmations!