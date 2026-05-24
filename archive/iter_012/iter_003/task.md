You are a research implementation agent. Write and run the experiment script `src/run_phase12_experiments.py` immediately.

Here is the exact structure and implementation logic for `src/run_phase12_experiments.py`:
1. Use the existing imports from `src/run_phase11_experiments.py` (including `PhysicsSandbox` from `src.environment`, `NonParametricJEPASpatial` and `calculate_centroid_and_variance as calculate_centroid_and_variance_ds` from `src.models_dual_stream`, and the `ReplayBuffer`, `set_seed`, `fit_linear_probe`, and `CLTSMotorController` from `src.motor`).
2. Write a standardized passive test evaluation function:
```python
def evaluate_branch_phase12(model, seed, device):
    set_seed(seed + 5000)
    test_env = PhysicsSandbox(N=4, seed=seed + 5000)
    test_env.masses[3] *= 2.0  # Apply 2x mass perturbation on the novel object
    test_obs = test_env.reset()
    test_history = collections.deque(maxlen=4)
    test_history.append(test_obs)
    
    test_x_hist = []
    test_x_target = []
    test_y_4 = []
    
    for _ in range(203):
        obs_t, info_t = test_env.step({"acc": 0.0, "push": False})
        test_history.append(obs_t)
        if len(test_history) == 4:
            test_x_hist.append(np.stack(list(test_history)[:3], axis=0))
            test_x_target.append(test_history[3])
            test_y_4.append(info_t["positions"][3])
            
    test_x_hist_t = torch.from_numpy(np.stack(test_x_hist, axis=0)).float().to(device)
    test_x_target_t = torch.from_numpy(np.stack(test_x_target, axis=0)).float().to(device)
    test_y_4_arr = np.array(test_y_4)
    
    y_probe_train = test_y_4_arr[:100]
    y_probe_test = test_y_4_arr[100:]
    
    model.eval()
    with torch.no_grad():
        loss_dict, _, _ = model(test_x_hist_t, test_x_target_t)
        test_sim_loss = loss_dict["sim_loss"].item()

        z_target_coord, z_target_dyn = model.encoder(test_x_target_t)
        z_3 = z_target_coord[:, 3].cpu().numpy()
        
        a_spatial = model.encoder.forward_spatial(test_x_target_t)
        centroids, variances = model.calculate_centroid_and_variance(a_spatial)
        x_mean_3 = centroids[:, 3].cpu().numpy()
        var_3 = variances[:, 3].cpu().numpy()
        
        z_active_coord = torch.abs(z_target_coord[:, :4]).cpu().numpy()
        e_a_3 = np.mean(z_active_coord[:, 3])
        e_a_all = np.mean(z_active_coord)
            
    # Fit linear probes
    w_cent, b_cent = fit_linear_probe(x_mean_3[:100], y_probe_train)
    y_pred_cent = x_mean_3[100:] * w_cent + b_cent
    mse_cent = np.mean((y_probe_test - y_pred_cent)**2)
    
    # Calculate Pearson correlations
    r_centroid = np.corrcoef(x_mean_3, test_y_4_arr)[0, 1]
    abs_r_centroid = abs(r_centroid) if not np.isnan(r_centroid) else 0.0
    
    mean_var_3 = np.mean(var_3)
    std_x_mean_3 = np.std(x_mean_3)
    
    # Standard collapse criterion (from Phase 10/11)
    has_collapsed = not (e_a_3 >= 0.1 * e_a_all and std_x_mean_3 > 5.0)
    
    return {
        "test_sim_loss": test_sim_loss,
        "abs_r_centroid": abs_r_centroid,
        "mse_cent": mse_cent,
        "mean_var_3": mean_var_3,
        "std_x_mean_3": std_x_mean_3,
        "collapsed": has_collapsed
    }
```
3. Run the 5-seed comparative sweep over `[42, 123, 456, 789, 999]`:
   - Bootstrap training `NonParametricJEPASpatial` passively on N=3 for 1500 steps.
   - For each branch at step 1501 (Arm F-Passive, Arm F-Random, Arm G-CLTS):
     - Transition to N=4 sandbox and double the mass of the 4th object (object index 3).
     - Run the respective motor actions.
     - For Arm G (CLTS): Use `CLTSMotorController` initialized with default arguments. On each training step, compute:
       ```python
       obs_t = torch.from_numpy(branch_history[-1]).float().unsqueeze(0).to(device)
       with torch.no_grad():
           z_hist_coord_flat, z_hist_dyn_flat = branch_model.encoder(obs_t) # wait, to get z_pred and z_target for surprise, do a full forward pass on a batch or use the last step transition!
       ```
       Actually, to feed surprise-modulated inputs to CLTS, we can compute surprise of the current transition online:
       ```python
       # Let's feed the controller the most recent predicted and target latents
       # By passing: z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, d_t, centroids
       # For step t, we have branch_history containing the last 4 frames.
       # We can run branch_model(x_hist_t, x_target_t) where x_hist_t is the history of the last 3 frames, and x_target_t is the current frame.
       # This returns loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn).
       # We can pass these directly to clts_controller.get_action!
       ```
       Yes! This is perfect!
     - Save checkpoints of model at `[1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000]`.
     - Evaluate each checkpoint on the standardized passive evaluation sequence of seed+5000 to record `test_sim_loss` over time.
     - Record the pointer's coordinate trajectory during step 1501 to 3000 to compute Shannon entropy of pointer's positions (using 16 bins).
4. Save the compiled results in a Pandas DataFrame and save to `archive/iter_012/results/summary_phase12.csv`.
5. Aggregate evaluation curves (average `test_sim_loss` across seeds at each checkpoint step) for the three arms and plot them. Save the plot to `archive/iter_012/results/auc_recovery_curves.png`.
6. Compute the Area Under the Curve (AUC) for both offline test simulation loss curves and online training surprise curves.
7. Print out an audit report checking the three falsification criteria and save it.

Write and execute the script. Do NOT spend time on excess directory searches or planning; write the script file directly and run it.