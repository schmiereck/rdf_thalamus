Implement `src/train.py` to run the Phase 1 scientific evaluation of Thalamus from end-to-end.

Here are the detailed specifications for `src/train.py` and the experiment:

1. Replay Buffer & Training Loop Setup:
   - Create a `ReplayBuffer` of capacity 2000 that stores transitions (x_hist, x_target), where x_hist has shape (3, 3, 128) and x_target has shape (3, 128).
   - Implement a seeding function `set_seed(seed)` that seeds random, numpy, and torch.
   - For a given model and seed, run training on PhysicsSandbox(N=2) for 1500 steps.
   - Transition to PhysicsSandbox(N=3) at step 1500, clear the replay buffer and history deque, and train on N=3 for 3500 steps (total 5000 steps).
   - Use batch size 32, learning rate 1e-3, and Adam optimizer.
   - For 'dynamic' (DynamicJEPA), call `model.update_recruitment_logic(sim_loss.item())` at each training step to handle GDASR recruitment with k=4, cooldown=500, stabilization_period=200.

2. Evaluation and Metrics:
   - At the end of training (after step 5000), evaluate the model on a separate test environment PhysicsSandbox(N=3, seed=seed+10000) for 100 transitions.
   - Compute final test sim_loss, standard deviations for all active dimensions, and mean absolute correlation on this evaluation batch.
   - For 'dynamic' (if active_dims >= 3), also compute the cross-correlation matrix on the evaluation batch and log the correlations r_0_2 (between dim 0 and 2) and r_1_2 (between dim 1 and 2) to measure the orthogonality of the newly recruited dimension.
   - Save step-by-step logs for each step to `archive/iter_002/runs/{model_type}_seed{seed}.csv` with columns: step, loss, sim_loss, var_loss, cov_loss, active_dims, std_dim0..7, mean_abs_corr.

3. Aggregation and Analysis:
   - Run the 15 runs: 3 models ('b1', 'b1_large', 'dynamic') across 5 seeds (42, 123, 456, 789, 999).
   - Collect and aggregate results across seeds. Compute mean and std for: final prediction error, recruitment step, mean absolute correlation, and orthogonality.
   - Test the three preregistered falsification criteria:
     1. Did 'dynamic' always recruit a 3rd dimension (d_t=3) within 3500 steps of N=3?
     2. Is the final N=3 prediction error of 'dynamic' at least 30% lower than 'b1'?
     3. Did representation collapse occur (any active dim std <= 0.1 or mean absolute corr >= 0.3)?
   - Generate summary.csv in `archive/iter_002/results/`.
   - Generate high-quality learning curve plots and save as PNG in `archive/iter_002/results/learning_curves.png`.

Write this complete pipeline into `src/train.py` and run it using `.venv\\Scripts\\python.exe src/train.py`. Make sure to output the detailed results, the pre-registered falsification criteria checklist, and any logging statements to stdout so that I can see them.