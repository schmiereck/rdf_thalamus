Implement src/train.py to execute the full Phase 1 scientific evaluation of Thalamus across 5 seeds and 3 model configurations.

Specifically:
1. Implement src/train.py which includes:
   - A replay buffer of size 5000 storing transition sequences of length H+1 (H=3 history frames + 1 target frame).
   - A training loop that:
     - Initializes the environment and model.
     - Runs the N=2 physics sandbox for 2500 steps, adding transitions to the buffer, and performing a training step each time.
     - Tracks the EMA of prediction error (sim_loss.item()) during the last 1000 steps of N=2 to establish the baseline error mean and standard deviation.
     - At step 2500, transitions to N=3. Clears the replay buffer of N=2 transitions to ensure the model immediately encounters pure N=3 inputs and surprise triggers can operate on clean data.
     - Runs the N=3 sandbox for 5000 steps, continuing online training.
     - Logs: step, loss, sim_loss, var_loss, cov_loss, active_dimensions (d_t), and active dimension stats (std of each dimension, and mean absolute correlation between dimensions calculated on the training batch).
   - A evaluation function that evaluates the model on a fixed test batch of 100 sequences from a separate test environment (both N=2 and N=3) and computes average sim_loss, dimension stds, and absolute cross-dimension correlations.
2. Run the 15 experiments:
   - 3 model types: 'b1' (FixedJEPA, d_t=2), 'b1_large' (FixedJEPA, d_t=3), 'dynamic' (DynamicJEPA, starting at d_t=2, adaptive trigger with k=4 and cooldown=500, with 200 step stop-gradient stabilization).
   - 5 independent random seeds (42, 123, 456, 789, 999).
   - Ensure the same seed is used for torch, numpy, and python random, and also passes the seed to the environment so the environments are identical across models for a given seed.
3. For each run:
   - Save the step-by-step training logs to `archive/iter_002/runs/{model_type}_seed{seed}.csv`.
4. After completing all 15 runs, write an aggregation and analysis script (or embed it in src/train.py) that:
   - Computes the mean and standard deviation across seeds of:
     - Step of recruitment (for 'dynamic'). Did it always recruit?
     - Final prediction error (sim_loss) on N=3 (averaged over the last 1000 steps).
     - Latent standard deviation per active dimension.
     - Mean absolute correlation between active dimensions.
     - Projection/orthogonality of the newly recruited dimension onto the stable 2D subspace (for 'dynamic').
   - Tests the three pre-registered falsification criteria:
     1. Did 'dynamic' always recruit a 3rd dimension within 5000 steps of N=3?
     2. Is the final N=3 prediction error of 'dynamic' at least 30% lower than 'b1'?
     3. Did representation collapse occur (defined as standard deviation <= 0.1 or mean absolute correlation >= 0.3)?
   - Compares the learning speed / adaptation curve of 'dynamic' after recruitment versus 'b1_large'.
   - Generates CSV summaries in `archive/iter_002/results/` (e.g., `summary.csv`, `learning_curves.csv`).
   - Generates high-quality learning curve plots and saves them as PNGs in `archive/iter_002/results/`.
5. Run the entire experimental pipeline and verify that all outputs are generated. Write out a clear summary of the results and status.