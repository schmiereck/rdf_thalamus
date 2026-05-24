You are a high-performing autonomous ML research agent. Your task is to implement and execute the systematic evaluation and comparison sweep for Phase 3.

Please perform the following steps carefully:

1. Create a Python script `src/train_eval_closed_loop.py` that implements:
   - A deterministic replay buffer.
   - Training functions for the three models: M_active, M_no_motor, and M_random over 5 seeds (42, 123, 456, 789, 999).
     - Each training runs for 5000 steps.
     - Starts with PhysicsSandbox(N=2, seed=seed). Transitions to PhysicsSandbox(N=3, seed=seed) at step 1501, resetting history and clearing the buffer, prefilling with 100 transitions.
     - For M_active, apply the staged training schedule:
       - Steps 1-1000: Decoupled random motor actions (sample acc in [-10, 10], push with probability 0.1, independent of model).
       - Steps 1001-3000: Lower-layer tracking active (query SubsumptionMotorController from src/motor.py, but disable push by overriding action['push'] = False).
       - Steps 3001-5000: Full subsumption motorics active (query SubsumptionMotorController from src/motor.py and use action as-is).
     - For M_no_motor, keep the pointer stationary throughout training (acc=0, push=False).
     - For M_random, keep the controller in random ablation mode throughout training (acc in [-10, 10], push with probability 0.1).
     - Save training logs as CSV (fields: step, loss, l2_sim_loss, token_locus, l2_locked, overlap) and trained models (.pt state_dicts) to `archive/iter_005/runs/`.
   - The Standardized Collision Benchmark:
     - Deterministically pre-generate 100 20-step trajectories of collision events with randomized hidden masses (e.g. 2.0 vs 12.0, decoupled from radius). Ensure object 0 and object 1 are guaranteed to collide (e.g., pos0=40.0, vel0=3.0, r0=5.0 vs pos1=88.0, vel1=-3.0, r1=5.0).
     - For each model and seed, load the trained model and evaluate post-collision L2 prediction loss (L2 similarity loss `l2_sim_loss`) over steps 8-20 (indices 5-17 in the 18 transitions) of these 100 trajectories, with `priming_mode="self"`. Average across the trajectories.
   - The Representation Ablation Control:
     - For M_active and each seed, run a test environment (N=3, seed=seed+10000) for 100 steps under three control configurations:
       1. Normal (no ablation).
       2. Random network ablation (SubsumptionMotorController with `ablation="random"`).
       3. Spatial attention shuffling (SubsumptionMotorController with `ablation="shuffle"`).
     - Measure and return the average physical tracking overlap of object 0 over the 100 steps.
   - Priming Comparison:
     - For M_active and each seed, run the test environment (N=3, seed=seed+10000) for 100 steps in:
       1. Primed attention mode: priming_mode="external" with target color fed as external query.
       2. Self-generated attention mode: priming_mode="self" with no external query.
     - Measure the average `l2_sim_loss` over the 100 steps for both modes and compute their ratio (Self / Primed).

2. Optimize execution speed of the 15 training runs by executing them in parallel using concurrent.futures.ProcessPoolExecutor (e.g., 3 processes in parallel). Set `torch.set_num_threads(2)` in each worker process to avoid CPU thrashing.

3. After running the full sweep:
   - Create `archive/iter_005/results/summary.csv` compiling the summary statistics (mean & std across 5 seeds) for all three models on the key metrics:
     - Normal tracking overlap (`overlap_mean`, `overlap_std`).
     - Tracking overlap under random ablation (`overlap_ablation_random_mean`, `overlap_ablation_random_std`).
     - Tracking overlap under spatial attention shuffling (`overlap_ablation_shuffle_mean`, `overlap_ablation_shuffle_std`).
     - Post-collision L2 prediction loss on collision benchmark (`collision_l2_loss_mean`, `collision_l2_loss_std`).
     - Test environment overall L2 prediction loss under self-generation (`self_l2_loss_mean`, `self_l2_loss_std`).
     - Test environment overall L2 prediction loss under priming (`primed_l2_loss_mean`, `primed_l2_loss_std`).
     - Ratio of self-generated to primed L2 loss (`self_to_primed_ratio`).
   - Create `archive/iter_005/results/falsification_report.md` performing the pre-registered falsification audit on the 4 falsification criteria:
     - Criterion 1: Is tracking overlap O_track of M_active >= 70.0%?
     - Criterion 2: Is post-collision L2 prediction loss ratio L_collision(M_active) / L_control < 0.65 for both controls (M_random and M_no_motor)?
     - Criterion 3: Is self-generated loss / primed loss ratio <= 1.15?
     - Criterion 4: Is M_active overall test L2 prediction loss <= B1 model baseline (0.0452)?
     - Declare whether each criterion passes or fails, and summarize the overall scientific verdict.

4. Print out the progress and complete evaluation metrics to stdout so that it is visible in the logs.
