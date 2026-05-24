Implement and execute the systematic evaluation and comparison sweep for Phase 3.
1. Create src/train_eval_closed_loop.py which:
   - Trains M_active, M_no_motor, and M_random over 5 seeds (42, 123, 456, 789, 999).
   - For M_active, applies the staged training schedule: Steps 0-1000: Decoupled random motor actions; Steps 1000-3000: Lower-layer tracking active (push disabled); Steps 3000-5000: Full subsumption motorics active.
   - For M_no_motor, keeps the pointer stationary (acc=0, push=False).
   - For M_random, keeps the controller in random ablation mode throughout.
   - Saves CSV training logs and trained models to archive/iter_005/runs/.
   - Implements the Standardized Collision Benchmark: Pre-generates 100 deterministic 20-step trajectories of collision events with randomized hidden masses (e.g., 2.0 vs 12.0, decoupled from radius). Evaluates the post-collision L2 prediction loss of all 3 models on these identical trajectories.
   - Implements the Representation Ablation Control: Evaluates tracking overlap of M_active under random network ablation and spatial attention shuffling.
   - Measures self-generated vs. primed attention prediction loss ratio.
2. Execute the full sweep.
3. Perform the pre-registered falsification audit and compile summary.csv and all results under archive/iter_005/results/.
4. Provide a comprehensive summary of the findings in the final agent result.