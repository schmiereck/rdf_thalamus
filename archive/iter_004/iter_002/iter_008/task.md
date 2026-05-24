Create two Python scripts: 'src/train_thalamus.py' and 'src/compile_results.py' to implement our robust distributed systematic evaluation framework.

1. 'src/train_thalamus.py' must accept '--model' and '--seed' arguments. When executed, it must:
   - Initialize the environment PhysicsSandbox with the given seed.
   - Set torch.set_num_threads(2).
   - Train the model for 5000 steps.
     - Steps 1-1500: N=2, priming_mode="external" using object 0 color as query.
     - Step 1501: Transition to N=3, clear and prefill replay buffer with 100 transitions.
     - Steps 1501-5000: N=3, priming_mode="self", and record tracking overlap.
   - Save the step logs to 'archive/iter_004/runs/{model_type}_seed{seed}.csv'.
   - Evaluate the model at step 5000 on a separate environment (seed = seed + 10000) with N=3 for 100 steps under self-generated priming.
   - Save the evaluation metrics ('model_type', 'seed', 'final_test_l2_loss', 'final_test_overlap') to 'archive/iter_004/runs/{model_type}_seed{seed}_eval.json'.

2. 'src/compile_results.py' must:
   - Load all 15 evaluation JSON files and 15 run CSV files.
   - Compute the aggregated mean and standard deviation of L2 test prediction loss and tracking overlap for each configuration ('gated', 'nongated', 'b1').
   - Run Levene's test on L2 test prediction loss between 'gated' and 'nongated'.
   - Determine the training step where the rolling average (window 100) of L2 loss stabilizes below 0.08 for each configuration.
   - Compute average training tracking overlap (steps 1501-5000) across the 5 seeds.
   - Check all three pre-registered falsification criteria:
     - Criterion 1: Does gated have a lower standard deviation of L2 test loss than nongated? (Compute Levene p-value).
     - Criterion 2: Does gated reach stable L2 loss < 0.08 in fewer steps than nongated?
     - Criterion 3: Does gated maintain target tracking overlap > 0.85 and reduce prediction loss on the target object by >= 15% compared to B1?
   - Save results to 'archive/iter_004/results/summary.csv' and 'archive/iter_004/results/execution_log.txt'.
   - Generate and save the three plots: 'learning_curves.png', 'tracking_overlap.png', and 'token_traces.png' to 'archive/iter_004/results/'.

Create these scripts and verify that they compile successfully. Do not start the full training sweep yet; we will run that next.