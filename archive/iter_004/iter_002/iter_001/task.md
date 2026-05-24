Create and run 'src/train_thalamus.py' to perform the systematic 5-seed sweep evaluation for Phase 2 (Thalamic Gating) exactly as specified.

Write a complete Python training script that implements:
1. Replay Buffer class containing (x_hist, x_target, color_0, pos_0).
2. Seeding utility and pre-filling logic for N=2 and N=3.
3. Training loop: 5000 steps per seed.
   - For steps 1-1500: N=2, priming_mode="external" using object 0 color.
   - At step 1501: N=3, clear and prefill replay buffer with 100 transitions.
   - For steps 1501-5000: N=3, priming_mode="self", and record the physical tracking overlap.
   - Optimize active/trainable parameters with Adam (lr=1e-3) at each training step.
4. Evaluation loop:
   - Evaluates at step 5000 on a separate test environment with N=3 and seed = seed + 10000 for 100 steps.
   - Uses self-generated priming 'priming_mode="self"'.
   - Measures final test L2 prediction loss ('l2_sim_loss' for gated/nongated, and 'sim_loss' for B1) and final test tracking overlap (for gated and nongated).
5. Aggregate and analyze the results across 5 seeds:
   - Compute the mean and standard deviation of L2 test loss for 'gated', 'nongated', and 'b1'.
   - Run Levene's test on the test L2 prediction loss between 'gated' and 'nongated' to compute the p-value (or use the manual Levene's test implementation as a fallback).
   - Compute the stable step (where rolling average of window 100 or raw L2 loss falls below 0.08 and remains below 0.08).
   - Compute the average target tracking overlap during steps 1501-5000 and the final test.
   - Check all three pre-registered falsification criteria.
   - Save the summary results to 'archive/iter_004/results/summary.csv' and write an execution log.
6. Generate three visual plots under 'archive/iter_004/results/':
   - 'learning_curves.png': Compare the L2 prediction loss over training steps for all three models (mean +/- std).
   - 'tracking_overlap.png': Compare the physical tracking overlap over steps 1501-5000 for 'gated' vs 'nongated'.
   - 'token_traces.png': Plot the attention token locus over training steps for 'gated' (e.g. for seed 42 and mean across seeds).

Run this script and verify that it completes successfully without any issues. If any errors are encountered, fix them in 'src/train_thalamus.py' and rerun. Print out a comprehensive final report summarizing the results.