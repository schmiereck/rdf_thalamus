Write a python script 'src/run_sweep.py' to run and analyze the full comparison sweep.

Specifically, the script must:
1. Identify which of the 15 experiments (models: ['gated', 'nongated', 'b1'] x seeds: [42, 123, 456, 789, 999]) have not yet been run by checking for the existence of 'archive/iter_004/runs/{model}_seed{seed}_eval.json'.
2. Run each missing experiment by calling 'python src/train_thalamus.py --model {model} --seed {seed}'.
3. Read the results from all 15 experiments (both the step-by-step CSV run logs in 'archive/iter_004/runs/' and the eval JSONs) and compile them.
4. Perform the pre-registered falsification audit against the three criteria:
   - Criterion 1: Levene's test p-value and comparison of standard deviations of L2 test prediction loss between 'gated' and 'nongated'.
   - Criterion 2: Sample efficiency comparison (average step to reach L2 prediction loss < 0.08 and remain there).
   - Check Criterion 3: Average physical tracking overlap for 'gated' vs 'nongated' during steps 1501-5000 and test, and check if gated's prediction loss is at least 15% lower than B1's loss.
5. Create and save the three requested plots under 'archive/iter_004/results/':
   - 'learning_curves.png'
   - 'tracking_overlap.png'
   - 'token_traces.png'
6. Save the aggregated stats to 'archive/iter_004/results/summary.csv' and print out a detailed scientific analysis.

Write and execute 'src/run_sweep.py' to complete the Phase 2 research track and output the results.