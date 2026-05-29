Check the number of CPUs on this machine. Then, modify src/run_phase0_sfa_multistep.py to run the 26 seed-arm combinations in parallel using concurrent.futures.ProcessPoolExecutor.

Make sure:
1. It flattens the nested loop (arm x seed) into a list of tasks.
2. It uses max_workers based on the CPU count (e.g. min(cpu_count() - 1, 8) or similar, or allow user override). Note that each worker process should use device 'cpu' (or 'cuda' if available).
3. The worker function runs `run_single(arm, seed, device, dry_run)` and returns the results list and logs (or saves them within the worker and returns the results list).
4. Since PyTorch uses multiple threads by default inside each process, set `torch.set_num_threads(1)` at the start of each worker process to avoid thread oversubscription and maximize parallel speedup on CPU.
5. Compile and save the final summarized and aggregated CSVs exactly as before after all tasks finish.
6. Verify that `--dry-run` still works perfectly and executes all 26 runs in parallel in just a few seconds.