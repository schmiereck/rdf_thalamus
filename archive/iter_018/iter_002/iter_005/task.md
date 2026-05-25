You are an AI executor sub-agent for iter_id 18.2.5. We have discovered a critical bug in `src/run_phase18_experiments.py`: when loading the passive model from disk cache, its `model.d_t` remains at 2 (since it was initialized to 2 and never set to 3), whereas passive pre-training ends with `d_t = 3`. This caused the active training sweep to start with `d_t = 2`, preventing proposals and WUP checks from executing correctly in the control sweep.

### Task:

1. Edit `src/run_phase18_experiments.py` to fix the disk cache loading. Set `model.d_t = 3` right after loading:
   ```python
   if os.path.exists(cache_path):
       print(f"     [passive] loading cached model for seed {seed} from {cache_path}")
       model.load_state_dict(torch.load(cache_path, map_location=device))
       model.d_t = 3  # <-- ADD THIS CRITICAL LINE!
       model = model.to(device)
       return model, {"S_bar_end": 0.05, "final_d_t": 3}
   ```
2. Double-check if there are any other places where `model.d_t` is assumed to be 3 after loading. (In `run_active_branch`, `branch_model = base_model_template.clone()` is called, which will correctly inherit `model.d_t = 3`).
3. Check if PyTorch can use CUDA (GPU) on this machine. If CUDA is available, ensure the script is running on the GPU to speed it up.
4. Execute `python src/run_phase18_experiments.py` and verify that:
   - WUP probation is successfully triggered in BOTH the transition sweep and the control sweep!
   - For control sweep, the Noisy-TV distractor's lack of predictable pattern should yield $\rho \approx 1.0$, which gets rejected by the prediction-trend gate (EG-MDL Arms S and S_alt), while the baseline Arm P might recruit it (yielding the expected falsification result).
5. Verify that all 4 results files under `archive/iter_018/results/` are updated and fully populated.
6. Print the updated falsification audit results from the run. Let's do this!