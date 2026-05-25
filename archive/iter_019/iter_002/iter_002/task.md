Implement and execute Phase 19 experiments.

You must:
1. Re-write src/itag.py to use the exact python code provided in the prompt (using torch for identify_surprising_positions and numpy for compute_itag and compute_isag). Ensure that identify_surprising_positions sorts and returns top indices.
2. Create src/run_phase19_experiments.py by adapting src/run_phase18_experiments.py:
   - Arms:
     - Arm A: WUP-MDL (W=100), no ITAG pre-filter. Gated by MDL Ratio < 1.0.
     - Arm B: ITAG pre-filter (tau=0.3, W_t=20) + WUP-MDL (W=100).
     - Arm C: ITAG-only gating (tau=0.3, W_t=20), no WUP. Gated only by ITAG > 0.3 at step 1800.
   - Sweeps:
     - Sweep 1: transition (N=3 -> 4, 4th object mass doubled)
     - Sweep 2: control (N=3 + Noisy-TV)
     - Sweep 3: structured_distractor (N=3 + Sinusoidal Oscillator)
   - Ensure the training protocol is identical to Phase 18:
     - Load pre-existing cached models from `cache/` (e.g. cache/passive_model_seed_{seed}.pt).
     - Train CLTS online for 1500 steps (steps 1501..3000) for each of the 3 arms x 3 sweeps x 5 seeds = 45 branches.
   - For ITAG & ISAG computation and logging:
     - Maintain a `pixel_buffer = collections.deque(maxlen=50)` during active training to store the raw pixel frames (obs).
     - For steps 1781 to 1800 (inclusive), at each step compute:
       - target_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
       - a_spatial = model.encoder.forward_spatial(target_t)
       - prediction_error_map = a_spatial.norm(dim=1)
       - surprising_positions = identify_surprising_positions(prediction_error_map, top_k=16)
       - itag_score = compute_itag(list(pixel_buffer), surprising_positions, window=20)
       - isag_score = compute_isag(list(pixel_buffer), surprising_positions)
       - Log these scores per step, so we can aggregate them across the 20 steps and 5 seeds (100 values per sweep per arm).
     - At step 1800:
       - Use the itag_score computed at step 1800 to make the gating decisions for Arms B and C.
   - Collect and save results to archive/iter_019/results/:
     - Compute Cohen's d between ITAG distributions (20 steps x 5 seeds = 100 values) for:
       - C1: Transition vs Noisy-TV
       - C2: Transition vs Structured Distractor
     - Check Falsification Criteria:
       - C1: Cohen's d (Transition vs Noisy-TV) >= 1.5. (OK if >= 1.5, Falsified if < 1.5)
       - C2: Cohen's d (Transition vs Structured Distractor) < 1.5. (OK if < 1.5, Falsified if >= 1.5. Expected: FALSIFIED!)
       - C3: Gating performance on Noisy-TV (false recruitment <= 20%, genuine recruitment >= 80% on Arm B)
       - C4: Gating performance on Structured Distractors (false recruitment <= 20%, genuine recruitment >= 80% on Arm B)
       - C5: Scope-reduction trigger (if C2 and C4 both failed, dynamic recruitment is disabled).
     - Save:
       - `archive/iter_019/results/summary_phase19.csv`
       - `archive/iter_019/results/adaptation_curves_phase19.png`
       - `archive/iter_019/results/audit_results_phase19.json`
       - `archive/iter_019/results/phase19_report.md`
3. RUN the experiments by executing `python src/run_phase19_experiments.py`.
4. Ensure all code compiles and runs on CPU/GPU seamlessly. Check if the outputs are correctly saved.
