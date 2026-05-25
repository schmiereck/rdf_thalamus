We will implement the Phase 19 experiment script by copying the existing Phase 18 experiment script and modifying it, which is extremely efficient and avoids token limit problems.

Here are the exact instructions to copy and modify:

1. COPY `src/run_phase18_experiments.py` to `src/run_phase19_experiments.py` using bash/sh.
2. MODIFY `src/run_phase19_experiments.py`:
   - Import `identify_surprising_positions`, `compute_itag`, `compute_isag` from `src.itag`.
   - Update `main()` to define the three arms:
     - Arm A: ("Arm A (WUP-MDL Baseline)", 100, "mdl", None)
     - Arm B: ("Arm B (ITAG+MDL)", 100, "mdl", None)
     - Arm C: ("Arm C (ITAG-only)", 100, "itag_only", None)
   - Update `main()` to run three sweeps:
     - Sweep 1: `transition` (N=4, noisy_tv=False, structured_distractor=False)
     - Sweep 2: `control` (N=3, noisy_tv=True, structured_distractor=False)
     - Sweep 3: `structured_distractor` (N=3, noisy_tv=False, structured_distractor=True)
   - Update `run_active_branch` to support these three sweeps in environment setup:
     ```python
     if sweep_type == "transition":
         branch_env = PhysicsSandbox(N=4, seed=seed + 1000, noisy_tv=False)
         branch_env.masses[3] *= 2.0
         n_test_eval = 4
     elif sweep_type == "control":
         branch_env = PhysicsSandbox(N=3, seed=seed + 1000, noisy_tv=True)
         n_test_eval = 3
     elif sweep_type == "structured_distractor":
         branch_env = PhysicsSandbox(N=3, seed=seed + 1000, noisy_tv=False, structured_distractor=True)
         n_test_eval = 3
     ```
   - In `run_active_branch`, maintain a pixel history buffer:
     - `pixel_history = collections.deque(maxlen=100)`
     - When initializing, append the observations currently in `branch_history` to `pixel_history`.
     - In the step loop, after `obs, info = branch_env.step(action)`, append `obs` to `pixel_history`.
     - Create a list to store `itag_scores_eval` and `isag_scores_eval` for this branch run.
   - For steps 1781 to 1800 (inclusive), at each step, compute:
     - `target_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)` (where `obs` is the current step's observation)
     - `a_spatial = branch_model.encoder.forward_spatial(target_t)`
     - `prediction_error_map = a_spatial.norm(dim=1)`
     - `surprising_positions = identify_surprising_positions(prediction_error_map, top_k=16)`
     - `itag_score = compute_itag(list(pixel_history), surprising_positions, window=20)`
     - `isag_score = compute_isag(list(pixel_history), surprising_positions)`
     - Append these to `itag_scores_eval` and `isag_scores_eval` list.
   - At step 1800, use `itag_score = itag_scores_eval[-1]` for the gating decisions:
     - For Arm A:
       - Proceed to WUP probation normally: `probationary = True`, `probation_end_step = step + wup_window`, `branch_model.d_t = 4`, reset error buffer.
     - For Arm B:
       - If `itag_score > 0.3`:
         Proceed to WUP probation: `probationary = True`, `probation_end_step = step + wup_window`, `branch_model.d_t = 4`, reset error buffer.
       - Else:
         Reject immediately: `probationary = False`, `branch_model.d_t = 3`, `active_transition_accepted = False`, `next_proposal_check = step + 50`.
     - For Arm C:
       - If `itag_score > 0.3`:
         Immediately recruit: `active_transition_accepted = True`, `active_transition_accepted_step = step`, `branch_model.d_t = 4`, reset error buffer, `post_recruitment_audit_active = True`.
       - Else:
         Reject immediately: `probationary = False`, `branch_model.d_t = 3`, `active_transition_accepted = False`, `next_proposal_check = step + 50`.
   - Make sure that the WUP probation check at step 1900 is ONLY run if `gating` is not `"itag_only"`. For `"itag_only"`, it was already decided at step 1800, so we do nothing at step 1900.
   - Ensure the logged results dictionary returned by `run_active_branch` includes the logged `itag_scores_eval` and `isag_scores_eval` (convert them to standard float lists).
   - In `main()`, aggregate these scores across the 5 seeds and 20 steps (100 values per arm/sweep combination) to compute Cohen's d:
     - C1: Transition Sweep vs Noisy-TV Control (overall or per arm)
     - C2: Transition Sweep vs Structured Distractor Control
   - Update the Falsification Audit in `main()` to check criteria C1 to C5 exactly as pre-registered in `src/pre_registration.md`.
   - Ensure all results are saved to `archive/iter_019/results/`:
     - `summary_phase19.csv`
     - `adaptation_curves_phase19.png`
     - `audit_results_phase19.json`
     - `phase19_report.md`
3. RUN `python src/run_phase19_experiments.py` to perform the whole suite of experiments.
4. Verify all files are correctly created and contain proper results.
