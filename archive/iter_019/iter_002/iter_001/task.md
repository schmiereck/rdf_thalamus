Implement Phase 19: ITAG (Input-Level Temporal AutoCorrelation Gating) with Structured Distractor.

Here are the precise tasks you must perform:

1. READ src/pre_registration.md and follow it strictly.
2. CREATE src/itag.py exactly as specified in the prompt.
3. MODIFY src/environment.py to implement the structured distractor (Sinusoidal Oscillator):
   - Add `structured_distractor=False` as a parameter to PhysicsSandbox's `__init__`.
   - Store it as `self.structured_distractor`.
   - Initialize state variables in `__init__`: `self.sd_center = 64.0`, `self.sd_amplitude = 10.0`, `self.sd_omega = 0.03`, `self.sd_phase = 0.0`, `self.sd_color = np.array([0.5, 0.5, 0.5])`, `self.sd_radius = 5.0`, `self.sd_t = 0`, `self.sd_pos = 64.0`.
   - Initialize these variables in `reset()` using np.random (if structured_distractor is True):
     - `self.sd_center` uniform in [32.0, 96.0]
     - `self.sd_amplitude` uniform in [5.0, 15.0]
     - `self.sd_omega` uniform in [0.02, 0.05]
     - `self.sd_phase` uniform in [0, 2 * pi]
     - `self.sd_color` uniform in [0.3, 1.0] for 3 channels
     - `self.sd_radius` uniform in [3.0, 8.0]
     - `self.sd_t = 0`
     - `self.sd_pos` initialized using the t=0 equation: `self.sd_center + self.sd_amplitude * np.sin(self.sd_phase)`
   - In `step()`, update when structured_distractor is True:
     - `self.sd_t += 1`
     - `self.sd_pos = self.sd_center + self.sd_amplitude * np.sin(self.sd_omega * self.sd_t + self.sd_phase)`
   - Add to `info` dict in `step()` when structured_distractor is True:
     - `info["sd_pos"] = self.sd_pos`
     - `info["sd_color"] = self.sd_color.copy()`
     - `info["sd_radius"] = self.sd_radius`
   - In `render()`, render the structured distractor after all standard objects and pointer are rendered, using the same soft sigmoid continuous blending logic:
     ```python
     if self.structured_distractor:
         pos = self.sd_pos
         r = self.sd_radius
         color = self.sd_color
         d = np.abs(pixel_centers - pos)
         mask = 1.0 / (1.0 + np.exp((d - r) / self.sigma_blur))
         mask = mask[np.newaxis, :]
         color_expanded = color[:, np.newaxis]
         canvas = canvas * (1.0 - mask) + color_expanded * mask
     ```
4. CREATE src/run_phase19_experiments.py:
   - Adapt the structure of `src/run_phase18_experiments.py` to support three arms:
     - **Arm A (Baseline)**: WUP-MDL (W=100) with no ITAG pre-filter. Gated strictly by MDL Ratio < 1.0.
     - **Arm B (ITAG+MDL)**: ITAG pre-filter (tau=0.3, W_t=20) + WUP-MDL (W=100).
       - At step 1800, compute ITAG using the 20 raw pixel frames up to step 1800 (steps 1781..1800).
       - If ITAG > 0.3, enter WUP probation for steps 1801..1900, set d_t = 4, reset error buffers, record WUP errors. At step 1900, accept if MDL Ratio < 1.0 (else revert d_t = 3).
       - If ITAG <= 0.3, immediately reject (no WUP probation, d_t = 3, next_proposal_check = 1850).
     - **Arm C (ITAG-only)**: ITAG-only gating (tau=0.3, W_t=20), no WUP.
       - At step 1800, compute ITAG using the 20 raw pixel frames up to step 1800 (steps 1781..1800).
       - If ITAG > 0.3, immediately recruit (set d_t = 4, accept=True).
       - If ITAG <= 0.3, reject (no recruitment, d_t = 3, next_proposal_check = 1850).
   - Support three sweeps:
     1. `transition`: N=3 -> N=4 clean objects (with 4th object mass doubled)
     2. `control` (Noisy-TV): N=3 clean + 1 Noisy-TV entity
     3. `structured_distractor`: N=3 clean + 1 Sinusoidal Oscillator (with structured_distractor=True)
   - Use the pre-existing passive cached models in `cache/` (like train_passive_cached does) to avoid passive training from scratch.
   - For all steps from 1781 to 1800, compute and log both ITAG and ISAG scores at every single timestep. This will give a distribution of 20 scores per seed per condition, allowing robust Cohen's d computation!
   - To compute ITAG and ISAG:
     - Get the surprising positions using `identify_surprising_positions` with `a_spatial = model.encoder.forward_spatial(target_t)` and `prediction_error_map = a_spatial.norm(dim=1)`.
     - Collect raw pixel frames from `branch_history` (which contains elements of shape (3, 128)).
     - Compute ITAG and ISAG scores using `compute_itag()` and `compute_isag()`.
   - Perform the Falsification Audit after completing the 45 branches (3 sweeps x 3 arms x 5 seeds):
     - Compute Cohen's d between ITAG distributions (20 steps x 5 seeds = 100 values per sweep per arm or overall) for:
       - C1: Genuine Transition vs Noisy-TV
       - C2: Genuine Transition vs Structured Distractor
     - Compute false recruitment rates and genuine recruitment rates for all arms.
     - Evaluate C3, C4, C5 (Scope-Reduction Trigger).
   - Save all results:
     - `archive/iter_019/results/summary_phase19.csv`
     - `archive/iter_019/results/adaptation_curves_phase19.png`
     - `archive/iter_019/results/audit_results_phase19.json`
     - `archive/iter_019/results/phase19_report.md`
5. RUN the experiments by executing `python src/run_phase19_experiments.py`. Verify everything completes without error. Ensure all required files and plots are generated and saved correctly.
