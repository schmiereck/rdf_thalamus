Let's fix the bug in src/run_phase19_experiments.py and execute the full experiment suite.

Here is the exact task:
1. In `src/run_phase19_experiments.py`, edit the ITAG gating block at step 1800 so that Arm A is NOT subjected to any ITAG gating and proceeds to WUP probation unconditionally. Use `arm_name` to distinguish Arm A and Arm B.
   Specifically, update the `if step == 1800:` gating block:
   ```python
            if gating == "itag_only":
                # Arm C: ITAG-only gating
                if itag_score > 0.3:
                    active_transition_accepted = True
                    active_transition_accepted_step = step
                    branch_model.d_t = 4
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    post_recruitment_audit_active = True
                    post_recruitment_audit_step_count = 0
                    print(f"       [{arm_name} ITAG-only @ step {step}] ITAG={itag_score:.4f} > 0.3 -> IMMEDIATELY RECRUITED (3->4)")
                else:
                    probationary = False
                    branch_model.d_t = 3
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    active_transition_accepted = False
                    next_proposal_check = step + 50
                    print(f"       [{arm_name} ITAG-only @ step {step}] ITAG={itag_score:.4f} <= 0.3 -> IMMEDIATELY REJECTED")
            elif "Arm B" in arm_name:
                # Arm B: ITAG+MDL
                if itag_score > 0.3:
                    probationary = True
                    probation_end_step = step + wup_window
                    branch_model.d_t = 4
                    branch_model.steps_since_recruitment = 0
                    branch_model.reset_error_buffer()
                    wup_errors = []
                    print(f"       [{arm_name} ITAG+MDL @ step {step}] ITAG={itag_score:.4f} > 0.3 -> WUP PROBATION STARTED (W={wup_window}, end={probation_end_step})")
                else:
                    probationary = False
                    branch_model.d_t = 3
                    active_transition_accepted = False
                    next_proposal_check = step + 50
                    print(f"       [{arm_name} ITAG+MDL @ step {step}] ITAG={itag_score:.4f} <= 0.3 -> REJECTED BY ITAG PRE-FILTER")
            elif "Arm A" in arm_name:
                # Arm A: WUP-MDL Baseline (no ITAG pre-filter)
                probationary = True
                probation_end_step = step + wup_window
                branch_model.d_t = 4
                branch_model.steps_since_recruitment = 0
                branch_model.reset_error_buffer()
                wup_errors = []
                print(f"       [{arm_name} Baseline @ step {step}] WUP PROBATION STARTED (W={wup_window}, end={probation_end_step})")
   ```
2. Save and run `python src/run_phase19_experiments.py`.
3. Verify that the experiments complete successfully and that the outputs (the report, csv, png, and json) are correctly written.
4. Let's inspect the results in detail and print the summary.
