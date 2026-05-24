1. Write the pre-registration file `src/pre_registration.md` according to the corrected hypothesis, falsification criteria, and methods.

2. Update `src/models.py`:
   - Change default `cov_weight` to `25.0` in the `forward` signature of both `FixedJEPA` and `DynamicJEPA`.
   - Import `collections` if not already imported.
   - Change `self.error_buffer = []` to `self.error_buffer = collections.deque(maxlen=500)` in `DynamicJEPA.__init__`.
   - Add a method to `DynamicJEPA`:
     ```python
     def reset_error_buffer(self):
         self.error_buffer.clear()
         self.ema_error = None
     ```

3. Update `src/train.py`:
   - Enforce the 1000-step representation-warmup across all models. Specifically:
     - During `step` from 1 to 1000, do NOT call `update_recruitment_logic` for the `dynamic` model.
     - At `step == 1001`, if `model_type == 'dynamic'`, call `model.reset_error_buffer()`.
     - Only call `model.update_recruitment_logic(sim_loss_val)` when `step > 1000` (so after warmup).
     - Ensure `cov_weight=25.0` is passed / used for all models (b1, b1_large, dynamic) in both training and evaluation forward passes.
     - Do NOT programmatically reset or clear the error buffer or change any parameters during the N=2 to N=3 transition at step 1501. The sliding window of size 500 must discard the older, lower-error history naturally.
     - Change all hardcoded output directories from `archive/iter_002` to `archive/iter_003` (including runs, results, plots, and summary.csv).

4. Run `python src/test_integration.py` to verify the code works and integration tests pass successfully. If there are any errors or issues, fix them.

5. Run `python src/train.py` to perform the 15-experiment evaluation suite. It will save all run CSVs, the final summary.csv, and learning curves plot to `archive/iter_003`.

6. Log a message confirming completion and print the aggregated results from summary.csv.