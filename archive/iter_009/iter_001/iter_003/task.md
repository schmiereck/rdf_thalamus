You need to fix a small plotting bug in `src/run_phase9_experiments.py` and refine the scientific report text, then run the experiment.

Specifically:
1. Open `src/run_phase9_experiments.py`.
2. Find the line:
   `ax1.set_title("EWMA Surprise ($\bar{S}_t$) Trajectory for Arm C", fontsize=12, fontweight="bold")`
   Change it to:
   `ax1.set_title(r"EWMA Surprise ($\bar{S}_t$) Trajectory for Arm C", fontsize=12, fontweight="bold")`
   (Note the raw string prefix `r`!).

3. Find where the markdown report content is defined in `src/run_phase9_experiments.py`.
   Refine the text under Section 2 (Detailed Analysis of Falsification and Sanity Check Criteria) to accurately and scientifically address the borderline results:
   - For Criterion 2: State that Arm C achieved an average centroid decoding MSE of 71.4021, which marginally fails (falsifies) the aggressive adjusted pre-registered threshold of <= 70.0 (though it is a massive improvement over Arm B's 106.8739 and extremely close to Arm A's 69.1121).
   - For the Curriculum Activity Sanity Check: State that the average final penalty weight \lambda_T reached 0.0498, which is just below the pre-registered sanity check of 0.05. Explain that according to the pre-registered sanity check mandate, this must be reported as a borderline failure of the curriculum to fully activate, not a successful resolution of the trade-off. This scientific finding suggests that the curriculum parameters (such as the scaling factor \gamma or initial EWMA surprise) require slight tuning to fully optimize the transition profile.

4. Run `python src/run_phase9_experiments.py` to train all 3 arms across the 5 seeds, evaluate them, save the raw results to `archive/iter_009/results/summary_phase9.csv`, and save the completed figures and report in `archive/iter_009/results/`.

Double check that no other raw strings containing math expressions with backslashes are un-prefixed. Then, run the training and verify success.