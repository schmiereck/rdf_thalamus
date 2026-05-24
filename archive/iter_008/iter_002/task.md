1. Create the experiment script src/run_phase8_experiments.py.
This script must:
  - Implement a 5-seed comparative sweep using seeds [42, 123, 456, 789, 999].
  - For each seed:
    a. Train a base DynamicJEPASpatial passively on N=3 for 1500 steps to let dt increase from 2 to 3 and stabilize.
    b. Clone this trained base model into 5 branches:
       - Control (Ground-Truth targets, lambda_spatial=0.0)
       - Exp lambda=0 (Output-as-input targets, lambda_spatial=0.0)
       - Exp lambda=0.01 (Output-as-input targets, lambda_spatial=0.01)
       - Exp lambda=0.1 (Output-as-input targets, lambda_spatial=0.1)
       - Exp lambda=1.0 (Output-as-input targets, lambda_spatial=1.0)
    c. Train each branch on N=4 for 1500 steps (steps 1501 to 3000).
       - During N=4 training, update the controller target every step:
         * Control: target_pos = env.positions[3]
         * Exp branches: target_pos = model.encoder.forward_spatial(obs)[0, 3].item() (using the spatial centroid computed from the current step's 1D image of the novel object represented by recruited channel 3).
       - For the Exp branches, the loss must include the spatial bottleneck penalty:
         loss_dict, _, _ = model(x_hist_t, x_target_t, lambda_spatial=lambda_spatial, k_chan=3).
       - Log the step where the 4th dimension is recruited (dt increases to 4).
    d. Evaluate each branch on a fresh N=4 test set of 200 transitions from seed + 5000:
       - Measure the Pearson correlation |r| between channel 3's centroid (and activation) and the true coordinate of the 4th object (y_4).
       - Measure the post-hoc linear decoding MSE of y_4 from channel 3's activation.
       - Measure the average soft spatial variance of channel 3's activations over the test set.
       - Check Criterion 5:
         * Activation magnitude E[|a_3|] >= 0.1 * E[|a_all|] (where E[|a_all|] is the average absolute activation across active channels 0 to 3).
         * Centroid temporal standard deviation std(x_mean_3) > 5.0 pixels.
         * Report whether representation collapse occurred based on these.
  - Compile the results across all seeds, calculate means and confidence intervals or standard deviations, and save the results as a CSV at archive/iter_008/results/summary_phase8.csv.
  - Generate a performance comparison plot and save it to archive/iter_008/results/performance_comparison.png.
  - Write a comprehensive Markdown report saved as archive/iter_008/results/phase8_report.md that documents the results, compares against the 5 pre-registered Falsification Criteria, and performs the sensitivity analysis across lambda.

2. Run the script src/run_phase8_experiments.py and ensure it runs successfully, logging its output.