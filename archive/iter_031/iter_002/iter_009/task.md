Execute the full 60-run production suite (3 arms x 20 seeds) by running the training script:
`python src/run_iter031_partA.py --workers 6`

This will train and evaluate all models and seeds.
Make sure to monitor progress, ensure no worker crashes, save all results and checkpoints to the correct locations under `archive/iter_031/results/`, and finally generate the summary CSV and markdown analysis file `archive/iter_031/results/partA_analysis.md`.
Report the final results and summary stats when finished.