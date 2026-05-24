1. Execute the python script src/run_phase8_experiments.py using Python.
This will run the full 5-seed sweep across the 5 branches, generate the summaries, plot, and the comprehensive markdown report.
2. Confirm that the following output files are successfully written:
   - archive/iter_008/results/summary_phase8.csv
   - archive/iter_008/results/performance_comparison.png
   - archive/iter_008/results/phase8_report.md
3. Print or extract the final average metrics for each of the 5 configurations (Pearson correlation |r|, linear decoding MSE, soft spatial variance, recruitment rate, and collapse/non-collapse counts) so we can review them in our analysis.