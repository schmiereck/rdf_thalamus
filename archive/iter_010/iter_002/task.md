1. Write src/run_phase10_experiments.py, which executes the systematic 5-seed sweep of Phase 10:
  - Seeds: [42, 123, 456, 789, 999].
  - Environments: PhysicsSandbox with parameterizable variation.
  - Three Arms:
    a. Arm A: Gentle static bottleneck (lambda_spatial=0.01), single-stream.
    b. Arm C: Dynamic single-stream DSMC.
    c. Arm D: Dual-Stream Decoupled Thalamus (DSDT) using DualStreamJEPASpatial and dynamic lambda_spatial weight.
  - For each seed:
    1. Passive N=3 training for 1500 steps on both standard DynamicJEPASpatial and DualStreamJEPASpatial.
    2. Clone base models and train Arm A, Arm C, and Arm D on N=4 environment for steps 1501 to 3000.
    3. Update pointer target pos using recruited channel 3's soft centroid in closed-loop control.
    4. Save learning curves (lambda, surprise) for plotting.
    5. Evaluate each branch on a fresh N=4 test set of 200 transitions. For Arm D, also evaluate with mask_coord=True to prove the Construction-vs-Empirical information flow.
  - Compile average metrics (Pearson |r| centroid, linear probe decoding MSE, soft spatial variance, test simulation prediction loss, and representation collapse rates).
  - Save summary_phase10.csv to archive/iter_010/results/summary_phase10.csv.
  - Generate trajectories and performance comparison plots and save to archive/iter_010/results/.
  - Audit results against the 3 pre-registered Falsification Criteria, and perform the information flow control test (comparing masked vs unmasked test loss for Arm D).
  - Save a comprehensive report to archive/iter_010/results/phase10_report.md.
2. Run src/run_phase10_experiments.py and log its complete output.