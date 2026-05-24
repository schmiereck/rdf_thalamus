Update `src/pre_registration.md` and implement the 5-arm Phase 11 sweep script `src/run_phase11_experiments.py` to evaluate the Plasticity-Adaptability Conflict in Arm E and evaluate Arm F's performance.

1. Update `src/pre_registration.md`:
   - Keep the original hypothesis and falsification criteria, but add the updated ones requested by the Strategic Research Manager's Notes:
     - **Generalization Penalty for Arm E (PDRC)**: Arm E must be falsified if, upon introducing a 4th novel object in Stage 2 (after $T_{ground}=1500$), its coordinate-centroid correlation drops below $0.25$ or its centroid decoding MSE exceeds $85.0$ (proving that freezing weights breaks adaptation to novelty).
     - **Arm F (Non-Parametric Soft-Argmax Projection) Evaluation**: Introduce Arm F's hypothesis: by deriving coordinates directly as a differentiable, non-parametric projection (soft-argmax) over the predictive dynamics channel, Arm F avoids gradient interference, remains fully grounded, and maintains full plasticity/adaptability to novel objects without needing a non-biological freezing schedule. Compare Arm F against Arm E, Arm D, and Arm A.

2. Implement `src/run_phase11_experiments.py` with 5 arms:
   - **Arm A**: Gentle single-stream bottleneck ($\lambda = 0.01$).
   - **Arm C**: Dynamic single-stream DSMC ($\lambda = \text{dynamic}$).
   - **Arm D**: Dual-Stream Decoupled Thalamus (DSDT) with immediate decoupling ($\lambda = \text{dynamic}$).
   - **Arm E**: Progressive Decoupling with Representational Consolidation (PDRC) ($\lambda = \text{dynamic}$). Stage 1 on $N=3$ ($0 \le t < 1500$) runs jointly with all weights active. Stage 2 on $N=4$ ($1500 \le t < 3000$) freezes the coordinate head weights (`encoder.conv_spatial_coord`) and detaches coordinates before the predictor.
   - **Arm F**: Non-Parametric Soft-Argmax Projection ($\lambda = \text{dynamic}$). It uses a single-backbone encoder (`NonParametricJEPASpatial`) and derives $z^{coord}$ and $z^{dyn}$ non-parametrically from the same spatial activation maps.
   - For each seed (42, 123, 456, 789, 999):
     - Train 4 dedicated base models passively on $N=3$ for 1500 steps: Standard base model, Dual base model, PDRC base model (in `stage=1`), and NonParametric base model.
     - Clone/instantiate the 5 arms and train under active probing on $N=4$ for steps 1501 to 3000 (PDController tracks the 4th recruited channel, spatial bottleneck is applied to channel 3).
     - At step 1501, set Arm E's stage to 2 (`arm_e_model.stage = 2` which freezes the coordinate head weights and activates stop-gradients).
     - Evaluate all arms on a fresh test set of 200 passive steps on $N=4$.

3. Save results and generate plots:
   - Save the full results table to `archive/iter_011/results/summary_phase11.csv`.
   - Compile a comprehensive scientific report in `archive/iter_011/results/phase11_report.md`.
   - Generate Plot 1: Decoded vs. Ground Truth position of the 4th object for Seed 42 across all 5 arms. Save to `archive/iter_011/results/performance_comparison_phase11.png`.
   - Generate Plot 2: Surprise and Lambda trajectories for Arm C, D, E, F. Save to `archive/iter_011/results/dsmc_trajectories_phase11.png`.
   - Print the final mean table to stdout. Run the experiment to completion!