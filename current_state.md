# Current Research State
Phase: Phase 13 Complete (Explicit Position Encodings Evaluated and Falsified)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 13 goal was to evaluate whether adding explicit pixel-position encodings (Linear Normalized or Sinusoidal) to the input of the convolutional backbone resolves the active-perception coordinate representation drift observed in Phase 12.

## Confirmed
- STATISTICAL NON-INFERIORITY NOT FORMALLY REJECTED (iter_13.4): Welch's t-test comparing post-collision test simulation loss at step 3000 did not reject non-inferiority under n=5 seeds for either Arm H (p=0.553) or Arm I (p=0.150) vs Arm G. However, the mean simulation losses were physically worse across the board, showing that explicit positional encoding degrades the optimization landscape.
- ROBUST SPATIAL COVERAGE (iter_13.4): Pointer spatial coverage entropy under CLTS active control remained highly stable and wide across all arms (G=3.955, H=3.955, I=3.952), showing that active probing and environment exploration were fully functional.

## Refuted / Falsified
- RE-IDENTITY/COORDINATE DRIFT UNMITIGATED (iter_13.4): The centroid decoding MSE of the novel object remained far above the falsification limit of 75.0 (Arm H: 87.50, Arm I: 88.11 vs Arm G: 85.85), proving that explicit positional input does not resolve representational drift under active physical perturbations (Criterion 1 Falsified).
- RECONSTRUCTION VS PHYSICS OPTIMIZATION INTERFERENCE (iter_13.4): Under sinusoidal positional encoding (Arm I), the post-collision test simulation loss degraded severely to 0.0911 (exceeding the 0.050 limit) and the training curve AUC worsened dramatically. This confirms that adding raw coordinate channels introduces an optimization shortcut that distracts the network from learning physical dynamics (Criterion 3 Falsified).
- LOSS OF SPATIAL REPRESENTATIONAL STABILITY (iter_13.4): Under linear positional encoding (Arm H), the soft spatial variance of the coordinate encoder expanded to 10.71 (exceeding the limit of 10.0), showing a complete destabilization of spatial tightness (Criterion 2 Falsified).

## Best Result
- Original RGB CLTS (Arm G): Test Sim Loss: 0.0483, Soft Spatial Variance: 8.67, Centroid Decoding MSE: 85.85, Pointer Spatial Entropy: 3.96.

## In Progress
- Phase 13 has successfully delivered a clear negative result: adding explicit coordinate channels creates a "position shortcut" that harms, rather than helps, unsupervised spatial bottle-necked dynamics networks.

## Open Questions
- Can contrastive coordinate regularization prevent active-perception representational drift without resorting to supervised coordinate readouts?
- What are the architectural trade-offs of transitioning the Thalamus project into a multi-dimensional (2D/3D) environment?
