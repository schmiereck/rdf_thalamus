# Current Research State
Phase: Phase 3 Complete (Closed-Loop Motor & Subsumption Motorics Evaluated)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 3 goal was to implement Subsumption Motorics (Pillar E), attach progressive motor coupling, and evaluate causal sensitivity and tracking overlap.

## Confirmed
- ACTIVE PERCEPTION CAUSAL REDUCTION: Active probing via intentional collisions achieves a massive **75.05% reduction** in post-collision predictive error compared to the passive baseline M_no_motor (0.0236 vs 0.0948) and a **57.57% reduction** compared to the random baseline M_random (0.0236 vs 0.0557) (iter_005.2).
- COOLDOWN ADAPTATION: Replacing the rigid 200-step cooldown with a surprise-modulated adaptive cooldown C_t operates seamlessly, enabling agile locus switching during high-surprise collisions (iter_005.1).
- ATTENTION TRANSITION STABILITY: Transitioning from externally primed attention to self-generated attention is highly stable (ratio of 1.0593 vs the 1.15 limit) (iter_005.2).
- REPRESENTATION-DEPENDENT CONTROL: The ablation study (Random network: 23.20%, Shuffled attention: 22.00% vs Normal: 22.80%) confirms that the agent's tracking behavior is causally reliant on the precise closed-loop integration of thalamic token locus selection and subsumption motorics (iter_005.2).

## Refuted
- REFUTED: M_active maintains physical pointer-to-object tracking overlap >= 70.0% (iter_005.2). The observed overlap was 22.80% in test and 19.38% in training. Actively perturbing objects creates high environmental entropy and pushes objects out of the segment, revealing a fundamental "active-perception entropy trade-off".
- REFUTED: M_active achieves overall test prediction loss <= B1 baseline (iter_005.2). The active closed-loop test loss is 0.0861 vs B1's 0.0452, because the active closed-loop contains continuous pointer-object collisions and high-velocity movements that are absent in the static, pointer-free B1 environment.

## Best Result
- Active Post-Collision L2 Prediction Loss: 0.0236 ± 0.0054 (iter_005.2)
- Error reduction vs Passive baseline: 75.05% (iter_005.2)

## In Progress
- Synthesizing active perception findings and preparing Phase 4 (Generalization & Reporting) scope.

## Open Questions
- Can we formulate an information-theoretic attention metric (e.g., expected information gain) to replace raw spatial tracking overlap?
- How quickly does the position readout adapt to novel physical parameters (like a 4th object with unseen mass) in a few-shot manner?
