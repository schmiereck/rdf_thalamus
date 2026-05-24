# Current Research State
Phase: Phase 5 Complete (Active Probing vs. Passive Observation Evaluated)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 5 goal was to compare Passive Observation (Control) against Active Probing (Experimental) during the N=3 -> N=4 generalization transition under a 5-seed comparison sweep, keeping representation training 100% unsupervised.

## Confirmed
- RELIABLE RECRUITMENT TIMELINE (iter_007.1): Active physical probing of a novel entity triggers capacity recruitment with 100% reliability at step 1501 exactly, whereas passive observation is erratic (60% recruitment rate, with 2 of 5 seeds failing to recruit completely).
- COGNITIVE DECODABILITY ADVANTAGE (iter_007.1): Post-hoc linear readouts on frozen representations show that active probing consistently improves the decodability of the novel object's physical coordinates across 100% of the seeds, achieving a 19.9% average reduction in position prediction MSE.
- REPRESENTATION DECORRELATION (iter_007.1): Active physical interaction acts as a powerful decorrelation force, reducing cross-dimension latent overlap by 29.8% (average r_cross decreased from 0.439 to 0.308), indicating that active probing forces the recruited dimension to represent unique, non-redundant state spaces.

## Refuted
- REFUTED (iter_007.1): Unsupervised active probing universally guarantees a high absolute Pearson correlation (|r| >= 0.40) or a large correlation improvement (\Delta |r| >= 0.25) on a single isolated recruited dimension. Due to high seed-to-seed variance, absolute correlation ranged from 0.005 to 0.562, indicating that spatial coordinates sometimes distribute across multiple dimensions.

## Best Result
- Active Probing Dimension Recruitment Rate: 100% (iter_007.1)
- Active Probing Average Position Decoding MSE: 73.65 (vs. 91.97 for Passive) (iter_007.1)
- Active Probing Cross-Dimension Decorrelation: 0.308 (vs. 0.439 for Passive) (iter_007.1)

## In Progress
- Investigating coordinate-pooling constraints or spatial bottlenecks to eliminate seed-to-seed variance in coordinate alignment.

## Open Questions
- How can we introduce a relative coordinate bottleneck or coordinate-pooling constraint to stabilize the alignment of spatial coordinates onto a single dedicated recruited dimension across all random seeds?
- Does the output-as-input loop (Pillar D's self-generated attention) interact with Active Probing to further stabilize spatial representations?
