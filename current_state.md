# Current Research State
Phase: Phase 8 Complete (Spatial Bottleneck & Closed-Loop Active Probing Swept)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") inside a 1D physics environment. Phase 8 goal was to implement and evaluate an unsupervised spatial localization bottleneck coupled with closed-loop output-as-input active probing across a 5-seed comparative sweep.

## Confirmed
- REPRODUCIBLE VARIANCE REDUCTION (iter_008.3): Minimizing the soft spatial variance of the recruited channel reduces its spatial activation spread by 95.0% (from 1209.33 to 60.29 for lambda=0.1), forcing the recruited channel to act as a highly localized spatial spotlight.
- COMPLETE COLLAPSE PREVENTION (iter_008.3): Introducing the local spatial variance penalty completely eliminates representation collapse, reducing the collapse rate from 40.0% in Control to 0.0% across all 15 runs of the active-bottleneck branches (lambda in [0.01, 0.1, 1.0]).
- CRITICAL LOCALIZATION-CAPACITY TRADE-OFF (iter_008.3): There is a clear trade-off between spatial localization and cognitive predictive capacity. A gentle bottleneck (lambda = 0.01) yields the best coordinate decoding MSE of 69.11 (a 16.9% reduction over Control's 83.12, and a 6.2% reduction over Phase 7's 73.65). Stronger bottlenecks (lambda >= 0.1) restrict the representations too severely, causing decoding MSE to degrade (106.87 for lambda=0.1).

## Refuted
- REFUTED (iter_008.3): A strong unsupervised spatial bottleneck (lambda >= 0.1) universally improves the post-hoc linear decoding accuracy of physical coordinates. Instead, excessive regularization degrades coordinate decodability by restricting the predictive capacity of the latent channel.
- REFUTED (iter_008.3): Unsupervised active probing and spatial bottlenecks can easily achieve an absolute Pearson correlation of |r| >= 0.60 or post-hoc decoding MSE < 55.0 within 1500 steps. High-dimensional coordinate alignment onto a single dimension remains noisy (average |r| = 0.2907, minimum seed |r| = 0.0059 for lambda=0.1).

## Best Result
- Active-Bottleneck (lambda=0.01) Average Centroid Decoding MSE: 69.11 (vs. 83.12 for Control) (iter_008.3)
- Active-Bottleneck (lambda=0.1) Soft Spatial Variance: 60.29 (vs. 1209.33 for Control) (iter_008.3)
- Active-Bottleneck Representation Collapse Rate: 0.0% (vs. 40.0% for Control) (iter_008.3)

## In Progress
- Exploring adaptive, surprise-modulated curriculums for lambda to resolve the localization-prediction trade-off.

## Open Questions
- Can an adaptive curriculum for lambda (starting at 0 and rising to 0.01 as surprise falls) achieve tight localization without degrading predictive accuracy?
- How does adding multi-scale temporal predictive context (predicting multiple steps into the future) affect coordinate specialization?
