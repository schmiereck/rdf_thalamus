# Current Research State
Phase: Phase 18 Complete (EG-MDL Falsified, Cold-Start Optimization Transient Discovered)

## Goal
Design and evaluate a novel dynamic representation network ("Thalamus") with hierarchical
abstraction, dynamic dimension recruitment, and thalamic gating. Current focus: solving the
distractor-rejection problem for dimension recruitment gating.

## Confirmed
- FALSIFICATION OF EG-MDL HYPOTHESIS (iter_18.2): The prediction-trend gate ρ = E_late/E_early
  provides zero discriminative power between genuine objects and Noisy-TV. All arms (P, S, S_alt)
  recruit the Noisy-TV distractor in 5/5 seeds. Falsified on Criterion C2 (false recruitment > 20%).
- COLD-START OPTIMIZATION TRANSIENT (iter_18.2): When the predictor's 4th-dimension weights are
  randomly initialized, E_early is enormous (26-314 across seeds) and E_late drops to near-zero
  (0.07-0.43) regardless of signal type, driving ρ to ~0.001-0.006. This transient dominates
  any learnability signal and renders ρ useless as a discriminative metric.
- WUP-MDL REMAINS BEST BASELINE (iter_18.2): Arm P achieves 100% recruitment (5/5), centroid
  MSE 55.58, test sim loss 0.0877. But 100% false recruitment on Noisy-TV persists.
- NO θ-SENSITIVITY (iter_18.2): Both θ=0.90 and θ=0.85 are universally passed (ρ < 0.01
  << 0.85). The problem is metric confound, not threshold calibration.
- THREE DISTINCT COLD-START PATHOLOGIES (iter_17, iter_18):
  1. Encoder cold-start (Phase 17, ESUG): Chaotic encoder output → gate rejects genuine objects
  2. Predictor cold-start (Phase 15-16, MDL): Cold predictor fails MDL → gate rejects (solved by WUP)
  3. Optimization transient cold-start (Phase 18, EG-MDL): Rapid weight adaptation → ρ→0 for all signals

## Refuted / Falsified
- PREDICTION-TREND GATE ρ (iter_18.2): Cannot distinguish structured from unstructured signals
  when the predictor is cold-started because the optimization transient dominates.
- ESUG (iter_17.1): Encoder-only gating without warm-up rejected genuine objects due to encoder
  cold-start.
- PVU GATING (iter_16): Physical-variance-uncorrelated gating rejected all dimensions due to
  high physical correlation in 1D space.
- SA-CCR (iter_15): Surprise-adaptive covariance regularization caused destructive interference
  between learning and attention signals.

## Best Result
- Arm P (WUP-MDL, W=100): Centroid Decoding MSE: 55.58, Test Sim Loss: 0.0877, 100% recruitment
  (iter_18.2). But 100% false recruitment on Noisy-TV control.

## In Progress
- Designing a post-transient ρ metric with extended WUP window (W=500) to test whether
  learnability differences emerge after cold-start transient saturation.
- Exploring warm-started predictor approaches to bypass the optimization transient entirely.

## Open Questions
1. Can a post-transient ρ (computed on steps 250-375 vs 375-500 of W=500 WUP) separate
   genuine objects from Noisy-TV?
2. Can warm-starting the predictor by weight transfer from existing dimensions avoid the
   cold-start optimization transient?
3. Is there a non-predictor-based gating metric (spectral flatness, spatial coherence,
   cross-dimension MI) that avoids all cold-start pathologies?
4. Should the Noisy-TV scenario be treated as fundamentally out-of-scope for predictor-based
   gating, requiring architectural solutions instead?
5. Can pre-allocated shadow dimensions (always encoding but only sometimes promoted)
   provide an alternative recruitment paradigm?
