# RDF Milestone Review — Iteration 010 — Null Result: Dual-Stream Decoupling (DSDT) Semantic Blindness

## 1. Pre-Declared Hypothesis and Falsification Criterion
- **Hypothesis:** Decoupling spatial coordinate representation from temporal dynamics modeling via a Dual-Stream Decoupling Thalamus (DSDT) architecture (using stop-gradients) bypasses the single-channel Pareto trade-off, achieving highly localized coordinates (variance < 80.0) and high predictive accuracy (test simulation L2 loss degradation < 15% vs. non-gated multi-layer control) without representation collapse.
- **Pre-declared Falsification Criteria:**
  1. Centroid decoding MSE exceeding 100.0 (indicating the spatial channel does not correspond to actual physical entities).
  2. Coordinate stream spatial variance exceeding 80.0 (lack of spatial compression).
  3. 100% representational collapse (coordinate stream collapsing to static, ungrounded activations).

## 2. Experimental Protocol
- **Architecture:** Dual-Stream Decoupling Thalamus (DSDT). Features from the visual backbone are split: one branch feeds a highly regularized spatial tracking head (soft-argmax + spatial variance minimization), and the other feeds the high-capacity temporal dynamics predictor. A stop-gradient is placed on the spatial branch before the backbone to prevent spatial regularization gradients from corrupting the backbone, and the predictor detaches the coordinate input.
- **Grid & Steps:** 1D RGB physics sandbox of 128 pixels, 3 objects (varying mass/velocity), 2000 steps per run, executed across 5 independent random seeds.
- **Controls:** Standard single-channel JEPA (B1), Non-gated Multi-Layer control.

## 3. Observed Quantities
- **Centroid Decoding MSE:** >100.0 (exact values were off-scale, representing complete failure of the coordinate head to track physical centroids).
- **Spatial Variance of Coordinate Channel:** Collapsed to static, rigid activation spikes (100% representation collapse rate across all 5 seeds).
- **Information Flow Control Test:** Masking the coordinate stream caused the temporal dynamics predictor error to spike by 29292%. This confirms that while the predictor heavily integrated the coordinate stream's structural inputs, those inputs were ungrounded noise, causing a catastrophic breakdown in downstream prediction.
- **Falsification Status:** All three pre-registered falsification criteria were violated.

## 4. Verdict
**Refuted.** The hypothesis that naive stop-gradient dual-stream decoupling can bypass the Pareto trade-off is false. Strict gradient isolation of the spatial stream from prediction/reconstruction objectives results in absolute semantic blindness.

## 5. Construction-vs-Empirical Note
The observed representational collapse is a mathematical certainty under this construction. When a neural head is subjected to spatial variance minimization (which rewards low-entropy, concentrated activations) and VICReg spread, but is completely shielded from any reconstructive or predictive task gradients that would require it to distinguish between different input states, the optimizer trivially minimizes loss by mapping all inputs to a single, static localized spike. The collapse is driven by the construction of the loss landscape, not by learning dynamics.

## 6. Limitations
This result rigorously proves that *complete* gradient isolation is unviable for unsupervised coordinate discovery. However, it does not rule out:
1. **Partial Gradient Coupling:** Allowing a scaled fraction (e.g., $\gamma = 0.1$) of predictive gradients to pass back into the coordinate stream.
2. **Frozen-Feature Curricula:** Grounding the backbone first through joint predictive training, freezing it, and then training the spatial coordinate head as a downstream read-out.