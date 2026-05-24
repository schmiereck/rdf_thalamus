Create 'src/thalamus.py' and implement the Thalamic Gated representation network and a non-gated control network.

Follow these specific instructions precisely to ensure alignment with the pre-registration and strategic corrections:

1. Class definitions:
   - SegmentEncoder(d_max=8): Maps segment '(B, 3, 32)' to '(B, d_max)' using 1D convolution layers.
   - SegmentPredictor(d_max=8, h=3): Predicts next segment latent state from a history of h * d_max values.
   - L2Encoder(d_max=8): Maps aggregated/concatenated active L1 latents of all 4 segments (total size 4 * d_max) to L2 latent space of size d_max.
   - L2Predictor(d_max=8, h=3): Predicts next L2 latent state.
   - ThalamusNet(d_max=8, h=3, cooldown=200):
     - Implements 4 L1 segments (each of size 32) and 1 global L2 layer.
     - Implements local temporal surprise calculation for each L1 segment and L2.
     - Surprise Watchdog: Keeps EMA of mean and variance of raw surprise for all 5 loci (4 L1 segments, 1 L2), and implements a Soft-Z-Score Normalization using var_floor=1e-4 and epsilon=1e-4.
     - Robust Relative Stability Lock: Keeps track of overall L1 running surprise EMA and minimum recorded L1 surprise. Computes 'theta_conv = max(0.25, 1.5 * L1_surprise_min)'. If L1 running surprise is greater than theta_conv, L2 is locked (attention token cannot be routed to L2 and L2 gradients are frozen/untrained).
     - Token Routing & Cooldown: Tracks active token locus in {0, 1, 2, 3, 4}. Implements a cooldown of 200 steps (token cannot change unless 200 steps have passed since last change or at startup).
     - Plasticity Gating: When backpropagating, only the layer holding the token has gradients enabled and weights updated. Enable/disable gradients dynamically or gate the optimizer step.
     - Priming & Self-Generation: Supports external query (color of target object) and self-generated query (from a linear color_readout mapping L2 latent to 3 RGB values, with stop-gradient on L2). Computes segment color similarity and biases L1 segment surprise.
     - Physical Tracking Overlap: Maps current attention locus to 1D sandbox physical coordinate and returns 1.0 if it falls within the same 32-pixel segment as the target object, else 0.0.
   - NonGatedControlNet(d_max=8, h=3):
     - Parallel structure to ThalamusNet (2 layers, 4 segments in L1, 1 L2), but with BOTH layers continuously trained without any Thalamic gating, stability lock, token routing, or plasticity gating. It still computes the physical tracking overlap for evaluation purposes using the color query.

2. Create a test section or a separate test file to verify that ThalamusNet and NonGatedControlNet compile, can perform forward passes on dummy tensors of shape (B, H, 3, 128) and target tensors of shape (B, 3, 128), can run backprop, and successfully gate gradients (i.e., verifying that when L1 holds the token, L2 parameters receive zero gradients, and vice versa).

Verify and execute this task. Write robust, clean, and well-commented code.