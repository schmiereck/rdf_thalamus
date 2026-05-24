Implement Arm E (PDRCJEPASpatial) and Arm F (NonParametricJEPASpatial) models in `src/models_dual_stream.py`.

1. `PDRCJEPASpatial` details:
   - Inherit from or implement similarly to `DualStreamJEPASpatial`.
   - It should have a flag/property `stage` (1 or 2).
   - In Stage 1:
     - The spatial feature map of the coordinate stream (`forward_spatial`) should NOT detach the input `x` before `conv_spatial_coord`. It should allow gradients to flow back to conv1-4.
     - The coordinate representations `z_hist_coord` and `z_target_coord` should NOT be detached before passing to the predictor or in similarity loss (allow prediction gradients to flow back to the coordinate stream head and conv1-4).
     - All parameters should have `requires_grad=True`.
   - In Stage 2:
     - Set `requires_grad=False` on `encoder.conv_spatial_coord` parameters.
     - The spatial feature map of the coordinate stream (`forward_spatial`) should detach `x` before `conv_spatial_coord`.
     - Inject a stop-gradient at the output of the coordinate stream before feeding into the predictor: detach `z_hist_coord` and `z_target_coord` (making them detached inputs/targets).
     - The dynamics stream and predictor should remain fully trainable.

2. `NonParametricJEPASpatial` details:
   - It uses a single encoder that produces `a_spatial` of shape `(B, d_max, 128)`.
   - $z^{dyn}$ is computed as the global average pooling of `a_spatial` over the spatial dimension: `a_spatial.mean(dim=-1)`.
   - $z^{coord}$ is computed as the non-parametric spatial soft-argmax (centroids) over `a_spatial`: `calculate_centroid_and_variance(a_spatial)[0]`.
   - Use the same `DualStreamPredictor` to predict both $z^{coord}$ and $z^{dyn}$ of the target frame.
   - All losses (similarity, variance, covariance on both streams) and spatial bottleneck loss are computed.
   - There are no separate coordinate encoder heads or stop-gradients between the coordinate stream and the encoder backbone. All encoder parameters are jointly trained.

3. Update `src/test_models_dual_stream.py` to add comprehensive tests for both new models, checking forward pass shapes, gradient flow/specificity in Stage 1 and Stage 2 of PDRCJEPASpatial, and the non-parametric gradient flow of NonParametricJEPASpatial. Run the tests to verify correct implementation.