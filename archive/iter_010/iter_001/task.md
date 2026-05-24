1. Create src/models_dual_stream.py, implementing the Dual-Stream Decoupled Thalamus (DSDT) architecture components:
  - calculate_centroid_and_variance helper function.
  - DualStreamEncoder class with forward_spatial, forward_dynamics, and forward methods.
  - DualStreamPredictor class with support for mask_coord option.
  - DualStreamJEPASpatial class, which subclasses nn.Module and implements the complete dynamic dual-stream JEPA model (with recruitment, stabilization, dual-stream VICReg loss, spatial variance loss, and self-cloning capabilities).
2. Create src/test_models_dual_stream.py to perform shape checks, forward pass checks, backprop specificity checks (ensuring backprop through loss_spatial only flows to the coordinate head of conv_spatial_coord, while prediction gradients on the dynamics stream flow to conv_spatial_dyn and conv1-4), and verify that mask_coord correctly zeros out coordinate input.
3. Run src/test_models_dual_stream.py and output the results.