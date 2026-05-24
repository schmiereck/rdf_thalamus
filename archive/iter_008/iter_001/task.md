1. Read src/pre_registration.md and append Criterion 5: 'Criterion 5 (Non-Collapse & Activity Threshold): The hypothesis is falsified if the recruited channel k undergoes representation collapse. Specifically, its mean activation magnitude must remain active (e.g., E[|a_k|] >= 0.1 * 1/C sum_c E[|a_c|]) and its temporal standard deviation must be non-trivial (std(x_mean_k) > 5.0 pixels), proving it has not collapsed into an inactive channel or a static, non-responsive spatial spike.'
Also pre-register the hyperparameter search space lambda in {0.01, 0.1, 1.0} and state the default lambda = 0.1 selection. Commit these changes to src/pre_registration.md.

2. Create src/models_spatial.py containing the updated DynamicJEPASpatial architecture. It should have:
  - An Encoder whose forward_spatial(x) returns a (B, d_max, 128) spatial feature map by mapping (B, 3, 128) through conv1-4 to (B, 128, 8), applying a Conv1d(128, d_max, 1) and linear interpolation to size 128. Its forward(x) should return the global average (mean) of the spatial feature map over the spatial dimension, returning a (B, d_max) vector.
  - A helper method or function to calculate centroid and variance for any channel:
    p_c = F.softmax(a_spatial, dim=-1) # shape (B, d_max, 128)
    coords = torch.arange(128, device=a_spatial.device, dtype=torch.float32)
    x_mean = torch.sum(coords * p_c, dim=-1) # shape (B, d_max)
    var = torch.sum(((coords.unsqueeze(0).unsqueeze(1) - x_mean.unsqueeze(-1)) ** 2) * p_c, dim=-1) # shape (B, d_max)
  - The forward pass should compute the standard VICReg losses, and additionally, if we pass a spatial bottleneck weight lambda_spatial > 0, it should add loss_spatial = lambda_spatial * Var_k to the loss, where k = d_t - 1 (the recruited channel index), or a specified channel k. Make sure to return loss_spatial in the output loss dict.
  - Retain the GDASR recruitment logic from DynamicJEPA.

3. Write a small unit test script test_models_spatial.py and run it to verify that the forward pass works and that backprop through the spatial loss only affects the k-th channel of the spatial conv layer.