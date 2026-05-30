Write the class `ReconVICRegSeparateDyn` into `src/models_recon.py` in the current project workspace.

The class should implement the Reconstruction+VICReg architecture:
- Encoder is `SeparateDynEncoder` from `src/models_separate_dyn.py`.
- Decoder deconv head on dyn spatial features maps (B, d_max, 8) -> (B, 3, 128) using the following layers:
  ConvTranspose1d(d_max, 128, k=5, s=2, p=2, op=1) -> ReLU ->
  ConvTranspose1d(128, 64, k=5, s=2, p=2, op=1) -> ReLU ->
  ConvTranspose1d(64, 32, k=5, s=2, p=2, op=1) -> ReLU ->
  ConvTranspose1d(32, 3, k=5, s=2, p=2, op=1)
- Forward pass:
  1. Encode x_target -> z_coord, z_dyn, and also get a_dyn spatial features (B, d_max, 8) before mean pooling. Run this manually by using self.encoder modules:
     ```python
     coord_features = self.encoder._forward_coord_backbone(x_target)
     a_spatial = self.encoder.conv_spatial(coord_features)
     a_spatial = F.interpolate(a_spatial, size=128, mode='linear', align_corners=False)
     z_target_coord, _ = calculate_centroid_and_variance(a_spatial)

     dyn_features = self.encoder._forward_dyn_backbone(x_target)
     a_dyn = self.encoder.conv_identity_dyn(dyn_features) # (B, d_max, 8)
     z_target_dyn = a_dyn.mean(dim=-1) # (B, d_max)
     ```
  2. Decode: x_recon = decoder(a_dyn)
  3. Recon loss: MSE(x_recon, x_target)
  4. VICReg: var_loss + cov_loss on z_target_dyn (active dims), plus z_target_coord if coord_vicreg=True. Use standard calc_var_loss and calc_cov_loss.
  5. JEPA readout: predictor with stop-gradient on encoder output (surprise readout only, sim_weight=1.0). Make sure to detach input z_hist_coord, z_hist_dyn, and target z_target_coord, z_target_dyn in predictor/similarity loss to prevent guidance of encoder.
  6. Total loss: recon_weight * recon_loss + var_weight * var_loss + cov_weight * cov_loss + sim_weight * sim_loss
- Must expose all properties needed by the evaluation pipeline: `encoder`, `d_t`, `d_max`, `sub_features`, `color_probe_weight`, `color_probe_bias`, `id_contrastive_proj`, `gdasr_growth_points`, and have `update_recruitment_logic` and `clone` methods.
- Write a small test script, execute it to make sure the model imports and the forward pass operates perfectly with no NaNs/errors. Clean up any temporary test files afterwards.
- Verify that the created file is written to `src/models_recon.py` in the actual project workspace.