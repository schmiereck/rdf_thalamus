Write the class `ReconVICRegSeparateDyn` into `src/models_recon.py` in the current project workspace.

The class should implement the Reconstruction+VICReg architecture:
- Encoder is `SeparateDynEncoder` from `src/models_separate_dyn.py`.
- Decoder deconv head on dyn spatial features maps (B, d_max, 8) -> (B, 3, 128) using the following layers:
  ConvTranspose1d(d_max, 128, k=5, s=2, p=2, op=1) -> ReLU ->
  ConvTranspose1d(128, 64, k=5, s=2, p=2, op=1) -> ReLU ->
  ConvTranspose1d(64, 32, k=5, s=2, p=2, op=1) -> ReLU ->
  ConvTranspose1d(32, 3, k=5, s=2, p=2, op=1)
- Must support forward pass reconstructing `x_target` and computing VICReg + JEPA predictor (surprise readout) losses.
- Must expose all properties needed by the evaluation pipeline: `encoder`, `d_t`, `d_max`, `sub_features`, `color_probe_weight`, `color_probe_bias`, `id_contrastive_proj`, `gdasr_growth_points`.
- Write a small test script and execute it to make sure the model imports and the forward pass operates perfectly. Do not leave temporary test files in the workspace (or clean them up).