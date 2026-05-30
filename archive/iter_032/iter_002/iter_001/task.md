Modify `src/models_separate_dyn.py` to:
1. Update `SeparateDynEncoder.__init__`:
   - If `self.dyn_readout == "centroid_gated"`, set `self.conv_identity_dyn = nn.Conv1d(128, d_max * sub_features, kernel_size=1)`
   - Otherwise, set `self.conv_identity_dyn = nn.Conv1d(128, d_max, kernel_size=1)`
   Keep all other backbone conv layers completely unchanged.
2. Update `SeparateDynEncoder.forward(self, x)`:
   - Produce coord backbone feature: `coord_features = self._forward_coord_backbone(x)`
   - Produce a_spatial: `a_spatial = self.conv_spatial(coord_features)`
   - Interpolate a_spatial: `a_spatial = F.interpolate(a_spatial, size=128, mode='linear', align_corners=False)`
   - Compute coordinate centroids and variance: `z_coord, _ = calculate_centroid_and_variance(a_spatial)`
   - Produce dyn backbone feature: `dyn_features = self._forward_dyn_backbone(x)`
   - Produce a_dyn: `a_dyn = self.conv_identity_dyn(dyn_features)`
   - Implement the three readout modes:
     a. `self.dyn_readout == "mean"`: `z_dyn = a_dyn.mean(dim=-1)`
     b. `self.dyn_readout == "centroid_gated"` with `self.sub_features == 1`:
        - Detach spatial attention weight: `p_c = F.softmax(a_spatial, dim=-1).detach()`
        - Interpolate a_dyn from 8 to 128: `a_dyn_128 = F.interpolate(a_dyn, size=128, mode='linear', align_corners=False)`
        - Compute centroid-gated z_dyn using einsum: `z_dyn = torch.einsum('bcs,bcs->bc', p_c, a_dyn_128)`
     c. `self.dyn_readout == "centroid_gated"` with `self.sub_features > 1`:
        - Detach spatial attention weight: `p_c = F.softmax(a_spatial, dim=-1).detach()`
        - Interpolate a_dyn from 8 to 128: `a_dyn_128 = F.interpolate(a_dyn, size=128, mode='linear', align_corners=False)`
        - Reshape a_dyn_128 to `(B, d_max, K, 128)` where `K = self.sub_features` and `d_max = self.d_max`: `a_dyn_128 = a_dyn_128.reshape(x.shape[0], self.d_max, self.sub_features, 128)`
        - Compute centroid-gated z_dyn using einsum: `z_dyn = torch.einsum('bcs,bcks->bck', p_c, a_dyn_128)`
        - Reshape z_dyn to `(B, d_max * K)`: `z_dyn = z_dyn.reshape(x.shape[0], self.d_max * self.sub_features)`
     If any other `dyn_readout`, raise a ValueError.
   - Return `z_coord, z_dyn`.
3. Check and ensure that `NonParametricJEPASpatialSeparateDyn` is preserved and correctly delegates to `SeparateDynEncoder` and `NonParametricJEPASpatial` as before. Ensure there are no other modifications that break backward compatibility.

Do this change in-place in `src/models_separate_dyn.py`.