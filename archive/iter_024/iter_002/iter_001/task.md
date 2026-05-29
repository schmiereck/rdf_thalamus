Modify src/models_dual_stream.py to add temporal contrastive learning functionality to NonParametricJEPASpatial.

Specifically:
1. In `NonParametricJEPASpatial.__init__()`, add parameters `contrastive_weight=25.0` and `temperature=0.1` and save them as `self.contrastive_weight` and `self.temperature`.
2. In `NonParametricJEPASpatial.forward()`, add parameters `contrastive_weight=None` and `temperature=None` to the signature. Resolve them as:
   _contrastive_weight = contrastive_weight if contrastive_weight is not None else self.contrastive_weight
   _temperature = temperature if temperature is not None else self.temperature
3. Modify the primary objective checks so that `if self.primary_objective == 'sfa':` becomes `if self.primary_objective in ['sfa', 'contrastive']:` or similar.
4. In that branch, if `primary_objective == 'contrastive'`, compute NT-Xent loss:
   - z_anchor = F.normalize(z_target_dyn[:, :d_t_dyn], dim=-1)
   - z_positive = F.normalize(z_hist_dyn[:, -1, :d_t_dyn], dim=-1) (last history step)
   - sim_matrix = torch.matmul(z_anchor, z_positive.T) / _temperature
   - labels = torch.arange(B, device=x_target.device)
   - contrastive_loss = F.cross_entropy(sim_matrix, labels)
   Otherwise set contrastive_loss = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype).
5. Compute base_loss dynamically:
   - If self.primary_objective == 'contrastive':
     base_loss = _contrastive_weight * contrastive_loss + var_weight * var_loss + cov_weight * cov_loss + sim_weight * sim_loss
   - Else:
     base_loss = _sfa_weight * sfa_loss + var_weight * var_loss + cov_weight * cov_loss + sim_weight * sim_loss
6. Add "contrastive_loss" to the returned loss dict in this branch.
7. Update `NonParametricJEPASpatial.clone()` to pass `contrastive_weight=self.contrastive_weight, temperature=self.temperature` to the constructor.
8. Run pytest on `src/test_models_dual_stream.py` (specifically tests relating to NonParametricJEPASpatial or general dual stream tests) to make sure there are no syntax errors and all baseline tests pass. Add a new test case to `src/test_models_dual_stream.py` that tests the "contrastive" primary objective of `NonParametricJEPASpatial` to make sure it doesn't crash and returns contrastive_loss correctly.