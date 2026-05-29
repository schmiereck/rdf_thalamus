Please add a unit test for contrastive mode to src/test_models_dual_stream.py.

Add a function:
```python
def test_contrastive_mode():
    print("Testing contrastive mode...")
    B, H, C, W = 4, 3, 3, 128
    d_max = 8
    model = NonParametricJEPASpatial(
        d_max=d_max, h=H,
        primary_objective="contrastive",
        contrastive_weight=15.0,
        temperature=0.2
    )
    assert model.primary_objective == "contrastive"
    assert model.contrastive_weight == 15.0
    assert model.temperature == 0.2

    # Test cloning
    cloned = model.clone()
    assert cloned.primary_objective == "contrastive"
    assert cloned.contrastive_weight == 15.0
    assert cloned.temperature == 0.2

    # Test forward pass
    x_hist = torch.randn(B, H, C, W)
    x_target = torch.randn(B, C, W)
    loss_dict, (z_pred_c, z_pred_d), (z_target_c, z_target_d) = model(x_hist, x_target)

    assert "contrastive_loss" in loss_dict
    assert loss_dict["contrastive_loss"].item() >= 0.0
    # check that we can override in forward pass
    loss_dict_override, _, _ = model(x_hist, x_target, contrastive_weight=10.0, temperature=0.1)
    assert "contrastive_loss" in loss_dict_override
    print("Contrastive mode tests: PASSED")
```

And call `test_contrastive_mode()` at the end of `src/test_models_dual_stream.py` inside the `if __name__ == "__main__":` block.
Verify that the tests run and pass by running `python src/test_models_dual_stream.py`!