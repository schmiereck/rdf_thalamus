import torch
import numpy as np
from src.models_dual_stream import (
    calculate_centroid_and_variance,
    DualStreamEncoder,
    DualStreamPredictor,
    DualStreamJEPASpatial
)

def test_shapes():
    B = 4
    H = 3
    C = 3
    W = 128
    d_max = 8
    
    encoder = DualStreamEncoder(d_max=d_max)
    x = torch.randn(B, C, W)
    
    # 1. forward_spatial
    a_spatial = encoder.forward_spatial(x)
    assert a_spatial.shape == (B, d_max, W), f"Expected {(B, d_max, W)}, got {a_spatial.shape}"
    
    # 2. forward_dynamics
    z_dyn = encoder.forward_dynamics(x)
    assert z_dyn.shape == (B, d_max), f"Expected {(B, d_max)}, got {z_dyn.shape}"
    
    # 3. forward
    z_coord, z_dyn_f = encoder(x)
    assert z_coord.shape == (B, d_max), f"Expected {(B, d_max)}, got {z_coord.shape}"
    assert z_dyn_f.shape == (B, d_max), f"Expected {(B, d_max)}, got {z_dyn_f.shape}"
    
    # 4. Predictor shapes
    predictor = DualStreamPredictor(d_max=d_max, h=H)
    z_coord_hist = torch.randn(B, H, d_max)
    z_dyn_hist = torch.randn(B, H, d_max)
    pred_coord, pred_dyn = predictor(z_coord_hist, z_dyn_hist, d_t=3)
    assert pred_coord.shape == (B, d_max), f"Expected {(B, d_max)}, got {pred_coord.shape}"
    assert pred_dyn.shape == (B, d_max), f"Expected {(B, d_max)}, got {pred_dyn.shape}"
    
    # Inactive dimensions (>= d_t) should be zeroed out
    assert torch.all(pred_coord[:, 3:] == 0.0), "Inactive dimensions in pred_coord not zeroed out"
    assert torch.all(pred_dyn[:, 3:] == 0.0), "Inactive dimensions in pred_dyn not zeroed out"
    
    # 5. DualStreamJEPA shapes
    jepa = DualStreamJEPASpatial(d_max=d_max, h=H)
    x_hist = torch.randn(B, H, C, W)
    x_target = torch.randn(B, C, W)
    
    loss_dict, (z_pred_c, z_pred_d), (z_target_c, z_target_d) = jepa(x_hist, x_target, lambda_spatial=1.0)
    assert "loss" in loss_dict, "Missing loss in output dict"
    assert "loss_spatial" in loss_dict, "Missing loss_spatial in output dict"
    assert "sim_loss" in loss_dict, "Missing sim_loss in output dict"
    assert "var_loss" in loss_dict, "Missing var_loss in output dict"
    assert "cov_loss" in loss_dict, "Missing cov_loss in output dict"
    
    assert z_pred_c.shape == (B, d_max), f"Expected {(B, d_max)}, got {z_pred_c.shape}"
    assert z_pred_d.shape == (B, d_max), f"Expected {(B, d_max)}, got {z_pred_d.shape}"
    assert z_target_c.shape == (B, d_max), f"Expected {(B, d_max)}, got {z_target_c.shape}"
    assert z_target_d.shape == (B, d_max), f"Expected {(B, d_max)}, got {z_target_d.shape}"
    print("Shape and forward pass checks: PASSED")

def test_backprop_specificity():
    B = 4
    H = 3
    C = 3
    W = 128
    d_max = 8
    
    # Create model
    jepa = DualStreamJEPASpatial(d_max=d_max, h=H)
    x_hist = torch.randn(B, H, C, W)
    x_target = torch.randn(B, C, W)
    
    # Zero gradients
    jepa.zero_grad()
    
    # Run forward with lambda_spatial > 0
    loss_dict, _, _ = jepa(x_hist, x_target, lambda_spatial=1.0)
    loss_spatial = loss_dict["loss_spatial"]
    loss_spatial.backward()
    
    # Verify gradient existence and correctness for loss_spatial:
    # Gradients should ONLY flow to the coordinate head conv_spatial_coord
    assert jepa.encoder.conv_spatial_coord.weight.grad is not None, "conv_spatial_coord should have gradients"
    assert torch.any(jepa.encoder.conv_spatial_coord.weight.grad != 0), "conv_spatial_coord gradients should be non-zero"
    
    assert jepa.encoder.conv1.weight.grad is None or torch.all(jepa.encoder.conv1.weight.grad == 0), "conv1 should NOT have gradients from loss_spatial"
    assert jepa.encoder.conv2.weight.grad is None or torch.all(jepa.encoder.conv2.weight.grad == 0), "conv2 should NOT have gradients from loss_spatial"
    assert jepa.encoder.conv3.weight.grad is None or torch.all(jepa.encoder.conv3.weight.grad == 0), "conv3 should NOT have gradients from loss_spatial"
    assert jepa.encoder.conv4.weight.grad is None or torch.all(jepa.encoder.conv4.weight.grad == 0), "conv4 should NOT have gradients from loss_spatial"
    assert jepa.encoder.conv_spatial_dyn.weight.grad is None or torch.all(jepa.encoder.conv_spatial_dyn.weight.grad == 0), "conv_spatial_dyn should NOT have gradients from loss_spatial"
    
    # Reset gradients
    jepa.zero_grad()
    
    # Run forward again and backprop through dynamics stream prediction loss only
    loss_dict, _, _ = jepa(x_hist, x_target, lambda_spatial=0.0)
    sim_loss_dyn = loss_dict["sim_loss_dyn"]
    sim_loss_dyn.backward()
    
    # Gradients on dynamics stream should flow to conv_spatial_dyn and conv1-4
    assert jepa.encoder.conv_spatial_dyn.weight.grad is not None, "conv_spatial_dyn should have gradients from prediction"
    assert torch.any(jepa.encoder.conv_spatial_dyn.weight.grad != 0), "conv_spatial_dyn gradients should be non-zero"
    
    assert jepa.encoder.conv1.weight.grad is not None, "conv1 should have gradients from dynamics stream prediction"
    assert torch.any(jepa.encoder.conv1.weight.grad != 0), "conv1 gradients should be non-zero"
    assert jepa.encoder.conv2.weight.grad is not None, "conv2 should have gradients from dynamics stream prediction"
    assert torch.any(jepa.encoder.conv2.weight.grad != 0), "conv2 gradients should be non-zero"
    assert jepa.encoder.conv3.weight.grad is not None, "conv3 should have gradients from dynamics stream prediction"
    assert torch.any(jepa.encoder.conv3.weight.grad != 0), "conv3 gradients should be non-zero"
    assert jepa.encoder.conv4.weight.grad is not None, "conv4 should have gradients from dynamics stream prediction"
    assert torch.any(jepa.encoder.conv4.weight.grad != 0), "conv4 gradients should be non-zero"
    
    # And they should NOT flow to conv_spatial_coord (the coordinate head)
    assert jepa.encoder.conv_spatial_coord.weight.grad is None or torch.all(jepa.encoder.conv_spatial_coord.weight.grad == 0), "conv_spatial_coord should NOT have gradients from dynamics prediction"
    
    print("Backprop specificity checks: PASSED")

def test_mask_coord():
    B = 2
    H = 3
    d_max = 8
    
    predictor = DualStreamPredictor(d_max=d_max, h=H)
    
    # Create non-zero inputs
    z_coord_hist = torch.randn(B, H, d_max)
    z_dyn_hist = torch.randn(B, H, d_max)
    
    # 1. Output with mask_coord=True
    pred_c_m, pred_d_m = predictor(z_coord_hist, z_dyn_hist, d_t=3, mask_coord=True)
    
    # 2. Output with coordinate input replaced by zeros and mask_coord=False
    z_coord_zeros = torch.zeros_like(z_coord_hist)
    pred_c_z, pred_d_z = predictor(z_coord_zeros, z_dyn_hist, d_t=3, mask_coord=False)
    
    # Verify they are identical
    assert torch.allclose(pred_c_m, pred_c_z), "mask_coord=True output does not match zero-input output"
    assert torch.allclose(pred_d_m, pred_d_z), "mask_coord=True output does not match zero-input output"
    
    # 3. Output with non-zero coordinates and mask_coord=False
    pred_c_n, pred_d_n = predictor(z_coord_hist, z_dyn_hist, d_t=3, mask_coord=False)
    
    # Verify they are NOT identical to the masked ones (ensures coordinates are actually used when not masked)
    assert not torch.allclose(pred_c_m, pred_c_n), "mask_coord=False should change output when coord is non-zero"
    assert not torch.allclose(pred_d_m, pred_d_n), "mask_coord=False should change output when coord is non-zero"
    
    print("mask_coord verification: PASSED")

def test_self_cloning():
    jepa = DualStreamJEPASpatial(d_max=8, h=3)
    jepa.d_t = 4
    jepa.steps_since_recruitment = 123
    jepa.ema_error = 0.45
    
    cloned = jepa.clone()
    assert cloned is not jepa, "Cloned instance should be a separate object"
    assert cloned.d_t == jepa.d_t, "cloned.d_t mismatch"
    assert cloned.steps_since_recruitment == jepa.steps_since_recruitment, "cloned.steps_since_recruitment mismatch"
    assert cloned.ema_error == jepa.ema_error, "cloned.ema_error mismatch"
    
    # Verify state dict matches
    for p1, p2 in zip(jepa.parameters(), cloned.parameters()):
        assert torch.allclose(p1, p2), "Cloned model parameters do not match original model"
        
    print("Self-cloning verification: PASSED")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING DUAL-STREAM ARCHITECTURE TESTS")
    print("=" * 60)
    test_shapes()
    test_backprop_specificity()
    test_mask_coord()
    test_self_cloning()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
