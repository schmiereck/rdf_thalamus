import torch
import numpy as np
from src.models_dual_stream import (
    calculate_centroid_and_variance,
    add_positional_encoding,
    DualStreamEncoder,
    DualStreamPredictor,
    DualStreamJEPASpatial,
    PDRCEncoder,
    PDRCJEPASpatial,
    NonParametricEncoder,
    NonParametricJEPASpatial
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

def test_pdrc_jepa_spatial():
    print("Testing PDRCJEPASpatial...")
    B, H, C, W = 4, 3, 3, 128
    d_max = 8
    
    # 1. Initialize Stage 1
    model = PDRCJEPASpatial(d_max=d_max, h=H, stage=1)
    assert model.stage == 1, "Expected stage 1"
    
    # Verify requires_grad is True for all parameters in Stage 1
    for p in model.parameters():
        assert p.requires_grad, "All parameters must have requires_grad=True in Stage 1"
        
    x_hist = torch.randn(B, H, C, W)
    x_target = torch.randn(B, C, W)
    
    # Forward check
    loss_dict, (z_pred_c, z_pred_d), (z_target_c, z_target_d) = model(x_hist, x_target, lambda_spatial=1.0)
    assert z_pred_c.shape == (B, d_max)
    assert z_pred_d.shape == (B, d_max)
    assert z_target_c.shape == (B, d_max)
    assert z_target_d.shape == (B, d_max)
    
    # Test gradient flow in Stage 1
    model.zero_grad()
    loss_dict["sim_loss_coord"].backward()
    
    # Gradients should flow back to coordinate head
    assert model.encoder.conv_spatial_coord.weight.grad is not None
    assert torch.any(model.encoder.conv_spatial_coord.weight.grad != 0), "conv_spatial_coord should have gradients"
    # Gradients should also flow to conv1-4 (the backbone) through the coordinate stream
    assert model.encoder.conv1.weight.grad is not None
    assert torch.any(model.encoder.conv1.weight.grad != 0), "Backbone conv1 should have gradients from coordinate stream in Stage 1"
    
    # 2. Transition to Stage 2
    model.stage = 2
    assert model.stage == 2, "Expected stage 2"
    
    # Verify requires_grad setup for Stage 2
    for name, p in model.named_parameters():
        if "encoder.conv_spatial_coord" in name:
            assert not p.requires_grad, f"encoder.conv_spatial_coord parameter {name} should be frozen in Stage 2"
        else:
            assert p.requires_grad, f"Parameter {name} should be trainable in Stage 2"
            
    # Temporarily set requires_grad=True on conv_spatial_coord to test detachment in forward_spatial
    for p in model.encoder.conv_spatial_coord.parameters():
        p.requires_grad = True
        
    # Check that in forward_spatial, x is detached before conv_spatial_coord
    model.zero_grad()
    z_coord, _ = model.encoder(x_target)
    z_coord.sum().backward()
    
    assert model.encoder.conv_spatial_coord.weight.grad is not None
    assert torch.any(model.encoder.conv_spatial_coord.weight.grad != 0)
    # conv1-4 should NOT have gradients
    assert model.encoder.conv1.weight.grad is None or torch.all(model.encoder.conv1.weight.grad == 0), "Backbone should be detached from coordinate head in Stage 2"
    
    # Check that z_hist_coord and z_target_coord are detached before predictor/similarity loss in Stage 2
    model.zero_grad()
    loss_dict, _, _ = model(x_hist, x_target)
    loss_dict["sim_loss_coord"].backward()
    
    # No gradients should flow back to conv_spatial_coord
    assert model.encoder.conv_spatial_coord.weight.grad is None or torch.all(model.encoder.conv_spatial_coord.weight.grad == 0), "No gradients should flow back to coordinate head from sim_loss_coord in Stage 2"
    
    # Test clone in PDRCJEPASpatial
    cloned = model.clone()
    assert cloned.stage == model.stage, "Cloned model stage mismatch"
    assert cloned.encoder.stage == model.encoder.stage, "Cloned encoder stage mismatch"
    
    print("PDRCJEPASpatial tests: PASSED")

def test_non_parametric_jepa_spatial():
    print("Testing NonParametricJEPASpatial...")
    B, H, C, W = 4, 3, 3, 128
    d_max = 8
    
    model = NonParametricJEPASpatial(d_max=d_max, h=H)
    
    x_hist = torch.randn(B, H, C, W)
    x_target = torch.randn(B, C, W)
    
    # Forward check
    loss_dict, (z_pred_c, z_pred_d), (z_target_c, z_target_d) = model(x_hist, x_target, lambda_spatial=1.0)
    assert z_pred_c.shape == (B, d_max)
    assert z_pred_d.shape == (B, d_max)
    assert z_target_c.shape == (B, d_max)
    assert z_target_d.shape == (B, d_max)
    
    # Check that there is only one encoder without separate heads and all parameters are trainable
    for p in model.encoder.parameters():
        assert p.requires_grad, "All encoder parameters should be trainable"
        
    # Check gradient flow: Backprop through coordinate stream similarity loss
    model.zero_grad()
    loss_dict["sim_loss_coord"].backward()
    
    # Gradients should flow back to all layers of the encoder since there are no stop-gradients
    assert model.encoder.conv_spatial.weight.grad is not None
    assert torch.any(model.encoder.conv_spatial.weight.grad != 0)
    assert model.encoder.conv1.weight.grad is not None
    assert torch.any(model.encoder.conv1.weight.grad != 0)
    assert model.encoder.conv4.weight.grad is not None
    assert torch.any(model.encoder.conv4.weight.grad != 0)
    
    # Test clone in NonParametricJEPASpatial
    cloned = model.clone()
    assert isinstance(cloned.encoder, NonParametricEncoder)
    
    print("NonParametricJEPASpatial tests: PASSED")

def test_positional_encodings():
    print("Testing NonParametricJEPASpatial with positional encodings...")
    B, H, C, W = 4, 3, 3, 128
    d_max = 8
    
    for pe_type in ["linear", "sinusoidal"]:
        model = NonParametricJEPASpatial(d_max=d_max, h=H, pos_encoding=pe_type)
        assert model.pos_encoding == pe_type
        assert model.encoder.pos_encoding == pe_type
        
        # Check input channel size of self.conv1
        expected_in_channels = 4 if pe_type == "linear" else 7
        assert model.encoder.conv1.in_channels == expected_in_channels
        
        # Test clone propagates positional encoding
        cloned = model.clone()
        assert cloned.pos_encoding == pe_type
        assert cloned.encoder.pos_encoding == pe_type
        
        # Forward check
        x_hist = torch.randn(B, H, C, W)
        x_target = torch.randn(B, C, W)
        loss_dict, (z_pred_c, z_pred_d), (z_target_c, z_target_d) = model(x_hist, x_target)
        assert z_pred_c.shape == (B, d_max)
        assert z_pred_d.shape == (B, d_max)
        
    print("Positional encodings verification: PASSED")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING DUAL-STREAM ARCHITECTURE TESTS")
    print("=" * 60)
    test_shapes()
    test_backprop_specificity()
    test_mask_coord()
    test_self_cloning()
    test_pdrc_jepa_spatial()
    test_non_parametric_jepa_spatial()
    test_positional_encodings()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
