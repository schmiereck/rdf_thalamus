import torch
import torch.nn.functional as F
from src.models_spatial import DynamicJEPASpatial, calculate_centroid_and_variance

def test_spatial_centroid_and_variance():
    print("--- Testing calculate_centroid_and_variance ---")
    B, d_max, L = 2, 4, 128
    # Create non-trivial dummy spatial activations
    a_spatial = torch.randn(B, d_max, L)
    
    x_mean, var = calculate_centroid_and_variance(a_spatial)
    print("x_mean shape:", x_mean.shape)
    print("var shape:", var.shape)
    
    assert x_mean.shape == (B, d_max), f"Expected (B, d_max), got {x_mean.shape}"
    assert var.shape == (B, d_max), f"Expected (B, d_max), got {var.shape}"
    print("calculate_centroid_and_variance helper tests passed!\n")

def test_dynamic_jepa_spatial_forward_and_backward():
    print("--- Testing DynamicJEPASpatial Forward & Backward ---")
    B, H, C, W = 4, 3, 3, 128
    d_max = 8
    
    model = DynamicJEPASpatial(d_max=d_max, h=H)
    
    x_hist = torch.randn(B, H, C, W)
    x_target = torch.randn(B, C, W)
    
    # Run forward pass with lambda_spatial > 0 and k_chan specified
    k_chan = 3
    lambda_spatial = 1.5
    
    output, z_pred, z_target = model(
        x_hist, 
        x_target, 
        lambda_spatial=lambda_spatial, 
        k_chan=k_chan
    )
    
    print("Keys in output dict:", list(output.keys()))
    print("loss_spatial value:", output["loss_spatial"].item())
    print("total loss value:", output["loss"].item())
    
    assert "loss_spatial" in output, "loss_spatial missing from output dict"
    assert output["loss_spatial"].requires_grad, "loss_spatial should require gradients"
    
    # Test backward pass specificity
    model.zero_grad()
    loss_spatial = output["loss_spatial"]
    loss_spatial.backward()
    
    # Retrieve gradients for conv_spatial
    weight_grad = model.encoder.conv_spatial.weight.grad
    bias_grad = model.encoder.conv_spatial.bias.grad
    
    print("\nconv_spatial weight gradient shape:", weight_grad.shape)
    print("conv_spatial bias gradient shape:", bias_grad.shape)
    
    # Verify gradient specificity: only k-th channel should have non-zero gradients
    for j in range(d_max):
        w_grad_norm = weight_grad[j].abs().sum().item()
        b_grad_norm = bias_grad[j].abs().item()
        if j == k_chan:
            print(f"Channel {j} (Target k_chan): weight grad norm = {w_grad_norm:.6f}, bias grad norm = {b_grad_norm:.6f}")
            assert w_grad_norm > 0.0, f"Expected non-zero weight gradient on target channel {j}"
            assert b_grad_norm > 0.0, f"Expected non-zero bias gradient on target channel {j}"
        else:
            print(f"Channel {j} (Other channel): weight grad norm = {w_grad_norm:.6f}, bias grad norm = {b_grad_norm:.6f}")
            assert w_grad_norm == 0.0, f"Expected exactly zero weight gradient on non-target channel {j}, got {w_grad_norm}"
            assert b_grad_norm == 0.0, f"Expected exactly zero bias gradient on non-target channel {j}, got {b_grad_norm}"
            
    print("\nDynamicJEPASpatial tests passed successfully!")

if __name__ == "__main__":
    test_spatial_centroid_and_variance()
    test_dynamic_jepa_spatial_forward_and_backward()
