import torch
import torch.optim as optim
import numpy as np
import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models import Encoder, Predictor, FixedJEPA, DynamicJEPA

def test_physics_sandbox():
    print("--- Testing PhysicsSandbox ---")
    for N in [2, 3]:
        print(f"Testing with N={N}...")
        env = PhysicsSandbox(N=N, substeps=10, sigma_blur=0.5)
        
        # Test reset
        obs = env.reset()
        assert obs.shape == (3, 128), f"Expected shape (3, 128), got {obs.shape}"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Observation values must be in [0, 1]"
        
        # Run a few steps
        pos_history = []
        for step in range(20):
            obs, info = env.step()
            assert obs.shape == (3, 128), f"Step {step} observation shape is incorrect"
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Step observation values must be in [0, 1]"
            
            # Verify keys in info
            for key in ["positions", "velocities", "radii", "masses", "colors"]:
                assert key in info, f"Missing key {key} in step info"
                assert len(info[key]) == N, f"Key {key} should have length {N}"
            
            pos_history.append(info["positions"].copy())
            
        # Verify that things moved
        pos_history = np.array(pos_history)
        max_diff = np.max(np.abs(pos_history[-1] - pos_history[0]))
        assert max_diff > 0.01, f"Objects did not move enough: max diff is {max_diff}"
        
    print("PhysicsSandbox tests passed successfully!")

def test_jepa_models():
    print("\n--- Testing FixedJEPA and DynamicJEPA Models ---")
    B, H, C, W = 4, 3, 3, 128
    
    x_hist = torch.rand(B, H, C, W)
    x_target = torch.rand(B, C, W)
    
    # 1. Test FixedJEPA
    print("Testing FixedJEPA...")
    fixed_model = FixedJEPA(d_t=2, d_max=8, h=H)
    loss_dict, z_pred, z_target = fixed_model(x_hist, x_target)
    
    # Check output shapes
    assert z_pred.shape == (B, 8), f"z_pred shape is {z_pred.shape}"
    assert z_target.shape == (B, 8), f"z_target shape is {z_target.shape}"
    for key in ["loss", "sim_loss", "var_loss", "cov_loss"]:
        assert key in loss_dict, f"Missing loss key {key}"
        assert isinstance(loss_dict[key], torch.Tensor)
        assert loss_dict[key].ndim == 0, f"Loss key {key} should be scalar"
        assert not torch.isnan(loss_dict[key]), f"Loss {key} is NaN"
        
    # Check backward pass and optimization
    optimizer = optim.Adam(fixed_model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()
    print("FixedJEPA forward/backward/step completed without errors.")
    
    # 2. Test DynamicJEPA
    print("Testing DynamicJEPA...")
    dyn_model = DynamicJEPA(d_max=8, h=H, k=4, cooldown=500, stabilization_period=200)
    loss_dict, z_pred, z_target = dyn_model(x_hist, x_target)
    
    assert z_pred.shape == (B, 8)
    assert z_target.shape == (B, 8)
    for key in ["loss", "sim_loss", "var_loss", "cov_loss"]:
        assert key in loss_dict, f"Missing loss key {key}"
        assert not torch.isnan(loss_dict[key]), f"Loss {key} is NaN"
        
    optimizer = optim.Adam(dyn_model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()
    print("DynamicJEPA forward/backward/step completed without errors.")

def test_recruitment_and_stabilization():
    print("\n--- Testing DynamicJEPA Recruitment and Stabilization Logic ---")
    H = 3
    # Instantiate with short cooldown and stabilization
    model = DynamicJEPA(d_max=8, h=H, k=4, cooldown=5, stabilization_period=4)
    
    # Verify initial state
    assert model.d_t == 2, f"Initial d_t should be 2, got {model.d_t}"
    assert model.steps_since_recruitment >= 5, "Should start outside of cooldown"
    
    # 1. Fill error buffer with low stable error values
    # Error buffer should have at least 200 elements to trigger recruitment
    for _ in range(210):
        # We manually update with low error
        model.update_recruitment_logic(0.1)
        
    assert len(model.error_buffer) >= 200
    assert model.d_t == 2, "Should not recruit with stable low errors"
    
    # 2. Mock high prediction error to trigger recruitment
    # The baseline mean is 0.1, std is 0.0.
    # High error of 5.0 will exceed mean + 4 * std
    model.update_recruitment_logic(5.0)
    
    assert model.d_t == 3, f"Should have recruited to d_t = 3, got {model.d_t}"
    assert model.steps_since_recruitment == 0, f"steps_since_recruitment should be reset to 0, got {model.steps_since_recruitment}"
    
    # 3. Verify stabilization / stop-gradient logic is active
    # Since steps_since_recruitment = 0 < stabilization_period = 4, stabilization is active.
    assert model.steps_since_recruitment < model.stabilization_period, "Stabilization should be active"
    
    # We will hook the outputs of the encoder to verify that dimensions < d_t - 1 (dim 0 and 1)
    # have zero gradients flowing back to them, while dimension 2 (the new dimension) does.
    captured_tensors = []
    def hook_fn(module, input, output):
        captured_tensors.append(output)
        output.retain_grad()
        
    handle = model.encoder.fc.register_forward_hook(hook_fn)
    
    B, C, W = 4, 3, 128
    x_hist = torch.rand(B, H, C, W)
    x_target = torch.rand(B, C, W)
    
    # Run forward pass during stabilization
    loss_dict, z_pred, z_target = model(x_hist, x_target)
    
    # Retain grad on inputs/outputs if needed
    # We run backward on the total loss
    loss = loss_dict["loss"]
    loss.backward()
    
    # Clean up the hook
    handle.remove()
    
    assert len(captured_tensors) > 0, "No encoder output was captured"
    
    for out in captured_tensors:
        # out has shape (N, 8)
        # Check its gradient
        grad = out.grad
        assert grad is not None, "Gradient of encoder output is None"
        
        # Dimensions 0 and 1 (d_t - 1 = 2, so index < 2) should have 0 gradient due to detach
        # Let's verify this!
        grad_dim0_mean = torch.abs(grad[:, 0]).mean().item()
        grad_dim1_mean = torch.abs(grad[:, 1]).mean().item()
        grad_dim2_mean = torch.abs(grad[:, 2]).mean().item()
        
        print(f"Captured output gradient means per dimension: dim0={grad_dim0_mean:.4f}, dim1={grad_dim1_mean:.4f}, dim2={grad_dim2_mean:.4f}")
        
        assert grad_dim0_mean == 0.0, f"Expected grad_dim0 to be 0.0, got {grad_dim0_mean}"
        assert grad_dim1_mean == 0.0, f"Expected grad_dim1 to be 0.0, got {grad_dim1_mean}"
        assert grad_dim2_mean > 0.0, f"Expected grad_dim2 (new dimension) to have gradients, got {grad_dim2_mean}"
        
    print("Stabilization stop-gradient logic tested successfully!")

if __name__ == "__main__":
    test_physics_sandbox()
    test_jepa_models()
    test_recruitment_and_stabilization()
    print("\nAll integration tests passed successfully!")
