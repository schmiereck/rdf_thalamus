import torch
import numpy as np
from environment import PhysicsSandbox
from thalamus import ThalamusNet
from motor import SubsumptionMotorController

def run_closed_loop_test():
    print("==================================================")
    print("Starting closed-loop integration test (Phase 3)...")
    print("==================================================")

    # 1. Initialize Sandbox Environment
    sandbox = PhysicsSandbox(N=2, substeps=10, seed=42)
    obs = sandbox.render()
    print(f"Environment initialized. Initial pointer pos: {sandbox.pointer_pos:.3f}, vel: {sandbox.pointer_vel:.3f}")

    # 2. Initialize Model (ThalamusNet)
    # We will use d_max=8, history h=3, and a small cooldown for testing token switching
    d_max = 8
    h = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThalamusNet(d_max=d_max, h=h, cooldown=5).to(device)
    model.eval() # Eval mode since we don't need grad update in this integration test
    print(f"ThalamusNet compiled on {device}.")

    # 3. Initialize Controller
    controller = SubsumptionMotorController(
        Kp=1.2, Kd=0.15, Kv=0.6, error_thresh=5.0, stability_thresh=0.05,
        push_cooldown=10, push_magnitude=15.0, dt=1.0, ablation=None
    )
    print("SubsumptionMotorController initialized successfully.")

    # 4. Prepare Observation History (h=3 frames)
    # Duplicate initial observation to fill the buffer
    x_hist = [obs.copy() for _ in range(h)]

    print("\n--- Starting Closed Loop Simulation for 30 steps ---")
    for step_idx in range(1, 31):
        # Build Tensors
        x_hist_tensor = torch.tensor(np.stack(x_hist), dtype=torch.float32, device=device).unsqueeze(0) # (1, H, 3, 128)
        x_target_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) # (1, 3, 128)

        # Model Forward Pass
        with torch.no_grad():
            loss_dict, z_pred_segments, z_pred_l2 = model(x_hist_tensor, x_target_tensor)

        # Extract variables
        locus = loss_dict["token_locus"]
        delta_E = loss_dict.get("delta_E", 0.0)
        pos_loss = loss_dict["pos_loss"].item()
        current_cooldown = loss_dict["current_cooldown"]

        # Controller computes action
        action = controller.get_action(model, obs, sandbox.step()[1], z_pred_segments, delta_E=delta_E)

        # Step Environment
        obs, info = sandbox.step(action)

        # Update History Buffer (FIFO)
        x_hist.pop(0)
        x_hist.append(obs.copy())

        # Compute current error for reporting
        with torch.no_grad():
            centroid = model.compute_attended_centroid(x_target_tensor, locus=locus).item()
        error = centroid - info["pointer_pos"]

        print(f"Step {step_idx:02d} | "
              f"Attn Locus: {locus} | "
              f"Target Centroid: {centroid:6.2f} | "
              f"Pointer Pos: {info['pointer_pos']:6.2f} (Vel: {info['pointer_vel']:6.2f}) | "
              f"Err: {error:6.2f} | "
              f"Acc: {action['acc']:6.2f} (Push: {str(action['push']):5s}) | "
              f"Cooldown: {current_cooldown:.1f} | "
              f"Pos Loss: {pos_loss:.4f}")

    print("\n==================================================")
    print("Closed-loop integration test COMPLETED successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_closed_loop_test()
