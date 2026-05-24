import os
import sys
import csv
import json
import random
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

# Set threads to prevent CPU thrashing
torch.set_num_threads(2)

# Ensure src directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment import PhysicsSandbox
from src.models import FixedJEPA
from src.thalamus import ThalamusNet, NonGatedControlNet

class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, x_hist, x_target, color_0, pos_0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (x_hist, x_target, color_0, pos_0)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        x_hist, x_target, color_0, pos_0 = zip(*batch)
        return (np.stack(x_hist, axis=0), 
                np.stack(x_target, axis=0), 
                np.stack(color_0, axis=0), 
                np.array(pos_0, dtype=np.float32))

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prefill_buffer(env, replay_buffer, history, num_transitions):
    if len(history) == 0:
        obs = env.reset()
        history.append(obs)
        
    while len(replay_buffer) < num_transitions:
        obs, info = env.step()
        history.append(obs)
        if len(history) == 4:
            x_hist = np.stack(list(history)[:3], axis=0)
            x_target = history[3]
            color_0 = info["colors"][0]
            pos_0 = info["positions"][0]
            replay_buffer.push(x_hist, x_target, color_0, pos_0)

def main():
    parser = argparse.ArgumentParser(description="Train Thalamus / JEPA Model")
    parser.add_argument("--model", type=str, required=True, choices=["gated", "nongated", "b1"], help="Model type")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    args = parser.parse_args()

    model_type = args.model
    seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"RUNNING EXPERIMENT: Model = {model_type:<10} | Seed = {seed}")
    
    set_seed(seed)
    
    # Initialize Model
    if model_type == "gated":
        model = ThalamusNet(d_max=8, h=3, cooldown=200)
    elif model_type == "nongated":
        model = NonGatedControlNet(d_max=8, h=3)
    elif model_type == "b1":
        model = FixedJEPA(d_t=2, d_max=8, h=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Environment (starts with N=2)
    env = PhysicsSandbox(N=2, seed=seed)
    obs = env.reset()
    
    history = collections.deque(maxlen=4)
    history.append(obs)
    
    replay_buffer = ReplayBuffer(capacity=2000)
    prefill_buffer(env, replay_buffer, history, num_transitions=100)
    
    logs = []
    
    # Training loop: 5000 steps
    for step in range(1, 5001):
        if step == 1501:
            # Transition to PhysicsSandbox(N=3)
            print(f"[Step {step}] Transitioning to PhysicsSandbox(N=3)...")
            env = PhysicsSandbox(N=3, seed=seed)
            obs = env.reset()
            history.clear()
            history.append(obs)
            replay_buffer.clear()
            prefill_buffer(env, replay_buffer, history, num_transitions=100)
            
        # Environment interaction
        obs, info = env.step()
        history.append(obs)
        if len(history) == 4:
            x_hist = np.stack(list(history)[:3], axis=0)
            x_target = history[3]
            color_0 = info["colors"][0]
            pos_0 = info["positions"][0]
            replay_buffer.push(x_hist, x_target, color_0, pos_0)
            
        # Draw batch
        batch_x_hist_np, batch_x_target_np, batch_color_0_np, batch_pos_0_np = replay_buffer.sample(32)
        batch_x_hist = torch.tensor(batch_x_hist_np, dtype=torch.float32).to(device)
        batch_x_target = torch.tensor(batch_x_target_np, dtype=torch.float32).to(device)
        batch_color_0 = torch.tensor(batch_color_0_np, dtype=torch.float32).to(device)
        batch_pos_0 = torch.tensor(batch_pos_0_np, dtype=torch.float32).to(device)
        
        # Determine priming mode
        priming_mode = "external" if step <= 1500 else "self"
        
        model.train()
        optimizer.zero_grad()
        
        if model_type in ["gated", "nongated"]:
            loss_dict, _, _ = model(
                batch_x_hist, 
                batch_x_target, 
                external_query=batch_color_0, 
                priming_mode=priming_mode
            )
            loss = loss_dict["loss"]
            try:
                loss.backward()
            except RuntimeError as e:
                print("\n=== CRASH DIAGNOSTICS ===")
                print(f"Step: {step}")
                print(f"Model: {model_type}")
                print(f"Token locus: {model.token_locus}")
                print(f"L2 locked: {model.l2_locked}")
                print(f"Loss value: {loss.item() if hasattr(loss, 'item') else loss}")
                print(f"Loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'No requires_grad attribute'}")
                print(f"Loss grad_fn: {loss.grad_fn if hasattr(loss, 'grad_fn') else 'No grad_fn attribute'}")
                print("\nParameter States:")
                for name, p in model.named_parameters():
                    print(f"  {name}: requires_grad={p.requires_grad}")
                print("==========================\n")
                raise e
            if hasattr(model, "zero_inactive_gradients"):
                model.zero_inactive_gradients()
            optimizer.step()
            
            l2_sim_loss_val = loss_dict["l2_sim_loss"].item()
            token_locus_val = loss_dict.get("token_locus", -1)
            l2_locked_val = int(loss_dict.get("l2_locked", False))
        else:
            # B1 (FixedJEPA)
            loss_dict, _, _ = model(batch_x_hist, batch_x_target)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            
            l2_sim_loss_val = loss_dict["sim_loss"].item()
            token_locus_val = -1
            l2_locked_val = -1
            
        # Compute tracking overlap for steps 1501-5000
        overlap_val = 0.0
        if step > 1500 and model_type in ["gated", "nongated"]:
            overlap_tensor = model.compute_physical_tracking_overlap(batch_pos_0, batch_x_target, external_query=batch_color_0)
            overlap_val = overlap_tensor.mean().item()
            
        logs.append({
            "step": step,
            "loss": loss.item(),
            "l2_sim_loss": l2_sim_loss_val,
            "token_locus": token_locus_val,
            "l2_locked": l2_locked_val,
            "overlap": overlap_val
        })
        
        if step % 1000 == 0:
            print(f"Step {step:4d}/5000 | Loss: {loss.item():.4f} | L2 Sim Loss: {l2_sim_loss_val:.4f} | Overlap: {overlap_val:.4f} | Locus: {token_locus_val}")
            
    # Save training runs
    os.makedirs("archive/iter_004/runs", exist_ok=True)
    df_run = pd.DataFrame(logs)
    df_run.to_csv(f"archive/iter_004/runs/{model_type}_seed{seed}.csv", index=False)
    print(f"Saved run log to archive/iter_004/runs/{model_type}_seed{seed}.csv")
    
    # 2. Evaluation on Separate Environment (seed + 10000) with N=3 for 100 steps
    print(f"Evaluating {model_type} on separate test environment (seed={seed+10000})...")
    test_env = PhysicsSandbox(N=3, seed=seed+10000)
    test_obs = test_env.reset()
    test_history = collections.deque(maxlen=4)
    test_history.append(test_obs)
    
    test_x_hist_list = []
    test_x_target_list = []
    test_color_0_list = []
    test_target_pos_list = []
    
    while len(test_x_hist_list) < 100:
        obs_t, info_t = test_env.step()
        test_history.append(obs_t)
        if len(test_history) == 4:
            x_hist_t = np.stack(list(test_history)[:3], axis=0)
            x_target_t = test_history[3]
            test_x_hist_list.append(x_hist_t)
            test_x_target_list.append(x_target_t)
            test_color_0_list.append(info_t["colors"][0])
            test_target_pos_list.append(info_t["positions"][0])
            
    test_x_hist = torch.tensor(np.stack(test_x_hist_list, axis=0), dtype=torch.float32).to(device)
    test_x_target = torch.tensor(np.stack(test_x_target_list, axis=0), dtype=torch.float32).to(device)
    test_color_0 = torch.tensor(np.stack(test_color_0_list, axis=0), dtype=torch.float32).to(device)
    test_target_pos = torch.tensor(np.array(test_target_pos_list), dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        if model_type == "b1":
            test_loss_dict, _, _ = model(test_x_hist, test_x_target)
            final_test_l2_loss = test_loss_dict["sim_loss"].item()
            final_test_overlap = 0.0
        elif model_type in ["gated", "nongated"]:
            test_loss_dict, _, _ = model(test_x_hist, test_x_target, priming_mode="self")
            final_test_l2_loss = test_loss_dict["l2_sim_loss"].item()
            overlap_tensor = model.compute_physical_tracking_overlap(test_target_pos, test_x_target, external_query=test_color_0)
            final_test_overlap = overlap_tensor.mean().item()
            
    print(f"Evaluation results | Test L2 Loss: {final_test_l2_loss:.4f} | Test Tracking Overlap: {final_test_overlap:.4f}")
    
    # Save evaluation JSON
    eval_metrics = {
        "model_type": model_type,
        "seed": seed,
        "final_test_l2_loss": final_test_l2_loss,
        "final_test_overlap": final_test_overlap
    }
    
    eval_json_path = f"archive/iter_004/runs/{model_type}_seed{seed}_eval.json"
    with open(eval_json_path, "w") as f_json:
        json.dump(eval_metrics, f_json, indent=4)
    print(f"Saved evaluation JSON to {eval_json_path}")

if __name__ == "__main__":
    main()
