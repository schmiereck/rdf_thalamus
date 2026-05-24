Implement and execute the systematic evaluation and comparison sweep for Phase 3 by creating and running the script `src/train_eval_closed_loop.py`.

### Detailed Tasks:

1. **Create `src/train_eval_closed_loop.py`**:
   The script must be fully self-contained and implement the following components:

   - **Imports**:
     ```python
     import os
     import sys
     import csv
     import json
     import random
     import collections
     import numpy as np
     import pandas as pd
     import torch
     import torch.nn as nn
     import torch.nn.functional as F
     import torch.optim as optim
     import concurrent.futures

     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
     from src.environment import PhysicsSandbox
     from src.thalamus import ThalamusNet
     from src.motor import SubsumptionMotorController
     ```

   - **Deterministic Replay Buffer**:
     ```python
     class ReplayBuffer:
         def __init__(self, capacity=2000, seed=42):
             self.capacity = capacity
             self.buffer = []
             self.position = 0
             self.rng = random.Random(seed)

         def push(self, x_hist, x_target, color_0, pos_0):
             if len(self.buffer) < self.capacity:
                 self.buffer.append(None)
             self.buffer[self.position] = (x_hist, x_target, color_0, pos_0)
             self.position = (self.position + 1) % self.capacity

         def sample(self, batch_size):
             batch = self.rng.sample(self.buffer, batch_size)
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
     ```

   - **Prefill Buffer Function**:
     ```python
     def prefill_buffer(env, replay_buffer, history, num_transitions):
         if len(history) == 0:
             obs = env.reset()
             history.append(obs)
         last_info = None
         while len(replay_buffer) < num_transitions:
             obs, info = env.step()
             history.append(obs)
             last_info = info
             if len(history) == 4:
                 x_hist = np.stack(list(history)[:3], axis=0)
                 x_target = history[3]
                 color_0 = info["colors"][0]
                 pos_0 = info["positions"][0]
                 replay_buffer.push(x_hist, x_target, color_0, pos_0)
         return last_info
     ```

   - **Global Seed Setter**:
     ```python
     def set_seed(seed):
         random.seed(seed)
         np.random.seed(seed)
         torch.manual_seed(seed)
         if torch.cuda.is_available():
             torch.cuda.manual_seed_all(seed)
         torch.backends.cudnn.deterministic = True
         torch.backends.cudnn.benchmark = False
     ```

   - **Training Worker Function** (`train_worker(model_name, seed)`):
     - Sets `torch.set_num_threads(2)`
     - Sets seeds: `set_seed(seed)`
     - Sets `device = torch.device("cpu")`
     - Instantiates `model = ThalamusNet(d_max=8, h=3, cooldown=200).to(device)`
     - Instantiates `optimizer = optim.Adam(model.parameters(), lr=1e-3)`
     - Instantiates `replay_buffer = ReplayBuffer(capacity=2000, seed=seed)`
     - Instantiates `env = PhysicsSandbox(N=2, seed=seed)`
     - Pre-fills replay buffer with 100 transitions, saving the last `info` from pre-filling.
     - Sets up `controller = SubsumptionMotorController()` and resets it.
     - Runs training loop for exactly 5000 steps:
       - Transition to `PhysicsSandbox(N=3, seed=seed)` at step 1501, resetting history, clearing buffer, and prefilling with 100 transitions. Reset controller.
       - Chooses environment action based on `model_name`:
         - For `M_active`:
           - If `step <= 1000`, sample continuous `acc` in `[-10, 10]`, `push` with probability 0.1.
           - If `step > 1000`:
             Query model with current history in `eval` mode and `torch.no_grad()` to get `z_pred_segments` and `delta_E` (from `loss_dict`). Set `priming_mode = "external" if step <= 1500 else "self"`.
             Query controller `get_action(model, obs, info, z_pred_segments, delta_E)`.
             If `step <= 3000`, override `action["push"] = False`.
         - For `M_no_motor`: stationary pointer (`acc=0.0`, `push=False`).
         - For `M_random`: random actions (`acc` in `[-10, 10]`, `push` with probability 0.1).
       - Steps environment with action to get `obs, info`.
       - Pushes new transition `(x_hist_new, x_target_new, color_0, pos_0)` to replay buffer.
       - Samples batch of 32 from replay buffer.
       - Performs standard model gradient update (`model.train()`, `optimizer.zero_grad()`, backprop, `model.zero_inactive_gradients()`, `optimizer.step()`). Set `priming_mode = "external" if step <= 1500 else "self"`.
       - Computes average tracking overlap for the batch for step > 1500 (using `model.compute_physical_tracking_overlap`).
       - Logs training metrics to CSV: `archive/iter_005/runs/{model_name}_seed{seed}.csv` (fields: `step`, `loss`, `l2_sim_loss`, `token_locus`, `l2_locked`, `overlap`).
       - Saves trained model to `archive/iter_005/runs/{model_name}_seed{seed}.pt`.

   - **Parallel Execution**:
     - Run the 15 training runs in parallel using `concurrent.futures.ProcessPoolExecutor(max_workers=3)`.

   - **Deterministic Collision Benchmark**:
     - Deterministically pre-generate 100 20-step trajectories of N=2 collision events using a fixed seed (e.g. 42) for randomizing hidden masses in `[2.0, 12.0]` and ensuring object 0 and object 1 are guaranteed to collide (e.g., `pos0=40.0`, `vel0=3.0`, `r0=5.0` vs `pos1=88.0`, `vel1=-3.0`, `r1=5.0`). Bounces pointer away (`pos=120.0`, `vel=0.0`).
     - Load the trained model for each run (model_name and seed) and evaluate post-collision L2 prediction loss (`l2_sim_loss` of the global latent space) over steps 8-20 (indices 5-17 in the 18 transitions) of the 100 trajectories under `priming_mode="self"`. Average across all trajectories.

   - **Representation Ablation Control**:
     - For `M_active` and each seed, run the test environment (N=3, seed=seed+10000) for 100 steps under three control configurations:
       1. Normal (no ablation).
       2. Random network ablation (`ablation="random"`).
       3. Spatial attention shuffling (`ablation="shuffle"`).
     - Measure and return the average physical tracking overlap of object 0 over the 100 steps.

   - **Priming Comparison**:
     - For `M_active` and each seed, run the test environment (N=3, seed=seed+10000) for 100 steps in:
       1. Primed attention mode: `priming_mode="external"` with target color fed as external query.
       2. Self-generated attention mode: `priming_mode="self"` with no external query.
     - Measure and return average `l2_sim_loss` over the 100 steps and compute their ratio (Self / Primed).

   - **Results Generation**:
     - Create `archive/iter_005/results/summary.csv` compiling mean and std across 5 seeds for all three models.
     - Create `archive/iter_005/results/falsification_report.md` performing the pre-registered falsification audit on the 4 falsification criteria.

2. **Run `src/train_eval_closed_loop.py`**:
   Execute the script and capture/print the full output showing training progress and evaluation results. Make sure all results files are correctly saved to `archive/iter_005/results/`.

Please execute the task completely, run the python script, ensure all files are written, and print the summary results and falsification report.