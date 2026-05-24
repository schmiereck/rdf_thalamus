You are a research implementation agent. Write and execute the code for Iteration 12 immediately.

1. Implement the `CLTSMotorController` inside `src/motor.py` by appending it.
Here is the implementation of `CLTSMotorController`:
```python
class CLTSMotorController:
    def __init__(self, Kp=2.0, Kd=0.5, Kv=0.5, push_cooldown=15, push_magnitude=15.0, dt=1.0):
        self.Kp = Kp
        self.Kd = Kd
        self.Kv = Kv
        self.push_cooldown = push_cooldown
        self.push_magnitude = push_magnitude
        self.dt = dt
        
        self.prev_error = {}
        self.prev_target_pos = {}
        self.push_cooldown_timer = 0
        
        # Running statistics for surprise per channel
        self.mu = np.zeros(8)
        self.sigma = np.ones(8)
        self.ema_alpha = 0.05
        
        self.token_locus = 0
        self.attention_cooldown = 0
        self.attention_cooldown_max = 15

    def reset(self):
        self.prev_error = {}
        self.prev_target_pos = {}
        self.push_cooldown_timer = 0
        self.token_locus = 0
        self.attention_cooldown = 0
        self.mu = np.zeros(8)
        self.sigma = np.ones(8)

    def get_action(self, model, obs, info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, d_t, centroids):
        surprises = []
        for c in range(d_t):
            err_coord = torch.mean((z_pred_coord[:, c] - z_target_coord[:, c])**2).item()
            err_dyn = torch.mean((z_pred_dyn[:, c] - z_target_dyn[:, c])**2).item()
            s = err_coord + err_dyn
            surprises.append(s)
            
            # Update running statistics (online EMA mean and variance)
            self.mu[c] = (1 - self.ema_alpha) * self.mu[c] + self.ema_alpha * s
            diff_sq = (s - self.mu[c]) ** 2
            var_c = (1 - self.ema_alpha) * (self.sigma[c]**2) + self.ema_alpha * diff_sq
            self.sigma[c] = np.sqrt(var_c + 1e-8)
            
        norm_surprises = []
        for c in range(d_t):
            norm_s = (surprises[c] - self.mu[c]) / (self.sigma[c] + 1e-8)
            norm_surprises.append(norm_s)
            
        if self.attention_cooldown > 0:
            self.attention_cooldown -= 1
        else:
            self.token_locus = int(np.argmax(norm_surprises[:d_t]))
            self.attention_cooldown = self.attention_cooldown_max
            
        target_pos = centroids[0, self.token_locus].item()
        pointer_pos = info["pointer_pos"]
        pointer_vel = info["pointer_vel"]
        
        error = target_pos - pointer_pos
        prev_err = self.prev_error.get(self.token_locus, error)
        dedt = (error - prev_err) / self.dt
        self.prev_error[self.token_locus] = error
        a_reflexive = self.Kp * error + self.Kd * dedt
        
        prev_target = self.prev_target_pos.get(self.token_locus, target_pos)
        v_target = (target_pos - prev_target) / self.dt
        self.prev_target_pos[self.token_locus] = target_pos
        a_predictive = self.Kv * (v_target - pointer_vel)
        
        if self.push_cooldown_timer > 0:
            self.push_cooldown_timer -= 1
            
        surprise_val = surprises[self.token_locus]
        surprise_threshold = self.mu[self.token_locus] + 1.0 * self.sigma[self.token_locus]
        
        push = False
        if surprise_val > surprise_threshold and self.push_cooldown_timer == 0:
            direction = 1.0 if error >= 0 else -1.0
            acc = self.push_magnitude * direction
            if abs(error) <= 6.0:
                push = True
                self.push_cooldown_timer = self.push_cooldown
        else:
            acc = a_reflexive + a_predictive
            
        return {"acc": float(acc), "push": push}, self.token_locus, surprises
```

2. Create `src/run_phase12_experiments.py` which:
   - Imports `NonParametricJEPASpatial` from `src.models_dual_stream`.
   - Imports `PhysicsSandbox` from `src.environment`.
   - Bootstraps training of `NonParametricJEPASpatial` passively on N=3 for 1500 steps (for 5 seeds: [42, 123, 456, 789, 999]).
   - Clones model at step 1501 into three arms: F-Passive, F-Random, G-CLTS.
   - At 1501, initializes N=4 sandbox and abruptly multiplies the mass of the 4th object (object 3) by 2.0.
   - Trains each arm from steps 1501 to 3000 using their respective motor policies:
     - F-Passive: `{"acc": 0.0, "push": False}`
     - F-Random: `{"acc": uniform(-10, 10), "push": random < 0.1}`
     - G-CLTS: uses `CLTSMotorController` (pass model, obs, info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, d_t, centroids). Note: the controller should run after the forward pass, and the resulting action is used for the next step.
   - Saves model checkpoints at [1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000].
   - Evaluates each checkpoint on a standardized passive test set (N=4, 2x mass perturbation on object 3) generated with seed+5000.
   - Computes:
     - Test simulation loss at step 3000.
     - Centroid decoding MSE on the novel 4th object at step 3000.
     - Soft spatial variance of the 4th coordinate channel at step 3000.
     - Pearson correlation between predicted centroid and true position of 4th object.
     - Spatial coverage entropy of the pointer.
     - Offline AUC of prediction error (simulation loss) on the standardized passive evaluation sequence across the checkpoints from 1501 to 3000.
   - Audits the pre-registered falsification criteria:
     - Criterion 1: Does CLTS achieve >=20% lower post-collision latent prediction MSE than Arm F-Random?
     - Criterion 2: Is soft spatial variance <=20.0 and centroid decoding MSE <=85.0 for CLTS?
     - Criterion 3: Is AUC adaptation time for CLTS shorter by >=15% compared to F-Random?
   - Saves summary CSV to `archive/iter_012/results/summary_phase12.csv`.
   - Plots adaptation test curves (average across seeds of standardized test simulation loss at checkpoint steps) and saves to `archive/iter_012/results/auc_recovery_curves.png`.

Execute the script and ensure it completes successfully, producing all the required files.