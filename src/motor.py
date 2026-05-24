import numpy as np
import torch

class SubsumptionMotorController:
    def __init__(self, Kp=1.0, Kd=0.1, Kv=0.5, error_thresh=4.0, stability_thresh=0.05, 
                 push_cooldown=15, push_magnitude=20.0, dt=1.0, ablation=None):
        """
        Subsumption Motor Controller with 3 hierarchical layers.
        
        Args:
            Kp (float): Proportional gain for reflexive PD tracking.
            Kd (float): Derivative gain for reflexive PD tracking.
            Kv (float): Velocity gain for predictive kinematics.
            error_thresh (float): Threshold under which the pointer is close enough to target to trigger boredom.
            stability_thresh (float): Threshold under which surprise change is stable.
            push_cooldown (int): Cooldown steps after a push is triggered.
            push_magnitude (float): Override acceleration magnitude during a deliberate push.
            dt (float): Control loop time step.
            ablation (str, optional): "random" or "shuffle".
        """
        self.Kp = Kp
        self.Kd = Kd
        self.Kv = Kv
        self.error_thresh = error_thresh
        self.stability_thresh = stability_thresh
        self.push_cooldown = push_cooldown
        self.push_magnitude = push_magnitude
        self.dt = dt
        self.ablation = ablation
        
        self.prev_error = None
        self.push_cooldown_timer = 0
        
    def reset(self):
        self.prev_error = None
        self.push_cooldown_timer = 0
        
    def get_action(self, model, obs, info, z_pred_segments, delta_E=0.0):
        """
        Computes the continuous acceleration and push decision.
        
        Args:
            model (nn.Module): The ThalamusNet or NonGatedControlNet instance.
            obs (np.ndarray): The current canvas observation (3, 128).
            info (dict): Diagnostic info from the environment step.
            z_pred_segments (list of torch.Tensor): Predicted latents from the model's forward pass.
            delta_E (float): Temporal change in local surprise (e.g. raw_surprise_t - raw_surprise_{t-1}).
        Returns:
            action (dict): {'acc': float, 'push': bool}
        """
        # --- Ablation: random ---
        if self.ablation == "random":
            acc = float(np.random.uniform(-10.0, 10.0))
            push = bool(np.random.rand() < 0.1)
            return {"acc": acc, "push": push}
            
        # Determine attention locus
        locus = getattr(model, "token_locus", 4)
        
        # --- Ablation: shuffle ---
        if self.ablation == "shuffle":
            # Shuffle locus to a random segment instead of the true active segment
            locus = int(np.random.randint(0, 4))
            
        # 1. Compute visual centroid of attended segment
        device = next(model.parameters()).device
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) # (1, 3, 128)
        
        with torch.no_grad():
            centroid_tensor = model.compute_attended_centroid(obs_tensor, locus=locus)
            target_pos = centroid_tensor.item()
            
        pointer_pos = info["pointer_pos"]
        pointer_vel = info["pointer_vel"]
        
        # 2. Lower Layer: Reflexive PD Tracking
        error = target_pos - pointer_pos
        if self.prev_error is None:
            self.prev_error = error
            
        dedt = (error - self.prev_error) / self.dt
        a_reflexive = self.Kp * error + self.Kd * dedt
        
        # 3. Middle Layer: Predictive Kinematics
        # Find segment to predict
        if locus == 4:
            # Fallback to segment with highest mean visual intensity
            seg_means = [obs[:, i*32:(i+1)*32].mean() for i in range(4)]
            pred_locus = int(np.argmax(seg_means))
        else:
            pred_locus = locus
            
        z_pred_attn = z_pred_segments[pred_locus] # (1, d_max) or (B, d_max)
        if z_pred_attn.dim() > 2:
            z_pred_attn = z_pred_attn[0]
            
        with torch.no_grad():
            pred_pos_future = model.pos_readout(z_pred_attn).mean().item()
            
        # Target predicted velocity
        v_predicted_target = (pred_pos_future - target_pos) / self.dt
        a_predictive = self.Kv * (v_predicted_target - pointer_vel)
        
        # 4. Upper Layer: Deliberate Push Perturbation
        if self.push_cooldown_timer > 0:
            self.push_cooldown_timer -= 1
            
        bored = (abs(error) < self.error_thresh) and (abs(delta_E) < self.stability_thresh)
        
        push = False
        if bored and self.push_cooldown_timer == 0:
            push = True
            direction = 1.0 if error >= 0 else -1.0
            acc = self.push_magnitude * direction
            self.push_cooldown_timer = self.push_cooldown
        else:
            # PD + Predictive kinematics control
            acc = a_reflexive + a_predictive
            
        self.prev_error = error
        
        return {"acc": float(acc), "push": push}


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
