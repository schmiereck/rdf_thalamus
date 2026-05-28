import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

def calculate_centroid_and_variance(a_spatial):
    """
    Computes spatial centroid and soft spatial variance for each channel.
    Args:
        a_spatial (Tensor): shape (B, d_max, 128)
    Returns:
        x_mean (Tensor): shape (B, d_max)
        var (Tensor): shape (B, d_max)
    """
    p_c = F.softmax(a_spatial, dim=-1) # shape (B, d_max, 128)
    coords = torch.arange(128, device=a_spatial.device, dtype=torch.float32)
    x_mean = torch.sum(coords * p_c, dim=-1) # shape (B, d_max)
    var = torch.sum(((coords.unsqueeze(0).unsqueeze(1) - x_mean.unsqueeze(-1)) ** 2) * p_c, dim=-1) # shape (B, d_max)
    return x_mean, var

def add_positional_encoding(x, pos_encoding="none"):
    if pos_encoding == "none":
        return x
    B, _, W = x.shape
    device = x.device
    coords = torch.arange(W, dtype=torch.float32, device=device)
    if pos_encoding == "linear":
        pos_linear = coords / max(1.0, float(W - 1))
        pos_linear = pos_linear.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        return torch.cat([x, pos_linear], dim=1)
    elif pos_encoding == "sinusoidal":
        pos_sin_10 = torch.sin(coords / 10.0)
        pos_cos_10 = torch.cos(coords / 10.0)
        pos_sin_100 = torch.sin(coords / 100.0)
        pos_cos_100 = torch.cos(coords / 100.0)
        pos_embeds = torch.stack([pos_sin_10, pos_cos_10, pos_sin_100, pos_cos_100], dim=0)
        pos_embeds = pos_embeds.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([x, pos_embeds], dim=1)
    return x

class DualStreamEncoder(nn.Module):
    def __init__(self, d_max=8):
        """
        1D CNN mapping input of shape (B, 3, 128) to decoupled coordinate and dynamics streams.
        """
        super().__init__()
        self.d_max = d_max
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        
        # Dual stream heads
        self.conv_spatial_coord = nn.Conv1d(128, d_max, kernel_size=1)
        self.conv_spatial_dyn = nn.Conv1d(128, d_max, kernel_size=1)
        
    def forward_spatial(self, x):
        """
        Returns spatial feature map of shape (B, d_max, 128) for the coordinate stream.
        To ensure backprop through loss_spatial only flows to conv_spatial_coord,
        we detach the output of conv4 before passing to conv_spatial_coord.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) # (B, 128, 8)
        
        # Detach before conv_spatial_coord to stop gradients flowing to conv1-4
        x_coord = self.conv_spatial_coord(x.detach())  # (B, d_max, 8)
        x_coord = F.interpolate(x_coord, size=128, mode='linear', align_corners=False) # (B, d_max, 128)
        return x_coord

    def forward_dynamics(self, x):
        """
        Returns the dynamics representation of shape (B, d_max).
        Gradients on this stream flow back to conv_spatial_dyn and conv1-4.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) # (B, 128, 8)
        
        x_dyn = self.conv_spatial_dyn(x) # (B, d_max, 8)
        return x_dyn.mean(dim=-1) # (B, d_max)

    def forward(self, x):
        """
        Returns both streams:
        z_coord: soft centroids, shape (B, d_max)
        z_dyn: dynamics representations, shape (B, d_max)
        """
        a_spatial = self.forward_spatial(x)
        z_coord, _ = calculate_centroid_and_variance(a_spatial)
        z_dyn = self.forward_dynamics(x)
        return z_coord, z_dyn


class DualStreamPredictor(nn.Module):
    def __init__(self, d_max=8, d_dyn=None, h=3):
        """
        MLP forecasting target z_{t+1} of size (B, d_max + d_dyn)
        from history of active latent states (B, H * (d_max + d_dyn)).
        """
        super().__init__()
        self.d_max = d_max
        self.d_dyn = d_dyn if d_dyn is not None else d_max
        self.h = h
        total_in = h * (d_max + self.d_dyn)
        total_out = d_max + self.d_dyn
        self.net = nn.Sequential(
            nn.Linear(total_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, total_out)
        )
        
    def forward(self, z_coord_history, z_dyn_history, d_t, mask_coord=False, d_t_dyn=None):
        """
        Args:
            z_coord_history (Tensor): Shape (B, H, D_max)
            z_dyn_history (Tensor): Shape (B, H, D_dyn)
            d_t (int): Current active latent dimension for coord stream.
            mask_coord (bool): If True, zero out the coordinate input.
            d_t_dyn (int or None): Current active latent dimension for dyn stream.
                                   Defaults to d_t if None.
        """
        if d_t_dyn is None:
            d_t_dyn = d_t
            
        # Option to zero out coordinate input completely
        if mask_coord:
            z_coord_history = torch.zeros_like(z_coord_history)
            
        # Zero out inactive dimensions (index >= d_t for coord, >= d_t_dyn for dyn)
        mask_c = torch.zeros_like(z_coord_history)
        mask_c[:, :, :d_t] = 1.0
        z_coord_active = z_coord_history * mask_c
        
        mask_d = torch.zeros_like(z_dyn_history)
        mask_d[:, :, :d_t_dyn] = 1.0
        z_dyn_active = z_dyn_history * mask_d
        
        # Concatenate coordinate and dynamics histories along feature dimension: (B, H, d_max + d_dyn)
        z_history = torch.cat([z_coord_active, z_dyn_active], dim=-1)
        
        # Flatten history: (B, H * (d_max + d_dyn))
        z_history_flat = z_history.reshape(-1, self.h * (self.d_max + self.d_dyn))
        
        # Predict both streams
        pred = self.net(z_history_flat) # (B, d_max + d_dyn)
        
        # Split into coord and dyn predictions
        pred_coord, pred_dyn = torch.split(pred, self.d_max, dim=-1)
        
        # Zero out inactive dimensions in the outputs
        out_mask_coord = torch.zeros_like(pred_coord)
        out_mask_coord[:, :d_t] = 1.0
        pred_coord_active = pred_coord * out_mask_coord
        
        out_mask_dyn = torch.zeros_like(pred_dyn)
        out_mask_dyn[:, :d_t_dyn] = 1.0
        pred_dyn_active = pred_dyn * out_mask_dyn
        
        return pred_coord_active, pred_dyn_active


class DualStreamJEPASpatial(nn.Module):
    def __init__(self, d_max=8, h=3, k=4, cooldown=300, stabilization_period=100):
        super().__init__()
        self.d_max = d_max
        self.h = h
        self.k = k
        self.cooldown = cooldown
        self.stabilization_period = stabilization_period
        
        self.encoder = DualStreamEncoder(d_max=d_max)
        self.predictor = DualStreamPredictor(d_max=d_max, h=h)
        
        # Dynamic state tracking
        self.d_t = 2
        self.steps_since_recruitment = cooldown  # start outside of cooldown
        self.error_buffer = collections.deque(maxlen=500)  # collects EMA of error during stable periods
        self.ema_error = None
        self.ema_alpha = 0.05
        self.gdasr_log_only = False
        self.gdasr_growth_points = []

    def reset_error_buffer(self):
        self.error_buffer.clear()
        self.ema_error = None

    def calculate_centroid_and_variance(self, a_spatial):
        return calculate_centroid_and_variance(a_spatial)

    def forward(self, x_hist, x_target, sim_weight=25.0, var_weight=25.0, cov_weight=25.0, lambda_spatial=0.0, k_chan=None, mask_coord=False):
        """
        Args:
            x_hist (Tensor): shape (B, H, 3, 128)
            x_target (Tensor): shape (B, 3, 128)
            sim_weight (float): VICReg invariance loss weight
            var_weight (float): VICReg variance loss weight
            cov_weight (float): VICReg covariance loss weight
            lambda_spatial (float): spatial bottleneck loss weight
            k_chan (int, optional): channel index to minimize spatial variance on. Defaults to d_t - 1.
            mask_coord (bool): whether to zero out coordinate history in predictor.
        """
        B, H, C, W = x_hist.shape
        x_hist_flat = x_hist.reshape(B * H, C, W)
        
        # Encode history
        z_hist_coord_flat, z_hist_dyn_flat = self.encoder(x_hist_flat)
        z_hist_coord = z_hist_coord_flat.reshape(B, H, self.d_max)
        z_hist_dyn = z_hist_dyn_flat.reshape(B, H, self.d_max)
        
        # Encode target
        z_target_coord, z_target_dyn = self.encoder(x_target)
        
        # Apply stabilization stop-gradient logic to history and target representations
        if self.steps_since_recruitment < self.stabilization_period and self.d_t > 1:
            z_hist_coord_stable = torch.cat([
                z_hist_coord[:, :, :self.d_t-1].detach(),
                z_hist_coord[:, :, self.d_t-1:]
            ], dim=-1)
            z_hist_dyn_stable = torch.cat([
                z_hist_dyn[:, :, :self.d_t-1].detach(),
                z_hist_dyn[:, :, self.d_t-1:]
            ], dim=-1)
            
            z_target_coord_stable = torch.cat([
                z_target_coord[:, :self.d_t-1].detach(),
                z_target_coord[:, self.d_t-1:]
            ], dim=-1)
            z_target_dyn_stable = torch.cat([
                z_target_dyn[:, :self.d_t-1].detach(),
                z_target_dyn[:, self.d_t-1:]
            ], dim=-1)
        else:
            z_hist_coord_stable = z_hist_coord
            z_hist_dyn_stable = z_hist_dyn
            
            z_target_coord_stable = z_target_coord
            z_target_dyn_stable = z_target_dyn
            
        # Predictive Coupling & Stop-Gradients on coordinate stream
        # This prevents prediction gradients from flowing to the coordinate stream representation
        z_hist_coord_pred = z_hist_coord_stable.detach()
        z_target_coord_pred = z_target_coord_stable.detach()
        
        # Predict target representations
        z_pred_coord, z_pred_dyn = self.predictor(
            z_hist_coord_pred, 
            z_hist_dyn_stable, 
            self.d_t, 
            mask_coord=mask_coord
        )
        
        # Apply stabilization stop-gradient to predictor outputs
        if self.steps_since_recruitment < self.stabilization_period and self.d_t > 1:
            z_pred_coord_stable = torch.cat([
                z_pred_coord[:, :self.d_t-1].detach(),
                z_pred_coord[:, self.d_t-1:]
            ], dim=-1)
            z_pred_dyn_stable = torch.cat([
                z_pred_dyn[:, :self.d_t-1].detach(),
                z_pred_dyn[:, self.d_t-1:]
            ], dim=-1)
        else:
            z_pred_coord_stable = z_pred_coord
            z_pred_dyn_stable = z_pred_dyn
            
        # Select active dimensions
        z_pred_coord_active = z_pred_coord_stable[:, :self.d_t]
        z_target_coord_active = z_target_coord_stable[:, :self.d_t]
        
        z_pred_dyn_active = z_pred_dyn_stable[:, :self.d_t]
        z_target_dyn_active = z_target_dyn_stable[:, :self.d_t]
        
        # 1. Similarity (Invariance) Loss
        sim_loss_coord = F.mse_loss(z_pred_coord_active, z_target_coord_pred[:, :self.d_t])
        sim_loss_dyn = F.mse_loss(z_pred_dyn_active, z_target_dyn_active)
        sim_loss = sim_loss_coord + sim_loss_dyn
        
        # 2. Variance Loss
        def calc_var_loss(x, gamma=1.0, eps=1e-4):
            mean = x.mean(dim=0)
            var = torch.mean((x - mean)**2, dim=0)
            std = torch.sqrt(var + eps)
            return torch.mean(F.relu(gamma - std))
            
        var_loss_coord = 0.5 * (calc_var_loss(z_pred_coord_active) + calc_var_loss(z_target_coord_active))
        var_loss_dyn = 0.5 * (calc_var_loss(z_pred_dyn_active) + calc_var_loss(z_target_dyn_active))
        var_loss = var_loss_coord + var_loss_dyn
        
        # 3. Covariance Loss
        def calc_cov_loss(x):
            B, d = x.shape
            if B <= 1 or d <= 1:
                return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            mean = x.mean(dim=0, keepdim=True)
            x_centered = x - mean
            cov = (x_centered.T @ x_centered) / (B - 1)
            diag = torch.diagonal(cov)
            off_diag = cov - torch.diag(diag)
            return (off_diag ** 2).sum() / d
            
        cov_loss_coord = 0.5 * (calc_cov_loss(z_pred_coord_active) + calc_cov_loss(z_target_coord_active))
        cov_loss_dyn = 0.5 * (calc_cov_loss(z_pred_dyn_active) + calc_cov_loss(z_target_dyn_active))
        cov_loss = cov_loss_coord + cov_loss_dyn
        
        base_loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
        
        # 4. Spatial Bottleneck Loss
        if lambda_spatial > 0:
            z_target_spatial = self.encoder.forward_spatial(x_target)
            _, var_all = self.calculate_centroid_and_variance(z_target_spatial)
            if k_chan is None:
                k_chan = self.d_t - 1
            var_k = var_all[:, k_chan]
            loss_spatial = lambda_spatial * var_k.mean()
            loss = base_loss + loss_spatial
        else:
            loss_spatial = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
            loss = base_loss
            
        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "sim_loss_coord": sim_loss_coord,
            "sim_loss_dyn": sim_loss_dyn,
            "var_loss": var_loss,
            "var_loss_coord": var_loss_coord,
            "var_loss_dyn": var_loss_dyn,
            "cov_loss": cov_loss,
            "cov_loss_coord": cov_loss_coord,
            "cov_loss_dyn": cov_loss_dyn,
            "loss_spatial": loss_spatial
        }, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn)

    def update_recruitment_logic(self, error_val, target_dim=None, step=None):
        if target_dim is None:
            target_dim = self.d_t

        if self.ema_error is None:
            self.ema_error = error_val
        else:
            self.ema_error = self.ema_alpha * error_val + (1.0 - self.ema_alpha) * self.ema_error
            
        self.steps_since_recruitment += 1
        
        if self.d_t == target_dim:
            self.error_buffer.append(self.ema_error)
            
        if self.d_t == target_dim and self.steps_since_recruitment > self.cooldown:
            if len(self.error_buffer) >= 200:
                mean = np.mean(self.error_buffer)
                std = np.std(self.error_buffer)
                if self.ema_error > mean + self.k * std:
                    if self.gdasr_log_only:
                        self.steps_since_recruitment = 0
                        self.gdasr_growth_points.append({
                            "step": step,
                            "ema_error": self.ema_error,
                            "mean": mean,
                            "std": std
                        })
                        print(f"[GDASR LOG-ONLY] Growth point detected at d_t={self.d_t}, "
                              f"error {self.ema_error:.4f} (baseline mean={mean:.4f}, std={std:.4f})")
                    else:
                        self.d_t = target_dim + 1
                        self.steps_since_recruitment = 0
                        print(f"[GDASR] Recruited dimension! d_t increased to {self.d_t} at error {self.ema_error:.4f} (baseline mean={mean:.4f}, std={std:.4f})")

    def clone(self):
        """
        Self-cloning capability: Returns a copy of this model with identical parameters and dynamic state.
        """
        import copy
        cloned = DualStreamJEPASpatial(
            d_max=self.d_max,
            h=self.h,
            k=self.k,
            cooldown=self.cooldown,
            stabilization_period=self.stabilization_period
        )
        cloned.d_t = self.d_t
        cloned.load_state_dict(self.state_dict())
        cloned.steps_since_recruitment = self.steps_since_recruitment
        cloned.error_buffer = copy.deepcopy(self.error_buffer)
        cloned.ema_error = self.ema_error
        cloned.gdasr_log_only = self.gdasr_log_only
        cloned.gdasr_growth_points = copy.deepcopy(self.gdasr_growth_points)
        return cloned


class PDRCEncoder(DualStreamEncoder):
    def __init__(self, d_max=8):
        super().__init__(d_max=d_max)
        self.stage = 1

    def forward_spatial(self, x):
        """
        Returns spatial feature map of shape (B, d_max, 128) for the coordinate stream.
        To ensure backprop through loss_spatial only flows to conv_spatial_coord,
        we detach the output of conv4 before passing to conv_spatial_coord in Stage 2.
        In Stage 1, we allow gradients to flow back to conv1-4.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) # (B, 128, 8)
        
        if self.stage == 2:
            x_coord = self.conv_spatial_coord(x.detach())
        else:
            x_coord = self.conv_spatial_coord(x)
            
        x_coord = F.interpolate(x_coord, size=128, mode='linear', align_corners=False) # (B, d_max, 128)
        return x_coord


class PDRCJEPASpatial(DualStreamJEPASpatial):
    def __init__(self, d_max=8, h=3, k=4, cooldown=300, stabilization_period=100, stage=1):
        super().__init__(d_max=d_max, h=h, k=k, cooldown=cooldown, stabilization_period=stabilization_period)
        self.encoder = PDRCEncoder(d_max=d_max)
        self._stage = stage
        self.stage = stage # Trigger property setter to configure requires_grad

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, value):
        if value not in [1, 2]:
            raise ValueError("Stage must be 1 or 2.")
        self._stage = value
        self.encoder.stage = value
        
        # Stage 1: All parameters should have requires_grad=True
        # Stage 2: Set requires_grad=False on encoder.conv_spatial_coord parameters
        #          The dynamics stream and predictor should remain fully trainable
        if value == 1:
            for p in self.parameters():
                p.requires_grad = True
        elif value == 2:
            for name, p in self.named_parameters():
                if "encoder.conv_spatial_coord" in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def forward(self, x_hist, x_target, sim_weight=25.0, var_weight=25.0, cov_weight=25.0, lambda_spatial=0.0, k_chan=None, mask_coord=False):
        B, H, C, W = x_hist.shape
        x_hist_flat = x_hist.reshape(B * H, C, W)
        
        # Encode history
        z_hist_coord_flat, z_hist_dyn_flat = self.encoder(x_hist_flat)
        z_hist_coord = z_hist_coord_flat.reshape(B, H, self.d_max)
        z_hist_dyn = z_hist_dyn_flat.reshape(B, H, self.d_max)
        
        # Encode target
        z_target_coord, z_target_dyn = self.encoder(x_target)
        
        # Stabilization stop-gradient logic depending on stage
        if self.stage == 1:
            # Stage 1: No detaching at all on coordinate representations
            z_hist_coord_stable = z_hist_coord
            z_target_coord_stable = z_target_coord
        else:
            # Stage 2: Apply standard stabilization stop-gradients to coordinate stream
            if self.steps_since_recruitment < self.stabilization_period and self.d_t > 1:
                z_hist_coord_stable = torch.cat([
                    z_hist_coord[:, :, :self.d_t-1].detach(),
                    z_hist_coord[:, :, self.d_t-1:]
                ], dim=-1)
                z_target_coord_stable = torch.cat([
                    z_target_coord[:, :self.d_t-1].detach(),
                    z_target_coord[:, self.d_t-1:]
                ], dim=-1)
            else:
                z_hist_coord_stable = z_hist_coord
                z_target_coord_stable = z_target_coord

        # Dynamics stream always has stabilization stop-gradient logic
        if self.steps_since_recruitment < self.stabilization_period and self.d_t > 1:
            z_hist_dyn_stable = torch.cat([
                z_hist_dyn[:, :, :self.d_t-1].detach(),
                z_hist_dyn[:, :, self.d_t-1:]
            ], dim=-1)
            z_target_dyn_stable = torch.cat([
                z_target_dyn[:, :self.d_t-1].detach(),
                z_target_dyn[:, self.d_t-1:]
            ], dim=-1)
        else:
            z_hist_dyn_stable = z_hist_dyn
            z_target_dyn_stable = z_target_dyn

        # Predictive Coupling & Stop-Gradients on coordinate stream
        if self.stage == 1:
            # Stage 1: Do NOT detach before passing to predictor or in similarity loss
            z_hist_coord_pred = z_hist_coord_stable
            z_target_coord_pred = z_target_coord_stable
        else:
            # Stage 2: Detach z_hist_coord and z_target_coord before predictor or similarity loss
            z_hist_coord_pred = z_hist_coord_stable.detach()
            z_target_coord_pred = z_target_coord_stable.detach()

        # Predict target representations
        z_pred_coord, z_pred_dyn = self.predictor(
            z_hist_coord_pred, 
            z_hist_dyn_stable, 
            self.d_t, 
            mask_coord=mask_coord
        )

        # Apply stabilization stop-gradient to predictor outputs
        if self.stage == 1:
            # Stage 1: Do NOT detach coordinate predictor output
            z_pred_coord_stable = z_pred_coord
        else:
            if self.steps_since_recruitment < self.stabilization_period and self.d_t > 1:
                z_pred_coord_stable = torch.cat([
                    z_pred_coord[:, :self.d_t-1].detach(),
                    z_pred_coord[:, self.d_t-1:]
                ], dim=-1)
            else:
                z_pred_coord_stable = z_pred_coord

        if self.steps_since_recruitment < self.stabilization_period and self.d_t > 1:
            z_pred_dyn_stable = torch.cat([
                z_pred_dyn[:, :self.d_t-1].detach(),
                z_pred_dyn[:, self.d_t-1:]
            ], dim=-1)
        else:
            z_pred_dyn_stable = z_pred_dyn

        # Select active dimensions
        z_pred_coord_active = z_pred_coord_stable[:, :self.d_t]
        z_target_coord_active = z_target_coord_stable[:, :self.d_t]
        
        z_pred_dyn_active = z_pred_dyn_stable[:, :self.d_t]
        z_target_dyn_active = z_target_dyn_stable[:, :self.d_t]
        
        # 1. Similarity (Invariance) Loss
        sim_loss_coord = F.mse_loss(z_pred_coord_active, z_target_coord_pred[:, :self.d_t])
        sim_loss_dyn = F.mse_loss(z_pred_dyn_active, z_target_dyn_active)
        sim_loss = sim_loss_coord + sim_loss_dyn
        
        # 2. Variance Loss
        def calc_var_loss(x, gamma=1.0, eps=1e-4):
            mean = x.mean(dim=0)
            var = torch.mean((x - mean)**2, dim=0)
            std = torch.sqrt(var + eps)
            return torch.mean(F.relu(gamma - std))
            
        var_loss_coord = 0.5 * (calc_var_loss(z_pred_coord_active) + calc_var_loss(z_target_coord_active))
        var_loss_dyn = 0.5 * (calc_var_loss(z_pred_dyn_active) + calc_var_loss(z_target_dyn_active))
        var_loss = var_loss_coord + var_loss_dyn
        
        # 3. Covariance Loss
        def calc_cov_loss(x):
            B, d = x.shape
            if B <= 1 or d <= 1:
                return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            mean = x.mean(dim=0, keepdim=True)
            x_centered = x - mean
            cov = (x_centered.T @ x_centered) / (B - 1)
            diag = torch.diagonal(cov)
            off_diag = cov - torch.diag(diag)
            return (off_diag ** 2).sum() / d
            
        cov_loss_coord = 0.5 * (calc_cov_loss(z_pred_coord_active) + calc_cov_loss(z_target_coord_active))
        cov_loss_dyn = 0.5 * (calc_cov_loss(z_pred_dyn_active) + calc_cov_loss(z_target_dyn_active))
        cov_loss = cov_loss_coord + cov_loss_dyn
        
        base_loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
        
        # 4. Spatial Bottleneck Loss
        if lambda_spatial > 0:
            z_target_spatial = self.encoder.forward_spatial(x_target)
            _, var_all = self.calculate_centroid_and_variance(z_target_spatial)
            if k_chan is None:
                k_chan = self.d_t - 1
            var_k = var_all[:, k_chan]
            loss_spatial = lambda_spatial * var_k.mean()
            loss = base_loss + loss_spatial
        else:
            loss_spatial = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
            loss = base_loss
            
        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "sim_loss_coord": sim_loss_coord,
            "sim_loss_dyn": sim_loss_dyn,
            "var_loss": var_loss,
            "var_loss_coord": var_loss_coord,
            "var_loss_dyn": var_loss_dyn,
            "cov_loss": cov_loss,
            "cov_loss_coord": cov_loss_coord,
            "cov_loss_dyn": cov_loss_dyn,
            "loss_spatial": loss_spatial
        }, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn)

    def clone(self):
        import copy
        cloned = PDRCJEPASpatial(
            d_max=self.d_max,
            h=self.h,
            k=self.k,
            cooldown=self.cooldown,
            stabilization_period=self.stabilization_period,
            stage=self.stage
        )
        cloned.d_t = self.d_t
        cloned.load_state_dict(self.state_dict())
        cloned.steps_since_recruitment = self.steps_since_recruitment
        cloned.error_buffer = copy.deepcopy(self.error_buffer)
        cloned.ema_error = self.ema_error
        return cloned


class NonParametricEncoder(nn.Module):
    def __init__(self, d_max=8, pos_encoding="none", dyn_readout="mean", sub_features=1, dyn_source="spatial"):
        super().__init__()
        self.d_max = d_max
        self.pos_encoding = pos_encoding
        self.dyn_readout = dyn_readout
        self.sub_features = sub_features
        self.dyn_source = dyn_source
        
        if pos_encoding == "none":
            in_channels = 3
        elif pos_encoding == "linear":
            in_channels = 4
        elif pos_encoding == "sinusoidal":
            in_channels = 7
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv_spatial = nn.Conv1d(128, d_max, kernel_size=1)
        
        if dyn_readout == "centroid_gated":
            if dyn_source == "conv4" and sub_features == 1:
                self.dyn_proj = nn.Linear(128, 1)
            else:
                self.conv_identity = nn.Conv1d(128, d_max * sub_features, kernel_size=1)

    @property
    def d_dyn(self):
        return self.d_max * self.sub_features

    def _forward_backbone(self, x):
        """Returns backbone features (B, 128, 8)"""
        x = add_positional_encoding(x, self.pos_encoding)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

    def forward_spatial(self, x):
        """
        Returns spatial feature map of shape (B, d_max, 128) for the encoder.
        """
        features = self._forward_backbone(x)
        x = self.conv_spatial(features)
        x = F.interpolate(x, size=128, mode='linear', align_corners=False)
        return x

    def forward(self, x):
        """
        Returns both streams:
        z_coord: soft centroids, shape (B, d_max)
        z_dyn: dynamics representations, shape (B, d_dyn) where d_dyn = d_max * sub_features
        """
        features = self._forward_backbone(x)
        a_spatial = self.conv_spatial(features)
        a_spatial = F.interpolate(a_spatial, size=128, mode='linear', align_corners=False)
        z_coord, _ = calculate_centroid_and_variance(a_spatial)
        B = x.shape[0]
        
        if self.dyn_readout == "mean":
            z_dyn = a_spatial.mean(dim=-1)
        elif self.dyn_readout == "centroid_gated":
            p_c = F.softmax(a_spatial, dim=-1)  # spatial attention
            
            if self.dyn_source == "conv4" and self.sub_features == 1:
                # Conv4-backed readout: attend conv4 features at centroid position
                features_interp = F.interpolate(features, size=128, mode='linear', align_corners=False)  # (B, 128, 128)
                attended = torch.bmm(p_c.detach(), features_interp.transpose(1, 2))  # (B, d_max, 128)
                attended_flat = attended.reshape(B * self.d_max, 128)
                z_dyn = self.dyn_proj(attended_flat).reshape(B, self.d_max)  # (B, d_max)
            elif self.sub_features > 1:
                # Multi-sub-feature readout
                a_identity = self.conv_identity(features)  # (B, d_max*K, 8)
                a_identity = F.interpolate(a_identity, size=128, mode='linear', align_corners=False)  # (B, d_max*K, 128)
                a_identity = a_identity.reshape(B, self.d_max, self.sub_features, 128)  # (B, d_max, K, 128)
                z_dyn = torch.einsum('bcs,bcks->bck', p_c.detach(), a_identity)  # (B, d_max, K)
                z_dyn = z_dyn.reshape(B, self.d_max * self.sub_features)  # (B, d_max * K)
            else:
                # Standard K=1 centroid_gated
                a_identity = self.conv_identity(features)
                a_identity = F.interpolate(a_identity, size=128, mode='linear', align_corners=False)
                z_dyn = torch.sum(a_identity * p_c.detach(), dim=-1)  # stop-gradient on attention
        else:
            raise ValueError(f"Unknown dyn_readout: {self.dyn_readout}")
        
        return z_coord, z_dyn




class NonParametricJEPASpatial(nn.Module):
    def __init__(self, d_max=8, h=3, k=4, cooldown=300, stabilization_period=100, pos_encoding="none",
                 primary_objective="jepa", sfa_weight=25.0, gdasr_log_only=True, dyn_readout="mean",
                 sub_features=1, dyn_source="spatial"):
        super().__init__()
        self.d_max = d_max
        self.h = h
        self.k = k
        self.cooldown = cooldown
        self.stabilization_period = stabilization_period
        self.pos_encoding = pos_encoding
        self.primary_objective = primary_objective
        self.sfa_weight = sfa_weight
        self.gdasr_log_only = gdasr_log_only
        self.dyn_readout = dyn_readout
        self.sub_features = sub_features
        self.dyn_source = dyn_source
        
        self.encoder = NonParametricEncoder(
            d_max=d_max, pos_encoding=pos_encoding, dyn_readout=dyn_readout,
            sub_features=sub_features, dyn_source=dyn_source
        )
        self.predictor = DualStreamPredictor(d_max=d_max, d_dyn=self.encoder.d_dyn, h=h)
        
        # Dynamic state tracking
        self.d_t = 2
        self.steps_since_recruitment = cooldown  # start outside of cooldown
        self.error_buffer = collections.deque(maxlen=500)
        self.ema_error = None
        self.ema_alpha = 0.05
        self.gdasr_growth_points = []

    def reset_error_buffer(self):
        self.error_buffer.clear()
        self.ema_error = None

    def calculate_centroid_and_variance(self, a_spatial):
        return calculate_centroid_and_variance(a_spatial)

    def forward(self, x_hist, x_target, sim_weight=25.0, var_weight=25.0, cov_weight=25.0, lambda_spatial=0.0, k_chan=None, mask_coord=False, ccr_mode='none', ccr_smooth_weight=10.0, ccr_spatial_weight=10.0, d_t_predict=None, sfa_weight=None):
        dt_pred = d_t_predict if d_t_predict is not None else self.d_t
        _sfa_weight = sfa_weight if sfa_weight is not None else self.sfa_weight
        d_dyn = self.encoder.d_dyn
        d_t_dyn = self.d_t * self.sub_features

        B, H, C, W = x_hist.shape
        x_hist_flat = x_hist.reshape(B * H, C, W)

        # Encode history
        z_hist_coord_flat, z_hist_dyn_flat = self.encoder(x_hist_flat)
        z_hist_coord = z_hist_coord_flat.reshape(B, H, self.d_max)
        z_hist_dyn = z_hist_dyn_flat.reshape(B, H, d_dyn)

        # Encode target
        z_target_coord, z_target_dyn = self.encoder(x_target)

        if self.primary_objective == "sfa":
            # SFA MODE
            z_prev_dyn = z_hist_dyn[:, -1]  # (B, d_dyn)

            z_target_dyn_active = z_target_dyn[:, :d_t_dyn]
            z_prev_dyn_active = z_prev_dyn[:, :d_t_dyn]
            sfa_loss = F.mse_loss(z_target_dyn_active, z_prev_dyn_active.detach())

            def calc_var_loss(x, gamma=1.0, eps=1e-4):
                mean = x.mean(dim=0)
                var = torch.mean((x - mean) ** 2, dim=0)
                std = torch.sqrt(var + eps)
                return torch.mean(F.relu(gamma - std))

            def calc_cov_loss(x):
                Bc, d = x.shape
                if Bc <= 1 or d <= 1:
                    return torch.tensor(0.0, device=x.device, dtype=x.dtype)
                mean = x.mean(dim=0, keepdim=True)
                x_centered = x - mean
                cov = (x_centered.T @ x_centered) / (Bc - 1)
                diag = torch.diagonal(cov)
                off_diag = cov - torch.diag(diag)
                return (off_diag ** 2).sum() / d

            var_loss_dyn = calc_var_loss(z_target_dyn_active)
            var_loss_coord = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
            var_loss = var_loss_dyn

            cov_loss_dyn = calc_cov_loss(z_target_dyn_active)
            cov_loss_coord = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
            cov_loss = cov_loss_dyn

            # JEPA readout with stop-gradient
            z_target_sfa_dyn = z_target_dyn_active.detach()
            z_hist_sfa_dyn = z_hist_dyn.detach()
            z_hist_sfa_coord = z_hist_coord.detach()

            z_pred_coord, z_pred_dyn = self.predictor(
                z_hist_sfa_coord,
                z_hist_sfa_dyn,
                dt_pred,
                mask_coord=mask_coord,
                d_t_dyn=dt_pred * self.sub_features
            )

            # Prediction losses on active dims
            z_pred_coord_pred = z_pred_coord[:, :dt_pred]
            z_target_coord_pred_dt = z_target_coord[:, :dt_pred]
            z_pred_dyn_pred = z_pred_dyn[:, :d_t_dyn]
            z_target_dyn_pred = z_target_dyn_active.detach()

            sim_loss_coord = F.mse_loss(z_pred_coord_pred, z_target_coord_pred_dt)
            sim_loss_dyn = F.mse_loss(z_pred_dyn_pred, z_target_dyn_pred)
            sim_loss = sim_loss_coord + sim_loss_dyn

            base_loss = _sfa_weight * sfa_loss + var_weight * var_loss + cov_weight * cov_loss + sim_weight * sim_loss

            # Spatial bottleneck
            if lambda_spatial > 0:
                z_target_spatial = self.encoder.forward_spatial(x_target)
                _, var_all = self.calculate_centroid_and_variance(z_target_spatial)
                if k_chan is None:
                    k_chan = self.d_t - 1
                var_k = var_all[:, k_chan]
                loss_spatial = lambda_spatial * var_k.mean()
                loss = base_loss + loss_spatial
            else:
                loss_spatial = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
                loss = base_loss

            # CCR
            ccr_smooth_loss = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
            ccr_spatial_loss = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)

            if ccr_mode != 'none':
                z_all_coord = torch.cat([z_hist_coord, z_target_coord.unsqueeze(1)], dim=1)
                z_all_norm = z_all_coord[:, :, :self.d_t] / 127.0
                diffs = z_all_norm[:, 1:] - z_all_norm[:, :-1]
                ccr_smooth_loss = torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-8).mean()

                if ccr_mode == 'hinge':
                    hinge_losses = []
                    for f in range(4):
                        z_f = z_all_norm[:, f]
                        if self.d_t > 1:
                            diff = torch.abs(z_f.unsqueeze(2) - z_f.unsqueeze(1))
                            triu_indices = torch.triu_indices(self.d_t, self.d_t, offset=1, device=z_f.device)
                            diff_pairs = diff[:, triu_indices[0], triu_indices[1]]
                            hinge = F.relu(0.15 - diff_pairs)
                            hinge_losses.append(hinge.mean())
                        else:
                            hinge_losses.append(torch.tensor(0.0, device=z_f.device, dtype=z_f.dtype))
                    ccr_spatial_loss = torch.stack(hinge_losses).mean()
                elif ccr_mode == 'covariance':
                    cov_losses = []
                    for f in range(4):
                        z_f = z_all_norm[:, f]
                        cov_losses.append(calc_cov_loss(z_f))
                    ccr_spatial_loss = torch.stack(cov_losses).mean()

                ccr_total = ccr_smooth_weight * ccr_smooth_loss + ccr_spatial_weight * ccr_spatial_loss
                loss = loss + ccr_total

            return {
                "loss": loss,
                "sim_loss": sim_loss,
                "sim_loss_coord": sim_loss_coord,
                "sim_loss_dyn": sim_loss_dyn,
                "var_loss": var_loss,
                "var_loss_coord": var_loss_coord,
                "var_loss_dyn": var_loss_dyn,
                "cov_loss": cov_loss,
                "cov_loss_coord": cov_loss_coord,
                "cov_loss_dyn": cov_loss_dyn,
                "loss_spatial": loss_spatial,
                "sfa_loss": sfa_loss,
                "ccr_smooth_loss": ccr_smooth_loss,
                "ccr_spatial_loss": ccr_spatial_loss
            }, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn)

        else:
            # JEPA MODE (default, backward compatible)
            z_hist_coord_pred = z_hist_coord
            z_target_coord_pred = z_target_coord

            z_pred_coord, z_pred_dyn = self.predictor(
                z_hist_coord_pred,
                z_hist_dyn,
                dt_pred,
                mask_coord=mask_coord,
                d_t_dyn=dt_pred * self.sub_features
            )

            # For JEPA mode with K>1, dt_pred dims for coord, dt_pred*K for dyn
            d_t_pred_dyn = dt_pred * self.sub_features
            z_pred_coord_pred = z_pred_coord[:, :dt_pred]
            z_target_coord_pred_dt = z_target_coord_pred[:, :dt_pred]

            z_pred_dyn_pred = z_pred_dyn[:, :d_t_pred_dyn]
            z_target_dyn_pred = z_target_dyn[:, :d_t_pred_dyn]

            z_pred_coord_active = z_pred_coord[:, :self.d_t]
            z_target_coord_active = z_target_coord[:, :self.d_t]

            z_pred_dyn_active = z_pred_dyn[:, :d_t_dyn]
            z_target_dyn_active = z_target_dyn[:, :d_t_dyn]

            sim_loss_coord = F.mse_loss(z_pred_coord_pred, z_target_coord_pred_dt)
            sim_loss_dyn = F.mse_loss(z_pred_dyn_pred, z_target_dyn_pred)
            sim_loss = sim_loss_coord + sim_loss_dyn

            def calc_var_loss(x, gamma=1.0, eps=1e-4):
                mean = x.mean(dim=0)
                var = torch.mean((x - mean) ** 2, dim=0)
                std = torch.sqrt(var + eps)
                return torch.mean(F.relu(gamma - std))

            def calc_cov_loss(x):
                Bc, d = x.shape
                if Bc <= 1 or d <= 1:
                    return torch.tensor(0.0, device=x.device, dtype=x.dtype)
                mean = x.mean(dim=0, keepdim=True)
                x_centered = x - mean
                cov = (x_centered.T @ x_centered) / (Bc - 1)
                diag = torch.diagonal(cov)
                off_diag = cov - torch.diag(diag)
                return (off_diag ** 2).sum() / d

            var_loss_coord = 0.5 * (calc_var_loss(z_pred_coord_active) + calc_var_loss(z_target_coord_active))
            var_loss_dyn = 0.5 * (calc_var_loss(z_pred_dyn_active) + calc_var_loss(z_target_dyn_active))
            var_loss = var_loss_coord + var_loss_dyn

            cov_loss_coord = 0.5 * (calc_cov_loss(z_pred_coord_active) + calc_cov_loss(z_target_coord_active))
            cov_loss_dyn = 0.5 * (calc_cov_loss(z_pred_dyn_active) + calc_cov_loss(z_target_dyn_active))
            cov_loss = cov_loss_coord + cov_loss_dyn

            base_loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss

            if lambda_spatial > 0:
                z_target_spatial = self.encoder.forward_spatial(x_target)
                _, var_all = self.calculate_centroid_and_variance(z_target_spatial)
                if k_chan is None:
                    k_chan = self.d_t - 1
                var_k = var_all[:, k_chan]
                loss_spatial = lambda_spatial * var_k.mean()
                loss = base_loss + loss_spatial
            else:
                loss_spatial = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
                loss = base_loss

            ccr_smooth_loss = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
            ccr_spatial_loss = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)

            if ccr_mode != 'none':
                z_all_coord = torch.cat([z_hist_coord, z_target_coord.unsqueeze(1)], dim=1)
                z_all_norm = z_all_coord[:, :, :self.d_t] / 127.0
                diffs = z_all_norm[:, 1:] - z_all_norm[:, :-1]
                ccr_smooth_loss = torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-8).mean()

                if ccr_mode == 'hinge':
                    hinge_losses = []
                    for f in range(4):
                        z_f = z_all_norm[:, f]
                        if self.d_t > 1:
                            diff = torch.abs(z_f.unsqueeze(2) - z_f.unsqueeze(1))
                            triu_indices = torch.triu_indices(self.d_t, self.d_t, offset=1, device=z_f.device)
                            diff_pairs = diff[:, triu_indices[0], triu_indices[1]]
                            hinge = F.relu(0.15 - diff_pairs)
                            hinge_losses.append(hinge.mean())
                        else:
                            hinge_losses.append(torch.tensor(0.0, device=z_f.device, dtype=z_f.dtype))
                    ccr_spatial_loss = torch.stack(hinge_losses).mean()
                elif ccr_mode == 'covariance':
                    cov_losses = []
                    for f in range(4):
                        z_f = z_all_norm[:, f]
                        cov_losses.append(calc_cov_loss(z_f))
                    ccr_spatial_loss = torch.stack(cov_losses).mean()

                ccr_total = ccr_smooth_weight * ccr_smooth_loss + ccr_spatial_weight * ccr_spatial_loss
                loss = loss + ccr_total

            return {
                "loss": loss,
                "sim_loss": sim_loss,
                "sim_loss_coord": sim_loss_coord,
                "sim_loss_dyn": sim_loss_dyn,
                "var_loss": var_loss,
                "var_loss_coord": var_loss_coord,
                "var_loss_dyn": var_loss_dyn,
                "cov_loss": cov_loss,
                "cov_loss_coord": cov_loss_coord,
                "cov_loss_dyn": cov_loss_dyn,
                "loss_spatial": loss_spatial,
                "ccr_smooth_loss": ccr_smooth_loss,
                "ccr_spatial_loss": ccr_spatial_loss
            }, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn)

    def update_recruitment_logic(self, error_val, target_dim=None, step=None):
        if target_dim is None:
            target_dim = self.d_t

        if self.ema_error is None:
            self.ema_error = error_val
        else:
            self.ema_error = self.ema_alpha * error_val + (1.0 - self.ema_alpha) * self.ema_error
            
        self.steps_since_recruitment += 1
        
        if self.d_t == target_dim:
            self.error_buffer.append(self.ema_error)

        # Log-only mode (M3): detect growth points without changing d_t
        if self.gdasr_log_only:
            if self.d_t == target_dim and self.steps_since_recruitment > self.cooldown:
                if len(self.error_buffer) >= 200:
                    mean = np.mean(self.error_buffer)
                    std = np.std(self.error_buffer)
                    if self.ema_error > mean + self.k * std:
                        self.steps_since_recruitment = 0
                        self.gdasr_growth_points.append({
                            "step": step,
                            "ema_error": self.ema_error,
                            "mean": mean,
                            "std": std
                        })
                        print(f"[GDASR LOG-ONLY] Growth point detected at d_t={self.d_t}, "
                              f"error {self.ema_error:.4f} (baseline mean={mean:.4f}, std={std:.4f})")
        else:
            # Active recruitment mode (backward compatible)
            if self.d_t == target_dim and self.steps_since_recruitment > self.cooldown:
                if len(self.error_buffer) >= 200:
                    mean = np.mean(self.error_buffer)
                    std = np.std(self.error_buffer)
                    if self.ema_error > mean + self.k * std:
                        self.d_t = target_dim + 1
                        self.steps_since_recruitment = 0
                        print(f"[GDASR] Recruited dimension! d_t increased to {self.d_t} at error {self.ema_error:.4f} (baseline mean={mean:.4f}, std={std:.4f})")

    def clone(self):
        import copy
        cloned = NonParametricJEPASpatial(
            d_max=self.d_max,
            h=self.h,
            k=self.k,
            cooldown=self.cooldown,
            stabilization_period=self.stabilization_period,
            pos_encoding=self.pos_encoding,
            primary_objective=self.primary_objective,
            sfa_weight=self.sfa_weight,
            gdasr_log_only=self.gdasr_log_only,
            dyn_readout=self.dyn_readout,
            sub_features=self.sub_features,
            dyn_source=self.dyn_source
        )
        cloned.d_t = self.d_t
        cloned.load_state_dict(self.state_dict())
        cloned.steps_since_recruitment = self.steps_since_recruitment
        cloned.error_buffer = copy.deepcopy(self.error_buffer)
        cloned.ema_error = self.ema_error
        cloned.gdasr_growth_points = copy.deepcopy(self.gdasr_growth_points)
        return cloned
