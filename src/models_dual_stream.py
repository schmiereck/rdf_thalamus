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
    def __init__(self, d_max=8, h=3):
        """
        MLP forecasting target z_{t+1} of size (B, 2 * D_max)
        from history of active latent states (B, H * 2 * D_max).
        """
        super().__init__()
        self.d_max = d_max
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(h * 2 * d_max, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * d_max)
        )
        
    def forward(self, z_coord_history, z_dyn_history, d_t, mask_coord=False):
        """
        Args:
            z_coord_history (Tensor): Shape (B, H, D_max)
            z_dyn_history (Tensor): Shape (B, H, D_max)
            d_t (int): Current active latent dimension.
            mask_coord (bool): If True, zero out the coordinate input.
        """
        # Option to zero out coordinate input completely
        if mask_coord:
            z_coord_history = torch.zeros_like(z_coord_history)
            
        # Zero out inactive dimensions (index >= d_t)
        mask_c = torch.zeros_like(z_coord_history)
        mask_c[:, :, :d_t] = 1.0
        z_coord_active = z_coord_history * mask_c
        
        mask_d = torch.zeros_like(z_dyn_history)
        mask_d[:, :, :d_t] = 1.0
        z_dyn_active = z_dyn_history * mask_d
        
        # Concatenate coordinate and dynamics histories along feature dimension: (B, H, 2 * d_max)
        z_history = torch.cat([z_coord_active, z_dyn_active], dim=-1)
        
        # Flatten history: (B, H * 2 * d_max)
        z_history_flat = z_history.reshape(-1, self.h * 2 * self.d_max)
        
        # Predict both streams
        pred = self.net(z_history_flat) # (B, 2 * d_max)
        
        # Split into coord and dyn predictions
        pred_coord, pred_dyn = torch.split(pred, self.d_max, dim=-1)
        
        # Zero out inactive dimensions in the outputs
        out_mask = torch.zeros_like(pred_coord)
        out_mask[:, :d_t] = 1.0
        
        pred_coord_active = pred_coord * out_mask
        pred_dyn_active = pred_dyn * out_mask
        
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

    def update_recruitment_logic(self, error_val, target_dim=None):
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
    def __init__(self, d_max=8, pos_encoding="none"):
        super().__init__()
        self.d_max = d_max
        self.pos_encoding = pos_encoding
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

    def forward_spatial(self, x):
        """
        Returns spatial feature map of shape (B, d_max, 128) for the encoder.
        """
        x = add_positional_encoding(x, self.pos_encoding)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) # (B, 128, 8)
        x = self.conv_spatial(x) # (B, d_max, 8)
        x = F.interpolate(x, size=128, mode='linear', align_corners=False) # (B, d_max, 128)
        return x

    def forward(self, x):
        """
        Returns both streams:
        z_coord: soft centroids, shape (B, d_max)
        z_dyn: dynamics representations computed as mean of a_spatial over spatial dim, shape (B, d_max)
        """
        a_spatial = self.forward_spatial(x)
        z_coord, _ = calculate_centroid_and_variance(a_spatial)
        z_dyn = a_spatial.mean(dim=-1)
        return z_coord, z_dyn


class NonParametricJEPASpatial(nn.Module):
    def __init__(self, d_max=8, h=3, k=4, cooldown=300, stabilization_period=100, pos_encoding="none"):
        super().__init__()
        self.d_max = d_max
        self.h = h
        self.k = k
        self.cooldown = cooldown
        self.stabilization_period = stabilization_period
        self.pos_encoding = pos_encoding
        
        self.encoder = NonParametricEncoder(d_max=d_max, pos_encoding=pos_encoding)
        self.predictor = DualStreamPredictor(d_max=d_max, h=h)
        
        # Dynamic state tracking
        self.d_t = 2
        self.steps_since_recruitment = cooldown  # start outside of cooldown
        self.error_buffer = collections.deque(maxlen=500)  # collects EMA of error during stable periods
        self.ema_error = None
        self.ema_alpha = 0.05

    def reset_error_buffer(self):
        self.error_buffer.clear()
        self.ema_error = None

    def calculate_centroid_and_variance(self, a_spatial):
        return calculate_centroid_and_variance(a_spatial)

    def forward(self, x_hist, x_target, sim_weight=25.0, var_weight=25.0, cov_weight=25.0, lambda_spatial=0.0, k_chan=None, mask_coord=False, ccr_mode='none', ccr_smooth_weight=10.0, ccr_spatial_weight=10.0, d_t_predict=None):
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
            ccr_mode (str): CCR mode ('none', 'hinge', 'covariance').
            ccr_smooth_weight (float): CCR smoothness loss weight.
            ccr_spatial_weight (float): CCR spatial loss weight.
            d_t_predict (int or None): Number of dimensions the predictor should predict.
                If None, defaults to self.d_t (standard behavior). If set to a value
                less than self.d_t, the predictor only receives/outputs d_t_predict
                active dimensions, keeping its higher-index dimensions untrained.
                This is used by the ESUG evaluation protocol to isolate the encoder's
                new-dimension signal from predictor confounds.
        """
        # Determine effective prediction dimension
        dt_pred = d_t_predict if d_t_predict is not None else self.d_t
        
        B, H, C, W = x_hist.shape
        x_hist_flat = x_hist.reshape(B * H, C, W)
        
        # Encode history
        z_hist_coord_flat, z_hist_dyn_flat = self.encoder(x_hist_flat)
        z_hist_coord = z_hist_coord_flat.reshape(B, H, self.d_max)
        z_hist_dyn = z_hist_dyn_flat.reshape(B, H, self.d_max)
        
        # Encode target
        z_target_coord, z_target_dyn = self.encoder(x_target)
        
        # In NonParametricJEPASpatial, there are no stop-gradients/detaches on the coordinate stream
        z_hist_coord_pred = z_hist_coord
        z_target_coord_pred = z_target_coord
        
        # Predict target representations using dt_pred for the predictor's active dim mask
        z_pred_coord, z_pred_dyn = self.predictor(
            z_hist_coord_pred, 
            z_hist_dyn, 
            dt_pred,  # <-- use dt_pred instead of self.d_t
            mask_coord=mask_coord
        )
        
        # Compute coordinate/dynamics prediction over dt_pred dimensions
        z_pred_coord_pred = z_pred_coord[:, :dt_pred]
        z_target_coord_pred_dt = z_target_coord_pred[:, :dt_pred]
        
        z_pred_dyn_pred = z_pred_dyn[:, :dt_pred]
        z_target_dyn_pred = z_target_dyn[:, :dt_pred]
        
        # Select all self.d_t active dimensions for variance and covariance losses
        z_pred_coord_active = z_pred_coord[:, :self.d_t]
        z_target_coord_active = z_target_coord[:, :self.d_t]
        
        z_pred_dyn_active = z_pred_dyn[:, :self.d_t]
        z_target_dyn_active = z_target_dyn[:, :self.d_t]
        
        # 1. Similarity (Invariance) Loss — computed only over dt_pred dimensions
        sim_loss_coord = F.mse_loss(z_pred_coord_pred, z_target_coord_pred_dt)
        sim_loss_dyn = F.mse_loss(z_pred_dyn_pred, z_target_dyn_pred)
        sim_loss = sim_loss_coord + sim_loss_dyn
        
        # 2. Variance Loss — computed over all self.d_t dimensions
        def calc_var_loss(x, gamma=1.0, eps=1e-4):
            mean = x.mean(dim=0)
            var = torch.mean((x - mean)**2, dim=0)
            std = torch.sqrt(var + eps)
            return torch.mean(F.relu(gamma - std))
            
        var_loss_coord = 0.5 * (calc_var_loss(z_pred_coord_active) + calc_var_loss(z_target_coord_active))
        var_loss_dyn = 0.5 * (calc_var_loss(z_pred_dyn_active) + calc_var_loss(z_target_dyn_active))
        var_loss = var_loss_coord + var_loss_dyn
        
        # 3. Covariance Loss — computed over all self.d_t dimensions
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
            
        # 5. Contrastive Coordinate Regularization (CCR)
        ccr_smooth_loss = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
        ccr_spatial_loss = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
        
        if ccr_mode != 'none':
            z_all_coord = torch.cat([z_hist_coord, z_target_coord.unsqueeze(1)], dim=1) # (B, 4, d_max)
            z_all_norm = z_all_coord[:, :, :self.d_t] / 127.0 # (B, 4, d_t)
            
            diffs = z_all_norm[:, 1:] - z_all_norm[:, :-1] # (B, 3, d_t)
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

    def update_recruitment_logic(self, error_val, target_dim=None):
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
            pos_encoding=self.pos_encoding
        )
        cloned.d_t = self.d_t
        cloned.load_state_dict(self.state_dict())
        cloned.steps_since_recruitment = self.steps_since_recruitment
        cloned.error_buffer = copy.deepcopy(self.error_buffer)
        cloned.ema_error = self.ema_error
        return cloned
