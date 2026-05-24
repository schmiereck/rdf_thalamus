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

class Encoder(nn.Module):
    def __init__(self, d_max=8):
        """
        1D CNN mapping input of shape (B, 3, 128) to latent space.
        """
        super().__init__()
        self.d_max = d_max
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv_spatial = nn.Conv1d(128, d_max, kernel_size=1)
        
    def forward_spatial(self, x):
        """
        Returns a (B, d_max, 128) spatial feature map.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) # (B, 128, 8)
        x = self.conv_spatial(x)  # (B, d_max, 8)
        x = F.interpolate(x, size=128, mode='linear', align_corners=False) # (B, d_max, 128)
        return x

    def forward(self, x):
        """
        Returns the global average (mean) of the spatial feature map over the spatial dimension,
        returning a (B, d_max) vector.
        """
        feat_map = self.forward_spatial(x)
        return feat_map.mean(dim=-1)

class Predictor(nn.Module):
    def __init__(self, d_max=8, h=3):
        """
        MLP forecasting latent states z_{t+1} of size (B, D_max)
        from history of active latent states (B, H * D_max).
        Inactive dimensions are padded/zeroed out.
        """
        super().__init__()
        self.d_max = d_max
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(h * d_max, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, d_max)
        )
        
    def forward(self, z_history, d_t):
        """
        Args:
            z_history (Tensor): Shape (B, H, D_max)
            d_t (int): Current active latent dimension.
        """
        # Zero out inactive dimensions (index >= d_t)
        mask = torch.zeros_like(z_history)
        mask[:, :, :d_t] = 1.0
        z_history_active = z_history * mask
        
        # Flatten history
        z_history_flat = z_history_active.reshape(-1, self.h * self.d_max)
        
        # Predict
        z_pred = self.net(z_history_flat)
        
        # Zero out inactive dimensions in the output
        out_mask = torch.zeros_like(z_pred)
        out_mask[:, :d_t] = 1.0
        z_pred_active = z_pred * out_mask
        
        return z_pred_active

class DynamicJEPASpatial(nn.Module):
    def __init__(self, d_max=8, h=3, k=4, cooldown=500, stabilization_period=200):
        """
        Dynamic Joint Embedding Predictive Architecture with GDASR and Spatial Bottleneck capabilities.
        """
        super().__init__()
        self.d_max = d_max
        self.h = h
        self.k = k
        self.cooldown = cooldown
        self.stabilization_period = stabilization_period
        
        self.encoder = Encoder(d_max=d_max)
        self.predictor = Predictor(d_max=d_max, h=h)
        
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

    def forward(self, x_hist, x_target, sim_weight=25.0, var_weight=25.0, cov_weight=25.0, lambda_spatial=0.0, k_chan=None):
        """
        Args:
            x_hist (Tensor): shape (B, H, 3, 128)
            x_target (Tensor): shape (B, 3, 128)
            sim_weight (float): VICReg invariance loss weight
            var_weight (float): VICReg variance loss weight
            cov_weight (float): VICReg covariance loss weight
            lambda_spatial (float): spatial bottleneck loss weight
            k_chan (int, optional): specified channel index to minimize spatial variance on. Defaults to d_t - 1.
        """
        B, H, C, W = x_hist.shape
        x_hist_flat = x_hist.reshape(B * H, C, W)
        z_hist_flat = self.encoder(x_hist_flat)
        z_hist = z_hist_flat.reshape(B, H, self.d_max)
        
        z_target = self.encoder(x_target)
        
        # Apply stabilization stop-gradient logic to history and target representations
        # During stabilization, detach the first d_t - 1 dimensions.
        if self.steps_since_recruitment < self.stabilization_period and self.d_t > 1:
            z_hist_stable = torch.cat([
                z_hist[:, :, :self.d_t-1].detach(),
                z_hist[:, :, self.d_t-1:]
            ], dim=-1)
            z_target_stable = torch.cat([
                z_target[:, :self.d_t-1].detach(),
                z_target[:, self.d_t-1:]
            ], dim=-1)
        else:
            z_hist_stable = z_hist
            z_target_stable = z_target
            
        z_pred = self.predictor(z_hist_stable, self.d_t)
        
        # Apply stabilization stop-gradient logic to predictor output
        if self.steps_since_recruitment < self.stabilization_period and self.d_t > 1:
            z_pred_stable = torch.cat([
                z_pred[:, :self.d_t-1].detach(),
                z_pred[:, self.d_t-1:]
            ], dim=-1)
        else:
            z_pred_stable = z_pred
            
        # Active dimensions only
        z_pred_active = z_pred_stable[:, :self.d_t]
        z_target_active = z_target_stable[:, :self.d_t]
        
        # 1. Invariance (similarity) loss
        sim_loss = F.mse_loss(z_pred_active, z_target_active)
        
        # 2. Variance loss
        def calc_var_loss(x, gamma=1.0, eps=1e-4):
            mean = x.mean(dim=0)
            var = torch.mean((x - mean)**2, dim=0)
            std = torch.sqrt(var + eps)
            return torch.mean(F.relu(gamma - std))
            
        var_loss = 0.5 * (calc_var_loss(z_pred_active) + calc_var_loss(z_target_active))
        
        # 3. Covariance loss
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
            
        cov_loss = 0.5 * (calc_cov_loss(z_pred_active) + calc_cov_loss(z_target_active))
        
        base_loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
        
        # 4. Spatial Bottleneck Loss
        if lambda_spatial > 0:
            # Get target spatial activations
            z_target_spatial = self.encoder.forward_spatial(x_target)
            # Calculate centroid and variance
            _, var_all = self.calculate_centroid_and_variance(z_target_spatial)
            
            # Select channel k
            if k_chan is None:
                k_chan = self.d_t - 1
            
            # Var_k for the specified/recruited channel (shape B)
            var_k = var_all[:, k_chan]
            loss_spatial = lambda_spatial * var_k.mean()
            loss = base_loss + loss_spatial
        else:
            loss_spatial = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
            loss = base_loss
            
        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "loss_spatial": loss_spatial
        }, z_pred, z_target

    def update_recruitment_logic(self, error_val, target_dim=None):
        """
        Updates EMA of prediction error and checks recruitment criteria.
        Args:
            error_val (float): Current prediction error (sim_loss.item())
            target_dim (int, optional): The dimension we are recruiting from. Defaults to self.d_t.
        """
        if target_dim is None:
            target_dim = self.d_t

        # 1. Update EMA
        if self.ema_error is None:
            self.ema_error = error_val
        else:
            self.ema_error = self.ema_alpha * error_val + (1.0 - self.ema_alpha) * self.ema_error
            
        # 2. Increment steps
        self.steps_since_recruitment += 1
        
        # 3. Buffer stable target_dim errors
        if self.d_t == target_dim:
            self.error_buffer.append(self.ema_error)
            
        # 4. Check recruitment condition
        if self.d_t == target_dim and self.steps_since_recruitment > self.cooldown:
            if len(self.error_buffer) >= 200:
                mean = np.mean(self.error_buffer)
                std = np.std(self.error_buffer)
                if self.ema_error > mean + self.k * std:
                    # Recruit new dimension
                    self.d_t = target_dim + 1
                    self.steps_since_recruitment = 0
                    print(f"[GDASR] Recruited dimension! d_t increased to {self.d_t} at error {self.ema_error:.4f} (baseline mean={mean:.4f}, std={std:.4f})")
