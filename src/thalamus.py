import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SegmentEncoder(nn.Module):
    def __init__(self, d_max=8):
        """
        Maps visual segment of shape (B, 3, 32) to latent space of shape (B, d_max).
        Uses 1D convolution layers.
        """
        super().__init__()
        self.d_max = d_max
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, d_max)

    def forward(self, x):
        # x: (B, 3, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        z = self.fc(x)
        return z

class SegmentPredictor(nn.Module):
    def __init__(self, d_max=8, h=3):
        """
        Predicts next segment latent state from a history of h * d_max values.
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

    def forward(self, z_history):
        # z_history: (B, h, d_max) or (B, h * d_max)
        if z_history.dim() == 3:
            z_history = z_history.reshape(-1, self.h * self.d_max)
        return self.net(z_history)

class L2Encoder(nn.Module):
    def __init__(self, d_max=8):
        """
        Maps aggregated/concatenated active L1 latents of all 4 segments (4 * d_max)
        to L2 latent space of size d_max.
        """
        super().__init__()
        self.d_max = d_max
        self.net = nn.Sequential(
            nn.Linear(4 * d_max, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, d_max)
        )

    def forward(self, x):
        # x: (B, 4 * d_max) or (B, 4, d_max)
        if x.dim() == 3:
            x = x.reshape(-1, 4 * self.d_max)
        return self.net(x)

class L2Predictor(nn.Module):
    def __init__(self, d_max=8, h=3):
        """
        Predicts next L2 latent state.
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

    def forward(self, z_history):
        # z_history: (B, h, d_max) or (B, h * d_max)
        if z_history.dim() == 3:
            z_history = z_history.reshape(-1, self.h * self.d_max)
        return self.net(z_history)

class ThalamusNet(nn.Module):
    def __init__(self, d_max=8, h=3, cooldown=200):
        super().__init__()
        self.d_max = d_max
        self.h = h
        self.cooldown = cooldown

        # 4 L1 segment modules
        self.l1_encoders = nn.ModuleList([SegmentEncoder(d_max) for _ in range(4)])
        self.l1_predictors = nn.ModuleList([SegmentPredictor(d_max, h) for _ in range(4)])

        # Global L2 modules
        self.l2_encoder = L2Encoder(d_max)
        self.l2_predictor = L2Predictor(d_max, h)

        # Color Readout for self-generation
        self.color_readout = nn.Linear(d_max, 3)

        # Token Routing & Cooldown
        self.token_locus = 0  # locus in {0, 1, 2, 3, 4}
        self.steps_since_change = cooldown  # startup can change immediately

        # Surprise Watchdog EMAs (0..3: L1 segments, 4: L2)
        self.register_buffer('surprise_mean', torch.zeros(5))
        self.register_buffer('surprise_var', torch.ones(5))
        self.ema_decay = 0.05
        self.initialized_watchdog = False

        # Robust Relative Stability Lock
        self.register_buffer('l1_running_surprise_ema', torch.tensor(1.0))
        self.register_buffer('l1_surprise_min', torch.tensor(float('inf')))
        self.initialized_stability_lock = False
        self.l2_locked = False

        self.last_query = None

    def calc_vicreg_loss(self, z_pred, z_target, sim_weight=25.0, var_weight=25.0, cov_weight=25.0):
        # Invariance (similarity)
        sim_loss = F.mse_loss(z_pred, z_target)

        # Variance loss
        def calc_var_loss(x, gamma=1.0, eps=1e-4):
            mean = x.mean(dim=0)
            var = torch.mean((x - mean)**2, dim=0)
            std = torch.sqrt(var + eps)
            return torch.mean(F.relu(gamma - std))

        var_loss = 0.5 * (calc_var_loss(z_pred) + calc_var_loss(z_target))

        # Covariance loss
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

        cov_loss = 0.5 * (calc_cov_loss(z_pred) + calc_cov_loss(z_target))

        loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
        return loss, sim_loss, var_loss, cov_loss

    def update_plasticity_gating(self):
        """
        Plasticity Gating: dynamically sets requires_grad based on active token locus.
        If L1 active (locus 0-3), only L1 parameters are trainable.
        If L2 active (locus 4), only L2 parameters (and color readout) are trainable.
        If L2 is locked, L2 parameters are always frozen.
        """
        l1_active = (self.token_locus in [0, 1, 2, 3])
        l2_active = (self.token_locus == 4) and not self.l2_locked

        for param in self.l1_encoders.parameters():
            param.requires_grad = l1_active
        for param in self.l1_predictors.parameters():
            param.requires_grad = l1_active

        for param in self.l2_encoder.parameters():
            param.requires_grad = l2_active
        for param in self.l2_predictor.parameters():
            param.requires_grad = l2_active
        for param in self.color_readout.parameters():
            param.requires_grad = l2_active

    def zero_inactive_gradients(self):
        """
        Ensures inactive layers receive zero gradients.
        """
        l1_active = (self.token_locus in [0, 1, 2, 3])
        l2_active = (self.token_locus == 4) and not self.l2_locked

        if not l1_active:
            for param in self.l1_encoders.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            for param in self.l1_predictors.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        if not l2_active:
            for param in self.l2_encoder.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            for param in self.l2_predictor.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            for param in self.color_readout.parameters():
                if param.grad is not None:
                    param.grad.zero_()

    def forward(self, x_hist, x_target, external_query=None, priming_mode="none", similarity_bias_weight=1.0, sim_weight=25.0, var_weight=25.0, cov_weight=25.0):
        """
        Args:
            x_hist: (B, H, 3, 128) - history of observations
            x_target: (B, 3, 128) - target observation
            external_query: (B, 3) or (3,) or None - query color
            priming_mode: "external", "self", or "none"
        """
        B, H, C, W = x_hist.shape
        assert W == 128, f"Expected width 128, got {W}"
        assert H == self.h, f"Expected history length {self.h}, got {H}"

        # 1. Encode L1 segments
        z_hist_segments = []
        for i in range(4):
            x_hist_seg = x_hist[:, :, :, i*32:(i+1)*32] # (B, H, 3, 32)
            x_hist_seg_flat = x_hist_seg.reshape(B * H, 3, 32)
            z_hist_seg_flat = self.l1_encoders[i](x_hist_seg_flat)
            z_hist_seg = z_hist_seg_flat.reshape(B, H, self.d_max)
            z_hist_segments.append(z_hist_seg)

        z_target_segments = []
        for i in range(4):
            x_target_seg = x_target[:, :, i*32:(i+1)*32] # (B, 3, 32)
            z_target_seg = self.l1_encoders[i](x_target_seg)
            z_target_segments.append(z_target_seg)

        # 2. Predict L1 segments
        z_pred_segments = []
        for i in range(4):
            z_pred_seg = self.l1_predictors[i](z_hist_segments[i])
            z_pred_segments.append(z_pred_seg)

        # 3. Process L2 layer
        z_hist_l2_list = []
        for t in range(H):
            l1_latents_t = torch.cat([z_hist_segments[i][:, t, :] for i in range(4)], dim=-1) # (B, 4 * d_max)
            z_hist_l2_t = self.l2_encoder(l1_latents_t)
            z_hist_l2_list.append(z_hist_l2_t)

        z_hist_l2 = torch.stack(z_hist_l2_list, dim=1) # (B, H, d_max)
        z_pred_l2 = self.l2_predictor(z_hist_l2) # (B, d_max)

        z_target_l1_all = torch.cat(z_target_segments, dim=-1)
        z_target_l2 = self.l2_encoder(z_target_l1_all)

        # 4. Compute temporal surprise (detached)
        raw_surprise_list = []
        for i in range(4):
            raw_s_i = torch.mean((z_target_segments[i].detach() - z_pred_segments[i].detach()) ** 2, dim=-1)
            raw_surprise_list.append(raw_s_i.mean())

        raw_s_l2 = torch.mean((z_target_l2.detach() - z_pred_l2.detach()) ** 2, dim=-1)
        raw_surprise_list.append(raw_s_l2.mean())

        raw_surprises = torch.stack(raw_surprise_list) # (5,)

        # 5. Priming & Self-Generation
        query = None
        if priming_mode == "external" and external_query is not None:
            query = external_query
            if query.dim() == 1:
                query = query.unsqueeze(0).expand(B, -1)
        elif priming_mode == "self":
            query = self.color_readout(z_pred_l2.detach())

        self.last_query = query.detach() if query is not None else None

        # Bias surprise based on color similarity
        if query is not None:
            for i in range(4):
                C_i = x_target[:, :, i*32:(i+1)*32].mean(dim=-1) # (B, 3)
                sim_i = -torch.mean((C_i - query)**2, dim=-1).mean()
                raw_surprises[i] = raw_surprises[i] + similarity_bias_weight * sim_i

        # 6. Surprise Watchdog
        if not self.initialized_watchdog:
            self.surprise_mean.copy_(raw_surprises)
            self.surprise_var.fill_(1.0)
            self.initialized_watchdog = True
        else:
            delta = raw_surprises - self.surprise_mean
            self.surprise_mean.add_(self.ema_decay * delta)
            self.surprise_var.add_(self.ema_decay * (delta ** 2 - self.surprise_var))

        std_watchdog = torch.sqrt(self.surprise_var + 1e-4)
        normalized_surprises = (raw_surprises - self.surprise_mean) / (std_watchdog + 1e-4)

        # 7. Robust Relative Stability Lock
        L1_surprise_current = raw_surprises[:4].mean()
        if not self.initialized_stability_lock:
            self.l1_running_surprise_ema.copy_(L1_surprise_current)
            self.l1_surprise_min.copy_(L1_surprise_current)
            self.initialized_stability_lock = True
        else:
            delta_l1 = L1_surprise_current - self.l1_running_surprise_ema
            self.l1_running_surprise_ema.add_(self.ema_decay * delta_l1)
            if self.l1_running_surprise_ema < self.l1_surprise_min:
                self.l1_surprise_min.copy_(self.l1_running_surprise_ema)

        theta_conv = torch.max(torch.tensor(0.25, device=raw_surprises.device), 1.5 * self.l1_surprise_min)
        self.l2_locked = (self.l1_running_surprise_ema > theta_conv).item()

        # 8. Token Routing & Cooldown
        self.steps_since_change += 1
        if self.steps_since_change >= self.cooldown:
            if self.l2_locked:
                candidate_locus = torch.argmax(normalized_surprises[:4]).item()
            else:
                candidate_locus = torch.argmax(normalized_surprises).item()

            if candidate_locus != self.token_locus:
                self.token_locus = candidate_locus
                self.steps_since_change = 0

        # 9. Plasticity Gating
        self.update_plasticity_gating()

        # 10. Compute VICReg losses
        l1_losses = []
        l1_sims = []
        l1_vars = []
        l1_covs = []
        for i in range(4):
            loss_i, sim_i, var_i, cov_i = self.calc_vicreg_loss(z_pred_segments[i], z_target_segments[i], sim_weight, var_weight, cov_weight)
            l1_losses.append(loss_i)
            l1_sims.append(sim_i)
            l1_vars.append(var_i)
            l1_covs.append(cov_i)

        total_l1_loss = torch.stack(l1_losses).mean()
        l2_loss, l2_sim, l2_var, l2_cov = self.calc_vicreg_loss(z_pred_l2, z_target_l2, sim_weight, var_weight, cov_weight)

        # Return active layer loss
        active_loss = total_l1_loss if (self.token_locus in [0, 1, 2, 3]) else l2_loss

        loss_dict = {
            "loss": active_loss,
            "total_l1_loss": total_l1_loss,
            "l2_loss": l2_loss,
            "l1_sim_loss": torch.stack(l1_sims).mean(),
            "l1_var_loss": torch.stack(l1_vars).mean(),
            "l1_cov_loss": torch.stack(l1_covs).mean(),
            "l2_sim_loss": l2_sim,
            "l2_var_loss": l2_var,
            "l2_cov_loss": l2_cov,
            "token_locus": self.token_locus,
            "l2_locked": self.l2_locked,
            "theta_conv": theta_conv.item()
        }

        return loss_dict, z_pred_segments, z_pred_l2

    def compute_physical_tracking_overlap(self, target_position, x_target, external_query=None):
        """
        Maps current attention locus to 1D sandbox physical coordinate and returns 1.0
        if it falls within the same 32-pixel segment as the target object, else 0.0.
        """
        B = x_target.shape[0]

        query = None
        if hasattr(self, 'last_query') and self.last_query is not None:
            query = self.last_query
        elif external_query is not None:
            query = external_query
            if query.dim() == 1:
                query = query.unsqueeze(0).expand(B, -1)

        attn_segments = []
        if self.token_locus in [0, 1, 2, 3]:
            attn_segments = [self.token_locus] * B
        else:
            if query is not None:
                for b in range(B):
                    best_sim = -float('inf')
                    best_seg = 0
                    q_b = query[b]
                    for i in range(4):
                        C_i = x_target[b, :, i*32:(i+1)*32].mean(dim=-1) # (3,)
                        sim = -torch.mean((C_i - q_b)**2).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_seg = i
                    attn_segments.append(best_seg)
            else:
                attn_segments = [0] * B

        if isinstance(target_position, (int, float)):
            target_positions = [target_position] * B
        elif isinstance(target_position, torch.Tensor):
            target_positions = target_position.cpu().tolist()
            if not isinstance(target_positions, list):
                target_positions = [target_positions]
        elif isinstance(target_position, np.ndarray):
            target_positions = target_position.tolist()
            if not isinstance(target_positions, list):
                target_positions = [target_positions]
        else:
            target_positions = list(target_position)

        overlaps = []
        for b in range(B):
            pos = target_positions[b]
            target_seg = int(pos // 32)
            target_seg = max(0, min(3, target_seg))
            overlap_val = 1.0 if attn_segments[b] == target_seg else 0.0
            overlaps.append(overlap_val)

        if len(overlaps) == 1 and isinstance(target_position, (int, float)):
            return overlaps[0]
        return torch.tensor(overlaps, device=x_target.device, dtype=torch.float32)


class NonGatedControlNet(nn.Module):
    def __init__(self, d_max=8, h=3):
        super().__init__()
        self.d_max = d_max
        self.h = h

        # 4 L1 segment modules
        self.l1_encoders = nn.ModuleList([SegmentEncoder(d_max) for _ in range(4)])
        self.l1_predictors = nn.ModuleList([SegmentPredictor(d_max, h) for _ in range(4)])

        # Global L2 modules
        self.l2_encoder = L2Encoder(d_max)
        self.l2_predictor = L2Predictor(d_max, h)

        # Color Readout
        self.color_readout = nn.Linear(d_max, 3)

        self.last_query = None

    def calc_vicreg_loss(self, z_pred, z_target, sim_weight=25.0, var_weight=25.0, cov_weight=25.0):
        # Invariance (similarity)
        sim_loss = F.mse_loss(z_pred, z_target)

        # Variance loss
        def calc_var_loss(x, gamma=1.0, eps=1e-4):
            mean = x.mean(dim=0)
            var = torch.mean((x - mean)**2, dim=0)
            std = torch.sqrt(var + eps)
            return torch.mean(F.relu(gamma - std))

        var_loss = 0.5 * (calc_var_loss(z_pred) + calc_var_loss(z_target))

        # Covariance loss
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

        cov_loss = 0.5 * (calc_cov_loss(z_pred) + calc_cov_loss(z_target))

        loss = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
        return loss, sim_loss, var_loss, cov_loss

    def forward(self, x_hist, x_target, external_query=None, priming_mode="none", similarity_bias_weight=1.0, sim_weight=25.0, var_weight=25.0, cov_weight=25.0):
        B, H, C, W = x_hist.shape
        assert W == 128, f"Expected width 128, got {W}"
        assert H == self.h, f"Expected history length {self.h}, got {H}"

        # 1. Encode L1 segments
        z_hist_segments = []
        for i in range(4):
            x_hist_seg = x_hist[:, :, :, i*32:(i+1)*32]
            x_hist_seg_flat = x_hist_seg.reshape(B * H, 3, 32)
            z_hist_seg_flat = self.l1_encoders[i](x_hist_seg_flat)
            z_hist_seg = z_hist_seg_flat.reshape(B, H, self.d_max)
            z_hist_segments.append(z_hist_seg)

        z_target_segments = []
        for i in range(4):
            x_target_seg = x_target[:, :, i*32:(i+1)*32]
            z_target_seg = self.l1_encoders[i](x_target_seg)
            z_target_segments.append(z_target_seg)

        # 2. Predict L1 segments
        z_pred_segments = []
        for i in range(4):
            z_pred_seg = self.l1_predictors[i](z_hist_segments[i])
            z_pred_segments.append(z_pred_seg)

        # 3. Process L2 layer
        z_hist_l2_list = []
        for t in range(H):
            l1_latents_t = torch.cat([z_hist_segments[i][:, t, :] for i in range(4)], dim=-1)
            z_hist_l2_t = self.l2_encoder(l1_latents_t)
            z_hist_l2_list.append(z_hist_l2_t)

        z_hist_l2 = torch.stack(z_hist_l2_list, dim=1)
        z_pred_l2 = self.l2_predictor(z_hist_l2)

        z_target_l1_all = torch.cat(z_target_segments, dim=-1)
        z_target_l2 = self.l2_encoder(z_target_l1_all)

        # 4. Color query tracking
        query = None
        if priming_mode == "external" and external_query is not None:
            query = external_query
            if query.dim() == 1:
                query = query.unsqueeze(0).expand(B, -1)
        elif priming_mode == "self":
            query = self.color_readout(z_pred_l2.detach())

        self.last_query = query.detach() if query is not None else None

        # 5. Compute VICReg losses (both layers continuously trained)
        l1_losses = []
        l1_sims = []
        l1_vars = []
        l1_covs = []
        for i in range(4):
            loss_i, sim_i, var_i, cov_i = self.calc_vicreg_loss(z_pred_segments[i], z_target_segments[i], sim_weight, var_weight, cov_weight)
            l1_losses.append(loss_i)
            l1_sims.append(sim_i)
            l1_vars.append(var_i)
            l1_covs.append(cov_i)

        total_l1_loss = torch.stack(l1_losses).mean()
        l2_loss, l2_sim, l2_var, l2_cov = self.calc_vicreg_loss(z_pred_l2, z_target_l2, sim_weight, var_weight, cov_weight)

        active_loss = total_l1_loss + l2_loss

        loss_dict = {
            "loss": active_loss,
            "total_l1_loss": total_l1_loss,
            "l2_loss": l2_loss,
            "l1_sim_loss": torch.stack(l1_sims).mean(),
            "l1_var_loss": torch.stack(l1_vars).mean(),
            "l1_cov_loss": torch.stack(l1_covs).mean(),
            "l2_sim_loss": l2_sim,
            "l2_var_loss": l2_var,
            "l2_cov_loss": l2_cov,
        }

        return loss_dict, z_pred_segments, z_pred_l2

    def compute_physical_tracking_overlap(self, target_position, x_target, external_query=None):
        """
        Computes the physical tracking overlap using the color query for evaluation.
        """
        B = x_target.shape[0]

        query = None
        if hasattr(self, 'last_query') and self.last_query is not None:
            query = self.last_query
        elif external_query is not None:
            query = external_query
            if query.dim() == 1:
                query = query.unsqueeze(0).expand(B, -1)

        attn_segments = []
        if query is not None:
            for b in range(B):
                best_sim = -float('inf')
                best_seg = 0
                q_b = query[b]
                for i in range(4):
                    C_i = x_target[b, :, i*32:(i+1)*32].mean(dim=-1) # (3,)
                    sim = -torch.mean((C_i - q_b)**2).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_seg = i
                attn_segments.append(best_seg)
        else:
            attn_segments = [0] * B

        if isinstance(target_position, (int, float)):
            target_positions = [target_position] * B
        elif isinstance(target_position, torch.Tensor):
            target_positions = target_position.cpu().tolist()
            if not isinstance(target_positions, list):
                target_positions = [target_positions]
        elif isinstance(target_position, np.ndarray):
            target_positions = target_position.tolist()
            if not isinstance(target_positions, list):
                target_positions = [target_positions]
        else:
            target_positions = list(target_position)

        overlaps = []
        for b in range(B):
            pos = target_positions[b]
            target_seg = int(pos // 32)
            target_seg = max(0, min(3, target_seg))
            overlap_val = 1.0 if attn_segments[b] == target_seg else 0.0
            overlaps.append(overlap_val)

        if len(overlaps) == 1 and isinstance(target_position, (int, float)):
            return overlaps[0]
        return torch.tensor(overlaps, device=x_target.device, dtype=torch.float32)


if __name__ == "__main__":
    print("Running self-contained verification suite for Thalamic Gated and Non-Gated networks...")
    
    # Setup dummy data
    B = 4
    H = 3
    x_hist = torch.rand(B, H, 3, 128)
    x_target = torch.rand(B, 3, 128)
    external_query = torch.rand(B, 3)
    target_positions = torch.tensor([15.0, 45.0, 75.0, 105.0]) # Segments 0, 1, 2, 3
    
    # 1. Verify ThalamusNet compilation and forward pass
    print("\n--- Verifying ThalamusNet ---")
    thalamus_net = ThalamusNet(d_max=8, h=H, cooldown=2)
    print("ThalamusNet compiled successfully.")
    
    loss_dict, z_pred_segments, z_pred_l2 = thalamus_net(
        x_hist, x_target, external_query=external_query, priming_mode="external"
    )
    print("Forward pass successful.")
    print("Keys in loss_dict:", list(loss_dict.keys()))
    print(f"Token locus: {loss_dict['token_locus']}")
    print(f"L2 locked: {loss_dict['l2_locked']}")
    
    # Verify backward pass and plasticity gating (gradient routing)
    # Test L1 holds token
    print("\nTesting Gated Gradient routing:")
    thalamus_net.token_locus = 1  # L1 holds token
    thalamus_net.l2_locked = False
    thalamus_net.steps_since_change = -100 # Prevent token override in forward pass
    thalamus_net.update_plasticity_gating()
    
    # Reset grads
    thalamus_net.zero_grad()
    loss_dict_l1, _, _ = thalamus_net(x_hist, x_target)
    loss_dict_l1["loss"].backward()
    thalamus_net.zero_inactive_gradients() # Force zero inactive
    
    # Check gradients
    l1_has_grad = False
    for p in thalamus_net.l1_encoders.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            l1_has_grad = True
            break
            
    l2_has_grad = False
    for p in thalamus_net.l2_encoder.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            l2_has_grad = True
            break
            
    print(f"When L1 holds token: L1 Has Gradients = {l1_has_grad}, L2 Has Gradients = {l2_has_grad}")
    assert l1_has_grad, "L1 parameters must receive gradients when L1 holds the token."
    assert not l2_has_grad, "L2 parameters must receive zero gradients when L1 holds the token."
    
    # Test L2 holds token
    thalamus_net.token_locus = 4  # L2 holds token
    thalamus_net.l2_locked = False
    thalamus_net.steps_since_change = -100 # Prevent token override in forward pass
    thalamus_net.update_plasticity_gating()
    
    # Reset grads
    thalamus_net.zero_grad()
    loss_dict_l2, _, _ = thalamus_net(x_hist, x_target)
    loss_dict_l2["loss"].backward()
    thalamus_net.zero_inactive_gradients() # Force zero inactive
    
    l1_has_grad = False
    for p in thalamus_net.l1_encoders.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            l1_has_grad = True
            break
            
    l2_has_grad = False
    for p in thalamus_net.l2_encoder.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            l2_has_grad = True
            break
            
    print(f"When L2 holds token: L1 Has Gradients = {l1_has_grad}, L2 Has Gradients = {l2_has_grad}")
    assert not l1_has_grad, "L1 parameters must receive zero gradients when L2 holds the token."
    assert l2_has_grad, "L2 parameters must receive gradients when L2 holds the token."
    
    # Verify physical tracking overlap
    overlap = thalamus_net.compute_physical_tracking_overlap(target_positions, x_target, external_query=external_query)
    print(f"ThalamusNet physical tracking overlap: {overlap}")
    assert overlap.shape == (B,), f"Expected shape ({B},), got {overlap.shape}"
    
    # 2. Verify NonGatedControlNet compilation and forward pass
    print("\n--- Verifying NonGatedControlNet ---")
    control_net = NonGatedControlNet(d_max=8, h=H)
    print("NonGatedControlNet compiled successfully.")
    
    loss_dict_c, _, _ = control_net(
        x_hist, x_target, external_query=external_query, priming_mode="external"
    )
    print("Forward pass successful.")
    print("Keys in loss_dict:", list(loss_dict_c.keys()))
    
    # Verify backward pass (both layers trained continuously)
    control_net.zero_grad()
    loss_dict_c["loss"].backward()
    
    l1_has_grad_c = False
    for p in control_net.l1_encoders.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            l1_has_grad_c = True
            break
            
    l2_has_grad_c = False
    for p in control_net.l2_encoder.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            l2_has_grad_c = True
            break
            
    print(f"Non-Gated control: L1 Has Gradients = {l1_has_grad_c}, L2 Has Gradients = {l2_has_grad_c}")
    assert l1_has_grad_c and l2_has_grad_c, "Both layers must continuously receive gradients in Non-Gated Baseline."
    
    overlap_c = control_net.compute_physical_tracking_overlap(target_positions, x_target, external_query=external_query)
    print(f"NonGatedControlNet physical tracking overlap: {overlap_c}")
    assert overlap_c.shape == (B,), f"Expected shape ({B},), got {overlap_c.shape}"
    
    print("\nAll self-contained verifications PASSED successfully!")
