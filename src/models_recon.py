"""
Reconstruction + VICReg architecture using SeparateDynEncoder.

Provides:
  - ReconVICRegSeparateDyn: Encoder (SeparateDynEncoder) + Decoder (deconv head)
    with VICReg losses and JEPA predictor readout.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from src.models_separate_dyn import SeparateDynEncoder
from src.models_dual_stream import (
    calculate_centroid_and_variance,
    add_positional_encoding,
    DualStreamPredictor,
)


def calc_var_loss(x, gamma=1.0, eps=1e-4):
    mean = x.mean(dim=0)
    var = torch.mean((x - mean) ** 2, dim=0)
    std = torch.sqrt(var + eps)
    return torch.mean(F.relu(gamma - std))


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


class ReconDecoder(nn.Module):
    """
    Deconvolutional decoder head mapping (B, d_max, 8) -> (B, 3, 128).
    """

    def __init__(self, d_max=8):
        super().__init__()
        self.d_max = d_max
        self.deconv1 = nn.ConvTranspose1d(d_max, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, a_dyn):
        """
        Args:
            a_dyn (Tensor): shape (B, d_max, 8)
        Returns:
            x_recon (Tensor): shape (B, 3, 128)
        """
        x = F.relu(self.deconv1(a_dyn))   # (B, 128, 16)
        x = F.relu(self.deconv2(x))       # (B, 64, 32)
        x = F.relu(self.deconv3(x))       # (B, 32, 64)
        x = self.deconv4(x)               # (B, 3, 128)
        return x


class ReconVICRegSeparateDyn(nn.Module):
    """
    Reconstruction + VICReg architecture with SeparateDynEncoder.

    Forward pass:
      1. Encode x_target -> z_coord, z_dyn (manual encoder steps)
      2. Decode a_dyn spatial features -> x_recon
      3. Recon loss: MSE(x_recon, x_target)
      4. VICReg: var_loss + cov_loss on active dims
      5. JEPA readout: predictor with stop-gradient (surprise only)
      6. Total: recon_weight * recon_loss + var_weight * var_loss +
                 cov_weight * cov_loss + sim_weight * sim_loss
    """

    def __init__(self, d_max=8, h=3, k=4, cooldown=300, stabilization_period=100,
                 pos_encoding="none", dyn_readout="mean", sub_features=1,
                 dyn_source="spatial", coord_vicreg=True,
                 recon_weight=1.0, var_weight=25.0, cov_weight=25.0, sim_weight=1.0):
        super().__init__()
        self.d_max = d_max
        self.h = h
        self.k = k
        self.cooldown = cooldown
        self.stabilization_period = stabilization_period
        self.pos_encoding = pos_encoding
        self.dyn_readout = dyn_readout
        self.sub_features = sub_features
        self.dyn_source = dyn_source
        self.coord_vicreg = coord_vicreg
        self.recon_weight = recon_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.sim_weight = sim_weight

        # Encoder
        self.encoder = SeparateDynEncoder(
            d_max=d_max, pos_encoding=pos_encoding, dyn_readout=dyn_readout,
            sub_features=sub_features, dyn_source=dyn_source
        )

        # Decoder
        self.decoder = ReconDecoder(d_max=d_max)

        # Predictor (JEPA readout)
        self.predictor = DualStreamPredictor(d_max=d_max, d_dyn=self.encoder.d_dyn, h=h)

        # Color probe head (for supervised color regression)
        self.color_probe_weight = nn.Parameter(torch.randn(d_max, 3) * 0.01)
        self.color_probe_bias = nn.Parameter(torch.zeros(d_max, 3))

        # ID contrastive projection head (scalar channel -> 8D embedding)
        self.id_contrastive_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # Dynamic state tracking
        self.d_t = 2
        self.steps_since_recruitment = cooldown
        self.error_buffer = collections.deque(maxlen=500)
        self.ema_error = None
        self.ema_alpha = 0.05
        self.gdasr_growth_points = []

    @property
    def d_dyn(self):
        return self.encoder.d_dyn

    def forward(self, x_hist, x_target):
        """
        Args:
            x_hist (Tensor): shape (B, H, 3, 128)
            x_target (Tensor): shape (B, 3, 128)
        Returns:
            loss_dict (dict): containing all loss terms
            (z_pred_coord, z_pred_dyn): predictor outputs
            (z_target_coord, z_target_dyn): target encoder outputs
        """
        B, H, C, W = x_hist.shape
        x_hist_flat = x_hist.reshape(B * H, C, W)

        # --- Encode target manually ---
        coord_features = self.encoder._forward_coord_backbone(x_target)
        a_spatial = self.encoder.conv_spatial(coord_features)
        a_spatial = F.interpolate(a_spatial, size=128, mode='linear', align_corners=False)
        z_target_coord, _ = calculate_centroid_and_variance(a_spatial)

        dyn_features = self.encoder._forward_dyn_backbone(x_target)
        a_dyn = self.encoder.conv_identity_dyn(dyn_features)  # (B, d_max, 8)
        z_target_dyn = a_dyn.mean(dim=-1)                     # (B, d_max)

        # --- Decode ---
        x_recon = self.decoder(a_dyn)  # (B, 3, 128)
        recon_loss = F.mse_loss(x_recon, x_target)

        # --- VICReg on target representations ---
        d_t_dyn = self.d_t * self.sub_features
        z_target_dyn_active = z_target_dyn[:, :d_t_dyn]
        var_loss_dyn = calc_var_loss(z_target_dyn_active)
        cov_loss_dyn = calc_cov_loss(z_target_dyn_active)

        if self.coord_vicreg:
            z_target_coord_active = z_target_coord[:, :self.d_t]
            var_loss_coord = calc_var_loss(z_target_coord_active)
            cov_loss_coord = calc_cov_loss(z_target_coord_active)
        else:
            var_loss_coord = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)
            cov_loss_coord = torch.tensor(0.0, device=x_target.device, dtype=x_target.dtype)

        var_loss = var_loss_dyn + var_loss_coord
        cov_loss = cov_loss_dyn + cov_loss_coord

        # --- JEPA readout with stop-gradient (surprise only) ---
        # Encode history
        z_hist_coord_flat, z_hist_dyn_flat = self.encoder(x_hist_flat)
        z_hist_coord = z_hist_coord_flat.reshape(B, H, self.d_max)
        z_hist_dyn = z_hist_dyn_flat.reshape(B, H, self.d_dyn)

        # Detach encoder outputs to prevent guidance of encoder
        z_hist_coord_sg = z_hist_coord.detach()
        z_hist_dyn_sg = z_hist_dyn.detach()
        z_target_coord_sg = z_target_coord.detach()
        z_target_dyn_sg = z_target_dyn.detach()

        z_pred_coord, z_pred_dyn = self.predictor(
            z_hist_coord_sg,
            z_hist_dyn_sg,
            self.d_t,
            d_t_dyn=d_t_dyn
        )

        # Similarity loss on active dims (surprise readout only)
        z_pred_coord_active = z_pred_coord[:, :self.d_t]
        z_target_coord_active_sg = z_target_coord_sg[:, :self.d_t]
        z_pred_dyn_active = z_pred_dyn[:, :d_t_dyn]
        z_target_dyn_active_sg = z_target_dyn_sg[:, :d_t_dyn]

        sim_loss_coord = F.mse_loss(z_pred_coord_active, z_target_coord_active_sg)
        sim_loss_dyn = F.mse_loss(z_pred_dyn_active, z_target_dyn_active_sg)
        sim_loss = sim_loss_coord + sim_loss_dyn

        # --- Total loss ---
        loss = (
            self.recon_weight * recon_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
            + self.sim_weight * sim_loss
        )

        loss_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "var_loss": var_loss,
            "var_loss_dyn": var_loss_dyn,
            "var_loss_coord": var_loss_coord,
            "cov_loss": cov_loss,
            "cov_loss_dyn": cov_loss_dyn,
            "cov_loss_coord": cov_loss_coord,
            "sim_loss": sim_loss,
            "sim_loss_coord": sim_loss_coord,
            "sim_loss_dyn": sim_loss_dyn,
        }

        return loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn)

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
                    self.d_t = target_dim + 1
                    self.steps_since_recruitment = 0
                    print(f"[GDASR] Recruited dimension! d_t increased to {self.d_t} "
                          f"at error {self.ema_error:.4f} (baseline mean={mean:.4f}, std={std:.4f})")

    def clone(self):
        import copy
        cloned = ReconVICRegSeparateDyn(
            d_max=self.d_max,
            h=self.h,
            k=self.k,
            cooldown=self.cooldown,
            stabilization_period=self.stabilization_period,
            pos_encoding=self.pos_encoding,
            dyn_readout=self.dyn_readout,
            sub_features=self.sub_features,
            dyn_source=self.dyn_source,
            coord_vicreg=self.coord_vicreg,
            recon_weight=self.recon_weight,
            var_weight=self.var_weight,
            cov_weight=self.cov_weight,
            sim_weight=self.sim_weight,
        )
        cloned.d_t = self.d_t
        cloned.load_state_dict(self.state_dict())
        cloned.steps_since_recruitment = self.steps_since_recruitment
        cloned.error_buffer = copy.deepcopy(self.error_buffer)
        cloned.ema_error = self.ema_error
        cloned.gdasr_growth_points = copy.deepcopy(self.gdasr_growth_points)
        return cloned
