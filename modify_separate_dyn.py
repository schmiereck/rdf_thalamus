with open('src/models_separate_dyn.py', 'r') as f:
    content = f.read()

# 1. Modify the __init__ of SeparateDynEncoder
old_init = "        self.conv_identity_dyn = nn.Conv1d(128, d_max, kernel_size=1)"
new_init = """        if self.dyn_readout == "centroid_gated":
            self.conv_identity_dyn = nn.Conv1d(128, d_max * sub_features, kernel_size=1)
        else:
            self.conv_identity_dyn = nn.Conv1d(128, d_max, kernel_size=1)"""

if old_init in content:
    content = content.replace(old_init, new_init)
    print("Successfully replaced old_init")
else:
    print("Warning: old_init not found!")

# 2. Modify the forward of SeparateDynEncoder
old_forward = """        # Dyn stream (independent backbone)
        dyn_features = self._forward_dyn_backbone(x)          # (B, 128, 8)
        a_dyn = self.conv_identity_dyn(dyn_features)           # (B, d_max, 8)
        z_dyn = a_dyn.mean(dim=-1)                             # (B, d_max)

        return z_coord, z_dyn"""

new_forward = """        # Dyn stream (independent backbone)
        dyn_features = self._forward_dyn_backbone(x)          # (B, 128, 8)
        a_dyn = self.conv_identity_dyn(dyn_features)           # (B, d_max, 8) or (B, d_max * sub_features, 8)

        if self.dyn_readout == "mean":
            z_dyn = a_dyn.mean(dim=-1)                         # (B, d_max)
        elif self.dyn_readout == "centroid_gated":
            p_c = F.softmax(a_spatial, dim=-1) # shape (B, d_max, 128)
            p_c_detached = p_c.detach()
            B = x.shape[0]
            if self.sub_features == 1:
                a_dyn_128 = F.interpolate(a_dyn, size=128, mode='linear', align_corners=False) # (B, d_max, 128)
                z_dyn = torch.einsum('bcs,bcs->bc', p_c_detached, a_dyn_128) # (B, d_max)
            else:
                a_dyn_128 = F.interpolate(a_dyn, size=128, mode='linear', align_corners=False) # (B, d_max * K, 128)
                a_dyn_128 = a_dyn_128.reshape(B, self.d_max, self.sub_features, 128)
                z_dyn = torch.einsum('bcs,bcks->bck', p_c_detached, a_dyn_128) # (B, d_max, K)
                z_dyn = z_dyn.reshape(B, self.d_max * self.sub_features) # (B, d_max * K)
        else:
            raise ValueError(f"Unknown dyn_readout: {self.dyn_readout}")

        return z_coord, z_dyn"""

if old_forward in content:
    content = content.replace(old_forward, new_forward)
    print("Successfully replaced old_forward")
else:
    print("Warning: old_forward not found!")

with open('src/models_separate_dyn.py', 'w') as f:
    f.write(content)

print("Modification complete!")
