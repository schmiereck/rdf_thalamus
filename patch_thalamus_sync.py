import os
import subprocess

path = "src/thalamus.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replacement 1: update_plasticity_gating
old_1 = """    def update_plasticity_gating(self):
        \"\"\"
        Plasticity Gating: dynamically sets requires_grad based on active token locus.
        If L1 active (locus 0-3), only L1 parameters are trainable.
        If L2 active (locus 4), only L2 parameters (and color readout) are trainable.
        If L2 is locked, L2 parameters are always frozen, and we fall back to training L1.
        \"\"\"
        if self.token_locus == 4 and self.l2_locked:
            l1_active = True
            l2_active = False
        else:
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
            param.requires_grad = l2_active"""

new_1 = """    def update_plasticity_gating(self):
        \"\"\"
        Plasticity Gating: dynamically sets requires_grad based on active token locus.
        If L1 active (locus 0-3), only L1 parameters are trainable.
        If L2 active (locus 4), only L2 parameters (and color readout) are trainable.
        If L2 is locked, L2 parameters are always frozen, and we fall back to training L1.
        \"\"\"
        locus = getattr(self, "current_forward_locus", self.token_locus)
        locked = getattr(self, "current_forward_locked", self.l2_locked)
        
        if locus == 4 and locked:
            l1_active = True
            l2_active = False
        else:
            l1_active = (locus in [0, 1, 2, 3])
            l2_active = (locus == 4) and not locked

        for param in self.l1_encoders.parameters():
            param.requires_grad = l1_active
        for param in self.l1_predictors.parameters():
            param.requires_grad = l1_active

        for param in self.l2_encoder.parameters():
            param.requires_grad = l2_active
        for param in self.l2_predictor.parameters():
            param.requires_grad = l2_active
        for param in self.color_readout.parameters():
            param.requires_grad = l2_active"""

# Replacement 2: zero_inactive_gradients
old_2 = """    def zero_inactive_gradients(self):
        \"\"\"
        Ensures inactive layers receive zero gradients.
        \"\"\"
        if self.token_locus == 4 and self.l2_locked:
            l1_active = True
            l2_active = False
        else:
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
                    param.grad.zero_()"""

new_2 = """    def zero_inactive_gradients(self):
        \"\"\"
        Ensures inactive layers receive zero gradients.
        \"\"\"
        locus = getattr(self, "current_forward_locus", self.token_locus)
        locked = getattr(self, "current_forward_locked", self.l2_locked)
        
        if locus == 4 and locked:
            l1_active = True
            l2_active = False
        else:
            l1_active = (locus in [0, 1, 2, 3])
            l2_active = (locus == 4) and not locked

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
                    param.grad.zero_()"""

# Replacement 3: forward function beginning
old_3 = """    def forward(self, x_hist, x_target, external_query=None, priming_mode="none", similarity_bias_weight=1.0, sim_weight=25.0, var_weight=25.0, cov_weight=25.0):
        \"\"\"
        Args:
            x_hist: (B, H, 3, 128) - history of observations
            x_target: (B, 3, 128) - target observation
            external_query: (B, 3) or (3,) or None - query color
            priming_mode: "external", "self", or "none"
        \"\"\"
        B, H, C, W = x_hist.shape"""

new_3 = """    def forward(self, x_hist, x_target, external_query=None, priming_mode="none", similarity_bias_weight=1.0, sim_weight=25.0, var_weight=25.0, cov_weight=25.0):
        \"\"\"
        Args:
            x_hist: (B, H, 3, 128) - history of observations
            x_target: (B, 3, 128) - target observation
            external_query: (B, 3) or (3,) or None - query color
            priming_mode: "external", "self", or "none"
        \"\"\"
        self.current_forward_locus = self.token_locus
        self.current_forward_locked = self.l2_locked
        self.update_plasticity_gating()

        B, H, C, W = x_hist.shape"""

# Replacement 4: active_loss calculation
old_4 = """        # Return active layer loss
        if self.token_locus == 4 and self.l2_locked:
            active_loss = total_l1_loss
        else:
            active_loss = total_l1_loss if (self.token_locus in [0, 1, 2, 3]) else l2_loss"""

new_4 = """        # Return active layer loss
        if self.current_forward_locus == 4 and self.current_forward_locked:
            active_loss = total_l1_loss
        else:
            active_loss = total_l1_loss if (self.current_forward_locus in [0, 1, 2, 3]) else l2_loss"""

# Verify and apply replacements
failed = False
for name, old in [("1", old_1), ("2", old_2), ("3", old_3), ("4", old_4)]:
    # Standardize line endings just in case
    old_norm = old.replace("\r\n", "\n")
    content_norm = content.replace("\r\n", "\n")
    count = content_norm.count(old_norm)
    if count != 1:
        print(f"Error: Found {count} occurrences of block {name}")
        failed = True

if failed:
    exit(1)

content_norm = content_norm.replace(old_1.replace("\r\n", "\n"), new_1.replace("\r\n", "\n"))
content_norm = content_norm.replace(old_2.replace("\r\n", "\n"), new_2.replace("\r\n", "\n"))
content_norm = content_norm.replace(old_3.replace("\r\n", "\n"), new_3.replace("\r\n", "\n"))
content_norm = content_norm.replace(old_4.replace("\r\n", "\n"), new_4.replace("\r\n", "\n"))

# Write back with platform-appropriate line endings (or keep original style)
with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(content_norm)

print("Patch applied successfully.")

# Run verification suite in src/thalamus.py
print("Running verification suite...")
res = subprocess.run([".venv/Scripts/python.exe", "src/thalamus.py"], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout)
print("STDERR:")
print(res.stderr)

if res.returncode == 0:
    print("Verification PASSED!")
    exit(0)
else:
    print("Verification FAILED!")
    exit(res.returncode)
