import os
import subprocess

path = "src/thalamus.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replacement 1:
old_1 = """    def update_plasticity_gating(self):
        \"\"\"
        Plasticity Gating: dynamically sets requires_grad based on active token locus.
        If L1 active (locus 0-3), only L1 parameters are trainable.
        If L2 active (locus 4), only L2 parameters (and color readout) are trainable.
        If L2 is locked, L2 parameters are always frozen.
        \"\"\"
        l1_active = (self.token_locus in [0, 1, 2, 3])
        l2_active = (self.token_locus == 4) and not self.l2_locked"""

new_1 = """    def update_plasticity_gating(self):
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
            l2_active = (self.token_locus == 4) and not self.l2_locked"""

# Replacement 2:
old_2 = """    def zero_inactive_gradients(self):
        \"\"\"
        Ensures inactive layers receive zero gradients.
        \"\"\"
        l1_active = (self.token_locus in [0, 1, 2, 3])
        l2_active = (self.token_locus == 4) and not self.l2_locked"""

new_2 = """    def zero_inactive_gradients(self):
        \"\"\"
        Ensures inactive layers receive zero gradients.
        \"\"\"
        if self.token_locus == 4 and self.l2_locked:
            l1_active = True
            l2_active = False
        else:
            l1_active = (self.token_locus in [0, 1, 2, 3])
            l2_active = (self.token_locus == 4) and not self.l2_locked"""

# Replacement 3:
old_3 = """        # Return active layer loss
        active_loss = total_l1_loss if (self.token_locus in [0, 1, 2, 3]) else l2_loss"""

new_3 = """        # Return active layer loss
        if self.token_locus == 4 and self.l2_locked:
            active_loss = total_l1_loss
        else:
            active_loss = total_l1_loss if (self.token_locus in [0, 1, 2, 3]) else l2_loss"""

# Verify counts
failed = False
for name, old in [("1", old_1), ("2", old_2), ("3", old_3)]:
    count = content.count(old)
    if count != 1:
        print(f"Error: Found {count} occurrences of block {name}")
        failed = True

if failed:
    exit(1)

content = content.replace(old_1, new_1)
content = content.replace(old_2, new_2)
content = content.replace(old_3, new_3)

with open(path, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(content)

print("Patch applied successfully.")

# Run verification
print("Running verification...")
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
