Create a file 'patch_thalamus.py' that patches 'src/thalamus.py' to implement the L1 plasticity fallback when L2 is locked.

The script must do the following:
1. Read the contents of 'src/thalamus.py'.
2. Perform these string replacements:

Replacement 1:
Find:
```python
    def update_plasticity_gating(self):
        \"\"\"
        Plasticity Gating: dynamically sets requires_grad based on active token locus.
        If L1 active (locus 0-3), only L1 parameters are trainable.
        If L2 active (locus 4), only L2 parameters (and color readout) are trainable.
        If L2 is locked, L2 parameters are always frozen.
        \"\"\"
        l1_active = (self.token_locus in [0, 1, 2, 3])
        l2_active = (self.token_locus == 4) and not self.l2_locked
```
Replace with:
```python
    def update_plasticity_gating(self):
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
```

Replacement 2:
Find:
```python
    def zero_inactive_gradients(self):
        \"\"\"
        Ensures inactive layers receive zero gradients.
        \"\"\"
        l1_active = (self.token_locus in [0, 1, 2, 3])
        l2_active = (self.token_locus == 4) and not self.l2_locked
```
Replace with:
```python
    def zero_inactive_gradients(self):
        \"\"\"
        Ensures inactive layers receive zero gradients.
        \"\"\"
        if self.token_locus == 4 and self.l2_locked:
            l1_active = True
            l2_active = False
        else:
            l1_active = (self.token_locus in [0, 1, 2, 3])
            l2_active = (self.token_locus == 4) and not self.l2_locked
```

Replacement 3:
Find:
```python
        # Return active layer loss
        active_loss = total_l1_loss if (self.token_locus in [0, 1, 2, 3]) else l2_loss
```
Replace with:
```python
        # Return active layer loss
        if self.token_locus == 4 and self.l2_locked:
            active_loss = total_l1_loss
        else:
            active_loss = total_l1_loss if (self.token_locus in [0, 1, 2, 3]) else l2_loss
```

3. Save the modified content back to 'src/thalamus.py'.
4. Run '.venv/Scripts/python.exe src/thalamus.py' to verify that all verification tests still pass successfully.

Write and execute this patcher script, and verify it completes with exit code 0.