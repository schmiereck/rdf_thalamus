Create 'patch_thalamus_sync.py' to implement the synchronized plasticity gating in 'src/thalamus.py'.

The patch must modify 'src/thalamus.py' as follows:

1. Replace the entire 'update_plasticity_gating' method with:
```python
    def update_plasticity_gating(self):
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
            param.requires_grad = l2_active
```

2. Replace the entire 'zero_inactive_gradients' method with:
```python
    def zero_inactive_gradients(self):
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
                    param.grad.zero_()
```

3. In the 'forward' method of 'ThalamusNet', insert these two lines at the very beginning of the function:
```python
        self.current_forward_locus = self.token_locus
        self.current_forward_locked = self.l2_locked
        self.update_plasticity_gating()
```

4. Verify that in 'forward' of 'ThalamusNet', the 'active_loss' calculation uses 'self.current_forward_locus' and 'self.current_forward_locked':
```python
        # Return active layer loss
        if self.current_forward_locus == 4 and self.current_forward_locked:
            active_loss = total_l1_loss
        else:
            active_loss = total_l1_loss if (self.current_forward_locus in [0, 1, 2, 3]) else l2_loss
```

Save these changes, and run the self-contained verification suite to verify that ThalamusNet compiles and passes all gradient gating unit tests.