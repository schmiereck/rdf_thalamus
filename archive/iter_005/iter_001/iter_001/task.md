Implement Phase 3 closed loop components by following these exact instructions:

1. Update `src/environment.py`:
   - Incorporate a 1D physical pointer in the `PhysicsSandbox` class.
   - Initialize pointer state variables: `self.pointer_pos = 64.0`, `self.pointer_vel = 0.0`, `self.pointer_radius = 4.0`, `self.pointer_mass = 10.0`, `self.pointer_color = np.array([1.0, 1.0, 1.0])` (white pointer).
   - In `reset()`, reset the pointer position and velocity, and resolve initial overlaps of both the objects and the pointer in a loop (similar to standard overlap resolution).
   - In `step()`, accept an optional `action` dictionary containing continuous acceleration 'acc' (float) and boolean 'push'.
   - Apply acceleration `acc` during integration substeps (`self.pointer_vel += acc * dt`).
   - If `push` is True, apply an instantaneous velocity boost (direction of nearest object, with magnitude 5.0).
   - Treat the pointer as a physical object during the substep collision resolution phase: append its position, velocity, radius, and mass to arrays of length N+1, perform sorting, overlap resolution, and elastic momentum exchanges, and copy the values back.
   - Render the pointer onto the 128 RGB pixel canvas using the same soft continuous alpha blending formula as other objects.
   - Include the pointer state in the returned `info` dictionary: "pointer_pos", "pointer_vel", "pointer_radius", "pointer_mass", "pointer_color".

2. Update `src/thalamus.py`:
   - In `ThalamusNet` and `NonGatedControlNet`, add a trainable linear position readout layer: `self.pos_readout = nn.Linear(d_max, 1)` mapping L1 latents to target position.
   - Add a mathematically clean `compute_attended_centroid(self, x_target, locus=None)` function to both classes that extracts the spatial centroid purely from visual activations of the attended segment (no ground truth).
     - It should mean/sum across color channels to compute pixel intensities of the 32-pixel segment, and compute the spatial centroid using a weighted average of pixel coordinates (coords = 0.5 to 31.5) weighted by pixel intensity. If intensity is near zero, default to the center of the segment (16.0). Add `locus * 32` to find the global centroid. If `locus` is 4 (L2), fall back to finding the segment with the highest mean visual intensity and use its centroid.
   - Integrate `pos_readout` training into the forward pass of both networks. Compute MSE loss between predicted and target position for all 4 segments at each forward step. Scale the loss down by dividing coordinates by 128.0 (normalized MSE) to keep it in a small, stable range (e.g. 0.01 - 0.1) so it doesn't overpower VICReg losses.
   - Update plasticity gating and zeroing of inactive gradients in `ThalamusNet` to correctly freeze/unfreeze `pos_readout` parameters depending on the active layer (trainable when L1 is active).
   - Implement the surprise-modulated adaptive cooldown $C_t$:
     - Store `self.prev_E` and `self.current_cooldown_val`.
     - In the token routing section, calculate the temporal change in local surprise: $E_t$ is `raw_surprises[self.token_locus]`, $\Delta E_t = E_t - E_{t-1}$.
     - Update the cooldown: $C_t = \text{clip}( \alpha / (|\Delta E_t| + \epsilon), 10, 30 )$ where $\alpha$ is `self.cooldown_alpha` (default to 1.0) and $\epsilon = 1e-5$. Save the computed $C_t$ to `self.current_cooldown_val`.
     - Use $C_t$ instead of the hardcoded `self.cooldown` value to gate token change.

3. Create `src/motor.py`:
   - Implement the 3-layer Subsumption Motor Controller:
     - **Lower Layer (Reflexive PD Tracking)**: Measures error between computed visual centroid of attended segment and pointer position. Calculates continuous acceleration: $a_{reflexive} = K_p * e_t + K_d * (e_t - e_{t-1}) / dt$.
     - **Middle Layer (Predictive Kinematics)**: Feeds predicted latent segment representations into `pos_readout` to estimate future target position, computes predicted velocity of target, and calculates feedforward anticipation: $a_{predictive} = K_v * (v_{predicted\_target} - v_{pointer})$.
     - **Upper Layer (Deliberate Push Perturbation)**: Checks "boredom" (e.g., error < error_thresh and delta_E is stable). If bored and cooldown is finished, triggers a push command (`push = True`) and overrides lower layers to push strongly towards the target object.
     - **Ablation Controls**: Support `ablation="random"` (returns random action) and `ablation="shuffle"` (shuffles attention token locus to a random segment instead of the true active segment for visual feedback).

4. Create `src/test_closed_loop.py`:
   - An integration test that runs the environment, ThalamusNet, and SubsumptionMotorController in a closed loop for 30 simulation steps.
   - Verify that all components compile, forward passes succeed, the pointer moves, and values update without errors.
   - Run the integration test and report the output.
