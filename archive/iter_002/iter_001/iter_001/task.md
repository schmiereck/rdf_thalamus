Implement Thalamus Phase 1 components and an integration test.

Specifically, do the following:

1. Create `src/environment.py`:
   - Implement a class `PhysicsSandbox` simulating N elastic colliding objects in a continuous 1D space mapped to a 1D array of 128 RGB pixels.
   - The environment must support varying object counts (N=2 and N=3) and parameter randomization (colors, sizes, masses) to prevent dark-room collapse.
   - Objects should have position x, radius r (from [3.0, 8.0]), mass m = r (or randomized), color (random RGB [0.3, 1.0]), and velocity v (from [-2.0, 2.0], non-zero).
   - The physics simulation must use sub-stepping (e.g., M=10 sub-steps per environment step) for extreme stability.
   - Implement continuous elastic collision resolution between adjacent objects (when sorted by position) that conserves both momentum and kinetic energy, and resolves overlap to prevent objects sticking together.
   - Implement correct boundary bounces (reverse velocity and reset position to boundary).
   - Rendering: Output observation of shape (3, 128) using continuous soft blending (anti-aliasing) to ensure smooth transitions and gradients.
   - Provide `reset()` and `step()` methods. `step()` should return `(observation, info)` and update positions/velocities.

2. Create `src/models.py`:
   - Implement `Encoder` (1D CNN) mapping input (B, 3, 128) to latent space (B, D_max) with D_max = 8.
   - Implement `Predictor` (MLP) forecasting latent states z_{t+1} of size (B, D_max) from history of active latent states (B, H * D_max) with H = 3. Let's pad/zero-out inactive dimensions.
   - Implement `FixedJEPA` (with fixed d_t=2 or d_t=3).
   - Implement `DynamicJEPA` implementing GDASR: starts with d_t=2, tracks EMA of prediction error. Define an adaptive recruitment trigger based on stable 2-object error mean and standard deviation: recruit if error > mean + k * std, with k=4 and cooldown N_cooldown=500.
   - Implement VICReg loss:
     - Invariance loss: Mean Squared Error of prediction in active latent space.
     - Variance loss: push standard deviation of active dimensions to be >= 1.
     - Covariance loss: minimize cross-correlations between active dimensions.
     - Ensure variance/covariance calculations are done on the batch dimension.
   - Implement weight freezing/stop-gradient on existing dimensions when a new dimension is recruited for a short stabilization period (e.g., 200 steps). During stabilization, detach the first d_t - 1 dimensions of both encoder and predictor outputs.

3. Create `src/test_integration.py`:
   - Write a script that instantiates `PhysicsSandbox` with N=2 and N=3, verifying that objects move and collide, and observations are shape (3, 128) with values in [0, 1].
   - Instantiates `FixedJEPA` and `DynamicJEPA`.
   - Simulates a training/evaluation sequence: computes forward pass, computes VICReg loss, runs backward pass, and checks that parameters are updated and no NaNs or shape errors occur.
   - Tests `DynamicJEPA` recruitment trigger and stabilization logic (e.g., mock a high prediction error to trigger recruitment, verify d_t increases from 2 to 3, and check that stabilization stop-gradients are active).

Verify the integration tests run and pass perfectly. Do not start full experiments yet.
