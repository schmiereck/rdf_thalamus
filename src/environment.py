import numpy as np

class PhysicsSandbox:
    def __init__(self, N=2, substeps=10, sigma_blur=0.5, seed=None, pixel_noise_std=0.0, noisy_tv=False):
        """
        PhysicsSandbox simulates N elastic colliding objects plus 1 physical pointer in a 1D space [0, 128].
        Mapped to a 1D array of 128 RGB pixels.
        
        Args:
            N (int): Number of objects (usually 2 or 3).
            substeps (int): Number of integration sub-steps per environment step.
            sigma_blur (float): Softness of the continuous rendering.
            seed (int, optional): Random seed.
            pixel_noise_std (float): Standard deviation of global pixel-level Gaussian noise.
            noisy_tv (bool): Whether to simulate a Noisy-TV entity that random-walks and flickers.
        """
        self.N = N
        self.substeps = substeps
        self.sigma_blur = sigma_blur
        self.seed = seed
        self.pixel_noise_std = pixel_noise_std
        self.noisy_tv = noisy_tv
        
        # State variables for standard objects
        self.positions = np.zeros(N)
        self.velocities = np.zeros(N)
        self.radii = np.zeros(N)
        self.masses = np.zeros(N)
        self.colors = np.zeros((N, 3))
        
        # State variables for Noisy-TV entity
        self.noisy_tv_pos = 64.0
        self.noisy_tv_radius = 5.0
        self.noisy_tv_color = np.array([0.5, 0.5, 0.5])
        
        # Pointer state variables
        self.pointer_pos = 64.0
        self.pointer_vel = 0.0
        self.pointer_radius = 4.0
        self.pointer_mass = 10.0
        self.pointer_color = np.array([1.0, 1.0, 1.0])
        
        self.reset(seed=seed)
        
    def reset(self, seed=None):
        """
        Reset environment with randomized parameters (sizes, colors, masses, positions, velocities)
        and reset pointer position & velocity, then resolve all overlaps.
        """
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        W = 128.0 / self.N
        for i in range(self.N):
            # Sample radius and mass
            r = np.random.uniform(3.0, 8.0)
            self.radii[i] = r
            self.masses[i] = r  # Mass is proportional to radius/size
            
            # Sample color from [0.3, 1.0] for each channel
            self.colors[i] = np.random.uniform(0.3, 1.0, size=(3,))
            
            # Position inside the segmented interval to prevent initial overlaps
            low = i * W + r
            high = (i + 1) * W - r
            # Just in case the segment is too small (should not happen with N=2 or 3, but let's be safe)
            if low >= high:
                low = i * W + W / 2.0
                high = low
            self.positions[i] = np.random.uniform(low, high)
            
            # Velocity from [-2.0, 2.0], non-zero
            v = np.random.uniform(0.5, 2.0)
            if np.random.rand() < 0.5:
                v = -v
            self.velocities[i] = v
            
        # Reset pointer position and velocity
        self.pointer_pos = 64.0
        self.pointer_vel = 0.0
        self.pointer_radius = 4.0
        self.pointer_mass = 10.0
        self.pointer_color = np.array([1.0, 1.0, 1.0])
        
        # Reset Noisy-TV entity if active
        if self.noisy_tv:
            self.noisy_tv_radius = np.random.uniform(3.0, 8.0)
            self.noisy_tv_pos = np.random.uniform(self.noisy_tv_radius, 128.0 - self.noisy_tv_radius)
            self.noisy_tv_color = np.random.uniform(0.3, 1.0, size=(3,))
        
        # Resolve initial overlaps of both the objects and the pointer in a loop
        temp_positions = np.concatenate([self.positions, [self.pointer_pos]])
        temp_radii = np.concatenate([self.radii, [self.pointer_radius]])
        temp_masses = np.concatenate([self.masses, [self.pointer_mass]])
        
        for _ in range(50):
            # Boundary correction
            for i in range(len(temp_positions)):
                if temp_positions[i] - temp_radii[i] < 0.0:
                    temp_positions[i] = temp_radii[i]
                elif temp_positions[i] + temp_radii[i] > 128.0:
                    temp_positions[i] = 128.0 - temp_radii[i]
            
            # Overlap resolution
            sort_idx = np.argsort(temp_positions)
            for idx_in_sort in range(len(temp_positions) - 1):
                i = sort_idx[idx_in_sort]
                j = sort_idx[idx_in_sort + 1]
                
                dist = temp_positions[j] - temp_positions[i]
                min_dist = temp_radii[i] + temp_radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    m_inv_i = 1.0 / temp_masses[i]
                    m_inv_j = 1.0 / temp_masses[j]
                    sum_inv_m = m_inv_i + m_inv_j
                    
                    temp_positions[i] -= overlap * (m_inv_i / sum_inv_m)
                    temp_positions[j] += overlap * (m_inv_j / sum_inv_m)

        # Assign positions back
        self.positions = temp_positions[:-1]
        self.pointer_pos = temp_positions[-1]
            
        return self.render()

    def step(self, action=None):
        """
        Step the environment by 1.0 time unit using self.substeps, incorporating optional actions.
        
        Args:
            action (dict, optional): containing continuous acceleration 'acc' (float) and boolean 'push'.
        Returns:
            observation (np.ndarray): Shape (3, 128)
            info (dict): Useful diagnostic info
        """
        if action is not None:
            acc = action.get('acc', 0.0)
            push = action.get('push', False)
        else:
            acc = 0.0
            push = False
            
        if push:
            if len(self.positions) > 0:
                dists = self.positions - self.pointer_pos
                nearest_idx = np.argmin(np.abs(dists))
                diff = dists[nearest_idx]
                direction = 1.0 if diff >= 0 else -1.0
                self.pointer_vel = direction * 5.0
                
        dt = 1.0 / self.substeps
        
        for _ in range(self.substeps):
            # Apply acceleration
            self.pointer_vel += acc * dt
            
            # Pack all N+1 objects
            temp_positions = np.concatenate([self.positions, [self.pointer_pos]])
            temp_velocities = np.concatenate([self.velocities, [self.pointer_vel]])
            temp_radii = np.concatenate([self.radii, [self.pointer_radius]])
            temp_masses = np.concatenate([self.masses, [self.pointer_mass]])
            
            # 1. Update positions
            temp_positions += temp_velocities * dt
            
            # 2. Boundary bounces & position correction
            for i in range(len(temp_positions)):
                if temp_positions[i] - temp_radii[i] < 0.0:
                    temp_positions[i] = temp_radii[i]
                    if temp_velocities[i] < 0.0:
                        temp_velocities[i] = -temp_velocities[i]
                elif temp_positions[i] + temp_radii[i] > 128.0:
                    temp_positions[i] = 128.0 - temp_radii[i]
                    if temp_velocities[i] > 0.0:
                        temp_velocities[i] = -temp_velocities[i]
            
            # 3. Resolve elastic collisions between adjacent objects
            sort_idx = np.argsort(temp_positions)
            for idx_in_sort in range(len(temp_positions) - 1):
                i = sort_idx[idx_in_sort]
                j = sort_idx[idx_in_sort + 1]
                
                dist = temp_positions[j] - temp_positions[i]
                min_dist = temp_radii[i] + temp_radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    m_inv_i = 1.0 / temp_masses[i]
                    m_inv_j = 1.0 / temp_masses[j]
                    sum_inv_m = m_inv_i + m_inv_j
                    
                    # Resolve overlap proportionally to inverse masses
                    temp_positions[i] -= overlap * (m_inv_i / sum_inv_m)
                    temp_positions[j] += overlap * (m_inv_j / sum_inv_m)
                    
                    # Conserve momentum and kinetic energy if moving towards each other
                    if temp_velocities[i] > temp_velocities[j]:
                        v1 = temp_velocities[i]
                        v2 = temp_velocities[j]
                        m1 = temp_masses[i]
                        m2 = temp_masses[j]
                        
                        temp_velocities[i] = (v1 * (m1 - m2) + 2.0 * m2 * v2) / (m1 + m2)
                        temp_velocities[j] = (v2 * (m2 - m1) + 2.0 * m1 * v1) / (m1 + m2)
                        
            # Additional boundary check after collision resolution to ensure no object is pushed out
            for i in range(len(temp_positions)):
                if temp_positions[i] - temp_radii[i] < 0.0:
                    temp_positions[i] = temp_radii[i]
                    if temp_velocities[i] < 0.0:
                        temp_velocities[i] = -temp_velocities[i]
                elif temp_positions[i] + temp_radii[i] > 128.0:
                    temp_positions[i] = 128.0 - temp_radii[i]
                    if temp_velocities[i] > 0.0:
                        temp_velocities[i] = -temp_velocities[i]
                        
            # Unpack back to state variables
            self.positions = temp_positions[:-1]
            self.velocities = temp_velocities[:-1]
            self.pointer_pos = temp_positions[-1]
            self.pointer_vel = temp_velocities[-1]

        # Update Noisy-TV state if active
        if self.noisy_tv:
            # Random walk motion: Gaussian step with std 2.0
            step_noise = np.random.normal(0.0, 2.0)
            self.noisy_tv_pos += step_noise
            # Keep inside environment boundaries [radius, 128 - radius]
            self.noisy_tv_pos = np.clip(self.noisy_tv_pos, self.noisy_tv_radius, 128.0 - self.noisy_tv_radius)
            # Flicker with maximum entropy: color randomly sampled from [0.3, 1.0]
            self.noisy_tv_color = np.random.uniform(0.3, 1.0, size=(3,))

        obs = self.render()
        
        info = {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "radii": self.radii.copy(),
            "masses": self.masses.copy(),
            "colors": self.colors.copy(),
            "pointer_pos": self.pointer_pos,
            "pointer_vel": self.pointer_vel,
            "pointer_radius": self.pointer_radius,
            "pointer_mass": self.pointer_mass,
            "pointer_color": self.pointer_color.copy(),
        }
        
        if self.noisy_tv:
            info["noisy_tv_pos"] = self.noisy_tv_pos
            info["noisy_tv_color"] = self.noisy_tv_color.copy()
            info["noisy_tv_radius"] = self.noisy_tv_radius
            
        return obs, info

    def render(self):
        """
        Render continuous 1D space to a (3, 128) array with soft continuous blending.
        Includes original objects, the white physical pointer, and optionally the Noisy-TV entity.
        """
        pixel_centers = np.arange(128) + 0.5  # shape (128,)
        canvas = np.zeros((3, 128))
        
        # Combine objects, pointer, and optional Noisy-TV for rendering depth sorting
        if self.noisy_tv:
            temp_positions = np.concatenate([self.positions, [self.pointer_pos], [self.noisy_tv_pos]])
            temp_radii = np.concatenate([self.radii, [self.pointer_radius], [self.noisy_tv_radius]])
            temp_colors = np.concatenate([self.colors, [self.pointer_color], [self.noisy_tv_color]], axis=0)
        else:
            temp_positions = np.concatenate([self.positions, [self.pointer_pos]])
            temp_radii = np.concatenate([self.radii, [self.pointer_radius]])
            temp_colors = np.concatenate([self.colors, [self.pointer_color]], axis=0)
        
        # Sort objects and pointer by position to blend left-to-right (depth order)
        sorted_indices = np.argsort(temp_positions)
        for idx in sorted_indices:
            pos = temp_positions[idx]
            r = temp_radii[idx]
            color = temp_colors[idx]  # shape (3,)
            
            # Distance from pixel centers to object center
            d = np.abs(pixel_centers - pos)
            
            # Sigmoid continuous blending
            mask = 1.0 / (1.0 + np.exp((d - r) / self.sigma_blur))  # shape (128,)
            mask = mask[np.newaxis, :]  # shape (1, 128)
            color_expanded = color[:, np.newaxis]  # shape (3, 1)
            
            # Alpha blend onto canvas
            canvas = canvas * (1.0 - mask) + color_expanded * mask
            
        # Apply global pixel noise if specified
        if self.pixel_noise_std > 0.0:
            noise = np.random.normal(0.0, self.pixel_noise_std, size=canvas.shape)
            canvas = np.clip(canvas + noise, 0.0, 1.0)
            
        return canvas
