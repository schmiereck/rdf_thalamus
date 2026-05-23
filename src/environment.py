import numpy as np

class PhysicsSandbox:
    def __init__(self, N=2, substeps=10, sigma_blur=0.5):
        """
        PhysicsSandbox simulates N elastic colliding objects in a 1D space [0, 128].
        Mapped to a 1D array of 128 RGB pixels.
        
        Args:
            N (int): Number of objects (usually 2 or 3).
            substeps (int): Number of integration sub-steps per environment step.
            sigma_blur (float): Softness of the continuous rendering.
        """
        self.N = N
        self.substeps = substeps
        self.sigma_blur = sigma_blur
        
        # State variables
        self.positions = np.zeros(N)
        self.velocities = np.zeros(N)
        self.radii = np.zeros(N)
        self.masses = np.zeros(N)
        self.colors = np.zeros((N, 3))
        
        self.reset()
        
    def reset(self):
        """
        Reset environment with randomized parameters (sizes, colors, masses, positions, velocities).
        """
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
            
        return self.render()

    def step(self):
        """
        Step the environment by 1.0 time unit using self.substeps.
        Returns:
            observation (np.ndarray): Shape (3, 128)
            info (dict): Useful diagnostic info
        """
        dt = 1.0 / self.substeps
        
        for _ in range(self.substeps):
            # 1. Update positions
            self.positions += self.velocities * dt
            
            # 2. Boundary bounces & position correction
            for i in range(self.N):
                if self.positions[i] - self.radii[i] < 0.0:
                    self.positions[i] = self.radii[i]
                    if self.velocities[i] < 0.0:
                        self.velocities[i] = -self.velocities[i]
                elif self.positions[i] + self.radii[i] > 128.0:
                    self.positions[i] = 128.0 - self.radii[i]
                    if self.velocities[i] > 0.0:
                        self.velocities[i] = -self.velocities[i]
            
            # 3. Resolve elastic collisions between adjacent objects
            sort_idx = np.argsort(self.positions)
            for idx_in_sort in range(self.N - 1):
                i = sort_idx[idx_in_sort]
                j = sort_idx[idx_in_sort + 1]
                
                dist = self.positions[j] - self.positions[i]
                min_dist = self.radii[i] + self.radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    m_inv_i = 1.0 / self.masses[i]
                    m_inv_j = 1.0 / self.masses[j]
                    sum_inv_m = m_inv_i + m_inv_j
                    
                    # Resolve overlap proportionally to inverse masses
                    self.positions[i] -= overlap * (m_inv_i / sum_inv_m)
                    self.positions[j] += overlap * (m_inv_j / sum_inv_m)
                    
                    # Conserve momentum and kinetic energy if moving towards each other
                    if self.velocities[i] > self.velocities[j]:
                        v1 = self.velocities[i]
                        v2 = self.velocities[j]
                        m1 = self.masses[i]
                        m2 = self.masses[j]
                        
                        self.velocities[i] = (v1 * (m1 - m2) + 2.0 * m2 * v2) / (m1 + m2)
                        self.velocities[j] = (v2 * (m2 - m1) + 2.0 * m1 * v1) / (m1 + m2)
                        
            # Additional boundary check after collision resolution to ensure no object is pushed out
            for i in range(self.N):
                if self.positions[i] - self.radii[i] < 0.0:
                    self.positions[i] = self.radii[i]
                    if self.velocities[i] < 0.0:
                        self.velocities[i] = -self.velocities[i]
                elif self.positions[i] + self.radii[i] > 128.0:
                    self.positions[i] = 128.0 - self.radii[i]
                    if self.velocities[i] > 0.0:
                        self.velocities[i] = -self.velocities[i]

        obs = self.render()
        info = {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "radii": self.radii.copy(),
            "masses": self.masses.copy(),
            "colors": self.colors.copy(),
        }
        return obs, info

    def render(self):
        """
        Render continuous 1D space to a (3, 128) array with soft continuous blending.
        """
        pixel_centers = np.arange(128) + 0.5  # shape (128,)
        canvas = np.zeros((3, 128))
        
        # Sort objects by position to blend left-to-right (depth order)
        sorted_indices = np.argsort(self.positions)
        for idx in sorted_indices:
            pos = self.positions[idx]
            r = self.radii[idx]
            color = self.colors[idx]  # shape (3,)
            
            # Distance from pixel centers to object center
            d = np.abs(pixel_centers - pos)
            
            # Sigmoid continuous blending
            # When d == r, mask is 0.5. Inside, mask -> 1.0. Outside, mask -> 0.0.
            # We use self.sigma_blur to control the steepness.
            mask = 1.0 / (1.0 + np.exp((d - r) / self.sigma_blur))  # shape (128,)
            mask = mask[np.newaxis, :]  # shape (1, 128)
            color_expanded = color[:, np.newaxis]  # shape (3, 1)
            
            # Alpha blend onto canvas
            canvas = canvas * (1.0 - mask) + color_expanded * mask
            
        return canvas
