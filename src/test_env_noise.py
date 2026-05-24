import numpy as np
from environment import PhysicsSandbox

def test_noisy_physics_sandbox():
    print("Initializing PhysicsSandbox with N=3, pixel_noise_std=0.15, and noisy_tv=True...")
    env = PhysicsSandbox(N=3, pixel_noise_std=0.15, noisy_tv=True, seed=42)
    
    obs = env.reset()
    assert obs.shape == (3, 128), f"Expected observation shape (3, 128), got {obs.shape}"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Rendered pixels must be clipped to [0.0, 1.0]"
    
    print("\nInitial reset completed successfully.")
    print(f"Noisy-TV initial position: {env.noisy_tv_pos:.3f}")
    print(f"Noisy-TV initial radius: {env.noisy_tv_radius:.3f}")
    print(f"Noisy-TV initial color: {env.noisy_tv_color}")
    
    # Track statistics over 50 steps
    tv_positions = []
    tv_colors = []
    obs_means = []
    obs_stds = []
    
    num_steps = 50
    for step_idx in range(1, num_steps + 1):
        action = {"acc": np.random.uniform(-1.0, 1.0), "push": (step_idx % 10 == 0)}
        obs, info = env.step(action)
        
        # Verify info dictionary has required Noisy-TV keys
        assert "noisy_tv_pos" in info, "Expected 'noisy_tv_pos' in info dict"
        assert "noisy_tv_color" in info, "Expected 'noisy_tv_color' in info dict"
        assert "noisy_tv_radius" in info, "Expected 'noisy_tv_radius' in info dict"
        
        tv_positions.append(info["noisy_tv_pos"])
        tv_colors.append(info["noisy_tv_color"])
        obs_means.append(np.mean(obs))
        obs_stds.append(np.std(obs))
        
        # Verify bounds of Noisy-TV position
        r = info["noisy_tv_radius"]
        pos = info["noisy_tv_pos"]
        assert r <= pos <= 128.0 - r, f"Noisy-TV position {pos} went out of bounds [{r}, {128.0 - r}]"
        
    tv_positions = np.array(tv_positions)
    tv_colors = np.array(tv_colors)
    
    # Calculate stats
    pos_diffs = np.diff(tv_positions)
    mean_step = np.mean(pos_diffs)
    std_step = np.std(pos_diffs)
    color_variance = np.var(tv_colors, axis=0)
    
    print("\n--- Statistics over 50 Steps ---")
    print(f"Noisy-TV position range: [{np.min(tv_positions):.3f}, {np.max(tv_positions):.3f}]")
    print(f"Noisy-TV random walk step mean: {mean_step:.4f}, step std: {std_step:.4f}")
    print(f"Noisy-TV color variance per channel (RGB): {color_variance}")
    print(f"Average observation mean intensity: {np.mean(obs_means):.4f}")
    print(f"Average observation standard deviation: {np.mean(obs_stds):.4f}")
    
    # Verify that the random walk indeed occurred
    assert std_step > 0.1, "Noisy-TV position did not move / simulate a random walk properly"
    # Verify color flickering variance is high
    assert np.all(color_variance > 0.01), f"Noisy-TV color didn't flicker with sufficient variance, got {color_variance}"
    
    print("\nAll assertions passed! The noisy physics sandbox is working perfectly.")

if __name__ == "__main__":
    test_noisy_physics_sandbox()
