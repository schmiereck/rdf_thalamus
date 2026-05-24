1. Rewrite and commit the pre-registration file `src/pre_registration.md` to reflect the Strategic Research Manager's instructions:
   - Formulate a mathematically rigorous, relativistic falsification criterion for attention tracking overlap under noise: $Overlap_{noise} \ge 0.8 \times Overlap_{clean}$.
   - Distinguish clearly between global high-frequency noise and localized structured entropic noise (the Noisy-TV entity: a 4th block that behaves as a random-walk generator and flickers with maximum entropy).
   - Establish generalization baselines: evaluate dynamic dimension recruitment against $B1$ (FixedJEPA, $d_t=3$) and $B1\_large$ (FixedJEPA, $d_t=4$) on the $N=3 \to N=4$ transition, showing that dynamic recruitment delivers comparable or superior adaptation and that the recruited 4th dimension correlates with the 4th object's state.

2. Modify `src/environment.py` to support:
   - Global high-frequency background noise: add a parameter `pixel_noise_std` (float) to `__init__`. If `pixel_noise_std > 0.0`, add pixel-level Gaussian noise to the rendered canvas and clip to [0.0, 1.0].
   - Localized structured entropic noise: add a parameter `noisy_tv` (bool) to `__init__`. If `noisy_tv=True`, simulate a Noisy-TV entity that behaves as a random-walk generator (moves by Gaussian steps and remains inbounds) and flickers with maximum entropy (color is randomly sampled from [0.3, 1.0] at each environment step). It should render as a soft continuous block in the same way as other physical entities.

3. Write a short script `src/test_env_noise.py` to verify that the noisy physics sandbox executes, renders properly with the Noisy-TV entity and pixel noise, and prints some stats. Run it and verify it works without error.