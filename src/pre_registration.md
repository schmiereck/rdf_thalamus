# RDF Scientific Pre-Registration (Revision 006)

*   **Iteration:** 006
*   **Pre-Registration File:** src/pre_registration.md

## 1. Hypothesis
In the Phase 4 Generalization and Noise Robustness evaluation, we investigate how the ThalamusNet architecture performs under localized structured and global high-frequency noise, and how its dynamic dimension recruitment mechanism adapts during an $N=3 \to N=4$ object transition.

We hypothesize:
1. **Dynamic Recruitment & Generalization:** When transitioning from $N=3$ objects (on which the network is pre-trained) to $N=4$ objects, ThalamusNet will dynamically recruit a 4th latent dimension $d_t=4$ within 500 timesteps of the 4th object's introduction. This recruited 4th dimension will dynamically adapt to correlate strongly with the 4th object's physical state (correlation $|r| \ge 0.7$).
2. **Superior/Comparable Adaptation:** Compared to the fixed-capacity baselines $B1$ (FixedJEPA, $d_t=3$) and $B1\_large$ (FixedJEPA, $d_t=4$), ThalamusNet with dynamic recruitment will deliver comparable or superior adaptation (measured by post-transition prediction loss on the 4th object, showing at least 30% reduction over $B1$).
3. **Attentional Resilience to Noise:** The Thalamus watch-dog mechanism is resilient to both global high-frequency noise and localized structured entropic noise (the Noisy-TV distractor), successfully maintaining tracking overlap without getting trapped.

## 2. Mathematically Rigorous, Relativistic Falsification Criterion

To evaluate the resilience of the attention tracking mechanism under noise, we define a mathematically rigorous, relativistic tracking overlap metric. Let $A_t \in \mathbb{R}^{128}$ be the continuous attention weight distribution across the 1D space at timestep $t$, and let $M_{t, \text{physical}} \in \{0, 1\}^{128}$ be the ground-truth binary mask representing the spatial extent (intervals occupied by the continuous physical objects, excluding any noise distractors or background).

The continuous attention tracking overlap is defined as:
$$Overlap = \frac{\sum_{i=1}^{128} A_t(i) \cdot M_{t, \text{physical}}(i)}{\sum_{i=1}^{128} A_t(i)}$$

Let $Overlap_{\text{clean}}$ denote the mean tracking overlap computed over a test episode of $T$ steps in a clean environment. Let $Overlap_{\text{noise}}$ denote the mean tracking overlap computed under the same physical trajectory but subjected to noise (either global pixel noise or the Noisy-TV distractor).

The attention tracking hypothesis is **proven false** if:
1. Under any noise condition, the relative attention tracking efficiency drops below 80% of its clean baseline, formulated as the relativistic falsification criterion:
   $$Overlap_{\text{noise}} < 0.8 \times Overlap_{\text{clean}}$$
2. In the transition $N=3 \to N=4$, ThalamusNet fails to recruit a 4th latent dimension within 500 timesteps, or the recruited dimension's activity does not correlate with the 4th object's state (Pearson $|r| < 0.7$).
3. The post-transition prediction loss of the dynamic recruitment model is significantly worse than that of $B1$ (FixedJEPA, $d_t=3$) or fails to match/surpass $B1\_large$ (FixedJEPA, $d_t=4$).

## 3. Noise Taxonomy: Global vs. Localized Structured Entropic Noise

To establish a clear boundary between non-structural background fluctuations and structured non-physical distractors, we distinguish:

### A. Global High-Frequency Background Noise
*   **Mechanism:** Additive independent and identically distributed (i.i.d.) Gaussian pixel-level noise:
    $$\text{Canvas}_{\text{noisy}} = \text{clip}(\text{Canvas}_{\text{clean}} + \eta, 0.0, 1.0), \quad \eta \sim \mathcal{N}(0, \sigma^2)$$
    where $\sigma$ (the standard deviation `pixel_noise_std`) controls the noise intensity.
*   **Characteristics:** High-frequency, spatially and temporally uncorrelated, zero-mean. It lacks spatial coherence and object-like persistence.

### B. Localized Structured Entropic Noise (The Noisy-TV Entity)
*   **Mechanism:** A 4th continuous block (rendered in the same soft continuous manner as physical entities) which behaves as an independent random-walk generator and flickers with maximum color entropy.
    *   **Motion:** Update step governed by a random walk inside boundaries:
        $$x_t^{\text{TV}} = \text{clip}(x_{t-1}^{\text{TV}} + \epsilon_t, r^{\text{TV}}, 128 - r^{\text{TV}}), \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{\text{step}}^2)$$
    *   **Appearance (Flicker):** Maximum entropy color sampling at each environment step:
        $$C_t^{\text{TV}} \sim \mathcal{U}(0.3, 1.0)^3$$
*   **Characteristics:** Spatially continuous and localized (has a cohesive physical shape and rendering), but completely unpredictable in its dynamics (random walk) and appearance (color flicker). It acts as a powerful attention-trap (the "Noisy-TV" problem) for standard curiosity or prediction-error-based attention mechanisms.

## 4. Generalization Baselines & Dimension Recruitment

During the $N=3 \to N=4$ transition, we compare our model against two fixed-dimension baselines:
1.  **Baseline 1 ($B1$ - FixedJEPA, $d_t=3$):** Represents a model with rigid, under-parameterized latent space. It is forced to compress 4 physical objects into 3 latent channels, causing high prediction error and representation overlap.
2.  **Baseline 1 Large ($B1\_large$ - FixedJEPA, $d_t=4$):** Represents a model pre-trained with over-parameterized latent space ($d_t=4$). It has the capacity for the 4th object but did not dynamically recruit it, potentially leading to unspecialized or noisy 4th-channel representations during the $N=3$ phase.

We evaluate whether:
-   **Dynamic Recruitment adaptation:** ThalamusNet dynamically expands its active latent space $d_t = 3 \to 4$ upon detecting the 4th object, delivering comparable or superior adaptation speed and prediction loss relative to $B1$ and $B1\_large$.
-   **Latent-to-State Alignment:** The newly recruited 4th latent channel aligns with the physical trajectory of the 4th object, demonstrating target-specific representation learning rather than capturing random-walk or global pixel noise.
