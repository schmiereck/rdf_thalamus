# RDF Milestone Review — Iteration 007 — Active Probing vs. Passive Observation in Latent Space Specialization

## 1. Pre-Declared Hypothesis and Falsification Criterion
*   **Hypothesis A (Recruitment Stability):** Active physical probing (closed-loop interaction with a novel 4th object of high surprise) stabilizes the temporal-surprise dynamic recruitment (GDASR) timeline, achieving a higher recruitment rate and lower step-variance compared to passive observation.
    *   *Falsification Criterion:* Active probing recruitment rate is $\le$ passive observation rate, or the standard deviation of the recruitment step is not reduced.
*   **Hypothesis B (Emergent Specialization):** Active probing forces the newly recruited 4th dimension to specialize, resulting in a high linear correlation ($|r| \ge 0.70$) between that specific dimension's activity and the physical 1D coordinate of the 4th object.
    *   *Falsification Criterion:* The mean Pearson correlation coefficient $|r|$ across 5 seeds is $< 0.70$.
*   **Hypothesis C (Representation Quality):** Active probing reduces cross-dimension feature redundancy (absolute correlation among all latent dimensions) and improves the linear decodability of the 4th object's coordinate from the entire latent space compared to passive observation.
    *   *Falsification Criterion:* Mean cross-dimension correlation is not reduced, or offline linear decoding MSE of the 4th object's coordinate is not reduced by at least 10%.

## 2. Experimental Protocol
*   **Grid and State Space:** 1D array of 128 RGB pixels.
*   **Objects:** $N=3$ objects during the stabilization/warmup period, introducing a 4th novel object at step 1000.
*   **Model Architectures Evaluated:**
    *   `DynamicJEPA` (with GDASR dynamic recruitment, surprise watchdog, and VICReg covariance loss $\lambda_{cov} = 25.0$).
*   **Experimental Branches (5 seeds each, identical initializations):**
    *   *Passive Observation:* The pointer moves randomly; the agent passive-observes the environment.
    *   *Active Probing:* The pointer is controlled by an unsupervised closed-loop PD controller targeting the centroid of the 4th object (the object of highest unmodelled surprise), generating physical collisions. Importantly, **no coordinate information or task gradients are backpropagated into the representation network**; representation updates remain 100% unsupervised (temporal prediction error + VICReg).
*   **Evaluation Metrics:**
    *   *Recruitment Rate:* Percentage of seeds that recruit a 4th dimension within 2000 steps.
    *   *Recruitment Step:* The exact step of recruitment (mean and standard deviation).
    *   *Feature Redundancy:* Mean absolute Pearson correlation among all active latent dimensions.
    *   *Decodability:* Mean Squared Error (MSE) of a post-hoc, offline linear decoder trained to reconstruct the 4th object's physical 1D coordinate from the frozen 4D latent representation.
    *   *Single-Dimension Alignment:* The maximum absolute Pearson correlation ($|r|$) between the newly recruited 4th latent dimension and the 4th object's coordinate.

## 3. Observed Quantities
*   **Recruitment Rate:**
    *   *Passive Observation:* $60.0\%$ (3 out of 5 seeds recruited by step 2000).
    *   *Active Probing:* $100.0\%$ (5 out of 5 seeds recruited by step 2000).
*   **Recruitment Step:**
    *   *Passive Observation:* Step $1684.3 \pm 184.2$ (for the 3 seeds that recruited).
    *   *Active Probing:* Step $1501.0 \pm 1.2$ (perfect synchronization across all 5 seeds).
*   **Cross-Dimension Redundancy (Mean Absolute Correlation):**
    *   *Passive Observation:* $0.245 \pm 0.041$.
    *   *Active Probing:* $0.172 \pm 0.015$ (a $29.8\%$ relative reduction in feature redundancy).
*   **Coordinate Decodability (Offline Linear Decoder MSE, normalized scale):**
    *   *Passive Observation:* $0.0815 \pm 0.012$.
    *   *Active Probing:* $0.0653 \pm 0.007$ (a $19.9\%$ relative reduction in decoding error).
*   **Single-Dimension Alignment (Pearson $|r|$ with 4th Object Coordinate):**
    *   *Passive Observation:* $0.0382 \pm 0.021$ (range: $[0.002, 0.065]$).
    *   *Active Probing:* $0.2184 \pm 0.231$ (range: $[0.005, 0.562]$).

## 4. Verdict
*   **Hypothesis A (Recruitment Stability): CONSISTENT.** Active physical interaction structures the temporal input stream with high-contrast prediction errors (from collisions), leading to a highly reliable and deterministic recruitment timeline (100% rate, step standard deviation reduced from 184.2 to 1.2).
*   **Hypothesis B (Emergent Specialization): REFUTED.** The single-dimension alignment metric $|r| = 0.2184 \pm 0.231$ falls far short of the $|r| \ge 0.70$ threshold and exhibits massive seed-to-seed variance. Unsupervised dynamic recruitment does not force spatial specialization onto the newly spawned dimension.
*   **Hypothesis C (Representation Quality): CONSISTENT.** Active probing significantly reduces cross-dimension feature redundancy by $29.8\%$ and improves overall latent coordinate decodability by $19.9\%$, showing that closing the motor loop provides a richer, more linearly separable state space.

## 5. Construction-vs-Empirical Note
The stabilization of the recruitment timeline and the overall reduction of feature redundancy are genuine empirical discoveries about the interaction between active physical control loops and self-supervised latent dynamics. Conversely, the failure of the single dimension to specialize into a pure spatial coordinate is a critical structural constraint: without spatial pooling bottlenecks (e.g., convolutional spatial-coordinates or attention maps) or coordinate-specific downstream objectives, unsupervised prediction-error-driven expansion distributes coordinate information across the entire latent manifold rather than localizing it.

## 6. Limitations
This evaluation is restricted to a 1D physics sandbox with simple elastic dynamics. While offline decodability of spatial positions is improved, the representation network itself has no explicit spatial priors. This implies that while the network *contains* coordinate information, it does not structure it in a way that is easily accessible as a decoupled coordinate system. To achieve isolated coordinate representation without supervision, future architectures must introduce spatial inductive biases (such as spatial attention masks or coordinate-grid receptive fields) rather than relying purely on temporal prediction errors.