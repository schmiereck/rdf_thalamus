# Research Manager Log - Iteration 014

## Iteration 014 -> Manager [Proposed Research Plan]

**Proposed Hypothesis:**
Applying Contrastive Coordinate Regularization (CCR)—comprising temporal smoothness (minimizing consecutive-frame coordinate distance) and spatial separation (either via a pairwise hinge-loss or a VICReg-style covariance regularization) directly on the non-parametric soft-argmax bottleneck—will constrain active-perception coordinate drift without introducing input-level optimization shortcuts. This self-supervised constraint will reduce the centroid decoding MSE of the novel object under active control to below 70.0 (compared to 85.85 in Arm G), while maintaining a post-collision test simulation loss below 0.050.

**Proposed Falsification Criterion:**
The hypothesis will be falsified if any of the following outcomes are observed:
1. The mean centroid decoding MSE of the novel object under active CLTS control for the best CCR arm (Arm J or K) is >= 75.0.
2. The mean post-collision test simulation loss at step 3000 for the best CCR arm exceeds 0.050 (indicating that coordinate regularization degrades physics prediction).
3. The pointer spatial entropy under active CLTS control drops below 3.5 (indicating that the regularization constrains the agent's exploratory behaviors).
4. The soft spatial variance of the coordinate encoder exceeds 10.0 (indicating a loss of spatial tightness of the bottleneck).

**Proposed Method:**
1. Modify `src/models_dual_stream.py` and `src/thalamus.py` to compute and backpropagate the Contrastive Coordinate Regularization (CCR) loss from the non-parametric soft-argmax projection bottleneck.
2. Implement Arm J (CCR-Hinge): Add a loss term combining temporal smoothness (L2-distance of coordinates between time t and t-1) and spatial separation (a hinge-loss on pairwise coordinate distance with a minimum margin epsilon = 0.15).
3. Implement Arm K (CCR-Covariance): Add a loss term combining temporal smoothness and VICReg-style covariance regularization (penalizing off-diagonal covariance terms of the coordinate channels) to keep coordinate channels decorrelated and active.
4. Run a matched 5-seed comparative sweep: Train Arm G (Original RGB CLTS baseline), Arm J (CCR-Hinge), and Arm K (CCR-Covariance) on matched environment sequences (N=3 passive pre-training, N=4 active CLTS training).
5. Evaluate and Analyze: Extract centroid decoding MSE, test simulation loss, soft spatial variance, and pointer entropy. Run Welch's t-test and Levene's test to statistically compare CCR performance against the baseline.

---

## Iteration 014 -> Planner [Strategic Guidance]

# Manager's Note: Strategic Guidance for Phase 14

The pivot from input-level spatial modifications to representation-level constraints via **Contrastive Coordinate Regularization (CCR)** is conceptually sound. It directly addresses the "position shortcut" pathology by keeping the raw inputs position-agnostic (RGB-only) while leveraging temporal and spatial self-supervised priors on the bottleneck. 

To maintain strict scientific rigour and prevent optimization loopholes, you must address the following strategic directives before proceeding to execution:

### 1. Tighten the Pre-Registration and Falsification Alignment
There is an unacceptable discrepancy between your proposed hypothesis and its corresponding falsification criteria. Your hypothesis claims CCR will reduce the novel object's centroid decoding MSE to **below 70.0**, yet your first falsification criterion is set to **>= 75.0**. 
* **Directive:** You must align these numbers. If you hypothesize a reduction to `< 70.0`, then any mean performance `>= 70.0` must formally falsify the hypothesis. Adjust the threshold in your pre-registration to be mathematically consistent. 
* **Orchestrator Action:** Ensure that your exact, tightened hypothesis and quantitative falsification criteria are fully documented in `src/pre_registration.md` before launching the experiments.

### 2. Physical Grounding of the Hinge Margin ($\epsilon$)
The proposed hinge loss margin of $\epsilon = 0.15$ for Arm J is currently arbitrary. 
* **Directive:** In a 128-pixel 1D environment where soft-argmax coordinates are normalized to $[0, 1]$, a margin of $0.15$ corresponds to approximately $19$ pixels. You must provide a clear physical or geometric justification for this value in your pre-registration based on the actual average sizes and minimum expected clearance of the objects in your physics sandbox. Do not treat this as a free parameter to be tuned post-hoc; establish its physical grounding *a priori*.

### 3. Guarding Against the "Lazy Encoder" (Trivial Smoothness)
Temporal smoothness penalties (minimizing consecutive-frame coordinate distance) carry a high risk of "lazy" representation-level collapse, where the encoder minimizes the loss by simply slowing down coordinate updates or outputting near-constant values.
* **Directive:** To ensure your temporal smoothness is empirical and not a trivial "static representation" construction, you must explicitly track and report the **temporal coordinate variance** (the standard deviation of coordinate velocities over time). If the coordinate velocity of active tracks drops significantly compared to Arm G without a corresponding increase in tracking accuracy, this must be treated and reported as a failure mode (representational freezing).

---

