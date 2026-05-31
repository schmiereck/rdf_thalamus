## Task: Implement and Run the REVISED iter_034 Benchmark (v2)

The original mass-estimation MAPE benchmark was falsified because pointer-object collisions corrupt the least-squares mass estimation (ORACLE_MAPE=1.005 > PASSIVE_MAPE=0.597 — inverted). The root cause: `m_i = 10 * -Δv_pointer / Δv_obj` is extremely noise-sensitive when Δv_obj is small, and hundreds of such estimates overwhelm the least-squares system.

Read `src/run_iter034_benchmark.py` first to understand the existing implementation, then create a REVISED version `src/run_iter034_benchmark_v2.py`.

### Key Changes from v1 → v2

**1. PRIMARY METRIC: Mean Absolute Log-Ratio Error (MALRE)**

Instead of MAPE from absolute mass estimation, measure how well mass RATIOS between object pairs can be estimated from object-object collisions only.

For each object pair (i,j) where i < j (3 pairs for N=3):
- Collect all object-object collisions between i and j
- For each collision, compute ratio estimate: `r = -Δv_j / Δv_i` (this equals m_i/m_j for elastic collisions)
- Filter: only use collisions where `|Δv_i| > 0.5 AND |Δv_j| > 0.5` (avoid near-zero denominators)
- Take the MEDIAN of r across all valid collisions for that pair → `r_hat_ij`
- Compute error: `e_ij = |log(r_hat_ij) - log(m_true_i / m_true_j)|`
- For pairs with 0 valid collisions: `e_ij = 2.0` (maximum penalty, since true ratios are typically 0.5-2.5, log-ratio range ≈ ±1.0)

MALRE = mean(e_ij across all 3 pairs) — LOWER IS BETTER

Why this works:
- Object-object collisions preserve the mass ratio even across multiple substep impulses (the sum of impulses preserves the ratio when each impulse satisfies elastic collision formulas)
- MEDIAN is robust to outlier micro-collisions
- Log-ratio error is symmetric and doesn't require absolute mass anchoring
- More diverse collisions (ORACLE) → more observations per pair → better MEDIAN estimates → lower MALRE

**2. SECONDARY METRIC: Absolute Mass Estimation (robust)**

After estimating ratios, anchor to absolute mass using the BEST pointer-object collision per object:
- For each object, find the pointer-object collision with the LARGEST `|Δv_obj|` (most reliable estimate)
- Compute `m_hat_i = 10 * (-Δv_pointer) / Δv_obj` from that single best collision
- Scale all mass estimates to be consistent: adjust the common scale factor to minimize MSE against the anchor estimates
- If an object has NO pointer-object collisions, estimate its mass from the ratio with an anchored object
- MAPE = mean(|m_hat_i - m_true_i| / m_true_i)

**3. COVERAGE METRIC (tertiary)**

For N=3 objects, there are 3 object-pair types and 3 pointer-object pair types = 6 total.
Coverage = (number of pair types with ≥3 collisions) / 6

**4. GATES (revised)**

- G1 (non-degeneracy): PASSIVE_MALRE - ORACLE_MALRE ≥ 0.3, lower 95% bootstrap CI ≥ 0.1
- G2 (coverage gradient): PASSIVE_MALRE - RANDOM_MALRE ≥ 0.1, lower 95% bootstrap CI > 0
- G3 (ordering): ORACLE_MALRE < RANDOM_MALRE < PASSIVE_MALRE (means)
- G4 (coverage sanity): ORACLE_coverage - PASSIVE_coverage ≥ 0.2

**5. SANITY CHECKS (revised)**

- S1: ORACLE achieves ≥3 pointer-object collisions per object (mean across seeds)
- S2: ORACLE has ≥10 object-object collisions per pair (mean across pairs and seeds)
- S3: ≥90% of logged collision events show |Δv| > 0.5 px/step
- S4: ORACLE achieves ≥1 pointer-object collision per object per seed
- S5: ORACLE pointer stays in bounds [0, 128] for ≥95% of steps

**6. VELOCITY NOISE: REMOVED**

Do NOT inject velocity observation noise. The log-ratio estimation with MEDIAN is inherently robust to the noise that comes from substep physics. Adding artificial noise would just add unnecessary variance. This also addresses the Manager's concern: in iter_035, the LEARNED agent observes the pixel observation (not ground-truth velocities), so there IS observation noise already.

### Implementation Details

ENVIRONMENT: `PhysicsSandbox(N=3)`, 2000 steps, 8 seeds [7, 31, 53, 71, 83, 97, 113, 163]. Same as v1.

THREE CONDITIONS: Same as v1 (ORACLE-TARGETED, RANDOM, PASSIVE).

COLLISION DETECTION: Same as v1. Use ground-truth velocities (no noise injection).

**MALRE Computation:**
```python
def compute_malre(collisions, true_masses, n_objects=3):
    pair_errors = []
    for i in range(n_objects):
        for j in range(i+1, n_objects):
            # Collect ratio estimates from obj-obj collisions
            pair_collisions = [c for c in collisions 
                              if (c[1] == i and c[2] == j) or (c[1] == j and c[2] == i)]
            
            ratios = []
            for c in pair_collisions:
                step, ci, cj, vi_pre, vj_pre, vi_post, vj_post = c
                dvi = vi_post - vi_pre
                dvj = vj_post - vj_pre
                if abs(dvi) > 0.5 and abs(dvj) > 0.5:
                    r = -dvj / dvi  # = m_i / m_j
                    if r > 0:  # physical ratio should be positive
                        ratios.append(r)
            
            if len(ratios) >= 1:
                r_hat = np.median(ratios)
                true_ratio = true_masses[i] / true_masses[j]
                error = abs(np.log(r_hat) - np.log(true_ratio))
            else:
                error = 2.0  # maximum penalty for unobserved pair
            
            pair_errors.append(error)
    
    return np.mean(pair_errors)
```

**Absolute Mass Estimation (secondary):**
```python
def estimate_absolute_masses(collisions, true_masses, n_objects=3):
    # Step 1: Estimate mass ratios from obj-obj collisions (MEDIAN per pair)
    ratio_estimates = {}  # (i,j) -> median ratio
    for i in range(n_objects):
        for j in range(i+1, n_objects):
            pair_collisions = [c for c in collisions 
                              if (c[1] == i and c[2] == j) or (c[1] == j and c[2] == i)]
            ratios = []
            for c in pair_collisions:
                step, ci, cj, vi_pre, vj_pre, vi_post, vj_post = c
                dvi = vi_post - vi_pre
                dvj = vj_post - vj_pre
                if abs(dvi) > 0.5 and abs(dvj) > 0.5:
                    r = -dvj / dvi
                    if r > 0:
                        ratios.append(r)
            if ratios:
                ratio_estimates[(i,j)] = np.median(ratios)
    
    # Step 2: Find best anchor (pointer-object collision with largest |dv_obj|)
    anchor_masses = {}  # obj_idx -> estimated mass
    po_collisions = [c for c in collisions 
                     if c[1] == n_objects or c[2] == n_objects]
    
    for obj_idx in range(n_objects):
        obj_po = [c for c in po_collisions 
                  if c[1] == obj_idx or c[2] == obj_idx]
        best_dv = 0
        best_mass = None
        for c in obj_po:
            step, ci, cj, vi_pre, vj_pre, vi_post, vj_post = c
            if ci == n_objects:  # pointer is entity i
                dv_pointer = vi_post - vi_pre
                dv_obj = vj_post - vj_pre
            else:  # pointer is entity j
                dv_pointer = vj_post - vj_pre
                dv_obj = vi_post - vi_pre
            
            if abs(dv_obj) > abs(best_dv) and abs(dv_obj) > 0.5:
                m_est = 10.0 * (-dv_pointer) / dv_obj
                if m_est > 0:  # physical mass must be positive
                    best_dv = dv_obj
                    best_mass = m_est
        
        if best_mass is not None:
            anchor_masses[obj_idx] = best_mass
    
    # Step 3: Build mass estimates
    m_hat = np.full(n_objects, np.nan)
    
    # Use anchors first
    for obj_idx, mass in anchor_masses.items():
        m_hat[obj_idx] = mass
    
    # Propagate via ratios
    for (i,j), ratio in ratio_estimates.items():
        if not np.isnan(m_hat[i]) and np.isnan(m_hat[j]):
            m_hat[j] = m_hat[i] / ratio
        elif np.isnan(m_hat[i]) and not np.isnan(m_hat[j]):
            m_hat[i] = m_hat[j] * ratio
    
    # Fill remaining with prior
    for k in range(n_objects):
        if np.isnan(m_hat[k]):
            m_hat[k] = 5.5
    
    return m_hat
```

**Output:** Write to `archive/iter_034/results_v2/`:
- `per_run.csv`
- `summary.csv`
- `analysis.md`

**Also:** Update `src/pre_registration.md` to reflect the v2 metric changes. Add a section "v2 Revision" explaining the MAPE failure and the MALRE replacement.

RUN THE EXPERIMENT. 24 runs, each 2000 steps with N=3 objects. Should take ~5 minutes.

After running, print clear summary:
1. Per-condition MALRE (mean, std, per-seed)
2. Per-condition MAPE (secondary, mean, std)
3. Per-condition coverage
4. All sanity check results
5. All gate results
6. Whether the benchmark is validated or falsified