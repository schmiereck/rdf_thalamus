Create `src/run_iter034_v2.py` and run it. This is a REVISED benchmark after v1 (MAPE) was falsified due to pointer-object collision noise sensitivity.

IMPORTANT: Do NOT read any files. Write the script directly from the spec below. Then run it.

```python
#!/usr/bin/env python3
"""iter_034 v2: Dynamics-Learning Benchmark with MALRE (Mean Absolute Log-Ratio Error).

v1 MAPE was falsified: PASSIVE=0.597 < RANDOM=0.999 < ORACLE=1.005 (inverted).
Root cause: pointer-object mass estimates are extremely noisy (m_i = 10*(-dv_ptr)/dv_obj
blows up when dv_obj is small), and hundreds of such rows overwhelm least-squares.

v2 fix: use MEDIAN of mass-RATIO estimates from object-object collisions only.
This is robust because: (a) ratio = -dv_j/dv_i is stable across substep impulses,
(b) MEDIAN filters outliers, (c) object-object collisions are more informative per event.
"""
import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import PhysicsSandbox

SEEDS = [7, 31, 53, 71, 83, 97, 113, 163]
N_OBJECTS = 3
N_STEPS = 2000
THRESHOLD = 4.0
DV_THRESHOLD = 0.5
POINTER_IDX = N_OBJECTS
POINTER_MASS = 10.0
KP, KD = 2.0, 0.5
PUSH_DISTANCE = 6.0
PUSH_VEL = 5.0
PUSH_COOLDOWN_STEPS = 15
N_BOOTSTRAP = 10000
OUTPUT_DIR = os.path.join('archive', 'iter_034', 'results_v2')


def detect_collisions(step, pre_pos, pre_vel, post_pos, post_vel, radii):
    colls = []
    sort_idx = np.argsort(post_pos)
    for k in range(len(sort_idx) - 1):
        i, j = int(sort_idx[k]), int(sort_idx[k+1])
        dist = post_pos[j] - post_pos[i]
        if dist < radii[i] + radii[j] + THRESHOLD:
            dvi = post_vel[i] - pre_vel[i]
            dvj = post_vel[j] - pre_vel[j]
            if abs(dvi) > DV_THRESHOLD and abs(dvj) > DV_THRESHOLD:
                colls.append((step, i, j, float(pre_vel[i]), float(pre_vel[j]),
                              float(post_vel[i]), float(post_vel[j])))
    return colls


def compute_malre(collisions, true_masses):
    """Mean Absolute Log-Ratio Error from obj-obj collisions, using MEDIAN per pair."""
    pair_errors = []
    for i in range(N_OBJECTS):
        for j in range(i+1, N_OBJECTS):
            ratios = []
            for c in collisions:
                step, ci, cj, vi_pre, vj_pre, vi_post, vj_post = c
                if (ci == i and cj == j) or (ci == j and cj == i):
                    dvi = vi_post - vi_pre
                    dvj = vj_post - vj_pre
                    if abs(dvi) > 0.5 and abs(dvj) > 0.5:
                        r = -dvj / dvi
                        if r > 0:
                            ratios.append(r)
            if len(ratios) >= 1:
                r_hat = np.median(ratios)
                true_ratio = true_masses[i] / true_masses[j]
                error = abs(np.log(r_hat) - np.log(true_ratio))
            else:
                error = 2.0
            pair_errors.append(error)
    return float(np.mean(pair_errors)), pair_errors


def compute_mape_robust(collisions, true_masses):
    """Absolute mass estimation using best pointer-object collision as anchor."""
    # Step 1: Estimate mass ratios from obj-obj collisions
    ratio_estimates = {}
    for i in range(N_OBJECTS):
        for j in range(i+1, N_OBJECTS):
            ratios = []
            for c in collisions:
                step, ci, cj, vi_pre, vj_pre, vi_post, vj_post = c
                if (ci == i and cj == j) or (ci == j and cj == i):
                    dvi = vi_post - vi_pre
                    dvj = vj_post - vj_pre
                    if abs(dvi) > 0.5 and abs(dvj) > 0.5:
                        r = -dvj / dvi
                        if r > 0:
                            ratios.append(r)
            if ratios:
                ratio_estimates[(i,j)] = np.median(ratios)
    
    # Step 2: Find best anchor per object
    anchor_masses = {}
    for obj_idx in range(N_OBJECTS):
        best_dv = 0
        best_mass = None
        for c in collisions:
            step, ci, cj, vi_pre, vj_pre, vi_post, vj_post = c
            if ci == obj_idx and cj == POINTER_IDX:
                dv_obj = vi_post - vi_pre
                dv_ptr = vj_post - vj_pre
            elif cj == obj_idx and ci == POINTER_IDX:
                dv_obj = vj_post - vj_pre
                dv_ptr = vi_post - vi_pre
            else:
                continue
            if abs(dv_obj) > abs(best_dv) and abs(dv_obj) > 0.5:
                m_est = 10.0 * (-dv_ptr) / dv_obj
                if m_est > 0:
                    best_dv = dv_obj
                    best_mass = m_est
        if best_mass is not None:
            anchor_masses[obj_idx] = best_mass
    
    # Step 3: Build estimates
    m_hat = np.full(N_OBJECTS, np.nan)
    for idx, mass in anchor_masses.items():
        m_hat[idx] = mass
    for (i,j), ratio in ratio_estimates.items():
        if not np.isnan(m_hat[i]) and np.isnan(m_hat[j]):
            m_hat[j] = m_hat[i] / ratio
        elif np.isnan(m_hat[i]) and not np.isnan(m_hat[j]):
            m_hat[i] = m_hat[j] * ratio
    for k in range(N_OBJECTS):
        if np.isnan(m_hat[k]):
            m_hat[k] = 5.5
    
    mape = float(np.mean(np.abs(m_hat - true_masses) / true_masses))
    return mape, m_hat


def compute_coverage(collisions):
    """Fraction of pair types (obj-obj + pointer-obj) with >=3 collisions."""
    n_covered = 0
    n_total = 0
    # Object-object pairs
    for i in range(N_OBJECTS):
        for j in range(i+1, N_OBJECTS):
            n_total += 1
            count = sum(1 for c in collisions if (c[1]==i and c[2]==j) or (c[1]==j and c[2]==i))
            if count >= 3:
                n_covered += 1
    # Pointer-object pairs
    for i in range(N_OBJECTS):
        n_total += 1
        count = sum(1 for c in collisions if (c[1]==i and c[2]==POINTER_IDX) or (c[1]==POINTER_IDX and c[2]==i))
        if count >= 3:
            n_covered += 1
    return n_covered / n_total if n_total > 0 else 0.0


def bootstrap_ci_gap(a_vals, b_vals, n_boot=N_BOOTSTRAP):
    """Paired bootstrap CI for mean(b) - mean(a)."""
    n = len(a_vals)
    gaps = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        gaps.append(np.mean(b_vals[idx]) - np.mean(a_vals[idx]))
    gaps = np.array(gaps)
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def run_single(condition, seed):
    env = PhysicsSandbox(N=N_OBJECTS, substeps=10, seed=seed)
    rng = np.random.RandomState(seed * 1000 + hash(condition) % 10000)
    
    obj_ptr_colls = [0, 0, 0]
    push_cooldown = 0
    prev_error = None
    forced_target = None
    collisions = []
    ptr_oob = 0
    
    for step in range(N_STEPS):
        pre_vel = np.concatenate([env.velocities.copy(), [env.pointer_vel]])
        pre_pos = np.concatenate([env.positions.copy(), [env.pointer_pos]])
        
        if condition == 'ORACLE':
            if forced_target is not None:
                target = forced_target
                forced_target = None
            else:
                target = int(np.argmin(obj_ptr_colls))
            error = env.positions[target] - env.pointer_pos
            d_error = (error - prev_error) if prev_error is not None else 0.0
            prev_error = error
            acc = KP * error + KD * d_error
            if push_cooldown > 0:
                push_cooldown -= 1
            if abs(error) <= PUSH_DISTANCE and push_cooldown == 0:
                env.pointer_vel = PUSH_VEL * np.sign(error)
                push_cooldown = PUSH_COOLDOWN_STEPS
                counts = np.array(obj_ptr_colls, dtype=float)
                counts[target] = np.inf
                forced_target = int(np.argmin(counts))
            action = {'acc': acc, 'push': False}
        elif condition == 'RANDOM':
            acc = rng.uniform(-10, 10)
            push = rng.rand() < 0.1
            action = {'acc': acc, 'push': push}
        else:  # PASSIVE
            action = {'acc': 0.0, 'push': False}
        
        obs, info = env.step(action)
        post_vel = np.concatenate([info["velocities"].copy(), [info["pointer_vel"]]])
        post_pos = np.concatenate([info["positions"].copy(), [info["pointer_pos"]]])
        radii = np.concatenate([env.radii.copy(), [env.pointer_radius]])
        
        if not (0 <= env.pointer_pos <= 128):
            ptr_oob += 1
        
        step_colls = detect_collisions(step, pre_pos, pre_vel, post_pos, post_vel, radii)
        collisions.extend(step_colls)
        
        if condition == 'ORACLE':
            for c in step_colls:
                _, ci, cj, *_ = c
                if ci == POINTER_IDX and cj < N_OBJECTS:
                    obj_ptr_colls[cj] += 1
                elif cj == POINTER_IDX and ci < N_OBJECTS:
                    obj_ptr_colls[ci] += 1
    
    true_masses = env.masses.copy()
    malre, pair_errors = compute_malre(collisions, true_masses)
    mape, m_hat = compute_mape_robust(collisions, true_masses)
    coverage = compute_coverage(collisions)
    
    n_total = len(collisions)
    n_ptr_obj = sum(1 for c in collisions if c[1]==POINTER_IDX or c[2]==POINTER_IDX)
    n_obj_obj = n_total - n_ptr_obj
    
    per_obj_ptr = [0]*N_OBJECTS
    for c in collisions:
        if c[1]==POINTER_IDX and c[2]<N_OBJECTS: per_obj_ptr[c[2]] += 1
        elif c[2]==POINTER_IDX and c[1]<N_OBJECTS: per_obj_ptr[c[1]] += 1
    
    per_pair_obj = {}
    for i in range(N_OBJECTS):
        for j in range(i+1, N_OBJECTS):
            cnt = sum(1 for c in collisions if (c[1]==i and c[2]==j) or (c[1]==j and c[2]==i))
            per_pair_obj[(i,j)] = cnt
    
    return {
        'condition': condition, 'seed': seed,
        'malre': malre, 'mape': mape, 'coverage': coverage,
        'm_hat': m_hat.tolist(), 'true_masses': true_masses.tolist(),
        'pair_errors': pair_errors,
        'n_total': n_total, 'n_ptr_obj': n_ptr_obj, 'n_obj_obj': n_obj_obj,
        'per_obj_ptr': per_obj_ptr, 'per_pair_obj': {f"{k[0]}_{k[1]}": v for k,v in per_pair_obj.items()},
        'ptr_oob_frac': ptr_oob / N_STEPS,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("iter_034 v2: MALRE Benchmark")
    results = []
    for cond in ['ORACLE', 'RANDOM', 'PASSIVE']:
        for seed in SEEDS:
            print(f"  {cond:8s} seed={seed:3d} ...", end=" ", flush=True)
            r = run_single(cond, seed)
            results.append(r)
            print(f"MALRE={r['malre']:.4f} MAPE={r['mape']:.4f} cov={r['coverage']:.2f}")
    
    by_cond = {c: [r for r in results if r['condition']==c] for c in ['ORACLE','RANDOM','PASSIVE']}
    
    # Sanity checks
    oracle_po_per_obj = np.array([r['per_obj_ptr'] for r in by_cond['ORACLE']])
    s1 = bool(np.all(oracle_po_per_obj.mean(axis=0) >= 3))
    
    oracle_oo_per_pair = np.array([[r['per_pair_obj'][f"{i}_{j}"] for i,j in [(0,1),(0,2),(1,2)]] for r in by_cond['ORACLE']])
    s2 = bool(np.all(oracle_oo_per_pair.mean(axis=0) >= 10))
    s3 = True  # collision detector enforces this
    s4 = bool(all(all(c >= 1 for c in r['per_obj_ptr']) for r in by_cond['ORACLE']))
    s5 = bool(max(r['ptr_oob_frac'] for r in by_cond['ORACLE']) <= 0.05)
    
    all_sanity = s1 and s2 and s3 and s4 and s5
    
    # Extract metrics
    oracle_malre = np.array([r['malre'] for r in by_cond['ORACLE']])
    random_malre = np.array([r['malre'] for r in by_cond['RANDOM']])
    passive_malre = np.array([r['malre'] for r in by_cond['PASSIVE']])
    
    oracle_mape = np.array([r['mape'] for r in by_cond['ORACLE']])
    random_mape = np.array([r['mape'] for r in by_cond['RANDOM']])
    passive_mape = np.array([r['mape'] for r in by_cond['PASSIVE']])
    
    oracle_cov = np.array([r['coverage'] for r in by_cond['ORACLE']])
    passive_cov = np.array([r['coverage'] for r in by_cond['PASSIVE']])
    
    # Gates
    g1_gap = float(np.mean(passive_malre - oracle_malre))
    g1_lo, g1_hi = bootstrap_ci_gap(oracle_malre, passive_malre)
    g1_pass = g1_gap >= 0.3 and g1_lo >= 0.1
    
    g2_gap = float(np.mean(passive_malre - random_malre))
    g2_lo, g2_hi = bootstrap_ci_gap(random_malre, passive_malre)
    g2_pass = g2_gap >= 0.1 and g2_lo > 0
    
    g3_pass = float(np.mean(oracle_malre)) < float(np.mean(random_malre)) < float(np.mean(passive_malre))
    
    g4_gap = float(np.mean(oracle_cov - passive_cov))
    g4_pass = g4_gap >= 0.2
    
    # Print results
    print("\n" + "="*70)
    print("SANITY CHECKS")
    for name, val in [("S1 (≥3 ptr-obj/object)", s1), ("S2 (≥10 obj-obj/pair)", s2),
                       ("S3 (≥90% |Δv|>0.5)", s3), ("S4 (≥1 ptr-obj/object/seed)", s4),
                       ("S5 (ptr in bounds)", s5)]:
        print(f"  {name}: {'PASS' if val else 'FAIL'}")
    print(f"  All sanity: {'PASS' if all_sanity else 'FAIL'}")
    
    print("\nPER-CONDITION RESULTS")
    for cond in ['ORACLE','RANDOM','PASSIVE']:
        vals = [r['malre'] for r in by_cond[cond]]
        mapes = [r['mape'] for r in by_cond[cond]]
        covs = [r['coverage'] for r in by_cond[cond]]
        print(f"  {cond:8s}: MALRE={np.mean(vals):.4f}±{np.std(vals):.4f}  "
              f"MAPE={np.mean(mapes):.4f}±{np.std(mapes):.4f}  "
              f"COV={np.mean(covs):.2f}")
    
    print("\nPER-SEED MALRE")
    print(f"{'Seed':>6} | {'ORACLE':>8} | {'RANDOM':>8} | {'PASSIVE':>8}")
    for i, seed in enumerate(SEEDS):
        print(f"{seed:>6} | {oracle_malre[i]:.4f}   | {random_malre[i]:.4f}   | {passive_malre[i]:.4f}")
    
    print("\nGATES")
    print(f"  G1 (PASSIVE-ORACLE MALRE≥0.3, CI≥0.1): {'PASS' if g1_pass else 'FAIL'}  gap={g1_gap:.4f} CI=[{g1_lo:.4f},{g1_hi:.4f}]")
    print(f"  G2 (PASSIVE-RANDOM MALRE≥0.1, CI>0):    {'PASS' if g2_pass else 'FAIL'}  gap={g2_gap:.4f} CI=[{g2_lo:.4f},{g2_hi:.4f}]")
    print(f"  G3 (ORACLE<RANDOM<PASSIVE):              {'PASS' if g3_pass else 'FAIL'}")
    print(f"  G4 (ORACLE-PASSIVE coverage≥0.2):        {'PASS' if g4_pass else 'FAIL'}  gap={g4_gap:.2f}")
    
    validated = all_sanity and g1_pass and g2_pass and g3_pass and g4_pass
    print(f"\nBENCHMARK {'VALIDATED' if validated else 'FALSIFIED'}")
    
    # Save per_run.csv
    with open(os.path.join(OUTPUT_DIR, 'per_run.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition','seed','malre','mape','coverage','n_total','n_ptr_obj','n_obj_obj',
                     'po_obj0','po_obj1','po_obj2',
                     'oo_01','oo_02','oo_12',
                     'm_hat0','m_hat1','m_hat2',
                     'true_m0','true_m1','true_m2'])
        for r in results:
            w.writerow([r['condition'], r['seed'], r['malre'], r['mape'], r['coverage'],
                        r['n_total'], r['n_ptr_obj'], r['n_obj_obj'],
                        *r['per_obj_ptr'],
                        r['per_pair_obj']['0_1'], r['per_pair_obj']['0_2'], r['per_pair_obj']['1_2'],
                        *r['m_hat'], *r['true_masses']])
    
    # Save analysis
    with open(os.path.join(OUTPUT_DIR, 'analysis.md'), 'w') as f:
        f.write("# iter_034 v2 Benchmark Analysis (MALRE)\n\n")
        f.write("## v1 Failure\nMAPE was falsified (PASSIVE=0.597 < RANDOM=0.999 < ORACLE=1.005).\n")
        f.write("Pointer-object collision mass estimates overwhelmed least-squares.\n\n")
        f.write("## v2 Metric: MALRE\nMean Absolute Log-Ratio Error from MEDIAN of obj-obj collision ratios.\n\n")
        f.write("## Per-Seed MALRE\n\n")
        f.write("| Seed | ORACLE | RANDOM | PASSIVE |\n|------|--------|--------|--------|\n")
        for i, seed in enumerate(SEEDS):
            f.write(f"| {seed} | {oracle_malre[i]:.4f} | {random_malre[i]:.4f} | {passive_malre[i]:.4f} |\n")
        f.write(f"\n## Summary\n\n")
        for cond in ['ORACLE','RANDOM','PASSIVE']:
            vals = [r['malre'] for r in by_cond[cond]]
            f.write(f"- {cond}: MALRE={np.mean(vals):.4f}±{np.std(vals):.4f}\n")
        f.write(f"\n## Gates\n- G1: gap={g1_gap:.4f} CI=[{g1_lo:.4f},{g1_hi:.4f}] {'PASS' if g1_pass else 'FAIL'}\n")
        f.write(f"- G2: gap={g2_gap:.4f} CI=[{g2_lo:.4f},{g2_hi:.4f}] {'PASS' if g2_pass else 'FAIL'}\n")
        f.write(f"- G3: {'PASS' if g3_pass else 'FAIL'}\n- G4: gap={g4_gap:.2f} {'PASS' if g4_pass else 'FAIL'}\n")
        f.write(f"\n## Result: {'VALIDATED' if validated else 'FALSIFIED'}\n")
    
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
```

Write this exact script to `src/run_iter034_v2.py`, then run it with `python src/run_iter034_v2.py`.