"""
iter_034 Dynamics-Learning Benchmark

Implements the pre-registered experiment to validate a mass-estimation MAPE metric
with ground-truth collision logs and injected velocity observation noise.
"""

import os
import sys
import csv
import numpy as np
import torch

# Set torch threads as specified
torch.set_num_threads(4)

# Import environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import PhysicsSandbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = [7, 31, 53, 71, 83, 97, 113, 163]
CONDITIONS = ['ORACLE', 'RANDOM', 'PASSIVE']
N_STEPS = 2000
N_OBJECTS = 3
POINTER_IDX = N_OBJECTS  # entity index of pointer in concatenated arrays
THRESHOLD = 4.0
DV_THRESHOLD = 0.5
SIGMA_VEL = 0.5
POINTER_MASS = 10.0
KP = 2.0
KD = 0.5
PUSH_DISTANCE = 6.0
PUSH_VEL = 5.0
PUSH_COOLDOWN_STEPS = 15
N_BOOTSTRAP = 10000
OUTPUT_DIR = os.path.join('archive', 'iter_034', 'results')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_radii(env):
    """Return radii array for all entities (objects + pointer)."""
    return np.concatenate([env.radii.copy(), [env.pointer_radius]])


def detect_collisions(step, pre_positions, pre_velocities, post_positions, post_velocities, radii):
    """
    Detect collisions between adjacent entities sorted by post-position.
    Returns list of collision tuples:
        (step, entity_i, entity_j, v_i_pre, v_j_pre, v_i_post, v_j_post)
    """
    collisions = []
    sort_idx = np.argsort(post_positions)
    for k in range(len(sort_idx) - 1):
        i = int(sort_idx[k])
        j = int(sort_idx[k + 1])

        dist = post_positions[j] - post_positions[i]
        radii_sum = radii[i] + radii[j]

        if dist < radii_sum + THRESHOLD:
            dv_i = post_velocities[i] - pre_velocities[i]
            dv_j = post_velocities[j] - pre_velocities[j]
            if abs(dv_i) > DV_THRESHOLD and abs(dv_j) > DV_THRESHOLD:
                collisions.append((
                    step, i, j,
                    float(pre_velocities[i]), float(pre_velocities[j]),
                    float(post_velocities[i]), float(post_velocities[j])
                ))
    return collisions


def estimate_masses_from_collisions(collisions, noise_rng):
    """
    Estimate object masses from collision events with velocity noise.
    Returns (m_hat, noisy_collisions, objects_with_collisions).
    """
    objects_with_collisions = set()
    A_rows = []
    b_rows = []
    noisy_collisions = []

    for c in collisions:
        step, i, j, vi_pre, vj_pre, vi_post, vj_post = c

        # Add noise
        vi_pre_n = vi_pre + noise_rng.normal(0, SIGMA_VEL)
        vj_pre_n = vj_pre + noise_rng.normal(0, SIGMA_VEL)
        vi_post_n = vi_post + noise_rng.normal(0, SIGMA_VEL)
        vj_post_n = vj_post + noise_rng.normal(0, SIGMA_VEL)
        noisy_collisions.append((step, i, j, vi_pre_n, vj_pre_n, vi_post_n, vj_post_n))

        dvi = vi_post_n - vi_pre_n
        dvj = vj_post_n - vj_pre_n

        if i < N_OBJECTS:
            objects_with_collisions.add(i)
        if j < N_OBJECTS:
            objects_with_collisions.add(j)

        if i == POINTER_IDX or j == POINTER_IDX:
            # Pointer-object collision: direct mass estimate
            if i == POINTER_IDX:
                obj_idx = j
                dv_pointer = dvi
                dv_obj = dvj
            else:
                obj_idx = i
                dv_pointer = dvj
                dv_obj = dvi

            if abs(dv_obj) > 1e-8:
                m_est = POINTER_MASS * (-dv_pointer) / dv_obj
                row = np.zeros(N_OBJECTS)
                row[obj_idx] = 1.0
                A_rows.append(row)
                b_rows.append(m_est)
        else:
            # Object-object collision: ratio constraint
            if abs(dvi) > 1e-8:
                ratio = -dvj / dvi
                row = np.zeros(N_OBJECTS)
                row[i] = 1.0
                row[j] = -ratio
                A_rows.append(row)
                b_rows.append(0.0)

    if len(A_rows) > 0:
        A = np.array(A_rows)
        b = np.array(b_rows)
        m_hat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    else:
        m_hat = np.full(N_OBJECTS, 5.5)

    # Apply prior for objects with zero observed collisions
    for k in range(N_OBJECTS):
        if k not in objects_with_collisions:
            m_hat[k] = 5.5

    return m_hat, noisy_collisions, objects_with_collisions


def compute_mape(m_hat, true_masses):
    """Mean Absolute Percentage Error across objects."""
    return np.mean(np.abs(m_hat - true_masses) / true_masses)


def predict_post_velocities(vi_pre, vj_pre, mi, mj):
    """Elastic collision velocity prediction."""
    vi_pred = (vi_pre * (mi - mj) + 2.0 * mj * vj_pre) / (mi + mj)
    vj_pred = (vj_pre * (mj - mi) + 2.0 * mi * vi_pre) / (mi + mj)
    return vi_pred, vj_pred


def velocity_prediction_mse(train_collisions, test_collisions, true_masses):
    """
    Estimate masses from train collisions, predict post-collision velocities
    on test collisions, return MSE.
    """
    # Use a dummy RNG for train estimation (deterministic, but train collisions
    # already have noise from the main estimation; we re-estimate from scratch)
    # The noise seed is irrelevant here because we re-add noise to the TRUE
    # velocities in train_collisions. To keep it consistent, we use a fixed seed.
    train_rng = np.random.RandomState(42)
    m_hat_train, _, _ = estimate_masses_from_collisions(train_collisions, train_rng)

    if len(test_collisions) == 0:
        return float('nan')

    mse_list = []
    for c in test_collisions:
        step, i, j, vi_pre, vj_pre, vi_post, vj_post = c

        if i < N_OBJECTS:
            mi = m_hat_train[i]
        else:
            mi = POINTER_MASS

        if j < N_OBJECTS:
            mj = m_hat_train[j]
        else:
            mj = POINTER_MASS

        vi_pred, vj_pred = predict_post_velocities(vi_pre, vj_pre, mi, mj)
        mse_list.append((vi_pred - vi_post) ** 2)
        mse_list.append((vj_pred - vj_post) ** 2)

    return np.mean(mse_list)


def bootstrap_ci_gap(values_a, values_b, n_bootstrap=N_BOOTSTRAP):
    """
    Paired bootstrap CI for mean(b) - mean(a).
    Returns (lower_95, upper_95).
    """
    n = len(values_a)
    gaps = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        gap = np.mean(values_b[idx]) - np.mean(values_a[idx])
        gaps.append(gap)
    gaps = np.array(gaps)
    lower = float(np.percentile(gaps, 2.5))
    upper = float(np.percentile(gaps, 97.5))
    return lower, upper


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(condition, seed):
    """Run one episode and return metrics dict."""
    env = PhysicsSandbox(N=N_OBJECTS, substeps=10, seed=seed)
    np.random.seed(seed)  # seed policy randomness for reproducibility

    # Condition-specific state
    if condition == 'ORACLE':
        obj_pointer_collisions = [0, 0, 0]
        push_cooldown = 0
        prev_error = None
        forced_next_target = None

    collisions = []
    pointer_oob_steps = 0
    pointer_positions = []

    for step in range(N_STEPS):
        # Save pre-state
        pre_velocities = np.concatenate([env.velocities.copy(), [env.pointer_vel]])
        pre_positions = np.concatenate([env.positions.copy(), [env.pointer_pos]])

        # Determine action
        if condition == 'ORACLE':
            # Target selection
            if forced_next_target is not None:
                target_idx = forced_next_target
                forced_next_target = None
            else:
                target_idx = int(np.argmin(obj_pointer_collisions))

            # PD control
            error = env.positions[target_idx] - env.pointer_pos
            if prev_error is not None:
                d_error = error - prev_error
            else:
                d_error = 0.0
            acc = KP * error + KD * d_error
            prev_error = error

            # Decrement cooldown
            if push_cooldown > 0:
                push_cooldown -= 1

            # Push logic
            if abs(error) <= PUSH_DISTANCE and push_cooldown == 0:
                env.pointer_vel = PUSH_VEL * np.sign(error)
                push_cooldown = PUSH_COOLDOWN_STEPS
                # Switch target to next least-observed
                counts = np.array(obj_pointer_collisions, dtype=float)
                counts[target_idx] = np.inf
                forced_next_target = int(np.argmin(counts))

            action = {'acc': acc, 'push': False}

        elif condition == 'RANDOM':
            acc = np.random.uniform(-10, 10)
            push = np.random.rand() < 0.1
            action = {'acc': acc, 'push': push}

        else:  # PASSIVE
            action = {'acc': 0.0, 'push': False}

        # Step environment
        obs, info = env.step(action)

        # Save post-state
        post_velocities = np.concatenate([info["velocities"].copy(), [info["pointer_vel"]]])
        post_positions = np.concatenate([info["positions"].copy(), [info["pointer_pos"]]])

        pointer_positions.append(float(env.pointer_pos))
        if not (0.0 <= env.pointer_pos <= 128.0):
            pointer_oob_steps += 1

        # Collision detection
        radii = get_radii(env)
        step_collisions = detect_collisions(
            step, pre_positions, pre_velocities,
            post_positions, post_velocities, radii
        )
        collisions.extend(step_collisions)

        # Update ORACLE collision counts
        if condition == 'ORACLE':
            for c in step_collisions:
                _, i, j, _, _, _, _ = c
                if i == POINTER_IDX and j < N_OBJECTS:
                    obj_pointer_collisions[j] += 1
                elif j == POINTER_IDX and i < N_OBJECTS:
                    obj_pointer_collisions[i] += 1

    # True masses
    true_masses = env.masses.copy()

    # Mass estimation with noise
    noise_seed = seed * 100 + hash(condition) % 10000
    noise_rng = np.random.RandomState(noise_seed)
    m_hat, noisy_collisions, objects_with_collisions = estimate_masses_from_collisions(collisions, noise_rng)
    mape = compute_mape(m_hat, true_masses)

    # Per-object MAPE
    per_object_mape = np.abs(m_hat - true_masses) / true_masses

    # Collision breakdown
    n_collisions_total = len(collisions)
    n_pointer_obj = sum(1 for c in collisions if c[1] == POINTER_IDX or c[2] == POINTER_IDX)
    n_obj_obj = n_collisions_total - n_pointer_obj

    # Per-object pointer collision count (for ORACLE sanity checks)
    per_object_pointer_collisions = [0, 0, 0]
    for c in collisions:
        if c[1] == POINTER_IDX and c[2] < N_OBJECTS:
            per_object_pointer_collisions[c[2]] += 1
        elif c[2] == POINTER_IDX and c[1] < N_OBJECTS:
            per_object_pointer_collisions[c[1]] += 1

    # Velocity prediction (secondary metric)
    if len(collisions) >= 2:
        steps = [c[0] for c in collisions]
        split_step = int(np.percentile(steps, 80))
        train_collisions = [c for c in collisions if c[0] <= split_step]
        test_collisions = [c for c in collisions if c[0] > split_step]
        vel_pred_mse = velocity_prediction_mse(train_collisions, test_collisions, true_masses)
    else:
        vel_pred_mse = float('nan')
        train_collisions = []
        test_collisions = []

    # Sanity check S3: fraction of collisions with |dv| > 0.5
    # By construction our detector requires this, but we verify anyway.
    s3_valid = 0
    s3_total = 0
    for c in collisions:
        _, i, j, vi_pre, vj_pre, vi_post, vj_post = c
        if abs(vi_post - vi_pre) > DV_THRESHOLD and abs(vj_post - vj_pre) > DV_THRESHOLD:
            s3_valid += 1
        s3_total += 1
    s3_fraction = s3_valid / s3_total if s3_total > 0 else 1.0

    return {
        'condition': condition,
        'seed': seed,
        'mape': float(mape),
        'mape_obj0': float(per_object_mape[0]),
        'mape_obj1': float(per_object_mape[1]),
        'mape_obj2': float(per_object_mape[2]),
        'm_hat_obj0': float(m_hat[0]),
        'm_hat_obj1': float(m_hat[1]),
        'm_hat_obj2': float(m_hat[2]),
        'true_masses': true_masses.tolist(),
        'n_collisions_total': n_collisions_total,
        'n_pointer_obj': n_pointer_obj,
        'n_obj_obj': n_obj_obj,
        'per_object_pointer_collisions': per_object_pointer_collisions,
        'pointer_oob_fraction': pointer_oob_steps / N_STEPS,
        'vel_pred_mse': float(vel_pred_mse),
        's3_fraction': float(s3_fraction),
        'collisions': collisions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("iter_034 Dynamics-Learning Benchmark")
    print("=" * 70)

    # Run all experiments
    results = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            print(f"Running {condition:8s} seed={seed:3d} ...", end=" ", flush=True)
            res = run_single(condition, seed)
            results.append(res)
            print(f"MAPE={res['mape']:.4f}  Collisions={res['n_collisions_total']:3d}")

    # Organize by condition
    by_condition = {cond: [r for r in results if r['condition'] == cond] for cond in CONDITIONS}

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    sanity = {}

    # S1: ORACLE achieves >=3 pointer-object collisions per object (mean across seeds)
    oracle_per_obj = np.array([r['per_object_pointer_collisions'] for r in by_condition['ORACLE']])
    s1_mean_per_obj = oracle_per_obj.mean(axis=0)
    sanity['S1'] = {
        'pass': bool(np.all(s1_mean_per_obj >= 3)),
        'detail': f"Mean per-object pointer collisions: {s1_mean_per_obj.tolist()}",
    }

    # S2: ORACLE pointer-object collision count >= PASSIVE (per seed, paired)
    oracle_po = np.array([r['n_pointer_obj'] for r in by_condition['ORACLE']])
    passive_po = np.array([r['n_pointer_obj'] for r in by_condition['PASSIVE']])
    s2_paired = oracle_po >= passive_po
    sanity['S2'] = {
        'pass': bool(np.all(s2_paired)),
        'detail': f"ORACLE po={oracle_po.tolist()}, PASSIVE po={passive_po.tolist()}",
    }

    # S3: >=90% of logged collision events show |dv| > 0.5
    s3_fractions = [r['s3_fraction'] for r in results]
    s3_min = min(s3_fractions)
    sanity['S3'] = {
        'pass': bool(s3_min >= 0.90),
        'detail': f"Min fraction across all runs: {s3_min:.4f}",
    }

    # S4: ORACLE achieves >=1 pointer-object collision per object per seed
    s4_pass = True
    s4_detail = []
    for r in by_condition['ORACLE']:
        po = r['per_object_pointer_collisions']
        ok = all(c >= 1 for c in po)
        s4_pass = s4_pass and ok
        s4_detail.append(f"seed={r['seed']}: {po}")
    sanity['S4'] = {
        'pass': bool(s4_pass),
        'detail': "; ".join(s4_detail),
    }

    # S5: ORACLE pointer stays in bounds [0, 128] for >=95% of steps
    s5_oob = [r['pointer_oob_fraction'] for r in by_condition['ORACLE']]
    s5_max_oob = max(s5_oob)
    sanity['S5'] = {
        'pass': bool(s5_max_oob <= 0.05),
        'detail': f"Max OOB fraction across seeds: {s5_max_oob:.6f}",
    }

    # -----------------------------------------------------------------------
    # Gates
    # -----------------------------------------------------------------------
    oracle_mape = np.array([r['mape'] for r in by_condition['ORACLE']])
    random_mape = np.array([r['mape'] for r in by_condition['RANDOM']])
    passive_mape = np.array([r['mape'] for r in by_condition['PASSIVE']])

    # G1: RANDOM_MAPE - ORACLE_MAPE >= 0.15, lower 95% bootstrap CI >= 0.05
    g1_gap = random_mape - oracle_mape
    g1_mean_gap = float(np.mean(g1_gap))
    g1_lower, g1_upper = bootstrap_ci_gap(oracle_mape, random_mape)
    gate_g1 = (g1_mean_gap >= 0.15) and (g1_lower >= 0.05)

    # G2: PASSIVE_MAPE - RANDOM_MAPE >= 0.05, lower 95% CI > 0
    g2_gap = passive_mape - random_mape
    g2_mean_gap = float(np.mean(g2_gap))
    g2_lower, g2_upper = bootstrap_ci_gap(random_mape, passive_mape)
    gate_g2 = (g2_mean_gap >= 0.05) and (g2_lower > 0)

    # G3: ORACLE_MAPE < RANDOM_MAPE < PASSIVE_MAPE (means)
    gate_g3 = (np.mean(oracle_mape) < np.mean(random_mape) < np.mean(passive_mape))

    # -----------------------------------------------------------------------
    # Write per_run.csv
    # -----------------------------------------------------------------------
    per_run_path = os.path.join(OUTPUT_DIR, 'per_run.csv')
    with open(per_run_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'condition', 'seed', 'mape', 'mape_obj0', 'mape_obj1', 'mape_obj2',
            'm_hat_obj0', 'm_hat_obj1', 'm_hat_obj2',
            'true_mass0', 'true_mass1', 'true_mass2',
            'n_collisions_total', 'n_pointer_obj', 'n_obj_obj',
            'po_collisions_obj0', 'po_collisions_obj1', 'po_collisions_obj2',
            'pointer_oob_fraction', 'vel_pred_mse'
        ])
        for r in results:
            writer.writerow([
                r['condition'], r['seed'], r['mape'],
                r['mape_obj0'], r['mape_obj1'], r['mape_obj2'],
                r['m_hat_obj0'], r['m_hat_obj1'], r['m_hat_obj2'],
                r['true_masses'][0], r['true_masses'][1], r['true_masses'][2],
                r['n_collisions_total'], r['n_pointer_obj'], r['n_obj_obj'],
                r['per_object_pointer_collisions'][0],
                r['per_object_pointer_collisions'][1],
                r['per_object_pointer_collisions'][2],
                r['pointer_oob_fraction'], r['vel_pred_mse']
            ])

    # -----------------------------------------------------------------------
    # Write summary.csv
    # -----------------------------------------------------------------------
    summary_path = os.path.join(OUTPUT_DIR, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'condition', 'mean_mape', 'std_mape', 'min_mape', 'max_mape',
            'mean_vel_pred_mse', 'mean_n_collisions', 'mean_n_pointer_obj', 'mean_n_obj_obj'
        ])
        for cond in CONDITIONS:
            mapes = [r['mape'] for r in by_condition[cond]]
            vmses = [r['vel_pred_mse'] for r in by_condition[cond] if not np.isnan(r['vel_pred_mse'])]
            ncols = [r['n_collisions_total'] for r in by_condition[cond]]
            npo = [r['n_pointer_obj'] for r in by_condition[cond]]
            noo = [r['n_obj_obj'] for r in by_condition[cond]]
            writer.writerow([
                cond,
                np.mean(mapes), np.std(mapes), np.min(mapes), np.max(mapes),
                np.mean(vmses) if vmses else float('nan'),
                np.mean(ncols), np.mean(npo), np.mean(noo)
            ])

    # -----------------------------------------------------------------------
    # Write sanity_checks.txt
    # -----------------------------------------------------------------------
    sanity_path = os.path.join(OUTPUT_DIR, 'sanity_checks.txt')
    with open(sanity_path, 'w') as f:
        f.write("iter_034 Sanity Checks\n")
        f.write("=" * 50 + "\n\n")
        all_pass = True
        for key in ['S1', 'S2', 'S3', 'S4', 'S5']:
            status = "PASS" if sanity[key]['pass'] else "FAIL"
            if not sanity[key]['pass']:
                all_pass = False
            f.write(f"{key}: {status}\n")
            f.write(f"  {sanity[key]['detail']}\n\n")
        f.write(f"\nAll sanity checks passed: {all_pass}\n")

    # -----------------------------------------------------------------------
    # Write analysis.md
    # -----------------------------------------------------------------------
    analysis_path = os.path.join(OUTPUT_DIR, 'analysis.md')
    with open(analysis_path, 'w') as f:
        f.write("# iter_034 Dynamics-Learning Benchmark Analysis\n\n")

        f.write("## Experimental Setup\n\n")
        f.write(f"- Environment: PhysicsSandbox(N={N_OBJECTS}), {N_STEPS} steps\n")
        f.write(f"- Seeds: {SEEDS}\n")
        f.write(f"- Conditions: {CONDITIONS}\n")
        f.write(f"- Velocity noise: sigma_vel={SIGMA_VEL}\n")
        f.write(f"- Collision threshold: {THRESHOLD}, dv_threshold: {DV_THRESHOLD}\n\n")

        f.write("## Per-Seed MAPE Results\n\n")
        f.write("| Seed | ORACLE | RANDOM | PASSIVE |\n")
        f.write("|------|--------|--------|---------|\n")
        for idx, seed in enumerate(SEEDS):
            f.write(f"| {seed:4d} | {oracle_mape[idx]:.4f} | {random_mape[idx]:.4f} | {passive_mape[idx]:.4f} |\n")

        f.write("\n## Summary Statistics\n\n")
        f.write("| Condition | Mean MAPE | Std MAPE | Min MAPE | Max MAPE |\n")
        f.write("|-----------|-----------|----------|----------|----------|\n")
        for cond in CONDITIONS:
            mapes = [r['mape'] for r in by_condition[cond]]
            f.write(f"| {cond:9s} | {np.mean(mapes):.4f} | {np.std(mapes):.4f} | {np.min(mapes):.4f} | {np.max(mapes):.4f} |\n")

        f.write("\n## Gates\n\n")
        f.write(f"**G1** (RANDOM - ORACLE >= 0.15, CI lower >= 0.05):\n")
        f.write(f"- Mean gap: {g1_mean_gap:.4f}\n")
        f.write(f"- 95% Bootstrap CI: [{g1_lower:.4f}, {g1_upper:.4f}]\n")
        f.write(f"- Result: {'PASS' if gate_g1 else 'FAIL'}\n\n")

        f.write(f"**G2** (PASSIVE - RANDOM >= 0.05, CI lower > 0):\n")
        f.write(f"- Mean gap: {g2_mean_gap:.4f}\n")
        f.write(f"- 95% Bootstrap CI: [{g2_lower:.4f}, {g2_upper:.4f}]\n")
        f.write(f"- Result: {'PASS' if gate_g2 else 'FAIL'}\n\n")

        f.write(f"**G3** (ORACLE < RANDOM < PASSIVE means):\n")
        f.write(f"- ORACLE mean: {np.mean(oracle_mape):.4f}\n")
        f.write(f"- RANDOM mean: {np.mean(random_mape):.4f}\n")
        f.write(f"- PASSIVE mean: {np.mean(passive_mape):.4f}\n")
        f.write(f"- Result: {'PASS' if gate_g3 else 'FAIL'}\n\n")

        f.write("## Sanity Checks\n\n")
        for key in ['S1', 'S2', 'S3', 'S4', 'S5']:
            status = "PASS" if sanity[key]['pass'] else "FAIL"
            f.write(f"- **{key}**: {status} — {sanity[key]['detail']}\n")

        f.write("\n## Conclusion\n\n")
        all_gates = gate_g1 and gate_g2 and gate_g3
        all_sanity = all(sanity[k]['pass'] for k in sanity)
        if all_sanity and all_gates:
            f.write("**Benchmark VALIDATED.** All sanity checks and gates pass.\n")
        elif not all_sanity:
            f.write("**Benchmark FALSIFIED (F3).** One or more sanity checks failed.\n")
        else:
            f.write("**Benchmark FALSIFIED.** Sanity checks pass but one or more gates failed.\n")

    # -----------------------------------------------------------------------
    # Print summary to console
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    for key in ['S1', 'S2', 'S3', 'S4', 'S5']:
        status = "PASS" if sanity[key]['pass'] else "FAIL"
        print(f"  {key}: {status} — {sanity[key]['detail']}")

    print("\n" + "=" * 70)
    print("PER-CONDITION MAPE")
    print("=" * 70)
    for cond in CONDITIONS:
        mapes = [r['mape'] for r in by_condition[cond]]
        print(f"  {cond:8s}: mean={np.mean(mapes):.4f} std={np.std(mapes):.4f}")
        for r in by_condition[cond]:
            print(f"            seed={r['seed']:3d}  mape={r['mape']:.4f}")

    print("\n" + "=" * 70)
    print("GATES")
    print("=" * 70)
    print(f"  G1 (RANDOM-ORACLE>=0.15, CI>=0.05):  {'PASS' if gate_g1 else 'FAIL'}  "
          f"gap={g1_mean_gap:.4f} CI=[{g1_lower:.4f}, {g1_upper:.4f}]")
    print(f"  G2 (PASSIVE-RANDOM>=0.05, CI>0):     {'PASS' if gate_g2 else 'FAIL'}  "
          f"gap={g2_mean_gap:.4f} CI=[{g2_lower:.4f}, {g2_upper:.4f}]")
    print(f"  G3 (ORACLE<RANDOM<PASSIVE):          {'PASS' if gate_g3 else 'FAIL'}")

    print("\n" + "=" * 70)
    all_gates = gate_g1 and gate_g2 and gate_g3
    all_sanity = all(sanity[k]['pass'] for k in sanity)
    if all_sanity and all_gates:
        print("BENCHMARK VALIDATED")
    elif not all_sanity:
        print("BENCHMARK FALSIFIED (sanity check failure)")
    else:
        print("BENCHMARK FALSIFIED (gate failure)")
    print("=" * 70)


if __name__ == '__main__':
    main()
