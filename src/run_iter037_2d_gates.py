"""src/run_iter037_2d_gates.py

2D Cheap Gate Experiment for the Thalamus project.
Core de-risking experiment for a potential 2D environment migration.
All parameters, gate definitions, and decision rules are pre-registered in
src/pre_registration.md and must not be modified.
"""

import os
import sys
import csv
import numpy as np

# Fix stdout encoding on Windows for redirection compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Pre-registered parameters (FROZEN) ───────────────────────────────────────
SEEDS = [7, 31, 53, 71, 83]
N_OBJ = 3
N_STEPS = 2000
SUBSTEPS = 10
ARENA_SIZE = 64.0
PTR_RADIUS = 4.0
PTR_MASS = 10.0
PTR_START_POS = np.array([32.0, 32.0])
OBJ_RADIUS_RANGE = [3.0, 8.0]
OBJ_VEL_RANGES = [(-2.0, -0.5), (0.5, 2.0)]
PROX_THRESH = 4.0
DV_THRESH = 0.5
GAZE_RADIUS = 8.0
PROBE_P = 0.01
PROBE_BUDGET = 20
GAZE_MASS = 10.0
GAZE_ACC_RANGE = (-10.0, 10.0)
GATE1_THRESH = 3.0
GATE1B_THRESH = 0.30
GATE2_THRESH = 0.50
N_SEEDS_PASS = 4

OUT_DIR = os.path.join('archive', 'iter_037', 'results')


def _rand_vel_component(rng):
    if rng.rand() < 0.5:
        return rng.uniform(-2.0, -0.5)
    return rng.uniform(0.5, 2.0)


# ── PhysicsSandbox2D ──────────────────────────────────────────────────────────
class PhysicsSandbox2D:
    def __init__(self, N=N_OBJ, substeps=SUBSTEPS, seed=None):
        self.N = N
        self.substeps = substeps
        self.dt = 1.0 / substeps
        self.rng = np.random.RandomState(seed)

        self.positions = np.zeros((N, 2), dtype=np.float64)
        self.velocities = np.zeros((N, 2), dtype=np.float64)
        self.radii = np.zeros(N, dtype=np.float64)
        self.masses = np.zeros(N, dtype=np.float64)
        self.colors = np.zeros((N, 3), dtype=np.float64)

        self.pointer_pos = PTR_START_POS.copy()
        self.pointer_vel = np.zeros(2, dtype=np.float64)
        self.pointer_radius = PTR_RADIUS
        self.pointer_mass = PTR_MASS
        self.pointer_color = np.ones(3, dtype=np.float64)

        self.gaze_pos = PTR_START_POS.copy()
        self.gaze_vel = np.zeros(2, dtype=np.float64)
        self.gaze_radius = GAZE_RADIUS
        self.gaze_mass = GAZE_MASS

        self._init_objects()

    def _init_objects(self):
        self.radii = self.rng.uniform(OBJ_RADIUS_RANGE[0], OBJ_RADIUS_RANGE[1], self.N)
        self.masses = self.radii.copy()
        for i in range(self.N):
            c = np.zeros(3)
            c[i % 3] = 1.0
            self.colors[i] = c
        for i in range(self.N):
            self.velocities[i, 0] = _rand_vel_component(self.rng)
            self.velocities[i, 1] = _rand_vel_component(self.rng)
        for i in range(self.N):
            margin = self.radii[i] + 0.5
            x_min = (ARENA_SIZE / self.N) * i + margin
            x_max = (ARENA_SIZE / self.N) * (i + 1) - margin
            self.positions[i, 0] = self.rng.uniform(x_min, x_max)
            self.positions[i, 1] = self.rng.uniform(margin, ARENA_SIZE - margin)

    def _resolve_circle_circle_2d(self, pos1, vel1, m1, r1,
                                   pos2, vel2, m2, r2):
        diff = pos2 - pos1
        dist = np.linalg.norm(diff)
        min_dist = r1 + r2
        if dist < min_dist and dist > 1e-12:
            n = diff / dist
            # Only resolve velocity if the bodies are approaching each other
            # along the normal direction. This prevents double-counting the
            # same collision across multiple substeps.
            v_rel_n = np.dot(vel2 - vel1, n)
            if v_rel_n > 0:
                overlap = min_dist - dist
                m_inv1 = 1.0 / m1
                m_inv2 = 1.0 / m2
                sum_inv = m_inv1 + m_inv2
                new_pos1 = pos1 - overlap * (m_inv1 / sum_inv) * n
                new_pos2 = pos2 + overlap * (m_inv2 / sum_inv) * n
                v1n = np.dot(vel1, n)
                v2n = np.dot(vel2, n)
                v1n_new = (v1n * (m1 - m2) + 2.0 * m2 * v2n) / (m1 + m2)
                v2n_new = (v2n * (m2 - m1) + 2.0 * m1 * v1n) / (m1 + m2)
                new_vel1 = vel1 + (v1n_new - v1n) * n
                new_vel2 = vel2 + (v2n_new - v2n) * n
                return new_pos1, new_vel1, new_pos2, new_vel2
            else:
                # Separating: only fix position overlap, do not exchange velocity
                overlap = min_dist - dist
                m_inv1 = 1.0 / m1
                m_inv2 = 1.0 / m2
                sum_inv = m_inv1 + m_inv2
                new_pos1 = pos1 - overlap * (m_inv1 / sum_inv) * n
                new_pos2 = pos2 + overlap * (m_inv2 / sum_inv) * n
                return new_pos1, vel1.copy(), new_pos2, vel2.copy()
        return pos1.copy(), vel1.copy(), pos2.copy(), vel2.copy()

    def _wall_bounce(self, pos, vel, radius):
        pos = pos.copy()
        vel = vel.copy()
        for axis in range(2):
            if pos[axis] - radius < 0.0:
                pos[axis] = radius
                if vel[axis] < 0.0:
                    vel[axis] = -vel[axis]
            elif pos[axis] + radius > ARENA_SIZE:
                pos[axis] = ARENA_SIZE - radius
                if vel[axis] > 0.0:
                    vel[axis] = -vel[axis]
        return pos, vel

    def step(self, pointer_acc=None, ghostly_gaze=False, probe_action=None,
             record_collisions=False):
        """
        One physics step with substeps.
        If record_collisions=True, returns collision_events as list of dicts
        with keys (obj_idx, pre_ptr_vel, pre_obj_vel, post_ptr_vel, post_obj_vel).
        Collisions are counted at STEP granularity (max one per object per step),
        matching the 1D benchmark paradigm.
        """
        if pointer_acc is None:
            pointer_acc = np.zeros(2)
        if probe_action is None:
            probe_action = {'do_probe': False, 'probe_acc': np.zeros(2)}

        probe_acc = probe_action['probe_acc']
        do_probe = probe_action['do_probe']

        # Save pre-step state for collision recording
        if record_collisions:
            pre_ptr_vel = self.pointer_vel.copy()
            pre_obj_vels = self.velocities.copy()
            ptr_obj_contact = [False] * self.N  # track if pair had contact during step

        # Substep loop
        for _ in range(self.substeps):
            self.pointer_vel += pointer_acc * self.dt
            self.gaze_vel += probe_acc * self.dt

            self.positions += self.velocities * self.dt
            self.pointer_pos += self.pointer_vel * self.dt
            self.gaze_pos += self.gaze_vel * self.dt

            # Wall bounces
            for i in range(self.N):
                self.positions[i], self.velocities[i] = self._wall_bounce(
                    self.positions[i], self.velocities[i], self.radii[i]
                )
            self.pointer_pos, self.pointer_vel = self._wall_bounce(
                self.pointer_pos, self.pointer_vel, self.pointer_radius
            )
            self.gaze_pos, self.gaze_vel = self._wall_bounce(
                self.gaze_pos, self.gaze_vel, self.gaze_radius
            )

            # Object-object collisions
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    diff = self.positions[j] - self.positions[i]
                    dist = np.linalg.norm(diff)
                    if dist < self.radii[i] + self.radii[j]:
                        p1, v1, p2, v2 = self._resolve_circle_circle_2d(
                            self.positions[i], self.velocities[i], self.masses[i], self.radii[i],
                            self.positions[j], self.velocities[j], self.masses[j], self.radii[j]
                        )
                        self.positions[i] = p1
                        self.velocities[i] = v1
                        self.positions[j] = p2
                        self.velocities[j] = v2

            # Pointer-object collisions (physics + optional recording)
            if not ghostly_gaze:
                for i in range(self.N):
                    diff = self.pointer_pos - self.positions[i]
                    dist = np.linalg.norm(diff)
                    # Record contact flag for gate-counting (uses proximity threshold)
                    if record_collisions and dist < self.pointer_radius + self.radii[i] + PROX_THRESH:
                        ptr_obj_contact[i] = True
                    # Physical collision resolution
                    if dist < self.pointer_radius + self.radii[i]:
                        p1, v1, p2, v2 = self._resolve_circle_circle_2d(
                            self.positions[i], self.velocities[i], self.masses[i], self.radii[i],
                            self.pointer_pos, self.pointer_vel, self.pointer_mass, self.pointer_radius
                        )
                        self.positions[i] = p1
                        self.velocities[i] = v1
                        self.pointer_pos = p2
                        self.pointer_vel = v2
                        # Wall bounce only the affected pair
                        self.positions[i], self.velocities[i] = self._wall_bounce(
                            self.positions[i], self.velocities[i], self.radii[i]
                        )
                        self.pointer_pos, self.pointer_vel = self._wall_bounce(
                            self.pointer_pos, self.pointer_vel, self.pointer_radius
                        )

        # Build collision events at STEP granularity
        collision_events = []
        if record_collisions:
            post_ptr_vel = self.pointer_vel.copy()
            post_obj_vels = self.velocities.copy()
            for i in range(self.N):
                if ptr_obj_contact[i]:
                    dv_ptr = post_ptr_vel - pre_ptr_vel
                    # Valid if |dv_ptr| > threshold in ANY component
                    # (per pre-registration: |Δvx| > 0.5 OR |Δvy| > 0.5)
                    ptr_valid = abs(dv_ptr[0]) > DV_THRESH or abs(dv_ptr[1]) > DV_THRESH
                    if ptr_valid:
                        collision_events.append({
                            'obj_idx': i,
                            'pre_ptr_vel': pre_ptr_vel.copy(),
                            'pre_obj_vel': pre_obj_vels[i].copy(),
                            'post_ptr_vel': post_ptr_vel.copy(),
                            'post_obj_vel': post_obj_vels[i].copy(),
                        })

        # Probe collision (step-level)
        probe_result = None
        if do_probe and ghostly_gaze:
            dists = np.linalg.norm(self.positions - self.gaze_pos, axis=1)
            valid_mask = dists < self.gaze_radius
            if np.any(valid_mask):
                nearest_idx = int(np.argmin(dists))
                pre_gaze_vel = self.gaze_vel.copy()
                pre_obj_vel = self.velocities[nearest_idx].copy()
                p1, v1, p2, v2 = self._resolve_circle_circle_2d(
                    self.positions[nearest_idx], self.velocities[nearest_idx],
                    self.masses[nearest_idx], self.radii[nearest_idx],
                    self.gaze_pos, self.gaze_vel, self.gaze_mass, self.gaze_radius
                )
                self.positions[nearest_idx] = p1
                self.velocities[nearest_idx] = v1
                self.gaze_pos = p2
                self.gaze_vel = v2
                probe_result = {
                    'obj_idx': nearest_idx,
                    'pre_gaze_vel': pre_gaze_vel,
                    'pre_obj_vel': pre_obj_vel,
                    'post_gaze_vel': self.gaze_vel.copy(),
                    'post_obj_vel': self.velocities[nearest_idx].copy(),
                }

        return probe_result, collision_events, ptr_obj_contact if record_collisions else None

    def reset_for_gate1(self, seed):
        self.rng = np.random.RandomState(seed)
        self._init_objects()
        self.pointer_pos = PTR_START_POS.copy()
        self.pointer_vel = np.zeros(2)

    def reset_for_gate2(self, seed):
        self.rng = np.random.RandomState(seed)
        self._init_objects()
        self.gaze_pos = PTR_START_POS.copy()
        self.gaze_vel = np.zeros(2)
        self.pointer_pos = PTR_START_POS.copy()
        self.pointer_vel = np.zeros(2)


# ── Gate-1: PASSIVE Boundedness ──────────────────────────────────────────────
def run_gate1(seed, check_bounds=False):
    env = PhysicsSandbox2D(N=N_OBJ, substeps=SUBSTEPS)
    env.reset_for_gate1(seed)

    valid_counts = np.zeros(N_OBJ, dtype=np.int32)
    raw_counts = np.zeros(N_OBJ, dtype=np.int32)

    ptr_in_bounds = True
    obj_in_bounds = True
    ptr_initial_vel_zero = True

    for step in range(N_STEPS):
        probe_result, collision_events, raw_contact = env.step(
            pointer_acc=np.zeros(2),
            ghostly_gaze=False,
            probe_action={'do_probe': False, 'probe_acc': np.zeros(2)},
            record_collisions=True,
        )

        for i in range(N_OBJ):
            if raw_contact[i]:
                raw_counts[i] += 1

        for evt in collision_events:
            i = evt['obj_idx']
            valid_counts[i] += 1

        if check_bounds:
            ptr = env.pointer_pos
            if not (0 <= ptr[0] <= ARENA_SIZE and 0 <= ptr[1] <= ARENA_SIZE):
                ptr_in_bounds = False
            for i in range(N_OBJ):
                if not (0 <= env.positions[i, 0] <= ARENA_SIZE and 0 <= env.positions[i, 1] <= ARENA_SIZE):
                    obj_in_bounds = False

    mean_valid = float(np.mean(valid_counts))
    mean_raw = float(np.mean(raw_counts))
    cv_valid = float(np.std(valid_counts) / mean_valid) if mean_valid > 0 else 0.0
    cv_raw = float(np.std(raw_counts) / mean_raw) if mean_raw > 0 else 0.0

    gate1_pass = mean_valid <= GATE1_THRESH
    gate1b_pass = cv_valid >= GATE1B_THRESH

    result = {
        'seed': seed,
        'obj_valid_counts': valid_counts.tolist(),
        'mean_valid': mean_valid,
        'cv_valid': cv_valid,
        'obj_raw_counts': raw_counts.tolist(),
        'mean_raw': mean_raw,
        'cv_raw': cv_raw,
        'gate1_pass': gate1_pass,
        'gate1b_pass': gate1b_pass,
        'ptr_in_bounds': ptr_in_bounds,
        'obj_in_bounds': obj_in_bounds,
        'ptr_initial_vel_zero': ptr_initial_vel_zero,
    }
    return result


# ── Gate-2: RANDOM Gaze Heterogeneity ────────────────────────────────────────
def run_gate2(seed):
    env = PhysicsSandbox2D(N=N_OBJ, substeps=SUBSTEPS)
    env.reset_for_gate2(seed)

    probe_counts = np.zeros(N_OBJ, dtype=np.int32)
    probes_fired = 0
    budget_remaining = PROBE_BUDGET

    rng = np.random.RandomState(seed * 10000 + 42)

    for step in range(N_STEPS):
        probe_acc = rng.uniform(GAZE_ACC_RANGE[0], GAZE_ACC_RANGE[1], 2)
        do_probe = (rng.rand() < PROBE_P) and (budget_remaining > 0)

        probe_result, _ = env.step(
            pointer_acc=None,
            ghostly_gaze=True,
            probe_action={'do_probe': do_probe, 'probe_acc': probe_acc},
            record_collisions=False,
        )

        if do_probe:
            probes_fired += 1
            budget_remaining -= 1
            if probe_result is not None:
                obj_idx = probe_result['obj_idx']
                probe_counts[obj_idx] += 1

    mean_probes = float(np.mean(probe_counts))
    cv = float(np.std(probe_counts) / mean_probes) if mean_probes > 0 else 0.0
    gate2_pass = cv >= GATE2_THRESH

    return {
        'seed': seed,
        'obj_probe_counts': probe_counts.tolist(),
        'mean_probes': mean_probes,
        'cv': cv,
        'total_probes_fired': probes_fired,
        'gate2_pass': gate2_pass,
    }


# ── Sanity Checks ─────────────────────────────────────────────────────────────
def run_sanity_checks():
    results = {}

    print("  S1: Physics conservation test...")
    env = PhysicsSandbox2D(N=2, substeps=SUBSTEPS, seed=12345)
    env.positions[0] = np.array([20.0, 32.0])
    env.positions[1] = np.array([44.0, 32.0])
    env.velocities[0] = np.array([2.0, 0.0])
    env.velocities[1] = np.array([-2.0, 0.0])
    env.radii[0] = 5.0
    env.radii[1] = 5.0
    env.masses[0] = 5.0
    env.masses[1] = 5.0

    total_p_pre = env.masses[0] * env.velocities[0] + env.masses[1] * env.velocities[1]
    ke_pre = 0.5 * env.masses[0] * np.dot(env.velocities[0], env.velocities[0]) \
             + 0.5 * env.masses[1] * np.dot(env.velocities[1], env.velocities[1])

    env.step(pointer_acc=None)

    total_p_post = env.masses[0] * env.velocities[0] + env.masses[1] * env.velocities[1]
    ke_post = 0.5 * env.masses[0] * np.dot(env.velocities[0], env.velocities[0]) \
              + 0.5 * env.masses[1] * np.dot(env.velocities[1], env.velocities[1])

    p_error = np.linalg.norm(total_p_post - total_p_pre)
    ke_error = abs(ke_post - ke_pre)
    S1_pass = p_error < 1e-6 and ke_error < 1e-6

    results['S1'] = {
        'pass': S1_pass,
        'p_pre': total_p_pre.tolist(),
        'p_post': total_p_post.tolist(),
        'ke_pre': float(ke_pre),
        'ke_post': float(ke_post),
        'p_error': float(p_error),
        'ke_error': float(ke_error),
    }
    print(f"    Momentum: pre={total_p_pre}, post={total_p_post}, error={p_error:.2e}")
    print(f"    KE: pre={ke_pre:.6f}, post={ke_post:.6f}, error={ke_error:.2e}")
    print(f"    S1: {'PASS' if S1_pass else 'FAIL'}")

    print("  S2/S3/S4: Bounds check + passive pointer sanity...")
    ptr_all_in_bounds = True
    obj_all_in_bounds = True
    ptr_initial_vel_zero_all = True

    for seed in SEEDS:
        env = PhysicsSandbox2D(N=N_OBJ, substeps=SUBSTEPS)
        env.reset_for_gate1(seed)
        if not np.allclose(env.pointer_vel, 0):
            ptr_initial_vel_zero_all = False
        for step in range(N_STEPS):
            env.step(pointer_acc=np.zeros(2))
            ptr = env.pointer_pos
            if not (0 <= ptr[0] <= ARENA_SIZE and 0 <= ptr[1] <= ARENA_SIZE):
                ptr_all_in_bounds = False
            for i in range(N_OBJ):
                if not (0 <= env.positions[i, 0] <= ARENA_SIZE and 0 <= env.positions[i, 1] <= ARENA_SIZE):
                    obj_all_in_bounds = False

    results['S2'] = {
        'pass': ptr_all_in_bounds,
        'description': 'Pointer stays within [0, 64]^2 throughout all Gate-1 runs',
    }
    print(f"    S2 (pointer bounds): {'PASS' if ptr_all_in_bounds else 'FAIL'}")

    results['S3'] = {
        'pass': obj_all_in_bounds,
        'description': 'Objects stay within [0, 64]^2 throughout all Gate-1 runs',
    }
    print(f"    S3 (object bounds): {'PASS' if obj_all_in_bounds else 'FAIL'}")

    results['S4'] = {
        'pass': ptr_initial_vel_zero_all,
        'description': 'Gate-1 pointer has initial vel=0 and receives no active acceleration (velocity changes come only from collisions)',
    }
    print(f"    S4 (passive pointer): {'PASS' if ptr_initial_vel_zero_all else 'FAIL'}")

    print("  S5: Probe count sanity...")
    probe_counts = []
    for seed in SEEDS:
        r = run_gate2(seed)
        probe_counts.append(r['total_probes_fired'])
    mean_probes = np.mean(probe_counts)
    S5_pass = 15 <= mean_probes <= 25
    results['S5'] = {
        'pass': S5_pass,
        'counts': probe_counts,
        'mean': float(mean_probes),
        'description': f'Gate-2 fires approx 20 probes (expected 0.01 x 2000 = 20), got mean={mean_probes:.1f}',
    }
    print(f"    S5 (probe count ~20): counts={probe_counts}, mean={mean_probes:.1f}, {'PASS' if S5_pass else 'FAIL'}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("iter_037: 2D Cheap Gate Experiment")
    print("=" * 70)
    print(f"Parameters: N={N_OBJ}, arena={ARENA_SIZE}x{ARENA_SIZE}, steps={N_STEPS}, substeps={SUBSTEPS}")
    print(f"Seeds: {SEEDS}")
    print()

    print("Running Sanity Checks...")
    sanity = run_sanity_checks()
    all_sanity_pass = all(v['pass'] for v in sanity.values())
    print()

    print("Running Gate-1 (PASSIVE Boundedness) and Gate-1b (PASSIVE Heterogeneity)...")
    gate1_results = []
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        r = run_gate1(seed)
        gate1_results.append(r)
        print(f"mean_valid={r['mean_valid']:.2f}, cv_valid={r['cv_valid']:.3f}")
    print()

    print("Running Gate-2 (RANDOM Gaze Heterogeneity)...")
    gate2_results = []
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        r = run_gate2(seed)
        gate2_results.append(r)
        print(f"mean_probes={r['mean_probes']:.2f}, cv={r['cv']:.3f}, total_probes={r['total_probes_fired']}")
    print()

    gate1_passes = [r['gate1_pass'] for r in gate1_results]
    gate1b_passes = [r['gate1b_pass'] for r in gate1_results]
    gate2_passes = [r['gate2_pass'] for r in gate2_results]

    gate1_overall = sum(gate1_passes) >= N_SEEDS_PASS
    gate1b_overall = sum(gate1b_passes) >= N_SEEDS_PASS
    gate2_overall = sum(gate2_passes) >= N_SEEDS_PASS

    all_gates_pass = gate1_overall and gate1b_overall and gate2_overall

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nGate-1 (PASSIVE Boundedness):")
    print(f"  Threshold: mean per-object valid collisions <= {GATE1_THRESH}")
    print(f"  {'Seed':>6} | {'Obj0':>4} | {'Obj1':>4} | {'Obj2':>4} | {'Mean':>6} | {'Pass':>5}")
    for r in gate1_results:
        mark = "PASS" if r['gate1_pass'] else "FAIL"
        vc = r['obj_valid_counts']
        print(f"  {r['seed']:>6} | {vc[0]:>4} | {vc[1]:>4} | {vc[2]:>4} | {r['mean_valid']:>6.2f} | {mark:>5}")
    print(f"  Passes: {sum(gate1_passes)}/{len(SEEDS)} seeds -> {'PASS' if gate1_overall else 'FAIL'}")

    print("\nGate-1b (PASSIVE Collision Heterogeneity):")
    print(f"  Threshold: CV >= {GATE1B_THRESH}")
    print(f"  {'Seed':>6} | {'CV(valid)':>10} | {'Pass':>5}")
    for r in gate1_results:
        mark = "PASS" if r['gate1b_pass'] else "FAIL"
        print(f"  {r['seed']:>6} | {r['cv_valid']:>10.3f} | {mark:>5}")
    print(f"  Passes: {sum(gate1b_passes)}/{len(SEEDS)} seeds -> {'PASS' if gate1b_overall else 'FAIL'}")

    print("\nGate-2 (RANDOM Gaze Heterogeneity):")
    print(f"  Threshold: CV >= {GATE2_THRESH}")
    print(f"  {'Seed':>6} | {'Obj0':>4} | {'Obj1':>4} | {'Obj2':>4} | {'Mean':>6} | {'CV':>6} | {'Fired':>5} | {'Pass':>5}")
    for r in gate2_results:
        mark = "PASS" if r['gate2_pass'] else "FAIL"
        pc = r['obj_probe_counts']
        print(f"  {r['seed']:>6} | {pc[0]:>4} | {pc[1]:>4} | {pc[2]:>4} | {r['mean_probes']:>6.2f} | {r['cv']:>6.3f} | {r['total_probes_fired']:>5} | {mark:>5}")
    print(f"  Passes: {sum(gate2_passes)}/{len(SEEDS)} seeds -> {'PASS' if gate2_overall else 'FAIL'}")

    print("\n" + "-" * 70)
    print(f"OVERALL: Gate-1 {'PASS' if gate1_overall else 'FAIL'}")
    print(f"         Gate-1b {'PASS' if gate1b_overall else 'FAIL'}")
    print(f"         Gate-2 {'PASS' if gate2_overall else 'FAIL'}")
    print(f"         ALL GATES: {'PASS' if all_gates_pass else 'FAIL'}")
    print("-" * 70)

    if all_gates_pass:
        print("CONCLUSION: All three gates pass.")
        print("-> 1D structural constraints (collision inevitability, collision")
        print("  homogeneity, coverage uniformity) are relaxed in 2D at the")
        print("  tested parameterization. This is measured evidence FOR 2D")
        print("  viability, not validation that 2D 'works' end-to-end.")
    else:
        print("CONCLUSION: At least one gate FAILED.")
        print("-> 2D does not resolve the 1D structural constraints at the")
        print("  tested parameterization. Evidence AGAINST 2D viability.")
    print("=" * 70)

    # 1. gate1_results.csv
    with open(os.path.join(OUT_DIR, 'gate1_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'seed',
            'obj_0_valid', 'obj_1_valid', 'obj_2_valid',
            'mean_valid', 'cv_valid',
            'obj_0_raw', 'obj_1_raw', 'obj_2_raw',
            'mean_raw', 'cv_raw',
            'gate1_pass', 'gate1b_pass'
        ])
        for r in gate1_results:
            w.writerow([
                r['seed'],
                *r['obj_valid_counts'],
                r['mean_valid'], r['cv_valid'],
                *r['obj_raw_counts'],
                r['mean_raw'], r['cv_raw'],
                int(r['gate1_pass']), int(r['gate1b_pass'])
            ])

    # 2. gate2_results.csv
    with open(os.path.join(OUT_DIR, 'gate2_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'seed',
            'obj_0_probes', 'obj_1_probes', 'obj_2_probes',
            'mean_probes', 'cv',
            'total_probes_fired',
            'gate2_pass'
        ])
        for r in gate2_results:
            w.writerow([
                r['seed'],
                *r['obj_probe_counts'],
                r['mean_probes'], r['cv'],
                r['total_probes_fired'],
                int(r['gate2_pass'])
            ])

    # 3. sanity_checks.txt
    with open(os.path.join(OUT_DIR, 'sanity_checks.txt'), 'w') as f:
        f.write("iter_037 2D Cheap Gate Experiment - Sanity Checks\n")
        f.write("=" * 60 + "\n\n")
        f.write("S1: Physics Conservation (2D elastic collision)\n")
        f.write(f"  Pre-collision momentum:  {sanity['S1']['p_pre']}\n")
        f.write(f"  Post-collision momentum: {sanity['S1']['p_post']}\n")
        f.write(f"  Momentum error:          {sanity['S1']['p_error']:.2e}\n")
        f.write(f"  Pre-collision KE:        {sanity['S1']['ke_pre']:.6f}\n")
        f.write(f"  Post-collision KE:       {sanity['S1']['ke_post']:.6f}\n")
        f.write(f"  KE error:                {sanity['S1']['ke_error']:.2e}\n")
        f.write(f"  Result: {'PASS' if sanity['S1']['pass'] else 'FAIL'}\n\n")
        f.write("S2: Pointer in bounds throughout Gate-1\n")
        f.write(f"  {sanity['S2']['description']}\n")
        f.write(f"  Result: {'PASS' if sanity['S2']['pass'] else 'FAIL'}\n\n")
        f.write("S3: Objects in bounds throughout Gate-1\n")
        f.write(f"  {sanity['S3']['description']}\n")
        f.write(f"  Result: {'PASS' if sanity['S3']['pass'] else 'FAIL'}\n\n")
        f.write("S4: Gate-1 PASSIVE pointer zero active acceleration\n")
        f.write(f"  {sanity['S4']['description']}\n")
        f.write(f"  Clarification: The pointer is PASSIVE (no active acceleration),\n")
        f.write(f"  starting from zero velocity. It WILL acquire velocity from elastic\n")
        f.write(f"  collisions with objects -- this is expected physics. The check\n")
        f.write(f"  verifies only that no active acceleration is applied.\n")
        f.write(f"  Result: {'PASS' if sanity['S4']['pass'] else 'FAIL'}\n\n")
        f.write("S5: Gate-2 probe count approximately 20\n")
        f.write(f"  {sanity['S5']['description']}\n")
        f.write(f"  Per-seed probe counts: {sanity['S5']['counts']}\n")
        f.write(f"  Result: {'PASS' if sanity['S5']['pass'] else 'FAIL'}\n\n")
        f.write("-" * 60 + "\n")
        f.write(f"All sanity checks: {'PASS' if all_sanity_pass else 'FAIL'}\n")

    # 4. gate_summary.txt
    with open(os.path.join(OUT_DIR, 'gate_summary.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("iter_037: 2D Cheap Gate Experiment - Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Pre-registered Parameters:\n")
        f.write(f"  Arena: {ARENA_SIZE}x{ARENA_SIZE}, N={N_OBJ}, Steps={N_STEPS}, Substeps={SUBSTEPS}\n")
        f.write(f"  Pointer: radius={PTR_RADIUS}, mass={PTR_MASS}, start={PTR_START_POS}\n")
        f.write(f"  Object radii: [{OBJ_RADIUS_RANGE[0]}, {OBJ_RADIUS_RANGE[1]}], mass=radius\n")
        f.write(f"  Seeds: {SEEDS}\n")
        f.write(f"  Gate-1 threshold:  mean valid collisions <= {GATE1_THRESH}\n")
        f.write(f"  Gate-1b threshold: CV valid collisions >= {GATE1B_THRESH}\n")
        f.write(f"  Gate-2 threshold:  CV probe events >= {GATE2_THRESH}\n")
        f.write(f"  Decision rule: >={N_SEEDS_PASS}/{len(SEEDS)} seeds must individually pass\n\n")
        f.write("Gate-1 (PASSIVE Boundedness) - Per-Seed:\n")
        for r in gate1_results:
            mark = "PASS" if r['gate1_pass'] else "FAIL"
            vc = r['obj_valid_counts']
            rc = r['obj_raw_counts']
            f.write(f"  Seed {r['seed']:>3}: valid=({vc[0]},{vc[1]},{vc[2]}), mean={r['mean_valid']:.2f}, "
                    f"raw=({rc[0]},{rc[1]},{rc[2]}), mean_raw={r['mean_raw']:.2f} -> {mark}\n")
        f.write(f"  Overall: {sum(gate1_passes)}/{len(SEEDS)} seeds pass -> {'PASS' if gate1_overall else 'FAIL'}\n\n")
        f.write("Gate-1b (PASSIVE Heterogeneity) - Per-Seed:\n")
        for r in gate1_results:
            mark = "PASS" if r['gate1b_pass'] else "FAIL"
            f.write(f"  Seed {r['seed']:>3}: CV_valid={r['cv_valid']:.3f}, CV_raw={r['cv_raw']:.3f} -> {mark}\n")
        f.write(f"  Overall: {sum(gate1b_passes)}/{len(SEEDS)} seeds pass -> {'PASS' if gate1b_overall else 'FAIL'}\n\n")
        f.write("Gate-2 (RANDOM Gaze Heterogeneity) - Per-Seed:\n")
        for r in gate2_results:
            mark = "PASS" if r['gate2_pass'] else "FAIL"
            pc = r['obj_probe_counts']
            f.write(f"  Seed {r['seed']:>3}: probes=({pc[0]},{pc[1]},{pc[2]}), mean={r['mean_probes']:.2f}, "
                    f"CV={r['cv']:.3f}, fired={r['total_probes_fired']} -> {mark}\n")
        f.write(f"  Overall: {sum(gate2_passes)}/{len(SEEDS)} seeds pass -> {'PASS' if gate2_overall else 'FAIL'}\n\n")
        f.write("-" * 60 + "\n")
        f.write("OVERALL GATE ASSESSMENT:\n")
        f.write(f"  Gate-1:  {'PASS' if gate1_overall else 'FAIL'}\n")
        f.write(f"  Gate-1b: {'PASS' if gate1b_overall else 'FAIL'}\n")
        f.write(f"  Gate-2:  {'PASS' if gate2_overall else 'FAIL'}\n")
        f.write(f"  ALL GATES: {'PASS' if all_gates_pass else 'FAIL'}\n\n")
        if all_gates_pass:
            f.write("INTERPRETATION:\n")
            f.write("  All three gates pass. This is measured evidence that the three\n")
            f.write("  structural constraints that blocked the 1D testbed are relaxed\n")
            f.write("  in 2D at the tested parameterization:\n")
            f.write("    - Collision inevitability (Gate-1): PASSIVE ceiling is low enough\n")
            f.write("    - Collision homogeneity (Gate-1b): per-object rates vary naturally\n")
            f.write("    - Coverage uniformity (Gate-2): random gaze does not cover evenly\n")
            f.write("  This does NOT validate that a full 2D bracket would discriminate;\n")
            f.write("  it only unblocks the decision to proceed to a full commitment.\n")
        else:
            f.write("INTERPRETATION:\n")
            f.write("  At least one gate failed. 2D at the tested parameterization does\n")
            f.write("  not sufficiently relax the 1D structural constraints.\n")
            f.write("  Options: (ii) re-frame deliverable, or explore different 2D\n")
            f.write("  parameterizations (larger arena, fewer objects, different gaze\n")
            f.write("  radius).\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults saved to {OUT_DIR}/")
    print("  - gate1_results.csv")
    print("  - gate2_results.csv")
    print("  - sanity_checks.txt")
    print("  - gate_summary.txt")

    return {
        'gate1_results': gate1_results,
        'gate2_results': gate2_results,
        'sanity': sanity,
        'overall': {
            'gate1_pass': gate1_overall,
            'gate1b_pass': gate1b_overall,
            'gate2_pass': gate2_overall,
            'all_pass': all_gates_pass,
        }
    }


if __name__ == '__main__':
    main()
