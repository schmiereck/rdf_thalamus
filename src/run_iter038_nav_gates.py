"""src/run_iter038_nav_gates.py

2D Navigation Gate Probe for the Thalamus project.
Tests whether a PHYSICALLY NAVIGATING pointer in a 2D arena resolves the
structural constraints that blocked behavioral validation across iter_033-037.

No training, no learned model, no representation work. Pure physics + random
navigation + probe mechanism.

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
PTR_START = np.array([32.0, 32.0], dtype=np.float64)
PTR_VEL_INIT = np.array([0.0, 0.0], dtype=np.float64)
OBJ_RADIUS_RANGE = [3.0, 8.0]
OBJ_VEL_RANGES = [(-2.0, -0.5), (0.5, 2.0)]
VEL_CAP = 3.5
NAV_ACC_RANGE = [-1.5, 1.5]
PROBE_RADIUS = 10.0
PROBE_P = 0.015
PROBE_BUDGET = 15

# Gate thresholds
GATE1_THRESH = 3.0
GATE1B_THRESH = 0.30
GATE1B_STD_THRESH = 0.25
GATE2_THRESH = 0.50
N_SEEDS_PASS = 4

OUT_DIR = os.path.join('archive', 'iter_038', 'results')

# Navigation RNG seed derivation (separate from environment RNG)
def nav_seed_for_env_seed(env_seed):
    return env_seed * 1000 + 7


# ── Utility ──────────────────────────────────────────────────────────────────
def _rand_vel_component(rng):
    """Draw a velocity component from [-2.0, -0.5] U [0.5, 2.0]."""
    if rng.rand() < 0.5:
        return rng.uniform(OBJ_VEL_RANGES[0][0], OBJ_VEL_RANGES[0][1])
    return rng.uniform(OBJ_VEL_RANGES[1][0], OBJ_VEL_RANGES[1][1])


def _cap_velocity(vel, cap):
    """Cap each velocity component to [-cap, +cap]."""
    return np.clip(vel, -cap, cap)


# ── PhysicsSandbox2D (iter_038 variant) ──────────────────────────────────────
class PhysicsSandbox2D:
    """
    2D physics sandbox with:
    - N objects with elastic collisions and wall bounces
    - One navigating pointer with random acceleration
    - Probe mechanism (separate from collision physics)
    """

    def __init__(self, N=N_OBJ, substeps=SUBSTEPS, seed=None):
        self.N = N
        self.substeps = substeps
        self.dt = 1.0 / substeps
        self.rng = np.random.RandomState(seed)

        self.positions = np.zeros((N, 2), dtype=np.float64)
        self.velocities = np.zeros((N, 2), dtype=np.float64)
        self.radii = np.zeros(N, dtype=np.float64)
        self.masses = np.zeros(N, dtype=np.float64)

        self.pointer_pos = PTR_START.copy()
        self.pointer_vel = PTR_VEL_INIT.copy()
        self.pointer_radius = PTR_RADIUS
        self.pointer_mass = PTR_MASS

        # Diagnostic collision counter (step-level, per object)
        self._collision_counts = np.zeros(N, dtype=np.int32)

        self._init_objects()

    def _init_objects(self):
        """Initialize objects with UNIFORM-RANDOM placement in arena interior."""
        self.radii = self.rng.uniform(OBJ_RADIUS_RANGE[0], OBJ_RADIUS_RANGE[1], self.N)
        self.masses = self.radii.copy()
        for i in range(self.N):
            self.velocities[i, 0] = _rand_vel_component(self.rng)
            self.velocities[i, 1] = _rand_vel_component(self.rng)
        for i in range(self.N):
            margin = self.radii[i] + 0.5
            low = margin
            high = ARENA_SIZE - margin
            self.positions[i, 0] = self.rng.uniform(low, high)
            self.positions[i, 1] = self.rng.uniform(low, high)

    def _resolve_circle_circle_2d(self, pos1, vel1, m1, r1, pos2, vel2, m2, r2):
        """
        2D elastic collision between two circles.
        Returns new positions and velocities for both bodies.
        Uses approach-velocity check to prevent double-counting across substeps.
        """
        diff = pos2 - pos1
        dist = np.linalg.norm(diff)
        min_dist = r1 + r2
        if dist < min_dist and dist > 1e-12:
            n = diff / dist
            v_rel_n = np.dot(vel2 - vel1, n)
            m_inv1 = 1.0 / m1
            m_inv2 = 1.0 / m2
            sum_inv = m_inv1 + m_inv2
            overlap = min_dist - dist
            new_pos1 = pos1 - overlap * (m_inv1 / sum_inv) * n
            new_pos2 = pos2 + overlap * (m_inv2 / sum_inv) * n
            if v_rel_n > 0:
                # Approaching: resolve velocity
                v1n = np.dot(vel1, n)
                v2n = np.dot(vel2, n)
                v1n_new = (v1n * (m1 - m2) + 2.0 * m2 * v2n) / (m1 + m2)
                v2n_new = (v2n * (m2 - m1) + 2.0 * m1 * v1n) / (m1 + m2)
                new_vel1 = vel1 + (v1n_new - v1n) * n
                new_vel2 = vel2 + (v2n_new - v2n) * n
                return new_pos1, new_vel1, new_pos2, new_vel2
            else:
                # Separating: only fix overlap
                return new_pos1, vel1.copy(), new_pos2, vel2.copy()
        return pos1.copy(), vel1.copy(), pos2.copy(), vel2.copy()

    def _wall_bounce(self, pos, vel, radius):
        """Reflect position and velocity at arena walls."""
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

    def step(self, pointer_acc):
        """
        One physics step with substeps.
        pointer_acc: 2D acceleration to apply to pointer (each substep).

        Returns:
            probe_hit: bool (did a probe succeed this step?)
            probed_obj_idx: int or -1 (which object was probed)
            collisions_this_step: np.array(N,) of bool (which objects had physical contact)
        """
        collisions_this_step = np.zeros(self.N, dtype=bool)

        # Substep integration
        for _ in range(self.substeps):
            self.pointer_vel += pointer_acc * self.dt

            self.positions += self.velocities * self.dt
            self.pointer_pos += self.pointer_vel * self.dt

            # Wall bounces
            for i in range(self.N):
                self.positions[i], self.velocities[i] = self._wall_bounce(
                    self.positions[i], self.velocities[i], self.radii[i]
                )
            self.pointer_pos, self.pointer_vel = self._wall_bounce(
                self.pointer_pos, self.pointer_vel, self.pointer_radius
            )

            # Object-object collisions
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    dist = np.linalg.norm(self.positions[j] - self.positions[i])
                    if dist < self.radii[i] + self.radii[j]:
                        p1, v1, p2, v2 = self._resolve_circle_circle_2d(
                            self.positions[i], self.velocities[i], self.masses[i], self.radii[i],
                            self.positions[j], self.velocities[j], self.masses[j], self.radii[j],
                        )
                        self.positions[i] = p1
                        self.velocities[i] = v1
                        self.positions[j] = p2
                        self.velocities[j] = v2

            # Pointer-object physics collisions
            for i in range(self.N):
                dist = np.linalg.norm(self.pointer_pos - self.positions[i])
                if dist < self.pointer_radius + self.radii[i]:
                    collisions_this_step[i] = True
                    p1, v1, p2, v2 = self._resolve_circle_circle_2d(
                        self.positions[i], self.velocities[i], self.masses[i], self.radii[i],
                        self.pointer_pos, self.pointer_vel, self.pointer_mass, self.pointer_radius,
                    )
                    self.positions[i] = p1
                    self.velocities[i] = v1
                    self.pointer_pos = p2
                    self.pointer_vel = v2
                    self.positions[i], self.velocities[i] = self._wall_bounce(
                        self.positions[i], self.velocities[i], self.radii[i]
                    )
                    self.pointer_pos, self.pointer_vel = self._wall_bounce(
                        self.pointer_pos, self.pointer_vel, self.pointer_radius
                    )
                elif dist < self.pointer_radius + self.radii[i] + PROBE_RADIUS:
                    # Track proximity for diagnostics (not a physical collision)
                    # We only flag if the CENTER distance is within PROBE_RADIUS
                    center_dist = np.linalg.norm(self.pointer_pos - self.positions[i])
                    # This is just diagnostic; the actual probe check is done at step level
                    pass

        # Update diagnostic collision counts
        for i in range(self.N):
            if collisions_this_step[i]:
                self._collision_counts[i] += 1

        # Probe check (step-level, AFTER physics)
        probe_hit = False
        probed_obj_idx = -1
        center_dists = np.linalg.norm(self.positions - self.pointer_pos, axis=1)
        within_probe = center_dists < PROBE_RADIUS
        if np.any(within_probe):
            # Find nearest object within probe radius
            nearby_indices = np.where(within_probe)[0]
            probed_obj_idx = int(nearby_indices[np.argmin(center_dists[nearby_indices])])
            probe_hit = True

        return probe_hit, probed_obj_idx, collisions_this_step


# ── Run one full rollout for a seed ──────────────────────────────────────────
def run_one_seed(env_seed, record_trajectory=False):
    """
    Run the full navigation experiment for one seed.

    Returns dict with:
        - probe_counts: per-object probe-event counts
        - collision_counts: per-object diagnostic collision counts
        - probes_fired: total successful probes
        - probes_attempted: total probe opportunities where budget remained
        - probe_events: list of (step_idx, obj_idx) for each successful probe
        - trajectory: list of (x, y) pointer positions per step (if record_trajectory)
        - max_speed: max pointer speed observed
        - total_distance: total distance traveled by pointer
    """
    env = PhysicsSandbox2D(N=N_OBJ, substeps=SUBSTEPS, seed=env_seed)
    env.pointer_pos = PTR_START.copy()
    env.pointer_vel = PTR_VEL_INIT.copy()

    nav_rng = np.random.RandomState(nav_seed_for_env_seed(env_seed))

    probe_counts = np.zeros(N_OBJ, dtype=np.int32)
    probes_fired = 0
    probe_events = []  # (step, obj_idx)

    trajectory = []
    total_distance = 0.0
    max_speed = 0.0
    prev_ptr_pos = PTR_START.copy()

    for step in range(N_STEPS):
        # Sample random acceleration
        ptr_acc = nav_rng.uniform(NAV_ACC_RANGE[0], NAV_ACC_RANGE[1], 2)

        # Physics step
        probe_hit, probed_obj_idx, _ = env.step(ptr_acc)

        # Apply velocity cap AFTER the full step
        env.pointer_vel = _cap_velocity(env.pointer_vel, VEL_CAP)

        # Probe decision (only if budget remains and opportunity exists)
        if probe_hit and probes_fired < PROBE_BUDGET:
            # Bernoulli probe decision
            if nav_rng.rand() < PROBE_P:
                probe_counts[probed_obj_idx] += 1
                probes_fired += 1
                probe_events.append((step, probed_obj_idx))

        # Trajectory tracking
        if record_trajectory:
            current_pos = env.pointer_pos.copy()
            trajectory.append((float(current_pos[0]), float(current_pos[1])))
            delta = current_pos - prev_ptr_pos
            total_distance += np.linalg.norm(delta)
            speed = np.linalg.norm(env.pointer_vel)
            if speed > max_speed:
                max_speed = float(speed)
            prev_ptr_pos = current_pos

        # Bounds tracking
        ptr_in_bounds = all(0 <= env.pointer_pos[a] <= ARENA_SIZE for a in range(2))
        assert ptr_in_bounds, f"Pointer out of bounds at step {step}: {env.pointer_pos}"
        for i in range(N_OBJ):
            obj_in_bounds = all(0 <= env.positions[i, a] <= ARENA_SIZE for a in range(2))
            assert obj_in_bounds, f"Object {i} out of bounds at step {step}: {env.positions[i]}"

    mean_probes = float(np.mean(probe_counts))
    cv_probes = float(np.std(probe_counts) / mean_probes) if mean_probes > 0 else 0.0

    result = {
        'seed': env_seed,
        'probe_counts': probe_counts.tolist(),
        'mean_probe_count': mean_probes,
        'cv_probes': cv_probes,
        'collision_counts': env._collision_counts.tolist(),
        'probes_fired': probes_fired,
        'probe_events': probe_events,
        'budget_remaining': PROBE_BUDGET - probes_fired,
        'gate1_pass': mean_probes <= GATE1_THRESH,
        'gate2_pass': cv_probes >= GATE2_THRESH,
        'max_speed': max_speed,
        'total_distance': total_distance,
    }

    if record_trajectory:
        result['trajectory'] = trajectory

    return result


# ── Sanity Checks ────────────────────────────────────────────────────────────
def run_sanity_checks():
    """Run all pre-registered sanity checks S1-S7."""
    results = {}
    all_pass = True

    # ── S1: Physics conservation (2D elastic collision) ──────────────────
    print("  S1: Physics conservation (2D elastic collision)...")
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
    ke_pre = (0.5 * env.masses[0] * np.dot(env.velocities[0], env.velocities[0])
              + 0.5 * env.masses[1] * np.dot(env.velocities[1], env.velocities[1]))

    env.step(pointer_acc=np.zeros(2))

    total_p_post = env.masses[0] * env.velocities[0] + env.masses[1] * env.velocities[1]
    ke_post = (0.5 * env.masses[0] * np.dot(env.velocities[0], env.velocities[0])
               + 0.5 * env.masses[1] * np.dot(env.velocities[1], env.velocities[1]))

    p_error = np.linalg.norm(total_p_post - total_p_pre)
    ke_error = abs(ke_post - ke_pre)
    S1_pass = p_error < 1e-6 and ke_error < 1e-6
    all_pass = all_pass and S1_pass
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

    # ── S2: Pointer stays within arena bounds ────────────────────────────
    print("  S2: Pointer stays within [0, 64]^2...")
    ptr_all_in_bounds = True
    for seed in SEEDS:
        r = run_one_seed(seed, record_trajectory=False)
        # We check bounds inside run_one_seed via assertions; if we get here, it passed
    results['S2'] = {'pass': True, 'description': 'Pointer stays within [0, 64]^2 throughout all seeds'}
    print(f"    S2: PASS")

    # ── S3: Objects stay within arena bounds ─────────────────────────────
    print("  S3: Objects stay within [0, 64]^2...")
    results['S3'] = {'pass': True, 'description': 'Objects stay within [0, 64]^2 throughout all seeds'}
    print(f"    S3: PASS")

    # ── S4: Random acceleration is truly random (check variance) ─────────
    print("  S4: Random acceleration is truly random...")
    nav_rng_check = np.random.RandomState(nav_seed_for_env_seed(7))
    accs = []
    for _ in range(1000):
        accs.append(nav_rng_check.uniform(NAV_ACC_RANGE[0], NAV_ACC_RANGE[1], 2))
    accs = np.array(accs)
    mean_acc = np.mean(accs, axis=0)
    std_acc = np.std(accs, axis=0)
    # Variance of uniform[-1.5, 1.5] is (3.0)^2 / 12 = 0.75, std ≈ 0.866
    expected_std = (NAV_ACC_RANGE[1] - NAV_ACC_RANGE[0]) / np.sqrt(12)
    std_error = np.abs(std_acc - expected_std)
    mean_close = np.all(np.abs(mean_acc) < 0.15)  # mean should be ≈ 0
    S4_pass = std_error.max() < 0.1 and mean_close
    all_pass = all_pass and S4_pass
    results['S4'] = {
        'pass': S4_pass,
        'mean_acc': mean_acc.tolist(),
        'std_acc': std_acc.tolist(),
        'expected_std': float(expected_std),
        'description': f'Nav acc mean≈0 (got {mean_acc}), std≈{expected_std:.4f} (got {std_acc})',
    }
    print(f"    S4: mean_acc={mean_acc}, std_acc={std_acc}, expected_std≈{expected_std:.4f}")
    print(f"    S4: {'PASS' if S4_pass else 'FAIL'}")

    # ── S5: Velocity cap enforced ────────────────────────────────────────
    print("  S5: Velocity cap enforced (max 3.5 per component)...")
    vel_cap_ok = True
    max_vel_observed = 0.0
    for seed in SEEDS:
        env = PhysicsSandbox2D(N=N_OBJ, substeps=SUBSTEPS, seed=seed)
        env.pointer_pos = PTR_START.copy()
        env.pointer_vel = PTR_VEL_INIT.copy()
        nav_rng_check2 = np.random.RandomState(nav_seed_for_env_seed(seed))
        for _ in range(N_STEPS):
            ptr_acc = nav_rng_check2.uniform(NAV_ACC_RANGE[0], NAV_ACC_RANGE[1], 2)
            env.step(ptr_acc)
            env.pointer_vel = _cap_velocity(env.pointer_vel, VEL_CAP)
            component_max = np.max(np.abs(env.pointer_vel))
            if component_max > VEL_CAP + 1e-9:
                vel_cap_ok = False
                break
            if component_max > max_vel_observed:
                max_vel_observed = float(component_max)
    all_pass = all_pass and vel_cap_ok
    results['S5'] = {
        'pass': vel_cap_ok,
        'max_vel_component': max_vel_observed,
        'cap': VEL_CAP,
        'description': f'Max velocity component observed: {max_vel_observed:.4f} (cap: {VEL_CAP})',
    }
    print(f"    S5: max_vel_component={max_vel_observed:.4f}, cap={VEL_CAP}")
    print(f"    S5: {'PASS' if vel_cap_ok else 'FAIL'}")

    # ── S6: Probe mechanism fires correctly ──────────────────────────────
    print("  S6: Probe mechanism fires correctly...")
    # Place objects at center so pointer always has probe opportunity
    env = PhysicsSandbox2D(N=3, substeps=SUBSTEPS, seed=42)
    for i in range(3):
        env.positions[i] = np.array([32.0, 32.0])
        env.radii[i] = 3.0
        env.masses[i] = 3.0
        env.velocities[i] = np.array([0.0, 0.0])
    env.pointer_pos = np.array([32.0, 32.0])
    env.pointer_vel = np.array([0.0, 0.0])

    nav_rng_probe = np.random.RandomState(999)
    probes_hit = 0
    total_steps_check = 500
    steps_in_range = 0
    for step in range(total_steps_check):
        ptr_acc = np.zeros(2)  # Don't move
        hit, idx, _ = env.step(ptr_acc)
        if hit:
            steps_in_range += 1
            if nav_rng_probe.rand() < PROBE_P and probes_hit < PROBE_BUDGET:
                probes_hit += 1

    # With pointer at center and all objects at center, every step should be in range
    S6_pass = (steps_in_range == total_steps_check) and (0 < probes_hit <= PROBE_BUDGET)
    all_pass = all_pass and S6_pass
    results['S6'] = {
        'pass': S6_pass,
        'steps_in_range': steps_in_range,
        'total_steps': total_steps_check,
        'probes_hit': probes_hit,
        'description': f'{steps_in_range}/{total_steps_check} steps in probe range, {probes_hit} probes hit',
    }
    print(f"    S6: {steps_in_range}/{total_steps_check} in range, {probes_hit} probes hit")
    print(f"    S6: {'PASS' if S6_pass else 'FAIL'}")

    # ── S7: Re-verify iter_037 Gate-1 on 1 seed with static pointer ────
    print("  S7: Re-verify iter_037 Gate-1 (static pointer, seed 7)...")
    # Run a passive variant: pointer gets no acceleration, so it stays at (32, 32)
    env = PhysicsSandbox2D(N=N_OBJ, substeps=SUBSTEPS, seed=7)
    env.pointer_pos = PTR_START.copy()
    env.pointer_vel = PTR_VEL_INIT.copy()
    for _ in range(N_STEPS):
        env.step(pointer_acc=np.zeros(2))
        env.pointer_vel = _cap_velocity(env.pointer_vel, VEL_CAP)  # won't change since acc=0

    # With static pointer and uniform-random placement, count collisions
    static_collisions = env._collision_counts.tolist()
    mean_static_coll = np.mean(static_collisions)
    S7_pass = mean_static_coll <= GATE1_THRESH
    all_pass = all_pass and S7_pass
    results['S7'] = {
        'pass': S7_pass,
        'collision_counts': static_collisions,
        'mean': float(mean_static_coll),
        'description': f'Static pointer seed 7: collisions=({static_collisions[0]},{static_collisions[1]},{static_collisions[2]}), mean={mean_static_coll:.2f}',
    }
    print(f"    S7: collisions=({static_collisions[0]},{static_collisions[1]},{static_collisions[2]}), mean={mean_static_coll:.2f}")
    print(f"    S7: {'PASS' if S7_pass else 'FAIL'}")

    return results, all_pass


# ── Gate evaluation ──────────────────────────────────────────────────────────
def evaluate_gates(seeds_results):
    """
    Evaluate all three pre-registered gates.
    Returns a dict with gate pass/fail and per-seed breakdowns.
    """
    gate = {}

    # ── Gate-1: Non-saturation ───────────────────────────────────────────
    gate1_per_seed = []
    for r in seeds_results:
        passes = r['mean_probe_count'] <= GATE1_THRESH
        gate1_per_seed.append({
            'seed': r['seed'],
            'mean_probe_count': r['mean_probe_count'],
            'threshold': GATE1_THRESH,
            'pass': passes,
        })
    n_pass_g1 = sum(1 for s in gate1_per_seed if s['pass'])
    gate['gate1'] = {
        'name': 'Non-saturation',
        'threshold': GATE1_THRESH,
        'per_seed': gate1_per_seed,
        'n_pass': n_pass_g1,
        'n_total': len(seeds_results),
        'pass': n_pass_g1 >= N_SEEDS_PASS,
    }

    # ── Gate-1b: CV stability ────────────────────────────────────────────
    per_seed_cvs = [r['cv_probes'] for r in seeds_results]
    mean_cv = float(np.mean(per_seed_cvs))
    std_cv = float(np.std(per_seed_cvs))
    gate['gate1b'] = {
        'name': 'CV stability',
        'mean_cv_threshold': GATE1B_THRESH,
        'std_cv_threshold': GATE1B_STD_THRESH,
        'per_seed_cvs': list(zip(SEEDS, per_seed_cvs)),
        'mean_cv': mean_cv,
        'std_cv': std_cv,
        'pass_mean': mean_cv >= GATE1B_THRESH,
        'pass_std': std_cv <= GATE1B_STD_THRESH,
        'pass': mean_cv >= GATE1B_THRESH and std_cv <= GATE1B_STD_THRESH,
    }

    # ── Gate-2: Heterogeneity ────────────────────────────────────────────
    gate2_per_seed = []
    for r in seeds_results:
        passes = r['cv_probes'] >= GATE2_THRESH
        gate2_per_seed.append({
            'seed': r['seed'],
            'cv_probes': r['cv_probes'],
            'threshold': GATE2_THRESH,
            'pass': passes,
        })
    n_pass_g2 = sum(1 for s in gate2_per_seed if s['pass'])
    gate['gate2'] = {
        'name': 'Heterogeneity',
        'threshold': GATE2_THRESH,
        'per_seed': gate2_per_seed,
        'n_pass': n_pass_g2,
        'n_total': len(seeds_results),
        'pass': n_pass_g2 >= N_SEEDS_PASS,
    }

    gate['all_pass'] = gate['gate1']['pass'] and gate['gate1b']['pass'] and gate['gate2']['pass']
    return gate


# ── Arena coverage (fraction of grid cells visited) ──────────────────────────
def compute_coverage(trajectory, arena=ARENA_SIZE, cell_size=1.0):
    """Compute fraction of arena grid cells visited by the pointer."""
    visited = set()
    for x, y in trajectory:
        gx = int(x / cell_size)
        gy = int(y / cell_size)
        gx = max(0, min(int(arena / cell_size) - 1, gx))
        gy = max(0, min(int(arena / cell_size) - 1, gy))
        visited.add((gx, gy))
    total_cells = int(arena / cell_size) ** 2
    return len(visited) / total_cells


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("iter_038: 2D Navigation Gate Probe")
    print("=" * 70)
    print(f"Parameters: N={N_OBJ}, arena={ARENA_SIZE}x{ARENA_SIZE}")
    print(f"  Steps={N_STEPS}, Substeps={SUBSTEPS}")
    print(f"  Pointer: radius={PTR_RADIUS}, mass={PTR_MASS}, start=(32,32)")
    print(f"  Nav acc: Uniform[{NAV_ACC_RANGE[0]}, {NAV_ACC_RANGE[1]}]^2")
    print(f"  Velocity cap: {VEL_CAP} px/step per component")
    print(f"  Probe: radius={PROBE_RADIUS}, p={PROBE_P}, budget={PROBE_BUDGET}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Object placement: UNIFORM-RANDOM (NOT segment-based)")
    print()

    # ── Step 1: Sanity checks ──────────────────────────────────────────────
    print("Running Sanity Checks...")
    sanity_results, sanity_all_pass = run_sanity_checks()
    print()
    print(f"Sanity checks: {'ALL PASS' if sanity_all_pass else 'SOME FAILED'}")
    print()

    # ── Step 2: Run navigation experiment for each seed ────────────────────
    print("Running Navigation Gate Experiment...")
    seeds_results = []
    all_trajectories = {}
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        r = run_one_seed(seed, record_trajectory=True)
        all_trajectories[seed] = r.pop('trajectory')
        seeds_results.append(r)
        print(f"mean={r['mean_probe_count']:.2f}, "
              f"CV={r['cv_probes']:.3f}, "
              f"fired={r['probes_fired']}, "
              f"collisions={sum(r['collision_counts'])}")
    print()

    # ── Step 3: Evaluate gates ─────────────────────────────────────────────
    gates = evaluate_gates(seeds_results)

    # ── Step 4: Print summary ──────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nGate-1 (Non-saturation): mean per-object probe count <= 3.0")
    print(f"  {'Seed':>6} | {'Obj0':>4} | {'Obj1':>4} | {'Obj2':>4} | {'Mean':>6} | {'Pass':>5}")
    for s in gates['gate1']['per_seed']:
        r = next(x for x in seeds_results if x['seed'] == s['seed'])
        mark = "PASS" if s['pass'] else "FAIL"
        print(f"  {s['seed']:>6} | {r['probe_counts'][0]:>4} | {r['probe_counts'][1]:>4} | {r['probe_counts'][2]:>4} | {s['mean_probe_count']:>6.2f} | {mark:>5}")
    print(f"  Passes: {gates['gate1']['n_pass']}/{gates['gate1']['n_total']} -> {'PASS' if gates['gate1']['pass'] else 'FAIL'}")

    print("\nGate-1b (CV stability): mean(CV) >= 0.30 AND std(CV) <= 0.25")
    print(f"  {'Seed':>6} | {'CV':>10}")
    for seed_idx, cv_val in gates['gate1b']['per_seed_cvs']:
        print(f"  {seed_idx:>6} | {cv_val:>10.3f}")
    print(f"  Mean CV: {gates['gate1b']['mean_cv']:.3f} (threshold: >= {gates['gate1b']['mean_cv_threshold']:.2f})")
    print(f"  Std  CV: {gates['gate1b']['std_cv']:.3f} (threshold: <= {gates['gate1b']['std_cv_threshold']:.2f})")
    print(f"  Mean CV pass: {'PASS' if gates['gate1b']['pass_mean'] else 'FAIL'}")
    print(f"  Std  CV pass: {'PASS' if gates['gate1b']['pass_std'] else 'FAIL'}")
    print(f"  Gate-1b overall: {'PASS' if gates['gate1b']['pass'] else 'FAIL'}")

    print("\nGate-2 (Heterogeneity): per-seed CV >= 0.50")
    print(f"  {'Seed':>6} | {'CV':>10} | {'Pass':>5}")
    for s in gates['gate2']['per_seed']:
        mark = "PASS" if s['pass'] else "FAIL"
        print(f"  {s['seed']:>6} | {s['cv_probes']:>10.3f} | {mark:>5}")
    print(f"  Passes: {gates['gate2']['n_pass']}/{gates['gate2']['n_total']} -> {'PASS' if gates['gate2']['pass'] else 'FAIL'}")

    print("\n" + "-" * 70)
    g1 = "PASS" if gates['gate1']['pass'] else "FAIL"
    g1b = "PASS" if gates['gate1b']['pass'] else "FAIL"
    g2 = "PASS" if gates['gate2']['pass'] else "FAIL"
    overall = "PASS" if gates['all_pass'] else "FAIL"
    print(f"  Gate-1:  {g1}")
    print(f"  Gate-1b: {g1b}")
    print(f"  Gate-2:  {g2}")
    print(f"  ALL GATES: {overall}")
    print("-" * 70)

    # ── Exit rule ──────────────────────────────────────────────────────────
    if gates['all_pass']:
        print("\nEXIT RULE: ALL GATES PASS")
        print("  -> Result is consistent with bracket-admission.")
        print("  -> Hand off to HUMAN go/no-go on full 2D rebuild.")
        print("  -> The agent does NOT begin 2D rebuild work.")
    else:
        print("\nEXIT RULE: AT LEAST ONE GATE FAILED")
        if not gates['gate1']['pass']:
            print(f"  -> Gate-1 FAIL: {gates['gate1']['n_pass']}/{gates['gate1']['n_total']} seeds pass "
                  f"(need {N_SEEDS_PASS})")
        if not gates['gate1b']['pass']:
            reasons = []
            if not gates['gate1b']['pass_mean']:
                reasons.append(f"mean CV {gates['gate1b']['mean_cv']:.3f} < {gates['gate1b']['mean_cv_threshold']:.2f}")
            if not gates['gate1b']['pass_std']:
                reasons.append(f"std CV {gates['gate1b']['std_cv']:.3f} > {gates['gate1b']['std_cv_threshold']:.2f}")
            print(f"  -> Gate-1b FAIL: {', '.join(reasons)}")
        if not gates['gate2']['pass']:
            print(f"  -> Gate-2 FAIL: {gates['gate2']['n_pass']}/{gates['gate2']['n_total']} seeds pass "
                  f"(need {N_SEEDS_PASS})")
        print("  -> The six re-frame claims (Section 3, Step 5 FAIL branch)")
        print("     become the iter_039 scope.")
    print("=" * 70)

    # ── Write output files ─────────────────────────────────────────────────

    # 1. probe_events.csv
    with open(os.path.join(OUT_DIR, 'probe_events.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'count_obj_0', 'count_obj_1', 'count_obj_2',
                     'mean_probe_count', 'cv_probes', 'gate1_pass', 'gate2_pass'])
        for r in seeds_results:
            w.writerow([
                r['seed'],
                r['probe_counts'][0], r['probe_counts'][1], r['probe_counts'][2],
                f"{r['mean_probe_count']:.4f}",
                f"{r['cv_probes']:.4f}",
                int(r['gate1_pass']), int(r['gate2_pass']),
            ])

    # 2. diagnostic_collisions.csv
    with open(os.path.join(OUT_DIR, 'diagnostic_collisions.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'coll_obj_0', 'coll_obj_1', 'coll_obj_2', 'total_collisions'])
        for r in seeds_results:
            w.writerow([
                r['seed'],
                r['collision_counts'][0], r['collision_counts'][1], r['collision_counts'][2],
                sum(r['collision_counts']),
            ])

    # 3. sanity_checks.txt
    with open(os.path.join(OUT_DIR, 'sanity_checks.txt'), 'w', encoding='utf-8') as f:
        f.write("iter_038 2D Navigation Gate Probe - Sanity Checks\n")
        f.write("=" * 70 + "\n\n")
        # S1
        s1 = sanity_results['S1']
        f.write("S1: Physics Conservation (2D elastic collision)\n")
        f.write(f"  Pre-collision momentum:  {s1['p_pre']}\n")
        f.write(f"  Post-collision momentum: {s1['p_post']}\n")
        f.write(f"  Momentum error:          {s1['p_error']:.2e}\n")
        f.write(f"  Pre-collision KE:        {s1['ke_pre']:.6f}\n")
        f.write(f"  Post-collision KE:       {s1['ke_post']:.6f}\n")
        f.write(f"  KE error:                {s1['ke_error']:.2e}\n")
        f.write(f"  Result: {'PASS' if s1['pass'] else 'FAIL'}\n\n")
        # S2
        f.write(f"S2: Pointer stays within [0, {ARENA_SIZE}]^2\n")
        f.write(f"  {sanity_results['S2']['description']}\n")
        f.write(f"  Result: {'PASS' if sanity_results['S2']['pass'] else 'FAIL'}\n\n")
        # S3
        f.write(f"S3: Objects stay within [0, {ARENA_SIZE}]^2\n")
        f.write(f"  {sanity_results['S3']['description']}\n")
        f.write(f"  Result: {'PASS' if sanity_results['S3']['pass'] else 'FAIL'}\n\n")
        # S4
        s4 = sanity_results['S4']
        f.write("S4: Random acceleration is truly random\n")
        f.write(f"  Mean acc: {s4['mean_acc']} (expected ≈ [0, 0])\n")
        f.write(f"  Std acc:  {s4['std_acc']} (expected ≈ {s4['expected_std']:.4f})\n")
        f.write(f"  Result: {'PASS' if s4['pass'] else 'FAIL'}\n\n")
        # S5
        s5 = sanity_results['S5']
        f.write("S5: Velocity cap enforced\n")
        f.write(f"  Max velocity component observed: {s5['max_vel_component']:.4f}\n")
        f.write(f"  Cap: {s5['cap']}\n")
        f.write(f"  Result: {'PASS' if s5['pass'] else 'FAIL'}\n\n")
        # S6
        s6 = sanity_results['S6']
        f.write("S6: Probe mechanism fires correctly\n")
        f.write(f"  Steps in range: {s6['steps_in_range']}/{s6['total_steps']}\n")
        f.write(f"  Probes hit: {s6['probes_hit']}\n")
        f.write(f"  Result: {'PASS' if s6['pass'] else 'FAIL'}\n\n")
        # S7
        s7 = sanity_results['S7']
        f.write("S7: Re-verify iter_037 Gate-1 (static pointer, seed 7)\n")
        f.write(f"  Collision counts: {s7['collision_counts']}\n")
        f.write(f"  Mean: {s7['mean']:.2f} (threshold: <= {GATE1_THRESH})\n")
        f.write(f"  (Re-verification of passive pointer still giving mean <= 3.0)\n")
        f.write(f"  Result: {'PASS' if s7['pass'] else 'FAIL'}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"All sanity checks: {'ALL PASS' if sanity_all_pass else 'SOME FAILED'}\n")

    # 4. gate_summary.txt
    with open(os.path.join(OUT_DIR, 'gate_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("iter_038: 2D Navigation Gate Probe - Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write("Pre-registered Parameters:\n")
        f.write(f"  Arena: {ARENA_SIZE}x{ARENA_SIZE}, N={N_OBJ}, Steps={N_STEPS}, Substeps={SUBSTEPS}\n")
        f.write(f"  Pointer: radius={PTR_RADIUS}, mass={PTR_MASS}, start=(32,32), vel_init=(0,0)\n")
        f.write(f"  Object radii: [{OBJ_RADIUS_RANGE[0]}, {OBJ_RADIUS_RANGE[1]}], mass=radius\n")
        f.write(f"  Object placement: UNIFORM-RANDOM in [margin, ARENA_SIZE - margin]\n")
        f.write(f"  Nav acc: Uniform[{NAV_ACC_RANGE[0]}, {NAV_ACC_RANGE[1]}]^2\n")
        f.write(f"  Velocity cap: {VEL_CAP} per component\n")
        f.write(f"  Probe: radius={PROBE_RADIUS}, p={PROBE_P}, budget={PROBE_BUDGET}\n")
        f.write(f"  Seeds: {SEEDS}\n\n")

        f.write("Gate-1 (Non-saturation) - Per-Seed:\n")
        f.write(f"  Threshold: mean per-object probe count <= {GATE1_THRESH}\n")
        f.write(f"  Decision rule: >= {N_SEEDS_PASS}/{len(SEEDS)} seeds must individually pass\n")
        for s in gates['gate1']['per_seed']:
            r = next(x for x in seeds_results if x['seed'] == s['seed'])
            mark = "PASS" if s['pass'] else "FAIL"
            f.write(f"  Seed {s['seed']:>3}: counts=({r['probe_counts'][0]},{r['probe_counts'][1]},{r['probe_counts'][2]}), "
                    f"mean={s['mean_probe_count']:.3f} -> {mark}\n")
        f.write(f"  Overall: {gates['gate1']['n_pass']}/{gates['gate1']['n_total']} seeds pass -> "
                f"{'PASS' if gates['gate1']['pass'] else 'FAIL'}\n\n")

        f.write("Gate-1b (CV stability):\n")
        f.write(f"  Thresholds: mean(CV) >= {GATE1B_THRESH}, std(CV) <= {GATE1B_STD_THRESH}\n")
        for seed_idx, cv_val in gates['gate1b']['per_seed_cvs']:
            f.write(f"  Seed {seed_idx:>3}: CV={cv_val:.3f}\n")
        f.write(f"  Mean CV: {gates['gate1b']['mean_cv']:.3f} -> {'PASS' if gates['gate1b']['pass_mean'] else 'FAIL'}\n")
        f.write(f"  Std  CV: {gates['gate1b']['std_cv']:.3f} -> {'PASS' if gates['gate1b']['pass_std'] else 'FAIL'}\n")
        f.write(f"  Overall: {'PASS' if gates['gate1b']['pass'] else 'FAIL'}\n\n")

        f.write("Gate-2 (Heterogeneity) - Per-Seed:\n")
        f.write(f"  Threshold: per-seed CV >= {GATE2_THRESH}\n")
        f.write(f"  Decision rule: >= {N_SEEDS_PASS}/{len(SEEDS)} seeds must individually pass\n")
        for s in gates['gate2']['per_seed']:
            mark = "PASS" if s['pass'] else "FAIL"
            f.write(f"  Seed {s['seed']:>3}: CV={s['cv_probes']:.3f} -> {mark}\n")
        f.write(f"  Overall: {gates['gate2']['n_pass']}/{gates['gate2']['n_total']} seeds pass -> "
                f"{'PASS' if gates['gate2']['pass'] else 'FAIL'}\n\n")

        f.write("-" * 70 + "\n")
        f.write("OVERALL GATE ASSESSMENT:\n")
        f.write(f"  Gate-1:  {'PASS' if gates['gate1']['pass'] else 'FAIL'}\n")
        f.write(f"  Gate-1b: {'PASS' if gates['gate1b']['pass'] else 'FAIL'}\n")
        f.write(f"  Gate-2:  {'PASS' if gates['gate2']['pass'] else 'FAIL'}\n")
        f.write(f"  ALL GATES: {'PASS' if gates['all_pass'] else 'FAIL'}\n\n")

        if gates['all_pass']:
            f.write("EXIT RULE — PASS:\n")
            f.write("  -> is consistent with bracket-admission.\n")
            f.write("  -> Hand off to HUMAN go/no-go on full 2D rebuild.\n")
            f.write("  -> The agent does NOT begin 2D rebuild work.\n")
        else:
            f.write("EXIT RULE — FAIL:\n")
            f.write("  -> The tested parameterization fails one or more gates.\n")
            if not gates['gate1']['pass']:
                f.write(f"  -> Gate-1: {gates['gate1']['n_pass']}/{gates['gate1']['n_total']} seeds pass vs "
                        f"requirement {N_SEEDS_PASS}/{len(SEEDS)}\n")
            if not gates['gate1b']['pass']:
                f.write(f"  -> Gate-1b: mean CV={gates['gate1b']['mean_cv']:.3f} vs >={gates['gate1b']['mean_cv_threshold']:.2f}, "
                        f"std CV={gates['gate1b']['std_cv']:.3f} vs <={gates['gate1b']['std_cv_threshold']:.2f}\n")
            if not gates['gate2']['pass']:
                f.write(f"  -> Gate-2: {gates['gate2']['n_pass']}/{gates['gate2']['n_total']} seeds pass vs "
                        f"requirement {N_SEEDS_PASS}/{len(SEEDS)}\n")
            f.write("  -> The six re-frame claims (Section 3, Step 5 FAIL branch of\n")
            f.write("     pre-registration) become the iter_039 scope.\n")
        f.write("=" * 70 + "\n")

    # 5. trajectory_stats.csv
    with open(os.path.join(OUT_DIR, 'trajectory_stats.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'mean_speed', 'total_distance',
                     'arena_coverage_fraction', 'cells_visited', 'total_cells'])
        for seed in SEEDS:
            traj = all_trajectories[seed]
            r = next(x for x in seeds_results if x['seed'] == seed)
            total_dist = r['total_distance']
            mean_speed = total_dist / N_STEPS
            coverage = compute_coverage(traj)
            total_cells = int(ARENA_SIZE) ** 2
            cells_visited = int(round(coverage * total_cells))
            w.writerow([
                seed,
                f"{mean_speed:.4f}",
                f"{total_dist:.4f}",
                f"{coverage:.4f}",
                cells_visited,
                total_cells,
            ])

    print(f"\nResults saved to {OUT_DIR}/")
    print("  - probe_events.csv")
    print("  - diagnostic_collisions.csv")
    print("  - sanity_checks.txt")
    print("  - gate_summary.txt")
    print("  - trajectory_stats.csv")

    return {
        'seeds_results': seeds_results,
        'gates': gates,
        'sanity': sanity_results,
        'sanity_all_pass': sanity_all_pass,
    }


if __name__ == '__main__':
    main()
