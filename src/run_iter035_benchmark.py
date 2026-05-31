#!/usr/bin/env python3
"""
iter_035: Pass-Through Physics Benchmark

Tests whether perception-driven targeting is load-bearing for mass estimation
in an environment where objects pass through each other (no obj-obj collisions).
Only pointer-object collisions remain, so the agent MUST target objects to get
information about their masses.
"""
import os, sys, csv, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import PhysicsSandbox

# ── Constants ──
SEEDS = [7, 31, 53, 71, 83, 97, 113, 163]
N_OBJ = 3
N_STEPS = 2000
PTR_IDX = N_OBJ  # pointer index in concatenated arrays
PTR_MASS = 10.0
KP, KD = 2.0, 0.5
PUSH_DIST = 6.0
PUSH_VEL = 5.0
PUSH_CD = 15
MAX_PUSHES = 15
PROX_THRESH = 4.0
DV_THRESH = 0.5
N_BOOT = 10000
OUT = os.path.join('archive', 'iter_035', 'results')


# ── PassThroughPhysicsSandbox ──
class PassThroughPhysicsSandbox(PhysicsSandbox):
    """Objects pass through each other; only pointer-object collisions are elastic."""
    
    def step(self, action=None):
        if action is not None:
            acc = action.get('acc', 0.0)
            push = action.get('push', False)
        else:
            acc = 0.0
            push = False
            
        if push and len(self.positions) > 0:
            dists = self.positions - self.pointer_pos
            nearest_idx = np.argmin(np.abs(dists))
            diff = dists[nearest_idx]
            direction = 1.0 if diff >= 0 else -1.0
            self.pointer_vel = direction * PUSH_VEL
                
        dt = 1.0 / self.substeps
        
        for _ in range(self.substeps):
            self.pointer_vel += acc * dt
            
            # Pack all N+1 entities
            temp_pos = np.concatenate([self.positions, [self.pointer_pos]])
            temp_vel = np.concatenate([self.velocities, [self.pointer_vel]])
            temp_rad = np.concatenate([self.radii, [self.pointer_radius]])
            temp_mass = np.concatenate([self.masses, [self.pointer_mass]])
            
            # Update positions
            temp_pos += temp_vel * dt
            
            # Boundary bounces for ALL entities
            for i in range(len(temp_pos)):
                if temp_pos[i] - temp_rad[i] < 0.0:
                    temp_pos[i] = temp_rad[i]
                    if temp_vel[i] < 0.0:
                        temp_vel[i] = -temp_vel[i]
                elif temp_pos[i] + temp_rad[i] > 128.0:
                    temp_pos[i] = 128.0 - temp_rad[i]
                    if temp_vel[i] > 0.0:
                        temp_vel[i] = -temp_vel[i]
            
            # Resolve collisions — ONLY pointer-object pairs
            sort_idx = np.argsort(temp_pos)
            for k in range(len(sort_idx) - 1):
                i = int(sort_idx[k])
                j = int(sort_idx[k + 1])
                
                # SKIP object-object collisions (pass-through)
                if i < self.N and j < self.N:
                    continue
                
                dist = temp_pos[j] - temp_pos[i]
                min_dist = temp_rad[i] + temp_rad[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    m_inv_i = 1.0 / temp_mass[i]
                    m_inv_j = 1.0 / temp_mass[j]
                    sum_inv_m = m_inv_i + m_inv_j
                    
                    temp_pos[i] -= overlap * (m_inv_i / sum_inv_m)
                    temp_pos[j] += overlap * (m_inv_j / sum_inv_m)
                    
                    if temp_vel[i] > temp_vel[j]:
                        v1 = temp_vel[i]
                        v2 = temp_vel[j]
                        m1 = temp_mass[i]
                        m2 = temp_mass[j]
                        temp_vel[i] = (v1 * (m1 - m2) + 2.0 * m2 * v2) / (m1 + m2)
                        temp_vel[j] = (v2 * (m2 - m1) + 2.0 * m1 * v1) / (m1 + m2)
                        
            # Additional boundary check after collision resolution
            for i in range(len(temp_pos)):
                if temp_pos[i] - temp_rad[i] < 0.0:
                    temp_pos[i] = temp_rad[i]
                    if temp_vel[i] < 0.0:
                        temp_vel[i] = -temp_vel[i]
                elif temp_pos[i] + temp_rad[i] > 128.0:
                    temp_pos[i] = 128.0 - temp_rad[i]
                    if temp_vel[i] > 0.0:
                        temp_vel[i] = -temp_vel[i]
                        
            self.positions = temp_pos[:-1].copy()
            self.velocities = temp_vel[:-1].copy()
            self.pointer_pos = temp_pos[-1]
            self.pointer_vel = temp_vel[-1]

        # Noisy-TV / Structured Distractor updates (inherited)
        if self.noisy_tv:
            step_noise = np.random.normal(0.0, 2.0)
            self.noisy_tv_pos += step_noise
            self.noisy_tv_pos = np.clip(self.noisy_tv_pos, self.noisy_tv_radius, 128.0 - self.noisy_tv_radius)
            self.noisy_tv_color = np.random.uniform(0.3, 1.0, size=(3,))
        if self.structured_distractor:
            self.sd_t += 1
            self.sd_pos = self.sd_center + self.sd_amplitude * np.sin(self.sd_omega * self.sd_t + self.sd_phase)

        obs = self.render()
        info = {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "radii": self.radii.copy(),
            "masses": self.masses.copy(),
            "colors": self.colors.copy(),
            "pointer_pos": self.pointer_pos,
            "pointer_vel": self.pointer_vel,
            "pointer_radius": self.pointer_radius,
            "pointer_mass": self.pointer_mass,
            "pointer_color": self.pointer_color.copy(),
        }
        if self.noisy_tv:
            info["noisy_tv_pos"] = self.noisy_tv_pos
            info["noisy_tv_color"] = self.noisy_tv_color.copy()
            info["noisy_tv_radius"] = self.noisy_tv_radius
        if self.structured_distractor:
            info["sd_pos"] = self.sd_pos
            info["sd_color"] = self.sd_color.copy()
            info["sd_radius"] = self.sd_radius
        return obs, info


# ── Collision Detection ──
def detect_ptr_obj_collisions(step, pre_ptr_vel, pre_obj_vels, post_ptr_vel, post_obj_vels,
                               ptr_pos, obj_positions, ptr_radius, obj_radii):
    """Detect pointer-object collisions by proximity + velocity change."""
    colls = []
    for i in range(N_OBJ):
        d = abs(ptr_pos - obj_positions[i])
        if d < ptr_radius + obj_radii[i] + PROX_THRESH:
            dv_ptr = post_ptr_vel - pre_ptr_vel
            dv_obj = post_obj_vels[i] - pre_obj_vels[i]
            if abs(dv_ptr) > DV_THRESH and abs(dv_obj) > DV_THRESH:
                colls.append((step, i, float(pre_ptr_vel), float(pre_obj_vels[i]),
                              float(post_ptr_vel), float(post_obj_vels[i]),
                              float(dv_ptr), float(dv_obj)))
    return colls


# ── POMLRE Metric ──
def compute_pomlre(collisions, true_masses):
    """Per-Object Median Log-Ratio Error."""
    errors = []
    per_obj_n_valid = []
    per_obj_errors = []
    
    for i in range(N_OBJ):
        obj_colls = [c for c in collisions if c[1] == i and abs(c[6]) > 1.0]  # |dv_obj| > 1.0
        # Note: c[6] = dv_obj, c[5] = dv_ptr
        # Actually c indices: (step, obj_i, pre_ptr, pre_obj, post_ptr, post_obj, dv_ptr, dv_obj)
        valid = [c for c in obj_colls if abs(c[7]) > 1.0]  # |Δv_obj| > 1.0
        
        if len(valid) >= 3:
            m_ests = [-PTR_MASS * c[6] / c[7] for c in valid]  # -M_ptr * dv_ptr / dv_obj
            m_hat = float(np.median(m_ests))
            if m_hat > 0 and true_masses[i] > 0:
                err = abs(np.log(m_hat / true_masses[i]))
            else:
                err = 2.0
        elif len(valid) >= 1:
            m_ests = [-PTR_MASS * c[6] / c[7] for c in valid]
            m_hat = float(np.mean(m_ests))
            if m_hat > 0 and true_masses[i] > 0:
                err = abs(np.log(m_hat / true_masses[i]))
            else:
                err = 2.0
        else:
            err = 2.0
        
        errors.append(err)
        per_obj_n_valid.append(len(valid))
        per_obj_errors.append(err)
    
    return float(np.mean(errors)), per_obj_n_valid, per_obj_errors


# ── Bootstrap CI ──
def bootstrap_ci(a_vals, b_vals, n_boot=N_BOOT):
    """Paired bootstrap CI for mean(b) - mean(a). Returns (lower_95, upper_95)."""
    n = len(a_vals)
    gaps = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        gaps.append(float(np.mean(b_vals[idx]) - np.mean(a_vals[idx])))
    gaps = np.array(gaps)
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


# ── Single Episode ──
def run_episode(condition, seed):
    env = PassThroughPhysicsSandbox(N=N_OBJ, substeps=10, seed=seed)
    rng = np.random.RandomState(seed * 1000 + hash(condition) % 10000)
    
    # ORACLE state
    if condition == 'ORACLE':
        obj_ptr_colls = [0, 0, 0]
        pushes_left = MAX_PUSHES
        push_cd = 0
        prev_err = None
        target = 0  # start with object 0
    
    # RANDOM state
    if condition == 'RANDOM':
        pushes_left = MAX_PUSHES
    
    all_colls = []
    ptr_oob = 0
    pushes_used = 0
    obj_pushes = [0, 0, 0]  # per-object push count
    
    for step in range(N_STEPS):
        # Save pre-state
        pre_ptr_vel = env.pointer_vel
        pre_obj_vels = env.velocities.copy()
        
        # Compute action
        if condition == 'ORACLE':
            target = int(np.argmin(obj_ptr_colls))
            error = env.positions[target] - env.pointer_pos
            d_error = (error - prev_err) if prev_err is not None else 0.0
            prev_err = error
            acc = KP * error + KD * d_error
            
            if push_cd > 0:
                push_cd -= 1
            
            do_push = False
            if abs(error) <= PUSH_DIST and push_cd == 0 and pushes_left > 0:
                do_push = True
                pushes_left -= 1
                pushes_used += 1
                obj_pushes[target] += 1
                push_cd = PUSH_CD
                # Switch target to next least-collided
                counts = np.array(obj_ptr_colls, dtype=float)
                counts[target] = np.inf
                target = int(np.argmin(counts))
                prev_err = None
            
            action = {'acc': acc, 'push': False}
            if do_push:
                env.pointer_vel = PUSH_VEL * np.sign(error)
                
        elif condition == 'RANDOM':
            acc = rng.uniform(-10, 10)
            do_push = rng.rand() < 0.1 and pushes_left > 0
            if do_push:
                pushes_left -= 1
                pushes_used += 1
            action = {'acc': acc, 'push': do_push}
            
        else:  # PASSIVE
            action = {'acc': 0.0, 'push': False}
        
        obs, info = env.step(action)
        
        post_ptr_vel = env.pointer_vel
        post_obj_vels = info['velocities'].copy()
        
        if not (0 <= env.pointer_pos <= 128):
            ptr_oob += 1
        
        # Detect collisions
        step_colls = detect_ptr_obj_collisions(
            step, pre_ptr_vel, pre_obj_vels, post_ptr_vel, post_obj_vels,
            env.pointer_pos, env.positions, env.pointer_radius, env.radii
        )
        all_colls.extend(step_colls)
        
        # Update ORACLE collision counts
        if condition == 'ORACLE':
            for c in step_colls:
                obj_idx = c[1]
                obj_ptr_colls[obj_idx] += 1
    
    true_masses = env.masses.copy()
    pomlre, per_obj_n_valid, per_obj_errors = compute_pomlre(all_colls, true_masses)
    
    # Per-object valid collision counts
    per_obj_total_colls = [0] * N_OBJ
    for c in all_colls:
        per_obj_total_colls[c[1]] += 1
    
    # S3: fraction of collision events with |dv_obj| > 1.0
    n_valid = sum(1 for c in all_colls if abs(c[7]) > 1.0)
    n_total = len(all_colls)
    s3_frac = n_valid / n_total if n_total > 0 else 1.0
    
    return {
        'condition': condition, 'seed': seed,
        'pomlre': pomlre,
        'per_obj_n_valid': per_obj_n_valid,
        'per_obj_errors': per_obj_errors,
        'per_obj_total_colls': per_obj_total_colls,
        'true_masses': true_masses.tolist(),
        'pushes_used': pushes_used,
        'obj_pushes': obj_pushes,
        'ptr_oob_frac': ptr_oob / N_STEPS,
        'n_collisions': n_total,
        's3_frac': s3_frac,
    }


# ── Analytical Ceiling ──
def analytical_ceiling():
    """Simulate PASSIVE to estimate expected collisions per object."""
    print("\n" + "="*60)
    print("ANALYTICAL CEILING GATE")
    print("="*60)
    
    test_seeds = [7, 31, 53, 71, 83]
    n_valid_per_obj = []
    
    for seed in test_seeds:
        r = run_episode('PASSIVE', seed)
        n_valid_per_obj.append(r['per_obj_n_valid'])
    
    n_valid_arr = np.array(n_valid_per_obj)
    mean_per_obj = n_valid_arr.mean(axis=0)
    overall_mean = n_valid_arr.mean()
    
    print(f"PASSIVE mean valid collisions per object: {mean_per_obj.tolist()}")
    print(f"Overall mean: {overall_mean:.2f}")
    
    gate_pass = overall_mean < 3.0
    
    with open(os.path.join(OUT, 'analytical_ceiling.txt'), 'w') as f:
        f.write(f"PASSIVE mean valid collisions per object: {mean_per_obj.tolist()}\n")
        f.write(f"Overall mean: {overall_mean:.2f}\n")
        f.write(f"Gate (mean < 3.0): {'PASS' if gate_pass else 'FAIL'}\n")
        if not gate_pass:
            f.write("Environment redesign FAILED: PASSIVE gets enough collisions without targeting.\n")
    
    return gate_pass, mean_per_obj, overall_mean


# ── Main ──
def main():
    os.makedirs(OUT, exist_ok=True)
    
    print("="*70)
    print("iter_035 Pass-Through Physics Benchmark")
    print("="*70)
    
    # Quick test of PassThroughPhysicsSandbox
    print("\nQuick test of PassThroughPhysicsSandbox...")
    env = PassThroughPhysicsSandbox(N=N_OBJ, substeps=10, seed=42)
    # Objects should pass through each other
    # Set two objects at same position moving toward each other
    env.positions = np.array([50.0, 52.0, 90.0])
    env.velocities = np.array([2.0, -2.0, 1.0])
    env.radii = np.array([5.0, 5.0, 4.0])
    env.masses = np.array([5.0, 5.0, 4.0])
    env.pointer_pos = 20.0
    env.pointer_vel = 0.0
    
    pre_pos = env.positions.copy()
    obs, info = env.step({'acc': 0.0, 'push': False})
    post_pos = info['positions'].copy()
    
    # Objects 0 and 1 should have passed through each other
    passed_through = not (abs(post_pos[0] - post_pos[1]) > abs(pre_pos[0] - pre_pos[1]))
    print(f"  Pre positions: {pre_pos}")
    print(f"  Post positions: {post_pos}")
    print(f"  Objects pass through: {'YES' if passed_through else 'NO (checking velocity exchange)'}")
    
    # Better check: if objects 0 and 1 were approaching (v0>v1), and they passed through,
    # their velocities should NOT have been exchanged
    pre_v0, pre_v1 = 2.0, -2.0
    post_v0 = info['velocities'][0]
    post_v1 = info['velocities'][1]
    no_exchange = abs(post_v0 - pre_v0) < 0.1 and abs(post_v1 - pre_v1) < 0.1
    print(f"  Pre v0={pre_v0}, v1={pre_v1}")
    print(f"  Post v0={post_v0:.4f}, v1={post_v1:.4f}")
    print(f"  No velocity exchange (pass-through works): {'YES' if no_exchange else 'NO - objects collided!'}")
    
    if not no_exchange:
        print("ERROR: PassThroughPhysicsSandbox is not working correctly!")
        print("Objects are still colliding with each other. Aborting.")
        return
    
    print("\nPassThroughPhysicsSandbox test passed. Proceeding to analytical ceiling.")
    
    # Step 1: Analytical ceiling
    ceiling_pass, mean_per_obj, overall_mean = analytical_ceiling()
    
    if not ceiling_pass:
        print("\nANALYTICAL CEILING GATE FAILED.")
        print(f"PASSIVE gets {overall_mean:.2f} valid collisions per object on average.")
        print("Environment redesign does not make perception load-bearing for PASSIVE.")
        print("Stopping. This is the finding.")
        return
    
    print(f"\nAnalytical ceiling gate PASSED (PASSIVE mean={overall_mean:.2f} < 3.0)")
    print("Proceeding to full experiment...")
    
    # Step 2: Full experiment
    results = []
    for cond in ['ORACLE', 'RANDOM', 'PASSIVE']:
        for seed in SEEDS:
            print(f"  {cond:8s} seed={seed:3d} ...", end=" ", flush=True)
            r = run_episode(cond, seed)
            results.append(r)
            print(f"POMLRE={r['pomlre']:.4f}  colls={r['n_collisions']:3d}  valid={r['per_obj_n_valid']}")
    
    by_cond = {c: [r for r in results if r['condition']==c] for c in ['ORACLE','RANDOM','PASSIVE']}
    
    # ── Sanity Checks ──
    oracle = by_cond['ORACLE']
    
    # S1: ≥3 valid ptr-obj collisions per object (mean across seeds)
    s1_per_obj = np.array([r['per_obj_n_valid'] for r in oracle]).mean(axis=0)
    s1_pass = bool(np.all(s1_per_obj >= 3))
    
    # S2: push budget utilization ≥80%
    s2_pushes = np.array([r['pushes_used'] for r in oracle])
    s2_pass = bool(s2_pushes.mean() >= 12)
    
    # S3: ≥80% of events have |dv_obj| > 1.0
    s3_pass = all(r['s3_frac'] >= 0.8 for r in oracle)
    
    # S4: no single object receives >80% of pushes
    s4_pass = True
    for r in oracle:
        if r['pushes_used'] > 0:
            max_frac = max(r['obj_pushes']) / r['pushes_used']
            if max_frac > 0.8:
                s4_pass = False
                break
    
    # S5: pointer stays in bounds ≥95%
    s5_pass = all(r['ptr_oob_frac'] <= 0.05 for r in oracle)
    
    sanity = {
        'S1': (s1_pass, f"Mean per-obj valid colls: {s1_per_obj.tolist()}"),
        'S2': (s2_pass, f"Mean pushes used: {s2_pushes.mean():.1f}/{MAX_PUSHES}"),
        'S3': (s3_pass, f"Min s3_frac: {min(r['s3_frac'] for r in oracle):.3f}"),
        'S4': (s4_pass, f"Max push fraction to single obj: {max(max(r['obj_pushes'])/max(r['pushes_used'],1) for r in oracle if r['pushes_used']>0):.3f}"),
        'S5': (s5_pass, f"Max OOB frac: {max(r['ptr_oob_frac'] for r in oracle):.4f}"),
    }
    
    all_sanity = all(v[0] for v in sanity.values())
    
    # ── Metrics ──
    oracle_pomlre = np.array([r['pomlre'] for r in by_cond['ORACLE']])
    random_pomlre = np.array([r['pomlre'] for r in by_cond['RANDOM']])
    passive_pomlre = np.array([r['pomlre'] for r in by_cond['PASSIVE']])
    
    # ── Gates ──
    # F1: RANDOM - ORACLE >= 0.15
    f1_gap = float(np.mean(random_pomlre - oracle_pomlre))
    f1_pass = f1_gap >= 0.15
    
    # F2: lower bound of two-sided 95% CI > 0
    f2_lo, f2_hi = bootstrap_ci(oracle_pomlre, random_pomlre)
    f2_pass = f2_lo > 0
    
    # F3: all sanity checks
    f3_pass = all_sanity
    
    # F4: ordering PASSIVE > RANDOM > ORACLE
    f4_pass = (np.mean(passive_pomlre) > np.mean(random_pomlre) > np.mean(oracle_pomlre))
    
    # ── Coverage-vs-Estimation Decomposition ──
    # Coverage-only: mean valid events per object per condition
    oracle_n_valid = np.array([r['per_obj_n_valid'] for r in by_cond['ORACLE']])
    random_n_valid = np.array([r['per_obj_n_valid'] for r in by_cond['RANDOM']])
    passive_n_valid = np.array([r['per_obj_n_valid'] for r in by_cond['PASSIVE']])
    
    oracle_cov_mean = oracle_n_valid.mean()
    random_cov_mean = random_n_valid.mean()
    passive_cov_mean = passive_n_valid.mean()
    
    # Estimation-only: restrict to seed×obj cells where BOTH ORACLE and RANDOM have ≥3 valid events
    est_only_oracle_errors = []
    est_only_random_errors = []
    
    for si, seed in enumerate(SEEDS):
        o_r = by_cond['ORACLE'][si]
        r_r = by_cond['RANDOM'][si]
        for oi in range(N_OBJ):
            if o_r['per_obj_n_valid'][oi] >= 3 and r_r['per_obj_n_valid'][oi] >= 3:
                est_only_oracle_errors.append(o_r['per_obj_errors'][oi])
                est_only_random_errors.append(r_r['per_obj_errors'][oi])
    
    if len(est_only_oracle_errors) > 0:
        est_only_gap = float(np.mean(est_only_random_errors) - np.mean(est_only_oracle_errors))
    else:
        est_only_gap = float('nan')
    
    est_only_gate = est_only_gap >= 0.05 if not np.isnan(est_only_gap) else False
    
    # ── Print Results ──
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    for k, (v, d) in sanity.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'} — {d}")
    print(f"  All sanity: {'PASS' if all_sanity else 'FAIL'}")
    
    print("\n" + "="*70)
    print("PER-CONDITION POMLRE")
    print("="*70)
    for cond in ['ORACLE','RANDOM','PASSIVE']:
        vals = [r['pomlre'] for r in by_cond[cond]]
        print(f"  {cond:8s}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}")
    
    print("\nPer-seed POMLRE:")
    print(f"{'Seed':>6} | {'ORACLE':>8} | {'RANDOM':>8} | {'PASSIVE':>8}")
    for i, seed in enumerate(SEEDS):
        print(f"{seed:>6} | {oracle_pomlre[i]:.4f}   | {random_pomlre[i]:.4f}   | {passive_pomlre[i]:.4f}")
    
    print("\n" + "="*70)
    print("GATES")
    print("="*70)
    print(f"  F1 (RANDOM-ORACLE >= 0.15):      {'PASS' if f1_pass else 'FAIL'}  gap={f1_gap:.4f}")
    print(f"  F2 (CI lower > 0):               {'PASS' if f2_pass else 'FAIL'}  CI=[{f2_lo:.4f}, {f2_hi:.4f}]")
    print(f"  F3 (sanity checks):              {'PASS' if f3_pass else 'FAIL'}")
    print(f"  F4 (PASSIVE>RANDOM>ORACLE):      {'PASS' if f4_pass else 'FAIL'}")
    print(f"  Est-only gap >= 0.05:            {'PASS' if est_only_gate else 'FAIL'}  gap={est_only_gap:.4f}")
    
    # ── Coverage Decomposition ──
    print("\n" + "="*70)
    print("COVERAGE VS ESTIMATION DECOMPOSITION")
    print("="*70)
    print(f"  Coverage (mean valid events per object):")
    print(f"    ORACLE: {oracle_cov_mean:.2f}")
    print(f"    RANDOM: {random_cov_mean:.2f}")
    print(f"    PASSIVE: {passive_cov_mean:.2f}")
    print(f"  Estimation-only POMLRE (matched-coverage cells):")
    print(f"    n_cells: {len(est_only_oracle_errors)}")
    print(f"    ORACLE est-only mean: {np.mean(est_only_oracle_errors):.4f}" if est_only_oracle_errors else "    ORACLE est-only: N/A")
    print(f"    RANDOM est-only mean: {np.mean(est_only_random_errors):.4f}" if est_only_random_errors else "    RANDOM est-only: N/A")
    print(f"    Est-only gap: {est_only_gap:.4f}")
    
    # ── Overall Conclusion ──
    all_gates = f1_pass and f2_pass and f3_pass and f4_pass
    hypothesis_holds = all_gates and est_only_gate
    
    print("\n" + "="*70)
    if hypothesis_holds:
        print("HYPOTHESIS SUPPORTED: Pass-through environment makes perception-driven")
        print("targeting load-bearing for mass-estimation under a finite budget.")
    elif all_gates and not est_only_gate:
        print("MIXED RESULT: ORACLE wins by coverage, not by perception-quality discrimination.")
        print(f"Full POMLRE gap = {f1_gap:.4f} (≥0.15), but estimation-only gap = {est_only_gap:.4f} (<0.05).")
    else:
        print("HYPOTHESIS FALSIFIED.")
        if not f1_pass:
            print(f"  F1 failed: gap={f1_gap:.4f} < 0.15")
        if not f2_pass:
            print(f"  F2 failed: CI lower={f2_lo:.4f} ≤ 0")
        if not f3_pass:
            print(f"  F3 failed: sanity checks")
        if not f4_pass:
            print(f"  F4 failed: ordering violated")
        print("  Null finding: perception is not behaviorally load-bearing under")
        print("  full observation even with pass-through dynamics.")
        print("  ESCALATION: Pull foveated-gaze mechanism (Section 8.2) forward.")
    print("="*70)
    
    # ── Save Files ──
    # per_run.csv
    with open(os.path.join(OUT, 'per_run.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition','seed','pomlre','pushes_used','n_collisions',
                     'n_valid_0','n_valid_1','n_valid_2',
                     'err_0','err_1','err_2',
                     'true_m0','true_m1','true_m2',
                     'ptr_oob_frac','s3_frac'])
        for r in results:
            w.writerow([r['condition'], r['seed'], r['pomlre'],
                        r['pushes_used'], r['n_collisions'],
                        *r['per_obj_n_valid'], *r['per_obj_errors'],
                        *r['true_masses'], r['ptr_oob_frac'], r['s3_frac']])
    
    # sanity_checks.txt
    with open(os.path.join(OUT, 'sanity_checks.txt'), 'w') as f:
        f.write("iter_035 Sanity Checks\n")
        f.write("="*50 + "\n\n")
        for k, (v, d) in sanity.items():
            f.write(f"{k}: {'PASS' if v else 'FAIL'}\n  {d}\n\n")
        f.write(f"\nAll sanity checks: {'PASS' if all_sanity else 'FAIL'}\n")
    
    # analysis.md
    with open(os.path.join(OUT, 'analysis.md'), 'w') as f:
        f.write("# iter_035 Pass-Through Benchmark Analysis\n\n")
        f.write("## Environment\nPass-through physics: objects pass through each other, only pointer-object collisions are elastic.\n")
        f.write(f"N={N_OBJ} objects, {N_STEPS} steps, {MAX_PUSHES} push budget.\n\n")
        
        f.write("## Analytical Ceiling\n")
        f.write(f"PASSIVE mean valid collisions per object: {overall_mean:.2f}\n")
        f.write(f"Gate (< 3.0): {'PASS' if ceiling_pass else 'FAIL'}\n\n")
        
        f.write("## Per-Seed POMLRE\n\n")
        f.write("| Seed | ORACLE | RANDOM | PASSIVE |\n|------|--------|--------|--------|\n")
        for i, seed in enumerate(SEEDS):
            f.write(f"| {seed} | {oracle_pomlre[i]:.4f} | {random_pomlre[i]:.4f} | {passive_pomlre[i]:.4f} |\n")
        
        f.write(f"\n## Summary\n\n")
        for cond in ['ORACLE','RANDOM','PASSIVE']:
            vals = [r['pomlre'] for r in by_cond[cond]]
            f.write(f"- {cond}: POMLRE={np.mean(vals):.4f}±{np.std(vals):.4f}\n")
        
        f.write(f"\n## Gates\n\n")
        f.write(f"- F1 (gap≥0.15): {'PASS' if f1_pass else 'FAIL'}, gap={f1_gap:.4f}\n")
        f.write(f"- F2 (CI lower>0): {'PASS' if f2_pass else 'FAIL'}, CI=[{f2_lo:.4f},{f2_hi:.4f}]\n")
        f.write(f"- F3 (sanity): {'PASS' if f3_pass else 'FAIL'}\n")
        f.write(f"- F4 (ordering): {'PASS' if f4_pass else 'FAIL'}\n")
        f.write(f"- Est-only gap≥0.05: {'PASS' if est_only_gate else 'FAIL'}, gap={est_only_gap:.4f}\n")
        
        f.write(f"\n## Coverage vs Estimation\n\n")
        f.write(f"Mean valid events per object: ORACLE={oracle_cov_mean:.2f}, RANDOM={random_cov_mean:.2f}, PASSIVE={passive_cov_mean:.2f}\n")
        f.write(f"Estimation-only (matched cells): n={len(est_only_oracle_errors)}, gap={est_only_gap:.4f}\n")
        
        f.write(f"\n## Conclusion\n\n")
        if hypothesis_holds:
            f.write("HYPOTHESIS SUPPORTED.\n")
        elif all_gates and not est_only_gate:
            f.write("MIXED: ORACLE wins by coverage, not by perception-quality discrimination.\n")
        else:
            f.write("HYPOTHESIS FALSIFIED. Null finding: perception not load-bearing even with pass-through.\n")
            f.write("Escalation: pull foveated-gaze (Section 8.2) forward.\n")
    
    print(f"\nResults saved to {OUT}/")


if __name__ == '__main__':
    main()
