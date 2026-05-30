#!/usr/bin/env python3
"""Iter_033: Three-Condition Oracle Bracket Experiment."""
import os, sys, csv, json, collections, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models_separate_dyn import NonParametricJEPASpatialSeparateDyn
from src.motor import CLTSMotorController
from src.environment import PhysicsSandbox

# ─── Configuration ───
SEEDS = [7, 17, 31, 53, 71, 83, 97, 101, 107, 113, 137, 163]
CONDITIONS = ["random", "learned_vicreg", "learned_sfa", "oracle"]
D_T = 3
D_MAX = 8
EVAL_STEPS = 2000
PERTURB_STEP = 1000
MASS_MULTIPLIER = 1.5
WARMUP_STEPS = 3
HISTORY_LEN = 4
COLLISION_DIST_THRESHOLD = 4.0
COLLISION_VELOCITY_CHANGE_THRESHOLD = 1.0
POST_COLLISION_WINDOW = 15
CKPT_DIR = "archive/iter_029/results/checkpoints"
RESULTS_DIR = "archive/iter_033/results"

def build_model():
    model = NonParametricJEPASpatialSeparateDyn(
        d_max=D_MAX, h=3, k=4, cooldown=300, stabilization_period=100,
        pos_encoding="none", primary_objective="jepa", sfa_weight=25.0,
        gdasr_log_only=True, dyn_readout="mean", sub_features=1,
        dyn_source="spatial", mask_dyn_sim=True, coord_vicreg=True,
    )
    model.d_t = D_T
    return model

def get_channel_to_obj_mapping(centroids_np, positions, d_t):
    mapping = {}
    for ch in range(d_t):
        val = centroids_np[ch]
        closest_obj = int(np.argmin(np.abs(positions - val)))
        mapping[ch] = closest_obj
    return mapping

def detect_collision(info, prev_velocities):
    pos_diff = abs(info["positions"][0] - info["positions"][1])
    radii_sum = info["radii"][0] + info["radii"][1]
    if pos_diff >= (radii_sum + COLLISION_DIST_THRESHOLD):
        return False, None
    vel_changes = np.abs(info["velocities"] - prev_velocities)
    max_change = np.max(vel_changes)
    if max_change > COLLISION_VELOCITY_CHANGE_THRESHOLD:
        return True, int(np.argmax(vel_changes))
    return False, None

def run_random(seed):
    env = PhysicsSandbox(N=2, seed=seed)
    controller = CLTSMotorController()
    model = build_model()
    model.eval()
    obs = env.reset()
    history = collections.deque(maxlen=HISTORY_LEN)
    history.append(obs)
    prev_velocities = env.velocities.copy()
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        prev_velocities = info["velocities"].copy()
    device = torch.device("cpu")
    tracking_errors, collision_events, attended_per_step, perturbation_attended = [], [], [], []
    surprise_log_coord, surprise_log_dyn = [], []
    for step in range(EVAL_STEPS):
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER
        obs_tensor = torch.tensor(history[-1], dtype=torch.float32).unsqueeze(0)
        x_hist = torch.from_numpy(np.stack(list(history)[:3], axis=0)).float().unsqueeze(0)
        x_target = torch.from_numpy(history[-1]).float().unsqueeze(0)
        with torch.no_grad():
            z_coord, z_dyn = model.encoder(obs_tensor)
            centroids = z_coord
            loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(x_hist, x_target, d_t_predict=min(D_T, D_MAX))
        # Zero EMA before action
        controller.mu[:] = 0.0
        controller.sigma[:] = 1.0
        controller.attention_cooldown = 0
        action, locus, surprises = controller.get_action(
            None, history[-1], info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, D_T, centroids)
        # Override locus to random
        random_locus = int(np.random.randint(0, D_T))
        locus = random_locus
        controller.token_locus = random_locus
        # Log surprise decomposition
        for c in range(D_T):
            surprise_log_coord.append(torch.mean((z_pred_coord[:, c] - z_target_coord[:, c])**2).item())
            surprise_log_dyn.append(torch.mean((z_pred_dyn[:, c] - z_target_dyn[:, c])**2).item())
        obs, info = env.step(action)
        history.append(obs)
        centroids_np = centroids[0, :D_T].cpu().numpy()
        ch2obj = get_channel_to_obj_mapping(centroids_np, info["positions"], D_T)
        attended_obj = ch2obj.get(locus, -1)
        target_centroid = centroids_np[locus]
        tracking_errors.append(abs(info["pointer_pos"] - target_centroid))
        attended_per_step.append((step, attended_obj))
        is_collision, max_change_obj = detect_collision(info, prev_velocities)
        if is_collision:
            collision_events.append((step, max_change_obj))
        if PERTURB_STEP <= step <= PERTURB_STEP + 99:
            perturbation_attended.append(1 if attended_obj == 0 else 0)
        prev_velocities = info["velocities"].copy()
    # Compute selectivity
    post_coll_vb = compute_selectivity_vb(collision_events, attended_per_step, EVAL_STEPS)
    return {
        "tracking_error": float(np.mean(tracking_errors)),
        "selectivity_vb": post_coll_vb,
        "perturbation_selectivity": float(np.mean(perturbation_attended)) if perturbation_attended else 0.0,
        "collision_count": len(collision_events),
        "mean_surprise_coord": float(np.mean(surprise_log_coord)) if surprise_log_coord else 0.0,
        "mean_surprise_dyn": float(np.mean(surprise_log_dyn)) if surprise_log_dyn else 0.0,
    }

def run_learned(seed, ckpt_path):
    env = PhysicsSandbox(N=2, seed=seed)
    controller = CLTSMotorController()
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    obs = env.reset()
    history = collections.deque(maxlen=HISTORY_LEN)
    history.append(obs)
    prev_velocities = env.velocities.copy()
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        prev_velocities = info["velocities"].copy()
    tracking_errors, collision_events, attended_per_step, perturbation_attended = [], [], [], []
    surprise_log_coord, surprise_log_dyn = [], []
    for step in range(EVAL_STEPS):
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER
        obs_tensor = torch.tensor(history[-1], dtype=torch.float32).unsqueeze(0)
        x_hist = torch.from_numpy(np.stack(list(history)[:3], axis=0)).float().unsqueeze(0)
        x_target = torch.from_numpy(history[-1]).float().unsqueeze(0)
        with torch.no_grad():
            z_coord, z_dyn = model.encoder(obs_tensor)
            centroids = z_coord
            loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(x_hist, x_target, d_t_predict=min(D_T, D_MAX))
        action, locus, surprises = controller.get_action(
            None, history[-1], info, z_pred_coord, z_target_coord, z_pred_dyn, z_target_dyn, D_T, centroids)
        for c in range(D_T):
            surprise_log_coord.append(torch.mean((z_pred_coord[:, c] - z_target_coord[:, c])**2).item())
            surprise_log_dyn.append(torch.mean((z_pred_dyn[:, c] - z_target_dyn[:, c])**2).item())
        obs, info = env.step(action)
        history.append(obs)
        centroids_np = centroids[0, :D_T].cpu().numpy()
        ch2obj = get_channel_to_obj_mapping(centroids_np, info["positions"], D_T)
        attended_obj = ch2obj.get(locus, -1)
        target_centroid = centroids_np[locus]
        tracking_errors.append(abs(info["pointer_pos"] - target_centroid))
        attended_per_step.append((step, attended_obj))
        is_collision, max_change_obj = detect_collision(info, prev_velocities)
        if is_collision:
            collision_events.append((step, max_change_obj))
        if PERTURB_STEP <= step <= PERTURB_STEP + 99:
            perturbation_attended.append(1 if attended_obj == 0 else 0)
        prev_velocities = info["velocities"].copy()
    post_coll_vb = compute_selectivity_vb(collision_events, attended_per_step, EVAL_STEPS)
    return {
        "tracking_error": float(np.mean(tracking_errors)),
        "selectivity_vb": post_coll_vb,
        "perturbation_selectivity": float(np.mean(perturbation_attended)) if perturbation_attended else 0.0,
        "collision_count": len(collision_events),
        "mean_surprise_coord": float(np.mean(surprise_log_coord)) if surprise_log_coord else 0.0,
        "mean_surprise_dyn": float(np.mean(surprise_log_dyn)) if surprise_log_dyn else 0.0,
    }

def run_oracle(seed):
    env = PhysicsSandbox(N=2, seed=seed)
    controller = CLTSMotorController()
    obs = env.reset()
    history = collections.deque(maxlen=HISTORY_LEN)
    history.append(obs)
    prev_velocities = env.velocities.copy()
    prev_positions = env.positions.copy()
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        prev_positions = info["positions"].copy()
        prev_velocities = info["velocities"].copy()
    # Initialize prev_positions from info after warmup
    prev_positions = info["positions"].copy()
    prev_velocities = info["velocities"].copy()
    tracking_errors, collision_events, attended_per_step, perturbation_attended = [], [], [], []
    surprise_log_coord, surprise_log_dyn = [], []
    for step in range(EVAL_STEPS):
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER
        positions = info["positions"]
        velocities = info["velocities"]
        colors = info["colors"]
        # Construct ground-truth tensors
        z_coord = torch.zeros(1, D_MAX)
        z_coord[0, :min(D_T, len(positions))] = torch.tensor(positions[:min(D_T, len(positions))], dtype=torch.float32)
        z_dyn = torch.zeros(1, D_MAX)
        for i in range(min(D_T, len(colors))):
            z_dyn[0, i] = torch.tensor(np.mean(colors[i]), dtype=torch.float32)
        z_pred_coord = torch.zeros(1, D_MAX)
        z_pred_coord[0, :min(D_T, len(prev_positions))] = torch.tensor(
            prev_positions[:min(D_T, len(prev_positions))] + prev_velocities[:min(D_T, len(prev_velocities))] * 1.0,
            dtype=torch.float32)
        z_pred_dyn = z_dyn.clone()
        centroids = z_coord.clone()
        action, locus, surprises = controller.get_action(
            None, history[-1], info, z_pred_coord, z_coord, z_pred_dyn, z_dyn, D_T, centroids)
        # Log surprise decomposition
        for c in range(D_T):
            surprise_log_coord.append(torch.mean((z_pred_coord[:, c] - z_coord[:, c])**2).item())
            surprise_log_dyn.append(0.0)  # ORACLE has zero dyn error by construction
        obs, info = env.step(action)
        history.append(obs)
        centroids_np = centroids[0, :D_T].cpu().numpy()
        ch2obj = get_channel_to_obj_mapping(centroids_np, info["positions"], D_T)
        attended_obj = ch2obj.get(locus, -1)
        target_centroid = centroids_np[locus]
        tracking_errors.append(abs(info["pointer_pos"] - target_centroid))
        attended_per_step.append((step, attended_obj))
        is_collision, max_change_obj = detect_collision(info, prev_velocities)
        if is_collision:
            collision_events.append((step, max_change_obj))
        if PERTURB_STEP <= step <= PERTURB_STEP + 99:
            perturbation_attended.append(1 if attended_obj == 0 else 0)
        prev_positions = info["positions"].copy()
        prev_velocities = info["velocities"].copy()
    post_coll_vb = compute_selectivity_vb(collision_events, attended_per_step, EVAL_STEPS)
    return {
        "tracking_error": float(np.mean(tracking_errors)),
        "selectivity_vb": post_coll_vb,
        "perturbation_selectivity": float(np.mean(perturbation_attended)) if perturbation_attended else 0.0,
        "collision_count": len(collision_events),
        "mean_surprise_coord": float(np.mean(surprise_log_coord)) if surprise_log_coord else 0.0,
        "mean_surprise_dyn": 0.0,  # ORACLE: zero dyn surprise by construction
    }

def compute_selectivity_vb(collision_events, attended_per_step, eval_steps):
    post_coll_steps = []
    for coll_step, max_change_obj in collision_events:
        for s in range(coll_step + 1, coll_step + POST_COLLISION_WINDOW + 1):
            if 0 <= s < eval_steps:
                post_coll_steps.append((s, max_change_obj))
    if not post_coll_steps:
        return 0.0
    total, count = 0, 0
    step_to_attn = {s: obj for s, obj in attended_per_step}
    for s, max_obj in post_coll_steps:
        attn_obj = step_to_attn.get(s)
        if attn_obj is not None:
            total += 1
            if attn_obj == max_obj:
                count += 1
    return count / max(total, 1)

def bootstrap_g(learned_vals, random_vals, oracle_vals, n_bootstrap=10000):
    n = len(learned_vals)
    rng = np.random.RandomState(42)
    gs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        l = np.mean(learned_vals[idx])
        r = np.mean(random_vals[idx])
        o = np.mean(oracle_vals[idx])
        denom = o - r
        if abs(denom) < 1e-10:
            gs.append(np.nan)
        else:
            gs.append((l - r) / denom)
    gs_arr = np.array(gs)
    gs_valid = gs_arr[~np.isnan(gs_arr)]
    if len(gs_valid) == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmean(gs_arr)), float(np.percentile(gs_valid, 2.5)), float(np.percentile(gs_valid, 97.5))

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    print(f"Seeds: {SEEDS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"d_t={D_T}, d_max={D_MAX}, eval_steps={EVAL_STEPS}")
    all_results = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            label = f"{condition} | seed={seed}"
            print(f"[RUN] {label} ...", end=" ", flush=True)
            if condition == "random":
                result = run_random(seed)
            elif condition == "learned_vicreg":
                ckpt = os.path.join(CKPT_DIR, f"a_vicreg-only_control_seed{seed}.pt")
                result = run_learned(seed, ckpt)
            elif condition == "learned_sfa":
                ckpt = os.path.join(CKPT_DIR, f"b_sfavicreg,_sfa_5.0_seed{seed}.pt")
                result = run_learned(seed, ckpt)
            elif condition == "oracle":
                result = run_oracle(seed)
            else:
                raise ValueError(f"Unknown condition: {condition}")
            row = {"condition": condition, "seed": seed, **result}
            all_results.append(row)
            print(f"sel_vb={result['selectivity_vb']:.4f} track={result['tracking_error']:.2f} "
                  f"pert={result['perturbation_selectivity']:.4f} coll={result['collision_count']} "
                  f"s_coord={result['mean_surprise_coord']:.4f} s_dyn={result['mean_surprise_dyn']:.4f}")

    df = pd.DataFrame(all_results)
    per_run_path = os.path.join(RESULTS_DIR, "per_run.csv")
    df.to_csv(per_run_path, index=False)
    print(f"\nSaved per-run results to {per_run_path}")

    # ─── Summary Statistics ───
    summary_rows = []
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        row = {
            "condition": condition,
            "n_seeds": len(sub),
            "mean_selectivity_vb": sub["selectivity_vb"].mean(),
            "std_selectivity_vb": sub["selectivity_vb"].std(ddof=1) if len(sub) > 1 else 0.0,
            "mean_tracking_error": sub["tracking_error"].mean(),
            "std_tracking_error": sub["tracking_error"].std(ddof=1) if len(sub) > 1 else 0.0,
            "mean_perturbation_sel": sub["perturbation_selectivity"].mean(),
            "std_perturbation_sel": sub["perturbation_selectivity"].std(ddof=1) if len(sub) > 1 else 0.0,
            "mean_surprise_coord": sub["mean_surprise_coord"].mean(),
            "mean_surprise_dyn": sub["mean_surprise_dyn"].mean(),
        }
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # ─── Compute g ratios ───
    random_sel = df[df["condition"] == "random"].set_index("seed")["selectivity_vb"]
    vicreg_sel = df[df["condition"] == "learned_vicreg"].set_index("seed")["selectivity_vb"]
    sfa_sel = df[df["condition"] == "learned_sfa"].set_index("seed")["selectivity_vb"]
    oracle_sel = df[df["condition"] == "oracle"].set_index("seed")["selectivity_vb"]

    # Align by seed
    common_seeds = sorted(set(random_sel.index) & set(vicreg_sel.index) & set(sfa_sel.index) & set(oracle_sel.index))
    r_vals = random_sel[common_seeds].values
    vr_vals = vicreg_sel[common_seeds].values
    sfa_vals = sfa_sel[common_seeds].values
    o_vals = oracle_sel[common_seeds].values

    # Ordering sanity check
    r_mean = np.mean(r_vals)
    o_mean = np.mean(o_vals)
    vr_mean = np.mean(vr_vals)
    sfa_mean = np.mean(sfa_vals)

    print("\n" + "="*60)
    print("ORACLE BRACKET ANALYSIS")
    print("="*60)
    print(f"\nRaw triple (selectivity_vb):")
    print(f"  RANDOM:         {r_mean:.4f} +/- {np.std(r_vals, ddof=1):.4f}")
    print(f"  LEARNED-VICReg: {vr_mean:.4f} +/- {np.std(vr_vals, ddof=1):.4f}")
    print(f"  LEARNED-SFA:    {sfa_mean:.4f} +/- {np.std(sfa_vals, ddof=1):.4f}")
    print(f"  ORACLE:         {o_mean:.4f} +/- {np.std(o_vals, ddof=1):.4f}")

    ordering_ok = (o_mean >= r_mean)
    print(f"\nOrdering sanity check: ORACLE({o_mean:.4f}) >= RANDOM({r_mean:.4f}) -> {ordering_ok}")

    # Branch (c) check
    oracle_random_gap = o_mean - r_mean
    branch_c = abs(oracle_random_gap) < 0.10
    print(f"Branch (c) check: |ORACLE - RANDOM| = |{oracle_random_gap:.4f}| < 0.10 -> {branch_c}")

    # g ratios
    if not branch_c and ordering_ok:
        g_vr_mean, g_vr_lo, g_vr_hi = bootstrap_g(vr_vals, r_vals, o_vals)
        g_sfa_mean, g_sfa_lo, g_sfa_hi = bootstrap_g(sfa_vals, r_vals, o_vals)
        print(f"\ng_vicreg = {g_vr_mean:.4f} (95% CI: [{g_vr_lo:.4f}, {g_vr_hi:.4f}])")
        print(f"g_sfa    = {g_sfa_mean:.4f} (95% CI: [{g_sfa_lo:.4f}, {g_sfa_hi:.4f}])")

        # Branch assignment
        for name, g_mean, g_lo, g_hi in [("VICReg", g_vr_mean, g_vr_lo, g_vr_hi), ("SFA", g_sfa_mean, g_sfa_lo, g_sfa_hi)]:
            if g_mean >= 0.70 and g_lo >= 0.50:
                branch = "(a) consistent with sufficiency"
            elif g_mean <= 0.20:
                branch = "(b) representation provably limits behavior"
            elif 0.20 < g_mean < 0.70:
                branch = "(d) partial sufficiency"
            else:
                branch = "ambiguous"
            print(f"  {name}: {branch}")
    elif branch_c:
        print("\nBRANCH (c) FIRED: Task/motor protocol is the bottleneck, NOT perception.")
        print("The behavioral-pivot strategy is invalidated for this protocol.")
    else:
        print("\nOrdering violated — g ratio is not meaningful.")

    # ─── Per-seed table for primary metric ───
    print(f"\nPer-seed selectivity_vb:")
    print(f"{'Seed':>6} | {'RANDOM':>8} | {'VICReg':>8} | {'SFA':>8} | {'ORACLE':>8}")
    for s in common_seeds:
        print(f"{s:>6} | {random_sel[s]:.4f}   | {vicreg_sel[s]:.4f}   | {sfa_sel[s]:.4f}   | {oracle_sel[s]:.4f}")

    # ─── Surprise decomposition ───
    print(f"\nSurprise decomposition (mean across seeds):")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        print(f"  {condition:>15}: coord={sub['mean_surprise_coord'].mean():.4f}, dyn={sub['mean_surprise_dyn'].mean():.4f}")

    # ─── Secondary metrics ───
    print(f"\nSecondary metrics (mean +/- std):")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        print(f"  {condition:>15}: tracking={sub['tracking_error'].mean():.2f}+/-{sub['tracking_error'].std(ddof=1):.2f}, "
              f"pert_sel={sub['perturbation_selectivity'].mean():.4f}+/-{sub['perturbation_selectivity'].std(ddof=1):.4f}")

    # ─── Write Analysis Report ───
    analysis_lines = []
    analysis_lines.append("# Iter_033 — Three-Condition Oracle Bracket Analysis\n\n")
    analysis_lines.append("## Raw Triple (Primary Metric: Post-Collision Selectivity V-B)\n\n")
    analysis_lines.append("| Condition | Mean | Std | n |\n")
    analysis_lines.append("|----------|------|-----|---|\n")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        analysis_lines.append(f"| {condition} | {sub['selectivity_vb'].mean():.4f} | {sub['selectivity_vb'].std(ddof=1):.4f} | {len(sub)} |\n")
    analysis_lines.append(f"\n**Ordering sanity check:** ORACLE({o_mean:.4f}) >= RANDOM({r_mean:.4f}) = {ordering_ok}\n\n")
    analysis_lines.append(f"**Branch (c) check:** |ORACLE - RANDOM| = |{oracle_random_gap:.4f}| < 0.10 = {branch_c}\n\n")

    if not branch_c and ordering_ok:
        analysis_lines.append("## g-Ratio Analysis\n\n")
        analysis_lines.append(f"| Arm | g | 95% CI Lower | 95% CI Upper | Branch |\n")
        analysis_lines.append(f"|-----|---|-------------|-------------|--------|\n")
        for name, g_mean, g_lo, g_hi in [("VICReg", g_vr_mean, g_vr_lo, g_vr_hi), ("SFA", g_sfa_mean, g_sfa_lo, g_sfa_hi)]:
            if g_mean >= 0.70 and g_lo >= 0.50:
                branch = "(a)"
            elif g_mean <= 0.20:
                branch = "(b)"
            elif 0.20 < g_mean < 0.70:
                branch = "(d)"
            else:
                branch = "ambiguous"
            analysis_lines.append(f"| {name} | {g_mean:.4f} | {g_lo:.4f} | {g_hi:.4f} | {branch} |\n")
    elif branch_c:
        analysis_lines.append("## BRANCH (c) FIRED\n\n")
        analysis_lines.append("The task or motor protocol is the bottleneck, NOT perception.\n")
        analysis_lines.append("The behavioral-pivot strategy is invalidated for this protocol.\n")

    analysis_lines.append("\n## Per-Seed Primary Metric\n\n")
    analysis_lines.append("| Seed | RANDOM | VICReg | SFA | ORACLE |\n")
    analysis_lines.append("|------|--------|--------|-----|--------|\n")
    for s in common_seeds:
        analysis_lines.append(f"| {s} | {random_sel[s]:.4f} | {vicreg_sel[s]:.4f} | {sfa_sel[s]:.4f} | {oracle_sel[s]:.4f} |\n")

    analysis_lines.append("\n## Surprise Decomposition\n\n")
    analysis_lines.append("| Condition | Mean Surprise Coord | Mean Surprise Dyn |\n")
    analysis_lines.append("|-----------|--------------------|------------------|\n")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        analysis_lines.append(f"| {condition} | {sub['mean_surprise_coord'].mean():.4f} | {sub['mean_surprise_dyn'].mean():.4f} |\n")

    analysis_lines.append("\n## Secondary Metrics\n\n")
    analysis_lines.append("| Condition | Tracking Error (mean) | Tracking Error (std) | Pert Sel (mean) | Pert Sel (std) |\n")
    analysis_lines.append("|-----------|----------------------|---------------------|-----------------|---------------|\n")
    for condition in CONDITIONS:
        sub = df[df["condition"] == condition]
        te_m = sub['tracking_error'].mean()
        te_s = sub['tracking_error'].std(ddof=1)
        ps_m = sub['perturbation_selectivity'].mean()
        ps_s = sub['perturbation_selectivity'].std(ddof=1)
        analysis_lines.append(f"| {condition} | {te_m:.2f} | {te_s:.2f} | {ps_m:.4f} | {ps_s:.4f} |\n")

    analysis_path = os.path.join(RESULTS_DIR, "analysis.md")
    with open(analysis_path, "w") as f:
        f.writelines(analysis_lines)
    print(f"\nSaved analysis to {analysis_path}")
    print(f"\nTotal runs: {len(all_results)}")

if __name__ == "__main__":
    main()
