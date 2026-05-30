#!/usr/bin/env python3
"""
Iter_031 Part B: Protocol Calibration — CLTS Evaluation on Collision-Sparse N=2 Environment.

Tests 5 seeds × 3 conditions (surprise-driven, frozen, random) × 2 d_t settings (2 and 3).
Includes mass perturbation at step 1000 (1.5× multiplier on object 0).
"""
import os
import sys
import csv
import json
import collections
import warnings
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models_separate_dyn import NonParametricJEPASpatialSeparateDyn
from src.motor import CLTSMotorController
from src.environment import PhysicsSandbox

# ─── Configuration ───────────────────────────────────────────────────────────
SEEDS = [7, 31, 97, 113, 137]
DT_SETTINGS = [2, 3]
CONDITIONS = ["surprise-driven", "frozen", "random"]
EVAL_STEPS = 2000
PERTURB_STEP = 1000
MASS_MULTIPLIER = 1.5
CHECKPOINT_DIR = "archive/iter_029/results/checkpoints"
RESULTS_DIR = "archive/iter_031/results"
WARMUP_STEPS = 3
HISTORY_LEN = 4

# Collision detection parameters
COLLISION_DIST_THRESHOLD = 4.0  # extra margin beyond radii sum
COLLISION_VELOCITY_CHANGE_THRESHOLD = 1.0
POST_COLLISION_WINDOW = 15  # steps 1..15 after collision


# ─── Helper Functions ─────────────────────────────────────────────────────────
def get_channel_to_obj_mapping_robust(centroids, positions, d_t):
    mapping = {}
    centroid_vals = centroids[0, :d_t].cpu().numpy()
    for ch in range(d_t):
        val = centroid_vals[ch]
        closest_obj = int(np.argmin(np.abs(positions - val)))
        mapping[ch] = closest_obj
    return mapping


def detect_collision(info, prev_velocities):
    """Detect object-object collision event.
    Returns: (is_collision: bool, max_change_obj: int or None)"""
    pos_diff = abs(info["positions"][0] - info["positions"][1])
    radii_sum = info["radii"][0] + info["radii"][1]
    close_enough = pos_diff < (radii_sum + COLLISION_DIST_THRESHOLD)

    if not close_enough:
        return False, None

    # Check velocity change
    vel_changes = np.abs(info["velocities"] - prev_velocities)
    max_change = np.max(vel_changes)
    if max_change > COLLISION_VELOCITY_CHANGE_THRESHOLD:
        max_change_obj = int(np.argmax(vel_changes))
        return True, max_change_obj

    return False, None


def run_single_evaluation(model, seed, condition, d_t, device):
    """Run one evaluation: one seed, one condition, one d_t."""
    env = PhysicsSandbox(N=2, seed=seed)
    d_t_eval = d_t

    # Create controller
    controller = CLTSMotorController()

    # Reset environment and pre-fill history
    obs = env.reset()
    history = collections.deque(maxlen=HISTORY_LEN)
    history.append(obs)

    # Warmup: 3 steps of zero action to fill deque to length 4
    prev_velocities = env.velocities.copy()
    for _ in range(WARMUP_STEPS):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)
        prev_velocities = info["velocities"].copy()

    # Metrics storage
    tracking_errors = []
    collision_events = []  # list of (step, max_change_obj)
    attended_objects_per_step = []  # (step, attended_obj_index)
    perturbation_attended = []  # for steps 1000-1099: was object 0 attended?

    for step in range(EVAL_STEPS):
        # Mass perturbation at step 1000
        if step == PERTURB_STEP:
            env.masses[0] *= MASS_MULTIPLIER

        # Run model on the current observation (last in history)
        obs = history[3]
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            z_coord, z_dyn = model.encoder(obs_tensor)
            centroids = z_coord  # (1, d_max)

        # Predict next frame for CLTS metrics
        x_hist = torch.from_numpy(np.stack(list(history)[:3], axis=0)).float().unsqueeze(0).to(device)
        x_target = torch.from_numpy(history[3]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(
                x_hist, x_target, d_t_predict=min(d_t_eval, model.d_max)
            )

        # ─── Locus override (before controller.get_action) ───
        if condition == "frozen":
            controller.token_locus = 0
            controller.attention_cooldown = controller.attention_cooldown_max
        elif condition == "random":
            controller.token_locus = int(np.random.randint(0, d_t_eval))
            controller.attention_cooldown = controller.attention_cooldown_max

        # Call controller.get_action:
        action, locus, surprises = controller.get_action(
            model, history[3], info,
            z_pred_coord, z_target_coord,
            z_pred_dyn, z_target_dyn,
            d_t_eval, centroids,
        )

        # Take action -> get new info with updated velocities for collision detection
        obs, info = env.step(action)
        history.append(obs)

        # Compute tracking error: abs(pointer_pos - centroid of locus)
        target_centroid = centroids[0, locus].item()
        tracking_err = abs(info["pointer_pos"] - target_centroid)
        tracking_errors.append(tracking_err)

        # Detect collision (compare new velocities to previous step's)
        is_collision, max_change_obj = detect_collision(info, prev_velocities)
        if is_collision:
            collision_events.append((step, max_change_obj))

        # Map channel to object using robust mapping
        ch2obj = get_channel_to_obj_mapping_robust(centroids, info["positions"], d_t_eval)
        attended_obj = ch2obj.get(locus, -1)
        attended_objects_per_step.append((step, attended_obj))

        # Perturbation tracking (steps 1000-1099)
        if step >= PERTURB_STEP and step <= PERTURB_STEP + 99:
            perturbation_attended.append(1 if attended_obj == 0 else 0)

        # Update previous velocities for next step's collision detection
        prev_velocities = info["velocities"].copy()

    # ─── Compute Metrics ──────────────────────────────────────────────────

    # 1. Tracking Error (mean)
    mean_tracking_error = float(np.mean(tracking_errors))

    # 2. Collision count per 100 steps (divide by 20 since 2000 steps = 20 periods of 100)
    collision_count = len(collision_events)
    collisions_per_100 = collision_count / 20.0

    # 3. Post-collision attention selectivity
    # Build set of post-collision steps for each collision
    post_collision_steps = []  # list of (step, any_colliding_obj_indices, max_change_obj)
    for coll_step, max_change_obj in collision_events:
        for s in range(coll_step + 1, coll_step + POST_COLLISION_WINDOW + 1):
            if 0 <= s < EVAL_STEPS:
                # Find all colliding objects (both objects are "colliding")
                post_collision_steps.append((s, [0, 1], max_change_obj))

    # Version A: Any colliding object attended
    if post_collision_steps:
        version_a_count = 0
        version_a_total = 0
        version_b_count = 0
        version_b_total = 0
        for s, any_colliding, max_obj in post_collision_steps:
            # Find attended object for step s
            attn_obj_at_s = None
            for (step_idx, attn_obj) in attended_objects_per_step:
                if step_idx == s:
                    attn_obj_at_s = attn_obj
                    break
            if attn_obj_at_s is not None:
                # Version A: attended object is in [0, 1]
                version_a_total += 1
                if attn_obj_at_s in any_colliding:
                    version_a_count += 1
                # Version B: attended object matches max_change_obj
                version_b_total += 1
                if attn_obj_at_s == max_obj:
                    version_b_count += 1
        selectivity_version_a = version_a_count / max(version_a_total, 1)
        selectivity_version_b = version_b_count / max(version_b_total, 1)
    else:
        selectivity_version_a = 0.0
        selectivity_version_b = 0.0

    # 4. Perturbation attention selectivity
    if len(perturbation_attended) > 0:
        perturbation_selectivity = float(np.mean(perturbation_attended))
    else:
        perturbation_selectivity = 0.0

    return {
        "mean_tracking_error": mean_tracking_error,
        "collision_count": collision_count,
        "collisions_per_100": collisions_per_100,
        "selectivity_version_a": selectivity_version_a,
        "selectivity_version_b": selectivity_version_b,
        "perturbation_selectivity": perturbation_selectivity,
        "post_coll_steps_total_a": selectivity_version_a != 0.0,  # dummy for debugging
        "post_coll_steps_count_a": version_a_total if post_collision_steps else 0,
        "post_coll_steps_count_b": version_b_total if post_collision_steps else 0,
        "perturbation_steps": len(perturbation_attended),
        "tracking_errors_raw": tracking_errors,
        "collision_events_detail": collision_events,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(4)
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"d_t settings: {DT_SETTINGS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Evaluation steps per run: {EVAL_STEPS}")
    print(f"Mass perturbation: at step {PERTURB_STEP}, multiply object 0 mass by {MASS_MULTIPLIER}")
    print()

    # ─── Model Loading ────────────────────────────────────────────────────
    def build_model():
        model = NonParametricJEPASpatialSeparateDyn(
            d_max=8, h=3, k=4, cooldown=300, stabilization_period=100,
            pos_encoding="none", primary_objective="jepa",
            sfa_weight=25.0, gdasr_log_only=True,
            dyn_readout="mean", sub_features=1, dyn_source="spatial",
            mask_dyn_sim=True, coord_vicreg=True,
        )
        model.d_t = 3  # matching the checkpoint
        return model

    print("Loading checkpoints...")
    models_per_seed = {}
    for seed in SEEDS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"a_vicreg-only_control_seed{seed}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = build_model()
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        model.to(device)
        models_per_seed[seed] = model
        print(f"  Loaded seed={seed} from {ckpt_path}")
    print()

    # ─── Run Evaluation Sweep ─────────────────────────────────────────────
    all_results = []

    for d_t_val in DT_SETTINGS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                run_label = f"d_t={d_t_val} | seed={seed} | {condition}"
                print(f"[RUN] {run_label} ...")
                model = models_per_seed[seed]
                model.d_t = d_t_val  # set active channel count for evaluation

                result = run_single_evaluation(model, seed, condition, d_t_val, device)

                row = {
                    "d_t": d_t_val,
                    "seed": seed,
                    "condition": condition,
                    "mean_tracking_error": result["mean_tracking_error"],
                    "collision_count": result["collision_count"],
                    "collisions_per_100": result["collisions_per_100"],
                    "selectivity_version_a": result["selectivity_version_a"],
                    "selectivity_version_b": result["selectivity_version_b"],
                    "perturbation_selectivity": result["perturbation_selectivity"],
                    "post_coll_total_a_steps": result["post_coll_steps_count_a"],
                    "post_coll_total_b_steps": result["post_coll_steps_count_b"],
                    "perturbation_window_steps": result["perturbation_steps"],
                }
                all_results.append(row)
                print(f"  -> tracking_err={result['mean_tracking_error']:.2f}, "
                      f"collisions={result['collision_count']}, "
                      f"sel_b={result['selectivity_version_b']:.4f}, "
                      f"pert_sel={result['perturbation_selectivity']:.4f}")

    print(f"\nTotal runs completed: {len(all_results)}")

    # ─── Write Per-Seed CSV ───────────────────────────────────────────────
    per_seed_path = os.path.join(RESULTS_DIR, "partB_per_seed.csv")
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(per_seed_path, index=False, encoding="utf-8")
    print(f"Saved per-seed results to {per_seed_path}")

    # ─── Compute Summary Statistics ──────────────────────────────────────
    summary_rows = []
    for d_t_val in DT_SETTINGS:
        sub = df_all[df_all["d_t"] == d_t_val]
        for condition in CONDITIONS:
            sub_cond = sub[sub["condition"] == condition]
            n = len(sub_cond)
            if n == 0:
                continue
            row = {
                "d_t": d_t_val,
                "condition": condition,
                "n_seeds": n,
                "mean_tracking_error": sub_cond["mean_tracking_error"].mean(),
                "std_tracking_error": sub_cond["mean_tracking_error"].std(ddof=1) if n > 1 else 0.0,
                "mean_collisions_per_100": sub_cond["collisions_per_100"].mean(),
                "std_collisions_per_100": sub_cond["collisions_per_100"].std(ddof=1) if n > 1 else 0.0,
                "mean_selectivity_version_a": sub_cond["selectivity_version_a"].mean(),
                "std_selectivity_version_a": sub_cond["selectivity_version_a"].std(ddof=1) if n > 1 else 0.0,
                "mean_selectivity_version_b": sub_cond["selectivity_version_b"].mean(),
                "std_selectivity_version_b": sub_cond["selectivity_version_b"].std(ddof=1) if n > 1 else 0.0,
                "mean_perturbation_selectivity": sub_cond["perturbation_selectivity"].mean(),
                "std_perturbation_selectivity": sub_cond["perturbation_selectivity"].std(ddof=1) if n > 1 else 0.0,
            }
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "partB_summary.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Saved summary results to {summary_path}")

    # ─── Gate Evaluation (G1-G3) ─────────────────────────────────────────
    gate_results = {}

    for d_t_val in DT_SETTINGS:
        d_t_name = f"d_t={d_t_val}"
        sub = df_summary[(df_summary["d_t"] == d_t_val)]

        sd_row = sub[sub["condition"] == "surprise-driven"].iloc[0]
        frozen_row = sub[sub["condition"] == "frozen"].iloc[0]
        random_row = sub[sub["condition"] == "random"].iloc[0]

        # G1_tracking: surprise_driven ≤ random_mean - 1*random_std
        random_track_mean = random_row["mean_tracking_error"]
        random_track_std = random_row["std_tracking_error"]
        sd_track = sd_row["mean_tracking_error"]
        g1_threshold = random_track_mean - 1.0 * random_track_std
        g1_pass = sd_track <= g1_threshold

        # G2_collision_selectivity (Version B): sd ≥ random_max × 1.5
        # Use the mean of random as baseline
        random_sel_b_mean = random_row["mean_selectivity_version_b"]
        sd_sel_b = sd_row["mean_selectivity_version_b"]
        g2_threshold = random_sel_b_mean * 1.5
        g2_pass = sd_sel_b >= g2_threshold

        # G3_perturbation_selectivity: sd ≥ random × 1.5
        random_pert_mean = random_row["mean_perturbation_selectivity"]
        sd_pert = sd_row["mean_perturbation_selectivity"]
        g3_threshold = random_pert_mean * 1.5
        g3_pass = sd_pert >= g3_threshold

        gate_results[d_t_name] = {
            "g1_pass": g1_pass,
            "g1_sd_tracking": sd_track,
            "g1_random_track_mean": random_track_mean,
            "g1_random_track_std": random_track_std,
            "g1_threshold": g1_threshold,
            "g2_pass": g2_pass,
            "g2_sd_sel_b": sd_sel_b,
            "g2_random_sel_b_mean": random_sel_b_mean,
            "g2_threshold": g2_threshold,
            "g3_pass": g3_pass,
            "g3_sd_pert": sd_pert,
            "g3_random_pert_mean": random_pert_mean,
            "g3_threshold": g3_threshold,
            "frozen_row": frozen_row.to_dict(),
            "random_row": random_row.to_dict(),
            "sd_row": sd_row.to_dict(),
        }

        print(f"\n[Gate Evaluation for {d_t_name}]")
        print(f"  G1_tracking: sd={sd_track:.2f} <= {g1_threshold:.2f} "
              f"(random_mean={random_track_mean:.2f}, random_std={random_track_std:.2f}) -> {'PASS' if g1_pass else 'FAIL'}")
        print(f"  G2_collision_sel(B): sd={sd_sel_b:.4f} >= {g2_threshold:.4f} "
              f"(random_sel_b_mean={random_sel_b_mean:.4f}) -> {'PASS' if g2_pass else 'FAIL'}")
        print(f"  G3_perturbation_sel: sd={sd_pert:.4f} >= {g3_threshold:.4f} "
              f"(random_pert_mean={random_pert_mean:.4f}) -> {'PASS' if g3_pass else 'FAIL'}")

    # Determine best d_t setting
    best_dt = None
    best_passes = -1
    for d_t_val in DT_SETTINGS:
        d_t_name = f"d_t={d_t_val}"
        passes = sum([
            gate_results[d_t_name]["g1_pass"],
            gate_results[d_t_name]["g2_pass"],
            gate_results[d_t_name]["g3_pass"],
        ])
        if passes > best_passes:
            best_passes = passes
            best_dt = d_t_val

    # ─── Write Analysis Report ───────────────────────────────────────────
    analysis_path = os.path.join(RESULTS_DIR, "partB_analysis.md")
    lines = []
    lines.append("# Iter_031 Part B — CLTS Protocol Calibration Analysis\n\n")
    lines.append("## Overview\n\n")
    lines.append("This report calibrates the CLTS evaluation protocol on a collision-sparse N=2 environment\n")
    lines.append("with a subtle mass perturbation at step 1000 (1.5× multiplier on object 0).\n")
    lines.append("Results cover 5 seeds (`[7, 31, 97, 113, 137]`) and 3 conditions (surprise-driven, frozen, random)\n")
    lines.append("for both `d_t = 2` and `d_t = 3` settings.\n\n")
    lines.append("Checkpoints: `archive/iter_029/results/checkpoints/a_vicreg-only_control_seed{}.pt`\n\n")

    lines.append("## Baselines: Frozen and Random Conditions\n\n")
    for d_t_val in DT_SETTINGS:
        lines.append(f"### d_t = {d_t_val}\n\n")
        sub = df_summary[(df_summary["d_t"] == d_t_val)]
        for cond in ["frozen", "random"]:
            crow = sub[sub["condition"] == cond].iloc[0]
            lines.append(f"**{cond}** ({len(crow)} seeds):\n")
            lines.append(f"- Tracking error: {crow['mean_tracking_error']:.2f} ± {crow['std_tracking_error']:.2f} px\n")
            lines.append(f"- Collisions per 100 steps: {crow['mean_collisions_per_100']:.2f} ± {crow['std_collisions_per_100']:.2f}\n")
            lines.append(f"- Collision selectivity (Version A): {crow['mean_selectivity_version_a']:.4f} ± {crow['std_selectivity_version_a']:.4f}\n")
            lines.append(f"- Collision selectivity (Version B): {crow['mean_selectivity_version_b']:.4f} ± {crow['std_selectivity_version_b']:.4f}\n")
            lines.append(f"- Perturbation selectivity: {crow['mean_perturbation_selectivity']:.4f} ± {crow['std_perturbation_selectivity']:.4f}\n\n")

    lines.append("## Surprise-Driven Performance\n\n")
    for d_t_val in DT_SETTINGS:
        lines.append(f"### d_t = {d_t_val}\n\n")
        sub = df_summary[(df_summary["d_t"] == d_t_val)]
        sdrow = sub[sub["condition"] == "surprise-driven"].iloc[0]
        lines.append(f"- Tracking error: {sdrow['mean_tracking_error']:.2f} ± {sdrow['std_tracking_error']:.2f} px\n")
        lines.append(f"- Collisions per 100 steps: {sdrow['mean_collisions_per_100']:.2f} ± {sdrow['std_collisions_per_100']:.2f}\n")
        lines.append(f"- Collision selectivity (Version A): {sdrow['mean_selectivity_version_a']:.4f} ± {sdrow['std_selectivity_version_a']:.4f}\n")
        lines.append(f"- Collision selectivity (Version B): {sdrow['mean_selectivity_version_b']:.4f} ± {sdrow['std_selectivity_version_b']:.4f}\n")
        lines.append(f"- Perturbation selectivity: {sdrow['mean_perturbation_selectivity']:.4f} ± {sdrow['std_perturbation_selectivity']:.4f}\n\n")

        # Per-seed table
        lines.append("| Seed | Tracking Error | Collisions | Sel (V-A) | Sel (V-B) | Pert Sel |\n")
        lines.append("|------|---------------|------------|------------|------------|----------|\n")
        per_seed_rows = df_all[(df_all["d_t"] == d_t_val) & (df_all["condition"] == "surprise-driven")]
        for _, srow in per_seed_rows.iterrows():
            lines.append(f"| {srow['seed']} | {srow['mean_tracking_error']:.2f} | "
                         f"{srow['collisions_per_100']:.2f} | "
                         f"{srow['selectivity_version_a']:.4f} | "
                         f"{srow['selectivity_version_b']:.4f} | "
                         f"{srow['perturbation_selectivity']:.4f} |\n")
        lines.append("\n")

    lines.append("\n## Gate Evaluation (G1–G3)\n\n")
    lines.append("| Gate | d_t | Criterion | Measured (SD) | Threshold | Result |\n")
    lines.append("|------|-----|-----------|----------------|-----------|--------|\n")

    for d_t_val in DT_SETTINGS:
        d_t_name = f"d_t={d_t_val}"
        gr = gate_results[d_t_name]

        g1_line = f"| G1_tracking | {d_t_name} | SD tracking ≤ random_mean − 1*random_std | {gr['g1_sd_tracking']:.2f} | {gr['g1_threshold']:.2f} | {'PASS' if gr['g1_pass'] else 'FAIL'} |"
        lines.append(g1_line + "\n")
        g2_line = f"| G2_collision_sel(B) | {d_t_name} | SD sel_B ≥ random × 1.5 | {gr['g2_sd_sel_b']:.4f} | {gr['g2_threshold']:.4f} | {'PASS' if gr['g2_pass'] else 'FAIL'} |"
        lines.append(g2_line + "\n")
        g3_line = f"| G3_perturbation_sel | {d_t_name} | SD pert_sel ≥ random × 1.5 | {gr['g3_sd_pert']:.4f} | {gr['g3_threshold']:.4f} | {'PASS' if gr['g3_pass'] else 'FAIL'} |"
        lines.append(g3_line + "\n")

    lines.append("\n")

    # Overall gate summary
    lines.append("### Gate Pass Summary\n\n")
    lines.append("| d_t | G1 | G2 | G3 | Total Passed |\n")
    lines.append("|-----|----|----|----|-------------|\n")
    for d_t_val in DT_SETTINGS:
        d_t_name = f"d_t={d_t_val}"
        gr = gate_results[d_t_name]
        total = sum([gr["g1_pass"], gr["g2_pass"], gr["g3_pass"]])
        lines.append(f"| {d_t_name} | {'✓' if gr['g1_pass'] else '✗'} | "
                     f"{'✓' if gr['g2_pass'] else '✗'} | "
                     f"{'✓' if gr['g3_pass'] else '✗'} | "
                     f"{total} |\n")
    lines.append("\n")

    lines.append("## Protocol Recommendation\n\n")
    if best_passes == 3:
        lines.append(f"**Recommended protocol setting: `d_t = {best_dt}`**\n\n")
        lines.append(f"All three gates (G1–G3) pass at d_t = {best_dt}. This setting is recommended for\n")
        lines.append(f"deployment in subsequent experiments.\n\n")
    elif best_passes == 2:
        lines.append(f"**Provisional recommendation: `d_t = {best_dt}`**\n\n")
        lines.append(f"Two out of three gates pass at d_t = {best_dt}. One gate fails. See analysis below.\n\n")
    else:
        lines.append(f"**No setting passes all gates.**\n\n")
        lines.append(f"Best performance: d_t = {best_dt} with {best_passes}/3 gates passing.\n\n")

    # Detailed gate analysis per d_t for both settings
    for d_t_val in DT_SETTINGS:
        d_t_name = f"d_t={d_t_val}"
        gr = gate_results[d_t_name]
        lines.append(f"### Analysis for {d_t_name}\n\n")

        all_pass = gr["g1_pass"] and gr["g2_pass"] and gr["g3_pass"]
        if all_pass:
            lines.append(f"✅ All gates pass at {d_t_name}.\n")
            if d_t_val == 2:
                lines.append(f"  - d_t=2 maps exactly to the 2 physical objects, yielding precise channel-to-object correspondence.\n")
            lines.append("\n")
        else:
            lines.append(f"❌ Not all gates pass at {d_t_name}.\n")
            if not gr["g1_pass"]:
                lines.append(f"  - G1 fails: surprise-driven tracking ({gr['g1_sd_tracking']:.2f} px) is not better than random "
                             f"(mean ± std = {gr['g1_random_track_mean']:.2f} ± {gr['g1_random_track_std']:.2f}). "
                             f"This suggests the CLTS mechanism is not providing a tracking advantage over random.\n")
            if not gr["g2_pass"]:
                lines.append(f"  - G2 fails: post-collision selectivity (V-B) for surprise-driven ({gr['g2_sd_sel_b']:.4f}) "
                             f"is not ≥ 1.5× random baseline ({gr['g2_random_sel_b_mean']:.4f}). "
                             f"The mechanism does not preferentially attend to the max-velocity-change colliding object.\n")
            if not gr["g3_pass"]:
                lines.append(f"  - G3 fails: post-perturbation selectivity for surprise-driven ({gr['g3_sd_pert']:.4f}) "
                             f"is not ≥ 1.5× random baseline ({gr['g3_random_pert_mean']:.4f}). "
                             f"The mechanism does not preferentially attend to the perturbed object.\n")
            lines.append("\n")

    lines.append("---\n")
    lines.append("*Analysis generated by `src/run_iter031_partB.py`\n")

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"\nSaved analysis report to {analysis_path}")


if __name__ == "__main__":
    main()
