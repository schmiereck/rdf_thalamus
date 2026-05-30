#!/usr/bin/env python3
"""
ARM 1 — Integration Smoke-Test for Iteration 030.

Loads pre-trained checkpoints from iter_029 and runs them through CLTSMotorController
closed-loop evaluation. Tests whether representation quality is sufficient for
downstream thalamic gating and motor behavior.

Conditions per seed:
  1. CLTS-SFA: SFA+VICReg checkpoint, surprise-driven attention
  2. CLTS-VICReg: VICReg-only checkpoint, surprise-driven attention
  3. CLTS-Frozen: SFA+VICReg checkpoint, token_locus forced to 0
  4. CLTS-Random: SFA+VICReg checkpoint, token_locus random
"""

import os
import sys
import csv
import json
import random
import collections
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import PhysicsSandbox
from src.models_separate_dyn import NonParametricJEPASpatialSeparateDyn
from src.motor import CLTSMotorController


torch.set_num_threads(2)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
FRESH_SEEDS = [101, 103, 107, 109, 131, 137, 139, 151, 157, 163]
HARD_SEEDS = [53, 71]
ALL_SEEDS = FRESH_SEEDS + HARD_SEEDS
CONDITIONS = ["CLTS-SFA", "CLTS-VICReg", "CLTS-Frozen", "CLTS-Random"]

SFA_CKPT_PATTERN = "archive/iter_029/results/checkpoints/b_sfavicreg,_sfa_5.0_seed{}.pt"
VICREG_CKPT_PATTERN = "archive/iter_029/results/checkpoints/a_vicreg-only_control_seed{}.pt"
NUM_STEPS = 2000
WARMUP_STEPS = 200
EVAL_START = 200
PERTURBATION_STEP = 1000
DEVICE = torch.device("cpu")


def load_model(condition, seed):
    """Create model and load checkpoint. Returns None if checkpoint missing."""
    if condition in ("CLTS-SFA", "CLTS-Frozen", "CLTS-Random"):
        ckpt_path = SFA_CKPT_PATTERN.format(seed)
        primary = "sfa"
        sfa_w = 5.0
    else:  # CLTS-VICReg
        ckpt_path = VICREG_CKPT_PATTERN.format(seed)
        primary = "jepa"
        sfa_w = 25.0

    if not os.path.exists(ckpt_path):
        print(f"    SKIP seed={seed} condition={condition}: checkpoint not found: {ckpt_path}")
        return None

    model = NonParametricJEPASpatialSeparateDyn(
        d_max=8, h=3, k=4, cooldown=300, stabilization_period=100,
        pos_encoding="none", primary_objective=primary,
        sfa_weight=sfa_w, gdasr_log_only=True,
        dyn_readout="mean", sub_features=1, dyn_source="spatial",
        mask_dyn_sim=True, coord_vicreg=True,
    )
    model.d_t = 3
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def get_channel_to_obj_mapping(centroids, positions, d_t):
    """Map each active channel to a physical object by sorting both lists."""
    centroid_vals = centroids[0, :d_t].cpu().numpy()
    channel_order = np.argsort(centroid_vals)
    obj_order = np.argsort(positions)
    mapping = {}
    for rank, ch in enumerate(channel_order):
        mapping[int(ch)] = int(obj_order[rank])
    return mapping


def run_closed_loop(seed, condition):
    """
    Run a single seed + condition through the full 2000-step protocol.
    Returns a dict with all metrics, or None if checkpoint missing.
    """
    model = load_model(condition, seed)
    if model is None:
        return None

    env = PhysicsSandbox(N=3, seed=seed)
    controller = CLTSMotorController(Kp=2.0, Kd=0.5, Kv=0.5)
    controller.reset()

    # Pre-fill history with 4 frames (zero-action warm-up)
    history = collections.deque(maxlen=4)
    obs = env.render()
    history.append(obs)
    info = None
    for _ in range(3):
        obs, info = env.step({"acc": 0.0, "push": False})
        history.append(obs)

    d_t = model.d_t
    prev_velocities = info["velocities"].copy()

    # Evaluation-time storages (steps >= EVAL_START)
    tracking_errors = []
    surprise_values = []
    token_loci = []
    mappings = []          # list of dicts: channel -> object
    collision_events = []  # (eval_step, involved_objects)

    perturbation_applied = False
    perturbation_switch = 0
    perturbation_switch_step = None
    obj0_channel_at_perturbation = None

    for step in range(NUM_STEPS):
        # ---- Build model inputs from observation history ----
        x_hist_np = np.stack(list(history)[:3], axis=0)   # (3, 3, 128)
        x_target_np = history[3]                           # (3, 128)

        x_hist_t = torch.from_numpy(x_hist_np).float().unsqueeze(0).to(DEVICE)
        x_target_t = torch.from_numpy(x_target_np).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            loss_dict, (z_pred_coord, z_pred_dyn), (z_target_coord, z_target_dyn) = model(
                x_hist_t, x_target_t, d_t_predict=d_t
            )
            z_coord, z_dyn = model.encoder(x_target_t)

        centroids = z_coord[:, :d_t]

        # ---- Override attention locus for controls / warm-up ----
        in_warmup = step < WARMUP_STEPS
        if in_warmup:
            controller.token_locus = 0
            controller.attention_cooldown = controller.attention_cooldown_max
        elif condition == "CLTS-Frozen":
            controller.token_locus = 0
            controller.attention_cooldown = controller.attention_cooldown_max
        elif condition == "CLTS-Random":
            controller.token_locus = int(np.random.randint(0, d_t))
            controller.attention_cooldown = controller.attention_cooldown_max

        # ---- Controller action ----
        action, locus, surprises = controller.get_action(
            model, history[3], info,
            z_pred_coord, z_target_coord,
            z_pred_dyn, z_target_dyn,
            d_t, centroids,
        )

        # ---- Mass perturbation at step 1000 ----
        if step == PERTURBATION_STEP:
            env.masses[0] *= 3.0
            env.pointer_pos = env.positions[0] + (
                5.0 if env.positions[0] < 64.0 else -5.0
            )
            action["push"] = True
            perturbation_applied = True
            # Identify which channel tracks object 0 right now
            mapping_now = get_channel_to_obj_mapping(centroids, info["positions"], d_t)
            for ch, obj in mapping_now.items():
                if obj == 0:
                    obj0_channel_at_perturbation = ch
                    break

        # ---- Step environment ----
        obs, info = env.step(action)
        history.append(obs)

        # ---- Metrics collection (post warm-up) ----
        if step >= EVAL_START:
            target_pos = centroids[0, locus].item()
            pointer_pos = info["pointer_pos"]
            tracking_errors.append(abs(pointer_pos - target_pos))
            surprise_values.append(surprises[locus])
            token_loci.append(locus)

            mapping = get_channel_to_obj_mapping(centroids, info["positions"], d_t)
            mappings.append(mapping)

            # Collision detection: object-object contacts only
            # Use velocity change > 2.0 as primary signal, but filter out
            # boundary bounces and require proximity between affected objects.
            vel_change = np.abs(info["velocities"] - prev_velocities)
            changed = np.where(vel_change > 2.0)[0]
            if changed.size > 0:
                involved = set()
                for i in changed:
                    pos_i = info["positions"][i]
                    r_i = info["radii"][i]
                    # Skip boundary bounces
                    if pos_i - r_i < 2.0 or pos_i + r_i > 126.0:
                        continue
                    for j in range(env.N):
                        if i == j:
                            continue
                        pos_j = info["positions"][j]
                        r_j = info["radii"][j]
                        # Also skip if j is at boundary
                        if pos_j - r_j < 2.0 or pos_j + r_j > 126.0:
                            continue
                        dist = abs(pos_i - pos_j)
                        min_dist = r_i + r_j
                        if dist < min_dist + 4.0:
                            involved.add(int(i))
                            involved.add(int(j))
                if involved:
                    collision_events.append((step, sorted(involved)))

            # Perturbation switch check (within 20 steps after step 1000)
            if perturbation_applied and perturbation_switch == 0 and 1000 < step <= 1020:
                ch_obj0 = None
                for ch, obj in mapping.items():
                    if obj == 0:
                        ch_obj0 = ch
                        break
                if ch_obj0 is not None and locus == ch_obj0:
                    perturbation_switch = 1
                    perturbation_switch_step = step

        prev_velocities = info["velocities"].copy()

    # ---- Post-run analysis ----
    # Collision switch rate
    collision_switches = 0
    for coll_step, involved in collision_events:
        switched = False
        # Check steps coll_step+1 through coll_step+15 (but only within eval range)
        for check_step in range(coll_step + 1, min(coll_step + 16, NUM_STEPS)):
            if check_step < EVAL_START:
                continue
            idx = check_step - EVAL_START
            if idx >= len(mappings):
                break
            mapped_obj = mappings[idx].get(token_loci[idx])
            if mapped_obj is not None and mapped_obj in involved:
                switched = True
                break
        if switched:
            collision_switches += 1

    collision_switch_rate = (
        collision_switches / len(collision_events) if collision_events else 0.0
    )

    result = {
        "seed": seed,
        "condition": condition,
        "is_hard_seed": seed in HARD_SEEDS,
        "tracking_error_mean": float(np.mean(tracking_errors)) if tracking_errors else float("nan"),
        "tracking_error_std": float(np.std(tracking_errors)) if tracking_errors else float("nan"),
        "collision_switch_rate": float(collision_switch_rate),
        "collision_switch_count": len(collision_events),
        "perturbation_switch": int(perturbation_switch),
        "perturbation_switch_step": int(perturbation_switch_step) if perturbation_switch_step is not None else None,
        "mean_surprise": float(np.mean(surprise_values)) if surprise_values else float("nan"),
        "num_eval_steps": len(tracking_errors),
    }
    return result


def evaluate_all():
    os.makedirs("archive/iter_030/results", exist_ok=True)

    all_results = []
    total = len(ALL_SEEDS) * len(CONDITIONS)
    idx = 0

    for seed in ALL_SEEDS:
        for condition in CONDITIONS:
            idx += 1
            t0 = time.time()
            print(f"[{idx}/{total}] seed={seed}  condition={condition} ... ", end="", flush=True)
            res = run_closed_loop(seed, condition)
            elapsed = time.time() - t0
            if res is not None:
                all_results.append(res)
                print(f"done in {elapsed:.1f}s | tracking_err={res['tracking_error_mean']:.2f} "
                      f"coll_rate={res['collision_switch_rate']:.2f} "
                      f"pert_switch={res['perturbation_switch']}")
            else:
                print("SKIPPED (no checkpoint)")

    # ---- Write per-seed CSV ----
    per_seed_path = "archive/iter_030/results/arm1_per_seed.csv"
    with open(per_seed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "condition", "is_hard_seed",
            "tracking_error_mean", "tracking_error_std",
            "collision_switch_rate", "collision_switch_count",
            "perturbation_switch", "perturbation_switch_step",
            "mean_surprise", "num_eval_steps",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nWrote per-seed results to {per_seed_path}")

    # ---- Build summary aggregrates ----
    def _mean_std(values):
        arr = np.array(values)
        return float(np.mean(arr)), float(np.std(arr))

    summary_rows = []
    for condition in CONDITIONS:
        fresh = [r for r in all_results if r["condition"] == condition and not r["is_hard_seed"]]
        hard = [r for r in all_results if r["condition"] == condition and r["is_hard_seed"]]

        def _agg(records):
            if not records:
                return {}
            te_mean, te_std = _mean_std([r["tracking_error_mean"] for r in records])
            csr_mean, csr_std = _mean_std([r["collision_switch_rate"] for r in records])
            ps_mean = float(np.mean([r["perturbation_switch"] for r in records]))
            ms_mean, ms_std = _mean_std([r["mean_surprise"] for r in records])
            return {
                "n": len(records),
                "tracking_error_mean": te_mean,
                "tracking_error_std": te_std,
                "collision_switch_rate_mean": csr_mean,
                "collision_switch_rate_std": csr_std,
                "perturbation_switch_rate": ps_mean,
                "mean_surprise": ms_mean,
                "mean_surprise_std": ms_std,
            }

        summary_rows.append({"condition": condition, "seed_group": "fresh", **_agg(fresh)})
        summary_rows.append({"condition": condition, "seed_group": "hard", **_agg(hard)})

    summary_path = "archive/iter_030/results/arm1_integration_smoke_test.csv"
    with open(summary_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Wrote summary to {summary_path}")

    # ---- Gate evaluation ----
    fresh_results = [r for r in all_results if not r["is_hard_seed"]]

    def _cond_mean(field, cond):
        vals = [r[field] for r in fresh_results if r["condition"] == cond]
        return float(np.mean(vals)) if vals else float("nan")

    g1_sfa = _cond_mean("tracking_error_mean", "CLTS-SFA")
    g1_vic = _cond_mean("tracking_error_mean", "CLTS-VICReg")
    g1_pass_sfa = g1_sfa < 20.0
    g1_pass_vic = g1_vic < 20.0

    g2_sfa = _cond_mean("collision_switch_rate", "CLTS-SFA")
    g2_vic = _cond_mean("collision_switch_rate", "CLTS-VICReg")
    g2_rand = _cond_mean("collision_switch_rate", "CLTS-Random")
    g2_froz = _cond_mean("collision_switch_rate", "CLTS-Frozen")
    g2_max_control = max(g2_rand, g2_froz)
    g2_pass_sfa = g2_sfa >= g2_max_control + 0.15
    g2_pass_vic = g2_vic >= g2_max_control + 0.15

    g3_sfa = _cond_mean("perturbation_switch", "CLTS-SFA")
    g3_vic = _cond_mean("perturbation_switch", "CLTS-VICReg")
    g3_rand = _cond_mean("perturbation_switch", "CLTS-Random")
    g3_froz = _cond_mean("perturbation_switch", "CLTS-Frozen")
    g3_max_control = max(g3_rand, g3_froz)
    g3_pass_sfa = g3_sfa >= g3_max_control + 0.15
    g3_pass_vic = g3_vic >= g3_max_control + 0.15

    sfa_passes = sum([g1_pass_sfa, g2_pass_sfa, g3_pass_sfa])
    vic_passes = sum([g1_pass_vic, g2_pass_vic, g3_pass_vic])

    # ---- Write analysis markdown ----
    md_lines = []
    md_lines.append("# ARM 1 — Integration Smoke-Test Analysis (Iter 030)\n")
    md_lines.append("## Protocol\n")
    md_lines.append(f"- Seeds: fresh {FRESH_SEEDS}\n")
    md_lines.append(f"- Hard seeds (reported separately): {HARD_SEEDS}\n")
    md_lines.append(f"- Conditions: {CONDITIONS}\n")
    md_lines.append(f"- Evaluation steps per run: {EVAL_START}–{NUM_STEPS}\n")
    md_lines.append(f"- Total seeds evaluated (fresh only for gates): {len(FRESH_SEEDS)}\n\n")

    md_lines.append("## Summary Statistics (Fresh Seeds)\n\n")
    md_lines.append("| Condition | n | Tracking Error (mean±std) | Collision Switch (mean±std) | Perturbation Switch Rate | Mean Surprise (mean±std) |\n")
    md_lines.append("|-----------|---|---------------------------|----------------------------|--------------------------|--------------------------|\n")
    for row in summary_rows:
        if row.get("seed_group") != "fresh":
            continue
        md_lines.append(
            f"| {row['condition']} | {row.get('n', 0)} | "
            f"{row.get('tracking_error_mean', 0):.2f}±{row.get('tracking_error_std', 0):.2f} | "
            f"{row.get('collision_switch_rate_mean', 0):.2f}±{row.get('collision_switch_rate_std', 0):.2f} | "
            f"{row.get('perturbation_switch_rate', 0):.2f} | "
            f"{row.get('mean_surprise', 0):.2f}±{row.get('mean_surprise_std', 0):.2f} |\n"
        )

    md_lines.append("\n## Gate Evaluation (Fresh Seeds Only)\n\n")
    md_lines.append(f"**G1 (Tracking Functionality):** mean tracking error < 20 pixels\n")
    md_lines.append(f"- CLTS-SFA: {g1_sfa:.2f} pixels → {'PASS' if g1_pass_sfa else 'FAIL'}\n")
    md_lines.append(f"- CLTS-VICReg: {g1_vic:.2f} pixels → {'PASS' if g1_pass_vic else 'FAIL'}\n\n")

    md_lines.append(f"**G2 (Attention Validity — Collision):** switch-rate ≥ max(control) + 0.15\n")
    md_lines.append(f"- CLTS-SFA: {g2_sfa:.2f} vs control max {g2_max_control:.2f} → {'PASS' if g2_pass_sfa else 'FAIL'}\n")
    md_lines.append(f"- CLTS-VICReg: {g2_vic:.2f} vs control max {g2_max_control:.2f} → {'PASS' if g2_pass_vic else 'FAIL'}\n")
    md_lines.append(f"- CLTS-Frozen: {g2_froz:.2f}\n")
    md_lines.append(f"- CLTS-Random: {g2_rand:.2f}\n\n")

    md_lines.append(f"**G3 (Causal Sensitivity — Mass Perturbation):** switch-rate ≥ max(control) + 0.15\n")
    md_lines.append(f"- CLTS-SFA: {g3_sfa:.2f} vs control max {g3_max_control:.2f} → {'PASS' if g3_pass_sfa else 'FAIL'}\n")
    md_lines.append(f"- CLTS-VICReg: {g3_vic:.2f} vs control max {g3_max_control:.2f} → {'PASS' if g3_pass_vic else 'FAIL'}\n")
    md_lines.append(f"- CLTS-Frozen: {g3_froz:.2f}\n")
    md_lines.append(f"- CLTS-Random: {g3_rand:.2f}\n\n")

    md_lines.append("## Decision Rules\n\n")
    md_lines.append(f"- CLTS-SFA gates passed: {sfa_passes}/3\n")
    md_lines.append(f"- CLTS-VICReg gates passed: {vic_passes}/3\n\n")

    if sfa_passes >= 2:
        md_lines.append("**CLTS-SFA verdict: REPRESENTATION SUFFICIENT** — project advances.\n\n")
    else:
        md_lines.append("**CLTS-SFA verdict: REPRESENTATION INSUFFICIENT** — objective hunt justified.\n\n")

    if sfa_passes >= 2 and vic_passes >= 2:
        md_lines.append("- Both CLTS-SFA and CLTS-VICReg pass ≥2 gates → **M2 demoted** (identity decodability does NOT bottleneck downstream).\n")
    elif sfa_passes >= 2 and vic_passes < 2:
        md_lines.append("- CLTS-SFA passes, CLTS-VICReg fails → **identity encoding matters**; 0.30 search justified.\n")
    elif sfa_passes < 2 and vic_passes < 2:
        md_lines.append("- Both fail → **representation truly insufficient**.\n")
    else:
        md_lines.append("- CLTS-VICReg passes but CLTS-SFA does not → **investigate anomaly**.\n")

    md_lines.append("\n## Per-Seed Raw Results\n\n")
    md_lines.append("| seed | cond | hard | track_err | coll_rate | coll_count | pert_switch | pert_step | surprise |\n")
    md_lines.append("|------|------|------|-----------|-----------|------------|-------------|-----------|----------|\n")
    for r in all_results:
        md_lines.append(
            f"| {r['seed']} | {r['condition']} | {r['is_hard_seed']} | "
            f"{r['tracking_error_mean']:.2f} | {r['collision_switch_rate']:.2f} | "
            f"{r['collision_switch_count']} | {r['perturbation_switch']} | "
            f"{r['perturbation_switch_step'] if r['perturbation_switch_step'] is not None else '-'} | "
            f"{r['mean_surprise']:.4f} |\n"
        )

    analysis_path = "archive/iter_030/results/arm1_analysis.md"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))
    print(f"Wrote analysis to {analysis_path}")


if __name__ == "__main__":
    evaluate_all()
