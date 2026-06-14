#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_ROOT = PROJECT_ROOT / "Mujoco"
if str(MUJOCO_ROOT) not in sys.path:
    sys.path.insert(0, str(MUJOCO_ROOT))

from raccoon_grasp_multicolor_scene_dataset import SyncSimRaccoonDataset  # noqa: E402
from raccoon_stack_dataset import (  # noqa: E402
    STACK_TEMPLATE,
    ordered_color_pairs,
    sample_stack_object_specs,
)


def load_lift_eval_core():
    core_path = Path(__file__).resolve().with_name("09_eval_v11_rollout_core.py")
    spec = importlib.util.spec_from_file_location("raccoon_lift_eval_core", core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import rollout core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_lift_eval_core()
DEFAULT_BASE_MODEL_PATH = CORE.DEFAULT_BASE_MODEL_PATH
DEFAULT_MODEL_PATH = CORE.DEFAULT_MODEL_PATH


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def classify_stack_result(result):
    if result.get("exception") is not None:
        return "exception"
    if result.get("final_strict_stack_success"):
        if result.get("ever_non_source_robot_touch"):
            return "strict_stack_success_with_non_source_touch"
        return "strict_stack_success_clean"
    if not result.get("ever_source_lift_success"):
        if result.get("ever_source_contact_success"):
            return "source_contact_no_lift"
        return "no_source_grasp_or_lift"
    if result.get("final_stack_xy_distance", 999.0) > result.get("stack_xy_threshold", 0.020):
        return "lifted_but_not_over_base"
    if result.get("final_stack_z_delta", -999.0) < result.get("stack_z_threshold", 0.014):
        return "over_base_but_not_on_top"
    if not result.get("final_gripper_open"):
        return "on_top_but_not_released"
    return "stack_failed_other"


def gripper_open(env):
    return bool(float(env.data.qpos[4]) > (env.GRIP_OPEN - 0.030))


def run_one_stack_rollout(
    env: SyncSimRaccoonDataset,
    processor,
    model,
    device: str,
    object_specs: dict,
    source_color: str,
    base_color: str,
    episode_index: int,
    scene_id: int,
    args,
    report_dir: Path,
):
    source_body_name = object_specs[source_color]["body_name"]
    base_body_name = object_specs[base_color]["body_name"]
    instruction = STACK_TEMPLATE.format(source_color=source_color, base_color=base_color)

    env.reset_episode(object_specs=object_specs, target_color=source_color)
    env.lockh()
    if args.initial_settle_seconds > 0:
        env.settle_steps(args.initial_settle_seconds)

    source_initial_pose = env.get_object_pose(source_body_name).astype(np.float64)
    base_initial_pose = env.get_object_pose(base_body_name).astype(np.float64)
    obs = env.get_observation(source_body_name)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    rollout_dir = report_dir / f"episode_{episode_index:04d}_{source_color}_on_{base_color}"
    if args.save_frames:
        CORE.save_frame(obs["image"], rollout_dir / "frame_000_start.png", f"start {source_color}->{base_color}")

    step_records = []
    ever_source_contact_success = False
    ever_source_lift_success = False
    ever_non_source_robot_touch = False
    ever_base_robot_touch = False
    first_source_contact_step = None
    first_source_lift_step = None
    first_strict_stack_step = None
    exception_text = None
    success_hold_steps = 0

    for step_idx in range(args.max_steps):
        image = Image.fromarray(obs["image"]).convert("RGB")
        if args.center_crop:
            image = CORE.center_crop_image(image, crop_scale=args.crop_scale)

        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
        if hasattr(action, "tolist"):
            action = action.tolist()

        try:
            execution = CORE.execute_delta_action(
                env=env,
                action=action,
                speed=args.speed,
                max_delta_xyz=args.max_delta_xyz,
                delta_scale=args.delta_scale,
                shrink_ratio=args.shrink_ratio,
                max_retries=args.max_retries,
                use_pitch_action=args.use_pitch_action,
                pitch_settle_seconds_on_change=args.pitch_settle_seconds_on_change,
                pitch_change_threshold=args.pitch_change_threshold,
            )
        except Exception as exc:
            exception_text = str(exc)
            break

        env.settle_steps(args.step_seconds)
        obs = env.get_observation(source_body_name)

        source_pose = env.get_object_pose(source_body_name).astype(np.float64)
        base_pose = env.get_object_pose(base_body_name).astype(np.float64)
        source_lift_delta = float(source_pose[2] - source_initial_pose[2])
        stack_xy_distance = float(math.hypot(source_pose[0] - base_pose[0], source_pose[1] - base_pose[1]))
        stack_z_delta = float(source_pose[2] - base_pose[2])
        source_contact_success = bool(
            env.is_target_grasp_success(source_body_name, touch_threshold=args.touch_threshold)
        )
        source_lift_success = bool(source_lift_delta >= args.lift_threshold)
        final_open_now = gripper_open(env)
        strict_stack_success = bool(
            stack_xy_distance <= args.stack_xy_threshold
            and stack_z_delta >= args.stack_z_threshold
            and final_open_now
        )
        touching_colors = CORE.get_body_touching_robot(env)
        non_source_touching_colors = [color for color in touching_colors if color != source_color]
        base_robot_touch = bool(base_color in touching_colors)

        if source_contact_success and first_source_contact_step is None:
            first_source_contact_step = step_idx
        if source_lift_success and first_source_lift_step is None:
            first_source_lift_step = step_idx
        if strict_stack_success and first_strict_stack_step is None:
            first_strict_stack_step = step_idx

        ever_source_contact_success = bool(ever_source_contact_success or source_contact_success)
        ever_source_lift_success = bool(ever_source_lift_success or source_lift_success)
        ever_non_source_robot_touch = bool(ever_non_source_robot_touch or non_source_touching_colors)
        ever_base_robot_touch = bool(ever_base_robot_touch or base_robot_touch)

        record = {
            "step": step_idx,
            "action": [float(v) for v in action[:7]],
            "execution": execution,
            "ee_pose": [float(v) for v in obs["ee_pose"]],
            "source_pose": source_pose.tolist(),
            "base_pose": base_pose.tolist(),
            "source_lift_delta": source_lift_delta,
            "source_contact_success": source_contact_success,
            "source_lift_success": source_lift_success,
            "stack_xy_distance": stack_xy_distance,
            "stack_z_delta": stack_z_delta,
            "gripper_open": final_open_now,
            "strict_stack_success": strict_stack_success,
            "touching_colors": touching_colors,
            "non_source_touching_colors": non_source_touching_colors,
            "base_robot_touch": base_robot_touch,
            "gripper_state": float(obs["gripper_state"]),
        }
        step_records.append(record)

        if args.save_frames and (
            step_idx % args.save_frames_every == 0
            or step_idx == args.max_steps - 1
            or strict_stack_success
        ):
            label = (
                f"t={step_idx} {source_color}->{base_color} "
                f"xy={stack_xy_distance:.3f} dz={stack_z_delta:.3f} open={int(final_open_now)}"
            )
            CORE.save_frame(obs["image"], rollout_dir / f"frame_{step_idx + 1:03d}.png", label)

        if strict_stack_success:
            success_hold_steps += 1
            if success_hold_steps >= args.stop_after_success_hold_steps:
                break
        else:
            success_hold_steps = 0

    source_final_pose = env.get_object_pose(source_body_name).astype(np.float64)
    base_final_pose = env.get_object_pose(base_body_name).astype(np.float64)
    source_lift_delta_final = float(source_final_pose[2] - source_initial_pose[2])
    final_stack_xy_distance = float(
        math.hypot(source_final_pose[0] - base_final_pose[0], source_final_pose[1] - base_final_pose[1])
    )
    final_stack_z_delta = float(source_final_pose[2] - base_final_pose[2])
    final_gripper_open = gripper_open(env)
    final_source_contact_success = bool(
        env.is_target_grasp_success(source_body_name, touch_threshold=args.touch_threshold)
    )
    final_strict_stack_success = bool(
        final_stack_xy_distance <= args.stack_xy_threshold
        and final_stack_z_delta >= args.stack_z_threshold
        and final_gripper_open
    )
    final_touching_colors = CORE.get_body_touching_robot(env)

    result = {
        "episode_index": int(episode_index),
        "scene_id": int(scene_id),
        "source_color": str(source_color),
        "base_color": str(base_color),
        "source_body_name": str(source_body_name),
        "base_body_name": str(base_body_name),
        "instruction": str(instruction),
        "prompt": prompt,
        "object_specs": SyncSimRaccoonDataset.specs_to_meta(object_specs),
        "num_steps": len(step_records),
        "exception": exception_text,
        "source_z_initial": float(source_initial_pose[2]),
        "source_z_final": float(source_final_pose[2]),
        "base_z_initial": float(base_initial_pose[2]),
        "base_z_final": float(base_final_pose[2]),
        "source_lift_delta_final": source_lift_delta_final,
        "final_stack_xy_distance": final_stack_xy_distance,
        "final_stack_z_delta": final_stack_z_delta,
        "final_gripper_open": final_gripper_open,
        "ever_source_contact_success": ever_source_contact_success,
        "ever_source_lift_success": ever_source_lift_success,
        "ever_non_source_robot_touch": ever_non_source_robot_touch,
        "ever_base_robot_touch": ever_base_robot_touch,
        "final_source_contact_success": final_source_contact_success,
        "final_strict_stack_success": final_strict_stack_success,
        "final_touching_colors": final_touching_colors,
        "first_source_contact_step": first_source_contact_step,
        "first_source_lift_step": first_source_lift_step,
        "first_strict_stack_step": first_strict_stack_step,
        "stack_xy_threshold": float(args.stack_xy_threshold),
        "stack_z_threshold": float(args.stack_z_threshold),
        "gif_path": None,
        "steps": step_records,
    }
    result["failure_class"] = classify_stack_result(result)

    if args.save_frames:
        CORE.save_frame(
            obs["image"],
            rollout_dir / "frame_final.png",
            f"final {source_color}->{base_color} xy={final_stack_xy_distance:.3f} dz={final_stack_z_delta:.3f}",
        )
        is_failure_for_gif = bool((not final_strict_stack_success) or exception_text is not None)
        should_make_gif = bool(args.make_gif and (not args.gif_failures_only or is_failure_for_gif))
        if should_make_gif:
            gif_path = CORE.make_gif_from_frames(
                frame_dir=rollout_dir,
                gif_path=report_dir / f"{report_dir.name}_episode_{episode_index:04d}_{source_color}_on_{base_color}.gif",
                duration_ms=args.gif_duration_ms,
            )
            result["gif_path"] = str(gif_path) if gif_path is not None else None
        if args.delete_png_frames:
            shutil.rmtree(rollout_dir, ignore_errors=True)

    return result


def rate(results, key):
    n = len(results)
    return float(sum(bool(r[key]) for r in results) / n) if n else 0.0


def summarize_results(results):
    n = len(results)
    summary = {
        "num_rollouts": n,
        "final_strict_stack_success_count": int(sum(bool(r["final_strict_stack_success"]) for r in results)),
        "final_strict_stack_success_rate": rate(results, "final_strict_stack_success"),
        "ever_source_contact_success_count": int(sum(bool(r["ever_source_contact_success"]) for r in results)),
        "ever_source_contact_success_rate": rate(results, "ever_source_contact_success"),
        "ever_source_lift_success_count": int(sum(bool(r["ever_source_lift_success"]) for r in results)),
        "ever_source_lift_success_rate": rate(results, "ever_source_lift_success"),
        "final_gripper_open_count": int(sum(bool(r["final_gripper_open"]) for r in results)),
        "final_gripper_open_rate": rate(results, "final_gripper_open"),
        "non_source_robot_touch_count": int(sum(bool(r["ever_non_source_robot_touch"]) for r in results)),
        "non_source_robot_touch_rate": rate(results, "ever_non_source_robot_touch"),
        "base_robot_touch_count": int(sum(bool(r["ever_base_robot_touch"]) for r in results)),
        "base_robot_touch_rate": rate(results, "ever_base_robot_touch"),
        "exception_count": int(sum(r.get("exception") is not None for r in results)),
        "mean_final_stack_xy_distance": float(np.mean([r["final_stack_xy_distance"] for r in results])) if n else 0.0,
        "median_final_stack_xy_distance": float(np.median([r["final_stack_xy_distance"] for r in results])) if n else 0.0,
        "mean_final_stack_z_delta": float(np.mean([r["final_stack_z_delta"] for r in results])) if n else 0.0,
        "median_final_stack_z_delta": float(np.median([r["final_stack_z_delta"] for r in results])) if n else 0.0,
        "mean_source_lift_delta_final": float(np.mean([r["source_lift_delta_final"] for r in results])) if n else 0.0,
        "mean_steps": float(np.mean([r["num_steps"] for r in results])) if n else 0.0,
        "failure_class_counts": dict(Counter(r["failure_class"] for r in results)),
        "instruction_counts": dict(Counter(r["instruction"] for r in results)),
        "by_source_color": {},
        "by_base_color": {},
        "by_pair": {},
    }

    for key_name, output_name in (("source_color", "by_source_color"), ("base_color", "by_base_color")):
        for color in sorted({r[key_name] for r in results}):
            subset = [r for r in results if r[key_name] == color]
            m = len(subset)
            summary[output_name][color] = {
                "n": m,
                "final_strict_stack_success_rate": float(sum(bool(r["final_strict_stack_success"]) for r in subset) / m)
                if m
                else 0.0,
                "ever_source_lift_success_rate": float(sum(bool(r["ever_source_lift_success"]) for r in subset) / m)
                if m
                else 0.0,
                "mean_final_stack_xy_distance": float(np.mean([r["final_stack_xy_distance"] for r in subset])) if m else 0.0,
                "mean_final_stack_z_delta": float(np.mean([r["final_stack_z_delta"] for r in subset])) if m else 0.0,
            }

    for pair in sorted({(r["source_color"], r["base_color"]) for r in results}):
        subset = [r for r in results if (r["source_color"], r["base_color"]) == pair]
        m = len(subset)
        pair_key = f"{pair[0]}_on_{pair[1]}"
        summary["by_pair"][pair_key] = {
            "n": m,
            "final_strict_stack_success_rate": float(sum(bool(r["final_strict_stack_success"]) for r in subset) / m)
            if m
            else 0.0,
            "ever_source_lift_success_rate": float(sum(bool(r["ever_source_lift_success"]) for r in subset) / m)
            if m
            else 0.0,
            "failure_class_counts": dict(Counter(r["failure_class"] for r in subset)),
        }

    return summary


def write_csv(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_index",
        "scene_id",
        "source_color",
        "base_color",
        "instruction",
        "num_steps",
        "exception",
        "failure_class",
        "ever_source_contact_success",
        "ever_source_lift_success",
        "final_source_contact_success",
        "final_strict_stack_success",
        "final_gripper_open",
        "ever_non_source_robot_touch",
        "ever_base_robot_touch",
        "source_lift_delta_final",
        "final_stack_xy_distance",
        "final_stack_z_delta",
        "first_source_contact_step",
        "first_source_lift_step",
        "first_strict_stack_step",
        "final_touching_colors",
        "gif_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fieldnames}
            row["final_touching_colors"] = json.dumps(row["final_touching_colors"], ensure_ascii=False)
            writer.writerow(row)


def write_markdown(summary, args, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OpenVLA Stack Rollout Evaluation",
        "",
        f"- model: `{args.model_path}`",
        f"- adapter: `{args.adapter_path}`",
        f"- rollouts: `{summary['num_rollouts']}`",
        f"- max steps: `{args.max_steps}`",
        f"- step seconds: `{args.step_seconds}`",
        f"- stack xy threshold: `{args.stack_xy_threshold}`",
        f"- stack z threshold: `{args.stack_z_threshold}`",
        f"- object x range: `{args.object_x_min}` / `{args.object_x_max}`",
        f"- object y range: `{args.object_y_min}` / `{args.object_y_max}`",
        "",
        "## Summary",
        "",
        f"- final strict stack success: `{summary['final_strict_stack_success_count']}/{summary['num_rollouts']}` "
        f"({summary['final_strict_stack_success_rate']:.3f})",
        f"- ever source contact: `{summary['ever_source_contact_success_count']}/{summary['num_rollouts']}` "
        f"({summary['ever_source_contact_success_rate']:.3f})",
        f"- ever source lift: `{summary['ever_source_lift_success_count']}/{summary['num_rollouts']}` "
        f"({summary['ever_source_lift_success_rate']:.3f})",
        f"- final gripper open: `{summary['final_gripper_open_count']}/{summary['num_rollouts']}` "
        f"({summary['final_gripper_open_rate']:.3f})",
        f"- non-source robot touch: `{summary['non_source_robot_touch_count']}/{summary['num_rollouts']}` "
        f"({summary['non_source_robot_touch_rate']:.3f})",
        f"- base robot touch: `{summary['base_robot_touch_count']}/{summary['num_rollouts']}` "
        f"({summary['base_robot_touch_rate']:.3f})",
        f"- exceptions: `{summary['exception_count']}`",
        f"- mean final stack xy distance: `{summary['mean_final_stack_xy_distance']:.6f}m`",
        f"- mean final stack z delta: `{summary['mean_final_stack_z_delta']:.6f}m`",
        f"- mean source lift delta final: `{summary['mean_source_lift_delta_final']:.6f}m`",
        f"- mean steps: `{summary['mean_steps']:.2f}`",
        "",
        "## Failure Classes",
        "",
    ]
    for name, count in sorted(summary["failure_class_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: `{count}`")

    lines.extend(["", "## By Source Color", ""])
    for color, item in sorted(summary["by_source_color"].items()):
        lines.append(
            f"- {color}: n `{item['n']}`, stack `{item['final_strict_stack_success_rate']:.3f}`, "
            f"source-lift `{item['ever_source_lift_success_rate']:.3f}`, "
            f"xy `{item['mean_final_stack_xy_distance']:.6f}m`, dz `{item['mean_final_stack_z_delta']:.6f}m`"
        )

    lines.extend(["", "## By Base Color", ""])
    for color, item in sorted(summary["by_base_color"].items()):
        lines.append(
            f"- {color}: n `{item['n']}`, stack `{item['final_strict_stack_success_rate']:.3f}`, "
            f"source-lift `{item['ever_source_lift_success_rate']:.3f}`, "
            f"xy `{item['mean_final_stack_xy_distance']:.6f}m`, dz `{item['mean_final_stack_z_delta']:.6f}m`"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA checkpoint on cylinder stack commands.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter_path", type=Path, default=None)
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--base_model_path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--xml_path", type=Path, default=MUJOCO_ROOT / "Raccoon_colored_cylinder.xml")
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--num_rollouts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--step_seconds", type=float, default=0.10)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.10)
    parser.add_argument("--speed", type=int, default=150)
    parser.add_argument("--max_delta_xyz", type=float, default=0.12)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--shrink_ratio", type=float, default=0.15)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--use_pitch_action", action="store_true")
    parser.add_argument("--pitch_settle_seconds_on_change", type=float, default=0.0)
    parser.add_argument("--pitch_change_threshold", type=float, default=0.25)
    parser.add_argument("--touch_threshold", type=float, default=0.1)
    parser.add_argument("--lift_threshold", type=float, default=0.010)
    parser.add_argument("--stack_xy_threshold", type=float, default=0.020)
    parser.add_argument("--stack_z_threshold", type=float, default=0.014)
    parser.add_argument("--object_x_min", type=float, default=-0.10)
    parser.add_argument("--object_x_max", type=float, default=0.10)
    parser.add_argument("--object_y_min", type=float, default=0.135)
    parser.add_argument("--object_y_max", type=float, default=0.180)
    parser.add_argument("--min_object_distance", type=float, default=0.045)
    parser.add_argument("--workspace_z_min", type=float, default=0.016)
    parser.add_argument("--workspace_z_max", type=float, default=0.10)
    parser.set_defaults(center_crop=True, save_frames=True)
    parser.add_argument("--center_crop", dest="center_crop", action="store_true")
    parser.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    parser.add_argument("--crop_scale", type=float, default=0.9)
    parser.add_argument("--save_frames", dest="save_frames", action="store_true")
    parser.add_argument("--no_save_frames", dest="save_frames", action="store_false")
    parser.add_argument("--save_frames_every", type=int, default=5)
    parser.add_argument("--make_gif", action="store_true")
    parser.add_argument("--gif_failures_only", action="store_true")
    parser.add_argument("--delete_png_frames", action="store_true")
    parser.add_argument("--gif_duration_ms", type=int, default=120)
    parser.add_argument("--stop_after_success_hold_steps", type=int, default=5)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_frames_every <= 0:
        raise ValueError("--save_frames_every must be positive")
    if args.gif_duration_ms <= 0:
        raise ValueError("--gif_duration_ms must be positive")
    if args.make_gif and not args.save_frames:
        print("[INFO] --make_gif requires frames; enabling --save_frames.", flush=True)
        args.save_frames = True
    if args.gif_failures_only and not args.make_gif:
        raise ValueError("--gif_failures_only requires --make_gif")

    run_name = args.run_name or f"v11_stack_rollout_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    report_dir = PROJECT_ROOT / "reports" / run_name
    diagnostics_json = PROJECT_ROOT / "diagnostics" / f"{run_name}.json"
    diagnostics_csv = PROJECT_ROOT / "diagnostics" / f"{run_name}.csv"
    diagnostics_md = PROJECT_ROOT / "diagnostics" / f"{run_name}.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    args.model_path = CORE.resolve_adapter_run_dir(args.model_path, args.adapter_path)
    CORE.RolloutWorkspace.z_min = float(args.workspace_z_min)
    CORE.RolloutWorkspace.z_max = float(args.workspace_z_max)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] run_name={run_name}", flush=True)
    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] gpu_before={CORE.get_gpu_memory_snapshot()}", flush=True)
    print(f"[INFO] loading model: {args.model_path}", flush=True)
    if args.adapter_path is not None:
        print(f"[INFO] loading adapter: {args.adapter_path}", flush=True)
        print(f"[INFO] loading base model: {args.base_model_path}", flush=True)
    processor, model = CORE.load_model(
        args.model_path,
        device=device,
        adapter_path=args.adapter_path,
        base_model_path=args.base_model_path,
        merge_adapter=args.merge_adapter,
    )
    print(f"[INFO] gpu_after_load={CORE.get_gpu_memory_snapshot()}", flush=True)

    rng = np.random.default_rng(args.seed)
    colors = list(SyncSimRaccoonDataset.CYLINDER_COLORS)
    pairs = ordered_color_pairs(colors)
    env = SyncSimRaccoonDataset(
        xml_path=str(args.xml_path),
        image_size=(256, 256),
        camera_name=args.camera_name,
        use_viewer=False,
    )

    results = []
    try:
        for rollout_idx in range(1, args.num_rollouts + 1):
            scene_id = rollout_idx
            if (rollout_idx - 1) % len(pairs) == 0:
                pair_order = list(rng.permutation(len(pairs)))
            pair = pairs[int(pair_order[(rollout_idx - 1) % len(pairs)])]
            source_color, base_color = pair
            object_specs = sample_stack_object_specs(
                rng=rng,
                source_color=source_color,
                base_color=base_color,
                colors=colors,
                x_range=(args.object_x_min, args.object_x_max),
                y_range=(args.object_y_min, args.object_y_max),
                min_distance=args.min_object_distance,
            )
            start = time.perf_counter()
            instruction = STACK_TEMPLATE.format(source_color=source_color, base_color=base_color)
            print(
                f"[ROLLOUT {rollout_idx:04d}] scene={scene_id:04d} pair={source_color}_on_{base_color} "
                f"instruction='{instruction}'",
                flush=True,
            )
            result = run_one_stack_rollout(
                env=env,
                processor=processor,
                model=model,
                device=device,
                object_specs=object_specs,
                source_color=source_color,
                base_color=base_color,
                episode_index=rollout_idx,
                scene_id=scene_id,
                args=args,
                report_dir=report_dir,
            )
            result["wall_time_s"] = time.perf_counter() - start
            results.append(result)
            print(
                f"[RESULT {rollout_idx:04d}] pair={source_color}_on_{base_color} "
                f"stack={result['final_strict_stack_success']} "
                f"lift={result['ever_source_lift_success']} "
                f"open={result['final_gripper_open']} "
                f"xy={result['final_stack_xy_distance']:.4f} "
                f"dz={result['final_stack_z_delta']:.4f} "
                f"class={result['failure_class']} "
                f"steps={result['num_steps']} exception={result['exception']}",
                flush=True,
            )
    finally:
        env.close()

    summary = summarize_results(results)
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gpu_after_eval": CORE.get_gpu_memory_snapshot(),
        "summary": summary,
        "results": results,
    }
    diagnostics_json.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    write_csv(results, diagnostics_csv)
    write_markdown(summary, args, diagnostics_md)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[SUMMARY]", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[INFO] wrote_json={diagnostics_json}", flush=True)
    print(f"[INFO] wrote_csv={diagnostics_csv}", flush=True)
    print(f"[INFO] wrote_md={diagnostics_md}", flush=True)
    print(f"[INFO] report_dir={report_dir}", flush=True)


if __name__ == "__main__":
    main()
