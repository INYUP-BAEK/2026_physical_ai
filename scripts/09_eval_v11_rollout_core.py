#!/usr/bin/env python3
import argparse
import csv
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

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_ROOT = PROJECT_ROOT / "Mujoco"
if str(MUJOCO_ROOT) not in sys.path:
    sys.path.insert(0, str(MUJOCO_ROOT))

from raccoon_grasp_multicolor_scene_dataset import (  # noqa: E402
    DEFAULT_LIFT_EXTENDED_TEMPLATES,
    SyncSimRaccoonDataset,
)


DEFAULT_BASE_MODEL_PATH = Path(
    "/root/.cache/huggingface/hub/models--openvla--openvla-7b/"
    "snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
)
DEFAULT_MODEL_PATH = DEFAULT_BASE_MODEL_PATH


class RolloutWorkspace:
    x_min = -0.16
    x_max = 0.16
    y_min = 0.11
    y_max = 0.20
    z_min = 0.016
    z_max = 0.10

    @classmethod
    def clip_xyz(cls, xyz):
        x, y, z = [float(v) for v in xyz]
        return np.asarray(
            [
                np.clip(x, cls.x_min, cls.x_max),
                np.clip(y, cls.y_min, cls.y_max),
                np.clip(z, cls.z_min, cls.z_max),
            ],
            dtype=np.float64,
        )


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def center_crop_image(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    if crop_scale >= 1.0:
        return image
    width, height = image.size
    side_scale = math.sqrt(float(crop_scale))
    crop_w = max(1, int(round(width * side_scale)))
    crop_h = max(1, int(round(height * side_scale)))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h)).resize(image.size, Image.BICUBIC)


def load_model(
    model_path: Path,
    device: str,
    adapter_path: Optional[Path] = None,
    base_model_path: Optional[Path] = None,
    merge_adapter: bool = False,
):
    processor_source = model_path if (model_path / "preprocessor_config.json").exists() else (base_model_path or model_path)
    processor = AutoProcessor.from_pretrained(str(processor_source), trust_remote_code=True)
    load_path = base_model_path if adapter_path is not None else model_path
    model = AutoModelForVision2Seq.from_pretrained(
        str(load_path),
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        if merge_adapter:
            model = model.merge_and_unload()
    model = model.to(device)
    stats_path = model_path / "dataset_statistics.json"
    if stats_path.exists():
        attach_norm_stats(model, json.loads(stats_path.read_text(encoding="utf-8")))
    model.eval()
    return processor, model


def resolve_adapter_run_dir(model_path: Path, adapter_path: Optional[Path]) -> Path:
    if adapter_path is None:
        return model_path
    if model_path != DEFAULT_MODEL_PATH:
        return model_path

    candidate = PROJECT_ROOT / "openvla/openvla-runs" / adapter_path.name
    if candidate.exists():
        return candidate
    return model_path


def get_gpu_memory_snapshot():
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
    }


def attach_norm_stats(model, norm_stats):
    model.norm_stats = norm_stats
    for attr_path in (
        ("base_model",),
        ("base_model", "model"),
    ):
        obj = model
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            obj.norm_stats = norm_stats


def get_body_touching_robot(env: SyncSimRaccoonDataset):
    touching = []
    for color, body_name in env.CYLINDER_BODY_BY_COLOR.items():
        try:
            if env.is_body_touching_robot(body_name):
                touching.append(color)
        except Exception:
            continue
    return touching


def execute_delta_action(
    env: SyncSimRaccoonDataset,
    action,
    speed: int,
    max_delta_xyz: float,
    delta_scale: float,
    shrink_ratio: float,
    max_retries: int,
    use_pitch_action: bool = False,
    pitch_settle_seconds_on_change: float = 0.0,
    pitch_change_threshold: float = 0.25,
):
    raw = np.asarray(action[:7], dtype=np.float64)
    raw_delta = raw[:3].copy()
    pitch_cmd = float(np.clip(raw[4], 0.0, 1.0))
    pitch_before = float(getattr(env, "gripper_pitch_alpha", 0.0))
    if use_pitch_action:
        env.set_gripper_pitch_alpha(pitch_cmd)
        if (
            pitch_settle_seconds_on_change > 0.0
            and abs(pitch_cmd - pitch_before) >= float(pitch_change_threshold)
        ):
            env.settle_steps(float(pitch_settle_seconds_on_change))
    scaled_delta = np.clip(raw_delta * float(delta_scale), -max_delta_xyz, max_delta_xyz)
    ee_before = np.asarray(env.get_ee_pose(), dtype=np.float64)

    tried = []
    chosen_target = None
    chosen_delta = None
    cur_delta = scaled_delta.copy()
    last_error = None

    for retry_idx in range(max_retries + 1):
        target_xyz = RolloutWorkspace.clip_xyz(ee_before + cur_delta)
        try:
            env.move_to(float(target_xyz[0]) * 100.0, float(target_xyz[1]) * 100.0, float(target_xyz[2]) * 100.0, speed=speed)
            chosen_target = target_xyz
            chosen_delta = cur_delta.copy()
            tried.append(
                {
                    "retry_index": retry_idx,
                    "target_xyz": target_xyz.tolist(),
                    "delta_xyz": cur_delta.tolist(),
                    "ok": True,
                    "error": None,
                }
            )
            break
        except Exception as exc:
            last_error = str(exc)
            tried.append(
                {
                    "retry_index": retry_idx,
                    "target_xyz": target_xyz.tolist(),
                    "delta_xyz": cur_delta.tolist(),
                    "ok": False,
                    "error": str(exc),
                }
            )
            cur_delta *= 1.0 - float(shrink_ratio)

    if chosen_target is None:
        raise RuntimeError(
            "IK failed after retries: "
            f"ee={ee_before.tolist()} raw_delta={raw_delta.tolist()} scaled_delta={scaled_delta.tolist()} "
            f"last_error={last_error}"
        )

    gripper_cmd = float(raw[6])
    if gripper_cmd >= 0.5:
        env.close_gripper()
    else:
        env.open_gripper()

    return {
        "raw_action": raw.tolist(),
        "raw_delta_xyz": raw_delta.tolist(),
        "scaled_delta_xyz": scaled_delta.tolist(),
        "executed_delta_xyz": chosen_delta.tolist(),
        "target_xyz": chosen_target.tolist(),
        "ee_before": ee_before.tolist(),
        "gripper_cmd": gripper_cmd,
        "pitch_cmd": pitch_cmd,
        "pitch_before": pitch_before,
        "use_pitch_action": bool(use_pitch_action),
        "retry_count": len(tried) - 1,
        "tried": tried,
    }


def save_frame(image_rgb, path: Path, label: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(image_rgb)
    if label:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 255, 255))
    image.save(path)


def make_gif_from_frames(frame_dir: Path, gif_path: Path, duration_ms: int = 120) -> Optional[Path]:
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        return None

    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(duration_ms),
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()
    return gif_path


def run_one_rollout(
    env: SyncSimRaccoonDataset,
    processor,
    model,
    device: str,
    object_specs: dict,
    target_color: str,
    instruction: str,
    episode_index: int,
    scene_id: int,
    args,
    report_dir: Path,
):
    target_body_name = object_specs[target_color]["body_name"]
    env.reset_episode(object_specs=object_specs, target_color=target_color)
    env.lockh()
    if args.initial_settle_seconds > 0:
        env.settle_steps(args.initial_settle_seconds)

    target_initial_pose = env.get_object_pose(target_body_name).astype(np.float64)
    obs = env.get_observation(target_body_name)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    rollout_dir = report_dir / f"episode_{episode_index:04d}_{target_color}"
    if args.save_frames:
        save_frame(obs["image"], rollout_dir / "frame_000_start.png", f"start {target_color}")

    step_records = []
    ever_contact_success = False
    ever_pose_lift_success = False
    ever_strict_lift_success = False
    ever_wrong_color_touch = False
    first_contact_step = None
    first_pose_lift_step = None
    first_strict_lift_step = None
    exception_text = None
    extra_hold_steps = 0

    for step_idx in range(args.max_steps):
        image = Image.fromarray(obs["image"]).convert("RGB")
        if args.center_crop:
            image = center_crop_image(image, crop_scale=args.crop_scale)

        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
        if hasattr(action, "tolist"):
            action = action.tolist()

        try:
            execution = execute_delta_action(
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
        obs = env.get_observation(target_body_name)

        target_pose = env.get_object_pose(target_body_name).astype(np.float64)
        lift_delta = float(target_pose[2] - target_initial_pose[2])
        contact_success = bool(env.is_target_grasp_success(target_body_name, touch_threshold=args.touch_threshold))
        pose_lift_success = bool(lift_delta >= args.lift_threshold)
        strict_lift_success = bool(contact_success and lift_delta >= args.lift_threshold)
        touching_colors = get_body_touching_robot(env)
        wrong_touching_colors = [color for color in touching_colors if color != target_color]
        wrong_color_touch = bool(wrong_touching_colors)

        if contact_success and first_contact_step is None:
            first_contact_step = step_idx
        if pose_lift_success and first_pose_lift_step is None:
            first_pose_lift_step = step_idx
        if strict_lift_success and first_strict_lift_step is None:
            first_strict_lift_step = step_idx

        ever_contact_success = bool(ever_contact_success or contact_success)
        ever_pose_lift_success = bool(ever_pose_lift_success or pose_lift_success)
        ever_strict_lift_success = bool(ever_strict_lift_success or strict_lift_success)
        ever_wrong_color_touch = bool(ever_wrong_color_touch or wrong_color_touch)

        record = {
            "step": step_idx,
            "action": [float(v) for v in action[:7]],
            "execution": execution,
            "ee_pose": [float(v) for v in obs["ee_pose"]],
            "target_pose": target_pose.tolist(),
            "target_lift_delta": lift_delta,
            "contact_success": contact_success,
            "pose_lift_success": pose_lift_success,
            "strict_lift_success": strict_lift_success,
            "touching_colors": touching_colors,
            "wrong_touching_colors": wrong_touching_colors,
            "gripper_state": float(obs["gripper_state"]),
        }
        step_records.append(record)

        if args.save_frames and (
            step_idx % args.save_frames_every == 0
            or step_idx == args.max_steps - 1
            or strict_lift_success
        ):
            label = f"t={step_idx} {target_color} lift={lift_delta:.3f} g={record['action'][6]:.2f}"
            save_frame(obs["image"], rollout_dir / f"frame_{step_idx + 1:03d}.png", label)

        if strict_lift_success:
            extra_hold_steps += 1
            if extra_hold_steps >= args.stop_after_success_hold_steps:
                break
        else:
            extra_hold_steps = 0

    final_target_pose = env.get_object_pose(target_body_name).astype(np.float64)
    final_lift_delta = float(final_target_pose[2] - target_initial_pose[2])
    final_contact_success = bool(env.is_target_grasp_success(target_body_name, touch_threshold=args.touch_threshold))
    final_pose_lift_success = bool(final_lift_delta >= args.lift_threshold)
    final_strict_lift_success = bool(final_contact_success and final_lift_delta >= args.lift_threshold)
    final_touching_colors = get_body_touching_robot(env)

    gif_path = None
    if args.save_frames:
        save_frame(obs["image"], rollout_dir / "frame_final.png", f"final {target_color} lift={final_lift_delta:.3f}")
        clean_success = bool(final_strict_lift_success and not ever_wrong_color_touch and exception_text is None)
        if args.failure_gif_criteria == "clean":
            is_failure_for_gif = not clean_success
        elif args.failure_gif_criteria == "pose":
            is_failure_for_gif = bool((not final_pose_lift_success) or exception_text is not None)
        else:
            is_failure_for_gif = bool((not final_strict_lift_success) or exception_text is not None)
        should_make_gif = bool(args.make_gif and (not args.gif_failures_only or is_failure_for_gif))
        if should_make_gif:
            gif_path = make_gif_from_frames(
                frame_dir=rollout_dir,
                gif_path=report_dir / f"{report_dir.name}_episode_{episode_index:04d}_{target_color}.gif",
                duration_ms=args.gif_duration_ms,
            )
        if args.delete_png_frames:
            shutil.rmtree(rollout_dir, ignore_errors=True)

    return {
        "episode_index": int(episode_index),
        "scene_id": int(scene_id),
        "target_color": str(target_color),
        "target_body_name": str(target_body_name),
        "instruction": str(instruction),
        "prompt": prompt,
        "object_specs": SyncSimRaccoonDataset.specs_to_meta(object_specs),
        "num_steps": len(step_records),
        "exception": exception_text,
        "target_z_initial": float(target_initial_pose[2]),
        "target_z_final": float(final_target_pose[2]),
        "target_lift_delta_final": final_lift_delta,
        "ever_contact_success": ever_contact_success,
        "ever_pose_lift_success": ever_pose_lift_success,
        "ever_strict_lift_success": ever_strict_lift_success,
        "ever_wrong_color_touch": ever_wrong_color_touch,
        "final_contact_success": final_contact_success,
        "final_pose_lift_success": final_pose_lift_success,
        "final_strict_lift_success": final_strict_lift_success,
        "final_touching_colors": final_touching_colors,
        "first_contact_step": first_contact_step,
        "first_pose_lift_step": first_pose_lift_step,
        "first_strict_lift_step": first_strict_lift_step,
        "gif_path": str(gif_path) if gif_path is not None else None,
        "steps": step_records,
    }


def summarize_results(results):
    n = len(results)
    colors = sorted({r["target_color"] for r in results})

    def rate(key):
        return float(sum(bool(r[key]) for r in results) / n) if n else 0.0

    summary = {
        "num_rollouts": n,
        "ever_contact_success_count": int(sum(bool(r["ever_contact_success"]) for r in results)),
        "ever_contact_success_rate": rate("ever_contact_success"),
        "ever_pose_lift_success_count": int(sum(bool(r["ever_pose_lift_success"]) for r in results)),
        "ever_pose_lift_success_rate": rate("ever_pose_lift_success"),
        "ever_strict_lift_success_count": int(sum(bool(r["ever_strict_lift_success"]) for r in results)),
        "ever_strict_lift_success_rate": rate("ever_strict_lift_success"),
        "final_contact_success_count": int(sum(bool(r["final_contact_success"]) for r in results)),
        "final_contact_success_rate": rate("final_contact_success"),
        "final_pose_lift_success_count": int(sum(bool(r["final_pose_lift_success"]) for r in results)),
        "final_pose_lift_success_rate": rate("final_pose_lift_success"),
        "final_strict_lift_success_count": int(sum(bool(r["final_strict_lift_success"]) for r in results)),
        "final_strict_lift_success_rate": rate("final_strict_lift_success"),
        "wrong_color_touch_count": int(sum(bool(r["ever_wrong_color_touch"]) for r in results)),
        "wrong_color_touch_rate": rate("ever_wrong_color_touch"),
        "exception_count": int(sum(r.get("exception") is not None for r in results)),
        "mean_final_lift_delta": float(np.mean([r["target_lift_delta_final"] for r in results])) if n else 0.0,
        "median_final_lift_delta": float(np.median([r["target_lift_delta_final"] for r in results])) if n else 0.0,
        "mean_steps": float(np.mean([r["num_steps"] for r in results])) if n else 0.0,
        "by_color": {},
        "instruction_counts": dict(Counter(r["instruction"] for r in results)),
    }

    for color in colors:
        subset = [r for r in results if r["target_color"] == color]
        m = len(subset)
        summary["by_color"][color] = {
            "n": m,
            "ever_contact_success_rate": float(sum(bool(r["ever_contact_success"]) for r in subset) / m) if m else 0.0,
            "ever_pose_lift_success_rate": float(sum(bool(r["ever_pose_lift_success"]) for r in subset) / m) if m else 0.0,
            "ever_strict_lift_success_rate": float(sum(bool(r["ever_strict_lift_success"]) for r in subset) / m) if m else 0.0,
            "final_pose_lift_success_rate": float(sum(bool(r["final_pose_lift_success"]) for r in subset) / m) if m else 0.0,
            "final_strict_lift_success_rate": float(sum(bool(r["final_strict_lift_success"]) for r in subset) / m) if m else 0.0,
            "wrong_color_touch_rate": float(sum(bool(r["ever_wrong_color_touch"]) for r in subset) / m) if m else 0.0,
            "mean_final_lift_delta": float(np.mean([r["target_lift_delta_final"] for r in subset])) if m else 0.0,
        }

    return summary


def write_csv(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_index",
        "scene_id",
        "target_color",
        "instruction",
        "num_steps",
        "exception",
        "ever_contact_success",
        "ever_pose_lift_success",
        "ever_strict_lift_success",
        "final_contact_success",
        "final_pose_lift_success",
        "final_strict_lift_success",
        "ever_wrong_color_touch",
        "target_lift_delta_final",
        "first_contact_step",
        "first_pose_lift_step",
        "first_strict_lift_step",
        "final_touching_colors",
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
        "# OpenVLA Closed-loop Rollout Evaluation",
        "",
        f"- model: `{args.model_path}`",
        f"- rollouts: `{summary['num_rollouts']}`",
        f"- max steps: `{args.max_steps}`",
        f"- step seconds: `{args.step_seconds}`",
        f"- center crop: `{args.center_crop}`",
        f"- max delta xyz: `{args.max_delta_xyz}`",
        f"- workspace z min/max: `{RolloutWorkspace.z_min}` / `{RolloutWorkspace.z_max}`",
        f"- lift threshold: `{args.lift_threshold}`",
        "",
        "## Summary",
        "",
        f"- ever contact success: `{summary['ever_contact_success_count']}/{summary['num_rollouts']}` ({summary['ever_contact_success_rate']:.3f})",
        f"- ever pose lift success: `{summary['ever_pose_lift_success_count']}/{summary['num_rollouts']}` ({summary['ever_pose_lift_success_rate']:.3f})",
        f"- ever strict lift success: `{summary['ever_strict_lift_success_count']}/{summary['num_rollouts']}` ({summary['ever_strict_lift_success_rate']:.3f})",
        f"- final pose lift success: `{summary['final_pose_lift_success_count']}/{summary['num_rollouts']}` ({summary['final_pose_lift_success_rate']:.3f})",
        f"- final strict lift success: `{summary['final_strict_lift_success_count']}/{summary['num_rollouts']}` ({summary['final_strict_lift_success_rate']:.3f})",
        f"- wrong-color touch: `{summary['wrong_color_touch_count']}/{summary['num_rollouts']}` ({summary['wrong_color_touch_rate']:.3f})",
        f"- exceptions: `{summary['exception_count']}`",
        f"- mean final lift delta: `{summary['mean_final_lift_delta']:.6f}m`",
        f"- mean steps: `{summary['mean_steps']:.2f}`",
        "",
        "## By Color",
        "",
    ]
    for color, color_summary in sorted(summary["by_color"].items()):
        lines.append(
            f"- {color}: n `{color_summary['n']}`, "
            f"pose-final `{color_summary['final_pose_lift_success_rate']:.3f}`, "
            f"strict-ever `{color_summary['ever_strict_lift_success_rate']:.3f}`, "
            f"strict-final `{color_summary['final_strict_lift_success_rate']:.3f}`, "
            f"wrong-touch `{color_summary['wrong_color_touch_rate']:.3f}`, "
            f"mean final lift `{color_summary['mean_final_lift_delta']:.6f}m`"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA checkpoint in closed-loop RaccoonBot MuJoCo rollouts.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter_path", type=Path, default=None)
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--base_model_path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--xml_path", type=Path, default=MUJOCO_ROOT / "Raccoon_colored_cylinder.xml")
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--num_rollouts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--step_seconds", type=float, default=0.10)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.10)
    parser.add_argument("--speed", type=int, default=150)
    parser.add_argument("--max_delta_xyz", type=float, default=0.12)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--shrink_ratio", type=float, default=0.15)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--use_pitch_action",
        action="store_true",
        help="Interpret action[4] dpitch as gripper pitch_alpha command: 0 horizontal, 1 vertical.",
    )
    parser.add_argument(
        "--pitch_settle_seconds_on_change",
        type=float,
        default=0.0,
        help="If >0, settle this long before XYZ motion when pitch_alpha changes by the threshold.",
    )
    parser.add_argument(
        "--pitch_change_threshold",
        type=float,
        default=0.25,
        help="Pitch command delta threshold that triggers --pitch_settle_seconds_on_change.",
    )
    parser.add_argument("--touch_threshold", type=float, default=0.1)
    parser.add_argument("--lift_threshold", type=float, default=0.010)
    parser.add_argument("--object_x_min", type=float, default=-0.10)
    parser.add_argument("--object_x_max", type=float, default=0.10)
    parser.add_argument("--object_y_min", type=float, default=0.16)
    parser.add_argument("--object_y_max", type=float, default=0.195)
    parser.add_argument("--min_object_distance", type=float, default=0.042)
    parser.add_argument("--workspace_z_min", type=float, default=0.016)
    parser.add_argument("--workspace_z_max", type=float, default=0.10)
    parser.set_defaults(center_crop=True, save_frames=True)
    parser.add_argument("--center_crop", dest="center_crop", action="store_true")
    parser.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    parser.add_argument("--crop_scale", type=float, default=0.9)
    parser.add_argument("--save_frames", dest="save_frames", action="store_true")
    parser.add_argument("--no_save_frames", dest="save_frames", action="store_false")
    parser.add_argument("--save_frames_every", type=int, default=5)
    parser.add_argument("--make_gif", action="store_true", help="Build an episode GIF from saved rollout frames.")
    parser.add_argument(
        "--gif_failures_only",
        action="store_true",
        help="When making GIFs, keep GIFs only for failed episodes.",
    )
    parser.add_argument(
        "--failure_gif_criteria",
        choices=("strict", "clean", "pose"),
        default="strict",
        help=(
            "Failure definition for --gif_failures_only. "
            "'strict' means final strict lift failed; 'pose' ignores contact sensor and only checks object lift; "
            "'clean' also treats wrong-color touch as failure."
        ),
    )
    parser.add_argument(
        "--delete_png_frames",
        action="store_true",
        help="Delete per-episode PNG frame directories after optional GIF creation.",
    )
    parser.add_argument("--gif_duration_ms", type=int, default=120)
    parser.add_argument("--stop_after_success_hold_steps", type=int, default=3)
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

    run_name = args.run_name or f"v11_close_stable_rollout_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    report_dir = PROJECT_ROOT / "reports" / run_name
    diagnostics_json = PROJECT_ROOT / "diagnostics" / f"{run_name}.json"
    diagnostics_csv = PROJECT_ROOT / "diagnostics" / f"{run_name}.csv"
    diagnostics_md = PROJECT_ROOT / "diagnostics" / f"{run_name}.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    args.model_path = resolve_adapter_run_dir(args.model_path, args.adapter_path)
    RolloutWorkspace.z_min = float(args.workspace_z_min)
    RolloutWorkspace.z_max = float(args.workspace_z_max)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] run_name={run_name}", flush=True)
    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] gpu_before={get_gpu_memory_snapshot()}", flush=True)
    print(f"[INFO] loading model: {args.model_path}", flush=True)
    if args.adapter_path is not None:
        print(f"[INFO] loading adapter: {args.adapter_path}", flush=True)
        print(f"[INFO] loading base model: {args.base_model_path}", flush=True)
    processor, model = load_model(
        args.model_path,
        device=device,
        adapter_path=args.adapter_path,
        base_model_path=args.base_model_path,
        merge_adapter=args.merge_adapter,
    )
    print(f"[INFO] gpu_after_load={get_gpu_memory_snapshot()}", flush=True)

    rng = np.random.default_rng(args.seed)
    colors = list(SyncSimRaccoonDataset.CYLINDER_COLORS)
    env = SyncSimRaccoonDataset(
        xml_path=str(args.xml_path),
        image_size=(256, 256),
        camera_name=args.camera_name,
        use_viewer=False,
    )

    results = []
    rollout_index = 0
    scene_id = 0
    try:
        while rollout_index < args.num_rollouts:
            scene_id += 1
            object_specs = SyncSimRaccoonDataset.sample_object_specs(
                rng=rng,
                colors=colors,
                x_range=(args.object_x_min, args.object_x_max),
                y_range=(args.object_y_min, args.object_y_max),
                min_distance=args.min_object_distance,
            )
            target_colors = list(rng.permutation(colors))
            for target_color in target_colors:
                if rollout_index >= args.num_rollouts:
                    break
                template = str(rng.choice(DEFAULT_LIFT_EXTENDED_TEMPLATES))
                instruction = template.format(color=target_color)
                rollout_index += 1
                start = time.perf_counter()
                print(
                    f"[ROLLOUT {rollout_index:04d}] scene={scene_id:04d} color={target_color} "
                    f"instruction='{instruction}'",
                    flush=True,
                )
                result = run_one_rollout(
                    env=env,
                    processor=processor,
                    model=model,
                    device=device,
                    object_specs=object_specs,
                    target_color=target_color,
                    instruction=instruction,
                    episode_index=rollout_index,
                    scene_id=scene_id,
                    args=args,
                    report_dir=report_dir,
                )
                result["wall_time_s"] = time.perf_counter() - start
                results.append(result)
                print(
                    f"[RESULT {rollout_index:04d}] color={target_color} "
                    f"ever_contact={result['ever_contact_success']} "
                    f"ever_strict_lift={result['ever_strict_lift_success']} "
                    f"final_strict_lift={result['final_strict_lift_success']} "
                    f"wrong_touch={result['ever_wrong_color_touch']} "
                    f"lift_final={result['target_lift_delta_final']:.4f} "
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
        "gpu_after_eval": get_gpu_memory_snapshot(),
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
