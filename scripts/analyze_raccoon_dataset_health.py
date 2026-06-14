from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence


RACCOON_L1_CM = 8.25
RACCOON_L2_CM = 10.0
RACCOON_L3_CM = 10.0
RACCOON_L4_CM = 8.0
IDLE_FILTER_THRESHOLDS_M = (0.0005, 0.0010, 0.0020)
IDLE_FILTER_JOINT_THRESHOLD_RAD = 0.01
IDLE_FILTER_GRIPPER_THRESHOLD = 1e-4


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def numeric_summary(values: Sequence[float]) -> str:
    if not values:
        return "n=0"
    mean = sum(values) / len(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return f"n={len(values)} min={min(values):.6f} mean={mean:.6f} max={max(values):.6f} std={std:.6f}"


def ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0/0"
    return f"{count}/{total} ({count / total:.3f})"


def fk_ee_pose(joint_angles: Sequence[Any]) -> List[float]:
    if len(joint_angles) < 3:
        raise ValueError(f"joint_angles must have at least 3 values, got {len(joint_angles)}")

    th1, th2, th3 = [float(v) for v in joint_angles[:3]]
    r = -RACCOON_L2_CM * math.sin(th2) - RACCOON_L3_CM * math.sin(th2 + th3)
    z = RACCOON_L1_CM + RACCOON_L2_CM * math.cos(th2) + RACCOON_L3_CM * math.cos(th2 + th3)
    r_tip = r + RACCOON_L4_CM

    return [
        -math.sin(th1) * r_tip / 100.0,
        math.cos(th1) * r_tip / 100.0,
        z / 100.0,
    ]


def iter_raw_meta(raw_root: Path) -> Iterable[Dict[str, Any]]:
    for meta_path in sorted(raw_root.glob("episode_*/meta.json")):
        yield read_json(meta_path)


def first_close_index(steps: Sequence[Dict[str, Any]], threshold: float = 0.5) -> Optional[int]:
    for index, step in enumerate(steps):
        action = step.get("action", [])
        if len(action) >= 4 and float(action[3]) >= threshold:
            return index
    return None


def raw_target_xy(meta: Dict[str, Any]) -> Optional[List[float]]:
    goal_xy = meta.get("goal_xy")
    if isinstance(goal_xy, list) and len(goal_xy) >= 2:
        return [float(goal_xy[0]), float(goal_xy[1])]
    return None


def analyze_raw(raw_root: Path, expected_y_max: float) -> None:
    metas = list(iter_raw_meta(raw_root))
    print(f"\n[RAW] {raw_root}")
    print(f"episodes: {len(metas)}")
    if not metas:
        return

    colors = Counter(str(meta.get("target_color", "")) for meta in metas)
    modes = Counter(str(meta.get("trajectory_mode_effective", meta.get("trajectory_mode", ""))) for meta in metas)
    templates = Counter(str(meta.get("instruction_template", meta.get("instruction", ""))) for meta in metas)
    successes = sum(bool(meta.get("success", False)) for meta in metas)
    strict = sum(bool(meta.get("strict_lift_success", False)) for meta in metas)
    scene_counts = Counter()
    scene_targets: Dict[str, set[str]] = defaultdict(set)

    target_xs: List[float] = []
    target_ys: List[float] = []
    lift_deltas: List[float] = []
    close_steps: List[int] = []
    logged_close_xy_errors: List[float] = []
    fk_close_xy_errors: List[float] = []
    fk_step_delta_norms: List[float] = []
    estimated_lengths_by_idle_threshold: Dict[float, List[int]] = {
        threshold: [] for threshold in IDLE_FILTER_THRESHOLDS_M
    }
    y_over = 0

    for meta in metas:
        scene_id = meta.get("scene_id")
        if scene_id is not None:
            scene_key = str(scene_id)
            scene_counts[scene_key] += 1
            scene_targets[scene_key].add(str(meta.get("target_color", "")))

        xy = raw_target_xy(meta)
        if xy is not None:
            target_xs.append(xy[0])
            target_ys.append(xy[1])
            if xy[1] > expected_y_max:
                y_over += 1

        if meta.get("target_lift_delta") is not None:
            lift_deltas.append(float(meta["target_lift_delta"]))

        steps = meta.get("steps", [])
        close_i = first_close_index(steps) if isinstance(steps, list) else None

        if isinstance(steps, list) and steps:
            estimated_kept = {threshold: 1 for threshold in IDLE_FILTER_THRESHOLDS_M}
            for index in range(len(steps) - 1):
                curr = steps[index]
                nxt = steps[index + 1]
                curr_fk = fk_ee_pose(curr.get("joint_angles", []))
                next_fk = fk_ee_pose(nxt.get("joint_angles", []))
                delta_norm = math.sqrt(sum((next_fk[i] - curr_fk[i]) ** 2 for i in range(3)))
                curr_joints = [float(v) for v in curr.get("joint_angles", [])]
                next_joints = [float(v) for v in nxt.get("joint_angles", [])]
                if len(curr_joints) == len(next_joints):
                    joint_delta_norm = math.sqrt(
                        sum((next_joints[i] - curr_joints[i]) ** 2 for i in range(len(curr_joints)))
                    )
                else:
                    joint_delta_norm = float("inf")
                gripper_delta = abs(
                    float(nxt.get("gripper_state", 0.0)) - float(curr.get("gripper_state", 0.0))
                )
                fk_step_delta_norms.append(delta_norm)
                for threshold in IDLE_FILTER_THRESHOLDS_M:
                    is_idle = (
                        delta_norm < threshold
                        and joint_delta_norm < IDLE_FILTER_JOINT_THRESHOLD_RAD
                        and gripper_delta < IDLE_FILTER_GRIPPER_THRESHOLD
                    )
                    if not is_idle:
                        estimated_kept[threshold] += 1
            for threshold, kept_count in estimated_kept.items():
                estimated_lengths_by_idle_threshold[threshold].append(kept_count)

        if close_i is None or xy is None:
            continue

        close_steps.append(close_i)
        close_step = steps[close_i]
        logged_ee = close_step.get("ee_pose", [])
        if len(logged_ee) >= 3:
            logged_close_xy_errors.append(math.hypot(float(logged_ee[0]) - xy[0], float(logged_ee[1]) - xy[1]))
        fk_ee = fk_ee_pose(close_step.get("joint_angles", []))
        fk_close_xy_errors.append(math.hypot(fk_ee[0] - xy[0], fk_ee[1] - xy[1]))

    print(f"success: {ratio(successes, len(metas))}")
    print(f"strict_lift_success: {ratio(strict, len(metas))}")
    print(f"target_color: {dict(colors)}")
    print(f"trajectory_mode_effective: {dict(modes)}")
    print(f"instruction_templates: {len(templates)} unique")
    if scene_counts:
        multi_target_scenes = sum(len(targets) > 1 for targets in scene_targets.values())
        colors_per_scene = [len(targets) for targets in scene_targets.values()]
        print(f"scenes: {len(scene_counts)}")
        print(f"episodes_per_scene: {numeric_summary(list(scene_counts.values()))}")
        print(f"target_colors_per_scene: {numeric_summary(colors_per_scene)}")
        print(f"scenes_with_multiple_targets: {ratio(multi_target_scenes, len(scene_counts))}")
    print(f"target_x: {numeric_summary(target_xs)}")
    print(f"target_y: {numeric_summary(target_ys)}")
    print(f"target_y > {expected_y_max:.3f}: {ratio(y_over, len(target_ys))}")
    print(f"target_lift_delta: {numeric_summary(lift_deltas)}")
    print(f"first_close_index: {dict(Counter(close_steps))}")
    print(f"logged_close_xy_error: {numeric_summary(logged_close_xy_errors)}")
    print(f"fk_close_xy_error: {numeric_summary(fk_close_xy_errors)}")
    print(f"fk_step_delta_norm: {numeric_summary(fk_step_delta_norms)}")
    print(
        "idle_filter_estimate_criteria: "
        f"joint_delta_norm < {IDLE_FILTER_JOINT_THRESHOLD_RAD} rad, "
        f"gripper_delta < {IDLE_FILTER_GRIPPER_THRESHOLD}"
    )
    for threshold, lengths in estimated_lengths_by_idle_threshold.items():
        print(f"estimated_len_drop_idle_{threshold * 1000:.1f}mm: {numeric_summary(lengths)}")


def iter_episode_jsons(intermediate_root: Path) -> Iterable[Path]:
    yield from sorted(intermediate_root.glob("*/episode_*/episode.json"))


def analyze_intermediate(intermediate_root: Path) -> None:
    episode_paths = list(iter_episode_jsons(intermediate_root))
    print(f"\n[INTERMEDIATE] {intermediate_root}")
    print(f"episodes: {len(episode_paths)}")
    if not episode_paths:
        return

    lengths: List[int] = []
    close_indices: List[int] = []
    z_by_phase: Dict[str, List[float]] = defaultdict(list)
    gripper_values: List[float] = []
    action_xyz_norms: List[float] = []
    instructions = Counter()
    scene_counts = Counter()
    scene_targets: Dict[str, set[str]] = defaultdict(set)

    for path in episode_paths:
        ep = read_json(path)
        steps = ep.get("steps", [])
        lengths.append(len(steps))
        metadata = ep.get("episode_metadata", {})
        scene_id = metadata.get("scene_id")
        if scene_id is not None:
            scene_key = str(scene_id)
            scene_counts[scene_key] += 1
            instruction_text = str(steps[0].get("language_instruction", "")) if steps else ""
            for color in ("red", "blue", "green", "yellow"):
                if color in instruction_text.lower():
                    scene_targets[scene_key].add(color)
        if steps:
            instructions[str(steps[0].get("language_instruction", ""))] += 1

        close_i = None
        for index, step in enumerate(steps):
            action = step.get("action", [])
            if len(action) >= 7 and float(action[6]) >= 0.5:
                close_i = index
                break
        if close_i is not None:
            close_indices.append(close_i)

        for index, step in enumerate(steps):
            action = step.get("action", [])
            if len(action) < 7:
                continue
            z = float(action[2])
            g = float(action[6])
            xyz_norm = math.sqrt(sum(float(action[i]) ** 2 for i in range(3)))
            action_xyz_norms.append(xyz_norm)
            gripper_values.append(g)
            if close_i is None or index < close_i:
                z_by_phase["pre_close"].append(z)
            else:
                z_by_phase["post_close"].append(z)
            if close_i is not None:
                if index == close_i:
                    z_by_phase["first_close"].append(z)
                elif index == close_i + 1:
                    z_by_phase["after_close_1"].append(z)
                elif close_i + 2 <= index <= close_i + 5:
                    z_by_phase["after_close_2_5"].append(z)

    print(f"episode_length: {numeric_summary(lengths)}")
    print(f"first_close_index: {dict(Counter(close_indices))}")
    print(f"gripper_close_ratio: {ratio(sum(v >= 0.5 for v in gripper_values), len(gripper_values))}")
    print(f"instructions: {len(instructions)} unique")
    print(f"action_xyz_norm: {numeric_summary(action_xyz_norms)}")
    for threshold in IDLE_FILTER_THRESHOLDS_M:
        small_count = sum(v < threshold for v in action_xyz_norms)
        print(f"action_xyz_norm < {threshold * 1000:.1f}mm: {ratio(small_count, len(action_xyz_norms))}")
    if scene_counts:
        multi_target_scenes = sum(len(targets) > 1 for targets in scene_targets.values())
        colors_per_scene = [len(targets) for targets in scene_targets.values()]
        print(f"scenes: {len(scene_counts)}")
        print(f"episodes_per_scene: {numeric_summary(list(scene_counts.values()))}")
        print(f"target_colors_per_scene: {numeric_summary(colors_per_scene)}")
        print(f"scenes_with_multiple_targets: {ratio(multi_target_scenes, len(scene_counts))}")
    for phase, values in sorted(z_by_phase.items()):
        positive = sum(v > 0 for v in values)
        negative = sum(v < 0 for v in values)
        print(
            f"{phase}_z: {numeric_summary(values)} "
            f"positive={ratio(positive, len(values))} negative={ratio(negative, len(values))}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RaccoonBot raw/intermediate dataset health.")
    parser.add_argument("--raw-root", type=Path, default=None, help="Raw episode root with episode_*/meta.json")
    parser.add_argument("--intermediate-root", type=Path, default=None, help="RLDS intermediate root")
    parser.add_argument("--expected-y-max", type=float, default=0.20, help="Report target_y values above this bound")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_root is None and args.intermediate_root is None:
        raise SystemExit("Pass --raw-root, --intermediate-root, or both.")
    if args.raw_root is not None:
        analyze_raw(args.raw_root, expected_y_max=args.expected_y_max)
    if args.intermediate_root is not None:
        analyze_intermediate(args.intermediate_root)


if __name__ == "__main__":
    main()
