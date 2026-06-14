from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RACCOON_L1_CM = 8.25
RACCOON_L2_CM = 10.0
RACCOON_L3_CM = 10.0
RACCOON_L4_CM = 8.0
IDLE_FILTER_THRESHOLDS_M = (0.0005, 0.0010, 0.0020)
IDLE_FILTER_JOINT_THRESHOLD_RAD = 0.01
IDLE_FILTER_GRIPPER_THRESHOLD = 1e-4
COLORS = {
    "red": "#d62728",
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "yellow": "#bcbd22",
}


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def numeric_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    return {
        "n": len(values),
        "min": min(values),
        "mean": mean(values),
        "max": max(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def fmt_stats(values: Sequence[float]) -> str:
    stats = numeric_stats(values)
    if stats["n"] == 0:
        return "n=0"
    return (
        f"n={int(stats['n'])}, min={stats['min']:.6f}, "
        f"mean={stats['mean']:.6f}, max={stats['max']:.6f}, std={stats['std']:.6f}"
    )


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


def iter_episode_jsons(intermediate_root: Path) -> Iterable[Path]:
    yield from sorted(intermediate_root.glob("*/episode_*/episode.json"))


def first_close_index(steps: Sequence[Dict[str, Any]], action_dim: int) -> Optional[int]:
    for index, step in enumerate(steps):
        action = step.get("action", [])
        if len(action) > action_dim and float(action[action_dim]) >= 0.5:
            return index
    return None


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def bar_chart(counter: Counter, title: str, ylabel: str, out_path: Path) -> None:
    labels = list(counter.keys())
    values = [counter[label] for label in labels]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color="#4c78a8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    save_fig(out_path)


def histogram(
    values: Sequence[float],
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 40,
    marker: Optional[float] = None,
    marker_label: Optional[str] = None,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(values, bins=bins, color="#4c78a8", alpha=0.82)
    if marker is not None:
        plt.axvline(marker, color="#d62728", linestyle="--", label=marker_label or f"{marker:g}")
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    save_fig(out_path)


def analyze_raw(raw_root: Path, expected_y_max: float, out_dir: Path, title_prefix: str) -> Dict[str, Any]:
    metas = list(iter_raw_meta(raw_root))
    summary: Dict[str, Any] = {"root": str(raw_root), "episodes": len(metas)}
    if not metas:
        return summary

    target_xs: List[float] = []
    target_ys: List[float] = []
    colors: List[str] = []
    lift_deltas: List[float] = []
    logged_close_errors: List[float] = []
    fk_close_errors: List[float] = []
    fk_step_delta_norms: List[float] = []
    estimated_lengths_by_threshold = {threshold: [] for threshold in IDLE_FILTER_THRESHOLDS_M}
    color_counts = Counter()
    template_counts = Counter()
    mode_counts = Counter()
    scene_counts = Counter()
    scene_targets: Dict[str, set[str]] = defaultdict(set)

    for meta in metas:
        color = str(meta.get("target_color", "unknown"))
        color_counts[color] += 1
        colors.append(color)
        template_counts[str(meta.get("instruction_template", meta.get("instruction", "")))] += 1
        mode_counts[str(meta.get("trajectory_mode_effective", meta.get("trajectory_mode", "")))] += 1
        scene_id = meta.get("scene_id")
        if scene_id is not None:
            scene_key = str(scene_id)
            scene_counts[scene_key] += 1
            scene_targets[scene_key].add(color)

        goal_xy = meta.get("goal_xy", [])
        if len(goal_xy) >= 2:
            target_xs.append(float(goal_xy[0]))
            target_ys.append(float(goal_xy[1]))

        if meta.get("target_lift_delta") is not None:
            lift_deltas.append(float(meta["target_lift_delta"]))

        steps = meta.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue

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
            gripper_delta = abs(float(nxt.get("gripper_state", 0.0)) - float(curr.get("gripper_state", 0.0)))
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
            estimated_lengths_by_threshold[threshold].append(kept_count)

        close_i = first_close_index(steps, action_dim=3)
        if close_i is None or len(goal_xy) < 2:
            continue
        close_step = steps[close_i]
        logged_ee = close_step.get("ee_pose", [])
        if len(logged_ee) >= 2:
            logged_close_errors.append(
                math.hypot(float(logged_ee[0]) - float(goal_xy[0]), float(logged_ee[1]) - float(goal_xy[1]))
            )
        fk_ee = fk_ee_pose(close_step.get("joint_angles", []))
        fk_close_errors.append(math.hypot(fk_ee[0] - float(goal_xy[0]), fk_ee[1] - float(goal_xy[1])))

    raw_dir = out_dir / "raw"
    bar_chart(color_counts, f"{title_prefix} target color count", "episodes", raw_dir / "target_color_count.png")
    histogram(target_ys, f"{title_prefix} target y distribution", "target y (m)", raw_dir / "target_y_hist.png", marker=expected_y_max, marker_label=f"expected y max={expected_y_max:.2f}m")
    histogram(lift_deltas, f"{title_prefix} target lift delta", "lift delta (m)", raw_dir / "target_lift_delta_hist.png")
    histogram(fk_step_delta_norms, f"{title_prefix} FK step delta norm", "delta norm (m)", raw_dir / "fk_step_delta_norm_hist.png", marker=0.0005, marker_label="0.5mm idle threshold")

    if target_xs and target_ys:
        plt.figure(figsize=(6, 6))
        for color in sorted(set(colors)):
            xs = [x for x, c in zip(target_xs, colors) if c == color]
            ys = [y for y, c in zip(target_ys, colors) if c == color]
            plt.scatter(xs, ys, s=12, alpha=0.7, label=color, color=COLORS.get(color, "#555555"))
        plt.axhline(expected_y_max, color="#d62728", linestyle="--", linewidth=1.2, label=f"y={expected_y_max:.2f}m")
        plt.title(f"{title_prefix} target xy")
        plt.xlabel("target x (m)")
        plt.ylabel("target y (m)")
        plt.legend()
        plt.axis("equal")
        save_fig(raw_dir / "target_xy_scatter.png")

    if logged_close_errors or fk_close_errors:
        plt.figure(figsize=(8, 4.5))
        if logged_close_errors:
            plt.hist(logged_close_errors, bins=40, alpha=0.55, label="logged ee_pose")
        if fk_close_errors:
            plt.hist(fk_close_errors, bins=40, alpha=0.75, label="FK ee_pose")
        plt.title(f"{title_prefix} close xy error")
        plt.xlabel("xy error at first close (m)")
        plt.ylabel("count")
        plt.legend()
        save_fig(raw_dir / "close_xy_error_hist.png")

    if scene_counts:
        target_counts_per_scene = [len(targets) for targets in scene_targets.values()]
        histogram(
            target_counts_per_scene,
            f"{title_prefix} target colors per scene",
            "target color count per scene",
            raw_dir / "target_colors_per_scene_hist.png",
            bins=max(4, max(target_counts_per_scene)),
        )

    idle_rows = []
    for threshold, lengths in estimated_lengths_by_threshold.items():
        idle_rows.append({"threshold_m": threshold, **numeric_stats(lengths)})
    with open(raw_dir / "estimated_idle_drop_lengths.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold_m", "n", "min", "mean", "max", "std"])
        writer.writeheader()
        writer.writerows(idle_rows)

    summary.update(
        {
            "target_color": dict(color_counts),
            "trajectory_mode": dict(mode_counts),
            "instruction_templates": len(template_counts),
            "target_y": numeric_stats(target_ys),
            "target_y_over_expected": sum(y > expected_y_max for y in target_ys),
            "target_lift_delta": numeric_stats(lift_deltas),
            "logged_close_xy_error": numeric_stats(logged_close_errors),
            "fk_close_xy_error": numeric_stats(fk_close_errors),
            "fk_step_delta_norm": numeric_stats(fk_step_delta_norms),
            "estimated_idle_drop_lengths": {str(k): numeric_stats(v) for k, v in estimated_lengths_by_threshold.items()},
            "scenes": len(scene_counts),
            "scenes_with_multiple_targets": sum(len(targets) > 1 for targets in scene_targets.values()),
        }
    )
    return summary


def analyze_intermediate(intermediate_root: Path, out_dir: Path, title_prefix: str) -> Dict[str, Any]:
    episode_paths = list(iter_episode_jsons(intermediate_root))
    summary: Dict[str, Any] = {"root": str(intermediate_root), "episodes": len(episode_paths)}
    if not episode_paths:
        return summary

    lengths: List[int] = []
    action_norms: List[float] = []
    gripper_values: List[float] = []
    z_by_phase: Dict[str, List[float]] = defaultdict(list)
    instruction_counts = Counter()
    scene_counts = Counter()
    scene_targets: Dict[str, set[str]] = defaultdict(set)

    for path in episode_paths:
        ep = read_json(path)
        steps = ep.get("steps", [])
        lengths.append(len(steps))
        if steps:
            instruction = str(steps[0].get("language_instruction", ""))
            instruction_counts[instruction] += 1
        metadata = ep.get("episode_metadata", {})
        scene_id = metadata.get("scene_id")
        if scene_id is not None:
            scene_key = str(scene_id)
            scene_counts[scene_key] += 1
            instruction = str(steps[0].get("language_instruction", "")) if steps else ""
            for color in COLORS:
                if color in instruction.lower():
                    scene_targets[scene_key].add(color)

        close_i = first_close_index(steps, action_dim=6)
        for index, step in enumerate(steps):
            action = step.get("action", [])
            if len(action) < 7:
                continue
            xyz_norm = math.sqrt(sum(float(action[i]) ** 2 for i in range(3)))
            action_norms.append(xyz_norm)
            gripper_values.append(float(action[6]))
            z = float(action[2])
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

    intermediate_dir = out_dir / "intermediate"
    histogram(lengths, f"{title_prefix} episode length", "steps", intermediate_dir / "episode_length_hist.png", bins=20)
    histogram(action_norms, f"{title_prefix} action xyz norm", "action xyz norm (m)", intermediate_dir / "action_xyz_norm_hist.png", marker=0.0005, marker_label="0.5mm")
    bar_chart(instruction_counts, f"{title_prefix} instruction count", "episodes", intermediate_dir / "instruction_count.png")

    phase_order = ["pre_close", "first_close", "after_close_1", "after_close_2_5", "post_close"]
    phase_values = [z_by_phase[phase] for phase in phase_order if z_by_phase[phase]]
    phase_labels = [phase for phase in phase_order if z_by_phase[phase]]
    if phase_values:
        plt.figure(figsize=(9, 4.8))
        plt.boxplot([[v * 1000.0 for v in values] for values in phase_values], labels=phase_labels, showfliers=False)
        plt.axhline(0.0, color="#555555", linewidth=1.0)
        plt.title(f"{title_prefix} z action by phase")
        plt.ylabel("z action (mm)")
        plt.xticks(rotation=18, ha="right")
        save_fig(intermediate_dir / "z_action_by_phase_boxplot.png")

    if scene_counts:
        target_counts_per_scene = [len(targets) for targets in scene_targets.values()]
        histogram(
            target_counts_per_scene,
            f"{title_prefix} intermediate target colors per scene",
            "target color count per scene",
            intermediate_dir / "target_colors_per_scene_hist.png",
            bins=max(4, max(target_counts_per_scene)),
        )

    summary.update(
        {
            "episode_length": numeric_stats(lengths),
            "action_xyz_norm": numeric_stats(action_norms),
            "action_xyz_norm_under_0.5mm": sum(v < 0.0005 for v in action_norms),
            "actions": len(action_norms),
            "gripper_close_ratio": (
                sum(v >= 0.5 for v in gripper_values) / len(gripper_values) if gripper_values else 0.0
            ),
            "instructions": len(instruction_counts),
            "z_by_phase": {phase: numeric_stats(values) for phase, values in z_by_phase.items()},
            "scenes": len(scene_counts),
            "scenes_with_multiple_targets": sum(len(targets) > 1 for targets in scene_targets.values()),
        }
    )
    return summary


def write_index(out_dir: Path, raw_summary: Optional[Dict[str, Any]], intermediate_summary: Optional[Dict[str, Any]]) -> None:
    lines = [
        "# Raccoon Dataset Visualization Index",
        "",
        "이 디렉터리는 `scripts/visualize_raccoon_dataset_health.py`로 생성된 데이터셋 진단 그림입니다.",
        "",
    ]
    if raw_summary:
        lines.extend(
            [
                "## Raw Dataset",
                "",
                f"- root: `{raw_summary.get('root')}`",
                f"- episodes: `{raw_summary.get('episodes')}`",
                f"- target_y: {fmt_stats_from_dict(raw_summary.get('target_y', {}))}",
                f"- target_y_over_expected: `{raw_summary.get('target_y_over_expected')}`",
                f"- fk_close_xy_error: {fmt_stats_from_dict(raw_summary.get('fk_close_xy_error', {}))}",
                f"- scenes_with_multiple_targets: `{raw_summary.get('scenes_with_multiple_targets')}/{raw_summary.get('scenes')}`",
                "",
                "주요 파일: `raw/target_xy_scatter.png`, `raw/target_y_hist.png`, `raw/close_xy_error_hist.png`, `raw/fk_step_delta_norm_hist.png`",
                "",
            ]
        )
    if intermediate_summary:
        actions = int(intermediate_summary.get("actions", 0))
        under = int(intermediate_summary.get("action_xyz_norm_under_0.5mm", 0))
        under_ratio = under / actions if actions else 0.0
        lines.extend(
            [
                "## Intermediate Dataset",
                "",
                f"- root: `{intermediate_summary.get('root')}`",
                f"- episodes: `{intermediate_summary.get('episodes')}`",
                f"- episode_length: {fmt_stats_from_dict(intermediate_summary.get('episode_length', {}))}",
                f"- action_xyz_norm: {fmt_stats_from_dict(intermediate_summary.get('action_xyz_norm', {}))}",
                f"- action_xyz_norm < 0.5mm: `{under}/{actions} ({under_ratio:.3f})`",
                f"- scenes_with_multiple_targets: `{intermediate_summary.get('scenes_with_multiple_targets')}/{intermediate_summary.get('scenes')}`",
                "",
                "주요 파일: `intermediate/episode_length_hist.png`, `intermediate/action_xyz_norm_hist.png`, `intermediate/z_action_by_phase_boxplot.png`",
                "",
            ]
        )

    lines.extend(
        [
            "## 해석 기준",
            "",
            "- `target_y`가 rollout workspace 상한인 0.20m를 넘지 않아야 합니다.",
            "- `fk_close_xy_error`는 millimeter scale이어야 합니다.",
            "- `action_xyz_norm < 0.5mm` 비율이 너무 크면 정지/settle frame이 과대표집된 것입니다.",
            "- `z_action_by_phase_boxplot.png`에서 close 이후 z action이 양수로 이동해야 lift 라벨이 살아 있습니다.",
            "- same-scene multi-target 수치가 높을수록 색상 언어 grounding에 유리합니다.",
            "",
        ]
    )
    (out_dir / "visualization_index.md").write_text("\n".join(lines), encoding="utf-8")


def fmt_stats_from_dict(stats: Dict[str, Any]) -> str:
    if not stats or int(stats.get("n", 0)) == 0:
        return "n=0"
    return (
        f"n={int(stats['n'])}, min={float(stats['min']):.6f}, "
        f"mean={float(stats['mean']):.6f}, max={float(stats['max']):.6f}, std={float(stats['std']):.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PNG diagnostics for Raccoon raw/intermediate datasets.")
    parser.add_argument("--raw-root", type=Path, default=None, help="Raw dataset root with episode_*/meta.json")
    parser.add_argument("--intermediate-root", type=Path, default=None, help="Intermediate root with split/episode_*/episode.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to save figures and summaries")
    parser.add_argument("--expected-y-max", type=float, default=0.20)
    parser.add_argument("--title-prefix", type=str, default="Raccoon")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_root is None and args.intermediate_root is None:
        raise SystemExit("Pass --raw-root, --intermediate-root, or both.")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = None
    intermediate_summary = None
    if args.raw_root is not None:
        raw_summary = analyze_raw(args.raw_root.resolve(), args.expected_y_max, out_dir, args.title_prefix)
    if args.intermediate_root is not None:
        intermediate_summary = analyze_intermediate(args.intermediate_root.resolve(), out_dir, args.title_prefix)

    summary = {
        "raw": raw_summary,
        "intermediate": intermediate_summary,
        "idle_filter": {
            "thresholds_m": list(IDLE_FILTER_THRESHOLDS_M),
            "joint_threshold_rad": IDLE_FILTER_JOINT_THRESHOLD_RAD,
            "gripper_threshold": IDLE_FILTER_GRIPPER_THRESHOLD,
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_index(out_dir, raw_summary, intermediate_summary)
    print(f"[OK] visualization written to: {out_dir}")


if __name__ == "__main__":
    main()
