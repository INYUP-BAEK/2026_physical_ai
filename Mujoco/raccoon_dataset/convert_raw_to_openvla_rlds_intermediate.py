from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Raw episode folder  ->  OpenVLA-friendly RLDS intermediate
# ============================================================
# Input (raw)
#   raw_root/
#     episode_000001/
#       frame_000000.png
#       frame_000001.png
#       ...
#       meta.json
#
# Output (intermediate)
#   out_root/
#     dataset_info.json
#     manifest_train.jsonl
#     manifest_val.jsonl               # only if val_ratio > 0
#     train/
#       episode_000001/
#         images/
#           frame_000000.png
#           ...
#         episode.json
#     val/
#       episode_000123/
#         images/
#         episode.json
#
# episode.json schema (builder-friendly, RLDS-like)
# {
#   "episode_metadata": {...},
#   "steps": [
#      {
#        "observation": {
#           "image": "images/frame_000000.png",
#           "state": [8 dims],
#           ... debug fields ...
#        },
#        "action": [7 dims],
#        "language_instruction": "...",
#        "reward": 0.0,
#        "discount": 1.0,
#        "is_first": true,
#        "is_last": false,
#        "is_terminal": false,
#        "timestep": 0,
#        ... raw/debug fields ...
#      }
#   ]
# }
#
# OpenVLA convention used here:
#   state  = [q1..q7(padded), gripper]                       -> 8 dims
#   action = [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd] -> 7 dims
#
# For this 4-axis robot:
#   - state still uses padded joint state for observation
#   - action can use either next-state command-space EE delta or raw waypoint
#     command delta from either FK(joint_angles) or meta.json["ee_pose"],
#     depending on --ee_pose_source and --action_label_source
#   - FK(joint_angles) matches the Raccoon move_to()/IK endpoint convention
#   - rotational deltas are zero-filled by default
#   - v12 can preserve raw action[4] as dpitch/pitch_alpha when --include_pitch_action is set
#   - gripper action uses raw step["action"][3] (0=open, 1=close)
#   - next_ee_delta final step action uses zero delta + current gripper_cmd

RACCOON_L1_CM = 8.25
RACCOON_L2_CM = 10.0
RACCOON_L3_CM = 10.0
RACCOON_L4_CM = 8.0


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_float_list(values: List[Any]) -> List[float]:
    return [float(v) for v in values]


def pad_joint_state(joint_angles: List[Any], gripper_state: float, joint_pad_dim: int = 7) -> List[float]:
    joints = to_float_list(joint_angles)
    if len(joints) > joint_pad_dim:
        raise ValueError(f"joint_angles length {len(joints)} exceeds joint_pad_dim {joint_pad_dim}")
    padded = joints + [0.0] * (joint_pad_dim - len(joints))
    return padded + [float(gripper_state)]


def raccoon_fk_ee_pose_from_joints(joint_angles: List[Any]) -> List[float]:
    """
    Compute the command-space EE endpoint in meters from RaccoonBot joint angles.

    This mirrors SyncSimRaccoonEnv.get_ee_pose() and the IK convention used by
    move_to(). The raw generator's older Link4 body pose is offset from this
    endpoint, which makes close-pose analysis and delta execution disagree.
    """
    joints = to_float_list(joint_angles)
    if len(joints) < 3:
        raise ValueError(f"joint_angles must have at least 3 values for FK, got {len(joints)}")

    th1, th2, th3 = joints[:3]
    r = -RACCOON_L2_CM * math.sin(th2) - RACCOON_L3_CM * math.sin(th2 + th3)
    z = RACCOON_L1_CM + RACCOON_L2_CM * math.cos(th2) + RACCOON_L3_CM * math.cos(th2 + th3)
    r_tip = r + RACCOON_L4_CM

    x_cm = -math.sin(th1) * r_tip
    y_cm = math.cos(th1) * r_tip
    z_cm = z

    return [x_cm / 100.0, y_cm / 100.0, z_cm / 100.0]


def resolve_ee_pose(step: Dict[str, Any], ee_pose_source: str) -> List[float]:
    if ee_pose_source == "fk":
        return raccoon_fk_ee_pose_from_joints(step.get("joint_angles", []))
    if ee_pose_source == "logged":
        ee_pose = to_float_list(step.get("ee_pose", []))
        if len(ee_pose) < 3:
            raise ValueError(f"logged ee_pose must have at least 3 dims, got {len(ee_pose)}")
        return ee_pose[:3]
    raise ValueError(f"Unsupported ee_pose_source: {ee_pose_source}")


def gripper_action_from_raw(step: Dict[str, Any]) -> float:
    """
    Raw step["action"] is [target_x, target_y, target_z, gripper_cmd].
    Keep the last element as gripper command.
    0.0=open, 1.0=close.
    """
    raw_action = step.get("action", [0.0, 0.0, 0.0, 0.0])
    if len(raw_action) < 4:
        return 0.0
    return 1.0 if float(raw_action[3]) >= 0.5 else 0.0


def pitch_action_from_raw_step(step: Dict[str, Any]) -> float:
    raw_action = to_float_list(step.get("action", []))
    if len(raw_action) >= 5:
        return float(max(0.0, min(1.0, raw_action[4])))
    return 0.0


def pitch_action_from_raw_waypoint(raw_waypoint_action: List[Any]) -> float:
    raw_action = to_float_list(raw_waypoint_action)
    if len(raw_action) >= 5:
        return float(max(0.0, min(1.0, raw_action[4])))
    return 0.0


def find_first_raw_close_index(raw_steps: List[Dict[str, Any]]) -> Optional[int]:
    for i, step in enumerate(raw_steps):
        if raw_step_gripper_command(step) >= 0.5:
            return i
    return None


def make_pre_close_promotion_indices(raw_steps: List[Dict[str, Any]], promote_pre_close_steps: int) -> set[int]:
    if promote_pre_close_steps <= 0:
        return set()

    first_close = find_first_raw_close_index(raw_steps)
    if first_close is None:
        return set()

    start = max(0, first_close - int(promote_pre_close_steps))
    return {
        i
        for i in range(start, first_close)
        if raw_step_gripper_command(raw_steps[i]) < 0.5
    }


def expand_stack_release_open_indices(
    raw_steps: List[Dict[str, Any]],
    kept_indices: List[int],
    promoted_pre_close_indices: set[int],
    task_type: str,
    stack_release_open_repeat: int,
) -> Tuple[List[int], int]:
    """
    Repeat only stack release/open supervision after the first close.

    Stack failures were mostly "object is over the base, but the gripper never
    opens". Repeating the post-close open frames strengthens that final release
    decision without flooding lift-only episodes with extra open labels.
    """
    repeat = max(0, int(stack_release_open_repeat))
    if repeat <= 0 or str(task_type) != "stack":
        return kept_indices, 0

    first_close = find_first_raw_close_index(raw_steps)
    if first_close is None:
        return kept_indices, 0

    expanded: List[int] = []
    duplicates = 0
    for raw_i in kept_indices:
        expanded.append(raw_i)
        is_post_close = raw_i > first_close
        is_open_cmd = effective_gripper_command(raw_steps, raw_i, promoted_pre_close_indices) < 0.5
        if is_post_close and is_open_cmd:
            expanded.extend([raw_i] * repeat)
            duplicates += repeat

    return expanded, duplicates


def effective_gripper_command(
    raw_steps: List[Dict[str, Any]],
    raw_index: int,
    promoted_pre_close_indices: set[int],
) -> float:
    if raw_index in promoted_pre_close_indices:
        return 1.0
    return 1.0 if raw_step_gripper_command(raw_steps[raw_index]) >= 0.5 else 0.0


def ee_delta_action(
    curr_ee_pose: List[Any],
    next_ee_pose: Optional[List[Any]],
) -> List[float]:
    """
    Build EEF_POS action:
      [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
    Raw ee_pose is assumed to contain at least [x, y, z].
    Since raw data does not include EE orientation, rotational deltas are zero-filled.
    """
    curr = to_float_list(curr_ee_pose)
    if len(curr) < 3:
        raise ValueError(f"ee_pose must have at least 3 dims, got {len(curr)}")

    if next_ee_pose is None:
        dpos = [0.0, 0.0, 0.0]
    else:
        nxt = to_float_list(next_ee_pose)
        if len(nxt) < 3:
            raise ValueError(f"next ee_pose must have at least 3 dims, got {len(nxt)}")
        dpos = [float(n - c) for c, n in zip(curr[:3], nxt[:3])]

    return [dpos[0], dpos[1], dpos[2], 0.0, 0.0, 0.0]


def command_delta_action(
    curr_ee_pose: List[Any],
    raw_waypoint_action: List[Any],
    include_pitch_action: bool = False,
) -> List[float]:
    """
    Build a controller-command delta:
      [raw_waypoint_xyz - current_ee_xyz, 0,0,0]

    The raw generator stores the expert waypoint command in step["action"] as
    [target_x, target_y, target_z, gripper]. This label is aligned with
    closed-loop deployment where the policy output is interpreted as the next
    controller target delta. In contrast, next_ee_delta labels the actual
    movement after one 10Hz control tick, which can under-drive rollout control
    when reused as a target command.
    """
    curr = to_float_list(curr_ee_pose)
    raw_action = to_float_list(raw_waypoint_action)
    if len(curr) < 3:
        raise ValueError(f"ee_pose must have at least 3 dims, got {len(curr)}")
    if len(raw_action) < 3:
        raise ValueError(f"raw waypoint action must have at least 3 dims, got {len(raw_action)}")

    dpitch = pitch_action_from_raw_waypoint(raw_waypoint_action) if include_pitch_action else 0.0
    return [
        float(raw_action[0] - curr[0]),
        float(raw_action[1] - curr[1]),
        float(raw_action[2] - curr[2]),
        0.0,
        float(dpitch),
        0.0,
    ]


def is_idle_transition(
    curr_step: Dict[str, Any],
    next_step: Optional[Dict[str, Any]],
    min_joint_delta_norm: float,
    min_gripper_delta: float,
    min_ee_delta_norm: float,
    ee_pose_source: str,
) -> bool:
    if next_step is None:
        return False

    curr_joint = to_float_list(curr_step.get("joint_angles", []))
    next_joint = to_float_list(next_step.get("joint_angles", []))
    if len(curr_joint) != len(next_joint):
        return False

    dq = [n - c for c, n in zip(curr_joint, next_joint)]
    joint_delta_norm = sum(v * v for v in dq) ** 0.5
    grip_delta = abs(float(next_step.get("gripper_state", 0.0)) - float(curr_step.get("gripper_state", 0.0)))

    curr_ee = resolve_ee_pose(curr_step, ee_pose_source)
    next_ee = resolve_ee_pose(next_step, ee_pose_source)
    if len(curr_ee) >= 3 and len(next_ee) >= 3:
        dee = [n - c for c, n in zip(curr_ee[:3], next_ee[:3])]
        ee_delta_norm = sum(v * v for v in dee) ** 0.5
    else:
        ee_delta_norm = float("inf")

    return (
        joint_delta_norm < min_joint_delta_norm
        and grip_delta < min_gripper_delta
        and ee_delta_norm < min_ee_delta_norm
    )


def copy_episode_images(raw_episode_dir: Path, out_images_dir: Path, referenced_files: List[str]) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    for image_file in referenced_files:
        src = raw_episode_dir / image_file
        dst = out_images_dir / image_file
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        shutil.copy2(src, dst)


def raw_step_gripper_command(raw_step: Dict[str, Any]) -> float:
    action = to_float_list(raw_step.get("action", []))
    if len(action) >= 4:
        return float(action[3])
    return float(raw_step.get("gripper_state", 0.0))


def transition_z_action(
    curr_step: Dict[str, Any],
    next_step: Optional[Dict[str, Any]],
    ee_pose_source: str,
    action_label_source: str,
    include_pitch_action: bool = False,
) -> float:
    curr_ee_pose = resolve_ee_pose(curr_step, ee_pose_source)
    next_ee_pose = resolve_ee_pose(next_step, ee_pose_source) if next_step is not None else None
    raw_waypoint_action = to_float_list(curr_step.get("action", []))

    if action_label_source == "next_ee_delta":
        ee_delta = ee_delta_action(curr_ee_pose=curr_ee_pose, next_ee_pose=next_ee_pose)
    elif action_label_source == "command_delta":
        ee_delta = command_delta_action(
            curr_ee_pose=curr_ee_pose,
            raw_waypoint_action=raw_waypoint_action,
            include_pitch_action=include_pitch_action,
        )
    else:
        raise ValueError(f"Unsupported action_label_source: {action_label_source}")

    return float(ee_delta[2]) if len(ee_delta) >= 3 else 0.0


def transition_xyz_action(
    curr_step: Dict[str, Any],
    next_step: Optional[Dict[str, Any]],
    ee_pose_source: str,
    action_label_source: str,
    include_pitch_action: bool = False,
) -> List[float]:
    curr_ee_pose = resolve_ee_pose(curr_step, ee_pose_source)
    next_ee_pose = resolve_ee_pose(next_step, ee_pose_source) if next_step is not None else None
    raw_waypoint_action = to_float_list(curr_step.get("action", []))

    if action_label_source == "next_ee_delta":
        ee_delta = ee_delta_action(curr_ee_pose=curr_ee_pose, next_ee_pose=next_ee_pose)
    elif action_label_source == "command_delta":
        ee_delta = command_delta_action(
            curr_ee_pose=curr_ee_pose,
            raw_waypoint_action=raw_waypoint_action,
            include_pitch_action=include_pitch_action,
        )
    else:
        raise ValueError(f"Unsupported action_label_source: {action_label_source}")

    return [float(x) for x in ee_delta[:3]]


def convert_episode(
    raw_episode_dir: Path,
    out_episode_dir: Path,
    joint_pad_dim: int = 7,
    include_failed: bool = False,
    drop_idle_steps: bool = False,
    min_joint_delta_norm: float = 1e-4,
    min_gripper_delta: float = 1e-4,
    min_ee_delta_norm: float = 1e-6,
    keep_debug_fields: bool = True,
    ee_pose_source: str = "fk",
    action_label_source: str = "command_delta",
    drop_post_close_hold_steps: int = 0,
    drop_closed_gripper_small_z_actions: bool = False,
    closed_gripper_min_z_action: float = 0.002,
    closed_gripper_min_xy_action: float = 0.002,
    promote_pre_close_steps: int = 0,
    initial_close_min_z_action: Optional[float] = None,
    include_pitch_action: bool = False,
    stack_release_open_repeat: int = 0,
) -> Optional[Dict[str, Any]]:
    meta_path = raw_episode_dir / "meta.json"
    if not meta_path.exists():
        print(f"[WARN] skip {raw_episode_dir.name}: meta.json not found")
        return None

    meta = read_json(meta_path)
    success = bool(meta.get("success", False))
    if not include_failed and not success:
        print(f"[SKIP] {raw_episode_dir.name}: failed episode")
        return None

    raw_steps = meta.get("steps", [])
    if not raw_steps:
        print(f"[WARN] skip {raw_episode_dir.name}: empty steps")
        return None

    promoted_pre_close_indices = make_pre_close_promotion_indices(raw_steps, promote_pre_close_steps)
    first_raw_close_index = find_first_raw_close_index(raw_steps)
    protected_close_indices = set(promoted_pre_close_indices)
    if first_raw_close_index is not None:
        protected_close_indices.add(first_raw_close_index)

    kept_indices: List[int] = []
    first_close_seen = False
    post_close_closed_seen = 0
    dropped_idle_steps = 0
    dropped_post_close_hold_steps = 0
    dropped_closed_small_z_steps = 0
    promoted_pre_close_kept_steps = 0
    duplicated_stack_release_open_steps = 0

    for i in range(len(raw_steps)):
        curr_step = raw_steps[i]
        next_step = raw_steps[i + 1] if i + 1 < len(raw_steps) else None
        is_final_raw_step = i == len(raw_steps) - 1
        gripper_cmd = effective_gripper_command(raw_steps, i, promoted_pre_close_indices)
        is_closed_cmd = gripper_cmd >= 0.5
        is_first_close = bool(is_closed_cmd and not first_close_seen)

        if drop_idle_steps and i < len(raw_steps) - 1 and i not in protected_close_indices:
            if is_idle_transition(
                curr_step,
                next_step,
                min_joint_delta_norm,
                min_gripper_delta,
                min_ee_delta_norm,
                ee_pose_source,
            ):
                dropped_idle_steps += 1
                continue

        if is_closed_cmd:
            if is_first_close:
                first_close_seen = True
            else:
                post_close_closed_seen += 1
                if not is_final_raw_step:
                    xyz_action = transition_xyz_action(
                        curr_step=curr_step,
                        next_step=next_step,
                        ee_pose_source=ee_pose_source,
                        action_label_source=action_label_source,
                        include_pitch_action=include_pitch_action,
                    )
                    z_action = float(xyz_action[2])
                    xy_action_norm = (float(xyz_action[0]) ** 2 + float(xyz_action[1]) ** 2) ** 0.5
                    is_small_z = abs(z_action) < closed_gripper_min_z_action
                    is_small_xy = xy_action_norm < closed_gripper_min_xy_action
                    if (
                        drop_post_close_hold_steps > 0
                        and post_close_closed_seen <= drop_post_close_hold_steps
                        and is_small_z
                        and is_small_xy
                        and i not in protected_close_indices
                    ):
                        dropped_post_close_hold_steps += 1
                        continue
                    if (
                        drop_closed_gripper_small_z_actions
                        and is_small_z
                        and is_small_xy
                        and i not in protected_close_indices
                    ):
                        dropped_closed_small_z_steps += 1
                        continue
        kept_indices.append(i)
        if i in promoted_pre_close_indices:
            promoted_pre_close_kept_steps += 1

    if not kept_indices:
        print(f"[WARN] skip {raw_episode_dir.name}: all steps filtered out")
        return None

    kept_steps = [raw_steps[i] for i in kept_indices]

    if kept_indices[-1] != len(raw_steps) - 1:
        kept_steps.append(raw_steps[-1])
        kept_indices.append(len(raw_steps) - 1)

    kept_indices, duplicated_stack_release_open_steps = expand_stack_release_open_indices(
        raw_steps=raw_steps,
        kept_indices=kept_indices,
        promoted_pre_close_indices=promoted_pre_close_indices,
        task_type=str(meta.get("task_type", "grasp")),
        stack_release_open_repeat=stack_release_open_repeat,
    )
    kept_steps = [raw_steps[i] for i in kept_indices]

    out_images_dir = out_episode_dir / "images"
    referenced_files = [str(step["image_file"]) for step in kept_steps]
    copy_episode_images(raw_episode_dir, out_images_dir, referenced_files)

    episode_steps: List[Dict[str, Any]] = []
    num_steps = len(kept_steps)
    instruction = str(meta.get("instruction", ""))

    for local_i, raw_i in enumerate(kept_indices):
        curr = raw_steps[raw_i]
        next_raw_i = kept_indices[local_i + 1] if local_i + 1 < len(kept_indices) else None
        nxt = raw_steps[next_raw_i] if next_raw_i is not None else None

        state = pad_joint_state(
            joint_angles=curr.get("joint_angles", []),
            gripper_state=float(curr.get("gripper_state", 0.0)),
            joint_pad_dim=joint_pad_dim,
        )

        curr_ee_pose = resolve_ee_pose(curr, ee_pose_source)
        next_ee_pose = resolve_ee_pose(nxt, ee_pose_source) if nxt is not None else None
        raw_waypoint_action = to_float_list(curr.get("action", []))
        if raw_i in promoted_pre_close_indices:
            if len(raw_waypoint_action) < 4:
                raw_waypoint_action = (raw_waypoint_action + [0.0, 0.0, 0.0, 0.0])[:4]
            raw_waypoint_action[3] = 1.0
        if action_label_source == "next_ee_delta":
            ee_delta = ee_delta_action(curr_ee_pose=curr_ee_pose, next_ee_pose=next_ee_pose)
        elif action_label_source == "command_delta":
            ee_delta = command_delta_action(
                curr_ee_pose=curr_ee_pose,
                raw_waypoint_action=raw_waypoint_action,
                include_pitch_action=include_pitch_action,
            )
        else:
            raise ValueError(f"Unsupported action_label_source: {action_label_source}")
        grip_cmd = effective_gripper_command(raw_steps, raw_i, promoted_pre_close_indices)
        if (
            initial_close_min_z_action is not None
            and raw_i in protected_close_indices
            and grip_cmd >= 0.5
        ):
            ee_delta[2] = max(float(ee_delta[2]), float(initial_close_min_z_action))
        action = ee_delta + [grip_cmd]

        is_last = local_i == (num_steps - 1)
        is_terminal = bool(is_last and success)
        reward = 1.0 if is_terminal else 0.0
        discount = 0.0 if is_terminal else 1.0

        step_item: Dict[str, Any] = {
            "observation": {
                "image": f"images/{curr['image_file']}",
                "state": state,
            },
            "action": action,
            "language_instruction": instruction,
            "reward": reward,
            "discount": discount,
            "is_first": local_i == 0,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "timestep": int(curr.get("t", local_i)),
            "raw_index": int(raw_i),
            "raw_waypoint_action": raw_waypoint_action,
        }

        if keep_debug_fields:
            step_item["observation"]["joint_angles_raw"] = to_float_list(curr.get("joint_angles", []))
            step_item["observation"]["gripper_state_raw"] = float(curr.get("gripper_state", 0.0))
            step_item["observation"]["ee_pose"] = to_float_list(curr_ee_pose)
            step_item["observation"]["ee_pose_source"] = ee_pose_source
            if ee_pose_source == "fk":
                step_item["observation"]["ee_pose_logged"] = to_float_list(curr.get("ee_pose", []))
            step_item["observation"]["object_pose"] = to_float_list(curr.get("object_pose", []))

        episode_steps.append(step_item)

    episode_json = {
        "episode_metadata": {
            "episode_id": int(meta.get("episode_id", -1)),
            "scene_id": meta.get("scene_id"),
            "task_type": meta.get("task_type", "grasp"),
            "instruction": instruction,
            "success": success,
            "target_color": meta.get("target_color"),
            "source_color": meta.get("source_color"),
            "base_color": meta.get("base_color"),
            "goal_xy": to_float_list(meta.get("goal_xy", [])),
            "box_init_xy": to_float_list(meta.get("box_init_xy", [])),
            "box_init_yaw": float(meta.get("box_init_yaw", 0.0)),
            "raw_episode_dir": raw_episode_dir.name,
            "num_steps_raw": len(raw_steps),
            "num_steps_converted": len(episode_steps),
            "num_steps_after_filters": len(episode_steps),
            "dropped_idle_steps": int(dropped_idle_steps),
            "dropped_post_close_hold_steps": int(dropped_post_close_hold_steps),
            "dropped_closed_small_z_steps": int(dropped_closed_small_z_steps),
            "promoted_pre_close_steps": int(promoted_pre_close_kept_steps),
            "duplicated_stack_release_open_steps": int(duplicated_stack_release_open_steps),
            "promote_pre_close_steps": int(promote_pre_close_steps),
            "stack_release_open_repeat": int(stack_release_open_repeat),
            "drop_post_close_hold_steps": int(drop_post_close_hold_steps),
            "drop_closed_gripper_small_z_actions": bool(drop_closed_gripper_small_z_actions),
            "closed_gripper_min_z_action": float(closed_gripper_min_z_action),
            "closed_gripper_min_xy_action": float(closed_gripper_min_xy_action),
            "initial_close_min_z_action": (
                None if initial_close_min_z_action is None else float(initial_close_min_z_action)
            ),
            "include_pitch_action": bool(include_pitch_action),
            "joint_state_dim": joint_pad_dim + 1,
            "eef_action_dim": 7,
            "action_semantics": {
                "type": "EEF_POS",
                "ee_position_action": (
                    "next command-space ee_pose[:3] - current command-space ee_pose[:3]"
                    if action_label_source == "next_ee_delta"
                    else "raw waypoint action[:3] - current command-space ee_pose[:3]"
                ),
                "ee_pose_source": ee_pose_source,
                "action_label_source": action_label_source,
                "ee_rotation_action": (
                    "dpitch stores raw waypoint action[4] pitch_alpha command; droll/dyaw are 0"
                    if include_pitch_action
                    else "[0,0,0] because raw data has no EE orientation deltas"
                ),
                "gripper_action": "raw step['action'][3] mapped to 0=open, 1=close",
            },
        },
        "steps": episode_steps,
    }

    write_json(out_episode_dir / "episode.json", episode_json)

    return {
        "episode_id": int(meta.get("episode_id", -1)),
        "scene_id": meta.get("scene_id"),
        "instruction": instruction,
        "success": success,
        "raw_episode_dir": raw_episode_dir.name,
        "relative_episode_json": str((out_episode_dir / "episode.json").as_posix()),
        "num_steps_raw": len(raw_steps),
        "num_steps_converted": len(episode_steps),
        "dropped_idle_steps": int(dropped_idle_steps),
        "dropped_post_close_hold_steps": int(dropped_post_close_hold_steps),
        "dropped_closed_small_z_steps": int(dropped_closed_small_z_steps),
        "promoted_pre_close_steps": int(promoted_pre_close_kept_steps),
        "duplicated_stack_release_open_steps": int(duplicated_stack_release_open_steps),
    }


def episode_split_key(episode_dir: Path, split_by_scene: bool) -> str:
    if not split_by_scene:
        return episode_dir.name

    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        return episode_dir.name

    try:
        meta = read_json(meta_path)
    except Exception:
        return episode_dir.name

    scene_id = meta.get("scene_id")
    if scene_id is None:
        return episode_dir.name
    return f"scene:{scene_id}"


def make_split_lists(
    episode_dirs: List[Path],
    val_ratio: float,
    seed: int,
    split_by_scene: bool = False,
) -> Tuple[List[Path], List[Path]]:
    episode_dirs = list(episode_dirs)
    rng = random.Random(seed)

    if val_ratio <= 0.0:
        return sorted(episode_dirs), []

    if split_by_scene:
        groups: Dict[str, List[Path]] = {}
        for episode_dir in episode_dirs:
            groups.setdefault(episode_split_key(episode_dir, split_by_scene=True), []).append(episode_dir)

        group_keys = sorted(groups)
        rng.shuffle(group_keys)
        val_group_count = int(round(len(group_keys) * val_ratio))
        val_group_count = max(1, val_group_count) if len(group_keys) > 1 else 0
        val_group_keys = set(group_keys[:val_group_count])

        train_dirs = []
        val_dirs = []
        for group_key, group_dirs in groups.items():
            if group_key in val_group_keys:
                val_dirs.extend(group_dirs)
            else:
                train_dirs.extend(group_dirs)
    else:
        rng.shuffle(episode_dirs)
        val_count = int(round(len(episode_dirs) * val_ratio))
        val_count = max(1, val_count) if len(episode_dirs) > 1 else 0
        val_dirs = episode_dirs[:val_count]
        train_dirs = episode_dirs[val_count:]

    return train_dirs, val_dirs


def convert_dataset(
    raw_root: Path,
    out_root: Path,
    joint_pad_dim: int = 7,
    include_failed: bool = False,
    val_ratio: float = 0.0,
    seed: int = 42,
    drop_idle_steps: bool = False,
    min_joint_delta_norm: float = 1e-4,
    min_gripper_delta: float = 1e-4,
    min_ee_delta_norm: float = 1e-6,
    keep_debug_fields: bool = True,
    ee_pose_source: str = "fk",
    action_label_source: str = "command_delta",
    split_by_scene: bool = False,
    drop_post_close_hold_steps: int = 0,
    drop_closed_gripper_small_z_actions: bool = False,
    closed_gripper_min_z_action: float = 0.002,
    closed_gripper_min_xy_action: float = 0.002,
    promote_pre_close_steps: int = 0,
    initial_close_min_z_action: Optional[float] = None,
    include_pitch_action: bool = False,
    stack_release_open_repeat: int = 0,
) -> None:
    raw_root = raw_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted([p for p in raw_root.glob("episode_*") if p.is_dir()])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found under: {raw_root}")

    train_dirs, val_dirs = make_split_lists(
        episode_dirs,
        val_ratio=val_ratio,
        seed=seed,
        split_by_scene=split_by_scene,
    )

    train_manifest: List[Dict[str, Any]] = []
    val_manifest: List[Dict[str, Any]] = []

    split_map = [("train", train_dirs, train_manifest)]
    if val_dirs:
        split_map.append(("val", val_dirs, val_manifest))

    for split_name, split_dirs, manifest in split_map:
        for raw_episode_dir in split_dirs:
            out_episode_dir = out_root / split_name / raw_episode_dir.name
            out_episode_dir.mkdir(parents=True, exist_ok=True)

            result = convert_episode(
                raw_episode_dir=raw_episode_dir,
                out_episode_dir=out_episode_dir,
                joint_pad_dim=joint_pad_dim,
                include_failed=include_failed,
                drop_idle_steps=drop_idle_steps,
                min_joint_delta_norm=min_joint_delta_norm,
                min_gripper_delta=min_gripper_delta,
                min_ee_delta_norm=min_ee_delta_norm,
                keep_debug_fields=keep_debug_fields,
                ee_pose_source=ee_pose_source,
                action_label_source=action_label_source,
                drop_post_close_hold_steps=drop_post_close_hold_steps,
                drop_closed_gripper_small_z_actions=drop_closed_gripper_small_z_actions,
                closed_gripper_min_z_action=closed_gripper_min_z_action,
                closed_gripper_min_xy_action=closed_gripper_min_xy_action,
                promote_pre_close_steps=promote_pre_close_steps,
                initial_close_min_z_action=initial_close_min_z_action,
                include_pitch_action=include_pitch_action,
                stack_release_open_repeat=stack_release_open_repeat,
            )
            if result is None:
                shutil.rmtree(out_episode_dir, ignore_errors=True)
                continue

            result["split"] = split_name
            result["relative_episode_json"] = str((Path(split_name) / raw_episode_dir.name / "episode.json").as_posix())
            manifest.append(result)
            print(
                f"[OK] {split_name}/{raw_episode_dir.name} | "
                f"steps {result['num_steps_raw']} -> {result['num_steps_converted']} | success={result['success']}"
            )

    write_jsonl(out_root / "manifest_train.jsonl", train_manifest)
    if val_dirs:
        write_jsonl(out_root / "manifest_val.jsonl", val_manifest)

    dataset_info = {
        "format_name": "openvla_rlds_intermediate_eef_pos",
        "raw_root": str(raw_root),
        "joint_state_dim": joint_pad_dim + 1,
        "eef_action_dim": 7,
        "joint_dims_padded_to": joint_pad_dim,
        "train_episodes": len(train_manifest),
        "val_episodes": len(val_manifest),
        "include_failed": include_failed,
        "drop_idle_steps": drop_idle_steps,
        "split_by_scene": split_by_scene,
        "min_joint_delta_norm": min_joint_delta_norm,
        "min_gripper_delta": min_gripper_delta,
        "min_ee_delta_norm": min_ee_delta_norm,
        "ee_pose_source": ee_pose_source,
        "action_label_source": action_label_source,
        "drop_post_close_hold_steps": int(drop_post_close_hold_steps),
        "drop_closed_gripper_small_z_actions": bool(drop_closed_gripper_small_z_actions),
        "closed_gripper_min_z_action": float(closed_gripper_min_z_action),
        "closed_gripper_min_xy_action": float(closed_gripper_min_xy_action),
        "promote_pre_close_steps": int(promote_pre_close_steps),
        "stack_release_open_repeat": int(stack_release_open_repeat),
        "initial_close_min_z_action": (
            None if initial_close_min_z_action is None else float(initial_close_min_z_action)
        ),
        "include_pitch_action": bool(include_pitch_action),
        "schema": {
            "episode_json": {
                "episode_metadata": {
                    "episode_id": "int",
                    "scene_id": "int|null",
                    "task_type": "str",
                    "instruction": "str",
                    "success": "bool",
                    "target_color": "str|null",
                    "source_color": "str|null",
                    "base_color": "str|null",
                    "goal_xy": "list[float]",
                    "box_init_xy": "list[float]",
                    "box_init_yaw": "float",
                    "raw_episode_dir": "str",
                    "num_steps_raw": "int",
                    "num_steps_converted": "int",
                    "dropped_idle_steps": "int",
                    "dropped_post_close_hold_steps": "int",
                    "dropped_closed_small_z_steps": "int",
                    "promoted_pre_close_steps": "int",
                },
                "steps": {
                    "observation.image": "relative image path",
                    "observation.state": f"list[float] length {joint_pad_dim + 1}",
                    "action": "list[float] length 7 = [dx,dy,dz,droll,dpitch,dyaw,gripper_cmd]",
                    "language_instruction": "str",
                    "reward": "float",
                    "discount": "float",
                    "is_first": "bool",
                    "is_last": "bool",
                    "is_terminal": "bool",
                    "timestep": "int",
                },
            }
        },
        "notes": [
            "state = [joint_angles padded to 7, gripper_state]",
            (
                "action = [next command-space ee_pose[:3] - current command-space ee_pose[:3], 0,0,0, gripper_cmd]"
                if action_label_source == "next_ee_delta"
                else "action = [raw waypoint action[:3] - current command-space ee_pose[:3], 0,0,0, gripper_cmd]"
            ),
            "ee_pose_source=fk recomputes the endpoint from joint angles to match rollout-time delta execution",
            (
                "v12 pitch action enabled: action[4] stores raw pitch_alpha command "
                "(0 horizontal, 1 vertical) for joint4/gripper orientation"
                if include_pitch_action
                else "rotational deltas are zero-filled because raw data does not include EE orientation deltas"
            ),
            "raw waypoint action is preserved in each step as raw_waypoint_action for debugging",
            (
                f"promote_pre_close_steps={int(promote_pre_close_steps)} relabels the final open "
                "frames before the first raw close as close commands to improve close-transition stability"
                if promote_pre_close_steps > 0
                else "promote_pre_close_steps=0 leaves raw gripper commands unchanged"
            ),
            (
                f"stack_release_open_repeat={int(stack_release_open_repeat)} repeats post-close stack open frames "
                "to strengthen release supervision"
                if stack_release_open_repeat > 0
                else "stack_release_open_repeat=0 leaves stack release/open frames unweighted"
            ),
            (
                f"initial_close_min_z_action={float(initial_close_min_z_action)} enforces immediate lift labels "
                "on promoted close frames and the first raw close frame"
                if initial_close_min_z_action is not None
                else "initial_close_min_z_action=None leaves initial close z labels unchanged"
            ),
            "This is an intermediate format; your TFDS/RLDS builder should load observation.image from disk and emit actual RLDS records.",
        ],
    }
    write_json(out_root / "dataset_info.json", dataset_info)

    print("\nDone.")
    print(f"train episodes: {len(train_manifest)}")
    print(f"val episodes  : {len(val_manifest)}")
    print(f"output root   : {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw robot episodes to OpenVLA RLDS intermediate format.")
    parser.add_argument("--raw_root", type=str, default="./dataset_raw_4color_dynamic_center_camera_visible_grasp", help="Root directory containing episode_*/meta.json")
    parser.add_argument("--out_root", type=str, default="./rlds_out", help="Output directory for intermediate dataset")
    parser.add_argument("--joint_pad_dim", type=int, default=7, help="Pad joint states to this many joints")
    parser.add_argument("--include_failed", action="store_true", help="Include failed episodes too")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Validation split ratio, e.g. 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--drop_idle_steps", action="store_true", help="Drop tiny-motion transitions")
    parser.add_argument(
        "--split_by_scene",
        action="store_true",
        help="Keep episodes with the same raw scene_id in the same train/val split.",
    )
    parser.add_argument("--min_joint_delta_norm", type=float, default=1e-4, help="Idle threshold for joint delta norm")
    parser.add_argument("--min_gripper_delta", type=float, default=1e-4, help="Idle threshold for gripper delta")
    parser.add_argument("--min_ee_delta_norm", type=float, default=1e-6, help="Idle threshold for ee delta norm")
    parser.add_argument("--no_debug_fields", action="store_true", help="Do not keep ee/object/raw debug fields")
    parser.add_argument(
        "--ee_pose_source",
        choices=("fk", "logged"),
        default="fk",
        help="Use FK(joint_angles) command-space EE pose for action labels, or legacy logged meta['ee_pose'].",
    )
    parser.add_argument(
        "--action_label_source",
        choices=("next_ee_delta", "command_delta"),
        default="command_delta",
        help=(
            "next_ee_delta labels actual next-step EE motion; command_delta labels "
            "raw waypoint target minus current EE pose, which better matches rollout-time delta commands."
        ),
    )
    parser.add_argument(
        "--drop_post_close_hold_steps",
        type=int,
        default=0,
        help=(
            "Drop up to this many low-z closed-gripper transitions after the first close command. "
            "The first close frame is always kept."
        ),
    )
    parser.add_argument(
        "--drop_closed_gripper_small_z_actions",
        action="store_true",
        help=(
            "Drop non-terminal closed-gripper transitions whose labeled z action is smaller than "
            "--closed_gripper_min_z_action. The first close frame is always kept."
        ),
    )
    parser.add_argument(
        "--closed_gripper_min_z_action",
        type=float,
        default=0.002,
        help="Minimum absolute z action for closed-gripper transitions to remain in the training set.",
    )
    parser.add_argument(
        "--closed_gripper_min_xy_action",
        type=float,
        default=0.002,
        help=(
            "Closed-gripper transitions are treated as removable holds only when both "
            "abs(z action) and xy action norm are below their thresholds. This preserves "
            "stack/place horizontal transport while still dropping true post-close holds."
        ),
    )
    parser.add_argument(
        "--promote_pre_close_steps",
        type=int,
        default=0,
        help=(
            "Relabel this many final open-gripper raw frames before the first close command as close commands. "
            "Use this to strengthen close-transition supervision without regenerating successful raw episodes."
        ),
    )
    parser.add_argument(
        "--initial_close_min_z_action",
        type=float,
        default=None,
        help=(
            "If set, promoted pre-close frames and the first raw close frame get at least this positive z action. "
            "This makes initial closed-frame supervision point toward lift instead of continuing down."
        ),
    )
    parser.add_argument(
        "--include_pitch_action",
        action="store_true",
        help=(
            "Preserve raw action[4] as OpenVLA action dpitch. "
            "Used by v12 pitch-aware demonstrations where action[4] is pitch_alpha."
        ),
    )
    parser.add_argument(
        "--stack_release_open_repeat",
        type=int,
        default=0,
        help=(
            "For stack episodes only, repeat post-close open/release frames this many extra times. "
            "This strengthens place-and-release supervision without changing lift episodes."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        joint_pad_dim=args.joint_pad_dim,
        include_failed=args.include_failed,
        val_ratio=args.val_ratio,
        seed=args.seed,
        drop_idle_steps=args.drop_idle_steps,
        min_joint_delta_norm=args.min_joint_delta_norm,
        min_gripper_delta=args.min_gripper_delta,
        min_ee_delta_norm=args.min_ee_delta_norm,
        keep_debug_fields=not args.no_debug_fields,
        ee_pose_source=args.ee_pose_source,
        action_label_source=args.action_label_source,
        split_by_scene=args.split_by_scene,
        drop_post_close_hold_steps=args.drop_post_close_hold_steps,
        drop_closed_gripper_small_z_actions=args.drop_closed_gripper_small_z_actions,
        closed_gripper_min_z_action=args.closed_gripper_min_z_action,
        closed_gripper_min_xy_action=args.closed_gripper_min_xy_action,
        promote_pre_close_steps=args.promote_pre_close_steps,
        initial_close_min_z_action=args.initial_close_min_z_action,
        include_pitch_action=args.include_pitch_action,
        stack_release_open_repeat=args.stack_release_open_repeat,
    )
