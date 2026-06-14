import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from raccoon_grasp_multicolor_scene_dataset import (
    DatasetLogger,
    SyncSimRaccoonDataset,
    record_final_grasp_alignment_metrics,
    record_lift_success_metrics,
)


STACK_TEMPLATE = "stack the {source_color} cylinder on the {base_color} cylinder"


def ordered_color_pairs(colors):
    colors = tuple(colors)
    return [(src, base) for src in colors for base in colors if src != base]


def balanced_pair_counts(num_episodes, pairs):
    base = int(num_episodes) // len(pairs)
    remainder = int(num_episodes) % len(pairs)
    return {
        pair: base + (1 if idx < remainder else 0)
        for idx, pair in enumerate(pairs)
    }


def sample_stack_object_specs(
    rng,
    source_color,
    base_color,
    colors=None,
    x_range=None,
    y_range=None,
    min_distance=0.045,
    max_tries=1000,
):
    colors = tuple(colors or SyncSimRaccoonDataset.CYLINDER_COLORS)
    x_range = x_range or SyncSimRaccoonDataset.DEFAULT_OBJECT_X_RANGE
    y_range = y_range or SyncSimRaccoonDataset.DEFAULT_OBJECT_Y_RANGE

    if source_color == base_color:
        raise ValueError("source_color and base_color must be different for stacking")
    for color in (source_color, base_color):
        if color not in colors:
            raise ValueError(f"{color} is not included in colors={colors}")

    x_min, x_max = [float(v) for v in x_range]
    y_min, y_max = [float(v) for v in y_range]
    placed = []
    specs = {}

    for _ in range(max_tries):
        base_x = float(rng.uniform(max(x_min + 0.030, -0.040), min(x_max - 0.030, 0.040)))
        base_y = float(rng.uniform(max(y_min, 0.145), min(y_max, 0.175)))

        side = float(rng.choice([-1.0, 1.0]))
        source_gap = float(rng.uniform(0.055, 0.075))
        source_x = base_x + side * source_gap
        if source_x < x_min + 0.010 or source_x > x_max - 0.010:
            source_x = base_x - side * source_gap
        source_y = base_y + float(rng.uniform(-0.014, 0.014))

        if not (x_min <= source_x <= x_max and y_min <= source_y <= y_max):
            continue
        if math.hypot(source_x - base_x, source_y - base_y) < min_distance:
            continue

        specs[source_color] = {
            "body_name": SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR[source_color],
            "x": float(source_x),
            "y": float(source_y),
            "yaw": float(rng.uniform(-math.pi, math.pi)),
        }
        specs[base_color] = {
            "body_name": SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR[base_color],
            "x": float(base_x),
            "y": float(base_y),
            "yaw": float(rng.uniform(-math.pi, math.pi)),
        }
        placed = [np.array([source_x, source_y]), np.array([base_x, base_y])]
        break
    else:
        raise RuntimeError("failed to sample source/base stack pair")

    for color in colors:
        if color in specs:
            continue
        for _ in range(max_tries):
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))
            xy = np.array([x, y])
            if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed):
                specs[color] = {
                    "body_name": SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR[color],
                    "x": x,
                    "y": y,
                    "yaw": float(rng.uniform(-math.pi, math.pi)),
                }
                placed.append(xy)
                break
        else:
            raise RuntimeError("failed to place remaining colored cylinders for stack scene")

    return {color: specs[color] for color in colors}


def make_stack_plan(rc, source_x, source_y, base_x, base_y):
    grasp_plan = rc.make_grasp_plan(source_x, source_y)

    travel_z_candidates = (0.075, 0.070, 0.065, 0.060)
    place_z_candidates = (0.036, 0.038, 0.034, 0.040, 0.032)
    travel_z = None
    place_z = None
    for candidate_travel_z in travel_z_candidates:
        if not rc.is_action_reachable([base_x, base_y, candidate_travel_z, 1.0]):
            continue
        for candidate_place_z in place_z_candidates:
            required = (
                [base_x, base_y, candidate_travel_z, 1.0],
                [base_x, base_y, 0.050, 1.0],
                [base_x, base_y, candidate_place_z, 1.0],
                [base_x, base_y, candidate_place_z, 0.0],
                [base_x, base_y, candidate_travel_z, 0.0],
            )
            if all(rc.is_action_reachable(action) for action in required):
                travel_z = candidate_travel_z
                place_z = candidate_place_z
                break
        if travel_z is not None:
            break

    if travel_z is None or place_z is None:
        raise ValueError(f"unreachable stack place waypoints for base=({base_x:.4f}, {base_y:.4f})")

    return grasp_plan + [
        [base_x, base_y, travel_z, 1.0],
        [base_x, base_y, travel_z, 1.0],
        [base_x, base_y, 0.050, 1.0],
        [base_x, base_y, place_z, 1.0],
        [base_x, base_y, place_z, 0.0],
        [base_x, base_y, place_z, 0.0],
        [base_x, base_y, travel_z, 0.0],
    ]


def record_stack_success_metrics(logger, rc, source_body_name, base_body_name):
    source_pose = rc.get_object_pose(source_body_name)
    base_pose = rc.get_object_pose(base_body_name)
    final_xy_dist = float(math.hypot(source_pose[0] - base_pose[0], source_pose[1] - base_pose[1]))
    final_z_delta = float(source_pose[2] - base_pose[2])
    gripper_open = bool(float(rc.data.qpos[4]) > (rc.GRIP_OPEN - 0.030))

    logger.meta["stack_source_final_pose"] = [float(x) for x in source_pose]
    logger.meta["stack_base_final_pose"] = [float(x) for x in base_pose]
    logger.meta["stack_final_xy_distance"] = final_xy_dist
    logger.meta["stack_final_z_delta"] = final_z_delta
    logger.meta["stack_gripper_open_final"] = gripper_open

    stack_success = bool(final_xy_dist <= 0.020 and final_z_delta >= 0.014 and gripper_open)
    logger.meta["strict_stack_success"] = stack_success
    return stack_success


def run_stack_episode_and_record(
    rc,
    logger,
    episode_id,
    instruction,
    object_specs,
    source_color,
    base_color,
    speed=150,
    settle_seconds_per_action=0.8,
    initial_settle_seconds=0.1,
    hz=10,
    touch_threshold=0.1,
    max_close_ee_z=0.025,
    max_close_xy_error=0.006,
    scene_id=None,
):
    if max_close_ee_z is not None and float(max_close_ee_z) < 0.0:
        max_close_ee_z = None
    if max_close_xy_error is not None and float(max_close_xy_error) < 0.0:
        max_close_xy_error = None

    source_spec = object_specs[source_color]
    base_spec = object_specs[base_color]
    source_body_name = source_spec["body_name"]
    base_body_name = base_spec["body_name"]
    source_x = float(source_spec["x"])
    source_y = float(source_spec["y"])
    base_x = float(base_spec["x"])
    base_y = float(base_spec["y"])

    rc.reset_episode(object_specs=object_specs, target_color=source_color)
    rc.lockh()
    if initial_settle_seconds > 0:
        rc.settle_steps(seconds=initial_settle_seconds)

    logger.start_episode(
        episode_id=episode_id,
        instruction=instruction,
        instruction_template=STACK_TEMPLATE,
        instruction_mode="stack_single_template",
        task_type="stack",
        goal_xy=[base_x, base_y],
        box_init_xy=[source_x, source_y],
        box_init_yaw=float(source_spec["yaw"]),
        target_color=source_color,
        target_body_name=source_body_name,
        all_object_init_poses=SyncSimRaccoonDataset.specs_to_meta(object_specs),
    )
    logger.meta["source_color"] = str(source_color)
    logger.meta["base_color"] = str(base_color)
    logger.meta["source_body_name"] = str(source_body_name)
    logger.meta["base_body_name"] = str(base_body_name)
    logger.meta["scene_id"] = int(scene_id) if scene_id is not None else None
    logger.meta["trajectory_mode"] = "v11_stack_extension"
    logger.meta["trajectory_mode_effective"] = "v11_stack_extension"

    try:
        plan = make_stack_plan(rc, source_x, source_y, base_x, base_y)
        waypoint_steps = [
            6, 3, 4, 12, 4, 1, 8, 8, 2,
            10, 4, 8, 6, 6, 4, 6,
        ]
        logger.meta["waypoint_steps"] = waypoint_steps
        logger.meta["grasp_target_z"] = float(next(action[2] for action in plan if float(action[3]) >= 0.5))
        logger.meta["place_target_z"] = float(plan[-3][2])
        logger.meta["place_target_xy"] = [base_x, base_y]

        obs = rc.get_observation(object_body_name=source_body_name)
        step_counter = 0
        for action_idx, action in enumerate(plan):
            rc.execute_action(action, speed=speed)
            for _ in range(int(waypoint_steps[action_idx])):
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )
                rc.settle_steps(seconds=1.0 / hz)
                obs = rc.get_observation(object_body_name=source_body_name)
                step_counter += 1

        # Let the released source settle briefly before the terminal frame.
        rc.settle_steps(seconds=0.35)
        obs = rc.get_observation(object_body_name=source_body_name)
        logger.log_step(
            step_idx=step_counter,
            image_rgb=obs["image"],
            joint_angles=obs["joint_angles"],
            gripper_state=obs["gripper_state"],
            object_pose=obs["object_pose"],
            ee_pose=obs["ee_pose"],
            action=plan[-1],
            is_first=False,
            is_last=True,
        )

        close_pose = next([float(x) for x in action[:3]] for action in plan if float(action[3]) >= 0.5)
        close_step = record_final_grasp_alignment_metrics(
            logger=logger,
            target_x=source_x,
            target_y=source_y,
            base_close_pose=close_pose,
        )
        strict_lift_success = record_lift_success_metrics(
            logger=logger,
            existing_success=True,
            close_step=close_step,
        )
        stack_success = record_stack_success_metrics(
            logger=logger,
            rc=rc,
            source_body_name=source_body_name,
            base_body_name=base_body_name,
        )

        close_ee_z = logger.meta.get("close_ee_z")
        close_xy_error = logger.meta.get("close_xy_distance_to_target")
        close_quality_success = True
        if max_close_ee_z is not None and close_ee_z is not None:
            close_quality_success = close_quality_success and float(close_ee_z) <= float(max_close_ee_z)
        if max_close_xy_error is not None and close_xy_error is not None:
            close_quality_success = close_quality_success and float(close_xy_error) <= float(max_close_xy_error)
        logger.meta["max_close_ee_z"] = None if max_close_ee_z is None else float(max_close_ee_z)
        logger.meta["max_close_xy_error"] = None if max_close_xy_error is None else float(max_close_xy_error)
        logger.meta["close_quality_success"] = bool(close_quality_success)

        success = bool(strict_lift_success and stack_success and close_quality_success)
        logger.finalize_episode(success=success)
        return success
    except Exception:
        logger.abort_episode()
        raise


def collect_stack_dataset(
    xml_path,
    dataset_root,
    num_episodes,
    colors=("red", "blue", "green", "yellow"),
    seed=None,
    max_attempts=None,
    keep_failed=False,
    camera_name="front_view",
    speed=150,
    settle_seconds_per_action=0.8,
    initial_settle_seconds=0.1,
    hz=10,
    object_x_range=SyncSimRaccoonDataset.DEFAULT_OBJECT_X_RANGE,
    object_y_range=SyncSimRaccoonDataset.DEFAULT_OBJECT_Y_RANGE,
    min_object_distance=0.045,
    max_close_ee_z=0.025,
    max_close_xy_error=0.006,
):
    colors = tuple(colors)
    rng = np.random.default_rng(seed)
    max_attempts = max_attempts or max(num_episodes * 25, num_episodes + 100)
    pairs = ordered_color_pairs(colors)
    target_pair_counts = balanced_pair_counts(num_episodes, pairs)
    pair_success_counts = {pair: 0 for pair in pairs}
    success_count = 0
    attempt_count = 0
    scene_id = 0

    logger = DatasetLogger(root_dir=dataset_root, keep_failed=keep_failed, overwrite_existing=True)
    rc = SyncSimRaccoonDataset(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=False,
    )

    try:
        while success_count < num_episodes and attempt_count < max_attempts:
            attempt_count += 1
            scene_id += 1
            remaining_pairs = [
                pair
                for pair in pairs
                if pair_success_counts[pair] < target_pair_counts[pair]
            ]
            if not remaining_pairs:
                break
            min_count = min(pair_success_counts[pair] for pair in remaining_pairs)
            candidate_pairs = [
                pair
                for pair in remaining_pairs
                if pair_success_counts[pair] == min_count
            ]
            source_color, base_color = candidate_pairs[int(rng.integers(0, len(candidate_pairs)))]
            instruction = STACK_TEMPLATE.format(source_color=source_color, base_color=base_color)
            episode_id = attempt_count if keep_failed else success_count + 1

            try:
                object_specs = sample_stack_object_specs(
                    rng=rng,
                    source_color=source_color,
                    base_color=base_color,
                    colors=colors,
                    x_range=object_x_range,
                    y_range=object_y_range,
                    min_distance=min_object_distance,
                )
                success = run_stack_episode_and_record(
                    rc=rc,
                    logger=logger,
                    episode_id=episode_id,
                    instruction=instruction,
                    object_specs=object_specs,
                    source_color=source_color,
                    base_color=base_color,
                    speed=speed,
                    settle_seconds_per_action=settle_seconds_per_action,
                    initial_settle_seconds=initial_settle_seconds,
                    hz=hz,
                    max_close_ee_z=max_close_ee_z,
                    max_close_xy_error=max_close_xy_error,
                    scene_id=scene_id,
                )
                if success:
                    success_count += 1
                    pair_success_counts[(source_color, base_color)] += 1
                print(
                    f"[Attempt {attempt_count:04d}] scene_id={scene_id:06d} | "
                    f"episode_id={episode_id:06d} | task_type='stack' | "
                    f"source='{source_color}' | base='{base_color}' | success={success} | "
                    f"success_count={success_count}/{num_episodes} | "
                    f"pair_count={pair_success_counts[(source_color, base_color)]}/"
                    f"{target_pair_counts[(source_color, base_color)]}"
                )
            except Exception as exc:
                print(
                    f"[Attempt {attempt_count:04d}] scene_id={scene_id:06d} | "
                    f"task_type='stack' | source='{source_color}' | base='{base_color}' | exception: {exc}"
                )
    finally:
        rc.close()

    if success_count < num_episodes:
        raise RuntimeError(
            f"Only generated {success_count}/{num_episodes} successful stack episodes "
            f"after {attempt_count} attempts"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Collect V11-extended cylinder stacking demonstrations.")
    parser.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--dataset_root", type=str, default="raccoon_stack_v11_extension")
    parser.add_argument("--num_episodes", type=int, default=40)
    parser.add_argument("--colors", nargs="+", default=list(SyncSimRaccoonDataset.CYLINDER_COLORS))
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--keep_failed", action="store_true")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--speed", type=int, default=150)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.8)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.1)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--object_x_min", type=float, default=SyncSimRaccoonDataset.DEFAULT_OBJECT_X_RANGE[0])
    parser.add_argument("--object_x_max", type=float, default=SyncSimRaccoonDataset.DEFAULT_OBJECT_X_RANGE[1])
    parser.add_argument("--object_y_min", type=float, default=0.135)
    parser.add_argument("--object_y_max", type=float, default=0.180)
    parser.add_argument("--min_object_distance", type=float, default=0.045)
    parser.add_argument("--max_close_ee_z", type=float, default=0.025)
    parser.add_argument("--max_close_xy_error", type=float, default=0.006)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_stack_dataset(
        xml_path=args.xml_path,
        dataset_root=args.dataset_root,
        num_episodes=args.num_episodes,
        colors=tuple(args.colors),
        seed=args.seed,
        max_attempts=args.max_attempts,
        keep_failed=args.keep_failed,
        camera_name=args.camera_name,
        speed=args.speed,
        settle_seconds_per_action=args.settle_seconds_per_action,
        initial_settle_seconds=args.initial_settle_seconds,
        hz=args.hz,
        object_x_range=(args.object_x_min, args.object_x_max),
        object_y_range=(args.object_y_min, args.object_y_max),
        min_object_distance=args.min_object_distance,
        max_close_ee_z=args.max_close_ee_z,
        max_close_xy_error=args.max_close_xy_error,
    )
