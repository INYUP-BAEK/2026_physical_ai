import argparse
import os
import json
import math
import shutil
from pathlib import Path

import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


DEFAULT_LANGUAGE_EXTENDED_TEMPLATES = (
    "grasp the {color} cylinder",
    "pick up the {color} cylinder",
    "grab the {color} cylinder",
    "hold the {color} cylinder",
    "move to the {color} cylinder and grasp it",
)

DEFAULT_LIFT_EXTENDED_TEMPLATES = (
    "grasp the {color} cylinder",
    "pick up the {color} cylinder",
    "lift the {color} cylinder",
    "raise the {color} cylinder",
    "grab the {color} cylinder",
    "grab and lift the {color} cylinder",
    "grasp and lift the {color} cylinder",
    "lift and hold the {color} cylinder",
    "pick up and hold the {color} cylinder",
    "take hold of the {color} cylinder",
    "move to the {color} cylinder and lift it",
    "pick up the cylinder that is {color}",
    "pick up only the {color} cylinder",
    "grasp only the {color} cylinder",
    "lift the {color} cylinder without touching the others",
)


class DatasetLogger:
    """
    Raw dataset logger.
    Saves:
      dataset_root/
        episode_000001/
          frame_000000.png
          frame_000001.png
          ...
          meta.json
    """
    def __init__(self, root_dir="dataset_raw", keep_failed=False, overwrite_existing=True):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_failed = keep_failed
        self.overwrite_existing = overwrite_existing
        self.episode_dir = None
        self.meta = None

    def start_episode(
        self,
        episode_id,
        instruction,
        goal_xy,
        box_init_xy,
        box_init_yaw,
        task_type="pick",
        instruction_template=None,
        instruction_mode="baseline",
        target_color=None,
        target_body_name=None,
        all_object_init_poses=None,
    ):
        episode_name = f"episode_{episode_id:06d}"
        self.episode_dir = self.root_dir / episode_name
        if self.episode_dir.exists():
            if not self.overwrite_existing:
                raise FileExistsError(f"Episode directory already exists: {self.episode_dir}")
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        self.episode_dir.mkdir(parents=True, exist_ok=False)

        self.meta = {
            "episode_id": int(episode_id),
            "instruction": str(instruction),
            "instruction_template": str(instruction_template if instruction_template is not None else instruction),
            "instruction_mode": str(instruction_mode),
            "task_type": str(task_type),
            # grasp-only에서는 별도 place goal이 없으므로 초기 box 위치를 goal_xy로 둔다.
            # 기존 intermediate/RLDS 변환 코드와 호환되도록 2차원 필드는 유지한다.
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "box_init_xy": [float(box_init_xy[0]), float(box_init_xy[1])],
            "box_init_yaw": float(box_init_yaw),
            "success": False,
            "steps": []
        }

        if target_color is not None:
            self.meta["target_color"] = str(target_color)
        if target_body_name is not None:
            self.meta["target_body_name"] = str(target_body_name)
        if all_object_init_poses is not None:
            self.meta["all_object_init_poses"] = all_object_init_poses

    def log_step(
        self,
        step_idx,
        image_rgb,
        joint_angles,
        gripper_state,
        object_pose,
        ee_pose,
        action,
        is_first=False,
        is_last=False,
    ):
        image_file = f"frame_{step_idx:06d}.png"
        image_path = self.episode_dir / image_file
        Image.fromarray(image_rgb).save(image_path)

        step_data = {
            "t": int(step_idx),
            "image_file": image_file,
            "joint_angles": [float(x) for x in joint_angles],
            "gripper_state": float(gripper_state),
            "object_pose": [float(x) for x in object_pose],
            "ee_pose": [float(x) for x in ee_pose],
            "action": [float(x) for x in action],
            "is_first": bool(is_first),
            "is_last": bool(is_last),
        }
        self.meta["steps"].append(step_data)

    def finalize_episode(self, success, exception_text=None):
        self.meta["success"] = bool(success)
        if exception_text is not None:
            self.meta["exception"] = str(exception_text)

        meta_path = self.episode_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

        if (not success) and (not self.keep_failed):
            shutil.rmtree(self.episode_dir, ignore_errors=True)

    def abort_episode(self):
        if self.episode_dir is not None and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)


class SyncSimRaccoonDataset:
    """
    Synchronous MuJoCo dataset collector for RaccoonBot.

    Key design choices:
    - No background simulation thread
    - No real-time sleep-based settling
    - Main loop only: command -> run N mj_step -> render/save
    - Safe with viewer=False (physics still advances)
    """

    MAX_SPEEDS = [2.2, 2.3, 2.3, 2.3]
    GRIPPER_SPEED = 15.0

    # Uploaded move_to code style uses centimeter-scale IK constants.
    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0

    MODE_POSITION = 0
    MODE_VELOCITY = 1

    GRIP_OPEN = 0.15701
    GRIP_CLOSE = -0.85

    GRIP_MODE_FREE = 0
    GRIP_MODE_HORZ = 1
    GRIP_MODE_VERT = 2
    GRIP_MODE_INTERP = 3

    CYLINDER_BODY_BY_COLOR = {
        "red": "target_object",
        "blue": "target_object_blue",
        "green": "target_object_green",
        "yellow": "target_object_yellow",
    }
    CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())

    # Workspace used when all four colored cylinders are visible at once.
    # Compared with the previous x=(-0.18, 0.18), y=(0.10, 0.18), this keeps
    # objects slightly farther forward and more centered left-to-right.
    DEFAULT_OBJECT_X_RANGE = (-0.10, 0.10)
    DEFAULT_OBJECT_Y_RANGE = (0.16, 0.195)
    DEFAULT_MIN_OBJECT_DISTANCE = 0.042

    DEEP_GRASP_Z = 0.016
    DEEP_PREGRASP_Z = 0.045

    def __init__(self, xml_path, image_size=(256, 256), camera_name=None, use_viewer=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xml 파일을 찾을 수 없습니다: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        self.camera_name = camera_name
        self.use_viewer = use_viewer

        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.target_angles = [0.0] * 4
        self.current_setpoints = [0.0] * 5
        self.joint_velocities = [0.0] * 4
        self.joint_control_mode = [self.MODE_POSITION] * 4
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE
        self.gripper_pitch_alpha = 0.0
        self.active_object_body_name = self.CYLINDER_BODY_BY_COLOR["red"]

        for i in range(4):
            self.joint_velocities[i] = self.MAX_SPEEDS[i] * 0.7

        # Initialize all colored cylinders in the scene. Dataset collection will
        # randomize these positions for every episode.
        self.reset_episode(
            object_specs=self.make_default_object_specs(),
            target_color="red",
        )

    # ---------- kinematics / commands ----------

    def _calc_inv_kinematics(self, x, y, z):
        """
        Inputs are in centimeters, matching the uploaded move_to code style.
        Returns [j1, j2, j3, j4] in degrees.
        """
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float)):
            if (-28.0 <= x <= 28.0) and (-15 <= y <= 28.0) and (0 <= z <= 36.25):
                x, y = y, -x
                th1 = math.atan2(y, x)
                c1 = math.cos(th1)
                s1 = math.sin(th1)
                x = x - self.L4 * c1
                y = y - self.L4 * s1
                zL1 = z - self.L1
                c3 = (x * x + y * y + zL1 * zL1 - self.L2 * self.L2 - self.L3 * self.L3) / (2 * self.L2 * self.L3)
                c32 = c3 * c3
                if c32 > 1:
                    c32 = 1
                s3 = -math.sqrt(1 - c32)
                th3 = math.atan2(s3, c3)
                M1 = c3 * self.L3 + self.L2
                M2 = z - self.L1
                M3 = s3 * self.L3
                M4 = c1 * x + s1 * y
                c2 = M1 * M2 - M3 * M4
                s2 = -M2 * M3 - M1 * M4
                th2 = math.atan2(s2, c2)
                th1 = math.degrees(th1)
                th2 = math.degrees(th2)
                th3 = math.degrees(th3)
                th4 = -(th2 + th3) - 90

                if th1 < -120 or th1 > 120:
                    return None
                if th2 < -90 or th2 > 30:
                    return None
                if th3 < -150 or th3 > 0:
                    return None

                return [th1, th2, th3, th4]
            return None
        return None

    def degree_to(self, joints, degrees, speed=70):
        j_list = joints if isinstance(joints, (list, tuple)) else [joints]
        d_list = degrees if isinstance(degrees, (list, tuple)) else [degrees]

        if len(d_list) == 1 and len(j_list) > 1:
            d_list = d_list * len(j_list)

        for j, deg in zip(j_list, d_list):
            idx = j - 1
            if 0 <= idx < 4:
                self.joint_control_mode[idx] = self.MODE_POSITION
                self.target_angles[idx] = np.radians(deg)
                percent = np.clip(speed, 0.0, 100.0)
                self.joint_velocities[idx] = (percent / 100.0) * self.MAX_SPEEDS[idx]

    def move_to(self, x_cm, y_cm, z_cm, speed=70):
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm)
        if angles is None:
            raise ValueError(f"도달할 수 없는 좌표입니다: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self.degree_to([1, 2, 3, 4], angles[:4], speed)

    def open_gripper(self):
        self.gripper_target = self.GRIP_OPEN

    def close_gripper(self):
        self.gripper_target = self.GRIP_CLOSE

    def set_gripper_pitch_alpha(self, alpha):
        self.gripper_pitch_alpha = float(np.clip(alpha, 0.0, 1.0))
        self.gripper_mode = self.GRIP_MODE_INTERP

    def lockh(self):
        self.gripper_pitch_alpha = 0.0
        self.gripper_mode = self.GRIP_MODE_HORZ

    def lockv(self):
        self.gripper_pitch_alpha = 1.0
        self.gripper_mode = self.GRIP_MODE_VERT

    def unlock(self):
        if self.gripper_mode != self.GRIP_MODE_FREE:
            self.target_angles[3] = self.data.qpos[3]
            self.gripper_mode = self.GRIP_MODE_FREE

    def execute_action(self, action, speed=70):
        """
        action = [target_x_m, target_y_m, target_z_m, gripper]
        v12 pitch-aware action = [target_x_m, target_y_m, target_z_m, gripper, pitch_alpha]
        where pitch_alpha=0.0 is horizontal and pitch_alpha=1.0 is vertical.
        """
        target_x, target_y, target_z, gripper = action[:4]
        if len(action) >= 5:
            self.set_gripper_pitch_alpha(float(action[4]))

        # move_to convention is centimeters.
        self.move_to(target_x * 100.0, target_y * 100.0, target_z * 100.0, speed=speed)

        if gripper >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    def is_action_reachable(self, action):
        if len(action) < 3:
            return False
        target_x, target_y, target_z = action[:3]
        return self._calc_inv_kinematics(
            float(target_x) * 100.0,
            float(target_y) * 100.0,
            float(target_z) * 100.0,
        ) is not None

    def first_unreachable_action(self, plan):
        for idx, action in enumerate(plan):
            if not self.is_action_reachable(action):
                return idx, action
        return None, None

    def make_reachable_direct_lift_actions(
        self,
        base_close,
        delta_z_pairs=(
            (0.020, 0.040),
            (0.020, 0.035),
            (0.018, 0.032),
            (0.015, 0.030),
            (0.012, 0.024),
            (0.010, 0.020),
        ),
    ):
        for mid_delta_z, high_delta_z in delta_z_pairs:
            lift_mid = [base_close[0], base_close[1], float(base_close[2]) + float(mid_delta_z), 1.0]
            lift_high = [base_close[0], base_close[1], float(base_close[2]) + float(high_delta_z), 1.0]
            if self.is_action_reachable(lift_mid) and self.is_action_reachable(lift_high):
                return lift_mid, lift_high, float(mid_delta_z), float(high_delta_z)
        return None, None, None, None

    # ---------- synchronous stepping ----------

    def _apply_controls_once(self):
        dt = self.model.opt.timestep

        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])
                if self.gripper_mode == self.GRIP_MODE_HORZ:
                    pitch_alpha = 0.0
                elif self.gripper_mode == self.GRIP_MODE_VERT:
                    pitch_alpha = 1.0
                else:
                    pitch_alpha = float(np.clip(self.gripper_pitch_alpha, 0.0, 1.0))
                desired = base_angle - np.radians(90.0 + 90.0 * pitch_alpha)

                error = desired - self.current_setpoints[i]
                speed_rad_s = self.MAX_SPEEDS[i]
                limit_step = speed_rad_s * dt
                step = np.clip(error, -limit_step, limit_step)
                self.current_setpoints[i] += step
            else:
                if self.joint_control_mode[i] == self.MODE_VELOCITY:
                    self.current_setpoints[i] += self.joint_velocities[i] * dt
                else:
                    error = self.target_angles[i] - self.current_setpoints[i]
                    if abs(error) > 1e-4:
                        max_step = abs(self.joint_velocities[i]) * dt
                        step_val = np.clip(error, -max_step, max_step)
                        self.current_setpoints[i] += step_val

            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        # Gripper stop-on-contact logic from uploaded code.
        try:
            touch_L = self.data.sensor("sensor_L").data[0]
            touch_R = self.data.sensor("sensor_R").data[0]
            is_touched = (touch_L > 0.1) and (touch_R > 0.1)
        except Exception:
            is_touched = False

        if self.gripper_target == self.GRIP_CLOSE and is_touched:
            self.gripper_target = self.data.qpos[4] - 0.028

        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_step = self.GRIPPER_SPEED * dt
            g_move = np.clip(g_err, -g_step, g_step)
            self.current_setpoints[4] += g_move

        self.data.ctrl[4] = self.current_setpoints[4]

    def step_n(self, n_steps):
        for _ in range(int(n_steps)):
            self._apply_controls_once()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def steps_for_seconds(self, seconds):
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def settle_steps(self, seconds=2.0):
        self.step_n(self.steps_for_seconds(seconds))

    # ---------- rendering / state ----------

    def get_robot_state(self):
        joint_angles = [float(self.data.qpos[i]) for i in range(4)]
        gripper_state = float(self.data.qpos[4])
        return {
            "joint_angles": joint_angles,
            "gripper_state": gripper_state
        }

    def get_object_pose(self, body_name="target_object"):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])

        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def get_ee_pose(self):
        """
        Return the command-space end-effector pose in meters.

        This matches the FK convention used by move_to()/IK and by the rollout
        environment. The MuJoCo Link4 body is offset from the actual command
        point, so using Link4.xpos here makes action labels and close-alignment
        metrics disagree with execution.
        """
        th1 = float(self.data.qpos[0])
        th2 = float(self.data.qpos[1])
        th3 = float(self.data.qpos[2])

        r = -self.L2 * math.sin(th2) - self.L3 * math.sin(th2 + th3)
        z = self.L1 + self.L2 * math.cos(th2) + self.L3 * math.cos(th2 + th3)
        r_tip = r + self.L4

        x_cm = -math.sin(th1) * r_tip
        y_cm = math.cos(th1) * r_tip
        z_cm = z

        return [x_cm / 100.0, y_cm / 100.0, z_cm / 100.0]

    def render_rgb(self):
        cam_id = self.camera_name if self.camera_name is not None else -1
        self.renderer.update_scene(self.data, camera=cam_id)
        image = self.renderer.render()
        return image.copy()

    def get_observation(self, object_body_name=None):
        if object_body_name is None:
            object_body_name = self.active_object_body_name

        rs = self.get_robot_state()
        obj = self.get_object_pose(object_body_name)
        img = self.render_rgb()
        ee_pose_list = self.get_ee_pose()

        return {
            "image": img,
            "joint_angles": rs["joint_angles"],
            "gripper_state": rs["gripper_state"],
            "object_pose": obj,
            "ee_pose": ee_pose_list,
        }

    # ---------- reset / success ----------

    def reset_object_pose(self, body_name="target_object", x=0.15, y=0.15, z=0.02, yaw=0.0):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        jnt_adr = self.model.body_jntadr[body_id]
        jnt_num = self.model.body_jntnum[body_id]
        if jnt_num < 1:
            raise ValueError(f"{body_name} has no joint")

        joint_id = jnt_adr
        qpos_adr = self.model.jnt_qposadr[joint_id]

        # freejoint qpos = [x, y, z, qw, qx, qy, qz]
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

        # Zero object joint velocities if present.
        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    @classmethod
    def make_default_object_specs(cls):
        """
        Deterministic fallback placement for initialization only.
        Dataset collection uses sample_object_specs() for randomized positions.
        """
        x_values = np.linspace(
            cls.DEFAULT_OBJECT_X_RANGE[0] * 0.75,
            cls.DEFAULT_OBJECT_X_RANGE[1] * 0.75,
            len(cls.CYLINDER_COLORS),
        )
        y_center = float(sum(cls.DEFAULT_OBJECT_Y_RANGE) / 2.0)
        return {
            color: {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                "x": float(x_values[idx]),
                "y": y_center,
                "yaw": 0.0,
            }
            for idx, color in enumerate(cls.CYLINDER_COLORS)
        }

    @classmethod
    def sample_object_specs(
        cls,
        rng,
        colors=None,
        x_range=None,
        y_range=None,
        yaw_range=(-np.pi / 4, np.pi / 4),
        min_distance=None,
        max_tries=1000,
    ):
        """
        Randomly place all colored cylinders in the visible workspace.

        Defaults intentionally narrow the spawn area compared with the older
        single-object collector:
          - x: -0.18~0.18  ->  -0.10~0.10
          - y:  0.10~0.18  ->   0.16~0.20
        A minimum XY distance prevents blocks from overlapping or touching.
        """
        colors = tuple(colors or cls.CYLINDER_COLORS)
        x_range = x_range or cls.DEFAULT_OBJECT_X_RANGE
        y_range = y_range or cls.DEFAULT_OBJECT_Y_RANGE
        min_distance = cls.DEFAULT_MIN_OBJECT_DISTANCE if min_distance is None else min_distance

        if len(colors) == 0:
            raise ValueError("colors는 비어 있을 수 없습니다.")
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

        specs = {}
        placed_xy = []
        # Shuffle placement order so one color is not systematically favored.
        placement_order = list(colors)
        rng.shuffle(placement_order)

        for color in placement_order:
            if color not in cls.CYLINDER_BODY_BY_COLOR:
                raise ValueError(f"지원하지 않는 색상입니다: {color}")

            for _ in range(max_tries):
                x = float(rng.uniform(x_range[0], x_range[1]))
                y = float(rng.uniform(y_range[0], y_range[1]))
                xy = np.array([x, y], dtype=np.float64)

                if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                    specs[color] = {
                        "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                        "x": x,
                        "y": y,
                        "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                    }
                    placed_xy.append(xy)
                    break
            else:
                raise RuntimeError(
                    "색상 cylinder 4개를 겹치지 않게 배치하지 못했습니다. "
                    f"x_range={x_range}, y_range={y_range}, min_distance={min_distance}를 확인하세요."
                )

        # Return in canonical color order for stable metadata.
        return {color: specs[color] for color in colors}

    @staticmethod
    def specs_to_meta(object_specs):
        return {
            color: {
                "body_name": str(spec["body_name"]),
                "xy": [float(spec["x"]), float(spec["y"])],
                "yaw": float(spec["yaw"]),
            }
            for color, spec in object_specs.items()
        }

    def reset_colored_objects(self, object_specs, target_color):
        """
        Place every colored cylinder in the scene. The target color controls
        which body is used for object_pose logging and grasp trajectory target.
        """
        if target_color not in object_specs:
            raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

        self.active_object_body_name = object_specs[target_color]["body_name"]

        for color, spec in object_specs.items():
            body_name = spec["body_name"]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"body not found for color '{color}': {body_name}")

            self.reset_object_pose(
                body_name,
                x=spec["x"],
                y=spec["y"],
                z=0.02,
                yaw=spec["yaw"],
            )

    def reset_episode(self, object_specs, target_color="red"):
        home = np.radians([0.0, -10.0, -140.0, 60.0])

        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION

        self.data.qvel[:] = 0.0

        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE
        self.gripper_pitch_alpha = 0.0

        self.reset_colored_objects(object_specs=object_specs, target_color=target_color)
        mujoco.mj_forward(self.model, self.data)

        # Short stabilization after reset.
        self.step_n(20)

    def get_gripper_touch_state(self):
        """
        Return whether the left/right gripper touch sensors are in contact.
        If the XML does not expose these sensors, this returns False for both sides.
        """
        try:
            touch_l = float(self.data.sensor("sensor_L").data[0])
            touch_r = float(self.data.sensor("sensor_R").data[0])
        except Exception:
            touch_l = 0.0
            touch_r = 0.0

        return touch_l, touch_r

    def is_grasp_success(self, touch_threshold=0.1, require_closed=True):
        """
        Grasp-only success criterion.
        The episode is considered successful when both gripper touch sensors detect contact.
        Optionally also require the gripper to have moved away from its fully-open position.
        """
        touch_l, touch_r = self.get_gripper_touch_state()
        both_touched = (touch_l > touch_threshold) and (touch_r > touch_threshold)

        if not require_closed:
            return bool(both_touched)

        # Make sure this is not just an accidental touch while the gripper is still fully open.
        gripper_is_closing_or_closed = float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01)
        return bool(both_touched and gripper_is_closing_or_closed)

    def is_body_touching_robot(self, body_name, ignored_geom_names=("floor",)):
        """
        Return True when the requested object body is in contact with a non-floor,
        non-cylinder body. This makes success target-specific when all four
        colored cylinders are present: touching the wrong color does not count.
        """
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if target_body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        cylinder_body_ids = set()
        for cylinder_body_name in self.CYLINDER_BODY_BY_COLOR.values():
            cylinder_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cylinder_body_name)
            if cylinder_body_id != -1:
                cylinder_body_ids.add(cylinder_body_id)

        ignored_geom_names = set(ignored_geom_names or [])

        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])

            if target_body_id not in (body1, body2):
                continue

            other_geom = geom2 if body1 == target_body_id else geom1
            other_body = body2 if body1 == target_body_id else body1

            other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_geom_name in ignored_geom_names:
                continue

            # Do not count target-object contact with another colored cylinder
            # as a grasp. We only want contacts against the robot/gripper.
            if other_body in cylinder_body_ids:
                continue

            return True

        return False

    def is_target_grasp_success(self, target_body_name, touch_threshold=0.1, require_closed=True):
        """
        Success for the multi-cylinder scene. Both gripper touch sensors must be
        active, the gripper must be closing/closed, and the prompted target body
        must be the object contacting the robot.
        """
        return bool(
            self.is_grasp_success(touch_threshold=touch_threshold, require_closed=require_closed)
            and self.is_body_touching_robot(target_body_name)
        )

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            close_fn = getattr(self.renderer, "close", None)
            if close_fn is not None:
                close_fn()
            self.renderer = None

    # ---------- current lift plan ----------

    def make_grasp_plan(self, box_x, box_y):
        z_above = 0.10
        base_above = [box_x, box_y, z_above, 0]
        pre_grasp = [box_x, box_y, self.DEEP_PREGRASP_Z, 0]
        deep_descend = [box_x, box_y, self.DEEP_GRASP_Z, 0]
        deep_close = [box_x, box_y, self.DEEP_GRASP_Z, 1]
        lift_mid, lift_high, _, _ = self.make_reachable_direct_lift_actions(deep_close)
        if lift_high is None:
            raise ValueError(f"Unreachable immediate lift waypoints for target=({box_x:.4f}, {box_y:.4f})")

        return [
            base_above,
            base_above,     # Settle at safe z for XY alignment.
            pre_grasp,
            deep_descend,
            deep_descend,   # Hold open at deep z before closing.
            deep_close,     # First close command only.
            lift_mid,       # First closed-frame target should already be lift.
            lift_high,
            lift_high,      # Brief high hold.
        ]

    def make_pitch_grasp_plan(self, box_x, box_y, pitch_alpha=0.0):
        pitch_alpha = float(np.clip(pitch_alpha, 0.0, 1.0))

        def action(x, y, z, gripper, pitch):
            return [float(x), float(y), float(z), float(gripper), float(pitch)]

        if pitch_alpha >= 0.75:
            raise ValueError(
                "Full vertical cylinder grasp was abandoned after smoke tests: "
                "the round cylinder slips out even when the gripper is visually aligned. "
                "Use the V11 horizontal close/lift trajectory for final training."
            )

        z_above = 0.10
        command_x = float(box_x)
        command_y = float(box_y)
        pregrasp_z = self.DEEP_PREGRASP_Z
        grasp_z = self.DEEP_GRASP_Z

        base_above_h = action(command_x, command_y, z_above, 0, 0.0)
        base_above_oriented = action(command_x, command_y, z_above, 0, pitch_alpha)
        pre_grasp = action(command_x, command_y, pregrasp_z, 0, pitch_alpha)
        deep_descend = action(command_x, command_y, grasp_z, 0, pitch_alpha)
        deep_close = action(command_x, command_y, grasp_z, 1, pitch_alpha)
        lift_mid, lift_high, _, _ = self.make_reachable_direct_lift_actions(deep_close[:4])
        if lift_high is None:
            raise ValueError(f"Unreachable immediate lift waypoints for target=({box_x:.4f}, {box_y:.4f})")
        lift_mid = action(lift_mid[0], lift_mid[1], lift_mid[2], 1, pitch_alpha)
        lift_high = action(lift_high[0], lift_high[1], lift_high[2], 1, pitch_alpha)

        return [
            base_above_h,
            base_above_oriented,  # At safe height, give joint4 time to rotate before descent.
            pre_grasp,
            deep_descend,
            deep_descend,
            deep_close,
            lift_mid,
            lift_high,
            lift_high,
        ]

    @staticmethod
    def should_use_vertical_grasp(
        object_specs,
        target_color,
        radial_x_tolerance=0.018,
        blocking_y_margin=0.030,
    ):
        target = object_specs[target_color]
        target_x = float(target["x"])
        target_y = float(target["y"])
        blockers = []
        for color, spec in object_specs.items():
            if color == target_color:
                continue
            dx = abs(float(spec["x"]) - target_x)
            dy = target_y - float(spec["y"])
            if dx <= float(radial_x_tolerance) and dy >= float(blocking_y_margin):
                blockers.append(color)
        return bool(blockers), blockers

    @classmethod
    def sample_v12_radial_interference_specs(
        cls,
        rng,
        colors=None,
        x_range=None,
        y_range=None,
        min_pair_gap=0.044,
    ):
        colors = tuple(colors or cls.CYLINDER_COLORS)
        x_range = x_range or cls.DEFAULT_OBJECT_X_RANGE
        y_range = y_range or (0.145, cls.DEFAULT_OBJECT_Y_RANGE[1])
        if len(colors) < 2:
            raise ValueError("v12 radial interference scene requires at least two colors")

        pair = list(rng.choice(colors, size=2, replace=False))
        near_color, far_color = pair[0], pair[1]
        remaining = [color for color in colors if color not in pair]

        x_min, x_max = [float(v) for v in x_range]
        y_min, y_max = [float(v) for v in y_range]
        lane_x = float(rng.uniform(max(x_min + 0.020, -0.045), min(x_max - 0.020, 0.045)))
        near_y = float(rng.uniform(y_min, min(y_min + 0.012, y_max - min_pair_gap)))
        far_y = float(min(y_max, near_y + float(rng.uniform(min_pair_gap, min_pair_gap + 0.010))))

        specs = {
            near_color: {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[near_color],
                "x": lane_x + float(rng.uniform(-0.003, 0.003)),
                "y": near_y,
                "z": 0.02,
                "yaw": float(rng.uniform(-math.pi, math.pi)),
            },
            far_color: {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[far_color],
                "x": lane_x + float(rng.uniform(-0.003, 0.003)),
                "y": far_y,
                "z": 0.02,
                "yaw": float(rng.uniform(-math.pi, math.pi)),
            },
        }

        side_slots = [
            (max(x_min, lane_x - 0.075), float(rng.uniform(y_min + 0.010, y_max - 0.006))),
            (min(x_max, lane_x + 0.075), float(rng.uniform(y_min + 0.010, y_max - 0.006))),
        ]
        rng.shuffle(side_slots)
        for color, (x, y) in zip(remaining, side_slots):
            specs[color] = {
                "body_name": cls.CYLINDER_BODY_BY_COLOR[color],
                "x": float(x),
                "y": float(y),
                "z": 0.02,
                "yaw": float(rng.uniform(-math.pi, math.pi)),
            }

        return {color: specs[color] for color in colors}


def record_final_grasp_alignment_metrics(logger, target_x, target_y, base_close_pose):
    if logger.meta is None:
        return None

    logger.meta["base_close_pose"] = [float(x) for x in base_close_pose]
    logger.meta["desired_close_pose"] = [float(x) for x in base_close_pose]
    logger.meta["first_gripper_close_step"] = None
    logger.meta["close_ee_pose"] = None
    logger.meta["close_dx_to_target"] = None
    logger.meta["close_dy_to_target"] = None
    logger.meta["close_xy_distance_to_target"] = None
    logger.meta["close_ee_z"] = None

    for step in logger.meta.get("steps", []):
        action = step.get("action", [])
        ee_pose = step.get("ee_pose", [])
        if len(action) < 4 or len(ee_pose) < 3:
            continue
        if float(action[3]) < 0.5:
            continue

        close_ee_pose = [float(x) for x in ee_pose[:3]]
        close_dx = close_ee_pose[0] - float(target_x)
        close_dy = close_ee_pose[1] - float(target_y)
        logger.meta["first_gripper_close_step"] = int(step.get("t", 0))
        logger.meta["close_ee_pose"] = close_ee_pose
        logger.meta["close_dx_to_target"] = close_dx
        logger.meta["close_dy_to_target"] = close_dy
        logger.meta["close_xy_distance_to_target"] = float(math.hypot(close_dx, close_dy))
        logger.meta["close_ee_z"] = close_ee_pose[2]
        return step

    return None


def record_lift_success_metrics(logger, existing_success, close_step=None):
    if logger.meta is None:
        return False

    steps = logger.meta.get("steps", [])
    logger.meta["target_z_initial"] = None
    logger.meta["target_z_at_close"] = None
    logger.meta["target_z_final"] = None
    logger.meta["target_lift_delta"] = None
    logger.meta["strict_lift_success"] = False

    if not steps:
        return False

    initial_pose = steps[0].get("object_pose", [])
    final_pose = steps[-1].get("object_pose", [])
    if len(initial_pose) < 3 or len(final_pose) < 3:
        return False

    target_z_initial = float(initial_pose[2])
    target_z_final = float(final_pose[2])
    target_z_at_close = None
    if close_step is not None:
        close_pose = close_step.get("object_pose", [])
        if len(close_pose) >= 3:
            target_z_at_close = float(close_pose[2])

    target_lift_delta = target_z_final - target_z_initial
    strict_lift_success = bool(existing_success and target_lift_delta >= 0.010)

    logger.meta["target_z_initial"] = target_z_initial
    logger.meta["target_z_at_close"] = target_z_at_close
    logger.meta["target_z_final"] = target_z_final
    logger.meta["target_lift_delta"] = target_lift_delta
    logger.meta["strict_lift_success"] = strict_lift_success
    return strict_lift_success


def run_episode_and_record(
    rc: SyncSimRaccoonDataset,
    logger: DatasetLogger,
    episode_id: int,
    instruction: str,
    object_specs: dict,
    target_color: str = "red",
    speed: int = 70,
    settle_seconds_per_action: float = 2.0,
    initial_settle_seconds: float = 0.3,
    hz: int = 10,
    touch_threshold: float = 0.1,
    instruction_template=None,
    instruction_mode: str = "baseline",
    trajectory_mode: str = "final_align_lift_deep_immediate",
    require_lift_success: bool = False,
    max_close_ee_z: float = 0.025,
    max_close_xy_error: float = 0.006,
    scene_id=None,
):
    if max_close_ee_z is not None and float(max_close_ee_z) < 0.0:
        max_close_ee_z = None
    if max_close_xy_error is not None and float(max_close_xy_error) < 0.0:
        max_close_xy_error = None

    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

    target_spec = object_specs[target_color]
    target_body_name = target_spec["body_name"]
    target_x = float(target_spec["x"])
    target_y = float(target_spec["y"])
    target_yaw = float(target_spec["yaw"])

    rc.reset_episode(object_specs=object_specs, target_color=target_color)
    rc.lockh()

    # Let newly reset free-joint cylinders fall/settle before capturing frame_000000.
    # Without this, the first saved image can show cylinders slightly floating while
    # later frames look normal after one physics step.
    if initial_settle_seconds > 0:
        rc.settle_steps(seconds=initial_settle_seconds)

    logger.start_episode(
        episode_id=episode_id,
        instruction=instruction,
        instruction_template=instruction_template,
        instruction_mode=instruction_mode,
        task_type="grasp",
        goal_xy=[target_x, target_y],
        box_init_xy=[target_x, target_y],
        box_init_yaw=target_yaw,
        target_color=target_color,
        target_body_name=target_body_name,
        all_object_init_poses=SyncSimRaccoonDataset.specs_to_meta(object_specs),
    )
    if trajectory_mode not in ("final_align_lift_deep_immediate", "v12_pitch_adaptive"):
        raise ValueError(
            "Supported trajectory_mode values: 'final_align_lift_deep_immediate', 'v12_pitch_adaptive'."
        )

    logger.meta["trajectory_mode"] = trajectory_mode
    logger.meta["trajectory_mode_effective"] = trajectory_mode
    logger.meta["scene_id"] = int(scene_id) if scene_id is not None else None
    logger.meta["lift_target_z"] = None
    logger.meta["lift_delta_z"] = None
    logger.meta["post_close_hold_count"] = 0
    logger.meta["waypoint_steps"] = None
    logger.meta["lift_mid_target_z"] = None
    logger.meta["lift_high_target_z"] = None
    logger.meta["lift_mid_delta_z"] = None
    logger.meta["lift_high_delta_z"] = None
    logger.meta["lift_ramp_mode"] = "deep_immediate_lift_after_close"
    logger.meta["grasp_target_z"] = None
    logger.meta["pregrasp_target_z"] = None
    logger.meta["pitch_action_enabled"] = bool(trajectory_mode == "v12_pitch_adaptive")
    logger.meta["pitch_alpha_target"] = 0.0
    logger.meta["vertical_grasp_required"] = False
    logger.meta["vertical_grasp_blockers"] = []

    try:
        # The prompt decides which cylinder to grasp. All four cylinders are
        # visible, but the trajectory is aimed only at the prompted color.
        if trajectory_mode == "v12_pitch_adaptive":
            vertical_required, vertical_blockers = rc.should_use_vertical_grasp(
                object_specs=object_specs,
                target_color=target_color,
            )
            pitch_alpha = 1.0 if vertical_required else 0.0
            plan = rc.make_pitch_grasp_plan(target_x, target_y, pitch_alpha=pitch_alpha)
            logger.meta["pitch_alpha_target"] = float(pitch_alpha)
            logger.meta["vertical_grasp_required"] = bool(vertical_required)
            logger.meta["vertical_grasp_blockers"] = [str(color) for color in vertical_blockers]
        else:
            plan = rc.make_grasp_plan(target_x, target_y)
        close_action = next(action for action in plan if float(action[3]) >= 0.5)
        lift_actions = [
            action
            for action in plan
            if float(action[3]) >= 0.5 and float(action[2]) > float(close_action[2])
        ]
        logger.meta["grasp_target_z"] = float(close_action[2])
        logger.meta["pregrasp_target_z"] = float(rc.DEEP_PREGRASP_Z)
        if lift_actions:
            logger.meta["lift_mid_target_z"] = float(lift_actions[0][2])
            logger.meta["lift_high_target_z"] = float(lift_actions[-1][2])
            logger.meta["lift_mid_delta_z"] = float(lift_actions[0][2] - close_action[2])
            logger.meta["lift_high_delta_z"] = float(lift_actions[-1][2] - close_action[2])
            logger.meta["lift_delta_z"] = float(lift_actions[-1][2] - close_action[2])
            logger.meta["lift_target_z"] = float(lift_actions[-1][2])

        if trajectory_mode == "v12_pitch_adaptive":
            if logger.meta["vertical_grasp_required"]:
                waypoint_steps = [
                    6,   # Move to vertical-grasp XY at safe z while horizontal.
                    10,  # Rotate joint4 to vertical at safe z.
                    4,   # Extra orientation settle before any descent.
                    6,   # Descend to vertical pre-grasp z.
                    12,  # Final z descent after pitch is already settled.
                    4,   # Open settle at vertical close z.
                    1,   # One close-command frame only.
                    8,   # Immediate lift mid after first close.
                    8,   # Lift high.
                    2,   # Brief high hold.
                ]
            else:
                waypoint_steps = [
                    6,   # Safe-z approach with horizontal gripper.
                    3,   # Brief orientation settle.
                    4,   # Pre-grasp transition.
                    12,  # Deep descent.
                    4,   # Open settle at deep z before closing.
                    1,   # One close-command frame only.
                    8,   # Immediate lift mid after first close.
                    8,   # Lift high.
                    2,   # Brief high hold.
                ]
        else:
            waypoint_steps = [
                6,   # Safe-z approach.
                3,   # Brief safe-z alignment settle.
                4,   # Pre-grasp transition.
                12,  # Deep descent.
                4,   # Open settle at deep z before closing.
                1,   # One close-command frame only.
                8,   # Immediate lift mid after first close.
                8,   # Lift high.
                2,   # Brief high hold.
            ]
        logger.meta["waypoint_steps"] = waypoint_steps

        # Initial observation.
        obs = rc.get_observation()
        dt = 1.0 / hz
        step_counter = 0

        for action_idx, action in enumerate(plan):
            # Set control target to current waypoint.
            rc.execute_action(action, speed=speed)

            # Capture continuous observations at specified Hz while moving toward the target.
            num_frames = int(waypoint_steps[action_idx])

            for _ in range(num_frames):
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

                # Advance physics by dt seconds.
                rc.settle_steps(seconds=dt)

                # Observe after stepping.
                obs = rc.get_observation()
                step_counter += 1

        # Record terminal observation.
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

        contact_success = rc.is_target_grasp_success(
            target_body_name=target_body_name,
            touch_threshold=touch_threshold,
        )
        effective_close_pose = next(
            [float(x) for x in action[:3]]
            for action in plan
            if float(action[3]) >= 0.5
        )
        close_step = record_final_grasp_alignment_metrics(
            logger=logger,
            target_x=target_x,
            target_y=target_y,
            base_close_pose=effective_close_pose,
        )
        strict_lift_success = record_lift_success_metrics(
            logger=logger,
            existing_success=contact_success,
            close_step=close_step,
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

        success = strict_lift_success if require_lift_success else contact_success
        if require_lift_success:
            success = bool(success and close_quality_success)
        logger.finalize_episode(success=success)
        return success

    except Exception as e:
        logger.abort_episode()
        raise e


def _balanced_target_counts(num_episodes, colors):
    """
    Return per-color episode targets. If num_episodes is divisible by the
    number of colors, the split is exactly equal. Otherwise the remainder is
    distributed one-by-one to the first colors.
    """
    base = num_episodes // len(colors)
    remainder = num_episodes % len(colors)
    return {
        color: base + (1 if idx < remainder else 0)
        for idx, color in enumerate(colors)
    }


def _parse_episode_id_from_dir(episode_dir):
    prefix = "episode_"
    name = Path(episode_dir).name
    if not name.startswith(prefix):
        return None

    suffix = name[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _target_counts_reached(target_counts, success_counts):
    return all(success_counts.get(color, 0) >= target for color, target in target_counts.items())


def _next_unused_episode_id(dataset_root, start_episode_id):
    dataset_root = Path(dataset_root)
    episode_id = max(1, int(start_episode_id))
    while (dataset_root / f"episode_{episode_id:06d}").exists():
        episode_id += 1
    return episode_id


def _cleanup_incomplete_episode_dirs(dataset_root):
    dataset_root = Path(dataset_root)
    removed_episode_dirs = []
    if not dataset_root.exists():
        return removed_episode_dirs

    for episode_dir in sorted(dataset_root.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        if (episode_dir / "meta.json").exists():
            continue
        shutil.rmtree(episode_dir, ignore_errors=True)
        removed_episode_dirs.append(episode_dir.name)
    return removed_episode_dirs


def _compact_episode_numbering(dataset_root):
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return []

    episode_entries = []
    for episode_dir in sorted(dataset_root.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        episode_id = _parse_episode_id_from_dir(episode_dir)
        if episode_id is None:
            continue
        episode_entries.append((episode_id, episode_dir))

    moves = []
    for new_episode_id, (old_episode_id, episode_dir) in enumerate(sorted(episode_entries), start=1):
        if old_episode_id != new_episode_id:
            moves.append((episode_dir, old_episode_id, new_episode_id))

    if not moves:
        return []

    temp_moves = []
    for move_idx, (episode_dir, old_episode_id, new_episode_id) in enumerate(moves, start=1):
        tmp_dir = dataset_root / f".renumber_tmp_{old_episode_id:06d}_{move_idx:06d}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        episode_dir.rename(tmp_dir)
        temp_moves.append((tmp_dir, old_episode_id, new_episode_id))

    compacted = []
    for tmp_dir, old_episode_id, new_episode_id in temp_moves:
        new_dir = dataset_root / f"episode_{new_episode_id:06d}"
        if new_dir.exists():
            raise FileExistsError(f"Cannot compact episode numbering; destination already exists: {new_dir}")
        tmp_dir.rename(new_dir)

        meta_path = new_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["episode_id"] = int(new_episode_id)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        compacted.append((old_episode_id, new_episode_id))

    return compacted


def _scan_resume_state(dataset_root, colors):
    dataset_root = Path(dataset_root)
    colors = tuple(colors)
    success_counts = {color: 0 for color in colors}
    max_episode_id = 0
    max_scene_id = 0
    scanned_meta_count = 0

    if not dataset_root.exists():
        return success_counts, max_episode_id, max_scene_id, scanned_meta_count

    for episode_dir in sorted(dataset_root.glob("episode_*")):
        if not episode_dir.is_dir():
            continue

        dir_episode_id = _parse_episode_id_from_dir(episode_dir)
        if dir_episode_id is not None:
            max_episode_id = max(max_episode_id, dir_episode_id)

        meta_path = episode_dir / "meta.json"
        if not meta_path.exists():
            continue

        scanned_meta_count += 1
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            print(f"[WARN] resume scan skipped unreadable meta: {meta_path} | {exc}")
            continue

        try:
            max_episode_id = max(max_episode_id, int(meta.get("episode_id", 0)))
        except (TypeError, ValueError):
            pass

        try:
            scene_id = meta.get("scene_id")
            if scene_id is not None:
                max_scene_id = max(max_scene_id, int(scene_id))
        except (TypeError, ValueError):
            pass

        if not bool(meta.get("success", False)):
            continue

        target_color = str(meta.get("target_color", ""))
        if target_color in success_counts:
            success_counts[target_color] += 1

    return success_counts, max_episode_id, max_scene_id, scanned_meta_count


def _sample_remaining_color(rng, target_counts, success_counts):
    remaining_colors = []
    remaining_weights = []

    for color, target_count in target_counts.items():
        remaining = target_count - success_counts[color]
        if remaining > 0:
            remaining_colors.append(color)
            remaining_weights.append(remaining)

    if not remaining_colors:
        return None

    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_weights /= remaining_weights.sum()
    return str(rng.choice(remaining_colors, p=remaining_weights))


def _remaining_colors(target_counts, success_counts):
    return [
        color
        for color, target_count in target_counts.items()
        if success_counts.get(color, 0) < target_count
    ]


def _select_instruction_template(
    rng,
    instruction_template,
    instruction_templates,
    instruction_templates_mode="language_extended",
):
    if instruction_templates is None:
        return instruction_template, "baseline"

    templates = tuple(instruction_templates)
    if len(templates) == 0:
        raise ValueError("instruction_templates는 비어 있을 수 없습니다.")
    return str(rng.choice(templates)), str(instruction_templates_mode)


def collect_dataset(
    xml_path="Raccoon_colored_cylinder.xml",
    dataset_root="raccoon_grasp_colored_cylinder",
    num_episodes=100,
    colors=("red", "blue", "green", "yellow"),
    instruction_template="grasp the {color} cylinder",
    instruction_templates=None,
    instruction_templates_mode="language_extended",
    keep_failed=False,
    use_viewer=False,
    camera_name="front_view",
    speed=150,
    settle_seconds_per_action=0.8,
    initial_settle_seconds=0.3,
    hz=10,
    touch_threshold=0.1,
    seed=None,
    max_attempts=None,
    object_x_range=SyncSimRaccoonDataset.DEFAULT_OBJECT_X_RANGE,
    object_y_range=SyncSimRaccoonDataset.DEFAULT_OBJECT_Y_RANGE,
    min_object_distance=SyncSimRaccoonDataset.DEFAULT_MIN_OBJECT_DISTANCE,
    trajectory_mode="final_align_lift_deep_immediate",
    require_lift_success=False,
    max_close_ee_z=0.025,
    max_close_xy_error=0.006,
    resume=False,
    scene_reuse_all_colors=False,
    scene_color_max_failures=3,
    restart_sim_every_attempts=0,
    restart_sim_after_fail_streak=0,
    v12_interference_scenes=False,
):
    """
    Collect a balanced grasp dataset for colored cylinders.

    Each episode contains all four colored cylinders at randomized positions.
    The instruction selects which colored cylinder is the target, and the robot
    executes the grasp plan toward that target color only.
    Passing instruction_templates enables language-extended mode and randomly
    selects one template per episode. Leaving it as None preserves the original
    single-template baseline behavior.

    Default behavior with keep_failed=False:
    - Saves exactly num_episodes successful episodes when possible.
    - Balances successful episodes across colors according to target_counts.
      For num_episodes=500 and 4 colors, this yields 125 episodes per color.
    - Failed episodes are discarded and retried with the remaining color quota.
    - Before frame_000000 is captured, the scene is stepped for
      initial_settle_seconds so free-joint cylinders are already resting on the table.

    Position defaults are constrained relative to the old single-object range:
    - old x range: -0.18~0.18  ->  new x range: -0.10~0.10
    - old y range:  0.10~0.18  ->  new y range:  0.16~0.20

    If keep_failed=True, failed episodes are also saved, so the final folder can
    contain more than num_episodes attempts and the all-attempt ratio may differ.
    """
    colors = tuple(colors)
    valid_colors = set(SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR.keys())
    unknown_colors = [color for color in colors if color not in valid_colors]
    if unknown_colors:
        raise ValueError(f"지원하지 않는 색상입니다: {unknown_colors}. 지원 색상: {sorted(valid_colors)}")

    if len(colors) == 0:
        raise ValueError("colors는 비어 있을 수 없습니다.")

    target_counts = _balanced_target_counts(num_episodes, colors)
    rng = np.random.default_rng(seed)

    success_counts = {color: 0 for color in colors}
    next_episode_id = 1
    resume_scene_count_start = 0
    if resume:
        removed_episode_dirs = _cleanup_incomplete_episode_dirs(dataset_root)
        if removed_episode_dirs:
            preview = ", ".join(removed_episode_dirs[:5])
            suffix = "" if len(removed_episode_dirs) <= 5 else f", ... (+{len(removed_episode_dirs) - 5})"
            print(f"[RESUME] removed incomplete episode dirs without meta.json: {preview}{suffix}")

        compacted_episode_ids = _compact_episode_numbering(dataset_root)
        if compacted_episode_ids:
            preview = ", ".join(
                f"{old_id:06d}->{new_id:06d}"
                for old_id, new_id in compacted_episode_ids[:8]
            )
            suffix = (
                ""
                if len(compacted_episode_ids) <= 8
                else f", ... (+{len(compacted_episode_ids) - 8})"
            )
            print(f"[RESUME] compacted episode numbering: {preview}{suffix}")

        success_counts, max_episode_id, max_scene_id, scanned_meta_count = _scan_resume_state(dataset_root, colors)
        next_episode_id = _next_unused_episode_id(dataset_root, max_episode_id + 1)
        resume_scene_count_start = max_scene_id
        if seed is not None and scanned_meta_count > 0:
            # A resumed run should not replay the exact same sampled scenes from
            # the start of the original seed.
            resume_seed = int(seed) + int(max_episode_id) * 1009 + int(max_scene_id) * 9176
            rng = np.random.default_rng(resume_seed)
            print(f"[RESUME] rng reseeded for continuation: {resume_seed}")
        print(f"[RESUME] scanned meta files: {scanned_meta_count}")
        print(f"[RESUME] existing successful counts: {success_counts}")
        print(f"[RESUME] max existing episode id: {max_episode_id:06d}")
        print(f"[RESUME] max existing scene id: {max_scene_id:06d}")
        print(f"[RESUME] next unused episode id: {next_episode_id:06d}")

        if _target_counts_reached(target_counts, success_counts):
            print(
                "[RESUME] Existing successful episodes already meet or exceed "
                f"the target counts {target_counts}. Nothing to generate."
            )
            return

    if max_attempts is None:
        # Prevent infinite loops if grasp repeatedly fails.
        max_attempts = max(num_episodes * 20, num_episodes + 100)

    rc = None

    def restart_sim(reason):
        nonlocal rc
        if rc is not None:
            rc.close()
        rc = SyncSimRaccoonDataset(
            xml_path=xml_path,
            image_size=(256, 256),
            camera_name=camera_name,
            use_viewer=use_viewer,
        )
        print(f"[SIM_RESTART] attempt={attempt_count:04d} | reason={reason}")

    logger = DatasetLogger(
        root_dir=dataset_root,
        keep_failed=keep_failed,
        overwrite_existing=not resume,
    )

    attempt_count = 0
    consecutive_failures = 0
    last_sim_restart_attempt = -1
    scene_count = resume_scene_count_start

    print(f"Target color counts: {target_counts}")
    if scene_reuse_all_colors:
        print("Scene reuse mode: each sampled object layout is attempted for all remaining colors.")
        if scene_color_max_failures and scene_color_max_failures > 0:
            print(
                "Scene rejection mode: abandon and rollback a scene after "
                f"{scene_color_max_failures} failures for the same color."
            )
    if restart_sim_every_attempts and restart_sim_every_attempts > 0:
        print(f"Sim restart mode: recreate MuJoCo every {restart_sim_every_attempts} attempts.")
    if restart_sim_after_fail_streak and restart_sim_after_fail_streak > 0:
        print(f"Sim restart mode: recreate MuJoCo after {restart_sim_after_fail_streak} consecutive failures.")

    restart_sim(reason="initial")

    try:
        while (not _target_counts_reached(target_counts, success_counts)) and attempt_count < max_attempts:
            if scene_reuse_all_colors:
                target_colors_for_scene = _remaining_colors(target_counts, success_counts)
                if not target_colors_for_scene:
                    break
                target_colors_for_scene = list(rng.permutation(target_colors_for_scene))
                scene_count += 1
            else:
                target_color = _sample_remaining_color(rng, target_counts, success_counts)
                if target_color is None:
                    break
                target_colors_for_scene = [target_color]
                scene_count += 1

            try:
                if v12_interference_scenes:
                    object_specs = SyncSimRaccoonDataset.sample_v12_radial_interference_specs(
                        rng=rng,
                        colors=colors,
                        x_range=object_x_range,
                        y_range=object_y_range,
                    )
                else:
                    object_specs = SyncSimRaccoonDataset.sample_object_specs(
                        rng=rng,
                        colors=colors,
                        x_range=object_x_range,
                        y_range=object_y_range,
                        min_distance=min_object_distance,
                    )
            except RuntimeError as exc:
                print(f"[SCENE_SAMPLE_RETRY] scene_id={scene_count:06d} | reason={exc}")
                continue
            current_scene_successes = []
            abandon_scene = False

            def rollback_current_scene(reason):
                nonlocal next_episode_id
                if not current_scene_successes:
                    print(f"[SCENE_REJECT] scene_id={scene_count:06d} | reason={reason} | no saved successes")
                    return

                removed = []
                removed_episode_ids = []
                for episode_dir, color in reversed(current_scene_successes):
                    shutil.rmtree(episode_dir, ignore_errors=True)
                    if success_counts.get(color, 0) > 0:
                        success_counts[color] -= 1
                    removed.append(episode_dir.name)
                    parsed_episode_id = _parse_episode_id_from_dir(episode_dir)
                    if parsed_episode_id is not None:
                        removed_episode_ids.append(parsed_episode_id)

                if resume:
                    if removed_episode_ids:
                        next_episode_id = _next_unused_episode_id(
                            logger.root_dir,
                            min(removed_episode_ids),
                        )
                    else:
                        next_episode_id = _next_unused_episode_id(logger.root_dir, next_episode_id)
                current_scene_successes.clear()

                print(
                    f"[SCENE_REJECT] scene_id={scene_count:06d} | reason={reason} | "
                    f"removed_success_episodes={list(reversed(removed))} | "
                    f"next_episode_id={next_episode_id:06d} | success_counts={success_counts}"
                )

            for target_color in target_colors_for_scene:
                if abandon_scene:
                    break
                if attempt_count >= max_attempts or _target_counts_reached(target_counts, success_counts):
                    break
                if success_counts[target_color] >= target_counts[target_color]:
                    continue

                color_done_for_scene = False
                color_failures_for_scene = 0
                while (
                    not color_done_for_scene
                    and not abandon_scene
                    and attempt_count < max_attempts
                    and not _target_counts_reached(target_counts, success_counts)
                    and success_counts[target_color] < target_counts[target_color]
                ):
                    attempt_count += 1

                    selected_template, instruction_mode = _select_instruction_template(
                        rng=rng,
                        instruction_template=instruction_template,
                        instruction_templates=instruction_templates,
                        instruction_templates_mode=instruction_templates_mode,
                    )
                    instruction = selected_template.format(color=target_color)

                    # With keep_failed=False, failed attempts are deleted, so reusing the
                    # next successful episode id keeps folder numbering compact.
                    if resume:
                        episode_id = _next_unused_episode_id(logger.root_dir, next_episode_id)
                    elif keep_failed:
                        episode_id = attempt_count
                    else:
                        episode_id = sum(success_counts.values()) + 1

                    success = False
                    try:
                        success = run_episode_and_record(
                            rc=rc,
                            logger=logger,
                            episode_id=episode_id,
                            instruction=instruction,
                            instruction_template=selected_template,
                            instruction_mode=instruction_mode,
                            object_specs=object_specs,
                            target_color=target_color,
                            speed=speed,
                            settle_seconds_per_action=settle_seconds_per_action,
                            initial_settle_seconds=initial_settle_seconds,
                            hz=hz,
                            touch_threshold=touch_threshold,
                            trajectory_mode=trajectory_mode,
                            require_lift_success=require_lift_success,
                            max_close_ee_z=max_close_ee_z,
                            max_close_xy_error=max_close_xy_error,
                            scene_id=scene_count,
                        )

                        if success:
                            success_counts[target_color] += 1
                            consecutive_failures = 0
                            color_done_for_scene = True
                            current_scene_successes.append(
                                (logger.root_dir / f"episode_{episode_id:06d}", target_color)
                            )
                        else:
                            consecutive_failures += 1
                            color_failures_for_scene += 1
                            color_done_for_scene = not scene_reuse_all_colors
                        if resume:
                            if success or keep_failed:
                                next_episode_id = episode_id + 1
                            else:
                                next_episode_id = episode_id

                        print(
                            f"[Attempt {attempt_count:04d}] scene_id={scene_count:06d} | "
                            f"episode_id={episode_id:06d} | task_type='grasp' | "
                            f"color='{target_color}' | "
                            f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f}) | "
                            f"instruction_mode='{instruction_mode}' | instruction='{instruction}' | success={success} | "
                            f"success_counts={success_counts}"
                        )
                    except Exception as e:
                        consecutive_failures += 1
                        color_failures_for_scene += 1
                        color_done_for_scene = not scene_reuse_all_colors
                        if resume:
                            next_episode_id = _next_unused_episode_id(logger.root_dir, episode_id)
                        print(
                            f"[Attempt {attempt_count:04d}] scene_id={scene_count:06d} | "
                            f"task_type='grasp' | color='{target_color}' | exception: {e}"
                        )

                    if (
                        scene_reuse_all_colors
                        and scene_color_max_failures
                        and scene_color_max_failures > 0
                        and color_failures_for_scene >= scene_color_max_failures
                    ):
                        rollback_current_scene(
                            reason=(
                                f"color_{target_color}_failed_"
                                f"{color_failures_for_scene}_times"
                            )
                        )
                        abandon_scene = True
                        color_done_for_scene = True

                    restart_reasons = []
                    if (
                        restart_sim_every_attempts
                        and restart_sim_every_attempts > 0
                        and attempt_count > 0
                        and attempt_count % restart_sim_every_attempts == 0
                    ):
                        restart_reasons.append(f"periodic_{restart_sim_every_attempts}_attempts")
                    if (
                        restart_sim_after_fail_streak
                        and restart_sim_after_fail_streak > 0
                        and consecutive_failures >= restart_sim_after_fail_streak
                    ):
                        restart_reasons.append(f"fail_streak_{consecutive_failures}")
                    if (
                        restart_reasons
                        and attempt_count != last_sim_restart_attempt
                        and attempt_count < max_attempts
                        and not _target_counts_reached(target_counts, success_counts)
                    ):
                        restart_sim(reason="+".join(restart_reasons))
                        last_sim_restart_attempt = attempt_count
                        consecutive_failures = 0

    finally:
        if rc is not None:
            rc.close()

    total_success = sum(success_counts.values())
    print(f"완료: success episodes = {total_success}/{num_episodes}, attempts = {attempt_count}")
    print(f"샘플링 scene 수: {scene_count}")
    print(f"색상별 성공 episode 수: {success_counts}")

    if not _target_counts_reached(target_counts, success_counts):
        print(
            "주의: max_attempts에 도달해서 목표 episode 수를 모두 채우지 못했습니다. "
            "max_attempts를 늘리거나 grasp 성공 조건/동작 파라미터를 확인하세요."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Collect RaccoonBot colored-cylinder grasp demonstrations.")
    parser.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--dataset_root", type=str, default="raccoon_grasp_colored_cylinder")
    parser.add_argument("--num_episodes", type=int, default=400)
    parser.add_argument(
        "--instruction_mode",
        choices=("baseline", "language_extended", "lift_extended"),
        default="baseline",
    )
    parser.add_argument("--instruction_template", type=str, default="grasp the {color} cylinder")
    parser.add_argument("--instruction_templates", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep_failed", action="store_true", help="Keep failed episode directories for debugging.")
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--speed", type=int, default=150)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.8)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.1)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--touch_threshold", type=float, default=0.1)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--object_x_min", type=float, default=SyncSimRaccoonDataset.DEFAULT_OBJECT_X_RANGE[0])
    parser.add_argument("--object_x_max", type=float, default=SyncSimRaccoonDataset.DEFAULT_OBJECT_X_RANGE[1])
    parser.add_argument("--object_y_min", type=float, default=SyncSimRaccoonDataset.DEFAULT_OBJECT_Y_RANGE[0])
    parser.add_argument("--object_y_max", type=float, default=SyncSimRaccoonDataset.DEFAULT_OBJECT_Y_RANGE[1])
    parser.add_argument("--min_object_distance", type=float, default=SyncSimRaccoonDataset.DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument(
        "--max_close_ee_z",
        type=float,
        default=0.025,
        help="Require first close EE z to be at or below this value when --require_lift_success is set.",
    )
    parser.add_argument(
        "--max_close_xy_error",
        type=float,
        default=0.006,
        help="Require first close XY error to be at or below this value when --require_lift_success is set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume into dataset_root without overwriting existing episode directories.",
    )
    parser.add_argument(
        "--scene_reuse_all_colors",
        action="store_true",
        help="Sample one object layout, then collect target episodes for every remaining color in that same scene.",
    )
    parser.add_argument(
        "--scene_color_max_failures",
        type=int,
        default=3,
        help=(
            "When scene_reuse_all_colors is enabled, abandon the current scene "
            "after this many failures for the same target color. 0 disables scene rejection."
        ),
    )
    parser.add_argument(
        "--restart_sim_every_attempts",
        type=int,
        default=0,
        help="Recreate the MuJoCo model/data/renderer after this many attempts. 0 disables periodic restarts.",
    )
    parser.add_argument(
        "--restart_sim_after_fail_streak",
        type=int,
        default=0,
        help="Recreate the MuJoCo model/data/renderer after this many consecutive failed attempts. 0 disables this guard.",
    )
    parser.add_argument(
        "--require_lift_success",
        action="store_true",
        help="Only count/save episodes when strict_lift_success is true. Default preserves contact-based success.",
    )
    parser.add_argument(
        "--trajectory_mode",
        choices=["final_align_lift_deep_immediate", "v12_pitch_adaptive"],
        default="final_align_lift_deep_immediate",
        help=(
            "final_align_lift_deep_immediate: V11 horizontal baseline. "
            "v12_pitch_adaptive: add pitch_alpha action, using vertical grasp for radial interference targets."
        ),
    )
    parser.add_argument(
        "--v12_interference_scenes",
        action="store_true",
        help=(
            "Sample hard scenes with a near/far pair on the same robot-radial line. "
            "The far target is labeled with vertical gripper pitch in v12_pitch_adaptive mode."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    instruction_templates = None
    instruction_templates_mode = args.instruction_mode
    if args.instruction_mode == "language_extended":
        instruction_templates = args.instruction_templates or DEFAULT_LANGUAGE_EXTENDED_TEMPLATES
    elif args.instruction_mode == "lift_extended":
        instruction_templates = args.instruction_templates or DEFAULT_LIFT_EXTENDED_TEMPLATES

    collect_dataset(
        xml_path=args.xml_path,
        dataset_root=args.dataset_root,
        num_episodes=args.num_episodes,
        colors=("red", "blue", "green", "yellow"),
        instruction_template=args.instruction_template,
        instruction_templates=instruction_templates,
        instruction_templates_mode=instruction_templates_mode,
        keep_failed=args.keep_failed,
        use_viewer=args.use_viewer,
        camera_name=args.camera_name,
        speed=args.speed,
        settle_seconds_per_action=args.settle_seconds_per_action,
        initial_settle_seconds=args.initial_settle_seconds,
        hz=args.hz,
        touch_threshold=args.touch_threshold,
        seed=args.seed,
        max_attempts=args.max_attempts,
        object_x_range=(args.object_x_min, args.object_x_max),
        object_y_range=(args.object_y_min, args.object_y_max),
        min_object_distance=args.min_object_distance,
        trajectory_mode=args.trajectory_mode,
        require_lift_success=args.require_lift_success,
        max_close_ee_z=args.max_close_ee_z,
        max_close_xy_error=args.max_close_xy_error,
        resume=args.resume,
        scene_reuse_all_colors=args.scene_reuse_all_colors,
        scene_color_max_failures=args.scene_color_max_failures,
        restart_sim_every_attempts=args.restart_sim_every_attempts,
        restart_sim_after_fail_streak=args.restart_sim_after_fail_streak,
        v12_interference_scenes=args.v12_interference_scenes,
    )
