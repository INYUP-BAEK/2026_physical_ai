#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/v11_generate_lift_dataset.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/Mujoco/raccoon_grasp_v10_lift_immediate_1200}"
NUM_EPISODES="${NUM_EPISODES:-1200}"
SEED="${SEED:-909}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-60000}"
RESTART_SIM_EVERY_ATTEMPTS="${RESTART_SIM_EVERY_ATTEMPTS:-120}"
RESTART_SIM_AFTER_FAIL_STREAK="${RESTART_SIM_AFTER_FAIL_STREAK:-40}"
SCENE_COLOR_MAX_FAILURES="${SCENE_COLOR_MAX_FAILURES:-3}"
TRAJECTORY_MODE="${TRAJECTORY_MODE:-final_align_lift_deep_immediate}"
MIN_OBJECT_DISTANCE="${MIN_OBJECT_DISTANCE:-0.042}"
MAX_CLOSE_EE_Z="${MAX_CLOSE_EE_Z:-0.025}"
MAX_CLOSE_XY_ERROR="${MAX_CLOSE_XY_ERROR:-0.006}"
OBJECT_X_MIN="${OBJECT_X_MIN:--0.10}"
OBJECT_X_MAX="${OBJECT_X_MAX:-0.10}"
OBJECT_Y_MIN="${OBJECT_Y_MIN:-0.16}"
OBJECT_Y_MAX="${OBJECT_Y_MAX:-0.195}"

python "${PROJECT_ROOT}/Mujoco/raccoon_grasp_multicolor_scene_dataset.py" \
  --xml_path "${PROJECT_ROOT}/Mujoco/Raccoon_colored_cylinder.xml" \
  --dataset_root "${DATASET_ROOT}" \
  --num_episodes "${NUM_EPISODES}" \
  --seed "${SEED}" \
  --max_attempts "${MAX_ATTEMPTS}" \
  --trajectory_mode "${TRAJECTORY_MODE}" \
  --instruction_mode lift_extended \
  --settle_seconds_per_action 0.8 \
  --initial_settle_seconds 0.1 \
  --hz 10 \
  --object_x_min "${OBJECT_X_MIN}" \
  --object_x_max "${OBJECT_X_MAX}" \
  --object_y_min "${OBJECT_Y_MIN}" \
  --object_y_max "${OBJECT_Y_MAX}" \
  --min_object_distance "${MIN_OBJECT_DISTANCE}" \
  --max_close_ee_z "${MAX_CLOSE_EE_Z}" \
  --max_close_xy_error "${MAX_CLOSE_XY_ERROR}" \
  --require_lift_success \
  --scene_reuse_all_colors \
  --scene_color_max_failures "${SCENE_COLOR_MAX_FAILURES}" \
  --restart_sim_every_attempts "${RESTART_SIM_EVERY_ATTEMPTS}" \
  --restart_sim_after_fail_streak "${RESTART_SIM_AFTER_FAIL_STREAK}" \
  --resume \
  "$@"
