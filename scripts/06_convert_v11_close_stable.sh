#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/v11_convert_close_stable_rlds.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/Mujoco/raccoon_grasp_v10_lift_immediate_1200}"
ACTION_LABEL_SOURCE="${ACTION_LABEL_SOURCE:-command_delta}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_${ACTION_LABEL_SOURCE}}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-42}"
DROP_IDLE_STEPS="${DROP_IDLE_STEPS:-1}"
MIN_EE_DELTA_NORM="${MIN_EE_DELTA_NORM:-0.0005}"
MIN_JOINT_DELTA_NORM="${MIN_JOINT_DELTA_NORM:-0.01}"
MIN_GRIPPER_DELTA="${MIN_GRIPPER_DELTA:-0.0001}"
DROP_POST_CLOSE_HOLD_STEPS="${DROP_POST_CLOSE_HOLD_STEPS:-2}"
DROP_CLOSED_GRIPPER_SMALL_Z_ACTIONS="${DROP_CLOSED_GRIPPER_SMALL_Z_ACTIONS:-1}"
CLOSED_GRIPPER_MIN_Z_ACTION="${CLOSED_GRIPPER_MIN_Z_ACTION:-0.002}"
PROMOTE_PRE_CLOSE_STEPS="${PROMOTE_PRE_CLOSE_STEPS:-3}"
INITIAL_CLOSE_MIN_Z_ACTION="${INITIAL_CLOSE_MIN_Z_ACTION:-0.004}"

EXTRA_CONVERT_ARGS=()
if [[ "${DROP_IDLE_STEPS}" == "1" ]]; then
  EXTRA_CONVERT_ARGS+=(
    --drop_idle_steps
    --min_ee_delta_norm "${MIN_EE_DELTA_NORM}"
    --min_joint_delta_norm "${MIN_JOINT_DELTA_NORM}"
    --min_gripper_delta "${MIN_GRIPPER_DELTA}"
  )
fi

if [[ "${DROP_POST_CLOSE_HOLD_STEPS}" != "0" ]]; then
  EXTRA_CONVERT_ARGS+=(
    --drop_post_close_hold_steps "${DROP_POST_CLOSE_HOLD_STEPS}"
  )
fi

if [[ "${DROP_CLOSED_GRIPPER_SMALL_Z_ACTIONS}" == "1" ]]; then
  EXTRA_CONVERT_ARGS+=(
    --drop_closed_gripper_small_z_actions
    --closed_gripper_min_z_action "${CLOSED_GRIPPER_MIN_Z_ACTION}"
  )
fi

python "${PROJECT_ROOT}/Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py" \
  --raw_root "${RAW_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
  --ee_pose_source fk \
  --action_label_source "${ACTION_LABEL_SOURCE}" \
  --split_by_scene \
  --promote_pre_close_steps "${PROMOTE_PRE_CLOSE_STEPS}" \
  --initial_close_min_z_action "${INITIAL_CLOSE_MIN_Z_ACTION}" \
  "${EXTRA_CONVERT_ARGS[@]}" \
  "$@"
