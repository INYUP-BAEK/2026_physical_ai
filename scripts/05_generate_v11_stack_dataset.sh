#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/v11_stack_generate.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/Mujoco/raccoon_stack_v11_extension_120}"
NUM_EPISODES="${NUM_EPISODES:-120}"
SEED="${SEED:-20260612}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2400}"
OBJECT_X_MIN="${OBJECT_X_MIN:--0.10}"
OBJECT_X_MAX="${OBJECT_X_MAX:-0.10}"
OBJECT_Y_MIN="${OBJECT_Y_MIN:-0.135}"
OBJECT_Y_MAX="${OBJECT_Y_MAX:-0.180}"
MIN_OBJECT_DISTANCE="${MIN_OBJECT_DISTANCE:-0.045}"
SETTLE_SECONDS_PER_ACTION="${SETTLE_SECONDS_PER_ACTION:-0.8}"
INITIAL_SETTLE_SECONDS="${INITIAL_SETTLE_SECONDS:-0.1}"
HZ="${HZ:-10}"
SPEED="${SPEED:-150}"
MAX_CLOSE_EE_Z="${MAX_CLOSE_EE_Z:-0.025}"
MAX_CLOSE_XY_ERROR="${MAX_CLOSE_XY_ERROR:-0.006}"

cd "${PROJECT_ROOT}"

echo "DATASET_ROOT=${DATASET_ROOT}"
echo "NUM_EPISODES=${NUM_EPISODES}"
echo "SEED=${SEED}"
echo "MAX_ATTEMPTS=${MAX_ATTEMPTS}"
echo "OBJECT_X_RANGE=${OBJECT_X_MIN},${OBJECT_X_MAX}"
echo "OBJECT_Y_RANGE=${OBJECT_Y_MIN},${OBJECT_Y_MAX}"

python Mujoco/raccoon_stack_dataset.py \
  --xml_path "${PROJECT_ROOT}/Mujoco/Raccoon_colored_cylinder.xml" \
  --dataset_root "${DATASET_ROOT}" \
  --num_episodes "${NUM_EPISODES}" \
  --seed "${SEED}" \
  --max_attempts "${MAX_ATTEMPTS}" \
  --object_x_min "${OBJECT_X_MIN}" \
  --object_x_max "${OBJECT_X_MAX}" \
  --object_y_min "${OBJECT_Y_MIN}" \
  --object_y_max "${OBJECT_Y_MAX}" \
  --min_object_distance "${MIN_OBJECT_DISTANCE}" \
  --speed "${SPEED}" \
  --settle_seconds_per_action "${SETTLE_SECONDS_PER_ACTION}" \
  --initial_settle_seconds "${INITIAL_SETTLE_SECONDS}" \
  --hz "${HZ}" \
  --max_close_ee_z "${MAX_CLOSE_EE_Z}" \
  --max_close_xy_error "${MAX_CLOSE_XY_ERROR}" \
  "$@"
