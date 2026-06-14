#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/merge_v11_stack_raw.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

BASE_RAW_ROOT="${BASE_RAW_ROOT:-${PROJECT_ROOT}/Mujoco/raccoon_grasp_v10_lift_immediate_1200}"
STACK_RAW_ROOT="${STACK_RAW_ROOT:-${PROJECT_ROOT}/Mujoco/raccoon_stack_v11_extension_120}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/Mujoco/raccoon_grasp_v11_plus_stack_raw}"
COPY_MODE="${COPY_MODE:-hardlink}"
OVERWRITE="${OVERWRITE:-1}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

echo "BASE_RAW_ROOT=${BASE_RAW_ROOT}"
echo "STACK_RAW_ROOT=${STACK_RAW_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "COPY_MODE=${COPY_MODE}"

python "${PROJECT_ROOT}/Mujoco/raccoon_dataset/merge_raw_datasets.py" \
  --input_roots "${BASE_RAW_ROOT}" "${STACK_RAW_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --copy_mode "${COPY_MODE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
