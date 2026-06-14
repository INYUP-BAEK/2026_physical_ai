#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/v11_tfds_close_stable_build.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

ACTION_LABEL_SOURCE="${ACTION_LABEL_SOURCE:-command_delta}"
export RACCOON_RLDS_INTERMEDIATE_ROOT="${RACCOON_RLDS_INTERMEDIATE_ROOT:-${PROJECT_ROOT}/Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_${ACTION_LABEL_SOURCE}}"
TFDS_OUT="${TFDS_OUT:-${PROJECT_ROOT}/tensorflow_datasets}"
TFDS_WORK_DIR="${TFDS_WORK_DIR:-${PROJECT_ROOT}/tensorflow_datasets_tmp_build}"

TFDS_BUILD_ARGS=("$@")
HAS_DATA_DIR=0
for arg in "$@"; do
  case "${arg}" in
    --data_dir|--data_dir=*)
      HAS_DATA_DIR=1
      ;;
  esac
done

if [[ "${HAS_DATA_DIR}" == "0" ]]; then
  TFDS_BUILD_ARGS+=(--data_dir "${TFDS_WORK_DIR}")
  TFDS_SRC="${TFDS_WORK_DIR}"
else
  TFDS_SRC="${TFDS_SRC:-${TFDS_OUT}}"
fi

cd "${PROJECT_ROOT}/Mujoco/rlds_dataset_builder/raccoon_pick_place"

echo "RACCOON_RLDS_INTERMEDIATE_ROOT=${RACCOON_RLDS_INTERMEDIATE_ROOT}"
echo "ACTION_LABEL_SOURCE=${ACTION_LABEL_SOURCE}"
echo "TFDS_SRC=${TFDS_SRC}"
echo "TFDS_OUT=${TFDS_OUT}"
tfds build --overwrite "${TFDS_BUILD_ARGS[@]}"

if [[ -d "${TFDS_SRC}" ]]; then
  if [[ "$(realpath -m "${TFDS_SRC}")" != "$(realpath -m "${TFDS_OUT}")" ]]; then
    rm -rf "${TFDS_OUT}"
    mv "${TFDS_SRC}" "${TFDS_OUT}"
  fi
else
  echo "WARNING: TFDS source directory not found after build: ${TFDS_SRC}"
  echo "Set TFDS_SRC to the actual tensorflow_datasets output directory if needed."
fi
