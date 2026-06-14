#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}/openvla"
export PYTHONPATH="${PROJECT_ROOT}/openvla:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

DEFAULT_BASE_MODEL_PATH="/root/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${DEFAULT_BASE_MODEL_PATH}}"
MODEL_PATH="${MODEL_PATH:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
UNNORM_KEY="${UNNORM_KEY:-raccoon_pick_place}"
MERGE_ADAPTER="${MERGE_ADAPTER:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ -n "${ADAPTER_PATH}" ]]; then
    MODEL_PATH="${PROJECT_ROOT}/openvla/openvla-runs/$(basename "${ADAPTER_PATH}")"
  else
    echo "ERROR: Set MODEL_PATH, or set ADAPTER_PATH so MODEL_PATH can be inferred." >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ "${MERGE_ADAPTER,,}" == "1" || "${MERGE_ADAPTER,,}" == "true" || "${MERGE_ADAPTER,,}" == "yes" ]]; then
  EXTRA_ARGS+=(--merge_adapter)
fi

python openvla_server.py \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${ADAPTER_PATH}" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --default-unnorm-key "${UNNORM_KEY}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
