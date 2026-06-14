#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/v11_train_lora_close_stable.log}"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

cd "${PROJECT_ROOT}/openvla"
export PYTHONPATH="${PROJECT_ROOT}/openvla:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

WANDB_MODE="${WANDB_MODE:-disabled}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEFAULT_OPENVLA_7B_PATH="openvla/openvla-7b"
LOCAL_OPENVLA_7B_PATH="${LOCAL_OPENVLA_7B_PATH:-/root/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f}"
if [[ -z "${VLA_PATH:-}" && -f "${LOCAL_OPENVLA_7B_PATH}/config.json" ]]; then
  VLA_PATH="${LOCAL_OPENVLA_7B_PATH}"
else
  VLA_PATH="${VLA_PATH:-${DEFAULT_OPENVLA_7B_PATH}}"
fi

DATASET_NAME="${DATASET_NAME:-raccoon_pick_place}"
LORA_RANK="${LORA_RANK:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
IMAGE_AUG="${IMAGE_AUG:-true}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100000}"
MAX_STEPS="${MAX_STEPS:-15000}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
RUN_ID_NOTE="${RUN_ID_NOTE:-v11-close-stable-promote3-1200eps-15000steps-b4ga4}"
REFRESH_DATASET_STATS="${REFRESH_DATASET_STATS:-1}"
MERGE_LORA_CHECKPOINT="${MERGE_LORA_CHECKPOINT:-0}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-0}"
INIT_LORA_ADAPTER_PATH="${INIT_LORA_ADAPTER_PATH:-}"

EXTRA_TRAIN_ARGS=()
if [[ -n "${INIT_LORA_ADAPTER_PATH}" ]]; then
  if [[ ! -f "${INIT_LORA_ADAPTER_PATH}/adapter_config.json" ]]; then
    echo "ERROR: INIT_LORA_ADAPTER_PATH does not look like a LoRA adapter directory: ${INIT_LORA_ADAPTER_PATH}" >&2
    exit 1
  fi
  EXTRA_TRAIN_ARGS+=(--init_lora_adapter_path "${INIT_LORA_ADAPTER_PATH}")
fi

if [[ "${MERGE_LORA_CHECKPOINT,,}" == "0" || "${MERGE_LORA_CHECKPOINT,,}" == "false" || "${MERGE_LORA_CHECKPOINT,,}" == "no" ]]; then
  MERGE_LORA_CHECKPOINT_ARG="false"
else
  MERGE_LORA_CHECKPOINT_ARG="true"
fi

if [[ "${SAVE_LATEST_CHECKPOINT_ONLY,,}" == "0" || "${SAVE_LATEST_CHECKPOINT_ONLY,,}" == "false" || "${SAVE_LATEST_CHECKPOINT_ONLY,,}" == "no" ]]; then
  SAVE_LATEST_CHECKPOINT_ONLY_ARG="false"
else
  SAVE_LATEST_CHECKPOINT_ONLY_ARG="true"
fi

if [[ "${REFRESH_DATASET_STATS}" == "1" ]]; then
  find "${PROJECT_ROOT}/tensorflow_datasets/raccoon_pick_place" \
    -name 'dataset_statistics_*.json' \
    -type f \
    -delete 2>/dev/null || true
fi

echo "VLA_PATH=${VLA_PATH}"
echo "HF_HOME=${HF_HOME}"
echo "RUN_ID_NOTE=${RUN_ID_NOTE}"
echo "LORA_RANK=${LORA_RANK}"
echo "LORA_DROPOUT=${LORA_DROPOUT}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRAD_ACCUMULATION_STEPS=${GRAD_ACCUMULATION_STEPS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "IMAGE_AUG=${IMAGE_AUG}"
echo "SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "MERGE_LORA_CHECKPOINT=${MERGE_LORA_CHECKPOINT_ARG}"
echo "SAVE_LATEST_CHECKPOINT_ONLY=${SAVE_LATEST_CHECKPOINT_ONLY_ARG}"
echo "INIT_LORA_ADAPTER_PATH=${INIT_LORA_ADAPTER_PATH:-<fresh-lora>}"

WANDB_MODE="${WANDB_MODE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${PROJECT_ROOT}/tensorflow_datasets" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${PROJECT_ROOT}/openvla/openvla-runs" \
  --adapter_tmp_dir "${PROJECT_ROOT}/openvla/openvla-adapter-tmp" \
  --lora_rank "${LORA_RANK}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --image_aug "${IMAGE_AUG}" \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --merge_lora_checkpoint "${MERGE_LORA_CHECKPOINT_ARG}" \
  --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY_ARG}" \
  --run_id_note "${RUN_ID_NOTE}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "$@"
