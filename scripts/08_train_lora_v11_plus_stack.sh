#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_INIT_LORA_ADAPTER_PATH="${PROJECT_ROOT}/openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt"

export RUN_ID_NOTE="${RUN_ID_NOTE:-v11-plus-stack-from-v11-close004-promote3-release2}"
export INIT_LORA_ADAPTER_PATH="${INIT_LORA_ADAPTER_PATH:-${DEFAULT_INIT_LORA_ADAPTER_PATH}}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/v11_plus_stack_train_lora.log}"

"${SCRIPT_DIR}/08_train_lora_v11.sh" "$@"
