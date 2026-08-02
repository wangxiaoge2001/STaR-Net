#!/usr/bin/env bash
set -euo pipefail

# Example launcher for the reorganized repository.
# 1. Update CONFIG_PATH to the YAML you want to run.
# 2. Update RUN_NAME and GPU_IDS if needed.

GPU_IDS="${GPU_IDS:-0}"
RUN_NAME="${RUN_NAME:-}"
TRAIN_SCRIPT="train.py"
CONFIG_PATH="${CONFIG_PATH:-./configs/STaR-Net/train_starnet.yaml}"

cmd=(python "$TRAIN_SCRIPT" --config "$CONFIG_PATH" --gpu "$GPU_IDS")
if [[ -n "$RUN_NAME" ]]; then
  cmd+=(--name "$RUN_NAME")
fi

"${cmd[@]}"
