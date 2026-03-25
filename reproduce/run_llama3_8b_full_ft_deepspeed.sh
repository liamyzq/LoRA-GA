#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}
DATASET_NAME=${DATASET_NAME:-meta_math}
MODEL_CONFIG=${MODEL_CONFIG:-llama3_8b_instruct}
WANDB_PROJECT=${WANDB_PROJECT:-lora_ga_full_ft}
FLASH_ATTENTION=${FLASH_ATTENTION:-false}

cd "$SCRIPT_DIR"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

deepspeed --num_gpus="$NUM_GPUS" --master_port="$MASTER_PORT" run_exp.py \
  model="$MODEL_CONFIG" \
  +peft=full_ft \
  +dataset_name="$DATASET_NAME" \
  ++flash_attention="$FLASH_ATTENTION" \
  ++gradient_checkpointing=true \
  ++wandb.project="$WANDB_PROJECT" \
  ++deepspeed.enabled=true \
  ++deepspeed.config="$SCRIPT_DIR/deepspeed/zero3_bf16.json" \
  "$@"
