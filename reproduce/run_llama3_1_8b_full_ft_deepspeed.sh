#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}
DATASET_NAME=${DATASET_NAME:-meta_math}
MODEL_CONFIG=${MODEL_CONFIG:-llama3_1_8b}
WANDB_PROJECT=${WANDB_PROJECT:-finetune_llama3_full}
FLASH_ATTENTION=${FLASH_ATTENTION:-false}
SEED=${SEED:-9}
EVAL_TASK=${EVAL_TASK:-gsm8k}
MONITOR_INTERVAL_MINUTES=${MONITOR_INTERVAL_MINUTES:-5}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-$SCRIPT_DIR/deepspeed/zero2_bf16.json}
MMLU_DATA_DIR=${MMLU_DATA_DIR:-}
MMLU_NTRAIN=${MMLU_NTRAIN:-5}

cd "$SCRIPT_DIR"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

cmd=(
  deepspeed --num_gpus="$NUM_GPUS" --master_port="$MASTER_PORT" run_exp.py
  model="$MODEL_CONFIG"
  +peft=full_ft
  +dataset_name="$DATASET_NAME"
  ++seed="$SEED"
  ++flash_attention="$FLASH_ATTENTION"
  ++gradient_checkpointing=true
  ++wandb.project="$WANDB_PROJECT"
  ++evaluation.enabled=true
  ++evaluation.task="$EVAL_TASK"
  ++monitor.enabled=true
  ++monitor.interval_minutes="$MONITOR_INTERVAL_MINUTES"
  ++deepspeed.enabled=true
  ++deepspeed.config="$DEEPSPEED_CONFIG"
  ++evaluation.ntrain="$MMLU_NTRAIN"
)

if [[ -n "$MMLU_DATA_DIR" ]]; then
  cmd+=(++evaluation.data_dir="$MMLU_DATA_DIR")
fi

cmd+=("$@")

"${cmd[@]}"
