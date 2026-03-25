# Full FT: Llama 3 8B Instruct

This repo now supports full fine-tuning `meta-llama/Meta-Llama-3-8B-Instruct` with DeepSpeed through the legacy `reproduce/` pipeline.

## Setup

From the repo root:

```bash
cd /home/mlw0719/LoRA-GA
DS_BUILD_OPS=0 uv sync
```

Notes:
- This uses the local editable [`peft`](/home/mlw0719/LoRA-GA/peft).
- `transformers` is capped at `<=4.45.2`.
- `flash-attn` is optional and not required for the full-FT launch below.

## Launch

Example on GPUs `0,1,2`:

```bash
cd /home/mlw0719/LoRA-GA
CUDA_VISIBLE_DEVICES=0,1,2 \
NUM_GPUS=3 \
MASTER_PORT=29606 \
WANDB_PROJECT=finetune_llama3_full \
WANDB_ENTITY_OVERRIDE=zeqiye-northwestern-university \
FLASH_ATTENTION=false \
uv run --directory reproduce ./run_llama3_8b_full_ft_deepspeed.sh \
++seed=9 ++model.learning_rate=2e-5
```

Equivalent tmux launch:

```bash
tmux new-session -d -s lora_ga_llama3_fullft \
"bash -lc 'cd /home/mlw0719/LoRA-GA && \
CUDA_VISIBLE_DEVICES=0,1,2 NUM_GPUS=3 MASTER_PORT=29606 \
WANDB_PROJECT=finetune_llama3_full \
WANDB_ENTITY_OVERRIDE=zeqiye-northwestern-university \
FLASH_ATTENTION=false \
uv run --directory reproduce ./run_llama3_8b_full_ft_deepspeed.sh \
++seed=9 ++model.learning_rate=2e-5'"
```

Attach:

```bash
tmux attach -t lora_ga_llama3_fullft
```

## What It Uses

- launcher: [/home/mlw0719/LoRA-GA/reproduce/run_llama3_8b_full_ft_deepspeed.sh](/home/mlw0719/LoRA-GA/reproduce/run_llama3_8b_full_ft_deepspeed.sh)
- model config: [/home/mlw0719/LoRA-GA/reproduce/conf/model/llama3_8b_instruct.yaml](/home/mlw0719/LoRA-GA/reproduce/conf/model/llama3_8b_instruct.yaml)
- DeepSpeed config: [/home/mlw0719/LoRA-GA/reproduce/deepspeed/zero3_bf16.json](/home/mlw0719/LoRA-GA/reproduce/deepspeed/zero3_bf16.json)

## Current Caveat

The launch path works and reaches real training, but the tested `3 x A6000` configuration hit CUDA OOM during backward with the current settings. If needed, reduce batch/length or add ZeRO offload.
