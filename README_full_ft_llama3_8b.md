# Full FT: Llama 3 8B Instruct

This repo supports full fine-tuning `meta-llama/Meta-Llama-3-8B-Instruct` with DeepSpeed through `reproduce/`.

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
- The launcher runs GSM8K evaluation after training and writes a final report automatically.

## Launch

Example on GPUs `0,1,2`:

```bash
cd /home/mlw0719/LoRA-GA
CUDA_VISIBLE_DEVICES=0,1,2 \
NUM_GPUS=3 \
MASTER_PORT=29606 \
SEED=2030 \
MONITOR_INTERVAL_MINUTES=5 \
WANDB_PROJECT=finetune_llama3_full \
WANDB_ENTITY_OVERRIDE=zeqiye-northwestern-university \
FLASH_ATTENTION=false \
uv run --directory reproduce ./run_llama3_8b_full_ft_deepspeed.sh \
++model.learning_rate=2e-5
```

Equivalent tmux launch:

```bash
tmux new-session -d -s lora_ga_llama3_fullft \
"bash -lc 'cd /home/mlw0719/LoRA-GA && \
CUDA_VISIBLE_DEVICES=0,1,2 NUM_GPUS=3 MASTER_PORT=29606 \
SEED=2030 MONITOR_INTERVAL_MINUTES=5 \
WANDB_PROJECT=finetune_llama3_full \
WANDB_ENTITY_OVERRIDE=zeqiye-northwestern-university \
FLASH_ATTENTION=false \
uv run --directory reproduce ./run_llama3_8b_full_ft_deepspeed.sh \
++model.learning_rate=2e-5'"
```

Attach:

```bash
tmux attach -t lora_ga_llama3_fullft
```

## What It Uses

- launcher: [/home/mlw0719/LoRA-GA/reproduce/run_llama3_8b_full_ft_deepspeed.sh](/home/mlw0719/LoRA-GA/reproduce/run_llama3_8b_full_ft_deepspeed.sh)
- model config: [/home/mlw0719/LoRA-GA/reproduce/conf/model/llama3_8b_instruct.yaml](/home/mlw0719/LoRA-GA/reproduce/conf/model/llama3_8b_instruct.yaml)
- DeepSpeed config: [/home/mlw0719/LoRA-GA/reproduce/deepspeed/zero2_bf16.json](/home/mlw0719/LoRA-GA/reproduce/deepspeed/zero2_bf16.json)

## Runtime Notes

Training can run with multi-GPU DeepSpeed. After training, rank 0 reloads the saved checkpoint and runs GSM8K evaluation on a single device.

## Outputs

After a run, the main process writes:

- `results/.../run_report.json`
- `results/.../run_report.txt`
- `results/.../gsm8k_eval.json`

These include total runtime, periodic GPU-memory/utilization summaries, and the GSM8K result.
