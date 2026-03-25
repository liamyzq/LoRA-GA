# Full FT: Llama 3.1 8B

This repo supports full fine-tuning `meta-llama/Llama-3.1-8B` with DeepSpeed through `reproduce/`.

## Setup

From the repo root:

```bash
cd /home/mlw0719/LoRA-GA
DS_BUILD_OPS=0 uv sync
```

Notes:
- This uses the local editable [`peft`](/home/mlw0719/LoRA-GA/peft).
- This also installs the vendored [`human-eval`](/home/mlw0719/LoRA-GA/human-eval) package from this repo, so HumanEval does not depend on any external checkout.
- `transformers` is capped at `<=4.45.2`.
- `flash-attn` is optional and not required for the full-FT launch below.
- The launcher can run `gsm8k`, `humaneval`, or `mmlu` after training and writes a final report automatically.

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
uv run --directory reproduce ./run_llama3_1_8b_full_ft_deepspeed.sh \
++model.learning_rate=2e-5
```

To switch to HumanEval:

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
EVAL_TASK=humaneval \
uv run --directory reproduce ./run_llama3_1_8b_full_ft_deepspeed.sh \
++model.learning_rate=2e-5
```

To switch to MMLU:

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
EVAL_TASK=mmlu \
MMLU_DATA_DIR=/path/to/mmlu_data \
uv run --directory reproduce ./run_llama3_1_8b_full_ft_deepspeed.sh \
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
uv run --directory reproduce ./run_llama3_1_8b_full_ft_deepspeed.sh \
++model.learning_rate=2e-5'"
```

Attach:

```bash
tmux attach -t lora_ga_llama3_fullft
```

## What It Uses

- launcher: [/home/mlw0719/LoRA-GA/reproduce/run_llama3_1_8b_full_ft_deepspeed.sh](/home/mlw0719/LoRA-GA/reproduce/run_llama3_1_8b_full_ft_deepspeed.sh)
- model config: [/home/mlw0719/LoRA-GA/reproduce/conf/model/llama3_1_8b.yaml](/home/mlw0719/LoRA-GA/reproduce/conf/model/llama3_1_8b.yaml)
- DeepSpeed config: [/home/mlw0719/LoRA-GA/reproduce/deepspeed/zero2_bf16.json](/home/mlw0719/LoRA-GA/reproduce/deepspeed/zero2_bf16.json)

## Runtime Notes

Training can run with multi-GPU DeepSpeed. After training, rank 0 reloads the saved checkpoint and runs evaluation on a single device. `EVAL_TASK=gsm8k` reports accuracy, `EVAL_TASK=humaneval` reports pass@1, and `EVAL_TASK=mmlu` reports accuracy with category breakdowns.

## Outputs

After a run, the main process writes:

- `results/.../run_report.json`
- `results/.../run_report.txt`
- `results/.../gsm8k_eval.json`
- `results/.../humaneval/..._humaneval_samples.jsonl`
- `results/.../humaneval/..._humaneval_samples.jsonl_results.jsonl`
- `results/.../mmlu_eval.json`

These include total runtime, periodic GPU-memory/utilization summaries, and the final evaluation result.
