import os
import re
import sys
from pathlib import Path

import torch
from fire import Fire
from tqdm import tqdm

from utils import initialize_text_to_text_model, model_inference


ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""


def _human_eval_imports():
    try:
        from human_eval.data import HUMAN_EVAL, read_problems, write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness

        return HUMAN_EVAL, read_problems, write_jsonl, evaluate_functional_correctness
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = []
        candidates.append(repo_root / "human-eval")
        human_eval_path = os.environ.get("HUMAN_EVAL_PATH")
        if human_eval_path:
            candidates.append(Path(human_eval_path))

        for candidate in candidates:
            if not candidate.exists():
                continue
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            try:
                from human_eval.data import HUMAN_EVAL, read_problems, write_jsonl
                from human_eval.evaluation import evaluate_functional_correctness

                return HUMAN_EVAL, read_problems, write_jsonl, evaluate_functional_correctness
            except ModuleNotFoundError:
                continue

    raise ModuleNotFoundError(
        "human_eval is not available. Run `uv sync` from the repo root to "
        "install the vendored human-eval package."
    )


def post_process(text):
    text = text.replace("```", "")
    text = text.replace("\t", "    ")
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', "", text, flags=re.DOTALL)
    text = "\n".join([line.rstrip() for line in text.splitlines() if line.strip()])
    lines = text.split("\n")
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r"^( *)", line)
        spaces_for_each_line.append(len(match.group(1)) if match else 0)
    try:
        def_line = [i for i, line in enumerate(lines) if "def" in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except Exception:
        def_line_space = 0

    rank_unique_spaces = sorted(set(spaces_for_each_line))
    indentation_level = {}
    level = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            level += 1
            indentation_level[space] = level

    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())
    return "\n".join(new_lines)


def generate_one_completion(model, tokenizer, model_type, prompt, template=True):
    prompt_in = ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt) if template else prompt
    pred_text = model_inference(
        model,
        tokenizer,
        prompt_in,
        model_type,
        max_target_length=512,
    )
    return post_process(pred_text)


def _model_slug(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return f"{model_path.parent.name}_{model_path.name}".replace("/", "_")
    return model_name.replace("/", "_")


def evaluate_humaneval_model(
    model_name,
    tokenizer_name=None,
    flash_attention=False,
    bf16=True,
    num_samples_per_task=1,
    output_dir=None,
    cleanup=False,
    n_workers=4,
    timeout=3.0,
    problem_file=None,
    max_tasks=None,
):
    human_eval_default, read_problems, write_jsonl, evaluate_functional_correctness = (
        _human_eval_imports()
    )

    problem_path = problem_file or human_eval_default
    problems = read_problems(problem_path)
    if max_tasks is not None:
        selected_items = list(problems.items())[:max_tasks]
        problems = dict(selected_items)

    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        model_name,
        model_type,
        bf16,
        use_peft=False,
        tokenizer=tokenizer_name or model_name,
        flash_attention=flash_attention,
        device_map=None,
    )
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.do_sample = False

    samples = []
    for task_id in tqdm(problems, desc="Tasks"):
        prompt = problems[task_id]["prompt"]
        for _ in range(num_samples_per_task):
            samples.append(
                {
                    "task_id": task_id,
                    "completion": generate_one_completion(
                        model,
                        tokenizer,
                        model_type,
                        prompt,
                    ),
                }
            )

    output_root = Path(output_dir or "./humaneval_samples")
    output_root.mkdir(parents=True, exist_ok=True)
    sample_file = output_root / (
        f"{_model_slug(model_name)}_nsamples{num_samples_per_task}_humaneval_samples.jsonl"
    )
    write_jsonl(str(sample_file), samples)

    eval_problem_file = problem_path
    if max_tasks is not None:
        eval_problem_file = output_root / "humaneval_problem_subset.jsonl"
        write_jsonl(
            str(eval_problem_file),
            [
                {"task_id": task_id, **problem}
                for task_id, problem in problems.items()
            ],
        )

    results = evaluate_functional_correctness(
        str(sample_file),
        k=[1],
        n_workers=n_workers,
        timeout=timeout,
        problem_file=str(eval_problem_file),
    )
    result_file = Path(f"{sample_file}_results.jsonl")

    payload = {
        "metric_name": "pass@1",
        "metric_value": float(results["pass@1"]),
        "num_tasks": len(problems),
        "num_samples_per_task": num_samples_per_task,
        "sample_file": str(sample_file),
        "result_file": str(result_file),
    }

    if cleanup:
        sample_file.unlink(missing_ok=True)
        result_file.unlink(missing_ok=True)
        if max_tasks is not None:
            Path(eval_problem_file).unlink(missing_ok=True)

    return payload


def main(model_name, num_sample=1, output_dir=None):
    results = evaluate_humaneval_model(
        model_name,
        tokenizer_name=model_name,
        flash_attention=False,
        bf16=True,
        num_samples_per_task=num_sample,
        output_dir=output_dir,
    )
    print(results)


if __name__ == "__main__":
    Fire(main)
