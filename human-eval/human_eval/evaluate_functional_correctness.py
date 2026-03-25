import fire
import sys
import os
from pathlib import Path
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

REPO_ROOT = Path(__file__).resolve().parents[2]
HUMANEVAL_RESULTS_DIR = REPO_ROOT / "subspace_method" / "humaneval_eval_results"


def entry_point(
    sample_file: str,
    k: str = "1",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
    # Save result: a dict containing pass@1, extract this and write it to a txt
    # not existing, create it
    HUMANEVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_txt = HUMANEVAL_RESULTS_DIR / "humaneval_results.txt"
    if not output_txt.exists():
        with output_txt.open("w") as f:
            f.write(f"{sample_file}\n pass@{k}: {results}\n")
    else:
        with output_txt.open("a") as f:
            f.write(f"{sample_file}\n pass@{k}: {results}\n")


def main():
    fire.Fire(entry_point)


sys.exit(main())
