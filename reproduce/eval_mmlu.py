# adapt from https://github.com/declare-lab/instruct-eval/blob/main/mmlu.py
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

from data import template_wo_input
from utils import initialize_text_to_text_model


MMLU_DATA_DIR = Path(__file__).resolve().parent / "data_cache" / "mmlu_data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    return " " + " ".join(subject.split("_"))


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{get_choices()[j]}. {df.iloc[idx, j + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = (
        "The following are multiple choice questions (with answers) about "
        f"{format_subject(subject)}.\n\n"
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def run_mmlu_choice(model, tokenizer, prompt: str, use_template=False):
    text = template_wo_input.format_map(dict(instruction=prompt)) if use_template else prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    choice_tokens = [tokenizer(choice).input_ids[-1] for choice in get_choices()]
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(0)[-1].flatten()
    probs = torch.softmax(logits[choice_tokens], dim=0)
    pred = get_choices()[torch.argmax(probs).item()]
    return pred, probs.cpu().numpy()


def _load_mmlu_model(model_name, tokenizer_name=None, flash_attention=False, bf16=True):
    model, tokenizer = initialize_text_to_text_model(
        model_name,
        "CausalLM",
        bf16,
        use_peft=False,
        tokenizer=tokenizer_name or model_name,
        flash_attention=flash_attention,
        device_map=None,
    )
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()
    return model, tokenizer


def evaluate_mmlu_model(
    model_name,
    tokenizer_name=None,
    flash_attention=False,
    bf16=True,
    ntrain=5,
    data_dir=None,
    results_path=None,
):
    data_root = Path(data_dir) if data_dir is not None else MMLU_DATA_DIR
    test_dir = data_root / "test"
    dev_dir = data_root / "dev"
    if not test_dir.exists() or not dev_dir.exists():
        raise FileNotFoundError(
            f"MMLU data not found under {data_root}. Expected dev/ and test/ directories."
        )

    model, tokenizer = _load_mmlu_model(
        model_name,
        tokenizer_name=tokenizer_name,
        flash_attention=flash_attention,
        bf16=bf16,
    )
    subjects = sorted(
        f.split("_test.csv")[0]
        for f in os.listdir(test_dir)
        if f.endswith("_test.csv")
    )

    all_cors = []
    cat_cors = {cat: [] for cat in get_categories()}
    subcat_cors = {
        subcat: []
        for subcat_list in get_subcategories().values()
        for subcat in subcat_list
    }

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(dev_dir / f"{subject}_dev.csv", header=None)
        if ntrain != -1:
            dev_df = dev_df[:ntrain]
        test_df = pd.read_csv(test_dir / f"{subject}_test.csv", header=None)
        cors = []
        for i in range(test_df.shape[0]):
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, ntrain)
            pred, _ = run_mmlu_choice(model, tokenizer, train_prompt + prompt_end)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            cors.append(pred.strip().startswith(label))
        cors = np.array(cors)
        all_cors.append(cors)
        for subcat in get_subcategories()[subject]:
            subcat_cors[subcat].append(cors)
            for cat, cat_subcats in get_categories().items():
                if subcat in cat_subcats:
                    cat_cors[cat].append(cors)

    weighted_acc = float(np.mean(np.concatenate(all_cors)))
    cat_result = {
        cat: float(np.mean(np.concatenate(values)))
        for cat, values in cat_cors.items()
        if values
    }
    subcat_result = {
        subcat: float(np.mean(np.concatenate(values)))
        for subcat, values in subcat_cors.items()
        if values
    }
    payload = {
        "metric_name": "accuracy",
        "metric_value": weighted_acc,
        "category_metrics": cat_result,
        "subcategory_metrics": subcat_result,
        "num_subjects": len(subjects),
        "data_dir": str(data_root),
    }
    if results_path is not None:
        results_file = Path(results_path)
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("w") as f:
            json.dump(payload, f, indent=2)
    return payload


def main(model_name, ntrain=5, data_dir=str(MMLU_DATA_DIR)):
    results = evaluate_mmlu_model(
        model_name,
        tokenizer_name=model_name,
        flash_attention=False,
        bf16=True,
        ntrain=ntrain,
        data_dir=data_dir,
        results_path=f"{model_name.replace('/', '_')}_mmlu_results.json",
    )
    print(results)


if __name__ == "__main__":
    Fire(main)
