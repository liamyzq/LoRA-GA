from data import load_gsm8k
from utils import model_inference, initialize_text_to_text_model
from fire import Fire
import re
import os
import json
import torch
from tqdm import tqdm

def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r'####\s*(\d+)'
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
    else:
        print(text)
        result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0


def evaluate_gsm8k_model(
    model_name,
    tokenizer_name=None,
    flash_attention=False,
    bf16=True,
    results_path=None,
):
    _, _, test_set = load_gsm8k()
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
    total = 0
    correct = 0
    t = tqdm(test_set)
    for example in t:
        pred_text = model_inference(
            model,
            tokenizer,
            example["x"],
            model_type,
            max_target_length=512,
        )
        gt = extract_num(example["y"])
        pred = extract_num(pred_text)
        correct += int(gt == pred)
        total += 1
        t.set_description(f"Accuracy: {correct / total * 100:02f}%")

    accuracy = correct / total if total else 0.0
    results = {
        "model_name": model_name,
        "num_examples": total,
        "accuracy": accuracy,
    }
    if results_path is not None:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
    return results

def main(model_name):
    results = evaluate_gsm8k_model(
        model_name,
        tokenizer_name=model_name,
        flash_attention=False,
        bf16=True,
    )
    print("Acc:", results["accuracy"])
    # append to gsm8k_results.txt (create if not exists)
    if not os.path.exists("gsm8k_results.txt"):
        with open("gsm8k_results.txt", "w") as f:
            f.write("Model Acc\n")
    with open("gsm8k_results.txt", "a") as f:
        f.write(f"{model_name} {results['accuracy']}\n")

if __name__ == "__main__":
    Fire(main)
