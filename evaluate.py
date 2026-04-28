import os
import json
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm
from codebleu import calc_codebleu

def load_model(model_dir, tokenizer_dir):
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model = model.to("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    return tokenizer, model


def generate(model, tokenizer, buggy_code, max_new_tokens=256):
    device = model.device
    inputs = tokenizer(
        buggy_code,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def exact_match(pred, gold):
    return pred.strip() == gold.strip()


def evaluate_pipeline(model_dir, tokenizer_dir, test_data):
    tokenizer, model = load_model(model_dir, tokenizer_dir)

    preds = []
    refs = []
    em_total = 0

    print(f"\nEvaluating: {model_dir}")

    for ex in tqdm(test_data):
        buggy = ex["buggy"]
        fixed = ex["fixed"]

        pred = generate(model, tokenizer, buggy)
        preds.append(pred)
        refs.append(fixed)

        if exact_match(pred, fixed):
            em_total += 1

    # Compute metrics
    exact_match_score = em_total / len(test_data)

    codebleu_score = calc_codebleu(
        refs,
        preds,
        lang="java"
    )["codebleu"]

    return exact_match_score, codebleu_score


def main():
    # load test set
    dataset = load_dataset("google/code_x_glue_cc_code_refinement", name="medium")
    test_data = dataset["test"]

    # -----------------------------
    # load paths to models
    TOKENIZER_DIR = "./t5_tokenizer"
    MODEL_PRETRAINED = "./finetuned_pretrained_model"
    MODEL_FINETUNE_ONLY = "./finetuned_only_model"


    # -----------------------------
    # Evaluate both pipelines
    # -----------------------------
    results = {}

    em, cb = evaluate_pipeline(MODEL_PRETRAINED, TOKENIZER_DIR, test_data)
    results["finetuned_pretrained"] = { "exact_match": em, "codebleu": cb}

    em, cb = evaluate_pipeline(MODEL_FINETUNE_ONLY, TOKENIZER_DIR, test_data)
    results["finetuned_only"] = {"exact_match": em, "codebleu": cb}

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFinal Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
