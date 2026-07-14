"""
Evaluate fine-tuned model on GSM8K test set using Exact Match (EM) metric.

Uses batch inference for faster evaluation.

Usage:
    python scripts/eval_gsm8k.py \
        --model_name_or_path /path/to/merged_model \
        --data_path ../data/gsm8k_test.json \
        --batch_size 8 \
        --output_file results/gsm8k_eval_results.json
"""

import json
import re
import torch
import argparse
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_answer(text):
    """Extract the final numeric answer after '####' from model output."""
    answer_pattern = re.compile(r"####\s*(-?\d+\.?\d*)")
    match = answer_pattern.search(text)
    if match:
        return match.group(1)
    # Fallback: try to find the last number in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def build_prompt(question):
    """Build Qwen chat format prompt for GSM8K."""
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step and provide the final answer after '####'."},
        {"role": "user", "content": f"Question: {question}\nPlease solve this problem step by step and put the final answer after '####'."}
    ]
    return messages


def evaluate(args):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load test data (JSONL format: one JSON object per line)
    test_data = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    correct = 0
    total = 0
    results = []
    batch_size = args.batch_size

    # Pre-build all prompts
    all_prompts = []
    all_ground_truths = []
    all_true_answers = []
    for item in test_data:
        question = item["instruction"]
        ground_truth = item["output"]
        true_answer = extract_answer(ground_truth)
        messages = build_prompt(question)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        all_prompts.append(prompt)
        all_ground_truths.append(ground_truth)
        all_true_answers.append(true_answer)

    # Batch inference
    start_time = time.time()
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Evaluating"):
        batch_prompts = all_prompts[i:i + batch_size]
        batch_truths = all_ground_truths[i:i + batch_size]
        batch_true_answers = all_true_answers[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        # Decode and evaluate each sample in batch
        input_lengths = inputs.input_ids.shape[1]
        for j, generated_ids in enumerate(outputs):
            generated = tokenizer.decode(generated_ids[input_lengths:], skip_special_tokens=True)
            pred_answer = extract_answer(generated)

            true_answer = batch_true_answers[j]
            is_correct = False
            if true_answer is not None and pred_answer is not None:
                try:
                    is_correct = float(true_answer) == float(pred_answer)
                except ValueError:
                    is_correct = str(true_answer).strip() == str(pred_answer).strip()

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": batch_prompts[j],
                "ground_truth": batch_truths[j],
                "prediction": generated.strip(),
                "extracted_true_answer": true_answer,
                "extracted_pred_answer": pred_answer,
                "correct": is_correct
            })

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"GSM8K Evaluation Results")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy (EM): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s per sample)")
    print(f"{'='*50}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time_seconds": elapsed,
        "results": results
    }
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GSM8K with Exact Match")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--data_path", type=str, default=None, help="Path to test data (default: auto-detect relative to script location)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (default: auto-detect relative to script location)")
    args = parser.parse_args()
    
    # Auto-detect default paths relative to script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.data_path is None:
        args.data_path = os.path.join(base_dir, "data", "gsm8k_test.json")
    if args.output_file is None:
        args.output_file = os.path.join(base_dir, "results", "gsm8k_eval_results.json")
    
    evaluate(args)
