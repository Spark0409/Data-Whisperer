"""
Evaluation script for experiment pipeline.
Evaluates a fine-tuned model on a test set.
"""
import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def extract_gsm8k_answer(text):
    """Extract the final answer from GSM8K format (#### <number>)."""
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    return match.group(1) if match else None


def extract_dialogsum_answer(text):
    """For dialogsum, use the full generated text as answer."""
    return text.strip()


def extract_bioinstruct_answer(text):
    """For bioinstruct, use the full generated text as answer."""
    return text.strip()


ANSWER_EXTRACTORS = {
    "gsm8k": extract_gsm8k_answer,
    "dialogsum": extract_dialogsum_answer,
    "bioinstruct": extract_bioinstruct_answer,
}


def prepare_prompts(test_data, args):
    """Prepare all prompts and ground truth answers."""
    prompts = []
    true_answers = []
    questions = []
    
    for item in test_data:
        if args.dataset == "gsm8k":
            question = item['question']
            true_answer = extract_gsm8k_answer(item['answer'])
            prompt = f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'
        elif args.dataset == "dialogsum":
            question = f'Summarize the following dialogue:\n{item["dialogue"]}'
            true_answer = item['summary']
            prompt = f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'
        elif args.dataset == "bioinstruct":
            user_content = item['instruction']
            if item.get('input'):
                user_content += f'\nInput: {item["input"]}'
            question = user_content
            true_answer = item['output']
            prompt = f'<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n'
        else:
            question = item.get('question', '')
            true_answer = item.get('answer', '')
            prompt = f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'
        
        prompts.append(prompt)
        true_answers.append(true_answer)
        questions.append(question)
    
    return prompts, true_answers, questions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the base pretrained model")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to the fine-tuned LoRA weights")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the test set JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "dialogsum", "bioinstruct"],
                        help="Dataset type")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum input length")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Batch size for evaluation generation")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model
    print(f"Loading base model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Left padding for batch generation
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
    )
    
    print(f"Loading LoRA weights from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    
    # Prepare all prompts
    print("Preparing prompts...")
    prompts, true_answers, questions = prepare_prompts(test_data, args)
    
    # Get answer extractor
    extract_answer = ANSWER_EXTRACTORS.get(args.dataset, extract_gsm8k_answer)
    
    # Evaluate in batches
    results = []
    correct = 0
    total = 0
    
    batch_size = args.eval_batch_size
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    print(f"Evaluating on {len(test_data)} samples (batch_size={batch_size}, {num_batches} batches)...")
    
    pbar = tqdm(total=len(prompts), desc="Evaluating", unit="sample")
    
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        batch_questions = questions[batch_idx:batch_idx + batch_size]
        batch_true = true_answers[batch_idx:batch_idx + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            truncation=True,
            padding='longest',
            max_length=args.max_length,
        ).to(model.device)
        
        # Generate batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode each sample in the batch
        for i, (question, true_answer) in enumerate(zip(batch_questions, batch_true)):
            prompt_len = inputs.input_ids[i].shape[0]
            generated = tokenizer.decode(
                outputs[i][prompt_len:],
                skip_special_tokens=True
            )
            
            if args.dataset == "gsm8k":
                pred_answer = extract_gsm8k_answer(generated)
                is_correct = (pred_answer is not None and true_answer is not None 
                             and pred_answer == true_answer)
            else:
                pred_answer = generated
                is_correct = None
            
            if is_correct is not None:
                if is_correct:
                    correct += 1
                total += 1
            
            results.append({
                'question': question,
                'true_answer': true_answer,
                'predicted_answer': pred_answer,
                'generated_text': generated,
                'correct': is_correct,
            })
        
        # Update progress bar
        pbar.update(len(batch_prompts))
        if total > 0:
            pbar.set_postfix(acc=f"{correct/total:.3f}")
    
    pbar.close()
    
    # Compute accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Save results
    output = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'details': results,
    }
    
    output_file = os.path.join(args.output_dir, 'results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
