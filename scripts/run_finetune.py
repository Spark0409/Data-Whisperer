"""
Fine-tuning script for experiment pipeline.
Trains a model on a selected subset using LoRA.
"""
import json
import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


def format_conversation_qwen(conv):
    """Format a conversation in Qwen2.5 ChatML format."""
    text = ''
    for msg in conv['conversations']:
        if msg['role'] == 'user':
            text += f'<|im_start|>user\n{msg["content"]}<|im_end|>\n'
        elif msg['role'] == 'assistant':
            text += f'<|im_start|>assistant\n{msg["content"]}<|im_end|>\n'
    text += '<|im_start|>assistant\n'
    return text


def convert_gsm8k_to_conversations(data):
    """Convert GSM8K data to conversation format."""
    conversations = []
    for item in data:
        conversations.append({
            'conversations': [
                {'role': 'user', 'content': item['question']},
                {'role': 'assistant', 'content': item['answer']},
            ]
        })
    return conversations


def convert_dialogsum_to_conversations(data):
    """Convert DialogSum data to conversation format."""
    conversations = []
    for item in data:
        conversations.append({
            'conversations': [
                {'role': 'user', 'content': f'Summarize the following dialogue:\n{item["dialogue"]}'},
                {'role': 'assistant', 'content': item['summary']},
            ]
        })
    return conversations


def convert_bioinstruct_to_conversations(data):
    """Convert BioInstruct data to conversation format."""
    conversations = []
    for item in data:
        user_content = item['instruction']
        if item.get('input'):
            user_content += f'\nInput: {item["input"]}'
        conversations.append({
            'conversations': [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': item['output']},
            ]
        })
    return conversations


DATASET_CONVERTERS = {
    "gsm8k": convert_gsm8k_to_conversations,
    "dialogsum": convert_dialogsum_to_conversations,
    "bioinstruct": convert_bioinstruct_to_conversations,
}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune a model on a selected subset")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained model")
    parser.add_argument("--subset_file", type=str, required=True,
                        help="Path to the subset JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for fine-tuned model")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "dialogsum", "bioinstruct"],
                        help="Dataset type")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to {args.seed}")

    
    # Load subset data
    print(f"Loading subset from {args.subset_file}...")
    with open(args.subset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Convert to conversation format
    converter = DATASET_CONVERTERS.get(args.dataset, convert_gsm8k_to_conversations)
    conversations = converter(data)
    
    # Save training data for reference
    train_data_file = os.path.join(args.output_dir, "train_data.json")
    with open(train_data_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    # Load tokenizer and model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Format texts
    texts = [format_conversation_qwen(conv) for conv in conversations]
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=args.max_length,
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Load model with LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
    )
    
    # Enable gradient checkpointing BEFORE applying LoRA
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, 'checkpoints'),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='no',
        report_to='none',
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    lora_output_path = os.path.join(args.output_dir, 'lora_model')
    model.save_pretrained(lora_output_path)
    tokenizer.save_pretrained(lora_output_path)
    print(f"Model saved to {lora_output_path}")
    
    # Save training config
    config = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "num_samples": len(data),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
