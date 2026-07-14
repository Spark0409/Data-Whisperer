"""
Convert GSM8K data from Data Whisperer format to Llama-Factory format.

Data Whisperer GSM8K format:
{
    "question": "...",
    "answer": "step-by-step solution\n#### <number>"
}

Llama-Factory SFT format:
{
    "instruction": "...",
    "output": "..."
}
"""

import json
import os
import sys

def convert_gsm8k(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            record = {
                "instruction": item["question"],
                "output": item["answer"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Converted {len(data)} samples from {input_path} to {output_path}")

if __name__ == "__main__":
    import sys
    # Get the base directory (llama_factory directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.dirname(base_dir)  # Data-Whisperer root
    
    # Convert train set
    convert_gsm8k(
        os.path.join(project_dir, "data", "gsm8k", "train.json"),
        os.path.join(base_dir, "data", "gsm8k_train.json")
    )
    # Convert test set
    convert_gsm8k(
        os.path.join(project_dir, "data", "gsm8k", "test.json"),
        os.path.join(base_dir, "data", "gsm8k_test.json")
    )
