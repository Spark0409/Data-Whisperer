"""
Phase 2: Fine-tune and Evaluate Only.
Run this in the Llama-Factory environment.

Reads pre-selected JSONL files from /root/experiments/selected_data/
and runs training + evaluation for each.

Usage:
    python scripts/run_finetune.py [--dry_run] [--resume <exp_id>]
"""

import json
import os
import sys
import time
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# (method, heuristic_strategy, keep_ratio, finetune_seeds)
# finetune_seeds: 微调阶段的随机种子（微调本身有随机性，需要重复3次取平均）
EXPERIMENTS = [
    ("random", None, 0.05, [42, 123, 456]),
    ("random", None, 0.10, [42, 123, 456]),
    ("random", None, 0.20, [42, 123, 456]),
    ("heuristic", "quality_score", 0.05, [42, 123, 456]),
    ("heuristic", "quality_score", 0.10, [42, 123, 456]),
    ("heuristic", "quality_score", 0.20, [42, 123, 456]),
    # DataWhisperer: 筛选只用 seed=42（确定性方法），微调用 3 个 seed 取平均
    ("datawhisperer", None, 0.05, [42, 123, 456]),
    ("datawhisperer", None, 0.10, [42, 123, 456]),
    ("datawhisperer", None, 0.20, [42, 123, 456]),
]

MODEL_PATH = "/mnt/models/Qwen2.5-3B-Instruct"
TRAIN_CONFIG = os.path.join(BASE_DIR, "configs", "train_qwen_gsm8k.yaml")
EVAL_SCRIPT = os.path.join(BASE_DIR, "scripts", "eval_gsm8k.py")

LLAMA_DATA_DIR = os.path.join(BASE_DIR, "data")
LLAMA_TRAIN_DATA = os.path.join(LLAMA_DATA_DIR, "gsm8k_train.json")
LLAMA_TEST_DATA = os.path.join(LLAMA_DATA_DIR, "gsm8k_test.json")
DATASET_INFO = os.path.join(LLAMA_DATA_DIR, "dataset_info.json")

OUTPUT_ROOT = "/root"
SELECTED_DIR = os.path.join(OUTPUT_ROOT, "experiments", "selected_data")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "experiments")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Helper Functions
# ============================================================

def run_command(cmd: List[str], desc: str, cwd: str = None) -> bool:
    print(f"\n{'='*60}", flush=True)
    print(f"[{desc}]", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        result = subprocess.run(cmd, cwd=cwd or BASE_DIR, timeout=7200)
        if result.returncode != 0:
            print(f"  ERROR: command exited with code {result.returncode}", flush=True)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (exceeded 2 hours)", flush=True)
        return False
    except Exception as e:
        print(f"  EXCEPTION: {e}", flush=True)
        return False


def copy_to_llama_factory():
    """Copy data files to Llama-Factory."""
    llama_factory_data = "/LLaMA-Factory/data"
    cmds = [
        f"cp {LLAMA_TRAIN_DATA} {llama_factory_data}/gsm8k_train.json",
        f"cp {LLAMA_TEST_DATA} {llama_factory_data}/gsm8k_test.json",
        f"cp {DATASET_INFO} {llama_factory_data}/dataset_info.json",
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: copy failed: {result.stderr.strip()}")
    return True


def run_training(seed: int) -> bool:
    """Run LoRA fine-tuning with a specific seed."""
    # Dynamically set seed in YAML config
    import yaml
    with open(TRAIN_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    config['seed'] = seed
    # Also set output_dir to include seed to avoid conflicts
    config['output_dir'] = f"/root/output/qwen2.5-3b-gsm8k-lora-seed{seed}"
    with open(TRAIN_CONFIG, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return run_command(
        ["llamafactory-cli", "train", TRAIN_CONFIG],
        f"LoRA fine-tuning (seed={seed})",
        cwd="/LLaMA-Factory"
    )


def merge_lora(seed: int) -> bool:
    """Merge LoRA weights."""
    adapter_path = os.path.join(OUTPUT_ROOT, "output", f"qwen2.5-3b-gsm8k-lora-seed{seed}")
    merged_dir = os.path.join(OUTPUT_ROOT, "output", f"qwen2.5-3b-gsm8k-lora-merged-seed{seed}")
    return run_command(
        [
            "llamafactory-cli", "export",
            "--model_name_or_path", MODEL_PATH,
            "--adapter_name_or_path", adapter_path,
            "--template", "qwen",
            "--finetuning_type", "lora",
            "--export_dir", merged_dir,
            "--export_device", "auto",
        ],
        f"Merge LoRA weights (seed={seed})"
    )


def evaluate_model(output_file: str, seed: int) -> Dict:
    """Evaluate on GSM8K test set."""
    merged_model_path = os.path.join(OUTPUT_ROOT, "output", f"qwen2.5-3b-gsm8k-lora-merged-seed{seed}")
    result = subprocess.run(
        [
            sys.executable, EVAL_SCRIPT,
            "--model_name_or_path", merged_model_path,
            "--batch_size", "8",
            "--output_file", output_file,
        ],
        cwd=BASE_DIR,
    )

    accuracy = 0.0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
            accuracy = data.get("accuracy", accuracy)

    print(f"  Evaluation accuracy: {accuracy:.4f}", flush=True)
    return {"accuracy": accuracy, "output_file": output_file}


def run_single_experiment(exp_id: str, selected_path: str) -> Dict:
    """Run training + evaluation for one experiment."""
    # Extract seed from exp_id (e.g., "random_none_r0.05_s42" -> 42)
    seed = int(exp_id.split("_s")[-1])

    print(f"\n{'#'*70}")
    print(f"# Experiment: {exp_id}")
    print(f"# Selected data: {selected_path}")
    print(f"# Finetune seed: {seed}")
    print(f"{'#'*70}")

    start_time = time.time()
    result = {
        "exp_id": exp_id,
        "status": "failed",
        "accuracy": None,
        "error": None,
        "duration_seconds": None,
    }

    try:
        # Step 1: Copy selected data to training data path
        if not os.path.exists(selected_path):
            result["error"] = f"Selected data not found: {selected_path}"
            return result

        os.makedirs(LLAMA_DATA_DIR, exist_ok=True)
        subprocess.run(f"cp {selected_path} {LLAMA_TRAIN_DATA}", shell=True, check=True)
        print(f"  Copied selected data to {LLAMA_TRAIN_DATA}")

        # Step 2: Copy to Llama-Factory
        copy_to_llama_factory()

        # Step 3: Train (with seed from exp_id)
        if not run_training(seed):
            result["error"] = "Training failed"
            return result

        # Step 4: Merge LoRA
        if not merge_lora(seed):
            result["error"] = "LoRA merge failed"
            return result

        # Step 5: Evaluate
        eval_output = os.path.join(RESULTS_DIR, f"{exp_id}_eval.json")
        eval_result = evaluate_model(eval_output, seed)

        result["status"] = "completed"
        result["accuracy"] = eval_result["accuracy"]

    except Exception as e:
        result["error"] = str(e)
        print(f"  EXCEPTION: {e}")

    duration = time.time() - start_time
    result["duration_seconds"] = duration
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f}min)")
    print(f"  Status: {result['status']}")
    if result["accuracy"] is not None:
        print(f"  Accuracy: {result['accuracy']:.4f}")

    return result


def print_summary_table(all_results: List[Dict]):
    """Print a summary table of all experiment results."""
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        # Parse exp_id to extract method and ratio
        parts = r["exp_id"].split("_")
        method = parts[0]
        heuristic = parts[1] if parts[1] != "none" else None
        ratio = float(parts[2].lstrip("r"))
        key = (method, heuristic, ratio)
        groups[key].append(r)

    print(f"\n{'Method':<20} {'Ratio':<8} {'Run1':<10} {'Run2':<10} {'Run3':<10} {'Mean':<10} {'Std':<10}  (Accuracy)")
    print(f"{'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    results_summary = []
    for (method, heuristic, ratio), runs in sorted(groups.items()):
        accuracies = [r["accuracy"] for r in runs if r["accuracy"] is not None]

        method_str = f"{method}"
        if heuristic:
            method_str += f"({heuristic})"

        if len(accuracies) == 3:
            mean_acc = sum(accuracies) / 3
            std_acc = (sum((a - mean_acc) ** 2 for a in accuracies) / 3) ** 0.5
            print(f"{method_str:<20} {ratio:<8.2f} {accuracies[0]:<10.4f} {accuracies[1]:<10.4f} {accuracies[2]:<10.4f} {mean_acc:<10.4f} {std_acc:<10.4f}")
            results_summary.append({
                "method": method,
                "heuristic": heuristic,
                "keep_ratio": ratio,
                "accuracies": accuracies,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
            })
        else:
            acc_strs = [f"{a:.4f}" if a is not None else "FAIL" for a in accuracies]
            print(f"{method_str:<20} {ratio:<8.2f} {' '.join(f'{s:<10}' for s in acc_strs)} {'N/A':<10} {'N/A':<10}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")

    # Print LaTeX table
    print(f"\n\nLaTeX Table:")
    print(f"{'='*70}")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Method & Ratio & Run 1 & Run 2 & Run 3 & Mean $\pm$ Std \\")
    print(r"\midrule")
    for s in results_summary:
        method_str = s["method"]
        if s["heuristic"]:
            method_str += f" ({s['heuristic']})"
        print(f"{method_str} & {s['keep_ratio']:.2f} & {s['accuracies'][0]:.4f} & {s['accuracies'][1]:.4f} & {s['accuracies'][2]:.4f} & {s['mean_accuracy']:.4f} $\\pm$ {s['std_accuracy']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{GSM8K accuracy comparison across data selection strategies}")
    print(r"\label{tab:gsm8k_selection}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tune and evaluate")
    parser.add_argument("--dry_run", action="store_true", help="Print plan without running")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a specific experiment ID")
    args = parser.parse_args()

    # Build list of all experiments
    all_experiments = []
    for method, heuristic, ratio, seeds in EXPERIMENTS:
        for seed in seeds:
            exp_id = f"{method}_{'quality' if heuristic else 'none'}_r{ratio:.2f}_s{seed}"
            # DataWhisperer 筛选是确定性的，只用 seed=42 的数据
            # 微调 seed 只用于命名（区分重复实验），筛选数据始终用 seed=42
            if method == "datawhisperer":
                selected_path = os.path.join(SELECTED_DIR, f"{method}_{'quality' if heuristic else 'none'}_r{ratio:.2f}_s42.jsonl")
            else:
                selected_path = os.path.join(SELECTED_DIR, f"{exp_id}.jsonl")
            all_experiments.append((exp_id, selected_path))

    total = len(all_experiments)
    print(f"Fine-tune Plan:")
    print(f"  Total experiments: {total}")
    print(f"  Selected data dir: {SELECTED_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")

    if args.dry_run:
        print("\nDry run mode. No experiments will be executed.")
        return

    # Check which selected data files exist
    missing = []
    for exp_id, path in all_experiments:
        if not os.path.exists(path):
            missing.append(exp_id)
    if missing:
        print(f"\nWARNING: {len(missing)} selected data files not found:")
        for m in missing:
            print(f"  - {m}")
        print("Run run_selection.py first in the Data-Whisperer environment.\n")

    # Run experiments
    all_results = []
    exp_counter = 0
    resume_found = args.resume is None

    for exp_id, selected_path in all_experiments:
        exp_counter += 1

        if args.resume and not resume_found:
            if exp_id == args.resume:
                resume_found = True
            else:
                print(f"\nSkipping {exp_id} (resuming from {args.resume})")
                continue

        if not os.path.exists(selected_path):
            print(f"\nSkipping {exp_id} (selected data not found: {selected_path})")
            continue

        print(f"\n\n{'='*70}")
        print(f"Experiment {exp_counter}/{total}")

        result = run_single_experiment(exp_id, selected_path)
        all_results.append(result)

        # Save intermediate results
        interim_path = os.path.join(RESULTS_DIR, "interim_results.json")
        with open(interim_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print_summary_table(all_results)

    # Final save
    final_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to: {final_path}")


if __name__ == "__main__":
    main()
