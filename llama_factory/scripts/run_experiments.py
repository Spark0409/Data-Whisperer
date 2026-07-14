"""
GSM8K Data Selection Experiment Runner.

Runs experiments with different data selection strategies and keep ratios,
each repeated 3 times with different random seeds.

Strategies: random, heuristic, datawhisperer
Ratios: 0.05, 0.10, 0.20
Repeats: 3 per (strategy, ratio) combination

Usage:
    python scripts/run_experiments.py [--dry_run]
"""

import json
import os
import sys
import time
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Tuple


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

EXPERIMENTS = [
    # (method, heuristic_strategy, keep_ratio, seeds)
    # Random
    ("random", None, 0.05, [42, 123, 456]),
    ("random", None, 0.10, [42, 123, 456]),
    ("random", None, 0.20, [42, 123, 456]),
    # Heuristic (quality_score)
    ("heuristic", "quality_score", 0.05, [42, 123, 456]),
    ("heuristic", "quality_score", 0.10, [42, 123, 456]),
    ("heuristic", "quality_score", 0.20, [42, 123, 456]),
    # DataWhisperer
    ("datawhisperer", None, 0.05, [42, 123, 456]),
    ("datawhisperer", None, 0.10, [42, 123, 456]),
    ("datawhisperer", None, 0.20, [42, 123, 456]),
]

# Model and training config (fixed across all experiments)
MODEL_PATH = "/mnt/models/Qwen2.5-3B-Instruct"
TRAIN_CONFIG = os.path.join(BASE_DIR, "configs", "train_qwen_gsm8k.yaml")
CONVERT_SCRIPT = os.path.join(BASE_DIR, "scripts", "convert_gsm8k.py")
SELECT_SCRIPT = os.path.join(BASE_DIR, "scripts", "select_data.py")
PRUNING_SCRIPT = os.path.join(PROJECT_DIR, "pruning", "pruning.py")
EVAL_SCRIPT = os.path.join(BASE_DIR, "scripts", "eval_gsm8k.py")

# Data paths
ORIGINAL_TRAIN_DATA = os.path.join(PROJECT_DIR, "data", "gsm8k", "train.json")
ORIGINAL_TEST_DATA = os.path.join(PROJECT_DIR, "data", "gsm8k", "test.json")
LLAMA_DATA_DIR = os.path.join(BASE_DIR, "data")
LLAMA_TRAIN_DATA = os.path.join(LLAMA_DATA_DIR, "gsm8k_train.json")
LLAMA_TEST_DATA = os.path.join(LLAMA_DATA_DIR, "gsm8k_test.json")
DATASET_INFO = os.path.join(LLAMA_DATA_DIR, "dataset_info.json")

# Output (use /root to avoid filling up the system partition)
OUTPUT_ROOT = "/root"
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "experiments")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Helper Functions
# ============================================================

def run_command(cmd: List[str], desc: str, cwd: str = None, env: dict = None) -> bool:
    """Run a command and return success status.
    
    Output is streamed in real-time to both stdout and the log file
    (via the parent process's redirected stdout).
    """
    print(f"\n{'='*60}", flush=True)
    print(f"[{desc}]", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or BASE_DIR,
            env=env,
            # Don't capture output — let it flow directly to stdout/stderr
            # so it appears in real-time in the log file when running via nohup
            timeout=7200,  # 2 hours max per experiment
        )
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


def convert_data():
    """Step 1: Convert original GSM8K data to Llama-Factory JSONL format."""
    return run_command(
        ["python", CONVERT_SCRIPT],
        "Convert GSM8K data"
    )


def select_data_random(keep_ratio: float, seed: int) -> bool:
    """Step 2a: Random data selection."""
    return run_command(
        [
            "python", SELECT_SCRIPT,
            "--input_path", LLAMA_TRAIN_DATA,
            "--output_path", LLAMA_TRAIN_DATA.replace(".json", "_selected.json"),
            "--method", "random",
            "--keep_ratio", str(keep_ratio),
            "--seed", str(seed),
        ],
        f"Random selection (ratio={keep_ratio}, seed={seed})"
    )


def select_data_heuristic(keep_ratio: float, heuristic: str, seed: int) -> bool:
    """Step 2b: Heuristic data selection."""
    return run_command(
        [
            "python", SELECT_SCRIPT,
            "--input_path", LLAMA_TRAIN_DATA,
            "--output_path", LLAMA_TRAIN_DATA.replace(".json", "_selected.json"),
            "--method", "heuristic",
            "--heuristic", heuristic,
            "--keep_ratio", str(keep_ratio),
            "--keep_long",
            "--seed", str(seed),
        ],
        f"Heuristic selection (strategy={heuristic}, ratio={keep_ratio}, seed={seed})"
    )


def select_data_datawhisperer(keep_ratio: float, seed: int) -> bool:
    """Step 2c: DataWhisperer data selection using existing pruning pipeline."""
    dw_output_dir = os.path.join(LLAMA_DATA_DIR, f"datawhisperer_output_seed{seed}")
    os.makedirs(dw_output_dir, exist_ok=True)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.path.join(PROJECT_DIR, 'pruning')}:{env.get('PYTHONPATH', '')}"
    
    # Run DataWhisperer pruning
    # Paper settings:
    #   nd (batch_train) = 10 (default)
    #   nq (batch_test)  = 5  (default)
    #   parallel_batches = 15 (task-specific for GSM8K)
    #   temperature = 0 (deterministic ICL)
    cmd = [
        sys.executable, PRUNING_SCRIPT,
        "--model_path", MODEL_PATH,
        "--model_type", "qwen",
        "--model_name", "Qwen2.5-3B-Instruct",
        "--data_path", ORIGINAL_TRAIN_DATA,
        "--dataset", "gsm8k",
        "--method", "datawhisperer",
        "--parallel_batches", "15",     # task-specific batch size for GSM8K
        "--batch_train", "10",          # nd = 10 (default)
        "--batch_test", "5",            # nq = 5 (default)
        "--max_token", "8192",
        "--k_folds", "2",
        "--metric", "exact_match",
        "--output_filtered_path", dw_output_dir,
        "--gpu_index", "0",
    ]
    
    success = run_command(cmd, f"DataWhisperer pruning (seed={seed})", cwd=PROJECT_DIR, env=env)
    if not success:
        return False
    
    # Load DataWhisperer output, select top-k, convert to JSONL
    dw_output = os.path.join(dw_output_dir, "dat_whisperer.json")
    if not os.path.exists(dw_output):
        print(f"  ERROR: DataWhisperer output not found: {dw_output}")
        return False
    
    with open(dw_output, 'r') as f:
        data = json.load(f)
    
    keep_count = max(1, int(len(data) * keep_ratio))
    selected = data[:keep_count]
    
    print(f"  DataWhisperer: selected {keep_count}/{len(data)} (ratio={keep_ratio})")
    print(f"  Score range: {selected[-1]['score']:.4f} ~ {selected[0]['score']:.4f}")
    
    # Convert to JSONL
    output_path = LLAMA_TRAIN_DATA.replace(".json", "_selected.json")
    with open(output_path, 'w') as f:
        for item in selected:
            record = {
                "instruction": item["question"],
                "output": item["answer"],
                "score": item["score"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"  Converted to Llama-Factory format: {output_path}")
    return True


def copy_to_llama_factory():
    """Step 3: Copy data files to Llama-Factory."""
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


def run_training() -> bool:
    """Step 4: Run LoRA fine-tuning."""
    return run_command(
        ["llamafactory-cli", "train", TRAIN_CONFIG],
        "LoRA fine-tuning",
        cwd="/LLaMA-Factory"
    )


def merge_lora() -> bool:
    """Step 5: Merge LoRA weights."""
    merged_dir = os.path.join(OUTPUT_ROOT, "output", "qwen2.5-3b-gsm8k-lora-merged")
    return run_command(
        [
            "llamafactory-cli", "export",
            "--model_name_or_path", MODEL_PATH,
            "--adapter_name_or_path", os.path.join(OUTPUT_ROOT, "output", "qwen2.5-3b-gsm8k-lora"),
            "--template", "qwen",
            "--finetuning_type", "lora",
            "--export_dir", merged_dir,
            "--export_device", "auto",
        ],
        "Merge LoRA weights"
    )


def evaluate_model(output_file: str) -> Dict:
    """Step 6: Evaluate on GSM8K test set. Returns accuracy dict."""
    merged_model_path = os.path.join(OUTPUT_ROOT, "output", "qwen2.5-3b-gsm8k-lora-merged")
    result = subprocess.run(
        [
            sys.executable, EVAL_SCRIPT,
            "--model_name_or_path", merged_model_path,
            "--batch_size", "8",
            "--output_file", output_file,
        ],
        cwd=BASE_DIR,
    )
    
    # Read accuracy from output file
    accuracy = 0.0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
            accuracy = data.get("accuracy", accuracy)
    
    print(f"  Evaluation accuracy: {accuracy:.4f}", flush=True)
    return {"accuracy": accuracy, "output_file": output_file}


def run_single_experiment(
    method: str,
    heuristic: str,
    keep_ratio: float,
    seed: int,
    exp_id: str,
) -> Dict:
    """
    Run a single experiment end-to-end.
    
    Returns:
        Dict with experiment results
    """
    print(f"\n{'#'*70}")
    print(f"# Experiment: {exp_id}")
    print(f"# Method={method}, Heuristic={heuristic}, Ratio={keep_ratio}, Seed={seed}")
    print(f"{'#'*70}")
    
    start_time = time.time()
    result = {
        "exp_id": exp_id,
        "method": method,
        "heuristic": heuristic,
        "keep_ratio": keep_ratio,
        "seed": seed,
        "status": "failed",
        "accuracy": None,
        "error": None,
        "duration_seconds": None,
    }
    
    try:
        # Step 1: Convert data
        if not convert_data():
            result["error"] = "Data conversion failed"
            return result
        
        # Step 2: Data selection
        if method == "random":
            success = select_data_random(keep_ratio, seed)
        elif method == "heuristic":
            success = select_data_heuristic(keep_ratio, heuristic, seed)
        elif method == "datawhisperer":
            success = select_data_datawhisperer(keep_ratio, seed)
        else:
            result["error"] = f"Unknown method: {method}"
            return result
        
        if not success:
            result["error"] = f"Data selection failed ({method})"
            return result
        
        # Replace training data with selected subset
        selected_file = LLAMA_TRAIN_DATA.replace(".json", "_selected.json")
        if os.path.exists(selected_file):
            os.replace(selected_file, LLAMA_TRAIN_DATA)
            print(f"  Replaced training data with selected subset")
        
        # Step 3: Copy to Llama-Factory
        copy_to_llama_factory()
        
        # Step 4: Train
        if not run_training():
            result["error"] = "Training failed"
            return result
        
        # Step 5: Merge LoRA
        if not merge_lora():
            result["error"] = "LoRA merge failed"
            return result
        
        # Step 6: Evaluate
        eval_output = os.path.join(RESULTS_DIR, f"{exp_id}_eval.json")
        eval_result = evaluate_model(eval_output)
        
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
    
    # Group by (method, ratio)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        key = (r["method"], r.get("heuristic"), r["keep_ratio"])
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
    parser = argparse.ArgumentParser(description="Run GSM8K data selection experiments")
    parser.add_argument("--dry_run", action="store_true", help="Print experiment plan without running")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a specific experiment ID")
    args = parser.parse_args()
    
    # Print experiment plan
    total_experiments = sum(len(seeds) for _, _, _, seeds in EXPERIMENTS)
    print(f"Experiment Plan:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Strategies: random, heuristic(quality_score), datawhisperer")
    print(f"  Ratios: 0.05, 0.10, 0.20")
    print(f"  Repeats: 3 per (strategy, ratio)")
    print(f"  Results dir: {RESULTS_DIR}")
    
    if args.dry_run:
        print("\nDry run mode. No experiments will be executed.")
        return
    
    # Run experiments
    all_results = []
    exp_counter = 0
    resume_found = args.resume is None  # If no resume, start from beginning
    
    for method, heuristic, ratio, seeds in EXPERIMENTS:
        for seed in seeds:
            exp_counter += 1
            exp_id = f"{method}_{'quality' if heuristic else 'none'}_r{ratio:.2f}_s{seed}"
            
            # Resume logic
            if args.resume and not resume_found:
                if exp_id == args.resume:
                    resume_found = True
                else:
                    print(f"\nSkipping {exp_id} (resuming from {args.resume})")
                    continue
            
            print(f"\n\n{'='*70}")
            print(f"Experiment {exp_counter}/{total_experiments}")
            
            result = run_single_experiment(method, heuristic, ratio, seed, exp_id)
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
