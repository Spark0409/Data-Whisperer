"""
Phase 1: Data Selection Only.
Run this in the Data-Whisperer environment.

Generates selected JSONL files for each (method, ratio, seed) combination.
Output: /root/experiments/selected_data/{exp_id}.jsonl

Usage:
    python scripts/run_selection.py [--dry_run]
"""

import json
import os
import sys
import subprocess
import argparse
from typing import List, Dict, Tuple


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

EXPERIMENTS = [
    # (method, heuristic_strategy, keep_ratio, seeds)
    # Random (需要不同 seed 产生随机性)
    ("random", None, 0.05, [42, 123, 456]),
    ("random", None, 0.10, [42, 123, 456]),
    ("random", None, 0.20, [42, 123, 456]),
    # Heuristic (quality_score) (需要不同 seed 打乱同分数据)
    ("heuristic", "quality_score", 0.05, [42, 123, 456]),
    ("heuristic", "quality_score", 0.10, [42, 123, 456]),
    ("heuristic", "quality_score", 0.20, [42, 123, 456]),
    # DataWhisperer (确定性方法，只需运行一次 seed=42，再从排序结果中取不同比例)
    ("datawhisperer", None, 0.05, [42]),
    ("datawhisperer", None, 0.10, [42]),
    ("datawhisperer", None, 0.20, [42]),
]

MODEL_PATH = "/mnt/models/Qwen2.5-3B-Instruct"
CONVERT_SCRIPT = os.path.join(BASE_DIR, "scripts", "convert_gsm8k.py")
SELECT_SCRIPT = os.path.join(BASE_DIR, "scripts", "select_data.py")
PRUNING_SCRIPT = os.path.join(PROJECT_DIR, "pruning", "pruning.py")

ORIGINAL_TRAIN_DATA = os.path.join(PROJECT_DIR, "data", "gsm8k", "train.json")
LLAMA_DATA_DIR = os.path.join(BASE_DIR, "data")
LLAMA_TRAIN_DATA = os.path.join(LLAMA_DATA_DIR, "gsm8k_train.json")

OUTPUT_ROOT = "/root"
SELECTED_DIR = os.path.join(OUTPUT_ROOT, "experiments", "selected_data")
os.makedirs(SELECTED_DIR, exist_ok=True)

# DataWhisperer pruning 需要特定 conda 环境（包含 qwen_vl_utils 等依赖）
# 如果为空，则使用当前 python 解释器
DATAWHISPERER_PYTHON = "/root/miniconda3/envs/dw/bin/python"


# ============================================================
# Helper Functions
# ============================================================

def run_command(cmd: List[str], desc: str, cwd: str = None, env: dict = None) -> bool:
    print(f"\n{'='*60}", flush=True)
    print(f"[{desc}]", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd or BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(f"  {line}", end="", flush=True)
        process.wait()
        if process.returncode != 0:
            print(f"  ERROR: command exited with code {process.returncode}", flush=True)
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
    return run_command(["python", CONVERT_SCRIPT], "Convert GSM8K data")


def select_data_random(keep_ratio: float, seed: int, output_path: str) -> bool:
    return run_command(
        [
            "python", SELECT_SCRIPT,
            "--input_path", LLAMA_TRAIN_DATA,
            "--output_path", output_path,
            "--method", "random",
            "--keep_ratio", str(keep_ratio),
            "--seed", str(seed),
        ],
        f"Random selection (ratio={keep_ratio}, seed={seed})"
    )


def select_data_heuristic(keep_ratio: float, heuristic: str, seed: int, output_path: str) -> bool:
    return run_command(
        [
            "python", SELECT_SCRIPT,
            "--input_path", LLAMA_TRAIN_DATA,
            "--output_path", output_path,
            "--method", "heuristic",
            "--heuristic", heuristic,
            "--keep_ratio", str(keep_ratio),
            "--keep_long",
            "--seed", str(seed),
        ],
        f"Heuristic selection (strategy={heuristic}, ratio={keep_ratio}, seed={seed})"
    )


def select_data_datawhisperer(keep_ratio: float, seed: int, output_path: str) -> bool:
    dw_output_dir = os.path.join(LLAMA_DATA_DIR, f"datawhisperer_output_seed{seed}")
    os.makedirs(dw_output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.path.join(PROJECT_DIR, 'pruning')}:{env.get('PYTHONPATH', '')}"

    python_path = DATAWHISPERER_PYTHON or sys.executable
    cmd = [
        python_path, PRUNING_SCRIPT,
        "--model_path", MODEL_PATH,
        "--model_type", "qwen",
        "--model_name", "Qwen2.5-3B-Instruct",
        "--data_path", ORIGINAL_TRAIN_DATA,
        "--dataset", "gsm8k",
        "--method", "datawhisperer",
        "--parallel_batches", "5",
        "--batch_train", "10",
        "--batch_test", "5",
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
    with open(output_path, 'w') as f:
        for item in selected:
            record = {
                "instruction": item["question"],
                "output": item["answer"],
                "score": item["score"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data selection only")
    parser.add_argument("--dry_run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    total = sum(len(seeds) for _, _, _, seeds in EXPERIMENTS)
    print(f"Data Selection Plan:")
    print(f"  Total selections: {total}")
    print(f"  Output dir: {SELECTED_DIR}")

    if args.dry_run:
        print("\nDry run mode. No selections will be executed.")
        return

    # Step 1: Convert data once
    print("\n" + "="*60)
    print("Step 1: Convert GSM8K data")
    print("="*60)
    if not convert_data():
        print("ERROR: Data conversion failed. Aborting.")
        sys.exit(1)

    # Step 2: Run all selections
    for method, heuristic, ratio, seeds in EXPERIMENTS:
        for seed in seeds:
            exp_id = f"{method}_{'quality' if heuristic else 'none'}_r{ratio:.2f}_s{seed}"
            output_path = os.path.join(SELECTED_DIR, f"{exp_id}.jsonl")

            if os.path.exists(output_path):
                print(f"\nSkipping {exp_id} (already exists: {output_path})")
                continue

            print(f"\n{'#'*70}")
            print(f"# Selection: {exp_id}")
            print(f"{'#'*70}")

            if method == "random":
                success = select_data_random(ratio, seed, output_path)
            elif method == "heuristic":
                success = select_data_heuristic(ratio, heuristic, seed, output_path)
            elif method == "datawhisperer":
                success = select_data_datawhisperer(ratio, seed, output_path)
            else:
                print(f"  Unknown method: {method}")
                success = False

            if success:
                print(f"  SUCCESS: {output_path}")
            else:
                print(f"  FAILED: {exp_id}")

    print(f"\n{'='*60}")
    print(f"All selections complete. Output in: {SELECTED_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
