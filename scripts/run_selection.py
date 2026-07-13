"""
Data selection script for experiments.
Supports three methods: datawhisperer, random, heuristic.
Outputs a subset.json file with the selected samples.
"""
import json
import os
import sys
import random
import re
from collections import Counter
from tqdm import tqdm


def select_datawhisperer(full_data, keep_ratio, output_dir, repeat, **kwargs):
    """
    Run Data Whisperer selection and extract top-k% samples.
    
    Data Whisperer is run via pruning/pruning.py (requires GPU).
    This function handles the post-processing: extract top-k from the scored file.
    """
    # 优先使用 dw_scored_file 参数（共享评分文件路径）
    scored_file = kwargs.get('dw_scored_file', None)
    if scored_file is None:
        scored_file = os.path.join(output_dir, "dat_whisperer.json")
    
    if not os.path.exists(scored_file):
        print(f"Error: Data Whisperer output not found at {scored_file}")
        return None
    
    print(f"Loading Data Whisperer scored results from {scored_file}...")
    with open(scored_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    keep = max(1, int(total * keep_ratio))
    subset = data[:keep]
    
    subset_file = os.path.join(output_dir, "subset.json")
    with open(subset_file, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    
    print(f"Data Whisperer: Selected {keep}/{total} samples (top {keep_ratio:.0%}) -> {subset_file}")
    return subset_file


def select_random(full_data, keep_ratio, output_dir, repeat, **kwargs):
    """Random selection: randomly sample keep_ratio of data."""
    total = len(full_data)
    keep = max(1, int(total * keep_ratio))
    
    random.seed(42 + repeat)
    indices = random.sample(range(total), keep)
    subset = [dict(full_data[i], score=random.random()) for i in indices]
    
    subset_file = os.path.join(output_dir, "subset.json")
    with open(subset_file, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    
    print(f"Random: Selected {keep}/{total} samples (seed={42 + repeat}) -> {subset_file}")
    return subset_file


def _get_sample_length(sample):
    """Get the total text length of a sample."""
    text = ""
    for key, value in sample.items():
        if isinstance(value, str):
            text += value + " "
    return len(text)


def _get_answer_length(sample):
    """Get the answer/output length of a sample."""
    for key in ['output', 'answer', 'summary']:
        if key in sample and isinstance(sample[key], str):
            return len(sample[key])
    return 0


def _quality_score(sample):
    """Compute a rule-based quality score for a sample."""
    score = 1.0
    text = ''
    for key in ['answer', 'question', 'output', 'instruction', 'dialogue', 'summary']:
        if key in sample and isinstance(sample[key], str):
            text += sample[key] + ' '
    
    if not text.strip():
        return 0.0
    
    text_len = len(text)
    if text_len < 10:
        score *= 0.3
    elif text_len < 50:
        score *= 0.7
    elif text_len > 2000:
        score *= 0.6
    elif text_len > 1000:
        score *= 0.8
    
    words = text.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            score *= 0.5
        elif unique_ratio < 0.5:
            score *= 0.8
    
    return score


def _dedup_samples(dataset, similarity_threshold=0.8):
    """
    Simple deduplication based on text similarity.
    Uses Jaccard similarity on word sets.
    Returns indices of samples to keep.
    """
    keep_indices = []
    
    # Pre-compute word sets for all samples
    word_sets = []
    for sample in dataset:
        text = ""
        for key, value in sample.items():
            if isinstance(value, str):
                text += value + " "
        word_sets.append(set(text.lower().split()))
    
    # Greedy deduplication
    for i, words_i in enumerate(word_sets):
        is_duplicate = False
        for j in keep_indices:
            words_j = word_sets[j]
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union > 0 and intersection / union > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)
    
    return keep_indices


HEURISTIC_SCORERS = {
    "length": _get_sample_length,
    "answer_length": _get_answer_length,
    "quality_score": _quality_score,
}


def select_heuristic(full_data, keep_ratio, output_dir, repeat, **kwargs):
    """
    Heuristic selection: support multiple strategies.
    
    Strategies:
    - length: Select samples based on input length
    - answer_length: Select samples based on answer/output length
    - quality_score: Select samples based on a rule-based quality score
    - dedup: Select samples based on deduplication (remove near-duplicates)
    """
    heuristic = kwargs.get('heuristic', 'quality_score')
    keep_long = kwargs.get('keep_long', 'true').lower() == 'true'
    similarity_threshold = float(kwargs.get('similarity_threshold', 0.8))
    
    total = len(full_data)
    
    if heuristic == 'dedup':
        # Deduplication: keep unique samples
        keep_indices = _dedup_samples(full_data, similarity_threshold)
        subset = [dict(full_data[i], score=1.0) for i in keep_indices]
        print(f"Deduplication completed. Kept {len(keep_indices)}/{total} unique samples.")
    else:
        # Score-based selection
        keep = max(1, int(total * keep_ratio))
        scorer = HEURISTIC_SCORERS.get(heuristic)
        if scorer is None:
            raise ValueError(f"Unknown heuristic: {heuristic}. Supported: {list(HEURISTIC_SCORERS.keys()) + ['dedup']}")
        
        print(f"Scoring {total} samples with heuristic '{heuristic}'...")
        scored = []
        for item in tqdm(full_data, desc="Scoring", unit="sample"):
            scored.append((scorer(item), item))
        
        scored.sort(key=lambda x: x[0], reverse=keep_long)
        
        subset = []
        for i in range(keep):
            item = dict(scored[i][1], score=scored[i][0])
            subset.append(item)
        
        print(f"Heuristic ({heuristic}): Selected {keep}/{total} samples.")
    
    subset_file = os.path.join(output_dir, "subset.json")
    with open(subset_file, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {subset_file}")
    return subset_file


SELECTION_METHODS = {
    "datawhisperer": select_datawhisperer,
    "random": select_random,
    "heuristic": select_heuristic,
}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data selection for experiments")
    parser.add_argument("--method", type=str, required=True,
                        choices=list(SELECTION_METHODS.keys()),
                        help="Selection method")
    parser.add_argument("--full_data", type=str, required=True,
                        help="Path to full dataset JSON")
    parser.add_argument("--keep_ratio", type=float, required=True,
                        help="Proportion of data to keep (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat index (for different random seeds)")
    parser.add_argument("--heuristic", type=str, default="quality_score",
                        choices=["length", "answer_length", "quality_score", "dedup"],
                        help="Heuristic strategy (only used when method=heuristic)")
    parser.add_argument("--keep_long", type=str, default="true",
                        choices=["true", "false"],
                        help="For heuristic: keep long samples (true) or short samples (false)")
    parser.add_argument("--similarity_threshold", type=float, default=0.8,
                        help="For dedup heuristic: Jaccard similarity threshold (0.0-1.0)")
    parser.add_argument("--dw_scored_file", type=str, default=None,
                        help="Path to Data Whisperer scored results file (only used when method=datawhisperer)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.full_data, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    selector = SELECTION_METHODS[args.method]
    subset_file = selector(
        full_data,
        keep_ratio=args.keep_ratio,
        output_dir=args.output_dir,
        repeat=args.repeat,
        heuristic=args.heuristic,
        keep_long=args.keep_long,
        similarity_threshold=args.similarity_threshold,
        dw_scored_file=args.dw_scored_file,
    )

    if subset_file:
        print(f"Selection complete. Subset saved to {subset_file}")
    else:
        print(f"Selection failed for method={args.method}")
        sys.exit(1)


if __name__ == "__main__":
    main()


