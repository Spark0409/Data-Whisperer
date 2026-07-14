"""
Data Selection Strategies for Llama-Factory Fine-tuning Pipeline.

Supports:
- Random: Randomly select a specified proportion of data
- Heuristic: Select data based on heuristic criteria (length, answer_length, quality_score, dedup)

Usage:
    python scripts/select_data.py \\
        --input_path ../data/gsm8k_train.json \\
        --output_path ../data/gsm8k_train_selected.json \\
        --method random \\
        --keep_ratio 0.5

    python scripts/select_data.py \\
        --input_path ../data/gsm8k_train.json \\
        --output_path ../data/gsm8k_train_selected.json \\
        --method heuristic \\
        --heuristic length \\
        --keep_long true \\
        --keep_ratio 0.5
"""

import json
import os
import re
import random
import argparse
from collections import Counter
from typing import List, Dict, Any, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file (one JSON object per line)."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    """Save data as JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {path}")


# ============================================================
# Random Selection
# ============================================================

def random_select(
    data: List[Dict[str, Any]],
    keep_ratio: float = 0.5,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Randomly select a proportion of data.
    
    Args:
        data: Input dataset
        keep_ratio: Proportion of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Selected subset with scores
    """
    total = len(data)
    keep_count = max(1, int(total * keep_ratio))
    
    random.seed(seed)
    indices = list(range(total))
    random.shuffle(indices)
    keep_indices = indices[:keep_count]
    
    selected = []
    for i in keep_indices:
        item = dict(data[i])
        item["score"] = random.random()
        selected.append(item)
    
    print(f"[Random] Selected {keep_count}/{total} samples (keep_ratio={keep_ratio})")
    return selected


# ============================================================
# Heuristic Selection
# ============================================================

def _get_text(sample: Dict[str, Any]) -> str:
    """Get all text content from a sample."""
    text = ""
    for key, value in sample.items():
        if isinstance(value, str):
            text += value + " "
    return text


def _get_sample_length(sample: Dict[str, Any]) -> int:
    """Get total text length of a sample."""
    return len(_get_text(sample))


def _get_answer_length(sample: Dict[str, Any]) -> int:
    """Get answer/output length of a sample."""
    for key in ['output', 'answer']:
        if key in sample and isinstance(sample[key], str):
            return len(sample[key])
    return 0


def _get_quality_score(sample: Dict[str, Any]) -> float:
    """
    Compute a rule-based quality score for a sample.
    Higher scores = higher quality.
    
    Criteria:
    1. Length score: penalize too short or too long samples
    2. Keyword diversity: penalize repetitive vocabulary
    3. Special character penalty: penalize excessive special chars
    4. Repetition penalty: penalize repeated n-grams
    """
    score = 1.0
    text = _get_text(sample)
    
    if not text.strip():
        return 0.0
    
    text_len = len(text)
    
    # 1. Length score
    if text_len < 10:
        score *= 0.3
    elif text_len < 50:
        score *= 0.7
    elif text_len > 2000:
        score *= 0.6
    elif text_len > 1000:
        score *= 0.8
    
    # 2. Keyword diversity
    words = text.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            score *= 0.5
        elif unique_ratio < 0.5:
            score *= 0.8
    
    # 3. Special character penalty
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    if words and special_chars / len(words) > 0.5:
        score *= 0.7
    
    # 4. Repetition detection (3-gram)
    words_list = text.lower().split()
    if len(words_list) >= 10:
        tri_grams = [' '.join(words_list[i:i+3]) for i in range(len(words_list)-2)]
        if tri_grams:
            most_common_count = Counter(tri_grams).most_common(1)[0][1]
            if most_common_count > len(tri_grams) * 0.3:
                score *= 0.6
    
    return score


def _dedup_samples(data: List[Dict[str, Any]], threshold: float = 0.8) -> List[int]:
    """
    Deduplicate samples based on Jaccard similarity of word sets.
    
    Args:
        data: Input dataset
        threshold: Jaccard similarity threshold (0.0-1.0)
    
    Returns:
        Indices of unique samples to keep
    """
    # Pre-compute word sets
    word_sets = []
    for sample in data:
        word_sets.append(set(_get_text(sample).lower().split()))
    
    # Greedy deduplication
    keep_indices = []
    for i, words_i in enumerate(word_sets):
        is_duplicate = False
        for j in keep_indices:
            words_j = word_sets[j]
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union > 0 and intersection / union > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)
    
    return keep_indices


def heuristic_select(
    data: List[Dict[str, Any]],
    heuristic: str = 'length',
    keep_long: bool = True,
    keep_ratio: float = 0.5,
    similarity_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Select data based on heuristic criteria.
    
    Args:
        data: Input dataset
        heuristic: Strategy - 'length', 'answer_length', 'quality_score', 'dedup'
        keep_long: For length-based: keep long (True) or short (False) samples
        keep_ratio: Proportion of data to keep (for non-dedup strategies)
        similarity_threshold: Jaccard threshold for dedup
    
    Returns:
        Selected subset with scores
    """
    total = len(data)
    
    if heuristic == 'dedup':
        keep_indices = _dedup_samples(data, similarity_threshold)
        selected = []
        for i in keep_indices:
            item = dict(data[i])
            item["score"] = 1.0
            selected.append(item)
        print(f"[Heuristic:dedup] Kept {len(keep_indices)}/{total} unique samples (threshold={similarity_threshold})")
        return selected
    
    # Score-based selection
    scores = []
    for sample in data:
        if heuristic == 'length':
            scores.append(_get_sample_length(sample))
        elif heuristic == 'answer_length':
            scores.append(_get_answer_length(sample))
        elif heuristic == 'quality_score':
            scores.append(_get_quality_score(sample))
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")
    
    # Sort by score
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=keep_long)
    
    keep_count = max(1, int(total * keep_ratio))
    keep_indices = [idx for idx, _ in indexed[:keep_count]]
    
    selected = []
    for i in keep_indices:
        item = dict(data[i])
        item["score"] = scores[i]
        selected.append(item)
    
    print(f"[Heuristic:{heuristic}] Selected {keep_count}/{total} samples (keep_long={keep_long}, keep_ratio={keep_ratio})")
    return selected


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Data selection for Llama-Factory fine-tuning")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--method", type=str, required=True, choices=['random', 'heuristic'],
                        help="Selection method")
    
    # Random args
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Heuristic args
    parser.add_argument("--heuristic", type=str, default='length',
                        choices=['length', 'answer_length', 'quality_score', 'dedup'],
                        help="Heuristic strategy")
    parser.add_argument("--keep_long", action='store_true', default=True,
                        help="Keep long samples (for length-based heuristics)")
    parser.add_argument("--no-keep_long", action='store_false', dest='keep_long',
                        help="Keep short samples")
    parser.add_argument("--keep_ratio", type=float, default=0.5,
                        help="Proportion of data to keep (0.0-1.0)")
    parser.add_argument("--similarity_threshold", type=float, default=0.8,
                        help="Jaccard similarity threshold for dedup (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_path}...")
    data = load_jsonl(args.input_path)
    print(f"Loaded {len(data)} samples")
    
    # Select
    if args.method == 'random':
        selected = random_select(data, args.keep_ratio, args.seed)
    elif args.method == 'heuristic':
        selected = heuristic_select(
            data,
            heuristic=args.heuristic,
            keep_long=args.keep_long,
            keep_ratio=args.keep_ratio,
            similarity_threshold=args.similarity_threshold
        )
    
    # Save
    save_jsonl(args.output_path, selected)
    
    # Print stats
    print(f"\nSelection summary:")
    print(f"  Method: {args.method}")
    if args.method == 'heuristic':
        print(f"  Heuristic: {args.heuristic}")
    print(f"  Original: {len(data)} samples")
    print(f"  Selected: {len(selected)} samples")
    print(f"  Keep ratio: {len(selected)/len(data):.2%}")


if __name__ == "__main__":
    main()
