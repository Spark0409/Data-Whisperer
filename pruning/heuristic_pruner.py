"""
Heuristic Pruner: Simple heuristic-based data selection strategies.

Supported strategies:
- length: Select samples based on input length (sample length)
- answer_length: Select samples based on answer/output length
- quality_score: Select samples based on a rule-based quality score
- dedup: Select samples based on deduplication (remove near-duplicates)
"""
import os
import re
import torch
from collections import Counter
from utils.utils import load_json, save_json, timer_decorator
from pruner import Pruner
from typing import List, Dict, Any, Optional


class HeuristicPruner(Pruner):
    """
    Heuristic data selection strategy.
    Supports multiple heuristic criteria for data selection.
    No model needed - purely heuristic-based selection.
    """
    def __init__(self, args: Any) -> None:
        # Skip the parent __init__ which loads a model (not needed for heuristic selection)
        self.args = args
        self.dataset = self.args.dataset
        # Heuristic strategy: length, answer_length, quality_score, dedup
        self.heuristic = getattr(self.args, 'heuristic', 'length')
        # Whether to keep long samples (True) or short samples (False)
        self.keep_long = getattr(self.args, 'keep_long', True)
        # Keep ratio
        self.keep_ratio = getattr(self.args, 'keep_ratio', 0.5)
        # For dedup: similarity threshold (0.0 to 1.0)
        self.similarity_threshold = getattr(self.args, 'similarity_threshold', 0.8)

    def predict_batch(self, demonstrations, val_samples):
        """Not used for heuristic selection."""
        pass

    def _get_sample_length(self, sample: Dict[str, Any]) -> int:
        """Get the total text length of a sample."""
        text = ""
        for key, value in sample.items():
            if isinstance(value, str):
                text += value + " "
        return len(text)

    def _get_answer_length(self, sample: Dict[str, Any]) -> int:
        """Get the answer/output length of a sample."""
        for key in ['output', 'answer', 'summary']:
            if key in sample and isinstance(sample[key], str):
                return len(sample[key])
        return 0

    def _get_quality_score(self, sample: Dict[str, Any]) -> float:
        """
        Compute a simple rule-based quality score for a sample.
        Higher scores indicate higher quality.
        
        Scoring criteria:
        1. Length score: samples that are too short or too long get lower scores
        2. Keyword diversity: more diverse vocabulary indicates higher quality
        3. Special character penalty: too many special characters lower the score
        4. Repetition penalty: repeated phrases lower the score
        """
        score = 1.0
        
        # Get the main text content
        text = ""
        for key in ['output', 'answer', 'summary', 'instruction', 'question', 'dialogue']:
            if key in sample and isinstance(sample[key], str):
                text += sample[key] + " "
        
        if not text.strip():
            return 0.0
        
        # 1. Length score (Gaussian-like: optimal around 100-500 chars)
        text_len = len(text)
        if text_len < 10:
            score *= 0.3  # Too short
        elif text_len < 50:
            score *= 0.7
        elif text_len > 2000:
            score *= 0.6  # Too long
        elif text_len > 1000:
            score *= 0.8
        
        # 2. Keyword diversity (unique words / total words)
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score *= 0.5  # Very repetitive
            elif unique_ratio < 0.5:
                score *= 0.8
        
        # 3. Special character penalty
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        if words and special_chars / len(words) > 0.5:
            score *= 0.7
        
        # 4. Repetition detection (check for repeated n-grams)
        words_list = text.lower().split()
        if len(words_list) >= 10:
            # Check 3-gram repetition
            tri_grams = [' '.join(words_list[i:i+3]) for i in range(len(words_list)-2)]
            if tri_grams:
                most_common_count = Counter(tri_grams).most_common(1)[0][1]
                if most_common_count > len(tri_grams) * 0.3:
                    score *= 0.6  # High repetition
        
        return score

    def _dedup_samples(self, dataset: List[Dict[str, Any]]) -> List[int]:
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
                # Compute Jaccard similarity
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                if union > 0 and intersection / union > self.similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)
        
        return keep_indices

    def evaluate(self, dataset, val_set=None, use_kfold=False):
        """
        Evaluate dataset using heuristic criteria and select top samples.
        """
        total_size = len(dataset)
        
        if self.heuristic == 'dedup':
            # Deduplication: keep unique samples
            keep_indices = self._dedup_samples(dataset)
            sorted_dataset_with_scores = [
                {**dataset[i], "score": 1.0}
                for i in keep_indices
            ]
            output_path = os.path.join(
                self.args.output_filtered_path,
                f"heuristic_dedup_th{self.similarity_threshold:.2f}.json"
            )
            print(f"Deduplication completed. Kept {len(keep_indices)}/{total_size} unique samples.")
        
        else:
            # Score-based selection
            scores = torch.zeros(total_size, dtype=torch.float16)
            
            for i, sample in enumerate(dataset):
                if self.heuristic == 'length':
                    scores[i] = self._get_sample_length(sample)
                elif self.heuristic == 'answer_length':
                    scores[i] = self._get_answer_length(sample)
                elif self.heuristic == 'quality_score':
                    scores[i] = self._get_quality_score(sample)
                else:
                    raise ValueError(f"Unknown heuristic: {self.heuristic}")
            
            # Sort by score
            if self.keep_long:
                sorted_idx = torch.argsort(scores, descending=True)
            else:
                sorted_idx = torch.argsort(scores, descending=False)
            
            # Keep top proportion
            keep_count = max(1, int(total_size * self.keep_ratio))
            keep_idx = sorted_idx[:keep_count]
            
            sorted_dataset_with_scores = [
                {
                    **dataset[i.item()],
                    "score": scores[i].item(),
                }
                for i in keep_idx
            ]
            
            output_path = os.path.join(
                self.args.output_filtered_path,
                f"heuristic_{self.heuristic}_keep{self.keep_ratio:.2f}.json"
            )
            print(f"Heuristic ({self.heuristic}) selection completed. Kept {keep_count}/{total_size} samples.")
        
        save_json(output_path, sorted_dataset_with_scores)
        print(f"Results saved to {output_path}")
        return output_path

    @timer_decorator
    def do_pruning(self):
        """
        Override do_pruning to skip model-dependent logic.
        Heuristic selection doesn't need a validation set or k-fold.
        """
        dataset = load_json(self.args.data_path)
        output_path = self.evaluate(dataset)
        print(f"Heuristic pruning completed. Filtered training data saved to {output_path}", flush=True)
