"""
Data Analyzer: Multi-dimensional feature analysis for selected vs unselected data.

This module provides tools to analyze and compare characteristics of data selected
by different pruning strategies (Data Whisperer, Random, Heuristic), helping to
understand what features distinguish high-value data.

Supported analysis dimensions:
1. Text statistical features (length, vocabulary diversity, etc.)
2. Reasoning structure features (steps, intermediate calculations, etc.)
3. Answer format features (format compliance, completeness, etc.)
4. Semantic similarity to target task (embedding-based)
5. Comparative analysis between selected and unselected data
"""

import re
import json
import os
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class FeatureStats:
    """Statistical summary of a single feature."""
    selected_mean: float = 0.0
    selected_std: float = 0.0
    selected_median: float = 0.0
    selected_q25: float = 0.0
    selected_q75: float = 0.0
    unselected_mean: float = 0.0
    unselected_std: float = 0.0
    unselected_median: float = 0.0
    unselected_q25: float = 0.0
    unselected_q75: float = 0.0
    effect_size: float = 0.0  # Cohen's d
    difference_pct: float = 0.0  # Percentage difference
    p_value: float = 0.0  # Statistical significance (Mann-Whitney U test)


@dataclass
class AnalysisResult:
    """Complete analysis result for a dataset comparison."""
    total_samples: int = 0
    selected_count: int = 0
    unselected_count: int = 0
    method: str = ""
    keep_ratio: float = 0.0
    feature_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    top_features: List[Tuple[str, Dict[str, float]]] = field(default_factory=list)
    summary: str = ""


class DataAnalyzer:
    """
    Multi-dimensional data feature analyzer.
    
    Analyzes and compares characteristics of selected vs unselected data samples
    to identify distinguishing features of high-value data.
    """

    # Dataset-specific field mappings
    DATASET_FIELDS = {
        "gsm8k": {
            "input": "question",
            "output": "answer",
            "extra_fields": [],
        },
        "dialogsum": {
            "input": "dialogue",
            "output": "summary",
            "extra_fields": [],
        },
        "bioinstruct": {
            "input": "instruction",
            "output": "output",
            "extra_fields": ["input"],
        },
        "llava_1k": {
            "input": "question",
            "output": "answer",
            "extra_fields": ["image"],
        },
    }

    def __init__(self, dataset_type: str = "gsm8k"):
        """
        Initialize the analyzer for a specific dataset type.
        
        Args:
            dataset_type: Type of dataset ('gsm8k', 'dialogsum', 'bioinstruct', 'llava_1k')
        """
        self.dataset_type = dataset_type
        self.fields = self.DATASET_FIELDS.get(dataset_type, self.DATASET_FIELDS["gsm8k"])

    # =========================================================================
    # 1. Text Statistical Features
    # =========================================================================

    def analyze_text_features(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze basic text statistical features of a sample.
        
        Features:
        - Input/output/total text length
        - Word count
        - Vocabulary diversity (unique words / total words)
        - Number of numeric values
        - Special character ratio
        - Average word length
        """
        input_text = str(sample.get(self.fields["input"], ""))
        output_text = str(sample.get(self.fields["output"], ""))
        full_text = input_text + " " + output_text

        input_words = input_text.split()
        output_words = output_text.split()
        full_words = full_text.split()

        features = {
            # Length features
            "input_length": len(input_text),
            "output_length": len(output_text),
            "total_length": len(full_text),
            "input_word_count": len(input_words),
            "output_word_count": len(output_words),
            "total_word_count": len(full_words),

            # Vocabulary diversity
            "input_vocab_diversity": (
                len(set(w.lower() for w in input_words)) / max(len(input_words), 1)
            ),
            "output_vocab_diversity": (
                len(set(w.lower() for w in output_words)) / max(len(output_words), 1)
            ),
            "full_vocab_diversity": (
                len(set(w.lower() for w in full_words)) / max(len(full_words), 1)
            ),

            # Numeric features
            "num_numbers_in_input": len(re.findall(r'\d+\.?\d*', input_text)),
            "num_numbers_in_output": len(re.findall(r'\d+\.?\d*', output_text)),
            "num_numbers_total": len(re.findall(r'\d+\.?\d*', full_text)),

            # Special character ratio
            "special_char_ratio_input": (
                len(re.findall(r'[^a-zA-Z0-9\s]', input_text)) / max(len(input_text), 1)
            ),
            "special_char_ratio_output": (
                len(re.findall(r'[^a-zA-Z0-9\s]', output_text)) / max(len(output_text), 1)
            ),

            # Average word length
            "avg_word_length_input": (
                np.mean([len(w) for w in input_words]) if input_words else 0
            ),
            "avg_word_length_output": (
                np.mean([len(w) for w in output_words]) if output_words else 0
            ),
        }

        return features

    # =========================================================================
    # 2. Reasoning Structure Features (GSM8K-specific)
    # =========================================================================

    def analyze_reasoning_structure(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze reasoning structure features.
        
        For GSM8K:
        - Number of reasoning steps (split by newlines/sentences)
        - Number of intermediate calculations (<<...>> markers)
        - Whether answer has proper format (####)
        - Average tokens per reasoning step
        - Presence of arithmetic operators
        
        For dialogsum:
        - Number of dialogue turns
        - Average utterance length
        
        For bioinstruct:
        - Instruction complexity (length, specificity)
        """
        output_text = str(sample.get(self.fields["output"], ""))
        input_text = str(sample.get(self.fields["input"], ""))

        if self.dataset_type == "gsm8k":
            return self._analyze_gsm8k_reasoning(output_text, input_text)
        elif self.dataset_type == "dialogsum":
            return self._analyze_dialogsum_structure(input_text, output_text)
        elif self.dataset_type == "bioinstruct":
            return self._analyze_bioinstruct_structure(sample)
        else:
            return {"reasoning_complexity": len(output_text.split())}

    def _analyze_gsm8k_reasoning(
        self, output_text: str, input_text: str
    ) -> Dict[str, float]:
        """GSM8K-specific reasoning analysis."""
        # Reasoning steps (split by newlines, filter empty)
        steps = [s.strip() for s in output_text.split('\n') if s.strip()]
        
        # Intermediate calculations (<<expr>>)
        intermediate_calcs = re.findall(r'<<([^>]+)>>', output_text)
        
        # Arithmetic expressions
        arithmetic_ops = re.findall(r'[\+\-\*/]', output_text)
        
        # Final answer format
        has_final_answer = '####' in output_text
        final_answer_match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', output_text)
        
        # Question complexity indicators
        question_numbers = re.findall(r'\d+\.?\d*', input_text)
        question_has_multiple_entities = len(set(question_numbers)) > 2
        
        return {
            "reasoning_steps": len(steps),
            "intermediate_calculations": len(intermediate_calcs),
            "arithmetic_operations": len(arithmetic_ops),
            "has_proper_format": float(has_final_answer),
            "has_extracted_answer": float(final_answer_match is not None),
            "avg_step_length": (
                float(np.mean([len(s) for s in steps])) if steps else 0.0
            ),
            "max_step_length": (
                float(max(len(s) for s in steps)) if steps else 0.0
            ),
            "question_complexity": float(question_has_multiple_entities),
            "output_newlines": output_text.count('\n'),
        }

    def _analyze_dialogsum_structure(
        self, dialogue: str, summary: str
    ) -> Dict[str, float]:
        """DialogSum-specific structure analysis."""
        # Count dialogue turns (speaker changes)
        speaker_pattern = re.findall(r'#\w+#:', dialogue)
        turns = len(speaker_pattern) if speaker_pattern else max(dialogue.count('\n'), 1)
        
        # Average utterance length
        utterances = [u.strip() for u in re.split(r'#\w+#:', dialogue) if u.strip()]
        avg_utterance_len = np.mean([len(u) for u in utterances]) if utterances else 0
        
        # Summary compression ratio
        compression_ratio = len(dialogue) / max(len(summary), 1)
        
        return {
            "dialogue_turns": turns,
            "avg_utterance_length": avg_utterance_len,
            "summary_compression_ratio": compression_ratio,
            "dialogue_length": len(dialogue),
            "summary_length": len(summary),
        }

    def _analyze_bioinstruct_structure(
        self, sample: Dict[str, Any]
    ) -> Dict[str, float]:
        """BioInstruct-specific structure analysis."""
        instruction = str(sample.get("instruction", ""))
        input_text = str(sample.get("input", ""))
        output_text = str(sample.get("output", ""))
        
        # Instruction specificity (contains specific medical terms)
        medical_terms = [
            'diagnosis', 'treatment', 'symptom', 'disease', 'patient',
            'clinical', 'therapy', 'medication', 'surgery', 'syndrome'
        ]
        term_count = sum(1 for term in medical_terms if term.lower() in instruction.lower())
        
        return {
            "instruction_length": len(instruction),
            "input_length": len(input_text),
            "output_length": len(output_text),
            "medical_term_count": term_count,
            "instruction_complexity": len(instruction.split()),
            "has_input_context": float(len(input_text.strip()) > 0),
        }

    # =========================================================================
    # 3. Answer Format Features
    # =========================================================================

    def analyze_answer_format(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze answer format quality and completeness.
        
        Features:
        - Whether answer contains reasoning process
        - Whether answer has proper final format
        - Answer completeness score
        - Format consistency score
        """
        output_text = str(sample.get(self.fields["output"], ""))
        
        # General format features
        has_content = len(output_text.strip()) > 0
        has_reasoning = len(output_text.split()) > 5
        
        # Dataset-specific format checks
        if self.dataset_type == "gsm8k":
            format_score = self._check_gsm8k_format(output_text)
        elif self.dataset_type == "dialogsum":
            format_score = self._check_summary_format(output_text)
        else:
            format_score = 1.0 if has_content else 0.0
        
        return {
            "has_content": float(has_content),
            "has_reasoning_process": float(has_reasoning),
            "format_score": format_score,
            "answer_completeness": min(len(output_text) / 100.0, 1.0),
        }

    def _check_gsm8k_format(self, answer: str) -> float:
        """Check GSM8K answer format compliance."""
        score = 0.0
        
        # Must have #### with number
        if re.search(r'####\s*[\d,]+(?:\.\d+)?', answer):
            score += 0.4
        
        # Should have step-by-step reasoning
        if answer.count('\n') >= 2:
            score += 0.3
        
        # Should have intermediate calculations
        if '<<' in answer and '>>' in answer:
            score += 0.3
        
        return score

    def _check_summary_format(self, summary: str) -> float:
        """Check summary format quality."""
        score = 0.0
        
        # Reasonable summary length (not too short, not too long)
        if 20 <= len(summary) <= 500:
            score += 0.3
        
        # Contains key information (not just filler)
        if len(summary.split()) >= 5:
            score += 0.3
        
        # Proper sentence structure
        if summary.count('.') >= 1:
            score += 0.2
        
        # No obvious issues
        if not summary.startswith(('Summary:', 'summary:')):
            score += 0.2
        
        return score

    # =========================================================================
    # 4. Semantic Similarity to Target Task
    # =========================================================================

    def compute_similarity_to_target(
        self,
        selected_texts: List[str],
        target_texts: List[str],
        model_name: str = 'all-MiniLM-L6-v2',
    ) -> np.ndarray:
        """
        Compute semantic similarity between selected data and target task data.
        
        Uses sentence-transformers to compute embedding similarity.
        Falls back to TF-IDF if sentence-transformers is not available.
        
        Args:
            selected_texts: List of texts from selected samples
            target_texts: List of texts from target task (e.g., test set)
            model_name: Sentence transformer model name
            
        Returns:
            Array of max similarity scores for each selected sample
        """
        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            selected_emb = model.encode(selected_texts, show_progress_bar=False)
            target_emb = model.encode(target_texts, show_progress_bar=False)
            
            # Normalize embeddings
            selected_emb = selected_emb / np.linalg.norm(selected_emb, axis=1, keepdims=True)
            target_emb = target_emb / np.linalg.norm(target_emb, axis=1, keepdims=True)
            
            # Cosine similarity: for each selected sample, find max similarity to any target
            similarities = np.max(selected_emb @ target_emb.T, axis=1)
            
        except ImportError:
            # Fallback to TF-IDF + cosine similarity
            similarities = self._tfidf_similarity(selected_texts, target_texts)
        
        return similarities

    def _tfidf_similarity(
        self, texts_a: List[str], texts_b: List[str]
    ) -> np.ndarray:
        """Compute TF-IDF based similarity as fallback."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            all_texts = texts_a + texts_b
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            a_matrix = tfidf_matrix[:len(texts_a)]
            b_matrix = tfidf_matrix[len(texts_a):]
            
            similarities = np.max(cosine_similarity(a_matrix, b_matrix), axis=1)
            return similarities
            
        except ImportError:
            print("Warning: sklearn not available. Returning zeros for similarity.")
            return np.zeros(len(texts_a))

    # =========================================================================
    # 5. Comparative Analysis
    # =========================================================================

    def extract_all_features(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract all features from a single sample.
        
        Combines text features, reasoning features, and format features
        into a single feature dictionary.
        """
        features = {}
        features.update(self.analyze_text_features(sample))
        features.update(self.analyze_reasoning_structure(sample))
        features.update(self.analyze_answer_format(sample))
        return features

    def compute_feature_stats(
        self, feature_values: List[float]
    ) -> Dict[str, float]:
        """Compute statistical summary for a list of feature values."""
        arr = np.array(feature_values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def compare_selected_vs_unselected(
        self,
        dataset: List[Dict[str, Any]],
        selected_indices: List[int],
        method: str = "unknown",
        keep_ratio: float = 0.0,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison between selected and unselected data.
        
        Args:
            dataset: Full dataset
            selected_indices: Indices of selected samples
            method: Selection method name (for reporting)
            keep_ratio: Proportion of data kept
            output_path: Path to save analysis results (JSON)
            
        Returns:
            Dictionary containing full analysis results
        """
        selected_set = set(selected_indices)
        selected_samples = [dataset[i] for i in range(len(dataset)) if i in selected_set]
        unselected_samples = [
            dataset[i] for i in range(len(dataset)) if i not in selected_set
        ]

        print(f"Analyzing {len(selected_samples)} selected vs "
              f"{len(unselected_samples)} unselected samples...")

        # Extract features for all samples
        selected_features = [self.extract_all_features(s) for s in selected_samples]
        unselected_features = [self.extract_all_features(s) for s in unselected_samples]

        # Get all feature names
        feature_names = list(selected_features[0].keys()) if selected_features else []

        # Compute comparison for each feature
        comparison = {}
        for name in feature_names:
            sel_values = [f[name] for f in selected_features]
            unsel_values = [f[name] for f in unselected_features]

            if not sel_values or not unsel_values:
                continue

            sel_stats = self.compute_feature_stats(sel_values)
            unsel_stats = self.compute_feature_stats(unsel_values)

            # Cohen's d effect size
            pooled_std = np.sqrt((sel_stats["std"] ** 2 + unsel_stats["std"] ** 2) / 2)
            effect_size = (
                (sel_stats["mean"] - unsel_stats["mean"]) / max(pooled_std, 1e-8)
            )

            # Percentage difference
            diff_pct = (
                (sel_stats["mean"] - unsel_stats["mean"]) / max(abs(unsel_stats["mean"]), 1e-8) * 100
            )

            # Mann-Whitney U test for significance
            try:
                from scipy.stats import mannwhitneyu
                _, p_value = mannwhitneyu(sel_values, unsel_values, alternative='two-sided')
            except ImportError:
                p_value = 0.0

            comparison[name] = {
                "selected": sel_stats,
                "unselected": unsel_stats,
                "effect_size": round(effect_size, 4),
                "difference_pct": round(diff_pct, 2),
                "p_value": round(float(p_value), 6),
                "significant": float(p_value < 0.05),
            }

        # Sort features by absolute effect size
        top_features = sorted(
            comparison.items(),
            key=lambda x: abs(x[1]["effect_size"]),
            reverse=True,
        )

        # Generate summary
        summary_lines = [
            f"Dataset: {self.dataset_type}",
            f"Method: {method}",
            f"Keep ratio: {keep_ratio:.0%}",
            f"Total samples: {len(dataset)}",
            f"Selected: {len(selected_samples)}, Unselected: {len(unselected_samples)}",
            "",
            "=== Top distinguishing features (by effect size) ===",
        ]

        for name, stats in top_features[:10]:
            arrow = "↑" if stats["effect_size"] > 0 else "↓"
            sig = "*" if stats["significant"] else ""
            summary_lines.append(
                f"  {arrow} {name}: "
                f"selected={stats['selected']['mean']:.2f}±{stats['selected']['std']:.2f} vs "
                f"unselected={stats['unselected']['mean']:.2f}±{stats['unselected']['std']:.2f} "
                f"(d={stats['effect_size']:+.3f}, Δ={stats['difference_pct']:+.1f}%){sig}"
            )

        summary = "\n".join(summary_lines)
        print("\n" + summary + "\n")

        # Build result
        result = {
            "metadata": {
                "dataset_type": self.dataset_type,
                "method": method,
                "keep_ratio": keep_ratio,
                "total_samples": len(dataset),
                "selected_count": len(selected_samples),
                "unselected_count": len(unselected_samples),
            },
            "feature_comparison": comparison,
            "top_features": [(name, comparison[name]) for name, _ in top_features],
            "summary": summary,
        }

        # Save to file if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Analysis results saved to {output_path}")

        return result

    # =========================================================================
    # 6. Batch Analysis for Experiment Results
    # =========================================================================

    def analyze_experiment_results(
        self,
        full_data_path: str,
        experiment_results_dir: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Analyze all experiment results in a directory structure.
        
        Expected directory structure:
            experiment_results_dir/
                {method}/
                    ratio_{ratio}/
                        repeat_{repeat}/
                            subset.json
        
        Args:
            full_data_path: Path to the full dataset JSON
            experiment_results_dir: Root directory of experiment results
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary mapping experiment configs to analysis results
        """
        import glob

        # Load full dataset
        with open(full_data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        # Find all subset.json files
        subset_files = glob.glob(
            os.path.join(experiment_results_dir, "*", "ratio_*", "repeat_*", "subset.json")
        )

        all_results = {}

        for subset_file in subset_files:
            # Parse method, ratio, repeat from path
            rel_path = os.path.relpath(subset_file, experiment_results_dir)
            parts = rel_path.replace('\\', '/').split('/')
            
            method = parts[0]
            ratio_str = parts[1].replace('ratio_', '')
            repeat_str = parts[2].replace('repeat_', '')
            
            try:
                ratio = float(ratio_str)
                repeat = int(repeat_str)
            except ValueError:
                print(f"Skipping invalid path: {subset_file}")
                continue

            # Load subset
            with open(subset_file, 'r', encoding='utf-8') as f:
                subset = json.load(f)

            # Find selected indices
            selected_indices = []
            for i, item in enumerate(full_data):
                # Match by content (question/answer)
                for j, sub_item in enumerate(subset):
                    if (sub_item.get(self.fields["input"]) == item.get(self.fields["input"])
                            and sub_item.get(self.fields["output"]) == item.get(self.fields["output"])):
                        selected_indices.append(i)
                        break

            if not selected_indices:
                print(f"Warning: No matching indices found for {subset_file}")
                continue

            # Run analysis
            analysis_output = os.path.join(
                output_dir, method, f"ratio_{ratio_str}", f"repeat_{repeat_str}", "analysis.json"
            )

            result = self.compare_selected_vs_unselected(
                full_data,
                selected_indices,
                method=method,
                keep_ratio=ratio,
                output_path=analysis_output,
            )

            key = f"{method}_r{ratio}_rep{repeat}"
            all_results[key] = result

        # Generate cross-method comparison
        self._generate_cross_method_comparison(all_results, output_dir)

        return all_results

    def _generate_cross_method_comparison(
        self,
        all_results: Dict[str, Any],
        output_dir: str,
    ) -> None:
        """
        Generate a comparison table across different methods and ratios.
        """
        # Group results by method and ratio
        grouped = {}
        for key, result in all_results.items():
            method = result["metadata"]["method"]
            ratio = result["metadata"]["keep_ratio"]
            
            if method not in grouped:
                grouped[method] = {}
            if ratio not in grouped[method]:
                grouped[method][ratio] = []
            
            grouped[method][ratio].append(result)

        # For each feature, compute average effect size per method/ratio
        feature_names = []
        for result in all_results.values():
            feature_names = list(result["feature_comparison"].keys())
            break

        cross_comparison = {}
        for feature in feature_names:
            cross_comparison[feature] = {}
            for method, ratios in grouped.items():
                cross_comparison[feature][method] = {}
                for ratio, results in ratios.items():
                    effect_sizes = [
                        r["feature_comparison"][feature]["effect_size"]
                        for r in results
                    ]
                    cross_comparison[feature][method][str(ratio)] = {
                        "mean_effect_size": round(np.mean(effect_sizes), 4),
                        "std_effect_size": round(np.std(effect_sizes), 4),
                    }

        # Save cross-method comparison
        output_path = os.path.join(output_dir, "cross_method_comparison.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cross_comparison, f, ensure_ascii=False, indent=2)

        print(f"Cross-method comparison saved to {output_path}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for data analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Analyzer: Analyze features of selected vs unselected data"
    )
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["gsm8k", "dialogsum", "bioinstruct", "llava_1k"],
        help="Dataset type"
    )
    parser.add_argument(
        "--full_data", type=str, required=True,
        help="Path to the full dataset JSON file"
    )
    parser.add_argument(
        "--subset", type=str, required=True,
        help="Path to the selected subset JSON file"
    )
    parser.add_argument(
        "--method", type=str, default="unknown",
        help="Selection method name (for reporting)"
    )
    parser.add_argument(
        "--keep_ratio", type=float, default=0.0,
        help="Proportion of data kept"
    )
    parser.add_argument(
        "--output", type=str, default="analysis_results.json",
        help="Path to save analysis results"
    )
    parser.add_argument(
        "--similarity", action="store_true",
        help="Compute semantic similarity to target task"
    )
    parser.add_argument(
        "--target_data", type=str, default=None,
        help="Path to target task data (for similarity computation)"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading full dataset from {args.full_data}...")
    with open(args.full_data, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    print(f"Loading subset from {args.subset}...")
    with open(args.subset, 'r', encoding='utf-8') as f:
        subset = json.load(f)

    # Find selected indices
    analyzer = DataAnalyzer(dataset_type=args.dataset)
    input_field = analyzer.fields["input"]
    output_field = analyzer.fields["output"]

    selected_indices = []
    for i, item in enumerate(full_data):
        for sub_item in subset:
            if (sub_item.get(input_field) == item.get(input_field)
                    and sub_item.get(output_field) == item.get(output_field)):
                selected_indices.append(i)
                break

    print(f"Found {len(selected_indices)} matching indices out of {len(subset)} subset samples.")

    if not selected_indices:
        print("Error: No matching samples found. Check that the subset is derived from the full dataset.")
        return

    # Run analysis
    result = analyzer.compare_selected_vs_unselected(
        full_data,
        selected_indices,
        method=args.method,
        keep_ratio=args.keep_ratio,
        output_path=args.output,
    )

    # Optional: compute semantic similarity
    if args.similarity and args.target_data:
        print(f"\nComputing semantic similarity to target task ({args.target_data})...")
        with open(args.target_data, 'r', encoding='utf-8') as f:
            target_data = json.load(f)

        selected_texts = [
            str(s.get(input_field, "")) + " " + str(s.get(output_field, ""))
            for s in subset
        ]
        target_texts = [
            str(t.get(input_field, "")) + " " + str(t.get(output_field, ""))
            for t in target_data
        ]

        similarities = analyzer.compute_similarity_to_target(selected_texts, target_texts)
        
        similarity_stats = {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
        }
        
        print(f"Similarity to target: mean={similarity_stats['mean_similarity']:.4f}, "
              f"std={similarity_stats['std_similarity']:.4f}")

        # Append similarity stats to result
        result["similarity_to_target"] = similarity_stats
        
        # Update output file
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nAnalysis complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
