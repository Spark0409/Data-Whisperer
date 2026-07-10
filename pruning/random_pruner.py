"""
Random Pruner: Randomly selects a specified proportion of data.
"""
import os
import random
import torch
from utils.utils import load_json, save_json, timer_decorator
from pruner import Pruner
from typing import List, Dict, Any, Optional


class RandomPruner(Pruner):
    """
    Random data selection strategy.
    Randomly selects a specified proportion of data from the dataset.
    No model needed - purely random selection.
    """
    def __init__(self, args: Any) -> None:
        # Skip the parent __init__ which loads a model (not needed for random selection)
        self.args = args
        self.dataset = self.args.dataset
        # Default keep ratio if not specified
        self.keep_ratio = getattr(self.args, 'keep_ratio', 0.5)
        self.seed = getattr(self.args, 'seed', 42)

    def predict_batch(self, demonstrations, val_samples):
        """Not used for random selection."""
        pass

    def evaluate(self, dataset, val_set=None, use_kfold=False):
        """
        Randomly select a proportion of data.
        """
        total_size = len(dataset)
        random.seed(self.seed)
        
        # Generate random scores
        scores = torch.rand(total_size, dtype=torch.float16)
        
        # Sort by random scores (descending)
        sorted_idx = torch.argsort(scores, descending=True)
        
        # Keep only the top keep_ratio proportion
        keep_count = int(total_size * self.keep_ratio)
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
            f"random_keep{self.keep_ratio:.2f}.json"
        )
        
        save_json(output_path, sorted_dataset_with_scores)
        print(f"Random selection completed. Kept {keep_count}/{total_size} samples. Results saved to {output_path}")
        return output_path

    @timer_decorator
    def do_pruning(self):
        """
        Override do_pruning to skip model-dependent logic.
        Random selection doesn't need a validation set or k-fold.
        """
        dataset = load_json(self.args.data_path)
        output_path = self.evaluate(dataset)
        print(f"Random pruning completed. Filtered training data saved to {output_path}", flush=True)
