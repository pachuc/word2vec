import numpy as np
import torch
from torch.utils.data import Dataset


class SkipGramDataset(Dataset):
    """
    Map-style dataset for precomputed skip-gram pairs.

    Supports memory-mapped loading for large datasets and handles
    negative sampling on-the-fly (which is fast).
    """

    def __init__(self, pairs_path: str, negative_sampler, num_negatives: int = 5,
                 mmap_mode: str = 'r'):
        """
        Args:
            pairs_path: Path to .npy file containing (center, context) pairs
            negative_sampler: NegativeSampler instance for generating negatives
            num_negatives: Number of negative samples per pair
            mmap_mode: Memory map mode ('r' for read-only, None to load into RAM)
        """
        self.pairs = np.load(pairs_path, mmap_mode=mmap_mode)
        self.negative_sampler = negative_sampler
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center_idx, context_idx = self.pairs[idx]

        # Sample negatives (fast operation)
        negatives = self.negative_sampler.sample(
            self.num_negatives,
            exclude_idx=int(context_idx)
        )

        return (
            torch.tensor(center_idx, dtype=torch.long),
            torch.tensor(context_idx, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long)
        )
